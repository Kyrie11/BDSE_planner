from __future__ import annotations

from typing import Any

import itertools
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

try:  # pragma: no cover - exercised only when nuPlan is installed
    from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
except Exception:  # pragma: no cover - lightweight/unit-test fallback
    class AbstractPlanner:  # type: ignore[no-redef]
        """Small compatibility shim for environments without nuPlan installed.

        nuPlan's real ``AbstractPlanner`` defines ``requires_scenario`` and
        routes ``compute_trajectory`` through ``compute_planner_trajectory``
        while recording runtimes.  The shim keeps the public method available
        for tests and local debugging without importing nuPlan.
        """

        requires_scenario: bool = False

        def compute_trajectory(self, current_input: Any):
            return self.compute_planner_trajectory(current_input)

        def generate_planner_report(self, clear_stats: bool = True):
            return None

from bdse.config import load_config
from bdse.utils import angle_wrap, configure_torch_for_device, resolve_torch_device, torch_load_any
from bdse.data.cache_schema import RuntimeFeatures
from bdse.data.nuplan_runtime_adapter import build_runtime_features_from_planner_input
from bdse.planner.candidate_generator import generate_candidate_bank
from bdse.planner.evidence_atoms import enumerate_evidence_atoms
from bdse.planner.evidence_queries import compute_query_features_for_pairs
from bdse.planner.hab import family_ids_from_atoms, select_topm_atoms_hab
from bdse.planner.fallback import runtime_safety_flags_from_runtime
from bdse.planner.pair_screen import build_runtime_pairs_from_base, build_rival_sets_from_base
from bdse.planner.selector import (
    SelectionResult,
    runtime_greedy_selector,
    runtime_greedy_selector_pair_conditioned,
    select_by_mode,
    structural_safety_mask,
)
from bdse.planner.tournament import run_tournament, run_pair_conditioned_tournament, selected_pair_sigma_from_action_variance


_PLANNER_DEVICE_LOCK = threading.Lock()
_PLANNER_DEVICE_COUNTER = itertools.count()
_NUPLAN_IMPORT_CACHE: dict[str, Any] = {}


def _cached_import(module: str, name: str) -> Any:
    key = f"{module}:{name}"
    if key not in _NUPLAN_IMPORT_CACHE:
        mod = __import__(module, fromlist=[name])
        _NUPLAN_IMPORT_CACHE[key] = getattr(mod, name)
    return _NUPLAN_IMPORT_CACHE[key]


def _maybe_shard_planner_device(device: str | None) -> str | None:
    """Optionally distribute per-simulation planner instances across visible CUDA devices.

    nuPlan builds one planner object per simulation.  With device="cuda", PyTorch
    otherwise puts every object on cuda:0.  Set BDSE_SHARD_PLANNERS_ACROSS_GPUS=1
    to assign cuda:0, cuda:1, ... round-robin inside a single nuPlan process.
    """
    requested = str(device or "auto").strip().lower()
    enabled = str(os.environ.get("BDSE_SHARD_PLANNERS_ACROSS_GPUS", "0")).lower() in {"1", "true", "yes", "on"}
    if not enabled or requested not in {"cuda", "auto", "gpu"}:
        return device
    try:
        import torch

        if not torch.cuda.is_available():
            return device
        n = int(torch.cuda.device_count())
        if n <= 1:
            return device
        with _PLANNER_DEVICE_LOCK:
            idx = next(_PLANNER_DEVICE_COUNTER) % n
        return f"cuda:{idx}"
    except Exception:
        return device


def runtime_query_diagnostics(pred: dict[str, Any], selected_atoms: list[int] | np.ndarray | None = None) -> dict[str, int]:
    """Return unambiguous runtime sparse-query counts.

    We separate the scores evaluated by the runtime model from the smaller
    certificate support eventually used by the tournament.  This avoids mixing
    action-conditioned atom queries with pair-conditioned delta queries.
    """
    topm = np.asarray(pred.get("top_m_atoms", []), dtype=np.int64).reshape(-1)
    actions = np.asarray(pred.get("queried_actions", []), dtype=np.int64).reshape(-1)
    runtime_pairs = np.asarray(pred.get("runtime_pairs", pred.get("pair_indices", [])), dtype=np.int64)
    runtime_pairs = runtime_pairs.reshape(-1, 2) if runtime_pairs.size else np.zeros((0, 2), dtype=np.int64)
    rival_pairs = np.asarray(pred.get("rival_pair_indices", []), dtype=np.int64)
    rival_pairs = rival_pairs.reshape(-1, 2) if rival_pairs.size else np.zeros((0, 2), dtype=np.int64)

    action_atom = int(pred.get("action_atom_query_count", len(topm) * len(actions)))
    selector_pair_atom = int(pred.get("selector_pair_atom_query_count", len(topm) * len(runtime_pairs)))
    tournament_pair_atom = int(pred.get("tournament_pair_atom_query_count", len(topm) * len(rival_pairs)))
    total = int(action_atom + selector_pair_atom + tournament_pair_atom)

    if selected_atoms is None:
        selected_count = 0
    else:
        selected_count = int(len(np.asarray(selected_atoms, dtype=np.int64).reshape(-1)))
    if len(rival_pairs):
        selected_certificate = int(selected_count * len(rival_pairs))
    else:
        selected_certificate = int(selected_count * len(actions))

    return {
        "proposal_atom_count": int(len(topm)),
        "queried_action_count": int(len(actions)),
        "runtime_pair_count": int(len(runtime_pairs)),
        "tournament_pair_count": int(len(rival_pairs)),
        "action_atom_query_count": action_atom,
        "selector_pair_atom_query_count": selector_pair_atom,
        "tournament_pair_atom_query_count": tournament_pair_atom,
        "sparse_query_count": total,
        "total_sparse_query_count": total,
        "selected_certificate_query_count": selected_certificate,
        "effective_query_count": selected_certificate,
    }


class BDSEPlannerCore:
    def __init__(self, model: Any | None = None, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or load_config()
        self.model = model

    def _rule_score_sparse(self, runtime: RuntimeFeatures, candidates, evidence_bank, atom_indices: np.ndarray, action_indices: np.ndarray, cfg: dict[str, Any] | None = None) -> np.ndarray:
        cfg = cfg or self.cfg
        atom_ids, action_ids, q = compute_query_features_for_pairs(evidence_bank.atoms, candidates, runtime, atom_indices, action_indices, cfg)
        g = np.zeros((evidence_bank.E, candidates.K), dtype=np.float32)
        for row, (ei, a) in enumerate(zip(atom_ids, action_ids)):
            atom = evidence_bank.atoms[int(ei)]
            feat = q[row]
            if atom.family in {"interaction", "reachability_interaction", "precedence"}:
                g[int(ei), int(a)] = max(0.0, 5.0 - float(feat[0]))
            elif atom.type == "red_light":
                g[int(ei), int(a)] = 50.0 * float(feat[7])
            elif atom.type == "drivable_area":
                g[int(ei), int(a)] = float(feat[6])
            elif atom.family in {"kinematic", "dynamic_regularity"}:
                g[int(ei), int(a)] = 0.1 * float(feat[9] + feat[10] + feat[11])
        g[:, ~candidates.valid_mask] = 0.0
        return g

    def _predict_runtime_certificate(self, runtime: RuntimeFeatures, candidates, evidence_bank, stage_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = stage_cfg or self.cfg
        if self.model is not None and hasattr(self.model, "predict_certificate_numpy"):
            return self.model.predict_certificate_numpy(runtime, candidates, evidence_bank, cfg)

        K = candidates.K
        J0 = np.square(candidates.trajectories[:, :, 1]).mean(axis=1).astype(np.float32)
        J0 += -0.05 * candidates.trajectories[:, -1, 0].astype(np.float32)
        J0[~candidates.valid_mask] = np.inf
        proposal_features = np.asarray(evidence_bank.proposal_features, dtype=np.float32)
        if proposal_features.ndim == 2 and proposal_features.shape[1] > 8:
            proposal_logits = 2.0 * proposal_features[:, 0] - proposal_features[:, 8] + proposal_features[:, 10]
        else:
            proposal_logits = np.zeros((evidence_bank.E,), dtype=np.float32)
        runtime_flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
        pairs, pair_weights = build_runtime_pairs_from_base(
            J0,
            candidates.valid_mask,
            runtime_flags,
            L0=int(cfg.get("tournament", {}).get("L_infer", 16)),
            eta0=float(cfg.get("selector", {}).get("eta_pred", 1.0)),
            lambda_near=float(cfg.get("selector", {}).get("lambda_near", 1.0)),
            lambda_safety=float(cfg.get("selector", {}).get("lambda_safety", 2.0)),
            bidirectional_pairs=bool(cfg.get("selector", {}).get("bidirectional_pairs", True)),
            reverse_pair_weight=float(cfg.get("selector", {}).get("reverse_pair_weight", 1.0)),
            pair_cap_multiplier=float(cfg.get("selector", {}).get("runtime_pair_cap_multiplier", 1.0)),
            candidate_trajectories=candidates.trajectories,
            maneuver_ids=candidates.maneuver_ids,
            progress_pair_count=int(cfg.get("selector", {}).get("progress_pair_count", 8)),
            maneuver_pair_count=int(cfg.get("selector", {}).get("maneuver_pair_count", 8)),
        )
        budget = float(cfg.get("evidence", {}).get("budget", 16))
        M = int(cfg.get("selector", {}).get("proposal_top_m", max(2 * int(budget), int(budget) + 1)))
        active = np.asarray(evidence_bank.active_mask, dtype=bool)
        costs = np.asarray(evidence_bank.budget_costs(), dtype=np.float32)
        family_ids = family_ids_from_atoms(evidence_bank.atoms, max_atoms=evidence_bank.E)
        topm, family_budget, hab_diag = select_topm_atoms_hab(
            proposal_logits,
            family_ids,
            active,
            costs,
            budget,
            M,
            family_scores=None,
            free_budget=cfg.get("selector", {}).get("hab_free_budget", None),
            reserve_fraction=float(cfg.get("selector", {}).get("hab_reserve_fraction", 0.2)),
            enabled=bool(cfg.get("selector", {}).get("hab_enabled", True)),
            min_family_slots=cfg.get("selector", {}).get("min_family_topm_slots", None),
        )
        try:
            raw_hard_mask = np.asarray(evidence_bank.hard_mask(), dtype=bool)[: evidence_bank.E]
        except Exception:
            raw_hard_mask = np.zeros((evidence_bank.E,), dtype=bool)
        mandatory_hard_mask = structural_safety_mask(
            raw_hard_mask,
            family_ids,
            active,
            include_feasibility=bool(cfg.get("selector", {}).get("structural_safety_include_feasibility", True)),
        )
        if bool(cfg.get("selector", {}).get("force_hard_topm", True)):
            forced = np.flatnonzero(mandatory_hard_mask)
            if forced.size:
                forced_cap = int(cfg.get("selector", {}).get("max_forced_hard_topm", max(1, M // 2)))
                forced = np.asarray(sorted(forced.tolist(), key=lambda i: (-float(proposal_logits[int(i)]), int(i)))[:forced_cap], dtype=np.int64)
                forced_set = set(forced.tolist())
                non_forced = [int(i) for i in np.asarray(topm, dtype=np.int64).reshape(-1).tolist() if int(i) not in forced_set]
                topm = np.asarray((forced.tolist() + non_forced)[:M], dtype=np.int64)
                hab_diag = dict(hab_diag)
                hab_diag["forced_hard_topm"] = int(forced.size)
        rival_sets = build_rival_sets_from_base(
            J0,
            candidates.valid_mask,
            runtime_flags,
            L_infer=int(cfg.get("tournament", {}).get("L_infer", 16)),
            eta0=float(cfg.get("selector", {}).get("eta_pred", 1.0)),
            candidate_trajectories=candidates.trajectories,
            maneuver_ids=candidates.maneuver_ids,
            progress_rivals=int(cfg.get("selector", {}).get("progress_rivals", 4)),
            maneuver_rivals=int(cfg.get("selector", {}).get("maneuver_rivals", 4)),
        )
        action_set: set[int] = set()
        for a_idx, rivals in enumerate(rival_sets):
            if not bool(candidates.valid_mask[a_idx]) or not rivals:
                continue
            action_set.add(int(a_idx))
            action_set.update(int(r) for r in rivals)
        if action_set:
            action_ids = np.asarray(sorted(action_set), dtype=np.int64)
        else:
            action_ids = np.unique(pairs.reshape(-1)) if len(pairs) else np.flatnonzero(candidates.valid_mask)[: max(1, int(cfg.get("tournament", {}).get("L_infer", 16)))]
        g = self._rule_score_sparse(runtime, candidates, evidence_bank, topm, action_ids, cfg)
        g_var = np.zeros_like(g, dtype=np.float32)
        return {
            "J0": J0,
            "g": g,
            "g_var": g_var,
            "proposal_logits": proposal_logits.astype(np.float32),
            "family_ids": family_ids,
            "family_budget_caps": family_budget.family_caps,
            "family_budgets": family_budget.family_budgets,
            "mandatory_atom_mask": mandatory_hard_mask.astype(bool),
            "mandatory_hard_atoms": np.flatnonzero(mandatory_hard_mask).astype(np.int64),
            "hab_diagnostics": hab_diag,
            "top_m_atoms": topm,
            "queried_actions": np.asarray(action_ids, dtype=np.int64),
            "action_atom_query_count": int(len(topm) * len(action_ids)),
            "selector_pair_atom_query_count": 0,
            "tournament_pair_atom_query_count": 0,
            "runtime_pair_count": int(len(pairs)),
            "tournament_pair_count": 0,
            "queried_pair_count": int(len(topm) * len(action_ids)),
            "runtime_pairs": pairs,
            "runtime_pair_weights": pair_weights,
        }

    def _predict_costs(self, runtime: RuntimeFeatures, candidates, evidence_bank) -> tuple[np.ndarray, np.ndarray]:
        pred = self._predict_runtime_certificate(runtime, candidates, evidence_bank)
        return pred["J0"], pred["g"]

    def _stage_cfg(self, budget: int | None = None, proposal_top_m: int | None = None, L_infer: int | None = None) -> dict[str, Any]:
        cfg = dict(self.cfg)
        cfg["evidence"] = dict(self.cfg.get("evidence", {}))
        cfg["selector"] = dict(self.cfg.get("selector", {}))
        cfg["tournament"] = dict(self.cfg.get("tournament", {}))
        if budget is not None:
            cfg["evidence"]["budget"] = int(budget)
        if proposal_top_m is not None:
            cfg["selector"]["proposal_top_m"] = int(proposal_top_m)
        if L_infer is not None:
            cfg["tournament"]["L_infer"] = int(L_infer)
        return cfg


    def _run_baseline_stage(self, mode: str, pred: dict[str, Any], runtime: RuntimeFeatures, candidates, evidence_bank, stage_cfg: dict[str, Any], atom_active: np.ndarray, family_ids: np.ndarray, runtime_flags: np.ndarray):
        """Internal planner baselines for mechanism ablations.

        These modes intentionally avoid the learned BDSE greedy selector so that
        closed-loop runs can isolate whether gains come from pair-conditioned
        interaction selection rather than simply using more evidence.
        """
        mode = str(mode or "bdse").lower().replace("-", "_")
        J0 = np.asarray(pred["J0"], dtype=np.float32)
        valid = np.asarray(candidates.valid_mask, dtype=bool)
        costs = np.asarray(evidence_bank.budget_costs(), dtype=np.float32)
        budget = float(stage_cfg.get("evidence", {}).get("budget", 16))
        atom_families = [str(getattr(a, "family", "all")) for a in evidence_bank.atoms]
        zero_g = np.zeros((evidence_bank.E, candidates.K), dtype=np.float32)

        def _run_selected_tournament(selected_atoms: list[int] | np.ndarray, *, force_action_sparse: bool = False):
            """Evaluate a baseline-selected evidence subset.

            For paper-facing selector ablations we want the selector to change
            while the downstream BDSE pair-action margin tournament remains the
            same as the method. Set runtime.baseline_pair_tournament=true to
            use neural pair-conditioned margins for the selected baseline atoms.
            Legacy/action-sparse baselines remain available by leaving that flag
            false.
            """
            runtime_cfg = stage_cfg.get("runtime", {}) if isinstance(stage_cfg, dict) else {}
            use_pair_baseline = (
                bool(runtime_cfg.get("baseline_pair_tournament", False))
                and bool(runtime_cfg.get("use_pair_conditioned_margins", stage_cfg.get("model", {}).get("pair_conditioned", True)))
                and not bool(force_action_sparse)
                and ("rival_pair_atom_delta" in pred or "pair_atom_delta" in pred)
                and ("rival_pair_indices" in pred or "pair_indices" in pred)
            )
            if use_pair_baseline:
                tournament_cfg = dict(stage_cfg)
                tournament_cfg["runtime_pair_margin_scale"] = float(pred.get("rival_pair_margin_scale", pred.get("pair_margin_scale", 100.0)))
                result = run_pair_conditioned_tournament(
                    J0,
                    pred.get("rival_pair_atom_delta", pred.get("pair_atom_delta")),
                    pred.get("rival_pair_indices", pred.get("pair_indices")),
                    selected_atoms,
                    valid,
                    runtime_flags,
                    tournament_cfg,
                    pair_atom_variance=pred.get("rival_pair_atom_var", pred.get("pair_atom_var", None)),
                    candidate_trajectories=candidates.trajectories,
                    maneuver_ids=candidates.maneuver_ids,
                )
                result.diagnostics["baseline_pair_tournament"] = True
                return result
            result = run_tournament(
                J0,
                np.asarray(pred.get("g", zero_g), dtype=np.float32),
                selected_atoms,
                valid,
                runtime_flags,
                stage_cfg,
                candidate_trajectories=candidates.trajectories,
                maneuver_ids=candidates.maneuver_ids,
            )
            result.diagnostics["baseline_pair_tournament"] = False
            return result

        if mode in {"oracle", "oracle_budget", "teacher_oracle"}:
            raise RuntimeError("oracle_budget is only available in offline diagnostics where teacher labels are present; use bdse.experiments.diagnostics/evaluate_open_loop oracle metrics.")

        if mode in {"external_policy", "external_score", "external_baseline"}:
            selected_arr = np.asarray(pred.get("external_selected_atoms", pred.get("top_m_atoms", [])), dtype=np.int64).reshape(-1)
            selected: list[int] = []
            spent = 0.0
            for i in selected_arr.tolist():
                if int(i) < 0 or int(i) >= evidence_bank.E or not bool(atom_active[int(i)]):
                    continue
                c = float(costs[int(i)]) if np.isfinite(float(costs[int(i)])) else 1.0
                if spent + c <= budget + 1e-6:
                    selected.append(int(i)); spent += c
            selection = SelectionResult(selected, float(spent), pred.get("runtime_pairs", np.zeros((0, 2), dtype=np.int64)), pred.get("runtime_pair_weights", np.zeros((0,), dtype=np.float32)), {"mode": mode, "spent_budget": float(spent), "external_variant": str(pred.get("external_variant", "unknown"))})
            # External baselines output a final candidate score under the same
            # evidence budget.  The tournament therefore uses J0 directly while
            # selection is retained for query/budget accounting.
            tournament = run_tournament(
                J0, zero_g, selection.selected, valid, runtime_flags, stage_cfg,
                candidate_trajectories=candidates.trajectories, maneuver_ids=candidates.maneuver_ids,
            )
            pred = dict(pred)
            pred["g"] = zero_g
            pred["baseline_mode"] = mode
            pred["baseline_pair_tournament"] = False
            pred["external_spent_budget"] = float(spent)
            return pred, selection, tournament

        if mode in {"base_only", "no_evidence", "no_selector"}:
            selection = SelectionResult([], 0.0, np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), {"mode": mode, "spent_budget": 0.0})
            # Pair-fair mode still uses the pair-action tournament with an empty evidence set.
            tournament = _run_selected_tournament(selection.selected)
            pred = dict(pred)
            pred["g"] = zero_g
            pred["baseline_mode"] = mode
            pred["baseline_pair_tournament"] = bool(tournament.diagnostics.get("baseline_pair_tournament", False))
            return pred, selection, tournament

        if mode in {"dense_full", "full_evidence", "dense_full_evidence"}:
            if self.model is not None and hasattr(self.model, "predict_dense_numpy"):
                dense = self.model.predict_dense_numpy(runtime, candidates, evidence_bank, stage_cfg)
                g_dense = np.asarray(dense["g"], dtype=np.float32)
                J_dense = np.asarray(dense["J0"], dtype=np.float32)
            else:
                g_dense = np.asarray(pred.get("g", zero_g), dtype=np.float32)
                J_dense = J0
            active = np.asarray(evidence_bank.active_mask, dtype=bool)
            selected = np.flatnonzero(active).astype(np.int64).tolist()
            selection = SelectionResult(selected, 0.0, np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), {"mode": mode, "spent_budget": float(costs[active].sum())})
            tournament = run_tournament(J_dense, g_dense, selection.selected, valid, runtime_flags, stage_cfg, candidate_trajectories=candidates.trajectories, maneuver_ids=candidates.maneuver_ids)
            pred = dict(pred)
            pred.update({"J0": J_dense, "g": g_dense, "baseline_mode": mode, "top_m_atoms": np.asarray(selected, dtype=np.int64), "queried_actions": np.flatnonzero(valid).astype(np.int64)})
            return pred, selection, tournament

        if mode in {"hard_safety_only", "hard_only", "safety_only"}:
            mandatory = np.asarray(pred.get("mandatory_atom_mask", np.zeros((evidence_bank.E,), dtype=bool)), dtype=bool)
            order = np.flatnonzero(mandatory & atom_active).astype(np.int64).tolist()
            selected: list[int] = []
            spent = 0.0
            for i in order:
                c = float(costs[int(i)])
                if np.isfinite(c) and spent + c <= budget + 1e-6:
                    selected.append(int(i)); spent += c
            selection = SelectionResult(selected, 0.0, np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), {"mode": mode, "spent_budget": float(spent)})
            tournament = _run_selected_tournament(selection.selected)
            pred = dict(pred); pred["baseline_mode"] = mode; pred["baseline_pair_tournament"] = bool(tournament.diagnostics.get("baseline_pair_tournament", False))
            return pred, selection, tournament

        selector_mode = {
            "random_budget": "random",
            "random": "random",
            "proposal_top": "proposal_top",
            "top_proposal": "proposal_top",
            "interaction_only": "interaction_only",
            "rule_map_only": "rule_map_only",
            "risk_only": "risk_only",
            "diversity": "diversity",
        }.get(mode, None)
        if selector_mode is None:
            raise ValueError(f"Unknown planner.baseline_mode={mode!r}")
        selection = select_by_mode(
            selector_mode,
            J0,
            np.asarray(pred.get("g", zero_g), dtype=np.float32),
            costs,
            valid,
            runtime_flags,
            budget,
            atom_families=atom_families,
            seed=int(stage_cfg.get("seed", 17)),
            atom_active_mask=atom_active,
            proposal_scores=pred.get("proposal_logits", None),
            mandatory_atom_mask=pred.get("mandatory_atom_mask", None),
        )
        tournament = _run_selected_tournament(selection.selected)
        pred = dict(pred); pred["baseline_mode"] = mode; pred["baseline_pair_tournament"] = bool(tournament.diagnostics.get("baseline_pair_tournament", False))
        return pred, selection, tournament

    def _run_certificate_stage(self, runtime: RuntimeFeatures, candidates, evidence_bank, stage_cfg: dict[str, Any]) -> tuple[dict[str, Any], Any, Any, np.ndarray]:
        pred = self._predict_runtime_certificate(runtime, candidates, evidence_bank, stage_cfg)
        J0, g = pred["J0"], pred["g"]
        g_var = pred.get("g_var", None)
        runtime_flags = runtime_safety_flags_from_runtime(runtime, candidates, stage_cfg)
        sel_cfg = stage_cfg.get("selector", {})
        tour_cfg = stage_cfg.get("tournament", {})
        atom_active = np.zeros((evidence_bank.E,), dtype=bool)
        topm = np.asarray(pred.get("top_m_atoms", np.flatnonzero(evidence_bank.active_mask)), dtype=np.int64)
        atom_active[topm[(topm >= 0) & (topm < evidence_bank.E)]] = True
        atom_active &= evidence_bank.active_mask
        family_ids = np.asarray(pred.get("family_ids", family_ids_from_atoms(evidence_bank.atoms, max_atoms=evidence_bank.E)), dtype=np.int64)
        family_caps = pred.get("family_budget_caps", None)
        baseline_mode = str(stage_cfg.get("planner", {}).get("baseline_mode", "bdse")).lower().replace("-", "_")
        if baseline_mode not in {"", "bdse", "pair_conditioned", "bdse_pair_conditioned"}:
            pred_b, selection_b, tournament_b = self._run_baseline_stage(baseline_mode, pred, runtime, candidates, evidence_bank, stage_cfg, atom_active, family_ids, runtime_flags)
            return pred_b, selection_b, tournament_b, atom_active
        use_pair_conditioned = bool(stage_cfg.get("runtime", {}).get("use_pair_conditioned_margins", stage_cfg.get("model", {}).get("pair_conditioned", True)))
        if use_pair_conditioned and "pair_atom_delta" in pred and "pair_indices" in pred:
            selection = runtime_greedy_selector_pair_conditioned(
                J0,
                pred["pair_atom_delta"],
                pred["pair_indices"],
                pred.get("runtime_pair_weights", np.ones((np.asarray(pred["pair_indices"]).reshape(-1, 2).shape[0],), dtype=np.float32)),
                evidence_bank.budget_costs(),
                candidates.valid_mask,
                runtime_flags,
                budget=float(stage_cfg.get("evidence", {}).get("budget", 16)),
                gamma_max=float(sel_cfg.get("normalized_gamma_max", 5.0) if bool(stage_cfg.get("model", {}).get("pair_margin_normalized", True)) else sel_cfg.get("gamma_max_default", 100.0)),
                eta_pred=float(sel_cfg.get("normalized_eta_pred", 0.1) if bool(stage_cfg.get("model", {}).get("pair_margin_normalized", True)) else sel_cfg.get("eta_pred", 1.0)),
                atom_active_mask=atom_active,
                pair_atom_variance=pred.get("pair_atom_var", None),
                beta_uncertainty=float(tour_cfg.get("beta_uncertainty", 0.0)),
                epsilon_cal=float(tour_cfg.get("epsilon_cal", stage_cfg.get("calibration", {}).get("epsilon_cal", 0.0))),
                lambda_info=float(sel_cfg.get("lambda_info", 0.0)),
                prior_atom_variance=sel_cfg.get("unqueried_atom_variance", None),
                family_ids=family_ids,
                family_budget_caps=family_caps,
                mandatory_atom_mask=pred.get("mandatory_atom_mask", None),
                mandatory_quota=int(sel_cfg.get("mandatory_hard_quota", 0)),
                min_selected_atoms=int(sel_cfg.get("min_selected_atoms", 0)),
                force_fill_budget=bool(sel_cfg.get("force_fill_budget", False)),
                normalize_margins=bool(stage_cfg.get("model", {}).get("pair_margin_normalized", True)),
                margin_scale=float(pred.get("pair_margin_scale", 100.0)),
                proposal_scores=pred.get("proposal_logits", None),
                proposal_fill_weight=float(sel_cfg.get("proposal_fill_weight", 0.25)),
                prioritize_mandatory_fill=bool(sel_cfg.get("prioritize_mandatory_fill", True)),
            )
            tournament_cfg = dict(stage_cfg)
            tournament_cfg["runtime_pair_margin_scale"] = float(pred.get("rival_pair_margin_scale", pred.get("pair_margin_scale", 100.0)))
            tournament = run_pair_conditioned_tournament(
                J0,
                pred.get("rival_pair_atom_delta", pred["pair_atom_delta"]),
                pred.get("rival_pair_indices", pred["pair_indices"]),
                selection.selected,
                candidates.valid_mask,
                runtime_flags,
                tournament_cfg,
                pair_atom_variance=pred.get("rival_pair_atom_var", pred.get("pair_atom_var", None)),
                candidate_trajectories=candidates.trajectories,
                maneuver_ids=candidates.maneuver_ids,
            )
        else:
            selection = runtime_greedy_selector(
                J0, g, evidence_bank.budget_costs(), candidates.valid_mask, runtime_flags,
                budget=float(stage_cfg.get("evidence", {}).get("budget", 16)),
                L_infer=int(tour_cfg.get("L_infer", 16)),
                gamma_max=float(sel_cfg.get("gamma_max_default", 100.0)),
                eta_pred=float(sel_cfg.get("eta_pred", 1.0)),
                lambda_near=float(sel_cfg.get("lambda_near", 1.0)),
                lambda_safety=float(sel_cfg.get("lambda_safety", 2.0)),
                atom_active_mask=atom_active,
                predicted_atom_variance=g_var,
                beta_uncertainty=float(tour_cfg.get("beta_uncertainty", 0.0)),
                epsilon_cal=float(tour_cfg.get("epsilon_cal", stage_cfg.get("calibration", {}).get("epsilon_cal", 0.0))),
                lambda_info=float(sel_cfg.get("lambda_info", 0.0)),
                prior_atom_variance=sel_cfg.get("unqueried_atom_variance", None),
                family_ids=family_ids,
                family_budget_caps=family_caps,
                mandatory_atom_mask=pred.get("mandatory_atom_mask", None),
                mandatory_quota=int(sel_cfg.get("mandatory_hard_quota", 0)),
                min_selected_atoms=int(sel_cfg.get("min_selected_atoms", 0)),
                force_fill_budget=bool(sel_cfg.get("force_fill_budget", False)),
                prioritize_mandatory_fill=bool(sel_cfg.get("prioritize_mandatory_fill", True)),
                bidirectional_pairs=bool(sel_cfg.get("bidirectional_pairs", True)),
                reverse_pair_weight=float(sel_cfg.get("reverse_pair_weight", 1.0)),
                pair_cap_multiplier=float(sel_cfg.get("runtime_pair_cap_multiplier", 1.0)),
                candidate_trajectories=candidates.trajectories,
                maneuver_ids=candidates.maneuver_ids,
                progress_pair_count=int(sel_cfg.get("progress_pair_count", 0)),
                maneuver_pair_count=int(sel_cfg.get("maneuver_pair_count", 0)),
            )
            sigma = selected_pair_sigma_from_action_variance(g_var, selection.selected, candidates.valid_mask)
            tournament = run_tournament(
                J0, g, selection.selected, candidates.valid_mask, runtime_flags, stage_cfg, sigma=sigma,
                candidate_trajectories=candidates.trajectories,
                maneuver_ids=candidates.maneuver_ids,
            )
        return pred, selection, tournament, atom_active

    def _needs_fallback(self, tournament, candidates, cfg: dict[str, Any]) -> bool:
        fcfg = cfg.get("fallback", {})
        if not bool(fcfg.get("enabled", True)):
            return False
        if float(tournament.diagnostics.get("delta_hat_B", 0.0)) < float(fcfg.get("tau_delta", 0.1)):
            return True
        if bool(tournament.diagnostics.get("selected_action_safety_flag", False)):
            return True
        safety_thr = float(fcfg.get("safety_lcb_min", 0.0))
        if float(tournament.diagnostics.get("safety_lcb_min", float("inf"))) < safety_thr:
            return True
        return False

    def plan_from_runtime(self, runtime: RuntimeFeatures) -> tuple[int, np.ndarray, dict[str, Any]]:
        profile_enabled = os.environ.get("BDSE_PROFILE_CLOSED_LOOP", "0").lower() in {"1", "true", "yes", "on"}
        timing_core: dict[str, float] = {}
        t = time.perf_counter()
        candidates = generate_candidate_bank(runtime, self.cfg)
        if profile_enabled:
            timing_core["candidate_generation_s"] = float(time.perf_counter() - t)
        if bool(self.cfg.get("preprocess", {}).get("candidate_aware_agent_selection", False)) and not bool(
            self.cfg.get("runtime", {}).get("skip_candidate_aware_agent_selection", False)
        ):
            from bdse.data.feature_builder import resort_runtime_agents_for_candidates

            t = time.perf_counter()
            runtime2 = resort_runtime_agents_for_candidates(runtime, candidates, self.cfg)
            if runtime2 is not runtime:
                runtime = runtime2
                candidates = generate_candidate_bank(runtime, self.cfg)
            if profile_enabled:
                timing_core["candidate_aware_agent_resort_s"] = float(time.perf_counter() - t)
        t = time.perf_counter()
        evidence_bank = enumerate_evidence_atoms(runtime, candidates, self.cfg)
        if profile_enabled:
            timing_core["evidence_enumeration_s"] = float(time.perf_counter() - t)
        base_budget = int(self.cfg.get("evidence", {}).get("budget", 16))
        base_M = int(self.cfg.get("selector", {}).get("proposal_top_m", max(2 * base_budget, base_budget + 1)))
        base_L = int(self.cfg.get("tournament", {}).get("L_infer", 16))
        stages: list[tuple[str, dict[str, Any]]] = [("base", self._stage_cfg(base_budget, base_M, base_L))]
        fcfg = self.cfg.get("fallback", {})
        if bool(fcfg.get("enabled", True)):
            L_stages = list(fcfg.get("rival_stages", [base_L, min(31, max(candidates.K - 1, 1))]))
            B_stages = list(fcfg.get("budget_stages", [base_budget, min(int(self.cfg.get("evidence", {}).get("max_atoms", 128)), max(base_budget * 2, base_budget + 1))]))
            max_extra_stages = fcfg.get("max_additional_stages", None)
            for L in L_stages:
                for B in B_stages:
                    M = int(max(int(self.cfg.get("selector", {}).get("proposal_top_m", base_M)), min(int(self.cfg.get("evidence", {}).get("max_atoms", 128)), int(float(fcfg.get("proposal_multiplier", 3.0)) * int(B)))))
                    name = f"fallback_L{int(L)}_B{int(B)}_M{int(M)}"
                    cfg_stage = self._stage_cfg(int(B), int(M), int(L))
                    if name != "base":
                        stages.append((name, cfg_stage))
                        if max_extra_stages is not None and len(stages) - 1 >= int(max_extra_stages):
                            break
                if max_extra_stages is not None and len(stages) - 1 >= int(max_extra_stages):
                    break
        best = None
        stage_records = []
        triggered = False
        for idx, (stage_name, cfg_stage) in enumerate(stages):
            t_stage = time.perf_counter()
            pred, selection, tournament, atom_active = self._run_certificate_stage(runtime, candidates, evidence_bank, cfg_stage)
            stage_elapsed = float(time.perf_counter() - t_stage)
            if profile_enabled:
                timing_core["certificate_stages_s"] = timing_core.get("certificate_stages_s", 0.0) + stage_elapsed
            qdiag = runtime_query_diagnostics(pred, selection.selected)
            qdiag.update({k: v for k, v in getattr(tournament, "diagnostics", {}).items() if k in {"normalized_margins", "margin_scale", "epsilon_cal", "pair_conditioned"}})
            stage_records.append({
                "stage": stage_name,
                "action": int(tournament.action_index),
                "delta_hat_B": float(tournament.diagnostics.get("delta_hat_B", 0.0)),
                "safety_lcb_min": float(tournament.diagnostics.get("safety_lcb_min", float("inf"))),
                "selected_atoms": list(map(int, selection.selected)),
                "top_m_atoms": list(map(int, np.asarray(pred.get("top_m_atoms", []), dtype=np.int64).tolist())),
                "queried_actions": list(map(int, np.asarray(pred.get("queried_actions", []), dtype=np.int64).tolist())),
                **qdiag,
                "hab": pred.get("hab_diagnostics", {}),
                **({"model_timing": pred.get("model_timing", {})} if profile_enabled else {}),
                **({"stage_elapsed_s": stage_elapsed} if profile_enabled else {}),
            })
            best = (stage_name, cfg_stage, pred, selection, tournament, atom_active)
            if idx == 0 and not self._needs_fallback(tournament, candidates, cfg_stage):
                break
            triggered = True
            if idx > 0 and not self._needs_fallback(tournament, candidates, cfg_stage):
                break
        assert best is not None
        stage_name, cfg_stage, pred, selection, tournament, atom_active = best
        action = int(tournament.action_index)
        t_post = time.perf_counter()
        runtime_flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg_stage)
        if profile_enabled:
            timing_core["final_safety_flags_s"] = float(time.perf_counter() - t_post)
        if triggered and bool(fcfg.get("rule_rerank_top_k", 5)):
            from bdse.planner.fallback import rule_based_runtime_scores, conservative_fallback_action
            t_rule = time.perf_counter()
            top_k = int(fcfg.get("rule_rerank_top_k", 5))
            top_actions = [int(a) for a in np.argsort(-tournament.scores)[:top_k] if candidates.valid_mask[int(a)]]
            rule_cost = rule_based_runtime_scores(runtime, candidates, cfg_stage, safety_flags=runtime_flags)
            safe_top = [a for a in top_actions if not runtime_flags[a]]
            if safe_top:
                best_rule = min(safe_top, key=lambda a: (float(rule_cost[a]), a))
                if float(rule_cost[best_rule]) + float(fcfg.get("rule_switch_margin", 0.0)) < float(rule_cost[action]) or runtime_flags[action]:
                    action = int(best_rule)
                    stage_name = stage_name + "+rule_rerank"
            elif runtime_flags[action]:
                action = int(conservative_fallback_action(candidates))
                stage_name = stage_name + "+conservative"
            if profile_enabled:
                timing_core["rule_rerank_s"] = float(time.perf_counter() - t_rule)
        trajectory = candidates.trajectories[action]
        qdiag = runtime_query_diagnostics(pred, selection.selected)
        qdiag.update({k: v for k, v in getattr(tournament, "diagnostics", {}).items() if k in {"normalized_margins", "margin_scale", "epsilon_cal", "pair_conditioned"}})
        diagnostics = {
            "action_index": action,
            "selected_atoms": selection.selected,
            "proposal_top_m_atoms": list(map(int, np.asarray(pred.get("top_m_atoms", []), dtype=np.int64).tolist())),
            "queried_actions": list(map(int, np.asarray(pred.get("queried_actions", []), dtype=np.int64).tolist())),
            **qdiag,
            "hab": pred.get("hab_diagnostics", {}),
            **({"model_timing": pred.get("model_timing", {})} if profile_enabled else {}),
            "selector": selection.diagnostics,
            "tournament": tournament.diagnostics,
            "fallback_stage": stage_name,
            "fallback_triggered": bool(triggered),
            "fallback_stage_records": stage_records,
            **({"timing_core": timing_core} if profile_enabled else {}),
        }
        return action, trajectory, diagnostics



def _json_safe(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class BDSEnuPlanPlanner(AbstractPlanner):
    requires_scenario: bool = False

    def __init__(
        self,
        model: Any | None = None,
        cfg: dict[str, Any] | None = None,
        checkpoint: str | None = None,
        config_path: str | None = None,
        device: str = "auto",
    ):
        cfg = cfg or load_config(config_path)
        device = _maybe_shard_planner_device(device)
        self.device = resolve_torch_device(device, context="BDSEnuPlanPlanner")
        configure_torch_for_device(self.device)
        external_enabled = bool((cfg.get("external_baseline", {}) or {}).get("enabled", False))
        if model is None and (checkpoint or external_enabled):
            # Load checkpoint tensors on CPU first to avoid accidentally putting
            # optimizer/RNG payloads from full training checkpoints on GPU.  Then
            # move only the model module to the resolved planner device.
            from bdse.external_baselines.model_factory import load_model_for_config

            model = load_model_for_config(checkpoint, cfg, self.device)
        elif model is not None and hasattr(model, "to"):
            model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()
        print(f"BDSEnuPlanPlanner device: {self.device}")
        self.core = BDSEPlannerCore(model=model, cfg=cfg)
        self._name = "BDSEPlanner"
        self._cached_local_trajectory = None
        self._cached_action_index = 0
        self._cached_replan_iteration_index = None
        self._cached_replan_time_s = None
        self._cached_replan_ego_pose = None

    def name(self) -> str:
        return self._name

    def observation_type(self):
        try:
            from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

            return DetectionsTracks
        except Exception:
            return None

    def initialize(self, initialization: Any) -> None:
        self.initialization = initialization

    def _current_iteration_index(self, current_input: Any) -> int:
        iteration = getattr(current_input, "iteration", None)
        try:
            return int(getattr(iteration, "index", -1)) if iteration is not None else -1
        except Exception:
            return -1

    def _write_closed_loop_diag(self, current_input: Any, action: int, diagnostics: dict[str, Any]) -> None:
        diag_path = os.environ.get("BDSE_CLOSED_LOOP_DIAG", "")
        if not diag_path:
            return
        try:
            iteration = getattr(current_input, "iteration", None)
            row = {
                "planner": self._name,
                "iteration_index": int(getattr(iteration, "index", -1)) if iteration is not None else -1,
                "time_s": float(getattr(iteration, "time_s", 0.0)) if iteration is not None else 0.0,
                "action_index": int(action),
                "diagnostics": _json_safe(diagnostics),
            }
            path = Path(diag_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception:
            pass

    def _planner_replan_interval_ticks(self) -> int:
        pcfg = self.core.cfg.get("planner", {}) if isinstance(self.core.cfg, dict) else {}
        # Keep exact old behavior unless the fast closed-loop config opts in.
        if os.environ.get("BDSE_FORCE_REPLAN_EVERY_TICK", "0").lower() in {"1", "true", "yes", "on"}:
            return 1
        env_value = os.environ.get("BDSE_REPLAN_INTERVAL_TICKS", "").strip()
        if env_value:
            try:
                return max(1, int(env_value))
            except Exception:
                pass
        try:
            return max(1, int(pcfg.get("replan_interval_ticks", 1)))
        except Exception:
            return 1

    def _planner_cache_cfg(self) -> dict[str, Any]:
        pcfg = self.core.cfg.get("planner", {}) if isinstance(self.core.cfg, dict) else {}
        return pcfg if isinstance(pcfg, dict) else {}

    def _current_ego_pose_time(self, current_input: Any) -> tuple[tuple[float, float, float] | None, float | None]:
        """Best-effort extraction of current rear-axle global pose and simulation time."""
        try:
            history = getattr(current_input, "history", None)
            ego_states = getattr(history, "ego_states", None)
            if ego_states is None or len(ego_states) == 0:
                return None, None
            state = ego_states[-1]
            rear = getattr(state, "rear_axle", state)
            pose = (float(getattr(rear, "x")), float(getattr(rear, "y")), float(getattr(rear, "heading")))
            tp = getattr(state, "time_point", None)
            time_s = None
            if tp is not None and hasattr(tp, "time_s"):
                time_s = float(getattr(tp, "time_s"))
            elif tp is not None and hasattr(tp, "time_us"):
                time_s = float(getattr(tp, "time_us")) * 1e-6
            else:
                iteration = getattr(current_input, "iteration", None)
                if iteration is not None and hasattr(iteration, "time_s"):
                    time_s = float(getattr(iteration, "time_s"))
            return pose, time_s
        except Exception:
            return None, None

    def _can_reuse_cached_plan(self, current_input: Any) -> bool:
        if getattr(self, "_cached_local_trajectory", None) is None:
            return False
        interval = self._planner_replan_interval_ticks()
        if interval <= 1:
            return False
        idx = self._current_iteration_index(current_input)
        last_idx = getattr(self, "_cached_replan_iteration_index", None)
        if last_idx is None or idx < 0 or idx <= int(last_idx):
            # A reset/non-monotonic index usually means a new scenario/simulation.
            return False
        if (idx - int(last_idx)) >= interval:
            return False

        # Optional guards make longer replan intervals safer: reuse the cached
        # rollout only while the ego has not moved/rotated too far from the state
        # where the expensive BDSE certificate was computed.  These guards are
        # disabled only if explicitly set to <=0 in the config/env.
        pcfg = self._planner_cache_cfg()
        max_dist = float(os.environ.get("BDSE_REPLAN_CACHE_MAX_DISTANCE_M", pcfg.get("replan_cache_max_distance_m", 8.0)))
        max_heading = float(os.environ.get("BDSE_REPLAN_CACHE_MAX_HEADING_RAD", pcfg.get("replan_cache_max_heading_rad", 0.8)))
        max_elapsed = float(os.environ.get("BDSE_REPLAN_CACHE_MAX_ELAPSED_S", pcfg.get("replan_cache_max_elapsed_s", 2.5)))
        if max_dist > 0.0 or max_heading > 0.0 or max_elapsed > 0.0:
            pose, time_s = self._current_ego_pose_time(current_input)
            last_pose = getattr(self, "_cached_replan_ego_pose", None)
            last_time = getattr(self, "_cached_replan_time_s", None)
            if pose is not None and last_pose is not None:
                dx = float(pose[0]) - float(last_pose[0])
                dy = float(pose[1]) - float(last_pose[1])
                if max_dist > 0.0 and (dx * dx + dy * dy) ** 0.5 > max_dist:
                    return False
                if max_heading > 0.0 and abs(float(angle_wrap(float(pose[2]) - float(last_pose[2])))) > max_heading:
                    return False
            if time_s is not None and last_time is not None:
                if max_elapsed > 0.0 and float(time_s) - float(last_time) > max_elapsed:
                    return False
        return True

    def compute_planner_trajectory(self, current_input: Any):
        """Compute a nuPlan-compatible trajectory from runtime-only inputs.

        When ``planner.replan_interval_ticks > 1`` the expensive BDSE core is
        evaluated only every N simulator ticks; intermediate ticks reuse the last
        local rollout and convert it relative to the *current* ego state.  This is
        a standard closed-loop evaluation speedup for slow research planners and
        keeps the old every-tick behavior by default.
        """
        profile_enabled = os.environ.get("BDSE_PROFILE_CLOSED_LOOP", "0").lower() in {"1", "true", "yes", "on"}
        t0 = time.perf_counter()
        idx = self._current_iteration_index(current_input)

        if self._can_reuse_cached_plan(current_input):
            trajectory = np.asarray(getattr(self, "_cached_local_trajectory"), dtype=np.float32)
            action = int(getattr(self, "_cached_action_index", 0))
            t1 = time.perf_counter()
            out_traj = self._to_nuplan_trajectory(trajectory, current_input)
            t2 = time.perf_counter()
            diagnostics: dict[str, Any] = {
                "action_index": action,
                "cached_plan": True,
                "reuse_from_iteration_index": int(getattr(self, "_cached_replan_iteration_index", -1)),
                "replan_interval_ticks": self._planner_replan_interval_ticks(),
                "cache_guarded": True,
            }
            if profile_enabled:
                diagnostics["timing"] = {
                    "runtime_from_planner_input_s": 0.0,
                    "core_plan_s": 0.0,
                    "to_nuplan_trajectory_s": float(t2 - t1),
                    "compute_planner_trajectory_total_s": float(t2 - t0),
                }
                diagnostics["timing_core"] = {"cached_plan_s": float(t2 - t0)}
            self._write_closed_loop_diag(current_input, action, diagnostics)
            return out_traj

        t_runtime = time.perf_counter()
        runtime = self._runtime_from_planner_input(current_input)
        t1 = time.perf_counter()
        action, trajectory, diagnostics = self.core.plan_from_runtime(runtime)
        t2 = time.perf_counter()
        out_traj = self._to_nuplan_trajectory(trajectory, current_input)
        t3 = time.perf_counter()
        # Cache the local rollout, not the absolute nuPlan trajectory, so reuse
        # remains anchored to the current ego state at the next simulator tick.
        self._cached_local_trajectory = np.asarray(trajectory, dtype=np.float32).copy()
        self._cached_action_index = int(action)
        self._cached_replan_iteration_index = int(idx)
        self._cached_replan_ego_pose, self._cached_replan_time_s = self._current_ego_pose_time(current_input)
        if profile_enabled:
            diagnostics = dict(diagnostics)
            timing = dict(diagnostics.get("timing", {}))
            timing.update({
                "runtime_from_planner_input_s": float(t1 - t_runtime),
                "core_plan_s": float(t2 - t1),
                "to_nuplan_trajectory_s": float(t3 - t2),
                "compute_planner_trajectory_total_s": float(t3 - t0),
            })
            diagnostics["timing"] = timing
            diagnostics["cached_plan"] = False
            diagnostics["replan_interval_ticks"] = self._planner_replan_interval_ticks()
        self._write_closed_loop_diag(current_input, int(action), diagnostics)
        return out_traj

    def _runtime_from_planner_input(self, current_input: Any) -> RuntimeFeatures:
        if isinstance(current_input, RuntimeFeatures):
            return current_input
        if hasattr(current_input, "runtime_features"):
            return current_input.runtime_features
        return build_runtime_features_from_planner_input(
            current_input=current_input,
            initialization=getattr(self, "initialization", None),
            cfg=self.core.cfg,
        )

    def _to_nuplan_trajectory(self, trajectory: np.ndarray, current_input: Any):
        """Convert a BDSE ego-local candidate rollout to nuPlan's trajectory type.

        The preferred path delegates pose-to-state conversion to nuPlan's ML
        planner utility. That utility applies the same relative-to-absolute pose
        transform and velocity/acceleration estimation used by nuPlan learned
        planners, which avoids frame mistakes in closed-loop comfort metrics.
        A manual fallback remains for tests and older/local nuPlan builds.
        """
        traj_arr = np.asarray(trajectory, dtype=np.float32)
        if traj_arr.ndim != 2 or traj_arr.shape[1] < 5 or len(traj_arr) == 0:
            raise ValueError(f"Expected BDSE trajectory with shape [T,5+], got {traj_arr.shape}")
        try:
            InterpolatedTrajectory = _cached_import("nuplan.planning.simulation.trajectory.interpolated_trajectory", "InterpolatedTrajectory")
            history = getattr(current_input, "history", None)
            ego_states = getattr(history, "ego_states", None)
            if ego_states is not None and len(ego_states) >= 2:
                try:
                    transform_predictions_to_states = _cached_import(
                        "nuplan.planning.simulation.planner.ml_planner.transform_utils",
                        "transform_predictions_to_states",
                    )
                    times = np.asarray(traj_arr[:, 4], dtype=np.float32)
                    if len(times) > 1:
                        diffs = np.diff(times)
                        diffs = diffs[np.isfinite(diffs) & (diffs > 1e-4)]
                        step_interval = float(np.median(diffs)) if diffs.size else float(times[0])
                    else:
                        step_interval = float(times[0]) if float(times[0]) > 1e-4 else 0.1
                    future_horizon = float(times[-1])
                    expected_steps = int(round(future_horizon / max(step_interval, 1e-4)))
                    poses = traj_arr[:, :3]
                    # nuPlan's helper constructs fixed timesteps from horizon and
                    # interval.  If a non-uniform trajectory slipped through, trim or
                    # pad poses to the fixed-timestep count rather than producing an
                    # inconsistent state/time list.
                    if expected_steps > 0 and expected_steps != len(poses):
                        if expected_steps < len(poses):
                            poses = poses[:expected_steps]
                        else:
                            pad = np.repeat(poses[-1:], expected_steps - len(poses), axis=0)
                            poses = np.concatenate([poses, pad], axis=0)
                    states = transform_predictions_to_states(
                        predicted_poses=poses.astype(np.float32),
                        ego_history=ego_states,
                        future_horizon=future_horizon,
                        step_interval=step_interval,
                        include_ego_state=True,
                    )
                    return InterpolatedTrajectory(states)
                except Exception:
                    # Fall through to the explicit implementation below. This keeps
                    # the planner usable with older devkit revisions whose helper
                    # signature differs while still surfacing conversion failures at
                    # the final boundary if manual conversion also fails.
                    pass

            EgoState = _cached_import("nuplan.common.actor_state.ego_state", "EgoState")
            TimePoint = _cached_import("nuplan.common.actor_state.state_representation", "TimePoint")
            StateSE2 = _cached_import("nuplan.common.actor_state.state_representation", "StateSE2")
            StateVector2D = _cached_import("nuplan.common.actor_state.state_representation", "StateVector2D")
            last = ego_states[-1] if ego_states is not None and len(ego_states) else None
            start_us = int(getattr(getattr(last, "time_point", None), "time_us", 0))
            rear = getattr(last, "rear_axle", last)
            ox = float(getattr(rear, "x", 0.0))
            oy = float(getattr(rear, "y", 0.0))
            oyaw = float(getattr(rear, "heading", 0.0))
            c = float(np.cos(oyaw))
            s = float(np.sin(oyaw))
            states = []
            if last is not None:
                states.append(last)
            times = np.asarray(traj_arr[:, 4], dtype=np.float32)
            speeds = np.asarray(traj_arr[:, 3], dtype=np.float32)
            if len(times) > 1:
                accel_lon = np.gradient(speeds, times, edge_order=1).astype(np.float32)
            else:
                accel_lon = np.asarray([0.0], dtype=np.float32)
            vehicle_params = getattr(getattr(self, "initialization", None), "vehicle_parameters", None)
            if vehicle_params is None and last is not None:
                vehicle_params = getattr(last, "car_footprint", None)
                vehicle_params = getattr(vehicle_params, "vehicle_parameters", None)
            if vehicle_params is None:
                try:
                    vehicle_params = _cached_import("nuplan.common.actor_state.vehicle_parameters", "get_pacifica_parameters")()
                except Exception:
                    vehicle_params = None
            for k, row in enumerate(traj_arr):
                # Candidate trajectories are represented in the ego-local frame.
                # nuPlan expects global SE2 poses, but DynamicCarState rear-axle
                # velocity/acceleration are expressed in the ego-body frame.
                lx, ly = float(row[0]), float(row[1])
                gx = ox + c * lx - s * ly
                gy = oy + s * lx + c * ly
                gyaw = float(angle_wrap(oyaw + float(row[2])))
                t_us = start_us + int(float(row[4]) * 1e6)
                state = EgoState.build_from_rear_axle(
                    rear_axle_pose=StateSE2(gx, gy, gyaw),
                    rear_axle_velocity_2d=StateVector2D(float(max(speeds[k], 0.0)), 0.0),
                    rear_axle_acceleration_2d=StateVector2D(float(accel_lon[k]), 0.0),
                    tire_steering_angle=0.0,
                    time_point=TimePoint(t_us),
                    vehicle_parameters=vehicle_params,
                    is_in_auto_mode=True,
                )
                states.append(state)
            return InterpolatedTrajectory(states)
        except ImportError:
            # Unit-test / non-nuPlan environments may not install nuPlan. In a real
            # nuPlan simulation this branch is not taken; callers can still inspect
            # the local-frame trajectory array.
            return trajectory
        except Exception as exc:
            raise RuntimeError(f"Failed to convert BDSE local trajectory to nuPlan InterpolatedTrajectory: {exc}") from exc
