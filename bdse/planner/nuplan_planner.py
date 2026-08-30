from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import atexit
import hashlib
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
from bdse.planner.fallback import runtime_safety_cache_scope, runtime_safety_diagnostics, runtime_safety_flags_from_runtime, runtime_safety_flag_components, runtime_risk_scores
from bdse.planner.pair_screen import build_runtime_pairs_from_base, build_rival_sets_from_base, restrict_pairs_to_viability_frontier
from bdse.planner.selector import (
    finalize_runtime_topm_policy,
    SelectionResult,
    runtime_greedy_selector,
    runtime_greedy_selector_pair_conditioned,
    select_by_mode,
    restrict_topm_to_decision_evidence,
    structural_safety_mask,
)
from bdse.planner.tournament import (
    run_tournament,
    run_pair_conditioned_tournament,
    selected_pair_sigma_from_action_variance,
    TournamentResult,
    _trajectory_utility_cost_np,
)
from bdse.planner.frontier_contrast_rebinding import frontier_contrast_rebind
from bdse.planner.full_bank_capacity_probe import full_bank_capacity_probe


_PLANNER_DEVICE_LOCK = threading.Lock()
_PLANNER_DEVICE_COUNTER = itertools.count()
_NUPLAN_IMPORT_CACHE: dict[str, Any] = {}
_SHARED_MODEL_CACHE_LOCK = threading.Lock()
_SHARED_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_DEVICE_INFERENCE_LOCKS: dict[str, threading.RLock] = {}
_CLOSED_LOOP_PROFILE_LOCK = threading.Lock()
_CLOSED_LOOP_PROFILE: dict[str, dict[str, float]] = {}
_CLOSED_LOOP_PROFILE_REGISTERED = False
_DIAG_WRITER_LOCK = threading.Lock()
_DIAG_WRITERS: dict[str, Any] = {}
_DIAG_WRITERS_REGISTERED = False


def _close_diag_writers() -> None:
    with _DIAG_WRITER_LOCK:
        for handle in list(_DIAG_WRITERS.values()):
            try:
                handle.flush(); handle.close()
            except Exception:
                pass
        _DIAG_WRITERS.clear()


def _append_diag_line(path: Path, text: str) -> None:
    """Append a diagnostic line without reopening the file every planner tick."""
    global _DIAG_WRITERS_REGISTERED
    key = str(path.resolve())
    with _DIAG_WRITER_LOCK:
        handle = _DIAG_WRITERS.get(key)
        if handle is None or getattr(handle, "closed", False):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8", buffering=1)
            _DIAG_WRITERS[key] = handle
        if not _DIAG_WRITERS_REGISTERED:
            atexit.register(_close_diag_writers)
            _DIAG_WRITERS_REGISTERED = True
        handle.write(text + "\n")


def _env_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _v50_live_candidate_identity(candidates: Any, action: int) -> dict[str, Any]:
    """Return an auditable *live-state* identity for one candidate slot.

    Candidate ``action`` integers are local indices into a bank that is rebuilt
    and state-dependently pruned on every planner tick.  They are therefore not
    stable proposal identifiers across two different planner states.  V50 only
    needs identity across CONTROL/TREATMENT at the *same* pre-intervention live
    state.  A quantized trajectory fingerprint plus the candidate semantics is
    a much stronger invariant for that paired comparison than comparing the
    local slot to an offline V49 slot number.

    The fingerprint is instrumentation only and never enters ranking, Q/P/E, or
    the deployed action decision.
    """
    a = int(action)
    if a < 0 or a >= int(getattr(candidates, "K", len(candidates.trajectories))):
        raise ValueError(f"V50 live proposal action out of candidate-bank range: {a}")
    traj = np.asarray(candidates.trajectories[a], dtype=np.float32)
    if not np.isfinite(traj).all():
        raise ValueError(f"V50 live proposal trajectory is non-finite for action {a}")
    # 1e-4 quantization is far below planning-scale geometry but avoids making
    # provenance depend on irrelevant last-bit differences between processes.
    quantized = np.round(traj.astype(np.float64), 4).astype(np.float32)
    h = hashlib.sha256()
    h.update(str(tuple(int(x) for x in quantized.shape)).encode("utf-8"))
    h.update(quantized.tobytes(order="C"))
    maneuver_id = int(np.asarray(candidates.maneuver_ids)[a])
    h.update(str(maneuver_id).encode("utf-8"))
    meta = dict(candidates.metadata[a] or {})
    theta = dict(candidates.theta[a] or {})
    return {
        "v50_live_proposal_fingerprint": h.hexdigest(),
        "v50_live_proposal_maneuver_id": maneuver_id,
        "v50_live_proposal_pool_original_index": int(meta.get("pool_original_index", -1)),
        "v50_live_proposal_maneuver": str(meta.get("maneuver", "")),
        "v50_live_proposal_theta": json.dumps(theta, sort_keys=True, default=str, separators=(",", ":")),
    }


def _v50_prefixed_identity(base: dict[str, Any], prefix: str) -> dict[str, Any]:
    mapping = {
        "v50_live_proposal_fingerprint": f"{prefix}_fingerprint",
        "v50_live_proposal_maneuver_id": f"{prefix}_maneuver_id",
        "v50_live_proposal_pool_original_index": f"{prefix}_pool_original_index",
        "v50_live_proposal_maneuver": f"{prefix}_maneuver",
        "v50_live_proposal_theta": f"{prefix}_theta",
    }
    return {mapping[k]: v for k, v in base.items()}


def _v50_prefixed_candidate_identity(candidates: Any, action: int, prefix: str) -> dict[str, Any]:
    return _v50_prefixed_identity(_v50_live_candidate_identity(candidates, action), prefix)


def _device_inference_lock(device: Any) -> threading.RLock:
    key = str(device)
    with _SHARED_MODEL_CACHE_LOCK:
        lock = _DEVICE_INFERENCE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _DEVICE_INFERENCE_LOCKS[key] = lock
        return lock


def _shared_model_cache_key(checkpoint: str | None, cfg: dict[str, Any], device: Any) -> tuple[str, str, str]:
    ckpt = str(Path(checkpoint).expanduser().resolve()) if checkpoint else "<external-or-none>"
    # The model block defines architecture.  Runtime differences belong to the
    # planner/core config and do not require duplicate CUDA model copies.
    model_sig = json.dumps({
        "model": cfg.get("model", {}),
        "external_baseline": cfg.get("external_baseline", {}),
    }, sort_keys=True, default=str)
    return ckpt, str(device), model_sig


def _record_closed_loop_profile(planner_name: str, diagnostics: dict[str, Any]) -> None:
    path = os.environ.get("BDSE_CLOSED_LOOP_PROFILE_JSON", "").strip()
    if not path:
        return
    flat: dict[str, float] = {}
    for group in ("timing", "timing_core"):
        values = diagnostics.get(group, {}) if isinstance(diagnostics, dict) else {}
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    flat[f"{group}.{key}"] = float(value)
    flat["cached_plan"] = float(bool(diagnostics.get("cached_plan", False)))
    with _CLOSED_LOOP_PROFILE_LOCK:
        state = _CLOSED_LOOP_PROFILE.setdefault(planner_name, {"calls": 0.0})
        state["calls"] = state.get("calls", 0.0) + 1.0
        for key, value in flat.items():
            state[f"sum.{key}"] = state.get(f"sum.{key}", 0.0) + value
            state[f"max.{key}"] = max(state.get(f"max.{key}", float("-inf")), value)


def _flush_closed_loop_profile() -> None:
    path = os.environ.get("BDSE_CLOSED_LOOP_PROFILE_JSON", "").strip()
    if not path:
        return
    try:
        out: dict[str, Any] = {}
        with _CLOSED_LOOP_PROFILE_LOCK:
            snapshot = {k: dict(v) for k, v in _CLOSED_LOOP_PROFILE.items()}
        for planner, state in snapshot.items():
            calls = max(float(state.get("calls", 0.0)), 1.0)
            row: dict[str, float] = {"calls": float(state.get("calls", 0.0))}
            for key, value in state.items():
                if key.startswith("sum."):
                    row[f"mean.{key[4:]}"] = float(value) / calls
                elif key.startswith("max."):
                    row[key] = float(value)
            out[planner] = row
        dest = Path(path).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def _register_closed_loop_profile_flush() -> None:
    global _CLOSED_LOOP_PROFILE_REGISTERED
    if _CLOSED_LOOP_PROFILE_REGISTERED:
        return
    _CLOSED_LOOP_PROFILE_REGISTERED = True
    atexit.register(_flush_closed_loop_profile)


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


def runtime_query_diagnostics(
    pred: dict[str, Any], selected_atoms: list[int] | np.ndarray | None = None
) -> dict[str, int | float]:
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
    # v34: the model scores the unique selector/tournament union in one call.
    # Report actual executed scores, while retaining the legacy decompositions.
    unique_pair_atom = int(pred.get("unique_pair_atom_query_count", selector_pair_atom + tournament_pair_atom))
    total = int(action_atom + unique_pair_atom)

    if selected_atoms is None:
        selected_count = 0
    else:
        selected_count = int(len(np.asarray(selected_atoms, dtype=np.int64).reshape(-1)))
    # V62's deployment-complete action bridge is action-conditioned: every
    # selected atom is queried on every valid action, while the rival graph only
    # defines tournament comparisons.  Reporting B*|pairs| here would silently
    # undercount the paper-facing fixed B*K interface.  Older/direct pair-only
    # paths retain their historical pair-based accounting.
    action_query_mode = str(pred.get("action_query_mode", "")).strip().lower()
    action_bridge_executed = action_atom > 0 and len(actions) > 0 and (
        bool(pred.get("action_query_mode_all_valid", False))
        or action_query_mode in {"all_valid", "rival_graph", "runtime_pairs", "fallback_top_l"}
    )
    if action_bridge_executed:
        selected_certificate = int(selected_count * len(actions))
    elif len(rival_pairs):
        # rival_pair_indices are canonicalized to one query per unordered pair.
        selected_certificate = int(selected_count * len(rival_pairs))
    else:
        selected_certificate = int(selected_count * len(actions))

    budget_atom_count = int(pred.get("configured_decision_budget_atom_count", selected_count))
    upstream_budget_atom_count = int(pred.get("upstream_configured_decision_budget_atom_count", budget_atom_count))
    acquisition_expansion = float(len(topm) / max(selected_count, 1)) if selected_count else float("nan")
    return {
        "proposal_atom_count": int(len(topm)),
        "proposal_candidate_atom_count": int(len(topm)),
        "queried_action_count": int(len(actions)),
        "action_query_mode_all_valid": float(bool(pred.get("action_query_mode_all_valid", False))),
        "valid_action_count": int(pred.get("valid_action_count", len(actions))),
        "queried_valid_action_fraction": float(pred.get("queried_valid_action_fraction", 0.0)),
        "runtime_pair_count": int(len(runtime_pairs)),
        "tournament_pair_count": int(len(rival_pairs)),
        "action_atom_query_count": action_atom,
        "acquisition_action_atom_query_count": action_atom,
        "proposal_to_certificate_atom_expansion": acquisition_expansion,
        "configured_decision_budget_atom_count": budget_atom_count,
        "upstream_configured_decision_budget_atom_count": upstream_budget_atom_count,
        "retained_interface_atom_budget_pass": float(selected_count <= budget_atom_count),
        "selector_pair_atom_query_count": selector_pair_atom,
        "tournament_pair_atom_query_count": tournament_pair_atom,
        "unique_pair_atom_query_count": unique_pair_atom,
        "actual_unique_pair_count": int(pred.get("actual_unique_pair_count", 0)),
        "sparse_query_count": total,
        "total_sparse_query_count": total,
        "selected_certificate_query_count": selected_certificate,
        "effective_query_count": selected_certificate,
        "decision_budget_atom_count": selected_count,
        "structural_safety_atom_count": int(pred.get("structural_safety_atom_count", len(np.asarray(pred.get("mandatory_hard_atoms", []), dtype=np.int64).reshape(-1)))),
        "decision_budget_excludes_structural_safety": int(bool(pred.get("structural_safety_bypass", False))),
        "structural_safety_include_feasibility": int(bool(pred.get("structural_safety_include_feasibility", False))),
        "structural_residual_enabled": int(bool(pred.get("structural_residual_enabled", False))),
        "structural_residual_weight": float(pred.get("structural_residual_weight", 0.0)),
        "base_prior_enabled": int(bool(pred.get("base_prior_enabled", False))),
        "base_prior_weight": float(pred.get("base_prior_weight", 0.0)),
        "base_prior_scale": float(pred.get("base_prior_scale", 0.0)),
        "base_prior_best_action": int(pred.get("base_prior_best_action", -1)),
        "learned_base_best_action": int(pred.get("learned_base_best_action", -1)),
        "base_prior_replaced_best": int(bool(pred.get("base_prior_replaced_best", False))),
        "pair_delta_calibration_enabled": int(bool(pred.get("pair_delta_calibration_enabled", False))),
        "pair_delta_selector_local_weight_mean": float(pred.get("pair_delta_selector_local_weight_mean", 0.0)),
        "pair_delta_selector_local_weight_p90": float(pred.get("pair_delta_selector_local_weight_p90", 0.0)),
        "pair_delta_selector_sign_disagreement_rate": float(pred.get("pair_delta_selector_sign_disagreement_rate", 0.0)),
        "pair_delta_tournament_local_weight_mean": float(pred.get("pair_delta_tournament_local_weight_mean", 0.0)),
        "pair_delta_tournament_sign_disagreement_rate": float(pred.get("pair_delta_tournament_sign_disagreement_rate", 0.0)),
        "pair_delta_selector_residual_trust_mean": float(pred.get("pair_delta_selector_residual_trust_mean", 0.0)),
        "pair_delta_selector_residual_trust_p90": float(pred.get("pair_delta_selector_residual_trust_p90", 0.0)),
        "pair_delta_selector_residual_sign_disagreement_rate": float(pred.get("pair_delta_selector_residual_sign_disagreement_rate", 0.0)),
        "pair_delta_tournament_residual_trust_mean": float(pred.get("pair_delta_tournament_residual_trust_mean", 0.0)),
        "pair_delta_tournament_residual_trust_p90": float(pred.get("pair_delta_tournament_residual_trust_p90", 0.0)),
        "pair_delta_tournament_residual_sign_disagreement_rate": float(pred.get("pair_delta_tournament_residual_sign_disagreement_rate", 0.0)),
        "pair_delta_selector_residual_pair_flip_proposal_rate": float(pred.get("pair_delta_selector_residual_pair_flip_proposal_rate", 0.0)),
        "pair_delta_tournament_residual_pair_flip_proposal_rate": float(pred.get("pair_delta_tournament_residual_pair_flip_proposal_rate", 0.0)),
        "pair_delta_selector_residual_pair_flip_allowed_rate": float(pred.get("pair_delta_selector_residual_pair_flip_allowed_rate", 0.0)),
        "pair_delta_tournament_residual_pair_flip_allowed_rate": float(pred.get("pair_delta_tournament_residual_pair_flip_allowed_rate", 0.0)),
        "pair_delta_selector_residual_pair_confident_flip_rate": float(pred.get("pair_delta_selector_residual_pair_confident_flip_rate", 0.0)),
        "pair_delta_tournament_residual_pair_confident_flip_rate": float(pred.get("pair_delta_tournament_residual_pair_confident_flip_rate", 0.0)),
        "pair_delta_selector_residual_pair_sign_preserved_rate": float(pred.get("pair_delta_selector_residual_pair_sign_preserved_rate", 0.0)),
        "pair_delta_tournament_residual_pair_sign_preserved_rate": float(pred.get("pair_delta_tournament_residual_pair_sign_preserved_rate", 0.0)),
        "pair_delta_selector_residual_aggregate_scale_mean": float(pred.get("pair_delta_selector_residual_aggregate_scale_mean", 0.0)),
        "pair_delta_tournament_residual_aggregate_scale_mean": float(pred.get("pair_delta_tournament_residual_aggregate_scale_mean", 0.0)),
        "pair_delta_selector_residual_aggregate_scale_p10": float(pred.get("pair_delta_selector_residual_aggregate_scale_p10", 0.0)),
        "pair_delta_tournament_residual_aggregate_scale_p10": float(pred.get("pair_delta_tournament_residual_aggregate_scale_p10", 0.0)),
        "pair_delta_selector_residual_aggregate_correction_abs_mean": float(pred.get("pair_delta_selector_residual_aggregate_correction_abs_mean", 0.0)),
        "pair_delta_tournament_residual_aggregate_correction_abs_mean": float(pred.get("pair_delta_tournament_residual_aggregate_correction_abs_mean", 0.0)),
        "pair_delta_selector_residual_aggregate_local_abs_mean": float(pred.get("pair_delta_selector_residual_aggregate_local_abs_mean", 0.0)),
        "pair_delta_tournament_residual_aggregate_local_abs_mean": float(pred.get("pair_delta_tournament_residual_aggregate_local_abs_mean", 0.0)),
        "viability_pair_weight_mean": float(pred.get("viability_viability_pair_weight_mean", pred.get("viability_pair_weight_mean", 0.0))),
        "stage_predict_ms": float(pred.get("stage_predict_ms", 0.0)),
        "stage_selector_ms": float(pred.get("stage_selector_ms", 0.0)),
        "stage_tournament_ms": float(pred.get("stage_tournament_ms", 0.0)),
        "stage_total_internal_ms": float(pred.get("stage_total_internal_ms", 0.0)),
    }


from bdse.planner.selected_outcome_probe import (
    SelectedOutcomeProbeState,
    apply_selected_outcome_probe,
)


class BDSEPlannerCore:
    def __init__(self, model: Any | None = None, cfg: dict[str, Any] | None = None, inference_lock: threading.RLock | None = None):
        self.cfg = cfg or load_config()
        self.model = model
        self.inference_lock = inference_lock
        self._selected_outcome_probe_state = SelectedOutcomeProbeState()

    def reset_selected_outcome_probe(self) -> None:
        self._selected_outcome_probe_state.reset()

    def __getstate__(self) -> dict[str, Any]:
        # nuPlan's SimulationLogCallback pickles the planner at the end of every
        # scene. ``threading.RLock`` is not serializable; V58 therefore spent
        # hours simulating and then marked every scene failed during log export.
        # The lock is process-local synchronization state and must not be stored.
        state = dict(self.__dict__)
        state["inference_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.inference_lock = None

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
            if self.inference_lock is None:
                return self.model.predict_certificate_numpy(runtime, candidates, evidence_bank, cfg)
            with self.inference_lock:
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
        selector_cfg_early = cfg.get("selector", {}) if isinstance(cfg, dict) else {}
        viability_pair_diag: dict[str, float] = {}
        if bool(selector_cfg_early.get("decision_pairs_within_viability_frontier", False)):
            risk_dict = runtime_risk_scores(runtime, candidates, cfg)
            pairs, pair_weights, viability_pair_diag = restrict_pairs_to_viability_frontier(
                pairs,
                pair_weights,
                candidates.valid_mask,
                runtime_flags,
                J0,
                hard_risk=risk_dict.get("hard", None),
                frontier_size=int(selector_cfg_early.get("all_flagged_frontier_size", 8)),
                single_safe_rivals=int(selector_cfg_early.get("single_safe_anchor_rivals", 8)),
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
        interaction_group_ids = np.full((evidence_bank.E,), -1, dtype=np.int64)
        for i, atom in enumerate(evidence_bank.atoms[: evidence_bank.E]):
            try:
                interaction_group_ids[i] = int(getattr(atom, "anchor", {}).get("agent_index", -1))
            except Exception:
                interaction_group_ids[i] = -1
        topm, mandatory_hard_mask, _, topm_policy_diag = finalize_runtime_topm_policy(
            topm,
            proposal_scores=proposal_logits,
            family_ids=family_ids,
            active_mask=active,
            max_size=M,
            selector_cfg=selector_cfg_early,
            raw_hard_mask=raw_hard_mask,
            interaction_group_ids=interaction_group_ids,
        )
        structural_safety_bypass = bool(selector_cfg_early.get("decision_budget_excludes_structural_safety", False))
        hab_diag = dict(hab_diag)
        hab_diag.update(topm_policy_diag)
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
            "structural_safety_bypass": bool(structural_safety_bypass),
            "structural_safety_atom_count": int(mandatory_hard_mask.sum()),
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
            **{f"viability_{k}": v for k, v in viability_pair_diag.items()},
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
                    predicted_atom_costs=np.asarray(pred.get("g", zero_g), dtype=np.float32),
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
                if self.inference_lock is None:
                    dense = self.model.predict_dense_numpy(runtime, candidates, evidence_bank, stage_cfg)
                else:
                    with self.inference_lock:
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
        stage_started = time.perf_counter()
        pred = self._predict_runtime_certificate(runtime, candidates, evidence_bank, stage_cfg)
        predict_finished = time.perf_counter()
        selector_started = predict_finished
        selection_finished = predict_finished
        tournament_started = predict_finished
        tournament_finished = predict_finished
        J0, g = pred["J0"], pred["g"]
        g_var = pred.get("g_var", None)
        runtime_flags = runtime_safety_flags_from_runtime(runtime, candidates, stage_cfg)
        # V64.3.42: value-specific deployment observables are instrumented only
        # when explicitly requested or when a V42 post-selection value mode is
        # active.  They are label-free and never participate in challenger
        # selection; the frozen RSMR winner is chosen before they are consumed.
        value_observable_matrix = None
        value_observable_names = None
        try:
            _ic = (((stage_cfg.get("runtime", {}) or {}).get("decisive_frontier_value", {}) or {}).get("incumbent_contrastive_extremal_recovery", {}) or {})
            _scir = (_ic.get("selection_conditioned_intervention_recovery", {}) or {})
            _vmode = str(_scir.get("post_selection_value_mode", "")).strip().lower()
            _need_vobs = bool(_ic.get("instrument_value_observables", False)) or bool(_ic.get("instrument_future_response_observables", False)) or _vmode in {
                "endpoint_potential_quality_observable",
                "endpoint_potential_risk_observable",
                "endpoint_potential_joint_observable",
                "endpoint_potential_joint_observable_shift",
                "endpoint_potential_quality_future_response_mean",
                "endpoint_potential_quality_future_response_robust",
                "endpoint_potential_quality_future_response_robust_shift",
            }
            if _need_vobs:
                from bdse.planner.value_observables import runtime_value_observable_costs
                value_observable_matrix, value_observable_names = runtime_value_observable_costs(runtime, candidates, stage_cfg)
        except Exception:
            # A V42 mode must fail closed.  Pure instrumentation on historical
            # arms is also expected to be exact, so propagate instead of hiding
            # malformed schemas.
            raise
        sel_cfg = stage_cfg.get("selector", {})
        tour_cfg = stage_cfg.get("tournament", {})
        atom_active = np.zeros((evidence_bank.E,), dtype=bool)
        topm = np.asarray(pred.get("top_m_atoms", np.flatnonzero(evidence_bank.active_mask)), dtype=np.int64)
        atom_active[topm[(topm >= 0) & (topm < evidence_bank.E)]] = True
        atom_active &= evidence_bank.active_mask
        structural_safety_bypass = bool(sel_cfg.get("decision_budget_excludes_structural_safety", False))
        structural_mask = np.asarray(pred.get("mandatory_atom_mask", np.zeros((evidence_bank.E,), dtype=bool)), dtype=bool).reshape(-1)
        if structural_mask.shape[0] < evidence_bank.E:
            structural_mask = np.pad(structural_mask, (0, evidence_bank.E - structural_mask.shape[0]), constant_values=False)
        structural_mask = structural_mask[: evidence_bank.E] & np.asarray(evidence_bank.active_mask, dtype=bool)
        decision_atom_active = atom_active & ~structural_mask if structural_safety_bypass else atom_active
        family_ids = np.asarray(pred.get("family_ids", family_ids_from_atoms(evidence_bank.atoms, max_atoms=evidence_bank.E)), dtype=np.int64)
        family_caps = pred.get("family_budget_caps", None)
        baseline_mode = str(stage_cfg.get("planner", {}).get("baseline_mode", "bdse")).lower().replace("-", "_")
        if baseline_mode not in {"", "bdse", "pair_conditioned", "bdse_pair_conditioned"}:
            pred_b, selection_b, tournament_b = self._run_baseline_stage(baseline_mode, pred, runtime, candidates, evidence_bank, stage_cfg, atom_active, family_ids, runtime_flags)
            return pred_b, selection_b, tournament_b, atom_active
        use_pair_conditioned = bool(stage_cfg.get("runtime", {}).get("use_pair_conditioned_margins", stage_cfg.get("model", {}).get("pair_conditioned", True)))
        if use_pair_conditioned and "pair_atom_delta" in pred and "pair_indices" in pred:
            action_utility_cost = None
            if (
                float(sel_cfg.get("action_utility_weight", 0.0)) > 0.0
                or float(sel_cfg.get("action_pair_utility_weight", 0.0)) > 0.0
                or str(sel_cfg.get("selector_cap_mode", "")).lower() in {"safety_gated_action_rank", "lcb_action_rank_hybrid", "hybrid_lcb_action_rank", "safe_action_rank", "margin_coreset", "signed_margin_coreset", "mars", "margin_preserving", "deployment_coreset", "deployment_aligned_coreset", "dacc", "exact_tournament_coreset", "lexicographic_deployment_coreset", "lex_dacc", "lexdacc", "path_relaxed_deployment_coreset", "pr_dacc", "prdacc", "beam_dacc", "counterfactual_budget_layer_coreset", "cbl_dacc", "cbldacc", "budget_layer_dacc", "stage_aware_budget_layer_coreset", "sab_dacc", "sabdacc", "stage_aware_dacc", "anytime_adverse_certificate", "one_sided_adverse_certificate", "aocc", "aobcc", "nested_certificate"}
            ):
                action_utility_cost = _trajectory_utility_cost_np(
                    candidates.trajectories,
                    candidates.valid_mask,
                    runtime_flags,
                    stage_cfg,
                )
            cap_mode = str(sel_cfg.get("selector_cap_mode", "legacy_abs")).lower()
            deployment_modes = {
                "deployment_coreset", "deployment_aligned_coreset", "dacc", "exact_tournament_coreset",
                "lexicographic_deployment_coreset", "lex_dacc", "lexdacc",
                "path_relaxed_deployment_coreset", "pr_dacc", "prdacc", "beam_dacc", "counterfactual_budget_layer_coreset", "cbl_dacc", "cbldacc", "budget_layer_dacc",
                "stage_aware_budget_layer_coreset", "sab_dacc", "sabdacc", "stage_aware_dacc",
            }
            adverse_modes = {
                "anytime_adverse_certificate", "one_sided_adverse_certificate",
                "aocc", "aobcc", "nested_certificate",
            }
            tournament_cfg = dict(stage_cfg)
            tournament_cfg["runtime_pair_margin_scale"] = float(pred.get("rival_pair_margin_scale", pred.get("pair_margin_scale", 100.0)))

            # v41 PR-DACC aligns the *cheap search graph* with the graph used by
            # the final deployment tournament.  v39/v40 evaluated candidate
            # subsets with the rival graph but screened them with the selector
            # graph, so beam branching could discard the useful deletion before
            # exact evaluation.  The rival deltas have already been queried; this
            # changes no neural-query count.
            # Dual certificate: AOCC searches and certifies the selected-local
            # evidence anchor.  Residual action uncertainty is handled only by
            # the downstream robust flip guard and cannot collapse the evidence
            # certificate when no residual action change is proposed.
            search_pair_delta = pred.get("certificate_pair_atom_delta", pred["pair_atom_delta"])
            search_pair_indices = pred["pair_indices"]
            search_pair_weights = pred.get(
                "runtime_pair_weights",
                np.ones((np.asarray(search_pair_indices).reshape(-1, 2).shape[0],), dtype=np.float32),
            )
            search_pair_variance = pred.get("certificate_pair_atom_var", pred.get("pair_atom_var", None))
            search_pair_margin_scale = float(pred.get("pair_margin_scale", 100.0))
            deployment_search_uses_rival_graph = False
            if cap_mode in deployment_modes and bool(sel_cfg.get("deployment_coreset_use_deployment_pair_graph", False)):
                rival_indices = np.asarray(pred.get("rival_pair_indices", []), dtype=np.int64)
                rival_indices = rival_indices.reshape(-1, 2) if rival_indices.size else np.zeros((0, 2), dtype=np.int64)
                rival_delta = np.asarray(pred.get("certificate_rival_pair_atom_delta", pred.get("rival_pair_atom_delta", [])), dtype=np.float32)
                if rival_indices.size and rival_delta.ndim == 2 and rival_delta.shape[1] == rival_indices.shape[0]:
                    search_pair_delta = rival_delta
                    search_pair_indices = rival_indices
                    # The final tournament is unweighted over its rival sets.
                    search_pair_weights = np.ones((rival_indices.shape[0],), dtype=np.float32)
                    search_pair_variance = pred.get("certificate_rival_pair_atom_var", pred.get("rival_pair_atom_var", None))
                    search_pair_margin_scale = float(pred.get("rival_pair_margin_scale", pred.get("pair_margin_scale", 100.0)))
                    deployment_search_uses_rival_graph = True

            deployment_evaluator = None
            anchor_tournament_cfg = dict(tournament_cfg)
            anchor_runtime_cfg = dict(anchor_tournament_cfg.get("runtime", {}) or {})
            anchor_runtime_cfg["disable_pair_residual_intervention"] = True
            anchor_runtime_cfg["pair_tournament_aggregation_mode"] = "evidence_action_potential"
            anchor_tournament_cfg["runtime"] = anchor_runtime_cfg
            if cap_mode in (deployment_modes | adverse_modes):
                def deployment_evaluator(selected_atoms: list[int]):
                    # Evaluate the exact downstream decision rule used after
                    # selection: final rival graph, normalized pair margins,
                    # soft-min tournament, safety guard, utility refinement and
                    # the all-flagged structural guard.  This is pure inference
                    # over already queried Top-M deltas and adds no model query.
                    trial = run_pair_conditioned_tournament(
                        J0,
                        pred.get("certificate_rival_pair_atom_delta", pred.get("rival_pair_atom_delta", pred["pair_atom_delta"])),
                        pred.get("rival_pair_indices", pred["pair_indices"]),
                        selected_atoms,
                        candidates.valid_mask,
                        runtime_flags,
                        anchor_tournament_cfg,
                        pair_atom_variance=pred.get("certificate_rival_pair_atom_var", None),
                        candidate_trajectories=candidates.trajectories,
                        maneuver_ids=candidates.maneuver_ids,
                        predicted_atom_costs=np.asarray(pred["g"], dtype=np.float32),
                        residual_action_potential=None,
                        residual_action_variance=None,
                        value_observable_matrix=value_observable_matrix,
                        value_observable_names=value_observable_names,
                    )
                    trial = self._apply_all_flagged_structural_guard(
                        trial, runtime, candidates, runtime_flags, stage_cfg
                    )
                    return int(trial.action_index), np.asarray(trial.scores), np.asarray(trial.margins), dict(trial.diagnostics)

            adverse_target_action = None
            if cap_mode in adverse_modes and deployment_evaluator is not None:
                # AOCC must preserve the action selected by the exact downstream
                # tournament, not a different Copeland-style proxy.  This full
                # Top-M evaluation reuses already queried deltas and adds no neural
                # evidence query.
                full_topm_atoms = np.flatnonzero(decision_atom_active).astype(np.int64).tolist()
                adverse_target_action = int(deployment_evaluator(full_topm_atoms)[0])

            calibration_cfg = stage_cfg.get("calibration", {}) if isinstance(stage_cfg, dict) else {}
            adverse_calibrated = bool(
                calibration_cfg.get("independent", False)
                and sel_cfg.get("adverse_certificate_calibrated", False)
            )

            selector_started = time.perf_counter()
            # FBIC keeps the historical B=16 AOCC construction frozen and only
            # expands the *post-selector retained interface*.  This avoids
            # confounding the capacity ceiling with a different greedy search.
            fbic_cfg_pre = sel_cfg.get("full_bank_capacity_probe", {}) or {}
            selector_budget = float(stage_cfg.get("evidence", {}).get("budget", 16))
            if bool(fbic_cfg_pre.get("enabled", False)):
                selector_budget = float(fbic_cfg_pre.get("baseline_selector_budget", 16))
            selection = runtime_greedy_selector_pair_conditioned(
                J0,
                search_pair_delta,
                search_pair_indices,
                search_pair_weights,
                evidence_bank.budget_costs(),
                candidates.valid_mask,
                runtime_flags,
                budget=selector_budget,
                gamma_max=float(sel_cfg.get("normalized_gamma_max", 5.0) if bool(stage_cfg.get("model", {}).get("pair_margin_normalized", True)) else sel_cfg.get("gamma_max_default", 100.0)),
                eta_pred=float(sel_cfg.get("normalized_eta_pred", 0.1) if bool(stage_cfg.get("model", {}).get("pair_margin_normalized", True)) else sel_cfg.get("eta_pred", 1.0)),
                atom_active_mask=decision_atom_active,
                pair_atom_variance=search_pair_variance,
                beta_uncertainty=float(tour_cfg.get("beta_uncertainty", 0.0)),
                epsilon_cal=float(tour_cfg.get("epsilon_cal", stage_cfg.get("calibration", {}).get("epsilon_cal", 0.0))),
                lambda_info=float(sel_cfg.get("lambda_info", 0.0)),
                prior_atom_variance=sel_cfg.get("unqueried_atom_variance", None),
                family_ids=family_ids,
                family_budget_caps=family_caps,
                mandatory_atom_mask=None if structural_safety_bypass else pred.get("mandatory_atom_mask", None),
                mandatory_quota=0 if structural_safety_bypass else int(sel_cfg.get("mandatory_hard_quota", 0)),
                min_selected_atoms=int(sel_cfg.get("min_selected_atoms", 0)),
                force_fill_budget=bool(sel_cfg.get("force_fill_budget", False)),
                normalize_margins=bool(stage_cfg.get("model", {}).get("pair_margin_normalized", True)),
                margin_scale=search_pair_margin_scale,
                proposal_scores=pred.get("proposal_logits", None),
                proposal_fill_weight=float(sel_cfg.get("proposal_fill_weight", 0.25)),
                prioritize_mandatory_fill=bool(sel_cfg.get("prioritize_mandatory_fill", True)),
                selector_cap_mode=str(sel_cfg.get("selector_cap_mode", "legacy_abs")),
                boundary_certificate_cap=sel_cfg.get("boundary_certificate_cap", None),
                base_margin_cap_multiplier=float(sel_cfg.get("base_margin_cap_multiplier", 1.0)),
                flip_bonus=float(sel_cfg.get("flip_bonus", 0.0)),
                flip_window=float(sel_cfg.get("flip_window", 0.5)),
                certify_margin=float(sel_cfg.get("certify_margin", 0.0)),
                flip_mode=str(sel_cfg.get("flip_mode", "hard")),
                flip_temperature=float(sel_cfg.get("flip_temperature", 0.08)),
                action_rank_certificate_weight=float(sel_cfg.get("action_rank_certificate_weight", 1.0)),
                action_rank_score_weight=float(sel_cfg.get("action_rank_score_weight", 0.0)),
                action_rank_gap_weight=float(sel_cfg.get("action_rank_gap_weight", 0.0)),
                action_rank_flip_weight=float(sel_cfg.get("action_rank_flip_weight", 0.0)),
                action_rank_softmin_tau=float(sel_cfg.get("action_rank_softmin_tau", 0.2)),
                action_utility_cost=action_utility_cost,
                action_utility_weight=float(sel_cfg.get("action_utility_weight", 0.0)),
                action_pair_utility_weight=float(sel_cfg.get("action_pair_utility_weight", 0.0)),
                action_rank_fast_greedy=bool(sel_cfg.get("action_rank_fast_greedy", False)),
                hybrid_lcb_budget_frac=float(sel_cfg.get("hybrid_lcb_budget_frac", 0.55)),
                hybrid_lcb_cap_mode=str(sel_cfg.get("hybrid_lcb_cap_mode", "legacy_abs")),
                hybrid_protect_lcb_seed=bool(sel_cfg.get("hybrid_protect_lcb_seed", True)),
                hybrid_min_action_budget_frac=float(sel_cfg.get("hybrid_min_action_budget_frac", 0.0)),
                hybrid_max_lcb_seed_atoms=int(sel_cfg.get("hybrid_max_lcb_seed_atoms", 0)),
                adaptive_hybrid_lcb_budget=bool(sel_cfg.get("adaptive_hybrid_lcb_budget", False)),
                adaptive_lcb_min_frac=float(sel_cfg.get("adaptive_lcb_min_frac", 0.45)),
                adaptive_lcb_max_frac=float(sel_cfg.get("adaptive_lcb_max_frac", 0.80)),
                adaptive_lcb_safety_weight=float(sel_cfg.get("adaptive_lcb_safety_weight", 0.25)),
                adaptive_lcb_fallback_weight=float(sel_cfg.get("adaptive_lcb_fallback_weight", 0.20)),
                adaptive_lcb_uncertainty_weight=float(sel_cfg.get("adaptive_lcb_uncertainty_weight", 0.10)),
                adaptive_lcb_boundary_action_weight=float(sel_cfg.get("adaptive_lcb_boundary_action_weight", 0.25)),
                adaptive_lcb_boundary_tau=float(sel_cfg.get("adaptive_lcb_boundary_tau", 0.35)),
                decision_family_boost=float(sel_cfg.get("decision_family_boost", 0.0)),
                decision_family_ids=sel_cfg.get("decision_family_ids", [2, 3]),
                decision_family_quota=int(sel_cfg.get("decision_family_quota", 0)),
                interaction_family_ids=sel_cfg.get("interaction_family_ids", [2, 3]),
                interaction_family_quota=int(sel_cfg.get("interaction_family_quota", 0)),
                soft_interaction_mask=pred.get("soft_interaction_mask", None),
                soft_interaction_quota=int(sel_cfg.get("soft_interaction_quota", 0)),
                interaction_group_ids=pred.get("interaction_group_ids", None),
                direction_invariant_interaction_weight=float(sel_cfg.get("direction_invariant_interaction_weight", 0.0)),
                direction_invariant_boundary_tau=float(sel_cfg.get("direction_invariant_boundary_tau", 0.35)),
                direction_invariant_flip_bonus=float(sel_cfg.get("direction_invariant_flip_bonus", 0.5)),
                collapse_reciprocal_pairs=bool(sel_cfg.get("collapse_reciprocal_pairs", True)),
                force_uncertainty_objective=bool(sel_cfg.get("force_uncertainty_objective", False)),
                adverse_certificate_beta=float(sel_cfg.get("adverse_certificate_beta", 1.0)),
                adverse_certificate_epsilon=float(sel_cfg.get("adverse_certificate_epsilon", 0.05)),
                adverse_certificate_prior_radius=float(sel_cfg.get("adverse_certificate_prior_radius", 0.10)),
                adverse_certificate_margin=float(sel_cfg.get("adverse_certificate_margin", 0.0)),
                adverse_certificate_stop_when_certified=bool(sel_cfg.get("adverse_certificate_stop_when_certified", True)),
                adverse_certificate_max_target_rivals=int(sel_cfg.get("adverse_certificate_max_target_rivals", 0)),
                adverse_certificate_target_action=adverse_target_action,
                adverse_certificate_calibrated=adverse_calibrated,
                adverse_certificate_fill_to_budget_after_certified=bool(
                    sel_cfg.get("adverse_certificate_fill_to_budget_after_certified", False)
                ),
                adverse_certificate_max_interaction_prefix_fraction=float(
                    sel_cfg.get("adverse_certificate_max_interaction_prefix_fraction", 1.0)
                ),
                margin_coreset_residual_weight=float(sel_cfg.get("margin_coreset_residual_weight", 1.0)),
                margin_coreset_sign_weight=float(sel_cfg.get("margin_coreset_sign_weight", 0.8)),
                margin_coreset_winner_weight=float(sel_cfg.get("margin_coreset_winner_weight", 1.5)),
                margin_coreset_action_weight=float(sel_cfg.get("margin_coreset_action_weight", 0.5)),
                margin_coreset_boundary_tau=float(sel_cfg.get("margin_coreset_boundary_tau", 0.35)),
                margin_coreset_huber_delta=float(sel_cfg.get("margin_coreset_huber_delta", 0.25)),
                margin_coreset_target_clip=float(sel_cfg.get("margin_coreset_target_clip", 3.0)),
                margin_coreset_swap_passes=int(sel_cfg.get("margin_coreset_swap_passes", 2)),
                deployment_evaluator=deployment_evaluator,
                deployment_coreset_exact_candidates=int(sel_cfg.get("deployment_coreset_exact_candidates", 8)),
                deployment_coreset_swap_passes=int(sel_cfg.get("deployment_coreset_swap_passes", 1)),
                deployment_coreset_score_weight=float(sel_cfg.get("deployment_coreset_score_weight", 1.0)),
                deployment_coreset_action_weight=float(sel_cfg.get("deployment_coreset_action_weight", 4.0)),
                deployment_coreset_gap_weight=float(sel_cfg.get("deployment_coreset_gap_weight", 2.0)),
                deployment_coreset_margin_weight=float(sel_cfg.get("deployment_coreset_margin_weight", 1.0)),
                deployment_coreset_lexicographic_action_preservation=bool(
                    sel_cfg.get("deployment_coreset_lexicographic_action_preservation", False)
                ),
                deployment_coreset_preservation_scan_candidates=int(
                    sel_cfg.get("deployment_coreset_preservation_scan_candidates", 0)
                ),
                deployment_coreset_repair_one_swap=bool(
                    sel_cfg.get("deployment_coreset_repair_one_swap", True)
                ),
                deployment_coreset_repair_two_swap_candidates=int(
                    sel_cfg.get("deployment_coreset_repair_two_swap_candidates", 0)
                ),
                deployment_coreset_beam_width=int(
                    sel_cfg.get("deployment_coreset_beam_width", 0)
                ),
                deployment_coreset_beam_branch=int(
                    sel_cfg.get("deployment_coreset_beam_branch", 0)
                ),
                deployment_coreset_beam_max_evaluations=int(
                    sel_cfg.get("deployment_coreset_beam_max_evaluations", 0)
                ),
                deployment_coreset_beam_mismatch_fraction=float(
                    sel_cfg.get("deployment_coreset_beam_mismatch_fraction", 0.35)
                ),
                deployment_coreset_budget_layer_width=int(
                    sel_cfg.get("deployment_coreset_budget_layer_width", 0)
                ),
                deployment_coreset_budget_layer_branch=int(
                    sel_cfg.get("deployment_coreset_budget_layer_branch", 0)
                ),
                deployment_coreset_budget_layer_iterations=int(
                    sel_cfg.get("deployment_coreset_budget_layer_iterations", 0)
                ),
                deployment_coreset_budget_layer_max_evaluations=int(
                    sel_cfg.get("deployment_coreset_budget_layer_max_evaluations", 0)
                ),
                deployment_coreset_budget_layer_exhaustive_first=bool(
                    sel_cfg.get("deployment_coreset_budget_layer_exhaustive_first", True)
                ),
                deployment_coreset_budget_layer_seed_count=int(
                    sel_cfg.get("deployment_coreset_budget_layer_seed_count", 0)
                ),
                deployment_coreset_budget_layer_diversity_distance=int(
                    sel_cfg.get("deployment_coreset_budget_layer_diversity_distance", 2)
                ),
            )

            # V64.3.30 Full-Bank Interface Capacity (FBIC) probe.
            #
            # V29 showed that same-B frontier-fidelity rebinding can change most
            # selected atoms and improve its own full-M compression objective
            # without improving fresh recovery.  The next pre-registered causal
            # question is therefore capacity, not another B=16 allocation proxy.
            # FBIC exposes the complete *already queried* decision Top-M bank to
            # downstream EAF/ICER when B=24 can pay it.  It is a one-point upper
            # ceiling, adds no evidence query, sees no teacher label, and is a
            # strict no-op in the all-flagged structural domain.
            fbic_cfg = sel_cfg.get("full_bank_capacity_probe", {}) or {}
            if bool(fbic_cfg.get("enabled", False)):
                valid_for_fbic = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)
                flags_for_fbic = np.asarray(runtime_flags, dtype=bool).reshape(-1)
                k_fbic = min(valid_for_fbic.shape[0], flags_for_fbic.shape[0])
                structural_domain_fbic = bool(
                    k_fbic > 0
                    and bool(valid_for_fbic[:k_fbic].any())
                    and not bool((valid_for_fbic[:k_fbic] & ~flags_for_fbic[:k_fbic]).any())
                )
                fbic_reference_atoms = np.flatnonzero(decision_atom_active).astype(np.int64).tolist()
                fbic_interface_budget = float(
                    fbic_cfg.get("interface_budget", stage_cfg.get("evidence", {}).get("budget", 16))
                )
                fbic_upstream_budget = float(stage_cfg.get("evidence", {}).get("budget", 16))
                fbic_result = full_bank_capacity_probe(
                    baseline_selected=selection.selected,
                    reference_atoms=fbic_reference_atoms,
                    atom_budget_costs=evidence_bank.budget_costs(),
                    # Important causal-isolation detail: the global evidence
                    # budget remains the frozen B=16 upstream setting.  FBIC's
                    # separate interface_budget opens only the post-selector
                    # retained view over atoms that were already queried.
                    budget=fbic_interface_budget,
                    structural_domain=structural_domain_fbic,
                    expected_top_m=int(fbic_cfg.get("expected_top_m", sel_cfg.get("proposal_top_m", 24))),
                )
                selection.selected = list(fbic_result.selected)
                selection.diagnostics.update(fbic_result.diagnostics)
                # Runtime query accounting historically reads the configured
                # decision budget from the model prediction.  For FBIC that
                # field must describe the *retained interface* ceiling (24),
                # while the frozen upstream selector budget remains separately
                # auditable as 16.  Otherwise a valid capacity-ceiling arm is
                # spuriously logged as a budget violation.
                pred = dict(pred)
                pred["upstream_configured_decision_budget_atom_count"] = int(round(fbic_upstream_budget))
                pred["configured_decision_budget_atom_count"] = int(round(fbic_interface_budget))
            else:
                selection.diagnostics["full_bank_capacity_probe_enabled"] = 0.0
                selection.diagnostics["full_bank_capacity_probe_attempted"] = 0.0
                selection.diagnostics["full_bank_capacity_probe_applied"] = 0.0

            # V64.3.29 Frontier-Contrast Rebinding (FCR).
            #
            # The frozen AOCC selection remains the incumbent interface.  FCR is
            # a post-EAF, teacher-free, same-cardinality refinement over the
            # already queried Top-M evidence only.  It is allowed to replace the
            # selected B-set *only* when it preserves both the full-M local anchor
            # and the exact downstream full-M target action while strictly
            # reducing complete DARM+EAF anchor-star compression error.  Any
            # contract failure is a no-op fallback to AOCC.  This deliberately
            # avoids reopening the historically exhausted learned acquisition or
            # DACC/beam/swap search branches.
            fcr_cfg = sel_cfg.get("frontier_contrast_rebinding", {}) or {}
            if bool(fcr_cfg.get("enabled", False)):
                fcr_runtime_cfg = stage_cfg.get("runtime", {}) if isinstance(stage_cfg, dict) else {}
                fcr_frontier_cfg = fcr_runtime_cfg.get("decisive_frontier_value", {}) or {}
                fcr_reference_atoms = np.flatnonzero(decision_atom_active).astype(np.int64).tolist()
                fcr_result = frontier_contrast_rebind(
                    baseline_selected=selection.selected,
                    reference_atoms=fcr_reference_atoms,
                    predicted_base_cost=J0,
                    predicted_atom_costs=np.asarray(pred["g"], dtype=np.float32),
                    pair_indices=np.asarray(pred.get("rival_pair_indices", pred["pair_indices"]), dtype=np.int64),
                    pair_atom_delta=np.asarray(pred.get("rival_pair_atom_delta", pred["pair_atom_delta"]), dtype=np.float32),
                    valid_mask=candidates.valid_mask,
                    atom_budget_costs=evidence_bank.budget_costs(),
                    budget=float(stage_cfg.get("evidence", {}).get("budget", 16)),
                    normalize_margins=bool(stage_cfg.get("model", {}).get("pair_margin_normalized", True)),
                    margin_scale=float(pred.get("rival_pair_margin_scale", pred.get("pair_margin_scale", 100.0))),
                    pair_delta_includes_local=bool(fcr_runtime_cfg.get("pair_tournament_pair_delta_includes_local", True)),
                    frontier_value_atom_factors=pred.get("frontier_value_atom_factors", None),
                    frontier_value_action_signed_factors=pred.get("frontier_value_action_signed_factors", None),
                    frontier_value_action_context_factors=pred.get("frontier_value_action_context_factors", None),
                    frontier_value_scale=float(fcr_frontier_cfg.get("scale", pred.get("frontier_value_scale", 1.0))),
                    deployment_evaluator=deployment_evaluator,
                    full_target_action=adverse_target_action,
                )
                selection.selected = list(fcr_result.selected)
                selection.diagnostics.update(fcr_result.diagnostics)
            else:
                selection.diagnostics["frontier_contrast_rebinding_enabled"] = 0.0
                selection.diagnostics["frontier_contrast_rebinding_accepted"] = 0.0

            selection_finished = time.perf_counter()
            selection.diagnostics["deployment_coreset_search_uses_rival_graph"] = bool(
                deployment_search_uses_rival_graph
            )
            selection.diagnostics["aocc_exact_tournament_target_active"] = bool(
                cap_mode in adverse_modes and adverse_target_action is not None
            )

            # V64.3.1 Decision-Aligned Exact Preservation Certificate (DA-EPC).
            #
            # The historical AOCC certificate is a one-sided pair-margin bound.
            # With the current exact deployment operator (action-potential
            # aggregation + hard safety + utility refinement), satisfying that
            # pair surrogate is neither necessary nor sufficient for preserving
            # the actual winner.  The planner already evaluates the exact full
            # Top-M target above using cached model outputs.  For the paper-facing
            # interface certificate, audit the *same* downstream operator on the
            # selected B atoms and certify iff the winner is literally identical.
            # This is a single deterministic audit, not DACC-style combinatorial
            # search/repair; it changes neither the selected atoms nor B and adds
            # no neural evidence query.  The pairwise AOCC bound is retained in
            # diagnostics as a robustness surrogate.
            certificate_mode = str(
                sel_cfg.get("evidence_certificate_mode", "pairwise_aocc")
            ).strip().lower()
            exact_preservation_modes = {
                "exact_downstream_winner_preservation",
                "exact_winner_preservation",
                "decision_aligned_exact",
                "da_epc",
            }
            exact_selected_anchor_action = None
            exact_evidence_certificate = None
            if (
                certificate_mode in exact_preservation_modes
                and deployment_evaluator is not None
                and adverse_target_action is not None
            ):
                exact_selected_anchor_action = int(deployment_evaluator(selection.selected)[0])
                exact_evidence_certificate = float(
                    exact_selected_anchor_action == int(adverse_target_action)
                )
                selection.diagnostics["aocc_pairwise_certified_pair_fraction_raw"] = float(
                    selection.diagnostics.get("aocc_certified_pair_fraction", float("nan"))
                )
                selection.diagnostics["exact_winner_preservation_target_action"] = int(
                    adverse_target_action
                )
                selection.diagnostics["exact_winner_preservation_selected_action"] = int(
                    exact_selected_anchor_action
                )
                selection.diagnostics["exact_winner_preservation_certificate"] = float(
                    exact_evidence_certificate
                )
                selection.diagnostics["evidence_certificate_fraction"] = float(
                    exact_evidence_certificate
                )
                selection.diagnostics["evidence_certificate_mode_exact_winner_preservation"] = 1.0
            else:
                selection.diagnostics["evidence_certificate_fraction"] = float(
                    selection.diagnostics.get("aocc_certified_pair_fraction", 1.0)
                )
                selection.diagnostics["evidence_certificate_mode_exact_winner_preservation"] = 0.0

            tournament_started = time.perf_counter()
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
                predicted_atom_costs=np.asarray(pred["g"], dtype=np.float32),
                residual_action_potential=pred.get("residual_action_potential", None),
                residual_action_variance=pred.get("residual_action_var", None),
                residual_set_atom_factors=pred.get("residual_set_atom_factors", None),
                residual_set_action_factors=pred.get("residual_set_action_factors", None),
                frontier_value_atom_factors=pred.get("frontier_value_atom_factors", None),
                frontier_value_action_signed_factors=pred.get("frontier_value_action_signed_factors", None),
                frontier_value_action_context_factors=pred.get("frontier_value_action_context_factors", None),
                frontier_value_scale=float(pred.get("frontier_value_scale", 1.0)),
                evidence_certificate_fraction=selection.diagnostics.get(
                    "evidence_certificate_fraction",
                    selection.diagnostics.get("aocc_certified_pair_fraction", None),
                ),
                selected_atom_family_ids=np.asarray(family_ids, dtype=np.int64)[np.asarray(selection.selected, dtype=np.int64)],
                selected_atom_type_names=[str(evidence_bank.atoms[int(i)].type) for i in np.asarray(selection.selected, dtype=np.int64).tolist()],
                value_observable_matrix=value_observable_matrix,
                value_observable_names=value_observable_names,
            )
        else:
            selector_started = time.perf_counter()
            selection = runtime_greedy_selector(
                J0, g, evidence_bank.budget_costs(), candidates.valid_mask, runtime_flags,
                budget=float(stage_cfg.get("evidence", {}).get("budget", 16)),
                L_infer=int(tour_cfg.get("L_infer", 16)),
                gamma_max=float(sel_cfg.get("gamma_max_default", 100.0)),
                eta_pred=float(sel_cfg.get("eta_pred", 1.0)),
                lambda_near=float(sel_cfg.get("lambda_near", 1.0)),
                lambda_safety=float(sel_cfg.get("lambda_safety", 2.0)),
                atom_active_mask=decision_atom_active,
                predicted_atom_variance=g_var,
                beta_uncertainty=float(tour_cfg.get("beta_uncertainty", 0.0)),
                epsilon_cal=float(tour_cfg.get("epsilon_cal", stage_cfg.get("calibration", {}).get("epsilon_cal", 0.0))),
                lambda_info=float(sel_cfg.get("lambda_info", 0.0)),
                prior_atom_variance=sel_cfg.get("unqueried_atom_variance", None),
                family_ids=family_ids,
                family_budget_caps=family_caps,
                mandatory_atom_mask=None if structural_safety_bypass else pred.get("mandatory_atom_mask", None),
                mandatory_quota=0 if structural_safety_bypass else int(sel_cfg.get("mandatory_hard_quota", 0)),
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
            selection_finished = time.perf_counter()
            sigma = selected_pair_sigma_from_action_variance(g_var, selection.selected, candidates.valid_mask)
            tournament_started = time.perf_counter()
            tournament = run_tournament(
                J0, g, selection.selected, candidates.valid_mask, runtime_flags, stage_cfg, sigma=sigma,
                candidate_trajectories=candidates.trajectories,
                maneuver_ids=candidates.maneuver_ids,
            )
        # Expose selector certificate diagnostics to the fallback controller.
        for key in (
            "aocc_certified_pair_fraction", "aocc_pairwise_certified_pair_fraction_raw",
            "aocc_final_deficit", "aocc_bound_calibrated", "aocc_target_action",
            "aocc_exact_tournament_target_active", "evidence_certificate_fraction",
            "evidence_certificate_mode_exact_winner_preservation",
            "exact_winner_preservation_certificate",
            "exact_winner_preservation_target_action",
            "exact_winner_preservation_selected_action",
        ):
            if key in selection.diagnostics:
                tournament.diagnostics[key] = selection.diagnostics[key]
        tournament = self._apply_all_flagged_structural_guard(
            tournament, runtime, candidates, runtime_flags, stage_cfg
        )
        tournament = self._finalize_pair_anchor_after_structural_guard(
            tournament, runtime, candidates, runtime_flags, stage_cfg
        )
        evidence_cert = float(
            tournament.diagnostics.get(
                "evidence_certificate_fraction",
                tournament.diagnostics.get("aocc_certified_pair_fraction", 1.0),
            )
        )
        raw_anchor_action = int(
            tournament.diagnostics.get(
                "pair_action_anchor_raw_anchor_action",
                tournament.diagnostics.get("pair_action_anchor_pre_structural_action", tournament.action_index),
            )
        )
        raw_proposed_action = int(
            tournament.diagnostics.get(
                "pair_action_anchor_raw_proposed_action",
                tournament.diagnostics.get("pair_action_anchor_proposed_action", tournament.action_index),
            )
        )
        residual_flip_proposed = bool(raw_proposed_action != raw_anchor_action)
        margin_cert = bool(
            tournament.diagnostics.get("pair_action_anchor_guard_margin_certificate_pass", True)
        )
        evidence_guard_cert = bool(
            tournament.diagnostics.get("pair_action_anchor_guard_evidence_certificate_pass", True)
        )
        dual_cfg = ((stage_cfg.get("runtime", {}) or {}).get("dual_certificate", {}) or {})
        evidence_flip_floor = float(
            dual_cfg.get("min_evidence_certificate_fraction_for_residual_flip", 1.0)
        )
        evidence_cert_pass = bool(
            evidence_guard_cert and evidence_cert + 1.0e-9 >= evidence_flip_floor
        )
        # Conditional certificate fields describe the learned residual proposal,
        # not a later structural guard.  No-proposal scenes are safe abstentions
        # and are reported separately from proposal-conditional pass rates.
        residual_cert = bool((not residual_flip_proposed) or margin_cert)
        dual_cert = bool((not residual_flip_proposed) or (margin_cert and evidence_cert_pass))
        tournament.diagnostics.update({
            "evidence_certificate_fraction": evidence_cert,
            "evidence_certificate_pass_for_residual_flip": evidence_cert_pass,
            "residual_flip_raw_anchor_action": raw_anchor_action,
            "residual_flip_raw_proposed_action": raw_proposed_action,
            "residual_flip_proposed": residual_flip_proposed,
            "residual_flip_deployed": bool(tournament.diagnostics.get("pair_action_anchor_deployed_flip", False)),
            "residual_flip_margin_certificate_pass": margin_cert,
            "residual_flip_certificate_pass": residual_cert,
            "residual_flip_certificate_pass_conditional": bool(margin_cert) if residual_flip_proposed else float("nan"),
            "dual_certificate_pass_conditional": bool(margin_cert and evidence_cert_pass) if residual_flip_proposed else float("nan"),
            "dual_certificate_deployment_certified": dual_cert,
        })
        tournament_finished = time.perf_counter()
        pred = dict(pred)
        pred.update({
            "stage_predict_ms": 1000.0 * (predict_finished - stage_started),
            "stage_selector_ms": 1000.0 * (selection_finished - selector_started),
            "stage_tournament_ms": 1000.0 * (tournament_finished - tournament_started),
            "stage_total_internal_ms": 1000.0 * (tournament_finished - stage_started),
        })
        selection.diagnostics.update({
            "structural_safety_bypass": bool(structural_safety_bypass),
            "structural_safety_atom_count": int(structural_mask.sum()),
            "decision_budget_atom_count": int(len(selection.selected)),
            "decision_topm_atom_count": int(decision_atom_active.sum()),
        })
        return pred, selection, tournament, decision_atom_active

    def _finalize_pair_anchor_after_structural_guard(
        self,
        tournament: TournamentResult,
        runtime: RuntimeFeatures,
        candidates,
        runtime_flags: np.ndarray,
        cfg: dict[str, Any],
    ) -> TournamentResult:
        """Compare candidate and anchor after identical structural post-processing."""
        diag = tournament.diagnostics
        scores = diag.pop("_pair_action_anchor_scores", None)
        margins = diag.pop("_pair_action_anchor_margins", None)
        pre_action = diag.pop("_pair_action_anchor_pre_structural_action", None)
        if scores is None or margins is None or pre_action is None:
            return tournament
        anchor = TournamentResult(
            action_index=int(pre_action),
            scores=np.asarray(scores, dtype=np.float32),
            margins=np.asarray(margins, dtype=np.float32),
            rival_sets=tournament.rival_sets,
            diagnostics={},
        )
        anchor = self._apply_all_flagged_structural_guard(anchor, runtime, candidates, runtime_flags, cfg)
        post_anchor = int(anchor.action_index)
        post_structural_reverted = False
        # Candidate and anchor pass through the same structural guard, but their
        # score arrays differ.  If the residual flip certificate rejected the
        # intervention, the structural tie-break must not reintroduce that flip.
        if (
            bool(diag.get("pair_action_anchor_guard_active", False))
            and not bool(diag.get("pair_action_anchor_guard_allowed_flip", False))
            and int(tournament.action_index) != post_anchor
        ):
            tournament.action_index = post_anchor
            post_structural_reverted = True
        diag.update({
            "pair_action_anchor_pre_structural_action": int(pre_action),
            "pair_action_anchor_action": post_anchor,
            "pair_action_anchor_post_structural_action": post_anchor,
            "pair_action_anchor_structural_guard_changed": bool(post_anchor != int(pre_action)),
            "pair_action_anchor_post_structural_reverted": bool(post_structural_reverted),
            "pair_action_anchor_deployed_flip": bool(int(tournament.action_index) != post_anchor),
        })
        return tournament


    def _apply_all_flagged_structural_guard(
        self,
        tournament,
        runtime: RuntimeFeatures,
        candidates,
        runtime_flags: np.ndarray,
        cfg: dict[str, Any],
    ):
        """Use continuous structural risk when every valid action is flagged.

        A boolean hard filter cannot discriminate an all-flagged candidate bank.
        V36 therefore reported a 3.5% selected-action flag rate even though there
        was no avoidable unsafe choice.  The guard keeps the learned certificate
        as the tie-breaker, but first restricts the choice to a near-minimum hard
        risk set using only deployment-time geometry.
        """
        gcfg = (((cfg.get("tournament", {}) or {}).get("all_flagged_risk_guard", {}) or {})
                if isinstance(cfg, dict) else {})
        valid = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)
        flags = np.asarray(runtime_flags, dtype=bool).reshape(-1)
        K = min(len(valid), len(flags), len(np.asarray(tournament.scores).reshape(-1)))
        valid = valid[:K]
        flags = flags[:K]
        safe_available = bool((valid & ~flags).any())
        all_flagged = bool(valid.any() and not safe_available)
        tournament.diagnostics["all_actions_safety_flagged"] = all_flagged
        if not all_flagged or not bool(gcfg.get("enabled", True)):
            selected = int(tournament.action_index)
            selected_flag = bool(flags[selected]) if 0 <= selected < K else True
            tournament.diagnostics["avoidable_selected_action_safety_flag"] = bool(selected_flag and safe_available)
            tournament.diagnostics["all_flagged_risk_guard_applied"] = False
            return tournament

        risks = runtime_risk_scores(runtime, candidates, cfg)
        hard = np.asarray(risks.get("hard", np.full((K,), np.inf)), dtype=np.float32).reshape(-1)[:K]
        soft = np.asarray(risks.get("soft", np.full((K,), np.inf)), dtype=np.float32).reshape(-1)[:K]
        red = np.asarray(risks.get("red_light", np.zeros((K,))), dtype=np.float32).reshape(-1)[:K]
        min_ttc = np.asarray(risks.get("min_ttc_s", np.full((K,), np.inf)), dtype=np.float32).reshape(-1)[:K]
        valid_idx = np.flatnonzero(valid & np.isfinite(hard))
        if valid_idx.size == 0:
            tournament.diagnostics["all_flagged_risk_guard_applied"] = False
            return tournament

        red_min = float(np.min(red[valid_idx]))
        red_tol = max(float(gcfg.get("red_tolerance", 1e-6)), 0.0)
        pool = valid_idx[red[valid_idx] <= red_min + red_tol]
        if pool.size == 0:
            pool = valid_idx
        hard_min = float(np.min(hard[pool]))
        finite_hard = hard[pool][np.isfinite(hard[pool])]
        scale = float(np.quantile(finite_hard, 0.75) - np.quantile(finite_hard, 0.25)) if finite_hard.size > 1 else 0.0
        slack = max(float(gcfg.get("hard_risk_abs_slack", 0.10)), float(gcfg.get("hard_risk_rel_slack", 0.08)) * max(scale, 1e-3))
        near = pool[hard[pool] <= hard_min + slack]
        if near.size:
            pool = near
        scores = np.asarray(tournament.scores, dtype=np.float32).reshape(-1)[:K]
        chosen = max(
            pool.tolist(),
            key=lambda a: (
                float(scores[int(a)]),
                -float(soft[int(a)]) if np.isfinite(soft[int(a)]) else -1e9,
                float(min_ttc[int(a)]) if np.isfinite(min_ttc[int(a)]) else 1e9,
                -int(a),
            ),
        )
        old = int(tournament.action_index)
        tournament.action_index = int(chosen)
        tournament.diagnostics.update({
            "all_flagged_risk_guard_applied": True,
            "all_flagged_action_before_guard": int(old),
            "all_flagged_guard_pool_size": int(len(pool)),
            "all_flagged_selected_hard_risk": float(hard[int(chosen)]),
            "all_flagged_min_hard_risk": float(hard_min),
            "all_flagged_hard_risk_regret": float(max(float(hard[int(chosen)] - hard_min), 0.0)),
            "selected_action_safety_flag": True,
            "avoidable_selected_action_safety_flag": False,
        })
        return tournament

    def _fallback_thresholds(self, cfg: dict[str, Any]) -> tuple[float, float]:
        fcfg = cfg.get("fallback", {})
        # Pair-conditioned margins are normalized before tournament scoring.  The
        # legacy raw-cost tau_delta=0.1 made v24 enter fallback on nearly every
        # closed-loop replan even after CACE improved proposal recall.  Use a
        # separate normalized threshold whenever the model uses normalized pair
        # margins; keep the old default for raw-cost ablations.
        normalized = bool(cfg.get("model", {}).get("pair_margin_normalized", False))
        if normalized:
            tau_delta = float(fcfg.get("tau_delta_normalized", fcfg.get("tau_delta", 0.01)))
            safety_thr = float(fcfg.get("safety_lcb_min_normalized", fcfg.get("safety_lcb_min", -0.02)))
        else:
            tau_delta = float(fcfg.get("tau_delta", 0.1))
            safety_thr = float(fcfg.get("safety_lcb_min", 0.0))
        return tau_delta, safety_thr

    def _fallback_reason(self, tournament, cfg: dict[str, Any]) -> str:
        fcfg = cfg.get("fallback", {})
        if not bool(fcfg.get("enabled", True)):
            return "disabled"
        tau_delta, safety_thr = self._fallback_thresholds(cfg)
        if bool(fcfg.get("trigger_on_uncertified_aocc", False)):
            certified_fraction = float(
                tournament.diagnostics.get(
                    "evidence_certificate_fraction",
                    tournament.diagnostics.get("aocc_certified_pair_fraction", 1.0),
                )
            )
            min_fraction = float(fcfg.get("min_aocc_certified_pair_fraction", 1.0))
            if certified_fraction < min_fraction:
                return "uncertified_aocc"
        if bool(tournament.diagnostics.get("selected_action_safety_flag", False)):
            return "selected_action_safety_flag"
        safety_lcb = float(tournament.diagnostics.get("safety_lcb_min", float("inf")))
        if safety_lcb < safety_thr:
            return "safety_lcb_min"
        delta = float(tournament.diagnostics.get("delta_hat_B", 0.0))
        trigger_low_delta = bool(fcfg.get("trigger_on_low_delta", True))
        if delta < tau_delta:
            if not trigger_low_delta:
                return "accepted_low_delta"
            if bool(fcfg.get("accept_low_delta_if_safe", False)):
                return "accepted_low_delta"
            return "low_delta"
        return "accepted"

    def _needs_fallback(self, tournament, candidates, cfg: dict[str, Any]) -> bool:
        return self._fallback_reason(tournament, cfg) not in {"disabled", "accepted", "accepted_low_delta"}

    def plan_from_runtime(self, runtime: RuntimeFeatures) -> tuple[int, np.ndarray, dict[str, Any]]:
        prediction_scope = getattr(self.model, "runtime_prediction_cache_scope", None)
        model_scope = prediction_scope() if callable(prediction_scope) else nullcontext(None)
        with runtime_safety_cache_scope() as safety_memo, model_scope as prediction_memo:
            action, trajectory, diagnostics = self._plan_from_runtime_impl(runtime)
            if os.environ.get("BDSE_PROFILE_CLOSED_LOOP", "0").lower() in {"1", "true", "yes", "on"}:
                timing = diagnostics.setdefault("timing_core", {})
                timing["runtime_safety_cache_hits"] = int(safety_memo.hits)
                timing["runtime_safety_cache_misses"] = int(safety_memo.misses)
                timing["runtime_safety_cache_entries"] = int(len(safety_memo.cache))
                if prediction_memo is not None:
                    timing["runtime_prediction_cache_hits"] = int(prediction_memo.hits)
                    timing["runtime_prediction_cache_misses"] = int(prediction_memo.misses)
                    timing["runtime_prediction_cache_entries"] = int(len(prediction_memo.cache))
            return action, trajectory, diagnostics

    def _plan_from_runtime_impl(self, runtime: RuntimeFeatures) -> tuple[int, np.ndarray, dict[str, Any]]:
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
            if max_extra_stages is None or int(max_extra_stages) > 0:
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
            qdiag.update({k: v for k, v in getattr(tournament, "diagnostics", {}).items() if k in {"normalized_margins", "margin_scale", "epsilon_cal", "pair_conditioned", "selected_action_safety_flag", "avoidable_selected_action_safety_flag", "all_actions_safety_flagged", "all_flagged_risk_guard_applied", "all_flagged_hard_risk_regret", "hard_filter_applied", "safe_action_available"}})
            safety_diag_stage = runtime_safety_diagnostics(runtime, candidates, cfg_stage)
            stage_records.append({
                "stage": stage_name,
                "action": int(tournament.action_index),
                "delta_hat_B": float(tournament.diagnostics.get("delta_hat_B", 0.0)),
                "safety_lcb_min": float(tournament.diagnostics.get("safety_lcb_min", float("inf"))),
                "fallback_reason": self._fallback_reason(tournament, cfg_stage),
                "runtime_safety": safety_diag_stage,
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
        safety_diag_final = runtime_safety_diagnostics(runtime, candidates, cfg_stage)
        if profile_enabled:
            timing_core["final_safety_flags_s"] = float(time.perf_counter() - t_post)

        # V64.3.50 TRAIN/fresh evidence probe.  This changes no deployed V49
        # science path because it is disabled unless an explicit probe config is
        # supplied.  With fallback disabled it creates a single paired intervention
        # on the first *live* full-set RSMR winner from the frozen selector:
        # treatment executes that live winner exactly once; control preserves the
        # incumbent.  Candidate action integers are state-local bank slots, so V50
        # also records a trajectory/semantic fingerprint for same-state pair identity.
        probe_input = dict(tournament.diagnostics)
        pcfg = cfg_stage.get("selected_outcome_probe", {}) if isinstance(cfg_stage, dict) else {}
        probe_enabled = bool((pcfg or {}).get("enabled", False))
        pre_probe_action = int(action)
        t_probe = time.perf_counter()
        identity_cache: dict[int, dict[str, Any]] = {}

        def v50_identity(action_index: int) -> dict[str, Any]:
            ai = int(action_index)
            if ai not in identity_cache:
                identity_cache[ai] = _v50_live_candidate_identity(candidates, ai)
            return identity_cache[ai]

        if probe_enabled:
            try:
                proposal_exists = bool(float(probe_input.get("decisive_frontier_icer_scir_proposal_exists", 0.0)) > 0.5)
                proposal_action = int(round(float(probe_input.get("decisive_frontier_icer_scir_proposal_action", -1))))
            except Exception:
                proposal_exists = False
                proposal_action = -1
            probe_input.update(_v50_prefixed_identity(v50_identity(pre_probe_action), "v50_pre_probe_action"))
            if proposal_exists:
                probe_input.update(v50_identity(proposal_action))
        action, selected_outcome_probe_diag = apply_selected_outcome_probe(
            action, probe_input, cfg_stage, self._selected_outcome_probe_state
        )
        if probe_enabled:
            selected_outcome_probe_diag.update(
                _v50_prefixed_identity(v50_identity(pre_probe_action), "v50_pre_probe_action")
            )
            selected_outcome_probe_diag.update(
                _v50_prefixed_identity(v50_identity(int(action)), "v50_post_probe_action")
            )
        if profile_enabled:
            timing_core["v50_probe_instrumentation_s"] = float(time.perf_counter() - t_probe)
        recovery_diag: dict[str, Any] = {}
        if triggered and bool(fcfg.get("rule_rerank_top_k", 5)):
            from bdse.planner.fallback import (
                conservative_fallback_action,
                rule_based_runtime_scores,
                runtime_risk_scores,
                runtime_safety_flag_components,
                viability_frontier_recovery_action,
            )
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
                recovery_cfg = ((((cfg_stage.get("fallback", {}) or {}).get("safe_progress_recovery", {}) or {}).get("viability_frontier", {}) or {}) if isinstance(cfg_stage, dict) else {})
                if bool(recovery_cfg.get("enabled", False)):
                    decision = viability_frontier_recovery_action(
                        candidates,
                        safety_flags=runtime_flags,
                        cfg=cfg_stage,
                        runtime=runtime,
                        tournament_scores=tournament.scores,
                        reference_action=action,
                    )
                    action = int(decision.action_index)
                    recovery_diag = dict(decision.diagnostics)
                    stage_name = stage_name + "+vcdsr"
                else:
                    action = int(conservative_fallback_action(candidates, safety_flags=runtime_flags, cfg=cfg_stage, runtime=runtime))
                    stage_name = stage_name + "+safe_progress"
            if profile_enabled:
                timing_core["rule_rerank_s"] = float(time.perf_counter() - t_rule)
        # v29 diagnostic fix: tournament.selected_action_safety_flag was computed
        # before rule_rerank / safe_progress could change the action.  Keep the
        # original value but expose final-action hard/soft flags and risk so the
        # closed-loop logs describe the actually deployed trajectory.
        try:
            comp_final = runtime_safety_flag_components(runtime, candidates, cfg_stage)
            risks_final = runtime_risk_scores(runtime, candidates, cfg_stage)
            final_runtime_safety = {
                "final_action_safety_flag": bool(runtime_flags[action]) if 0 <= int(action) < len(runtime_flags) else True,
                "final_action_hard_flag": bool(comp_final.get("hard", runtime_flags)[action]) if 0 <= int(action) < len(runtime_flags) else True,
                "final_action_soft_flag": bool(comp_final.get("soft", runtime_flags)[action]) if 0 <= int(action) < len(runtime_flags) else True,
                "final_action_hard_risk": float(risks_final.get("hard", [float("inf")])[action]) if 0 <= int(action) < len(runtime_flags) else float("inf"),
                "final_action_soft_risk": float(risks_final.get("soft", [float("inf")])[action]) if 0 <= int(action) < len(runtime_flags) else float("inf"),
                "final_action_hard_agent_risk": float(risks_final.get("hard_agent", risks_final.get("agent", [float("inf")]))[action]) if 0 <= int(action) < len(runtime_flags) else float("inf"),
                "final_action_hard_offroute_risk": float(risks_final.get("hard_off_route", risks_final.get("off_route", [float("inf")]))[action]) if 0 <= int(action) < len(runtime_flags) else float("inf"),
                "pre_recovery_action": int(tournament.action_index),
                **({f"recovery_{k}": v for k, v in recovery_diag.items()} if recovery_diag else {}),
            }
        except Exception:
            final_runtime_safety = {
                "final_action_safety_flag": bool(runtime_flags[action]) if 0 <= int(action) < len(runtime_flags) else True,
                "pre_recovery_action": int(tournament.action_index),
            }
        trajectory = candidates.trajectories[action]
        qdiag = runtime_query_diagnostics(pred, selection.selected)
        qdiag.update({k: v for k, v in getattr(tournament, "diagnostics", {}).items() if k in {"normalized_margins", "margin_scale", "epsilon_cal", "pair_conditioned", "selected_action_safety_flag", "avoidable_selected_action_safety_flag", "all_actions_safety_flagged", "all_flagged_risk_guard_applied", "all_flagged_hard_risk_regret", "hard_filter_applied", "safe_action_available"}})
        tournament_diag = dict(tournament.diagnostics)
        if "selected_action_safety_flag" in tournament_diag:
            tournament_diag["pre_recovery_selected_action_safety_flag"] = bool(tournament_diag.get("selected_action_safety_flag", False))
        tournament_diag.update(final_runtime_safety)
        tournament_diag["selected_action_safety_flag"] = bool(final_runtime_safety.get("final_action_safety_flag", False))
        diagnostics = {
            "action_index": action,
            "selected_atoms": selection.selected,
            "proposal_top_m_atoms": list(map(int, np.asarray(pred.get("top_m_atoms", []), dtype=np.int64).tolist())),
            "queried_actions": list(map(int, np.asarray(pred.get("queried_actions", []), dtype=np.int64).tolist())),
            **qdiag,
            "hab": pred.get("hab_diagnostics", {}),
            **({"model_timing": pred.get("model_timing", {})} if profile_enabled else {}),
            "selector": selection.diagnostics,
            "tournament": tournament_diag,
            "runtime_safety": {**safety_diag_final, **final_runtime_safety},
            "fallback_stage": stage_name,
            "fallback_triggered": bool(triggered),
            "fallback_reason": self._fallback_reason(tournament, cfg_stage),
            "fallback_stage_records": stage_records,
            "recovery": recovery_diag,
            "selected_outcome_probe": selected_outcome_probe_diag,
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
    # V50 batched evidence collection needs the native nuPlan scenario token in
    # runtime diagnostics so several scenarios can safely share one process.
    # Keep the historical planner interface unchanged unless the dedicated
    # instrumentation environment flag is present.
    requires_scenario: bool = _env_true("BDSE_REQUIRE_SCENARIO_FOR_DIAG", default=False)

    def __init__(
        self,
        model: Any | None = None,
        cfg: dict[str, Any] | None = None,
        checkpoint: str | None = None,
        config_path: str | None = None,
        device: str = "auto",
        scenario: Any | None = None,
    ):
        cfg = cfg or load_config(config_path)
        # Instrumentation-only identity.  It never enters the planner features,
        # candidate generation, selector, or treatment assignment.  nuPlan passes
        # the scenario only when ``requires_scenario`` is enabled.
        self._diagnostic_scenario_token = str(getattr(scenario, "token", "") or "")
        device = _maybe_shard_planner_device(device)
        self.device = resolve_torch_device(device, context="BDSEnuPlanPlanner")
        configure_torch_for_device(self.device)
        external_enabled = bool((cfg.get("external_baseline", {}) or {}).get("enabled", False))
        share_model = _env_true("BDSE_SHARE_MODEL_PER_PROCESS", default=False)
        inference_lock = _device_inference_lock(self.device) if _env_true("BDSE_SERIALIZE_GPU_INFERENCE", default=share_model) else None
        reused_model = False
        if model is None and (checkpoint or external_enabled):
            # nuPlan constructs one planner per simulation.  In a threaded shard,
            # loading ten identical CUDA models wastes memory and makes kernels
            # contend.  V56 can share one read-only eval model per process/device.
            from bdse.external_baselines.model_factory import load_model_for_config

            if share_model:
                key = _shared_model_cache_key(checkpoint, cfg, self.device)
                # Keep construction under the cache lock. nuPlan may initialize
                # many planner instances concurrently; releasing the lock before
                # load_model_for_config lets every worker allocate an identical
                # CUDA model and defeats the cache (or OOMs) before setdefault.
                with _SHARED_MODEL_CACHE_LOCK:
                    model = _SHARED_MODEL_CACHE.get(key)
                    if model is None:
                        model = load_model_for_config(checkpoint, cfg, self.device)
                        _SHARED_MODEL_CACHE[key] = model
                        reused_model = False
                    else:
                        reused_model = True
            else:
                model = load_model_for_config(checkpoint, cfg, self.device)
        elif model is not None and hasattr(model, "to"):
            model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()
        variant = str((cfg.get("external_baseline", {}) or {}).get("variant", "bdse")) if external_enabled else "bdse"
        params = 0
        model_param_devices: set[str] = set()
        if model is not None and hasattr(model, "parameters"):
            try:
                model_params = list(model.parameters())
                params = int(sum(int(p.numel()) for p in model_params))
                model_param_devices = {str(p.device) for p in model_params if int(p.numel()) > 0}
            except Exception:
                params = -1
        if str(self.device).startswith("cuda") and params > 0:
            if not model_param_devices or any(not dev.startswith("cuda") for dev in model_param_devices):
                raise RuntimeError(
                    f"Planner requested CUDA ({self.device}) but model parameters are on {sorted(model_param_devices)}. "
                    "Refusing a silent CPU closed-loop run."
                )
        cuda_alloc_mb = 0.0
        cuda_reserved_mb = 0.0
        if str(self.device).startswith("cuda"):
            try:
                import torch

                cuda_alloc_mb = float(torch.cuda.memory_allocated(self.device)) / (1024.0 ** 2)
                cuda_reserved_mb = float(torch.cuda.memory_reserved(self.device)) / (1024.0 ** 2)
            except Exception:
                pass
        print(
            f"[planner-ready] variant={variant} logical_device={self.device} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
            f"params={params} param_devices={sorted(model_param_devices)} "
            f"cuda_alloc={cuda_alloc_mb:.1f}MB cuda_reserved={cuda_reserved_mb:.1f}MB "
            f"shared_model={share_model} reused={reused_model} checkpoint={'yes' if checkpoint else 'no'}",
            flush=True,
        )
        self.core = BDSEPlannerCore(model=model, cfg=cfg, inference_lock=inference_lock)
        _register_closed_loop_profile_flush()
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
        # A nuPlan planner instance can be reused across scenarios.  The V50
        # one-shot intervention state is scenario-local and must never leak.
        self.core.reset_selected_outcome_probe()
        self._cached_local_trajectory = None
        self._cached_action_index = 0
        self._cached_replan_iteration_index = None
        self._cached_replan_time_s = None
        self._cached_replan_ego_pose = None

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
            diag_payload = diagnostics
            if _env_true("BDSE_SELECTED_OUTCOME_DIAG_ONLY", default=False):
                # V50 consumes only the selected-outcome probe certificate.
                # Serializing the complete selector/tournament/runtime-safety
                # diagnostics every tick creates substantial JSON/I/O overhead
                # and is scientifically redundant for this evidence collector.
                diag_payload = {
                    "selected_outcome_probe": (diagnostics.get("selected_outcome_probe", {}) or {}),
                }
                if _env_true("BDSE_V50_TIMING_TELEMETRY", default=False):
                    diag_payload["v50_timing"] = {
                        "timing": (diagnostics.get("timing", {}) or {}),
                        "timing_core": (diagnostics.get("timing_core", {}) or {}),
                        "model_timing": (diagnostics.get("model_timing", {}) or {}),
                        "cached_plan": bool(diagnostics.get("cached_plan", False)),
                    }
            row = {
                "planner": self._name,
                "scenario_token": str(getattr(self, "_diagnostic_scenario_token", "") or ""),
                "iteration_index": int(getattr(iteration, "index", -1)) if iteration is not None else -1,
                "time_s": float(getattr(iteration, "time_s", 0.0)) if iteration is not None else 0.0,
                "action_index": int(action),
                "diagnostics": _json_safe(diag_payload),
            }
            if _env_true("BDSE_REQUIRE_SCENARIO_FOR_DIAG", default=False) and not row["scenario_token"]:
                raise RuntimeError("V50 batched diagnostics require the native nuPlan scenario token")
            path = Path(diag_path).expanduser()
            _append_diag_line(path, json.dumps(row, sort_keys=True))
        except Exception as exc:
            # Never silently lose the runtime diagnostics used to validate the
            # safety/progress mechanism.  Hydra can change the worker cwd, so the
            # runner supplies an absolute path; any remaining failure is recorded
            # next to that path and can optionally fail the simulation.
            try:
                err_path = Path(str(diag_path) + ".error.log").expanduser()
                err_path.parent.mkdir(parents=True, exist_ok=True)
                with err_path.open("a", encoding="utf-8") as f:
                    f.write(f"{type(exc).__name__}: {exc}\n")
            except Exception:
                pass
            if os.environ.get("BDSE_STRICT_CLOSED_LOOP_DIAG", "0").lower() in {"1", "true", "yes", "on"}:
                raise

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
            _record_closed_loop_profile(self._name, diagnostics)
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
        _record_closed_loop_profile(self._name, diagnostics)
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
