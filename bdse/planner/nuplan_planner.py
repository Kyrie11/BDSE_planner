from __future__ import annotations

from typing import Any

import numpy as np

from bdse.config import load_config
from bdse.utils import angle_wrap
from bdse.data.cache_schema import RuntimeFeatures
from bdse.data.nuplan_runtime_adapter import build_runtime_features_from_planner_input
from bdse.planner.candidate_generator import generate_candidate_bank
from bdse.planner.evidence_atoms import enumerate_evidence_atoms
from bdse.planner.evidence_queries import compute_query_features_for_pairs
from bdse.planner.hab import family_ids_from_atoms, select_topm_atoms_hab
from bdse.planner.fallback import runtime_safety_flags_from_runtime
from bdse.planner.pair_screen import build_runtime_pairs_from_base, build_rival_sets_from_base
from bdse.planner.selector import runtime_greedy_selector, runtime_greedy_selector_pair_conditioned
from bdse.planner.tournament import run_tournament, run_pair_conditioned_tournament, selected_pair_sigma_from_action_variance


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
                gamma_max=float(sel_cfg.get("gamma_max_default", 100.0)),
                eta_pred=float(sel_cfg.get("eta_pred", 1.0)),
                atom_active_mask=atom_active,
                pair_atom_variance=pred.get("pair_atom_var", None),
                beta_uncertainty=float(tour_cfg.get("beta_uncertainty", 0.0)),
                epsilon_cal=float(tour_cfg.get("epsilon_cal", stage_cfg.get("calibration", {}).get("epsilon_cal", 0.0))),
                lambda_info=float(sel_cfg.get("lambda_info", 0.0)),
                prior_atom_variance=sel_cfg.get("unqueried_atom_variance", None),
                family_ids=family_ids,
                family_budget_caps=family_caps,
            )
            tournament = run_pair_conditioned_tournament(
                J0,
                pred.get("rival_pair_atom_delta", pred["pair_atom_delta"]),
                pred.get("rival_pair_indices", pred["pair_indices"]),
                selection.selected,
                candidates.valid_mask,
                runtime_flags,
                stage_cfg,
                pair_atom_variance=pred.get("rival_pair_atom_var", pred.get("pair_atom_var", None)),
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
                bidirectional_pairs=bool(sel_cfg.get("bidirectional_pairs", True)),
                reverse_pair_weight=float(sel_cfg.get("reverse_pair_weight", 1.0)),
                pair_cap_multiplier=float(sel_cfg.get("runtime_pair_cap_multiplier", 1.0)),
            )
            sigma = selected_pair_sigma_from_action_variance(g_var, selection.selected, candidates.valid_mask)
            tournament = run_tournament(J0, g, selection.selected, candidates.valid_mask, runtime_flags, stage_cfg, sigma=sigma)
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
        candidates = generate_candidate_bank(runtime, self.cfg)
        evidence_bank = enumerate_evidence_atoms(runtime, candidates, self.cfg)
        base_budget = int(self.cfg.get("evidence", {}).get("budget", 16))
        base_M = int(self.cfg.get("selector", {}).get("proposal_top_m", max(2 * base_budget, base_budget + 1)))
        base_L = int(self.cfg.get("tournament", {}).get("L_infer", 16))
        stages: list[tuple[str, dict[str, Any]]] = [("base", self._stage_cfg(base_budget, base_M, base_L))]
        fcfg = self.cfg.get("fallback", {})
        if bool(fcfg.get("enabled", True)):
            L_stages = list(fcfg.get("rival_stages", [base_L, min(31, max(candidates.K - 1, 1))]))
            B_stages = list(fcfg.get("budget_stages", [base_budget, min(int(self.cfg.get("evidence", {}).get("max_atoms", 128)), max(base_budget * 2, base_budget + 1))]))
            for L in L_stages:
                for B in B_stages:
                    M = int(max(int(self.cfg.get("selector", {}).get("proposal_top_m", base_M)), min(int(self.cfg.get("evidence", {}).get("max_atoms", 128)), int(float(fcfg.get("proposal_multiplier", 3.0)) * int(B)))))
                    name = f"fallback_L{int(L)}_B{int(B)}_M{int(M)}"
                    cfg_stage = self._stage_cfg(int(B), int(M), int(L))
                    if name != "base":
                        stages.append((name, cfg_stage))
        best = None
        stage_records = []
        triggered = False
        for idx, (stage_name, cfg_stage) in enumerate(stages):
            pred, selection, tournament, atom_active = self._run_certificate_stage(runtime, candidates, evidence_bank, cfg_stage)
            qdiag = runtime_query_diagnostics(pred, selection.selected)
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
        runtime_flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg_stage)
        if triggered and bool(fcfg.get("rule_rerank_top_k", 5)):
            from bdse.planner.fallback import rule_based_runtime_scores, conservative_fallback_action
            top_k = int(fcfg.get("rule_rerank_top_k", 5))
            top_actions = [int(a) for a in np.argsort(-tournament.scores)[:top_k] if candidates.valid_mask[int(a)]]
            rule_cost = rule_based_runtime_scores(runtime, candidates, cfg_stage)
            safe_top = [a for a in top_actions if not runtime_flags[a]]
            if safe_top:
                best_rule = min(safe_top, key=lambda a: (float(rule_cost[a]), a))
                if float(rule_cost[best_rule]) + float(fcfg.get("rule_switch_margin", 0.0)) < float(rule_cost[action]) or runtime_flags[action]:
                    action = int(best_rule)
                    stage_name = stage_name + "+rule_rerank"
            elif runtime_flags[action]:
                action = int(conservative_fallback_action(candidates))
                stage_name = stage_name + "+conservative"
        trajectory = candidates.trajectories[action]
        qdiag = runtime_query_diagnostics(pred, selection.selected)
        diagnostics = {
            "action_index": action,
            "selected_atoms": selection.selected,
            "proposal_top_m_atoms": list(map(int, np.asarray(pred.get("top_m_atoms", []), dtype=np.int64).tolist())),
            "queried_actions": list(map(int, np.asarray(pred.get("queried_actions", []), dtype=np.int64).tolist())),
            **qdiag,
            "hab": pred.get("hab_diagnostics", {}),
            "selector": selection.diagnostics,
            "tournament": tournament.diagnostics,
            "fallback_stage": stage_name,
            "fallback_triggered": bool(triggered),
            "fallback_stage_records": stage_records,
        }
        return action, trajectory, diagnostics


class BDSEnuPlanPlanner:
    def __init__(
        self,
        model: Any | None = None,
        cfg: dict[str, Any] | None = None,
        checkpoint: str | None = None,
        config_path: str | None = None,
        device: str = "cpu",
    ):
        cfg = cfg or load_config(config_path)
        if model is None and checkpoint:
            import torch
            from bdse.model.bdse_model import BDSEModel

            model = BDSEModel(cfg)
            ckpt = torch.load(checkpoint, map_location=device)
            state = ckpt.get("model", ckpt)
            current = model.state_dict()
            compatible = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
            model.load_state_dict(compatible, strict=False)
            model.to(torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu"))
            model.eval()
        self.core = BDSEPlannerCore(model=model, cfg=cfg)
        self._name = "BDSEPlanner"

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

    def compute_trajectory(self, current_input: Any):
        runtime = self._runtime_from_planner_input(current_input)
        _, trajectory, _ = self.core.plan_from_runtime(runtime)
        return self._to_nuplan_trajectory(trajectory, current_input)

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
        try:
            state_mod = __import__("nuplan.common.actor_state.ego_state", fromlist=["EgoState"])
            traj_mod = __import__("nuplan.planning.simulation.trajectory.interpolated_trajectory", fromlist=["InterpolatedTrajectory"])
            time_mod = __import__("nuplan.common.actor_state.state_representation", fromlist=["TimePoint", "StateSE2", "StateVector2D"])
            EgoState = getattr(state_mod, "EgoState")
            InterpolatedTrajectory = getattr(traj_mod, "InterpolatedTrajectory")
            TimePoint = getattr(time_mod, "TimePoint")
            StateSE2 = getattr(time_mod, "StateSE2")
            StateVector2D = getattr(time_mod, "StateVector2D")
            ego_state = getattr(current_input, "history", None)
            last = ego_state.ego_states[-1] if ego_state is not None and getattr(ego_state, "ego_states", None) else None
            start_us = int(getattr(getattr(last, "time_point", None), "time_us", 0))
            rear = getattr(last, "rear_axle", last)
            ox = float(getattr(rear, "x", 0.0))
            oy = float(getattr(rear, "y", 0.0))
            oyaw = float(getattr(rear, "heading", 0.0))
            c = float(np.cos(oyaw))
            s = float(np.sin(oyaw))
            states = []
            traj_arr = np.asarray(trajectory, dtype=np.float32)
            # Estimate local longitudinal velocity and acceleration for nuPlan state dynamics.
            local_vx = traj_arr[:, 3] * np.cos(traj_arr[:, 2])
            local_vy = traj_arr[:, 3] * np.sin(traj_arr[:, 2])
            t = traj_arr[:, 4]
            dt = np.maximum(np.gradient(t), 1e-3) if len(t) > 1 else np.asarray([0.1], dtype=np.float32)
            local_ax = np.gradient(local_vx) / dt if len(t) > 1 else np.asarray([0.0], dtype=np.float32)
            local_ay = np.gradient(local_vy) / dt if len(t) > 1 else np.asarray([0.0], dtype=np.float32)
            for k, row in enumerate(traj_arr):
                # Candidate trajectories are represented in the ego-local frame. nuPlan
                # expects global SE2 states in closed-loop simulation.
                lx, ly = float(row[0]), float(row[1])
                gx = ox + c * lx - s * ly
                gy = oy + s * lx + c * ly
                gyaw = float(angle_wrap(oyaw + float(row[2])))
                t_us = start_us + int(float(row[4]) * 1e6)
                if hasattr(EgoState, "build_from_rear_axle"):
                    lvx, lvy = float(local_vx[k]), float(local_vy[k])
                    lax, lay = float(local_ax[k]), float(local_ay[k])
                    gvx = c * lvx - s * lvy
                    gvy = s * lvx + c * lvy
                    gax = c * lax - s * lay
                    gay = s * lax + c * lay
                    vehicle_params = getattr(getattr(self, "initialization", None), "vehicle_parameters", None)
                    if vehicle_params is None and last is not None:
                        vehicle_params = getattr(last, "car_footprint", None)
                        vehicle_params = getattr(vehicle_params, "vehicle_parameters", None)
                    state = EgoState.build_from_rear_axle(
                        rear_axle_pose=StateSE2(gx, gy, gyaw),
                        rear_axle_velocity_2d=StateVector2D(gvx, gvy),
                        rear_axle_acceleration_2d=StateVector2D(gax, gay),
                        tire_steering_angle=0.0,
                        time_point=TimePoint(t_us),
                        vehicle_parameters=vehicle_params,
                    )
                else:
                    state = np.asarray([gx, gy, gyaw, row[3], row[4]], dtype=np.float32)
                states.append(state)
            return InterpolatedTrajectory(states)
        except ImportError:
            # Unit-test / non-nuPlan environments may not install nuPlan.  In a real
            # nuPlan simulation this branch is not taken; callers can still inspect the
            # local-frame trajectory array.
            return trajectory
        except Exception as exc:
            raise RuntimeError(f"Failed to convert BDSE local trajectory to nuPlan InterpolatedTrajectory: {exc}") from exc
