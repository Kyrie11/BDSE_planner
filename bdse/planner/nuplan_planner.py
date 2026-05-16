from __future__ import annotations

from typing import Any

import numpy as np

from bdse.config import load_config
from bdse.utils import angle_wrap
from bdse.data.cache_schema import RuntimeFeatures
from bdse.data.feature_builder import build_runtime_features_from_scenario
from bdse.planner.candidate_generator import generate_candidate_bank
from bdse.planner.evidence_atoms import enumerate_evidence_atoms
from bdse.planner.fallback import apply_fallback_if_needed, runtime_safety_flags_from_runtime
from bdse.planner.selector import runtime_greedy_selector
from bdse.planner.tournament import run_tournament


class BDSEPlannerCore:
    def __init__(self, model: Any | None = None, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or load_config()
        self.model = model

    def _predict_costs(self, runtime: RuntimeFeatures, candidates, evidence_bank) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            # Deterministic runtime-only fallback predictor for smoke tests and rule-only deployment.
            K = candidates.K
            J0 = np.square(candidates.trajectories[:, :, 1]).mean(axis=1).astype(np.float32)
            J0 += -0.05 * candidates.trajectories[:, -1, 0].astype(np.float32)
            g = np.zeros((evidence_bank.E, K), dtype=np.float32)
            for ei, atom in enumerate(evidence_bank.atoms):
                q = evidence_bank.query_features[ei]
                if atom.family == "interaction":
                    g[ei] = np.maximum(0.0, 5.0 - q[:, 0])
                elif atom.type == "red_light":
                    g[ei] = 50.0 * q[:, 7]
                elif atom.type == "drivable_area":
                    g[ei] = q[:, 6]
                elif atom.family == "kinematic":
                    g[ei] = 0.1 * (q[:, 9] + q[:, 10] + q[:, 11])
            g[:, ~candidates.valid_mask] = 0.0
            J0[~candidates.valid_mask] = np.inf
            return J0, g
        if hasattr(self.model, "predict_numpy"):
            return self.model.predict_numpy(runtime, candidates, evidence_bank)
        raise TypeError("BDSEPlannerCore model must expose predict_numpy(runtime,candidates,evidence_bank)")

    def plan_from_runtime(self, runtime: RuntimeFeatures) -> tuple[int, np.ndarray, dict[str, Any]]:
        candidates = generate_candidate_bank(runtime, self.cfg)
        evidence_bank = enumerate_evidence_atoms(runtime, candidates, self.cfg)
        J0, g = self._predict_costs(runtime, candidates, evidence_bank)
        runtime_flags = runtime_safety_flags_from_runtime(runtime, candidates, self.cfg)
        sel_cfg = self.cfg.get("selector", {})
        selection = runtime_greedy_selector(
            J0,
            g,
            evidence_bank.budget_costs(),
            candidates.valid_mask,
            runtime_flags,
            budget=float(self.cfg.get("evidence", {}).get("budget", 16)),
            L_infer=int(self.cfg.get("tournament", {}).get("L_infer", 16)),
            gamma_max=float(sel_cfg.get("gamma_max_default", 100.0)),
            eta_pred=float(sel_cfg.get("eta_pred", 1.0)),
            lambda_near=float(sel_cfg.get("lambda_near", 1.0)),
            lambda_safety=float(sel_cfg.get("lambda_safety", 2.0)),
            atom_active_mask=evidence_bank.active_mask,
        )
        tournament = run_tournament(J0, g, selection.selected, candidates.valid_mask, runtime_flags, self.cfg)
        fallback = apply_fallback_if_needed(runtime, candidates, J0, g, evidence_bank.budget_costs(), evidence_bank.active_mask, tournament, self.cfg)
        action = int(fallback.action_index)
        trajectory = candidates.trajectories[action]
        diagnostics = {
            "action_index": action,
            "selected_atoms": selection.selected,
            "selector": selection.diagnostics,
            "tournament": fallback.tournament.diagnostics,
            "fallback_stage": fallback.stage,
            "fallback_triggered": fallback.triggered,
        }
        return action, trajectory, diagnostics


class BDSEnuPlanPlanner:
    def __init__(self, model: Any | None = None, cfg: dict[str, Any] | None = None):
        self.core = BDSEPlannerCore(model=model, cfg=cfg)
        self._name = "BDSEPlanner"

    def name(self) -> str:
        return self._name

    def initialize(self, initialization: Any) -> None:
        self.initialization = initialization

    def compute_trajectory(self, current_input: Any):
        runtime = self._runtime_from_planner_input(current_input)
        _, trajectory, _ = self.core.plan_from_runtime(runtime)
        return self._to_nuplan_trajectory(trajectory, current_input)

    def _runtime_from_planner_input(self, current_input: Any) -> RuntimeFeatures:
        scenario = getattr(current_input, "scenario", None)
        iteration = int(getattr(getattr(current_input, "iteration", None), "index", 0))
        if scenario is not None:
            return build_runtime_features_from_scenario(scenario, iteration, self.core.cfg)
        if isinstance(current_input, RuntimeFeatures):
            return current_input
        if hasattr(current_input, "runtime_features"):
            return current_input.runtime_features
        raise TypeError("BDSEnuPlanPlanner.compute_trajectory expects nuPlan PlannerInput or RuntimeFeatures")

    def _to_nuplan_trajectory(self, trajectory: np.ndarray, current_input: Any):
        try:
            state_mod = __import__("nuplan.common.actor_state.ego_state", fromlist=["EgoState"])
            traj_mod = __import__("nuplan.planning.simulation.trajectory.interpolated_trajectory", fromlist=["InterpolatedTrajectory"])
            time_mod = __import__("nuplan.common.actor_state.state_representation", fromlist=["TimePoint", "StateSE2"])
            EgoState = getattr(state_mod, "EgoState")
            InterpolatedTrajectory = getattr(traj_mod, "InterpolatedTrajectory")
            TimePoint = getattr(time_mod, "TimePoint")
            StateSE2 = getattr(time_mod, "StateSE2")
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
            for row in trajectory:
                # Candidate trajectories are represented in the ego-local frame. nuPlan
                # expects global SE2 states in closed-loop simulation.
                lx, ly = float(row[0]), float(row[1])
                gx = ox + c * lx - s * ly
                gy = oy + s * lx + c * ly
                gyaw = float(angle_wrap(oyaw + float(row[2])))
                t_us = start_us + int(float(row[4]) * 1e6)
                if hasattr(EgoState, "build_from_rear_axle"):
                    state = EgoState.build_from_rear_axle(
                        rear_axle_pose=StateSE2(gx, gy, gyaw),
                        rear_axle_velocity_2d=None,
                        rear_axle_acceleration_2d=None,
                        tire_steering_angle=0.0,
                        time_point=TimePoint(t_us),
                        vehicle_parameters=getattr(getattr(self, "initialization", None), "vehicle_parameters", None),
                    )
                else:
                    state = np.asarray([gx, gy, gyaw, row[3], row[4]], dtype=np.float32)
                states.append(state)
            return InterpolatedTrajectory(states)
        except Exception:
            return trajectory
