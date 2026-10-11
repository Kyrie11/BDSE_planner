"""Opt-in nuPlan adapter; original frozen planner and config remain unchanged."""
from __future__ import annotations
import os
from bdse.experiments.planning_budget import instrument_core, measure_call
from bdse.planner.nuplan_planner import BDSEnuPlanPlanner


class ProfiledBDSEnuPlanPlanner(BDSEnuPlanPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._budget_prefix = os.environ.get("BDSE_BUDGET_TRACE_PREFIX", "").strip()
        self._budget_mode = os.environ.get("BDSE_BUDGET_PROFILE_MODE", "native")
        if self._budget_mode not in {"native", "attribution"}:
            raise ValueError("BDSE_BUDGET_PROFILE_MODE must be native or attribution")
        if self._budget_prefix:
            instrument_core(self.core, self._budget_mode == "attribution")
        self._budget_calls = 0

    def compute_planner_trajectory(self, current_input):
        if not self._budget_prefix:
            return super().compute_planner_trajectory(current_input)
        self._budget_calls += 1
        raw = os.environ.get("BDSE_BUDGET_DEADLINE_MS", "").strip()
        initialization = getattr(self, "initialization", None)
        goal = getattr(initialization, "mission_goal", None)
        # Runtime timestamp is simulated time; it is not a real-time arrival.
        iteration = getattr(current_input, "iteration", None)
        context = {"scope": "nuplan_planner_call", "sequence_index": self._budget_calls,
                   "warmup": self._budget_calls <= int(os.environ.get("BDSE_BUDGET_WARMUP_CALLS", "5")),
                   "iteration_index": self._current_iteration_index(current_input),
                   "simulation_time_s": getattr(iteration, "time_s", None),
                   "stream_id": str(id(self)), "budget": self.core.cfg.get("evidence", {}).get("budget"),
                   "proposal_top_m": self.core.cfg.get("selector", {}).get("proposal_top_m"),
                   "has_mission_goal": goal is not None,
                   "includes_sensor_pipeline": False,
                   "includes_runtime_feature_build": True,
                   "includes_nuplan_output_conversion": True,
                   "includes_profile_trace_write": False}
        result, _ = measure_call(lambda: super(ProfiledBDSEnuPlanPlanner, self).compute_planner_trajectory(current_input),
                                 device=self.device, mode=self._budget_mode, context=context,
                                 deadline_ms=float(raw) if raw else None,
                                 output_prefix=self._budget_prefix)
        return result
