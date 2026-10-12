"""nuPlan adapter that models delayed publication of a newly computed plan.

The simulator is synchronous, so wall-clock inference does not automatically
advance simulated vehicle state.  This adapter turns measured (or explicitly
specified) planning latency into an actuation contract: the incumbent trajectory
is followed during the delay and the new trajectory is aligned at the release
time.  It is an evaluation model, not a hard real-time scheduler.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np

from bdse.planner.nuplan_planner import BDSEnuPlanPlanner, _record_closed_loop_profile
from bdse.planner.latency_model import advance_local_trajectory, stitch_delayed_plan


class LatencyAwareBDSEnuPlanPlanner(BDSEnuPlanPlanner):
    """Evaluate the same planner with an explicit plan-publication delay."""

    def compute_planner_trajectory(self, current_input: Any):
        t0 = time.perf_counter()
        idx = self._current_iteration_index(current_input)
        _, now_s = self._current_ego_pose_time(current_input)
        runtime = self._runtime_from_planner_input(current_input)
        t1 = time.perf_counter()
        proposed_action, proposed, diagnostics = self.core.plan_from_runtime(runtime, current_input=current_input)
        t2 = time.perf_counter()
        measured_ms = 1000.0 * (t2 - t1)
        configured = os.environ.get("BDSE_SIMULATED_DECISION_LATENCY_MS", "").strip()
        delay_ms = float(configured) if configured else measured_ms
        if not math.isfinite(delay_ms) or delay_ms < 0:
            raise ValueError("BDSE_SIMULATED_DECISION_LATENCY_MS must be finite and non-negative")
        max_age_raw = os.environ.get("BDSE_MAX_PLAN_AGE_MS", "").strip()
        max_age_ms = float(max_age_raw) if max_age_raw else None

        incumbent = getattr(self, "_cached_local_trajectory", None)
        incumbent_action = int(getattr(self, "_cached_action_index", proposed_action))
        previous_time = getattr(self, "_cached_replan_time_s", None)
        if incumbent is not None and now_s is not None and previous_time is not None:
            incumbent = advance_local_trajectory(incumbent, max(0.0, float(now_s) - float(previous_time)))

        warm_start = incumbent is None
        late_invalid = bool(max_age_ms is not None and delay_ms > max_age_ms and incumbent is not None)
        if incumbent is None:
            published = np.asarray(proposed, dtype=np.float32)
            published_action = int(proposed_action)
            publication = "warm_start_proposal"
        elif late_invalid:
            published = np.asarray(incumbent, dtype=np.float32)
            published_action = incumbent_action
            publication = "retain_incumbent_late_proposal"
        else:
            published = stitch_delayed_plan(incumbent, proposed, delay_ms / 1000.0)
            published_action = int(proposed_action)
            publication = "delayed_aligned_proposal"

        out = self._to_nuplan_trajectory(published, current_input)
        t3 = time.perf_counter()
        self._cached_local_trajectory = published.copy()
        self._cached_action_index = int(published_action)
        self._cached_replan_iteration_index = int(idx)
        self._cached_replan_ego_pose, self._cached_replan_time_s = self._current_ego_pose_time(current_input)
        diagnostics = dict(diagnostics)
        timing = dict(diagnostics.get("timing", {}))
        timing.update({
            "runtime_from_planner_input_s": float(t1 - t0),
            "core_plan_s": float(t2 - t1),
            "to_nuplan_trajectory_s": float(t3 - t2),
            "compute_planner_trajectory_total_s": float(t3 - t0),
        })
        diagnostics.update({
            "timing": timing, "cached_plan": False,
            "latency_aware_closed_loop": True,
            "measured_core_latency_ms": measured_ms,
            "simulated_decision_latency_ms": delay_ms,
            "max_plan_age_ms": max_age_ms,
            "late_invalid_intervention": late_invalid,
            "latency_warm_start": warm_start,
            "proposal_action_index": int(proposed_action),
            "published_action_index": int(published_action),
            "publication_decision": publication,
        })
        _record_closed_loop_profile(self._name, diagnostics)
        self._write_closed_loop_diag(current_input, int(published_action), diagnostics)
        return out

