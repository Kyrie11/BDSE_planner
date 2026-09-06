from __future__ import annotations

"""Process-local nuPlan wrapper for V64.3.54 paired dynamic response collection.

Science-critical properties:
- the historical planner/probe implementation is not edited;
- treatment/control action semantics remain the frozen V50.5 one-shot probe;
- only current simulated ego states are logged;
- simulation is capped immediately after the first scheduled replan following
  the intervention, because V54 reuses the already metric-safe V50.5 full-horizon
  outcome labels and needs no new outcome metric;
- run_metric=false is required by the V54 collector.
"""

import json
import math
import os
import runpy
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_PATCH_MARKER = "_bdse_v64_3_54_dynamic_response_sidecar"
_TIME_PATCH_MARKER = "_bdse_v64_3_54_short_horizon"


def _ego_row(current_input: Any) -> dict[str, Any]:
    try:
        ego, _obs = current_input.history.current_state
        it = current_input.iteration
        rear = ego.rear_axle
        dyn = ego.dynamic_car_state
        speed = float(dyn.speed)
        vals = [float(rear.x), float(rear.y), float(rear.heading), speed]
        if any(not math.isfinite(v) for v in vals):
            raise ValueError("non-finite ego state")
        return {
            "iteration_index": int(it.index),
            "time_us": int(it.time_point.time_us),
            "ego_world": vals,
        }
    except Exception as exc:  # pragma: no cover - server API guard
        raise RuntimeError(f"V54 PDRM cannot extract current simulated ego state: {exc}") from exc


def install_short_horizon() -> None:
    from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import StepSimulationTimeController

    if getattr(StepSimulationTimeController, _TIME_PATCH_MARKER, False):
        return
    exposure = int(os.environ.get("BDSE_V54_EXPOSURE_TICKS", "-1"))
    if exposure < 1:
        raise RuntimeError("V54 PDRM requires BDSE_V54_EXPOSURE_TICKS>=1")
    original = StepSimulationTimeController.number_of_iterations

    def number_of_iterations(self: Any) -> int:
        n = int(original(self))
        # With nuPlan's reached_end condition, N=exposure+2 executes planner
        # iterations 0..exposure inclusive, then terminates.
        return min(n, exposure + 2)

    StepSimulationTimeController.number_of_iterations = number_of_iterations
    setattr(StepSimulationTimeController, _TIME_PATCH_MARKER, True)


def _resolve_nuplan_planner_class():
    """Resolve the actual nuPlan adapter and fail closed on interface drift.

    ``BDSEPlanner`` is only the runtime display name (``self._name``); the
    concrete adapter class exported by ``nuplan_planner.py`` is
    ``BDSEnuPlanPlanner``.  V54 patches the adapter process-locally so the
    historical planner source remains byte-identical.
    """
    from bdse.planner.nuplan_planner import BDSEnuPlanPlanner

    required = (
        "compute_planner_trajectory",
        "_planner_replan_interval_ticks",
        "_to_nuplan_trajectory",
    )
    missing = [name for name in required if not callable(getattr(BDSEnuPlanPlanner, name, None))]
    if missing:
        raise RuntimeError(
            "V54 PDRM nuPlan planner adapter interface drift: "
            + ", ".join(missing)
        )
    return BDSEnuPlanPlanner


def install_dynamic_response_sidecar() -> None:
    PlannerClass = _resolve_nuplan_planner_class()

    if getattr(PlannerClass, _PATCH_MARKER, False):
        return
    original = PlannerClass.compute_planner_trajectory

    def wrapped(self: Any, current_input: Any):
        exposure = int(os.environ.get("BDSE_V54_EXPOSURE_TICKS", "-1"))
        if exposure < 1:
            raise RuntimeError("V54 PDRM missing exposure ticks")
        actual_interval = int(self._planner_replan_interval_ticks())
        if actual_interval != exposure:
            raise RuntimeError(
                f"V54 PDRM operator-window mismatch planner replan_interval_ticks={actual_interval} expected={exposure}"
            )
        pre = _ego_row(current_input)
        idx = int(pre["iteration_index"])
        if idx < 0 or idx > exposure:
            return original(self, current_input)

        # At the terminal sample we need the state immediately *before* the first
        # scheduled replan, not the result of another expensive BDSE replan.  The
        # simulator still expects a trajectory return value for this final loop, so
        # return the already cached local rollout.  Simulation terminates right
        # afterwards and this returned trajectory is never part of the mediator.
        bypass_terminal_replan = bool(idx == exposure)
        if bypass_terminal_replan:
            cached = getattr(self, "_cached_local_trajectory", None)
            if cached is None:
                raise RuntimeError("V54 PDRM terminal state reached without cached one-shot rollout")
            out = self._to_nuplan_trajectory(cached, current_input)
        else:
            out = original(self, current_input)

        target = getattr(self.core, "_pior_bound_target", None)
        if not isinstance(target, dict):
            # At iteration 0 the target is bound inside the original planner call.
            target = getattr(self.core, "_pior_bound_target", None)
        if not isinstance(target, dict):
            raise RuntimeError(f"V54 PDRM missing manifest-bound probe target at iteration={idx}")
        tok = str(target.get("scenario_token", ""))
        if not tok:
            raise RuntimeError("V54 PDRM bound target lacks scenario_token")
        pcfg = ((self.core.cfg.get("selected_outcome_probe", {}) or {}) if isinstance(self.core.cfg, dict) else {})
        arm = str(pcfg.get("arm", "")).strip().lower()
        if arm not in {"treatment", "control"}:
            raise RuntimeError(f"V54 PDRM invalid probe arm {arm!r}")
        diag_path = os.environ.get("BDSE_CLOSED_LOOP_DIAG", "")
        if not diag_path:
            raise RuntimeError("V54 PDRM requires BDSE_CLOSED_LOOP_DIAG for sidecar location")
        path = Path(diag_path).with_name("v54_dynamic_response_events.jsonl")
        row = {
            "scenario_token": tok,
            "arm": arm,
            "iteration_index": idx,
            "time_us": int(pre["time_us"]),
            "ego_world": [float(v) for v in pre["ego_world"]],
            "exposure_ticks": exposure,
            "state_source": "simulated_ego_state_before_planner_compute",
            "post_intervention": bool(idx > 0),
            "terminal_replan_bypassed": bypass_terminal_replan,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return out

    PlannerClass.compute_planner_trajectory = wrapped
    setattr(PlannerClass, _PATCH_MARKER, True)


def main() -> None:
    if os.environ.get("BDSE_V54_DYNAMIC_RESPONSE") != "1":
        raise RuntimeError("V54 PDRM wrapper requires BDSE_V54_DYNAMIC_RESPONSE=1")
    install_short_horizon()
    install_dynamic_response_sidecar()
    print(
        "[BDSE-V54-PDRM] paired dynamic-response sidecar enabled; full outcome metrics disabled; planner/probe science unchanged",
        flush=True,
    )
    runpy.run_module("nuplan.planning.script.run_simulation", run_name="__main__")


if __name__ == "__main__":
    main()
