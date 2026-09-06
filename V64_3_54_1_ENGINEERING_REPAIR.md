# V64.3.54.1 engineering repair — nuPlan adapter binding

## Failure

The original V54 wrapper imported `BDSEPlanner` from `bdse.planner.nuplan_planner`.
That symbol does not exist. The concrete Hydra/nuPlan adapter is
`BDSEnuPlanPlanner`; `BDSEPlanner` is only the instance display name stored in
`self._name`.

The failure therefore occurs before any scenario simulation and carries no V54
scientific evidence.

## Repair

- `nuplan_v54_dynamic_response_run_simulation.py` now resolves
  `BDSEnuPlanPlanner`.
- It fail-closes if the adapter no longer exposes
  `compute_planner_trajectory`, `_planner_replan_interval_ticks`, or
  `_to_nuplan_trajectory`.
- Two V54 regression checks now verify the actual adapter symbol/interface and
  the process-local sidecar installation.
- `V64_3_54_SCIENCE_MANIFEST.sha256` is regenerated so the original launcher
  keeps its result-defining source lock.

## Scientific invariance

No algorithm or experiment semantics changed: RSMR, V52 effect support, V50.5
paired labels, the 502-scene population, treatment/control configs, exposure
window, dynamic profile definition, folds, loss, calibration, and preregistered
gates are unchanged. The original command remains valid.

A failed/partial previous V54 run may be rerun directly. The launcher uses
resume certificates; invalid/incomplete batch directories are replaced before
collection.
