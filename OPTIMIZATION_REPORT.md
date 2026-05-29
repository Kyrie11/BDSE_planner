# BDSE code optimization report

## What I changed

This optimized revision follows the paper/diagnosis priorities: first make the repository importable, then align runtime, evaluation, metrics, and ablations with the BDSE pipeline.

### P0 / correctness

- Restored `PairLabels` and `Sample` in `bdse/data/cache_schema.py`.
- Added `PairLabels.validate_positive_direction()` to catch corrupt pair labels early.
- Verified all Python sources compile with `python -m py_compile $(find bdse -name '*.py')`.
- Verified the regression suite: `26 passed, 1 warning`.

### Runtime adapter

- Rewrote `bdse/data/nuplan_runtime_adapter.py` to make deployment features closer to paper assumptions:
  - uses nuPlan map API when available;
  - reuses offline-style map extraction for route centerline, stop-line / traffic-light geometry, and drivable polygons;
  - transforms ego/agent states into the current ego frame consistently;
  - rotates global object velocity into ego-local coordinates;
  - reconstructs tracked-object history from `PlannerInput.history.observations` when available instead of repeating the current frame;
  - pads only selected agents as valid and records metadata about missing history.

### Planner / evaluation consistency

- Added `BDSEPlannerCore.plan_from_components(...)` so open-loop evaluation can use cached candidates/evidence while still running the same runtime proposal, sparse query, greedy certificate, tournament, and fallback path.
- Added richer planner diagnostics: selected atoms, Top-M atoms, queried actions, sparse query counts, selected-certificate query counts, tournament comparison counts, rival sets, fallback stages, and latency.
- Updated `evaluate_open_loop.py` to run the full planner core and optionally compute validation-only dense full-interface predictions.

### Metrics

- Reworked `bdse/metrics/bdse_metrics.py` to include:
  - correct `effective_query_count` based on selected atoms × queried actions;
  - `sparse_query_count` and `selected_certificate_query_count`;
  - `decisive_rival_recall` and `symmetric_decisive_rival_recall`;
  - proposal recall / selected critical-atom recall;
  - fallback/rule-rerank/conservative-fallback rates;
  - latency and expanded fallback-stage budget/rival sizes;
  - dense full-interface action match when available.

### Training robustness

- Removed a no-op NumPy assignment in `losses.py`.
- Added an optional oracle-to-predicted certificate schedule for `L_act` through `training.certificate_schedule`.
- Updated the training loop to pass `current_epoch` into loss computation.
- Made robust teacher fallback explicit: paper config can hard-fail when robust response-mode costs error instead of silently falling back to a single-world teacher.

### Model / candidate hygiene

- Added safe-fallback maneuver coverage to the candidate-set summary (`maneuver_id == 6`).
- Added `BDSEModel.predict_dense_numpy(...)` for validation-only full-interface diagnostics.
- Zeroed invalid padded candidate trajectories instead of copying the first valid trajectory.

### Configs / ablations / closed-loop hook

- Updated `paper.yaml` and `full_preprocess.yaml` toward the recommended paper-faithful setup: drivable polygons on, candidate-aware agent selection, Top-M=48, softmin tau=0.5, B=16, robust teacher hard-fail, weak demo prior.
- Replaced `ablations.py` with a runner that generates configs and can run open-loop sweeps for budget, proposal size, rival size, selector mode, evidence family, fallback, demo prior, and drivable polygons.
- Updated `evaluate_closed_loop.py` so it constructs a planner, can validate/export existing metric summaries, and can call a project-provided nuPlan runner via `module:function`.
- Updated `README.md` with the recommended preprocessing → diagnostics → training → calibration → evaluation → ablation flow.

## Validation run

```bash
cd /mnt/data/bdse_work/bdse
python -m py_compile $(find bdse -name '*.py')
pytest -q
```

Result:

```text
26 passed, 1 warning in ~7s
```

The warning is from PyTorch transformer nested-tensor settings in an existing padding-invariance test and does not indicate a failing test.

## Remaining limitation

The repository still does not bundle a full nuPlan Hydra closed-loop experiment config. I added a robust `--runner module:function` hook and metric-summary validation path, but a real closed-loop run still requires your nuPlan project’s simulation entrypoint to accept and run the constructed `BDSEnuPlanPlanner`.
