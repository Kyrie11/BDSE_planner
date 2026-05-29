# BDSE nuPlan implementation

This repository implements **Budgeted Decision-Sufficient Evidence (BDSE)** for nuPlan-style candidate-set planning.  The code in this package has been aligned with the paper's deployment constraint: the runtime planner may use current observations, route/map context, candidate rollouts, cheap atom features, Top-M proposal, sparse action-atom queries, greedy budgeted certificate selection, and a budgeted tournament.  It must not access logged future ego/agent labels inside `compute_trajectory()`.

## What was fixed in this version

The implementation now addresses the main paper/code mismatches that existed in the earlier package:

- **P0 cache/schema repair:** `PairLabels` and `Sample` dataclasses are restored, pair-label invariants are validated, and the package compiles/imports cleanly.
- **Runtime map/history adapter:** the nuPlan runtime adapter now tries to recover route/map/stop-line/drivable context from `initialization.map_api`, enriches red-light geometry, builds tracked agent history from observation history when available, rotates agent velocities into the ego frame, and sorts/pads agents deterministically instead of repeating the current frame blindly.
- **Open-loop uses the real planner core:** evaluation now runs `BDSEPlannerCore.plan_from_components(...)`, so fallback re-query stages, rule reranking, query accounting, and planner diagnostics are included.
- **Full-interface offline diagnostics:** `predict_dense_numpy(...)` is available for validation-only dense evidence scoring, while runtime still uses sparse Top-M/action queries.
- **BDSE metrics and ablations:** metrics now report decisive-rival recall, proposal/critical-atom recall, effective query counts, fallback/rerank rates, latency, and full-interface action match; the ablation runner can generate and optionally execute budget/proposal/rival/family/fallback/demo/drivable sweeps.
- **Strict teacher feasibility:** robust teacher labels use lexicographic hard feasibility through owned hard evidence atoms. Collision, off-drivable, wrong-way, and red-light events are no longer collapsed into one soft boolean.
- **Partition-preserving hard costs:** hard priorities are injected into the unique owning hard atom before normalization, so `J_T = J_base_T + sum_i g_i_T` still holds and residual margin labels remain closed.
- **Runtime sparse path:** deployment uses Top-M evidence proposal, base/cheap rival pair screening, sparse `q_i(a)` queries only for Top-M atoms and screened actions, greedy certificate selection, and tournament scoring.
- **No dense leakage into `L_act`:** training may still use dense offline query features for residual supervision, but the action loss masks unqueried atom-action entries and follows the same sparse pathway as deployment.
- **Proposal head includes `u(A_t)`:** atom proposal now receives a candidate-bank summary containing score entropy, top gap, near-tie fraction, progress/lateral/speed summaries, and maneuver coverage.
- **Oracle proposal targets:** proposal supervision now uses greedy marginal certificate gain rather than mean positive support.
- **Padding-safe scene encoder:** invalid agent/map/route/traffic/goal tokens are passed through a transformer key-padding mask, so padded token values do not affect the scene embedding.
- **Richer action/evidence features:** action encoding includes route-relative proxy progress, lateral offset, lateral envelope, and heading change; evidence tensors encode anchor geometry, lambda, budget, type/family, and cheap proposal features.
- **Fallback re-queries:** low-confidence or unsafe-certificate stages rerun proposal/query/selection/tournament with expanded rival size, proposal size, and budget before rule reranking or conservative fallback.
- **LCB diagnostics and calibration hook:** tournament diagnostics include `safety_lcb_min`; `bdse.experiments.calibrate` estimates validation-set `epsilon_cal` for one-sided certificate checks.
- **nuPlan trajectory conversion:** `BDSEnuPlanPlanner` now builds `InterpolatedTrajectory` using global pose, `StateVector2D` velocity, and acceleration instead of silently returning a numpy array in nuPlan environments.
- **Regression tests:** tests now cover sparse runtime behavior, Top-M selection limits, padding invariance, hard-priority teacher behavior, and fallback re-query expansion.

## Repository layout

```text
bdse/
  configs/
    default.yaml          # paper-oriented default runtime/training config
    paper.yaml            # alias/overrides for paper experiments
    smoke.yaml            # fast debugging only
    full_preprocess.yaml  # nuPlan preprocessing overrides
    ablations.yaml
  data/
    cache_schema.py
    state_schema.py               # canonical state-vector indices
    tensorizer.py                 # shared train/runtime tensorization
    feature_builder.py            # offline scenario/cache feature extraction
    label_builder.py              # label-only future and teacher sample construction
    nuplan_runtime_adapter.py     # PlannerInput -> RuntimeFeatures
  planner/
    response_modes.py             # logged/CV/CA/brake/yield/non-yield modes
    robust_teacher.py             # mean+CVaR robust candidate teacher with hard hierarchy
    evidence_atoms.py             # evidence atoms, hard ownership, raw local costs
    evidence_queries.py           # cheap proposal and sparse q_i(a) query features
    pair_screen.py                # runtime base/cheap rival screen
    selector.py                   # oracle and runtime greedy certificate selection
    certificate_selector.py       # sparse selector helper
    tournament.py                 # LCB-aware budgeted tournament
    fallback.py                   # runtime cheap safety flags and rule reranker
    nuplan_planner.py             # runtime planner core and nuPlan adapter
  model/
    scene_encoder.py
    action_encoder.py
    evidence_encoder.py
    bdse_model.py
    losses.py
  experiments/
    preprocess.py
    train.py
    evaluate_open_loop.py
    evaluate_closed_loop.py       # wrapper notes for external nuPlan Hydra integration
    calibrate.py                  # estimate epsilon_cal on validation data
    diagnostics.py
  tests/
```

## Environment

```bash
pip install -U pip
pip install -r requirements.txt
pip install -e .
pip install -e '.[test]'
```

For real nuPlan preprocessing or closed-loop simulation, install the official nuPlan devkit in the same environment:

```bash
pip install -e '.[nuplan]'
```

Set dataset paths before preprocessing/training:

```bash
export NUPLAN_DATA_ROOT=/path/to/nuplan/data/cache
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export NUPLAN_EXP_ROOT=/path/to/nuplan/exp
export BDSE_CACHE_ROOT=/path/to/bdse_cache
```

## Quick validation

Run the regression suite first:

```bash
pytest -q
```

Expected result in this package revision:

```text
26 passed, 1 warning
```

## Recommended experiment command flow

Use `paper.yaml` for the main paper-faithful setup. It enables candidate-aware agent selection, drivable polygons, robust teacher hard-fail checks, Top-M=48, B=16, and the oracle-to-predicted certificate schedule. Use `smoke.yaml` only for quick debugging.

### 1) Dataset preprocessing

```bash
python -m bdse.experiments.preprocess \
  --config bdse/configs/paper.yaml \
  --data-root "$NUPLAN_DATA_ROOT" \
  --maps-root "$NUPLAN_MAPS_ROOT" \
  --map-version nuplan-maps-v1.0 \
  --splits train val \
  --output-dir "$BDSE_CACHE_ROOT" \
  --scenario-stride 10 \
  --scenario-iteration-policy initial \
  --max-samples-per-log 512 \
  --max-samples-per-log-strategy uniform_blocks \
  --max-samples-per-log-block-size 64 \
  --num-workers 6 \
  --max-in-flight 6 \
  --scenario-builder-workers 8 \
  --teacher-cost-eval-stride 1 \
  --resume \
  --candidate-aware-agent-selection \
  --include-drivable-polygons \
  --no-include-crosswalks \
  --cache-local-scheduler \
  --cache-local-log-parallelism 1 \
  --temporal-frame-cache-max-entries 262144 \
  --temporal-frame-cache-individual-miss-threshold 32 \
  --temporal-frame-cache-coalesce-bulk \
  --skip-failed-samples \
  --profile \
  --profile-threshold-s 10.0
```

For a smaller validation-only smoke preprocess, replace `--splits train val` with `--splits val` and add `--max-files` / lower `--max-samples-per-log`.

Preprocessing stores runtime-only features, label-only futures, candidates, evidence atoms, proposal features, teacher labels, and pair labels. Runtime code consumes runtime features, candidate bank, evidence atoms, proposal features, and sparse queries only.

### 2) Dataset diagnostics

```bash
python -m bdse.experiments.diagnostics \
  --config bdse/configs/paper.yaml \
  --split val \
  --preprocessed-dir "$BDSE_CACHE_ROOT" \
  --max-scenarios 500 \
  --output outputs/diagnostics_val.json
```

Use `--recompute-hard-events` for a slower consistency check of hard-event labels.

### 3) Training

```bash
python -m bdse.experiments.train \
  --config bdse/configs/paper.yaml \
  --split train \
  --preprocessed-dir "$BDSE_CACHE_ROOT" \
  --output outputs/bdse_model.pt
```

For a short sanity run:

```bash
python -m bdse.experiments.train \
  --config bdse/configs/smoke.yaml \
  --split train \
  --preprocessed-dir "$BDSE_CACHE_ROOT" \
  --max-scenarios 128 \
  --output outputs/bdse_smoke.pt
```

Training losses:

- `L_base`: regress predicted base cost to `J_base_T`.
- `L_res`: offline residual-margin supervision over the full evidence bank.
- `L_rank`: logistic rank loss over positive teacher pairs.
- `L_prop`: BCE/listwise proposal loss from oracle marginal certificate gain.
- `L_act`: deployment-consistent sparse proposal/query/greedy/tournament action loss with optional oracle-to-predicted certificate schedule.
- `L_cal`: optional margin calibration surrogate when `training.loss_weights.calibration > 0`.

### 4) Post-hoc calibration

```bash
python -m bdse.experiments.calibrate \
  --config bdse/configs/paper.yaml \
  --checkpoint outputs/bdse_model.pt \
  --split val \
  --preprocessed-dir "$BDSE_CACHE_ROOT" \
  --delta 0.1 \
  --output outputs/calibration.json
```

Copy the reported value into your runtime config:

```yaml
tournament:
  epsilon_cal: <epsilon_cal from calibration.json>
fallback:
  safety_lcb_min: 0.0
```

### 5) Open-loop evaluation and BDSE diagnostics

```bash
python -m bdse.experiments.evaluate_open_loop \
  --config bdse/configs/paper.yaml \
  --checkpoint outputs/bdse_model.pt \
  --split val \
  --preprocessed-dir "$BDSE_CACHE_ROOT" \
  --max-scenarios 500 \
  --dense-full-interface \
  --write-details \
  --output outputs/open_loop_bdse_metrics.json
```

`--dense-full-interface` is validation-only. It computes dense evidence predictions for `full_interface_action_match` while the planner path still uses sparse Top-M/action queries and fallback diagnostics.

### 6) Closed-loop evaluation hook

The package constructs a real `BDSEnuPlanPlanner`; your nuPlan project must still provide its Hydra simulation runner. The wrapper accepts a `module:function` runner that receives `planner=...`, `cfg=...`, `challenge=...`, and `output_dir=...`.

```bash
python -m bdse.experiments.evaluate_closed_loop \
  --config bdse/configs/paper.yaml \
  --checkpoint outputs/bdse_model.pt \
  --challenge closed_loop_nonreactive_agents \
  --output-dir outputs/closed_loop_nonreactive \
  --run \
  --runner your_nuplan_project.bdse_runner:run_with_prebuilt_planner
```

To validate/export an already produced nuPlan metric summary:

```bash
python -m bdse.experiments.evaluate_closed_loop \
  --config bdse/configs/paper.yaml \
  --checkpoint outputs/bdse_model.pt \
  --metric-summary /path/to/nuplan_metric_summary.json \
  --output-dir outputs/closed_loop_nonreactive
```

### 7) Ablation sweeps

Generate configs and the plan only:

```bash
python -m bdse.experiments.ablations \
  --config bdse/configs/paper.yaml \
  --output-dir outputs/ablations
```

Run open-loop ablations over budget/proposal/rival size, selector family, evidence-family disables, fallback, demo prior, and drivable-polygons settings:

```bash
python -m bdse.experiments.ablations \
  --config bdse/configs/paper.yaml \
  --output-dir outputs/ablations \
  --run \
  --checkpoint outputs/bdse_model.pt \
  --preprocessed-dir "$BDSE_CACHE_ROOT" \
  --split val \
  --max-scenarios 500
```

## Runtime planner path

Use the planner core directly for runtime-feature tests:

```python
from bdse.config import load_config
from bdse.planner.nuplan_planner import BDSEPlannerCore

cfg = load_config("bdse/configs/full_preprocess.yaml")
core = BDSEPlannerCore(model=model, cfg=cfg)
action_index, local_trajectory, diagnostics = core.plan_from_runtime(runtime_features)
```

Runtime sequence:

```text
RuntimeFeatures
  -> generate_candidate_bank
  -> enumerate_evidence_atoms and cheap proposal features
  -> model/base scorer and proposal logits
  -> Top-M atoms
  -> base/cheap runtime pair screen
  -> sparse q_i(a) for Top-M atoms and screened actions
  -> local action-atom scores only for queried entries
  -> greedy certificate under budget B
  -> LCB-aware tournament
  -> fallback re-query stages if low confidence / unsafe certificate
  -> rule rerank or conservative fallback only if needed
```

`compute_trajectory()` never calls scenario future APIs and never reads logged future ego/agent states.

## nuPlan closed-loop integration

Instantiate the planner:

```python
from bdse.planner.nuplan_planner import BDSEnuPlanPlanner
planner = BDSEnuPlanPlanner(model=model, cfg=cfg)
```

The included `bdse.experiments.evaluate_closed_loop` script constructs the planner, can validate/export existing metric summaries, and can call a project-provided `module:function` runner. Full nuPlan closed-loop evaluation still needs your project’s Hydra `run_simulation` entrypoint to pass the constructed planner instance as a pre-built planner. This repository does not bundle a full Hydra simulation config.

Inside a nuPlan environment, `BDSEnuPlanPlanner.compute_trajectory()` returns `InterpolatedTrajectory` built from global `EgoState` objects with velocity and acceleration vectors. Outside a nuPlan environment, the converter returns the local numpy trajectory for unit tests and lightweight debugging.

## Important deployment notes

1. Keep `candidate.K`, `evidence.max_atoms`, model feature dimensions, and preprocessing config identical between cache generation and training.
2. Keep `evidence.precompute_dense_query_features: false` for runtime; dense query tensors are only for offline supervision and diagnostics.
3. Use validation calibration before trusting LCB-based fallback thresholds.
4. Treat `smoke.yaml` as a speed test, not a paper-quality setting.
5. If maps expose incomplete drivable polygons, keep the route-corridor fallback enabled; otherwise teacher labels may mark all candidates off-drivable.
6. Closed-loop results should always report fallback trigger rate, expanded budget/query count, latency, selected evidence count, and safety-LCB diagnostics together with nuPlan metrics.

## Known limits

- The repository provides the planner and scripts, but not a full nuPlan Hydra experiment bundle.
- Learned uncertainty heads are not implemented; the recommended runtime-safe option is validation quantile calibration through `bdse.experiments.calibrate`.
- Sparse query geometry is vectorized over the requested Top-M atoms and selected action ids. It avoids full `E × K` runtime scoring, but individual atom helpers may reuse candidate-level geometry internally for speed and numerical consistency.
