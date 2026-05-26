# BDSE nuPlan implementation

This repository implements **Budgeted Decision-Sufficient Evidence (BDSE)** for nuPlan-style candidate-set planning.  The code in this package has been aligned with the paper's deployment constraint: the runtime planner may use current observations, route/map context, candidate rollouts, cheap atom features, Top-M proposal, sparse action-atom queries, greedy budgeted certificate selection, and a budgeted tournament.  It must not access logged future ego/agent labels inside `compute_trajectory()`.

## What was fixed in this version

The implementation now addresses the main paper/code mismatches that existed in the earlier package:

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
```

## Quick validation

Run the regression suite first:

```bash
pytest -q
```

Expected result in this package revision:

```text
23 passed
```

## Preprocessing

Use `full_preprocess.yaml` or `paper.yaml` for data used in experiments. `smoke.yaml` is only for fast debugging.

Small validation preprocessing:

```bash
python -m bdse.experiments.preprocess \
  --config bdse/configs/full_preprocess.yaml \
  --data-root "$NUPLAN_DATA_ROOT" \
  --maps-root "$NUPLAN_MAPS_ROOT" \
  --map-version nuplan-maps-v1.0 \
  --splits val \
  --output-dir /path/to/bdse_cache/val \
  --num-workers 4 \
  --scenario-stride 10 \
  --teacher-cost-eval-stride 1 \
  --include-drivable-polygons \
  --candidate-aware-agent-selection \
  --profile \
  --profile-threshold-s 1.0
```

Training preprocessing:

```bash
python -m bdse.experiments.preprocess \
  --config bdse/configs/full_preprocess.yaml \
  --data-root "$NUPLAN_DATA_ROOT" \
  --maps-root "$NUPLAN_MAPS_ROOT" \
  --map-version nuplan-maps-v1.0 \
  --splits train \
  --output-dir /path/to/bdse_cache/train \
  --num-workers 8 \
  --scenario-stride 10 \
  --max-samples-per-log 256 \
  --teacher-cost-eval-stride 1 \
  --include-drivable-polygons \
  --candidate-aware-agent-selection \
  --profile \
  --profile-threshold-s 2.0
```

Preprocessing stores runtime-only features, label-only futures, candidates, evidence atoms, proposal features, teacher labels, and pair labels. Runtime code only consumes runtime features, candidate bank, evidence atoms, proposal features, and sparse queries.

## Training

```bash
python -m bdse.experiments.train \
  --config bdse/configs/full_preprocess.yaml \
  --split train \
  --preprocessed-dir /path/to/bdse_cache \
  --output outputs/bdse_model.pt
```

For a short sanity run:

```bash
python -m bdse.experiments.train \
  --config bdse/configs/smoke.yaml \
  --split train \
  --preprocessed-dir /path/to/bdse_cache \
  --max-scenarios 128 \
  --output outputs/bdse_smoke.pt
```

Training losses:

- `L_base`: regress predicted base cost to `J_base_T`.
- `L_res`: offline residual-margin supervision over full evidence bank.
- `L_rank`: logistic rank loss over positive teacher pairs.
- `L_prop`: BCE/listwise proposal loss from oracle marginal certificate gain.
- `L_act`: deployment-consistent sparse proposal/query/greedy/tournament action loss.
- `L_cal`: optional margin calibration surrogate when `training.loss_weights.calibration > 0`.

## Post-hoc calibration

Estimate a validation-set one-sided margin residual quantile:

```bash
python -m bdse.experiments.calibrate \
  --config bdse/configs/full_preprocess.yaml \
  --checkpoint outputs/bdse_model.pt \
  --split val \
  --preprocessed-dir /path/to/bdse_cache \
  --delta 0.1 \
  --output outputs/calibration.json
```

Then copy the reported value into your runtime config:

```yaml
tournament:
  epsilon_cal: <epsilon_cal from calibration.json>
fallback:
  safety_lcb_min: 0.0
```

This is the deployment-oriented substitute for an unimplemented learned uncertainty head. It is cheaper and easier to validate in a real-time planner: it uses a scalar validation quantile instead of expanding the runtime network output.

## Open-loop diagnostics

```bash
python -m bdse.experiments.evaluate_open_loop \
  --config bdse/configs/full_preprocess.yaml \
  --checkpoint outputs/bdse_model.pt \
  --split val \
  --max-scenarios 500 \
  --output outputs/open_loop_bdse_metrics.json
```

Teacher/evidence diagnostics:

```bash
python -m bdse.experiments.diagnostics \
  --config bdse/configs/full_preprocess.yaml \
  --split val \
  --preprocessed-dir /path/to/bdse_cache \
  --max-scenarios 500 \
  --output outputs/diagnostics_val.json
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

The included `bdse.experiments.evaluate_closed_loop` script only verifies planner construction and prints integration instructions. Full nuPlan closed-loop evaluation still needs your project’s Hydra `run_simulation` entrypoint to pass the constructed planner instance as a pre-built planner. This repository does not bundle a full Hydra simulation config.

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
