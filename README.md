# BDSE nuPlan implementation

This repository implements **Budgeted Decision-Sufficient Evidence (BDSE)** for nuPlan-style DB planning data. The implementation follows the paper and the landing guide: candidate-set teacher supervision, additive atom-level evidence costs, runtime/label-only separation, runtime predicted greedy selection, and pairwise tournament action choice.

## 1. Repository layout

```text
bdse/
  configs/
    default.yaml          # main experiment configuration
    ablations.yaml        # budget, rival, selector and fallback sweeps
  data/
    nuplan_dataset.py     # DB discovery and nuPlan ScenarioBuilder wrapper
    scenario_sampler.py   # split/folder-aware .db discovery under /data0/nuplan/data/cache
    feature_builder.py    # runtime-only feature extraction
    label_builder.py      # offline label-only future and teacher sample construction
    cache_schema.py       # sample/candidate/evidence/teacher/pair dataclasses
  planner/
    candidate_generator.py
    evidence_atoms.py
    teacher_cost.py
    pair_builder.py
    selector.py
    tournament.py
    fallback.py
    nuplan_planner.py
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
    evaluate_closed_loop.py
    diagnostics.py
    ablations.py
  metrics/
    bdse_metrics.py
    nuplan_metrics.py
  tests/
    pytest unit tests for the required BDSE invariants
```

## 2. Environment

Recommended base environment:

```bash
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

For real nuPlan training and closed-loop simulation, install the official nuPlan devkit in the same environment and make it importable. The code calls `NuPlanScenarioBuilder` when `use_devkit=True`.

Set dataset paths before preprocessing/training:

```bash
export NUPLAN_DATA_ROOT=/data0/nuplan/data/cache
export NUPLAN_MAPS_ROOT=/data0/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/data0/nuplan/exp
```

The default config already points to the user-provided DB cache root:

```yaml
paths:
  data_cache_root: /data0/nuplan/data/cache
  maps_root: /data0/nuplan/dataset/maps
  exp_root: /data0/nuplan/exp/bdse
```

## 3. Expected nuPlan DB organization

The loader treats each direct subfolder under `/data0/nuplan/data/cache` as a split/folder bucket and recursively discovers `.db` files:

```text
/data0/nuplan/data/cache/
  val/
    *.db
  train_boston/
    *.db
  train_singapore/
    *.db
  train_las_vegas/
    *.db
  train_pittsburgh/
    *.db
```

Folder names starting with `train` are normalized to split `train`; folder names starting with `val`, `valid`, or `validation` are normalized to split `val`. You can still select a specific folder with `--folders train_boston train_singapore`.

Check discovered splits:

```bash
python -m bdse.experiments.preprocess --list-splits
```

## 4. Core implementation guarantees

The code enforces the following BDSE invariants:

1. `compute_trajectory()` accepts only runtime planner input. It does not accept `label_future`, `future_agents`, logged future ego, teacher margins, or teacher action labels.
2. Teacher margin sign is fixed: `M_T(a,b) = J_T(b) - J_T(a)`. Positive means the first action is better.
3. Teacher cost has exactly two parts: `J_T = J_base_T + J_evid_T`, and `J_evid_T = sum_i g_i_T`.
4. Evidence is normalized atom by atom before summation: `g_i_T = w_tau * clip(r_i / (s_tau + eps), 0, gmax_tau)`.
5. Hard collision, red light, off-drivable, and wrong-way events are evidence atoms, not a separate hard gate.
6. Invalid padded candidates are excluded from teacher argmin, pair construction, selector objectives, tournament argmax, and metric denominators.
7. Runtime selector uses only predicted full-interface margins, predicted positive pairs, predicted caps, predicted weights, runtime safety flags, and candidate/evidence valid masks.

## 5. Preprocessing labels and teacher data

Preprocessing builds each sample as:

```python
Sample = {
    "runtime": RuntimeFeatures,
    "label_future": LabelOnlyFuture,
    "candidates": CandidateBank,
    "evidence_bank": EvidenceBank,
    "teacher": TeacherLabels,
    "pairs": PairLabels,
}
```

Run a small validation preprocessing job:

```bash
python -m bdse.experiments.preprocess \
  --config bdse/configs/full_preprocess.yaml \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --maps-root /data0/senzeyu2/dataset/nuplan/maps \
  --map-version nuplan-maps-v1.0 \
  --splits val \
  --output-dir /data0/senzeyu2/d[preprocess_v3.patch](../preprocess_v3.patch)ataset/nuplan/data/cache/val_set \
  --num-workers 4 \
  --scenario-stride 10 \
  --teacher-cost-eval-stride 1 \
  --include-drivable-polygons \
  --candidate-aware-agent-selection \
  --profile \
  --profile-threshold-s 0.2
```

Run training preprocessing on selected train folders:

```bash
python -m bdse.experiments.preprocess \
  --config bdse/configs/full_preprocess.yaml \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --maps-root /data0/senzeyu2/dataset/nuplan/maps \
  --map-version nuplan-maps-v1.0 \
  --splits train \
  --output-dir /data0/senzeyu2/dataset/nuplan/data/cache/bdse_full_qualityfix \
  --num-workers 4 \
  --scenario-stride 10 \
  --teacher-cost-eval-stride 1 \
  --include-drivable-polygons \
  --candidate-aware-agent-selection \
  --profile \
  --profile-threshold-s 1.0
```

During preprocessing:

- runtime features use past/current ego, past/current agents, current traffic lights, HD map, route roadblock ids, and mission goal;
- label futures use logged future ego and logged future agents for the offline teacher only;
- candidate bank uses the default `K=32`, `T=80`, horizon `8s`, step `0.1s` route-conditioned semantic lattice;
- evidence bank is capped by `N_inter<=64`, `N_map<=32`, `N_kin<=16`, `N_E<=128`;
- pair labels store better action first and validate residual closure.

## 6. Training

Run a small smoke training job:

```bash
python -m bdse.experiments.train \
  --split train \
  --max-files 1 \
  --max-scenarios 256 \
  --output outputs/bdse_model.pt
```

Full training uses the same command without `--max-*` limits. Main hyperparameters are in `bdse/configs/default.yaml`:

- hidden dim: `256`
- transformer layers: `4`
- heads: `8`
- candidates: `32`
- evidence atoms: `128`
- pair batch per scene: `128`
- optimizer: `AdamW`
- learning rate: `1e-4`
- weight decay: `1e-2`
- batch size: `64`
- epochs: `20`
- grad clip: `5.0`

Training loss:

```text
L_total = L_base + L_res + L_rank + 0.5 * L_sel + L_act
```

`L_rank` uses the full-interface predicted margin. `L_act` uses runtime-style selected evidence.

## 7. Open-loop diagnostics

Run BDSE-specific diagnostics with an already trained checkpoint:

```bash
python -m bdse.experiments.evaluate_open_loop \
  --checkpoint outputs/bdse_model.pt \
  --split val \
  --max-files 1 \
  --max-scenarios 100 \
  --output outputs/open_loop_bdse_metrics.json
```

Run teacher/evidence diagnostics using teacher costs as the predictor oracle:

```bash
python -m bdse.experiments.diagnostics \
  --config bdse/configs/full_preprocess.yaml \
  --split val \
  --preprocessed-dir /data0/senzeyu2/dataset/nuplan/data/cache/val_set \
  --max-scenarios 200 \
  --output outputs/diagnostics_val_200.json
```

Reported BDSE metrics include:

- teacher regret
- teacher action match
- full-interface match
- budget-vs-full match
- preserved margin error
- evidence sufficiency
- decision sufficiency
- selector value ratio
- hard-evidence recall
- effective query count
- fallback rate when running through the planner core

## 8. Closed-loop nuPlan integration

`BDSEnuPlanPlanner` implements the runtime planning path:

```python
from bdse.planner.nuplan_planner import BDSEnuPlanPlanner
planner = BDSEnuPlanPlanner(model=model, cfg=cfg)
```

The planner returns an 8-second trajectory at 10 Hz. In a nuPlan Hydra simulation entrypoint, pass the planner instance as a pre-built planner. The implementation keeps logged futures out of `compute_trajectory()`.

The closed-loop script is a lightweight entrypoint check:

```bash
python -m bdse.experiments.evaluate_closed_loop \
  --checkpoint outputs/bdse_model.pt \
  --challenge closed_loop_nonreactive_agents \
  --output-dir outputs/closed_loop
```

For full official metrics, run the planner through nuPlan devkit's simulation pipeline and aggregate nuPlan metrics: overall planning score, no at-fault collision, drivable-area compliance, route progress, speed-limit compliance, TTC, comfort, and latency.

## 9. Ablation plan

Generate the configured ablation plan:

```bash
python -m bdse.experiments.ablations --output outputs/ablation_plan.json
```

Supported switches are config-only:

- selector modes: runtime predicted, random, top magnitude, diversity, interaction-only, rule/map-only;
- budget sweep: `B in {4,8,16,24,32}`;
- rival sweep: `L_infer in {4,8,16,24,K-1}`;
- fallback modes: no fallback, rival expansion only, budget expansion only, rule top-K rerank, full fallback.

`teacher.separate_hard_gate=true` raises an error unless placed under an explicit debug-only invalid ablation block.

## 10. Tests

Run all required invariant tests:

```bash
pytest -q bdse/tests
```

The included tests cover:

1. runtime/label-only separation
2. margin sign and antisymmetry
3. teacher cost partition
4. atom-level additivity
5. hard event ownership
6. residual closure
7. invalid padding masks
8. oracle-vs-runtime selector input separation
9. selector monotonicity
10. tournament antisymmetry

## 11. Notes on planner-interface budget accounting

The main BDSE budget is the post-selection planner-interface atom budget `|S_B|`. The runtime selector may score all predicted atom costs to build `S_B`; diagnostics should report both:

```text
pre-selection scoring cost = N_E * K or N_E * |P_hat_plus|
post-selection query cost  = |S_B| * K or |S_B| * |P_infer|
```

This avoids conflating neural preselection compute with the deployed planner-interface query budget.
