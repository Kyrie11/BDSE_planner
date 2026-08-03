# Matched fixed-budget external baseline adapters

This package implements controlled **budget-compatible adapters**, not verbatim official reproductions. Every trainable adapter uses the same BDSE preprocessed scene tensors, candidate bank, teacher labels, evidence atoms, fixed evidence budget and planner shell. This gives a controlled comparison under the paper's interface constraint, but the names in tables must include `-inspired adapter` or `-style scorer`.

| Config variant | Reference | Retained core logic | Important omissions | Required paper label |
|---|---|---|---|---|
| `gameformer` | GameFormer, ICCV 2023 | scene Transformer; iterative level-k candidate/evidence refinement | official joint multi-agent prediction and hierarchical decoder targets | GameFormer-inspired budget adapter |
| `dtpp` | DTPP, 2023 | maneuver/tree branch cost plus trajectory refinement | ego-conditioned prediction, scenario tree and differentiable joint prediction-cost training | DTPP-inspired budget adapter |
| `plantf` | PlanTF, ICRA 2024 | pure Transformer imitation planner; state dropout | official object-token feature pipeline and full augmentation recipe | PlanTF-inspired budget adapter |
| `pluto` | PLUTO, 2024 | longitudinal/lateral cost decomposition; state dropout | auxiliary loss framework, contrastive imitation learning and official augmentations | PLUTO-inspired budget adapter |
| `pdm_closed` | PDM-Closed / tuPlan Garage, CoRL 2023 | route/centerline, progress, comfort and safety prior | official centerline proposal generation, IDM rollout and PDM scoring stack | PDM-Closed-style budget scorer |

The `pdm_closed` variant is deterministic and has no checkpoint. It must not be reported as the official PDM-Closed planner.

## Matched-data training

Use the repository wrapper instead of handwritten per-model commands:

```bash
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2
export BDSE_SPLIT_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v53_split
export EXTERNAL_OUT_ROOT=outputs/external
export GPUS=0,1
bash RUN_V60_EXTERNAL_BASELINES_MATCHED_TRAIN_2GPU.sh
```

The wrapper runs two models concurrently and writes exactly:

```text
outputs/external/gameformer_budgeted.best.pt
outputs/external/dtpp_budgeted.best.pt
outputs/external/plantf_budgeted.best.pt
outputs/external/pluto_budgeted.best.pt
```

These names match `RUN_V60_BUDGET_BASELINE_SWEEP.sh`. Do not pass `*.best.pt` to `--output`; the trainer appends `.best.pt` itself.

Each checkpoint contains a training manifest with train/validation path hashes, counts, split names, seed and common protocol. Evaluation scripts refuse to start if the four manifests differ or a checkpoint does not strictly match its adapter architecture.

## Speed changes

- Unit-cost evidence selection is a vectorized GPU top-k; the previous per-sample/per-atom GPU-to-CPU synchronization is removed.
- Selected evidence tokens are gathered in batch and use attention padding masks; the previous Python batch loop is removed.
- Loss aggregation synchronizes to CPU once per epoch rather than every batch.
- Progress-bar loss synchronization is throttled.
- Fused AdamW, TF32, AMP, deterministic workers, prefetching, warmup/cosine scheduling and early stopping are supported.
- Closed-loop uses deterministic token manifests, one worker per process, disables planner serialization, verifies successful/failed simulation counts, and schedules two systems concurrently on two GPUs.

## Evaluation

```bash
bash RUN_V60_EXTERNAL_OPEN_LOOP_2GPU.sh
bash RUN_V60_BUDGET_BASELINE_SWEEP.sh
bash RUN_V60_EXTERNAL_CL20_2GPU.sh
bash RUN_V60_EXTERNAL_CL50_2GPU.sh
```

For reactive evaluation set:

```bash
export CL_CHALLENGE=closed_loop_reactive_agents
bash RUN_V60_EXTERNAL_CL20_2GPU.sh
```

All comparison scripts use a shared ordered scenario/token list and retain every finite metric in CSV/JSON, not only the summary columns.
