# BDSE v28 DA-RBSR summary

## Motivation from v27 results

v27 solved part of the v26 failure mode: the hard-filter-safe availability improved and the fixed-budget runtime-only CL20 score recovered to the v25 no-fallback level. After finetuning, CL50 reached 0.2341 with safety fallback. However, the new bottleneck is drivable-area compliance (0.70 in CL50) and the planner still selects runtime-flagged actions in ~35--37% of replans. v27's tiered mask still treats soft-risk flags as tournament constraints whenever a soft-safe candidate exists, which can over-constrain the candidate set and push recovery/utility toward aggressive but route-unsafe trajectories.

## v28 algorithm: DA-RBSR

v28 changes the runtime guard from v27 T-RBSR to Dual-tier Adaptive Risk-Bounded Safety Recovery (DA-RBSR):

1. **Hard-constrained, soft-priced risk.** `runtime_safety.flag_mode=dual_tier` uses only hard violations for tournament hard masking and fallback triggering. Soft risks are retained for rule/rerank pricing and diagnostics rather than used as hard constraints.
2. **Drivable-aware utility/recovery tuning.** Utility and safe-progress recovery reduce progress/path reward and increase lateral/route penalties to address the v27 CL50 drivable bottleneck.
3. **Explicit hard/soft diagnostics.** Closed-loop diagnostics now log hard/soft/active safe availability and component counts, so the next run can directly test whether failures are caused by hard infeasibility, soft-risk overpricing, or recovery choice.

The evidence budget remains fixed: no expanded fallback evidence stage is used.

## Modified files

- `bdse/planner/fallback.py`
  - Added `dual_tier` flag mode.
  - Added `runtime_safety_diagnostics()`.
  - Kept soft-risk pricing in `rule_based_runtime_scores()` while hard flags act as constraints.
- `bdse/planner/nuplan_planner.py`
  - Logs runtime safety diagnostics at stage and final planner level.
- `bdse/configs/v28_bdse_darbsr_*_fast_cl.yaml`
  - New v28 runtime configs.
- `bdse/configs/v28_bdse_darbsr_train.yaml`
  - New v28 finetune config.
- `run_v28_darbsr.sh`
  - Uses v28 configs.
  - Supports export-style parameters.
  - Runs 4 open-loop / CL20 evaluations concurrently by default: two jobs on GPU 0 and two jobs on GPU 1.
  - Uses `RUN_CL20` and `RUN_CL50` explicitly.

## Validation

Local static/unit validation:

```text
44 passed, 2 warnings
```

## Recommended first checks

Runtime-only open loop with v27 checkpoint:

```bash
export SKIP_TRAIN=1
export V28_CKPT=outputs_v27/train/bdse_v27_trbsr.best.pt
export OPEN_LOOP_ONLY=1
export OPEN_PARALLEL4=1
bash run_v28_darbsr.sh
```

Runtime-only CL20 with v27 checkpoint:

```bash
export SKIP_TRAIN=1
export V28_CKPT=outputs_v27/train/bdse_v27_trbsr.best.pt
export RUN_CL20=1
export RUN_CL50=0
export CL_PARALLEL4=1
export CL_WORKERS_PER_RUN=2
bash run_v28_darbsr.sh
```

Finetune from v25 checkpoint:

```bash
export V25_CKPT_IN=outputs_v25/train/bdse_v25_dgcace.best.pt
export TRAIN_MAX_SCENARIOS=12000
export VAL_MAX_SCENARIOS=1000
export TRAIN_EPOCHS=5
export NPROC_PER_NODE=2
bash run_v28_darbsr.sh
```

CL50 with v28 checkpoint:

```bash
export SKIP_TRAIN=1
export V28_CKPT=outputs_v28/train/bdse_v28_darbsr.best.pt
export RUN_CL20=0
export RUN_CL50=1
export CL_WORKERS_PER_RUN=2
bash run_v28_darbsr.sh
```
