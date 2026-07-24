# v45 PB-RADS: Primary-Budget Exact Deployment Supervision

v45 is a validity and deployment-alignment correction of v44. It is deliberately
not another online subset-search heuristic. The goal is to obtain the first
clean test of the paper's core claim: under a fixed budget, train the model on
the exact evidence set used by the deployed selector.

## Why v44 is not a valid positive result

The uploaded v44 run has `loss=NaN` and `L_cert_frontier=NaN` in every epoch.
The source is the safe-frontier certificate loss: scenes with hard candidates
but no safe candidate evaluate a large sentinel difference inside `softplus`,
then multiply the resulting infinity by a zero mask (`inf * 0 -> NaN`). With AMP,
this can silently turn training into repeated optimizer-step skips.

The fast v44 schedule also provides only about 25% exact-selector supervision:
2 of 4 scenes per rank, every 2 steps. The weighted round-robin budget schedule
selects only one of B=8/16/24 per rank-step, so the primary B=16 path is trained
on only about 12.5% of scene-steps before the final short exact tail.

## v45 changes

1. Mask certificate safety/frontier terms before nonlinear evaluation, so
   invalid sentinel differences cannot generate NaN.
2. Abort all DDP ranks immediately if any scalar loss becomes non-finite.
3. Always optimize the primary B=16 deployment path and rotate one auxiliary
   budget (B=8 or B=24) with an unbiased primary-plus-aux schedule.
4. Run the exact runtime selector on every local scene and every optimizer step
   after the one-epoch oracle warm-up.
5. Distill stop-gradient runtime-selected masks into proposal logits. This gives
   the discrete selector a direct learned proposal signal while leaving the
   deployed fixed-budget selector unchanged.
6. Checkpoint by the fixed-budget critical score with dense validation, not by
   256-scene teacher match alone. Structural hard safety uses effective recall
   because those atoms are budget-exempt by design.
7. Add a strict gate that verifies finite training, exact-selector coverage,
   match gain, tail regret, winner/rival sign, evidence sufficiency, and p95
   latency before authorizing CL20.

## Recommended command

```bash
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/
export BDSE_VAL_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/

DETACH=1 \
GPUS=0,1 \
V30_CKPT_IN=outputs_v30/train/bdse_v30_pmvrbsr.best.pt \
OUT_ROOT=outputs_v45_pb_rads_exact_2gpu_v1 \
RUN_MODE=train_open_loop \
AUTO_RESUME=0 \
BATCH_SIZE_PER_GPU=4 \
NUM_WORKERS_PER_GPU=6 \
PREFETCH_FACTOR=2 \
SAVE_EVERY_N_STEPS=500 \
SELECTOR_SCENES_PER_RANK=0 \
SELECTOR_EVERY_N_STEPS=1 \
SELECTOR_FULL_LAST_N_STEPS=0 \
VAL_SCENARIOS=1000 \
VAL_DENSE_DIAGNOSTIC=1 \
bash run_v45_pb_rads.sh
```

`SELECTOR_SCENES_PER_RANK=0` means all local scenes.

## Strict open-loop gate

Reuse the frozen v30/MARS control evaluated on the exact same 1000 scenarios
and order. Then run:

```bash
python -m bdse.tools.check_v45_pb_rads_gate \
  outputs_v45_pb_rads_exact_2gpu_v1/open_loop/open_loop_v45_pb_rads.json \
  outputs_v43/open_loop/open_loop_v43_mars_control.json \
  --candidate-jsonl outputs_v45_pb_rads_exact_2gpu_v1/open_loop/open_loop_v45_pb_rads.jsonl \
  --control-jsonl outputs_v43/open_loop/open_loop_v43_mars_control.jsonl \
  --train-log outputs_v45_pb_rads_exact_2gpu_v1/train/bdse_v45_pb_rads.train_log.jsonl \
  --latency-target-ms 500
```

The 500 ms target matches the current 5-tick/10 Hz replan interval. A stricter
real-time claim requires a correspondingly lower target.

## Closed-loop authorization

Do not run CL20 as the next scientific step unless the strict gate passes. When
it passes:

```bash
export NUPLAN_ROOT=/data0/senzeyu2/dataset/nuplan
RUN_MODE=cl20 \
V45_CKPT=outputs_v45_pb_rads_exact_2gpu_v1/train/bdse_v45_pb_rads.best.pt \
OUT_ROOT=outputs_v45_pb_rads_exact_2gpu_v1 \
GPUS=0,1 \
bash run_v45_pb_rads.sh
```

Run CL100 only after CL20 is non-inferior on safety and improves the target
planning score:

```bash
RUN_MODE=cl100 \
V45_CKPT=outputs_v45_pb_rads_exact_2gpu_v1/train/bdse_v45_pb_rads.best.pt \
OUT_ROOT=outputs_v45_pb_rads_exact_2gpu_v1 \
GPUS=0,1 \
bash run_v45_pb_rads.sh
```
