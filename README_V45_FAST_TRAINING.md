# v45 Fast Training Path

## Root cause of the one-day / five-epoch run

The two-A30 DDP setup is valid. The slowdown is caused by the training loss, not
by model replication:

1. `primary_plus_aux` evaluates two budgets on every optimizer step.
2. For each budget, `_predicted_pair_certificate_masks()` detaches the current
   GPU predictions, synchronously copies them to CPU/NumPy, and runs the serial
   runtime MARS backward-elimination selector for every local scene.
3. With a local batch of four, this is eight exact CPU selector searches per
   rank per step. DDP then waits for the slower rank before gradient all-reduce.
4. At 50k scenarios and global batch eight there are about 6250 optimizer steps
   per epoch. A measured synthetic v45-sized batch needs roughly 0.57 seconds
   for one exact budget on four scenes, before model forward/backward and I/O.
5. Validation additionally runs 1000 scenes every two epochs and, when dense
   diagnostics are enabled, performs a second dense model inference per scene.

This is a CPU-bound synchronization design. Adding GPUs does not fix it.

## Optimized training design

The fast pathway keeps runtime evaluation unchanged.

- Every training step uses a GPU margin-damage surrogate. It starts from the
  full predicted Top-M signed margin field and scores each atom by the damage
  caused by removing it: residual error, sign flips, winner-certificate loss,
  and predicted action changes.
- The resulting ranking is nested, so B=8, B=16, and B=24 masks are produced in
  one accelerator pass.
- The exact runtime NumPy selector is still used as the ground-truth mask for
  proposal distillation, but only for one scene per rank every four steps by
  default.
- The training log reports `selector_fast_wall_time_s`,
  `selector_exact_wall_time_s`, `selector_exact_fraction`, and
  `selector_surrogate_exact_agreement`.
- Intermediate validation defaults to 500 scenes every three epochs without the
  dense second pass. `RUN_MODE=train_open_loop` still runs the final 1000-scene
  dense open-loop evaluation after training.
- Mid-epoch checkpoints default to every 2000 steps to reduce filesystem stalls.

## Recommended clean run

Do not reuse the slow-run output directory. For the final controlled experiment,
restart from the frozen v30 checkpoint so the training schedule is reproducible.

```bash
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/
export BDSE_VAL_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/

DETACH=1 \
GPUS=0,1 \
V30_CKPT_IN=outputs_v30/train/bdse_v30_pmvrbsr.best.pt \
OUT_ROOT=outputs_v45_pb_rads_fast_2gpu_v1 \
RUN_MODE=train_open_loop \
AUTO_RESUME=0 \
BATCH_SIZE_PER_GPU=4 \
NUM_WORKERS_PER_GPU=8 \
PREFETCH_FACTOR=3 \
SAVE_EVERY_N_STEPS=2000 \
EXACT_DISTILL_SCENES_PER_RANK=1 \
EXACT_DISTILL_EVERY_N_STEPS=4 \
VAL_SCENARIOS=500 \
VAL_EVERY_N_EPOCHS=3 \
VAL_DENSE_DIAGNOSTIC=0 \
bash run_v45_pb_rads_fast.sh
```

The final open-loop evaluation still uses `OPEN_LOOP_MAX_SCENARIOS=1000` and
keeps dense diagnostics enabled.

## Server-side selector benchmark

```bash
python -m bdse.tools.benchmark_v45_selector_training \
  --config bdse/configs/v45_bdse_pb_rads_train_fast_2gpu.yaml \
  --device cuda
```

## How to read the new timing metrics

- High `train_data_wait_ms_per_step`: increase workers/prefetch only if storage
  can sustain it; inspect disk utilization first.
- High `train_loss_ms_per_step` together with high
  `selector_exact_wall_time_s`: reduce exact-distillation frequency to every
  eight steps, but treat this as an ablation.
- Low data wait and low selector time but high `train_backward_step_ms_per_step`:
  the neural pair head is now the bottleneck; try batch six only after checking
  A30 memory headroom.
- `selector_surrogate_exact_agreement` should be tracked against open-loop gate
  metrics. A low value means the surrogate needs calibration; it is not a reason
  to silently return to all-scene exact CPU selection.

## Scientific reporting

The paper should describe the exact runtime MARS selector as the deployed
algorithm. The GPU surrogate is a training approximation, analogous to a
teacher-student or sampled structured-loss estimator. Report its exact-mask
sampling rate and include an exact-all-scenes small-data ablation.
