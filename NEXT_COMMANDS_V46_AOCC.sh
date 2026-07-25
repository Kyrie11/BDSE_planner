#!/usr/bin/env bash
# Run from the extracted bdse_v46_aocc repository root.
set -euo pipefail

export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/
export BDSE_VAL_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/
export BDSE_CALIB_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_calib_v2/  # must be disjoint from final gate scenarios

V30=outputs_v30/train/bdse_v30_pmvrbsr.best.pt
V45=outputs_v45_pb_rads_fast_2gpu_v1/train/bdse_v45_pb_rads_fast.best.pt

# -----------------------------------------------------------------------------
# Phase 0A: Re-evaluate the v45 checkpoint with the original MARS selector,
# but using the v46 evaluator to obtain pair-full-interface and stage latency.
# -----------------------------------------------------------------------------
GPUS=0,1 \
RUN_MODE=open_loop \
V46_CKPT="$V45" \
OUT_ROOT=outputs_v45_mars_pairfull_recheck \
EVAL_CONFIG=bdse/configs/v45_bdse_pb_rads_fast_cl.yaml \
OPEN_LOOP_MAX_SCENARIOS=1000 \
bash run_v46_aocc.sh

python -m bdse.tools.analyze_v46_failure_modes \
  outputs_v45_mars_pairfull_recheck/open_loop/open_loop_v46_aocc.jsonl \
  --output outputs_v45_mars_pairfull_recheck/open_loop/failure_modes.json

# -----------------------------------------------------------------------------
# Phase 0B: Selector-only replay. Use the same v45 checkpoint and AOCC selector.
# This isolates AOCC from representation/exact-training changes.
# -----------------------------------------------------------------------------
GPUS=0,1 \
RUN_MODE=open_loop \
V46_CKPT="$V45" \
OUT_ROOT=outputs_v45_ckpt_aocc_replay \
EVAL_CONFIG=bdse/configs/v46_bdse_aocc_cl.yaml \
OPEN_LOOP_MAX_SCENARIOS=1000 \
bash run_v46_aocc.sh

python -m bdse.tools.analyze_v46_failure_modes \
  outputs_v45_ckpt_aocc_replay/open_loop/open_loop_v46_aocc.jsonl \
  --output outputs_v45_ckpt_aocc_replay/open_loop/failure_modes.json

# -----------------------------------------------------------------------------
# Phase 1: Main v46 exact training. Do not warm-start from the fast v45 model.
# -----------------------------------------------------------------------------
DETACH=1 \
GPUS=0,1 \
V30_CKPT_IN="$V30" \
OUT_ROOT=outputs_v46_aocc_exact_2gpu_v1 \
RUN_MODE=train_open_loop \
AUTO_RESUME=0 \
BATCH_SIZE_PER_GPU=4 \
NUM_WORKERS_PER_GPU=6 \
PREFETCH_FACTOR=2 \
SAVE_EVERY_N_STEPS=500 \
SELECTOR_SCENES_PER_RANK=0 \
SELECTOR_EVERY_N_STEPS=1 \
SELECTOR_FULL_LAST_N_STEPS=0 \
EXACT_DISTILL_SCENES_PER_RANK=0 \
EXACT_DISTILL_EVERY_N_STEPS=1 \
VAL_SCENARIOS=1000 \
VAL_EVERY_N_EPOCHS=2 \
VAL_DENSE_DIAGNOSTIC=1 \
OPEN_LOOP_MAX_SCENARIOS=1000 \
bash run_v46_aocc.sh

# -----------------------------------------------------------------------------
# Phase 2: Calibrate the one-sided adverse bound on an independent cache/split.
# This command is run after the v46 checkpoint exists.
# -----------------------------------------------------------------------------
python -m bdse.tools.calibrate_v46_adverse_bounds \
  --config bdse/configs/v46_bdse_aocc_cl.yaml \
  --checkpoint outputs_v46_aocc_exact_2gpu_v1/train/bdse_v46_aocc.best.pt \
  --preprocessed-dir "$BDSE_CALIB_CACHE" \
  --split calib \
  --max-scenarios 2000 \
  --device cuda \
  --alpha 0.05 \
  --beta 1.0 \
  --prior-radius 0.10 \
  --output outputs_v46_aocc_exact_2gpu_v1/calibration/adverse_bound.json

python -m bdse.tools.apply_v46_calibration \
  --config-in bdse/configs/v46_bdse_aocc_cl.yaml \
  --calibration-json outputs_v46_aocc_exact_2gpu_v1/calibration/adverse_bound.json \
  --config-out outputs_v46_aocc_exact_2gpu_v1/calibration/v46_bdse_aocc_cl_calibrated.yaml

# -----------------------------------------------------------------------------
# Phase 3: Final disjoint open-loop evaluation with calibrated epsilon.
# -----------------------------------------------------------------------------
GPUS=0,1 \
RUN_MODE=open_loop \
V46_CKPT=outputs_v46_aocc_exact_2gpu_v1/train/bdse_v46_aocc.best.pt \
OUT_ROOT=outputs_v46_aocc_exact_2gpu_v1_calibrated \
EVAL_CONFIG=outputs_v46_aocc_exact_2gpu_v1/calibration/v46_bdse_aocc_cl_calibrated.yaml \
OPEN_LOOP_MAX_SCENARIOS=1000 \
bash run_v46_aocc.sh

python -m bdse.tools.analyze_v46_failure_modes \
  outputs_v46_aocc_exact_2gpu_v1_calibrated/open_loop/open_loop_v46_aocc.jsonl \
  --output outputs_v46_aocc_exact_2gpu_v1_calibrated/open_loop/failure_modes.json

# -----------------------------------------------------------------------------
# Phase 4: Strict gate. Summary and JSONL for the control MUST come from the
# same run root and exactly the same scenario-token set/order.
# -----------------------------------------------------------------------------
python -m bdse.tools.check_v46_aocc_gate \
  outputs_v46_aocc_exact_2gpu_v1_calibrated/open_loop/open_loop_v46_aocc.json \
  outputs_v43_sabdacc_runtime_v30ckpt/open_loop/open_loop_v43_mars_control.json \
  --candidate-jsonl outputs_v46_aocc_exact_2gpu_v1_calibrated/open_loop/open_loop_v46_aocc.jsonl \
  --control-jsonl outputs_v43_sabdacc_runtime_v30ckpt/open_loop/open_loop_v43_mars_control.jsonl \
  --train-log outputs_v46_aocc_exact_2gpu_v1/train/bdse_v46_aocc.train_log.jsonl \
  --latency-target-ms 500

# -----------------------------------------------------------------------------
# Phase 5: Closed loop only after the strict gate prints PASS.
# -----------------------------------------------------------------------------
# export NUPLAN_ROOT=/data0/senzeyu2/dataset/nuplan
# GPUS=0,1 \
# RUN_MODE=cl20 \
# V46_CKPT=outputs_v46_aocc_exact_2gpu_v1/train/bdse_v46_aocc.best.pt \
# OUT_ROOT=outputs_v46_aocc_exact_2gpu_v1_calibrated \
# EVAL_CONFIG=outputs_v46_aocc_exact_2gpu_v1/calibration/v46_bdse_aocc_cl_calibrated.yaml \
# bash run_v46_aocc.sh
