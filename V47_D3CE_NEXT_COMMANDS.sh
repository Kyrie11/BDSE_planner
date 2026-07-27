#!/usr/bin/env bash
set -euo pipefail

# Run this script from the v47 code root, or copy commands selectively.
# Mandatory paths.
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE_ORIGINAL="${BDSE_VAL_CACHE_ORIGINAL:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export BDSE_SPLIT_CACHE="${BDSE_SPLIT_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v47_split}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_v2}"
export V30_CKPT_IN="${V30_CKPT_IN:-outputs_v30/train/bdse_v30_pmvrbsr.best.pt}"
export OUT_ROOT="${OUT_ROOT:-outputs_v47_d3ce_exact_2gpu_v1}"
export CONTROL_CONFIG="${CONTROL_CONFIG:-bdse/configs/v43_bdse_mars_control_fast_cl.yaml}"
export CONTROL_CKPT="${CONTROL_CKPT:-outputs_v30/train/bdse_v30_pmvrbsr.best.pt}"
export CONTROL_ROOT="${CONTROL_ROOT:-outputs_v47_control_val_tune}"

# ---------------------------------------------------------------------------
# 0. Create a paper-grade validation protocol. The split is by nuPlan log group,
#    not random scenario row, to reduce temporal/scene leakage.
# ---------------------------------------------------------------------------
python -m bdse.tools.build_group_disjoint_calibration_split \
  --preprocessed-dir "$BDSE_VAL_CACHE_ORIGINAL" \
  --split val \
  --output-root "$BDSE_SPLIT_CACHE" \
  --calibration-fraction 0.20 \
  --seed 47

export BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE"

# ---------------------------------------------------------------------------
# 1. Main v47 training. Use val_tune only for checkpoint selection.
#    Main paper run starts from the frozen v30 checkpoint, not v46.
# ---------------------------------------------------------------------------
DETACH=1 \
GPUS=0,1 \
V30_CKPT_IN="$V30_CKPT_IN" \
OUT_ROOT="$OUT_ROOT" \
RUN_MODE=train_open_loop \
AUTO_RESUME=0 \
VAL_SPLIT=val_tune \
OPEN_LOOP_SPLIT=val_tune \
BATCH_SIZE_PER_GPU=4 \
NUM_WORKERS_PER_GPU=6 \
PREFETCH_FACTOR=2 \
SAVE_EVERY_N_STEPS=500 \
SELECTOR_SCENES_PER_RANK=0 \
SELECTOR_EVERY_N_STEPS=1 \
EXACT_DISTILL_SCENES_PER_RANK=0 \
EXACT_DISTILL_EVERY_N_STEPS=1 \
VAL_SCENARIOS=1000 \
VAL_EVERY_N_EPOCHS=2 \
VAL_DENSE_DIAGNOSTIC=1 \
OPEN_LOOP_MAX_SCENARIOS=1000 \
bash run_v47_d3ce.sh

# ---------------------------------------------------------------------------
# 2. Independent one-sided calibration on val_calib only.
# ---------------------------------------------------------------------------
mkdir -p "$OUT_ROOT/calibration"
python -m bdse.tools.calibrate_v47_adverse_bounds \
  --config bdse/configs/v47_bdse_d3ce_cl.yaml \
  --checkpoint "$OUT_ROOT/train/bdse_v47_d3ce.best.pt" \
  --preprocessed-dir "$BDSE_SPLIT_CACHE" \
  --split val_calib \
  --max-scenarios 5000 \
  --device cuda \
  --alpha 0.05 \
  --beta 1.0 \
  --prior-radius 0.10 \
  --provenance-json "$BDSE_SPLIT_CACHE/calibration_split_provenance.json" \
  --output "$OUT_ROOT/calibration/v47_adverse_calibration.json"

python -m bdse.tools.apply_v47_calibration \
  --config-in bdse/configs/v47_bdse_d3ce_cl.yaml \
  --calibration-json "$OUT_ROOT/calibration/v47_adverse_calibration.json" \
  --config-out "$OUT_ROOT/calibration/v47_bdse_d3ce_cl_calibrated.yaml"

# ---------------------------------------------------------------------------
# 3. Replay calibrated v47 on the same deterministic val_tune 1000 scenes.
# ---------------------------------------------------------------------------
GPUS=0,1 \
OUT_ROOT="$OUT_ROOT" \
RUN_MODE=open_loop \
V47_CKPT="$OUT_ROOT/train/bdse_v47_d3ce.best.pt" \
EVAL_CONFIG="$OUT_ROOT/calibration/v47_bdse_d3ce_cl_calibrated.yaml" \
VAL_SPLIT=val_tune \
OPEN_LOOP_SPLIT=val_tune \
OPEN_LOOP_MAX_SCENARIOS=1000 \
bash run_v47_d3ce.sh

# ---------------------------------------------------------------------------
# 4. Rebuild the frozen control on exactly the same val_tune rows.
#    A single-GPU command is used here for clarity; the scenario order is fixed.
# ---------------------------------------------------------------------------
mkdir -p "$CONTROL_ROOT/open_loop"
CUDA_VISIBLE_DEVICES=0 python -m bdse.experiments.evaluate_open_loop \
  --config "$CONTROL_CONFIG" \
  --checkpoint "$CONTROL_CKPT" \
  --split val_tune \
  --preprocessed-dir "$BDSE_SPLIT_CACHE" \
  --max-scenarios 1000 \
  --device cuda \
  --output "$CONTROL_ROOT/open_loop/control.json" \
  --per-sample-output "$CONTROL_ROOT/open_loop/control.jsonl"

# ---------------------------------------------------------------------------
# 5. Strict paired gate. Do not run closed-loop if this fails.
# ---------------------------------------------------------------------------
python -m bdse.tools.check_v47_d3ce_gate \
  "$OUT_ROOT/open_loop/open_loop_v47_d3ce.json" \
  "$CONTROL_ROOT/open_loop/control.json" \
  --candidate-jsonl "$OUT_ROOT/open_loop/open_loop_v47_d3ce.jsonl" \
  --control-jsonl "$CONTROL_ROOT/open_loop/control.jsonl" \
  --train-log "$OUT_ROOT/train/bdse_v47_d3ce.train_log.jsonl" \
  --latency-target-ms 500 \
  --min-match-gain 0.02 \
  --min-sufficiency-gain 0.01 \
  --min-pair-full-match 0.30 \
  --min-certified-pair-fraction 0.50

# ---------------------------------------------------------------------------
# 6. Only after PASS: paired CL20. Reuse exactly the same token JSON for every
#    compared planner. Increase to CL100 only after safety non-inferiority.
# ---------------------------------------------------------------------------
# NUPLAN_ROOT=/path/to/nuplan \
# BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" \
# CL_TOKEN_SPLIT=val_tune \
# GPUS=0,1 OUT_ROOT="$OUT_ROOT" RUN_MODE=cl20 \
# V47_CKPT="$OUT_ROOT/train/bdse_v47_d3ce.best.pt" \
# EVAL_CONFIG="$OUT_ROOT/calibration/v47_bdse_d3ce_cl_calibrated.yaml" \
# bash run_v47_d3ce.sh

# ---------------------------------------------------------------------------
# 7. Final test protocol: execute only after the complete test cache is built
#    with the same preprocessing/map features as val. The current partial test
#    diagnostics should fail this parity gate and must not be reported as final.
# ---------------------------------------------------------------------------
# python -m bdse.tools.check_dataset_diagnostics_parity \
#   /path/to/diagnostics_val.json \
#   /path/to/diagnostics_test_complete.json \
#   --min-scenarios <EXPECTED_COMPLETE_TEST_COUNT>
#
# CUDA_VISIBLE_DEVICES=0 python -m bdse.experiments.evaluate_open_loop \
#   --config "$OUT_ROOT/calibration/v47_bdse_d3ce_cl_calibrated.yaml" \
#   --checkpoint "$OUT_ROOT/train/bdse_v47_d3ce.best.pt" \
#   --split test --preprocessed-dir "$BDSE_TEST_CACHE" --device cuda \
#   --output "$OUT_ROOT/open_loop/open_loop_v47_d3ce_test.json" \
#   --per-sample-output "$OUT_ROOT/open_loop/open_loop_v47_d3ce_test.jsonl"
