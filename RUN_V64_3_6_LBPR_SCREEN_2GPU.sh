#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT to the immutable V62 foundation checkpoint}"
[[ -s "$FOUNDATION_CKPT" ]] || { echo "FOUNDATION_CKPT does not exist: $FOUNDATION_CKPT" >&2; exit 2; }
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export GPUS="${GPUS:-0,1}"; export NPROC_PER_NODE=2
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
export NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}"
export VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export MAX_TRAIN_SCENARIOS="${MAX_TRAIN_SCENARIOS:-12000}"; export VAL_SCENARIOS="${VAL_SCENARIOS:-500}"
export VAL_EVERY_N_EPOCHS=1; export VAL_BEFORE_TRAINING=1; export BEST_MIN_EPOCH=3; export AUTO_RESUME=0
export INIT_MODE=warm_start; export RUN_MODE=train
export TRAIN_CONFIG="bdse/configs/v64_3_6_cc_aocc_ccbr_lbpr_lea_daepc_screen_2gpu.yaml"
export EVAL_CONFIG="bdse/configs/v64_3_6_cc_aocc_ccbr_lbpr_lea_cl.yaml"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_6_lbpr_screen_2gpu_v1}"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"
python -m bdse.tools.validate_v64_pipeline_config --train-config "$TRAIN_CONFIG" --eval-config "$EVAL_CONFIG" --expected-family v64.3.6 --output "$OUT_ROOT/provenance/config_contract.json"
bash run_v64_saqa_bcc.sh
python -m bdse.tools.check_v64_3_6_dual_bottleneck_screen --train-log "$OUT_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl" --variant 'CCBR+LBPR+LEA' --output "$OUT_ROOT/provenance/v64_3_6_dual_screen.json"
