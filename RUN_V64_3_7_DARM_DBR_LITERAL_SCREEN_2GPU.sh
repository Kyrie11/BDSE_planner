#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT; recommended: outputs_v62_dcab_ewfc_fast_2gpu_v1/train/bdse_v62_dcab_ewfc.best.pt}"
[[ -s "$FOUNDATION_CKPT" ]] || { echo "FOUNDATION_CKPT does not exist: $FOUNDATION_CKPT" >&2; exit 2; }
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export GPUS="${GPUS:-0,1}" NPROC_PER_NODE=2 BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}" NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}" VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export MAX_TRAIN_SCENARIOS="${MAX_TRAIN_SCENARIOS:-12000}" VAL_SCENARIOS="${VAL_SCENARIOS:-500}" VAL_EVERY_N_EPOCHS=1 VAL_BEFORE_TRAINING=1 BEST_MIN_EPOCH=3 AUTO_RESUME=0
export INIT_MODE=warm_start RUN_MODE=train
export TRAIN_CONFIG="bdse/configs/v64_3_7_cc_aocc_darm_dbr_literal_daepc_screen_2gpu.yaml"
export EVAL_CONFIG="bdse/configs/v64_3_7_cc_aocc_darm_dbr_literal_cl.yaml"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_7_darm_dbr_literal_screen_2gpu_v1}"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"
python -m bdse.tools.validate_v64_pipeline_config --train-config "$TRAIN_CONFIG" --eval-config "$EVAL_CONFIG" --expected-family v64.3.7 --output "$OUT_ROOT/provenance/config_contract.json"
bash run_v64_saqa_bcc.sh
python -m bdse.tools.check_v64_3_7_darm_dbr_screen --train-log "$OUT_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl" --variant 'DARM+DBR-LITERAL' --output "$OUT_ROOT/provenance/v64_3_7_darm_dbr_screen.json"
