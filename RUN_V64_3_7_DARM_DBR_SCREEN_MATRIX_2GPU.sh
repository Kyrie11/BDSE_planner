#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT; recommended: outputs_v62_dcab_ewfc_fast_2gpu_v1/train/bdse_v62_dcab_ewfc.best.pt}"
[[ -s "$FOUNDATION_CKPT" ]] || { echo "FOUNDATION_CKPT does not exist: $FOUNDATION_CKPT" >&2; exit 2; }
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2}"
export GPUS="${GPUS:-0,1}" BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}" NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}" VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
ROOT="${MATRIX_ROOT:-outputs_v64_3_7_darm_dbr_screen_matrix_2gpu_v1}"; mkdir -p "$ROOT"
run_one(){ local tag="$1" script="$2"; local out="$ROOT/$tag"; if [[ -s "$out/provenance/v64_3_7_darm_dbr_screen.json" ]]; then echo "[v64.3.7] reuse $tag"; else OUT_ROOT="$out" bash "$script"; fi; }
run_one broad RUN_V64_3_7_DARM_DBR_BROAD_SCREEN_2GPU.sh
run_one literal RUN_V64_3_7_DARM_DBR_LITERAL_SCREEN_2GPU.sh
python -m bdse.tools.compare_v64_3_7_darm_dbr_screens \
  --screen "BROAD=$ROOT/broad/provenance/v64_3_7_darm_dbr_screen.json" \
  --screen "LITERAL=$ROOT/literal/provenance/v64_3_7_darm_dbr_screen.json" \
  --output "$ROOT/darm_dbr_screen_comparison.json"
echo "[v64.3.7] done: $ROOT/darm_dbr_screen_comparison.json"
