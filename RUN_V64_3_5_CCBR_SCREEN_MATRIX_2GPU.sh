#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT to the immutable matched foundation checkpoint}"
[[ -s "$FOUNDATION_CKPT" ]] || { echo "FOUNDATION_CKPT does not exist: $FOUNDATION_CKPT" >&2; exit 2; }
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2}"
export GPUS="${GPUS:-0,1}"
export NPROC_PER_NODE=2
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
export NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}"
export VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MATRIX_ROOT="${MATRIX_ROOT:-outputs_v64_3_5_ccbr_screen_matrix_2gpu_v1}"
mkdir -p "$MATRIX_ROOT"
run_screen() {
  local name="$1" out="$2" script="$3"
  local report="$out/provenance/critical_acquisition_screen.json"
  if [[ -s "$report" ]]; then
    echo "[v64.3.5] reuse completed $name screen"
  else
    OUT_ROOT="$out" bash "$script"
  fi
}
run_screen CCBR_NOLEA "$MATRIX_ROOT/ccbr_nolea" RUN_V64_3_5_CCBR_NOLEA_SCREEN_2GPU.sh
run_screen CCBR_LEA "$MATRIX_ROOT/ccbr_lea" RUN_V64_3_5_CCBR_LEA_SCREEN_2GPU.sh
python -m bdse.tools.compare_v64_3_5_ccbr_screens \
  --screen "CCBR_NOLEA=$MATRIX_ROOT/ccbr_nolea/provenance/critical_acquisition_screen.json" \
  --screen "CCBR_LEA=$MATRIX_ROOT/ccbr_lea/provenance/critical_acquisition_screen.json" \
  --output "$MATRIX_ROOT/acquisition_screen_comparison.json"
WINNER="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("winner") or "")' "$MATRIX_ROOT/acquisition_screen_comparison.json")"
echo "[v64.3.5] screen winner: ${WINNER:-NONE}"
if [[ "${AUTO_RUN_FULL:-0}" == "1" ]]; then
  [[ -n "$WINNER" ]] || { echo "No CCBR screen meets promotion thresholds; full training blocked." >&2; exit 4; }
  export PROMOTED_SCREEN_REPORT="$MATRIX_ROOT/acquisition_screen_comparison.json"
  case "$WINNER" in
    CCBR_NOLEA) exec bash RUN_V64_3_5_CCBR_NOLEA_FULL_PIPELINE_2GPU.sh ;;
    CCBR_LEA) exec bash RUN_V64_3_5_CCBR_LEA_FULL_PIPELINE_2GPU.sh ;;
    *) echo "Unknown winner $WINNER" >&2; exit 5 ;;
  esac
fi
echo "[v64.3.5] screening complete: $MATRIX_ROOT/acquisition_screen_comparison.json"
