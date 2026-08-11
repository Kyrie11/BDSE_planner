#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT}"; [[ -s "$FOUNDATION_CKPT" ]] || { echo "missing FOUNDATION_CKPT" >&2; exit 2; }
: "${PROMOTED_SCREEN_REPORT:?Set PROMOTED_SCREEN_REPORT to darm_dbr_screen_comparison.json}"; [[ -s "$PROMOTED_SCREEN_REPORT" ]] || exit 2
WINNER="$(python - "$PROMOTED_SCREEN_REPORT" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('winner') or '')
PY
)"
[[ -n "$WINNER" ]] || { echo "No V64.3.7 DARM-DBR screen meets full-promotion thresholds; full training blocked." >&2; exit 4; }
case "$WINNER" in
  BROAD) TAG=broad;; LITERAL) TAG=literal;; *) echo "Unknown winner $WINNER" >&2; exit 5;;
esac
export TRAIN_CONFIG="bdse/configs/v64_3_7_cc_aocc_darm_dbr_${TAG}_daepc_train_2gpu.yaml"
export EVAL_CONFIG="bdse/configs/v64_3_7_cc_aocc_darm_dbr_${TAG}_cl.yaml"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_7_darm_dbr_${TAG}_full_2gpu_v1}"
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2}"
export GPUS="${GPUS:-0,1}" NPROC_PER_NODE=2 BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}" NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}" VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export MAX_TRAIN_SCENARIOS="${MAX_TRAIN_SCENARIOS:-50000}" VAL_SCENARIOS="${VAL_SCENARIOS:-1000}" VAL_EVERY_N_EPOCHS=1 VAL_BEFORE_TRAINING=1 BEST_MIN_EPOCH=3 AUTO_RESUME=0 INIT_MODE=warm_start RUN_MODE=train_open_loop
python -m bdse.tools.validate_v64_pipeline_config --train-config "$TRAIN_CONFIG" --eval-config "$EVAL_CONFIG" --expected-family v64.3.7 --output "$OUT_ROOT.config_contract.json"
exec bash run_v64_saqa_bcc.sh
