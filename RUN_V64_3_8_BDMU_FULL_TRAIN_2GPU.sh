#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
: "${DARM_DBR_CKPT:?Set DARM_DBR_CKPT to the promoted V64.3.7 DARM+DBR-LITERAL full checkpoint}"
: "${BDMU_SCREEN_REPORT:?Set BDMU_SCREEN_REPORT to v64_3_8_bdmu_screen.json}"
: "${DARM_DBR_FULL_REPORT:?Set DARM_DBR_FULL_REPORT to the passed V64.3.7 full-pipeline audit JSON}"
[[ -s "$DARM_DBR_CKPT" ]] || { echo "missing DARM_DBR_CKPT" >&2; exit 2; }
[[ -s "$BDMU_SCREEN_REPORT" ]] || { echo "missing BDMU_SCREEN_REPORT" >&2; exit 2; }
[[ -s "$DARM_DBR_FULL_REPORT" ]] || { echo "missing DARM_DBR_FULL_REPORT" >&2; exit 2; }
python - "$BDMU_SCREEN_REPORT" "$DARM_DBR_FULL_REPORT" <<'PY'
import json,sys
for name,path in [('V64.3.8 BDMU screen',sys.argv[1]),('V64.3.7 DARM+DBR full',sys.argv[2])]:
    r=json.load(open(path))
    if not r.get('full_promotion', False):
        raise SystemExit(f'{name} did not meet promotion criteria; full BDMU training blocked.')
print('promoted BDMU epoch', json.load(open(sys.argv[1])).get('selected_epoch'))
PY
export FOUNDATION_CKPT="$DARM_DBR_CKPT"
export TRAIN_CONFIG="bdse/configs/v64_3_8_cc_aocc_bdmu_daepc_train_2gpu.yaml"
export EVAL_CONFIG="bdse/configs/v64_3_8_cc_aocc_bdmu_cl.yaml"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_8_bdmu_full_2gpu_v1}"
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export GPUS="${GPUS:-0,1}" NPROC_PER_NODE=2
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
export NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}"
export VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export MAX_TRAIN_SCENARIOS="${MAX_TRAIN_SCENARIOS:-50000}"
export VAL_SCENARIOS="${VAL_SCENARIOS:-1000}"
export OPEN_LOOP_MAX_SCENARIOS="${OPEN_LOOP_MAX_SCENARIOS:-1000}"
export VAL_EVERY_N_EPOCHS=1 VAL_BEFORE_TRAINING=1 BEST_MIN_EPOCH=2 AUTO_RESUME=0 INIT_MODE=warm_start RUN_MODE="${RUN_MODE:-train}"
# Preserve every full-run epoch so final model selection can enforce the paper's
# teacher-action-first / regret-no-harm Pareto rule on validation only.
export SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
python -m bdse.tools.validate_v64_pipeline_config \
  --train-config "$TRAIN_CONFIG" --eval-config "$EVAL_CONFIG" --expected-family v64.3.8 \
  --output "$OUT_ROOT.config_contract.json"
exec bash run_v64_saqa_bcc.sh
