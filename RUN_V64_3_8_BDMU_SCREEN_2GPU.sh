#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
: "${DARM_DBR_CKPT:?Set DARM_DBR_CKPT to the promoted V64.3.7 DARM+DBR-LITERAL full-training checkpoint}"
[[ -s "$DARM_DBR_CKPT" ]] || { echo "missing DARM_DBR_CKPT=$DARM_DBR_CKPT" >&2; exit 2; }
: "${DARM_DBR_FULL_REPORT:?Set DARM_DBR_FULL_REPORT to the V64.3.7 full-pipeline audit JSON}"
[[ -s "$DARM_DBR_FULL_REPORT" ]] || { echo "missing DARM_DBR_FULL_REPORT=$DARM_DBR_FULL_REPORT" >&2; exit 2; }
python - "$DARM_DBR_FULL_REPORT" <<'PY_GATE'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('full_promotion', False):
    raise SystemExit('V64.3.7 DARM+DBR did not reproduce the value/deployment gain on the full pipeline; BDMU acquisition training is blocked.')
print('validated DARM+DBR full epoch', r.get('selected_epoch'))
PY_GATE
export FOUNDATION_CKPT="$DARM_DBR_CKPT"
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export GPUS="${GPUS:-0,1}" NPROC_PER_NODE=2
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
export NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}"
export VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export MAX_TRAIN_SCENARIOS="${MAX_TRAIN_SCENARIOS:-12000}"
export VAL_SCENARIOS="${VAL_SCENARIOS:-500}"
export VAL_EVERY_N_EPOCHS=1 VAL_BEFORE_TRAINING=1 BEST_MIN_EPOCH=1 AUTO_RESUME=0 INIT_MODE=warm_start RUN_MODE=train
export TRAIN_CONFIG="bdse/configs/v64_3_8_cc_aocc_bdmu_daepc_screen_2gpu.yaml"
export EVAL_CONFIG="bdse/configs/v64_3_8_cc_aocc_bdmu_cl.yaml"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_8_bdmu_screen_2gpu_v1}"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"
python -m bdse.tools.validate_v64_pipeline_config \
  --train-config "$TRAIN_CONFIG" --eval-config "$EVAL_CONFIG" --expected-family v64.3.8 \
  --output "$OUT_ROOT/provenance/config_contract.json"
bash run_v64_saqa_bcc.sh
python -m bdse.tools.check_v64_3_8_bdmu_screen \
  --train-log "$OUT_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl" \
  --output "$OUT_ROOT/provenance/v64_3_8_bdmu_screen.json"
echo "[v64.3.8] screen report: $OUT_ROOT/provenance/v64_3_8_bdmu_screen.json"
