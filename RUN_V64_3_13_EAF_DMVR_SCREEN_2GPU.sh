#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

: "${DARM_DBR_CKPT:?Set DARM_DBR_CKPT to promoted V64.3.7 DARM+DBR-LITERAL full checkpoint}"
: "${DARM_DBR_FULL_REPORT:?Set DARM_DBR_FULL_REPORT to promoted V64.3.7 full report}"
: "${RET_BDMU_SCREEN_REPORT:?Set RET_BDMU_SCREEN_REPORT to V64.3.12 RET screen report}"
: "${CET_BDMU_SCREEN_REPORT:?Set CET_BDMU_SCREEN_REPORT to V64.3.12 CET screen report}"
for p in "$DARM_DBR_CKPT" "$DARM_DBR_FULL_REPORT" "$RET_BDMU_SCREEN_REPORT" "$CET_BDMU_SCREEN_REPORT"; do
  [[ -s "$p" ]] || { echo "missing required artifact: $p" >&2; exit 2; }
done
python - "$DARM_DBR_FULL_REPORT" "$RET_BDMU_SCREEN_REPORT" "$CET_BDMU_SCREEN_REPORT" <<'PY'
import json,sys
base=json.load(open(sys.argv[1])); ret=json.load(open(sys.argv[2])); cet=json.load(open(sys.argv[3]))
if not base.get('full_promotion', False):
    raise SystemExit('STOP: V64.3.7 DARM+DBR full value checkpoint is not promoted.')
if not ret.get('instrumentation_valid', False) or not cet.get('instrumentation_valid', False):
    raise SystemExit('STOP ENGINEERING: V64.3.12 RET/CET instrumentation invalid; value pivot cannot be interpreted.')
if not cet.get('exact_acquisition_exhausted', False) or not cet.get('pivot_to_value_frontier', False):
    raise SystemExit('STOP: V64.3.12 did not establish the terminal acquisition stop required by V64.3.13.')
print('validated terminal acquisition stop; pivoting to decisive value/frontier')
PY

# Reproduce the V64.3.12 anchor rather than warm-starting a failed RET/CET epoch.
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
export VAL_MAX_SCENARIOS_STRATEGY="${VAL_MAX_SCENARIOS_STRATEGY:-uniform_blocks}"
export MAX_SCENARIOS_STRATEGY="${MAX_SCENARIOS_STRATEGY:-first}"
export VAL_MODE="${VAL_MODE:-both}"
export VAL_EVERY_N_EPOCHS=1 VAL_BEFORE_TRAINING=1 BEST_MIN_EPOCH=1 AUTO_RESUME=0 INIT_MODE=warm_start RUN_MODE=train
export SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
export TRAIN_CONFIG="bdse/configs/v64_3_13_eaf_dmvr_daepc_screen_2gpu.yaml"
export EVAL_CONFIG="bdse/configs/v64_3_13_eaf_dmvr_cl.yaml"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"

python -m bdse.tools.validate_v64_pipeline_config \
  --train-config "$TRAIN_CONFIG" --eval-config "$EVAL_CONFIG" --expected-family v64.3.13 \
  --output "$OUT_ROOT/provenance/config_contract.json"
python -m bdse.tools.check_v64_3_13_eaf_dmvr_contract \
  --config "$TRAIN_CONFIG" --output "$OUT_ROOT/provenance/eaf_dmvr_contract.json"

bash run_v64_saqa_bcc.sh

python -m bdse.tools.validate_training_artifacts \
  --output-root "$OUT_ROOT" --require-epoch-checkpoint \
  --output "$OUT_ROOT/provenance/training_artifact_contract.json"
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
  --train-log "$OUT_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl" \
  --variant SCREEN --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_screen.json"
echo "[v64.3.13] screen report: $OUT_ROOT/provenance/v64_3_13_eaf_dmvr_screen.json"
