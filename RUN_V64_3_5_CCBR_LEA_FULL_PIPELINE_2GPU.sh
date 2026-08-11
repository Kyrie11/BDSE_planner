#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT to an existing immutable matched foundation checkpoint}"
[[ -s "$FOUNDATION_CKPT" ]] || { echo "FOUNDATION_CKPT does not exist: $FOUNDATION_CKPT" >&2; exit 2; }
: "${PROMOTED_SCREEN_REPORT:?Set PROMOTED_SCREEN_REPORT to the CCBR comparison/report that promoted this variant}"
[[ -s "$PROMOTED_SCREEN_REPORT" ]] || { echo "Missing promoted screen report: $PROMOTED_SCREEN_REPORT" >&2; exit 2; }
python - "$PROMOTED_SCREEN_REPORT" "CCBR_LEA" <<'PY_GATE'
import json,sys
p,name=sys.argv[1:]
d=json.load(open(p))
if 'winner' in d:
    ok=str(d.get('winner') or '').upper()==name
else:
    variant=str(d.get('variant') or '').upper().replace('-','_')
    ok=bool(d.get('continue_to_full_run')) and name in variant
if not ok:
    raise SystemExit(f'Full run blocked: {name} was not promoted by {p}')
PY_GATE
export FOUNDATION_POLICY=explicit
export REBUILD_FOUNDATION_IF_MISSING=0
export RECOVER_SAFE_FOUNDATION_COPIES=0
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE_ORIGINAL="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2}"
export BDSE_SPLIT_CACHE="${BDSE_SPLIT_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v64_3_5_split}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_5_ccbr_lea_daepc_2gpu_v1}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-bdse/configs/v64_3_5_cc_aocc_ccbr_lea_daepc_train_2gpu.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-bdse/configs/v64_3_5_cc_aocc_ccbr_cl.yaml}"
export FOUNDATION_CONTROL_CONFIG="${FOUNDATION_CONTROL_CONFIG:-bdse/configs/v64_3_5_cc_aocc_ccbr_anchor_control_cl.yaml}"
export LOCAL_CONTROL_CONFIG="${LOCAL_CONTROL_CONFIG:-bdse/configs/v64_3_5_cc_aocc_ccbr_local_control_cl.yaml}"
export TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-16}"
export TRAIN_NUM_WORKERS_PER_GPU="${TRAIN_NUM_WORKERS_PER_GPU:-12}"
export VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}"
export OPEN_LOOP_WORKERS_PER_GPU="${OPEN_LOOP_WORKERS_PER_GPU:-2}"
export GPUS="${GPUS:-0,1}"
export EXPECTED_V64_FAMILY=v64.3.5
export PIPELINE_DETACH="${PIPELINE_DETACH:-0}"
exec bash V64_3_CC_AOCC_APWCCA_NEXT_COMMANDS.sh
