#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_4_fpcca_f8_lba_daepc_fast_2gpu_v1}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-bdse/configs/v64_3_4_cc_aocc_fpcca_f8_lba_daepc_train_2gpu.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-bdse/configs/v64_3_4_cc_aocc_fpcca_f8_cl.yaml}"
export FOUNDATION_CONTROL_CONFIG="${FOUNDATION_CONTROL_CONFIG:-bdse/configs/v64_3_4_cc_aocc_fpcca_f8_anchor_control_cl.yaml}"
export LOCAL_CONTROL_CONFIG="${LOCAL_CONTROL_CONFIG:-bdse/configs/v64_3_4_cc_aocc_fpcca_f8_local_control_cl.yaml}"
export TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-16}"
export TRAIN_NUM_WORKERS_PER_GPU="${TRAIN_NUM_WORKERS_PER_GPU:-12}"
export VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}"
export OPEN_LOOP_WORKERS_PER_GPU="${OPEN_LOOP_WORKERS_PER_GPU:-2}"
export EXPECTED_V64_FAMILY=v64.3.4
export PIPELINE_DETACH="${PIPELINE_DETACH:-0}"
exec bash V64_3_CC_AOCC_APWCCA_NEXT_COMMANDS.sh
