#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Resume the already-trained V64 run from calibration/open-loop without changing
# its algorithm/config identity.  Use this first to obtain the official gate
# report before launching the V64.2 HCBE ablation.
export OUT_ROOT="${OUT_ROOT:-outputs_v64_saqa_bcc_fast_2gpu_v1}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-bdse/configs/v64_saqa_bcc_train_2gpu.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-bdse/configs/v64_saqa_bcc_cl.yaml}"
export PIPELINE_FORCE="${PIPELINE_FORCE:-0}"
export PIPELINE_DETACH="${PIPELINE_DETACH:-1}"
export RUN_CLOSED_LOOP_AFTER_GATE="${RUN_CLOSED_LOOP_AFTER_GATE:-0}"
export SKIP_V64_TRAINING="${SKIP_V64_TRAINING:-1}"
export V64_CANDIDATE_CHECKPOINT="${V64_CANDIDATE_CHECKPOINT:-$OUT_ROOT/train/bdse_v64_saqa_bcc.best.pt}"

exec bash "$SCRIPT_DIR/V64_SAQA_BCC_NEXT_COMMANDS.sh"
