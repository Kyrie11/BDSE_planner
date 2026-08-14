#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

: "${DARM_DBR_CKPT:?Set DARM_DBR_CKPT to promoted V64.3.7 DARM+DBR-LITERAL checkpoint}"
: "${DARM_DBR_FULL_REPORT:?Set DARM_DBR_FULL_REPORT to promoted V64.3.7 full report}"

export RET_OUT_ROOT="${RET_OUT_ROOT:-outputs_v64_3_12_ret_bdmu_screen_2gpu_v1}"
export CET_OUT_ROOT="${CET_OUT_ROOT:-outputs_v64_3_12_cet_bdmu_screen_2gpu_v1}"

# Arm A: exact runtime transmission targets, blanket current-B protection.
OUT_ROOT="$RET_OUT_ROOT" bash RUN_V64_3_12_RET_BDMU_SCREEN_2GPU.sh
RET_REPORT="$RET_OUT_ROOT/provenance/v64_3_12_ret_bdmu_screen.json"

# Arm B: same exact targets + controlled current-B exchange only when oracle-B drops it.
OUT_ROOT="$CET_OUT_ROOT" bash RUN_V64_3_12_CET_BDMU_SCREEN_2GPU.sh
CET_REPORT="$CET_OUT_ROOT/provenance/v64_3_12_cet_bdmu_screen.json"

python -m bdse.tools.compare_v64_3_12_ret_cet_screens \
  --ret "$RET_REPORT" --cet "$CET_REPORT" \
  --output v64_3_12_ret_cet_screen_comparison.json
cat v64_3_12_ret_cet_screen_comparison.json
