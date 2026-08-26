#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep the exact scientific V30 seed/configs.  The uploaded run stopped before
# fresh-token selection, so no new fresh population is needed for this pure
# engineering rerun.  Use a new output directory to avoid mixing invalid and
# repaired provenance.
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_30_eaf_icer_fbic_screen_2gpu_hotfix_v1}"
exec bash "$SCRIPT_DIR/RUN_V64_3_30_EAF_ICER_FBIC_SCREEN_2GPU.sh"
