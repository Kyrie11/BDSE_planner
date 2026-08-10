#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT to the immutable matched foundation checkpoint}"
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2}"
export GPUS="${GPUS:-0,1}"
export NPROC_PER_NODE=2
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
export NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}"
export VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

MATRIX_ROOT="${MATRIX_ROOT:-outputs_v64_3_4_acquisition_screen_matrix_2gpu_v1}"
mkdir -p "$MATRIX_ROOT"

run_screen() {
  local name="$1"; shift
  local out_root="$1"; shift
  local report="$out_root/provenance/critical_acquisition_screen.json"
  if [[ -s "$report" ]]; then
    echo "[v64.3.4] reuse completed screen $name: $report"
    return 0
  fi
  echo "[v64.3.4] run screen $name on GPUS=$GPUS"
  OUT_ROOT="$out_root" "$@"
}

# 1) Finish the unrun V64.3.3 hypothesis first: does within-literal-critical
# severity/value help the already-tested winner+rival representation?
run_screen APWRCCA_LCV \
  "$MATRIX_ROOT/apwrcca_lcv" \
  bash RUN_V64_3_4_APWRCCA_LCV_SCREEN_2GPU.sh

# 2) Representation-only causal ablation: remove the mostly-wrong single-winner
# anchor, but do not yet add literal boundary-attribution supervision.
run_screen FPCCA_NOLBA \
  "$MATRIX_ROOT/fpcca_nolba" \
  bash RUN_V64_3_4_FPCCA_NOLBA_ACTIVATION_SCREEN_2GPU.sh

# 3) Main V64.3.4 mechanism: same FPCCA representation + LBA.
run_screen FPCCA_LBA \
  "$MATRIX_ROOT/fpcca_lba" \
  bash RUN_V64_3_4_FPCCA_LBA_ACTIVATION_SCREEN_2GPU.sh

COMPARE_ARGS=(
  --screen "APWRCCA_LCV=$MATRIX_ROOT/apwrcca_lcv/provenance/critical_acquisition_screen.json"
  --screen "FPCCA_NOLBA=$MATRIX_ROOT/fpcca_nolba/provenance/critical_acquisition_screen.json"
  --screen "FPCCA_LBA=$MATRIX_ROOT/fpcca_lba/provenance/critical_acquisition_screen.json"
  --output "$MATRIX_ROOT/acquisition_screen_comparison.json"
)
python -m bdse.tools.compare_v64_3_4_acquisition_screens "${COMPARE_ARGS[@]}"

# F=8 is not a generic capacity sweep.  It is allowed only when F=6 fails and
# the *anchor* diagnostic says the literal boundary is frequently outside top-6.
WINNER="$(python - "$MATRIX_ROOT/acquisition_screen_comparison.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('winner') or '')
PY
)"
if [[ -z "$WINNER" ]]; then
  NEED_F8="$(python - "$MATRIX_ROOT/fpcca_lba/provenance/critical_acquisition_screen.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])).get('anchor_val_critical_boundary_in_base_top6_fraction')
print('1' if x is not None and float(x) < 0.70 else '0')
PY
)"
  if [[ "$NEED_F8" == "1" ]]; then
    echo "[v64.3.4] FPCCA F6 boundary support <0.70; run the pre-registered F8 support screen"
    run_screen FPCCA_F8_LBA \
      "$MATRIX_ROOT/fpcca_f8_lba" \
      bash RUN_V64_3_4_FPCCA_F8_LBA_ACTIVATION_SCREEN_2GPU.sh
    python -m bdse.tools.compare_v64_3_4_acquisition_screens \
      --screen "APWRCCA_LCV=$MATRIX_ROOT/apwrcca_lcv/provenance/critical_acquisition_screen.json" \
      --screen "FPCCA_NOLBA=$MATRIX_ROOT/fpcca_nolba/provenance/critical_acquisition_screen.json" \
      --screen "FPCCA_LBA=$MATRIX_ROOT/fpcca_lba/provenance/critical_acquisition_screen.json" \
      --screen "FPCCA_F8_LBA=$MATRIX_ROOT/fpcca_f8_lba/provenance/critical_acquisition_screen.json" \
      --output "$MATRIX_ROOT/acquisition_screen_comparison.json"
  fi
fi

WINNER="$(python - "$MATRIX_ROOT/acquisition_screen_comparison.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('winner') or '')
PY
)"
echo "[v64.3.4] screen winner: ${WINNER:-NONE}"

if [[ "${AUTO_RUN_FULL:-0}" == "1" ]]; then
  [[ -n "$WINNER" ]] || { echo "No screen meets promotion thresholds; full training is blocked." >&2; exit 4; }
  export RUN_CLOSED_LOOP_AFTER_GATE="${RUN_CLOSED_LOOP_AFTER_GATE:-1}"
  export RUN_DIAGNOSTIC_CL20_ON_GATE_FAIL="${RUN_DIAGNOSTIC_CL20_ON_GATE_FAIL:-1}"
  case "$WINNER" in
    APWRCCA_LCV) exec bash RUN_V64_3_4_APWRCCA_LCV_FULL_PIPELINE_2GPU.sh ;;
    FPCCA_NOLBA) exec bash RUN_V64_3_4_FPCCA_NOLBA_FULL_PIPELINE_2GPU.sh ;;
    FPCCA_LBA) exec bash RUN_V64_3_4_FPCCA_LBA_FULL_PIPELINE_2GPU.sh ;;
    FPCCA_F8_LBA) exec bash RUN_V64_3_4_FPCCA_F8_LBA_FULL_PIPELINE_2GPU.sh ;;
    *) echo "Unknown winner $WINNER" >&2; exit 5 ;;
  esac
fi

cat <<EOF
[v64.3.4] Screening finished. Inspect:
  $MATRIX_ROOT/acquisition_screen_comparison.json
To run the winning full pipeline after inspection, either rerun this script with
AUTO_RUN_FULL=1 (and set NUPLAN_ROOT if closed loop is enabled) or invoke the
matching RUN_V64_3_4_*_FULL_PIPELINE_2GPU.sh wrapper directly.
EOF
