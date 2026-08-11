#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT; recommended: outputs_v62_dcab_ewfc_fast_2gpu_v1/train/bdse_v62_dcab_ewfc.best.pt}"
[[ -s "$FOUNDATION_CKPT" ]] || { echo "FOUNDATION_CKPT does not exist: $FOUNDATION_CKPT" >&2; exit 2; }
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2}"
export GPUS="${GPUS:-0,1}" BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}" NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}" VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
ROOT="${MATRIX_ROOT:-outputs_v64_3_6_dual_bottleneck_screen_matrix_2gpu_v1}"; mkdir -p "$ROOT"
run_one(){ local tag="$1" script="$2"; local out="$ROOT/$tag"; if [[ -s "$out/provenance/v64_3_6_dual_screen.json" ]]; then echo "[v64.3.6] reuse $tag"; else OUT_ROOT="$out" bash "$script"; fi; }
run_one local RUN_V64_3_6_LOCAL_SCREEN_2GPU.sh
run_one lbpr RUN_V64_3_6_LBPR_SCREEN_2GPU.sh
FAMILY_NEEDED="$(python - "$ROOT/local/provenance/v64_3_6_dual_screen.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(int(bool(x.get('family_admission_ceiling_indicated')) or x.get('frozen_family_slot_oracle_topm_recall_max') is None))
PY
)"
if [[ "$FAMILY_NEEDED" == "1" || "${FORCE_BCHA:-0}" == "1" || "${RUN_FULL_2X2:-0}" == "1" ]]; then
  run_one bcha RUN_V64_3_6_BCHA_SCREEN_2GPU.sh
  SHOULD_JOINT="$(python - "$ROOT/lbpr/provenance/v64_3_6_dual_screen.json" "$ROOT/bcha/provenance/v64_3_6_dual_screen.json" <<'PY'
import json,sys
xs=[json.load(open(p)) for p in sys.argv[1:]]
print(int(any(x.get('meaningful_acquisition_gain') or x.get('meaningful_value_gain') for x in xs)))
PY
)"
  if [[ "$SHOULD_JOINT" == "1" || "${FORCE_JOINT:-0}" == "1" || "${RUN_FULL_2X2:-0}" == "1" ]]; then run_one bcha_lbpr RUN_V64_3_6_BCHA_LBPR_SCREEN_2GPU.sh; fi
fi
ARGS=(); for tag in local lbpr bcha bcha_lbpr; do p="$ROOT/$tag/provenance/v64_3_6_dual_screen.json"; [[ -s "$p" ]] && ARGS+=(--screen "${tag^^}=$p"); done
python -m bdse.tools.compare_v64_3_6_dual_screens "${ARGS[@]}" --output "$ROOT/dual_screen_comparison.json"
echo "[v64.3.6] done: $ROOT/dual_screen_comparison.json"
