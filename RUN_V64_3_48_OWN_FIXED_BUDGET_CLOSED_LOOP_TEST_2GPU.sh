#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export NUPLAN_ROOT="${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}"
export NUPLAN_MAP_ROOT="${NUPLAN_MAP_ROOT:-$NUPLAN_ROOT/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-$NUPLAN_ROOT/exp}"
export NUPLAN_TEST_DB_ROOT="${NUPLAN_TEST_DB_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits/test}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2}"
export V48_ROOT="${V48_ROOT:-outputs_v64_3_48_eaf_icer_ocrr_screen_2gpu_v1}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export OWN_CLOSED_LOOP_OUT_ROOT="${OWN_CLOSED_LOOP_OUT_ROOT:-outputs/closed_loop/v64_3_48_fixed_budget_test}"
export GPUS="${GPUS:-0,1}"
export BUDGETS="${BUDGETS:-8 16 24}"
export PROPOSAL_TOP_M="${PROPOSAL_TOP_M:-24}"
export CL_CHALLENGE="${CL_CHALLENGE:-closed_loop_nonreactive_agents}"
export CL_LIMIT="${CL_LIMIT:-0}"
export CL_WORKERS_PER_JOB="${CL_WORKERS_PER_JOB:-4}"

[[ -d "$NUPLAN_TEST_DB_ROOT" ]] || { echo "missing raw test DB directory: $NUPLAN_TEST_DB_ROOT" >&2; exit 2; }
if ! find "$NUPLAN_TEST_DB_ROOT" -maxdepth 1 -type f -name '*.db' -print -quit | grep -q .; then
  echo "No direct *.db files found in $NUPLAN_TEST_DB_ROOT" >&2; exit 2
fi
FIT="$V48_ROOT/provenance/v64_3_48_ocrr_fit.json"
SCREEN="$V48_ROOT/provenance/v64_3_48_eaf_icer_ocrr_double_fresh_screen.json"
REAUDIT="$V48_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_48.json"
for f in "$FIT" "$SCREEN" "$REAUDIT"; do [[ -s "$f" ]] || { echo "missing V48 provenance: $f" >&2; exit 2; }; done

read -r PREFERRED SELECTED_EPOCH < <(python - "$FIT" "$SCREEN" "$REAUDIT" <<'PY'
import json,sys
fit=json.load(open(sys.argv[1])); screen=json.load(open(sys.argv[2])); rea=json.load(open(sys.argv[3]))
if not bool(screen.get('pass', False)):
    raise SystemExit('V64.3.48 double-fresh screen did not pass; do not report V48 as the promoted own model.')
p=fit.get('nested_crossfit',{}).get('preferred_promotion_arm')
if p not in {'sign_nomult','sign_mult'}:
    raise SystemExit(f'invalid preferred arm: {p!r}')
print(p, int(rea['selected_epoch']))
PY
)
OWN_CONFIG="$V48_ROOT/provenance/v64_3_48_${PREFERRED}.yaml"
OWN_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"
[[ -s "$OWN_CONFIG" ]] || { echo "missing own config: $OWN_CONFIG" >&2; exit 2; }
[[ -s "$OWN_CKPT" ]] || { echo "missing own checkpoint: $OWN_CKPT" >&2; exit 2; }
echo "Resolved V48 own model: arm=$PREFERRED config=$OWN_CONFIG checkpoint=$OWN_CKPT"

# The suite CLI requires an external checkpoint root even for --systems bdse; it is not read for this system.
DUMMY_EXTERNAL_ROOT="${EXTERNAL_OUT_ROOT:-outputs/external_fixed_budget}"
python -m bdse.tools.run_fixed_budget_closed_loop_suite \
  --own-config "$OWN_CONFIG" --own-checkpoint "$OWN_CKPT" \
  --external-checkpoint-root "$DUMMY_EXTERNAL_ROOT" \
  --split-cache "$BDSE_TEST_CACHE" --token-split public_set_test --limit "$CL_LIMIT" \
  --nuplan-root "$NUPLAN_ROOT" --nuplan-map-root "$NUPLAN_MAP_ROOT" --nuplan-exp-root "$NUPLAN_EXP_ROOT" --nuplan-db-root "$NUPLAN_TEST_DB_ROOT" \
  --budgets $BUDGETS --proposal-top-m "$PROPOSAL_TOP_M" \
  --challenge "$CL_CHALLENGE" --output-root "$OWN_CLOSED_LOOP_OUT_ROOT" \
  --gpus "$GPUS" --workers-per-job "$CL_WORKERS_PER_JOB" --schedule-mode model_pairs --systems bdse --resume
