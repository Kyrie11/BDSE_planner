#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BDSE_ROOT="${BDSE_ROOT:-$SCRIPT_DIR}"
BDSE_ROOT="$(cd "$BDSE_ROOT" && pwd)"
[[ "$BDSE_ROOT" == "$SCRIPT_DIR" ]] || { echo "BDSE_ROOT must be repository root: $SCRIPT_DIR" >&2; exit 2; }
cd "$BDSE_ROOT"
export PYTHONPATH="$BDSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$BDSE_ROOT/outputs}"

# This is evaluation-only.  The converged BDSE/EAF-RSMR learned artifacts were
# frozen at B=16.  B=8/B=24 are cross-budget interface robustness ablations,
# not budget-specific retraining.  Do not use test results to refit EAF/RSMR.
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export V47_ROOT="${V47_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_47_eaf_icer_fsfr_screen_2gpu_v1}"
export OWN_CONFIG="${OWN_CONFIG:-$V47_ROOT/provenance/v64_3_47_rsmr.yaml}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"

export OWN_CL_OUT_ROOT="${OWN_CL_OUT_ROOT:-$OUTPUTS_ROOT/closed_loop/v64_3_56_own_frozen_budget_robustness}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2}"
export NUPLAN_ROOT="${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}"
export NUPLAN_MAP_ROOT="${NUPLAN_MAP_ROOT:-$NUPLAN_ROOT/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-$NUPLAN_ROOT/exp}"
export NUPLAN_TEST_DB_ROOT="${NUPLAN_TEST_DB_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits/test}"
export GPUS="${GPUS:-0,1}"
export BUDGETS="${BUDGETS:-8 16 24}"
export PROPOSAL_TOP_M="${PROPOSAL_TOP_M:-24}"
export CL_CHALLENGE="${CL_CHALLENGE:-closed_loop_nonreactive_agents}"
export CL_LIMIT="${CL_LIMIT:-0}"
export CL_WORKERS_PER_JOB="${CL_WORKERS_PER_JOB:-4}"
export CL_TOKEN_SCAN_WORKERS="${CL_TOKEN_SCAN_WORKERS:-8}"
export CL_TOKEN_PROGRESS_SECONDS="${CL_TOKEN_PROGRESS_SECONDS:-5}"
export CL_HEARTBEAT_SECONDS="${CL_HEARTBEAT_SECONDS:-15}"
export PYTHONUNBUFFERED=1

[[ -s "$OWN_CONFIG" ]] || { echo "missing frozen RSMR config: $OWN_CONFIG" >&2; exit 2; }
[[ -s "$EAF_TRAIN_LOG" ]] || { echo "missing V13 train log: $EAF_TRAIN_LOG" >&2; exit 2; }
[[ -d "$BDSE_TEST_CACHE/public_set_test" ]] || { echo "missing test cache: $BDSE_TEST_CACHE/public_set_test" >&2; exit 2; }
[[ -d "$NUPLAN_TEST_DB_ROOT" ]] || { echo "missing raw test DB root: $NUPLAN_TEST_DB_ROOT" >&2; exit 2; }
find "$NUPLAN_TEST_DB_ROOT" -maxdepth 1 -type f -name '*.db' -print -quit | grep -q . || { echo "no direct .db files under $NUPLAN_TEST_DB_ROOT" >&2; exit 2; }
sha256sum -c V64_3_56_SCIENCE_MANIFEST.sha256 >/dev/null

mkdir -p "$OWN_CL_OUT_ROOT/provenance"
if [[ -z "${EAF_CKPT:-}" ]]; then
  python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
    --train-log "$EAF_TRAIN_LOG" --variant CONVERGED_OWN_BUDGET_ROBUSTNESS \
    --output "$OWN_CL_OUT_ROOT/provenance/v64_3_13_reaudit.json" \
    > "$OWN_CL_OUT_ROOT/provenance/v64_3_13_reaudit.out"
  SELECTED_EPOCH="$(python - "$OWN_CL_OUT_ROOT/provenance/v64_3_13_reaudit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False):
    raise SystemExit('STOP: V13 frozen checkpoint prerequisites changed')
print(int(r['selected_epoch']))
PY
)"
  export EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"
fi
[[ -s "$EAF_CKPT" ]] || { echo "missing frozen EAF checkpoint: $EAF_CKPT" >&2; exit 2; }

python -m bdse.tools.run_fixed_budget_closed_loop_suite \
  --own-config "$OWN_CONFIG" --own-checkpoint "$EAF_CKPT" \
  --own-label "BDSE-EAF-RSMR (frozen B16 learned policy; cross-budget interface robustness)" \
  --split-cache "$BDSE_TEST_CACHE" --token-split public_set_test --limit "$CL_LIMIT" \
  --nuplan-root "$NUPLAN_ROOT" --nuplan-map-root "$NUPLAN_MAP_ROOT" --nuplan-exp-root "$NUPLAN_EXP_ROOT" --nuplan-db-root "$NUPLAN_TEST_DB_ROOT" \
  --budgets $BUDGETS --proposal-top-m "$PROPOSAL_TOP_M" \
  --challenge "$CL_CHALLENGE" --output-root "$OWN_CL_OUT_ROOT/results" \
  --gpus "$GPUS" --workers-per-job "$CL_WORKERS_PER_JOB" --schedule-mode queue \
  --token-scan-workers "$CL_TOKEN_SCAN_WORKERS" --token-progress-seconds "$CL_TOKEN_PROGRESS_SECONDS" \
  --heartbeat-seconds "$CL_HEARTBEAT_SECONDS" \
  --systems bdse --resume

echo "DONE: frozen BDSE cross-budget robustness B=[$BUDGETS] under $OWN_CL_OUT_ROOT"
