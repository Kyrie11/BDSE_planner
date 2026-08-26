#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export NUPLAN_ROOT="${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}"
export NUPLAN_MAP_ROOT="${NUPLAN_MAP_ROOT:-$NUPLAN_ROOT/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-$NUPLAN_ROOT/exp}"
export NUPLAN_TEST_DB_ROOT="${NUPLAN_TEST_DB_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits/test}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2}"
export EXTERNAL_OUT_ROOT="${EXTERNAL_OUT_ROOT:-outputs/external_fixed_budget}"
export CLOSED_LOOP_OUT_ROOT="${CLOSED_LOOP_OUT_ROOT:-outputs/closed_loop/fixed_budget_test}"
export OWN_CONFIG="${OWN_CONFIG:-}"
export OWN_CKPT="${OWN_CKPT:-}"
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
[[ -n "$OWN_CONFIG" && -s "$OWN_CONFIG" ]] || { echo 'Set OWN_CONFIG to the resolved deployed V64.3.48 config (preferred sign arm).' >&2; exit 2; }
[[ -n "$OWN_CKPT" && -s "$OWN_CKPT" ]] || { echo 'Set OWN_CKPT to the frozen EAF checkpoint selected by V64.3.48 provenance.' >&2; exit 2; }

python -m bdse.tools.run_fixed_budget_closed_loop_suite \
  --own-config "$OWN_CONFIG" \
  --own-checkpoint "$OWN_CKPT" \
  --external-checkpoint-root "$EXTERNAL_OUT_ROOT" \
  --split-cache "$BDSE_TEST_CACHE" \
  --token-split public_set_test \
  --limit "$CL_LIMIT" \
  --nuplan-root "$NUPLAN_ROOT" \
  --nuplan-map-root "$NUPLAN_MAP_ROOT" \
  --nuplan-exp-root "$NUPLAN_EXP_ROOT" \
  --nuplan-db-root "$NUPLAN_TEST_DB_ROOT" \
  --budgets $BUDGETS \
  --proposal-top-m "$PROPOSAL_TOP_M" \
  --challenge "$CL_CHALLENGE" \
  --output-root "$CLOSED_LOOP_OUT_ROOT" \
  --gpus "$GPUS" \
  --workers-per-job "$CL_WORKERS_PER_JOB" \
  --schedule-mode model_pairs \
  --resume
