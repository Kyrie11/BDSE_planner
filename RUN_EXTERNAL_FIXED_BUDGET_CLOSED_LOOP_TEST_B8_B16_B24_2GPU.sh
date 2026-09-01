#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BDSE_ROOT="${BDSE_ROOT:-$SCRIPT_DIR}"
BDSE_ROOT="$(cd "$BDSE_ROOT" && pwd)"
if [[ "$BDSE_ROOT" != "$SCRIPT_DIR" ]]; then
  echo "BDSE_ROOT must be the repository root containing this launcher: script=$SCRIPT_DIR BDSE_ROOT=$BDSE_ROOT" >&2
  exit 2
fi
cd "$BDSE_ROOT"
export PYTHONPATH="$BDSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$BDSE_ROOT/outputs}"

export NUPLAN_ROOT="${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}"
export NUPLAN_MAP_ROOT="${NUPLAN_MAP_ROOT:-$NUPLAN_ROOT/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-$NUPLAN_ROOT/exp}"
# User-provided raw nuPlan v1.1 TEST split: .db files are directly in this folder.
export NUPLAN_TEST_DB_ROOT="${NUPLAN_TEST_DB_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits/test}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2}"
export EXTERNAL_OUT_ROOT="${EXTERNAL_OUT_ROOT:-$OUTPUTS_ROOT/external_fixed_budget}"
export EXTERNAL_CL_OUT_ROOT="${EXTERNAL_CL_OUT_ROOT:-$OUTPUTS_ROOT/closed_loop/external_fixed_budget_test}"
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

[[ -d "$NUPLAN_TEST_DB_ROOT" ]] || { echo "missing raw test DB directory: $NUPLAN_TEST_DB_ROOT" >&2; exit 2; }
if ! find "$NUPLAN_TEST_DB_ROOT" -maxdepth 1 -type f -name '*.db' -print -quit | grep -q .; then
  echo "No direct *.db files found in $NUPLAN_TEST_DB_ROOT" >&2; exit 2
fi
[[ -d "$BDSE_TEST_CACHE/public_set_test" ]] || { echo "missing NPZ test cache split: $BDSE_TEST_CACHE/public_set_test" >&2; exit 2; }

echo "IMPORTANT: this script performs CLOSED-LOOP EVALUATION ONLY; it does not train models."
echo "Training command: bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh"
echo "Raw nuPlan test DB root: $NUPLAN_TEST_DB_ROOT"
echo "DB file count: $(find "$NUPLAN_TEST_DB_ROOT" -maxdepth 1 -type f -name '*.db' | wc -l)"
echo "Two-GPU schedule: model pairs, each model runs B=$BUDGETS sequentially on one fixed GPU."
echo "nuPlan workers per model job: $CL_WORKERS_PER_JOB"
echo "Progress heartbeat: every ${CL_HEARTBEAT_SECONDS}s; manifest scan workers: $CL_TOKEN_SCAN_WORKERS"

# Fail immediately if training has not produced the budget-specific checkpoints.
# The Python runner repeats this check for direct invocation, but doing it here
# avoids an expensive test-cache scan before reporting a simple missing-model error.
missing=0
for B in $BUDGETS; do
  for name in gameformer dtpp plantf pluto; do
    ckpt="$EXTERNAL_OUT_ROOT/B${B}/${name}_budgeted.best.pt"
    if [[ ! -s "$ckpt" ]]; then
      echo "MISSING CHECKPOINT: $ckpt" >&2
      missing=1
    fi
  done
done
if (( missing != 0 )); then
  echo >&2
  echo "Closed-loop evaluation cannot start because required trained checkpoints are missing." >&2
  echo "Run first: bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh" >&2
  exit 3
fi
echo "Checkpoint preflight: OK"

python -m bdse.tools.run_fixed_budget_closed_loop_suite \
  --external-checkpoint-root "$EXTERNAL_OUT_ROOT" \
  --split-cache "$BDSE_TEST_CACHE" --token-split public_set_test --limit "$CL_LIMIT" \
  --nuplan-root "$NUPLAN_ROOT" --nuplan-map-root "$NUPLAN_MAP_ROOT" --nuplan-exp-root "$NUPLAN_EXP_ROOT" --nuplan-db-root "$NUPLAN_TEST_DB_ROOT" \
  --budgets $BUDGETS --proposal-top-m "$PROPOSAL_TOP_M" \
  --challenge "$CL_CHALLENGE" --output-root "$EXTERNAL_CL_OUT_ROOT" \
  --gpus "$GPUS" --workers-per-job "$CL_WORKERS_PER_JOB" --schedule-mode model_pairs \
  --token-scan-workers "$CL_TOKEN_SCAN_WORKERS" --token-progress-seconds "$CL_TOKEN_PROGRESS_SECONDS" \
  --heartbeat-seconds "$CL_HEARTBEAT_SECONDS" \
  --systems gameformer dtpp plantf pluto pdm_closed_style --resume
