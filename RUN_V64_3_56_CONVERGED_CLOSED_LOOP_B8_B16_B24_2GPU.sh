#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BDSE_ROOT="${BDSE_ROOT:-$SCRIPT_DIR}"
BDSE_ROOT="$(cd "$BDSE_ROOT" && pwd)"
[[ "$BDSE_ROOT" == "$SCRIPT_DIR" ]] || { echo "BDSE_ROOT must be repository root: $SCRIPT_DIR" >&2; exit 2; }
cd "$BDSE_ROOT"
export PYTHONPATH="$BDSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$BDSE_ROOT/outputs}"

# Frozen internal-search conclusion: benchmark the strongest fully t0-deployable
# backbone (full-set RSMR).  V54-V56 oracle mediators are diagnostic only and
# are intentionally NOT used by this closed-loop runner.
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export V47_ROOT="${V47_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_47_eaf_icer_fsfr_screen_2gpu_v1}"
export OWN_CONFIG="${OWN_CONFIG:-$V47_ROOT/provenance/v64_3_47_rsmr.yaml}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"

export EXTERNAL_OUT_ROOT="${EXTERNAL_OUT_ROOT:-$OUTPUTS_ROOT/external_fixed_budget}"
export BENCH_OUT_ROOT="${BENCH_OUT_ROOT:-$OUTPUTS_ROOT/closed_loop/v64_3_56_converged_fixed_budget_test}"
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

# Preserve the exact converged V56 science source before benchmark-only changes.
sha256sum -c V64_3_56_SCIENCE_MANIFEST.sha256 >/dev/null

mkdir -p "$BENCH_OUT_ROOT/provenance"
if [[ -z "${EAF_CKPT:-}" ]]; then
  python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
    --train-log "$EAF_TRAIN_LOG" --variant CONVERGED_EXTERNAL_BENCHMARK \
    --output "$BENCH_OUT_ROOT/provenance/v64_3_13_reaudit_for_converged_benchmark.json" \
    > "$BENCH_OUT_ROOT/provenance/v64_3_13_reaudit_for_converged_benchmark.out"
  read -r SELECTED_EPOCH < <(python - "$BENCH_OUT_ROOT/provenance/v64_3_13_reaudit_for_converged_benchmark.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False):
    raise SystemExit('STOP benchmark: V13 frozen checkpoint prerequisites changed')
print(int(r['selected_epoch']))
PY
  )
  export EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"
fi
[[ -s "$EAF_CKPT" ]] || { echo "missing frozen EAF checkpoint: $EAF_CKPT" >&2; exit 2; }

# External trainable adapters MUST have a budget-specific checkpoint for every B.
missing=0
for B in $BUDGETS; do
  for name in gameformer dtpp plantf pluto; do
    p="$EXTERNAL_OUT_ROOT/B${B}/${name}_budgeted.best.pt"
    [[ -s "$p" ]] || { echo "MISSING external checkpoint: $p" >&2; missing=1; }
  done
done
if (( missing )); then
  echo "Run first: GPUS=$GPUS bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh" >&2
  exit 3
fi

cat > "$BENCH_OUT_ROOT/provenance/benchmark_semantics.txt" <<EOF
Internal search status: converged by falsification at V64.3.56.
Primary own method: frozen full-set V47 RSMR config + exact V13 EAF checkpoint; no V54/V55/V56 post-intervention oracle state is used.
Primary matched-interface comparison: B=16.
B=8 and B=24 for BDSE are frozen-policy cross-budget robustness ablations because the promoted/frozen own fit was developed at B=16.
External GameFormer/DTPP/PlanTF/PLUTO adapters are trained separately for each B by RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh.
PDM-Closed-style is a repository style baseline, not the official PDM-Closed implementation.
All systems use the exact same ordered public_set_test tokens and metric-safe serialized nuPlan metric callbacks; planner/simulation workers remain parallel.
EOF

python -m bdse.tools.run_fixed_budget_closed_loop_suite \
  --own-config "$OWN_CONFIG" --own-checkpoint "$EAF_CKPT" \
  --own-label "BDSE-EAF-RSMR (frozen t0-deployable backbone)" \
  --external-checkpoint-root "$EXTERNAL_OUT_ROOT" \
  --split-cache "$BDSE_TEST_CACHE" --token-split public_set_test --limit "$CL_LIMIT" \
  --nuplan-root "$NUPLAN_ROOT" --nuplan-map-root "$NUPLAN_MAP_ROOT" --nuplan-exp-root "$NUPLAN_EXP_ROOT" --nuplan-db-root "$NUPLAN_TEST_DB_ROOT" \
  --budgets $BUDGETS --proposal-top-m "$PROPOSAL_TOP_M" \
  --challenge "$CL_CHALLENGE" --output-root "$BENCH_OUT_ROOT/results" \
  --gpus "$GPUS" --workers-per-job "$CL_WORKERS_PER_JOB" --schedule-mode model_pairs \
  --token-scan-workers "$CL_TOKEN_SCAN_WORKERS" --token-progress-seconds "$CL_TOKEN_PROGRESS_SECONDS" \
  --heartbeat-seconds "$CL_HEARTBEAT_SECONDS" \
  --systems bdse gameformer dtpp plantf pluto pdm_closed_style --resume

python -m bdse.tools.summarize_v64_3_56_converged_benchmark \
  --input-json "$BENCH_OUT_ROOT/results/closed_loop_fixed_budget_all_metrics.json" \
  --output-root "$BENCH_OUT_ROOT/summary"

echo "DONE: primary B16 + B8/B16/B24 robustness sweep under $BENCH_OUT_ROOT"
