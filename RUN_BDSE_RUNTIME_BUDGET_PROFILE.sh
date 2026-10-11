#!/usr/bin/env bash
# Single-worker actual nuPlan calls; no artificial delay or deadline rejection.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
: "${OWN_CONFIG:?Set frozen deployment config}"
: "${EAF_CKPT:?Set frozen EAF checkpoint}"
: "${NUPLAN_DB_ROOT:?Set actual raw nuPlan DB directory; NPZ cache is insufficient}"
: "${SCENARIO_TOKENS_FILE:?Set an audited compatible raw-DB scenario token manifest}"
PROFILE_ROOT="${PROFILE_ROOT:-$SCRIPT_DIR/outputs/runtime_budget_profile_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$PROFILE_ROOT" ]] || { echo "PROFILE_ROOT must be new" >&2; exit 2; }
[[ -s "$OWN_CONFIG" && -s "$EAF_CKPT" && -s "$SCENARIO_TOKENS_FILE" ]] || { echo "missing config/checkpoint/token manifest" >&2; exit 2; }
mkdir -p "$PROFILE_ROOT"
export CUDA_VISIBLE_DEVICES="${PROFILE_GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export BDSE_SHARE_MODEL_PER_PROCESS=1
export BDSE_SERIALIZE_GPU_INFERENCE=1
export BDSE_SHARD_PLANNERS_ACROSS_GPUS=0
export BDSE_PROFILE_CLOSED_LOOP=1
export BDSE_PIOR_METRIC_ENGINE_SERIALIZATION=1
export BDSE_BUDGET_TRACE_PREFIX="$PROFILE_ROOT/trace"
export BDSE_BUDGET_PROFILE_MODE="${MODE:-native}"
export BDSE_BUDGET_WARMUP_CALLS="${WARMUP:-5}"
# Optional descriptive target, never enforced by this measurement code.
export BDSE_BUDGET_DEADLINE_MS="${DEADLINE_MS:-}"
python - "$OWN_CONFIG" "$PROFILE_ROOT/runtime_config.yaml" "${BUDGET:-16}" "${PROPOSAL_TOP_M:-24}" "${REPLAN_INTERVAL_TICKS:-}" <<'PY'
import sys,yaml
from bdse.config import load_config
cfg=load_config(sys.argv[1], {'evidence':{'budget':int(sys.argv[3])},'selector':{'proposal_top_m':int(sys.argv[4])}})
if cfg.get('selected_outcome_probe',{}).get('enabled',False):
    raise SystemExit('refuse TRAIN-only selected_outcome_probe config')
if int(sys.argv[4])<int(sys.argv[3]):raise SystemExit('proposal-top-m must be >= budget')
if sys.argv[5]:
    interval=int(sys.argv[5])
    if interval<1:raise SystemExit('replan interval must be positive')
    cfg.setdefault('planner',{})['replan_interval_ticks']=interval
with open(sys.argv[2],'w') as f:yaml.safe_dump(cfg,f,sort_keys=False)
PY
sha256sum "$OWN_CONFIG" "$EAF_CKPT" "$SCENARIO_TOKENS_FILE" "$PROFILE_ROOT/runtime_config.yaml" > "$PROFILE_ROOT/input_sha256.txt"
args=(--config "$PROFILE_ROOT/runtime_config.yaml" --checkpoint "$EAF_CKPT" --device cuda:0
  --challenge "${CHALLENGE:-closed_loop_reactive_agents}" --output-dir "$PROFILE_ROOT/closed_loop"
  --experiment-uid "budget_profile" --nuplan-module bdse.tools.nuplan_metric_safe_run_simulation
  --scenario-builder nuplan --worker single_machine_thread_pool --hydra-full-error
  --nuplan-data-root "${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}"
  --nuplan-map-root "${NUPLAN_MAP_ROOT:-/data0/senzeyu2/dataset/nuplan/maps}"
  --nuplan-exp-root "${NUPLAN_EXP_ROOT:-$PROFILE_ROOT/exp}"
  --nuplan-db-root "$NUPLAN_DB_ROOT" --scenario-tokens-file "$SCENARIO_TOKENS_FILE")
[[ "${DRY_RUN:-0}" == "1" ]] && args+=(--dry-run)
python -m bdse.experiments.evaluate_closed_loop "${args[@]}" -- \
  planner.bdse_planner._target_=bdse.planner.profiled_nuplan_planner.ProfiledBDSEnuPlanPlanner \
  scenario_filter.limit_total_scenarios="${CL_LIMIT:-100}" scenario_filter.shuffle=false \
  scenario_filter.log_names=null worker.max_workers=1 run_metric=true '~callback.simulation_log_callback'
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python -m bdse.tools.summarize_planning_budget --traces "$PROFILE_ROOT" \
    --deadlines-ms ${DEADLINES_MS:-50 100 200} --output "$PROFILE_ROOT/summary.json"
fi
