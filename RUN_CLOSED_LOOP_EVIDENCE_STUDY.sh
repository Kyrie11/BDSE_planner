#!/usr/bin/env bash
# Paired reactive and latency-aware nuPlan runs on the identical token manifest.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
EAF_CKPT="${EAF_CKPT:-}"
SCENARIO_TOKENS_FILE="${SCENARIO_TOKENS_FILE:-}"
missing=()
[[ -n "$EAF_CKPT" ]] || missing+=("EAF_CKPT (frozen checkpoint)")
[[ -n "$SCENARIO_TOKENS_FILE" ]] || missing+=("SCENARIO_TOKENS_FILE (ordered scenario-token manifest)")
if ((${#missing[@]})); then
  printf 'missing required environment inputs:\n' >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 2
fi
OWN_CONFIG="$(realpath -m -- "${OWN_CONFIG:-bdse/configs/deployment.yaml}")"
EAF_CKPT="$(realpath -m -- "$EAF_CKPT")"
RAW_ROOT="${NUPLAN_DB_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits/val}"
DATA_ROOT="${NUPLAN_DATA_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan}"
MAP_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
RAW_ROOT="$(realpath -m -- "$RAW_ROOT")"
SCENARIO_TOKENS_FILE="$(realpath -m -- "$SCENARIO_TOKENS_FILE")"
OUT="${CLOSED_LOOP_OUT:-outputs/evidence_closed_loop_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
python -m bdse.tools.validate_runtime_inputs --repo-root "$ROOT" \
  --config "$OWN_CONFIG" --checkpoint "$EAF_CKPT" --raw-db-root "$RAW_ROOT" \
  --token-manifest "$SCENARIO_TOKENS_FILE" --json-output "$OUT/input_validation.json"
common=(--config "${OWN_CONFIG:-bdse/configs/deployment.yaml}" --checkpoint "$EAF_CKPT" --device cuda:0
  --challenge closed_loop_reactive_agents --nuplan-module bdse.tools.nuplan_metric_safe_run_simulation
  --scenario-builder nuplan --worker single_machine_thread_pool --hydra-full-error
  --nuplan-data-root "$DATA_ROOT" --nuplan-map-root "$MAP_ROOT" --nuplan-exp-root "$OUT/exp"
  --nuplan-db-root "$RAW_ROOT" --scenario-tokens-file "$SCENARIO_TOKENS_FILE")
base_overrides=(scenario_filter.limit_total_scenarios="${CL_LIMIT:-100}" scenario_filter.shuffle=false
  scenario_filter.log_names=null worker.max_workers=1 run_metric=true '~callback.simulation_log_callback')
python -m bdse.experiments.evaluate_closed_loop "${common[@]}" \
  --output-dir "$OUT/reactive" --experiment-uid reactive -- \
  planner.bdse_planner._target_=bdse.planner.nuplan_planner.BDSEnuPlanPlanner "${base_overrides[@]}"
export BDSE_CLOSED_LOOP_DIAG="$OUT/latency_aware_events.jsonl"
export BDSE_CLOSED_LOOP_DIAG_MODE=full
export BDSE_SIMULATED_DECISION_LATENCY_MS="${SIMULATED_LATENCY_MS:-}"
export BDSE_MAX_PLAN_AGE_MS="${MAX_PLAN_AGE_MS:-}"
python -m bdse.experiments.evaluate_closed_loop "${common[@]}" \
  --output-dir "$OUT/latency_aware" --experiment-uid latency_aware -- \
  planner.bdse_planner._target_=bdse.planner.latency_aware_nuplan_planner.LatencyAwareBDSEnuPlanPlanner "${base_overrides[@]}"
echo "DONE $OUT"
