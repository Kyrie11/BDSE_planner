#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
OWN_CONFIG="${OWN_CONFIG:-bdse/configs/deployment.yaml}"
: "${EAF_CKPT:?Set EAF_CKPT to the frozen EAF checkpoint}"
CACHE_ROOT="${CACHE_ROOT:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
PROFILE_ROOT="${PROFILE_ROOT:-$SCRIPT_DIR/outputs/budget_profile_$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE_GPU="${PROFILE_GPU:-0}"
MODE="${MODE:-native}"
SWEEP_SEMANTICS="${SWEEP_SEMANTICS:-strict}"
LIMIT="${LIMIT:-1000}"
WARMUP="${WARMUP:-20}"
REPEATS="${REPEATS:-1}"
SEED="${SEED:-20261011}"
export CUDA_VISIBLE_DEVICES="$PROFILE_GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
[[ ! -e "$PROFILE_ROOT" ]] || { echo "PROFILE_ROOT must be new" >&2; exit 2; }
[[ -s "$OWN_CONFIG" ]] || { echo "missing deployment config: $OWN_CONFIG" >&2; exit 2; }
[[ -s "$EAF_CKPT" ]] || { echo "missing checkpoint: $EAF_CKPT" >&2; exit 2; }
read -r -a budgets <<< "${BUDGETS:-8 16 24}"
read -r -a proposals <<< "${PROPOSALS:-24 24 24}"
deadline_args=()
[[ -z "${DEADLINE_MS:-}" ]] || deadline_args=(--deadline-ms "$DEADLINE_MS")
[[ ${#budgets[@]} -eq ${#proposals[@]} ]] || { echo "BUDGETS/PROPOSALS length mismatch" >&2; exit 2; }
python -m bdse.tools.profile_planning_budget --cache-root "$CACHE_ROOT" --config "$OWN_CONFIG" \
  --output "$PROFILE_ROOT/workload" --limit "$LIMIT" --seed "$SEED" --list-only
for i in "${!budgets[@]}"; do
  budget="${budgets[$i]}"; proposal="${proposals[$i]}"
  python -m bdse.tools.profile_planning_budget --cache-root "$CACHE_ROOT" \
    --manifest "$PROFILE_ROOT/workload/workload_manifest.json" --config "$OWN_CONFIG" \
    --checkpoint "$EAF_CKPT" --device cuda:0 --budget "$budget" --proposal-top-m "$proposal" \
    --sweep-semantics "$SWEEP_SEMANTICS" \
    --mode "$MODE" --warmup "$WARMUP" --repeats "$REPEATS" --seed "$SEED" \
    "${deadline_args[@]}" \
    --output "$PROFILE_ROOT/B${budget}_M${proposal}_${MODE}"
done
python -m bdse.tools.summarize_planning_budget --traces "$PROFILE_ROOT" \
  --deadlines-ms ${DEADLINES_MS:-50 100 200} --periods-ms ${PERIODS_MS:-100 200} \
  --output "$PROFILE_ROOT/combined_summary.json"
echo "DONE $PROFILE_ROOT"
