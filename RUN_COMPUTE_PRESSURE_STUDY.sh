#!/usr/bin/env bash
# Controlled cached-core scaling study. Each arm is B:M:K:pool-K:response-modes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${EAF_CKPT:?Set EAF_CKPT to the frozen checkpoint}"
CONFIG="${OWN_CONFIG:-bdse/configs/deployment.yaml}"
CACHE_ROOT="${CACHE_ROOT:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
OUT="${PROFILE_ROOT:-outputs/compute_pressure_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$OUT" ]] || { echo "PROFILE_ROOT must be new: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="${PROFILE_GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -m bdse.tools.profile_planning_budget --cache-root "$CACHE_ROOT" --config "$CONFIG" \
  --output "$OUT/workload" --limit "${LIMIT:-1000}" --seed "${SEED:-20261011}" --list-only
# Keep arms below the observed ~30 active-atom ceiling; M=48/128 are saturated.
ARMS="${ARMS:-8:8:32:96:logged+cv 16:16:32:96:logged+cv 16:24:16:48:logged+cv 16:24:16:96:logged+cv 16:24:32:96:logged+cv 16:24:32:96:logged+cv+ca+brake+yield+nonyield 24:30:32:96:logged+cv+ca+brake+yield+nonyield}"
for block in $(seq 1 "${BLOCKS:-1}"); do
  mapfile -t ordered_arms < <(python - "$ARMS" "${SEED:-20261011}" "$block" <<'PY'
import random,sys
arms=sys.argv[1].split(); random.Random(int(sys.argv[2])+int(sys.argv[3])).shuffle(arms)
print("\n".join(arms))
PY
  )
  for arm in "${ordered_arms[@]}"; do
    IFS=: read -r B M K POOL_K MODES <<< "$arm"
    MODES_CSV="${MODES//+/,}"
    label="block${block}_B${B}_M${M}_K${K}_P${POOL_K}_R${MODES//+/-}"
    python -m bdse.tools.profile_planning_budget --cache-root "$CACHE_ROOT" \
      --manifest "$OUT/workload/workload_manifest.json" --config "$CONFIG" \
      --checkpoint "$EAF_CKPT" --device cuda:0 --budget "$B" --proposal-top-m "$M" \
      --candidate-k "$K" --candidate-pool-k "$POOL_K" --response-modes "$MODES_CSV" --sweep-semantics strict \
      --mode "${MODE:-native}" --warmup "${WARMUP:-20}" --repeats "${REPEATS:-1}" \
      --seed "${SEED:-20261011}" --output "$OUT/$label"
  done
done
python -m bdse.tools.summarize_planning_budget --traces "$OUT" \
  --deadlines-ms ${DEADLINES_MS:-50 100 200 500 1000} --periods-ms ${PERIODS_MS:-100 200 500} \
  --output "$OUT/combined_summary.json"
echo "DONE $OUT"
