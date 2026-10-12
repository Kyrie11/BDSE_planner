#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${EAF_CKPT:?Set EAF_CKPT to the frozen checkpoint}"
CONFIG="${OWN_CONFIG:-bdse/configs/deployment.yaml}"
CACHE_ROOT="${CACHE_ROOT:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
OUT="${EVIDENCE_OUT:-outputs/evidence_study_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
python -m bdse.experiments.evidence_protocol \
  --config "$CONFIG" --checkpoint "$EAF_CKPT" \
  --preprocessed-dir "$CACHE_ROOT" --split "${CACHE_SPLIT:-val}" \
  --max-scenarios "${EVIDENCE_SCENARIOS:-1000}" --device "${DEVICE:-cuda:0}" \
  --budgets ${EVIDENCE_BUDGETS:-2 4 8 12 16 24} \
  --budget-unit "${BUDGET_UNIT:-atoms}" \
  --stop-z ${STOP_Z:-0 1 1.645 1.96} \
  --stop-unqueried-radius "${STOP_UNQUERIED_RADIUS:-0.05}" \
  --physical-delta "${PHYSICAL_DELTA:-0.05}" \
  --intervention-delta "${INTERVENTION_DELTA:-0.05}" \
  --output "$OUT/summary.json" --per-scene-output "$OUT/per_scene.jsonl"
echo "DONE $OUT"
