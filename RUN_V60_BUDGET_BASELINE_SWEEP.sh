#!/usr/bin/env bash
set -euo pipefail

# Fast advisor-facing strict-budget comparison.  This evaluates the same first
# 1000 val_tune samples for every system and disables fallback, so B is the
# actual evidence-query budget rather than only the first stage of a cascade.
# BDSE uses selected-local/no-residual here because a residual conformal
# certificate must be recalibrated separately at each budget for paper claims.

: "${BDSE_SPLIT_CACHE:=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v53_split}"
: "${V60_CKPT:?Set V60_CKPT to the frozen V60 best checkpoint}"
: "${EXTERNAL_CKPT_ROOT:?Set EXTERNAL_CKPT_ROOT to the folder containing gameformer_budgeted.best.pt, dtpp_budgeted.best.pt, plantf_budgeted.best.pt and pluto_budgeted.best.pt}"
: "${OUT_ROOT:=outputs_v60_budget_baseline_sweep}"
: "${GPUS:=0,1}"
: "${BUDGETS:=8 16 24 32}"
: "${SWEEP_WORKERS_PER_GPU:=1}"

python -m bdse.tools.run_budget_baseline_sweep \
  --bdse-config bdse/configs/v60_dwapc_bfar_dbap_local_control_cl.yaml \
  --bdse-checkpoint "$V60_CKPT" \
  --external-checkpoint-root "$EXTERNAL_CKPT_ROOT" \
  --preprocessed-dir "$BDSE_SPLIT_CACHE" \
  --split val_tune --max-scenarios 1000 \
  --budgets $BUDGETS \
  --include-pdm-closed \
  --output-root "$OUT_ROOT" \
  --gpus "$GPUS" --workers-per-gpu "$SWEEP_WORKERS_PER_GPU" \
  --device cuda --resume

echo "CSV:      $OUT_ROOT/budget_sweep.csv"
echo "Markdown: $OUT_ROOT/budget_sweep.md"
echo "JSON:     $OUT_ROOT/budget_sweep.json"
