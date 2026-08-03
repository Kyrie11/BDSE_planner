#!/usr/bin/env bash
set -euo pipefail

# Paired strict-budget comparison.  Every system receives the same ordered
# val_tune samples, fallback is disabled, and the four trainable adapters must
# first pass the matched-data checkpoint validation.

: "${BDSE_SPLIT_CACHE:=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v53_split}"
: "${V60_CKPT:?Set V60_CKPT to the frozen V60 best checkpoint}"
: "${EXTERNAL_CKPT_ROOT:?Set EXTERNAL_CKPT_ROOT to outputs/external}"
: "${SWEEP_OUT:=${BUDGET_SWEEP_OUT:-${OUT_ROOT:-outputs_v60_budget_baseline_sweep}}}"
: "${GPUS:=0,1}"
: "${BUDGETS:=8 16 24 32}"
: "${SWEEP_WORKERS_PER_GPU:=1}"
: "${SWEEP_SPLIT:=val_tune}"
: "${SWEEP_SCENARIOS:=1000}"

mkdir -p "$SWEEP_OUT"
python -m bdse.tools.validate_external_checkpoint_suite \
  --checkpoint-root "$EXTERNAL_CKPT_ROOT" \
  --output "$SWEEP_OUT/external_checkpoint_suite_validation.json"

python -m bdse.tools.run_budget_baseline_sweep \
  --bdse-config bdse/configs/v60_dwapc_bfar_dbap_local_control_cl.yaml \
  --bdse-checkpoint "$V60_CKPT" \
  --external-checkpoint-root "$EXTERNAL_CKPT_ROOT" \
  --preprocessed-dir "$BDSE_SPLIT_CACHE" \
  --split "$SWEEP_SPLIT" --max-scenarios "$SWEEP_SCENARIOS" \
  --budgets $BUDGETS \
  --include-pdm-closed \
  --output-root "$SWEEP_OUT" \
  --gpus "$GPUS" --workers-per-gpu "$SWEEP_WORKERS_PER_GPU" \
  --device cuda --resume

echo "CSV:      $SWEEP_OUT/budget_sweep.csv"
echo "Markdown: $SWEEP_OUT/budget_sweep.md"
echo "JSON:     $SWEEP_OUT/budget_sweep.json"
