#!/usr/bin/env bash
set -euo pipefail
: "${V60_CKPT:?Set V60_CKPT to the frozen V60 best.pt}"
: "${V60_EVAL_CONFIG:=bdse/configs/v60_dwapc_bfar_dbap_cl.yaml}"
: "${EXTERNAL_CKPT_ROOT:=outputs/external}"
: "${BDSE_SPLIT_CACHE:?Set BDSE_SPLIT_CACHE}"
: "${OPEN_LOOP_OUT:=outputs/v60_external_compare/open_loop_val_tune_1000}"
: "${OPEN_LOOP_SPLIT:=val_tune}"
: "${OPEN_LOOP_SCENARIOS:=1000}"
: "${GPUS:=0,1}"
: "${OPEN_LOOP_WORKERS_PER_GPU:=1}"

python -m bdse.tools.validate_external_checkpoint_suite \
  --checkpoint-root "$EXTERNAL_CKPT_ROOT" \
  --output "$OPEN_LOOP_OUT/external_checkpoint_suite_validation.json"

python -m bdse.tools.run_external_open_loop_comparison \
  --v60-config "$V60_EVAL_CONFIG" --v60-checkpoint "$V60_CKPT" \
  --external-checkpoint-root "$EXTERNAL_CKPT_ROOT" \
  --preprocessed-dir "$BDSE_SPLIT_CACHE" --split "$OPEN_LOOP_SPLIT" \
  --max-scenarios "$OPEN_LOOP_SCENARIOS" \
  --output-root "$OPEN_LOOP_OUT" --gpus "$GPUS" \
  --workers-per-gpu "$OPEN_LOOP_WORKERS_PER_GPU" --device cuda --resume

cat "$OPEN_LOOP_OUT/open_loop_summary.md"
