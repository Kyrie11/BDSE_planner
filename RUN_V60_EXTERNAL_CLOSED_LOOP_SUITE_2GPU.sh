#!/usr/bin/env bash
set -euo pipefail
: "${NUPLAN_ROOT:?Set NUPLAN_ROOT}"
: "${BDSE_SPLIT_CACHE:?Set BDSE_SPLIT_CACHE}"
: "${V60_CKPT:?Set V60_CKPT to the frozen V60 best.pt}"
: "${V60_EVAL_CONFIG:=bdse/configs/v60_dwapc_bfar_dbap_cl.yaml}"
: "${EXTERNAL_CKPT_ROOT:=outputs/external}"
: "${CL_LIMIT:=20}"
: "${CL_CHALLENGE:=closed_loop_nonreactive_agents}"
: "${CL_TOKEN_SPLIT:=val_tune}"
: "${CL_TOKEN_SCAN_MAX:=2000}"
: "${CL_PROCESSES_PER_MODEL:=1}"
: "${GPUS:=0,1}"
: "${CL_SYSTEMS:=v60 gameformer dtpp plantf pluto pdm_closed_style}"
: "${CL_OUT:=outputs/v60_external_compare/${CL_CHALLENGE}_cl${CL_LIMIT}}"

python -m bdse.tools.validate_external_checkpoint_suite \
  --checkpoint-root "$EXTERNAL_CKPT_ROOT" \
  --output "$CL_OUT/external_checkpoint_suite_validation.json"

python -m bdse.tools.run_external_closed_loop_comparison \
  --v60-config "$V60_EVAL_CONFIG" --v60-checkpoint "$V60_CKPT" \
  --external-checkpoint-root "$EXTERNAL_CKPT_ROOT" \
  --split-cache "$BDSE_SPLIT_CACHE" --token-split "$CL_TOKEN_SPLIT" \
  --limit "$CL_LIMIT" --token-scan-max "$CL_TOKEN_SCAN_MAX" \
  --nuplan-root "$NUPLAN_ROOT" \
  --nuplan-db-root "${NUPLAN_DB_ROOT:-$NUPLAN_ROOT/data/cache/val/}" \
  --challenge "$CL_CHALLENGE" \
  --output-root "$CL_OUT" --gpus "$GPUS" \
  --processes-per-model "$CL_PROCESSES_PER_MODEL" \
  --systems $CL_SYSTEMS --device cuda --resume

cat "$CL_OUT/closed_loop_summary.md"
