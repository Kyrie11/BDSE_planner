#!/usr/bin/env bash
set -euo pipefail

# Small CL4 benchmark for choosing 1 vs 2 process copies per model/GPU.
# Uses only V60 and GameFormer so both GPUs stay occupied.  Choose the setting
# with lower total wall time and zero failures before launching CL20/CL50.

: "${NUPLAN_ROOT:?Set NUPLAN_ROOT}"
: "${BDSE_SPLIT_CACHE:?Set BDSE_SPLIT_CACHE}"
: "${V60_CKPT:?Set V60_CKPT}"
: "${EXTERNAL_CKPT_ROOT:=outputs/external}"
: "${V60_EVAL_CONFIG:=bdse/configs/v60_dwapc_bfar_dbap_cl.yaml}"
: "${GPUS:=0,1}"
: "${BENCH_ROOT:=outputs/v60_external_compare/cl_concurrency_benchmark}"

for p in 1 2; do
  out="$BENCH_ROOT/processes_per_model_${p}"
  python -m bdse.tools.run_external_closed_loop_comparison \
    --v60-config "$V60_EVAL_CONFIG" --v60-checkpoint "$V60_CKPT" \
    --external-checkpoint-root "$EXTERNAL_CKPT_ROOT" \
    --split-cache "$BDSE_SPLIT_CACHE" --token-split val_tune \
    --limit 4 --token-scan-max 200 \
    --nuplan-root "$NUPLAN_ROOT" \
    --nuplan-db-root "${NUPLAN_DB_ROOT:-$NUPLAN_ROOT/data/cache/val/}" \
    --challenge closed_loop_nonreactive_agents \
    --output-root "$out" --gpus "$GPUS" \
    --processes-per-model "$p" \
    --systems v60 gameformer --device cuda
  cat "$out/closed_loop_summary.md"
done
