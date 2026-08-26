#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2}"
export EXTERNAL_OUT_ROOT="${EXTERNAL_OUT_ROOT:-outputs/external_fixed_budget}"
export OPEN_LOOP_OUT_ROOT="${OPEN_LOOP_OUT_ROOT:-outputs/open_loop_test_fixed_budget}"
export GPUS="${GPUS:-0,1}"
export BUDGETS="${BUDGETS:-8 16 24}"
export PROPOSAL_TOP_M="${PROPOSAL_TOP_M:-24}"

[[ -d "$BDSE_TEST_CACHE" ]] || { echo "missing test cache: $BDSE_TEST_CACHE" >&2; exit 2; }
python -m bdse.tools.prepare_external_fixed_budget_configs \
  --output-root "$EXTERNAL_OUT_ROOT/configs" --budgets $BUDGETS --proposal-top-m "$PROPOSAL_TOP_M"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
[[ ${#GPU_ARR[@]} -ge 2 ]] || { echo "This script expects two GPUs, e.g. GPUS=0,1" >&2; exit 2; }
GPU0="${GPU_ARR[0]}"; GPU1="${GPU_ARR[1]}"

run_model_all_budgets(){
  local gpu="$1" name="$2"
  for B in $BUDGETS; do
    local cfg ckpt outdir
    outdir="$OPEN_LOOP_OUT_ROOT/B${B}"; mkdir -p "$outdir"
    cfg="$EXTERNAL_OUT_ROOT/configs/B${B}/external_${name}_budgeted_fast_cl.yaml"
    ckpt="$EXTERNAL_OUT_ROOT/B${B}/${name}_budgeted.best.pt"
    [[ -s "$ckpt" ]] || { echo "missing checkpoint: $ckpt" >&2; return 2; }
    echo "[open-loop] gpu=$gpu system=$name B=$B"
    CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop \
      --config "$cfg" --checkpoint "$ckpt" \
      --split public_set_test --preprocessed-dir "$BDSE_TEST_CACHE" --device cuda \
      --disable-dense-diagnostic \
      --output "$outdir/${name}.json" --per-sample-output "$outdir/${name}.jsonl" \
      > "$outdir/${name}.out" 2>&1
  done
}

run_pair(){
  local left="$1" right="$2"
  echo "=== open-loop pair: GPU$GPU0->$left | GPU$GPU1->$right ==="
  run_model_all_budgets "$GPU0" "$left" & local p0=$!
  run_model_all_budgets "$GPU1" "$right" & local p1=$!
  local failed=0
  wait "$p0" || { echo "FAILED: $left" >&2; failed=1; }
  wait "$p1" || { echo "FAILED: $right" >&2; failed=1; }
  (( failed == 0 )) || exit 2
}

run_pair gameformer dtpp
run_pair plantf pluto

# PDM-style is deterministic and now allocates no Transformer; run it on CPU.
for B in $BUDGETS; do
  outdir="$OPEN_LOOP_OUT_ROOT/B${B}"; mkdir -p "$outdir"
  cfg="$EXTERNAL_OUT_ROOT/configs/B${B}/external_pdm_closed_budgeted_fast_cl.yaml"
  echo "[open-loop] cpu system=pdm_closed B=$B"
  python -m bdse.experiments.evaluate_open_loop \
    --config "$cfg" --split public_set_test --preprocessed-dir "$BDSE_TEST_CACHE" \
    --device cpu --disable-dense-diagnostic \
    --output "$outdir/pdm_closed.json" --per-sample-output "$outdir/pdm_closed.jsonl" \
    > "$outdir/pdm_closed.out" 2>&1
done

echo "DONE: $OPEN_LOOP_OUT_ROOT"
