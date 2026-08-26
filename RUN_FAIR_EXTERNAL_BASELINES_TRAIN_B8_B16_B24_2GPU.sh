#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

# Two-A30 optimized training: GPU0/GPU1 each own one model for all budgets,
# then the next model pair starts.  This avoids model/GPU migration and keeps
# disk/cache access predictable.
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EXTERNAL_OUT_ROOT="${EXTERNAL_OUT_ROOT:-outputs/external_fixed_budget}"
export GPUS="${GPUS:-0,1}"
export BUDGETS="${BUDGETS:-8 16 24}"
export PROPOSAL_TOP_M="${PROPOSAL_TOP_M:-24}"
export NUM_WORKERS_PER_JOB="${NUM_WORKERS_PER_JOB:-6}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export TORCH_COMPILE="${TORCH_COMPILE:-1}"
export TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-reduce-overhead}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-100}"
# Keep paper-grade full validation by default. Set VAL_MAX_SCENARIOS to a
# positive number only for exploratory fast runs.
export VAL_MAX_SCENARIOS="${VAL_MAX_SCENARIOS:-0}"
export VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-1}"

# Two concurrent DataLoaders should not each spawn BLAS thread pools.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -d "$BDSE_TRAIN_CACHE" ]] || { echo "missing train cache: $BDSE_TRAIN_CACHE" >&2; exit 2; }
[[ -d "$BDSE_VAL_CACHE" ]] || { echo "missing val cache: $BDSE_VAL_CACHE" >&2; exit 2; }
mkdir -p "$EXTERNAL_OUT_ROOT/configs"

python -m bdse.tools.prepare_external_fixed_budget_configs \
  --output-root "$EXTERNAL_OUT_ROOT/configs" \
  --budgets $BUDGETS \
  --proposal-top-m "$PROPOSAL_TOP_M"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
[[ ${#GPU_ARR[@]} -ge 2 ]] || { echo "This script expects two GPUs, e.g. GPUS=0,1" >&2; exit 2; }
GPU0="${GPU_ARR[0]}"; GPU1="${GPU_ARR[1]}"

run_one_budget() {
  local gpu="$1" B="$2" name="$3"
  local cfg="$EXTERNAL_OUT_ROOT/configs/B${B}/external_${name}_budgeted.yaml"
  local outdir="$EXTERNAL_OUT_ROOT/B${B}"
  mkdir -p "$outdir"

  local batch=32 accum=1
  # Preserve the published/equivalent effective batch 128 for PlanTF/PLUTO.
  # These ~5M-parameter adapters are small enough that an A30 should normally
  # fit one 128-sample fp16 micro-batch after compact tensorization, eliminating
  # the old 32x4 accumulation overhead. Override to 64x2 or 32x4 if needed.
  if [[ "$name" == "plantf" || "$name" == "pluto" ]]; then
    batch="${PLAN_BATCH_SIZE:-128}"
    accum="${PLAN_GRAD_ACCUM:-1}"
  fi

  local compile_args=()
  if [[ "$TORCH_COMPILE" == "1" || "$TORCH_COMPILE" == "true" ]]; then
    compile_args+=(--compile --compile-mode "$TORCH_COMPILE_MODE" --compile-fallback)
  fi

  echo "[train] gpu=$gpu system=$name B=$B batch=$batch accum=$accum workers=$NUM_WORKERS_PER_JOB compile=$TORCH_COMPILE"
  CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.external_baselines.train \
    --config "$cfg" \
    --split train_boston train_pittsburgh train_singapore train_vegas_2 \
    --preprocessed-dir "$BDSE_TRAIN_CACHE" \
    --batch-size "$batch" \
    --grad-accum-steps "$accum" \
    --num-workers "$NUM_WORKERS_PER_JOB" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --device cuda --amp \
    --val-preprocessed-dir "$BDSE_VAL_CACHE" \
    --val-split val \
    --val-max-scenarios "$VAL_MAX_SCENARIOS" \
    --val-mode loss \
    --val-every-n-epochs "$VAL_EVERY_N_EPOCHS" \
    --warmup-epochs 3 \
    --scheduler cosine \
    --selection-metric val_action_ce \
    --early-stop-patience 0 \
    --log-every-n-steps "$LOG_EVERY_N_STEPS" \
    "${compile_args[@]}" \
    --log-file "$outdir/${name}.train_log.jsonl" \
    --output "$outdir/${name}_budgeted.pt" \
    > "$outdir/${name}.train.out" 2>&1
}

run_model_all_budgets() {
  local gpu="$1" name="$2"
  for B in $BUDGETS; do
    run_one_budget "$gpu" "$B" "$name"
  done
}

run_pair() {
  local left="$1" right="$2"
  echo "=== pair: GPU$GPU0->$left | GPU$GPU1->$right ==="
  run_model_all_budgets "$GPU0" "$left" & local p0=$!
  run_model_all_budgets "$GPU1" "$right" & local p1=$!
  local failed=0
  wait "$p0" || { echo "FAILED: $left" >&2; failed=1; }
  wait "$p1" || { echo "FAILED: $right" >&2; failed=1; }
  (( failed == 0 )) || exit 2
}

run_pair gameformer dtpp
run_pair plantf pluto

echo "DONE: budget-specific external checkpoints are under $EXTERNAL_OUT_ROOT/B{8,16,24}/"
