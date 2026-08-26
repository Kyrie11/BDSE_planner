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
export TORCH_COMPILE="${TORCH_COMPILE:-0}"
export TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-reduce-overhead}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-100}"
export TRAIN_PROGRESS_STYLE="${TRAIN_PROGRESS_STYLE:-lines}"
export STARTUP_PREFLIGHT_SAMPLES="${STARTUP_PREFLIGHT_SAMPLES:-2}"
export TRAIN_MAX_SCENARIOS="${TRAIN_MAX_SCENARIOS:-0}"
export TRAIN_MAX_SCENARIOS_PER_SPLIT="${TRAIN_MAX_SCENARIOS_PER_SPLIT:-0}"
export EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-0}"
export ONLY_FIRST_PAIR="${ONLY_FIRST_PAIR:-0}"
export PYTHONUNBUFFERED=1
# Keep paper-grade full validation by default. Set VAL_MAX_SCENARIOS to a
# positive number only for exploratory fast runs.
export VAL_MAX_SCENARIOS="${VAL_MAX_SCENARIOS:-0}"
export VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-1}"

# Two concurrent DataLoaders should not each spawn BLAS thread pools.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# Do not force PYTORCH_CUDA_ALLOC_CONF: expandable_segments is unsupported by some nuPlan-era PyTorch builds.
# If you have validated it on your environment, you may export it explicitly before launching this script.

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

echo "[train-launch] GPUs=$GPUS budgets=[$BUDGETS] compile=$TORCH_COMPILE workers_per_job=$NUM_WORKERS_PER_JOB"
CUDA_VISIBLE_DEVICES="$GPU0,$GPU1" python - <<'PY'
import os
import torch
print(f"[train-launch-env] torch={torch.__version__} torch_cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()} visible_count={torch.cuda.device_count()} alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the current Python/PyTorch environment")
if torch.cuda.device_count() < 2:
    raise SystemExit(f"Expected 2 visible CUDA devices, got {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"[train-launch-gpu] logical={i} name={p.name} capability={p.major}.{p.minor} mem_gib={p.total_memory/(1024**3):.1f}", flush=True)
PY

run_one_budget() {
  local gpu="$1" B="$2" name="$3"
  local cfg="$EXTERNAL_OUT_ROOT/configs/B${B}/external_${name}_budgeted.yaml"
  local outdir="$EXTERNAL_OUT_ROOT/B${B}"
  mkdir -p "$outdir"
  local marker="$outdir/.${name}_B${B}.run_started.$$"
  : > "$marker"

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
  local run_args=()
  if (( TRAIN_MAX_SCENARIOS > 0 )); then run_args+=(--max-scenarios "$TRAIN_MAX_SCENARIOS"); fi
  if (( TRAIN_MAX_SCENARIOS_PER_SPLIT > 0 )); then run_args+=(--max-scenarios-per-split "$TRAIN_MAX_SCENARIOS_PER_SPLIT"); fi
  if (( EPOCHS_OVERRIDE > 0 )); then run_args+=(--epochs "$EPOCHS_OVERRIDE"); fi

  echo "[train] START gpu=$gpu system=$name B=$B batch=$batch accum=$accum workers=$NUM_WORKERS_PER_JOB compile=$TORCH_COMPILE progress=$TRAIN_PROGRESS_STYLE"
  echo "[train] log=$outdir/${name}.train.out checkpoint=$outdir/${name}_budgeted.best.pt"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m bdse.external_baselines.train \
    --config "$cfg" \
    --split train_boston train_pittsburgh train_singapore train_vegas_2 \
    --preprocessed-dir "$BDSE_TRAIN_CACHE" \
    "${run_args[@]}" \
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
    --progress-style "$TRAIN_PROGRESS_STYLE" \
    --startup-preflight-samples "$STARTUP_PREFLIGHT_SAMPLES" \
    "${compile_args[@]}" \
    --log-file "$outdir/${name}.train_log.jsonl" \
    --output "$outdir/${name}_budgeted.pt" \
    2>&1 | tee "$outdir/${name}.train.out"
  local pipe_status=("${PIPESTATUS[@]}")
  local py_rc=${pipe_status[0]:-1}
  local tee_rc=${pipe_status[1]:-1}
  set -e
  if (( py_rc != 0 || tee_rc != 0 )); then
    echo "[train] FAILED gpu=$gpu system=$name B=$B python_exit=$py_rc tee_exit=$tee_rc" >&2
    echo "[train] --- tail of $outdir/${name}.train.out ---" >&2
    tail -n 80 "$outdir/${name}.train.out" >&2 || true
    echo "[train] --- end tail ---" >&2
    rm -f "$marker"
    if (( py_rc != 0 )); then return "$py_rc"; else return "$tee_rc"; fi
  fi
  [[ -s "$outdir/${name}_budgeted.best.pt" && "$outdir/${name}_budgeted.best.pt" -nt "$marker" ]] || {
    echo "[train] FAILED gpu=$gpu system=$name B=$B: training exited 0 but no fresh best checkpoint was produced" >&2
    rm -f "$marker"
    return 3
  }
  rm -f "$marker"
  echo "[train] DONE gpu=$gpu system=$name B=$B checkpoint=$outdir/${name}_budgeted.best.pt"
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
  wait "$p0" || { echo "[train-pair] FAILED model=$left (see the B-specific error above)" >&2; failed=1; }
  wait "$p1" || { echo "[train-pair] FAILED model=$right (see the B-specific error above)" >&2; failed=1; }
  (( failed == 0 )) || exit 2
}

run_pair gameformer dtpp
if [[ "$ONLY_FIRST_PAIR" == "1" || "$ONLY_FIRST_PAIR" == "true" ]]; then
  echo "DONE: ONLY_FIRST_PAIR requested; GameFormer/DTPP finished."
  exit 0
fi
run_pair plantf pluto

echo "DONE: budget-specific external checkpoints are under $EXTERNAL_OUT_ROOT/B{8,16,24}/"
