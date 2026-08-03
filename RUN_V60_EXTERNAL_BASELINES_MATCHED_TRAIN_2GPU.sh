#!/usr/bin/env bash
set -euo pipefail

# Matched-data training for the four trainable external baseline adapters.
# Two models run concurrently: gameformer/dtpp, then plantf/pluto.
# IMPORTANT: --output must end in *_budgeted.pt; train.py then writes the exact
# sweep-compatible *_budgeted.best.pt name (do not pass *.best.pt here).

: "${BDSE_TRAIN_CACHE:?Set BDSE_TRAIN_CACHE to bdse_train_v2}"
: "${BDSE_SPLIT_CACHE:?Set BDSE_SPLIT_CACHE to the frozen val_tune/val_calib cache}"
: "${EXTERNAL_OUT_ROOT:=outputs/external}"
: "${GPUS:=0,1}"
: "${MAX_TRAIN_SCENARIOS:=50000}"
: "${MAX_TRAIN_PER_SPLIT:=12500}"
: "${VAL_SPLIT:=val_tune}"
: "${VAL_SCENARIOS:=500}"
: "${EXTERNAL_EPOCHS:=30}"
: "${EXTERNAL_BATCH_SIZE:=32}"
: "${EXTERNAL_NUM_WORKERS:=10}"
: "${EXTERNAL_PREFETCH_FACTOR:=2}"
: "${EXTERNAL_LR:=0.0003}"
: "${EXTERNAL_WEIGHT_DECAY:=0.0001}"
: "${EXTERNAL_WARMUP_EPOCHS:=3}"
: "${EXTERNAL_VAL_EVERY:=3}"
: "${EXTERNAL_EARLY_STOP_PATIENCE:=3}"
: "${EXTERNAL_MIN_EPOCHS:=12}"
: "${EXTERNAL_SEED:=2026}"
: "${RESET_EXTERNAL_CHECKPOINTS:=0}"

IFS=',' read -r GPU0 GPU1 <<<"$GPUS"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$EXTERNAL_OUT_ROOT/logs"

if [[ "$RESET_EXTERNAL_CHECKPOINTS" == "1" ]]; then
  rm -f "$EXTERNAL_OUT_ROOT"/*_budgeted.pt \
        "$EXTERNAL_OUT_ROOT"/*_budgeted.best.pt \
        "$EXTERNAL_OUT_ROOT"/*_budgeted.data_manifest.json \
        "$EXTERNAL_OUT_ROOT"/*_budgeted.training_summary.json \
        "$EXTERNAL_OUT_ROOT"/*.train_log.jsonl
fi

train_one() {
  local name="$1" gpu="$2"
  local cfg="bdse/configs/external_${name}_budgeted.yaml"
  local out="$EXTERNAL_OUT_ROOT/${name}_budgeted.pt"
  local best="$EXTERNAL_OUT_ROOT/${name}_budgeted.best.pt"
  local log="$EXTERNAL_OUT_ROOT/${name}.train_log.jsonl"
  local resume_args=()
  if [[ -s "$out" ]]; then
    resume_args+=(--resume-from "$out")
  fi
  echo "[external] train $name on physical GPU $gpu; best=$best"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m bdse.external_baselines.train \
    --config "$cfg" \
    --split train_boston train_pittsburgh train_singapore train_vegas_2 \
    --preprocessed-dir "$BDSE_TRAIN_CACHE" \
    --max-scenarios "$MAX_TRAIN_SCENARIOS" \
    --max-scenarios-per-split "$MAX_TRAIN_PER_SPLIT" \
    --batch-size "$EXTERNAL_BATCH_SIZE" \
    --num-workers "$EXTERNAL_NUM_WORKERS" \
    --prefetch-factor "$EXTERNAL_PREFETCH_FACTOR" \
    --device cuda --amp --optimizer-fused \
    --lr "$EXTERNAL_LR" --weight-decay "$EXTERNAL_WEIGHT_DECAY" \
    --warmup-epochs "$EXTERNAL_WARMUP_EPOCHS" --scheduler cosine \
    --epochs "$EXTERNAL_EPOCHS" \
    --val-preprocessed-dir "$BDSE_SPLIT_CACHE" \
    --val-split "$VAL_SPLIT" \
    --val-max-scenarios "$VAL_SCENARIOS" \
    --val-mode loss --val-every-n-epochs "$EXTERNAL_VAL_EVERY" \
    --selection-metric val_action_ce \
    --early-stop-patience "$EXTERNAL_EARLY_STOP_PATIENCE" \
    --min-epochs "$EXTERNAL_MIN_EPOCHS" \
    --seed "$EXTERNAL_SEED" \
    --log-every-n-steps 25 \
    --output "$out" --log-file "$log" \
    "${resume_args[@]}" \
    > "$EXTERNAL_OUT_ROOT/logs/${name}.train.out" 2>&1
}

run_pair() {
  local a="$1" b="$2"
  train_one "$a" "$GPU0" & local p0=$!
  train_one "$b" "$GPU1" & local p1=$!
  local failed=0
  wait "$p0" || failed=1
  wait "$p1" || failed=1
  (( failed == 0 )) || { echo "training pair failed: $a/$b" >&2; exit 1; }
}

run_pair gameformer dtpp
run_pair plantf pluto

python -m bdse.tools.validate_external_checkpoint_suite \
  --checkpoint-root "$EXTERNAL_OUT_ROOT" \
  --expected-train-count "$MAX_TRAIN_SCENARIOS" \
  --expected-val-count "$VAL_SCENARIOS" \
  --output "$EXTERNAL_OUT_ROOT/external_checkpoint_suite_validation.json"

cat <<EOF
Matched external checkpoints are ready:
  $EXTERNAL_OUT_ROOT/gameformer_budgeted.best.pt
  $EXTERNAL_OUT_ROOT/dtpp_budgeted.best.pt
  $EXTERNAL_OUT_ROOT/plantf_budgeted.best.pt
  $EXTERNAL_OUT_ROOT/pluto_budgeted.best.pt
EOF
