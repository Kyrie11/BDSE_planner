#!/usr/bin/env bash
set -euo pipefail

: "${BDSE_TRAIN_CACHE:?Set BDSE_TRAIN_CACHE to the preprocessed training cache}"
: "${BDSE_VAL_CACHE:?Set BDSE_VAL_CACHE to the preprocessed validation cache}"

V30_CKPT_IN="${V30_CKPT_IN:-outputs_v30/train/bdse_v30_pmvrbsr.best.pt}"
OUT_ROOT="${OUT_ROOT:-outputs_v44_rads}"
RUN_MODE="${RUN_MODE:-train_open_loop}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
MAX_TRAIN_SCENARIOS="${MAX_TRAIN_SCENARIOS:-50000}"
VAL_SCENARIOS="${VAL_SCENARIOS:-1000}"
OPEN_LOOP_MAX_SCENARIOS="${OPEN_LOOP_MAX_SCENARIOS:-1000}"
EVAL_CONFIG="${EVAL_CONFIG:-bdse/configs/v44_bdse_rads_fast_cl.yaml}"

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT_ROOT/train" "$OUT_ROOT/open_loop" "$OUT_ROOT/closed_loop"
CKPT="$OUT_ROOT/train/bdse_v44_rads.pt"
BEST_CKPT="$OUT_ROOT/train/bdse_v44_rads.best.pt"

train() {
  python -m bdse.experiments.train \
    --config bdse/configs/v44_bdse_rads_train.yaml \
    --split train_boston train_pittsburgh train_singapore train_vegas_2 \
    --preprocessed-dir "$BDSE_TRAIN_CACHE" \
    --max-scenarios "$MAX_TRAIN_SCENARIOS" \
    --max-scenarios-per-split $((MAX_TRAIN_SCENARIOS / 4)) \
    --batch-size "${BATCH_SIZE:-8}" \
    --num-workers "${NUM_WORKERS:-8}" \
    --device "$DEVICE" \
    --amp \
    --warm-start-from "$V30_CKPT_IN" \
    --val-preprocessed-dir "$BDSE_VAL_CACHE" \
    --val-split val \
    --val-max-scenarios "$VAL_SCENARIOS" \
    --val-mode open_loop \
    --val-dense-diagnostic \
    --val-every-n-epochs 1 \
    --best-metric teacher_action_match \
    --best-metrics teacher_action_match teacher_regret full_interface_action_match \
    --log-file "$OUT_ROOT/train/bdse_v44_rads.train_log.jsonl" \
    --output "$CKPT"
}

open_loop() {
  local checkpoint="${V44_CKPT:-$BEST_CKPT}"
  python -m bdse.experiments.evaluate_open_loop \
    --config "$EVAL_CONFIG" \
    --checkpoint "$checkpoint" \
    --split val \
    --preprocessed-dir "$BDSE_VAL_CACHE" \
    --max-scenarios "$OPEN_LOOP_MAX_SCENARIOS" \
    --device "$DEVICE" \
    --output "$OUT_ROOT/open_loop/open_loop_v44_rads.json" \
    --per-sample-output "$OUT_ROOT/open_loop/open_loop_v44_rads.jsonl"
}

closed_loop_20() {
  : "${NUPLAN_ROOT:?Set NUPLAN_ROOT for closed-loop evaluation}"
  local checkpoint="${V44_CKPT:-$BEST_CKPT}"
  python -m bdse.experiments.evaluate_closed_loop \
    --config "$EVAL_CONFIG" \
    --checkpoint "$checkpoint" \
    --device "$DEVICE" \
    --challenge closed_loop_nonreactive_agents \
    --metric-aggregator closed_loop_nonreactive_agents_weighted_average \
    --output-dir "$OUT_ROOT/closed_loop/v44_rads_20" \
    --experiment-uid v44_rads_20 \
    --nuplan-module nuplan.planning.script.run_simulation \
    --scenario-builder nuplan \
    --worker single_machine_thread_pool \
    --hydra-full-error \
    --nuplan-data-root "$NUPLAN_ROOT" \
    --nuplan-map-root "$NUPLAN_ROOT/maps" \
    --nuplan-exp-root "$NUPLAN_ROOT/exp" \
    --nuplan-db-root "$NUPLAN_ROOT/data/cache/val/" \
    -- \
    scenario_filter.limit_total_scenarios=20 \
    scenario_filter.shuffle=false \
    worker.max_workers="${CL_WORKERS:-2}" \
    run_metric=true
}

case "$RUN_MODE" in
  train) train ;;
  open_loop) open_loop ;;
  train_open_loop) train; open_loop ;;
  cl20) closed_loop_20 ;;
  *) echo "Unknown RUN_MODE=$RUN_MODE (train|open_loop|train_open_loop|cl20)" >&2; exit 2 ;;
esac
