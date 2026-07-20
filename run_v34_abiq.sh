#!/usr/bin/env bash
set -euo pipefail

for kv in "$@"; do
  [[ "$kv" == *=* ]] && export "$kv"
done

# v34 ABIQ-BDSE: Antisymmetric Budgeted Influence Query planning.
# Phase order: runtime-only open-loop gate -> runtime CL20 -> head finetune -> CL50.
ROOT_DIR="$(pwd)"
export BDSE_TRAIN_CACHE=${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/}
export BDSE_VAL_CACHE=${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}
export NUPLAN_ROOT=${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}
export NUPLAN_DB_ROOT=${NUPLAN_DB_ROOT:-$NUPLAN_ROOT/data/cache/val}
export V30_CKPT_IN=${V30_CKPT_IN:-outputs_v30/train/bdse_v30_pmvrbsr.best.pt}
export CL_WORKERS_PER_RUN=${CL_WORKERS_PER_RUN:-2}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/outputs_v34_runtime_v30ckpt}
OUT_ROOT="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_ROOT")"
TRAIN_ROOT="$OUT_ROOT/train"
OPEN_ROOT="$OUT_ROOT/open_loop"
CL_ROOT="$OUT_ROOT/closed_loop"
LOG_ROOT="$OUT_ROOT/v34_logs"
mkdir -p "$TRAIN_ROOT" "$OPEN_ROOT" "$CL_ROOT" "$LOG_ROOT"

RUN_MODE=${RUN_MODE:-open_loop}
V34_CKPT=${V34_CKPT:-$V30_CKPT_IN}
V34_MAIN_CFG=${V34_MAIN_CFG:-bdse/configs/v34_bdse_abiq_balanced_fast_cl.yaml}

python -m py_compile $(find bdse -name '*.py')
python -m pytest -q \
  bdse/tests/test_tournament_antisymmetry.py \
  bdse/tests/test_selector_monotonicity.py \
  bdse/tests/test_runtime_selector_no_teacher.py \
  bdse/tests/test_followup_training_and_closed_loop.py \
  bdse/tests/test_v29_adaptive_min_violation.py \
  bdse/tests/test_v30_pareto_progress_recovery.py \
  bdse/tests/test_v31_rhvcdsr.py \
  bdse/tests/test_v32_cavr.py \
  bdse/tests/test_v33_carb.py \
  bdse/tests/test_v34_abiq.py

run_open_loop() {
  local gpu="$1" tag="$2" cfg="$3"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -m bdse.experiments.evaluate_open_loop \
      --config "$cfg" \
      --checkpoint "$V34_CKPT" \
      --split val \
      --preprocessed-dir "$BDSE_VAL_CACHE" \
      --max-scenarios "${OPEN_LOOP_MAX_SCENARIOS:-1000}" \
      --device cuda \
      --output "$OPEN_ROOT/open_loop_${tag}.json" \
      --per-sample-output "$OPEN_ROOT/open_loop_${tag}.jsonl" \
      > "$LOG_ROOT/${tag}.open_loop.out" 2>&1
  )
}

run_runtime_gate() {
  [[ -f "$V34_CKPT" ]] || { echo "Missing checkpoint: $V34_CKPT" >&2; exit 2; }
  run_open_loop "${GPU_OL_BALANCED:-0}" v34_abiq_balanced bdse/configs/v34_bdse_abiq_balanced_fast_cl.yaml &
  run_open_loop "${GPU_OL_HARD:-0}" v34_abiq_hard_guard bdse/configs/v34_bdse_abiq_hard_guard_fast_cl.yaml &
  run_open_loop "${GPU_OL_INTERACTION:-1}" v34_abiq_interaction_guard bdse/configs/v34_bdse_abiq_interaction_guard_fast_cl.yaml &
  run_open_loop "${GPU_OL_ACTION:-1}" v34_abiq_action_rank bdse/configs/v34_bdse_abiq_action_rank_fast_cl.yaml &
  wait

  set +e
  python -m bdse.tools.check_v34_runtime_gate \
    "$OPEN_ROOT/open_loop_v34_abiq_balanced.json" \
    "$OPEN_ROOT/open_loop_v34_abiq_hard_guard.json" \
    "$OPEN_ROOT/open_loop_v34_abiq_interaction_guard.json" \
    "$OPEN_ROOT/open_loop_v34_abiq_action_rank.json" \
    --write-best "$OPEN_ROOT/recommended_result_path.txt" \
    | tee "$OPEN_ROOT/runtime_gate.txt"
  gate_status=${PIPESTATUS[0]}
  set -e
  if [[ "$gate_status" -ne 0 && "${ENFORCE_RUNTIME_GATE:-1}" == "1" ]]; then
    echo "Runtime gate failed; do not run CL20 or training." >&2
    exit "$gate_status"
  fi
}

run_train() {
  local output="$TRAIN_ROOT/bdse_v34_abiq.pt"
  local init_mode=${TRAIN_INIT:-v30}
  local warm_args=()
  local epochs=${TRAIN_EPOCHS:-3}
  if [[ "$init_mode" == "clean" ]]; then
    # Final-paper clean training, only after runtime + CL20 are frozen.
    epochs=${TRAIN_EPOCHS:-12}
    echo "[train] clean v34 training (${epochs} epochs)"
  else
    [[ -f "$V30_CKPT_IN" ]] || { echo "Missing V30_CKPT_IN=$V30_CKPT_IN" >&2; exit 2; }
    warm_args=(--warm-start-from "$V30_CKPT_IN")
    echo "[train] controlled v34 head finetune from v30"
  fi
  torchrun --standalone --nproc_per_node=${NPROC_PER_NODE:-2} -m bdse.experiments.train \
    --config bdse/configs/v34_bdse_abiq_train.yaml \
    --split train --preprocessed-dir "$BDSE_TRAIN_CACHE" \
    --max-scenarios "${TRAIN_MAX_SCENARIOS:-12000}" \
    --val-split val --val-preprocessed-dir "$BDSE_VAL_CACHE" \
    --val-max-scenarios "${VAL_MAX_SCENARIOS:-1000}" \
    --epochs "$epochs" --batch-size "${TRAIN_BATCH_SIZE:-8}" \
    --num-workers "${TRAIN_NUM_WORKERS:-4}" --val-num-workers "${VAL_NUM_WORKERS:-2}" \
    --val-batch-size "${VAL_BATCH_SIZE:-8}" --val-mode open_loop \
    --val-every-n-epochs 1 --val-dense-diagnostic \
    --best-metric fixed_budget_critical_score \
    --best-metrics fixed_budget_critical_score teacher_action_match budget_vs_full_match selected_interaction_decisive_recall selected_hard_decisive_recall fallback_would_trigger_rate teacher_regret effective_query_count total_sparse_query_count \
    "${warm_args[@]}" --output "$output" --amp \
    --log-file "$LOG_ROOT/v34_abiq_train.jsonl" \
    > "$LOG_ROOT/v34_abiq_train.out" 2>&1
}

run_closed_loop_one() {
  local gpu="$1" tag="$2" cfg="$3" limit="$4"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export BDSE_STRICT_CLOSED_LOOP_DIAG=1
    export BDSE_CLOSED_LOOP_DIAG="$LOG_ROOT/${tag}.closed_loop_${limit}.diag.jsonl"
    rm -f "$BDSE_CLOSED_LOOP_DIAG"
    python -m bdse.experiments.evaluate_closed_loop \
      --config "$cfg" --checkpoint "$V34_CKPT" --device cuda \
      --challenge closed_loop_nonreactive_agents \
      --metric-aggregator closed_loop_nonreactive_agents_weighted_average \
      --output-dir "$CL_ROOT/${tag}_${limit}" --experiment-uid "bdse_${tag}_${limit}" \
      --nuplan-module nuplan.planning.script.run_simulation \
      --scenario-builder nuplan --worker single_machine_thread_pool --hydra-full-error \
      --nuplan-data-root "$NUPLAN_ROOT" --nuplan-map-root "$NUPLAN_ROOT/maps" \
      --nuplan-exp-root "$NUPLAN_ROOT/exp" --nuplan-db-root "$NUPLAN_DB_ROOT" -- \
      scenario_filter.limit_total_scenarios="$limit" scenario_filter.shuffle=false \
      worker.max_workers="$CL_WORKERS_PER_RUN" run_metric=true \
      > "$LOG_ROOT/${tag}.closed_loop_${limit}.out" 2>&1
    [[ -s "$BDSE_CLOSED_LOOP_DIAG" ]] || { echo "Missing diagnostics: $BDSE_CLOSED_LOOP_DIAG" >&2; exit 3; }
  )
}

run_closed_loop_group() {
  local limit="$1"
  [[ -f "$V34_CKPT" ]] || { echo "Missing checkpoint: $V34_CKPT" >&2; exit 2; }
  run_closed_loop_one "${GPU_CL_MAIN:-0}" v34_abiq_main "$V34_MAIN_CFG" "$limit" &
  run_closed_loop_one "${GPU_CL_SAFETY:-0}" v34_abiq_safety bdse/configs/v34_bdse_abiq_safety_fallback_fast_cl.yaml "$limit" &
  run_closed_loop_one "${GPU_CL_BBR:-1}" v34_abiq_bbr bdse/configs/v34_bdse_abiq_bbr_scur_fast_cl.yaml "$limit" &
  wait
  python -m bdse.tools.collect_closed_loop_metrics \
    "$CL_ROOT/v34_abiq_main_${limit}" "$CL_ROOT/v34_abiq_safety_${limit}" "$CL_ROOT/v34_abiq_bbr_${limit}" \
    --csv "$CL_ROOT/v34_${limit}_compare.csv"
}

case "$RUN_MODE" in
  open_loop) run_runtime_gate ;;
  train) run_train ;;
  cl20) run_closed_loop_group 20 ;;
  cl50) run_closed_loop_group 50 ;;
  *) echo "RUN_MODE must be open_loop|train|cl20|cl50" >&2; exit 2 ;;
esac
