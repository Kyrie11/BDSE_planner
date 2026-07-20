#!/usr/bin/env bash
set -euo pipefail

for kv in "$@"; do
  [[ "$kv" == *=* ]] && export "$kv"
done

# v36 SCIDE-BDSE: Safety-Complete Interaction-Decisive Evidence planning.
# Hard safety is a deterministic, budget-exempt viability channel.  B=16 is
# reserved for decision evidence inside the viable action frontier.
ROOT_DIR="$(pwd)"
export BDSE_TRAIN_CACHE=${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/}
export BDSE_VAL_CACHE=${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}
export NUPLAN_ROOT=${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}
export NUPLAN_DB_ROOT=${NUPLAN_DB_ROOT:-$NUPLAN_ROOT/data/cache/val}
export V30_CKPT_IN=${V30_CKPT_IN:-outputs_v30/train/bdse_v30_pmvrbsr.best.pt}
export CL_WORKERS_PER_RUN=${CL_WORKERS_PER_RUN:-2}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/outputs_v36_runtime_v30ckpt}
OUT_ROOT="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_ROOT")"
TRAIN_ROOT="$OUT_ROOT/train"
OPEN_ROOT="$OUT_ROOT/open_loop"
CL_ROOT="$OUT_ROOT/closed_loop"
LOG_ROOT="$OUT_ROOT/v36_logs"
mkdir -p "$TRAIN_ROOT" "$OPEN_ROOT" "$CL_ROOT" "$LOG_ROOT"

RUN_MODE=${RUN_MODE:-open_loop}
V36_CKPT=${V36_CKPT:-$V30_CKPT_IN}
V36_MAIN_CFG=${V36_MAIN_CFG:-bdse/configs/v36_bdse_scide_balanced_fast_cl.yaml}
V35_BASELINE_JSON=${V35_BASELINE_JSON:-$ROOT_DIR/outputs_v35_runtime_v30ckpt/open_loop/open_loop_v35_dice_hard7.json}

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
  bdse/tests/test_v34_abiq.py \
  bdse/tests/test_v35_dice.py \
  bdse/tests/test_v36_scide.py

run_open_loop() {
  local gpu="$1" tag="$2" cfg="$3"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -m bdse.experiments.evaluate_open_loop \
      --config "$cfg" \
      --checkpoint "$V36_CKPT" \
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
  [[ -f "$V36_CKPT" ]] || { echo "Missing checkpoint: $V36_CKPT" >&2; exit 2; }
  run_open_loop "${GPU_OL_BALANCED:-0}" v36_scide_balanced bdse/configs/v36_bdse_scide_balanced_fast_cl.yaml &
  run_open_loop "${GPU_OL_INTERACTION:-0}" v36_scide_interaction10 bdse/configs/v36_bdse_scide_interaction10_fast_cl.yaml &
  run_open_loop "${GPU_OL_INFLUENCE:-1}" v36_scide_influence bdse/configs/v36_bdse_scide_influence_fast_cl.yaml &
  run_open_loop "${GPU_OL_CONTROL:-1}" v36_scide_no_frontier bdse/configs/v36_bdse_scide_no_frontier_control_fast_cl.yaml &
  wait

  set +e
  baseline_args=()
  [[ -f "$V35_BASELINE_JSON" ]] && baseline_args=(--baseline "$V35_BASELINE_JSON")
  python -m bdse.tools.check_v36_runtime_gate \
    "$OPEN_ROOT/open_loop_v36_scide_balanced.json" \
    "$OPEN_ROOT/open_loop_v36_scide_interaction10.json" \
    "$OPEN_ROOT/open_loop_v36_scide_influence.json" \
    "$OPEN_ROOT/open_loop_v36_scide_no_frontier.json" \
    "${baseline_args[@]}" \
    --write-best "$OPEN_ROOT/recommended_result_path.txt" \
    | tee "$OPEN_ROOT/runtime_gate.txt"
  gate_status=${PIPESTATUS[0]}
  set -e
  if [[ "$gate_status" -ne 0 && "${ENFORCE_RUNTIME_GATE:-1}" == "1" ]]; then
    echo "Runtime gate failed; do not run CL20 or training." >&2
    exit "$gate_status"
  fi
}

run_interaction_only_ablation() {
  [[ -f "$V36_CKPT" ]] || { echo "Missing checkpoint: $V36_CKPT" >&2; exit 2; }
  run_open_loop "${GPU_OL_ABLATION:-0}" v36_scide_interaction_only bdse/configs/v36_bdse_scide_interaction_only_ablation_fast_cl.yaml
}

run_train() {
  local output="$TRAIN_ROOT/bdse_v36_scide.pt"
  local init_mode=${TRAIN_INIT:-v30}
  local warm_args=()
  local epochs=${TRAIN_EPOCHS:-3}
  if [[ "$init_mode" == "clean" ]]; then
    epochs=${TRAIN_EPOCHS:-12}
    echo "[train] clean v36 training (${epochs} epochs)"
  else
    [[ -f "$V30_CKPT_IN" ]] || { echo "Missing V30_CKPT_IN=$V30_CKPT_IN" >&2; exit 2; }
    warm_args=(--warm-start-from "$V30_CKPT_IN")
    echo "[train] controlled v36 head finetune from v30"
  fi
  torchrun --standalone --nproc_per_node=${NPROC_PER_NODE:-2} -m bdse.experiments.train \
    --config bdse/configs/v36_bdse_scide_train.yaml \
    --split train --preprocessed-dir "$BDSE_TRAIN_CACHE" \
    --max-scenarios "${TRAIN_MAX_SCENARIOS:-12000}" \
    --val-split val --val-preprocessed-dir "$BDSE_VAL_CACHE" \
    --val-max-scenarios "${VAL_MAX_SCENARIOS:-1000}" \
    --epochs "$epochs" --batch-size "${TRAIN_BATCH_SIZE:-8}" \
    --num-workers "${TRAIN_NUM_WORKERS:-4}" --val-num-workers "${VAL_NUM_WORKERS:-2}" \
    --val-batch-size "${VAL_BATCH_SIZE:-8}" --val-mode open_loop \
    --val-every-n-epochs 1 --val-dense-diagnostic \
    --best-metric fixed_budget_critical_score \
    --best-metrics fixed_budget_critical_score teacher_action_match budget_vs_full_match selected_soft_interaction_decisive_recall effective_interaction_decisive_recall effective_hard_decisive_recall fallback_would_trigger_rate teacher_regret effective_query_count total_sparse_query_count \
    "${warm_args[@]}" --output "$output" --amp \
    --log-file "$LOG_ROOT/v36_scide_train.jsonl" \
    > "$LOG_ROOT/v36_scide_train.out" 2>&1
}

run_closed_loop_one() {
  local gpu="$1" tag="$2" cfg="$3" limit="$4"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export BDSE_STRICT_CLOSED_LOOP_DIAG=1
    export BDSE_CLOSED_LOOP_DIAG="$LOG_ROOT/${tag}.closed_loop_${limit}.diag.jsonl"
    rm -f "$BDSE_CLOSED_LOOP_DIAG"
    python -m bdse.experiments.evaluate_closed_loop \
      --config "$cfg" --checkpoint "$V36_CKPT" --device cuda \
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
  [[ -f "$V36_CKPT" ]] || { echo "Missing checkpoint: $V36_CKPT" >&2; exit 2; }
  run_closed_loop_one "${GPU_CL_MAIN:-0}" v36_scide_main "$V36_MAIN_CFG" "$limit" &
  run_closed_loop_one "${GPU_CL_SAFETY:-0}" v36_scide_safety bdse/configs/v36_bdse_scide_safety_fallback_fast_cl.yaml "$limit" &
  run_closed_loop_one "${GPU_CL_BBR:-1}" v36_scide_bbr bdse/configs/v36_bdse_scide_bbr_scur_fast_cl.yaml "$limit" &
  wait
  python -m bdse.tools.collect_closed_loop_metrics \
    "$CL_ROOT/v36_scide_main_${limit}" "$CL_ROOT/v36_scide_safety_${limit}" "$CL_ROOT/v36_scide_bbr_${limit}" \
    --csv "$CL_ROOT/v36_${limit}_compare.csv"
}

case "$RUN_MODE" in
  open_loop) run_runtime_gate ;;
  interaction_only_ablation) run_interaction_only_ablation ;;
  train) run_train ;;
  cl20) run_closed_loop_group 20 ;;
  cl50) run_closed_loop_group 50 ;;
  *) echo "RUN_MODE must be open_loop|interaction_only_ablation|train|cl20|cl50" >&2; exit 2 ;;
esac
