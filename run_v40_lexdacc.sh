#!/usr/bin/env bash
set -euo pipefail
for kv in "$@"; do [[ "$kv" == *=* ]] && export "$kv"; done

ROOT_DIR="$(pwd)"
export BDSE_TRAIN_CACHE=${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/}
export BDSE_VAL_CACHE=${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}
export NUPLAN_ROOT=${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}
export NUPLAN_DB_ROOT=${NUPLAN_DB_ROOT:-$NUPLAN_ROOT/data/cache/val}
export V30_CKPT_IN=${V30_CKPT_IN:-outputs_v30/train/bdse_v30_pmvrbsr.best.pt}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/outputs_v40_lexdacc_runtime_v30ckpt}
OUT_ROOT="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_ROOT")"
OPEN_ROOT="$OUT_ROOT/open_loop"; CL_ROOT="$OUT_ROOT/closed_loop"; TRAIN_ROOT="$OUT_ROOT/train"; LOG_ROOT="$OUT_ROOT/v40_logs"
mkdir -p "$OPEN_ROOT" "$CL_ROOT" "$TRAIN_ROOT" "$LOG_ROOT"
RUN_MODE=${RUN_MODE:-open_loop}
V40_CKPT=${V40_CKPT:-$V30_CKPT_IN}
CL_WORKERS_PER_RUN=${CL_WORKERS_PER_RUN:-2}

python -m py_compile $(find bdse -name '*.py')
python -m pytest -q \
  bdse/tests/test_v38_finetune_pair_mask.py \
  bdse/tests/test_v38_mars.py \
  bdse/tests/test_v40_lexdacc.py \
  bdse/tests/test_tournament_antisymmetry.py \
  bdse/tests/test_runtime_selector_no_teacher.py

run_open_loop_one() {
  local gpu="$1" tag="$2" cfg="$3"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -m bdse.experiments.evaluate_open_loop \
      --config "$cfg" --checkpoint "$V40_CKPT" --split val \
      --preprocessed-dir "$BDSE_VAL_CACHE" \
      --max-scenarios "${OPEN_LOOP_MAX_SCENARIOS:-1000}" --device cuda \
      --output "$OPEN_ROOT/open_loop_${tag}.json" \
      --per-sample-output "$OPEN_ROOT/open_loop_${tag}.jsonl" \
      > "$LOG_ROOT/${tag}.open_loop.out" 2>&1
  )
}

run_open_loop() {
  [[ -f "$V40_CKPT" ]] || { echo "Missing checkpoint: $V40_CKPT" >&2; exit 2; }
  run_open_loop_one "${GPU_OL_DACC:-0}" v40_lexdacc bdse/configs/v40_bdse_lexdacc_fast_cl.yaml &
  run_open_loop_one "${GPU_OL_CONTROL:-1}" v40_mars_control bdse/configs/v40_bdse_mars_control_fast_cl.yaml &
  wait
  set +e
  python -m bdse.tools.check_v40_lexdacc_gate \
    "$OPEN_ROOT/open_loop_v40_lexdacc.json" \
    "$OPEN_ROOT/open_loop_v40_mars_control.json" \
    --write-best "$OPEN_ROOT/recommended_result_path.txt" | tee "$OPEN_ROOT/runtime_gate.txt"
  status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 && "${ENFORCE_RUNTIME_GATE:-1}" == "1" ]]; then
    echo "V40 Lex-DACC runtime gate failed; do not run closed loop or finetune." >&2
    exit "$status"
  fi
}

run_train() {
  [[ -f "$V30_CKPT_IN" ]] || { echo "Missing V30_CKPT_IN=$V30_CKPT_IN" >&2; exit 2; }
  # Controlled head finetune. Lex-DACC itself is runtime-only: do not invoke
  # train until the strict open-loop gate and CL100 both complete.
  torchrun --standalone --nproc_per_node=${NPROC_PER_NODE:-2} -m bdse.experiments.train \
    --config bdse/configs/v38_bdse_mars_train.yaml \
    --split train --preprocessed-dir "$BDSE_TRAIN_CACHE" \
    --max-scenarios "${TRAIN_MAX_SCENARIOS:-12000}" \
    --val-split val --val-preprocessed-dir "$BDSE_VAL_CACHE" \
    --val-max-scenarios "${VAL_MAX_SCENARIOS:-1000}" \
    --epochs "${TRAIN_EPOCHS:-3}" --batch-size "${TRAIN_BATCH_SIZE:-8}" \
    --num-workers "${TRAIN_NUM_WORKERS:-4}" --val-num-workers "${VAL_NUM_WORKERS:-2}" \
    --val-batch-size "${VAL_BATCH_SIZE:-8}" --val-mode open_loop \
    --val-every-n-epochs 1 --val-dense-diagnostic \
    --best-metric fixed_budget_critical_score \
    --warm-start-from "$V30_CKPT_IN" --output "$TRAIN_ROOT/bdse_v40_head_finetune.pt" --amp \
    --log-file "$LOG_ROOT/v40_finetune.jsonl" > "$LOG_ROOT/v40_finetune.out" 2>&1
}

run_closed_loop_one() {
  local gpu="$1" tag="$2" cfg="$3" limit="$4"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export BDSE_STRICT_CLOSED_LOOP_DIAG=1
    export BDSE_CLOSED_LOOP_DIAG="$LOG_ROOT/${tag}.closed_loop_${limit}.diag.jsonl"
    rm -f "$BDSE_CLOSED_LOOP_DIAG"
    python -m bdse.experiments.evaluate_closed_loop \
      --config "$cfg" --checkpoint "$V40_CKPT" --device cuda \
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

run_closed_loop() {
  local limit="$1"
  [[ -f "$V40_CKPT" ]] || { echo "Missing checkpoint: $V40_CKPT" >&2; exit 2; }
  run_closed_loop_one "${GPU_CL_DACC:-0}" v40_lexdacc bdse/configs/v40_bdse_lexdacc_fast_cl.yaml "$limit" &
  run_closed_loop_one "${GPU_CL_CONTROL:-1}" v40_mars_control bdse/configs/v40_bdse_mars_control_fast_cl.yaml "$limit" &
  run_closed_loop_one "${GPU_CL_FALLBACK:-0}" v40_lexdacc_fallback bdse/configs/v40_bdse_lexdacc_fallback_fast_cl.yaml "$limit" &
  wait
  python -m bdse.tools.collect_closed_loop_metrics \
    "$CL_ROOT/v40_lexdacc_${limit}" "$CL_ROOT/v40_mars_control_${limit}" "$CL_ROOT/v40_lexdacc_fallback_${limit}" \
    --csv "$CL_ROOT/v40_${limit}_compare.csv"
}

case "$RUN_MODE" in
  open_loop) run_open_loop ;;
  train) run_train ;;
  cl20) run_closed_loop 20 ;;
  cl50) run_closed_loop 50 ;;
  cl100) run_closed_loop 100 ;;
  *) echo "RUN_MODE must be open_loop|train|cl20|cl50|cl100" >&2; exit 2 ;;
esac
