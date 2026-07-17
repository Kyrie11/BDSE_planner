#!/usr/bin/env bash
set -euo pipefail

# v26 / RBSR: Risk-Bounded Safety Recovery
# Key changes over v25 DG-CACE:
#   1) runtime-flagged actions are hard-masked in tournament whenever an unflagged candidate exists;
#   2) low normalized delta no longer triggers fallback by itself;
#   3) fallback is fixed-budget by default: no expanded evidence stage, only safe-progress rule recovery;
#   4) optional v26 finetune adds safe-frontier certificate loss.

ROOT_DIR="$(pwd)"
export BDSE_TRAIN_CACHE=${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/}
export BDSE_VAL_CACHE=${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}
export NUPLAN_ROOT=${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}
export NUPLAN_DB_ROOT=${NUPLAN_DB_ROOT:-$NUPLAN_ROOT/data/cache/val}
export V25_CKPT_IN=${V25_CKPT_IN:-outputs_v25/train/bdse_v25_dgcace.best.pt}
export V24_CKPT_IN=${V24_CKPT_IN:-outputs_v24/train/bdse_v24_cace.best.pt}
export V11_CKPT=${V11_CKPT:-outputs/v11_train/bdse_v11_ta_selector.best.pt}
export CL_WORKERS_PER_RUN=${CL_WORKERS_PER_RUN:-2}
export BDSE_REPLAN_INTERVAL_TICKS=${BDSE_REPLAN_INTERVAL_TICKS:-5}
export BDSE_PROFILE_CLOSED_LOOP=${BDSE_PROFILE_CLOSED_LOOP:-1}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/outputs_v26}
TRAIN_ROOT="$OUT_ROOT/train"
OPEN_ROOT="$OUT_ROOT/open_loop"
CL_ROOT="$OUT_ROOT/closed_loop"
LOG_ROOT="$OUT_ROOT/v26_logs"
mkdir -p "$TRAIN_ROOT" "$OPEN_ROOT" "$CL_ROOT" "$LOG_ROOT"

python -m py_compile $(find bdse -name '*.py')
python -m pytest -q \
  bdse/tests/test_flip_rank_selector.py \
  bdse/tests/test_certificate_utility_refinement.py \
  bdse/tests/test_runtime_base_prior.py \
  bdse/tests/test_runtime_alignment_fixes.py \
  bdse/tests/test_tournament_antisymmetry.py \
  bdse/tests/test_selector_monotonicity.py \
  bdse/tests/test_family_and_safety_pair_fixes.py \
  bdse/tests/test_runtime_selector_no_teacher.py \
  bdse/tests/test_followup_training_and_closed_loop.py \
  bdse/tests/test_v19_behavior_actionrank.py \
  bdse/tests/test_v22_adaptive_hybrid_selector.py

TRAIN_OUTPUT="$TRAIN_ROOT/bdse_v26_rbsr.pt"
V26_CKPT=${V26_CKPT:-$TRAIN_ROOT/bdse_v26_rbsr.best.pt}

pick_warm_start() {
  if [[ -f "$V25_CKPT_IN" ]]; then echo "$V25_CKPT_IN"; return; fi
  if [[ -f "$V24_CKPT_IN" ]]; then echo "$V24_CKPT_IN"; return; fi
  if [[ -f "$V11_CKPT" ]]; then echo "$V11_CKPT"; return; fi
  echo ""; return 1
}

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  WARM_START="$(pick_warm_start)"
  if [[ -z "$WARM_START" ]]; then
    echo "Missing warm-start checkpoint. Set V25_CKPT_IN, V24_CKPT_IN, or V11_CKPT." >&2
    exit 1
  fi
  echo "[train] v26 RBSR finetune from $WARM_START"
  torchrun --standalone --nproc_per_node=${NPROC_PER_NODE:-2} -m bdse.experiments.train \
    --config bdse/configs/v26_bdse_rbsr_train.yaml \
    --split train \
    --preprocessed-dir "$BDSE_TRAIN_CACHE" \
    --max-scenarios "${TRAIN_MAX_SCENARIOS:-12000}" \
    --val-split val \
    --val-preprocessed-dir "$BDSE_VAL_CACHE" \
    --val-max-scenarios "${VAL_MAX_SCENARIOS:-1000}" \
    --epochs "${TRAIN_EPOCHS:-4}" \
    --batch-size "${TRAIN_BATCH_SIZE:-8}" \
    --num-workers "${TRAIN_NUM_WORKERS:-4}" \
    --val-num-workers "${VAL_NUM_WORKERS:-2}" \
    --val-batch-size "${VAL_BATCH_SIZE:-8}" \
    --val-mode open_loop \
    --val-every-n-epochs "${VAL_EVERY_N_EPOCHS:-1}" \
    --best-metrics auto bdse_score teacher_action_match budget_vs_full_match selected_interaction_decisive_recall selected_hard_decisive_recall fallback_would_trigger_rate teacher_regret \
    --warm-start-from "$WARM_START" \
    --output "$TRAIN_OUTPUT" \
    --amp \
    --log-file "$LOG_ROOT/v26_rbsr_train.jsonl" \
    > "$LOG_ROOT/v26_rbsr_train.out" 2>&1
fi

# For zero-training ablation, reuse the v25 checkpoint with v26 runtime guard/configs:
#   SKIP_TRAIN=1 V26_CKPT=outputs_v25/train/bdse_v25_dgcace.best.pt bash run_v26_rbsr.sh
if [[ ! -f "$V26_CKPT" ]]; then
  echo "Missing v26 checkpoint: $V26_CKPT" >&2
  echo "For runtime-only check, run: SKIP_TRAIN=1 V26_CKPT=$V25_CKPT_IN bash run_v26_rbsr.sh" >&2
  exit 1
fi

run_open_loop() {
  local gpu="$1" tag="$2" cfg="$3" ckpt="$4"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[open-loop] $tag on GPU $gpu"
    python -m bdse.experiments.evaluate_open_loop \
      --config "$cfg" \
      --checkpoint "$ckpt" \
      --split val \
      --preprocessed-dir "$BDSE_VAL_CACHE" \
      --max-scenarios "${OPEN_LOOP_MAX_SCENARIOS:-1000}" \
      --device cuda \
      --output "$OPEN_ROOT/open_loop_${tag}.json" \
      --per-sample-output "$OPEN_ROOT/open_loop_${tag}.jsonl" \
      > "$LOG_ROOT/${tag}.open_loop.out" 2>&1
  )
}

print_open_loop_compare() {
  python - "$OPEN_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
paths = [
    ('v26_rbsr_fixed_budget', root / 'open_loop_v26_rbsr_fixed_budget.json'),
    ('v26_rbsr_safety_fallback', root / 'open_loop_v26_rbsr_safety_fallback.json'),
    ('v26_rbsr_bbr_scur', root / 'open_loop_v26_rbsr_bbr_scur.json'),
    ('v26_rbsr_lcb_control', root / 'open_loop_v26_rbsr_lcb_control.json'),
]
keys = [
    'teacher_action_match','decision_sufficiency','budget_vs_full_match','teacher_regret',
    'fallback_would_trigger_rate','pair_sign_acc_winner_rival','pair_sign_acc_interaction','pair_sign_acc_hard',
    'selected_interaction_decisive_recall','selected_hard_decisive_recall','selected_decisive_atom_recall',
    'selector_adaptive_lcb_frac','selector_hybrid_lcb_seed_atoms','selector_hybrid_action_atoms',
    'selector_decision_family_selected','selector_pair_atom_query_count','tournament_pair_atom_query_count',
    'total_sparse_query_count','effective_query_count'
]
for name, path in paths:
    if not path.exists():
        continue
    d = json.load(open(path))
    print('\n' + name)
    for k in keys:
        print(f'{k}: {d.get(k)}')
PY
}

run_open_loop 0 v26_rbsr_fixed_budget bdse/configs/v26_bdse_rbsr_fixed_budget_fast_cl.yaml "$V26_CKPT" &
run_open_loop 1 v26_rbsr_safety_fallback bdse/configs/v26_bdse_rbsr_safety_fallback_fast_cl.yaml "$V26_CKPT" &
wait
run_open_loop 0 v26_rbsr_bbr_scur bdse/configs/v26_bdse_rbsr_bbr_scur_fast_cl.yaml "$V26_CKPT" &
run_open_loop 1 v26_rbsr_lcb_control bdse/configs/v26_bdse_rbsr_lcb_control_fast_cl.yaml "$V26_CKPT" &
wait
print_open_loop_compare

if [[ "${OPEN_LOOP_ONLY:-0}" == "1" ]]; then
  echo "OPEN_LOOP_ONLY=1, skipping closed-loop runs."
  exit 0
fi

run_closed_loop() {
  local gpu="$1" tag="$2" cfg="$3" ckpt="$4" limit="$5"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export BDSE_CLOSED_LOOP_DIAG="$LOG_ROOT/${tag}.closed_loop_${limit}.diag.jsonl"
    echo "[closed-loop-${limit}] $tag on GPU $gpu"
    python -m bdse.experiments.evaluate_closed_loop \
      --config "$cfg" \
      --checkpoint "$ckpt" \
      --device cuda \
      --challenge closed_loop_nonreactive_agents \
      --metric-aggregator closed_loop_nonreactive_agents_weighted_average \
      --output-dir "$CL_ROOT/${tag}_${limit}" \
      --experiment-uid "bdse_${tag}_${limit}" \
      --nuplan-module nuplan.planning.script.run_simulation \
      --scenario-builder nuplan \
      --worker single_machine_thread_pool \
      --hydra-full-error \
      --nuplan-data-root "$NUPLAN_ROOT" \
      --nuplan-map-root "$NUPLAN_ROOT/maps" \
      --nuplan-exp-root "$NUPLAN_ROOT/exp" \
      --nuplan-db-root "$NUPLAN_DB_ROOT" \
      -- \
      scenario_filter.limit_total_scenarios="$limit" \
      scenario_filter.shuffle=false \
      worker.max_workers="$CL_WORKERS_PER_RUN" \
      run_metric=true \
      > "$LOG_ROOT/${tag}.closed_loop_${limit}.out" 2>&1
  )
}

run_closed_loop 0 v26_rbsr_fixed_budget bdse/configs/v26_bdse_rbsr_fixed_budget_fast_cl.yaml "$V26_CKPT" 20 &
run_closed_loop 1 v26_rbsr_safety_fallback bdse/configs/v26_bdse_rbsr_safety_fallback_fast_cl.yaml "$V26_CKPT" 20 &
wait
run_closed_loop 0 v26_rbsr_bbr_scur bdse/configs/v26_bdse_rbsr_bbr_scur_fast_cl.yaml "$V26_CKPT" 20 &
run_closed_loop 1 v26_rbsr_lcb_control bdse/configs/v26_bdse_rbsr_lcb_control_fast_cl.yaml "$V26_CKPT" 20 &
wait

python -m bdse.tools.collect_closed_loop_metrics \
  "$CL_ROOT/v26_rbsr_fixed_budget_20" \
  "$CL_ROOT/v26_rbsr_safety_fallback_20" \
  "$CL_ROOT/v26_rbsr_bbr_scur_20" \
  "$CL_ROOT/v26_rbsr_lcb_control_20" \
  --csv "$CL_ROOT/v26_20_compare.csv"

python - "$LOG_ROOT" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
for path in sorted(root.glob('*.closed_loop_20.diag.jsonl')):
    replans = []
    reasons = Counter()
    selected_flags = 0
    hard_filter = 0
    for line in path.read_text(errors='ignore').splitlines():
        try:
            d = json.loads(line).get('diagnostics', {})
        except Exception:
            continue
        if d.get('cached_plan'):
            continue
        replans.append(d)
        reasons[str(d.get('fallback_reason', 'missing'))] += 1
        t = d.get('tournament', {}) or {}
        selected_flags += int(bool(t.get('selected_action_safety_flag', False)))
        hard_filter += int(bool(t.get('hard_filter_applied', False)))
    if not replans:
        continue
    triggered = sum(1 for d in replans if d.get('fallback_triggered')) / len(replans)
    print(path.name, 'replans=', len(replans), 'fallback_triggered_rate=', round(triggered, 4),
          'selected_action_safety_flag_rate=', round(selected_flags / len(replans), 4),
          'hard_filter_applied_rate=', round(hard_filter / len(replans), 4),
          'reasons=', dict(reasons))
PY

if [[ "${RUN_CL50:-0}" == "1" ]]; then
  run_closed_loop 0 v26_rbsr_fixed_budget bdse/configs/v26_bdse_rbsr_fixed_budget_fast_cl.yaml "$V26_CKPT" 50 &
  run_closed_loop 1 v26_rbsr_safety_fallback bdse/configs/v26_bdse_rbsr_safety_fallback_fast_cl.yaml "$V26_CKPT" 50 &
  wait
  python -m bdse.tools.collect_closed_loop_metrics \
    "$CL_ROOT/v26_rbsr_fixed_budget_50" \
    "$CL_ROOT/v26_rbsr_safety_fallback_50" \
    --csv "$CL_ROOT/v26_50_compare.csv"
fi

echo "Done. Key outputs:"
echo "  $TRAIN_ROOT/bdse_v26_rbsr.best.pt"
echo "  $OPEN_ROOT/open_loop_v26_*.json"
echo "  $CL_ROOT/v26_20_compare.csv"
echo "  $LOG_ROOT/*.closed_loop_20.diag.jsonl"
echo "Runtime-only first check: SKIP_TRAIN=1 V26_CKPT=$V25_CKPT_IN OPEN_LOOP_ONLY=1 bash run_v26_rbsr.sh"
echo "Runtime-only CL20: SKIP_TRAIN=1 V26_CKPT=$V25_CKPT_IN bash run_v26_rbsr.sh"
echo "Finetune: TRAIN_MAX_SCENARIOS=12000 VAL_MAX_SCENARIOS=1000 TRAIN_EPOCHS=4 bash run_v26_rbsr.sh"
