#!/usr/bin/env bash
set -euo pipefail

# v25 / DG-CACE:
#   - keep v24 CACE, but include hard/safety atoms on critical crossings;
#   - add deployment-gate-aware certificate gap losses;
#   - calibrate normalized closed-loop fallback thresholds so the fallback stage
#     does not dominate every replan;
#   - compare trained LCB, fixed control, adaptive branches, and a no-fallback ablation.

ROOT_DIR="$(pwd)"
export BDSE_TRAIN_CACHE=${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/}
export BDSE_VAL_CACHE=${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}
export NUPLAN_ROOT=${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}
export NUPLAN_DB_ROOT=${NUPLAN_DB_ROOT:-$NUPLAN_ROOT/data/cache/val}
export V24_CKPT_IN=${V24_CKPT_IN:-outputs_v24/train/bdse_v24_cace.best.pt}
export V11_CKPT=${V11_CKPT:-outputs/v11_train/bdse_v11_ta_selector.best.pt}
export CL_WORKERS_PER_RUN=${CL_WORKERS_PER_RUN:-2}
export BDSE_REPLAN_INTERVAL_TICKS=${BDSE_REPLAN_INTERVAL_TICKS:-5}
export BDSE_PROFILE_CLOSED_LOOP=${BDSE_PROFILE_CLOSED_LOOP:-1}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/outputs_v25}
TRAIN_ROOT="$OUT_ROOT/train"
OPEN_ROOT="$OUT_ROOT/open_loop"
CL_ROOT="$OUT_ROOT/closed_loop"
LOG_ROOT="$OUT_ROOT/v25_logs"
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

TRAIN_OUTPUT="$TRAIN_ROOT/bdse_v25_dgcace.pt"
V25_CKPT=${V25_CKPT:-$TRAIN_ROOT/bdse_v25_dgcace.best.pt}

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if [[ -f "$V24_CKPT_IN" ]]; then
    WARM_START="$V24_CKPT_IN"
  elif [[ -f "$V11_CKPT" ]]; then
    WARM_START="$V11_CKPT"
  else
    echo "Missing warm-start checkpoint. Set V24_CKPT_IN or V11_CKPT." >&2
    exit 1
  fi
  echo "[train] v25 DG-CACE finetune from $WARM_START"
  torchrun --standalone --nproc_per_node=2 -m bdse.experiments.train \
    --config bdse/configs/v25_bdse_dgcace_train.yaml \
    --split train \
    --preprocessed-dir "$BDSE_TRAIN_CACHE" \
    --max-scenarios "${TRAIN_MAX_SCENARIOS:-12000}" \
    --val-split val \
    --val-preprocessed-dir "$BDSE_VAL_CACHE" \
    --val-max-scenarios "${VAL_MAX_SCENARIOS:-1000}" \
    --epochs "${TRAIN_EPOCHS:-5}" \
    --batch-size "${TRAIN_BATCH_SIZE:-8}" \
    --num-workers "${TRAIN_NUM_WORKERS:-4}" \
    --val-num-workers "${VAL_NUM_WORKERS:-2}" \
    --val-batch-size "${VAL_BATCH_SIZE:-8}" \
    --val-mode open_loop \
    --val-every-n-epochs "${VAL_EVERY_N_EPOCHS:-1}" \
    --best-metrics auto bdse_score teacher_action_match budget_vs_full_match selected_interaction_decisive_recall selected_hard_decisive_recall teacher_regret \
    --warm-start-from "$WARM_START" \
    --output "$TRAIN_OUTPUT" \
    --amp \
    --log-file "$LOG_ROOT/v25_dgcace_train.jsonl" \
    > "$LOG_ROOT/v25_dgcace_train.out" 2>&1
fi

if [[ ! -f "$V25_CKPT" ]]; then
  echo "Missing v25 checkpoint: $V25_CKPT" >&2
  echo "Set V25_CKPT=/path/to/checkpoint or run without SKIP_TRAIN=1." >&2
  exit 1
fi

run_open_loop() {
  local gpu="$1"
  local tag="$2"
  local cfg="$3"
  local ckpt="$4"
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
    ('v25_lcb_legacy_dgcace', root / 'open_loop_v25_lcb_legacy_dgcace.json'),
    ('v25_dgcace_fixed50_control', root / 'open_loop_v25_dgcace_fixed50_control.json'),
    ('v25_dgcace_bbr_scur_tau35', root / 'open_loop_v25_dgcace_bbr_scur_tau35.json'),
    ('v25_dgcace_actionheavy_tau45', root / 'open_loop_v25_dgcace_actionheavy_tau45.json'),
    ('v25_dgcace_safety_tau30', root / 'open_loop_v25_dgcace_safety_tau30.json'),
    ('v25_dgcace_no_fallback_ablation', root / 'open_loop_v25_dgcace_no_fallback_ablation.json'),
]
keys = [
    'teacher_action_match','decision_sufficiency','budget_vs_full_match','teacher_regret',
    'fallback_would_trigger_rate','pair_sign_acc_winner_rival','pair_sign_acc_interaction','pair_sign_acc_hard',
    'selected_interaction_decisive_recall','selected_hard_decisive_recall','selected_decisive_atom_recall',
    'selector_adaptive_lcb_frac','selector_hybrid_lcb_seed_atoms','selector_hybrid_action_atoms',
    'selector_hybrid_min_action_budget','selector_hybrid_max_lcb_seed_atoms','selector_decision_family_selected',
    'selector_pair_atom_query_count','tournament_pair_atom_query_count','total_sparse_query_count','effective_query_count'
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

run_open_loop 0 v25_lcb_legacy_dgcace bdse/configs/v25_bdse_lcb_legacy_dgcace_fast_cl.yaml "$V25_CKPT" &
run_open_loop 1 v25_dgcace_fixed50_control bdse/configs/v25_bdse_dgcace_fixed50_control_fast_cl.yaml "$V25_CKPT" &
wait
run_open_loop 0 v25_dgcace_bbr_scur_tau35 bdse/configs/v25_bdse_dgcace_bbr_scur_tau35_fast_cl.yaml "$V25_CKPT" &
run_open_loop 1 v25_dgcace_actionheavy_tau45 bdse/configs/v25_bdse_dgcace_actionheavy_tau45_fast_cl.yaml "$V25_CKPT" &
wait
run_open_loop 0 v25_dgcace_safety_tau30 bdse/configs/v25_bdse_dgcace_safety_tau30_fast_cl.yaml "$V25_CKPT" &
run_open_loop 1 v25_dgcace_no_fallback_ablation bdse/configs/v25_bdse_dgcace_no_fallback_ablation_fast_cl.yaml "$V25_CKPT" &
wait
print_open_loop_compare

if [[ "${OPEN_LOOP_ONLY:-0}" == "1" ]]; then
  echo "OPEN_LOOP_ONLY=1, skipping closed-loop runs."
  exit 0
fi

run_closed_loop() {
  local gpu="$1"
  local tag="$2"
  local cfg="$3"
  local ckpt="$4"
  local limit="$5"
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

run_closed_loop 0 v25_lcb_legacy_dgcace bdse/configs/v25_bdse_lcb_legacy_dgcace_fast_cl.yaml "$V25_CKPT" 20 &
run_closed_loop 1 v25_dgcace_fixed50_control bdse/configs/v25_bdse_dgcace_fixed50_control_fast_cl.yaml "$V25_CKPT" 20 &
wait
run_closed_loop 0 v25_dgcace_bbr_scur_tau35 bdse/configs/v25_bdse_dgcace_bbr_scur_tau35_fast_cl.yaml "$V25_CKPT" 20 &
run_closed_loop 1 v25_dgcace_actionheavy_tau45 bdse/configs/v25_bdse_dgcace_actionheavy_tau45_fast_cl.yaml "$V25_CKPT" 20 &
wait
run_closed_loop 0 v25_dgcace_safety_tau30 bdse/configs/v25_bdse_dgcace_safety_tau30_fast_cl.yaml "$V25_CKPT" 20 &
run_closed_loop 1 v25_dgcace_no_fallback_ablation bdse/configs/v25_bdse_dgcace_no_fallback_ablation_fast_cl.yaml "$V25_CKPT" 20 &
wait

python -m bdse.tools.collect_closed_loop_metrics \
  "$CL_ROOT/v25_lcb_legacy_dgcace_20" \
  "$CL_ROOT/v25_dgcace_fixed50_control_20" \
  "$CL_ROOT/v25_dgcace_bbr_scur_tau35_20" \
  "$CL_ROOT/v25_dgcace_actionheavy_tau45_20" \
  "$CL_ROOT/v25_dgcace_safety_tau30_20" \
  "$CL_ROOT/v25_dgcace_no_fallback_ablation_20" \
  --csv "$CL_ROOT/v25_20_compare.csv"

python - "$LOG_ROOT" <<'PY'
import json, math, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
for path in sorted(root.glob('*.closed_loop_20.diag.jsonl')):
    replans = []
    reasons = Counter()
    for line in path.read_text(errors='ignore').splitlines():
        try:
            d = json.loads(line).get('diagnostics', {})
        except Exception:
            continue
        if d.get('cached_plan'):
            continue
        replans.append(d)
        reasons[str(d.get('fallback_reason', 'missing'))] += 1
    if not replans:
        continue
    triggered = sum(1 for d in replans if d.get('fallback_triggered')) / len(replans)
    print(path.name, 'replans=', len(replans), 'fallback_triggered_rate=', round(triggered, 4), 'reasons=', dict(reasons))
PY

if [[ "${RUN_CL50:-0}" == "1" ]]; then
  run_closed_loop 0 v25_dgcace_fixed50_control bdse/configs/v25_bdse_dgcace_fixed50_control_fast_cl.yaml "$V25_CKPT" 50 &
  run_closed_loop 1 v25_dgcace_bbr_scur_tau35 bdse/configs/v25_bdse_dgcace_bbr_scur_tau35_fast_cl.yaml "$V25_CKPT" 50 &
  wait
  run_closed_loop 0 v25_dgcace_actionheavy_tau45 bdse/configs/v25_bdse_dgcace_actionheavy_tau45_fast_cl.yaml "$V25_CKPT" 50 &
  run_closed_loop 1 v25_dgcace_safety_tau30 bdse/configs/v25_bdse_dgcace_safety_tau30_fast_cl.yaml "$V25_CKPT" 50 &
  wait
  python -m bdse.tools.collect_closed_loop_metrics \
    "$CL_ROOT/v25_dgcace_fixed50_control_50" \
    "$CL_ROOT/v25_dgcace_bbr_scur_tau35_50" \
    "$CL_ROOT/v25_dgcace_actionheavy_tau45_50" \
    "$CL_ROOT/v25_dgcace_safety_tau30_50" \
    --csv "$CL_ROOT/v25_50_compare.csv"
fi

echo "Done. Key outputs:"
echo "  $TRAIN_ROOT/bdse_v25_dgcace.best.pt"
echo "  $OPEN_ROOT/open_loop_v25_*.json"
echo "  $CL_ROOT/v25_20_compare.csv"
echo "  $LOG_ROOT/*.closed_loop_20.diag.jsonl"
echo "Fast precheck: TRAIN_MAX_SCENARIOS=3000 VAL_MAX_SCENARIOS=500 TRAIN_EPOCHS=2 OPEN_LOOP_ONLY=1 bash run_v25_dgcace.sh"
echo "Reuse ckpt: SKIP_TRAIN=1 V25_CKPT=$V25_CKPT bash run_v25_dgcace.sh"
