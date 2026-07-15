#!/usr/bin/env bash
set -euo pipefail

# Run from repository root after replacing bdse/ with BDSE_v22_adaptive_safety_seed.zip contents.
# v22 goal:
#   1) keep the v21 safety-gated LCB->ActionRank hybrid, but make the LCB seed budget adaptive;
#   2) increase LCB seed when safety/fallback/uncertainty risk is high;
#   3) reserve more ActionRank budget when near-boundary, low-safety pairs are dense;
#   4) add a decision-family gain boost in ActionRank refinement to better retain interaction evidence;
#   5) flatten nuPlan closed-loop output paths by using absolute output/log paths.

ROOT_DIR="$(pwd)"
export BDSE_TRAIN_CACHE=${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/}
export BDSE_VAL_CACHE=${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}
export NUPLAN_ROOT=${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}
export NUPLAN_DB_ROOT=${NUPLAN_DB_ROOT:-$NUPLAN_ROOT/data/cache/val}
export V11_CKPT=${V11_CKPT:-outputs/v11_train/bdse_v11_ta_selector.best.pt}
export CL_WORKERS_PER_RUN=${CL_WORKERS_PER_RUN:-2}
export BDSE_REPLAN_INTERVAL_TICKS=${BDSE_REPLAN_INTERVAL_TICKS:-5}
export BDSE_PROFILE_CLOSED_LOOP=${BDSE_PROFILE_CLOSED_LOOP:-1}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/outputs_v22}
OPEN_ROOT="$OUT_ROOT/open_loop"
CL_ROOT="$OUT_ROOT/closed_loop"
LOG_ROOT="$OUT_ROOT/v22_logs"
mkdir -p "$OPEN_ROOT" "$CL_ROOT" "$LOG_ROOT"

if [[ ! -f "$V11_CKPT" ]]; then
  echo "Missing checkpoint: $V11_CKPT" >&2
  exit 1
fi

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
    ('v22_lcb_legacy_replan5', root / 'open_loop_v22_lcb_legacy_replan5.json'),
    ('v22_adaptive_hybrid_scur_tau35', root / 'open_loop_v22_adaptive_hybrid_scur_tau35.json'),
    ('v22_adaptive_hybrid_safety_tau30', root / 'open_loop_v22_adaptive_hybrid_safety_tau30.json'),
    ('v22_adaptive_hybrid_progress_tau50', root / 'open_loop_v22_adaptive_hybrid_progress_tau50.json'),
    ('v22_fixed55_familyboost_ablation', root / 'open_loop_v22_fixed55_familyboost_ablation.json'),
]
keys = [
    'teacher_action_match','decision_sufficiency','budget_vs_full_match','teacher_regret',
    'fallback_would_trigger_rate','pair_sign_acc_winner_rival','pair_sign_acc_interaction','pair_sign_acc_hard',
    'selected_interaction_decisive_recall','selected_hard_decisive_recall','selected_decisive_atom_recall',
    'selector_hybrid_lcb_action_rank_active','selector_adaptive_lcb_frac','selector_adaptive_lcb_raw_frac',
    'selector_adaptive_safety_density','selector_adaptive_fallback_risk','selector_adaptive_boundary_density',
    'selector_adaptive_action_need','selector_hybrid_lcb_seed_atoms','selector_hybrid_action_atoms',
    'selector_decision_family_selected','selector_decision_family_boost',
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

run_open_loop 0 v22_lcb_legacy_replan5 bdse/configs/v22_bdse_lcb_legacy_replan5_fast_cl.yaml "$V11_CKPT" &
run_open_loop 1 v22_adaptive_hybrid_scur_tau35 bdse/configs/v22_bdse_adaptive_hybrid_scur_tau35_fast_cl.yaml "$V11_CKPT" &
wait
run_open_loop 0 v22_adaptive_hybrid_safety_tau30 bdse/configs/v22_bdse_adaptive_hybrid_safety_tau30_fast_cl.yaml "$V11_CKPT" &
run_open_loop 1 v22_adaptive_hybrid_progress_tau50 bdse/configs/v22_bdse_adaptive_hybrid_progress_tau50_fast_cl.yaml "$V11_CKPT" &
wait
run_open_loop 0 v22_fixed55_familyboost_ablation bdse/configs/v22_bdse_fixed55_familyboost_ablation_fast_cl.yaml "$V11_CKPT" &
wait
print_open_loop_compare

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

run_closed_loop 0 v22_lcb_legacy_replan5 bdse/configs/v22_bdse_lcb_legacy_replan5_fast_cl.yaml "$V11_CKPT" 20 &
run_closed_loop 1 v22_adaptive_hybrid_scur_tau35 bdse/configs/v22_bdse_adaptive_hybrid_scur_tau35_fast_cl.yaml "$V11_CKPT" 20 &
wait
run_closed_loop 0 v22_adaptive_hybrid_safety_tau30 bdse/configs/v22_bdse_adaptive_hybrid_safety_tau30_fast_cl.yaml "$V11_CKPT" 20 &
run_closed_loop 1 v22_adaptive_hybrid_progress_tau50 bdse/configs/v22_bdse_adaptive_hybrid_progress_tau50_fast_cl.yaml "$V11_CKPT" 20 &
wait
run_closed_loop 0 v22_fixed55_familyboost_ablation bdse/configs/v22_bdse_fixed55_familyboost_ablation_fast_cl.yaml "$V11_CKPT" 20 &
wait

python -m bdse.tools.collect_closed_loop_metrics \
  "$CL_ROOT/v22_lcb_legacy_replan5_20" \
  "$CL_ROOT/v22_adaptive_hybrid_scur_tau35_20" \
  "$CL_ROOT/v22_adaptive_hybrid_safety_tau30_20" \
  "$CL_ROOT/v22_adaptive_hybrid_progress_tau50_20" \
  "$CL_ROOT/v22_fixed55_familyboost_ablation_20" \
  --csv "$CL_ROOT/v22_20_compare.csv"

if [[ "${RUN_REPLAN8:-0}" == "1" ]]; then
  run_closed_loop 0 v22_adaptive_hybrid_scur_tau35_replan8 bdse/configs/v22_bdse_adaptive_hybrid_scur_tau35_replan8_fast_cl.yaml "$V11_CKPT" 20
  python -m bdse.tools.collect_closed_loop_metrics \
    "$CL_ROOT/v22_adaptive_hybrid_scur_tau35_replan8_20" \
    --csv "$CL_ROOT/v22_replan8_20_compare.csv"
fi

if [[ "${RUN_CL50:-0}" == "1" ]]; then
  run_closed_loop 0 v22_lcb_legacy_replan5 bdse/configs/v22_bdse_lcb_legacy_replan5_fast_cl.yaml "$V11_CKPT" 50 &
  run_closed_loop 1 v22_adaptive_hybrid_scur_tau35 bdse/configs/v22_bdse_adaptive_hybrid_scur_tau35_fast_cl.yaml "$V11_CKPT" 50 &
  wait
  run_closed_loop 0 v22_adaptive_hybrid_safety_tau30 bdse/configs/v22_bdse_adaptive_hybrid_safety_tau30_fast_cl.yaml "$V11_CKPT" 50 &
  run_closed_loop 1 v22_adaptive_hybrid_progress_tau50 bdse/configs/v22_bdse_adaptive_hybrid_progress_tau50_fast_cl.yaml "$V11_CKPT" 50 &
  wait
  python -m bdse.tools.collect_closed_loop_metrics \
    "$CL_ROOT/v22_lcb_legacy_replan5_50" \
    "$CL_ROOT/v22_adaptive_hybrid_scur_tau35_50" \
    "$CL_ROOT/v22_adaptive_hybrid_safety_tau30_50" \
    "$CL_ROOT/v22_adaptive_hybrid_progress_tau50_50" \
    --csv "$CL_ROOT/v22_50_compare.csv"
fi

if [[ "${RUN_CL100:-0}" == "1" ]]; then
  run_closed_loop 0 v22_lcb_legacy_replan5 bdse/configs/v22_bdse_lcb_legacy_replan5_fast_cl.yaml "$V11_CKPT" 100 &
  run_closed_loop 1 v22_adaptive_hybrid_scur_tau35 bdse/configs/v22_bdse_adaptive_hybrid_scur_tau35_fast_cl.yaml "$V11_CKPT" 100 &
  wait
  run_closed_loop 0 v22_adaptive_hybrid_safety_tau30 bdse/configs/v22_bdse_adaptive_hybrid_safety_tau30_fast_cl.yaml "$V11_CKPT" 100 &
  run_closed_loop 1 v22_adaptive_hybrid_progress_tau50 bdse/configs/v22_bdse_adaptive_hybrid_progress_tau50_fast_cl.yaml "$V11_CKPT" 100 &
  wait
  python -m bdse.tools.collect_closed_loop_metrics \
    "$CL_ROOT/v22_lcb_legacy_replan5_100" \
    "$CL_ROOT/v22_adaptive_hybrid_scur_tau35_100" \
    "$CL_ROOT/v22_adaptive_hybrid_safety_tau30_100" \
    "$CL_ROOT/v22_adaptive_hybrid_progress_tau50_100" \
    --csv "$CL_ROOT/v22_100_compare.csv"
fi

echo "Done. Key outputs:"
echo "  $OPEN_ROOT/open_loop_v22_*.json"
echo "  $CL_ROOT/v22_20_compare.csv"
echo "  $LOG_ROOT/*.diag.jsonl"
echo "  Closed-loop folders are now rooted at: $CL_ROOT/<tag>_<limit>/closed_loop_nonreactive_agents"
echo "Optional: RUN_REPLAN8=1 bash run_v22_adaptive_safety_seed.sh ; RUN_CL50=1 bash run_v22_adaptive_safety_seed.sh ; RUN_CL100=1 bash run_v22_adaptive_safety_seed.sh"
