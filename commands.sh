#!/usr/bin/env bash
set -euo pipefail

# Run from repository root after replacing bdse/ with BDSE_v19_clu_actionrank.zip contents.
# v19 focuses on zero-retrain diagnosis before any slow finetune:
#   1) fixes ActionRank's certificate cap so it is boundary-aware, not legacy abs(base)+eta;
#   2) adds closed-loop utility calibrated ActionRank (CLu-ActionRank);
#   3) reserves a small decision-family quota for interaction/precedence evidence;
#   4) includes legacy/boundary LCB controls to isolate the source of gains/losses.

export BDSE_TRAIN_CACHE=${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/}
export BDSE_VAL_CACHE=${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}
export NUPLAN_ROOT=${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}
export NUPLAN_DB_ROOT=${NUPLAN_DB_ROOT:-$NUPLAN_ROOT/data/cache/val}
export V11_CKPT=${V11_CKPT:-outputs/v11_train/bdse_v11_ta_selector.best.pt}
export CL_WORKERS_PER_RUN=${CL_WORKERS_PER_RUN:-2}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p outputs/open_loop outputs/closed_loop outputs/v19_logs

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
  bdse/tests/test_v19_behavior_actionrank.py

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
      --max-scenarios 1000 \
      --device cuda \
      --output "outputs/open_loop/open_loop_${tag}.json" \
      --per-sample-output "outputs/open_loop/open_loop_${tag}.jsonl" \
      > "outputs/v19_logs/${tag}.open_loop.out" 2>&1
  )
}

print_open_loop_compare() {
  python - <<'PY'
import json
paths = [
    ('v18_lcb_control', 'outputs/open_loop/open_loop_v18_lcb_control_scur_tau35.json'),
    ('v18_true_actionrank', 'outputs/open_loop/open_loop_v18_true_actionrank_scur_tau35.json'),
    ('v19_lcb_legacy', 'outputs/open_loop/open_loop_v19_lcb_legacy_control_scur_tau35.json'),
    ('v19_lcb_boundary', 'outputs/open_loop/open_loop_v19_lcb_boundary_control_scur_tau35.json'),
    ('v19_boundary_actionrank', 'outputs/open_loop/open_loop_v19_boundary_actionrank_scur_tau35.json'),
    ('v19_clu_actionrank', 'outputs/open_loop/open_loop_v19_clu_actionrank_scur_tau35.json'),
    ('v19_clu_progress', 'outputs/open_loop/open_loop_v19_clu_actionrank_progress_tau50.json'),
    ('v19_clu_safety', 'outputs/open_loop/open_loop_v19_clu_actionrank_safety_tau30.json'),
]
keys = [
    'teacher_action_match','decision_sufficiency','budget_vs_full_match','teacher_regret',
    'fallback_would_trigger_rate','pair_sign_acc_winner_rival','pair_sign_acc_interaction','pair_sign_acc_hard',
    'selected_interaction_decisive_recall','selected_hard_decisive_recall','selected_decisive_atom_recall',
    'selector_action_rank_active','selector_lcb_active','selector_pair_atom_query_count',
    'tournament_pair_atom_query_count','total_sparse_query_count','effective_query_count'
]
for name, path in paths:
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        continue
    print('\n' + name)
    for k in keys:
        print(f'{k}: {d.get(k)}')
PY
}

# Zero-retrain open-loop diagnosis.  Legacy LCB should reproduce the old v18 control path;
# boundary LCB isolates the certificate-cap fix; CLu variants test the new closed-loop-aligned selector.
run_open_loop 0 v19_lcb_legacy_control_scur_tau35 bdse/configs/v19_bdse_lcb_legacy_control_scur_tau35_fast_cl.yaml "$V11_CKPT" &
run_open_loop 1 v19_lcb_boundary_control_scur_tau35 bdse/configs/v19_bdse_lcb_boundary_control_scur_tau35_fast_cl.yaml "$V11_CKPT" &

run_open_loop 0 v19_boundary_actionrank_scur_tau35 bdse/configs/v19_bdse_boundary_actionrank_scur_tau35_fast_cl.yaml "$V11_CKPT" &
run_open_loop 1 v19_clu_actionrank_scur_tau35 bdse/configs/v19_bdse_clu_actionrank_scur_tau35_fast_cl.yaml "$V11_CKPT" &
wait
run_open_loop 0 v19_clu_actionrank_progress_tau50 bdse/configs/v19_bdse_clu_actionrank_progress_tau50_fast_cl.yaml "$V11_CKPT" &
run_open_loop 1 v19_clu_actionrank_safety_tau30 bdse/configs/v19_bdse_clu_actionrank_safety_tau30_fast_cl.yaml "$V11_CKPT" &
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
    echo "[closed-loop-${limit}] $tag on GPU $gpu"
    python -m bdse.experiments.evaluate_closed_loop \
      --config "$cfg" \
      --checkpoint "$ckpt" \
      --device cuda \
      --challenge closed_loop_nonreactive_agents \
      --metric-aggregator closed_loop_nonreactive_agents_weighted_average \
      --output-dir "outputs/closed_loop/${tag}_${limit}" \
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
      > "outputs/v19_logs/${tag}.closed_loop_${limit}.out" 2>&1
  )
}

# Keep closed-loop at 20 first so we can test the diagnosis cheaply.
run_closed_loop 0 v19_lcb_legacy_control_scur_tau35 bdse/configs/v19_bdse_lcb_legacy_control_scur_tau35_fast_cl.yaml "$V11_CKPT" 20 &
run_closed_loop 1 v19_lcb_boundary_control_scur_tau35 bdse/configs/v19_bdse_lcb_boundary_control_scur_tau35_fast_cl.yaml "$V11_CKPT" 20 &
run_closed_loop 0 v19_boundary_actionrank_scur_tau35 bdse/configs/v19_bdse_boundary_actionrank_scur_tau35_fast_cl.yaml "$V11_CKPT" 20 &
wait
run_closed_loop 1 v19_clu_actionrank_scur_tau35 bdse/configs/v19_bdse_clu_actionrank_scur_tau35_fast_cl.yaml "$V11_CKPT" 20 &
run_closed_loop 0 v19_clu_actionrank_progress_tau50 bdse/configs/v19_bdse_clu_actionrank_progress_tau50_fast_cl.yaml "$V11_CKPT" 20 &
run_closed_loop 1 v19_clu_actionrank_safety_tau30 bdse/configs/v19_bdse_clu_actionrank_safety_tau30_fast_cl.yaml "$V11_CKPT" 20 &
wait

python -m bdse.tools.collect_closed_loop_metrics \
  outputs/closed_loop/v18_lcb_control_scur_tau35_20 \
  outputs/closed_loop/v18_true_actionrank_scur_tau35_20 \
  outputs/closed_loop/v19_lcb_legacy_control_scur_tau35_20 \
  outputs/closed_loop/v19_lcb_boundary_control_scur_tau35_20 \
  outputs/closed_loop/v19_boundary_actionrank_scur_tau35_20 \
  outputs/closed_loop/v19_clu_actionrank_scur_tau35_20 \
  outputs/closed_loop/v19_clu_actionrank_progress_tau50_20 \
  outputs/closed_loop/v19_clu_actionrank_safety_tau30_20 \
  --csv outputs/closed_loop/v19_20_compare.csv

# Optional: after the 20-scenario result identifies the best two variants, run a slightly larger cheap check.
# This defaults to the main CLu variant vs boundary LCB control; change the tags if the 20-scenario table says otherwise.
if [[ "${RUN_CL50:-0}" == "1" ]]; then
  run_closed_loop 0 v19_lcb_boundary_control_scur_tau35 bdse/configs/v19_bdse_lcb_boundary_control_scur_tau35_fast_cl.yaml "$V11_CKPT" 50 &
  run_closed_loop 1 v19_clu_actionrank_scur_tau35 bdse/configs/v19_bdse_clu_actionrank_scur_tau35_fast_cl.yaml "$V11_CKPT" 50 &
  wait
  python -m bdse.tools.collect_closed_loop_metrics \
    outputs/closed_loop/v19_lcb_boundary_control_scur_tau35_50 \
    outputs/closed_loop/v19_clu_actionrank_scur_tau35_50 \
    --csv outputs/closed_loop/v19_50_compare.csv
fi

echo "Done. Key outputs:"
echo "  outputs/open_loop/open_loop_v19_*.json"
echo "  outputs/open_loop/open_loop_v19_*.jsonl"
echo "  outputs/closed_loop/v19_*_20"
echo "  outputs/closed_loop/v19_20_compare.csv"
echo "Optional larger check: RUN_CL50=1 bash run_v19_clu_actionrank_bdse.sh"
