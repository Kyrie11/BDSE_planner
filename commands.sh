#!/usr/bin/env bash
set -euo pipefail

# Run from repository root after replacing bdse/ with BDSE_optimized_v18_true_actionrank.zip contents.
# v18 fixes two issues:
#   1) true ActionRank dispatch was bypassed by pair_atom_variance/family caps in v15-v17;
#   2) finetuning from v11 best must warm-start weights, not resume epoch=30.

export BDSE_TRAIN_CACHE=${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/}
export BDSE_VAL_CACHE=${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}
export NUPLAN_ROOT=${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}
export V11_CKPT=${V11_CKPT:-outputs/v11_train/bdse_v11_ta_selector.best.pt}
export V18_CKPT=${V18_CKPT:-outputs/v18_train/bdse_v18_true_actionrank.best.pt}
export CL_WORKERS_PER_RUN=${CL_WORKERS_PER_RUN:-2}
export FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-5}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p outputs/open_loop outputs/closed_loop outputs/v18_logs outputs/v18_train

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
  bdse/tests/test_followup_training_and_closed_loop.py

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
      > "outputs/v18_logs/${tag}.open_loop.out" 2>&1
  )
}

print_open_loop_compare() {
  python - <<'PY'
import json
paths = [
    ('v11_trained', 'outputs/open_loop/open_loop_v11_ta_selector_best.json'),
    ('v12_w50', 'outputs/open_loop/open_loop_v12_rule_prior_w50.json'),
    ('v13_tau50', 'outputs/open_loop/open_loop_v13_scur_tau50.json'),
    ('v16_safety_tau35', 'outputs/open_loop/open_loop_v16_anchor_smooth_safety_tau35.json'),
    ('v17_selector', 'outputs/open_loop/open_loop_v17_actionrank_selector_keep_prior.json'),
    ('v17_scur_tau35', 'outputs/open_loop/open_loop_v17_actionrank_scur_tau35.json'),
    ('v17_safety_tau35', 'outputs/open_loop/open_loop_v17_actionrank_safety_tau35.json'),
    ('v18_true_selector', 'outputs/open_loop/open_loop_v18_true_actionrank_selector_keep_prior.json'),
    ('v18_true_scur', 'outputs/open_loop/open_loop_v18_true_actionrank_scur_tau35.json'),
    ('v18_true_progress', 'outputs/open_loop/open_loop_v18_true_actionrank_progress_tau50.json'),
    ('v18_true_safety', 'outputs/open_loop/open_loop_v18_true_actionrank_safety_tau35.json'),
    ('v18_lcb_control', 'outputs/open_loop/open_loop_v18_lcb_control_scur_tau35.json'),
    ('v18_finetuned', 'outputs/open_loop/open_loop_v18_finetuned_true_actionrank_scur_tau35.json'),
]
keys = ['teacher_action_match','decision_sufficiency','budget_vs_full_match','teacher_regret','fallback_would_trigger_rate','pair_sign_acc_winner_rival','pair_sign_acc_interaction','pair_sign_acc_hard','selected_interaction_decisive_recall','selected_hard_decisive_recall','selected_decisive_atom_recall','selector_action_rank_active','selector_lcb_active','selector_pair_atom_query_count','tournament_pair_atom_query_count','total_sparse_query_count','effective_query_count']
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

# Zero-retrain evaluation on the fixed v11 checkpoint.
run_open_loop 0 v18_true_actionrank_selector_keep_prior bdse/configs/v18_bdse_true_actionrank_selector_keep_prior_fast_cl.yaml "$V11_CKPT" &
run_open_loop 1 v18_true_actionrank_scur_tau35 bdse/configs/v18_bdse_true_actionrank_scur_tau35_fast_cl.yaml "$V11_CKPT" &
wait
run_open_loop 0 v18_true_actionrank_progress_tau50 bdse/configs/v18_bdse_true_actionrank_progress_tau50_fast_cl.yaml "$V11_CKPT" &
run_open_loop 1 v18_true_actionrank_safety_tau35 bdse/configs/v18_bdse_true_actionrank_safety_tau35_fast_cl.yaml "$V11_CKPT" &
wait
# Control: explicitly force the old LCB/uncertainty objective to verify v17's accidental path.
run_open_loop 0 v18_lcb_control_scur_tau35 bdse/configs/v18_bdse_lcb_control_scur_tau35_fast_cl.yaml "$V11_CKPT"
print_open_loop_compare

run_closed_loop_20() {
  local gpu="$1"
  local tag="$2"
  local cfg="$3"
  local ckpt="$4"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[closed-loop-20] $tag on GPU $gpu"
    python -m bdse.experiments.evaluate_closed_loop \
      --config "$cfg" \
      --checkpoint "$ckpt" \
      --device cuda \
      --challenge closed_loop_nonreactive_agents \
      --metric-aggregator closed_loop_nonreactive_agents_weighted_average \
      --output-dir "outputs/closed_loop/${tag}_20" \
      --experiment-uid "bdse_${tag}_20" \
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
      worker.max_workers="$CL_WORKERS_PER_RUN" \
      run_metric=true \
      > "outputs/v18_logs/${tag}.closed_loop_20.out" 2>&1
  )
}

# Run at most two closed-loop jobs concurrently: one per visible GPU.
run_closed_loop_20 0 v18_true_actionrank_selector_keep_prior bdse/configs/v18_bdse_true_actionrank_selector_keep_prior_fast_cl.yaml "$V11_CKPT" &
run_closed_loop_20 1 v18_true_actionrank_scur_tau35 bdse/configs/v18_bdse_true_actionrank_scur_tau35_fast_cl.yaml "$V11_CKPT" &
wait
run_closed_loop_20 0 v18_true_actionrank_progress_tau50 bdse/configs/v18_bdse_true_actionrank_progress_tau50_fast_cl.yaml "$V11_CKPT" &
run_closed_loop_20 1 v18_true_actionrank_safety_tau35 bdse/configs/v18_bdse_true_actionrank_safety_tau35_fast_cl.yaml "$V11_CKPT" &
wait
run_closed_loop_20 0 v18_lcb_control_scur_tau35 bdse/configs/v18_bdse_lcb_control_scur_tau35_fast_cl.yaml "$V11_CKPT"

python -m bdse.tools.collect_closed_loop_metrics \
  outputs/closed_loop/v12_rule_prior_w50_20 \
  outputs/closed_loop/v13_scur_tau50_20 \
  outputs/closed_loop/v16_anchor_smooth_safety_tau35_20 \
  outputs/closed_loop/v17_actionrank_selector_keep_prior_20 \
  outputs/closed_loop/v17_actionrank_scur_tau35_20 \
  outputs/closed_loop/v17_actionrank_progress_tau50_20 \
  outputs/closed_loop/v17_actionrank_safety_tau35_20 \
  outputs/closed_loop/v18_true_actionrank_selector_keep_prior_20 \
  outputs/closed_loop/v18_true_actionrank_scur_tau35_20 \
  outputs/closed_loop/v18_true_actionrank_progress_tau50_20 \
  outputs/closed_loop/v18_true_actionrank_safety_tau35_20 \
  outputs/closed_loop/v18_lcb_control_scur_tau35_20 \
  --csv outputs/closed_loop/v18_20_compare.csv

if [[ "${RUN_FINETUNE:-0}" == "1" ]]; then
  echo "[finetune] warm-starting from $V11_CKPT with v18 true ActionRank selector objective for $FINETUNE_EPOCHS epochs"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m bdse.experiments.train \
    --config bdse/configs/v18_bdse_true_actionrank_train.yaml \
    --split train_boston train_pittsburgh train_singapore train_vegas_2 \
    --preprocessed-dir "$BDSE_TRAIN_CACHE" \
    --max-scenarios 50000 \
    --max-scenarios-per-split 12500 \
    --batch-size 16 \
    --num-workers 12 \
    --prefetch-factor 1 \
    --device cuda \
    --amp \
    --val-preprocessed-dir "$BDSE_VAL_CACHE" \
    --val-split val \
    --val-max-scenarios 1000 \
    --val-mode loss \
    --val-every-n-epochs 1 \
    --best-metric val_loss \
    --best-metrics val_loss teacher_action_match full_interface_action_match teacher_regret pair_sign_acc_interaction pair_sign_acc_hard pair_sign_acc_winner_rival selected_interaction_decisive_recall selected_hard_decisive_recall budget_vs_full_match fallback_would_trigger_rate \
    --epochs "$FINETUNE_EPOCHS" \
    --warm-start-from "$V11_CKPT" \
    --save-every-n-epochs 0 \
    --log-file outputs/v18_train/bdse_v18_true_actionrank.train_log.jsonl \
    --output outputs/v18_train/bdse_v18_true_actionrank.pt \
    > outputs/v18_train/bdse_v18_true_actionrank.detached.out 2>&1

  run_open_loop 0 v18_finetuned_true_actionrank_scur_tau35 bdse/configs/v18_bdse_true_actionrank_scur_tau35_fast_cl.yaml "$V18_CKPT"
  run_closed_loop_20 0 v18_finetuned_true_actionrank_scur_tau35 bdse/configs/v18_bdse_true_actionrank_scur_tau35_fast_cl.yaml "$V18_CKPT"

  python -m bdse.tools.collect_closed_loop_metrics \
    outputs/closed_loop/v13_scur_tau50_20 \
    outputs/closed_loop/v17_actionrank_scur_tau35_20 \
    outputs/closed_loop/v18_true_actionrank_scur_tau35_20 \
    outputs/closed_loop/v18_finetuned_true_actionrank_scur_tau35_20 \
    --csv outputs/closed_loop/v18_finetune_20_compare.csv
  print_open_loop_compare
fi

echo "Done. Key outputs:"
echo "  outputs/open_loop/open_loop_v18_*.json"
echo "  outputs/closed_loop/v18_*_20"
echo "  outputs/closed_loop/v18_20_compare.csv"
echo "Optional real finetune: RUN_FINETUNE=1 FINETUNE_EPOCHS=5 bash run_v18_true_actionrank_bdse.sh"
