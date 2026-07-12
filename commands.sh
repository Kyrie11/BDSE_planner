#!/usr/bin/env bash
set -euo pipefail

# Run from repository root after replacing bdse/ with BDSE_optimized_v16_anchor_smooth.zip contents.
# This script only launches <=20-scenario closed-loop jobs. 100/500-scenario jobs are intentionally not included.
# Set RUN_FINETUNE=1 only after the zero-retrain v16 run shows improved open-loop/20-scenario signals.

export BDSE_TRAIN_CACHE=${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/}
export BDSE_VAL_CACHE=${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}
export NUPLAN_ROOT=${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}
export V11_CKPT=${V11_CKPT:-outputs/v11_train/bdse_v11_ta_selector.best.pt}
export V16_CKPT=${V16_CKPT:-outputs/v16_train/bdse_v16_anchor_smooth.best.pt}
export CL_WORKERS_PER_RUN=${CL_WORKERS_PER_RUN:-2}
export PYTHONUNBUFFERED=1
# Avoid CUDA memory fragmentation during closed-loop runs.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p outputs/open_loop outputs/closed_loop outputs/v16_logs outputs/v16_train

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
      > "outputs/v16_logs/${tag}.open_loop.out" 2>&1
  )
}

# Zero-retrain test using the trained v11 checkpoint. Two GPUs: two jobs at a time.
run_open_loop 0 v16_anchor_smooth_selector_keep_prior bdse/configs/v16_bdse_anchor_smooth_selector_keep_prior_fast_cl.yaml "$V11_CKPT" &
run_open_loop 1 v16_anchor_smooth_scur_tau35 bdse/configs/v16_bdse_anchor_smooth_scur_tau35_fast_cl.yaml "$V11_CKPT" &
wait
run_open_loop 0 v16_anchor_smooth_progress_tau50 bdse/configs/v16_bdse_anchor_smooth_progress_tau50_fast_cl.yaml "$V11_CKPT" &
run_open_loop 1 v16_anchor_smooth_safety_tau35 bdse/configs/v16_bdse_anchor_smooth_safety_tau35_fast_cl.yaml "$V11_CKPT" &
wait

python - <<'PY'
import json
paths = [
    ('v11_trained', 'outputs/open_loop/open_loop_v11_ta_selector_best.json'),
    ('v12_w50', 'outputs/open_loop/open_loop_v12_rule_prior_w50.json'),
    ('v14_tau20', 'outputs/open_loop/open_loop_v14_flipcap_scur_tau20_eps0.json'),
    ('v15_scur_tau35', 'outputs/open_loop/open_loop_v15_fliprank_scur_tau35.json'),
    ('v15_finetuned', 'outputs/open_loop/open_loop_v15_finetuned_fliprank_scur_tau35.json'),
    ('v16_selector', 'outputs/open_loop/open_loop_v16_anchor_smooth_selector_keep_prior.json'),
    ('v16_scur_tau35', 'outputs/open_loop/open_loop_v16_anchor_smooth_scur_tau35.json'),
    ('v16_progress_tau50', 'outputs/open_loop/open_loop_v16_anchor_smooth_progress_tau50.json'),
    ('v16_safety_tau35', 'outputs/open_loop/open_loop_v16_anchor_smooth_safety_tau35.json'),
]
keys = ['teacher_action_match','decision_sufficiency','budget_vs_full_match','teacher_regret','fallback_would_trigger_rate','pair_sign_acc_winner_rival','pair_sign_acc_interaction','pair_sign_acc_hard','selected_interaction_decisive_recall','selected_hard_decisive_recall','selected_decisive_atom_recall','selector_pair_count','runtime_pair_count','total_sparse_query_count']
for name, path in paths:
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        continue
    print('\n' + name)
    for k in keys:
        print(f'{k}: {d.get(k)}')
PY

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
      > "outputs/v16_logs/${tag}.closed_loop_20.out" 2>&1
  )
}

# v15 OOM was caused by launching multiple backgrounded inner nohup jobs. Here each job blocks within its subshell.
# Run at most two closed-loop jobs concurrently: one per A30. This is safer than four jobs for pair-conditioned BDSE.
run_closed_loop_20 0 v16_anchor_smooth_selector_keep_prior bdse/configs/v16_bdse_anchor_smooth_selector_keep_prior_fast_cl.yaml "$V11_CKPT" &
run_closed_loop_20 1 v16_anchor_smooth_scur_tau35 bdse/configs/v16_bdse_anchor_smooth_scur_tau35_fast_cl.yaml "$V11_CKPT" &
wait
run_closed_loop_20 0 v16_anchor_smooth_progress_tau50 bdse/configs/v16_bdse_anchor_smooth_progress_tau50_fast_cl.yaml "$V11_CKPT" &
run_closed_loop_20 1 v16_anchor_smooth_safety_tau35 bdse/configs/v16_bdse_anchor_smooth_safety_tau35_fast_cl.yaml "$V11_CKPT" &
wait

python -m bdse.tools.collect_closed_loop_metrics \
  outputs/closed_loop/v10_bdse_20 \
  outputs/closed_loop/v10_hard_safety_only_20 \
  outputs/closed_loop/v11_from_v10_ta_selector_20 \
  outputs/closed_loop/v11_trained_ta_selector_20 \
  outputs/closed_loop/v12_rule_prior_w50_20 \
  outputs/closed_loop/v13_scur_tau50_20 \
  outputs/closed_loop/v14_flipcap_scur_tau20_eps0_20 \
  outputs/closed_loop/v15_finetuned_fliprank_scur_tau35_20 \
  outputs/closed_loop/external_pdm_closed_20 \
  outputs/closed_loop/external_gameformer_20 \
  outputs/closed_loop/external_dtpp_20 \
  outputs/closed_loop/external_plantf_20 \
  outputs/closed_loop/external_pluto_20 \
  outputs/closed_loop/v16_anchor_smooth_selector_keep_prior_20 \
  outputs/closed_loop/v16_anchor_smooth_scur_tau35_20 \
  outputs/closed_loop/v16_anchor_smooth_progress_tau50_20 \
  outputs/closed_loop/v16_anchor_smooth_safety_tau35_20 \
  --csv outputs/closed_loop/v16_20_compare.csv

if [[ "${RUN_FINETUNE:-0}" == "1" ]]; then
  echo "[finetune] continuing from $V11_CKPT with v16 anchor-smooth selector objective"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m bdse.experiments.train \
    --config bdse/configs/v16_bdse_anchor_smooth_train.yaml \
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
    --epochs 15 \
    --resume-from "$V11_CKPT" \
    --save-every-n-epochs 0 \
    --log-file outputs/v16_train/bdse_v16_anchor_smooth.train_log.jsonl \
    --output outputs/v16_train/bdse_v16_anchor_smooth.pt \
    > outputs/v16_train/bdse_v16_anchor_smooth.detached.out 2>&1

  run_open_loop 0 v16_finetuned_anchor_smooth_scur_tau35 bdse/configs/v16_bdse_anchor_smooth_scur_tau35_fast_cl.yaml "$V16_CKPT"
  run_closed_loop_20 0 v16_finetuned_anchor_smooth_scur_tau35 bdse/configs/v16_bdse_anchor_smooth_scur_tau35_fast_cl.yaml "$V16_CKPT"
fi

echo "Done. Key outputs:"
echo "  outputs/open_loop/open_loop_v16_*.json"
echo "  outputs/closed_loop/v16_*_20"
echo "  outputs/closed_loop/v16_20_compare.csv"
echo "Optional finetune only after zero-retrain improves: RUN_FINETUNE=1 bash run_v16_anchor_smooth_bdse.sh"
