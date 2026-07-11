#!/usr/bin/env bash
set -euo pipefail

# Run from repository root after replacing bdse/ with BDSE_optimized_v14_flipcap.zip contents.
# This script only launches <=20-scenario closed-loop jobs. 100/500-scenario jobs are intentionally not included.

export BDSE_VAL_CACHE=${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}
export NUPLAN_ROOT=${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}
export V11_CKPT=${V11_CKPT:-outputs/v11_train/bdse_v11_ta_selector.best.pt}
export CL_WORKERS_PER_RUN=${CL_WORKERS_PER_RUN:-2}
export PYTHONUNBUFFERED=1

mkdir -p outputs/open_loop outputs/closed_loop outputs/v14_logs

if [[ ! -f "$V11_CKPT" ]]; then
  echo "Missing checkpoint: $V11_CKPT" >&2
  exit 1
fi

python -m py_compile $(find bdse -name '*.py')
python -m pytest -q \
  bdse/tests/test_certificate_utility_refinement.py \
  bdse/tests/test_runtime_base_prior.py \
  bdse/tests/test_runtime_alignment_fixes.py \
  bdse/tests/test_tournament_antisymmetry.py \
  bdse/tests/test_selector_monotonicity.py \
  bdse/tests/test_family_and_safety_pair_fixes.py \
  bdse/tests/test_runtime_selector_no_teacher.py \
  bdse/tests/test_followup_training_and_closed_loop.py

eval_open_loop() {
  local gpu="$1"
  local tag="$2"
  local cfg="$3"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[open-loop] $tag on GPU $gpu"
    python -m bdse.experiments.evaluate_open_loop \
      --config "$cfg" \
      --checkpoint "$V11_CKPT" \
      --split val \
      --preprocessed-dir "$BDSE_VAL_CACHE" \
      --max-scenarios 1000 \
      --device cuda \
      --output "outputs/open_loop/open_loop_${tag}.json" \
      --per-sample-output "outputs/open_loop/open_loop_${tag}.jsonl" \
      > "outputs/v14_logs/${tag}.open_loop.out" 2>&1
  )
}

# Two A30 GPUs: two open-loop jobs at a time.
eval_open_loop 0 v14_flipcap_scur_tau20_eps0 bdse/configs/v14_bdse_flipcap_scur_tau20_eps0_fast_cl.yaml &
eval_open_loop 1 v14_flipcap_scur_tau35_eps0 bdse/configs/v14_bdse_flipcap_scur_tau35_eps0_fast_cl.yaml &
wait
eval_open_loop 0 v14_flipcap_scur_tau50_eps0 bdse/configs/v14_bdse_flipcap_scur_tau50_eps0_fast_cl.yaml &
eval_open_loop 1 v14_flipcap_selector_only_tau20_eps0 bdse/configs/v14_bdse_flipcap_selector_only_tau20_eps0_fast_cl.yaml &
wait

python - <<'PY'
import json
paths = [
    ('v11_trained', 'outputs/open_loop/open_loop_v11_ta_selector_best.json'),
    ('v12_w50', 'outputs/open_loop/open_loop_v12_rule_prior_w50.json'),
    ('v13_tau20', 'outputs/open_loop/open_loop_v13_scur_tau20.json'),
    ('v13_tau35', 'outputs/open_loop/open_loop_v13_scur_tau35.json'),
    ('v14_tau20', 'outputs/open_loop/open_loop_v14_flipcap_scur_tau20_eps0.json'),
    ('v14_tau35', 'outputs/open_loop/open_loop_v14_flipcap_scur_tau35_eps0.json'),
    ('v14_tau50', 'outputs/open_loop/open_loop_v14_flipcap_scur_tau50_eps0.json'),
    ('v14_selector_only', 'outputs/open_loop/open_loop_v14_flipcap_selector_only_tau20_eps0.json'),
]
keys = ['teacher_action_match','decision_sufficiency','budget_vs_full_match','teacher_regret','fallback_would_trigger_rate','pair_sign_acc_winner_rival','pair_sign_acc_interaction','pair_sign_acc_hard','selected_interaction_decisive_recall','selected_hard_decisive_recall','total_sparse_query_count']
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
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[closed-loop-20] $tag on GPU $gpu"
    nohup python -m bdse.experiments.evaluate_closed_loop \
      --config "$cfg" \
      --checkpoint "$V11_CKPT" \
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
      > "outputs/v14_logs/${tag}.closed_loop_20.out" 2>&1
  )
}

# At most four closed-loop jobs concurrently.
run_closed_loop_20 0 v14_flipcap_scur_tau20_eps0 bdse/configs/v14_bdse_flipcap_scur_tau20_eps0_fast_cl.yaml &
run_closed_loop_20 1 v14_flipcap_scur_tau35_eps0 bdse/configs/v14_bdse_flipcap_scur_tau35_eps0_fast_cl.yaml &
run_closed_loop_20 0 v14_flipcap_scur_tau50_eps0 bdse/configs/v14_bdse_flipcap_scur_tau50_eps0_fast_cl.yaml &
run_closed_loop_20 1 v14_flipcap_selector_only_tau20_eps0 bdse/configs/v14_bdse_flipcap_selector_only_tau20_eps0_fast_cl.yaml &
wait

python -m bdse.tools.collect_closed_loop_metrics \
  outputs/closed_loop/v10_bdse_20 \
  outputs/closed_loop/v10_hard_safety_only_20 \
  outputs/closed_loop/v11_from_v10_ta_selector_20 \
  outputs/closed_loop/v11_trained_ta_selector_20 \
  outputs/closed_loop/v12_rule_prior_w50_20 \
  outputs/closed_loop/v13_scur_tau20_20 \
  outputs/closed_loop/v13_scur_tau50_20 \
  outputs/closed_loop/external_pdm_closed_20 \
  outputs/closed_loop/external_gameformer_20 \
  outputs/closed_loop/external_dtpp_20 \
  outputs/closed_loop/external_plantf_20 \
  outputs/closed_loop/external_pluto_20 \
  outputs/closed_loop/v14_flipcap_scur_tau20_eps0_20 \
  outputs/closed_loop/v14_flipcap_scur_tau35_eps0_20 \
  outputs/closed_loop/v14_flipcap_scur_tau50_eps0_20 \
  outputs/closed_loop/v14_flipcap_selector_only_tau20_eps0_20 \
  --csv outputs/closed_loop/v14_20_compare.csv

echo "Done. Key outputs:"
echo "  outputs/open_loop/open_loop_v14_flipcap_*.json"
echo "  outputs/closed_loop/v14_flipcap_*_20"
echo "  outputs/closed_loop/v14_20_compare.csv"
