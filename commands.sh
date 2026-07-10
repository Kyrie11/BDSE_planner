#!/usr/bin/env bash
set -euo pipefail

# Run from repository root after replacing bdse/ with BDSE_optimized_v12.zip contents.
# This script only runs <=20-scenario closed-loop jobs. Larger 100/500 jobs are intentionally not included.

export BDSE_VAL_CACHE=${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/}
export NUPLAN_ROOT=${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}
export V11_CKPT=${V11_CKPT:-outputs/v11_train/bdse_v11_ta_selector.best.pt}
export CL_WORKERS_PER_RUN=${CL_WORKERS_PER_RUN:-2}
export PYTHONUNBUFFERED=1

mkdir -p outputs/open_loop outputs/calibration outputs/closed_loop outputs/v12_logs

if [[ ! -f "$V11_CKPT" ]]; then
  echo "Missing checkpoint: $V11_CKPT" >&2
  exit 1
fi

python -m py_compile $(find bdse -name '*.py')
python -m pytest -q \
  bdse/tests/test_runtime_base_prior.py \
  bdse/tests/test_runtime_alignment_fixes.py \
  bdse/tests/test_tournament_antisymmetry.py \
  bdse/tests/test_selector_monotonicity.py \
  bdse/tests/test_family_and_safety_pair_fixes.py \
  bdse/tests/test_runtime_selector_no_teacher.py \
  bdse/tests/test_followup_training_and_closed_loop.py

patch_epsilon() {
  local cfg="$1"
  local cal_json="$2"
  python - "$cfg" "$cal_json" <<'PY'
import json, pathlib, sys
cfg_path = pathlib.Path(sys.argv[1])
cal = json.load(open(sys.argv[2]))
eps_raw = float(cal.get('epsilon_cal', cal.get('epsilon_cal_safety', 0.0)))
eps = min(eps_raw, 1.0)
lines = cfg_path.read_text().splitlines()
in_tournament = False
replaced = False
for i, line in enumerate(lines):
    if line.strip() == 'tournament:':
        in_tournament = True
        continue
    if in_tournament and line and not line.startswith(' '):
        if not replaced:
            lines.insert(i, f'  epsilon_cal: {eps}')
            replaced = True
        in_tournament = False
    if in_tournament and line.strip().startswith('epsilon_cal:'):
        lines[i] = f'  epsilon_cal: {eps}'
        replaced = True
        break
if in_tournament and not replaced:
    lines.append(f'  epsilon_cal: {eps}')
cfg_path.write_text('\n'.join(lines) + '\n')
print(f'{cfg_path}: epsilon_cal_raw={eps_raw} epsilon_cal_capped={eps}')
PY
}

eval_and_calibrate() {
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
      > "outputs/v12_logs/${tag}.open_loop.out" 2>&1

    echo "[calibration] $tag on GPU $gpu"
    python -m bdse.experiments.calibrate \
      --config "$cfg" \
      --checkpoint "$V11_CKPT" \
      --split val \
      --preprocessed-dir "$BDSE_VAL_CACHE" \
      --max-scenarios 1000 \
      --device cuda \
      --delta 0.1 \
      --output "outputs/calibration/calibration_${tag}.json" \
      > "outputs/v12_logs/${tag}.calibration.out" 2>&1
    patch_epsilon "$cfg" "outputs/calibration/calibration_${tag}.json"
  )
}

# Two A30 GPUs: run two eval/calibration jobs at a time.
eval_and_calibrate 0 v12_rule_prior_w25 bdse/configs/v12_bdse_rule_prior_w25_fast_cl.yaml &
eval_and_calibrate 1 v12_rule_prior_w50 bdse/configs/v12_bdse_rule_prior_w50_fast_cl.yaml &
wait
eval_and_calibrate 0 v12_rule_prior_w75 bdse/configs/v12_bdse_rule_prior_w75_fast_cl.yaml &
eval_and_calibrate 1 v12_rule_prior_w50_no_guard bdse/configs/v12_bdse_rule_prior_w50_no_guard_fast_cl.yaml &
wait

python - <<'PY'
import json
paths = [
    ('v11_trained', 'outputs/open_loop/open_loop_v11_ta_selector_best.json'),
    ('v12_w25', 'outputs/open_loop/open_loop_v12_rule_prior_w25.json'),
    ('v12_w50', 'outputs/open_loop/open_loop_v12_rule_prior_w50.json'),
    ('v12_w75', 'outputs/open_loop/open_loop_v12_rule_prior_w75.json'),
    ('v12_w50_no_guard', 'outputs/open_loop/open_loop_v12_rule_prior_w50_no_guard.json'),
]
keys = ['teacher_action_match','decision_sufficiency','budget_vs_full_match','teacher_regret','fallback_would_trigger_rate','pair_sign_acc_winner_rival','selected_interaction_decisive_recall','selected_hard_decisive_recall','total_sparse_query_count']
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
    python -m bdse.experiments.evaluate_closed_loop \
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
      > "outputs/v12_logs/${tag}.closed_loop_20.out" 2>&1
  )
}

# Up to four closed-loop jobs at once, as requested.
run_closed_loop_20 0 v12_rule_prior_w25 bdse/configs/v12_bdse_rule_prior_w25_fast_cl.yaml &
run_closed_loop_20 1 v12_rule_prior_w50 bdse/configs/v12_bdse_rule_prior_w50_fast_cl.yaml &
run_closed_loop_20 0 v12_rule_prior_w75 bdse/configs/v12_bdse_rule_prior_w75_fast_cl.yaml &
run_closed_loop_20 1 v12_rule_prior_w50_no_guard bdse/configs/v12_bdse_rule_prior_w50_no_guard_fast_cl.yaml &
wait

python -m bdse.tools.collect_closed_loop_metrics \
  outputs/closed_loop/v10_bdse_20 \
  outputs/closed_loop/v10_hard_safety_only_20 \
  outputs/closed_loop/v11_from_v10_ta_selector_20 \
  outputs/closed_loop/v11_trained_ta_selector_20 \
  outputs/closed_loop/external_pdm_closed_20 \
  outputs/closed_loop/external_gameformer_20 \
  outputs/closed_loop/external_dtpp_20 \
  outputs/closed_loop/external_plantf_20 \
  outputs/closed_loop/external_pluto_20 \
  outputs/closed_loop/v12_rule_prior_w25_20 \
  outputs/closed_loop/v12_rule_prior_w50_20 \
  outputs/closed_loop/v12_rule_prior_w75_20 \
  outputs/closed_loop/v12_rule_prior_w50_no_guard_20 \
  --csv outputs/closed_loop/v12_20_compare.csv

echo "Done. Key outputs:"
echo "  outputs/open_loop/open_loop_v12_rule_prior_*.json"
echo "  outputs/calibration/calibration_v12_rule_prior_*.json"
echo "  outputs/closed_loop/v12_*_20"
echo "  outputs/closed_loop/v12_20_compare.csv"
