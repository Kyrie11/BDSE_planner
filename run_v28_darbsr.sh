#!/usr/bin/env bash
set -euo pipefail

# Accept both `SKIP_TRAIN=1 bash run_v28_darbsr.sh` and
# `bash run_v28_darbsr.sh SKIP_TRAIN=1` to avoid accidental training.
for kv in "$@"; do
  if [[ "$kv" == *=* ]]; then export "$kv"; fi
done


# v28 / DA-RBSR: Dual-tier Adaptive Risk-Bounded Safety Recovery
# Key changes over v25 DG-CACE:
#   1) runtime-flagged actions are hard-masked in tournament whenever an unflagged candidate exists;
#   2) low normalized delta no longer triggers fallback by itself;
#   3) fallback is fixed-budget by default: no expanded evidence stage, only safe-progress rule recovery;
#   4) optional finetune keeps certificate losses but uses tiered runtime safety flags.

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

OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/outputs_v28}
TRAIN_ROOT="$OUT_ROOT/train"
OPEN_ROOT="$OUT_ROOT/open_loop"
CL_ROOT="$OUT_ROOT/closed_loop"
LOG_ROOT="$OUT_ROOT/v28_logs"
mkdir -p "$TRAIN_ROOT" "$OPEN_ROOT" "$CL_ROOT" "$LOG_ROOT"

# Optional unambiguous stage selector. This avoids relying on ambiguous legacy toggles.
#   RUN_MODE=open_loop : open-loop only
#   RUN_MODE=cl20      : closed-loop 20 only, skip open-loop by default
#   RUN_MODE=cl50      : closed-loop 50 only, skip open-loop by default
#   RUN_MODE=all       : open-loop + CL20, and CL50 if RUN_CL50=1
RUN_MODE=${RUN_MODE:-all}
case "$RUN_MODE" in
  open_loop) export OPEN_LOOP_ONLY=1; export RUN_CL20=0; export RUN_CL50=0 ;;
  cl20) export SKIP_OPEN_LOOP=${SKIP_OPEN_LOOP:-1}; export RUN_CL20=1; export RUN_CL50=0 ;;
  cl50) export SKIP_OPEN_LOOP=${SKIP_OPEN_LOOP:-1}; export RUN_CL20=0; export RUN_CL50=1 ;;
  all) ;;
  *) echo "Unknown RUN_MODE=$RUN_MODE. Use all|open_loop|cl20|cl50." >&2; exit 2 ;;
esac
echo "[run-mode] RUN_MODE=$RUN_MODE SKIP_OPEN_LOOP=${SKIP_OPEN_LOOP:-0} OPEN_LOOP_ONLY=${OPEN_LOOP_ONLY:-0} RUN_CL20=${RUN_CL20:-1} RUN_CL50=${RUN_CL50:-0}"

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

TRAIN_OUTPUT="$TRAIN_ROOT/bdse_v28_darbsr.pt"
V28_CKPT=${V28_CKPT:-$TRAIN_ROOT/bdse_v28_darbsr.best.pt}

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
  echo "[train] v28 DA-RBSR finetune from $WARM_START"
  torchrun --standalone --nproc_per_node=${NPROC_PER_NODE:-2} -m bdse.experiments.train \
    --config bdse/configs/v28_bdse_darbsr_train.yaml \
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
    --log-file "$LOG_ROOT/v28_darbsr_train.jsonl" \
    > "$LOG_ROOT/v28_darbsr_train.out" 2>&1
else
  echo "[train] SKIP_TRAIN=1, using existing checkpoint: $V28_CKPT"
fi

# For zero-training ablation, reuse the v25/v26 checkpoint with v28 runtime guard/configs:
#   SKIP_TRAIN=1 V28_CKPT=outputs_v25/train/bdse_v25_dgcace.best.pt bash run_v28_darbsr.sh
if [[ ! -f "$V28_CKPT" ]]; then
  echo "Missing v28 checkpoint: $V28_CKPT" >&2
  echo "For runtime-only check, run: SKIP_TRAIN=1 V28_CKPT=$V25_CKPT_IN bash run_v28_darbsr.sh" >&2
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
    ('v28_darbsr_fixed_budget', root / 'open_loop_v28_darbsr_fixed_budget.json'),
    ('v28_darbsr_safety_fallback', root / 'open_loop_v28_darbsr_safety_fallback.json'),
    ('v28_darbsr_bbr_scur', root / 'open_loop_v28_darbsr_bbr_scur.json'),
    ('v28_darbsr_lcb_control', root / 'open_loop_v28_darbsr_lcb_control.json'),
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


# Run four open-loop evaluations concurrently by default: two jobs on GPU 0 and two on GPU 1.
# Set OPEN_PARALLEL4=0 to use the old two-at-a-time schedule.
if [[ "${SKIP_OPEN_LOOP:-0}" != "1" ]]; then
  if [[ "${OPEN_PARALLEL4:-1}" == "1" ]]; then
    run_open_loop "${GPU_OL_FIXED:-0}" v28_darbsr_fixed_budget bdse/configs/v28_bdse_darbsr_fixed_budget_fast_cl.yaml "$V28_CKPT" &
    run_open_loop "${GPU_OL_SAFETY:-0}" v28_darbsr_safety_fallback bdse/configs/v28_bdse_darbsr_safety_fallback_fast_cl.yaml "$V28_CKPT" &
    run_open_loop "${GPU_OL_BBR:-1}" v28_darbsr_bbr_scur bdse/configs/v28_bdse_darbsr_bbr_scur_fast_cl.yaml "$V28_CKPT" &
    run_open_loop "${GPU_OL_LCB:-1}" v28_darbsr_lcb_control bdse/configs/v28_bdse_darbsr_lcb_control_fast_cl.yaml "$V28_CKPT" &
    wait
  else
    run_open_loop "${GPU_OL_FIXED:-0}" v28_darbsr_fixed_budget bdse/configs/v28_bdse_darbsr_fixed_budget_fast_cl.yaml "$V28_CKPT" &
    run_open_loop "${GPU_OL_SAFETY:-1}" v28_darbsr_safety_fallback bdse/configs/v28_bdse_darbsr_safety_fallback_fast_cl.yaml "$V28_CKPT" &
    wait
    run_open_loop "${GPU_OL_BBR:-0}" v28_darbsr_bbr_scur bdse/configs/v28_bdse_darbsr_bbr_scur_fast_cl.yaml "$V28_CKPT" &
    run_open_loop "${GPU_OL_LCB:-1}" v28_darbsr_lcb_control bdse/configs/v28_bdse_darbsr_lcb_control_fast_cl.yaml "$V28_CKPT" &
    wait
  fi
  print_open_loop_compare
else
  echo "SKIP_OPEN_LOOP=1, skipping open-loop runs."
fi

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


if [[ "${RUN_CL20:-1}" == "1" ]]; then
  if [[ "${CL_PARALLEL4:-1}" == "1" ]]; then
    run_closed_loop "${GPU_CL_FIXED:-0}" v28_darbsr_fixed_budget bdse/configs/v28_bdse_darbsr_fixed_budget_fast_cl.yaml "$V28_CKPT" 20 &
    run_closed_loop "${GPU_CL_SAFETY:-0}" v28_darbsr_safety_fallback bdse/configs/v28_bdse_darbsr_safety_fallback_fast_cl.yaml "$V28_CKPT" 20 &
    run_closed_loop "${GPU_CL_BBR:-1}" v28_darbsr_bbr_scur bdse/configs/v28_bdse_darbsr_bbr_scur_fast_cl.yaml "$V28_CKPT" 20 &
    run_closed_loop "${GPU_CL_LCB:-1}" v28_darbsr_lcb_control bdse/configs/v28_bdse_darbsr_lcb_control_fast_cl.yaml "$V28_CKPT" 20 &
    wait
  else
    run_closed_loop "${GPU_CL_FIXED:-0}" v28_darbsr_fixed_budget bdse/configs/v28_bdse_darbsr_fixed_budget_fast_cl.yaml "$V28_CKPT" 20 &
    run_closed_loop "${GPU_CL_SAFETY:-1}" v28_darbsr_safety_fallback bdse/configs/v28_bdse_darbsr_safety_fallback_fast_cl.yaml "$V28_CKPT" 20 &
    wait
    run_closed_loop "${GPU_CL_BBR:-0}" v28_darbsr_bbr_scur bdse/configs/v28_bdse_darbsr_bbr_scur_fast_cl.yaml "$V28_CKPT" 20 &
    run_closed_loop "${GPU_CL_LCB:-1}" v28_darbsr_lcb_control bdse/configs/v28_bdse_darbsr_lcb_control_fast_cl.yaml "$V28_CKPT" 20 &
    wait
  fi
fi

if [[ "${RUN_CL20:-1}" == "1" ]]; then
  python -m bdse.tools.collect_closed_loop_metrics \
    "$CL_ROOT/v28_darbsr_fixed_budget_20" \
    "$CL_ROOT/v28_darbsr_safety_fallback_20" \
    "$CL_ROOT/v28_darbsr_bbr_scur_20" \
    "$CL_ROOT/v28_darbsr_lcb_control_20" \
    --csv "$CL_ROOT/v28_20_compare.csv"
fi

python - "$LOG_ROOT" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
for path in sorted(list(root.glob('*.closed_loop_20.diag.jsonl')) + list(root.glob('*.closed_loop_50.diag.jsonl'))):
    replans = []
    reasons = Counter()
    selected_flags = 0
    hard_filter = 0
    hard_safe_avail = 0
    soft_safe_avail = 0
    active_safe_avail = 0
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
        rs = d.get('runtime_safety', {}) or {}
        selected_flags += int(bool(t.get('selected_action_safety_flag', False)))
        hard_filter += int(bool(t.get('hard_filter_applied', False)))
        hard_safe_avail += int(bool(rs.get('hard_safe_action_available', False)))
        soft_safe_avail += int(bool(rs.get('soft_safe_action_available', False)))
        active_safe_avail += int(bool(rs.get('active_safe_action_available', False)))
    if not replans:
        continue
    triggered = sum(1 for d in replans if d.get('fallback_triggered')) / len(replans)
    print(path.name, 'replans=', len(replans), 'fallback_triggered_rate=', round(triggered, 4),
          'selected_action_safety_flag_rate=', round(selected_flags / len(replans), 4),
          'hard_filter_applied_rate=', round(hard_filter / len(replans), 4),
          'hard_safe_avail_rate=', round(hard_safe_avail / len(replans), 4),
          'soft_safe_avail_rate=', round(soft_safe_avail / len(replans), 4),
          'active_safe_avail_rate=', round(active_safe_avail / len(replans), 4),
          'reasons=', dict(reasons))
PY

if [[ "${RUN_CL50:-0}" == "1" ]]; then
  if [[ "${RUN_CL50_ALL4:-1}" == "1" ]]; then
    run_closed_loop "${GPU_CL50_FIXED:-0}" v28_darbsr_fixed_budget bdse/configs/v28_bdse_darbsr_fixed_budget_fast_cl.yaml "$V28_CKPT" 50 &
    run_closed_loop "${GPU_CL50_SAFETY:-0}" v28_darbsr_safety_fallback bdse/configs/v28_bdse_darbsr_safety_fallback_fast_cl.yaml "$V28_CKPT" 50 &
    run_closed_loop "${GPU_CL50_BBR:-1}" v28_darbsr_bbr_scur bdse/configs/v28_bdse_darbsr_bbr_scur_fast_cl.yaml "$V28_CKPT" 50 &
    run_closed_loop "${GPU_CL50_LCB:-1}" v28_darbsr_lcb_control bdse/configs/v28_bdse_darbsr_lcb_control_fast_cl.yaml "$V28_CKPT" 50 &
    wait
    python -m bdse.tools.collect_closed_loop_metrics \
      "$CL_ROOT/v28_darbsr_fixed_budget_50" \
      "$CL_ROOT/v28_darbsr_safety_fallback_50" \
      "$CL_ROOT/v28_darbsr_bbr_scur_50" \
      "$CL_ROOT/v28_darbsr_lcb_control_50" \
      --csv "$CL_ROOT/v28_50_compare.csv"
  else
    run_closed_loop "${GPU_CL50_FIXED:-0}" v28_darbsr_fixed_budget bdse/configs/v28_bdse_darbsr_fixed_budget_fast_cl.yaml "$V28_CKPT" 50 &
    run_closed_loop "${GPU_CL50_SAFETY:-1}" v28_darbsr_safety_fallback bdse/configs/v28_bdse_darbsr_safety_fallback_fast_cl.yaml "$V28_CKPT" 50 &
    wait
    python -m bdse.tools.collect_closed_loop_metrics \
      "$CL_ROOT/v28_darbsr_fixed_budget_50" \
      "$CL_ROOT/v28_darbsr_safety_fallback_50" \
      --csv "$CL_ROOT/v28_50_compare.csv"
  fi
  python - "$LOG_ROOT" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
for path in sorted(root.glob('*.closed_loop_50.diag.jsonl')):
    replans = []
    reasons = Counter()
    selected_flags = hard_filter = hard_safe_avail = soft_safe_avail = active_safe_avail = 0
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
        rs = d.get('runtime_safety', {}) or {}
        selected_flags += int(bool(t.get('selected_action_safety_flag', False)))
        hard_filter += int(bool(t.get('hard_filter_applied', False)))
        hard_safe_avail += int(bool(rs.get('hard_safe_action_available', False)))
        soft_safe_avail += int(bool(rs.get('soft_safe_action_available', False)))
        active_safe_avail += int(bool(rs.get('active_safe_action_available', False)))
    if replans:
        triggered = sum(1 for d in replans if d.get('fallback_triggered')) / len(replans)
        print(path.name, 'replans=', len(replans), 'fallback_triggered_rate=', round(triggered, 4),
              'selected_action_safety_flag_rate=', round(selected_flags / len(replans), 4),
              'hard_filter_applied_rate=', round(hard_filter / len(replans), 4),
              'hard_safe_avail_rate=', round(hard_safe_avail / len(replans), 4),
              'soft_safe_avail_rate=', round(soft_safe_avail / len(replans), 4),
              'active_safe_avail_rate=', round(active_safe_avail / len(replans), 4),
              'reasons=', dict(reasons))
PY
fi

echo "Done. Key outputs:"
echo "  $TRAIN_ROOT/bdse_v28_darbsr.best.pt"
echo "  $OPEN_ROOT/open_loop_v28_*.json"
echo "  $CL_ROOT/v28_20_compare.csv"
echo "  $LOG_ROOT/*.closed_loop_20.diag.jsonl"
echo "Runtime-only first check: export SKIP_TRAIN=1 V28_CKPT=$V25_CKPT_IN RUN_MODE=open_loop; bash run_v28_darbsr.sh"
echo "Runtime-only CL20: export SKIP_TRAIN=1 V28_CKPT=$V25_CKPT_IN RUN_MODE=cl20; bash run_v28_darbsr.sh"
echo "Finetune: export TRAIN_MAX_SCENARIOS=12000 VAL_MAX_SCENARIOS=1000 TRAIN_EPOCHS=5 RUN_MODE=all; bash run_v28_darbsr.sh"

# CL50 after finetune: export SKIP_TRAIN=1 V28_CKPT=outputs_v28/train/bdse_v28_darbsr.best.pt RUN_MODE=cl50; bash run_v28_darbsr.sh
