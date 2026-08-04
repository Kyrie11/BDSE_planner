#!/usr/bin/env bash
set -euo pipefail

# v61.0 checkpoint-independent pipeline.  Resolve every relative path from the
# checked-in script directory so a stale working directory cannot silently run a
# different config or helper script.
PIPELINE_VERSION="v61.1-deployment-exact-hierarchical-winner-preservation"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "[v61] pipeline_version=$PIPELINE_VERSION script=$SCRIPT_DIR/$(basename "$0")"

# Mandatory data paths.
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE_ORIGINAL="${BDSE_VAL_CACHE_ORIGINAL:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export BDSE_SPLIT_CACHE="${BDSE_SPLIT_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v53_split}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_v2}"
export OUT_ROOT="${OUT_ROOT:-outputs_v61_dehab_bfar_dbap_fast_2gpu_v1}"
export FOUNDATION_ROOT="${FOUNDATION_ROOT:-$OUT_ROOT/factorized_anchor}"
export FOUNDATION_CONFIG="${FOUNDATION_CONFIG:-bdse/configs/v53_factorized_anchor_fast_2gpu.yaml}"
export FOUNDATION_SOURCE_CONFIG="${FOUNDATION_SOURCE_CONFIG:-$FOUNDATION_CONFIG}"

# There is deliberately no hard-coded outputs_v30 dependency.  FOUNDATION_CKPT
# is the canonical variable; V30_CKPT_IN remains only as a backwards-compatible
# alias consumed by run_v61_dehab_bfar_dbap.sh.
export V30_CKPT_IN="${V30_CKPT_IN:-}"
export FOUNDATION_CKPT="${FOUNDATION_CKPT:-$V30_CKPT_IN}"
export FOUNDATION_POLICY="${FOUNDATION_POLICY:-auto}"  # auto | rebuild | recover | explicit
export FOUNDATION_SEARCH_ROOT="${FOUNDATION_SEARCH_ROOT:-.}"
export RECOVER_SAFE_FOUNDATION_COPIES="${RECOVER_SAFE_FOUNDATION_COPIES:-1}"
export REBUILD_FOUNDATION_IF_MISSING="${REBUILD_FOUNDATION_IF_MISSING:-1}"
export ALLOW_ALGORITHM_CHECKPOINT_INIT="${ALLOW_ALGORITHM_CHECKPOINT_INIT:-0}"
export V61_INIT_MODE="${V61_INIT_MODE:-warm_start}"
export EXACT_SELECTOR_WORKERS_PER_RANK="${EXACT_SELECTOR_WORKERS_PER_RANK:-4}"
export EXACT_SELECTOR_CPU_BACKEND="${EXACT_SELECTOR_CPU_BACKEND:-process}"
export GPUS="${GPUS:-0,1}"
export FOUNDATION_CONTROL_CONFIG="${FOUNDATION_CONTROL_CONFIG:-bdse/configs/v61_dehab_bfar_dbap_anchor_control_cl.yaml}"
export LOCAL_CONTROL_CONFIG="${LOCAL_CONTROL_CONFIG:-bdse/configs/v61_dehab_bfar_dbap_local_control_cl.yaml}"
export CONTROL_CONFIG="${CONTROL_CONFIG:-$FOUNDATION_CONTROL_CONFIG}"  # compatibility alias
export CONTROL_CKPT="${CONTROL_CKPT:-}"
export FOUNDATION_CONTROL_ROOT="${FOUNDATION_CONTROL_ROOT:-$OUT_ROOT/control_foundation_matched}"
export LOCAL_CONTROL_ROOT="${LOCAL_CONTROL_ROOT:-$OUT_ROOT/control_local_same_checkpoint}"
export PIPELINE_DETACH="${PIPELINE_DETACH:-1}"
export PIPELINE_FORCE="${PIPELINE_FORCE:-0}"
export RUN_CLOSED_LOOP_AFTER_GATE="${RUN_CLOSED_LOOP_AFTER_GATE:-0}"
export RUN_DIAGNOSTIC_CL20_ON_GATE_FAIL="${RUN_DIAGNOSTIC_CL20_ON_GATE_FAIL:-0}"
export RUN_CL100_AFTER_CL20="${RUN_CL100_AFTER_CL20:-0}"
# Keep algorithmic closed-loop validation separate from the deployment-latency
# claim. Set to 1 only when CL should be blocked by the real-time target.
export ENFORCE_LATENCY_BEFORE_CL="${ENFORCE_LATENCY_BEFORE_CL:-0}"
export TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-8}"
export TRAIN_NUM_WORKERS_PER_GPU="${TRAIN_NUM_WORKERS_PER_GPU:-6}"
export FAST_SELECTOR_SCENES_PER_RANK="${FAST_SELECTOR_SCENES_PER_RANK:-1}"
export FAST_SELECTOR_EVERY_N_STEPS="${FAST_SELECTOR_EVERY_N_STEPS:-4}"
export CL_PROCESSES_PER_GPU="${CL_PROCESSES_PER_GPU:-2}"
export CL_WORKERS_PER_GPU=1

case "$FOUNDATION_POLICY" in
  auto|rebuild|recover|explicit) ;;
  *) echo "FOUNDATION_POLICY must be auto, rebuild, recover, or explicit" >&2; exit 2 ;;
esac

mkdir -p "$OUT_ROOT/logs"

# Detach the complete pipeline, not only the training child.  Detaching only
# run_v61_dehab_bfar_dbap.sh lets this parent continue immediately into calibration while
# training is still writing checkpoints.  The child below keeps all stages in
# one ordered session and survives an SSH/terminal disconnect.
if [[ "$PIPELINE_DETACH" == "1" && "${BDSE_PIPELINE_CHILD:-0}" != "1" ]]; then
  command -v setsid >/dev/null 2>&1 || {
    echo "PIPELINE_DETACH=1 requires the setsid command" >&2
    exit 2
  }
  export BDSE_PIPELINE_CHILD=1 PIPELINE_DETACH=0
  pipeline_log="$OUT_ROOT/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
  script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  setsid nohup bash "$script_path" "$@" </dev/null >>"$pipeline_log" 2>&1 &
  pipeline_pid=$!
  echo "$pipeline_pid" >"$OUT_ROOT/logs/pipeline.pid"
  echo "[v61] detached pipeline started pid=$pipeline_pid log=$pipeline_log"
  exit 0
fi

PIPELINE_LOCK_DIR="$OUT_ROOT/.v61_pipeline.lock"
if ! mkdir "$PIPELINE_LOCK_DIR" 2>/dev/null; then
  owner="$(cat "$PIPELINE_LOCK_DIR/pid" 2>/dev/null || echo unknown)"
  echo "A v61 pipeline is already using OUT_ROOT=$OUT_ROOT (pid=$owner)" >&2
  exit 3
fi
echo "$$" > "$PIPELINE_LOCK_DIR/pid"
trap 'rm -rf "$PIPELINE_LOCK_DIR"' EXIT INT TERM

is_fresh() {
  local output="$1"
  shift
  [[ "$PIPELINE_FORCE" != "1" && -s "$output" ]] || return 1
  local input
  for input in "$@"; do
    [[ -e "$input" && "$output" -nt "$input" ]] || return 1
  done
}

training_complete() {
  [[ "$PIPELINE_FORCE" != "1" ]] || return 1
  # The final model is written only after a clean full run or configured early
  # stop.  Its presence is therefore a stronger completion marker than forcing
  # the log to reach the nominal epoch count.
  [[ -s "$OUT_ROOT/train/bdse_v61_dehab_bfar_dbap.train_log.jsonl"      && -s "$OUT_ROOT/train/bdse_v61_dehab_bfar_dbap.pt"      && -s "$OUT_ROOT/train/bdse_v61_dehab_bfar_dbap.best.pt"      && "$OUT_ROOT/train/bdse_v61_dehab_bfar_dbap.pt" -nt bdse/configs/v61_dehab_bfar_dbap_train_2gpu.yaml ]]
}

# ---------------------------------------------------------------------------
# 0. Create a paper-grade validation protocol. The split is by nuPlan log group,
#    not random scenario row, to reduce temporal/scene leakage.
# ---------------------------------------------------------------------------
if [[ "$PIPELINE_FORCE" != "1" \
      && -s "$BDSE_SPLIT_CACHE/calibration_split_provenance.json" \
      && -s "$BDSE_SPLIT_CACHE/val_tune/manifest.jsonl" \
      && -s "$BDSE_SPLIT_CACHE/val_calib/manifest.jsonl" ]]; then
  echo "[v61] stage 0 already complete: reuse group-disjoint split"
else
  python -m bdse.tools.build_group_disjoint_calibration_split \
    --preprocessed-dir "$BDSE_VAL_CACHE_ORIGINAL" \
    --split val \
    --output-root "$BDSE_SPLIT_CACHE" \
    --calibration-fraction 0.20 \
    --seed 49
fi

export BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE"

# ---------------------------------------------------------------------------
# 0.5 Resolve a matched foundation without depending on outputs_v30.
#
# Resolution order under FOUNDATION_POLICY=auto:
#   1) explicit FOUNDATION_CKPT / V30_CKPT_IN, when the file exists;
#   2) a foundation already rebuilt under this OUT_ROOT;
#   3) a conservatively verified retained compatible anchor copy from outputs_v40..v53;
#   4) a fresh current-code factorized base+local anchor rebuild.
#
# Later D3CE/DBCE/DBAP/AOCC checkpoints are inventoried but rejected by default.
# They are algorithm-specific initializations and would confound the v61 paper
# comparison.  ALLOW_ALGORITHM_CHECKPOINT_INIT=1 is an explicit transfer-ablation
# escape hatch, never the default main-run protocol.
# ---------------------------------------------------------------------------
FOUNDATION_SOURCE=""
FOUNDATION_INVENTORY="$OUT_ROOT/provenance/foundation_checkpoint_inventory.json"
mkdir -p "$OUT_ROOT/provenance"

resolve_foundation_checkpoint() {
  local rebuilt_best="$FOUNDATION_ROOT/train/bdse_v53_factorized_anchor.best.pt"
  local selected=""

  if [[ "$FOUNDATION_POLICY" == "explicit" ]]; then
    [[ -n "$FOUNDATION_CKPT" && -s "$FOUNDATION_CKPT" ]] || {
      echo "FOUNDATION_POLICY=explicit requires an existing FOUNDATION_CKPT" >&2
      exit 2
    }
    FOUNDATION_SOURCE="explicit_external"
    return 0
  fi

  if [[ "$FOUNDATION_POLICY" == "auto" && -n "$FOUNDATION_CKPT" && -s "$FOUNDATION_CKPT" ]]; then
    FOUNDATION_SOURCE="explicit_external"
    return 0
  fi

  # Ignore inherited stale paths such as the deleted outputs_v30/... file.
  if [[ -n "$FOUNDATION_CKPT" && ! -s "$FOUNDATION_CKPT" ]]; then
    echo "[v61] ignore missing inherited foundation path: $FOUNDATION_CKPT"
    FOUNDATION_CKPT=""
  fi

  if [[ -s "$rebuilt_best" ]]; then
    FOUNDATION_CKPT="$rebuilt_best"
    FOUNDATION_SOURCE="rebuilt_current_code_reused"
    return 0
  fi

  if [[ "$FOUNDATION_POLICY" != "rebuild" && "$RECOVER_SAFE_FOUNDATION_COPIES" == "1" ]]; then
    local allow_args=()
    if [[ "$ALLOW_ALGORITHM_CHECKPOINT_INIT" == "1" ]]; then
      allow_args+=(--allow-algorithm-checkpoints)
      echo "[v61] WARNING: algorithm-specific checkpoint recovery enabled; use only as a transfer ablation" >&2
    fi
    selected="$(python -m bdse.tools.resolve_foundation_checkpoint       --config bdse/configs/v61_dehab_bfar_dbap_train_2gpu.yaml       --search-root "$FOUNDATION_SEARCH_ROOT"       --output-json "$FOUNDATION_INVENTORY"       --print-selected       "${allow_args[@]}")"
    if [[ -n "$selected" && -s "$selected" ]]; then
      FOUNDATION_CKPT="$selected"
      if [[ "$ALLOW_ALGORITHM_CHECKPOINT_INIT" == "1" ]]; then
        FOUNDATION_SOURCE="recovered_algorithm_transfer_ablation"
      else
        FOUNDATION_SOURCE="recovered_verified_compatible_anchor"
      fi
      echo "[v61] recovered foundation checkpoint=$FOUNDATION_CKPT source=$FOUNDATION_SOURCE"
      return 0
    fi
  else
    # Still write an inventory report when recovery is disabled, because it is
    # useful for documenting why retained v40-v49 checkpoints were not used.
    python -m bdse.tools.resolve_foundation_checkpoint       --config bdse/configs/v61_dehab_bfar_dbap_train_2gpu.yaml       --search-root "$FOUNDATION_SEARCH_ROOT"       --output-json "$FOUNDATION_INVENTORY" >/dev/null || true
  fi

  if [[ "$FOUNDATION_POLICY" == "recover" ]]; then
    echo "FOUNDATION_POLICY=recover found no conservatively safe v30 copy; see $FOUNDATION_INVENTORY" >&2
    exit 2
  fi
  if [[ "$REBUILD_FOUNDATION_IF_MISSING" != "1" ]]; then
    echo "No usable foundation checkpoint and rebuilding is disabled" >&2
    exit 2
  fi

  echo "[v61] no safe historical foundation found; rebuilding from random initialization"
  DETACH=0   GPUS="$GPUS"   INIT_MODE=scratch   V30_CKPT_IN=   OUT_ROOT="$FOUNDATION_ROOT"   FOUNDATION_OUT_ROOT="$FOUNDATION_ROOT"   FOUNDATION_CONFIG="$FOUNDATION_CONFIG"   RUN_MODE=foundation   AUTO_RESUME=1   VAL_SPLIT=val_tune   VAL_SCENARIOS=1000   BATCH_SIZE_PER_GPU="$TRAIN_BATCH_SIZE_PER_GPU"   NUM_WORKERS_PER_GPU="$TRAIN_NUM_WORKERS_PER_GPU"   PREFETCH_FACTOR=2   SAVE_EVERY_N_STEPS=1000   bash run_v61_dehab_bfar_dbap.sh

  [[ -s "$rebuilt_best" ]] || {
    echo "Foundation rebuild did not produce $rebuilt_best" >&2
    exit 2
  }
  FOUNDATION_CKPT="$rebuilt_best"
  FOUNDATION_SOURCE="rebuilt_current_code"
}

if [[ "$V61_INIT_MODE" == "warm_start" ]]; then
  resolve_foundation_checkpoint
  [[ -s "$FOUNDATION_CKPT" ]] || { echo "Resolved foundation is missing: $FOUNDATION_CKPT" >&2; exit 2; }
  V30_CKPT_IN="$FOUNDATION_CKPT"  # compatibility alias for the inner launcher
  if [[ -z "$CONTROL_CKPT" ]]; then
    CONTROL_CKPT="$FOUNDATION_CKPT"
  fi
else
  echo "[v61] V61_INIT_MODE=scratch is an ablation, not the paper main run."
  [[ -n "$CONTROL_CKPT" && -s "$CONTROL_CKPT" ]] || {
    echo "Scratch v61 still needs an independently trained matched CONTROL_CKPT for a paired gate" >&2
    exit 2
  }
  FOUNDATION_CKPT="$CONTROL_CKPT"
  V30_CKPT_IN="$FOUNDATION_CKPT"
  FOUNDATION_SOURCE="scratch_ablation_matched_control"
fi
export FOUNDATION_CKPT V30_CKPT_IN CONTROL_CKPT

echo "[v61] foundation_source=$FOUNDATION_SOURCE"
echo "[v61] foundation_checkpoint=$FOUNDATION_CKPT"
echo "[v61] control_checkpoint=$CONTROL_CKPT"

FOUNDATION_PROVENANCE="$OUT_ROOT/provenance/foundation_checkpoint.json"
if ! is_fresh "$FOUNDATION_PROVENANCE" "$FOUNDATION_CKPT" "$FOUNDATION_SOURCE_CONFIG"; then
  python -m bdse.tools.write_checkpoint_provenance     --checkpoint "$FOUNDATION_CKPT"     --config "$FOUNDATION_SOURCE_CONFIG"     --train-cache "$BDSE_TRAIN_CACHE"     --val-cache "$BDSE_SPLIT_CACHE"     --source "$FOUNDATION_SOURCE"     --output "$FOUNDATION_PROVENANCE"
fi

# ---------------------------------------------------------------------------
# 0.75 Foundation quality gate. A weak foundation makes residual results
#      uninterpretable, so the algorithm stage is blocked before any v61 update.
# ---------------------------------------------------------------------------
FOUNDATION_QUALITY_ROOT="$FOUNDATION_ROOT/quality"
FOUNDATION_QUALITY_JSON="$FOUNDATION_QUALITY_ROOT/open_loop.json"
FOUNDATION_QUALITY_JSONL="$FOUNDATION_QUALITY_ROOT/open_loop.jsonl"
mkdir -p "$FOUNDATION_QUALITY_ROOT"
if is_fresh "$FOUNDATION_QUALITY_JSON" "$FOUNDATION_CKPT" "$FOUNDATION_CONTROL_CONFIG" \
   && is_fresh "$FOUNDATION_QUALITY_JSONL" "$FOUNDATION_CKPT" "$FOUNDATION_CONTROL_CONFIG"; then
  echo "[v61] foundation quality replay already complete"
else
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python -m bdse.experiments.evaluate_open_loop \
    --config "$FOUNDATION_CONTROL_CONFIG" \
    --checkpoint "$FOUNDATION_CKPT" \
    --split val_tune \
    --preprocessed-dir "$BDSE_SPLIT_CACHE" \
    --max-scenarios 1000 \
    --device cuda \
    --output "$FOUNDATION_QUALITY_JSON" \
    --per-sample-output "$FOUNDATION_QUALITY_JSONL"
fi
ANCHOR_GATE_REPORT="$OUT_ROOT/provenance/v61_anchor_gate.json"
python -m bdse.tools.check_v53_anchor_quality "$FOUNDATION_QUALITY_JSON" \
  --jsonl "$FOUNDATION_QUALITY_JSONL" \
  --min-full-interface-match "${ANCHOR_MIN_FULL_INTERFACE_MATCH:-0.32}" \
  --min-base-winner-sign "${ANCHOR_MIN_BASE_WINNER_SIGN:-0.62}" \
  --min-dense-winner-sign "${ANCHOR_MIN_DENSE_WINNER_SIGN:-0.75}" \
  --min-dense-near-sign "${ANCHOR_MIN_DENSE_NEAR_SIGN:-0.65}" \
  --min-dense-all-sign "${ANCHOR_MIN_DENSE_ALL_SIGN:-0.68}" \
  --warn-max-full-interface-regret "${ANCHOR_WARN_MAX_FULL_INTERFACE_REGRET:-15000}" \
  --report-json "$ANCHOR_GATE_REPORT" \
  2>&1 | tee "$OUT_ROOT/logs/v61_anchor_quality_gate.out"

# ---------------------------------------------------------------------------
# 1. Main v61 training. Base/local encoders are frozen by config; only the
#    reset residual/uncertainty and proposal-family heads are trainable.
# ---------------------------------------------------------------------------
if training_complete; then
  echo "[v61] stage 1 already complete: reuse final/best checkpoints"
else
  DETACH=0 \
  GPUS="$GPUS" \
  V30_CKPT_IN="$V30_CKPT_IN" \
  INIT_MODE="$V61_INIT_MODE" \
  TRAIN_CONFIG=bdse/configs/v61_dehab_bfar_dbap_train_2gpu.yaml \
  EXACT_SELECTOR_WORKERS_PER_RANK="$EXACT_SELECTOR_WORKERS_PER_RANK" \
  EXACT_SELECTOR_CPU_BACKEND="$EXACT_SELECTOR_CPU_BACKEND" \
  OUT_ROOT="$OUT_ROOT" \
  RUN_MODE=train \
  AUTO_RESUME=0 \
  VAL_SPLIT=val_tune \
  OPEN_LOOP_SPLIT=val_tune \
  BATCH_SIZE_PER_GPU="$TRAIN_BATCH_SIZE_PER_GPU" \
  NUM_WORKERS_PER_GPU="$TRAIN_NUM_WORKERS_PER_GPU" \
  PREFETCH_FACTOR=2 \
  SAVE_EVERY_N_STEPS=1000 \
  SELECTOR_SCENES_PER_RANK="$FAST_SELECTOR_SCENES_PER_RANK" \
  SELECTOR_EVERY_N_STEPS="$FAST_SELECTOR_EVERY_N_STEPS" \
  SELECTOR_FULL_LAST_N_STEPS=48 \
  EXACT_DISTILL_SCENES_PER_RANK=0 \
  EXACT_DISTILL_EVERY_N_STEPS=1 \
  VAL_SCENARIOS=1000 \
  VAL_EVERY_N_EPOCHS=2 \
  VAL_DENSE_DIAGNOSTIC=1 \
  OPEN_LOOP_MAX_SCENARIOS=1000 \
  bash run_v61_dehab_bfar_dbap.sh
fi

BEST_CHECKPOINT="$OUT_ROOT/train/bdse_v61_dehab_bfar_dbap.best.pt"
[[ -s "$BEST_CHECKPOINT" ]] || { echo "Missing v61 best checkpoint: $BEST_CHECKPOINT" >&2; exit 2; }

# ---------------------------------------------------------------------------
# 2. One shared evidence calibration plus candidate-only residual calibration.
#    V57 repeated the same frozen local-evidence replay three times and still did
#    not calibrate the direct residual uncertainty used by the flip guard.  V61
#    collects both certificates in one paired candidate pass, sharded across GPUs.
# ---------------------------------------------------------------------------
mkdir -p "$OUT_ROOT/calibration/raw" "$OUT_ROOT/calibration/shards"
CALIBRATION_PROVENANCE="$BDSE_SPLIT_CACHE/calibration_split_provenance.json"
DUAL_CAL_JSON="$OUT_ROOT/calibration/v61_dual_certificate.json"
CAL_SHARD_ROOT="$OUT_ROOT/calibration/shards"

prepare_calibration_shards() {
  rm -rf "$CAL_SHARD_ROOT"
  mkdir -p "$CAL_SHARD_ROOT/gpu0/val" "$CAL_SHARD_ROOT/gpu1/val"
  python - "$BDSE_SPLIT_CACHE" "$CAL_SHARD_ROOT" <<'PY_CAL_SHARDS'
import json, sys
from pathlib import Path
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
root=Path(sys.argv[1]); out=Path(sys.argv[2])
paths=PreprocessedBDSEDataset(root, split=['val_calib'], max_scenarios=5000).build_index()
if not paths: raise SystemExit('No val_calib samples')
for sid in range(2):
    rows=[{'split':'val','path':str(p.resolve()),'original_index':i} for i,p in enumerate(paths) if i%2==sid]
    (out/f'gpu{sid}'/'val'/'manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
(out/'metadata.json').write_text(json.dumps({'num_scenarios':len(paths),'num_shards':2},indent=2),encoding='utf-8')
PY_CAL_SHARDS
}

if ! is_fresh "$DUAL_CAL_JSON" "$BEST_CHECKPOINT" bdse/configs/v61_dehab_bfar_dbap_cl.yaml "$CALIBRATION_PROVENANCE"; then
  prepare_calibration_shards
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python -m bdse.tools.calibrate_v61_dual_certificates \
    --raw-output "$OUT_ROOT/calibration/raw/gpu0.npz" \
    --config bdse/configs/v61_dehab_bfar_dbap_cl.yaml --checkpoint "$BEST_CHECKPOINT" \
    --preprocessed-dir "$CAL_SHARD_ROOT/gpu0" --split val --max-scenarios 5000 \
    --device cuda --alpha 0.05 --beta 0.0 --prior-radius 0.02 \
    > "$OUT_ROOT/logs/calibration_gpu0.out" 2>&1 & pid_cal0=$!
  CUDA_VISIBLE_DEVICES="$(echo "$GPUS" | cut -d, -f2)" python -m bdse.tools.calibrate_v61_dual_certificates \
    --raw-output "$OUT_ROOT/calibration/raw/gpu1.npz" \
    --config bdse/configs/v61_dehab_bfar_dbap_cl.yaml --checkpoint "$BEST_CHECKPOINT" \
    --preprocessed-dir "$CAL_SHARD_ROOT/gpu1" --split val --max-scenarios 5000 \
    --device cuda --alpha 0.05 --beta 0.0 --prior-radius 0.02 \
    > "$OUT_ROOT/logs/calibration_gpu1.out" 2>&1 & pid_cal1=$!
  wait "$pid_cal0"; wait "$pid_cal1"
  python -m bdse.tools.calibrate_v61_dual_certificates \
    --merge-raw "$OUT_ROOT/calibration/raw/gpu0.npz" "$OUT_ROOT/calibration/raw/gpu1.npz" \
    --alpha 0.05 --beta 0.0 --residual-epsilon-fallback 0.05 \
    --provenance-json "$CALIBRATION_PROVENANCE" --output "$DUAL_CAL_JSON"
else
  echo "[v61] reuse dual-certificate calibration"
fi

CAND_CAL_CFG="$OUT_ROOT/calibration/v61_dehab_bfar_dbap_candidate_calibrated.yaml"
LOCAL_CAL_CFG="$OUT_ROOT/calibration/v61_dehab_bfar_dbap_local_control_calibrated.yaml"
FOUND_CAL_CFG="$OUT_ROOT/calibration/v61_dehab_bfar_dbap_foundation_control_calibrated.yaml"
python -m bdse.tools.apply_v61_dual_calibration --config bdse/configs/v61_dehab_bfar_dbap_cl.yaml --calibration-json "$DUAL_CAL_JSON" --output "$CAND_CAL_CFG"
python -m bdse.tools.apply_v61_dual_calibration --config "$LOCAL_CONTROL_CONFIG" --calibration-json "$DUAL_CAL_JSON" --output "$LOCAL_CAL_CFG" --control
python -m bdse.tools.apply_v61_dual_calibration --config "$FOUNDATION_CONTROL_CONFIG" --calibration-json "$DUAL_CAL_JSON" --output "$FOUND_CAL_CFG" --control

# ---------------------------------------------------------------------------
# 3. Candidate/local/foundation open-loop in one bounded worker pool.
#    All systems use the same modulo shards; OPEN_LOOP_WORKERS_PER_GPU controls
#    overlap.  Start at 2 because V57 used <100 MB/model, then benchmark 1 vs 2.
# ---------------------------------------------------------------------------
OPEN_LOOP_WORKERS_PER_GPU="${OPEN_LOOP_WORKERS_PER_GPU:-2}"
SUITE_ROOT="$OUT_ROOT/open_loop/parallel_suite"
SUITE_REPORT="$SUITE_ROOT/parallel_open_loop_suite_report.json"
if ! is_fresh "$SUITE_REPORT" "$BEST_CHECKPOINT" "$FOUNDATION_CKPT" "$CAND_CAL_CFG" "$LOCAL_CAL_CFG" "$FOUND_CAL_CFG"; then
  rm -rf "$SUITE_ROOT"
  python -m bdse.tools.run_parallel_open_loop_suite \
    --system "candidate::$CAND_CAL_CFG::$BEST_CHECKPOINT" \
    --system "local::$LOCAL_CAL_CFG::$BEST_CHECKPOINT" \
    --system "foundation::$FOUND_CAL_CFG::$FOUNDATION_CKPT" \
    --preprocessed-dir "$BDSE_SPLIT_CACHE" --split val_tune --max-scenarios 1000 \
    --output-root "$SUITE_ROOT" --gpus "$GPUS" \
    --workers-per-gpu "$OPEN_LOOP_WORKERS_PER_GPU" --device cuda
fi
OPEN_LOOP_JSON="$SUITE_ROOT/candidate/metrics.json"
OPEN_LOOP_JSONL="$SUITE_ROOT/candidate/metrics.jsonl"
LOCAL_JSON="$SUITE_ROOT/local/metrics.json"
LOCAL_JSONL="$SUITE_ROOT/local/metrics.jsonl"
FOUND_JSON="$SUITE_ROOT/foundation/metrics.json"
FOUND_JSONL="$SUITE_ROOT/foundation/metrics.jsonl"

# ---------------------------------------------------------------------------
# 4. Three-tier gate with engineering-integrity checks.
# ---------------------------------------------------------------------------
PROTOCOL_GATE_MARKER="$OUT_ROOT/open_loop/.v61_protocol_gate_passed"
COMP_GATE_MARKER="$OUT_ROOT/open_loop/.v61_competitive_gate_passed"
GATE_REPORT="$OUT_ROOT/open_loop/v61_dehab_bfar_gate_report.json"
latency_gate_args=()
if [[ "$ENFORCE_LATENCY_BEFORE_CL" == "1" ]]; then latency_gate_args+=(--enforce-latency); fi
python -m bdse.tools.check_v61_dehab_bfar_dbap_gate \
  "$OPEN_LOOP_JSON" "$LOCAL_JSON" "$FOUND_JSON" \
  --candidate-jsonl "$OPEN_LOOP_JSONL" --local-control-jsonl "$LOCAL_JSONL" --foundation-control-jsonl "$FOUND_JSONL" \
  --train-log "$OUT_ROOT/train/bdse_v61_dehab_bfar_dbap.train_log.jsonl" \
  --train-config bdse/configs/v61_dehab_bfar_dbap_train_2gpu.yaml \
  --candidate-config "$CAND_CAL_CFG" \
  --local-control-config "$LOCAL_CAL_CFG" \
  --foundation-control-config "$FOUND_CAL_CFG" \
  --dual-calibration-json "$DUAL_CAL_JSON" \
  --report-json "$GATE_REPORT" --latency-target-ms "${V61_LATENCY_TARGET_MS:-500}" \
  "${latency_gate_args[@]}" 2>&1 | tee "$OUT_ROOT/logs/v61_dehab_bfar_dbap_gate.out"

read -r PROTOCOL_PASS MINIMUM_PASS COMPETITIVE_PASS < <(python - "$GATE_REPORT" <<'PY_GATE'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
print(int(bool(d.get('protocol_pass'))),int(bool(d.get('minimum_pass'))),int(bool(d.get('competitive_pass'))))
PY_GATE
)
[[ "$PROTOCOL_PASS" == "1" ]] || { echo "[v61] protocol gate failed; closed loop is invalid" >&2; exit 3; }
printf 'passed\n' > "$PROTOCOL_GATE_MARKER"
if [[ "$COMPETITIVE_PASS" == "1" ]]; then printf 'passed\n' > "$COMP_GATE_MARKER"; else rm -f "$COMP_GATE_MARKER"; fi

# ---------------------------------------------------------------------------
# 5. Paired CL20.  A minimum-gate failure no longer hides closed-loop evidence:
#    with a valid protocol, RUN_DIAGNOSTIC_CL20_ON_GATE_FAIL=1 runs a clearly
#    labelled diagnostic CL20.  It is not a publication PASS.
# ---------------------------------------------------------------------------
if [[ "$RUN_CLOSED_LOOP_AFTER_GATE" == "1" && ( "$MINIMUM_PASS" == "1" || "$RUN_DIAGNOSTIC_CL20_ON_GATE_FAIL" == "1" ) ]]; then
  mkdir -p "$OUT_ROOT/closed_loop"
  if [[ "$MINIMUM_PASS" != "1" ]]; then
    echo "[v61] minimum gate failed; running paired CL20 as diagnostic evidence only"
    printf 'diagnostic\n' > "$OUT_ROOT/closed_loop/.diagnostic_cl20"
  fi
  : "${NUPLAN_ROOT:?Set NUPLAN_ROOT before the gated closed-loop stage}"
  BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" CL_TOKEN_SPLIT=val_tune \
  GPUS="$GPUS" OUT_ROOT="$OUT_ROOT" RUN_MODE=cl20 \
  V61_CKPT="$BEST_CHECKPOINT" EVAL_CONFIG="$CAND_CAL_CFG" \
  bash run_v61_dehab_bfar_dbap.sh

  BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" CL_TOKEN_SPLIT=val_tune \
  GPUS="$GPUS" OUT_ROOT="$LOCAL_CONTROL_ROOT" RUN_MODE=cl20 \
  V61_CKPT="$BEST_CHECKPOINT" EVAL_CONFIG="$LOCAL_CAL_CFG" \
  bash run_v61_dehab_bfar_dbap.sh

  BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" CL_TOKEN_SPLIT=val_tune \
  GPUS="$GPUS" OUT_ROOT="$FOUNDATION_CONTROL_ROOT" RUN_MODE=cl20 \
  V61_CKPT="$FOUNDATION_CKPT" EVAL_CONFIG="$FOUND_CAL_CFG" \
  bash run_v61_dehab_bfar_dbap.sh

  # Hard closed-loop attribution check: all compared methods must use exactly
  # the same token list.  Persist the proof next to the candidate results.
  python - \
    "$OUT_ROOT/closed_loop/v61_dehab_bfar_dbap_20/closed_loop_combined_summary.json" \
    "$LOCAL_CONTROL_ROOT/closed_loop/v61_dehab_bfar_dbap_20/closed_loop_combined_summary.json" \
    "$FOUNDATION_CONTROL_ROOT/closed_loop/v61_dehab_bfar_dbap_20/closed_loop_combined_summary.json" \
    "$OUT_ROOT/closed_loop/v61_three_way_cl20_protocol.json" <<'PY_CL_PROTOCOL'
import json, sys
from pathlib import Path
paths = [Path(x) for x in sys.argv[1:4]]
rows = [json.loads(p.read_text()) for p in paths]
sha = [r.get("scenario_token_sha256") for r in rows]
counts = [int(r.get("scenario_count", -1)) for r in rows]
passed = len(set(sha)) == 1 and None not in sha and len(set(counts)) == 1 and counts[0] == 20
report = {
    "protocol_pass": passed,
    "scenario_token_sha256": sha,
    "scenario_count": counts,
    "summaries": [str(p) for p in paths],
}
out = Path(sys.argv[4])
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
if not passed:
    raise SystemExit(f"three-way CL20 token protocol mismatch: {report}")
print(json.dumps(report, sort_keys=True))
PY_CL_PROTOCOL

  if [[ "$RUN_CL100_AFTER_CL20" == "1" && "$COMPETITIVE_PASS" == "1" ]]; then
    BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" CL_TOKEN_SPLIT=val_tune \
    GPUS="$GPUS" OUT_ROOT="$OUT_ROOT" RUN_MODE=cl100 \
    V61_CKPT="$BEST_CHECKPOINT" EVAL_CONFIG="$CAND_CAL_CFG" \
    bash run_v61_dehab_bfar_dbap.sh
  elif [[ "$RUN_CL100_AFTER_CL20" == "1" ]]; then
    echo "[v61] CL100 requested but competitive gate failed; run paired CL20 first and inspect $GATE_REPORT"
  fi
else
  echo "[v61] closed-loop skipped (RUN_CLOSED_LOOP_AFTER_GATE=$RUN_CLOSED_LOOP_AFTER_GATE, minimum_pass=$MINIMUM_PASS, diagnostic=$RUN_DIAGNOSTIC_CL20_ON_GATE_FAIL)"
fi

# ---------------------------------------------------------------------------
# 7. Official test protocol.  Build integrity/leakage is a hard gate; natural
#    val->test distribution shift is reported, not "repaired".  A partial cache
#    may be used once as a frozen-checkpoint stress test, but never for tuning.
# ---------------------------------------------------------------------------
# TEST_DIAGNOSTICS=/path/to/diagnostics_test.json \
# VAL_DIAGNOSTICS=/path/to/diagnostics_val.json \
# TEST_CACHE="$BDSE_TEST_CACHE" TRAIN_CACHE="$BDSE_TRAIN_CACHE" \
# VAL_CACHE="$BDSE_VAL_CACHE_ORIGINAL" OUT="$OUT_ROOT/test_readiness.json" \
# bash CHECK_PARTIAL_TEST_SET.sh
#
# After preprocessing is complete, rerun BUILD_MATCHED_TEST_SET.sh with
# ALLOW_INCOMPLETE=0 and the train/val caches supplied. Only then report final
# test metrics. Do not use any test result for checkpoint or threshold choices.
#
# CUDA_VISIBLE_DEVICES=0 python -m bdse.experiments.evaluate_open_loop \
#   --config "$CAND_CAL_CFG" \
#   --checkpoint "$OUT_ROOT/train/bdse_v61_dehab_bfar_dbap.best.pt" \
#   --split public_set_test --preprocessed-dir "$BDSE_TEST_CACHE" --device cuda \
#   --output "$OUT_ROOT/open_loop/open_loop_v61_dehab_bfar_dbap_test.json" \
#   --per-sample-output "$OUT_ROOT/open_loop/open_loop_v61_dehab_bfar_dbap_test.jsonl"
