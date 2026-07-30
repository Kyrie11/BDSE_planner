#!/usr/bin/env bash
set -euo pipefail

# v51.1 checkpoint-independent pipeline.  Resolve every relative path from the
# checked-in script directory so a stale working directory cannot silently run a
# different config or helper script.
PIPELINE_VERSION="v51.0-foundation-anchored"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "[v51] pipeline_version=$PIPELINE_VERSION script=$SCRIPT_DIR/$(basename "$0")"

# Mandatory data paths.
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE_ORIGINAL="${BDSE_VAL_CACHE_ORIGINAL:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export BDSE_SPLIT_CACHE="${BDSE_SPLIT_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v51_split}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_v2}"
export OUT_ROOT="${OUT_ROOT:-outputs_v51_far_dbap_2gpu_v1}"
export FOUNDATION_ROOT="${FOUNDATION_ROOT:-$OUT_ROOT/foundation_anchor}"
export FOUNDATION_CONFIG="${FOUNDATION_CONFIG:-bdse/configs/v51_strong_foundation_anchor_2gpu.yaml}"

# There is deliberately no hard-coded outputs_v30 dependency.  FOUNDATION_CKPT
# is the canonical variable; V30_CKPT_IN remains only as a backwards-compatible
# alias consumed by run_v51_far_dbap.sh.
export V30_CKPT_IN="${V30_CKPT_IN:-}"
export FOUNDATION_CKPT="${FOUNDATION_CKPT:-$V30_CKPT_IN}"
export FOUNDATION_POLICY="${FOUNDATION_POLICY:-auto}"  # auto | rebuild | recover | explicit
export FOUNDATION_SEARCH_ROOT="${FOUNDATION_SEARCH_ROOT:-.}"
export RECOVER_SAFE_FOUNDATION_COPIES="${RECOVER_SAFE_FOUNDATION_COPIES:-1}"
export REBUILD_FOUNDATION_IF_MISSING="${REBUILD_FOUNDATION_IF_MISSING:-1}"
export ALLOW_ALGORITHM_CHECKPOINT_INIT="${ALLOW_ALGORITHM_CHECKPOINT_INIT:-0}"
export V51_INIT_MODE="${V51_INIT_MODE:-warm_start}"
export EXACT_SELECTOR_WORKERS_PER_RANK="${EXACT_SELECTOR_WORKERS_PER_RANK:-4}"
export EXACT_SELECTOR_CPU_BACKEND="${EXACT_SELECTOR_CPU_BACKEND:-process}"
export GPUS="${GPUS:-0,1}"
export FOUNDATION_CONTROL_CONFIG="${FOUNDATION_CONTROL_CONFIG:-bdse/configs/v51_far_dbap_anchor_control_cl.yaml}"
export LOCAL_CONTROL_CONFIG="${LOCAL_CONTROL_CONFIG:-bdse/configs/v51_far_dbap_local_control_cl.yaml}"
export CONTROL_CONFIG="${CONTROL_CONFIG:-$FOUNDATION_CONTROL_CONFIG}"  # compatibility alias
export CONTROL_CKPT="${CONTROL_CKPT:-}"
export FOUNDATION_CONTROL_ROOT="${FOUNDATION_CONTROL_ROOT:-$OUT_ROOT/control_foundation_matched}"
export LOCAL_CONTROL_ROOT="${LOCAL_CONTROL_ROOT:-$OUT_ROOT/control_local_same_checkpoint}"
export PIPELINE_DETACH="${PIPELINE_DETACH:-1}"
export PIPELINE_FORCE="${PIPELINE_FORCE:-0}"
export RUN_CLOSED_LOOP_AFTER_GATE="${RUN_CLOSED_LOOP_AFTER_GATE:-1}"
export RUN_CL100_AFTER_CL20="${RUN_CL100_AFTER_CL20:-0}"
# Keep algorithmic closed-loop validation separate from the deployment-latency
# claim. Set to 1 only when CL should be blocked by the real-time target.
export ENFORCE_LATENCY_BEFORE_CL="${ENFORCE_LATENCY_BEFORE_CL:-0}"

case "$FOUNDATION_POLICY" in
  auto|rebuild|recover|explicit) ;;
  *) echo "FOUNDATION_POLICY must be auto, rebuild, recover, or explicit" >&2; exit 2 ;;
esac

mkdir -p "$OUT_ROOT/logs"

# Detach the complete pipeline, not only the training child.  Detaching only
# run_v51_far_dbap.sh lets this parent continue immediately into calibration while
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
  echo "[v51] detached pipeline started pid=$pipeline_pid log=$pipeline_log"
  exit 0
fi

PIPELINE_LOCK_DIR="$OUT_ROOT/.v51_pipeline.lock"
if ! mkdir "$PIPELINE_LOCK_DIR" 2>/dev/null; then
  owner="$(cat "$PIPELINE_LOCK_DIR/pid" 2>/dev/null || echo unknown)"
  echo "A v51 pipeline is already using OUT_ROOT=$OUT_ROOT (pid=$owner)" >&2
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
  [[ -s "$OUT_ROOT/train/bdse_v51_far_dbap.train_log.jsonl"      && -s "$OUT_ROOT/train/bdse_v51_far_dbap.pt"      && -s "$OUT_ROOT/train/bdse_v51_far_dbap.best.pt"      && "$OUT_ROOT/train/bdse_v51_far_dbap.pt" -nt bdse/configs/v51_far_dbap_train_2gpu.yaml ]]
}

# ---------------------------------------------------------------------------
# 0. Create a paper-grade validation protocol. The split is by nuPlan log group,
#    not random scenario row, to reduce temporal/scene leakage.
# ---------------------------------------------------------------------------
if [[ "$PIPELINE_FORCE" != "1" \
      && -s "$BDSE_SPLIT_CACHE/calibration_split_provenance.json" \
      && -s "$BDSE_SPLIT_CACHE/val_tune/manifest.jsonl" \
      && -s "$BDSE_SPLIT_CACHE/val_calib/manifest.jsonl" ]]; then
  echo "[v51] stage 0 already complete: reuse group-disjoint split"
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
#   3) a conservatively verified retained v30 copy from outputs_v40..v51;
#   4) a fresh current-code v30-compatible foundation rebuild.
#
# Later D3CE/DBCE/DBAP/AOCC checkpoints are inventoried but rejected by default.
# They are algorithm-specific initializations and would confound the v51 paper
# comparison.  ALLOW_ALGORITHM_CHECKPOINT_INIT=1 is an explicit transfer-ablation
# escape hatch, never the default main-run protocol.
# ---------------------------------------------------------------------------
FOUNDATION_SOURCE=""
FOUNDATION_INVENTORY="$OUT_ROOT/provenance/foundation_checkpoint_inventory.json"
mkdir -p "$OUT_ROOT/provenance"

resolve_foundation_checkpoint() {
  local rebuilt_best="$FOUNDATION_ROOT/train/bdse_v51_foundation_anchor.best.pt"
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
    echo "[v51] ignore missing inherited foundation path: $FOUNDATION_CKPT"
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
      echo "[v51] WARNING: algorithm-specific checkpoint recovery enabled; use only as a transfer ablation" >&2
    fi
    selected="$(python -m bdse.tools.resolve_foundation_checkpoint       --config bdse/configs/v51_far_dbap_train_2gpu.yaml       --search-root "$FOUNDATION_SEARCH_ROOT"       --output-json "$FOUNDATION_INVENTORY"       --print-selected       "${allow_args[@]}")"
    if [[ -n "$selected" && -s "$selected" ]]; then
      FOUNDATION_CKPT="$selected"
      if [[ "$ALLOW_ALGORITHM_CHECKPOINT_INIT" == "1" ]]; then
        FOUNDATION_SOURCE="recovered_algorithm_transfer_ablation"
      else
        FOUNDATION_SOURCE="recovered_verified_v30_copy"
      fi
      echo "[v51] recovered foundation checkpoint=$FOUNDATION_CKPT source=$FOUNDATION_SOURCE"
      return 0
    fi
  else
    # Still write an inventory report when recovery is disabled, because it is
    # useful for documenting why retained v40-v49 checkpoints were not used.
    python -m bdse.tools.resolve_foundation_checkpoint       --config bdse/configs/v51_far_dbap_train_2gpu.yaml       --search-root "$FOUNDATION_SEARCH_ROOT"       --output-json "$FOUNDATION_INVENTORY" >/dev/null || true
  fi

  if [[ "$FOUNDATION_POLICY" == "recover" ]]; then
    echo "FOUNDATION_POLICY=recover found no conservatively safe v30 copy; see $FOUNDATION_INVENTORY" >&2
    exit 2
  fi
  if [[ "$REBUILD_FOUNDATION_IF_MISSING" != "1" ]]; then
    echo "No usable foundation checkpoint and rebuilding is disabled" >&2
    exit 2
  fi

  echo "[v51] no safe historical foundation found; rebuilding from random initialization"
  DETACH=0   GPUS="$GPUS"   INIT_MODE=scratch   V30_CKPT_IN=   OUT_ROOT="$FOUNDATION_ROOT"   FOUNDATION_OUT_ROOT="$FOUNDATION_ROOT"   FOUNDATION_CONFIG="$FOUNDATION_CONFIG"   RUN_MODE=foundation   AUTO_RESUME=1   VAL_SPLIT=val_tune   VAL_SCENARIOS=1000   BATCH_SIZE_PER_GPU=4   NUM_WORKERS_PER_GPU=4   PREFETCH_FACTOR=2   SAVE_EVERY_N_STEPS=2000   bash run_v51_far_dbap.sh

  [[ -s "$rebuilt_best" ]] || {
    echo "Foundation rebuild did not produce $rebuilt_best" >&2
    exit 2
  }
  FOUNDATION_CKPT="$rebuilt_best"
  FOUNDATION_SOURCE="rebuilt_current_code"
}

if [[ "$V51_INIT_MODE" == "warm_start" ]]; then
  resolve_foundation_checkpoint
  [[ -s "$FOUNDATION_CKPT" ]] || { echo "Resolved foundation is missing: $FOUNDATION_CKPT" >&2; exit 2; }
  V30_CKPT_IN="$FOUNDATION_CKPT"  # compatibility alias for the inner launcher
  if [[ -z "$CONTROL_CKPT" ]]; then
    CONTROL_CKPT="$FOUNDATION_CKPT"
  fi
else
  echo "[v51] V51_INIT_MODE=scratch is an ablation, not the paper main run."
  [[ -n "$CONTROL_CKPT" && -s "$CONTROL_CKPT" ]] || {
    echo "Scratch v51 still needs an independently trained matched CONTROL_CKPT for a paired gate" >&2
    exit 2
  }
  FOUNDATION_CKPT="$CONTROL_CKPT"
  V30_CKPT_IN="$FOUNDATION_CKPT"
  FOUNDATION_SOURCE="scratch_ablation_matched_control"
fi
export FOUNDATION_CKPT V30_CKPT_IN CONTROL_CKPT

echo "[v51] foundation_source=$FOUNDATION_SOURCE"
echo "[v51] foundation_checkpoint=$FOUNDATION_CKPT"
echo "[v51] control_checkpoint=$CONTROL_CKPT"

FOUNDATION_PROVENANCE="$OUT_ROOT/provenance/foundation_checkpoint.json"
if ! is_fresh "$FOUNDATION_PROVENANCE" "$FOUNDATION_CKPT" "$FOUNDATION_CONFIG"; then
  python -m bdse.tools.write_checkpoint_provenance     --checkpoint "$FOUNDATION_CKPT"     --config "$FOUNDATION_CONFIG"     --train-cache "$BDSE_TRAIN_CACHE"     --val-cache "$BDSE_SPLIT_CACHE"     --source "$FOUNDATION_SOURCE"     --output "$FOUNDATION_PROVENANCE"
fi

# ---------------------------------------------------------------------------
# 0.75 Foundation quality gate. A weak foundation makes residual results
#      uninterpretable, so the algorithm stage is blocked before any v51 update.
# ---------------------------------------------------------------------------
FOUNDATION_QUALITY_ROOT="$FOUNDATION_ROOT/quality"
FOUNDATION_QUALITY_JSON="$FOUNDATION_QUALITY_ROOT/open_loop.json"
FOUNDATION_QUALITY_JSONL="$FOUNDATION_QUALITY_ROOT/open_loop.jsonl"
mkdir -p "$FOUNDATION_QUALITY_ROOT"
if is_fresh "$FOUNDATION_QUALITY_JSON" "$FOUNDATION_CKPT" "$FOUNDATION_CONTROL_CONFIG" \
   && is_fresh "$FOUNDATION_QUALITY_JSONL" "$FOUNDATION_CKPT" "$FOUNDATION_CONTROL_CONFIG"; then
  echo "[v51] foundation quality replay already complete"
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
python -m bdse.tools.check_v51_foundation_quality "$FOUNDATION_QUALITY_JSON" \
  --min-teacher-match "${FOUNDATION_MIN_TEACHER_MATCH:-0.22}" \
  --min-full-interface-match "${FOUNDATION_MIN_FULL_INTERFACE_MATCH:-0.45}" \
  --min-pair-full-match "${FOUNDATION_MIN_PAIR_FULL_MATCH:-0.25}" \
  --min-winner-rival-sign "${FOUNDATION_MIN_WINNER_SIGN:-0.60}" \
  --min-near-tie-sign "${FOUNDATION_MIN_NEAR_SIGN:-0.45}" \
  --min-evidence-sufficiency "${FOUNDATION_MIN_SUFFICIENCY:-0.06}" \
  2>&1 | tee "$OUT_ROOT/logs/v51_foundation_quality_gate.out"

# ---------------------------------------------------------------------------
# 1. Main v51 training. Base/local encoders are frozen by config; only the
#    reset residual/uncertainty and proposal-family heads are trainable.
# ---------------------------------------------------------------------------
if training_complete; then
  echo "[v51] stage 1 already complete: reuse final/best checkpoints"
else
  DETACH=0 \
  GPUS="$GPUS" \
  V30_CKPT_IN="$V30_CKPT_IN" \
  INIT_MODE="$V51_INIT_MODE" \
  TRAIN_CONFIG=bdse/configs/v51_far_dbap_train_2gpu.yaml \
  EXACT_SELECTOR_WORKERS_PER_RANK="$EXACT_SELECTOR_WORKERS_PER_RANK" \
  EXACT_SELECTOR_CPU_BACKEND="$EXACT_SELECTOR_CPU_BACKEND" \
  OUT_ROOT="$OUT_ROOT" \
  RUN_MODE=train \
  AUTO_RESUME=0 \
  VAL_SPLIT=val_tune \
  OPEN_LOOP_SPLIT=val_tune \
  BATCH_SIZE_PER_GPU=4 \
  NUM_WORKERS_PER_GPU=4 \
  PREFETCH_FACTOR=2 \
  SAVE_EVERY_N_STEPS=2000 \
  SELECTOR_SCENES_PER_RANK=0 \
  SELECTOR_EVERY_N_STEPS=1 \
  EXACT_DISTILL_SCENES_PER_RANK=0 \
  EXACT_DISTILL_EVERY_N_STEPS=1 \
  VAL_SCENARIOS=1000 \
  VAL_EVERY_N_EPOCHS=1 \
  VAL_DENSE_DIAGNOSTIC=1 \
  OPEN_LOOP_MAX_SCENARIOS=1000 \
  bash run_v51_far_dbap.sh
fi

BEST_CHECKPOINT="$OUT_ROOT/train/bdse_v51_far_dbap.best.pt"
[[ -s "$BEST_CHECKPOINT" ]] || { echo "Missing v51 best checkpoint: $BEST_CHECKPOINT" >&2; exit 2; }

# ---------------------------------------------------------------------------
# 2. Independently calibrate all three compared systems on val_calib:
#      candidate, exact same-checkpoint local control, immutable foundation.
# ---------------------------------------------------------------------------
mkdir -p "$OUT_ROOT/calibration"
CALIBRATION_PROVENANCE="$BDSE_SPLIT_CACHE/calibration_split_provenance.json"
calibrate_system() {
  local label="$1" config_in="$2" checkpoint="$3" json_out="$4" config_out="$5"
  if ! is_fresh "$json_out" "$checkpoint" "$config_in" "$CALIBRATION_PROVENANCE"; then
    python -m bdse.tools.calibrate_v48_adverse_bounds \
      --config "$config_in" --checkpoint "$checkpoint" \
      --preprocessed-dir "$BDSE_SPLIT_CACHE" --split val_calib \
      --max-scenarios 5000 --device cuda --alpha 0.05 --beta 1.0 --prior-radius 0.02 \
      --provenance-json "$CALIBRATION_PROVENANCE" --output "$json_out"
  else
    echo "[v51] reuse $label calibration"
  fi
  if ! is_fresh "$config_out" "$json_out" "$config_in"; then
    python -m bdse.tools.apply_v48_calibration \
      --config-in "$config_in" --calibration-json "$json_out" --config-out "$config_out"
  fi
}

CAND_CAL_JSON="$OUT_ROOT/calibration/candidate_adverse.json"
CAND_CAL_CFG="$OUT_ROOT/calibration/v51_far_dbap_candidate_calibrated.yaml"
LOCAL_CAL_JSON="$OUT_ROOT/calibration/local_control_adverse.json"
LOCAL_CAL_CFG="$OUT_ROOT/calibration/v51_far_dbap_local_control_calibrated.yaml"
FOUND_CAL_JSON="$OUT_ROOT/calibration/foundation_control_adverse.json"
FOUND_CAL_CFG="$OUT_ROOT/calibration/v51_far_dbap_foundation_control_calibrated.yaml"
calibrate_system candidate bdse/configs/v51_far_dbap_cl.yaml "$BEST_CHECKPOINT" "$CAND_CAL_JSON" "$CAND_CAL_CFG"
calibrate_system local-control "$LOCAL_CONTROL_CONFIG" "$BEST_CHECKPOINT" "$LOCAL_CAL_JSON" "$LOCAL_CAL_CFG"
calibrate_system foundation-control "$FOUNDATION_CONTROL_CONFIG" "$FOUNDATION_CKPT" "$FOUND_CAL_JSON" "$FOUND_CAL_CFG"

# ---------------------------------------------------------------------------
# 3. Deterministic, token-paired val_tune replay for all three systems.
# ---------------------------------------------------------------------------
OPEN_LOOP_JSON="$OUT_ROOT/open_loop/open_loop_v51_far_dbap.json"
OPEN_LOOP_JSONL="$OUT_ROOT/open_loop/open_loop_v51_far_dbap.jsonl"
if ! is_fresh "$OPEN_LOOP_JSON" "$BEST_CHECKPOINT" "$CAND_CAL_CFG" \
   || ! is_fresh "$OPEN_LOOP_JSONL" "$BEST_CHECKPOINT" "$CAND_CAL_CFG"; then
  GPUS="$GPUS" OUT_ROOT="$OUT_ROOT" RUN_MODE=open_loop \
  V51_CKPT="$BEST_CHECKPOINT" EVAL_CONFIG="$CAND_CAL_CFG" \
  VAL_SPLIT=val_tune OPEN_LOOP_SPLIT=val_tune OPEN_LOOP_MAX_SCENARIOS=1000 \
  bash run_v51_far_dbap.sh
fi

mkdir -p "$LOCAL_CONTROL_ROOT/open_loop" "$FOUNDATION_CONTROL_ROOT/open_loop"
LOCAL_JSON="$LOCAL_CONTROL_ROOT/open_loop/control.json"
LOCAL_JSONL="$LOCAL_CONTROL_ROOT/open_loop/control.jsonl"
FOUND_JSON="$FOUNDATION_CONTROL_ROOT/open_loop/control.json"
FOUND_JSONL="$FOUNDATION_CONTROL_ROOT/open_loop/control.jsonl"
evaluate_control() {
  local label="$1" config="$2" checkpoint="$3" json_out="$4" jsonl_out="$5"
  if is_fresh "$json_out" "$config" "$checkpoint" && is_fresh "$jsonl_out" "$config" "$checkpoint"; then
    echo "[v51] reuse $label replay"
  else
    CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python -m bdse.experiments.evaluate_open_loop \
      --config "$config" --checkpoint "$checkpoint" \
      --split val_tune --preprocessed-dir "$BDSE_SPLIT_CACHE" --max-scenarios 1000 \
      --device cuda --output "$json_out" --per-sample-output "$jsonl_out"
  fi
}
evaluate_control local-control "$LOCAL_CAL_CFG" "$BEST_CHECKPOINT" "$LOCAL_JSON" "$LOCAL_JSONL"
evaluate_control foundation-control "$FOUND_CAL_CFG" "$FOUNDATION_CKPT" "$FOUND_JSON" "$FOUND_JSONL"

# ---------------------------------------------------------------------------
# 4. Three-way causal gate. Closed loop starts only when residual-only and
#    total-vs-foundation quality checks pass together with AOCC correctness.
#    Latency is always reported, but is a separate engineering gate by default
#    so a known runtime bottleneck cannot suppress algorithmic CL attribution.
# ---------------------------------------------------------------------------
GATE_MARKER="$OUT_ROOT/open_loop/.v51_far_dbap_gate_passed"
if is_fresh "$GATE_MARKER" "$OPEN_LOOP_JSON" "$OPEN_LOOP_JSONL" \
    "$LOCAL_JSON" "$LOCAL_JSONL" "$FOUND_JSON" "$FOUND_JSONL" \
    "$OUT_ROOT/train/bdse_v51_far_dbap.train_log.jsonl" \
    bdse/tools/check_v51_far_dbap_gate.py; then
  echo "[v51] three-way gate previously passed"
else
  latency_gate_args=()
  if [[ "$ENFORCE_LATENCY_BEFORE_CL" == "1" ]]; then
    latency_gate_args+=(--enforce-latency)
  fi
  python -m bdse.tools.check_v51_far_dbap_gate \
    "$OPEN_LOOP_JSON" "$LOCAL_JSON" "$FOUND_JSON" \
    --candidate-jsonl "$OPEN_LOOP_JSONL" \
    --local-control-jsonl "$LOCAL_JSONL" \
    --foundation-control-jsonl "$FOUND_JSONL" \
    --train-log "$OUT_ROOT/train/bdse_v51_far_dbap.train_log.jsonl" \
    --latency-target-ms "${V51_LATENCY_TARGET_MS:-500}" \
    "${latency_gate_args[@]}" \
    2>&1 | tee "$OUT_ROOT/logs/v51_far_dbap_gate.out"
  printf 'passed\n' >"$GATE_MARKER"
fi

# ---------------------------------------------------------------------------
# 5. After PASS: paired CL20 for candidate, local same-checkpoint control, and
#    immutable foundation control on the same deterministic token sequence.
# ---------------------------------------------------------------------------
if [[ "$RUN_CLOSED_LOOP_AFTER_GATE" == "1" ]]; then
  : "${NUPLAN_ROOT:?Set NUPLAN_ROOT before the gated closed-loop stage}"
  BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" CL_TOKEN_SPLIT=val_tune \
  GPUS="$GPUS" OUT_ROOT="$OUT_ROOT" RUN_MODE=cl20 \
  V51_CKPT="$BEST_CHECKPOINT" EVAL_CONFIG="$CAND_CAL_CFG" \
  bash run_v51_far_dbap.sh

  BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" CL_TOKEN_SPLIT=val_tune \
  GPUS="$GPUS" OUT_ROOT="$LOCAL_CONTROL_ROOT" RUN_MODE=cl20 \
  V51_CKPT="$BEST_CHECKPOINT" EVAL_CONFIG="$LOCAL_CAL_CFG" \
  bash run_v51_far_dbap.sh

  BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" CL_TOKEN_SPLIT=val_tune \
  GPUS="$GPUS" OUT_ROOT="$FOUNDATION_CONTROL_ROOT" RUN_MODE=cl20 \
  V51_CKPT="$FOUNDATION_CKPT" EVAL_CONFIG="$FOUND_CAL_CFG" \
  bash run_v51_far_dbap.sh

  if [[ "$RUN_CL100_AFTER_CL20" == "1" ]]; then
    BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" CL_TOKEN_SPLIT=val_tune \
    GPUS="$GPUS" OUT_ROOT="$OUT_ROOT" RUN_MODE=cl100 \
    V51_CKPT="$BEST_CHECKPOINT" EVAL_CONFIG="$CAND_CAL_CFG" \
    bash run_v51_far_dbap.sh
  fi
else
  echo "[v51] gate passed; closed-loop skipped because RUN_CLOSED_LOOP_AFTER_GATE=0"
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
#   --config "$OUT_ROOT/calibration/v51_bdse_dbap_ri_cl_calibrated.yaml" \
#   --checkpoint "$OUT_ROOT/train/bdse_v51_far_dbap.best.pt" \
#   --split public_set_test --preprocessed-dir "$BDSE_TEST_CACHE" --device cuda \
#   --output "$OUT_ROOT/open_loop/open_loop_v51_far_dbap_test.json" \
#   --per-sample-output "$OUT_ROOT/open_loop/open_loop_v51_far_dbap_test.jsonl"
