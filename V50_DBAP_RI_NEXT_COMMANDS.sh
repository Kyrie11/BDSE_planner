#!/usr/bin/env bash
set -euo pipefail

# v50.1 checkpoint-independent pipeline.  Resolve every relative path from the
# checked-in script directory so a stale working directory cannot silently run a
# different config or helper script.
PIPELINE_VERSION="v50.1-checkpoint-independent"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "[v50] pipeline_version=$PIPELINE_VERSION script=$SCRIPT_DIR/$(basename "$0")"

# Mandatory data paths.
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE_ORIGINAL="${BDSE_VAL_CACHE_ORIGINAL:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export BDSE_SPLIT_CACHE="${BDSE_SPLIT_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v50_split}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_v2}"
export OUT_ROOT="${OUT_ROOT:-outputs_v50_dbap_ri_checkpoint_independent_2gpu_v1}"
export FOUNDATION_ROOT="${FOUNDATION_ROOT:-$OUT_ROOT/foundation_v30_compatible}"
export FOUNDATION_CONFIG="${FOUNDATION_CONFIG:-bdse/configs/v50_rebuild_v30_from_scratch_2gpu.yaml}"

# There is deliberately no hard-coded outputs_v30 dependency.  FOUNDATION_CKPT
# is the canonical variable; V30_CKPT_IN remains only as a backwards-compatible
# alias consumed by run_v50_dbap_ri.sh.
export V30_CKPT_IN="${V30_CKPT_IN:-}"
export FOUNDATION_CKPT="${FOUNDATION_CKPT:-$V30_CKPT_IN}"
export FOUNDATION_POLICY="${FOUNDATION_POLICY:-auto}"  # auto | rebuild | recover | explicit
export FOUNDATION_SEARCH_ROOT="${FOUNDATION_SEARCH_ROOT:-.}"
export RECOVER_SAFE_FOUNDATION_COPIES="${RECOVER_SAFE_FOUNDATION_COPIES:-1}"
export REBUILD_FOUNDATION_IF_MISSING="${REBUILD_FOUNDATION_IF_MISSING:-1}"
export ALLOW_ALGORITHM_CHECKPOINT_INIT="${ALLOW_ALGORITHM_CHECKPOINT_INIT:-0}"
export V50_INIT_MODE="${V50_INIT_MODE:-warm_start}"
export EXACT_SELECTOR_WORKERS_PER_RANK="${EXACT_SELECTOR_WORKERS_PER_RANK:-4}"
export EXACT_SELECTOR_CPU_BACKEND="${EXACT_SELECTOR_CPU_BACKEND:-process}"
export GPUS="${GPUS:-0,1}"
export CONTROL_CONFIG="${CONTROL_CONFIG:-bdse/configs/v43_bdse_mars_control_fast_cl.yaml}"
export CONTROL_CKPT="${CONTROL_CKPT:-}"
export CONTROL_ROOT="${CONTROL_ROOT:-$OUT_ROOT/control_foundation_matched}"
export PIPELINE_DETACH="${PIPELINE_DETACH:-1}"
export PIPELINE_FORCE="${PIPELINE_FORCE:-0}"
export RUN_CLOSED_LOOP_AFTER_GATE="${RUN_CLOSED_LOOP_AFTER_GATE:-1}"
export RUN_CL100_AFTER_CL20="${RUN_CL100_AFTER_CL20:-0}"

case "$FOUNDATION_POLICY" in
  auto|rebuild|recover|explicit) ;;
  *) echo "FOUNDATION_POLICY must be auto, rebuild, recover, or explicit" >&2; exit 2 ;;
esac

mkdir -p "$OUT_ROOT/logs"

# Detach the complete pipeline, not only the training child.  Detaching only
# run_v50_dbap_ri.sh lets this parent continue immediately into calibration while
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
  echo "[v50] detached pipeline started pid=$pipeline_pid log=$pipeline_log"
  exit 0
fi

PIPELINE_LOCK_DIR="$OUT_ROOT/.v50_pipeline.lock"
if ! mkdir "$PIPELINE_LOCK_DIR" 2>/dev/null; then
  owner="$(cat "$PIPELINE_LOCK_DIR/pid" 2>/dev/null || echo unknown)"
  echo "A v50 pipeline is already using OUT_ROOT=$OUT_ROOT (pid=$owner)" >&2
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
  [[ -s "$OUT_ROOT/train/bdse_v50_dbap_ri.train_log.jsonl"      && -s "$OUT_ROOT/train/bdse_v50_dbap_ri.pt"      && -s "$OUT_ROOT/train/bdse_v50_dbap_ri.best.pt"      && "$OUT_ROOT/train/bdse_v50_dbap_ri.pt" -nt bdse/configs/v50_bdse_dbap_ri_train_2gpu.yaml ]]
}

# ---------------------------------------------------------------------------
# 0. Create a paper-grade validation protocol. The split is by nuPlan log group,
#    not random scenario row, to reduce temporal/scene leakage.
# ---------------------------------------------------------------------------
if [[ "$PIPELINE_FORCE" != "1" \
      && -s "$BDSE_SPLIT_CACHE/calibration_split_provenance.json" \
      && -s "$BDSE_SPLIT_CACHE/val_tune/manifest.jsonl" \
      && -s "$BDSE_SPLIT_CACHE/val_calib/manifest.jsonl" ]]; then
  echo "[v50] stage 0 already complete: reuse group-disjoint split"
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
#   3) a conservatively verified retained v30 copy from outputs_v40..v50;
#   4) a fresh current-code v30-compatible foundation rebuild.
#
# Later D3CE/DBCE/DBAP/AOCC checkpoints are inventoried but rejected by default.
# They are algorithm-specific initializations and would confound the v50 paper
# comparison.  ALLOW_ALGORITHM_CHECKPOINT_INIT=1 is an explicit transfer-ablation
# escape hatch, never the default main-run protocol.
# ---------------------------------------------------------------------------
FOUNDATION_SOURCE=""
FOUNDATION_INVENTORY="$OUT_ROOT/provenance/foundation_checkpoint_inventory.json"
mkdir -p "$OUT_ROOT/provenance"

resolve_foundation_checkpoint() {
  local rebuilt_best="$FOUNDATION_ROOT/train/bdse_v30_pmvrbsr_rebuilt.best.pt"
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
    echo "[v50] ignore missing inherited foundation path: $FOUNDATION_CKPT"
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
      echo "[v50] WARNING: algorithm-specific checkpoint recovery enabled; use only as a transfer ablation" >&2
    fi
    selected="$(python -m bdse.tools.resolve_foundation_checkpoint       --config bdse/configs/v50_bdse_dbap_ri_train_2gpu.yaml       --search-root "$FOUNDATION_SEARCH_ROOT"       --output-json "$FOUNDATION_INVENTORY"       --print-selected       "${allow_args[@]}")"
    if [[ -n "$selected" && -s "$selected" ]]; then
      FOUNDATION_CKPT="$selected"
      if [[ "$ALLOW_ALGORITHM_CHECKPOINT_INIT" == "1" ]]; then
        FOUNDATION_SOURCE="recovered_algorithm_transfer_ablation"
      else
        FOUNDATION_SOURCE="recovered_verified_v30_copy"
      fi
      echo "[v50] recovered foundation checkpoint=$FOUNDATION_CKPT source=$FOUNDATION_SOURCE"
      return 0
    fi
  else
    # Still write an inventory report when recovery is disabled, because it is
    # useful for documenting why retained v40-v49 checkpoints were not used.
    python -m bdse.tools.resolve_foundation_checkpoint       --config bdse/configs/v50_bdse_dbap_ri_train_2gpu.yaml       --search-root "$FOUNDATION_SEARCH_ROOT"       --output-json "$FOUNDATION_INVENTORY" >/dev/null || true
  fi

  if [[ "$FOUNDATION_POLICY" == "recover" ]]; then
    echo "FOUNDATION_POLICY=recover found no conservatively safe v30 copy; see $FOUNDATION_INVENTORY" >&2
    exit 2
  fi
  if [[ "$REBUILD_FOUNDATION_IF_MISSING" != "1" ]]; then
    echo "No usable foundation checkpoint and rebuilding is disabled" >&2
    exit 2
  fi

  echo "[v50] no safe historical foundation found; rebuilding from random initialization"
  DETACH=0   GPUS="$GPUS"   INIT_MODE=scratch   V30_CKPT_IN=   OUT_ROOT="$FOUNDATION_ROOT"   FOUNDATION_OUT_ROOT="$FOUNDATION_ROOT"   FOUNDATION_CONFIG="$FOUNDATION_CONFIG"   RUN_MODE=foundation   AUTO_RESUME=1   VAL_SPLIT=val_tune   VAL_SCENARIOS=1000   BATCH_SIZE_PER_GPU=4   NUM_WORKERS_PER_GPU=4   PREFETCH_FACTOR=2   SAVE_EVERY_N_STEPS=2000   bash run_v50_dbap_ri.sh

  [[ -s "$rebuilt_best" ]] || {
    echo "Foundation rebuild did not produce $rebuilt_best" >&2
    exit 2
  }
  FOUNDATION_CKPT="$rebuilt_best"
  FOUNDATION_SOURCE="rebuilt_current_code"
}

if [[ "$V50_INIT_MODE" == "warm_start" ]]; then
  resolve_foundation_checkpoint
  [[ -s "$FOUNDATION_CKPT" ]] || { echo "Resolved foundation is missing: $FOUNDATION_CKPT" >&2; exit 2; }
  V30_CKPT_IN="$FOUNDATION_CKPT"  # compatibility alias for the inner launcher
  if [[ -z "$CONTROL_CKPT" ]]; then
    CONTROL_CKPT="$FOUNDATION_CKPT"
  fi
else
  echo "[v50] V50_INIT_MODE=scratch is an ablation, not the paper main run."
  [[ -n "$CONTROL_CKPT" && -s "$CONTROL_CKPT" ]] || {
    echo "Scratch v50 still needs an independently trained matched CONTROL_CKPT for a paired gate" >&2
    exit 2
  }
  FOUNDATION_CKPT="$CONTROL_CKPT"
  V30_CKPT_IN="$FOUNDATION_CKPT"
  FOUNDATION_SOURCE="scratch_ablation_matched_control"
fi
export FOUNDATION_CKPT V30_CKPT_IN CONTROL_CKPT

echo "[v50] foundation_source=$FOUNDATION_SOURCE"
echo "[v50] foundation_checkpoint=$FOUNDATION_CKPT"
echo "[v50] control_checkpoint=$CONTROL_CKPT"

FOUNDATION_PROVENANCE="$OUT_ROOT/provenance/foundation_checkpoint.json"
if ! is_fresh "$FOUNDATION_PROVENANCE" "$FOUNDATION_CKPT" "$FOUNDATION_CONFIG"; then
  python -m bdse.tools.write_checkpoint_provenance     --checkpoint "$FOUNDATION_CKPT"     --config "$FOUNDATION_CONFIG"     --train-cache "$BDSE_TRAIN_CACHE"     --val-cache "$BDSE_SPLIT_CACHE"     --source "$FOUNDATION_SOURCE"     --output "$FOUNDATION_PROVENANCE"
fi

# ---------------------------------------------------------------------------
# 1. Main v50 training. Use val_tune only for checkpoint selection.
#    Main paper run starts from the frozen v30 checkpoint, not v46.
# ---------------------------------------------------------------------------
if training_complete; then
  echo "[v50] stage 1 already complete: reuse final/best checkpoints"
else
  DETACH=0 \
  GPUS="$GPUS" \
  V30_CKPT_IN="$V30_CKPT_IN" \
  INIT_MODE="$V50_INIT_MODE" \
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
  VAL_EVERY_N_EPOCHS=2 \
  VAL_DENSE_DIAGNOSTIC=1 \
  OPEN_LOOP_MAX_SCENARIOS=1000 \
  bash run_v50_dbap_ri.sh
fi

# ---------------------------------------------------------------------------
# 2. Independent one-sided calibration on val_calib only.
# ---------------------------------------------------------------------------
mkdir -p "$OUT_ROOT/calibration"
CALIBRATION_JSON="$OUT_ROOT/calibration/v50_adverse_calibration.json"
CALIBRATED_CONFIG="$OUT_ROOT/calibration/v50_bdse_dbap_cl_calibrated.yaml"
BEST_CHECKPOINT="$OUT_ROOT/train/bdse_v50_dbap_ri.best.pt"
if is_fresh \
  "$CALIBRATION_JSON" \
  "$BEST_CHECKPOINT" \
  "$BDSE_SPLIT_CACHE/calibration_split_provenance.json" \
  bdse/configs/v50_bdse_dbap_ri_cl.yaml; then
  echo "[v50] stage 2a already complete: reuse adverse calibration"
else
  python -m bdse.tools.calibrate_v48_adverse_bounds \
    --config bdse/configs/v50_bdse_dbap_ri_cl.yaml \
    --checkpoint "$BEST_CHECKPOINT" \
    --preprocessed-dir "$BDSE_SPLIT_CACHE" \
    --split val_calib \
    --max-scenarios 5000 \
    --device cuda \
    --alpha 0.05 \
    --beta 1.0 \
    --prior-radius 0.02 \
    --provenance-json "$BDSE_SPLIT_CACHE/calibration_split_provenance.json" \
    --output "$CALIBRATION_JSON"
fi

if is_fresh "$CALIBRATED_CONFIG" "$CALIBRATION_JSON" bdse/configs/v50_bdse_dbap_ri_cl.yaml; then
  echo "[v50] stage 2b already complete: reuse calibrated config"
else
  python -m bdse.tools.apply_v48_calibration \
    --config-in bdse/configs/v50_bdse_dbap_ri_cl.yaml \
    --calibration-json "$CALIBRATION_JSON" \
    --config-out "$CALIBRATED_CONFIG"
fi

# ---------------------------------------------------------------------------
# 3. Replay calibrated v50 on the same deterministic val_tune 1000 scenes.
# ---------------------------------------------------------------------------
OPEN_LOOP_JSON="$OUT_ROOT/open_loop/open_loop_v50_dbap_ri.json"
OPEN_LOOP_JSONL="$OUT_ROOT/open_loop/open_loop_v50_dbap_ri.jsonl"
if is_fresh "$OPEN_LOOP_JSON" "$BEST_CHECKPOINT" "$CALIBRATED_CONFIG" \
   && is_fresh "$OPEN_LOOP_JSONL" "$BEST_CHECKPOINT" "$CALIBRATED_CONFIG"; then
  echo "[v50] stage 3 already complete: reuse calibrated open-loop results"
else
  GPUS="$GPUS" \
  OUT_ROOT="$OUT_ROOT" \
  RUN_MODE=open_loop \
  V50_CKPT="$BEST_CHECKPOINT" \
  EVAL_CONFIG="$CALIBRATED_CONFIG" \
  VAL_SPLIT=val_tune \
  OPEN_LOOP_SPLIT=val_tune \
  OPEN_LOOP_MAX_SCENARIOS=1000 \
  bash run_v50_dbap_ri.sh
fi

# ---------------------------------------------------------------------------
# 4. Rebuild the frozen control on exactly the same val_tune rows.
#    A single-GPU command is used here for clarity; the scenario order is fixed.
# ---------------------------------------------------------------------------
mkdir -p "$CONTROL_ROOT/open_loop"
CONTROL_JSON="$CONTROL_ROOT/open_loop/control.json"
CONTROL_JSONL="$CONTROL_ROOT/open_loop/control.jsonl"
if is_fresh "$CONTROL_JSON" "$CONTROL_CONFIG" "$CONTROL_CKPT" \
      "$BDSE_SPLIT_CACHE/calibration_split_provenance.json" \
      "$BDSE_SPLIT_CACHE/val_tune/manifest.jsonl" \
   && is_fresh "$CONTROL_JSONL" "$CONTROL_CONFIG" "$CONTROL_CKPT" \
      "$BDSE_SPLIT_CACHE/calibration_split_provenance.json" \
      "$BDSE_SPLIT_CACHE/val_tune/manifest.jsonl"; then
  echo "[v50] stage 4 already complete: reuse frozen-control results"
else
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python -m bdse.experiments.evaluate_open_loop \
    --config "$CONTROL_CONFIG" \
    --checkpoint "$CONTROL_CKPT" \
    --split val_tune \
    --preprocessed-dir "$BDSE_SPLIT_CACHE" \
    --max-scenarios 1000 \
    --device cuda \
    --output "$CONTROL_JSON" \
    --per-sample-output "$CONTROL_JSONL"
fi

# ---------------------------------------------------------------------------
# 5. Strict paired gate. Do not run closed-loop if this fails.
# ---------------------------------------------------------------------------
GATE_MARKER="$OUT_ROOT/open_loop/.v50_dbap_ri_gate_passed"
if is_fresh \
  "$GATE_MARKER" \
  "$OPEN_LOOP_JSON" "$OPEN_LOOP_JSONL" \
  "$CONTROL_JSON" "$CONTROL_JSONL" \
  "$OUT_ROOT/train/bdse_v50_dbap_ri.train_log.jsonl"; then
  echo "[v50] stage 5 already complete: paired gate previously passed"
else
  python -m bdse.tools.check_v50_dbap_ri_gate \
    "$OPEN_LOOP_JSON" \
    "$CONTROL_JSON" \
    --candidate-jsonl "$OPEN_LOOP_JSONL" \
    --control-jsonl "$CONTROL_JSONL" \
    --train-log "$OUT_ROOT/train/bdse_v50_dbap_ri.train_log.jsonl" \
    --latency-target-ms 500 \
    --min-match-gain 0.02 \
    --min-sufficiency-gain 0.01 \
    --min-pair-full-match 0.30 \
    --min-local-pair-full-match 0.30 \
    --max-residual-interface-drop 0.01 \
    --max-harmful-residual-rate 0.05 \
    --min-certified-pair-fraction 0.50 \
    --min-budget-fill-fraction 0.95 \
    2>&1 | tee "$OUT_ROOT/logs/v50_dbap_ri_gate.out"
  printf 'passed\n' >"$GATE_MARKER"
fi

# ---------------------------------------------------------------------------
# 6. Only after PASS: paired CL20.  V47's script contained only a commented
#    example, so even a future PASS would not have started simulation.  V50
#    executes CL20 automatically unless RUN_CLOSED_LOOP_AFTER_GATE=0.
# ---------------------------------------------------------------------------
if [[ "$RUN_CLOSED_LOOP_AFTER_GATE" == "1" ]]; then
  : "${NUPLAN_ROOT:?Set NUPLAN_ROOT before the gated closed-loop stage}"
  BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" \
  CL_TOKEN_SPLIT=val_tune \
  GPUS="$GPUS" OUT_ROOT="$OUT_ROOT" RUN_MODE=cl20 \
  V50_CKPT="$OUT_ROOT/train/bdse_v50_dbap_ri.best.pt" \
  EVAL_CONFIG="$OUT_ROOT/calibration/v50_bdse_dbap_cl_calibrated.yaml" \
  bash run_v50_dbap_ri.sh

  if [[ "$RUN_CL100_AFTER_CL20" == "1" ]]; then
    BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" \
    CL_TOKEN_SPLIT=val_tune \
    GPUS="$GPUS" OUT_ROOT="$OUT_ROOT" RUN_MODE=cl100 \
    V50_CKPT="$OUT_ROOT/train/bdse_v50_dbap_ri.best.pt" \
    EVAL_CONFIG="$OUT_ROOT/calibration/v50_bdse_dbap_cl_calibrated.yaml" \
    bash run_v50_dbap_ri.sh
  fi
else
  echo "[v50] gate passed; closed-loop skipped because RUN_CLOSED_LOOP_AFTER_GATE=0"
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
#   --config "$OUT_ROOT/calibration/v50_bdse_dbap_ri_cl_calibrated.yaml" \
#   --checkpoint "$OUT_ROOT/train/bdse_v50_dbap_ri.best.pt" \
#   --split public_set_test --preprocessed-dir "$BDSE_TEST_CACHE" --device cuda \
#   --output "$OUT_ROOT/open_loop/open_loop_v50_dbap_ri_test.json" \
#   --per-sample-output "$OUT_ROOT/open_loop/open_loop_v50_dbap_ri_test.jsonl"
