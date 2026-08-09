#!/usr/bin/env bash
set -euo pipefail

# V64.3.1 provenance-correct checkpoint-independent pipeline.  Resolve every relative path from the
# checked-in script directory so a stale working directory cannot silently run a
# different config or helper script.
PIPELINE_VERSION="v64.3.1-cc-aocc-apwcca-engineering-corrected"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "[v64] pipeline_version=$PIPELINE_VERSION script=$SCRIPT_DIR/$(basename "$0")"

# Mandatory data paths.
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE_ORIGINAL="${BDSE_VAL_CACHE_ORIGINAL:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export BDSE_SPLIT_CACHE="${BDSE_SPLIT_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v53_split}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_1_cc_aocc_apwcca_fast_2gpu_v1}"
export FOUNDATION_ROOT="${FOUNDATION_ROOT:-$OUT_ROOT/factorized_anchor}"
export FOUNDATION_CONFIG="${FOUNDATION_CONFIG:-bdse/configs/v53_factorized_anchor_fast_2gpu.yaml}"
export FOUNDATION_SOURCE_CONFIG="${FOUNDATION_SOURCE_CONFIG:-$FOUNDATION_CONFIG}"

# There is deliberately no hard-coded outputs_v30 dependency.  FOUNDATION_CKPT
# is the canonical variable; V30_CKPT_IN remains only as a backwards-compatible
# alias consumed by run_v64_saqa_bcc.sh.
export V30_CKPT_IN="${V30_CKPT_IN:-}"
export FOUNDATION_CKPT="${FOUNDATION_CKPT:-$V30_CKPT_IN}"
export FOUNDATION_POLICY="${FOUNDATION_POLICY:-auto}"  # auto | rebuild | recover | explicit
export FOUNDATION_SEARCH_ROOT="${FOUNDATION_SEARCH_ROOT:-.}"
export RECOVER_SAFE_FOUNDATION_COPIES="${RECOVER_SAFE_FOUNDATION_COPIES:-1}"
export REBUILD_FOUNDATION_IF_MISSING="${REBUILD_FOUNDATION_IF_MISSING:-1}"
export ALLOW_ALGORITHM_CHECKPOINT_INIT="${ALLOW_ALGORITHM_CHECKPOINT_INIT:-0}"
export V64_INIT_MODE="${V64_INIT_MODE:-warm_start}"
export EXACT_SELECTOR_WORKERS_PER_RANK="${EXACT_SELECTOR_WORKERS_PER_RANK:-4}"
export EXACT_SELECTOR_CPU_BACKEND="${EXACT_SELECTOR_CPU_BACKEND:-process}"
export GPUS="${GPUS:-0,1}"
export FOUNDATION_CONTROL_CONFIG="${FOUNDATION_CONTROL_CONFIG:-bdse/configs/v64_3_cc_aocc_apwcca_anchor_control_cl.yaml}"
export LOCAL_CONTROL_CONFIG="${LOCAL_CONTROL_CONFIG:-bdse/configs/v64_3_cc_aocc_apwcca_local_control_cl.yaml}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-bdse/configs/v64_3_cc_aocc_apwcca_train_2gpu.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-bdse/configs/v64_3_cc_aocc_apwcca_cl.yaml}"
export QUERY_CACHE_AUDIT_REPORT="${QUERY_CACHE_AUDIT_REPORT:-$OUT_ROOT/provenance/query_cache_audit.json}"
export CONTROL_CONFIG="${CONTROL_CONFIG:-$FOUNDATION_CONTROL_CONFIG}"  # compatibility alias
export CONTROL_CKPT="${CONTROL_CKPT:-}"
export FOUNDATION_CONTROL_ROOT="${FOUNDATION_CONTROL_ROOT:-$OUT_ROOT/control_foundation_matched}"
export LOCAL_CONTROL_ROOT="${LOCAL_CONTROL_ROOT:-$OUT_ROOT/control_local_same_checkpoint}"
export PIPELINE_DETACH="${PIPELINE_DETACH:-1}"
export PIPELINE_FORCE="${PIPELINE_FORCE:-0}"
export RUN_CLOSED_LOOP_AFTER_GATE="${RUN_CLOSED_LOOP_AFTER_GATE:-0}"
export SKIP_V64_TRAINING="${SKIP_V64_TRAINING:-0}"
export V64_CANDIDATE_CHECKPOINT="${V64_CANDIDATE_CHECKPOINT:-}"
export RUN_DIAGNOSTIC_CL20_ON_GATE_FAIL="${RUN_DIAGNOSTIC_CL20_ON_GATE_FAIL:-1}"
export RUN_CL100_AFTER_CL20="${RUN_CL100_AFTER_CL20:-0}"
# Keep algorithmic closed-loop validation separate from the deployment-latency
# claim. Set to 1 only when CL should be blocked by the real-time target.
export ENFORCE_LATENCY_BEFORE_CL="${ENFORCE_LATENCY_BEFORE_CL:-0}"
export TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-16}"
export TRAIN_NUM_WORKERS_PER_GPU="${TRAIN_NUM_WORKERS_PER_GPU:-12}"
export TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-2}"
export VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}"
export FAST_SELECTOR_SCENES_PER_RANK="${FAST_SELECTOR_SCENES_PER_RANK:-1}"
export FAST_SELECTOR_EVERY_N_STEPS="${FAST_SELECTOR_EVERY_N_STEPS:-4}"
export BEST_MIN_EPOCH="${BEST_MIN_EPOCH:-3}"
export CALIBRATION_BETA="${CALIBRATION_BETA:-0.0}"
export CALIBRATION_PRIOR_RADIUS="${CALIBRATION_PRIOR_RADIUS:-0.02}"
export CL_PROCESSES_PER_GPU="${CL_PROCESSES_PER_GPU:-2}"
export CL_WORKERS_PER_GPU=1
export ANCHOR_WORKERS_PER_GPU="${ANCHOR_WORKERS_PER_GPU:-2}"
export CALIBRATION_WORKERS_PER_GPU="${CALIBRATION_WORKERS_PER_GPU:-2}"
export MAX_TRAIN_SCENARIOS="${MAX_TRAIN_SCENARIOS:-50000}"

case "$FOUNDATION_POLICY" in
  auto|rebuild|recover|explicit) ;;
  *) echo "FOUNDATION_POLICY must be auto, rebuild, recover, or explicit" >&2; exit 2 ;;
esac

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/provenance"

# Fail before any long-running work when an inherited TRAIN_CONFIG/EVAL_CONFIG
# points to an older algorithm family or violates the fixed planner-interface
# support contract.  This also makes the selected config auditable.
python -m bdse.tools.validate_v64_pipeline_config \
  --train-config "$TRAIN_CONFIG" \
  --eval-config "$EVAL_CONFIG" \
  --expected-family v64.3.3 \
  --output "$OUT_ROOT/provenance/v64_pipeline_config_contract.json"

# Detach the complete pipeline, not only the training child.  Detaching only
# run_v64_saqa_bcc.sh lets this parent continue immediately into calibration while
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
  echo "[v64] detached pipeline started pid=$pipeline_pid log=$pipeline_log"
  exit 0
fi

PIPELINE_LOCK_DIR="$OUT_ROOT/.v64_pipeline.lock"
if ! mkdir "$PIPELINE_LOCK_DIR" 2>/dev/null; then
  owner="$(cat "$PIPELINE_LOCK_DIR/pid" 2>/dev/null || echo unknown)"
  echo "A v64 pipeline is already using OUT_ROOT=$OUT_ROOT (pid=$owner)" >&2
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

TRAIN_CONTRACT_JSON="$OUT_ROOT/train/v64_3_training_contract.json"

training_complete() {
  [[ "$PIPELINE_FORCE" != "1" ]] || return 1
  [[ -s "$OUT_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl"       && -s "$OUT_ROOT/train/bdse_v64_saqa_bcc.pt"       && -s "$OUT_ROOT/train/bdse_v64_saqa_bcc.best.pt"       && -s "$TRAIN_CONTRACT_JSON" ]] || return 1
  python - "$TRAIN_CONFIG" "$FOUNDATION_CKPT" "$OUT_ROOT/train/bdse_v64_saqa_bcc.best.pt" "$TRAIN_CONTRACT_JSON" <<'PY_TRAIN_CONTRACT_CHECK'
import hashlib,json,sys
from pathlib import Path
cfg, ckpt, best, marker = map(Path, sys.argv[1:])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
try: d=json.loads(marker.read_text())
except Exception: raise SystemExit(1)
if d.get('train_config_sha256') != sha(cfg): raise SystemExit(1)
if d.get('foundation_checkpoint_sha256') != sha(ckpt): raise SystemExit(1)
if d.get('best_checkpoint_sha256') != sha(best): raise SystemExit(1)
if d.get('algorithm_family') != 'V64.3.1-CC-AOCC-AP-WCCA-DA-EPC': raise SystemExit(1)
PY_TRAIN_CONTRACT_CHECK
}

write_training_contract() {
  python - "$TRAIN_CONFIG" "$FOUNDATION_CKPT" "$OUT_ROOT/train/bdse_v64_saqa_bcc.best.pt" "$TRAIN_CONTRACT_JSON" <<'PY_TRAIN_CONTRACT_WRITE'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
cfg, foundation, best, out = map(Path, sys.argv[1:])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
payload={
 'algorithm_family':'V64.3.1-CC-AOCC-AP-WCCA-DA-EPC',
 'train_config_path':str(cfg.resolve()), 'train_config_sha256':sha(cfg),
 'foundation_checkpoint_path':str(foundation.resolve()), 'foundation_checkpoint_sha256':sha(foundation),
 'best_checkpoint_path':str(best.resolve()), 'best_checkpoint_sha256':sha(best),
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
}
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
PY_TRAIN_CONTRACT_WRITE
}

# ---------------------------------------------------------------------------
# 0. Create a paper-grade validation protocol. The split is by nuPlan log group,
#    not random scenario row, to reduce temporal/scene leakage.
# ---------------------------------------------------------------------------
if [[ "$PIPELINE_FORCE" != "1" \
      && -s "$BDSE_SPLIT_CACHE/calibration_split_provenance.json" \
      && -s "$BDSE_SPLIT_CACHE/val_tune/manifest.jsonl" \
      && -s "$BDSE_SPLIT_CACHE/val_calib/manifest.jsonl" ]]; then
  echo "[v64] stage 0 already complete: reuse group-disjoint split"
else
  python -m bdse.tools.build_group_disjoint_calibration_split \
    --preprocessed-dir "$BDSE_VAL_CACHE_ORIGINAL" \
    --split val \
    --output-root "$BDSE_SPLIT_CACHE" \
    --calibration-fraction 0.20 \
    --seed 49
fi

export BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE"

# Cached dense query tensors are an optional speed path, never an implicit one.
# A cache-based TRAIN_CONFIG is accepted only with a PASS report whose code and
# query-relevant config fingerprints still match this checkout.
TRAIN_QUERY_SOURCE="$(python - "$TRAIN_CONFIG" <<'PY_QUERY_SOURCE'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(str((cfg.get("runtime", {}) or {}).get("dense_query_feature_source", "cache_or_recompute")).strip().lower())
PY_QUERY_SOURCE
)"
case "$TRAIN_QUERY_SOURCE" in
  cache|cached|cache_only|cache_prefix_runtime_extension|verified_prefix_runtime_extension|cache_supported_prefix)
    [[ -s "$QUERY_CACHE_AUDIT_REPORT" ]] || {
      echo "TRAIN_CONFIG uses cached query features but QUERY_CACHE_AUDIT_REPORT is missing: $QUERY_CACHE_AUDIT_REPORT" >&2
      echo "Run bdse.tools.audit_cached_query_features first, or use the default runtime_recompute config." >&2
      exit 2
    }
    python -m bdse.tools.audit_cached_query_features \
      --config "$TRAIN_CONFIG" --verify-report "$QUERY_CACHE_AUDIT_REPORT"
    ;;
  runtime|runtime_recompute|recompute|canonical_runtime|cache_verified|verified_cache) ;;
  *) echo "Unsupported runtime.dense_query_feature_source=$TRAIN_QUERY_SOURCE" >&2; exit 2 ;;
esac

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
# They are algorithm-specific initializations and would confound the v64 paper
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
    echo "[v64] ignore missing inherited foundation path: $FOUNDATION_CKPT"
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
      echo "[v64] WARNING: algorithm-specific checkpoint recovery enabled; use only as a transfer ablation" >&2
    fi
    selected="$(python -m bdse.tools.resolve_foundation_checkpoint       --config "$TRAIN_CONFIG"       --search-root "$FOUNDATION_SEARCH_ROOT"       --output-json "$FOUNDATION_INVENTORY"       --print-selected       "${allow_args[@]}")"
    if [[ -n "$selected" && -s "$selected" ]]; then
      FOUNDATION_CKPT="$selected"
      if [[ "$ALLOW_ALGORITHM_CHECKPOINT_INIT" == "1" ]]; then
        FOUNDATION_SOURCE="recovered_algorithm_transfer_ablation"
      else
        FOUNDATION_SOURCE="recovered_verified_compatible_anchor"
      fi
      echo "[v64] recovered foundation checkpoint=$FOUNDATION_CKPT source=$FOUNDATION_SOURCE"
      return 0
    fi
  else
    # Still write an inventory report when recovery is disabled, because it is
    # useful for documenting why retained v40-v49 checkpoints were not used.
    python -m bdse.tools.resolve_foundation_checkpoint       --config "$TRAIN_CONFIG"       --search-root "$FOUNDATION_SEARCH_ROOT"       --output-json "$FOUNDATION_INVENTORY" >/dev/null || true
  fi

  if [[ "$FOUNDATION_POLICY" == "recover" ]]; then
    echo "FOUNDATION_POLICY=recover found no conservatively safe v30 copy; see $FOUNDATION_INVENTORY" >&2
    exit 2
  fi
  if [[ "$REBUILD_FOUNDATION_IF_MISSING" != "1" ]]; then
    echo "No usable foundation checkpoint and rebuilding is disabled" >&2
    exit 2
  fi

  echo "[v64] no safe historical foundation found; rebuilding from random initialization"
  DETACH=0   GPUS="$GPUS"   INIT_MODE=scratch   V30_CKPT_IN=   OUT_ROOT="$FOUNDATION_ROOT"   FOUNDATION_OUT_ROOT="$FOUNDATION_ROOT"   FOUNDATION_CONFIG="$FOUNDATION_CONFIG"   RUN_MODE=foundation   AUTO_RESUME=1   VAL_SPLIT=val_tune   VAL_SCENARIOS=1000   BATCH_SIZE_PER_GPU="$TRAIN_BATCH_SIZE_PER_GPU"   NUM_WORKERS_PER_GPU="$TRAIN_NUM_WORKERS_PER_GPU"   PREFETCH_FACTOR="$TRAIN_PREFETCH_FACTOR"   SAVE_EVERY_N_STEPS=1000   bash run_v64_saqa_bcc.sh

  [[ -s "$rebuilt_best" ]] || {
    echo "Foundation rebuild did not produce $rebuilt_best" >&2
    exit 2
  }
  FOUNDATION_CKPT="$rebuilt_best"
  FOUNDATION_SOURCE="rebuilt_current_code"
}

if [[ "$V64_INIT_MODE" == "warm_start" ]]; then
  resolve_foundation_checkpoint
  [[ -s "$FOUNDATION_CKPT" ]] || { echo "Resolved foundation is missing: $FOUNDATION_CKPT" >&2; exit 2; }
  V30_CKPT_IN="$FOUNDATION_CKPT"  # compatibility alias for the inner launcher
  if [[ -z "$CONTROL_CKPT" ]]; then
    CONTROL_CKPT="$FOUNDATION_CKPT"
  fi
else
  echo "[v64] V64_INIT_MODE=scratch is an ablation, not the paper main run."
  [[ -n "$CONTROL_CKPT" && -s "$CONTROL_CKPT" ]] || {
    echo "Scratch v64 still needs an independently trained matched CONTROL_CKPT for a paired gate" >&2
    exit 2
  }
  FOUNDATION_CKPT="$CONTROL_CKPT"
  V30_CKPT_IN="$FOUNDATION_CKPT"
  FOUNDATION_SOURCE="scratch_ablation_matched_control"
fi
export FOUNDATION_CKPT V30_CKPT_IN CONTROL_CKPT

echo "[v64] foundation_source=$FOUNDATION_SOURCE"
echo "[v64] foundation_checkpoint=$FOUNDATION_CKPT"
echo "[v64] control_checkpoint=$CONTROL_CKPT"

FOUNDATION_PROVENANCE="$OUT_ROOT/provenance/foundation_checkpoint.json"
if ! is_fresh "$FOUNDATION_PROVENANCE" "$FOUNDATION_CKPT" "$FOUNDATION_SOURCE_CONFIG"; then
  python -m bdse.tools.write_checkpoint_provenance     --checkpoint "$FOUNDATION_CKPT"     --config "$FOUNDATION_SOURCE_CONFIG"     --train-cache "$BDSE_TRAIN_CACHE"     --val-cache "$BDSE_SPLIT_CACHE"     --source "$FOUNDATION_SOURCE"     --output "$FOUNDATION_PROVENANCE"
fi

# ---------------------------------------------------------------------------
# 0.75 Foundation quality gate. A weak foundation makes residual results
#      uninterpretable, so the algorithm stage is blocked before any v64 update.
# ---------------------------------------------------------------------------
FOUNDATION_QUALITY_ROOT="$FOUNDATION_ROOT/quality"
FOUNDATION_QUALITY_JSON="$FOUNDATION_QUALITY_ROOT/open_loop.json"
FOUNDATION_QUALITY_JSONL="$FOUNDATION_QUALITY_ROOT/open_loop.jsonl"
mkdir -p "$FOUNDATION_QUALITY_ROOT"
if is_fresh "$FOUNDATION_QUALITY_JSON" "$FOUNDATION_CKPT" "$FOUNDATION_CONTROL_CONFIG" "$BDSE_SPLIT_CACHE/val_tune/manifest.jsonl" \
   && is_fresh "$FOUNDATION_QUALITY_JSONL" "$FOUNDATION_CKPT" "$FOUNDATION_CONTROL_CONFIG" "$BDSE_SPLIT_CACHE/val_tune/manifest.jsonl"; then
  echo "[v64] foundation quality replay already complete"
else
  # The uploaded V64.3 run spent ~55 min on this single-GPU 1000-scene replay.
  # It is embarrassingly parallel and does not alter evaluation semantics, so
  # use the same deterministic modulo-shard merger as the paired open-loop suite.
  ANCHOR_PAR_ROOT="$FOUNDATION_QUALITY_ROOT/parallel_suite"
  rm -rf "$ANCHOR_PAR_ROOT"
  python -m bdse.tools.run_parallel_open_loop_suite \
    --system "foundation::$FOUNDATION_CONTROL_CONFIG::$FOUNDATION_CKPT" \
    --preprocessed-dir "$BDSE_SPLIT_CACHE" --split val_tune --max-scenarios 1000 \
    --output-root "$ANCHOR_PAR_ROOT" --gpus "$GPUS" \
    --workers-per-gpu "$ANCHOR_WORKERS_PER_GPU" --device cuda
  cp -f "$ANCHOR_PAR_ROOT/foundation/metrics.json" "$FOUNDATION_QUALITY_JSON"
  cp -f "$ANCHOR_PAR_ROOT/foundation/metrics.jsonl" "$FOUNDATION_QUALITY_JSONL"
fi
ANCHOR_GATE_REPORT="$OUT_ROOT/provenance/v64_anchor_gate.json"
python -m bdse.tools.check_v53_anchor_quality "$FOUNDATION_QUALITY_JSON" \
  --jsonl "$FOUNDATION_QUALITY_JSONL" \
  --min-full-interface-match "${ANCHOR_MIN_FULL_INTERFACE_MATCH:-0.32}" \
  --min-base-winner-sign "${ANCHOR_MIN_BASE_WINNER_SIGN:-0.62}" \
  --min-dense-winner-sign "${ANCHOR_MIN_DENSE_WINNER_SIGN:-0.75}" \
  --min-dense-near-sign "${ANCHOR_MIN_DENSE_NEAR_SIGN:-0.65}" \
  --min-dense-all-sign "${ANCHOR_MIN_DENSE_ALL_SIGN:-0.68}" \
  --warn-max-full-interface-regret "${ANCHOR_WARN_MAX_FULL_INTERFACE_REGRET:-15000}" \
  --report-json "$ANCHOR_GATE_REPORT" \
  2>&1 | tee "$OUT_ROOT/logs/v64_anchor_quality_gate.out"

# ---------------------------------------------------------------------------
# 1. Main V64.3.1 training. Foundation + legacy HAB proposal/family/query-extension
#    are frozen; only AP-WCCA critical adapter + residual uncertainty heads train.
# ---------------------------------------------------------------------------
if [[ "$SKIP_V64_TRAINING" == "1" ]]; then
  [[ -n "$V64_CANDIDATE_CHECKPOINT" && -s "$V64_CANDIDATE_CHECKPOINT" ]] || {
    echo "SKIP_V64_TRAINING=1 requires a valid V64_CANDIDATE_CHECKPOINT" >&2
    exit 2
  }
  echo "[v64] stage 1 skipped by explicit checkpoint-evaluation mode: $V64_CANDIDATE_CHECKPOINT"
elif training_complete; then
  echo "[v64] stage 1 already complete: reuse final/best checkpoints"
else
  DETACH=0 \
  GPUS="$GPUS" \
  V30_CKPT_IN="$V30_CKPT_IN" \
  INIT_MODE="$V64_INIT_MODE" \
  TRAIN_CONFIG="$TRAIN_CONFIG" \
  EXACT_SELECTOR_WORKERS_PER_RANK="$EXACT_SELECTOR_WORKERS_PER_RANK" \
  EXACT_SELECTOR_CPU_BACKEND="$EXACT_SELECTOR_CPU_BACKEND" \
  OUT_ROOT="$OUT_ROOT" \
  RUN_MODE=train \
  AUTO_RESUME=0 \
  VAL_SPLIT=val_tune \
  OPEN_LOOP_SPLIT=val_tune \
  MAX_TRAIN_SCENARIOS="$MAX_TRAIN_SCENARIOS" \
  BATCH_SIZE_PER_GPU="$TRAIN_BATCH_SIZE_PER_GPU" \
  NUM_WORKERS_PER_GPU="$TRAIN_NUM_WORKERS_PER_GPU" \
  PREFETCH_FACTOR="$TRAIN_PREFETCH_FACTOR" \
  VAL_NUM_WORKERS_PER_GPU="$VAL_NUM_WORKERS_PER_GPU" \
  BEST_MIN_EPOCH="$BEST_MIN_EPOCH" \
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
  bash run_v64_saqa_bcc.sh
  write_training_contract
fi

# Old deliveries did not bind checkpoint reuse to the actual selected config.
# In checkpoint-evaluation mode we intentionally skip this marker; otherwise it
# must exist after either a fresh train or a provenance-matched reuse.
if [[ "$SKIP_V64_TRAINING" != "1" ]]; then
  training_complete || { echo "V64.3.1 training outputs exist but their config/foundation provenance does not match this run" >&2; exit 2; }
fi

BEST_CHECKPOINT="${V64_CANDIDATE_CHECKPOINT:-$OUT_ROOT/train/bdse_v64_saqa_bcc.best.pt}"
[[ -s "$BEST_CHECKPOINT" ]] || { echo "Missing v64 candidate checkpoint: $BEST_CHECKPOINT" >&2; exit 2; }
python - "$BEST_CHECKPOINT" "$OUT_ROOT/provenance/v64_candidate_checkpoint.json" <<'PY_CANDIDATE_CKPT'
import hashlib, json, sys
from pathlib import Path
src, out = map(Path, sys.argv[1:])
h = hashlib.sha256()
with src.open('rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
payload = {"path": str(src.resolve()), "sha256": h.hexdigest(), "size_bytes": src.stat().st_size}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
PY_CANDIDATE_CKPT

# ---------------------------------------------------------------------------
# 2. One shared evidence calibration plus candidate-only residual calibration.
#    V57 repeated the same frozen local-evidence replay three times and still did
#    not calibrate the direct residual uncertainty used by the flip guard.  V64
#    collects both certificates in one paired candidate pass, sharded across GPUs.
# ---------------------------------------------------------------------------
mkdir -p "$OUT_ROOT/calibration/raw" "$OUT_ROOT/calibration/shards"
CALIBRATION_PROVENANCE="$BDSE_SPLIT_CACHE/calibration_split_provenance.json"
DUAL_CAL_JSON="$OUT_ROOT/calibration/v64_dual_certificate.json"
CAL_SHARD_ROOT="$OUT_ROOT/calibration/shards"

prepare_calibration_shards() {
  IFS=',' read -r -a cal_gpus <<< "$GPUS"
  local ngpu="${#cal_gpus[@]}"
  [[ "$ngpu" -ge 2 ]] || { echo "V64 calibration requires at least two GPU ids, got GPUS=$GPUS" >&2; return 2; }
  local nshards=$(( ngpu * CALIBRATION_WORKERS_PER_GPU ))
  [[ "$nshards" -ge 2 ]] || nshards=2

  if [[ "$PIPELINE_FORCE" != "1" && -s "$CAL_SHARD_ROOT/metadata.json" ]]; then
    if python - "$CAL_SHARD_ROOT" "$nshards" <<'PY_CAL_CHECK'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); n=int(sys.argv[2])
try: meta=json.loads((root/'metadata.json').read_text())
except Exception: raise SystemExit(1)
if int(meta.get('num_shards',-1)) != n: raise SystemExit(1)
for sid in range(n):
    if not (root/f'shard{sid}'/'val'/'manifest.jsonl').is_file(): raise SystemExit(1)
PY_CAL_CHECK
    then
      echo "[v64] reuse $nshards calibration shard manifests"
      return 0
    fi
  fi
  rm -rf "$CAL_SHARD_ROOT"
  mkdir -p "$CAL_SHARD_ROOT"
  python - "$BDSE_SPLIT_CACHE" "$CAL_SHARD_ROOT" "$nshards" <<'PY_CAL_SHARDS'
import json, sys
from pathlib import Path
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
root=Path(sys.argv[1]); out=Path(sys.argv[2]); n=int(sys.argv[3])
paths=PreprocessedBDSEDataset(root, split=['val_calib'], max_scenarios=5000).build_index()
if not paths: raise SystemExit('No val_calib samples')
for sid in range(n):
    d=out/f'shard{sid}'/'val'; d.mkdir(parents=True,exist_ok=True)
    rows=[{'split':'val','path':str(p.resolve()),'original_index':i} for i,p in enumerate(paths) if i%n==sid]
    (d/'manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
(out/'metadata.json').write_text(json.dumps({'num_scenarios':len(paths),'num_shards':n},indent=2),encoding='utf-8')
PY_CAL_SHARDS
}

calibration_shard_fresh() {
  local sid="$1"
  local raw="$OUT_ROOT/calibration/raw/shard${sid}.npz"
  local manifest="$CAL_SHARD_ROOT/shard${sid}/val/manifest.jsonl"
  is_fresh "$raw" "$BEST_CHECKPOINT" "$EVAL_CONFIG" "$CALIBRATION_PROVENANCE" "$manifest" || return 1
  # File timestamps are insufficient when a package is unpacked with preserved
  # mtimes.  Bind every reusable calibration shard to the exact checkpoint and
  # config bytes that generated it.  Old V64.3 shards without these hashes are
  # intentionally treated as stale and recomputed.
  python - "$raw" "$BEST_CHECKPOINT" "$EVAL_CONFIG" <<'PY_CAL_RAW_SHA'
import hashlib,sys,numpy as np
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
try:
    z=np.load(sys.argv[1],allow_pickle=False)
    ch=str(z['source_checkpoint_sha256'][0])
    gh=str(z['source_config_sha256'][0])
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if ch==sha(sys.argv[2]) and gh==sha(sys.argv[3]) else 1)
PY_CAL_RAW_SHA
}

dual_calibration_fresh() {
  is_fresh "$DUAL_CAL_JSON" "$BEST_CHECKPOINT" "$EVAL_CONFIG" "$CALIBRATION_PROVENANCE" || return 1
  python - "$DUAL_CAL_JSON" "$BEST_CHECKPOINT" "$EVAL_CONFIG" <<'PY_CAL_MERGED_SHA'
import hashlib,json,sys
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
try: d=json.loads(Path(sys.argv[1]).read_text())
except Exception: raise SystemExit(1)
ok=(d.get('source_checkpoint_sha256')==sha(sys.argv[2]) and d.get('source_config_sha256')==sha(sys.argv[3]))
raise SystemExit(0 if ok else 1)
PY_CAL_MERGED_SHA
}

LAUNCHED_CALIBRATION_PID=""
launch_calibration_shard() {
  local sid="$1" gpu="$2"
  local raw="$OUT_ROOT/calibration/raw/shard${sid}.npz"
  local manifest="$CAL_SHARD_ROOT/shard${sid}/val/manifest.jsonl"
  local log="$OUT_ROOT/logs/calibration_shard${sid}_gpu${gpu}.out"
  if calibration_shard_fresh "$sid"; then
    echo "[v64] calibration shard $sid already complete"
    return 1
  fi
  rm -f "$raw" "$raw.failed"
  echo "[v64] launch calibration shard $sid on CUDA_VISIBLE_DEVICES=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m bdse.tools.calibrate_v64_3_dual_certificates \
    --raw-output "$raw" \
    --config "$EVAL_CONFIG" --checkpoint "$BEST_CHECKPOINT" \
    --preprocessed-dir "$CAL_SHARD_ROOT/shard${sid}" --split val --max-scenarios 5000 \
    --device cuda --alpha 0.05 --beta "$CALIBRATION_BETA" --prior-radius "$CALIBRATION_PRIOR_RADIUS" \
    > "$log" 2>&1 &
  LAUNCHED_CALIBRATION_PID="$!"
}

wait_calibration_shard() {
  local sid="$1" gpu="$2" pid="$3"
  local raw="$OUT_ROOT/calibration/raw/shard${sid}.npz"
  local log="$OUT_ROOT/logs/calibration_shard${sid}_gpu${gpu}.out"
  if wait "$pid"; then
    [[ -s "$raw" ]] || {
      echo "[v64] calibration shard $sid exited 0 but raw output is missing" >&2
      tail -80 "$log" >&2 || true
      return 1
    }
    echo "[v64] calibration shard $sid complete"
    return 0
  fi
  local status=$?
  printf 'exit_status=%s\n' "$status" > "$raw.failed"
  echo "[v64] calibration shard $sid failed with status=$status; tail follows" >&2
  tail -120 "$log" >&2 || true
  return "$status"
}

if ! dual_calibration_fresh; then
  prepare_calibration_shards
  IFS=',' read -r -a cal_gpus <<< "$GPUS"
  ngpu="${#cal_gpus[@]}"
  [[ "$ngpu" -ge 2 ]] || { echo "V64 calibration requires two distinct GPU ids in GPUS=$GPUS" >&2; exit 2; }
  num_cal_shards=$(( ngpu * CALIBRATION_WORKERS_PER_GPU ))
  declare -a cal_pids=() cal_pid_sids=() cal_pid_gpus=() raw_paths=()
  for ((sid=0; sid<num_cal_shards; sid++)); do
    gpu="${cal_gpus[$((sid % ngpu))]}"
    raw_paths+=("$OUT_ROOT/calibration/raw/shard${sid}.npz")
    if ! calibration_shard_fresh "$sid"; then
      launch_calibration_shard "$sid" "$gpu"
      cal_pids+=("$LAUNCHED_CALIBRATION_PID")
      cal_pid_sids+=("$sid")
      cal_pid_gpus+=("$gpu")
    fi
  done
  cal_failed=0
  for ((i=0; i<${#cal_pids[@]}; i++)); do
    wait_calibration_shard "${cal_pid_sids[$i]}" "${cal_pid_gpus[$i]}" "${cal_pids[$i]}" || cal_failed=1
  done
  [[ "$cal_failed" == "0" ]] || {
    echo "[v64] calibration incomplete. Re-run with PIPELINE_FORCE=0; completed shards will be reused." >&2
    exit 4
  }
  tmp_cal="$DUAL_CAL_JSON.tmp"
  rm -f "$tmp_cal"
  python -m bdse.tools.calibrate_v64_3_dual_certificates \
    --merge-raw "${raw_paths[@]}" \
    --alpha 0.05 --beta "$CALIBRATION_BETA" --prior-radius "$CALIBRATION_PRIOR_RADIUS" --residual-epsilon-fallback 0.05 \
    --provenance-json "$CALIBRATION_PROVENANCE" --output "$tmp_cal"
  [[ -s "$tmp_cal" ]] || { echo "[v64] merged calibration JSON missing" >&2; exit 4; }
  mv -f "$tmp_cal" "$DUAL_CAL_JSON"
else
  echo "[v64] reuse dual-certificate calibration"
fi

CAND_CAL_CFG="$OUT_ROOT/calibration/v64_saqa_bcc_candidate_calibrated.yaml"
LOCAL_CAL_CFG="$OUT_ROOT/calibration/v64_saqa_bcc_local_control_calibrated.yaml"
FOUND_CAL_CFG="$OUT_ROOT/calibration/v64_saqa_bcc_foundation_control_calibrated.yaml"
python -m bdse.tools.apply_v64_3_dual_calibration --config "$EVAL_CONFIG" --calibration-json "$DUAL_CAL_JSON" --output "$CAND_CAL_CFG"
python -m bdse.tools.apply_v64_3_dual_calibration --config "$LOCAL_CONTROL_CONFIG" --calibration-json "$DUAL_CAL_JSON" --output "$LOCAL_CAL_CFG" --control
python -m bdse.tools.apply_v64_3_dual_calibration --config "$FOUNDATION_CONTROL_CONFIG" --calibration-json "$DUAL_CAL_JSON" --output "$FOUND_CAL_CFG" --control

# ---------------------------------------------------------------------------
# 3. Candidate/local/foundation open-loop in one bounded worker pool.
#    All systems use the same modulo shards; OPEN_LOOP_WORKERS_PER_GPU controls
#    overlap.  Start at 2 because V57 used <100 MB/model, then benchmark 1 vs 2.
# ---------------------------------------------------------------------------
OPEN_LOOP_WORKERS_PER_GPU="${OPEN_LOOP_WORKERS_PER_GPU:-2}"
SUITE_ROOT="$OUT_ROOT/open_loop/parallel_suite"
SUITE_REPORT="$SUITE_ROOT/parallel_open_loop_suite_report.json"
open_loop_suite_fresh() {
  is_fresh "$SUITE_REPORT" "$BEST_CHECKPOINT" "$FOUNDATION_CKPT" "$CAND_CAL_CFG" "$LOCAL_CAL_CFG" "$FOUND_CAL_CFG" "$BDSE_SPLIT_CACHE/val_tune/manifest.jsonl" || return 1
  python - "$SUITE_REPORT" \
    "$CAND_CAL_CFG" "$BEST_CHECKPOINT" \
    "$LOCAL_CAL_CFG" "$BEST_CHECKPOINT" \
    "$FOUND_CAL_CFG" "$FOUNDATION_CKPT" <<'PY_SUITE_SHA'
import hashlib,json,sys
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
try: d=json.loads(Path(sys.argv[1]).read_text())
except Exception: raise SystemExit(1)
sources=d.get('system_sources') or {}
items=[('candidate',sys.argv[2],sys.argv[3]),('local',sys.argv[4],sys.argv[5]),('foundation',sys.argv[6],sys.argv[7])]
for name,cfg,ckpt in items:
    src=sources.get(name) or {}
    if src.get('config_sha256') != sha(cfg) or src.get('checkpoint_sha256') != sha(ckpt):
        raise SystemExit(1)
raise SystemExit(0)
PY_SUITE_SHA
}
if ! open_loop_suite_fresh; then
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

# Decision-aligned certificate audit is diagnostic-only.  It measures whether
# the AOCC pairwise certificate actually tracks the exact fixed-B winner of the
# downstream deployment operator.  It never changes the gate or certifies an
# otherwise uncertified scene.
CERT_ALIGNMENT_JSON="$SUITE_ROOT/candidate/certificate_action_alignment.json"
python -m bdse.tools.analyze_certificate_action_alignment \
  --jsonl "$OPEN_LOOP_JSONL" --output "$CERT_ALIGNMENT_JSON" \
  > "$OUT_ROOT/logs/v64_3_certificate_action_alignment.out"

# ---------------------------------------------------------------------------
# 4. Three-tier gate with engineering-integrity checks.
# ---------------------------------------------------------------------------
PROTOCOL_GATE_MARKER="$OUT_ROOT/open_loop/.v64_protocol_gate_passed"
COMP_GATE_MARKER="$OUT_ROOT/open_loop/.v64_competitive_gate_passed"
GATE_REPORT="$OUT_ROOT/open_loop/v64_saqa_bcc_gate_report.json"
latency_gate_args=()
if [[ "$ENFORCE_LATENCY_BEFORE_CL" == "1" ]]; then latency_gate_args+=(--enforce-latency); fi
python -m bdse.tools.check_v64_saqa_bcc_gate \
  "$OPEN_LOOP_JSON" "$LOCAL_JSON" "$FOUND_JSON" \
  --candidate-jsonl "$OPEN_LOOP_JSONL" --local-control-jsonl "$LOCAL_JSONL" --foundation-control-jsonl "$FOUND_JSONL" \
  --train-log "$OUT_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl" \
  --train-config "$TRAIN_CONFIG" \
  --candidate-config "$CAND_CAL_CFG" \
  --local-control-config "$LOCAL_CAL_CFG" \
  --foundation-control-config "$FOUND_CAL_CFG" \
  --dual-calibration-json "$DUAL_CAL_JSON" \
  --report-json "$GATE_REPORT" --latency-target-ms "${V64_LATENCY_TARGET_MS:-500}" \
  "${latency_gate_args[@]}" 2>&1 | tee "$OUT_ROOT/logs/v64_saqa_bcc_gate.out"

read -r PROTOCOL_PASS MINIMUM_PASS COMPETITIVE_PASS < <(python - "$GATE_REPORT" <<'PY_GATE'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
print(int(bool(d.get('protocol_pass'))),int(bool(d.get('minimum_pass'))),int(bool(d.get('competitive_pass'))))
PY_GATE
)
[[ "$PROTOCOL_PASS" == "1" ]] || { echo "[v64] protocol gate failed; closed loop is invalid" >&2; exit 3; }
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
    echo "[v64] minimum gate failed; running paired CL20 as diagnostic evidence only"
    printf 'diagnostic\n' > "$OUT_ROOT/closed_loop/.diagnostic_cl20"
  fi
  : "${NUPLAN_ROOT:?Set NUPLAN_ROOT before the gated closed-loop stage}"
  BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" CL_TOKEN_SPLIT=val_tune \
  GPUS="$GPUS" OUT_ROOT="$OUT_ROOT" RUN_MODE=cl20 \
  V64_CKPT="$BEST_CHECKPOINT" EVAL_CONFIG="$CAND_CAL_CFG" \
  bash run_v64_saqa_bcc.sh

  BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" CL_TOKEN_SPLIT=val_tune \
  GPUS="$GPUS" OUT_ROOT="$LOCAL_CONTROL_ROOT" RUN_MODE=cl20 \
  V64_CKPT="$BEST_CHECKPOINT" EVAL_CONFIG="$LOCAL_CAL_CFG" \
  bash run_v64_saqa_bcc.sh

  BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" CL_TOKEN_SPLIT=val_tune \
  GPUS="$GPUS" OUT_ROOT="$FOUNDATION_CONTROL_ROOT" RUN_MODE=cl20 \
  V64_CKPT="$FOUNDATION_CKPT" EVAL_CONFIG="$FOUND_CAL_CFG" \
  bash run_v64_saqa_bcc.sh

  # Hard closed-loop attribution check: all compared methods must use exactly
  # the same token list.  Persist the proof next to the candidate results.
  python - \
    "$OUT_ROOT/closed_loop/v64_saqa_bcc_20/closed_loop_combined_summary.json" \
    "$LOCAL_CONTROL_ROOT/closed_loop/v64_saqa_bcc_20/closed_loop_combined_summary.json" \
    "$FOUNDATION_CONTROL_ROOT/closed_loop/v64_saqa_bcc_20/closed_loop_combined_summary.json" \
    "$OUT_ROOT/closed_loop/v64_three_way_cl20_protocol.json" <<'PY_CL_PROTOCOL'
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
    V64_CKPT="$BEST_CHECKPOINT" EVAL_CONFIG="$CAND_CAL_CFG" \
    bash run_v64_saqa_bcc.sh
  elif [[ "$RUN_CL100_AFTER_CL20" == "1" ]]; then
    echo "[v64] CL100 requested but competitive gate failed; run paired CL20 first and inspect $GATE_REPORT"
  fi
else
  echo "[v64] closed-loop skipped (RUN_CLOSED_LOOP_AFTER_GATE=$RUN_CLOSED_LOOP_AFTER_GATE, minimum_pass=$MINIMUM_PASS, diagnostic=$RUN_DIAGNOSTIC_CL20_ON_GATE_FAIL)"
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
#   --checkpoint "$OUT_ROOT/train/bdse_v64_saqa_bcc.best.pt" \
#   --split public_set_test --preprocessed-dir "$BDSE_TEST_CACHE" --device cuda \
#   --output "$OUT_ROOT/open_loop/open_loop_v64_saqa_bcc_test.json" \
#   --per-sample-output "$OUT_ROOT/open_loop/open_loop_v64_saqa_bcc_test.jsonl"
