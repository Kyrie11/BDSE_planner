#!/usr/bin/env bash
set -euo pipefail

# Allow either:
#   GPUS=0,1 RUN_MODE=train_open_loop bash run_v50_dbap_ri.sh
# or:
#   bash run_v50_dbap_ri.sh GPUS=0,1 RUN_MODE=train_open_loop
for kv in "$@"; do
  [[ "$kv" == *=* ]] && export "$kv"
done

: "${BDSE_TRAIN_CACHE:?Set BDSE_TRAIN_CACHE to the preprocessed training cache}"
: "${BDSE_VAL_CACHE:?Set BDSE_VAL_CACHE to the preprocessed validation cache}"

FOUNDATION_CKPT="${FOUNDATION_CKPT:-${V30_CKPT_IN:-}}"
V30_CKPT_IN="$FOUNDATION_CKPT"  # backwards-compatible alias; no hard-coded outputs_v30 dependency
INIT_MODE="${INIT_MODE:-warm_start}"  # warm_start | scratch
FOUNDATION_CONFIG="${FOUNDATION_CONFIG:-bdse/configs/v50_rebuild_v30_from_scratch_2gpu.yaml}"
FOUNDATION_OUT_ROOT="${FOUNDATION_OUT_ROOT:-outputs_v30_rebuilt_current_code}"
OUT_ROOT="${OUT_ROOT:-outputs_v50_dbap_ri_2gpu}"
RUN_MODE="${RUN_MODE:-train_open_loop}"
DEVICE="${DEVICE:-cuda}"
GPUS="${GPUS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29545}"
MAX_TRAIN_SCENARIOS="${MAX_TRAIN_SCENARIOS:-50000}"
VAL_SCENARIOS="${VAL_SCENARIOS:-500}"
OPEN_LOOP_MAX_SCENARIOS="${OPEN_LOOP_MAX_SCENARIOS:-1000}"
EVAL_CONFIG="${EVAL_CONFIG:-bdse/configs/v50_bdse_dbap_ri_cl.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-bdse/configs/v50_bdse_dbap_ri_train_2gpu.yaml}"
# Paper-grade protocol: checkpoint/hyperparameter selection uses val_tune only;
# one-sided residual calibration uses the log-disjoint val_calib manifest only.
VAL_SPLIT="${VAL_SPLIT:-val_tune}"
OPEN_LOOP_SPLIT="${OPEN_LOOP_SPLIT:-$VAL_SPLIT}"
CL_TOKEN_SPLIT="${CL_TOKEN_SPLIT:-$OPEN_LOOP_SPLIT}"

# Make the per-GPU batch explicit.  The previous script silently defaulted to
# GLOBAL_BATCH_SIZE=24, which produced batch_per_gpu=12 whenever an environment
# assignment was lost by a wrapper/nohup command.
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
if (( BATCH_SIZE_PER_GPU < 1 )); then
  echo "BATCH_SIZE_PER_GPU must be >= 1" >&2
  exit 2
fi
GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NPROC_PER_NODE))
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-4}"
VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-2}"
VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-$BATCH_SIZE_PER_GPU}"
VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-3}"
VAL_DENSE_DIAGNOSTIC="${VAL_DENSE_DIAGNOSTIC:-0}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-0}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-2000}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_FROM="${RESUME_FROM:-}"
DETACH="${DETACH:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
EXACT_SELECTOR_THREADS_PER_RANK="${EXACT_SELECTOR_THREADS_PER_RANK:-4}" # compatibility alias
EXACT_SELECTOR_WORKERS_PER_RANK="${EXACT_SELECTOR_WORKERS_PER_RANK:-$EXACT_SELECTOR_THREADS_PER_RANK}"
EXACT_SELECTOR_CPU_BACKEND="${EXACT_SELECTOR_CPU_BACKEND:-process}"
SELECTOR_SCENES_PER_RANK="${SELECTOR_SCENES_PER_RANK:-0}"
SELECTOR_EVERY_N_STEPS="${SELECTOR_EVERY_N_STEPS:-1}"
SELECTOR_FULL_LAST_N_STEPS="${SELECTOR_FULL_LAST_N_STEPS:-0}"
EXACT_DISTILL_SCENES_PER_RANK="${EXACT_DISTILL_SCENES_PER_RANK:-0}"
EXACT_DISTILL_EVERY_N_STEPS="${EXACT_DISTILL_EVERY_N_STEPS:-1}"
CL_WORKERS_PER_GPU="${CL_WORKERS_PER_GPU:-2}"
CL_TOKEN_SCAN_MAX="${CL_TOKEN_SCAN_MAX:-2000}"

IFS=',' read -r GPU0 GPU1 GPU_EXTRA <<< "$GPUS"
if [[ -z "${GPU0:-}" || -z "${GPU1:-}" || -n "${GPU_EXTRA:-}" ]]; then
  echo "GPUS must contain exactly two comma-separated GPU ids, e.g. GPUS=0,1" >&2
  exit 2
fi
if [[ "$NPROC_PER_NODE" -ne 2 ]]; then
  echo "This script is designed for exactly two GPUs; set NPROC_PER_NODE=2." >&2
  exit 2
fi
if [[ "$DEVICE" != "cuda" ]]; then
  echo "The two-GPU path requires DEVICE=cuda." >&2
  exit 2
fi
if [[ "$INIT_MODE" != "warm_start" && "$INIT_MODE" != "scratch" ]]; then
  echo "INIT_MODE must be warm_start or scratch" >&2
  exit 2
fi
if (( EXACT_SELECTOR_WORKERS_PER_RANK < 1 )); then
  echo "EXACT_SELECTOR_WORKERS_PER_RANK must be >= 1" >&2
  exit 2
fi
case "$EXACT_SELECTOR_CPU_BACKEND" in
  sequential|thread|process) ;;
  *) echo "EXACT_SELECTOR_CPU_BACKEND must be sequential, thread, or process" >&2; exit 2 ;;
esac

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# Do not set expandable_segments by default: the uploaded log shows that the
# installed CUDA allocator does not support it.  A user-provided value is kept.
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF
fi

mkdir -p "$OUT_ROOT/train" "$OUT_ROOT/open_loop" "$OUT_ROOT/closed_loop" "$OUT_ROOT/logs"

# torchrun installs its own SIGHUP handler.  Plain `nohup ... &` can therefore
# still be terminated when an SSH session or parent process group disappears.
# DETACH=1 starts a new session with setsid and writes a stable launcher log.
if [[ "$DETACH" == "1" && "${BDSE_DETACHED_CHILD:-0}" != "1" ]]; then
  command -v setsid >/dev/null 2>&1 || { echo "DETACH=1 requires the setsid command" >&2; exit 2; }
  export BDSE_DETACHED_CHILD=1 DETACH=0
  launcher_log="$OUT_ROOT/logs/launcher_$(date +%Y%m%d_%H%M%S).log"
  script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  setsid nohup bash "$script_path" "$@" </dev/null >>"$launcher_log" 2>&1 &
  child_pid=$!
  echo "$child_pid" > "$OUT_ROOT/logs/train.pid"
  echo "[v50] detached session started pid=$child_pid log=$launcher_log"
  exit 0
fi

# A single output root must have a single writer.  V47 was accidentally launched
# twice and produced duplicate epoch rows plus competing checkpoint writes.
RUN_LOCK_DIR="$OUT_ROOT/.v50_run.lock"
if [[ "${ALLOW_CONCURRENT_OUT_ROOT:-0}" != "1" ]]; then
  if ! mkdir "$RUN_LOCK_DIR" 2>/dev/null; then
    owner="$(cat "$RUN_LOCK_DIR/pid" 2>/dev/null || echo unknown)"
    echo "Output root is already locked by pid=$owner: $OUT_ROOT" >&2
    echo "Use a fresh OUT_ROOT.  Set ALLOW_CONCURRENT_OUT_ROOT=1 only for read-only debugging." >&2
    exit 3
  fi
  echo "$$" > "$RUN_LOCK_DIR/pid"
  printf '%s\n' "$RUN_MODE" > "$RUN_LOCK_DIR/run_mode"
  trap 'rm -rf "$RUN_LOCK_DIR"' EXIT INT TERM
fi
CKPT="$OUT_ROOT/train/bdse_v50_dbap_ri.pt"
BEST_CKPT="$OUT_ROOT/train/bdse_v50_dbap_ri.best.pt"
LATEST_CKPT="$OUT_ROOT/train/bdse_v50_dbap_ri.latest.pt"

check_checkpoint() {
  local checkpoint="${1:-}"
  if [[ -z "$checkpoint" ]]; then
    echo "Missing checkpoint path: resolve FOUNDATION_CKPT before warm-start training" >&2
    exit 2
  fi
  [[ -f "$checkpoint" ]] || {
    echo "Missing checkpoint: $checkpoint" >&2
    exit 2
  }
}

discover_latest_checkpoint() {
  local latest_path="${1:-$LATEST_CKPT}"
  local checkpoint_dir_path="${2:-$OUT_ROOT/train/checkpoints}"
  local final_path="${3:-$CKPT}"
  python - "$latest_path" "$checkpoint_dir_path" "$final_path" <<'PY'
import sys
from pathlib import Path

import torch

latest = Path(sys.argv[1])
checkpoint_dir = Path(sys.argv[2])
legacy_final = Path(sys.argv[3])
candidates = [latest, legacy_final]
if checkpoint_dir.is_dir():
    candidates.extend(checkpoint_dir.glob("*.pt"))

best = None
for path in dict.fromkeys(candidates):
    if not path.is_file():
        continue
    try:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[v50] ignore unreadable checkpoint {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        continue
    # A final inference-only model is not resumable; optimizer and epoch state
    # are required to continue the same training run.
    if not isinstance(state, dict) or "optimizer" not in state or "epoch" not in state:
        continue
    next_epoch = int(state.get("next_epoch", int(state.get("epoch", -1)) + 1))
    next_batch = max(0, int(state.get("next_batch_index", 0)))
    score = (next_epoch, next_batch, path.stat().st_mtime_ns)
    if best is None or score > best[0]:
        best = (score, path)

if best is not None:
    print(best[1])
PY
}

wait_two() {
  local pid0="$1" pid1="$2" status=0
  wait "$pid0" || status=$?
  wait "$pid1" || status=$?
  return "$status"
}

train_2gpu() {
  local effective_global_batch=$((BATCH_SIZE_PER_GPU * NPROC_PER_NODE))
  local checkpoint_args=()
  local resume_checkpoint=""
  if [[ -n "$RESUME_FROM" ]]; then
    check_checkpoint "$RESUME_FROM"
    resume_checkpoint="$RESUME_FROM"
  elif [[ "$AUTO_RESUME" == "1" || "$AUTO_RESUME" == "true" ]]; then
    resume_checkpoint="$(discover_latest_checkpoint)"
  fi
  if [[ -n "$resume_checkpoint" ]]; then
    checkpoint_args+=(--resume-from "$resume_checkpoint")
    echo "[v50] resume checkpoint=$resume_checkpoint"
  elif [[ "$INIT_MODE" == "warm_start" ]]; then
    check_checkpoint "$V30_CKPT_IN"
    checkpoint_args+=(--warm-start-from "$V30_CKPT_IN")
    echo "[v50] warm-start checkpoint=$V30_CKPT_IN"
  else
    echo "[v50] training from random initialization (INIT_MODE=scratch)"
  fi
  echo "[v50] DDP training on physical GPUs $GPU0,$GPU1"
  echo "[v50] batch_per_gpu=$BATCH_SIZE_PER_GPU global_batch=$effective_global_batch workers_per_gpu=$NUM_WORKERS_PER_GPU"
  echo "[v50] train_config=$TRAIN_CONFIG val_split=$VAL_SPLIT val_scenarios=$VAL_SCENARIOS val_every=$VAL_EVERY_N_EPOCHS dense_val=$VAL_DENSE_DIAGNOSTIC"
  echo "[v50] auto_resume=$AUTO_RESUME save_every_n_steps=$SAVE_EVERY_N_STEPS init_mode=$INIT_MODE"
  echo "[v50] exact_selector_cpu_backend=$EXACT_SELECTOR_CPU_BACKEND workers_per_rank=$EXACT_SELECTOR_WORKERS_PER_RANK"
  echo "[v50-dbap] exact-budget nested certificate masks on every local scene"

  local val_dense_args=()
  if [[ "$VAL_DENSE_DIAGNOSTIC" == "1" || "$VAL_DENSE_DIAGNOSTIC" == "true" ]]; then
    val_dense_args+=(--val-dense-diagnostic)
  fi

  CUDA_VISIBLE_DEVICES="$GPUS" \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    --max_restarts=0 \
    -m bdse.experiments.train \
      --config "$TRAIN_CONFIG" \
      --split train_boston train_pittsburgh train_singapore train_vegas_2 \
      --preprocessed-dir "$BDSE_TRAIN_CACHE" \
      --max-scenarios "$MAX_TRAIN_SCENARIOS" \
      --max-scenarios-per-split $((MAX_TRAIN_SCENARIOS / 4)) \
      --batch-size "$BATCH_SIZE_PER_GPU" \
      --num-workers "$NUM_WORKERS_PER_GPU" \
      --prefetch-factor "$PREFETCH_FACTOR" \
      --selector-scenes-per-rank "$SELECTOR_SCENES_PER_RANK" \
      --selector-every-n-steps "$SELECTOR_EVERY_N_STEPS" \
      --selector-full-last-n-steps "$SELECTOR_FULL_LAST_N_STEPS" \
      --exact-distill-scenes-per-rank "$EXACT_DISTILL_SCENES_PER_RANK" \
      --exact-distill-every-n-steps "$EXACT_DISTILL_EVERY_N_STEPS" \
      --selector-cpu-workers "$EXACT_SELECTOR_WORKERS_PER_RANK" \
      --selector-cpu-backend "$EXACT_SELECTOR_CPU_BACKEND" \
      --device cuda \
      --amp \
      "${checkpoint_args[@]}" \
      --val-preprocessed-dir "$BDSE_VAL_CACHE" \
      --val-split "$VAL_SPLIT" \
      --val-max-scenarios "$VAL_SCENARIOS" \
      --val-mode open_loop \
      "${val_dense_args[@]}" \
      --val-every-n-epochs "$VAL_EVERY_N_EPOCHS" \
      --val-batch-size "$VAL_BATCH_SIZE_PER_GPU" \
      --val-num-workers "$VAL_NUM_WORKERS_PER_GPU" \
      --save-every-n-epochs "$SAVE_EVERY_N_EPOCHS" \
      --save-every-n-steps "$SAVE_EVERY_N_STEPS" \
      --best-metric fixed_budget_critical_score \
      --best-metrics fixed_budget_critical_score teacher_action_match teacher_regret full_interface_action_match \
      --log-file "$OUT_ROOT/train/bdse_v50_dbap_ri.train_log.jsonl" \
      --output "$CKPT" \
    2>&1 | tee -a "$OUT_ROOT/logs/train_2gpu.out"
}

train_foundation_2gpu() {
  local foundation_ckpt="$FOUNDATION_OUT_ROOT/train/bdse_v30_pmvrbsr_rebuilt.pt"
  local foundation_best="$FOUNDATION_OUT_ROOT/train/bdse_v30_pmvrbsr_rebuilt.best.pt"
  mkdir -p "$FOUNDATION_OUT_ROOT/train" "$FOUNDATION_OUT_ROOT/logs"
  if [[ "$AUTO_RESUME" != "0" && -s "$foundation_best" && -s "$foundation_ckpt" ]]; then
    echo "[v50-foundation] reuse rebuilt foundation checkpoint=$foundation_best"
    return 0
  fi
  local foundation_checkpoint_args=()
  if [[ "$AUTO_RESUME" != "0" ]]; then
    local foundation_resume=""
    foundation_resume="$(discover_latest_checkpoint \
      "$FOUNDATION_OUT_ROOT/train/bdse_v30_pmvrbsr_rebuilt.latest.pt" \
      "$FOUNDATION_OUT_ROOT/train/checkpoints" \
      "$foundation_ckpt")"
    if [[ -n "$foundation_resume" ]]; then
      foundation_checkpoint_args+=(--resume-from "$foundation_resume")
      echo "[v50-foundation] resume checkpoint=$foundation_resume"
    fi
  fi
  if [[ ${#foundation_checkpoint_args[@]} -eq 0 ]]; then
    echo "[v50-foundation] train v30-compatible foundation from random initialization"
  fi
  echo "[v50-foundation] config=$FOUNDATION_CONFIG output=$foundation_ckpt"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$((MASTER_PORT + 1))" \
    --max_restarts=0 \
    -m bdse.experiments.train \
      --config "$FOUNDATION_CONFIG" \
      --split train_boston train_pittsburgh train_singapore train_vegas_2 \
      --preprocessed-dir "$BDSE_TRAIN_CACHE" \
      --max-scenarios "$MAX_TRAIN_SCENARIOS" \
      --max-scenarios-per-split $((MAX_TRAIN_SCENARIOS / 4)) \
      --batch-size "$BATCH_SIZE_PER_GPU" \
      --num-workers "$NUM_WORKERS_PER_GPU" \
      --prefetch-factor "$PREFETCH_FACTOR" \
      --selector-cpu-workers 1 \
      --selector-cpu-backend sequential \
      --device cuda \
      --amp \
      "${foundation_checkpoint_args[@]}" \
      --val-preprocessed-dir "$BDSE_VAL_CACHE" \
      --val-split "$VAL_SPLIT" \
      --val-max-scenarios "$VAL_SCENARIOS" \
      --val-mode open_loop \
      --val-every-n-epochs 1 \
      --val-batch-size "$VAL_BATCH_SIZE_PER_GPU" \
      --val-num-workers "$VAL_NUM_WORKERS_PER_GPU" \
      --save-every-n-epochs 0 \
      --save-every-n-steps "$SAVE_EVERY_N_STEPS" \
      --best-metric teacher_action_match \
      --best-metrics teacher_action_match teacher_regret full_interface_action_match \
      --log-file "$FOUNDATION_OUT_ROOT/train/bdse_v30_pmvrbsr_rebuilt.train_log.jsonl" \
      --output "$foundation_ckpt" \
    2>&1 | tee -a "$FOUNDATION_OUT_ROOT/logs/train_foundation_2gpu.out"
  if [[ ! -s "$foundation_best" && -s "$foundation_ckpt" ]]; then
    echo "[v50-foundation] warning: metric-specific best was not emitted; use the clean final checkpoint" >&2
    cp -f "$foundation_ckpt" "$foundation_best"
  fi
  check_checkpoint "$foundation_best"
  echo "[v50-foundation] rebuilt best checkpoint=$foundation_best"
}

prepare_open_loop_shards() {
  local shard_root="$1"
  rm -rf "$shard_root"
  mkdir -p "$shard_root/gpu0/val" "$shard_root/gpu1/val"

  python - "$BDSE_VAL_CACHE" "$shard_root" "$OPEN_LOOP_MAX_SCENARIOS" "$OPEN_LOOP_SPLIT" <<'PY'
import json
import sys
from pathlib import Path

from bdse.data.nuplan_dataset import PreprocessedBDSEDataset

cache_root = Path(sys.argv[1])
shard_root = Path(sys.argv[2])
limit = int(sys.argv[3])
source_split = str(sys.argv[4])
paths = PreprocessedBDSEDataset(cache_root, split=[source_split], max_scenarios=limit).build_index()
if not paths:
    raise SystemExit("No validation samples found")

records = [[], []]
for index, path in enumerate(paths):
    shard = index % 2
    records[shard].append({
        "split": "val",
        "path": str(path.resolve()),
        "original_index": index,
    })

for shard in range(2):
    manifest = shard_root / f"gpu{shard}" / "val" / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records[shard]),
        encoding="utf-8",
    )

meta = {
    "source_split": source_split,
    "requested_scenarios": limit,
    "selected_scenarios": len(paths),
    "gpu0_scenarios": len(records[0]),
    "gpu1_scenarios": len(records[1]),
}
(shard_root / "shard_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, sort_keys=True))
PY
}

run_open_loop_shard() {
  local physical_gpu="$1" shard_id="$2" checkpoint="$3" shard_root="$4"
  local shard_cache="$shard_root/gpu${shard_id}/val"
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  python -m bdse.experiments.evaluate_open_loop \
    --config "$EVAL_CONFIG" \
    --checkpoint "$checkpoint" \
    --split val \
    --preprocessed-dir "$shard_cache" \
    --device cuda \
    --output "$OUT_ROOT/open_loop/open_loop_v50_dbap_ri.gpu${shard_id}.json" \
    --per-sample-output "$OUT_ROOT/open_loop/open_loop_v50_dbap_ri.gpu${shard_id}.jsonl" \
    > "$OUT_ROOT/logs/open_loop_gpu${shard_id}.out" 2>&1
}

merge_open_loop_shards() {
  local wall_seconds="$1"
  python - "$OUT_ROOT/open_loop" "$wall_seconds" <<'PY'
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
wall_seconds = float(sys.argv[2])
summary_paths = [root / f"open_loop_v50_dbap_ri.gpu{i}.json" for i in range(2)]
row_paths = [root / f"open_loop_v50_dbap_ri.gpu{i}.jsonl" for i in range(2)]

summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
shard_rows = []
for path in row_paths:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    shard_rows.append(rows)

# Restore the original alternating dataset order.
rows = []
for pair in itertools.zip_longest(*shard_rows):
    rows.extend(item for item in pair if item is not None)

summary_keys = set().union(*(s.keys() for s in summaries))
special = {
    "device",
    "cuda_peak_memory_mb",
    "planner_latency_ms_mean",
    "planner_latency_ms_p50",
    "planner_latency_ms_p90",
    "planner_latency_ms_p95",
    "planner_latency_ms_p99",
    "planner_latency_ms_max",
}
out = {}
for key in sorted(summary_keys - special):
    vals = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (bool, int, float)):
            value = float(value)
            if math.isfinite(value):
                vals.append(value)
    if vals:
        out[key] = float(np.mean(np.asarray(vals, dtype=np.float64)))

latencies = np.asarray(
    [float(row["planner_latency_ms"]) for row in rows if math.isfinite(float(row.get("planner_latency_ms", float("nan"))))],
    dtype=np.float64,
)
if latencies.size:
    out.update({
        "planner_latency_ms_mean": float(latencies.mean()),
        "planner_latency_ms_p50": float(np.quantile(latencies, 0.50)),
        "planner_latency_ms_p90": float(np.quantile(latencies, 0.90)),
        "planner_latency_ms_p95": float(np.quantile(latencies, 0.95)),
        "planner_latency_ms_p99": float(np.quantile(latencies, 0.99)),
        "planner_latency_ms_max": float(latencies.max()),
    })

peaks = [float(s.get("cuda_peak_memory_mb", float("nan"))) for s in summaries]
finite_peaks = [x for x in peaks if math.isfinite(x)]
out["device"] = "2xCUDA (independent open-loop shards)"
out["num_scenarios"] = len(rows)
out["parallel_evaluation_wall_time_s"] = wall_seconds
if finite_peaks:
    out["cuda_peak_memory_mb"] = max(finite_peaks)
for idx, peak in enumerate(peaks):
    if math.isfinite(peak):
        out[f"cuda_peak_memory_mb_gpu{idx}"] = peak

(root / "open_loop_v50_dbap_ri.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
with (root / "open_loop_v50_dbap_ri.jsonl").open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY
}

open_loop_2gpu() {
  local checkpoint="${V50_CKPT:-$BEST_CKPT}"
  check_checkpoint "$checkpoint"
  local shard_root="$OUT_ROOT/open_loop/.two_gpu_shards"
  prepare_open_loop_shards "$shard_root"

  echo "[v50] Open-loop source_split=$OPEN_LOOP_SPLIT across physical GPUs $GPU0 and $GPU1"
  local start_time end_time
  start_time=$(date +%s)
  run_open_loop_shard "$GPU0" 0 "$checkpoint" "$shard_root" & local pid0=$!
  run_open_loop_shard "$GPU1" 1 "$checkpoint" "$shard_root" & local pid1=$!
  wait_two "$pid0" "$pid1"
  end_time=$(date +%s)
  merge_open_loop_shards "$((end_time - start_time))"
}

prepare_closed_loop_token_shards() {
  local limit="$1" token_root="$2"
  rm -rf "$token_root"
  mkdir -p "$token_root"

  python - "$BDSE_VAL_CACHE" "$token_root" "$limit" "$CL_TOKEN_SCAN_MAX" "$CL_TOKEN_SPLIT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset

cache_root = Path(sys.argv[1])
token_root = Path(sys.argv[2])
limit = int(sys.argv[3])
scan_max = max(limit, int(sys.argv[4]))
source_split = str(sys.argv[5])
paths = PreprocessedBDSEDataset(cache_root, split=[source_split], max_scenarios=scan_max).build_index()

tokens = []
seen = set()
for path in paths:
    try:
        with np.load(path, allow_pickle=False) as z:
            value = z["scenario_token"]
            token = str(value.item() if value.shape == () else value.reshape(-1)[0])
    except Exception:
        continue
    if token and token not in seen:
        seen.add(token)
        tokens.append(token)
    if len(tokens) >= limit:
        break

if len(tokens) < limit:
    raise SystemExit(f"Only found {len(tokens)} unique scenario tokens; need {limit}. Increase CL_TOKEN_SCAN_MAX.")

shards = [tokens[0::2], tokens[1::2]]
(token_root / "scenario_tokens_all.json").write_text(json.dumps(tokens, indent=2), encoding="utf-8")
for shard_id, shard_tokens in enumerate(shards):
    (token_root / f"scenario_tokens_gpu{shard_id}.json").write_text(json.dumps(shard_tokens, indent=2), encoding="utf-8")
    override = "scenario_filter.scenario_tokens=" + json.dumps(shard_tokens, separators=(",", ":"))
    (token_root / f"scenario_tokens_gpu{shard_id}.override").write_text(override, encoding="utf-8")

meta = {
    "total": len(tokens),
    "gpu0": len(shards[0]),
    "gpu1": len(shards[1]),
    "source_split": source_split,
    "selection": "first unique tokens from deterministic source-cache order, alternated across GPUs",
}
(token_root / "shard_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, sort_keys=True))
PY
}

run_closed_loop_shard() {
  local physical_gpu="$1" shard_id="$2" limit="$3" checkpoint="$4" token_root="$5" run_root="$6"
  local token_override token_count
  token_override="$(cat "$token_root/scenario_tokens_gpu${shard_id}.override")"
  token_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$token_root/scenario_tokens_gpu${shard_id}.json")"

  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  python -m bdse.experiments.evaluate_closed_loop \
    --config "$EVAL_CONFIG" \
    --checkpoint "$checkpoint" \
    --device cuda \
    --challenge closed_loop_nonreactive_agents \
    --metric-aggregator closed_loop_nonreactive_agents_weighted_average \
    --output-dir "$run_root/gpu${shard_id}" \
    --experiment-uid "v50_dbap_ri_${limit}_gpu${shard_id}" \
    --nuplan-module nuplan.planning.script.run_simulation \
    --scenario-builder nuplan \
    --worker single_machine_thread_pool \
    --hydra-full-error \
    --nuplan-data-root "$NUPLAN_ROOT" \
    --nuplan-map-root "$NUPLAN_ROOT/maps" \
    --nuplan-exp-root "$NUPLAN_ROOT/exp" \
    --nuplan-db-root "${NUPLAN_DB_ROOT:-$NUPLAN_ROOT/data/cache/val/}" \
    -- \
    "$token_override" \
    scenario_filter.limit_total_scenarios="$token_count" \
    scenario_filter.shuffle=false \
    worker.max_workers="$CL_WORKERS_PER_GPU" \
    run_metric=true \
    > "$OUT_ROOT/logs/closed_loop_${limit}_gpu${shard_id}.out" 2>&1
}

merge_closed_loop_summaries() {
  local run_root="$1" token_root="$2" wall_seconds="$3"
  python - "$run_root" "$token_root" "$wall_seconds" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd

run_root = Path(sys.argv[1])
token_root = Path(sys.argv[2])
wall_seconds = float(sys.argv[3])

rows = []
for shard_id in range(2):
    files = sorted((run_root / f"gpu{shard_id}").glob("**/aggregator_metric/*.parquet"))
    if not files:
        raise SystemExit(f"No aggregator metric parquet found for GPU shard {shard_id}")
    df = pd.read_parquet(files[0])
    final = df[df["scenario"] == "final_score"]
    if final.empty:
        raise SystemExit(f"No final_score row in {files[0]}")
    count = len(json.loads((token_root / f"scenario_tokens_gpu{shard_id}.json").read_text(encoding="utf-8")))
    row = {"shard": f"gpu{shard_id}", "scenario_count": count, "metric_file": str(files[0])}
    for key, value in final.iloc[0].items():
        if isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
            row[str(key)] = float(value)
    rows.append(row)

weight_total = sum(int(row["scenario_count"]) for row in rows)
combined = {
    "shard": "combined_weighted",
    "scenario_count": weight_total,
    "parallel_closed_loop_wall_time_s": wall_seconds,
}
keys = set(rows[0]).intersection(rows[1]) - {"shard", "scenario_count", "metric_file"}
for key in sorted(keys):
    values = [row.get(key) for row in rows]
    if all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values):
        combined[key] = sum(float(row[key]) * int(row["scenario_count"]) for row in rows) / weight_total

all_rows = rows + [combined]
pd.DataFrame(all_rows).to_csv(run_root / "closed_loop_shard_and_combined_summary.csv", index=False)
(run_root / "closed_loop_combined_summary.json").write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
print(pd.DataFrame(all_rows).to_string(index=False))
PY
}

closed_loop_2gpu() {
  local limit="$1"
  : "${NUPLAN_ROOT:?Set NUPLAN_ROOT for closed-loop evaluation}"
  local checkpoint="${V50_CKPT:-$BEST_CKPT}"
  check_checkpoint "$checkpoint"

  local run_root="$OUT_ROOT/closed_loop/v50_dbap_ri_${limit}"
  local token_root="$run_root/scenario_token_shards"
  rm -rf "$run_root"
  mkdir -p "$run_root"
  prepare_closed_loop_token_shards "$limit" "$token_root"

  echo "[v50] Closed-loop token_source_split=$CL_TOKEN_SPLIT, $limit scenarios across physical GPUs $GPU0 and $GPU1"
  echo "[v50] Reuse $token_root/scenario_tokens_all.json for every compared method to keep the CL subset identical."
  local start_time end_time
  start_time=$(date +%s)
  run_closed_loop_shard "$GPU0" 0 "$limit" "$checkpoint" "$token_root" "$run_root" & local pid0=$!
  run_closed_loop_shard "$GPU1" 1 "$limit" "$checkpoint" "$token_root" "$run_root" & local pid1=$!
  wait_two "$pid0" "$pid1"
  end_time=$(date +%s)
  merge_closed_loop_summaries "$run_root" "$token_root" "$((end_time - start_time))"
}

case "$RUN_MODE" in
  foundation) train_foundation_2gpu ;;
  train) train_2gpu ;;
  open_loop) open_loop_2gpu ;;
  train_open_loop) train_2gpu; open_loop_2gpu ;;
  cl20) closed_loop_2gpu 20 ;;
  cl50) closed_loop_2gpu 50 ;;
  cl100) closed_loop_2gpu 100 ;;
  *)
    echo "Unknown RUN_MODE=$RUN_MODE (foundation|train|open_loop|train_open_loop|cl20|cl50|cl100)" >&2
    exit 2
    ;;
esac
