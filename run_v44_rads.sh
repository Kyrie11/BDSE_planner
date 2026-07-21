#!/usr/bin/env bash
set -euo pipefail

# Allow either:
#   GPUS=0,1 RUN_MODE=train_open_loop bash run_v44_rads_2gpu.sh
# or:
#   bash run_v44_rads_2gpu.sh GPUS=0,1 RUN_MODE=train_open_loop
for kv in "$@"; do
  [[ "$kv" == *=* ]] && export "$kv"
done

: "${BDSE_TRAIN_CACHE:?Set BDSE_TRAIN_CACHE to the preprocessed training cache}"
: "${BDSE_VAL_CACHE:?Set BDSE_VAL_CACHE to the preprocessed validation cache}"

V30_CKPT_IN="${V30_CKPT_IN:-outputs_v30/train/bdse_v30_pmvrbsr.best.pt}"
OUT_ROOT="${OUT_ROOT:-outputs_v44_rads_2gpu}"
RUN_MODE="${RUN_MODE:-train_open_loop}"
DEVICE="${DEVICE:-cuda}"
GPUS="${GPUS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29544}"
MAX_TRAIN_SCENARIOS="${MAX_TRAIN_SCENARIOS:-50000}"
VAL_SCENARIOS="${VAL_SCENARIOS:-1000}"
OPEN_LOOP_MAX_SCENARIOS="${OPEN_LOOP_MAX_SCENARIOS:-1000}"
EVAL_CONFIG="${EVAL_CONFIG:-bdse/configs/v44_bdse_rads_fast_cl.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-bdse/configs/v44_bdse_rads_train.yaml}"

# Preserve the original single-GPU global batch size by default:
#   old: 1 GPU x batch 8 = global batch 8
#   new: 2 GPUs x batch 4 = global batch 8
# For a faster but not strictly optimization-equivalent run, set
# BATCH_SIZE_PER_GPU=8 (global batch 16) and retune/record the LR if needed.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
if [[ -z "${BATCH_SIZE_PER_GPU:-}" ]]; then
  if (( GLOBAL_BATCH_SIZE % NPROC_PER_NODE != 0 )); then
    echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by NPROC_PER_NODE=$NPROC_PER_NODE" >&2
    exit 2
  fi
  BATCH_SIZE_PER_GPU=$((GLOBAL_BATCH_SIZE / NPROC_PER_NODE))
fi
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-6}"
VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-2}"
VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-$BATCH_SIZE_PER_GPU}"
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

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p "$OUT_ROOT/train" "$OUT_ROOT/open_loop" "$OUT_ROOT/closed_loop" "$OUT_ROOT/logs"
CKPT="$OUT_ROOT/train/bdse_v44_rads.pt"
BEST_CKPT="$OUT_ROOT/train/bdse_v44_rads.best.pt"

check_checkpoint() {
  local checkpoint="$1"
  [[ -f "$checkpoint" ]] || {
    echo "Missing checkpoint: $checkpoint" >&2
    exit 2
  }
}

wait_two() {
  local pid0="$1" pid1="$2" status=0
  wait "$pid0" || status=$?
  wait "$pid1" || status=$?
  return "$status"
}

train_2gpu() {
  check_checkpoint "$V30_CKPT_IN"
  local effective_global_batch=$((BATCH_SIZE_PER_GPU * NPROC_PER_NODE))
  echo "[v44] DDP training on physical GPUs $GPU0,$GPU1"
  echo "[v44] batch_per_gpu=$BATCH_SIZE_PER_GPU global_batch=$effective_global_batch workers_per_gpu=$NUM_WORKERS_PER_GPU"

  CUDA_VISIBLE_DEVICES="$GPUS" \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    -m bdse.experiments.train \
      --config "$TRAIN_CONFIG" \
      --split train_boston train_pittsburgh train_singapore train_vegas_2 \
      --preprocessed-dir "$BDSE_TRAIN_CACHE" \
      --max-scenarios "$MAX_TRAIN_SCENARIOS" \
      --max-scenarios-per-split $((MAX_TRAIN_SCENARIOS / 4)) \
      --batch-size "$BATCH_SIZE_PER_GPU" \
      --num-workers "$NUM_WORKERS_PER_GPU" \
      --device cuda \
      --amp \
      --warm-start-from "$V30_CKPT_IN" \
      --val-preprocessed-dir "$BDSE_VAL_CACHE" \
      --val-split val \
      --val-max-scenarios "$VAL_SCENARIOS" \
      --val-mode open_loop \
      --val-dense-diagnostic \
      --val-every-n-epochs 1 \
      --val-batch-size "$VAL_BATCH_SIZE_PER_GPU" \
      --val-num-workers "$VAL_NUM_WORKERS_PER_GPU" \
      --best-metric teacher_action_match \
      --best-metrics teacher_action_match teacher_regret full_interface_action_match \
      --log-file "$OUT_ROOT/train/bdse_v44_rads.train_log.jsonl" \
      --output "$CKPT" \
    2>&1 | tee "$OUT_ROOT/logs/train_2gpu.out"
}

prepare_open_loop_shards() {
  local shard_root="$1"
  rm -rf "$shard_root"
  mkdir -p "$shard_root/gpu0/val" "$shard_root/gpu1/val"

  python - "$BDSE_VAL_CACHE" "$shard_root" "$OPEN_LOOP_MAX_SCENARIOS" <<'PY'
import json
import sys
from pathlib import Path

from bdse.data.nuplan_dataset import PreprocessedBDSEDataset

cache_root = Path(sys.argv[1])
shard_root = Path(sys.argv[2])
limit = int(sys.argv[3])
paths = PreprocessedBDSEDataset(cache_root, split=["val"], max_scenarios=limit).build_index()
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
    --output "$OUT_ROOT/open_loop/open_loop_v44_rads.gpu${shard_id}.json" \
    --per-sample-output "$OUT_ROOT/open_loop/open_loop_v44_rads.gpu${shard_id}.jsonl" \
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
summary_paths = [root / f"open_loop_v44_rads.gpu{i}.json" for i in range(2)]
row_paths = [root / f"open_loop_v44_rads.gpu{i}.jsonl" for i in range(2)]

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

(root / "open_loop_v44_rads.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
with (root / "open_loop_v44_rads.jsonl").open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY
}

open_loop_2gpu() {
  local checkpoint="${V44_CKPT:-$BEST_CKPT}"
  check_checkpoint "$checkpoint"
  local shard_root="$OUT_ROOT/open_loop/.two_gpu_shards"
  prepare_open_loop_shards "$shard_root"

  echo "[v44] Open-loop evaluation split across physical GPUs $GPU0 and $GPU1"
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

  python - "$BDSE_VAL_CACHE" "$token_root" "$limit" "$CL_TOKEN_SCAN_MAX" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset

cache_root = Path(sys.argv[1])
token_root = Path(sys.argv[2])
limit = int(sys.argv[3])
scan_max = max(limit, int(sys.argv[4]))
paths = PreprocessedBDSEDataset(cache_root, split=["val"], max_scenarios=scan_max).build_index()

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
    "selection": "first unique tokens from deterministic val-cache order, alternated across GPUs",
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
    --experiment-uid "v44_rads_${limit}_gpu${shard_id}" \
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
  local checkpoint="${V44_CKPT:-$BEST_CKPT}"
  check_checkpoint "$checkpoint"

  local run_root="$OUT_ROOT/closed_loop/v44_rads_${limit}"
  local token_root="$run_root/scenario_token_shards"
  rm -rf "$run_root"
  mkdir -p "$run_root"
  prepare_closed_loop_token_shards "$limit" "$token_root"

  echo "[v44] Closed-loop $limit scenarios split across physical GPUs $GPU0 and $GPU1"
  echo "[v44] Reuse $token_root/scenario_tokens_all.json for every compared method to keep the CL subset identical."
  local start_time end_time
  start_time=$(date +%s)
  run_closed_loop_shard "$GPU0" 0 "$limit" "$checkpoint" "$token_root" "$run_root" & local pid0=$!
  run_closed_loop_shard "$GPU1" 1 "$limit" "$checkpoint" "$token_root" "$run_root" & local pid1=$!
  wait_two "$pid0" "$pid1"
  end_time=$(date +%s)
  merge_closed_loop_summaries "$run_root" "$token_root" "$((end_time - start_time))"
}

case "$RUN_MODE" in
  train) train_2gpu ;;
  open_loop) open_loop_2gpu ;;
  train_open_loop) train_2gpu; open_loop_2gpu ;;
  cl20) closed_loop_2gpu 20 ;;
  cl50) closed_loop_2gpu 50 ;;
  cl100) closed_loop_2gpu 100 ;;
  *)
    echo "Unknown RUN_MODE=$RUN_MODE (train|open_loop|train_open_loop|cl20|cl50|cl100)" >&2
    exit 2
    ;;
esac
