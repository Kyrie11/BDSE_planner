#!/usr/bin/env bash
set -euo pipefail

: "${BDSE_SPLIT_CACHE:?Set BDSE_SPLIT_CACHE}"
: "${CANDIDATE_CONFIG:?Set CANDIDATE_CONFIG to the calibrated V58 candidate YAML}"
: "${LOCAL_CONFIG:?Set LOCAL_CONFIG to the calibrated local-control YAML}"
: "${FOUNDATION_CONFIG:?Set FOUNDATION_CONFIG to the calibrated foundation-control YAML}"
: "${CANDIDATE_CKPT:?Set CANDIDATE_CKPT}"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT}"

GPUS="${GPUS:-0,1}"
BENCH_SPLIT="${BENCH_SPLIT:-val_eval}"
BENCH_SCENARIOS="${BENCH_SCENARIOS:-120}"
BENCH_ROOT="${BENCH_ROOT:-outputs_v58_open_loop_concurrency_benchmark}"
WORKER_SETTINGS="${WORKER_SETTINGS:-1 2 3}"
mkdir -p "$BENCH_ROOT"

for workers in $WORKER_SETTINGS; do
  root="$BENCH_ROOT/workers_per_gpu_${workers}"
  rm -rf "$root"
  python -m bdse.tools.run_parallel_open_loop_suite \
    --system "candidate::${CANDIDATE_CONFIG}::${CANDIDATE_CKPT}" \
    --system "local::${LOCAL_CONFIG}::${CANDIDATE_CKPT}" \
    --system "foundation::${FOUNDATION_CONFIG}::${FOUNDATION_CKPT}" \
    --preprocessed-dir "$BDSE_SPLIT_CACHE" \
    --split "$BENCH_SPLIT" \
    --max-scenarios "$BENCH_SCENARIOS" \
    --output-root "$root" \
    --gpus "$GPUS" \
    --workers-per-gpu "$workers" \
    --device cuda \
    --disable-dense-diagnostic \
    > "$root.stdout.log" 2>&1
  echo "finished workers_per_gpu=$workers"
done

python - "$BENCH_ROOT" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
rows = []
for report in sorted(root.glob("workers_per_gpu_*/parallel_open_loop_suite_report.json")):
    data = json.loads(report.read_text(encoding="utf-8"))
    rows.append({
        "workers_per_gpu": data["workers_per_gpu"],
        "num_worker_slots": data["num_worker_slots"],
        "suite_wall_time_s": data["suite_wall_time_s"],
        "paired_protocol_pass": data["paired_protocol_pass"],
        "candidate_latency_p95_ms": data["summaries"]["candidate"].get("planner_latency_ms_p95"),
    })
rows.sort(key=lambda row: row["suite_wall_time_s"])
(root / "benchmark_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
print(f"Recommended OPEN_LOOP_WORKERS_PER_GPU={rows[0]['workers_per_gpu']} for this machine/checkpoint")
PY
