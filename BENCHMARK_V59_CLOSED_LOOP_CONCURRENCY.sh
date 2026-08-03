#!/usr/bin/env bash
set -euo pipefail

: "${NUPLAN_ROOT:?Set NUPLAN_ROOT}"
: "${BDSE_SPLIT_CACHE:?Set BDSE_SPLIT_CACHE}"
: "${CANDIDATE_CONFIG:?Set CANDIDATE_CONFIG to a frozen calibrated candidate yaml}"
: "${CANDIDATE_CKPT:?Set CANDIDATE_CKPT to a frozen checkpoint}"
GPUS="${GPUS:-0,1}"
BENCH_ROOT="${BENCH_ROOT:-outputs_v59_closed_loop_concurrency_benchmark}"
PROCESS_SETTINGS="${PROCESS_SETTINGS:-1 2 3}"
mkdir -p "$BENCH_ROOT"

python - "$BENCH_ROOT" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1], 'benchmark_summary.json').write_text(json.dumps({'runs': []}, indent=2))
PY

for p in $PROCESS_SETTINGS; do
  out="$BENCH_ROOT/processes_per_gpu_${p}"
  rm -rf "$out"
  start=$(date +%s)
  BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE" CL_TOKEN_SPLIT=val_tune \
  GPUS="$GPUS" OUT_ROOT="$out" RUN_MODE=cl4 \
  CL_PROCESSES_PER_GPU="$p" CL_WORKERS_PER_GPU=1 CL_RENDER_SUMMARY_PDF=0 \
  V59_CKPT="$CANDIDATE_CKPT" EVAL_CONFIG="$CANDIDATE_CONFIG" \
  bash run_v59_fscip_bfar_dbap.sh
  stop=$(date +%s)
  python - "$BENCH_ROOT/benchmark_summary.json" "$p" "$((stop-start))" "$out/closed_loop/v59_fscip_bfar_dbap_4/closed_loop_combined_summary.json" <<'PY'
import json,sys
from pathlib import Path
report=Path(sys.argv[1]); d=json.loads(report.read_text())
summary=json.loads(Path(sys.argv[4]).read_text())
d['runs'].append({'processes_per_gpu':int(sys.argv[2]),'wall_time_s':float(sys.argv[3]),
                  'planner_reported_wall_time_s':float(summary.get('parallel_closed_loop_wall_time_s',float('nan'))),
                  'scenario_count':int(summary.get('scenario_count',0))})
report.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
done
python - "$BENCH_ROOT/benchmark_summary.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
valid=[r for r in d['runs'] if r.get('scenario_count')==4]
if valid: d['recommended_processes_per_gpu']=min(valid,key=lambda r:r['wall_time_s'])['processes_per_gpu']
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
print(json.dumps(d,indent=2,sort_keys=True))
PY
