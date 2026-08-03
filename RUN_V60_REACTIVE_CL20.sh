#!/usr/bin/env bash
set -euo pipefail

# Run publication-protocol reactive CL20 with an already frozen V60 checkpoint
# and calibrated configs.  This script never retrains and writes to a separate
# root so NR and R outputs cannot be mixed.
: "${SOURCE_OUT_ROOT:?Set SOURCE_OUT_ROOT to the completed/frozen V60 training output}"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT to the matched frozen foundation checkpoint}"
: "${NUPLAN_ROOT:?Set NUPLAN_ROOT}"
: "${BDSE_SPLIT_CACHE:?Set BDSE_SPLIT_CACHE to the frozen val_tune split cache}"

GPUS="${GPUS:-0,1}"
REACTIVE_OUT_ROOT="${REACTIVE_OUT_ROOT:-${SOURCE_OUT_ROOT}_reactive_cl20}"
CL_PROCESSES_PER_GPU="${CL_PROCESSES_PER_GPU:-2}"
CL_WORKERS_PER_GPU=1
CL_RENDER_SUMMARY_PDF="${CL_RENDER_SUMMARY_PDF:-0}"

CAND_CKPT="$SOURCE_OUT_ROOT/train/bdse_v60_dwapc_bfar_dbap.best.pt"
CAND_CFG="$SOURCE_OUT_ROOT/calibration/v60_dwapc_bfar_dbap_candidate_calibrated.yaml"
LOCAL_CFG="$SOURCE_OUT_ROOT/calibration/v60_dwapc_bfar_dbap_local_control_calibrated.yaml"
FOUND_CFG="$SOURCE_OUT_ROOT/calibration/v60_dwapc_bfar_dbap_foundation_control_calibrated.yaml"
for path in "$CAND_CKPT" "$CAND_CFG" "$LOCAL_CFG" "$FOUND_CFG" "$FOUNDATION_CKPT"; do
  [[ -f "$path" ]] || { echo "Missing frozen artifact: $path" >&2; exit 2; }
done

LOCAL_ROOT="$REACTIVE_OUT_ROOT/control_local_same_checkpoint"
FOUND_ROOT="$REACTIVE_OUT_ROOT/control_foundation_matched"
common=(
  CL_CHALLENGE=closed_loop_reactive_agents
  CL_METRIC_AGGREGATOR=closed_loop_reactive_agents_weighted_average
  CL_PROCESSES_PER_GPU="$CL_PROCESSES_PER_GPU"
  CL_WORKERS_PER_GPU=1
  CL_RENDER_SUMMARY_PDF="$CL_RENDER_SUMMARY_PDF"
  BDSE_SHARE_MODEL_PER_PROCESS=1
  BDSE_SERIALIZE_GPU_INFERENCE=0
  BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE"
  CL_TOKEN_SPLIT=val_tune
  GPUS="$GPUS"
  RUN_MODE=cl20
)

env "${common[@]}" OUT_ROOT="$REACTIVE_OUT_ROOT" V60_CKPT="$CAND_CKPT" EVAL_CONFIG="$CAND_CFG" \
  bash run_v60_dwapc_bfar_dbap.sh
env "${common[@]}" OUT_ROOT="$LOCAL_ROOT" V60_CKPT="$CAND_CKPT" EVAL_CONFIG="$LOCAL_CFG" \
  bash run_v60_dwapc_bfar_dbap.sh
env "${common[@]}" OUT_ROOT="$FOUND_ROOT" V60_CKPT="$FOUNDATION_CKPT" EVAL_CONFIG="$FOUND_CFG" \
  bash run_v60_dwapc_bfar_dbap.sh

python - \
  "$REACTIVE_OUT_ROOT/closed_loop/v60_dwapc_bfar_dbap_20/closed_loop_combined_summary.json" \
  "$LOCAL_ROOT/closed_loop/v60_dwapc_bfar_dbap_20/closed_loop_combined_summary.json" \
  "$FOUND_ROOT/closed_loop/v60_dwapc_bfar_dbap_20/closed_loop_combined_summary.json" \
  "$REACTIVE_OUT_ROOT/closed_loop/v60_three_way_reactive_cl20_protocol.json" <<'PY'
import json, sys
from pathlib import Path
paths = [Path(x) for x in sys.argv[1:4]]
rows = [json.loads(p.read_text()) for p in paths]
sha = [r.get("scenario_token_sha256") for r in rows]
counts = [int(r.get("scenario_count", -1)) for r in rows]
passed = len(set(sha)) == 1 and None not in sha and len(set(counts)) == 1 and counts[0] == 20
report = {
    "protocol": "closed_loop_reactive_agents",
    "protocol_pass": passed,
    "scenario_token_sha256": sha,
    "scenario_count": counts,
    "summaries": [str(p) for p in paths],
}
out = Path(sys.argv[4])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
if not passed:
    raise SystemExit(f"three-way reactive CL20 mismatch: {report}")
print(json.dumps(report, sort_keys=True))
PY
