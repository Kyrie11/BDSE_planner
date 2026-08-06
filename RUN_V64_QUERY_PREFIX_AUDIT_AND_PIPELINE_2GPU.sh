#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_saqa_bcc_fast_2gpu_v1}"
export QUERY_CACHE_AUDIT_REPORT="${QUERY_CACHE_AUDIT_REPORT:-$OUT_ROOT/provenance/query_prefix_cache_audit.json}"
MAIN_CONFIG="${MAIN_CONFIG:-bdse/configs/v64_2_saqa_bcc_hcbe_train_2gpu.yaml}"
SPEED_CONFIG="${SPEED_CONFIG:-bdse/configs/v64_saqa_bcc_train_2gpu_verified_prefix_cache.yaml}"
AUDIT_SCENARIOS="${AUDIT_SCENARIOS:-512}"

mkdir -p "$(dirname "$QUERY_CACHE_AUDIT_REPORT")" "$OUT_ROOT/logs"

echo "[v64] auditing only the checkpoint-supported 12-D query prefix; the 6-D extension is recomputed online"
set +e
python -m bdse.tools.audit_cached_query_features \
  --config "$SPEED_CONFIG" \
  --preprocessed-dir "$BDSE_TRAIN_CACHE" \
  --split train \
  --max-scenarios "$AUDIT_SCENARIOS" \
  --seed 64 \
  --tolerance 1.0e-5 \
  --output "$QUERY_CACHE_AUDIT_REPORT" \
  2>&1 | tee "$OUT_ROOT/logs/query_prefix_cache_audit.out"
audit_status=${PIPESTATUS[0]}
set -e

if [[ "$audit_status" -eq 0 ]]; then
  export TRAIN_CONFIG="$SPEED_CONFIG"
  query_path="cached_supported_prefix_plus_runtime_extension"
  echo "[v64] prefix-cache audit PASS: use cached 12-D prefix + runtime 6-D extension"
else
  export TRAIN_CONFIG="$MAIN_CONFIG"
  query_path="full_runtime_recompute"
  echo "[v64] prefix-cache audit did not pass; automatically fall back to full runtime recomputation"
  echo "[v64] this fallback is correctness-preserving and does not block training"
fi

python - "$QUERY_CACHE_AUDIT_REPORT" "$TRAIN_CONFIG" "$query_path" "$audit_status" \
  "$OUT_ROOT/provenance/v64_query_path_selection.json" <<'PY_QUERY_PATH'
import json, sys
from pathlib import Path
report, config, path_name, status, output = sys.argv[1:]
payload = {
    "audit_report": str(Path(report).resolve()),
    "audit_exit_status": int(status),
    "selected_train_config": str(Path(config).resolve()),
    "selected_query_path": path_name,
}
out = Path(output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY_QUERY_PATH

exec bash "$SCRIPT_DIR/V64_SAQA_BCC_NEXT_COMMANDS.sh"
