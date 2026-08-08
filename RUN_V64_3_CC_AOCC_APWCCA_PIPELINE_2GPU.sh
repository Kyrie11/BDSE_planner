#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_1_cc_aocc_apwcca_fast_2gpu_v1}"
export QUERY_CACHE_AUDIT_REPORT="${QUERY_CACHE_AUDIT_REPORT:-$OUT_ROOT/provenance/query_prefix_cache_audit.json}"

# V64.3.1 deliberately does NOT consume the historical generic MAIN_CONFIG /
# SPEED_CONFIG shell variables.  Those names are shared by earlier launchers and
# caused the uploaded "V64.3" run to train V64.2 while evaluating V64.3.
# Version-scoped overrides remain possible but are validated before any long run.
V64_3_MAIN_CONFIG="${V64_3_MAIN_CONFIG:-bdse/configs/v64_3_cc_aocc_apwcca_train_2gpu.yaml}"
V64_3_SPEED_CONFIG="${V64_3_SPEED_CONFIG:-bdse/configs/v64_3_cc_aocc_apwcca_train_2gpu_verified_prefix_cache.yaml}"
V64_3_EVAL_CONFIG="${V64_3_EVAL_CONFIG:-bdse/configs/v64_3_cc_aocc_apwcca_cl.yaml}"
AUDIT_SCENARIOS="${AUDIT_SCENARIOS:-512}"

if [[ -n "${MAIN_CONFIG:-}" || -n "${SPEED_CONFIG:-}" ]]; then
  echo "[v64.3.1] ignoring inherited generic MAIN_CONFIG/SPEED_CONFIG; use V64_3_MAIN_CONFIG/V64_3_SPEED_CONFIG explicitly" >&2
fi
MAIN_CONFIG="$V64_3_MAIN_CONFIG"
SPEED_CONFIG="$V64_3_SPEED_CONFIG"
export EVAL_CONFIG="$V64_3_EVAL_CONFIG"

mkdir -p "$(dirname "$QUERY_CACHE_AUDIT_REPORT")" "$OUT_ROOT/logs" "$OUT_ROOT/provenance"

# Validate both possible train branches before the query audit.  A stale V64.2
# config now stops immediately instead of silently becoming the training path.
python -m bdse.tools.validate_v64_pipeline_config \
  --train-config "$MAIN_CONFIG" --eval-config "$EVAL_CONFIG" \
  --expected-family v64.3.1 \
  --output "$OUT_ROOT/provenance/v64_3_main_config_contract.json"
python -m bdse.tools.validate_v64_pipeline_config \
  --train-config "$SPEED_CONFIG" --eval-config "$EVAL_CONFIG" \
  --expected-family v64.3.1 \
  --output "$OUT_ROOT/provenance/v64_3_speed_config_contract.json"

echo "[v64.3.1] auditing the checkpoint-supported 12-D query prefix; nominal AP-WCCA disables the 6-D extension"
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
  query_path="verified_cached_12d_prefix_zero_extension"
  echo "[v64.3.1] prefix-cache audit PASS: use verified cached 12-D prefix"
else
  export TRAIN_CONFIG="$MAIN_CONFIG"
  query_path="full_runtime_recompute"
  echo "[v64.3.1] prefix-cache audit did not pass; fall back to canonical runtime recomputation"
fi

# Validate the exact selected branch once more after resolution.  This catches
# future launcher edits that accidentally overwrite TRAIN_CONFIG later.
python -m bdse.tools.validate_v64_pipeline_config \
  --train-config "$TRAIN_CONFIG" --eval-config "$EVAL_CONFIG" \
  --expected-family v64.3.1 \
  --output "$OUT_ROOT/provenance/v64_3_selected_config_contract.json"

python - "$QUERY_CACHE_AUDIT_REPORT" "$TRAIN_CONFIG" "$query_path" "$audit_status" \
  "$OUT_ROOT/provenance/v64_query_path_selection.json" <<'PY_QUERY_PATH'
import hashlib, json, sys
from pathlib import Path
report, config, path_name, status, output = sys.argv[1:]
config_path = Path(config).resolve()
h = hashlib.sha256(config_path.read_bytes()).hexdigest()
payload = {
    "audit_report": str(Path(report).resolve()),
    "audit_exit_status": int(status),
    "selected_train_config": str(config_path),
    "selected_train_config_sha256": h,
    "selected_query_path": path_name,
    "expected_algorithm_family": "V64.3.1-CC-AOCC-AP-WCCA-DA-EPC",
}
out = Path(output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY_QUERY_PATH

exec bash "$SCRIPT_DIR/V64_3_CC_AOCC_APWCCA_NEXT_COMMANDS.sh"
