#!/usr/bin/env bash
set -euo pipefail

# A partial test cache is a one-shot frozen-checkpoint stress test only. It must
# never be used for checkpoint selection, threshold calibration, or gate tuning.
: "${BDSE_TEST_CACHE:?Set BDSE_TEST_CACHE to the partial/complete test cache}"
: "${TEST_DIAGNOSTICS:?Set TEST_DIAGNOSTICS to diagnostics_test.json}"
: "${VAL_DIAGNOSTICS:?Set VAL_DIAGNOSTICS to diagnostics_val.json}"
: "${V56_CHECKPOINT:?Set V56_CHECKPOINT to the already frozen V56 checkpoint}"
: "${V56_TEST_FROZEN_ACK:?Set V56_TEST_FROZEN_ACK=YES after freezing checkpoint/config/gates}"
[[ "$V56_TEST_FROZEN_ACK" == "YES" ]] || { echo "Refusing test evaluation: set V56_TEST_FROZEN_ACK=YES only after all tuning is frozen." >&2; exit 2; }

GPUS="${GPUS:-0,1}"
TEST_CONFIG="${TEST_CONFIG:-bdse/configs/v56_dcip_bfar_dbap_cl.yaml}"
TEST_OUT_ROOT="${TEST_OUT_ROOT:-outputs_v56_dcip_bfar_partial_test_frozen}"
TEST_MAX_SCENARIOS="${TEST_MAX_SCENARIOS:-10000}"
READINESS="$TEST_OUT_ROOT/provenance/partial_test_readiness.json"
mkdir -p "$TEST_OUT_ROOT/provenance" "$TEST_OUT_ROOT/logs"

args=(python -m bdse.tools.check_test_set_readiness
  --test-diagnostics "$TEST_DIAGNOSTICS"
  --val-diagnostics "$VAL_DIAGNOSTICS"
  --test-cache "$BDSE_TEST_CACHE"
  --allow-incomplete
  --output "$READINESS")
[[ -n "${BDSE_TRAIN_CACHE:-}" ]] && args+=(--train-cache "$BDSE_TRAIN_CACHE")
[[ -n "${BDSE_VAL_CACHE_ORIGINAL:-}" ]] && args+=(--val-cache "$BDSE_VAL_CACHE_ORIGINAL")
"${args[@]}"

python - "$READINESS" "$V56_CHECKPOINT" "$TEST_CONFIG" "$TEST_OUT_ROOT/provenance/frozen_test_protocol.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
readiness, ckpt, cfg, out = map(Path, sys.argv[1:])
d = json.loads(readiness.read_text())
if d.get("status") not in {"PRELIMINARY_PASS", "INTEGRITY_PASS_COMPLETION_UNVERIFIED"}:
    raise SystemExit(f"test readiness failed: {d.get('status')} {d.get('hard_failures')}")
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()
report={
  "protocol":"frozen_preliminary_test_no_tuning",
  "checkpoint":str(ckpt.resolve()), "checkpoint_sha256":sha(ckpt),
  "config":str(cfg.resolve()), "config_sha256":sha(cfg),
  "readiness":d,
  "warning":"Partial test results are exploratory stress-test evidence, not a final paper test result.",
}
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
PY

# Use a separate output root so test metrics can never overwrite validation
# artifacts or be consumed by the training/calibration pipeline.
BDSE_VAL_CACHE="$BDSE_TEST_CACHE" \
OUT_ROOT="$TEST_OUT_ROOT" \
GPUS="$GPUS" \
RUN_MODE=open_loop \
OPEN_LOOP_SPLIT=test \
OPEN_LOOP_MAX_SCENARIOS="$TEST_MAX_SCENARIOS" \
V56_CKPT="$V56_CHECKPOINT" \
EVAL_CONFIG="$TEST_CONFIG" \
bash run_v56_dcip_bfar_dbap.sh

echo "Frozen preliminary test complete: $TEST_OUT_ROOT"
