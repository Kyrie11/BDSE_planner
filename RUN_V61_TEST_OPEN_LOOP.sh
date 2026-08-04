#!/usr/bin/env bash
set -euo pipefail

# Evaluate an already-trained, already-calibrated V61 checkpoint on the test
# cache.  This script deliberately contains no calibration, checkpoint
# selection, gate-threshold fitting, or hyperparameter search.
#
# TEST_ROLE=development_test (default): permits an incomplete cache, but the
# result is diagnostic/development evidence only.  Once used to modify the
# algorithm, this split is no longer an untouched final test set.
#
# TEST_ROLE=final_test: requires completion evidence and performs a frozen
# one-shot evaluation.  Supply TEST_EXPECTED_SAMPLES or TEST_COMPLETION_MARKER.

: "${BDSE_TEST_CACHE:?Set BDSE_TEST_CACHE to the preprocessed test cache}"
: "${TEST_DIAGNOSTICS:?Set TEST_DIAGNOSTICS to diagnostics_test.json}"
: "${VAL_DIAGNOSTICS:?Set VAL_DIAGNOSTICS to diagnostics_val.json}"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT to the frozen foundation checkpoint}"

OUT_ROOT="${OUT_ROOT:-outputs_v61_dehab_bfar_dbap_2gpu}"
TEST_ROLE="${TEST_ROLE:-development_test}"
TEST_SPLIT="${TEST_SPLIT:-test}"
TEST_WORKERS_PER_GPU="${TEST_WORKERS_PER_GPU:-2}"
TEST_DEVICE="${TEST_DEVICE:-cuda}"
GPUS="${GPUS:-0,1}"

DIAGNOSTIC_TEST_SAMPLES="$(python - "$TEST_DIAGNOSTICS" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(d.get("num_samples", d.get("num_loaded", 0)) or 0))
PY
)"
if [[ "$TEST_ROLE" == "final_test" ]]; then
  # A final result must cover the complete diagnostics population rather than a
  # convenient 1k subset.  Development runs keep the fast 1k default.
  TEST_MAX_SCENARIOS="${TEST_MAX_SCENARIOS:-$DIAGNOSTIC_TEST_SAMPLES}"
else
  TEST_MAX_SCENARIOS="${TEST_MAX_SCENARIOS:-1000}"
fi

CANDIDATE_CKPT="${CANDIDATE_CKPT:-$OUT_ROOT/train/bdse_v61_dehab_bfar_dbap.best.pt}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-$OUT_ROOT/calibration/v61_dehab_bfar_dbap_candidate_calibrated.yaml}"
LOCAL_CONFIG="${LOCAL_CONFIG:-$OUT_ROOT/calibration/v61_dehab_bfar_dbap_local_control_calibrated.yaml}"
FOUNDATION_CONFIG="${FOUNDATION_CONFIG:-$OUT_ROOT/calibration/v61_dehab_bfar_dbap_foundation_control_calibrated.yaml}"
TEST_OUT_ROOT="${TEST_OUT_ROOT:-$OUT_ROOT/test_open_loop/$TEST_ROLE}"
READINESS_JSON="$TEST_OUT_ROOT/test_readiness.json"
PROVENANCE_JSON="$TEST_OUT_ROOT/test_evaluation_provenance.json"

case "$TEST_ROLE" in
  development_test|final_test) ;;
  *) echo "TEST_ROLE must be development_test or final_test" >&2; exit 2 ;;
esac
case "$TEST_DEVICE" in
  cuda|cpu) ;;
  *) echo "TEST_DEVICE must be cuda or cpu" >&2; exit 2 ;;
esac
for path in "$TEST_DIAGNOSTICS" "$VAL_DIAGNOSTICS" "$FOUNDATION_CKPT" \
            "$CANDIDATE_CKPT" "$CANDIDATE_CONFIG" "$LOCAL_CONFIG" "$FOUNDATION_CONFIG"; do
  [[ -s "$path" ]] || { echo "Missing or empty required file: $path" >&2; exit 2; }
done
mkdir -p "$TEST_OUT_ROOT"

readiness_args=(
  --test-diagnostics "$TEST_DIAGNOSTICS"
  --val-diagnostics "$VAL_DIAGNOSTICS"
  --test-cache "$BDSE_TEST_CACHE"
  --output "$READINESS_JSON"
)
[[ -n "${BDSE_TRAIN_CACHE:-}" ]] && readiness_args+=(--train-cache "$BDSE_TRAIN_CACHE")
[[ -n "${BDSE_VAL_CACHE:-}" ]] && readiness_args+=(--val-cache "$BDSE_VAL_CACHE")
[[ -n "${TEST_EXPECTED_SAMPLES:-}" ]] && readiness_args+=(--expected-samples "$TEST_EXPECTED_SAMPLES")
[[ -n "${TEST_COMPLETION_MARKER:-}" ]] && readiness_args+=(--completion-marker "$TEST_COMPLETION_MARKER")
[[ -n "${TEST_MIN_PRELIMINARY_SAMPLES:-}" ]] && readiness_args+=(--min-preliminary-samples "$TEST_MIN_PRELIMINARY_SAMPLES")
[[ -n "${TEST_MAX_FAILED_FRACTION:-}" ]] && readiness_args+=(--max-failed-fraction "$TEST_MAX_FAILED_FRACTION")

if [[ "$TEST_ROLE" == "final_test" ]]; then
  if [[ -z "${TEST_EXPECTED_SAMPLES:-}" && -z "${TEST_COMPLETION_MARKER:-}" ]]; then
    echo "final_test requires TEST_EXPECTED_SAMPLES or TEST_COMPLETION_MARKER" >&2
    exit 2
  fi
  : "${BDSE_TRAIN_CACHE:?final_test requires BDSE_TRAIN_CACHE for leakage audit}"
  : "${BDSE_VAL_CACHE:?final_test requires BDSE_VAL_CACHE for leakage audit}"
  readiness_args+=(--require-complete)
else
  readiness_args+=(--allow-incomplete)
  cat >&2 <<'WARNING'
[v61-test] DEVELOPMENT TEST ONLY: these results may be used for diagnosis, but
using them to change the algorithm consumes this split as development data.
Do not report them as untouched final-test performance.
WARNING
fi

python -m bdse.tools.check_test_set_readiness "${readiness_args[@]}"

rm -rf "$TEST_OUT_ROOT/suite"
python -m bdse.tools.run_parallel_open_loop_suite \
  --system "candidate::$CANDIDATE_CONFIG::$CANDIDATE_CKPT" \
  --system "local::$LOCAL_CONFIG::$CANDIDATE_CKPT" \
  --system "foundation::$FOUNDATION_CONFIG::$FOUNDATION_CKPT" \
  --preprocessed-dir "$BDSE_TEST_CACHE" \
  --split "$TEST_SPLIT" \
  --max-scenarios "$TEST_MAX_SCENARIOS" \
  --output-root "$TEST_OUT_ROOT/suite" \
  --gpus "$GPUS" \
  --workers-per-gpu "$TEST_WORKERS_PER_GPU" \
  --device "$TEST_DEVICE"

python - "$PROVENANCE_JSON" "$READINESS_JSON" "$TEST_OUT_ROOT/suite/parallel_open_loop_suite_report.json" \
  "$TEST_ROLE" "$TEST_SPLIT" "$TEST_MAX_SCENARIOS" "$BDSE_TEST_CACHE" \
  "$CANDIDATE_CKPT" "$FOUNDATION_CKPT" "$CANDIDATE_CONFIG" "$LOCAL_CONFIG" "$FOUNDATION_CONFIG" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output_path,
    readiness_path,
    suite_path,
    role,
    split,
    max_scenarios,
    cache,
    candidate_ckpt,
    foundation_ckpt,
    candidate_cfg,
    local_cfg,
    foundation_cfg,
) = sys.argv[1:]


def fingerprint(path_text: str) -> dict[str, object]:
    path = Path(path_text).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}

readiness = json.loads(Path(readiness_path).read_text(encoding="utf-8"))
suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
complete = bool(readiness.get("completion_verified", False))
evaluated_count = int(suite.get("num_scenarios_per_system", -1))
diagnostic_count = int(readiness.get("num_samples", -1))
full_test_evaluated = evaluated_count == diagnostic_count and diagnostic_count > 0
report = {
    "schema_version": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "evaluation_role": role,
    "test_split": split,
    "test_cache": str(Path(cache).expanduser().resolve()),
    "requested_max_scenarios": int(max_scenarios),
    "evaluated_scenarios_per_system": evaluated_count,
    "diagnostic_test_sample_count": diagnostic_count,
    "full_test_evaluated": full_test_evaluated,
    "readiness_status": readiness.get("status"),
    "completion_verified": complete,
    "paired_protocol_pass": bool(suite.get("paired_protocol_pass", False)),
    "checkpoint_selected_on_test": False,
    "calibration_fit_on_test": False,
    "thresholds_tuned_on_test": False,
    "ready_for_final_claim": bool(
        role == "final_test"
        and complete
        and readiness.get("status") == "INTEGRITY_PASS_COMPLETE"
        and suite.get("paired_protocol_pass", False)
        and full_test_evaluated
    ),
    "warning": (
        "Development-only test evidence. Any algorithm change informed by these results makes this split adaptive development data."
        if role == "development_test"
        else "Frozen one-shot final-test evaluation; do not use these results to select or modify the submitted method."
    ),
    "artifacts": {
        "candidate_checkpoint": fingerprint(candidate_ckpt),
        "foundation_checkpoint": fingerprint(foundation_ckpt),
        "candidate_config": fingerprint(candidate_cfg),
        "local_control_config": fingerprint(local_cfg),
        "foundation_control_config": fingerprint(foundation_cfg),
        "readiness_report": str(Path(readiness_path).resolve()),
        "suite_report": str(Path(suite_path).resolve()),
    },
}
Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
if role == "final_test" and not report["ready_for_final_claim"]:
    raise SystemExit("final-test provenance checks failed; see " + str(Path(output_path).resolve()))
PY

echo "[v61-test] suite: $TEST_OUT_ROOT/suite/parallel_open_loop_suite_report.json"
echo "[v61-test] provenance: $PROVENANCE_JSON"
