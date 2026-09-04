#!/usr/bin/env bash
set -euo pipefail

# V64.3.50.7: provenance-only repair for the V50.6 repaired PIOR fit.
# Scientific code and already-collected paired evidence are unchanged.
# This wrapper verifies byte identity of V50/V50.5/V50.6/V50.7 sources before
# invoking the exact V50.6 fit-only runner. It does NOT rerun 502x2 closed loop
# and does NOT consume untouched validation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BDSE_ROOT="${BDSE_ROOT:-$SCRIPT_DIR}"
cd "$BDSE_ROOT"
export PYTHONPATH="$BDSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$BDSE_ROOT/outputs}"

V49_ROOT="${V49_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1}"
V50_5_ROOT="${V50_5_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_50_5_eaf_icer_pior_train_2gpu_v1}"
OUT_ROOT="${OUT_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_50_7_eaf_icer_pior_provenance_repair_v1}"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"

# Fail closed on source identity before any scientific fit.
sha256sum -c V64_3_50_SOURCE_MANIFEST.sha256 \
  > "$OUT_ROOT/logs/v64_3_50_source_manifest_recheck.out"
sha256sum -c V64_3_50_5_ENGINEERING_MANIFEST.sha256 \
  > "$OUT_ROOT/logs/v64_3_50_5_engineering_manifest_recheck.out"
sha256sum -c V64_3_50_6_ENGINEERING_MANIFEST.sha256 \
  > "$OUT_ROOT/logs/v64_3_50_6_engineering_manifest_recheck.out"
sha256sum -c V64_3_50_7_ENGINEERING_MANIFEST.sha256 \
  > "$OUT_ROOT/logs/v64_3_50_7_engineering_manifest_recheck.out"

python - <<'PY' > "$OUT_ROOT/provenance/v64_3_50_7_source_identity.json"
import hashlib, json
from pathlib import Path

root = Path(".")
manifest = root / "V64_3_50_6_ENGINEERING_MANIFEST.sha256"
critical = {
    "bdse/tools/fit_v64_3_50_6_eaf_icer_pior.py",
    "bdse/tests/test_v64_3_50_6_pior_fit_repair.py",
    "RUN_V64_3_50_6_EAF_ICER_PIOR_FIT_REPAIR.sh",
}
expected = {}
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    sha, rel = line.split(None, 1)
    rel = rel.strip()
    if rel in critical:
        expected[rel] = sha
rows = []
ok = True
for rel in sorted(critical):
    p = root / rel
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    match = expected.get(rel) == actual
    ok &= match
    rows.append({"path": rel, "expected_sha256": expected.get(rel), "actual_sha256": actual, "match": match})
print(json.dumps({
    "schema": "v64.3.50.7-pior-source-identity-v1",
    "pass": bool(ok and len(expected) == len(critical)),
    "scientific_mechanism_unchanged": True,
    "result_defining_fit_is_exact_uploaded_v50_6_source": bool(ok and len(expected) == len(critical)),
    "critical_files": rows,
}, indent=2, sort_keys=True))
if not (ok and len(expected) == len(critical)):
    raise SystemExit("V50.7 ENGINEERING STOP: V50.6 result-defining source identity mismatch")
PY

cp V64_3_50_6_ENGINEERING_MANIFEST.sha256 \
  "$OUT_ROOT/provenance/V64_3_50_6_ENGINEERING_MANIFEST_VERIFIED.sha256"
cp V64_3_50_7_ENGINEERING_MANIFEST.sha256 \
  "$OUT_ROOT/provenance/V64_3_50_7_ENGINEERING_MANIFEST_VERIFIED.sha256"

# Invoke the exact frozen V50.6 fit-only path. Its scientific STOP remains a
# scientific STOP; this wrapper only closes provenance.
set +e
V49_ROOT="$V49_ROOT" V50_5_ROOT="$V50_5_ROOT" OUT_ROOT="$OUT_ROOT" \
  bash RUN_V64_3_50_6_EAF_ICER_PIOR_FIT_REPAIR.sh
STATUS=$?
set -e

python - "$OUT_ROOT" "$STATUS" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); status = int(sys.argv[2])
fit = out / "provenance/v64_3_50_6_pior_fit.json"
identity = out / "provenance/v64_3_50_7_source_identity.json"
ident = json.load(open(identity))
report = {
    "schema": "v64.3.50.7-pior-provenance-repair-v1",
    "provenance_pass": bool(ident.get("pass")),
    "v50_6_runner_exit_status": status,
    "fit_report_present": fit.is_file() and fit.stat().st_size > 0,
    "scientific_mechanism_unchanged": True,
    "paired_closed_loop_rerun": False,
    "untouched_validation_consumed": False,
}
if report["fit_report_present"]:
    d = json.load(open(fit))
    report["train_gate_pass"] = bool(d.get("train_gate_pass"))
    report["failure_diagnosis"] = d.get("nested_crossfit", {}).get("failure_diagnosis")
(out / "provenance/v64_3_50_7_provenance_repair_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

if [[ $STATUS -ne 0 ]]; then
  echo "V64.3.50.7 provenance PASS; exact V50.6 scientific STOP preserved. Do not consume untouched validation." >&2
  exit "$STATUS"
fi

echo "V64.3.50.7 provenance PASS and V50.6 TRAIN gate PASS. Freeze PIOR; next step is untouched paired closed-loop validation."
