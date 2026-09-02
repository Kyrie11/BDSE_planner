"""Fail-closed provenance audit for V64.3.50.5 metric-safe PIOR batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SAFE_MARKER = "[BDSE-PIOR-METRIC-SAFE]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed-loop-root", type=Path, required=True)
    ap.add_argument("--expected-scenarios", type=int, default=502)
    ap.add_argument("--output-report", type=Path, required=True)
    args = ap.parse_args()

    report = {
        "schema": "v64.3.50.5-pior-metric-safety-audit-v1",
        "expected_scenarios_per_arm": args.expected_scenarios,
        "arms": {},
    }
    passed = True
    for arm in ("control", "treatment"):
        batch_root = args.closed_loop_root / arm / "batches"
        certs = sorted(batch_root.glob("batch_*/.pior_batch_complete.json")) if batch_root.is_dir() else []
        n = 0
        missing_markers: list[str] = []
        bad_certs: list[str] = []
        metric_fail_logs: list[str] = []
        for cert_path in certs:
            try:
                cert = json.loads(cert_path.read_text(encoding="utf-8"))
            except Exception:
                bad_certs.append(str(cert_path))
                continue
            c = int(cert.get("scenario_count", 0) or 0)
            if not bool(cert.get("complete")) or int(cert.get("failed", -1)) != 0 or int(cert.get("successful", -1)) != c or int(cert.get("probe_fired_count", -1)) != c:
                bad_certs.append(str(cert_path))
                continue
            n += c
            log = cert_path.parent / "run.log"
            text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
            if SAFE_MARKER not in text:
                missing_markers.append(str(log))
            if "Metric Engine failed with:" in text or "drivable_area_compliance.py" in text and "AssertionError" in text:
                metric_fail_logs.append(str(log))
        arm_pass = n == args.expected_scenarios and not bad_certs and not missing_markers and not metric_fail_logs
        report["arms"][arm] = {
            "certificate_count": len(certs),
            "certified_scenarios": n,
            "bad_certificates": bad_certs,
            "missing_metric_safe_marker_logs": missing_markers,
            "metric_engine_failure_logs": metric_fail_logs,
            "pass": arm_pass,
        }
        passed = passed and arm_pass
    report["pass"] = passed
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("STOP V64.3.50.5: metric-safety provenance audit failed")


if __name__ == "__main__":
    main()
