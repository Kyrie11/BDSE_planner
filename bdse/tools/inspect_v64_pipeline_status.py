from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STAGES = (
    ("training", "train/bdse_v64_saqa_bcc.train_log.jsonl"),
    ("final_checkpoint", "train/bdse_v64_saqa_bcc.pt"),
    ("best_checkpoint", "train/bdse_v64_saqa_bcc.best.pt"),
    ("calibration", "calibration/v64_dual_certificate_calibration.json"),
    ("candidate_open_loop", "open_loop/candidate/metrics.json"),
    ("local_control_open_loop", "open_loop/local_control/metrics.json"),
    ("foundation_control_open_loop", "open_loop/foundation_control/metrics.json"),
    ("gate_report", "open_loop/v64_saqa_bcc_gate_report.json"),
)


def _tail(path: Path, max_chars: int = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    root = args.out_root
    artifacts = {name: {"path": rel, "exists": (root / rel).is_file() and (root / rel).stat().st_size > 0} for name, rel in STAGES}
    logs = sorted((root / "logs").glob("pipeline_*.log")) if (root / "logs").exists() else []
    errors: list[dict[str, Any]] = []
    patterns = [
        ("bash_unbound_variable", re.compile(r"line\s+(\d+):\s+([^:\n]+): unbound variable")),
        ("cache_audit_rejected", re.compile(r"audit report is not PASS|cached query features.*missing", re.I)),
        ("python_traceback", re.compile(r"Traceback \(most recent call last\):")),
    ]
    for log in logs:
        tail = _tail(log)
        for kind, pattern in patterns:
            match = pattern.search(tail)
            if match:
                errors.append({"kind": kind, "log": str(log), "match": match.group(0)})
    first_missing = next((name for name, _ in STAGES if not artifacts[name]["exists"]), None)
    train_last_epoch = None
    train_has_validation = False
    train_log = root / "train/bdse_v64_saqa_bcc.train_log.jsonl"
    if train_log.is_file():
        for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("epoch"), int):
                train_last_epoch = row["epoch"]
            train_has_validation = train_has_validation or any(str(k).startswith("val_") for k in row)
    inferred_stop_stage = None
    if any(item["kind"] == "bash_unbound_variable" for item in errors) and not artifacts["calibration"]["exists"]:
        inferred_stop_stage = "calibration_launch"
    elif any(item["kind"] == "cache_audit_rejected" for item in errors):
        inferred_stop_stage = "preflight_cache_audit"
    report = {
        "out_root": str(root),
        "artifacts": artifacts,
        "first_missing_artifact": first_missing,
        "inferred_pipeline_stop_stage": inferred_stop_stage,
        "training_log_last_epoch": train_last_epoch,
        "training_log_has_validation": train_has_validation,
        "pipeline_log_count": len(logs),
        "detected_errors": errors,
        "official_gate_evaluated": artifacts["gate_report"]["exists"],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
