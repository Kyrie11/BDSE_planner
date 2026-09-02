from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _arm(root: Path, arm: str, total: int) -> dict[str, Any]:
    batches_root = root / "closed_loop_train" / arm / "batches"
    certified = []
    incomplete = []
    if batches_root.is_dir():
        for b in sorted(p for p in batches_root.glob("batch_*") if p.is_dir()):
            cert_path = b / ".pior_batch_complete.json"
            if cert_path.is_file():
                cert = _load_json(cert_path)
                if not bool(cert.get("complete", False)):
                    raise RuntimeError(f"non-complete certificate present: {cert_path}")
                n = int(cert.get("scenario_count", 0))
                ok = (
                    int(cert.get("successful", -1)) == n
                    and int(cert.get("failed", -1)) == 0
                    and int(cert.get("probe_fired_count", -1)) == n
                )
                if not ok:
                    raise RuntimeError(f"invalid completion certificate: {cert_path}")
                certified.append({
                    "batch": b.name,
                    "scenario_count": n,
                    "wall_time_s": float(cert.get("wall_time_s", 0.0)),
                    "scenarios_per_wall_hour": float(cert.get("scenarios_per_wall_hour", 0.0)),
                    "probe_identity_audit": cert.get("probe_identity_audit", {}),
                })
            else:
                incomplete.append({
                    "batch": b.name,
                    "probe_events_observed_diagnostic_only": _jsonl_count(b / "pior_probe_events.jsonl"),
                    "has_run_log": (b / "run.log").is_file(),
                })
    certified_n = sum(x["scenario_count"] for x in certified)
    full_rates = [
        x["scenarios_per_wall_hour"] for x in certified
        if x["scenario_count"] >= 64 and x["scenarios_per_wall_hour"] > 0
    ]
    rate = statistics.median(full_rates) if full_rates else 0.0
    remaining = max(0, total - certified_n)
    return {
        "arm": arm,
        "certified_scenarios": certified_n,
        "required_scenarios": total,
        "certified_fraction": certified_n / total if total else 0.0,
        "remaining_if_resumed_from_certificates": remaining,
        "median_completed_full_batch_scenarios_per_hour": rate,
        "estimated_remaining_hours_from_certificates": remaining / rate if rate > 0 else None,
        "completed_batches": certified,
        "incomplete_batches_diagnostic_only": incomplete,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit V64.3.50 PIOR paired-collection progress without treating partial batches as scientific data.")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--expected-scenarios", type=int, default=502)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    root = args.output_root.resolve()
    total = int(args.expected_scenarios)
    arms = [_arm(root, "control", total), _arm(root, "treatment", total)]
    paired_certified = min(x["certified_scenarios"] for x in arms)
    estimates = [x["estimated_remaining_hours_from_certificates"] for x in arms if x["estimated_remaining_hours_from_certificates"] is not None]
    result = {
        "schema": "v64.3.50-pior-progress-audit-v1",
        "output_root": str(root),
        "expected_paired_scenarios": total,
        "paired_certified_scenarios": paired_certified,
        "paired_collection_complete": all(x["certified_scenarios"] == total for x in arms),
        "scientific_attribution_allowed": all(x["certified_scenarios"] == total for x in arms),
        "partial_batch_policy": "diagnostic_only; no partial probe/metric rows count toward the preregistered 502/502 data gate",
        "estimated_wall_hours_remaining_if_two_arms_run_in_parallel": max(estimates) if estimates else None,
        "arms": arms,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
