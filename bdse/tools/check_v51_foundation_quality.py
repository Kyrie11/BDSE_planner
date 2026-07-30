from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(d: dict[str, Any], key: str) -> float:
    try:
        value = float(d[key])
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stop FAR-DBAP before algorithm training when the rebuilt foundation is not decision-usable."
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--min-teacher-match", type=float, default=0.22)
    parser.add_argument("--min-full-interface-match", type=float, default=0.45)
    parser.add_argument("--min-pair-full-match", type=float, default=0.25)
    parser.add_argument("--min-winner-rival-sign", type=float, default=0.60)
    parser.add_argument("--min-near-tie-sign", type=float, default=0.45)
    parser.add_argument("--min-evidence-sufficiency", type=float, default=0.06)
    parser.add_argument("--max-latency-p95-ms", type=float, default=1500.0)
    args = parser.parse_args()

    data = _load(args.summary)
    requirements = {
        "teacher_action_match": (args.min_teacher_match, ">="),
        "full_interface_action_match": (args.min_full_interface_match, ">="),
        "pair_full_interface_action_match": (args.min_pair_full_match, ">="),
        "pair_sign_acc_winner_rival": (args.min_winner_rival_sign, ">="),
        "pair_sign_acc_near_tie": (args.min_near_tie_sign, ">="),
        "evidence_sufficiency": (args.min_evidence_sufficiency, ">="),
        "planner_latency_ms_p95": (args.max_latency_p95_ms, "<="),
    }
    failures: list[str] = []
    values: dict[str, float] = {}
    for key, (threshold, direction) in requirements.items():
        value = _finite(data, key)
        values[key] = value
        if not math.isfinite(value):
            failures.append(f"{key} is missing or non-finite")
        elif direction == ">=" and value + 1e-12 < threshold:
            failures.append(f"{key}={value:.6f} < {threshold:.6f}")
        elif direction == "<=" and value - 1e-12 > threshold:
            failures.append(f"{key}={value:.3f} > {threshold:.3f}")

    print(f"\nV51 foundation quality gate [{'PASS' if not failures else 'FAIL'}]")
    for key, value in values.items():
        print(f"  {key}: {value}")
    if failures:
        print("  The residual stage is intentionally blocked because its result would not be attributable to a strong anchor.")
        for failure in failures:
            print(f"  - {failure}")
    return 0 if not failures else 4


if __name__ == "__main__":
    raise SystemExit(main())
