from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(data: dict[str, Any], key: str) -> float:
    try:
        value = float(data[key])
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gate only the immutable V52 base+local anchor. Direct pair heads, "
            "proposal selection, and budgeted AOCC are intentionally excluded "
            "because they are reset/trained in the BFAR stage."
        )
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--min-full-interface-match", type=float, default=0.32)
    parser.add_argument("--min-base-winner-sign", type=float, default=0.62)
    parser.add_argument("--min-dense-winner-sign", type=float, default=0.75)
    parser.add_argument("--min-dense-near-sign", type=float, default=0.65)
    parser.add_argument("--min-dense-all-sign", type=float, default=0.68)
    parser.add_argument("--max-teacher-regret", type=float, default=12000.0)
    parser.add_argument("--max-latency-p95-ms", type=float, default=1500.0)
    args = parser.parse_args()

    data = _load(args.summary)
    requirements = {
        "full_interface_action_match": (args.min_full_interface_match, ">="),
        "base_pair_sign_acc_winner_rival": (args.min_base_winner_sign, ">="),
        "dense_pair_sign_acc_winner_rival": (args.min_dense_winner_sign, ">="),
        "dense_pair_sign_acc_near_tie": (args.min_dense_near_sign, ">="),
        "dense_pair_sign_acc_all": (args.min_dense_all_sign, ">="),
        "teacher_regret": (args.max_teacher_regret, "<="),
        "planner_latency_ms_p95": (args.max_latency_p95_ms, "<="),
    }
    values: dict[str, float] = {}
    failures: list[str] = []
    for key, (threshold, direction) in requirements.items():
        value = _finite(data, key)
        values[key] = value
        if not math.isfinite(value):
            failures.append(f"{key} is missing or non-finite")
        elif direction == ">=" and value + 1e-12 < threshold:
            failures.append(f"{key}={value:.6f} < {threshold:.6f}")
        elif direction == "<=" and value - 1e-12 > threshold:
            failures.append(f"{key}={value:.6f} > {threshold:.6f}")

    print(f"\nV52 immutable anchor quality gate [{'PASS' if not failures else 'FAIL'}]")
    for key, value in values.items():
        print(f"  {key}: {value}")
    print("  excluded by design: direct pair-head match, budgeted teacher match, evidence sufficiency, AOCC coverage")
    if failures:
        print("  BFAR training is blocked because the modules that will remain frozen are not yet decision-usable.")
        for failure in failures:
            print(f"  - {failure}")
    return 0 if not failures else 4


if __name__ == "__main__":
    raise SystemExit(main())
