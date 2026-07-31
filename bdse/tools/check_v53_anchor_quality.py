from __future__ import annotations

import argparse
import json
import math
import hashlib
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


def _validate_rows(path: Path, minimum: int) -> tuple[int, str, list[str]]:
    failures: list[str] = []
    keys: set[tuple[str, int]] = set()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("scenario_token", "")), int(row.get("timestamp_us", 0) or 0))
            if not key[0]:
                failures.append("empty scenario token in anchor replay")
                continue
            if key in keys:
                failures.append(f"duplicate anchor replay key: {key}")
                continue
            keys.add(key)
            count += 1
    if count < minimum:
        failures.append(f"anchor replay has {count} unique rows; expected at least {minimum}")
    digest = hashlib.sha256()
    for token, timestamp in sorted(keys):
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(timestamp).encode("ascii"))
        digest.update(b"\n")
    return count, digest.hexdigest(), failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gate only the immutable base+dense-local anchor used by V53. "
            "Budgeted regret, direct pair heads, proposal selection and AOCC are "
            "excluded because they change when the residual/selector runtime is configured."
        )
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--min-replay-rows", type=int, default=950)
    parser.add_argument("--min-full-interface-match", type=float, default=0.32)
    parser.add_argument("--min-base-winner-sign", type=float, default=0.62)
    parser.add_argument("--min-dense-winner-sign", type=float, default=0.75)
    parser.add_argument("--min-dense-near-sign", type=float, default=0.65)
    parser.add_argument("--min-dense-all-sign", type=float, default=0.68)
    parser.add_argument("--warn-max-full-interface-regret", type=float, default=15000.0)
    parser.add_argument("--warn-max-latency-p95-ms", type=float, default=1500.0)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    data = _load(args.summary)
    requirements = {
        "full_interface_action_match": args.min_full_interface_match,
        "base_pair_sign_acc_winner_rival": args.min_base_winner_sign,
        "dense_pair_sign_acc_winner_rival": args.min_dense_winner_sign,
        "dense_pair_sign_acc_near_tie": args.min_dense_near_sign,
        "dense_pair_sign_acc_all": args.min_dense_all_sign,
    }
    values: dict[str, float] = {}
    failures: list[str] = []
    warnings: list[str] = []
    for key, threshold in requirements.items():
        value = _finite(data, key)
        values[key] = value
        if not math.isfinite(value):
            failures.append(f"{key} is missing or non-finite")
        elif value + 1e-12 < threshold:
            failures.append(f"{key}={value:.6f} < {threshold:.6f}")

    row_count, row_fingerprint, row_failures = _validate_rows(args.jsonl, args.min_replay_rows)
    failures.extend(row_failures)
    anchor_regret = _finite(data, "full_interface_teacher_regret")
    latency = _finite(data, "planner_latency_ms_p95")
    if math.isfinite(anchor_regret) and anchor_regret > args.warn_max_full_interface_regret:
        warnings.append(
            f"full_interface_teacher_regret={anchor_regret:.3f} exceeds diagnostic target "
            f"{args.warn_max_full_interface_regret:.3f}"
        )
    if math.isfinite(latency) and latency > args.warn_max_latency_p95_ms:
        warnings.append(
            f"planner_latency_ms_p95={latency:.3f} exceeds diagnostic target "
            f"{args.warn_max_latency_p95_ms:.3f}"
        )

    passed = not failures
    report = {
        "gate": "v53_immutable_anchor",
        "passed": passed,
        "row_count": row_count,
        "row_key_sha256": row_fingerprint,
        "values": values,
        "full_interface_teacher_regret": anchor_regret if math.isfinite(anchor_regret) else None,
        "latency_p95_ms": latency if math.isfinite(latency) else None,
        "failures": failures,
        "warnings": warnings,
        "excluded_metrics": [
            "teacher_regret (budgeted action)",
            "direct pair-head metrics",
            "proposal/selector metrics",
            "AOCC metrics",
        ],
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nV53 immutable anchor quality gate [{'PASS' if passed else 'FAIL'}]")
    for key, value in values.items():
        print(f"  {key}: {value}")
    print(f"  replay rows: {row_count}; key_sha256={row_fingerprint}")
    print("  full-interface regret (diagnostic only): " + (f"{anchor_regret:.6f}" if math.isfinite(anchor_regret) else "unavailable in legacy summary"))
    print("  excluded by design: budgeted teacher regret, direct pair head, selector/AOCC")
    for warning in warnings:
        print(f"  ! WARNING: {warning}")
    for failure in failures:
        print(f"  - {failure}")
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
