from __future__ import annotations

import argparse
import json
from pathlib import Path


THRESHOLDS = {
    "selected_hard_decisive_recall": (">=", 0.60),
    "selected_interaction_decisive_recall": (">=", 0.32),
    "fallback_would_trigger_rate": ("<=", 0.02),
    "teacher_action_match": (">=", 0.215),
    "effective_query_count": ("<=", 8500.0),
    "total_sparse_query_count": ("<=", 33000.0),
}


NONREGRESSION_TOLERANCES = {
    "teacher_action_match": 0.003,
    "budget_vs_full_match": 0.005,
    "pair_sign_acc_winner_rival": 0.005,
    "pair_sign_acc_interaction": 0.005,
    "pair_sign_acc_hard": 0.005,
    "selected_decisive_atom_recall": 0.010,
}


def passes(row: dict[str, float], baseline: dict[str, float] | None = None) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for key, (op, threshold) in THRESHOLDS.items():
        value = row.get(key)
        if value is None:
            failures.append(f"{key}=missing")
            continue
        ok = float(value) >= threshold if op == ">=" else float(value) <= threshold
        if not ok:
            failures.append(f"{key}={float(value):.6g} {op} {threshold:g}")
    if baseline:
        for key, tol in NONREGRESSION_TOLERANCES.items():
            if key not in row or key not in baseline:
                continue
            floor = float(baseline[key]) - float(tol)
            if float(row[key]) < floor:
                failures.append(f"{key}={float(row[key]):.6g} >= baseline-tol {floor:.6g}")
        if "teacher_regret" in row and "teacher_regret" in baseline and float(baseline["teacher_regret"]) > 0:
            ceiling = float(baseline["teacher_regret"]) * 1.03
            if float(row["teacher_regret"]) > ceiling:
                failures.append(f"teacher_regret={float(row['teacher_regret']):.6g} <= baseline*1.03 {ceiling:.6g}")
    return not failures, failures


def normalized_margin(row: dict[str, float]) -> float:
    terms = [
        float(row.get("selected_hard_decisive_recall", 0.0)) / 0.60,
        float(row.get("selected_interaction_decisive_recall", 0.0)) / 0.32,
        float(row.get("teacher_action_match", 0.0)) / 0.215,
        8500.0 / max(float(row.get("effective_query_count", 1e12)), 1e-9),
        33000.0 / max(float(row.get("total_sparse_query_count", 1e12)), 1e-9),
        0.02 / max(float(row.get("fallback_would_trigger_rate", 0.0)), 1e-6),
    ]
    return min(terms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--write-best", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args()

    baseline = None
    if args.baseline is not None and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text())

    rows: list[tuple[Path, dict[str, float], bool, list[str]]] = []
    for path in args.paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        ok, failures = passes(data, baseline=baseline)
        rows.append((path, data, ok, failures))

    if not rows:
        print("No result JSON files found.")
        return 2

    feasible = [x for x in rows if x[2]]
    if feasible:
        # Feasible-first, then teacher match, joint recall, lower queries/regret.
        best = max(
            feasible,
            key=lambda x: (
                float(x[1].get("teacher_action_match", 0.0)),
                min(float(x[1].get("selected_hard_decisive_recall", 0.0)), float(x[1].get("selected_interaction_decisive_recall", 0.0))),
                float(x[1].get("selected_hard_decisive_recall", 0.0)) + float(x[1].get("selected_interaction_decisive_recall", 0.0)),
                -float(x[1].get("total_sparse_query_count", 1e12)),
                -float(x[1].get("teacher_regret", 1e12)),
            ),
        )
    else:
        best = max(rows, key=lambda x: normalized_margin(x[1]))

    print("\nV35 runtime gate")
    if args.baseline is not None:
        print(f"Non-regression baseline: {args.baseline if baseline is not None else 'missing; skipped'}")
    for path, row, ok, failures in rows:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {path.name}")
        for key in THRESHOLDS:
            print(f"  {key}: {row.get(key)}")
        if failures:
            print("  failures: " + "; ".join(failures))
    print(f"\nRecommended: {best[0].name} ({'feasible' if best[2] else 'closest but not feasible'})")
    if args.write_best:
        args.write_best.parent.mkdir(parents=True, exist_ok=True)
        args.write_best.write_text(str(best[0]) + "\n")
    return 0 if feasible else 3


if __name__ == "__main__":
    raise SystemExit(main())
