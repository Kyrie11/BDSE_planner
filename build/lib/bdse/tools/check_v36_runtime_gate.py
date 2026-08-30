from __future__ import annotations

import argparse
import json
from pathlib import Path


THRESHOLDS = {
    # Hard evidence is no longer part of the learned B=16 decision support.  It
    # must instead be completely represented by the structural safety channel.
    "structural_hard_decisive_coverage": (">=", 0.98),
    "effective_hard_decisive_recall": (">=", 0.98),
    # The budgeted selector is judged on the evidence it actually owns.
    "selected_soft_interaction_decisive_recall": (">=", 0.32),
    "effective_interaction_decisive_recall": (">=", 0.35),
    "fallback_would_trigger_rate": ("<=", 0.02),
    "selected_action_safety_flag_rate": ("<=", 0.005),
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
}


def _value(row: dict[str, float], key: str) -> float | None:
    val = row.get(key)
    return None if val is None else float(val)


def passes(row: dict[str, float], baseline: dict[str, float] | None = None) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for key, (op, threshold) in THRESHOLDS.items():
        value = _value(row, key)
        if value is None:
            failures.append(f"{key}=missing")
            continue
        ok = value >= threshold if op == ">=" else value <= threshold
        if not ok:
            failures.append(f"{key}={value:.6g} {op} {threshold:g}")
    # The runtime architecture must really be the two-channel SCIDE design.
    if float(row.get("decision_budget_excludes_structural_safety", 0.0)) < 0.5:
        failures.append("decision_budget_excludes_structural_safety!=1")
    if baseline:
        for key, tol in NONREGRESSION_TOLERANCES.items():
            if key not in row or key not in baseline:
                continue
            floor = float(baseline[key]) - float(tol)
            if float(row[key]) < floor:
                failures.append(f"{key}={float(row[key]):.6g} >= baseline-tol {floor:.6g}")
        # Compare architecture-equivalent effective recall against the old raw
        # selected recall.  The new safety channel should not hide a coverage loss.
        if "effective_selected_decisive_atom_recall" in row and "selected_decisive_atom_recall" in baseline:
            floor = float(baseline["selected_decisive_atom_recall"]) - 0.01
            if float(row["effective_selected_decisive_atom_recall"]) < floor:
                failures.append(
                    f"effective_selected_decisive_atom_recall={float(row['effective_selected_decisive_atom_recall']):.6g} >= baseline-tol {floor:.6g}"
                )
        if "effective_interaction_decisive_recall" in row and "selected_interaction_decisive_recall" in baseline:
            floor = float(baseline["selected_interaction_decisive_recall"]) - 0.005
            if float(row["effective_interaction_decisive_recall"]) < floor:
                failures.append(
                    f"effective_interaction_decisive_recall={float(row['effective_interaction_decisive_recall']):.6g} >= baseline-tol {floor:.6g}"
                )
        if "teacher_regret" in row and "teacher_regret" in baseline and float(baseline["teacher_regret"]) > 0:
            ceiling = float(baseline["teacher_regret"]) * 1.03
            if float(row["teacher_regret"]) > ceiling:
                failures.append(f"teacher_regret={float(row['teacher_regret']):.6g} <= baseline*1.03 {ceiling:.6g}")
    return not failures, failures


def normalized_margin(row: dict[str, float]) -> float:
    terms = []
    for key, (op, threshold) in THRESHOLDS.items():
        v = float(row.get(key, 0.0 if op == ">=" else 1e12))
        if op == ">=":
            terms.append(v / max(threshold, 1e-9))
        else:
            terms.append(threshold / max(v, 1e-6))
    return min(terms) if terms else 0.0


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
        best = max(
            feasible,
            key=lambda x: (
                float(x[1].get("teacher_action_match", 0.0)),
                float(x[1].get("selected_soft_interaction_decisive_recall", 0.0)),
                float(x[1].get("effective_selected_decisive_atom_recall", 0.0)),
                -float(x[1].get("total_sparse_query_count", 1e12)),
                -float(x[1].get("teacher_regret", 1e12)),
            ),
        )
    else:
        best = max(rows, key=lambda x: normalized_margin(x[1]))

    print("\nV36 SCIDE runtime gate")
    if args.baseline is not None:
        print(f"Non-regression baseline: {args.baseline if baseline is not None else 'missing; skipped'}")
    for path, row, ok, failures in rows:
        print(f"[{'PASS' if ok else 'FAIL'}] {path.name}")
        for key in THRESHOLDS:
            print(f"  {key}: {row.get(key)}")
        print(f"  raw selected_hard_decisive_recall (diagnostic only): {row.get('selected_hard_decisive_recall')}")
        print(f"  raw selected_interaction_decisive_recall (diagnostic only): {row.get('selected_interaction_decisive_recall')}")
        if failures:
            print("  failures: " + "; ".join(failures))
    print(f"\nRecommended: {best[0].name} ({'feasible' if best[2] else 'closest but not feasible'})")
    if args.write_best:
        args.write_best.parent.mkdir(parents=True, exist_ok=True)
        args.write_best.write_text(str(best[0]) + "\n")
    return 0 if feasible else 3


if __name__ == "__main__":
    raise SystemExit(main())
