from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


THRESHOLDS = {
    "structural_hard_decisive_coverage": (">=", 0.98),
    "effective_hard_decisive_recall": (">=", 0.98),
    "selected_soft_interaction_decisive_recall": (">=", 0.32),
    "effective_interaction_decisive_recall": (">=", 0.35),
    "fallback_would_trigger_rate": ("<=", 0.02),
    "avoidable_selected_action_safety_flag_rate": ("<=", 0.005),
    "teacher_action_match": (">=", 0.215),
    "budget_vs_full_match": (">=", 0.17),
    "effective_query_count": ("<=", 8500.0),
    "total_sparse_query_count": ("<=", 33000.0),
}

# Fallback tolerances are used only when paired baseline JSONL is unavailable.
# Summary fallback retains the strict v35/v37 tolerances.  Paired JSONL is
# preferred because it distinguishes real per-scenario regression from sampling
# noise without relaxing the absolute gates.
SUMMARY_TOLERANCES = {
    "teacher_action_match": 0.003,
    "budget_vs_full_match": 0.003,
    "pair_sign_acc_winner_rival": 0.005,
    "pair_sign_acc_interaction": 0.005,
    "pair_sign_acc_hard": 0.005,
}
PAIRED_NONINFERIORITY_MARGINS = {
    "teacher_action_match": 0.005,
    "budget_vs_full_match": 0.005,
    "pair_sign_acc_winner_rival": 0.005,
    "pair_sign_acc_interaction": 0.005,
    "pair_sign_acc_hard": 0.005,
}


def _value(row: dict[str, Any], key: str) -> float | None:
    val = row.get(key)
    if val is None:
        return None
    try:
        out = float(val)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _load_jsonl(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("scenario_token", "")), int(row.get("timestamp_us", 0) or 0))
        rows[key] = row
    return rows


def paired_noninferiority(
    candidate_jsonl: Path,
    baseline_jsonl: Path,
    metrics: dict[str, float],
) -> dict[str, dict[str, float]]:
    cand = _load_jsonl(candidate_jsonl)
    base = _load_jsonl(baseline_jsonl)
    common = sorted(set(cand) & set(base))
    out: dict[str, dict[str, float]] = {}
    if len(common) < 100:
        return out
    for metric, margin in metrics.items():
        diffs = []
        for key in common:
            a = _value(cand[key], metric)
            b = _value(base[key], metric)
            if a is None or b is None:
                continue
            diffs.append(a - b)
        if len(diffs) < 100:
            continue
        arr = np.asarray(diffs, dtype=np.float64)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
        lcb = mean - 1.645 * se
        out[metric] = {
            "n": float(arr.size),
            "mean_diff": mean,
            "one_sided_95_lcb": lcb,
            "margin": float(margin),
            "pass": float(lcb >= -float(margin)),
        }
    return out


def passes(
    row: dict[str, Any],
    *,
    path: Path,
    baseline: dict[str, Any] | None = None,
    paired: dict[str, dict[str, float]] | None = None,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for key, (op, threshold) in THRESHOLDS.items():
        value = _value(row, key)
        if value is None:
            failures.append(f"{key}=missing")
            continue
        ok = value >= threshold if op == ">=" else value <= threshold
        if not ok:
            failures.append(f"{key}={value:.6g} {op} {threshold:g}")

    if float(row.get("decision_budget_excludes_structural_safety", 0.0)) < 0.5:
        failures.append("decision_budget_excludes_structural_safety!=1")
    if float(row.get("structural_residual_enabled", 0.0)) < 0.5:
        failures.append("structural_residual_enabled!=1")
    if float(row.get("structural_safety_include_feasibility", 1.0)) > 0.5:
        failures.append("structural_safety_include_feasibility!=0")

    is_actionrank_control = "actionrank_control" in path.stem
    is_pair_only = "pair_only" in path.stem
    if not is_actionrank_control and float(row.get("selector_margin_coreset_active", 0.0)) < 0.5:
        failures.append("selector_margin_coreset_active!=1")
    if not is_actionrank_control:
        preserved = float(row.get("selector_margin_coreset_target_action_preserved", 0.0))
        sign_agreement = float(row.get("selector_margin_coreset_target_sign_agreement", 0.0))
        if preserved < 0.90:
            failures.append(f"selector_margin_coreset_target_action_preserved={preserved:.6g} >= 0.9")
        if sign_agreement < 0.90:
            failures.append(f"selector_margin_coreset_target_sign_agreement={sign_agreement:.6g} >= 0.9")
    if not is_pair_only and float(row.get("pair_delta_calibration_enabled", 0.0)) < 0.5:
        failures.append("pair_delta_calibration_enabled!=1")

    raw_flag = float(row.get("selected_action_safety_flag_rate", 0.0))
    all_flagged = float(row.get("all_actions_safety_flagged_rate", 0.0))
    if raw_flag > all_flagged + 0.001:
        failures.append(
            f"selected_action_safety_flag_rate={raw_flag:.6g} <= all_flagged+0.001 {all_flagged + 0.001:.6g}"
        )
    guard_rate = float(row.get("all_flagged_risk_guard_applied_rate", 0.0))
    if all_flagged > 0.0 and guard_rate + 0.001 < all_flagged:
        failures.append(
            f"all_flagged_risk_guard_applied_rate={guard_rate:.6g} >= all_flagged-0.001 {max(all_flagged - 0.001, 0.0):.6g}"
        )

    if baseline:
        paired = paired or {}
        for key, tol in SUMMARY_TOLERANCES.items():
            if key in paired:
                stat = paired[key]
                if stat.get("pass", 0.0) < 0.5:
                    failures.append(
                        f"paired {key} LCB={stat['one_sided_95_lcb']:.6g} >= -{stat['margin']:.6g}"
                    )
                continue
            if key not in row or key not in baseline:
                continue
            floor = float(baseline[key]) - float(tol)
            if float(row[key]) + 1e-12 < floor:
                failures.append(f"{key}={float(row[key]):.6g} >= baseline-tol {floor:.6g}")
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


def normalized_margin(row: dict[str, Any]) -> float:
    terms = []
    for key, (op, threshold) in THRESHOLDS.items():
        v = float(row.get(key, 0.0 if op == ">=" else 1e12))
        terms.append(v / max(threshold, 1e-9) if op == ">=" else threshold / max(v, 1e-6))
    return min(terms) if terms else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--write-best", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--baseline-jsonl", type=Path, default=None)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text()) if args.baseline is not None and args.baseline.exists() else None
    rows = []
    for path in args.paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        paired = {}
        cand_jsonl = path.with_suffix(".jsonl")
        if baseline is not None and args.baseline_jsonl is not None and args.baseline_jsonl.exists() and cand_jsonl.exists():
            paired = paired_noninferiority(cand_jsonl, args.baseline_jsonl, PAIRED_NONINFERIORITY_MARGINS)
        ok, failures = passes(data, path=path, baseline=baseline, paired=paired)
        rows.append((path, data, ok, failures, paired))
    if not rows:
        print("No result JSON files found.")
        return 2

    feasible = [x for x in rows if x[2]]
    if feasible:
        best = max(
            feasible,
            key=lambda x: (
                float(x[1].get("teacher_action_match", 0.0)),
                float(x[1].get("budget_vs_full_match", 0.0)),
                float(x[1].get("pair_sign_acc_interaction", 0.0)),
                float(x[1].get("selector_margin_coreset_target_action_preserved", 0.0)),
                -float(x[1].get("teacher_regret", 1e12)),
                -float(x[1].get("total_sparse_query_count", 1e12)),
            ),
        )
    else:
        best = max(rows, key=lambda x: (normalized_margin(x[1]), float(x[1].get("teacher_action_match", 0.0)), -float(x[1].get("teacher_regret", 1e12))))

    print("\nV38 MARS runtime gate")
    if args.baseline is not None:
        print(f"Non-regression baseline: {args.baseline if baseline is not None else 'missing; skipped'}")
    if args.baseline_jsonl is not None:
        print(f"Paired baseline JSONL: {args.baseline_jsonl if args.baseline_jsonl.exists() else 'missing; summary fallback'}")
    for path, row, ok, failures, paired in rows:
        print(f"[{'PASS' if ok else 'FAIL'}] {path.name}")
        for key in THRESHOLDS:
            print(f"  {key}: {row.get(key)}")
        print(f"  structural_residual_enabled: {row.get('structural_residual_enabled')}")
        print(f"  structural_safety_include_feasibility: {row.get('structural_safety_include_feasibility')}")
        print(f"  pair_delta_calibration_enabled: {row.get('pair_delta_calibration_enabled')}")
        print(f"  selector_margin_coreset_active: {row.get('selector_margin_coreset_active')}")
        print(f"  margin target action preserved: {row.get('selector_margin_coreset_target_action_preserved')}")
        print(f"  margin target sign agreement: {row.get('selector_margin_coreset_target_sign_agreement')}")
        print(f"  raw selected_action_safety_flag_rate: {row.get('selected_action_safety_flag_rate')}")
        print(f"  all_actions_safety_flagged_rate: {row.get('all_actions_safety_flagged_rate')}")
        if paired:
            for key, stat in paired.items():
                print(f"  paired {key}: diff={stat['mean_diff']:.6g}, LCB={stat['one_sided_95_lcb']:.6g}, margin={stat['margin']:.6g}")
        if failures:
            print("  failures: " + "; ".join(failures))
    print(f"\nRecommended: {best[0].name} ({'feasible' if best[2] else 'closest but not feasible'})")
    if args.write_best:
        args.write_best.parent.mkdir(parents=True, exist_ok=True)
        args.write_best.write_text(str(best[0]) + "\n")
    return 0 if feasible else 3


if __name__ == "__main__":
    raise SystemExit(main())
