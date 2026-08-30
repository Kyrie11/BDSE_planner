from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from bdse.tools.check_v38_runtime_gate import (
    PAIRED_NONINFERIORITY_MARGINS,
    THRESHOLDS,
    paired_noninferiority,
)


def _finite(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _scan_candidate(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "n": 0,
        "preserved": 0,
        "attempted": 0,
        "success": 0,
        "remaining": 0,
        "iterations_attempted_sum": 0.0,
        "iterations_attempted_min": math.inf,
        "evaluations_attempted_sum": 0.0,
        "unique_attempted_sum": 0.0,
        "stage_counts": Counter(),
        "rank1_remaining": 0,
        "remaining_tokens": [],
    }
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["n"] += 1
            preserved = _finite(row, "selector_deployment_coreset_target_action_preserved") >= 0.5
            attempted = _finite(row, "selector_deployment_coreset_budget_layer_attempted") >= 0.5
            success = _finite(row, "selector_deployment_coreset_budget_layer_success") >= 0.5
            stats["preserved"] += int(preserved)
            stats["attempted"] += int(attempted)
            stats["success"] += int(success)
            if attempted:
                it = _finite(row, "selector_deployment_coreset_budget_layer_iterations")
                stats["iterations_attempted_sum"] += it
                stats["iterations_attempted_min"] = min(stats["iterations_attempted_min"], it)
                stats["evaluations_attempted_sum"] += _finite(row, "selector_deployment_coreset_budget_layer_evaluations")
                stats["unique_attempted_sum"] += _finite(row, "selector_deployment_coreset_budget_layer_unique_states")
            if not preserved:
                stats["remaining"] += 1
                rank = int(_finite(row, "selector_deployment_coreset_budget_layer_best_target_rank"))
                stage = int(_finite(row, "selector_deployment_coreset_budget_layer_best_stage"))
                stats["stage_counts"][stage] += 1
                stats["rank1_remaining"] += int(rank == 1)
                stats["remaining_tokens"].append(str(row.get("scenario_token", "")))
    return stats


def _compare_previous(current: Path, previous: Path) -> dict[str, Any]:
    old: dict[tuple[str, int], tuple[bool, int, int, int]] = {}
    with previous.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("scenario_token", "")), int(row.get("timestamp_us", 0) or 0))
            old[key] = (
                _finite(row, "selector_deployment_coreset_target_action_preserved") >= 0.5,
                int(row.get("bdse_action", -1)), int(row.get("teacher_action", -2)), int(row.get("full_action", -2)),
            )
    out = {"recovered": 0, "lost": 0, "changed": 0, "teacher_net": 0, "full_net": 0}
    with current.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("scenario_token", "")), int(row.get("timestamp_us", 0) or 0))
            prev = old.get(key)
            if prev is None:
                continue
            old_ok, old_action, old_teacher, old_full = prev
            new_ok = _finite(row, "selector_deployment_coreset_target_action_preserved") >= 0.5
            out["recovered"] += int(not old_ok and new_ok)
            out["lost"] += int(old_ok and not new_ok)
            new_action = int(row.get("bdse_action", -1))
            if new_action != old_action:
                out["changed"] += 1
                out["teacher_net"] += int(new_action == int(row.get("teacher_action", -2))) - int(old_action == old_teacher)
                out["full_net"] += int(new_action == int(row.get("full_action", -2))) - int(old_action == old_full)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast strict runtime gate for v43 stage-aware budget-layer DACC")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--write-best", type=Path, default=None)
    parser.add_argument("--analysis-output", type=Path, default=None)
    parser.add_argument("--previous-jsonl", type=Path, default=None)
    args = parser.parse_args()
    if not args.candidate.exists() or not args.control.exists():
        print("Missing SAB-DACC/control result JSON")
        return 2

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    control = json.loads(args.control.read_text(encoding="utf-8"))
    cand_jsonl = args.candidate.with_suffix(".jsonl")
    control_jsonl = args.control.with_suffix(".jsonl")
    failures: list[str] = []

    for key, (op, threshold) in THRESHOLDS.items():
        value = float(candidate.get(key, float("nan")))
        ok = value >= threshold if op == ">=" else value <= threshold
        if not ok:
            failures.append(f"{key}={value} {op} {threshold}")

    required = {
        "decision_budget_excludes_structural_safety": 0.5,
        "structural_residual_enabled": 0.5,
        "pair_delta_calibration_enabled": 0.5,
        "selector_deployment_coreset_active": 0.99,
        "selector_deployment_coreset_lexicographic_active": 0.99,
        "selector_deployment_coreset_search_uses_rival_graph": 0.99,
        "selector_deployment_coreset_target_action_preserved": 0.95,
        "selector_deployment_coreset_budget_layer_width": 1.0,
        "selector_deployment_coreset_budget_layer_branch": 1.0,
        # This is the configured limit, not an all-scenario average of a
        # conditionally executed diagnostic.
        "selector_deployment_coreset_budget_layer_iteration_limit": 1.0,
    }
    for key, floor in required.items():
        value = float(candidate.get(key, 0.0))
        if value < floor:
            failures.append(f"{key}={value:.6g} >= {floor:.6g}")
    if float(candidate.get("selector_deployment_coreset_beam_width", 0.0)) > 1e-6:
        failures.append("deletion-lattice beam must remain disabled")
    if float(candidate.get("structural_safety_include_feasibility", 1.0)) > 0.5:
        failures.append("structural_safety_include_feasibility!=0")
    if float(candidate.get("selector_deployment_coreset_evaluator_missing", 0.0)) > 1e-6:
        failures.append("deployment evaluator missing in runtime")

    margins = dict(PAIRED_NONINFERIORITY_MARGINS)
    margins["teacher_action_match"] = 0.005
    margins["budget_vs_full_match"] = 0.002
    paired = paired_noninferiority(cand_jsonl, control_jsonl, margins) if cand_jsonl.exists() and control_jsonl.exists() else {}
    for key in ("teacher_action_match", "budget_vs_full_match", "pair_sign_acc_winner_rival"):
        stat = paired.get(key)
        if stat is None:
            tol = margins.get(key, 0.005)
            if key in candidate and key in control and float(candidate[key]) < float(control[key]) - tol:
                failures.append(f"{key}={candidate[key]} < control-tol={float(control[key]) - tol}")
        elif stat.get("pass", 0.0) < 0.5:
            failures.append(f"paired {key} LCB={stat['one_sided_95_lcb']:.6g} < -{stat['margin']:.6g}")

    scan = _scan_candidate(cand_jsonl) if cand_jsonl.exists() else None
    if scan is not None and scan["attempted"] > 0:
        # Validate that the conditional search actually ran.  v42 incorrectly
        # required the all-scenario mean iterations to exceed one, even though
        # 94.7% of scenes correctly had zero iterations because no repair was
        # needed.  The threshold is unchanged; the denominator is corrected.
        if scan["iterations_attempted_min"] < 1.0:
            failures.append(
                f"conditional budget-layer iterations min={scan['iterations_attempted_min']:.6g} >= 1"
            )

    print("\nV43 SAB-DACC runtime gate")
    print(f"[{'PASS' if not failures else 'FAIL'}] {args.candidate.name}")
    for key in (
        "teacher_action_match", "budget_vs_full_match", "teacher_regret",
        "pair_sign_acc_winner_rival", "pair_sign_acc_interaction",
        "selector_deployment_coreset_target_action_preserved",
        "selector_deployment_coreset_budget_layer_attempted",
        "selector_deployment_coreset_budget_layer_success",
        "selector_deployment_coreset_budget_layer_evaluations",
        "selector_deployment_coreset_budget_layer_iterations",
        "selector_deployment_coreset_budget_layer_iteration_limit",
        "selector_deployment_coreset_budget_layer_best_stage",
        "selector_deployment_coreset_budget_layer_best_stage_violation",
        "selector_deployment_coreset_evaluations",
        "effective_query_count", "total_sparse_query_count",
    ):
        print(f"  {key}: {candidate.get(key)}")
    if scan is not None:
        conditional_mean = scan["iterations_attempted_sum"] / max(scan["attempted"], 1)
        print(f"  conditional budget-layer iterations: mean={conditional_mean:.6g}, min={scan['iterations_attempted_min']:.6g}, n={scan['attempted']}")
    for key, stat in paired.items():
        print(f"  paired {key}: diff={stat['mean_diff']:.6g}, LCB={stat['one_sided_95_lcb']:.6g}, margin={stat['margin']:.6g}")
    if failures:
        print("  failures: " + "; ".join(failures))
    else:
        print("  stage-aware fixed-budget recovery is active and non-inferior to the frozen MARS control")

    if args.write_best:
        args.write_best.parent.mkdir(parents=True, exist_ok=True)
        args.write_best.write_text(str(args.candidate if not failures else args.control) + "\n", encoding="utf-8")

    if args.analysis_output is not None and scan is not None:
        lines = [
            f"scenarios: {scan['n']}",
            f"target_action_preserved: {scan['preserved']}/{scan['n']} = {scan['preserved']/max(scan['n'],1):.6f}",
            f"budget_layer_attempted: {scan['attempted']}",
            f"budget_layer_success: {scan['success']}",
            f"remaining_failures: {scan['remaining']}",
            f"conditional_iterations_mean: {scan['iterations_attempted_sum']/max(scan['attempted'],1):.3f}",
            f"conditional_iterations_min: {scan['iterations_attempted_min'] if scan['attempted'] else 0:.3f}",
            f"conditional_evaluations_mean: {scan['evaluations_attempted_sum']/max(scan['attempted'],1):.3f}",
            f"conditional_unique_states_mean: {scan['unique_attempted_sum']/max(scan['attempted'],1):.3f}",
            f"remaining_rank1_count: {scan['rank1_remaining']}",
            "remaining_stage_counts: " + ", ".join(f"{k}:{v}" for k, v in sorted(scan['stage_counts'].items())),
        ]
        if args.previous_jsonl is not None and args.previous_jsonl.exists():
            cmp = _compare_previous(cand_jsonl, args.previous_jsonl)
            lines.extend([
                f"previous_failures_recovered: {cmp['recovered']}",
                f"previous_preserved_lost: {cmp['lost']}",
                f"previous_actions_changed: {cmp['changed']}",
                f"teacher_match_net_on_changed: {cmp['teacher_net']}",
                f"full_match_net_on_changed: {cmp['full_net']}",
            ])
        lines.append("remaining_failure_tokens: " + (", ".join(scan["remaining_tokens"]) if scan["remaining_tokens"] else "none"))
        args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
        args.analysis_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
