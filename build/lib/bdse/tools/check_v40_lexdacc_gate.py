from __future__ import annotations

import argparse
import json
from pathlib import Path

from bdse.tools.check_v38_runtime_gate import PAIRED_NONINFERIORITY_MARGINS, THRESHOLDS, paired_noninferiority


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict runtime gate for v40 Lex-DACC")
    parser.add_argument("lexdacc", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--write-best", type=Path, default=None)
    args = parser.parse_args()
    if not args.lexdacc.exists() or not args.control.exists():
        print("Missing DACC/control result JSON")
        return 2
    lexdacc = json.loads(args.lexdacc.read_text())
    control = json.loads(args.control.read_text())
    failures: list[str] = []
    for key, (op, threshold) in THRESHOLDS.items():
        value = float(lexdacc.get(key, float("nan")))
        ok = value >= threshold if op == ">=" else value <= threshold
        if not ok:
            failures.append(f"{key}={value} {op} {threshold}")
    required = {
        "decision_budget_excludes_structural_safety": 0.5,
        "structural_residual_enabled": 0.5,
        "pair_delta_calibration_enabled": 0.5,
        "selector_deployment_coreset_active": 0.99,
        "selector_deployment_coreset_lexicographic_active": 0.99,
        "selector_deployment_coreset_target_action_preserved": 0.95,
    }
    for key, floor in required.items():
        value = float(lexdacc.get(key, 0.0))
        if value < floor:
            failures.append(f"{key}={value:.6g} >= {floor:.6g}")
    if float(lexdacc.get("structural_safety_include_feasibility", 1.0)) > 0.5:
        failures.append("structural_safety_include_feasibility!=0")
    if float(lexdacc.get("selector_deployment_coreset_evaluator_missing", 0.0)) > 1e-6:
        failures.append("deployment evaluator missing in runtime")

    lexdacc_jsonl = args.lexdacc.with_suffix(".jsonl")
    control_jsonl = args.control.with_suffix(".jsonl")
    margins = dict(PAIRED_NONINFERIORITY_MARGINS)
    margins["teacher_action_match"] = 0.005
    margins["budget_vs_full_match"] = 0.002
    paired = paired_noninferiority(lexdacc_jsonl, control_jsonl, margins) if lexdacc_jsonl.exists() and control_jsonl.exists() else {}
    for key in ("teacher_action_match", "budget_vs_full_match", "pair_sign_acc_winner_rival"):
        stat = paired.get(key)
        if stat is None:
            # Strict summary fallback when JSONL is unavailable.
            tol = margins.get(key, 0.005)
            if key in lexdacc and key in control and float(lexdacc[key]) < float(control[key]) - tol:
                failures.append(f"{key}={lexdacc[key]} < control-tol={float(control[key])-tol}")
        elif stat.get("pass", 0.0) < 0.5:
            failures.append(f"paired {key} LCB={stat['one_sided_95_lcb']:.6g} < -{stat['margin']:.6g}")

    print("\nV40 Lex-DACC runtime gate")
    print(f"[{'PASS' if not failures else 'FAIL'}] {args.lexdacc.name}")
    for key in (
        "teacher_action_match", "budget_vs_full_match", "teacher_regret",
        "pair_sign_acc_winner_rival", "pair_sign_acc_interaction",
        "selector_deployment_coreset_target_action_preserved",
        "selector_deployment_coreset_lexicographic_active",
        "selector_deployment_coreset_forced_action_flip_steps",
        "selector_deployment_coreset_preservation_scan_evaluations",
        "selector_deployment_coreset_repair_success",
        "selector_deployment_coreset_postfill_changed",
        "selector_deployment_coreset_postfill_reverted",
        "selector_deployment_coreset_score_rmse",
        "selector_deployment_coreset_evaluations",
        "effective_query_count", "total_sparse_query_count",
    ):
        print(f"  {key}: {lexdacc.get(key)}")
    for key, stat in paired.items():
        print(f"  paired {key}: diff={stat['mean_diff']:.6g}, LCB={stat['one_sided_95_lcb']:.6g}, margin={stat['margin']:.6g}")
    if failures:
        print("  failures: " + "; ".join(failures))
    else:
        print("  lexicographic deployment objective is active and non-inferior to the v39 MARS control")
    if args.write_best:
        args.write_best.parent.mkdir(parents=True, exist_ok=True)
        args.write_best.write_text(str(args.lexdacc if not failures else args.control) + "\n")
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
