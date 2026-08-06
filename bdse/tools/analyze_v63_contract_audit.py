from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, float("nan")))
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze V63 same-checkpoint query/base contract ablations")
    p.add_argument("--suite-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    systems = ["nominal", "no_base_prior", "no_structural_prior", "no_runtime_priors"]
    rows: dict[str, dict[str, Any]] = {}
    for name in systems:
        path = args.suite_root / name / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        d = load(path)
        rows[name] = {
            "teacher_action_match": finite(d, "teacher_action_match"),
            "model_dense_full_action_match": finite(d, "model_dense_full_action_match"),
            "deployment_dense_full_action_match": finite(d, "deployment_dense_full_action_match"),
            "model_dense_vs_deployment_dense_full_match": finite(d, "model_dense_vs_deployment_dense_full_match"),
            "hab_topm_dense_value_action_match": finite(d, "hab_topm_dense_value_action_match"),
            "hab_topm_dense_value_vs_runtime_sparse_full_match": finite(d, "hab_topm_dense_value_vs_runtime_sparse_full_match"),
            "dense_runtime_query_contract_pass": finite(d, "dense_runtime_query_contract_pass"),
            "dense_runtime_query_value_max_abs": finite(d, "dense_runtime_query_value_max_abs"),
            "budget_vs_sparse_full_match": finite(d, "budget_vs_sparse_full_match"),
            "proposal_decisive_atom_recall": finite(d, "proposal_decisive_atom_recall"),
            "selected_decisive_atom_recall": finite(d, "selected_decisive_atom_recall"),
            "teacher_exact_winner_flip_critical_recall_topm": finite(d, "teacher_exact_winner_flip_critical_recall_topm"),
            "teacher_exact_winner_flip_critical_recall_selected": finite(d, "teacher_exact_winner_flip_critical_recall_selected"),
            "teacher_exact_winner_flip_critical_scene_rate": finite(d, "teacher_exact_winner_flip_critical_scene_rate"),
            "base_prior_replaced_best": finite(d, "base_prior_replaced_best"),
            "proposal_to_certificate_atom_expansion": finite(d, "proposal_to_certificate_atom_expansion"),
            "planner_latency_ms_p95": finite(d, "planner_latency_ms_p95"),
        }
    nominal = rows["nominal"]
    best = max(
        systems,
        key=lambda name: (
            -1.0 if not math.isfinite(rows[name]["teacher_action_match"]) else rows[name]["teacher_action_match"],
            -1.0 if not math.isfinite(rows[name]["hab_topm_dense_value_vs_runtime_sparse_full_match"]) else rows[name]["hab_topm_dense_value_vs_runtime_sparse_full_match"],
        ),
    )
    conclusions: list[str] = []
    if finite(nominal, "dense_runtime_query_contract_pass") < 0.99:
        conclusions.append("QUERY_CONTRACT_FAIL: do not interpret bridge/selector metrics; retrain or fix runtime query parity first.")
    else:
        conclusions.append("QUERY_CONTRACT_PASS: dense and sparse local heads are numerically comparable.")
    if finite(nominal, "model_dense_vs_deployment_dense_full_match") < 0.95:
        conclusions.append("DEPLOYMENT_BASE_DRIFT: runtime priors materially change the learned foundation winner.")
    if best != "nominal" and rows[best]["teacher_action_match"] > nominal["teacher_action_match"] + 0.005:
        conclusions.append(f"PRIOR_ABLATION_POSITIVE: {best} improves teacher match by {rows[best]['teacher_action_match']-nominal['teacher_action_match']:+.6f} on the same checkpoint.")
    if finite(nominal, "budget_vs_sparse_full_match") >= 0.95:
        conclusions.append("SELECTOR_PRESERVED: B-atom selector remains faithful to the measured sparse interface.")
    report = {
        "audit": "v63_same_checkpoint_query_and_deployment_base_contract",
        "systems": rows,
        "best_same_checkpoint_variant": best,
        "conclusions": conclusions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
