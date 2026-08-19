from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bdse.planner.tournament import _ICER_TYPED_EVIDENCE_FEATURE_NAMES
from bdse.tools.check_v64_3_19_eaf_icer_screen import _icer_edge_diag, _metric_pack, _f, _auc
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _load_rows, _edge_groups, _path_diag, _guard_block_rate, _identity_rate


def _replacement_diag(path: str, safe: set[str], tau: float) -> dict[str, float]:
    edge_y: list[float] = []
    edge_score: list[float] = []
    gate_y: list[float] = []
    selected_y: list[float] = []
    typed_rows = typed_complete = 0
    groups = _edge_groups(path, safe)
    for rs in groups.values():
        if not rs:
            continue
        anchor = int(rs[0].get("anchor_action", -1))
        legacy = int(rs[0].get("raw_top_action", -1))
        selected = int(rs[0].get("icer_selected_action", legacy))
        inc = next((r for r in rs if int(r.get("challenger_action", -999)) == legacy), None)
        if inc is None or _f(inc, "icer_admissible", 0.0) < 0.5:
            continue
        inc_tm = _f(inc, "teacher_margin")
        if not math.isfinite(inc_tm):
            continue
        selected_delta = None
        for r in rs:
            ch = int(r.get("challenger_action", -1))
            if ch < 0 or ch in {anchor, legacy} or _f(r, "icer_admissible", 0.0) < 0.5:
                continue
            tm = _f(r, "teacher_margin")
            sup = _f(r, "icer_support_logit")
            sdom = _f(r, "icer_scalar_dominance_logit")
            risk = _f(r, "icer_replacement_regret_risk_logit")
            if not all(math.isfinite(v) for v in [tm, sup, sdom, risk]):
                continue
            # Correct V23's diagnostic population bug: deployment risk is only
            # meaningful after the frozen support+scalar-dominance eligibility.
            if not (sup > 0.0 and sdom > 0.0):
                continue
            d = tm - inc_tm
            edge_y.append(d); edge_score.append(risk)
            if risk > 0.0:
                gate_y.append(d)
            typed_rows += 1
            typed_complete += int(all(f"icer_typed_incumbent_{n}" in r and math.isfinite(_f(r, f"icer_typed_incumbent_{n}")) for n in _ICER_TYPED_EVIDENCE_FEATURE_NAMES))
            if ch == selected:
                selected_delta = d
        if selected not in {anchor, legacy} and selected_delta is not None:
            selected_y.append(float(selected_delta))
    yy = np.asarray(edge_y, dtype=float); ss = np.asarray(edge_score, dtype=float)
    gg = np.asarray(gate_y, dtype=float); sy = np.asarray(selected_y, dtype=float)
    material = sy < -float(tau)
    return {
        "deployment_eligible_edge_count": float(len(yy)),
        "risk_auc_on_deployment_eligible_edges": _auc([int(v > 0.0) for v in yy.tolist()], ss.tolist()) if len(yy) else float("nan"),
        "positive_gate_rate_on_deployment_eligible_edges": float(np.mean(ss > 0.0)) if len(ss) else float("nan"),
        "positive_gate_teacher_improvement_sum": float(gg.sum()) if len(gg) else 0.0,
        "positive_gate_precision": float(np.mean(gg > 0.0)) if len(gg) else float("nan"),
        "selected_direct_replacement_count": float(len(sy)),
        "selected_teacher_improvement_sum": float(sy.sum()) if len(sy) else 0.0,
        "selected_teacher_improvement_mean": float(sy.mean()) if len(sy) else float("nan"),
        "selected_worst_teacher_improvement": float(sy.min()) if len(sy) else float("nan"),
        "selected_material_negative_count": float(material.sum()) if len(sy) else 0.0,
        "selected_material_negative_rate": float(material.mean()) if len(sy) else 0.0,
        "selected_material_negative_excess_sum": float(np.maximum(-sy[material] - tau, 0.0).sum()) if np.any(material) else 0.0,
        "typed_feature_row_coverage": float(typed_complete / typed_rows) if typed_rows else float("nan"),
    }


def _incremental(new: str, control: str, M: dict[str, dict[str, float]], rep: dict[str, dict[str, float]], tau: float) -> bool:
    n, c = rep[new], rep[control]
    no_endpoint_regression = bool(
        M[new]["regret"] <= M[control]["regret"] * 1.02
        and M[new]["match"] >= M[control]["match"] - 0.005
        and M[new]["harmful"] <= M[control]["harmful"] + 0.005
    )
    mechanism_gain = bool(
        n["selected_material_negative_count"] < c["selected_material_negative_count"]
        or n["selected_material_negative_excess_sum"] < c["selected_material_negative_excess_sum"] - tau
        or n["selected_teacher_improvement_sum"] > c["selected_teacher_improvement_sum"] + tau
        or M[new]["regret"] <= M[control]["regret"] * 0.99
    )
    return bool(no_endpoint_regression and mechanism_gain)


def main() -> None:
    ap = argparse.ArgumentParser(description="One independent 500-scene V64.3.24 typed-tail causal block checker.")
    ap.add_argument("--split-name", required=True)
    ap.add_argument("--material-delta-threshold", type=float, default=0.004)
    for name in ["raw", "v20", "evidence-lcb", "evidence-tail", "typed-lcb", "typed-tail-dominance", "typed-tail-risk-first"]:
        ap.add_argument(f"--{name}-metrics", required=True)
        ap.add_argument(f"--{name}-rows", required=True)
        if name != "raw":
            ap.add_argument(f"--{name}-edges", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    tau = float(a.material_delta_threshold)
    if not math.isfinite(tau) or tau <= 0.0:
        raise SystemExit("invalid material delta threshold")

    tags = ["raw", "v20", "evidence_lcb", "evidence_tail", "typed_lcb", "typed_tail_dominance", "typed_tail_risk_first"]
    metrics = {t: json.load(open(getattr(a, t + "_metrics"))) for t in tags}
    rows = {t: _load_rows(getattr(a, t + "_rows")) for t in tags}
    tokens = set(rows["raw"])
    if len(tokens) < 480 or any(set(rows[t]) != tokens for t in tags[1:]):
        raise SystemExit("STOP DATA: paired fresh token identity mismatch")
    allflag = {t for t in tokens if _f(rows["raw"][t], "all_actions_safety_flagged_rate", 0.0) >= 0.5}
    safe = tokens - allflag
    edge_paths = {t: getattr(a, t + "_edges") for t in tags[1:]}
    ed = {t: _icer_edge_diag(Path(edge_paths[t]), safe) for t in tags[1:]}
    path = {t: _path_diag(rows["raw"], rows[t], edge_paths[t], safe, allflag) for t in tags[1:]}
    rep = {t: _replacement_diag(edge_paths[t], safe, tau) for t in ["evidence_lcb", "evidence_tail", "typed_lcb", "typed_tail_dominance", "typed_tail_risk_first"]}
    M = {t: _metric_pack(metrics[t]) for t in tags}
    main = "typed_tail_risk_first"
    c = ed[main]
    rmain = rep[main]
    anchor = {"match": _f(metrics["raw"], "selected_local_anchor_action_match"), "regret": _f(metrics["raw"], "selected_local_anchor_teacher_regret")}

    frozen_keys = [
        "selected_local_anchor_action_match", "pair_full_interface_action_match", "local_pair_full_interface_action_match",
        "evidence_certificate_fraction", "decision_budget_atom_count", "proposal_candidate_atom_count",
        "proposal_decisive_atom_recall", "selected_decisive_atom_recall", "effective_selected_decisive_atom_recall",
    ]
    frozen = {k: bool(math.isfinite(_f(metrics["raw"], k)) and math.isfinite(_f(metrics[main], k)) and abs(_f(metrics["raw"], k)-_f(metrics[main], k)) <= 1e-6) for k in frozen_keys}
    structural = {
        "all_flagged_scene_count": float(len(allflag)),
        "main_all_flagged_final_identity_vs_raw": _identity_rate(rows[main], rows["raw"], allflag),
        "main_all_flagged_delegation_rate": float(np.mean([_f(rows[main][t], "decisive_frontier_icer_structural_domain_delegated", 0.0) >= 0.5 for t in allflag])) if allflag else float("nan"),
        "main_safe_guard_block_rate": _guard_block_rate(rows[main], safe),
    }
    instrumentation = bool(
        c["scene_count"] >= 450 and c["admissible_edge_count"] >= 1800 and c["direct_counterfactual_dominance_edge_count"] >= 400
        and _f(metrics[main], "decisive_frontier_value_complete_star_coverage") >= 0.99 and all(frozen.values())
        and rmain["typed_feature_row_coverage"] >= 0.99
    )
    structural_ok = bool(
        len(allflag) >= 3 and structural["main_all_flagged_final_identity_vs_raw"] == 1.0
        and structural["main_all_flagged_delegation_rate"] == 1.0 and structural["main_safe_guard_block_rate"] <= 0.001
    )
    candidate_ok = bool(c["multi_admissible_proposal_rate"] >= 0.25 and c["admissible_candidates_per_proposal_mean"] >= 3.0)
    frozen_reliability_ok = bool(c["support_auc"] >= 0.65 and c["direct_counterfactual_dominance_auc"] >= 0.70)
    ret_path = path[main]["admissible_incumbent_to_anchor"]
    asymmetric_ok = bool(ret_path["count"] == 0 and abs(ret_path["regret_delta_sum"]) <= 1e-9)
    tail_safe = bool(
        rmain["selected_direct_replacement_count"] >= 8
        and rmain["selected_teacher_improvement_sum"] >= 0.0
        and rmain["selected_material_negative_count"] == 0.0
    )
    recovery_ok = bool(
        c["direct_incumbent_replacement_precision"] >= 0.60
        and c["direct_incumbent_opportunity_capture_rate"] >= 0.05
        and c["alternative_teacher_margin_mean"] > 0.0
    )

    # Causal decomposition.  A component stays in the eventual paper mainline
    # only if it adds under an otherwise matched operator/objective.
    # Keep both matched comparisons visible, but gate each mechanism in the
    # *main-chain context* in which it will be claimed.  Allowing an OR across
    # contexts would make a negative interaction look incremental and weaken
    # causal attribution (e.g. typed helps LCB but hurts once the tail objective
    # is present).
    representation_under_lcb_incremental = _incremental("typed_lcb", "evidence_lcb", M, rep, tau)
    representation_under_tail_incremental = _incremental("typed_tail_dominance", "evidence_tail", M, rep, tau)
    tail_without_typed_incremental = _incremental("evidence_tail", "evidence_lcb", M, rep, tau)
    tail_with_typed_incremental = _incremental("typed_tail_dominance", "typed_lcb", M, rep, tau)
    representation_incremental = bool(representation_under_tail_incremental)
    tail_objective_incremental = bool(tail_with_typed_incremental)
    rank_alignment_incremental = _incremental("typed_tail_risk_first", "typed_tail_dominance", M, rep, tau)

    beneficial_retention = M[main]["beneficial"] / max(M["raw"]["beneficial"], 1e-12) if M["raw"]["beneficial"] > 0 else float("nan")
    preservation = bool(
        M["raw"]["harmful"] - M[main]["harmful"] >= 0.05
        and beneficial_retention >= 0.35
        and M[main]["beneficial"] > M[main]["harmful"]
        and M[main]["flip"] >= 0.03 and M[main]["flip"] < M["raw"]["flip"]
    )
    endpoint = bool(
        M[main]["match"] >= anchor["match"] + 0.005
        and M[main]["regret"] <= M["raw"]["regret"] * 1.02
        and M[main]["regret"] <= M["v20"]["regret"] * 1.02
    )
    full = bool(
        instrumentation and structural_ok and candidate_ok and frozen_reliability_ok and asymmetric_ok
        and tail_safe and recovery_ok and representation_incremental and tail_objective_incremental
        and rank_alignment_incremental and preservation and endpoint
    )

    if full:
        next_action = "split_pass_freeze_typed_tail_risk_first_do_not_tune_wait_for_second_independent_block"
    elif not instrumentation:
        next_action = "engineering_or_typed_instrumentation_or_frozen_interface_failure"
    elif not structural_ok or not asymmetric_ok:
        next_action = "deployment_semantics_contract_failure_fix_engineering_do_not_change_reliability"
    elif not candidate_ok or not frozen_reliability_ok:
        next_action = "frozen_frontier_support_dominance_regression_do_not_tune_tail_model"
    elif not tail_safe:
        next_action = "typed_tail_still_leaks_material_negative_replacements_audit_missing_typed_observable_do_not_tune_K_tau_or_SE"
    elif not recovery_ok:
        next_action = "tail_safe_but_too_conservative_audit_support_coverage_and_typed_feature_degeneracy_no_threshold_sweep"
    elif not representation_incremental:
        next_action = "typed_representation_not_incremental_drop_it_from_mainline_and_keep_best_evidence_tail_control"
    elif not tail_objective_incremental:
        next_action = "material_tail_objective_not_incremental_drop_tail_penalty_keep_best_LCB_representation"
    elif not rank_alignment_incremental:
        next_action = "risk_first_extremal_ranking_not_incremental_revert_to_typed_tail_dominance_first"
    elif not preservation:
        next_action = "mechanism_tail_safe_but_intervention_preservation_failed_audit_nonmaterial_action_change_mass"
    else:
        next_action = "mechanism_chain_passes_but_endpoint_fails_do_not_add_mechanism_reproduce_and_audit_remaining_endpoint_error"

    report: dict[str, Any] = {
        "audit": "v64_3_24_eaf_icer_typed_tail_split", "split_name": a.split_name, "main_algorithm_arm": main,
        "full_split_pass": full, "instrumentation_valid": instrumentation, "deployment_semantics_valid": bool(structural_ok and asymmetric_ok),
        "candidate_support_valid": candidate_ok, "frozen_support_dominance_signal_valid": frozen_reliability_ok,
        "selected_path_material_tail_safe": tail_safe, "recovery_nontrivial": recovery_ok,
        "typed_representation_incremental": representation_incremental,
        "tail_objective_incremental": tail_objective_incremental,
        "risk_first_extremal_alignment_incremental": rank_alignment_incremental,
        "preservation_gain": preservation, "endpoint_gain": endpoint, "next_action": next_action,
        "material_delta_threshold": tau, "structural": structural, "replacement_diagnostics": rep,
        "edge_diagnostics": ed, "path_diagnostics": path,
        "metrics": {"anchor": anchor, **M, "main_beneficial_retention_vs_raw": beneficial_retention},
        "frozen_interface": frozen,
        "causal_comparisons": {
            "representation_under_lcb": {"contrast": "typed_lcb vs evidence_lcb", "incremental": bool(representation_under_lcb_incremental)},
            "representation_under_tail": {"contrast": "typed_tail_dominance vs evidence_tail", "incremental": bool(representation_under_tail_incremental), "main_chain_gate": True},
            "tail_without_typed": {"contrast": "evidence_tail vs evidence_lcb", "incremental": bool(tail_without_typed_incremental)},
            "tail_with_typed": {"contrast": "typed_tail_dominance vs typed_lcb", "incremental": bool(tail_with_typed_incremental), "main_chain_gate": True},
            "extremal_rank_alignment": {"contrast": "typed_tail_risk_first vs typed_tail_dominance", "incremental": bool(rank_alignment_incremental), "main_chain_gate": True},
        },
        "thresholds": {
            "support_auc_min": 0.65, "dominance_auc_min": 0.70, "typed_feature_coverage_min": 0.99,
            "selected_replacement_count_min": 8, "selected_material_negative_count_max": 0,
            "selected_teacher_improvement_sum_min": 0.0, "direct_replacement_precision_min": 0.60,
            "direct_capture_min": 0.05, "harmful_abs_reduction_min": 0.05,
            "beneficial_retention_min": 0.35, "match_over_anchor_min": 0.005, "regret_tolerance": 0.02,
        },
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
