from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _icer_edge_diag, _metric_pack, _f, _auc
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _load_rows, _edge_groups, _path_diag, _guard_block_rate, _identity_rate


def _replacement_risk_diag(path: str, safe: set[str]) -> dict[str, float]:
    y: list[float] = []
    score: list[float] = []
    positive_y: list[float] = []
    for rs in _edge_groups(path, safe).values():
        if not rs:
            continue
        anchor = int(rs[0].get("anchor_action", -1))
        legacy = int(rs[0].get("raw_top_action", -1))
        inc = next((r for r in rs if int(r.get("challenger_action", -999)) == legacy), None)
        if inc is None or _f(inc, "icer_admissible", 0.0) < 0.5:
            continue
        inc_tm = _f(inc, "teacher_margin")
        if not math.isfinite(inc_tm):
            continue
        for r in rs:
            ch = int(r.get("challenger_action", -1))
            if ch in {anchor, legacy} or _f(r, "icer_admissible", 0.0) < 0.5:
                continue
            d = _f(r, "teacher_margin") - inc_tm
            s = _f(r, "icer_replacement_regret_risk_logit")
            if math.isfinite(d) and math.isfinite(s):
                y.append(d); score.append(s)
                if s > 0.0:
                    positive_y.append(d)
    yy = np.asarray(y, dtype=float); ss = np.asarray(score, dtype=float); pp = np.asarray(positive_y, dtype=float)
    return {
        "edge_count": float(len(yy)),
        "local_lower_bound_auc": _auc([int(v > 0.0) for v in yy.tolist()], ss.tolist()) if len(yy) else float("nan"),
        "positive_gate_rate": float(np.mean(ss > 0.0)) if len(ss) else float("nan"),
        "positive_gate_teacher_improvement_sum": float(pp.sum()) if len(pp) else 0.0,
        "positive_gate_precision": float(np.mean(pp > 0.0)) if len(pp) else float("nan"),
    }


def _incremental(main_tag: str, control_tag: str, M: dict, ed: dict, path: dict) -> bool:
    m = ed[main_tag]; c = ed[control_tag]
    mr = path[main_tag]["direct_incumbent_to_alternative"]["regret_delta_sum"]
    cr = path[control_tag]["direct_incumbent_to_alternative"]["regret_delta_sum"]
    return bool(
        M[main_tag]["regret"] <= M[control_tag]["regret"] * 1.02
        and (
            mr < cr - 1e-6
            or m["direct_incumbent_replacement_precision"] >= c["direct_incumbent_replacement_precision"] + 0.01
            or M[main_tag]["regret"] <= M[control_tag]["regret"] * 0.99
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="One independent 500-scene V64.3.23 EAF-ICER-RCR block checker.")
    ap.add_argument("--split-name", required=True)
    for name in ["raw", "v20", "evidence-scalar", "evidence-rcr", "transition-rcr"]:
        ap.add_argument(f"--{name}-metrics", required=True)
        ap.add_argument(f"--{name}-rows", required=True)
        if name != "raw":
            ap.add_argument(f"--{name}-edges", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    tags = ["raw", "v20", "evidence_scalar", "evidence_rcr", "transition_rcr"]
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
    risk = {t: _replacement_risk_diag(edge_paths[t], safe) for t in ["evidence_scalar", "evidence_rcr", "transition_rcr"]}
    M = {t: _metric_pack(metrics[t]) for t in tags}
    anchor = {
        "match": _f(metrics["raw"], "selected_local_anchor_action_match"),
        "regret": _f(metrics["raw"], "selected_local_anchor_teacher_regret"),
    }

    # The paper/main mechanism is evidence-local RCR.  Planner-transition
    # conditioning is a controlled ablation and can never rescue a failed main.
    main_tag = "evidence_rcr"
    structural = {
        "all_flagged_scene_count": float(len(allflag)),
        "main_all_flagged_final_identity_vs_raw": _identity_rate(rows[main_tag], rows["raw"], allflag),
        "main_all_flagged_delegation_rate": float(np.mean([
            _f(rows[main_tag][t], "decisive_frontier_icer_structural_domain_delegated", 0.0) >= 0.5 for t in allflag
        ])) if allflag else float("nan"),
        "main_safe_guard_block_rate": _guard_block_rate(rows[main_tag], safe),
    }
    frozen_keys = [
        "selected_local_anchor_action_match", "pair_full_interface_action_match", "local_pair_full_interface_action_match",
        "evidence_certificate_fraction", "decision_budget_atom_count", "proposal_candidate_atom_count",
        "proposal_decisive_atom_recall", "selected_decisive_atom_recall", "effective_selected_decisive_atom_recall",
    ]
    frozen = {
        k: bool(math.isfinite(_f(metrics["raw"], k)) and math.isfinite(_f(metrics[main_tag], k)) and abs(_f(metrics["raw"], k) - _f(metrics[main_tag], k)) <= 1e-6)
        for k in frozen_keys
    }

    c = ed[main_tag]
    scalar = ed["evidence_scalar"]
    trans = ed["transition_rcr"]
    instrumentation = bool(
        c["scene_count"] >= 450
        and c["admissible_edge_count"] >= 1800
        and c["direct_counterfactual_dominance_edge_count"] >= 400
        and _f(metrics[main_tag], "decisive_frontier_value_complete_star_coverage") >= 0.99
        and all(frozen.values())
    )
    structural_ok = bool(
        len(allflag) >= 3
        and structural["main_all_flagged_final_identity_vs_raw"] == 1.0
        and structural["main_all_flagged_delegation_rate"] == 1.0
        and structural["main_safe_guard_block_rate"] <= 0.001
    )
    candidate_ok = bool(c["multi_admissible_proposal_rate"] >= 0.25 and c["admissible_candidates_per_proposal_mean"] >= 3.0)
    reliability_ok = bool(c["support_auc"] >= 0.65 and c["direct_counterfactual_dominance_auc"] >= 0.70)

    ret_path = path[main_tag]["admissible_incumbent_to_anchor"]
    rep_path = path[main_tag]["direct_incumbent_to_alternative"]
    asymmetric_operator_ok = bool(ret_path["count"] == 0 and abs(ret_path["regret_delta_sum"]) <= 1e-9)
    replacement_path_safe = bool(rep_path["count"] >= 8 and rep_path["regret_delta_sum"] <= 0.0)
    recovery_ok = bool(
        c["alternative_recovery_rate"] >= 0.03
        and c["alternative_recovery_precision"] >= 0.80
        and c["direct_incumbent_replacement_rate"] >= 0.02
        and c["direct_incumbent_replacement_precision"] >= 0.60
        and c["direct_incumbent_opportunity_capture_rate"] >= 0.08
        and c["alternative_teacher_margin_mean"] > 0.0
    )

    # Signed selected-evidence attribution is part of the main only if it adds
    # over an otherwise identical evidence-local scalar ranking arm.
    signed_incremental = _incremental("evidence_rcr", "evidence_scalar", M, ed, path)
    # Transition conditioning is deliberately not a promotion prerequisite.
    # It is absorbed only if it independently helps on both fresh blocks.
    transition_incremental = _incremental("transition_rcr", "evidence_rcr", M, ed, path)

    beneficial_retention = M[main_tag]["beneficial"] / max(M["raw"]["beneficial"], 1e-12) if M["raw"]["beneficial"] > 0 else float("nan")
    preservation = bool(
        M["raw"]["harmful"] - M[main_tag]["harmful"] >= 0.05
        and beneficial_retention >= 0.35
        and M[main_tag]["beneficial"] > M[main_tag]["harmful"]
        and M[main_tag]["flip"] >= 0.03
        and M[main_tag]["flip"] < M["raw"]["flip"]
    )
    endpoint = bool(
        M[main_tag]["match"] >= anchor["match"] + 0.005
        and M[main_tag]["regret"] <= M["raw"]["regret"] * 1.02
        and M[main_tag]["regret"] <= M["v20"]["regret"] * 1.02
    )
    full = bool(
        instrumentation and structural_ok and candidate_ok and reliability_ok and asymmetric_operator_ok
        and replacement_path_safe and recovery_ok and signed_incremental and preservation and endpoint
    )

    if full:
        next_action = "split_pass_freeze_evidence_local_RCR_do_not_tune_wait_for_second_independent_fresh_block"
    elif not instrumentation:
        next_action = "engineering_or_frozen_interface_failure"
    elif not structural_ok:
        next_action = "deployment_domain_semantics_failure_do_not_change_reliability"
    elif not candidate_ok or not reliability_ok:
        next_action = "frontier_or_frozen_support_dominance_regression_do_not_tune_local_risk"
    elif not asymmetric_operator_ok:
        next_action = "incumbent_default_preservation_contract_broken_engineering_failure"
    elif not replacement_path_safe:
        next_action = "evidence_local_regret_coherence_failed_on_selected_replacement_path_do_not_tune_K_or_zero_boundary_audit_local_tail_structure"
    elif not recovery_ok:
        next_action = "regret_safe_but_recovery_too_conservative_audit_evidence_neighborhood_support_without_threshold_sweep"
    elif not signed_incremental:
        next_action = "signed_profile_ranking_not_incremental_keep_evidence_local_scalar_RCR_as_candidate_do_not_tune_view_weights"
    elif not preservation:
        next_action = "replacement_path_safe_but_preservation_failed_keep_guards_frozen_audit_action_change_distribution"
    else:
        next_action = "mechanism_paths_pass_but_endpoint_fail_audit_remaining_endpoint_tail_before_any_representation_expansion"

    report = {
        "audit": "v64_3_23_eaf_icer_rcr_split",
        "split_name": a.split_name,
        "main_algorithm_arm": "evidence_local_rcr",
        "full_split_pass": full,
        "instrumentation_valid": instrumentation,
        "deployment_complete_domain_alignment": structural_ok,
        "candidate_support_valid": candidate_ok,
        "frozen_support_dominance_signal": reliability_ok,
        "incumbent_default_preservation_invariant": asymmetric_operator_ok,
        "selected_replacement_path_nonharmful": replacement_path_safe,
        "counterfactual_recovery_mechanism": recovery_ok,
        "signed_profile_ranking_incremental": signed_incremental,
        "transition_conditioning_incremental_diagnostic": transition_incremental,
        "transition_conditioning_required_for_promotion": False,
        "preservation_gain": preservation,
        "endpoint_gain": endpoint,
        "next_action": next_action,
        "structural": structural,
        "risk_diagnostics": risk,
        "edge_diagnostics": ed,
        "path_diagnostics": path,
        "metrics": {"anchor": anchor, **M, "main_beneficial_retention_vs_raw": beneficial_retention},
        "frozen_interface": frozen,
        "thresholds": {
            "support_auc_min": 0.65,
            "dominance_auc_min": 0.70,
            "incumbent_to_anchor_count_max": 0,
            "replacement_path_regret_delta_sum_max": 0.0,
            "direct_replacement_precision_min": 0.60,
            "direct_capture_min": 0.08,
            "harmful_abs_reduction_min": 0.05,
            "beneficial_retention_min": 0.35,
            "match_over_anchor_min": 0.005,
            "regret_vs_raw_tolerance": 0.02,
            "local_risk_edge_auc": "diagnostic_only_not_a_promotion_gate",
        },
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
