from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from bdse.tools.check_v64_3_19_eaf_icer_screen import _icer_edge_diag, _metric_pack, _f


def _load_rows(path: str) -> dict[str, dict[str, Any]]:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    out = {str(r.get("scenario_token", "")): r for r in rows}
    if len(out) != len(rows):
        raise SystemExit(f"duplicate scenario tokens in {path}")
    return out


def _eq_action(a: dict[str, Any], b: dict[str, Any], key: str = "bdse_action") -> bool:
    try:
        return int(a.get(key, -999999)) == int(b.get(key, -999998))
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="V64.3.20 deployment-complete ICER fresh causal screen checker.")
    for name in [
        "raw-metrics", "v19-scalar-metrics", "v20-scalar-metrics", "v20-dual-metrics",
        "raw-rows", "v19-scalar-rows", "v20-scalar-rows", "v20-dual-rows",
        "v19-scalar-edge-output", "v20-scalar-edge-output", "v20-dual-edge-output", "output",
    ]:
        ap.add_argument("--" + name, required=True)
    a = ap.parse_args()

    raw_m = json.load(open(a.raw_metrics)); v19_m = json.load(open(a.v19_scalar_metrics)); s_m = json.load(open(a.v20_scalar_metrics)); d_m = json.load(open(a.v20_dual_metrics))
    R, V19, S, D = map(_metric_pack, [raw_m, v19_m, s_m, d_m])
    v19e = _icer_edge_diag(Path(a.v19_scalar_edge_output)); se = _icer_edge_diag(Path(a.v20_scalar_edge_output)); de = _icer_edge_diag(Path(a.v20_dual_edge_output))
    rr = _load_rows(a.raw_rows); vr = _load_rows(a.v19_scalar_rows); sr = _load_rows(a.v20_scalar_rows); dr = _load_rows(a.v20_dual_rows)
    tokens = set(rr)
    if set(vr) != tokens or set(sr) != tokens or set(dr) != tokens:
        raise SystemExit("paired row token identity mismatch")

    all_flagged = [t for t in tokens if _f(rr[t], "all_actions_safety_flagged_rate", 0.0) >= 0.5]
    safe_domain = [t for t in tokens if t not in set(all_flagged)]
    def frac(xs: list[bool]) -> float:
        return float(sum(bool(x) for x in xs) / len(xs)) if xs else float("nan")

    structural = {
        "all_flagged_scene_count": float(len(all_flagged)),
        "v20_scalar_delegation_rate": frac([_f(sr[t], "decisive_frontier_icer_structural_domain_delegated", 0.0) >= .5 for t in all_flagged]),
        "v20_dual_delegation_rate": frac([_f(dr[t], "decisive_frontier_icer_structural_domain_delegated", 0.0) >= .5 for t in all_flagged]),
        "v20_scalar_preserve_legacy_rate": frac([int(sr[t].get("decisive_frontier_icer_selected_action", -1)) == int(sr[t].get("decisive_frontier_icer_legacy_selected_action", -2)) for t in all_flagged]),
        "v20_dual_preserve_legacy_rate": frac([int(dr[t].get("decisive_frontier_icer_selected_action", -1)) == int(dr[t].get("decisive_frontier_icer_legacy_selected_action", -2)) for t in all_flagged]),
        "v20_scalar_all_flagged_final_action_identity_vs_raw": frac([_eq_action(sr[t], rr[t]) for t in all_flagged]),
        "v20_dual_all_flagged_final_action_identity_vs_raw": frac([_eq_action(dr[t], rr[t]) for t in all_flagged]),
        "v20_scalar_safe_domain_selected_identity_vs_v19": frac([_eq_action(sr[t], vr[t], "decisive_frontier_icer_selected_action") for t in safe_domain]),
        "v20_scalar_safe_domain_final_identity_vs_v19": frac([_eq_action(sr[t], vr[t]) for t in safe_domain]),
        "v19_all_flagged_final_mismatch_vs_raw_rate": frac([not _eq_action(vr[t], rr[t]) for t in all_flagged]),
    }
    structural_support = len(all_flagged) >= 5
    deployment_complete = bool(
        structural_support
        and structural["v20_scalar_delegation_rate"] == 1.0
        and structural["v20_dual_delegation_rate"] == 1.0
        and structural["v20_scalar_preserve_legacy_rate"] == 1.0
        and structural["v20_dual_preserve_legacy_rate"] == 1.0
        and structural["v20_scalar_all_flagged_final_action_identity_vs_raw"] == 1.0
        and structural["v20_dual_all_flagged_final_action_identity_vs_raw"] == 1.0
        and structural["v20_scalar_safe_domain_selected_identity_vs_v19"] == 1.0
        and structural["v20_scalar_safe_domain_final_identity_vs_v19"] == 1.0
    )

    frozen_keys = ["selected_local_anchor_action_match", "pair_full_interface_action_match", "local_pair_full_interface_action_match", "evidence_certificate_fraction", "decision_budget_atom_count", "proposal_candidate_atom_count", "proposal_decisive_atom_recall", "selected_decisive_atom_recall", "effective_selected_decisive_atom_recall"]
    frozen = {k: bool(math.isfinite(_f(raw_m, k)) and math.isfinite(_f(d_m, k)) and abs(_f(raw_m, k) - _f(d_m, k)) <= 1e-6) for k in frozen_keys}
    instrumentation = de["scene_count"] >= 480 and de["admissible_edge_count"] >= 2048 and de["direct_counterfactual_dominance_edge_count"] >= 512 and _f(d_m, "decisive_frontier_value_complete_star_coverage") >= .99 and all(frozen.values())
    candidate_support = de["multi_admissible_proposal_rate"] >= .25 and de["admissible_candidates_per_proposal_mean"] >= 3.0
    fresh_signal = de["support_auc"] >= .65 and de["direct_counterfactual_dominance_auc"] >= .70
    recovery = de["alternative_recovery_rate"] >= .03 and de["alternative_recovery_precision"] >= .80 and de["direct_incumbent_replacement_rate"] >= .02 and de["direct_incumbent_replacement_precision"] >= .60 and de["direct_incumbent_opportunity_capture_rate"] >= .08 and de["selected_nonanchor_teacher_better_rate"] >= .80 and de["alternative_teacher_margin_mean"] > 0.0
    # The scalar V20 operator must preserve V19 safe-domain mechanism exactly; the
    # only allowed semantic difference is all-flagged structural-domain delegation.
    mechanism_identity = all(
        (math.isnan(float(v19e[k])) and math.isnan(float(se[k]))) or abs(float(v19e[k]) - float(se[k])) <= 1e-12
        for k in ["support_auc", "direct_counterfactual_dominance_auc", "direct_incumbent_replacement_rate", "direct_incumbent_replacement_precision", "direct_incumbent_opportunity_capture_rate"]
    )
    signed_profile_support = bool(
        de["direct_counterfactual_dominance_auc"] >= se["direct_counterfactual_dominance_auc"] + .005
        and de["alternative_recovery_precision"] >= se["alternative_recovery_precision"] - .005
        and de["direct_incumbent_replacement_precision"] >= se["direct_incumbent_replacement_precision"] - .03
        and de["direct_incumbent_opportunity_capture_rate"] >= se["direct_incumbent_opportunity_capture_rate"] - .02
        and D["regret"] <= S["regret"] * 1.02
        and D["match"] >= S["match"] - .005
        and D["harmful"] <= S["harmful"] + .005
    )
    deployment_alignment = D["guard_block"] <= .001
    retention = D["beneficial"] / max(R["beneficial"], 1e-12) if R["beneficial"] > 0 else float("nan")
    preservation = R["harmful"] - D["harmful"] >= .05 and retention >= .35 and D["beneficial"] > D["harmful"] and D["flip"] >= .03 and D["flip"] < R["flip"]
    anchor_match = _f(raw_m, "selected_local_anchor_action_match"); anchor_regret = _f(raw_m, "selected_local_anchor_teacher_regret")
    endpoint = D["match"] >= anchor_match + .005 and D["regret"] <= R["regret"] * 1.02 and D["match"] >= V19["match"] - .01
    full = bool(instrumentation and candidate_support and fresh_signal and recovery and mechanism_identity and signed_profile_support and deployment_complete and deployment_alignment and preservation and endpoint)

    if full:
        next_action = "independent_full_val_reproduction_freeze_V20_ICER_DC_heads_zero_thresholds_structural_domain_delegation_then_test_closed_loop_only_if_reproduced"
    elif not instrumentation:
        next_action = "engineering_stop_fix_V20_diagnostics_token_identity_or_frozen_interface"
    elif not deployment_complete:
        next_action = "deployment_complete_semantics_not_verified_audit_all_flagged_structural_guard_path_do_not_change_reliability_heads"
    elif not candidate_support or not fresh_signal or not recovery or not mechanism_identity:
        next_action = "V19_mechanism_failed_to_reproduce_in_safe_domain_do_not_tune_thresholds_audit_operator_or_data_identity"
    elif not signed_profile_support:
        next_action = "signed_profile_not_incremental_for_V20_demote_dual_main_keep_scalar_ICER_DC_and_redesign_attribution_only_if_needed"
    elif not deployment_alignment or not preservation:
        next_action = "V20_recovery_harms_guard_or_preservation_keep_structural_delegation_and_zero_thresholds_frozen_audit_false_positive_extremes"
    else:
        next_action = "V20_deployment_semantics_and_recovery_pass_but_endpoint_still_fails_now_audit_teacher_improvement_magnitude_ordering_without_selector_budget_or_threshold_changes"

    report = {
        "audit": "v64_3_20_eaf_icer_dc_screen",
        "full_promotion": full,
        "instrumentation_valid": instrumentation,
        "candidate_support_valid": candidate_support,
        "fresh_counterfactual_signal": fresh_signal,
        "v19_safe_domain_mechanism_identity": mechanism_identity,
        "signed_profile_incremental_support": signed_profile_support,
        "deployment_complete_structural_domain_support": deployment_complete,
        "deployment_alignment_invariant": deployment_alignment,
        "preservation_gain": preservation,
        "endpoint_gain": endpoint,
        "structural_domain_diagnostics": structural,
        "edge_diagnostics": {"v19_scalar_control": v19e, "v20_scalar": se, "v20_dual": de},
        "metrics": {"anchor": {"match": anchor_match, "regret": anchor_regret}, "raw": R, "v19_scalar_control": V19, "v20_scalar": S, "v20_dual": D, "dual_beneficial_retention_vs_raw": retention},
        "frozen_interface": frozen,
        "thresholds": {
            "all_flagged_min_scenes_for_structural_causal_support": 5,
            "fresh_support_auc_min": .65,
            "fresh_direct_dominance_auc_min": .70,
            "direct_incumbent_precision_min": .60,
            "direct_incumbent_capture_min": .08,
            "signed_profile_auc_gain_min": .005,
            "signed_profile_precision_max_drop": .03,
            "signed_profile_capture_max_drop": .02,
            "harmful_abs_reduction_vs_raw_min": .05,
            "beneficial_retention_min": .35,
            "teacher_match_over_anchor_min": .005,
            "regret_vs_raw_tolerance": .02,
        },
        "next_action": next_action,
        "interpretation": "V64.3.20 changes no learned head. It tests whether V19's mechanism-level success becomes a full endpoint success once all-flagged scenes preserve the frozen raw proposal and are delegated to the unchanged continuous structural-risk guard. Signed selected-evidence attribution remains incremental only if dual improves direct dominance discrimination without material replacement-precision/capture or endpoint harm.",
    }
    p = Path(a.output); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
