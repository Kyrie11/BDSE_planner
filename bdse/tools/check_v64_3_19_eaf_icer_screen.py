from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _f(d: dict[str, Any], k: str, default: float = float("nan")) -> float:
    try:
        return float(d.get(k, default))
    except Exception:
        return float(default)


def _mean(x: list[float]) -> float:
    return float(np.mean(x)) if x else float("nan")


def _auc(y: list[int], s: list[float]) -> float:
    yy = np.asarray(y, dtype=np.int64); ss = np.asarray(s, dtype=np.float64)
    good = np.isfinite(ss); yy, ss = yy[good], ss[good]
    pos, neg = int((yy == 1).sum()), int((yy == 0).sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(ss, kind="mergesort"); ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(ss) + 1, dtype=np.float64)
    _, inv, cnt = np.unique(ss, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ix = np.flatnonzero(inv == i); ranks[ix] = ranks[ix].mean()
    return float((ranks[yy == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _icer_edge_diag(path: Path, allowed_tokens: set[str] | None = None) -> dict[str, float]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if allowed_tokens is not None:
        rows = [r for r in rows if str(r.get("scenario_token", "")) in allowed_tokens]
    groups: dict[str, list[dict[str, Any]]] = {}
    support_y: list[int] = []; support_s: list[float] = []
    for r in rows:
        if _f(r, "icer_admissible", 0.0) >= 0.5:
            tm, sc = _f(r, "teacher_margin"), _f(r, "icer_support_logit")
            if math.isfinite(tm) and math.isfinite(sc):
                support_y.append(int(tm > 0.0)); support_s.append(sc)
        groups.setdefault(str(r.get("scenario_token", "")), []).append(r)

    proposal = selected_nonanchor = multi = alternatives = alt_good = 0
    cf_good = strict_raw_cf_good = opp = capture = fallback = changed = 0
    direct_incumbent_proposals = direct_incumbent_replacements = direct_incumbent_good = 0
    direct_incumbent_opp = direct_incumbent_capture = anchor_recoveries = anchor_recovery_good = 0
    admissible_counts: list[float] = []; selected_tm: list[float] = []; alt_tm: list[float] = []
    dom_y: list[int] = []; dom_s: list[float] = []; scalar_dom_s: list[float] = []; profile_dom_s: list[float] = []
    selected_good: list[float] = []
    for rs in groups.values():
        a = int(rs[0].get("anchor_action", -1)); lg = int(rs[0].get("raw_top_action", -1)); sel = int(rs[0].get("icer_selected_action", lg))
        by = {int(r.get("challenger_action", -1)): r for r in rs}
        adm = [r for r in rs if _f(r, "icer_admissible", 0.0) >= 0.5]
        if lg == a:
            continue
        proposal += 1; admissible_counts.append(float(len(adm))); multi += int(len(adm) >= 2)
        lr = by.get(lg); ltm = _f(lr or {}, "teacher_margin", 0.0); legacy_adm = bool(lr is not None and _f(lr, "icer_admissible", 0.0) >= 0.5)
        deployment_threshold = max(0.0, ltm) if legacy_adm else 0.0
        if legacy_adm:
            direct_incumbent_proposals += 1
            for r in adm:
                if int(r.get("challenger_action", -1)) == lg:
                    continue
                tm = _f(r, "teacher_margin"); ds = _f(r, "icer_dominance_logit")
                if math.isfinite(tm) and math.isfinite(ds):
                    dom_y.append(int(tm > deployment_threshold)); dom_s.append(ds)
                    scalar_dom_s.append(_f(r, "icer_scalar_dominance_logit")); profile_dom_s.append(_f(r, "icer_profile_dominance_logit"))
        sr = by.get(sel)
        if sel == a or sr is None:
            fallback += 1; selected_tm.append(0.0)
        else:
            stm = _f(sr, "teacher_margin", 0.0); selected_nonanchor += 1; selected_tm.append(stm); selected_good.append(float(stm > 0.0))
        changed += int(sel != lg)
        is_alt = bool(sel not in {a, lg} and sr is not None); alternatives += int(is_alt)
        if is_alt:
            stm = _f(sr or {}, "teacher_margin", 0.0); alt_tm.append(stm); alt_good += int(stm > 0.0)
            good = int(stm > deployment_threshold)
            cf_good += good; strict_raw_cf_good += int(stm > max(0.0, ltm))
            if legacy_adm:
                direct_incumbent_replacements += 1; direct_incumbent_good += good
            else:
                anchor_recoveries += 1; anchor_recovery_good += int(stm > 0.0)
        opportunity = any(int(r.get("challenger_action", -1)) not in {a, lg} and _f(r, "teacher_margin") > deployment_threshold for r in adm)
        opp += int(opportunity)
        capture += int(opportunity and is_alt and _f(sr or {}, "teacher_margin") > deployment_threshold)
        if legacy_adm:
            direct_incumbent_opp += int(opportunity)
            direct_incumbent_capture += int(opportunity and is_alt and _f(sr or {}, "teacher_margin") > deployment_threshold)
    div = lambda x, y: float(x / y) if y else float("nan")
    return {
        "scene_count": float(len(groups)), "proposal_scene_count": float(proposal),
        "admissible_edge_count": float(len(support_y)), "support_auc": _auc(support_y, support_s),
        "admissible_candidates_per_proposal_mean": _mean(admissible_counts), "multi_admissible_proposal_rate": div(multi, proposal),
        "direct_counterfactual_dominance_edge_count": float(len(dom_y)), "direct_counterfactual_dominance_auc": _auc(dom_y, dom_s),
        "scalar_counterfactual_dominance_auc": _auc(dom_y, scalar_dom_s),
        "profile_counterfactual_dominance_auc": _auc(dom_y, profile_dom_s) if any(math.isfinite(x) and abs(x) > 1e-12 for x in profile_dom_s) else float("nan"),
        "alternative_recovery_rate": div(alternatives, proposal), "alternative_recovery_precision": div(alt_good, alternatives),
        "alternative_teacher_margin_mean": _mean(alt_tm), "counterfactual_recovery_precision": div(cf_good, alternatives),
        "strict_raw_top_counterfactual_recovery_precision": div(strict_raw_cf_good, alternatives),
        "counterfactual_opportunity_rate": div(opp, proposal), "counterfactual_opportunity_capture_rate": div(capture, opp),
        "direct_incumbent_proposal_count": float(direct_incumbent_proposals),
        "direct_incumbent_replacement_rate": div(direct_incumbent_replacements, direct_incumbent_proposals),
        "direct_incumbent_replacement_precision": div(direct_incumbent_good, direct_incumbent_replacements),
        "direct_incumbent_opportunity_rate": div(direct_incumbent_opp, direct_incumbent_proposals),
        "direct_incumbent_opportunity_capture_rate": div(direct_incumbent_capture, direct_incumbent_opp),
        "anchor_recovery_rate_on_proposals": div(anchor_recoveries, proposal),
        "anchor_recovery_precision": div(anchor_recovery_good, anchor_recoveries),
        "selected_nonanchor_teacher_better_rate": _mean(selected_good), "selected_teacher_margin_mean_including_anchor": _mean(selected_tm),
        "anchor_fallback_rate": div(fallback, proposal), "proposal_changed_rate": div(changed, proposal),
    }


def _dacer_v18_diag(path: Path) -> dict[str, float]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows: groups.setdefault(str(r.get("scenario_token", "")), []).append(r)
    proposal = alt = good = cf = opp = cap = 0
    direct_prop = direct_alt = direct_good = direct_opp = direct_cap = 0
    for rs in groups.values():
        a = int(rs[0].get("anchor_action", -1)); lg = int(rs[0].get("raw_top_action", -1)); sel = int(rs[0].get("dacer_selected_action", lg))
        if lg == a: continue
        proposal += 1; by = {int(r.get("challenger_action", -1)): r for r in rs}; lr = by.get(lg); ltm = _f(lr or {}, "teacher_margin", 0.0)
        adm = [r for r in rs if _f(r, "dacer_admissible", 0.0) >= 0.5]; legacy_adm = bool(lr is not None and _f(lr, "dacer_admissible", 0.0) >= 0.5)
        direct_prop += int(legacy_adm)
        th = max(0.0, ltm) if legacy_adm else 0.0; sr = by.get(sel); is_alt = bool(sel not in {a, lg} and sr is not None); alt += int(is_alt)
        if is_alt:
            stm = _f(sr or {}, "teacher_margin", 0.0); good += int(stm > 0.0); cf += int(stm > th)
            if legacy_adm: direct_alt += 1; direct_good += int(stm > th)
        o = any(int(r.get("challenger_action", -1)) not in {a, lg} and _f(r, "teacher_margin") > th for r in adm); opp += int(o); cap += int(o and is_alt and _f(sr or {}, "teacher_margin") > th)
        if legacy_adm: direct_opp += int(o); direct_cap += int(o and is_alt and _f(sr or {}, "teacher_margin") > th)
    div = lambda x, y: float(x / y) if y else float("nan")
    return {"proposal_scene_count": float(proposal), "alternative_recovery_rate": div(alt, proposal), "alternative_recovery_precision": div(good, alt), "counterfactual_recovery_precision": div(cf, alt), "counterfactual_opportunity_capture_rate": div(cap, opp), "direct_incumbent_proposal_count": float(direct_prop), "direct_incumbent_replacement_rate": div(direct_alt, direct_prop), "direct_incumbent_replacement_precision": div(direct_good, direct_alt), "direct_incumbent_opportunity_capture_rate": div(direct_cap, direct_opp)}


def _metric_pack(m: dict[str, Any]) -> dict[str, float]:
    return {"match": _f(m, "teacher_action_match"), "regret": _f(m, "teacher_regret"), "harmful": _f(m, "harmful_pair_potential_intervention_rate"), "beneficial": _f(m, "beneficial_pair_potential_intervention_rate"), "flip": _f(m, "pair_potential_deployed_flip_rate"), "guard_block": _f(m, "pair_action_anchor_guard_blocked_flip", 0.0)}


def main() -> None:
    ap = argparse.ArgumentParser(description="V64.3.19 EAF-ICER pre-registered fresh causal screen checker.")
    for name in ["raw-metrics", "v18-metrics", "icer-scalar-metrics", "icer-dual-metrics", "v18-edge-output", "icer-scalar-edge-output", "icer-dual-edge-output", "icer-scalar-fit-report", "icer-dual-fit-report", "output"]:
        ap.add_argument("--" + name, required=True)
    a = ap.parse_args()
    raw = json.load(open(a.raw_metrics)); v18 = json.load(open(a.v18_metrics)); sm = json.load(open(a.icer_scalar_metrics)); dm = json.load(open(a.icer_dual_metrics))
    sf = json.load(open(a.icer_scalar_fit_report)); df = json.load(open(a.icer_dual_fit_report))
    ved = _dacer_v18_diag(Path(a.v18_edge_output)); sed = _icer_edge_diag(Path(a.icer_scalar_edge_output)); ded = _icer_edge_diag(Path(a.icer_dual_edge_output))
    R, V, S, D = map(_metric_pack, [raw, v18, sm, dm])
    anchor_match, anchor_regret = _f(raw, "selected_local_anchor_action_match"), _f(raw, "selected_local_anchor_teacher_regret")
    frozen_keys = ["selected_local_anchor_action_match", "pair_full_interface_action_match", "local_pair_full_interface_action_match", "evidence_certificate_fraction", "decision_budget_atom_count", "proposal_candidate_atom_count", "proposal_decisive_atom_recall", "selected_decisive_atom_recall", "effective_selected_decisive_atom_recall"]
    frozen = {k: bool(math.isfinite(_f(raw, k)) and math.isfinite(_f(dm, k)) and abs(_f(raw, k) - _f(dm, k)) <= 1e-6) for k in frozen_keys}

    instrumentation = ded["scene_count"] >= 480 and ded["admissible_edge_count"] >= 2048 and ded["direct_counterfactual_dominance_edge_count"] >= 512 and _f(dm, "decisive_frontier_value_complete_star_coverage") >= .99 and all(frozen.values())
    candidate_support = ded["multi_admissible_proposal_rate"] >= .25 and ded["admissible_candidates_per_proposal_mean"] >= 3.0
    train_signal = float(df.get("holdout_direct_counterfactual_dominance_auc", 0.0)) >= .70 and float(df.get("holdout_direct_incumbent_replacement_precision", 0.0)) >= .60 and float(df.get("holdout_direct_incumbent_opportunity_capture_rate", 0.0)) >= .08 and float(df.get("holdout_alternative_recovery_precision", 0.0)) >= .85 and float(sf.get("holdout_direct_counterfactual_dominance_auc", 0.0)) >= .70
    fresh_signal = math.isfinite(ded["support_auc"]) and ded["support_auc"] >= .65 and math.isfinite(ded["direct_counterfactual_dominance_auc"]) and ded["direct_counterfactual_dominance_auc"] >= .70
    recovery = ded["alternative_recovery_rate"] >= .03 and ded["alternative_recovery_precision"] >= .80 and ded["direct_incumbent_replacement_rate"] >= .02 and ded["direct_incumbent_replacement_precision"] >= .60 and ded["direct_incumbent_opportunity_capture_rate"] >= .08 and ded["selected_nonanchor_teacher_better_rate"] >= .80 and ded["alternative_teacher_margin_mean"] > 0.0
    recovery_gain_vs_v18 = ded["direct_incumbent_replacement_precision"] >= max(.60, ved["direct_incumbent_replacement_precision"] + .10) and ded["direct_incumbent_opportunity_capture_rate"] >= ved["direct_incumbent_opportunity_capture_rate"] - .01
    signed_profile_gain = (
        ded["direct_counterfactual_dominance_auc"] >= sed["direct_counterfactual_dominance_auc"] + .005
        or ded["direct_incumbent_replacement_precision"] >= sed["direct_incumbent_replacement_precision"] + .03
        or ded["direct_incumbent_opportunity_capture_rate"] >= sed["direct_incumbent_opportunity_capture_rate"] + .01
    ) and D["regret"] <= S["regret"] * 1.02 and D["match"] >= S["match"] - .005 and D["harmful"] <= S["harmful"] + .005
    deployment_alignment = D["guard_block"] <= .001
    retention = D["beneficial"] / max(R["beneficial"], 1e-12) if R["beneficial"] > 0 else float("nan")
    preservation = R["harmful"] - D["harmful"] >= .05 and retention >= .35 and D["beneficial"] > D["harmful"] and D["flip"] >= .03 and D["flip"] < R["flip"]
    endpoint = D["match"] >= anchor_match + .005 and D["regret"] <= R["regret"] * 1.02 and D["regret"] <= V["regret"] * 1.02 and D["match"] >= V["match"] - .005
    full = bool(instrumentation and candidate_support and train_signal and fresh_signal and recovery and recovery_gain_vs_v18 and signed_profile_gain and deployment_alignment and preservation and endpoint)

    if full:
        next_action = "independent_full_val_reproduction_with_ICER_support_dominance_maps_and_zero_thresholds_frozen_then_test_closed_loop_only_if_reproduced"
    elif not instrumentation:
        next_action = "engineering_stop_fix_ICER_diagnostics_token_identity_or_frozen_interface_mismatch"
    elif not candidate_support:
        next_action = "candidate_semantics_regressed_audit_guard_admissible_frontier_do_not_tune_thresholds_or_B_M"
    elif not train_signal or not fresh_signal:
        next_action = "incumbent_contrastive_representation_capacity_failed_keep_candidate_frontier_and_guards_frozen"
    elif not recovery or not recovery_gain_vs_v18:
        next_action = "counterfactual_precision_or_capture_still_bottlenecked_improve_incumbent_contrastive_evidence_interactions_not_selector_acquisition_or_thresholds"
    elif not signed_profile_gain:
        next_action = "dual_signed_profile_view_not_causally_helpful_keep_ICER_semantics_and_redesign_only_attribution_contrast_view"
    elif not deployment_alignment:
        next_action = "engineering_stop_preselection_admissibility_not_equivalent_to_final_guard"
    elif not preservation:
        next_action = "ICER_recovery_harms_preservation_keep_zero_thresholds_and_guards_frozen_audit_false_positive_extremes"
    else:
        next_action = "ICER_mechanism_passes_but_endpoint_not_reproduced_do_not_change_selector_or_budget_audit_teacher_improvement_ordering"

    report = {
        "audit": "v64_3_19_eaf_icer_screen", "full_promotion": full,
        "instrumentation_valid": instrumentation, "candidate_support_valid": candidate_support, "train_counterfactual_signal": train_signal, "fresh_counterfactual_signal": fresh_signal,
        "counterfactual_recovery_mechanism": recovery, "counterfactual_recovery_gain_vs_v18": recovery_gain_vs_v18, "signed_profile_dual_view_causal_support": signed_profile_gain,
        "deployment_alignment_invariant": deployment_alignment, "preservation_gain": preservation, "endpoint_gain": endpoint, "frozen_interface": frozen,
        "edge_diagnostics": {"v64_3_18_dacer_profile": ved, "icer_scalar": sed, "icer_dual": ded},
        "fit_diagnostics": {"icer_scalar": sf, "icer_dual": df},
        "metrics": {"anchor": {"match": anchor_match, "regret": anchor_regret}, "raw": R, "v64_3_18_dacer_profile": V, "icer_scalar": S, "icer_dual": D, "dual_beneficial_retention_vs_raw": retention},
        "thresholds": {"fresh_support_auc_min": .65, "fresh_direct_dominance_auc_min": .70, "alternative_recovery_rate_min": .03, "alternative_precision_min": .80, "direct_incumbent_replacement_rate_min": .02, "direct_incumbent_precision_min": .60, "direct_incumbent_capture_min": .08, "direct_incumbent_precision_gain_vs_v18_min": .10, "signed_profile_dominance_auc_gain_min": .005, "harmful_abs_reduction_vs_raw_min": .05, "beneficial_retention_min": .35, "teacher_match_over_anchor_min": .005, "regret_vs_raw_tolerance": .02, "post_selection_guard_block_max": .001},
        "next_action": next_action,
        "interpretation": "ICER is promotable only if the already-repaired deployment-admissible multi-challenger frontier remains healthy, a direct incumbent-contrastive reliability signal generalizes, direct replacements beat a deployment-admissible frozen incumbent with high precision and useful opportunity capture; anchor-recovery cases are reported separately, the fixed equal-weight signed-profile view adds causal value over scalar ICER, the unchanged final guard performs no hidden cleanup, and preservation/endpoint remain competitive. No validation threshold, loss-weight, B/M, acquisition, certificate, or support-head tuning is allowed."
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
