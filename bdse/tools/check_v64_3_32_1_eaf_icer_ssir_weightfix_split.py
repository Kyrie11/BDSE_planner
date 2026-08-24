from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _auc, _f, _icer_edge_diag, _metric_pack
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _identity_rate, _load_rows
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _replacement_tail_diag
from bdse.tools.check_v64_3_30_eaf_icer_fbic_split import _query_diag

CAT = -0.5
MAIN_CAPTURE_GAIN_MIN = 0.03
EPS = 1.0e-12


def _edge_groups(path: str, allowed: set[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            t = str(r.get("scenario_token", ""))
            if t in allowed:
                out.setdefault(t, []).append(r)
    return out


def _structural(rows: dict[str, dict[str, Any]], raw: dict[str, dict[str, Any]], flagged: set[str]) -> dict[str, Any]:
    return {
        "all_flagged_scene_count": len(flagged),
        "final_identity_vs_raw": _identity_rate(rows, raw, flagged),
        "icer_structural_delegation_rate": float(np.mean([_f(rows[t], "decisive_frontier_icer_structural_domain_delegated", 0.0) for t in flagged])) if flagged else 1.0,
    }


def _direct_candidate_view(edge_path: str, allowed: set[str]) -> dict[str, Any]:
    groups = _edge_groups(edge_path, allowed)
    labels: list[int] = []
    mu_scores: list[float] = []
    lcb_scores: list[float] = []
    continuous_y: list[float] = []
    continuous_mu: list[float] = []
    continuous_lcb: list[float] = []
    scales: list[float] = []
    direct_eligible_scenes = 0
    simultaneous_covered = 0
    scene_miscoverage: list[str] = []
    eligible_candidates = 0
    selected_y: list[float] = []
    selected_tokens: list[str] = []
    harmful_selected: list[str] = []

    for tok, rs in groups.items():
        if not rs:
            continue
        inc = int(rs[0].get("raw_top_action", -1))
        by = {int(r.get("challenger_action", -2)): r for r in rs}
        ir = by.get(inc)
        if ir is None or _f(ir, "icer_admissible", 0.0) < 0.5:
            continue
        inc_tm = _f(ir, "teacher_margin")
        if not math.isfinite(inc_tm):
            continue
        alts: list[tuple[int, float, float, float, float]] = []
        for r in rs:
            act = int(r.get("challenger_action", -2))
            if act == inc or _f(r, "icer_admissible", 0.0) < 0.5 or _f(r, "icer_support_logit", -math.inf) <= 0.0:
                continue
            y = _f(r, "teacher_margin") - inc_tm
            mu = _f(r, "icer_scir_predicted_improvement")
            lcb = _f(r, "icer_scir_lower_bound")
            scale = _f(r, "icer_scir_selection_scale", 1.0)
            if not (math.isfinite(y) and math.isfinite(mu) and math.isfinite(lcb) and math.isfinite(scale) and scale > 0.0):
                continue
            alts.append((act, y, mu, lcb, scale))
            labels.append(int(y > 0.0)); mu_scores.append(mu); lcb_scores.append(lcb)
            continuous_y.append(y); continuous_mu.append(mu); continuous_lcb.append(lcb); scales.append(scale)
        if not alts:
            continue
        direct_eligible_scenes += 1
        eligible_candidates += len(alts)
        ok = all(y >= lcb - 1.0e-9 for _, y, _, lcb, _ in alts)
        simultaneous_covered += int(ok)
        if not ok:
            scene_miscoverage.append(tok)
        prop_exists = _f(rs[0], "icer_scir_proposal_exists", 0.0) >= 0.5
        prop = int(rs[0].get("icer_scir_proposal_action", inc))
        amap = {act: y for act, y, _, _, _ in alts}
        if prop_exists and prop in amap:
            y = float(amap[prop]); selected_y.append(y); selected_tokens.append(tok)
            if y <= 0.0:
                harmful_selected.append(tok)

    yarr = np.asarray(selected_y, dtype=np.float64)
    neg = np.minimum(yarr, 0.0) if yarr.size else yarr
    cy = np.asarray(continuous_y, dtype=np.float64)
    cm = np.asarray(continuous_mu, dtype=np.float64)
    cl = np.asarray(continuous_lcb, dtype=np.float64)
    corr_mu = float(np.corrcoef(cy, cm)[0, 1]) if cy.size >= 2 and np.std(cy) > 0 and np.std(cm) > 0 else float("nan")
    corr_lcb = float(np.corrcoef(cy, cl)[0, 1]) if cy.size >= 2 and np.std(cy) > 0 and np.std(cl) > 0 else float("nan")
    return {
        "direct_eligible_scene_count": direct_eligible_scenes,
        "eligible_candidate_count": eligible_candidates,
        "eligible_candidates_per_direct_scene_mean": float(eligible_candidates / max(direct_eligible_scenes, 1)),
        "mean_improvement_sign_auc": _auc(labels, mu_scores),
        "lcb_sign_auc": _auc(labels, lcb_scores),
        "mean_vs_teacher_pearson": corr_mu,
        "lcb_vs_teacher_pearson": corr_lcb,
        "selection_scale_mean": float(np.mean(scales)) if scales else float("nan"),
        "selection_scale_max": float(np.max(scales)) if scales else float("nan"),
        "empirical_scene_simultaneous_coverage": float(simultaneous_covered / max(direct_eligible_scenes, 1)),
        "scene_simultaneous_miscoverage_count": len(scene_miscoverage),
        "selected_count": int(yarr.size),
        "selected_positive_count": int((yarr > 0.0).sum()) if yarr.size else 0,
        "selected_precision": float((yarr > 0.0).mean()) if yarr.size else float("nan"),
        "selected_teacher_improvement_sum": float(yarr.sum()) if yarr.size else 0.0,
        "selected_teacher_improvement_worst": float(yarr.min()) if yarr.size else float("nan"),
        "selected_negative_rms": float(np.sqrt(np.mean(neg * neg))) if yarr.size else 0.0,
        "selected_harmful_count": len(harmful_selected),
        "selected_harmful_is_subset_of_miscoverage": set(harmful_selected).issubset(set(scene_miscoverage)),
        "example_scene_miscoverage": scene_miscoverage[:10],
        "example_selected_harmful": harmful_selected[:10],
    }


def _ssir_runtime_contract(main_rows: dict[str, dict[str, Any]], main_edges: str, allowed: set[str]) -> dict[str, Any]:
    groups = _edge_groups(main_edges, allowed)
    wrong_winner: list[str] = []
    wrong_no_proposal: list[str] = []
    fallback: list[str] = []
    checked = 0
    inferred_q: list[float] = []
    for tok in sorted(allowed):
        rs = groups.get(tok, [])
        if not rs:
            continue
        inc = int(rs[0].get("raw_top_action", -1))
        by = {int(r.get("challenger_action", -2)): r for r in rs}
        ir = by.get(inc)
        if ir is None or _f(ir, "icer_admissible", 0.0) < 0.5:
            continue
        cands: list[tuple[int, float, float, float, int]] = []
        for r in rs:
            act = int(r.get("challenger_action", -2))
            if act == inc or _f(r, "icer_admissible", 0.0) < 0.5 or _f(r, "icer_support_logit", -math.inf) <= 0.0:
                continue
            lcb = _f(r, "icer_scir_lower_bound")
            mu = _f(r, "icer_scir_predicted_improvement")
            scale = _f(r, "icer_scir_selection_scale", 1.0)
            if math.isfinite(mu) and math.isfinite(lcb) and math.isfinite(scale) and scale > 0:
                inferred_q.append((mu - lcb) / scale)
            if math.isfinite(lcb) and lcb > 0.0:
                cands.append((act, lcb, _f(r, "icer_support_logit", -math.inf), _f(r, "raw_margin", -math.inf), int(_f(r, "dacer_utility_prior", 0.0) >= 0.5)))
        checked += 1
        expected = None
        if cands:
            expected = sorted(cands, key=lambda x: (-x[1], -x[2], -x[3], -x[4], x[0]))[0][0]
        prop_exists = _f(rs[0], "icer_scir_proposal_exists", 0.0) >= 0.5
        prop = int(rs[0].get("icer_scir_proposal_action", inc))
        baseline = int(round(_f(main_rows[tok], "decisive_frontier_icer_baseline_action", inc)))
        selected = int(round(_f(main_rows[tok], "decisive_frontier_icer_selected_action", baseline)))
        accepted = _f(main_rows[tok], "decisive_frontier_icer_scir_certificate_accepted", 0.0) >= 0.5
        if expected is None:
            if prop_exists or selected != baseline or accepted:
                wrong_no_proposal.append(tok)
        else:
            if not prop_exists or prop != expected or selected != expected or not accepted:
                wrong_winner.append(tok)
        # There is no two-stage fallback in SSIR: a positive-LCB winner is the
        # selected intervention, otherwise incumbent-default baseline is returned.
        if expected is None and selected not in {baseline}:
            fallback.append(tok)
    qarr = np.asarray([x for x in inferred_q if math.isfinite(x)], dtype=np.float64)
    return {
        "direct_domain_scenes_checked": checked,
        "positive_lcb_winner_contract_violation_count": len(wrong_winner),
        "no_positive_lcb_incumbent_default_violation_count": len(wrong_no_proposal),
        "fallback_violation_count": len(fallback),
        "selection_contract_valid": not wrong_winner and not wrong_no_proposal and not fallback,
        "inferred_quantile_mean": float(qarr.mean()) if qarr.size else float("nan"),
        "inferred_quantile_range": float(qarr.max() - qarr.min()) if qarr.size else float("nan"),
        "example_winner_violations": wrong_winner[:10],
        "example_no_proposal_violations": wrong_no_proposal[:10],
    }


def _mean_main_reordering(mean_edges: str, main_edges: str, allowed: set[str]) -> dict[str, Any]:
    mg = _edge_groups(mean_edges, allowed); sg = _edge_groups(main_edges, allowed)
    compared = changed = mean_prop = main_prop = 0
    for t in sorted(allowed):
        a = mg.get(t, []); b = sg.get(t, [])
        if not a or not b:
            continue
        ma = _f(a[0], "icer_scir_proposal_exists", 0.0) >= 0.5
        mb = _f(b[0], "icer_scir_proposal_exists", 0.0) >= 0.5
        pa = int(a[0].get("icer_scir_proposal_action", -1)); pb = int(b[0].get("icer_scir_proposal_action", -1))
        mean_prop += int(ma); main_prop += int(mb); compared += 1
        changed += int((ma, pa) != (mb, pb))
    return {
        "scene_count_compared": compared,
        "mean_proposal_count": mean_prop,
        "ssir_proposal_count": main_prop,
        "proposal_identity_changed_scene_count": changed,
        "proposal_identity_changed_scene_rate": float(changed / max(compared, 1)),
        "interpretation": "A nonzero change rate is intended in V32: candidate-specific lower bounds enter ordering before extremal selection, unlike V31 common-offset post-selection veto.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit one untouched V64.3.32.1 SSIR block: raw/V20/PRESERVE/MEAN/SSIR.")
    ap.add_argument("--split-name", required=True)
    for tag in ["raw", "v20", "preserve", "mean", "main"]:
        ap.add_argument(f"--{tag}-metrics", required=True)
        ap.add_argument(f"--{tag}-rows", required=True)
        if tag != "raw":
            ap.add_argument(f"--{tag}-edges", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    tags = ["raw", "v20", "preserve", "mean", "main"]
    metrics = {t: json.load(open(getattr(a, t + "_metrics"), encoding="utf-8")) for t in tags}
    rows = {t: _load_rows(getattr(a, t + "_rows")) for t in tags}
    tokens = set(rows["raw"])
    if len(tokens) != 500 or any(set(rows[t]) != tokens for t in tags[1:]):
        raise SystemExit("STOP DATA: V32 all five arms must contain exact paired 500-scene identity")
    flagged = {t for t in tokens if _f(rows["raw"][t], "all_actions_safety_flagged_rate", 0.0) >= 0.5}
    safe = tokens - flagged

    query = {t: _query_diag(rows["v20"], rows[t], tokens) for t in ["preserve", "mean", "main"]}
    query_ok = all(x["all_query_counts_exact_scene_parity"] for x in query.values())
    structural = {t: _structural(rows[t], rows["raw"], flagged) for t in ["v20", "preserve", "mean", "main"]}
    structural_ok = all((not flagged) or (structural[t]["final_identity_vs_raw"] == 1.0 and structural[t]["icer_structural_delegation_rate"] == 1.0) for t in ["preserve", "mean", "main"])

    runtime_contract = _ssir_runtime_contract(rows["main"], getattr(a, "main_edges"), safe)
    edge = {t: _icer_edge_diag(Path(getattr(a, t + "_edges")), safe) for t in ["v20", "preserve", "mean", "main"]}
    tail = {t: _replacement_tail_diag(rows["raw"], rows[t], getattr(a, t + "_edges"), safe) for t in ["v20", "preserve", "mean", "main"]}
    candidate = {t: _direct_candidate_view(getattr(a, t + "_edges"), safe) for t in ["mean", "main"]}
    reorder = _mean_main_reordering(getattr(a, "mean_edges"), getattr(a, "main_edges"), safe)
    M = {t: _metric_pack(metrics[t]) for t in tags}

    pres_capture = float(edge["preserve"]["direct_incumbent_opportunity_capture_rate"])
    main_capture = float(edge["main"]["direct_incumbent_opportunity_capture_rate"])
    meaningful_coverage = bool(math.isfinite(pres_capture) and math.isfinite(main_capture) and main_capture >= pres_capture + MAIN_CAPTURE_GAIN_MIN - EPS)
    main_tail = bool(
        tail["main"]["count"] >= 8
        and tail["main"]["teacher_improvement_sum"] >= -1.0e-9
        and math.isfinite(tail["main"]["teacher_improvement_worst"])
        and tail["main"]["teacher_improvement_worst"] > CAT
        and tail["main"]["teacher_negative_rms"] <= tail["preserve"]["teacher_negative_rms"] + EPS
        and edge["main"]["direct_incumbent_replacement_precision"] >= edge["preserve"]["direct_incumbent_replacement_precision"] - EPS
    )
    # The new bound is supposed to fix the selected path, not merely reproduce
    # the same mean winner.  It must be non-worse than MEAN on both tail axes and
    # strictly improve at least one tail/precision diagnostic unless MEAN selects
    # nothing.  Reordering itself is diagnostic, not a mandatory source of gain.
    mean_count = int(tail["mean"]["count"])
    tail_nonworse_mean = bool(
        tail["main"]["teacher_negative_rms"] <= tail["mean"]["teacher_negative_rms"] + EPS
        and (
            not math.isfinite(tail["mean"]["teacher_improvement_worst"])
            or tail["main"]["teacher_improvement_worst"] >= tail["mean"]["teacher_improvement_worst"] - EPS
        )
    )
    tail_strict_mean = bool(
        mean_count == 0
        or tail["main"]["teacher_negative_rms"] < tail["mean"]["teacher_negative_rms"] - 1.0e-9
        or (math.isfinite(tail["main"]["teacher_improvement_worst"]) and math.isfinite(tail["mean"]["teacher_improvement_worst"]) and tail["main"]["teacher_improvement_worst"] > tail["mean"]["teacher_improvement_worst"] + 1.0e-9)
        or edge["main"]["direct_incumbent_replacement_precision"] > edge["mean"]["direct_incumbent_replacement_precision"] + 1.0e-9
    )
    selected_path_incremental = bool(tail_nonworse_mean and tail_strict_mean)

    endpoint_vs_pres = bool(M["main"]["match"] >= M["preserve"]["match"] - 0.002 and M["main"]["regret"] <= M["preserve"]["regret"] * 1.005)
    endpoint_vs_v20 = bool(M["main"]["match"] >= M["v20"]["match"] - 0.002 and M["main"]["regret"] <= M["v20"]["regret"] * 1.005)
    endpoint = bool(endpoint_vs_pres and endpoint_vs_v20)

    theorem_instrumentation = bool(
        candidate["main"]["direct_eligible_scene_count"] >= 64
        and candidate["main"]["selected_harmful_is_subset_of_miscoverage"]
        and runtime_contract["inferred_quantile_range"] <= 1.0e-8
    )
    engineering = bool(query_ok and structural_ok and runtime_contract["selection_contract_valid"] and theorem_instrumentation)
    full = bool(engineering and meaningful_coverage and main_tail and selected_path_incremental and endpoint)

    if not engineering:
        nxt = "STOP_fix_SSIR_engineering_selection_or_calibration_contract_before_scientific_interpretation"
    elif not meaningful_coverage:
        nxt = "SSIR_does_not_add_3pp_direct_capture_over_preservation_control_stop_do_not_sweep_alpha_or_scale"
    elif not main_tail:
        nxt = "SSIR_coverage_exists_but_selected_path_tail_or_precision_fails_stop_do_not_threshold_rescue"
    elif not selected_path_incremental:
        nxt = "candidate_specific_simultaneous_bound_does_not_improve_mean_selected_tail_stop_revisit_risk_ordering_semantics"
    elif not endpoint:
        nxt = "direct_SSIR_mechanism_passes_but_endpoint_does_not_convert_audit_path_composition_before_new_representation"
    else:
        nxt = "if_second_fresh_block_also_passes_freeze_SSIR_and_run_exactly_one_independent_full_validation_reproduction"

    report = {
        "audit": "v64_3_32_1_eaf_icer_ssir_split",
        "split_name": a.split_name,
        "full_split_pass": full,
        "engineering_valid": engineering,
        "main_direct_capture_gain_over_preservation_control": main_capture - pres_capture if math.isfinite(main_capture) and math.isfinite(pres_capture) else float("nan"),
        "main_meaningful_direct_coverage_gain": meaningful_coverage,
        "main_selected_path_tail_safe_vs_preservation": main_tail,
        "ssir_selected_path_incremental_vs_mean_control": selected_path_incremental,
        "endpoint_noninferior_to_preservation_and_v20": endpoint,
        "endpoint_noninferior_vs_preservation_control": endpoint_vs_pres,
        "endpoint_noninferior_vs_v20": endpoint_vs_v20,
        "next_action": nxt,
        "query_parity_vs_v20": query,
        "structural": structural,
        "ssir_runtime_selection_contract": runtime_contract,
        "mean_to_ssir_reordering": reorder,
        "edge_diagnostics": edge,
        "candidate_level_diagnostics": candidate,
        "direct_selected_path_tail": tail,
        "metrics": M,
        "frozen_thresholds": {
            "catastrophic_teacher_improvement_threshold": CAT,
            "main_capture_gain_over_preservation_control_min": MAIN_CAPTURE_GAIN_MIN,
            "main_selected_count_min": 8,
            "main_selected_teacher_improvement_sum_min": 0.0,
            "main_precision_required_noninferior_to_preservation_control": True,
            "main_negative_rms_required_nonworse_than_preservation_control": True,
            "main_worst_required_above_catastrophic_threshold": True,
            "main_tail_required_nonworse_than_mean_and_strictly_better_on_at_least_one_tail_or_precision_metric": True,
            "main_match_tolerance_abs": 0.002,
            "main_regret_tolerance_relative": 0.005,
            "endpoint_must_pass_vs_both_preservation_control_and_v20": True,
            "no_pooled_AB_rescue": True,
            "conformal_alpha_frozen": 0.05,
            "no_alpha_scale_ridge_or_threshold_sweep": True,
        },
        "causal_control_note": "PRESERVE isolates admissible-incumbent default; MEAN is the exact V31 same-scene mean-ordering idea without a bound; SSIR changes the intervention decision by ranking positive candidate-specific simultaneous lower bounds. Thus fresh A/B can distinguish incumbent-default, mean ordering, and selection-stable risk ordering.",
        "coverage_note": "Empirical simultaneous coverage is a theorem diagnostic, not a hard 95% sample-frequency gate. The hard safety gates remain the actually selected path sum/worst/NegRMS/precision and independent A/B replication.",
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
