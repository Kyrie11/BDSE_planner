from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _f, _icer_edge_diag, _metric_pack
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _identity_rate, _load_rows
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _replacement_tail_diag
from bdse.tools.check_v64_3_30_eaf_icer_fbic_split import _query_diag
from bdse.tools.check_v64_3_33_eaf_icer_spcr_split import _containment

CAT = -0.5
NOOP_REDUCTION_MIN = 0.20
CAPTURE_TOL = 0.03
EPS = 1.0e-12


def _structural(rows, raw, flagged):
    return {
        "all_flagged_scene_count": len(flagged),
        "final_identity_vs_raw": _identity_rate(rows, raw, flagged),
        "icer_structural_delegation_rate": float(np.mean([_f(rows[t], "decisive_frontier_icer_structural_domain_delegated", 0.0) for t in flagged])) if flagged else 1.0,
    }


def _selected_policy_diag(edge_path: str, allowed: set[str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for line in Path(edge_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line); t = str(r.get("scenario_token", ""))
        if t in allowed:
            groups.setdefault(t, []).append(r)
    vals: list[float] = []; opp = cap = noop = oppsel = 0
    for rs in groups.values():
        if not rs:
            continue
        inc = int(rs[0].get("raw_top_action", -1)); by = {int(r.get("challenger_action", -2)): r for r in rs}; ir = by.get(inc)
        if ir is None or _f(ir, "icer_admissible", 0.0) < 0.5:
            continue
        itm = _f(ir, "teacher_margin")
        if not math.isfinite(itm):
            continue
        alts = []
        for r in rs:
            a = int(r.get("challenger_action", -2))
            if a == inc or _f(r, "icer_admissible", 0.0) < 0.5 or _f(r, "icer_support_logit", -math.inf) <= 0.0:
                continue
            y = _f(r, "teacher_margin") - itm
            if math.isfinite(y):
                alts.append((a, y))
        has = any(y > 0.0 for _, y in alts); opp += int(has)
        sel = int(rs[0].get("icer_selected_action", inc))
        if sel != inc and sel in by:
            y = _f(by[sel], "teacher_margin") - itm
            if math.isfinite(y):
                vals.append(float(y)); cap += int(has and y > 0.0); noop += int(not has); oppsel += int(has)
    a = np.asarray(vals, dtype=np.float64); neg = np.minimum(a, 0.0) if a.size else a
    return {
        "selected_count": int(a.size), "selected_positive_count": int((a > 0.0).sum()) if a.size else 0,
        "selected_precision": float(np.mean(a > 0.0)) if a.size else float("nan"),
        "teacher_improvement_sum": float(a.sum()) if a.size else 0.0,
        "teacher_improvement_worst": float(a.min()) if a.size else float("nan"),
        "teacher_negative_rms": float(np.sqrt(np.mean(neg * neg))) if a.size else 0.0,
        "catastrophic_count": int(np.sum(a <= CAT)) if a.size else 0,
        "positive_opportunity_scene_count": int(opp), "positive_capture_count": int(cap),
        "positive_capture_rate": float(cap / max(opp, 1)),
        "positive_opportunity_selected_count": int(oppsel),
        "no_positive_opportunity_false_intervention_count": int(noop),
    }


def _mechanism_gate(m: dict[str, Any], r: dict[str, Any]) -> dict[str, bool]:
    existence = bool(
        m["no_positive_opportunity_false_intervention_count"] <= (1.0 - NOOP_REDUCTION_MIN) * r["no_positive_opportunity_false_intervention_count"] + EPS
        and m["positive_capture_rate"] >= r["positive_capture_rate"] - CAPTURE_TOL - EPS
    )
    tail = bool(
        m["selected_count"] >= 8
        and m["teacher_improvement_sum"] >= -EPS
        and m["catastrophic_count"] == 0
        and math.isfinite(m["teacher_improvement_worst"])
        and m["teacher_improvement_worst"] > CAT
        and m["teacher_negative_rms"] <= r["teacher_negative_rms"] + EPS
    )
    return {"existence_and_capture": existence, "hard_tail": tail, "pass": bool(existence and tail)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit one V64.3.37 PVR fresh block")
    tags = ["raw", "v20", "preserve", "rsmr", "affine", "opvr"]
    ap.add_argument("--split-name", required=True)
    for tag in tags:
        ap.add_argument(f"--{tag}-metrics", required=True); ap.add_argument(f"--{tag}-rows", required=True)
        if tag != "raw":
            ap.add_argument(f"--{tag}-edges", required=True)
    ap.add_argument("--output", required=True); a = ap.parse_args()
    metrics = {t: json.load(open(getattr(a, t + "_metrics"))) for t in tags}
    rows = {t: _load_rows(getattr(a, t + "_rows")) for t in tags}; toks = set(rows["raw"])
    if len(toks) != 500 or any(set(rows[t]) != toks for t in tags[1:]):
        raise SystemExit("STOP DATA: V37 six arms must contain exact paired 500 scenes")
    flagged = {t for t in toks if _f(rows["raw"][t], "all_actions_safety_flagged_rate", 0.0) >= 0.5}; safe = toks - flagged
    q = {t: _query_diag(rows["v20"], rows[t], toks) for t in ["preserve", "rsmr", "affine", "opvr"]}
    query_ok = all(v["all_query_counts_exact_scene_parity"] for v in q.values())
    structural = {t: _structural(rows[t], rows["raw"], flagged) for t in ["v20", "preserve", "rsmr", "affine", "opvr"]}
    struct_ok = all((not flagged) or (structural[t]["final_identity_vs_raw"] == 1.0 and structural[t]["icer_structural_delegation_rate"] == 1.0) for t in ["preserve", "rsmr", "affine", "opvr"])
    ca = _containment(rows["rsmr"], rows["affine"], safe); co = _containment(rows["rsmr"], rows["opvr"], safe)
    edge = {t: _icer_edge_diag(Path(getattr(a, t + "_edges")), safe) for t in tags if t != "raw"}
    tail = {t: _replacement_tail_diag(rows["raw"], rows[t], getattr(a, t + "_edges"), safe) for t in tags if t != "raw"}
    policy = {t: _selected_policy_diag(getattr(a, t + "_edges"), safe) for t in ["rsmr", "affine", "opvr"]}
    M = {t: _metric_pack(metrics[t]) for t in tags}
    ag = _mechanism_gate(policy["affine"], policy["rsmr"]); og = _mechanism_gate(policy["opvr"], policy["rsmr"])
    preserve_cap = float(edge["preserve"]["direct_incumbent_opportunity_capture_rate"]); opvr_cap = float(edge["opvr"]["direct_incumbent_opportunity_capture_rate"])
    coverage = bool(math.isfinite(opvr_cap) and math.isfinite(preserve_cap) and opvr_cap >= preserve_cap + 0.03 - EPS)
    endp = bool(
        M["opvr"]["match"] >= M["preserve"]["match"] - 0.002 and M["opvr"]["regret"] <= M["preserve"]["regret"] * 1.005
        and M["opvr"]["match"] >= M["v20"]["match"] - 0.002 and M["opvr"]["regret"] <= M["v20"]["regret"] * 1.005
    )
    eng = bool(query_ok and struct_ok and ca["monotone_selected_policy_containment_valid"] and co["monotone_selected_policy_containment_valid"])
    full = bool(eng and og["pass"] and coverage and endp)
    if not eng:
        nxt = "STOP_fix_V37_engineering_or_frozen_winner_containment_before_scientific_interpretation"
    elif not og["existence_and_capture"]:
        nxt = "post_selection_value_does_not_reproduce_intervention_existence_gain_without_capture_loss_stop_do_not_expand_scene_geometry_or_context"
    elif not og["hard_tail"]:
        nxt = "proposal_value_sign_gain_exists_but_hard_selected_tail_still_fails_revisit_value_target_or_representation_not_RSMR_ordering"
    elif not coverage:
        nxt = "proposal_value_tail_is_safe_but_useful_recovery_coverage_is_insufficient"
    elif not endp:
        nxt = "direct_proposal_value_mechanism_passes_but_endpoint_does_not_convert_audit_runtime_path_composition"
    else:
        nxt = "if_second_fresh_block_also_passes_freeze_OPVR_and_run_one_independent_full_validation_reproduction"
    rep = {
        "audit": "v64_3_37_eaf_icer_pvr_split", "split_name": a.split_name, "full_split_pass": full,
        "engineering_valid": eng, "affine_score_only_mechanism_gate": ag, "orthogonal_proposal_value_mechanism_gate": og,
        "opvr_capture_gain_over_preserve": opvr_cap - preserve_cap if math.isfinite(opvr_cap) and math.isfinite(preserve_cap) else float("nan"),
        "opvr_meaningful_coverage": coverage, "endpoint_noninferior": endp, "next_action": nxt,
        "query_parity": q, "structural": structural, "affine_containment": ca, "opvr_containment": co,
        "edge_diagnostics": edge, "selected_policy_diagnostics": policy, "direct_selected_path_tail": tail, "metrics": M,
        "frozen_contract": {
            "catastrophic_threshold": CAT, "noop_reduction_fraction_min": NOOP_REDUCTION_MIN, "capture_tolerance": CAPTURE_TOL,
            "promotion_capture_gain_over_preserve_min": 0.03, "no_AB_pooling": True,
            "RSMR_winner_frozen_before_value_readout": True, "OPVR_residual_orthogonal_to_RSMR_score_direction": True,
            "no_second_best_fallback": True, "no_runtime_threshold_lambda_or_feature_sweep": True,
        },
    }
    Path(a.output).write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n"); print(json.dumps(rep, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
