from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from bdse.tools.check_v64_3_39_eaf_icer_cfsr_split import (
    EPS,
    _containment,
    _f,
    _icer_edge_diag,
    _load_rows,
    _mechanism_gate,
    _metric_pack,
    _query_diag,
    _replacement_tail_diag,
    _selected_policy_diag,
    _structural,
)

TAGS = [
    "raw",
    "v20",
    "preserve",
    "rsmr",
    "v42_quality",
    "q_anchor",
    "cv_anchor",
    "mean_anchor",
    "robust_anchor",
    "crvr",
]
VALUE_TAGS = ["v42_quality", "q_anchor", "cv_anchor", "mean_anchor", "robust_anchor", "crvr"]


def _same_final_action(a, b, tokens: set[str]) -> bool:
    for t in tokens:
        aa = int(_f(a[t], "bdse_action", -999))
        bb = int(_f(b[t], "bdse_action", -998))
        if aa != bb:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit one V64.3.43 CRVR untouched fresh block")
    ap.add_argument("--split-name", required=True)
    ap.add_argument("--fit-report", required=True)
    for tag in TAGS:
        x = tag.replace("_", "-")
        ap.add_argument(f"--{x}-metrics", dest=f"{tag}_metrics", required=True)
        ap.add_argument(f"--{x}-rows", dest=f"{tag}_rows", required=True)
        if tag != "raw":
            ap.add_argument(f"--{x}-edges", dest=f"{tag}_edges", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    fit = json.load(open(a.fit_report))
    promoted = str((fit.get("nested_crossfit", {}) or {}).get("promoted_arm") or "")
    if promoted not in {"q_anchor", "cv_anchor", "mean_anchor", "robust_anchor"}:
        raise SystemExit("STOP V43: TRAIN fit report does not contain a preregistered promoted arm")

    metrics = {t: json.load(open(getattr(a, t + "_metrics"))) for t in TAGS}
    rows = {t: _load_rows(getattr(a, t + "_rows")) for t in TAGS}
    toks = set(rows["raw"])
    if len(toks) != 500 or any(set(rows[t]) != toks for t in TAGS[1:]):
        raise SystemExit("STOP DATA: V43 arms must contain exact paired 500 scenes")

    flagged = {t for t in toks if _f(rows["raw"][t], "all_actions_safety_flagged_rate", 0.0) >= 0.5}
    safe = toks - flagged
    valtags = TAGS[2:]
    query = {t: _query_diag(rows["v20"], rows[t], toks) for t in valtags}
    structural = {t: _structural(rows[t], rows["raw"], flagged) for t in ["v20"] + valtags}
    containment = {t: _containment(rows["rsmr"], rows[t], safe) for t in VALUE_TAGS}
    promoted_identity = _same_final_action(rows[promoted], rows["crvr"], toks)
    engineering = bool(
        all(v["all_query_counts_exact_scene_parity"] for v in query.values())
        and all(
            (not flagged)
            or (
                structural[t]["final_identity_vs_raw"] == 1.0
                and structural[t]["icer_structural_delegation_rate"] == 1.0
            )
            for t in valtags
        )
        and all(x["monotone_selected_policy_containment_valid"] for x in containment.values())
        and promoted_identity
    )

    edge = {t: _icer_edge_diag(Path(getattr(a, t + "_edges")), safe) for t in TAGS if t != "raw"}
    policy = {t: _selected_policy_diag(getattr(a, t + "_edges"), safe) for t in ["rsmr"] + VALUE_TAGS}
    gates = {t: _mechanism_gate(policy[t], policy["rsmr"]) for t in VALUE_TAGS}
    M = {t: _metric_pack(metrics[t]) for t in TAGS}
    tail = {t: _replacement_tail_diag(rows["raw"], rows[t], getattr(a, t + "_edges"), safe) for t in TAGS if t != "raw"}

    preserve_cap = float(edge["preserve"]["direct_incumbent_opportunity_capture_rate"])
    main_cap = float(edge["crvr"]["direct_incumbent_opportunity_capture_rate"])
    coverage = bool(math.isfinite(main_cap) and math.isfinite(preserve_cap) and main_cap >= preserve_cap + 0.03 - EPS)
    endpoint = bool(
        M["crvr"]["match"] >= M["preserve"]["match"] - 0.002
        and M["crvr"]["regret"] <= M["preserve"]["regret"] * 1.005
        and M["crvr"]["match"] >= M["v20"]["match"] - 0.002
        and M["crvr"]["regret"] <= M["v20"]["regret"] * 1.005
    )
    full = bool(engineering and gates["crvr"]["pass"] and coverage and endpoint)

    if not engineering:
        nxt = "STOP_fix_V43_engineering_before_scientific_interpretation"
    elif gates[promoted]["pass"] and not gates["crvr"]["pass"]:
        nxt = "STOP_promoted_config_identity_or_runtime_replay_inconsistent"
    elif not gates["crvr"]["existence_and_capture"]:
        nxt = "counterfactual_response_value_does_not_reproduce_incumbent_exit_boundary_on_fresh"
    elif not gates["crvr"]["hard_tail"]:
        nxt = "counterfactual_response_value_recovers_capture_but_fresh_catastrophic_tail_remains"
    elif not coverage:
        nxt = "CRVR_direct_gate_passes_but_gain_over_preserve_is_insufficient"
    elif not endpoint:
        nxt = "CRVR_direct_gate_passes_but_endpoint_noninferiority_fails"
    else:
        nxt = "if_second_fresh_block_also_passes_freeze_CRVR_and_run_full_validation_plus_official_closed_loop"

    rep = {
        "audit": "v64_3_43_eaf_icer_crvr_split",
        "split_name": a.split_name,
        "promoted_train_arm": promoted,
        "full_split_pass": full,
        "engineering_valid": engineering,
        "promoted_config_identity_valid": promoted_identity,
        "mechanism_gates": gates,
        "crvr_capture_gain_over_preserve": main_cap - preserve_cap if math.isfinite(main_cap) and math.isfinite(preserve_cap) else float("nan"),
        "crvr_meaningful_coverage": coverage,
        "endpoint_noninferior": endpoint,
        "next_action": nxt,
        "query_parity": query,
        "structural": structural,
        "containment": containment,
        "edge_diagnostics": edge,
        "selected_policy_diagnostics": policy,
        "direct_selected_path_tail": tail,
        "metrics": M,
        "frozen_contract": {
            "RSMR_winner_frozen_before_any_value_readout": True,
            "analytic_cost_terms_normalized_by_exact_pair_margin_scale": True,
            "quality_cost_coefficients_fixed_plus_one": True,
            "selected_evidence_response_cost_coefficient_fixed_plus_one": True,
            "logged_future_forbidden_at_runtime": True,
            "no_selected_translation": True,
            "no_second_best_fallback": True,
            "no_AB_pooling": True,
            "fresh_gate_keeps_historical_zero_catastrophe_requirement": True,
        },
    }
    Path(a.output).write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
