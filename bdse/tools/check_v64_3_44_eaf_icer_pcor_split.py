from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from bdse.tools.check_v64_3_39_eaf_icer_cfsr_split import (
    EPS, _containment, _f, _icer_edge_diag, _load_rows, _mechanism_gate,
    _metric_pack, _query_diag, _replacement_tail_diag, _selected_policy_diag, _structural,
)

TAGS = [
    "raw", "v20", "preserve", "rsmr", "quality", "v43_future_mean", "v43_future_robust",
    "pc_response_mean", "pc_occupancy_mean", "pc_occupancy_robust", "pcor",
]
VALUE_TAGS = TAGS[4:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-name", required=True)
    for tag in TAGS:
        x = tag.replace("_", "-")
        ap.add_argument(f"--{x}-metrics", dest=f"{tag}_metrics", required=True)
        ap.add_argument(f"--{x}-rows", dest=f"{tag}_rows", required=True)
        if tag != "raw":
            ap.add_argument(f"--{x}-edges", dest=f"{tag}_edges", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    metrics = {t: json.load(open(getattr(a, t + "_metrics"))) for t in TAGS}
    rows = {t: _load_rows(getattr(a, t + "_rows")) for t in TAGS}
    toks = set(rows["raw"])
    if len(toks) != 500 or any(set(rows[t]) != toks for t in TAGS[1:]):
        raise SystemExit("STOP DATA: V44 arms must contain exact paired 500 scenes")

    flagged = {t for t in toks if _f(rows["raw"][t], "all_actions_safety_flagged_rate", 0) >= .5}
    safe = toks - flagged
    q = {t: _query_diag(rows["v20"], rows[t], toks) for t in TAGS[2:]}
    structural = {t: _structural(rows[t], rows["raw"], flagged) for t in ["v20"] + TAGS[2:]}
    cont = {t: _containment(rows["rsmr"], rows[t], safe) for t in VALUE_TAGS}
    eng = (
        all(v["all_query_counts_exact_scene_parity"] for v in q.values())
        and all((not flagged) or (structural[t]["final_identity_vs_raw"] == 1.0 and structural[t]["icer_structural_delegation_rate"] == 1.0) for t in TAGS[2:])
        and all(x["monotone_selected_policy_containment_valid"] for x in cont.values())
    )
    edge = {t: _icer_edge_diag(Path(getattr(a, t + "_edges")), safe) for t in TAGS if t != "raw"}
    policy = {t: _selected_policy_diag(getattr(a, t + "_edges"), safe) for t in ["rsmr"] + VALUE_TAGS}
    gates = {t: _mechanism_gate(policy[t], policy["rsmr"]) for t in VALUE_TAGS}
    M = {t: _metric_pack(metrics[t]) for t in TAGS}
    tail = {t: _replacement_tail_diag(rows["raw"], rows[t], getattr(a, t + "_edges"), safe) for t in TAGS if t != "raw"}

    pc = float(edge["preserve"]["direct_incumbent_opportunity_capture_rate"])
    mc = float(edge["pcor"]["direct_incumbent_opportunity_capture_rate"])
    coverage = math.isfinite(pc) and math.isfinite(mc) and mc >= pc + .03 - EPS
    endpoint = (
        M["pcor"]["match"] >= M["preserve"]["match"] - .002
        and M["pcor"]["regret"] <= M["preserve"]["regret"] * 1.005
        and M["pcor"]["match"] >= M["v20"]["match"] - .002
        and M["pcor"]["regret"] <= M["v20"]["regret"] * 1.005
    )
    full = bool(eng and gates["pcor"]["pass"] and coverage and endpoint)

    if not eng:
        nxt = "STOP_fix_V44_engineering_before_scientific_interpretation"
    elif gates["pc_response_mean"]["pass"] and not gates["pc_occupancy_mean"]["pass"]:
        nxt = "plan_conditioned_mode_reweighting_generalizes_without_support_extension_keep_reweight_only"
    elif gates["pc_occupancy_mean"]["pass"] and not gates["pc_occupancy_robust"]["pass"]:
        nxt = "ungated_plan_conditioned_occupancy_generalizes_but_CVaR_not_required_keep_mean"
    elif not gates["pcor"]["existence_and_capture"]:
        nxt = "plan_conditioning_support_still_does_not_close_zero_boundary_require_continuous_plan_conditioned_occupancy_or_behavior_prediction"
    elif not gates["pcor"]["hard_tail"]:
        nxt = "PCOR_recovers_capture_but_tail_requires_richer_continuous_response_support_not_threshold_tuning"
    elif not coverage:
        nxt = "PCOR_direct_gate_passes_but_gain_over_preserve_insufficient"
    elif not endpoint:
        nxt = "PCOR_direct_gate_passes_but_endpoint_noninferiority_fails"
    else:
        nxt = "if_second_fresh_block_also_passes_freeze_PCOR_and_run_full_validation_plus_official_nuPlan_closed_loop"

    rep = {
        "audit": "v64_3_44_eaf_icer_pcor_split",
        "split_name": a.split_name,
        "full_split_pass": full,
        "engineering_valid": eng,
        "mechanism_gates": gates,
        "pcor_capture_gain_over_preserve": mc - pc if math.isfinite(mc) and math.isfinite(pc) else float("nan"),
        "endpoint_noninferior": endpoint,
        "next_action": nxt,
        "query_parity": q,
        "structural": structural,
        "containment": cont,
        "edge_diagnostics": edge,
        "selected_policy_diagnostics": policy,
        "direct_selected_path_tail": tail,
        "metrics": M,
        "frozen_contract": {
            "RSMR_winner_frozen": True,
            "V42_QUALITY_frozen": True,
            "V43_future_controls_frozen": True,
            "TRAIN_only_behavior_supervision_no_teacher_value": True,
            "deployment_no_logged_future": True,
            "plan_conditioned_posterior_vs_ungated_occupancy_ablation_preregistered": True,
            "CAL500_translation_unit_slope": True,
            "no_second_best_fallback": True,
            "no_AB_pooling": True,
        },
    }
    Path(a.output).write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
