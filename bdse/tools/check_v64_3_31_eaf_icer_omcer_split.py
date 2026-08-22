from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from bdse.tools.check_v64_3_19_eaf_icer_screen import _icer_edge_diag, _metric_pack, _f
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _load_rows, _path_diag, _guard_block_rate, _identity_rate
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _replacement_tail_diag

CAT_THRESHOLD = -0.5


def _positive_count(tail: dict[str, Any]) -> int:
    count = int(tail.get("count", 0))
    precision = float(tail.get("teacher_positive_precision", float("nan")))
    return int(round(count * precision)) if math.isfinite(precision) else 0


def _tail_noninferior(main: dict[str, Any], ctrl: dict[str, Any], eps: float = 1e-9) -> bool:
    keys_le = ["regret_positive_rms", "worst_regret_increase", "teacher_negative_rms"]
    if any(not (math.isfinite(float(main.get(k, math.nan))) and math.isfinite(float(ctrl.get(k, math.nan)))) for k in keys_le):
        return False
    if not (math.isfinite(float(main.get("teacher_improvement_worst", math.nan))) and math.isfinite(float(ctrl.get("teacher_improvement_worst", math.nan)))):
        return False
    return bool(
        all(float(main[k]) <= float(ctrl[k]) + eps for k in keys_le)
        and float(main["teacher_improvement_worst"]) >= float(ctrl["teacher_improvement_worst"]) - eps
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="One untouched 500-scene V64.3.31 OMCER mechanism block.")
    ap.add_argument("--split-name", required=True)
    for name in ["raw", "v20", "aggregate-downside", "lock-downside", "omcer"]:
        ap.add_argument(f"--{name}-metrics", required=True)
        ap.add_argument(f"--{name}-rows", required=True)
        if name != "raw":
            ap.add_argument(f"--{name}-edges", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cli = ["raw", "v20", "aggregate-downside", "lock-downside", "omcer"]
    tags = [x.replace("-", "_") for x in cli]
    metrics = {t: json.load(open(getattr(args, t + "_metrics"), encoding="utf-8")) for t in tags}
    rows = {t: _load_rows(getattr(args, t + "_rows")) for t in tags}
    tokens = set(rows["raw"])
    if len(tokens) != 500 or any(set(rows[t]) != tokens for t in tags[1:]):
        raise SystemExit("STOP DATA: exact paired 500-scene identity required")

    all_flagged = {t for t in tokens if _f(rows["raw"][t], "all_actions_safety_flagged_rate", 0.0) >= 0.5}
    safe = tokens - all_flagged
    edge_paths = {t: getattr(args, t + "_edges") for t in tags[1:]}
    edge = {t: _icer_edge_diag(Path(edge_paths[t]), safe) for t in tags[1:]}
    path = {t: _path_diag(rows["raw"], rows[t], edge_paths[t], safe, all_flagged) for t in tags[1:]}
    tails = {t: _replacement_tail_diag(rows["raw"], rows[t], edge_paths[t], safe) for t in ["aggregate_downside", "lock_downside", "omcer"]}
    M = {t: _metric_pack(metrics[t]) for t in tags}

    structural = {
        "all_flagged_scene_count": len(all_flagged),
        "main_all_flagged_final_identity_vs_raw": _identity_rate(rows["omcer"], rows["raw"], all_flagged),
        "main_all_flagged_delegation_rate": (
            sum(_f(rows["omcer"][t], "decisive_frontier_icer_structural_domain_delegated", 0.0) >= 0.5 for t in all_flagged) / len(all_flagged)
            if all_flagged else float("nan")
        ),
        "main_safe_guard_block_rate": _guard_block_rate(rows["omcer"], safe),
    }
    structural_ok = bool(
        len(all_flagged) >= 3
        and structural["main_all_flagged_final_identity_vs_raw"] == 1.0
        and structural["main_all_flagged_delegation_rate"] == 1.0
        and structural["main_safe_guard_block_rate"] <= 0.001
    )

    main_path = path["omcer"]["direct_incumbent_to_alternative"]
    inc_anchor = path["omcer"]["admissible_incumbent_to_anchor"]
    asymmetric = bool(inc_anchor["count"] == 0 and abs(inc_anchor["regret_delta_sum"]) <= 1e-9)
    main_tail = tails["omcer"]
    catastrophe_free = bool(
        main_tail["count"] == 0
        or (math.isfinite(main_tail["teacher_improvement_worst"]) and main_tail["teacher_improvement_worst"] > CAT_THRESHOLD)
    )
    selected_path_safe = bool(main_path["count"] >= 8 and main_path["regret_delta_sum"] <= 0.0 and catastrophe_free)

    main_edge = edge["omcer"]
    v25_edge = edge["aggregate_downside"]
    lock_edge = edge["lock_downside"]
    main_pos = _positive_count(main_tail)
    v25_pos = _positive_count(tails["aggregate_downside"])
    lock_pos = _positive_count(tails["lock_downside"])
    coverage_vs_v25 = bool(
        main_edge["direct_incumbent_opportunity_capture_rate"] >= v25_edge["direct_incumbent_opportunity_capture_rate"] + 0.03
        and main_pos >= v25_pos + 5
    )
    coverage_vs_lock = bool(
        main_edge["direct_incumbent_opportunity_capture_rate"] >= lock_edge["direct_incumbent_opportunity_capture_rate"] + 0.03
        and main_pos >= lock_pos + 5
    )
    tail_vs_v25 = _tail_noninferior(main_tail, tails["aggregate_downside"])
    tail_vs_lock = _tail_noninferior(main_tail, tails["lock_downside"])

    instrumentation = bool(
        main_edge["scene_count"] >= 450
        and main_edge["admissible_edge_count"] >= 1800
        and main_edge["direct_counterfactual_dominance_edge_count"] >= 400
        and _f(metrics["omcer"], "decisive_frontier_value_complete_star_coverage") >= 0.99
    )
    frozen_signal = bool(main_edge["support_auc"] >= 0.65 and main_edge["direct_counterfactual_dominance_auc"] >= 0.70)
    preservation = bool(
        M["omcer"]["harmful"] <= M["raw"]["harmful"] + 0.005
        and M["omcer"]["flip"] <= M["raw"]["flip"] + 0.01
        and selected_path_safe
        and asymmetric
    )
    endpoint_noninferior = bool(
        M["omcer"]["match"] >= M["aggregate_downside"]["match"] - 0.002
        and M["omcer"]["regret"] <= M["aggregate_downside"]["regret"] * 1.005
        and M["omcer"]["regret"] <= M["raw"]["regret"] * 1.02
    )
    endpoint_signal = bool(
        M["omcer"]["match"] >= M["aggregate_downside"]["match"] + 0.002
        or M["omcer"]["regret"] < M["aggregate_downside"]["regret"] - 1e-6
    )

    full = bool(
        instrumentation and frozen_signal and structural_ok and asymmetric and selected_path_safe
        and coverage_vs_v25 and coverage_vs_lock and tail_vs_v25 and tail_vs_lock
        and preservation and endpoint_noninferior
    )
    if not instrumentation:
        nxt = "engineering_or_provenance_failure_stop_before_mechanism_interpretation"
    elif not frozen_signal:
        nxt = "frozen_frontier_support_or_dominance_regression_stop_OMCER"
    elif not structural_ok or not asymmetric:
        nxt = "deployment_or_incumbent_preservation_contract_failure_stop_OMCER"
    elif not selected_path_safe or not tail_vs_v25 or not tail_vs_lock:
        nxt = "OMCER_tail_failure_stop_no_K_cat_threshold_multiplier_or_zero_boundary_tuning"
    elif not coverage_vs_v25 or not coverage_vs_lock:
        nxt = "OMCER_safe_but_no_required_coverage_gain_stop_no_operator_margin_feature_combo_search"
    elif not endpoint_noninferior:
        nxt = "OMCER_mechanism_gain_without_endpoint_noninferiority_audit_final_guard_mediation"
    else:
        nxt = "split_pass_freeze_OMCER_wait_for_second_independent_block"

    report = {
        "audit": "v64_3_31_eaf_icer_omcer_split",
        "split_name": args.split_name,
        "full_split_pass": full,
        "instrumentation_valid": instrumentation,
        "frozen_support_dominance_signal": frozen_signal,
        "deployment_alignment": structural_ok,
        "incumbent_default_invariant": asymmetric,
        "selected_replacement_path_nonharmful": selected_path_safe,
        "selected_replacement_catastrophe_free": catastrophe_free,
        "safe_recovery_coverage_gain_over_V25_DRC": coverage_vs_v25,
        "safe_recovery_coverage_gain_over_proposal_lock_control": coverage_vs_lock,
        "selected_tail_noninferior_to_V25_DRC": tail_vs_v25,
        "selected_tail_noninferior_to_proposal_lock_control": tail_vs_lock,
        "endpoint_noninferior_to_V25_DRC": endpoint_noninferior,
        "endpoint_strict_signal_over_V25_DRC": endpoint_signal,
        "next_action": nxt,
        "structural": structural,
        "edge_diagnostics": edge,
        "path_diagnostics": path,
        "selected_replacement_tail_diagnostics": tails,
        "positive_direct_replacements": {
            "aggregate_downside": v25_pos,
            "lock_downside": lock_pos,
            "omcer": main_pos,
        },
        "metrics": M,
        "thresholds": {
            "B": 16, "M": 24, "K": [32, 64], "decision_boundary": 0.0,
            "catastrophic_delta_threshold": CAT_THRESHOLD,
            "capture_gain_min_pp": 3.0, "positive_replacement_gain_min": 5,
            "no_pooled_rescue": True,
        },
        "interpretation": (
            "V31 holds the B=16/M=24 evidence representation fixed and tests whether conditioning local tail risk on the exact "
            "support/scalar eligibility margin, then applying that risk before extremal selection, improves safe recovery coverage. "
            "The proposal-lock arm isolates the V30 post-extremal confirm/veto placement; V25 isolates the historical evidence-only downside certificate."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
