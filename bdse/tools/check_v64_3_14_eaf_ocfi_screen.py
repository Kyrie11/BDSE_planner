from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _get(d: dict[str, Any], key: str) -> float:
    return _f(d.get(key))


def _audit_variant(metrics: dict[str, Any], cal: dict[str, Any]) -> dict[str, Any]:
    raw = cal.get("raw_eval_subset_metrics", {}) or {}
    norm = str(cal.get("normalization", ""))
    q = _f(cal.get("calibration_quantile"))
    teacher0, teacher1 = _get(raw, "teacher_action_match"), _get(metrics, "teacher_action_match")
    regret0, regret1 = _get(raw, "teacher_regret"), _get(metrics, "teacher_regret")
    harm0, harm1 = _get(raw, "harmful_pair_potential_intervention_rate"), _get(metrics, "harmful_pair_potential_intervention_rate")
    benefit0, benefit1 = _get(raw, "beneficial_pair_potential_intervention_rate"), _get(metrics, "beneficial_pair_potential_intervention_rate")
    flip0, flip1 = _get(raw, "pair_potential_deployed_flip_rate"), _get(metrics, "pair_potential_deployed_flip_rate")
    anchor0, anchor1 = _get(raw, "selected_local_anchor_action_match"), _get(metrics, "selected_local_anchor_action_match")
    pairfull0, pairfull1 = _get(raw, "pair_full_interface_action_match"), _get(metrics, "pair_full_interface_action_match")
    localpair0, localpair1 = _get(raw, "local_pair_full_interface_action_match"), _get(metrics, "local_pair_full_interface_action_match")
    cert0, cert1 = _get(raw, "evidence_certificate_fraction"), _get(metrics, "evidence_certificate_fraction")

    teacher_delta = teacher1 - teacher0 if math.isfinite(teacher0) and math.isfinite(teacher1) else float("nan")
    regret_gain = (regret0 - regret1) / abs(regret0) if math.isfinite(regret0) and abs(regret0) > 1e-9 and math.isfinite(regret1) else float("nan")
    harm_reduction = harm0 - harm1 if math.isfinite(harm0) and math.isfinite(harm1) else float("nan")
    benefit_retention = benefit1 / benefit0 if math.isfinite(benefit0) and benefit0 > 1e-9 and math.isfinite(benefit1) else float("nan")
    flip_reduction = flip0 - flip1 if math.isfinite(flip0) and math.isfinite(flip1) else float("nan")

    calibration_valid = bool(
        int(cal.get("calibration_proposal_edge_count", 0)) >= 32
        and int(cal.get("calibration_group_count", 0)) > 0
        and int(cal.get("evaluation_group_count", 0)) > 0
        and math.isfinite(q) and q >= 0.0
    )
    runtime_instrumentation = bool(
        _get(metrics, "decisive_frontier_value_active") >= 0.99
        and _get(metrics, "decisive_frontier_value_complete_star_coverage") >= 0.99
        and _get(metrics, "decisive_frontier_ocfi_active") >= 0.99
        and math.isfinite(_get(metrics, "decisive_frontier_ocfi_calibration_radius"))
        and _get(metrics, "decisive_frontier_ocfi_calibration_radius") >= 0.0
        and math.isfinite(_get(metrics, "decisive_frontier_ocfi_calibration_quantile"))
    )
    frozen_interface = bool(
        math.isfinite(anchor0) and math.isfinite(anchor1) and abs(anchor1 - anchor0) <= 1e-6
        and (not math.isfinite(pairfull0) or not math.isfinite(pairfull1) or abs(pairfull1 - pairfull0) <= 1e-6)
        and (not math.isfinite(localpair0) or not math.isfinite(localpair1) or abs(localpair1 - localpair0) <= 1e-6)
        and (not math.isfinite(cert0) or not math.isfinite(cert1) or abs(cert1 - cert0) <= 1e-6)
    )
    preservation_gain = bool(
        math.isfinite(harm_reduction) and harm_reduction >= 0.01
        and (not math.isfinite(benefit_retention) or benefit_retention >= 0.50)
        and (not math.isfinite(flip_reduction) or flip_reduction > 0.0)
    )
    teacher_nonharm = math.isfinite(teacher_delta) and teacher_delta >= -0.004
    regret_nonharm = math.isfinite(regret_gain) and regret_gain >= -0.01
    endpoint_gain = bool(
        (math.isfinite(teacher_delta) and teacher_delta >= 0.005 and regret_nonharm)
        or (math.isfinite(regret_gain) and regret_gain >= 0.02 and teacher_nonharm)
    )
    full_promotion = bool(calibration_valid and runtime_instrumentation and frozen_interface and preservation_gain and endpoint_gain)

    if not calibration_valid:
        next_action = "repair_ocfi_split_calibration"
    elif not runtime_instrumentation:
        next_action = "repair_ocfi_runtime_instrumentation"
    elif not frozen_interface:
        next_action = "repair_ocfi_causal_isolation"
    elif not preservation_gain:
        next_action = "ocfi_failed_to_reduce_harm_test_representation_capacity"
    elif not endpoint_gain:
        next_action = "preservation_fixed_but_value_not_decisive_test_selective_representation_capacity"
    else:
        next_action = "promote_ocfi_full_calibration"

    return {
        "normalization": norm,
        "calibration_valid": calibration_valid,
        "runtime_instrumentation_valid": runtime_instrumentation,
        "frozen_interface": frozen_interface,
        "preservation_gain": preservation_gain,
        "endpoint_gain": endpoint_gain,
        "full_promotion": full_promotion,
        "next_action": next_action,
        "raw_eval_subset_metrics": raw,
        "metrics": metrics,
        "deltas": {
            "teacher_action_match": teacher_delta,
            "teacher_regret_relative_gain": regret_gain,
            "harmful_intervention_reduction": harm_reduction,
            "beneficial_intervention_retention": benefit_retention,
            "deployed_flip_reduction": flip_reduction,
        },
        "calibration": {
            "alpha": _f(cal.get("alpha")),
            "quantile": q,
            "attribution_scale_floor": _f(cal.get("attribution_scale_floor")),
            "proposal_edge_count": int(cal.get("calibration_proposal_edge_count", 0)),
        },
    }


def build(attribution_metrics: dict[str, Any], attribution_cal: dict[str, Any], constant_metrics: dict[str, Any], constant_cal: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        _audit_variant(attribution_metrics, attribution_cal),
        _audit_variant(constant_metrics, constant_cal),
    ]

    def score(x: dict[str, Any]) -> tuple[float, ...]:
        d = x["deltas"]
        prefer_attr = 1.0 if x["normalization"] == "attribution" else 0.0
        vals = [
            float(x["full_promotion"]),
            float(x["preservation_gain"]),
            float(x["endpoint_gain"]),
            _f(d.get("teacher_action_match")),
            _f(d.get("teacher_regret_relative_gain")),
            _f(d.get("harmful_intervention_reduction")),
            prefer_attr,
        ]
        return tuple(v if math.isfinite(v) else -9.0 for v in vals)

    selected = max(candidates, key=score)
    attr = next(x for x in candidates if x["normalization"] == "attribution")
    const = next(x for x in candidates if x["normalization"] == "none")
    attr_teacher = _f(attr["deltas"]["teacher_action_match"])
    const_teacher = _f(const["deltas"]["teacher_action_match"])
    attr_regret = _f(attr["deltas"]["teacher_regret_relative_gain"])
    const_regret = _f(const["deltas"]["teacher_regret_relative_gain"])
    attr_harm = _f(attr["deltas"]["harmful_intervention_reduction"])
    const_harm = _f(const["deltas"]["harmful_intervention_reduction"])
    attribution_specific_gain = bool(
        attr["full_promotion"]
        and (
            not const["full_promotion"]
            or (
                # If both controls promote, parity is not evidence for the
                # evidence-attribution mechanism.  Require a measurable
                # attribution-specific advantage in at least one endpoint or
                # preservation axis while remaining non-inferior on teacher
                # match and harmful-intervention reduction.
                (
                    (math.isfinite(attr_teacher) and math.isfinite(const_teacher) and attr_teacher >= const_teacher + 0.003)
                    or (math.isfinite(attr_regret) and math.isfinite(const_regret) and attr_regret >= const_regret + 0.01)
                    or (math.isfinite(attr_harm) and math.isfinite(const_harm) and attr_harm >= const_harm + 0.01)
                )
                and (not math.isfinite(attr_teacher) or not math.isfinite(const_teacher) or attr_teacher >= const_teacher - 0.002)
                and (not math.isfinite(attr_harm) or not math.isfinite(const_harm) or attr_harm >= const_harm - 0.002)
            )
        )
    )

    if selected["full_promotion"] and selected["normalization"] == "attribution" and attribution_specific_gain:
        next_action = "promote_attribution_scaled_ocfi_to_full_val_then_test_cl"
    elif selected["full_promotion"] and selected["normalization"] == "attribution":
        next_action = "ocfi_works_but_attribution_specific_gain_not_established_do_not_overclaim"
    elif selected["full_promotion"]:
        next_action = "constant_ocfi_works_but_attribution_scaling_not_supported_do_not_overclaim"
    else:
        next_action = selected["next_action"]

    return {
        "audit": "v64_3_14_eaf_ocfi_screen",
        "selected_normalization": selected["normalization"],
        "full_promotion": selected["full_promotion"],
        "attribution_specific_gain": attribution_specific_gain,
        "next_action": next_action,
        "selected": selected,
        "candidates": candidates,
        "thresholds": {
            "harmful_intervention_absolute_reduction": 0.01,
            "beneficial_intervention_retention": 0.50,
            "teacher_match_gain": 0.005,
            "teacher_match_nonharm_tolerance": 0.004,
            "teacher_regret_relative_gain": 0.02,
            "teacher_regret_nonharm_tolerance": 0.01,
            "frozen_interface_tolerance": 1e-6,
            "attribution_specific_teacher_advantage": 0.003,
            "attribution_specific_regret_advantage": 0.01,
            "attribution_specific_harm_reduction_advantage": 0.01,
            "attribution_specific_noninferiority_tolerance": 0.002,
        },
        "interpretation": (
            "The constant-radius branch controls for generic post-hoc thresholding. The attribution-scaled branch "
            "is paper-facing only if it improves the one-sided preservation/endpoint trade-off without changing the "
            "selected-local anchor, pair-full ceiling, evidence certificate, B=16 budget, or M=24 proposal frontier."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution-metrics", type=Path, required=True)
    ap.add_argument("--attribution-calibration", type=Path, required=True)
    ap.add_argument("--constant-metrics", type=Path, required=True)
    ap.add_argument("--constant-calibration", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    r = build(load(args.attribution_metrics), load(args.attribution_calibration), load(args.constant_metrics), load(args.constant_calibration))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(r, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
