from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _pick(row: dict[str, Any], key: str) -> float:
    return _finite(row.get(key))


def _delta(a: float, b: float) -> float:
    return b - a if math.isfinite(a) and math.isfinite(b) else float("nan")


def _summary(row: dict[str, Any]) -> dict[str, float]:
    return {
        "teacher": _pick(row, "val_teacher_action_match"),
        "teacher_regret": _pick(row, "val_teacher_regret"),
        "pairfull": _pick(row, "val_pair_full_interface_action_match"),
        "critical_topm": _pick(row, "val_teacher_exact_winner_flip_critical_recall_topm_micro"),
        "critical_selected": _pick(row, "val_teacher_exact_winner_flip_critical_recall_selected_micro"),
        "proposal_decisive": _pick(row, "val_proposal_decisive_atom_recall"),
        "certificate": _pick(row, "val_evidence_certificate_fraction"),
        # C1-M/C2-M: exact-HAB proposal layer.
        "c2_topm_capture": _pick(row, "val_bdmu_current_topm_utility_capture"),
        "c1_hab_oracle_capture": _pick(row, "val_bdmu_hab_oracle_topm_utility_capture"),
        # C1-B/C2-B: same fixed B=16 downstream selector, differing only in
        # learned-vs-utility-oracle exact HAB proposal support.
        "c2_budget_capture": _pick(row, "val_bdmu_current_budget_utility_capture"),
        "c1_budget_oracle_capture": _pick(row, "val_bdmu_oracle_budget_utility_capture"),
        "budget_transmission_gap": _pick(row, "val_bdmu_budget_transmission_gap"),
        "fixed_reference_capture": _pick(row, "val_bdmu_reference_selected_utility_capture"),
        "btp_rank_loss": _pick(row, "val_bdmu_budget_transmission_rank_loss"),
        "btp_pairs": _pick(row, "val_bdmu_budget_transmission_pairs"),
        "btp_scene_fraction": _pick(row, "val_bdmu_budget_transmission_scene_fraction"),
        "btp_positive_fraction": _pick(row, "val_bdmu_budget_transmission_positive_fraction"),
        "protected_negative_fraction": _pick(row, "val_bdmu_budget_protected_negative_fraction"),
        "budget_projection_exact_fraction": _pick(row, "val_bdmu_budget_projection_exact_fraction"),
        "budget_projection_topm_violation_fraction": _pick(row, "val_bdmu_budget_projection_topm_violation_fraction"),
        "budget_selector_surrogate_jaccard_current": _pick(row, "val_bdmu_budget_selector_surrogate_jaccard_current"),
        "budget_selector_surrogate_jaccard_oracle": _pick(row, "val_bdmu_budget_selector_surrogate_jaccard_oracle"),
        "exact_candidate_scene_fraction": _pick(row, "val_bdmu_budget_exact_candidate_scene_fraction"),
        "current_oracle_budget_jaccard": _pick(row, "val_bdmu_budget_current_oracle_jaccard"),
        "controlled_exchange_negative_fraction": _pick(row, "val_bdmu_budget_controlled_exchange_negative_fraction"),
        "controlled_exchange_pair_fraction": _pick(row, "val_bdmu_budget_controlled_exchange_pair_fraction"),
        # Training-only instrumentation proving the gradient target itself used
        # sampled exact-runtime B projection rather than the V64.3.11 surrogate.
        "train_budget_projection_exact_fraction": _pick(row, "bdmu_budget_projection_exact_fraction"),
        "train_exact_candidate_scene_fraction": _pick(row, "bdmu_budget_exact_candidate_scene_fraction"),
        "train_selector_surrogate_jaccard_current": _pick(row, "bdmu_budget_selector_surrogate_jaccard_current"),
        "train_selector_surrogate_jaccard_oracle": _pick(row, "bdmu_budget_selector_surrogate_jaccard_oracle"),
        "train_controlled_exchange_pair_fraction": _pick(row, "bdmu_budget_controlled_exchange_pair_fraction"),
        "runtime_topm_exact_fraction": _pick(row, "val_bdmu_runtime_topm_exact_fraction"),
        "adapter_delta_rms": _pick(row, "critical_adapter_parameter_delta_rms"),
        "adapter_residual_rms": _pick(row, "critical_proposal_residual_rms"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit V64.3.12 RET/CET-BDMU exact budget-transmission screen")
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-budget-oracle-gap", type=float, default=0.005)
    ap.add_argument("--min-budget-capture-gain", type=float, default=0.005)
    ap.add_argument("--min-budget-gap-closure", type=float, default=0.15)
    ap.add_argument("--min-teacher-gain", type=float, default=0.005)
    ap.add_argument("--min-regret-relative-gain", type=float, default=0.02)
    ap.add_argument("--teacher-nonworse-tol", type=float, default=0.004)
    ap.add_argument("--regret-nonworse-relative-tol", type=float, default=0.02)
    ap.add_argument("--proposal-decisive-nonworse-tol", type=float, default=0.010)
    ap.add_argument("--critical-topm-nonworse-tol", type=float, default=0.005)
    ap.add_argument("--critical-selected-nonworse-tol", type=float, default=0.005)
    ap.add_argument("--topm-capture-nonworse-tol", type=float, default=0.005)
    ap.add_argument("--variant", choices=["ret", "cet"], default="cet")
    ap.add_argument("--min-train-exact-fraction", type=float, default=1.0e-4)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.train_log).read_text(encoding="utf-8").splitlines() if x.strip()]
    anchors = [r for r in rows if int(r.get("epoch", 999999)) < 0]
    trained = [r for r in rows if int(r.get("epoch", -999999)) >= 0]
    if not anchors or not trained:
        raise SystemExit("RET/CET-BDMU screen requires epoch<0 anchor and trained epochs")
    anchor_row = anchors[-1]
    anchor = _summary(anchor_row)
    required = ["c2_topm_capture", "c1_hab_oracle_capture", "c2_budget_capture", "c1_budget_oracle_capture"]
    missing = [k for k in required if not math.isfinite(anchor[k])]
    if missing:
        raise SystemExit(
            "RET/CET-BDMU requires VAL_MODE=both with C1-M/C2-M/C1-B/C2-B fixed-reference metrics; "
            f"missing anchor metrics={missing}"
        )

    anchor_m_gap = max(anchor["c1_hab_oracle_capture"] - anchor["c2_topm_capture"], 0.0)
    anchor_b_gap = max(anchor["c1_budget_oracle_capture"] - anchor["c2_budget_capture"], 0.0)
    acquisition_capacity_available = anchor_b_gap >= args.min_budget_oracle_gap

    audited: list[dict[str, Any]] = []
    for row in trained:
        cur = _summary(row)
        deltas = {k: _delta(anchor[k], cur[k]) for k in anchor}
        budget_gain = deltas["c2_budget_capture"]
        topm_gain = deltas["c2_topm_capture"]
        closure = float("nan")
        if anchor_b_gap > 1e-9 and math.isfinite(budget_gain):
            closure = budget_gain / anchor_b_gap

        mechanism_nonharm = bool(
            math.isfinite(deltas["proposal_decisive"])
            and deltas["proposal_decisive"] >= -args.proposal_decisive_nonworse_tol
            and math.isfinite(deltas["critical_topm"])
            and deltas["critical_topm"] >= -args.critical_topm_nonworse_tol
            and math.isfinite(deltas["critical_selected"])
            and deltas["critical_selected"] >= -args.critical_selected_nonworse_tol
            and math.isfinite(topm_gain)
            and topm_gain >= -args.topm_capture_nonworse_tol
        )
        mechanism_gain = bool(
            acquisition_capacity_available
            and math.isfinite(budget_gain)
            and budget_gain >= args.min_budget_capture_gain
            and math.isfinite(closure)
            and closure >= args.min_budget_gap_closure
            and mechanism_nonharm
        )

        regret_rel = float("nan")
        if math.isfinite(anchor["teacher_regret"]) and abs(anchor["teacher_regret"]) > 1e-9 and math.isfinite(cur["teacher_regret"]):
            regret_rel = (anchor["teacher_regret"] - cur["teacher_regret"]) / abs(anchor["teacher_regret"])
        deployment_gain = bool(
            (math.isfinite(deltas["teacher"]) and deltas["teacher"] >= args.min_teacher_gain)
            or (math.isfinite(regret_rel) and regret_rel >= args.min_regret_relative_gain)
        )
        teacher_nonworse = math.isfinite(deltas["teacher"]) and deltas["teacher"] >= -args.teacher_nonworse_tol
        regret_nonworse = bool(
            math.isfinite(cur["teacher_regret"])
            and math.isfinite(anchor["teacher_regret"])
            and cur["teacher_regret"] <= anchor["teacher_regret"] * (1.0 + args.regret_nonworse_relative_tol)
        )
        instrumentation = bool(
            math.isfinite(cur["adapter_delta_rms"]) and cur["adapter_delta_rms"] > 1e-7
            and math.isfinite(cur["btp_scene_fraction"]) and cur["btp_scene_fraction"] >= 0.0
            and math.isfinite(cur["btp_pairs"])
            and math.isfinite(cur["c1_budget_oracle_capture"])
            and math.isfinite(cur["c2_budget_capture"])
            and math.isfinite(cur["budget_projection_exact_fraction"])
            and cur["budget_projection_exact_fraction"] >= 0.999
            and math.isfinite(cur["budget_projection_topm_violation_fraction"])
            and cur["budget_projection_topm_violation_fraction"] <= 1.0e-9
            and math.isfinite(cur["train_budget_projection_exact_fraction"])
            and cur["train_budget_projection_exact_fraction"] >= args.min_train_exact_fraction
            and math.isfinite(cur["train_exact_candidate_scene_fraction"])
            and cur["train_exact_candidate_scene_fraction"] > 0.0
            and math.isfinite(cur["train_selector_surrogate_jaccard_current"])
            and math.isfinite(cur["train_selector_surrogate_jaccard_oracle"])
            and (not math.isfinite(cur["runtime_topm_exact_fraction"]) or cur["runtime_topm_exact_fraction"] >= 0.999)
        )
        capacity_not_binding = bool(instrumentation and not acquisition_capacity_available)
        transmission_learned_endpoint_blocked = bool(instrumentation and mechanism_gain and not deployment_gain)
        exact_acquisition_exhausted = bool(
            args.variant == "cet" and instrumentation and acquisition_capacity_available and not mechanism_gain
        )
        pivot_to_value = bool(capacity_not_binding or transmission_learned_endpoint_blocked or exact_acquisition_exhausted)
        promotion = bool(
            instrumentation and mechanism_gain and deployment_gain and teacher_nonworse and regret_nonworse
        )
        audited.append({
            "epoch": int(row["epoch"]),
            "selected": cur,
            "deltas": deltas,
            "anchor_hab_oracle_gap": anchor_m_gap,
            "anchor_budget_oracle_gap": anchor_b_gap,
            "budget_capture_gain": budget_gain,
            "topm_capture_gain": topm_gain,
            "budget_oracle_gap_closure": closure,
            "teacher_regret_relative_gain": regret_rel,
            "instrumentation_valid": instrumentation,
            "acquisition_capacity_available": acquisition_capacity_available,
            "mechanism_nonharm": mechanism_nonharm,
            "mechanism_gain": mechanism_gain,
            "deployment_gain": deployment_gain,
            "teacher_nonworse": teacher_nonworse,
            "teacher_regret_nonworse": regret_nonworse,
            "full_promotion": promotion,
            "pivot_to_value_frontier": pivot_to_value,
            "acquisition_capacity_not_binding": capacity_not_binding,
            "transmission_learned_endpoint_blocked": transmission_learned_endpoint_blocked,
            "exact_acquisition_exhausted": exact_acquisition_exhausted,
        })

    def mechanism_score(r: dict[str, Any]) -> tuple[float, float, float, float, int]:
        return (
            float(bool(r["mechanism_gain"])),
            r["budget_oracle_gap_closure"] if math.isfinite(r["budget_oracle_gap_closure"]) else -1e9,
            r["budget_capture_gain"] if math.isfinite(r["budget_capture_gain"]) else -1e9,
            r["selected"]["proposal_decisive"] if math.isfinite(r["selected"]["proposal_decisive"]) else -1e9,
            -int(r["epoch"]),
        )

    def endpoint_score(r: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            float(bool(r["deployment_gain"])),
            r["selected"]["teacher"] if math.isfinite(r["selected"]["teacher"]) else -1e9,
            -(r["selected"]["teacher_regret"] if math.isfinite(r["selected"]["teacher_regret"]) else 1e18),
            -int(r["epoch"]),
        )

    valid = [r for r in audited if r["instrumentation_valid"]] or audited
    best_mech = max(valid, key=mechanism_score)
    best_endpoint = max(valid, key=endpoint_score)
    promoted = [r for r in valid if r["full_promotion"]]
    best = max(promoted, key=mechanism_score) if promoted else best_mech

    if best["acquisition_capacity_not_binding"]:
        diagnosis = "C1-B leaves < threshold budget-transmission headroom; acquisition is no longer the binding interface error. Pivot to value/frontier."
    elif best["mechanism_gain"] and not best["deployment_gain"]:
        diagnosis = "Exact-trained C2-B closes the B=16 oracle gap without endpoint gain; acquisition is causally cleared. Pivot to decisive value/frontier."
    elif not best["mechanism_gain"] and args.variant == "ret":
        diagnosis = "RET exact training did not move C2-B despite C1-B headroom. Run CET to test whether blanket current-B protection is the remaining transmission constraint."
    elif not best["mechanism_gain"]:
        diagnosis = "CET exact-runtime supervision plus controlled B-set exchange still did not move C2-B. This proposal-only acquisition branch is exhausted; pivot to decisive value/frontier rather than inventing another acquisition proxy."
    else:
        diagnosis = "RET/CET moved the exact budget-transmitted acquisition mediator and downstream endpoint."

    report = {
        "audit": "v64_3_12_ret_cet_bdmu_screen",
        "audit_version": "v64.3.12.0",
        "variant": args.variant,
        "objective": "fixed BDMU target -> exact HAB projection -> sampled exact-runtime B=16 training target -> slack/controlled budget exchange -> frozen DARM/DBR endpoint",
        "anchor_epoch": int(anchor_row.get("epoch", -1)),
        "anchor": anchor,
        "anchor_hab_oracle_gap": anchor_m_gap,
        "anchor_budget_oracle_gap": anchor_b_gap,
        "acquisition_capacity_available": acquisition_capacity_available,
        "epochs": audited,
        "selected_epoch": best["epoch"],
        "selected": best["selected"],
        "deltas": best["deltas"],
        "budget_oracle_gap_closure": best["budget_oracle_gap_closure"],
        "teacher_regret_relative_gain": best["teacher_regret_relative_gain"],
        "instrumentation_valid": best["instrumentation_valid"],
        "mechanism_nonharm": best["mechanism_nonharm"],
        "mechanism_gain": best["mechanism_gain"],
        "deployment_gain": best["deployment_gain"],
        "full_promotion": best["full_promotion"],
        "pivot_to_value_frontier": best["pivot_to_value_frontier"],
        "acquisition_capacity_not_binding": best["acquisition_capacity_not_binding"],
        "exact_acquisition_exhausted": best["exact_acquisition_exhausted"],
        "best_mechanism_epoch": best_mech["epoch"],
        "best_endpoint_epoch": best_endpoint["epoch"],
        "mechanism_endpoint_concordance": best_mech["epoch"] == best_endpoint["epoch"],
        "diagnosis": diagnosis,
        "thresholds": {
            "min_budget_oracle_gap": args.min_budget_oracle_gap,
            "min_budget_capture_gain": args.min_budget_capture_gain,
            "min_budget_gap_closure": args.min_budget_gap_closure,
            "min_teacher_gain": args.min_teacher_gain,
            "min_regret_relative_gain": args.min_regret_relative_gain,
            "teacher_nonworse_tolerance": args.teacher_nonworse_tol,
            "regret_nonworse_relative_tolerance": args.regret_nonworse_relative_tol,
            "proposal_decisive_nonworse_tolerance": args.proposal_decisive_nonworse_tol,
            "critical_topm_nonworse_tolerance": args.critical_topm_nonworse_tol,
            "critical_selected_nonworse_tolerance": args.critical_selected_nonworse_tol,
            "topm_capture_nonworse_tolerance": args.topm_capture_nonworse_tol,
        },
        "causal_protocol": {
            "C0_fixed_target": "same frozen-foundation decisive-margin utility target as training",
            "C1_M_HAB_ceiling": "val_bdmu_hab_oracle_topm_utility_capture",
            "C2_M_learned_HAB": "val_bdmu_current_topm_utility_capture",
            "C1_B_budget_transmission_ceiling": "val_bdmu_oracle_budget_utility_capture: exact-HAB utility oracle passed through the exact runtime B=16 pair-conditioned selector",
            "C2_B_learned_budget_transmission": "val_bdmu_current_budget_utility_capture: learned exact-HAB Top-M passed through the same exact runtime B=16 selector",
            "train_eval_selector_alignment": "training uses sampled stop-gradient exact runtime B projection on actionable scenes; validation uses exact projection on every scene. Surrogate masks are diagnostics only and their Jaccard is logged, not optimized.",
            "controlled_exchange": "CET may displace current B evidence only when the exact oracle-B intervention drops that atom; RET keeps blanket current-B protection.",
            "C3_endpoint": "teacher match/regret with B, DARM, DBR and foundation value frozen",
            "stop_rule": "RET failure proceeds once to CET. CET failure with valid exact training, negligible C1-B headroom, or C2-B gain without C3 all terminate acquisition tuning and pivot to decisive value/frontier.",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    keys = [
        "selected_epoch", "instrumentation_valid", "acquisition_capacity_available", "mechanism_gain",
        "deployment_gain", "full_promotion", "pivot_to_value_frontier", "acquisition_capacity_not_binding",
        "exact_acquisition_exhausted", "best_mechanism_epoch", "best_endpoint_epoch", "mechanism_endpoint_concordance", "diagnosis",
    ]
    print(json.dumps({k: report[k] for k in keys}, indent=2))
    return 0 if report["instrumentation_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
