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
    # The val_bdmu_* metrics come from val-mode=both and therefore use exactly
    # the same frozen-foundation BDMU reference target as training.  They are the
    # causal mechanism metrics.  val_teacher_bdmu_* is retained only as a
    # deployment-side descriptive diagnostic because its reference B-set is
    # endogenous to the current checkpoint.
    return {
        "teacher": _pick(row, "val_teacher_action_match"),
        "teacher_regret": _pick(row, "val_teacher_regret"),
        "pairfull": _pick(row, "val_pair_full_interface_action_match"),
        "critical_topm": _pick(row, "val_teacher_exact_winner_flip_critical_recall_topm_micro"),
        "critical_selected": _pick(row, "val_teacher_exact_winner_flip_critical_recall_selected_micro"),
        "proposal_decisive": _pick(row, "val_proposal_decisive_atom_recall"),
        "certificate": _pick(row, "val_evidence_certificate_fraction"),
        "fixed_topm_capture": _pick(row, "val_bdmu_current_topm_utility_capture"),
        "fixed_reference_capture": _pick(row, "val_bdmu_reference_selected_utility_capture"),
        "hab_oracle_capture": _pick(row, "val_bdmu_hab_oracle_topm_utility_capture"),
        "hab_oracle_gap": _pick(row, "val_bdmu_hab_oracle_gap"),
        "feasible_rank_loss": _pick(row, "val_bdmu_feasible_admission_rank_loss"),
        "feasible_rank_pairs": _pick(row, "val_bdmu_feasible_admission_pairs"),
        "feasible_same_family_pair_fraction": _pick(row, "val_bdmu_feasible_admission_same_family_pair_fraction"),
        "feasible_rank_scene_fraction": _pick(row, "val_bdmu_feasible_admission_scene_fraction"),
        "runtime_topm_exact_fraction": _pick(row, "val_bdmu_runtime_topm_exact_fraction"),
        "moving_topm_capture": _pick(row, "val_teacher_bdmu_topm_utility_capture"),
        "adapter_delta_rms": _pick(row, "critical_adapter_parameter_delta_rms"),
        "adapter_residual_rms": _pick(row, "critical_proposal_residual_rms"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit V64.3.10 HAP-BDMU HAB-feasible acquisition screen")
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-fixed-topm-gain", type=float, default=0.010)
    ap.add_argument("--min-oracle-gap", type=float, default=0.010)
    ap.add_argument("--min-oracle-gap-closure", type=float, default=0.10)
    ap.add_argument("--min-teacher-gain", type=float, default=0.005)
    ap.add_argument("--min-regret-relative-gain", type=float, default=0.02)
    ap.add_argument("--teacher-nonworse-tol", type=float, default=0.004)
    ap.add_argument("--regret-nonworse-relative-tol", type=float, default=0.02)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.train_log).read_text(encoding="utf-8").splitlines() if x.strip()]
    anchors = [r for r in rows if int(r.get("epoch", 999999)) < 0]
    trained = [r for r in rows if int(r.get("epoch", -999999)) >= 0]
    if not anchors or not trained:
        raise SystemExit("HAP-BDMU screen requires epoch<0 anchor and trained epochs")
    anchor_row = anchors[-1]
    anchor = _summary(anchor_row)
    required = ["fixed_topm_capture", "hab_oracle_capture", "hab_oracle_gap"]
    missing = [k for k in required if not math.isfinite(anchor[k])]
    if missing:
        raise SystemExit(
            "HAP-BDMU requires VAL_MODE=both so fixed-reference val-loss metrics are logged; "
            f"missing anchor metrics={missing}"
        )

    anchor_oracle_gap = max(anchor["hab_oracle_capture"] - anchor["fixed_topm_capture"], 0.0)
    acquisition_capacity_available = anchor_oracle_gap >= args.min_oracle_gap
    audited: list[dict[str, Any]] = []
    for row in trained:
        cur = _summary(row)
        deltas = {k: _delta(anchor[k], cur[k]) for k in anchor}
        fixed_gain = deltas["fixed_topm_capture"]
        closure = float("nan")
        if anchor_oracle_gap > 1e-9 and math.isfinite(fixed_gain):
            closure = fixed_gain / anchor_oracle_gap
        mechanism_gain = bool(
            acquisition_capacity_available
            and math.isfinite(fixed_gain)
            and fixed_gain >= args.min_fixed_topm_gain
            and math.isfinite(closure)
            and closure >= args.min_oracle_gap_closure
        )
        regret_rel = float("nan")
        if math.isfinite(anchor["teacher_regret"]) and abs(anchor["teacher_regret"]) > 1e-9 and math.isfinite(cur["teacher_regret"]):
            regret_rel = (anchor["teacher_regret"] - cur["teacher_regret"]) / abs(anchor["teacher_regret"])
        deployment_gain = bool(
            (math.isfinite(deltas["teacher"]) and deltas["teacher"] >= args.min_teacher_gain)
            or (math.isfinite(regret_rel) and regret_rel >= args.min_regret_relative_gain)
        )
        teacher_nonworse = math.isfinite(deltas["teacher"]) and deltas["teacher"] >= -args.teacher_nonworse_tol
        regret_nonworse = (
            math.isfinite(cur["teacher_regret"]) and math.isfinite(anchor["teacher_regret"])
            and cur["teacher_regret"] <= anchor["teacher_regret"] * (1.0 + args.regret_nonworse_relative_tol)
        )
        instrumentation = bool(
            math.isfinite(cur["adapter_delta_rms"]) and cur["adapter_delta_rms"] > 1e-7
            and math.isfinite(cur["feasible_rank_scene_fraction"]) and cur["feasible_rank_scene_fraction"] > 0.0
            and math.isfinite(cur["hab_oracle_capture"])
            and math.isfinite(cur["fixed_topm_capture"])
            and (not math.isfinite(cur["runtime_topm_exact_fraction"]) or cur["runtime_topm_exact_fraction"] >= 0.999)
        )
        promotion = bool(instrumentation and mechanism_gain and deployment_gain and teacher_nonworse and regret_nonworse)
        pivot_to_value = bool(instrumentation and mechanism_gain and not deployment_gain)
        capacity_not_binding = bool(instrumentation and not acquisition_capacity_available)
        audited.append({
            "epoch": int(row["epoch"]),
            "selected": cur,
            "deltas": deltas,
            "fixed_topm_gain": fixed_gain,
            "anchor_hab_oracle_gap": anchor_oracle_gap,
            "oracle_gap_closure": closure,
            "teacher_regret_relative_gain": regret_rel,
            "instrumentation_valid": instrumentation,
            "acquisition_capacity_available": acquisition_capacity_available,
            "mechanism_gain": mechanism_gain,
            "deployment_gain": deployment_gain,
            "teacher_nonworse": teacher_nonworse,
            "teacher_regret_nonworse": regret_nonworse,
            "full_promotion": promotion,
            "pivot_to_value_frontier": pivot_to_value,
            "acquisition_capacity_not_binding": capacity_not_binding,
        })

    def score(r: dict[str, Any]) -> tuple[float, float, float, float, int]:
        return (
            float(bool(r["full_promotion"])),
            r["oracle_gap_closure"] if math.isfinite(r["oracle_gap_closure"]) else -1e9,
            r["selected"]["teacher"] if math.isfinite(r["selected"]["teacher"]) else -1e9,
            -(r["selected"]["teacher_regret"] if math.isfinite(r["selected"]["teacher_regret"]) else 1e18),
            -int(r["epoch"]),
        )

    valid = [r for r in audited if r["instrumentation_valid"]] or audited
    best = max(valid, key=score)
    if best["acquisition_capacity_not_binding"]:
        diagnosis = "HAB-feasible utility oracle leaves too little acquisition headroom; do not tune proposal ranking."
    elif best["mechanism_gain"] and not best["deployment_gain"]:
        diagnosis = "HAB-feasible acquisition mechanism moved but teacher endpoint did not; pivot to value/frontier."
    elif not best["mechanism_gain"]:
        diagnosis = "Structured feasible-admission training did not close the fixed-reference HAB utility gap."
    else:
        diagnosis = "HAP-BDMU mechanism and deployment endpoint both improved."

    report = {
        "audit": "v64_3_10_hap_bdmu_screen",
        "audit_version": "v64.3.10.0",
        "objective": "fixed-reference decisive-margin utility -> exact HAB-feasible utility projection -> structured hard Top-M admission",
        "anchor_epoch": int(anchor_row.get("epoch", -1)),
        "anchor": anchor,
        "anchor_hab_oracle_gap": anchor_oracle_gap,
        "acquisition_capacity_available": acquisition_capacity_available,
        "epochs": audited,
        "selected_epoch": best["epoch"],
        "selected": best["selected"],
        "deltas": best["deltas"],
        "oracle_gap_closure": best["oracle_gap_closure"],
        "teacher_regret_relative_gain": best["teacher_regret_relative_gain"],
        "instrumentation_valid": best["instrumentation_valid"],
        "mechanism_gain": best["mechanism_gain"],
        "deployment_gain": best["deployment_gain"],
        "full_promotion": best["full_promotion"],
        "pivot_to_value_frontier": best["pivot_to_value_frontier"],
        "acquisition_capacity_not_binding": best["acquisition_capacity_not_binding"],
        "diagnosis": diagnosis,
        "thresholds": {
            "min_fixed_topm_gain": args.min_fixed_topm_gain,
            "min_oracle_gap": args.min_oracle_gap,
            "min_oracle_gap_closure": args.min_oracle_gap_closure,
            "min_teacher_gain": args.min_teacher_gain,
            "min_regret_relative_gain": args.min_regret_relative_gain,
            "teacher_nonworse_tolerance": args.teacher_nonworse_tol,
            "regret_nonworse_relative_tolerance": args.regret_nonworse_relative_tol,
        },
        "causal_protocol": {
            "C0_fixed_target": "val_bdmu_* uses the frozen-foundation reference B-set from the training objective",
            "C1_interface_ceiling": "val_bdmu_hab_oracle_topm_utility_capture projects the same utility through the exact frozen-family runtime HAB policy",
            "C2_learned_admission": "val_bdmu_current_topm_utility_capture measures the learned proposal under the same target/interface",
            "C3_downstream_endpoint": "teacher match/regret is evaluated with B, DARM and DBR frozen",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in [
        "selected_epoch", "instrumentation_valid", "acquisition_capacity_available",
        "mechanism_gain", "deployment_gain", "full_promotion", "pivot_to_value_frontier",
        "acquisition_capacity_not_binding", "diagnosis"
    ]}, indent=2))
    return 0 if report["instrumentation_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
