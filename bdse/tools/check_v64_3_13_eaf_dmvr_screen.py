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


def _summary(row: dict[str, Any]) -> dict[str, float]:
    keys = {
        "teacher": "val_teacher_action_match",
        "teacher_regret": "val_teacher_regret",
        "pairfull": "val_pair_full_interface_action_match",
        "localpair": "val_local_pair_full_interface_action_match",
        "proposal_decisive": "val_proposal_decisive_atom_recall",
        "critical_topm": "val_teacher_exact_winner_flip_critical_recall_topm_micro",
        "critical_selected": "val_teacher_exact_winner_flip_critical_recall_selected_micro",
        "certificate": "val_evidence_certificate_fraction",
        "base_winner_sign": "val_base_pair_sign_acc_winner_rival",
        "dense_winner_sign": "val_dense_pair_sign_acc_winner_rival",
        "legacy_pair_winner_sign": "val_pair_sign_acc_winner_rival",
        "frontier_pair_sign": "val_frontier_value_pair_sign_acc",
        "frontier_action_match": "val_frontier_value_action_match",
        "frontier_anchor_wrong": "val_frontier_value_anchor_wrong_fraction",
        "frontier_wrong_corrected": "val_frontier_value_wrong_anchor_corrected_fraction",
        "frontier_correct_preserved": "val_frontier_value_correct_anchor_preserved_fraction",
        "frontier_residual_rms": "val_frontier_value_residual_rms",
        "frontier_exact_scene_fraction": "val_frontier_value_exact_scene_fraction",
        "frontier_complete_star_coverage": "val_frontier_value_complete_star_coverage",
        "runtime_frontier_active": "val_decisive_frontier_value_active",
        "runtime_frontier_complete_star": "val_decisive_frontier_value_complete_star_coverage",
        "runtime_frontier_residual_rms": "val_decisive_frontier_value_residual_rms",
        "value_adapter_delta_rms": "frontier_value_adapter_parameter_delta_rms",
        "value_adapter_delta_max": "frontier_value_adapter_parameter_delta_max_abs",
        "critical_adapter_delta_rms": "critical_adapter_parameter_delta_rms",
        "dbr_adapter_delta_rms": "decisive_pair_adapter_parameter_delta_rms",
    }
    return {k: _f(row.get(v)) for k, v in keys.items()}


def _delta(a: float, b: float) -> float:
    return b - a if math.isfinite(a) and math.isfinite(b) else float("nan")


def build(rows: list[dict[str, Any]], variant: str = "SCREEN") -> dict[str, Any]:
    anchors = [r for r in rows if int(r.get("epoch", 99999)) < 0]
    trained = [r for r in rows if int(r.get("epoch", -99999)) >= 0]
    if not anchors or not trained:
        raise ValueError("EAF-DMVR audit requires an epoch<0 anchor and trained epochs")
    anchor_row = anchors[-1]
    anchor = _summary(anchor_row)
    required_anchor = ["teacher", "teacher_regret", "frontier_pair_sign", "frontier_action_match"]
    missing = [k for k in required_anchor if not math.isfinite(anchor[k])]
    if missing:
        raise ValueError(f"EAF-DMVR requires VAL_MODE=both and frontier-value metrics; missing anchor={missing}")

    audited: list[dict[str, Any]] = []
    for row in trained:
        cur = _summary(row)
        d = {k: _delta(anchor[k], cur[k]) for k in anchor}
        regret_rel = float("nan")
        if math.isfinite(anchor["teacher_regret"]) and abs(anchor["teacher_regret"]) > 1e-9 and math.isfinite(cur["teacher_regret"]):
            regret_rel = (anchor["teacher_regret"] - cur["teacher_regret"]) / abs(anchor["teacher_regret"])

        instrumentation = bool(
            math.isfinite(cur["value_adapter_delta_rms"]) and cur["value_adapter_delta_rms"] > 1e-7
            and math.isfinite(cur["frontier_residual_rms"]) and cur["frontier_residual_rms"] > 1e-6
            and math.isfinite(cur["frontier_exact_scene_fraction"]) and cur["frontier_exact_scene_fraction"] >= 0.999
            and math.isfinite(cur["frontier_complete_star_coverage"]) and cur["frontier_complete_star_coverage"] >= 0.999
            and (not math.isfinite(cur["runtime_frontier_active"]) or cur["runtime_frontier_active"] >= 0.99)
            and (not math.isfinite(cur["runtime_frontier_complete_star"]) or cur["runtime_frontier_complete_star"] >= 0.99)
        )
        acquisition_frozen = bool(
            (not math.isfinite(cur["critical_adapter_delta_rms"]) or abs(cur["critical_adapter_delta_rms"]) <= 1e-10)
            and (not math.isfinite(cur["dbr_adapter_delta_rms"]) or abs(cur["dbr_adapter_delta_rms"]) <= 1e-10)
            and (not math.isfinite(d["proposal_decisive"]) or abs(d["proposal_decisive"]) <= 1e-4)
            and (not math.isfinite(d["critical_topm"]) or abs(d["critical_topm"]) <= 1e-4)
            and (not math.isfinite(d["critical_selected"]) or abs(d["critical_selected"]) <= 1e-4)
        )
        correct_preservation = math.isfinite(cur["frontier_correct_preserved"]) and cur["frontier_correct_preserved"] >= 0.97
        mechanism_gain = bool(
            instrumentation
            and math.isfinite(d["frontier_pair_sign"]) and d["frontier_pair_sign"] >= 0.02
            and math.isfinite(d["frontier_action_match"]) and d["frontier_action_match"] >= 0.01
            and correct_preservation
        )
        teacher_nonharm = math.isfinite(d["teacher"]) and d["teacher"] >= -0.004
        regret_nonharm = math.isfinite(regret_rel) and regret_rel >= -0.01
        deployment_gain = bool(
            ((math.isfinite(d["teacher"]) and d["teacher"] >= 0.01) and regret_nonharm)
            or ((math.isfinite(regret_rel) and regret_rel >= 0.02) and teacher_nonharm)
        )
        full_promotion = bool(instrumentation and acquisition_frozen and mechanism_gain and deployment_gain)

        if not instrumentation:
            next_action = "repair_frontier_value_instrumentation"
        elif not acquisition_frozen:
            next_action = "repair_causal_isolation_do_not_interpret"
        elif not mechanism_gain:
            next_action = "selective_action_evidence_representation_capacity_test"
        elif not deployment_gain:
            next_action = "audit_frontier_to_final_preservation_guard_or_value_calibration"
        else:
            next_action = "promote_full"

        audited.append({
            "epoch": int(row.get("epoch", -1)),
            "current": cur,
            "deltas": d,
            "teacher_regret_relative_gain": regret_rel,
            "instrumentation_valid": instrumentation,
            "acquisition_frozen": acquisition_frozen,
            "correct_anchor_preservation": correct_preservation,
            "mechanism_gain": mechanism_gain,
            "deployment_gain": deployment_gain,
            "full_promotion": full_promotion,
            "next_action": next_action,
        })

    def score(x: dict[str, Any]) -> tuple[float, ...]:
        d=x["deltas"]
        return (
            float(x["full_promotion"]), float(x["mechanism_gain"]), float(x["deployment_gain"]),
            d.get("teacher", float("nan")) if math.isfinite(d.get("teacher", float("nan"))) else -9.0,
            x.get("teacher_regret_relative_gain", -9.0) if math.isfinite(x.get("teacher_regret_relative_gain", float("nan"))) else -9.0,
            d.get("frontier_action_match", -9.0) if math.isfinite(d.get("frontier_action_match", float("nan"))) else -9.0,
            d.get("frontier_pair_sign", -9.0) if math.isfinite(d.get("frontier_pair_sign", float("nan"))) else -9.0,
        )
    selected = max(audited, key=score)

    return {
        "audit": "v64_3_13_eaf_dmvr_screen",
        "variant": variant,
        "anchor_epoch": int(anchor_row.get("epoch", -1)),
        "selected_epoch": selected["epoch"],
        "anchor": anchor,
        "selected": selected["current"],
        "deltas": selected["deltas"],
        "teacher_regret_relative_gain": selected["teacher_regret_relative_gain"],
        "instrumentation_valid": selected["instrumentation_valid"],
        "acquisition_frozen": selected["acquisition_frozen"],
        "mechanism_gain": selected["mechanism_gain"],
        "deployment_gain": selected["deployment_gain"],
        "full_promotion": selected["full_promotion"],
        "next_action": selected["next_action"],
        "all_epochs": audited,
        "thresholds": {
            "frontier_pair_sign_gain": 0.02,
            "frontier_action_match_gain": 0.01,
            "correct_anchor_preserved_fraction": 0.97,
            "teacher_match_gain": 0.01,
            "teacher_match_nonharm_tolerance": 0.004,
            "teacher_regret_relative_gain": 0.02,
            "teacher_regret_relative_nonharm_tolerance": 0.01,
            "acquisition_metric_drift_tolerance": 1e-4,
            "complete_anchor_star_coverage": 0.999,
        },
        "interpretation": (
            "A negative EAF-DMVR mechanism result is not permission to reopen acquisition. "
            "If the new head is active but complete-frontier sign/action metrics do not improve, "
            "the next causal hypothesis is frozen action/evidence representation capacity. If the "
            "frontier mechanism improves but final teacher endpoint does not, inspect the downstream "
            "one-sided/structural preservation interface rather than creating another proposal loss."
        ),
    }


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-log", type=Path, required=True)
    ap.add_argument("--variant", default="SCREEN")
    ap.add_argument("--output", type=Path, required=True)
    args=ap.parse_args()
    rows=[json.loads(x) for x in args.train_log.read_text(encoding="utf-8").splitlines() if x.strip()]
    report=build(rows,args.variant)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
