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


def _delta(a: float, b: float) -> float:
    return b - a if math.isfinite(a) and math.isfinite(b) else float("nan")


def _pick(row: dict[str, Any], key: str) -> float:
    return _finite(row.get(key))


def _summary(row: dict[str, Any]) -> dict[str, float]:
    return {
        "teacher": _pick(row, "val_teacher_action_match"),
        "teacher_regret": _pick(row, "val_teacher_regret"),
        "pairfull": _pick(row, "val_pair_full_interface_action_match"),
        "pairfull_regret": _pick(row, "val_pair_full_teacher_regret"),
        "localpair": _pick(row, "val_local_pair_full_interface_action_match"),
        "critical_topm": _pick(row, "val_teacher_exact_winner_flip_critical_recall_topm_micro"),
        "critical_selected": _pick(row, "val_teacher_exact_winner_flip_critical_recall_selected_micro"),
        "proposal_decisive": _pick(row, "val_proposal_decisive_atom_recall"),
        "bdmu_topm_capture": _pick(row, "val_teacher_bdmu_topm_utility_capture"),
        "bdmu_selected_capture": _pick(row, "val_teacher_bdmu_selected_utility_capture"),
        "bdmu_missed_fraction": _pick(row, "val_teacher_bdmu_missed_utility_fraction"),
        "bdmu_margin_deficit": _pick(row, "val_teacher_bdmu_reference_margin_deficit"),
        "bdmu_scene_has_utility": _pick(row, "val_teacher_bdmu_scene_has_utility"),
        "bdmu_positive_atom_fraction": _pick(row, "val_teacher_bdmu_positive_atom_fraction"),
        "adapter_delta_rms": _pick(row, "critical_adapter_parameter_delta_rms"),
        "adapter_residual_rms": _pick(row, "critical_proposal_residual_rms"),
        "loss_fast_path": _pick(row, "bdmu_fast_path_active"),
        "train_loss_wall_s": _pick(row, "train_loss_wall_time_s"),
        "train_epoch_wall_s": _pick(row, "train_epoch_wall_time_s"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit V64.3.8 BDMU acquisition-isolation screen")
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-topm-utility-gain", type=float, default=0.02)
    ap.add_argument("--min-selected-utility-gain", type=float, default=0.01)
    ap.add_argument("--min-teacher-gain", type=float, default=0.005)
    ap.add_argument("--min-regret-relative-gain", type=float, default=0.02)
    ap.add_argument("--teacher-nonworse-tol", type=float, default=0.004)
    ap.add_argument("--regret-nonworse-relative-tol", type=float, default=0.02)
    args = ap.parse_args()

    path = Path(args.train_log)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    anchor_rows = [r for r in rows if int(r.get("epoch", 999999)) < 0]
    train_rows = [r for r in rows if int(r.get("epoch", -999999)) >= 0]
    if not anchor_rows or not train_rows:
        raise SystemExit("BDMU screen requires a val-before-training anchor (epoch < 0) and trained epochs")

    anchor_row = anchor_rows[-1]
    anchor = _summary(anchor_row)
    audited: list[dict[str, Any]] = []
    for row in train_rows:
        cur = _summary(row)
        d = {k: _delta(anchor[k], cur[k]) for k in anchor if k not in {"teacher_regret", "pairfull_regret", "train_loss_wall_s", "train_epoch_wall_s"}}
        d["teacher_regret"] = _delta(anchor["teacher_regret"], cur["teacher_regret"])
        d["pairfull_regret"] = _delta(anchor["pairfull_regret"], cur["pairfull_regret"])
        topm_mechanism = math.isfinite(d["bdmu_topm_capture"]) and d["bdmu_topm_capture"] >= args.min_topm_utility_gain
        selected_mechanism = math.isfinite(d["bdmu_selected_capture"]) and d["bdmu_selected_capture"] >= args.min_selected_utility_gain
        literal_support = math.isfinite(d["critical_topm"]) and d["critical_topm"] >= 0.01
        mechanism_gain = bool((topm_mechanism and selected_mechanism) or (topm_mechanism and literal_support))

        teacher_gain = math.isfinite(d["teacher"]) and d["teacher"] >= args.min_teacher_gain
        regret_rel = float("nan")
        if math.isfinite(anchor["teacher_regret"]) and abs(anchor["teacher_regret"]) > 1e-9 and math.isfinite(cur["teacher_regret"]):
            regret_rel = (anchor["teacher_regret"] - cur["teacher_regret"]) / abs(anchor["teacher_regret"])
        regret_gain = math.isfinite(regret_rel) and regret_rel >= args.min_regret_relative_gain
        deployment_gain = bool(teacher_gain or regret_gain)

        teacher_nonworse = math.isfinite(d["teacher"]) and d["teacher"] >= -args.teacher_nonworse_tol
        regret_nonworse = True
        if math.isfinite(anchor["teacher_regret"]) and abs(anchor["teacher_regret"]) > 1e-9 and math.isfinite(cur["teacher_regret"]):
            regret_nonworse = cur["teacher_regret"] <= anchor["teacher_regret"] * (1.0 + args.regret_nonworse_relative_tol)
        instrumentation = bool(
            math.isfinite(cur["bdmu_scene_has_utility"])
            and cur["bdmu_scene_has_utility"] > 0.0
            and math.isfinite(cur["adapter_delta_rms"])
            and cur["adapter_delta_rms"] > 1e-7
        )
        full_promotion = bool(instrumentation and mechanism_gain and deployment_gain and teacher_nonworse and regret_nonworse)
        audited.append({
            "epoch": int(row["epoch"]),
            "selected": cur,
            "deltas": d,
            "teacher_regret_relative_gain": regret_rel,
            "instrumentation_valid": instrumentation,
            "mechanism_gain": mechanism_gain,
            "deployment_gain": deployment_gain,
            "teacher_nonworse": teacher_nonworse,
            "teacher_regret_nonworse": regret_nonworse,
            "full_promotion": full_promotion,
        })

    # Mechanism-first selection prevents choosing an epoch solely because it got
    # lucky on a 500-scene action-match endpoint.  Among mechanism-valid epochs,
    # prioritize the paper endpoint, then regret, then BDMU capture.
    pool = [r for r in audited if r["mechanism_gain"] and r["instrumentation_valid"]] or audited
    def key(r: dict[str, Any]) -> tuple[float, float, float, int]:
        s = r["selected"]
        teacher = s["teacher"] if math.isfinite(s["teacher"]) else -1e9
        regret = s["teacher_regret"] if math.isfinite(s["teacher_regret"]) else 1e18
        capture = s["bdmu_topm_capture"] if math.isfinite(s["bdmu_topm_capture"]) else -1e9
        return (teacher, -regret, capture, -int(r["epoch"]))
    best = max(pool, key=key)

    report = {
        "audit": "v64_3_8_bdmu_screen",
        "objective": "cost-aware marginal reduction of teacher one-sided decisive-margin deficit under fixed B=16",
        "anchor_epoch": int(anchor_row.get("epoch", -1)),
        "anchor": anchor,
        "epochs": audited,
        "selected_epoch": best["epoch"],
        "selected": best["selected"],
        "deltas": best["deltas"],
        "instrumentation_valid": best["instrumentation_valid"],
        "mechanism_gain": best["mechanism_gain"],
        "deployment_gain": best["deployment_gain"],
        "teacher_nonworse": best["teacher_nonworse"],
        "teacher_regret_nonworse": best["teacher_regret_nonworse"],
        "full_promotion": best["full_promotion"],
        "thresholds": {
            "min_topm_utility_gain": args.min_topm_utility_gain,
            "min_selected_utility_gain": args.min_selected_utility_gain,
            "min_teacher_gain": args.min_teacher_gain,
            "min_regret_relative_gain": args.min_regret_relative_gain,
            "teacher_nonworse_tolerance": args.teacher_nonworse_tol,
            "regret_nonworse_relative_tolerance": args.regret_nonworse_relative_tol,
        },
        "scientific_note": "BDMU utility is the primary mechanism diagnostic; exact literal-critical recall remains a boundary stress-test rather than the training target. Full promotion still requires a fixed-budget teacher decision/regret gain, so utility improvement alone cannot promote the model.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["selected_epoch", "instrumentation_valid", "mechanism_gain", "deployment_gain", "full_promotion"]}, indent=2))


if __name__ == "__main__":
    main()
