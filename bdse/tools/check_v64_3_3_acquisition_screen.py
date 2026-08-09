from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite(row: dict, key: str):
    try:
        value = float(row[key])
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _last(rows: list[dict]) -> dict:
    candidates = [r for r in rows if int(r.get("epoch", -999)) >= 0]
    return candidates[-1] if candidates else {}


def _anchor(rows: list[dict]) -> dict:
    candidates = [r for r in rows if int(r.get("epoch", -999)) == -1]
    return candidates[-1] if candidates else {}


def _max(rows: list[dict], key: str):
    vals = [_finite(r, key) for r in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def build_report(rows: list[dict], variant: str) -> dict:
    anchor = _anchor(rows)
    last = _last(rows)
    report = {
        "screening_only": True,
        "variant": variant,
        "epochs_seen": sorted({int(r.get("epoch", -999)) for r in rows}),
        "activation_source": "parameter_delta_plus_wired_forward_and_acra",
        "critical_adapter_parameter_delta_rms_max": _max(rows, "critical_adapter_parameter_delta_rms"),
        "critical_adapter_parameter_delta_max_abs_max": _max(rows, "critical_adapter_parameter_delta_max_abs"),
        "critical_proposal_residual_rms_max": _max(rows, "critical_proposal_residual_rms"),
        "critical_proposal_residual_abs_mean_max": _max(rows, "critical_proposal_residual_abs_mean"),
        "acra_alignment_loss_max": _max(rows, "L_critical_adapter_residual_alignment"),
        "anchor_val_critical_topm_recall_macro": _finite(anchor, "val_teacher_exact_winner_flip_critical_recall_topm"),
        "last_val_critical_topm_recall_macro": _finite(last, "val_teacher_exact_winner_flip_critical_recall_topm"),
        "anchor_val_critical_topm_recall_micro": _finite(anchor, "val_teacher_exact_winner_flip_critical_recall_topm_micro"),
        "last_val_critical_topm_recall_micro": _finite(last, "val_teacher_exact_winner_flip_critical_recall_topm_micro"),
        "anchor_val_critical_selected_recall_micro": _finite(anchor, "val_teacher_exact_winner_flip_critical_recall_selected_micro"),
        "last_val_critical_selected_recall_micro": _finite(last, "val_teacher_exact_winner_flip_critical_recall_selected_micro"),
        "anchor_val_critical_scene_rate": _finite(anchor, "val_teacher_exact_winner_flip_critical_scene_rate"),
        "last_val_critical_scene_rate": _finite(last, "val_teacher_exact_winner_flip_critical_scene_rate"),
        "anchor_val_critical_count_mean": _finite(anchor, "val_teacher_exact_winner_flip_critical_count"),
        "last_val_critical_count_mean": _finite(last, "val_teacher_exact_winner_flip_critical_count"),
        "anchor_val_proposal_decisive_recall": _finite(anchor, "val_proposal_decisive_atom_recall"),
        "last_val_proposal_decisive_recall": _finite(last, "val_proposal_decisive_atom_recall"),
        "last_val_teacher_action_match": _finite(last, "val_teacher_action_match"),
    }
    for stem in ("critical_topm_recall_micro", "critical_selected_recall_micro", "proposal_decisive_recall"):
        a = report.get("anchor_val_" + stem)
        z = report.get("last_val_" + stem)
        report["delta_val_" + stem] = (z - a) if a is not None and z is not None else None

    required = [
        "critical_adapter_parameter_delta_rms_max",
        "critical_proposal_residual_rms_max",
        "acra_alignment_loss_max",
        "anchor_val_critical_topm_recall_micro",
        "last_val_critical_topm_recall_micro",
        "anchor_val_proposal_decisive_recall",
        "last_val_proposal_decisive_recall",
        "anchor_val_critical_count_mean",
    ]
    report["screen_instrumentation_valid"] = all(report.get(k) is not None for k in required)
    report["adapter_parameter_activated"] = bool(
        report["critical_adapter_parameter_delta_rms_max"] is not None
        and report["critical_adapter_parameter_delta_rms_max"] > 1.0e-9
    )
    report["adapter_forward_activated"] = bool(
        report["critical_proposal_residual_rms_max"] is not None
        and report["critical_proposal_residual_rms_max"] > 1.0e-8
    )
    report["acra_wired"] = bool(
        report["acra_alignment_loss_max"] is not None
        and report["acra_alignment_loss_max"] > 1.0e-10
    )
    report["literal_critical_support_nonempty"] = bool(
        report["anchor_val_critical_count_mean"] is not None
        and report["anchor_val_critical_count_mean"] > 0.0
    )
    report["continue_to_full_run"] = bool(
        report["screen_instrumentation_valid"]
        and report["adapter_parameter_activated"]
        and report["adapter_forward_activated"]
        and report["acra_wired"]
        and report["literal_critical_support_nonempty"]
        and report["delta_val_critical_topm_recall_micro"] is not None
        and report["delta_val_critical_topm_recall_micro"] > 0.0
        and report["delta_val_proposal_decisive_recall"] is not None
        and report["delta_val_proposal_decisive_recall"] >= -0.02
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.train_log.read_text().splitlines() if x.strip()]
    report = build_report(rows, args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
