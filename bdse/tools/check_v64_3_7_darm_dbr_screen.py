from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _max(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [v for row in rows if (v := _finite(row.get(key))) is not None]
    return max(vals) if vals else None


def _min(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [v for row in rows if (v := _finite(row.get(key))) is not None]
    return min(vals) if vals else None


def _delta(row: dict[str, Any], anchor: dict[str, Any], key: str) -> float | None:
    a, b = _finite(anchor.get(key)), _finite(row.get(key))
    return None if a is None or b is None else b - a


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty training log")
    anchor = next((r for r in rows if int(r.get("epoch", 0)) < 0), rows[0])
    post = [r for r in rows if int(r.get("epoch", -1)) >= 0] or [rows[-1]]
    keys = {
        "teacher": "val_teacher_action_match",
        "pairfull": "val_pair_full_interface_action_match",
        "localpair": "val_local_pair_full_interface_action_match",
        "budgetpair": "val_budget_vs_pair_full_match",
        "pairflip": "val_pair_full_to_budget_flip_rate",
        "topm": "val_teacher_exact_winner_flip_critical_recall_topm_micro",
        "selected": "val_teacher_exact_winner_flip_critical_recall_selected_micro",
        "proposal": "val_proposal_decisive_atom_recall",
        "beneficial": "val_beneficial_residual_intervention_rate",
        "harmful": "val_harmful_residual_intervention_rate",
    }
    enriched = []
    for row in post:
        d = {k: _delta(row, anchor, v) for k, v in keys.items()}
        pairfull, localpair = _finite(row.get(keys["pairfull"])), _finite(row.get(keys["localpair"]))
        advantage = None if pairfull is None or localpair is None else pairfull - localpair
        beneficial, harmful = _finite(row.get(keys["beneficial"])), _finite(row.get(keys["harmful"]))
        intervention_net = None if beneficial is None or harmful is None else beneficial - harmful
        value_gain = bool(
            d["pairfull"] is not None and d["pairfull"] >= 0.01
            and advantage is not None and advantage >= 0.005
            and d["teacher"] is not None and d["teacher"] >= -0.005
            and d["budgetpair"] is not None and d["budgetpair"] >= -0.02
            and (intervention_net is None or intervention_net >= 0.0)
        )
        full = bool(value_gain and d["teacher"] is not None and d["teacher"] >= 0.005)
        enriched.append((row, d, advantage, intervention_net, value_gain, full))

    def score(item: tuple) -> tuple:
        row, d, advantage, intervention_net, value_gain, full = item
        return (
            int(full), int(value_gain), d["teacher"] or -9.0, d["pairfull"] or -9.0,
            advantage or -9.0, intervention_net or -9.0,
        )

    row, d, advantage, intervention_net, value_gain, full = max(enriched, key=score)
    dbr_delta = _max(post, "decisive_pair_adapter_parameter_delta_rms") or 0.0
    dbr_rms = _max(post, "decisive_boundary_pair_residual_rms") or 0.0
    full_cov = _max(post, "decisive_anchor_full_pair_coverage")
    budget_cov = _max(post, "decisive_anchor_budget_pair_coverage")
    runtime_active = _min(post, "val_decisive_anchor_margin_active")
    anchor_teacher = _finite(anchor.get(keys["teacher"]))
    strong_anchor_restored = anchor_teacher is not None and anchor_teacher >= 0.24
    valid = bool(
        strong_anchor_restored
        and dbr_delta > 1e-7
        and dbr_rms > 1e-7
        and (full_cov is None or full_cov >= 0.20)
        and (runtime_active is None or runtime_active > 0.99)
    )
    return {
        "audit": "v64_3_7_darm_dbr_screen",
        "variant": variant,
        "valid": valid,
        "anchor_epoch": anchor.get("epoch"),
        "selected_epoch": row.get("epoch"),
        "anchor": {k: _finite(anchor.get(v)) for k, v in keys.items()},
        "selected": {k: _finite(row.get(v)) for k, v in keys.items()},
        "deltas": d,
        "pair_full_advantage_over_local": advantage,
        "residual_intervention_net": intervention_net,
        "meaningful_value_gain": bool(value_gain),
        "full_promotion": bool(full and valid),
        "strong_selected_local_anchor_restored": bool(strong_anchor_restored),
        "activation": {
            "dbr_parameter_delta_rms_max": dbr_delta,
            "dbr_residual_rms_max": dbr_rms,
            "decisive_anchor_full_pair_coverage_max": full_cov,
            "decisive_anchor_budget_pair_coverage_max": budget_cov,
            "runtime_darm_active_min": runtime_active,
        },
        "thresholds": {
            "anchor_teacher_match_floor": 0.24,
            "pair_full_gain": 0.01,
            "pair_full_over_local": 0.005,
            "teacher_stability_floor": -0.005,
            "full_teacher_gain": 0.005,
            "budget_vs_pair_full_floor": -0.02,
            "residual_intervention_net_floor": 0.0,
            "training_anchor_pair_coverage_floor": 0.20,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = build(load_rows(args.train_log), args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
