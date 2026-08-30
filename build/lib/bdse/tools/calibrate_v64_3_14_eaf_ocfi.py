from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _finite(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _group_key(row: dict[str, Any]) -> str:
    token = str(row.get("scenario_token", "")).strip()
    if token:
        return token
    return f"timestamp:{int(row.get('timestamp_us', 0) or 0)}"


def _hash_fraction(key: str, seed: str) -> float:
    payload = f"{seed}|{key}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    return value / float(2**64)


def _split_rows(rows: list[dict[str, Any]], calibration_fraction: float, seed: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (0.0 < calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be in (0,1)")
    cal: list[dict[str, Any]] = []
    ev: list[dict[str, Any]] = []
    assignment: dict[str, bool] = {}
    for row in rows:
        key = _group_key(row)
        is_cal = assignment.setdefault(key, _hash_fraction(key, seed) < calibration_fraction)
        (cal if is_cal else ev).append(row)
    if not cal or not ev:
        raise ValueError(f"group split produced an empty partition: calibration={len(cal)}, evaluation={len(ev)}")
    return cal, ev


def _conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    values = np.asarray(scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n <= 0:
        raise ValueError("no finite calibration scores")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    # Split-conformal finite-sample upper quantile: kth order statistic with
    # k=ceil((n+1)(1-alpha)), clipped to [1,n].  Negative over-estimation scores
    # are possible, but we never use calibration to *relax* the legacy guard.
    k = min(max(int(math.ceil((n + 1) * (1.0 - alpha))), 1), n)
    q = float(np.partition(values, k - 1)[k - 1])
    return max(q, 0.0), k


def _metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = np.asarray([_finite(r.get(key)) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _raw_eval_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "teacher_action_match",
        "teacher_regret",
        "pair_full_interface_action_match",
        "local_pair_full_interface_action_match",
        "evidence_certificate_fraction",
        "selected_local_anchor_action_match",
        "deployed_vs_selected_local_anchor_match",
        "pair_potential_deployed_flip_rate",
        "beneficial_pair_potential_intervention_rate",
        "harmful_pair_potential_intervention_rate",
        "decisive_frontier_value_active",
        "decisive_frontier_value_complete_star_coverage",
        "decisive_frontier_value_residual_rms",
    ]
    return {k: _metric_mean(rows, k) for k in keys}


def calibrate(
    rows: list[dict[str, Any]],
    *,
    normalization: str,
    alpha: float,
    calibration_fraction: float,
    split_seed: str,
    scale_floor_quantile: float,
    min_scale_floor: float,
) -> dict[str, Any]:
    normalization = str(normalization).strip().lower()
    if normalization not in {"attribution", "none"}:
        raise ValueError("normalization must be attribution or none")
    cal_rows, eval_rows = _split_rows(rows, calibration_fraction, split_seed)

    proposal_rows: list[dict[str, Any]] = []
    attr_scales: list[float] = []
    for row in cal_rows:
        anchor = int(row.get("raw_frontier_anchor_action", -1))
        proposed = int(row.get("raw_frontier_proposed_action", -1))
        pred_margin = _finite(row.get("pair_action_anchor_raw_margin"))
        teacher_margin = _finite(row.get("decisive_frontier_value_teacher_proposed_vs_anchor_margin"))
        attr = _finite(row.get("decisive_frontier_ocfi_proposed_attribution_scale"))
        active = _finite(row.get("decisive_frontier_value_active"))
        complete = _finite(row.get("decisive_frontier_value_complete_star_coverage"))
        if anchor < 0 or proposed < 0 or anchor == proposed:
            continue
        if not (math.isfinite(pred_margin) and math.isfinite(teacher_margin)):
            continue
        if not (math.isfinite(active) and active >= 0.5 and math.isfinite(complete) and complete >= 0.99):
            continue
        proposal_rows.append(row)
        if math.isfinite(attr) and attr > 0.0:
            attr_scales.append(attr)

    if len(proposal_rows) < 32:
        raise ValueError(
            f"too few valid raw EAF proposal edges for calibration: {len(proposal_rows)}; "
            "run the fixed runtime-instrumented replay on more validation scenes"
        )

    if normalization == "attribution":
        if len(attr_scales) < 16:
            raise ValueError("attribution normalization requested but too few finite positive attribution scales")
        q_floor = float(np.quantile(np.asarray(attr_scales, dtype=np.float64), scale_floor_quantile))
        scale_floor = max(float(min_scale_floor), q_floor)
    else:
        scale_floor = 1.0

    scores: list[float] = []
    overestimation: list[float] = []
    for row in proposal_rows:
        pred_margin = _finite(row.get("pair_action_anchor_raw_margin"))
        teacher_margin = _finite(row.get("decisive_frontier_value_teacher_proposed_vs_anchor_margin"))
        err = pred_margin - teacher_margin
        overestimation.append(err)
        if normalization == "attribution":
            attr = _finite(row.get("decisive_frontier_ocfi_proposed_attribution_scale"))
            scale = max(attr if math.isfinite(attr) else 0.0, scale_floor)
        else:
            scale = 1.0
        scores.append(err / scale)

    q, order_index = _conformal_quantile(np.asarray(scores, dtype=np.float64), alpha)
    cal_keys = {_group_key(r) for r in cal_rows}
    eval_keys = {_group_key(r) for r in eval_rows}
    if cal_keys & eval_keys:
        raise AssertionError("group-disjoint calibration/evaluation split violated")

    return {
        "audit": "v64_3_14_eaf_ocfi_split_calibration",
        "normalization": normalization,
        "alpha": float(alpha),
        "coverage_target": float(1.0 - alpha),
        "calibration_fraction": float(calibration_fraction),
        "split_seed": str(split_seed),
        "calibration_group_count": len(cal_keys),
        "evaluation_group_count": len(eval_keys),
        "calibration_row_count": len(cal_rows),
        "evaluation_row_count": len(eval_rows),
        "calibration_proposal_edge_count": len(proposal_rows),
        "attribution_scale_floor": float(scale_floor),
        "scale_floor_quantile": float(scale_floor_quantile),
        "conformal_order_index_1based": int(order_index),
        "calibration_quantile": float(q),
        "raw_overestimation_mean": float(np.mean(overestimation)),
        "raw_overestimation_p90": float(np.quantile(overestimation, 0.90)),
        "raw_overestimation_p95": float(np.quantile(overestimation, 0.95)),
        "raw_eval_subset_metrics": _raw_eval_summary(eval_rows),
        "calibration_tokens": sorted({_group_key(r) for r in cal_rows}),
        "evaluation_tokens": sorted({_group_key(r) for r in eval_rows}),
    }


def _write_calibrated_config(
    base_config: Path,
    output_config: Path,
    report: dict[str, Any],
    report_path: Path,
) -> None:
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    runtime = cfg.setdefault("runtime", {})
    frontier = runtime.setdefault("decisive_frontier_value", {})
    ocfi = frontier.setdefault("one_sided_intervention", {})
    ocfi.update({
        "enabled": True,
        "normalization": str(report["normalization"]),
        "calibration_quantile": float(report["calibration_quantile"]),
        "attribution_scale_floor": float(report["attribution_scale_floor"]),
        "additive_radius": 0.0,
        "require_frontier_active": True,
        "calibration_method": "split_conformal_overprediction_on_raw_anchor_challenger_edge",
        "calibration_alpha": float(report["alpha"]),
        "calibration_role": "frozen_group_disjoint_calibration",
    })
    exp = cfg.setdefault("experiment", {})
    exp["name"] = f"v64_3_14_eaf_ocfi_{report['normalization']}_calibrated"
    exp["evaluation_role"] = "candidate"
    exp["calibration_protocol"] = (
        "group-disjoint split-conformal one-sided over-estimation calibration on the raw "
        "selected-local anchor/challenger edge; calibration parameters are frozen before held-out evaluation"
    )
    prov = cfg.setdefault("provenance", {})
    prov["ocfi_calibration_report"] = str(report_path)
    prov["ocfi_calibration_normalization"] = str(report["normalization"])
    prov["ocfi_calibration_alpha"] = float(report["alpha"])
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit V64.3.14 EAF-OCFI one-sided split-conformal intervention calibration.")
    ap.add_argument("--per-sample", type=Path, required=True)
    ap.add_argument("--base-config", type=Path, required=True)
    ap.add_argument("--normalization", choices=["attribution", "none"], required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--calibration-fraction", type=float, default=0.40)
    ap.add_argument("--split-seed", default="v64.3.14-eaf-ocfi-v1")
    ap.add_argument("--scale-floor-quantile", type=float, default=0.10)
    ap.add_argument("--min-scale-floor", type=float, default=0.005)
    ap.add_argument("--output-report", type=Path, required=True)
    ap.add_argument("--output-config", type=Path, required=True)
    ap.add_argument("--calibration-token-file", type=Path, required=True)
    ap.add_argument("--evaluation-token-file", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(x) for x in args.per_sample.read_text(encoding="utf-8").splitlines() if x.strip()]
    report = calibrate(
        rows,
        normalization=args.normalization,
        alpha=args.alpha,
        calibration_fraction=args.calibration_fraction,
        split_seed=args.split_seed,
        scale_floor_quantile=args.scale_floor_quantile,
        min_scale_floor=args.min_scale_floor,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.calibration_token_file.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation_token_file.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_token_file.write_text("\n".join(report["calibration_tokens"]) + "\n", encoding="utf-8")
    args.evaluation_token_file.write_text("\n".join(report["evaluation_tokens"]) + "\n", encoding="utf-8")
    _write_calibrated_config(args.base_config, args.output_config, report, args.output_report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"calibration_tokens", "evaluation_tokens"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
