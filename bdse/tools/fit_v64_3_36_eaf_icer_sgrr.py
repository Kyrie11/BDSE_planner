from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.tools.fit_v64_3_33_eaf_icer_spcr import (
    BASE_FEATURE_NAMES, CAT, FEATURE_NAMES, FOLDS, RIDGE_LAMBDA,
    _diag, _finite, _fold, _read_edges, _scene_samples, _select,
)
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import (
    _extended_diag, _fit_regret_structured_margin, _structured_scores,
)

EPS = 1.0e-12
MIN_RESERVATION_CAL_PROPOSALS = 64
NOOP_REDUCTION_MIN = 0.20
CAPTURE_TOL = 0.03
CATASTROPHE_REDUCTION_MIN = 0.25
GEOMETRY_NAMES = [
    "reservation::top_score",
    "reservation::top_gap_to_runnerup_or_incumbent",
    "reservation::score_rms",
    "reservation::positive_score_fraction",
    "reservation::log_effective_competitor_mass",
]
BASEPOINT_NAMES = [f"reservation_incumbent::{n}" for n in BASE_FEATURE_NAMES] + ["reservation_incumbent::support_logit"]


def _scene_context(group: list[dict[str, Any]]) -> np.ndarray | None:
    if not group:
        return None
    inc = int(group[0].get("raw_top_action", -1))
    ir = next((r for r in group if int(r.get("challenger_action", -2)) == inc), None)
    if ir is None or _finite(ir, "icer_admissible", 0.0) < 0.5:
        return None
    base = np.asarray([_finite(ir, f"icer_feature_{n}") for n in BASE_FEATURE_NAMES], dtype=np.float64)
    sup = _finite(ir, "icer_support_logit")
    if not np.all(np.isfinite(base)) or not math.isfinite(sup):
        return None
    return np.concatenate([base, np.asarray([sup], dtype=np.float64)])


def _build(groups: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    scene: dict[str, list[dict[str, Any]]] = {}
    context: dict[str, np.ndarray] = {}
    for t, g in groups.items():
        ss = _scene_samples(g)
        cc = _scene_context(g)
        if ss and cc is not None:
            scene[t] = ss
            context[t] = cc
    return scene, context


def _geometry(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0 or not np.all(np.isfinite(s)):
        raise ValueError("V36 selection geometry requires finite challenger scores")
    order = np.sort(s)[::-1]
    top = float(order[0])
    second = float(order[1]) if order.size >= 2 else float("-inf")
    runner = max(0.0, second) if math.isfinite(second) else 0.0
    gap = top - runner
    rms = float(np.sqrt(np.mean(s * s)))
    pos_frac = float(np.mean(s > 0.0))
    log_mass = float(np.log(np.exp(np.clip(s - top, -60.0, 0.0)).sum()))
    return np.asarray([top, gap, rms, pos_frac, log_mass], dtype=np.float64)


def _fit_reservation(X: np.ndarray, target: np.ndarray, *, mode: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xx = np.asarray(X, dtype=np.float64)
    yy = np.asarray(target, dtype=np.float64).reshape(-1)
    if xx.ndim != 2 or xx.shape[0] != yy.size or yy.size < MIN_RESERVATION_CAL_PROPOSALS:
        raise ValueError("V36 reservation calibration population is malformed or too small")
    if np.any(~np.isfinite(xx)) or np.any(~np.isfinite(yy)) or np.any(yy < -EPS):
        raise ValueError("V36 reservation features/targets must be finite with nonnegative overprediction targets")
    scale = np.maximum(np.sqrt(np.mean(xx * xx, axis=0)), 1.0e-6)
    z = xx / scale[None, :]
    gram = z.T @ z + RIDGE_LAMBDA * np.eye(z.shape[1], dtype=np.float64)
    rhs = z.T @ yy
    w = np.linalg.solve(gram, rhs)
    pred = np.maximum(z @ w, 0.0)
    return w, scale, {
        "mode": mode,
        "ridge_lambda": RIDGE_LAMBDA,
        "sample_count": int(yy.size),
        "target_mean": float(yy.mean()),
        "target_rms": float(np.sqrt(np.mean(yy * yy))),
        "predicted_reservation_mean": float(pred.mean()),
        "fit_mse": float(np.mean((pred - yy) ** 2)),
        "weight_l2": float(np.linalg.norm(w)),
        "target_definition": "max(0, frozen_rsmr_selected_score - selected_teacher_improvement)",
    }


def _reservation_value(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, dict[str, Any]]) -> float:
    w, scale, _ = model
    z = np.asarray(x, dtype=np.float64) / np.maximum(np.asarray(scale, dtype=np.float64), 1.0e-6)
    return float(min(max(float(z @ w), 0.0), 40.0))


def _cal_samples(
    scene: dict[str, list[dict[str, Any]]], context: dict[str, np.ndarray], tokens: list[str], rsm_model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    geo: list[np.ndarray] = []
    bp: list[np.ndarray] = []
    target: list[float] = []
    used: list[str] = []
    for t in tokens:
        ss = scene.get(t, [])
        if not ss:
            continue
        score = _structured_scores(ss, rsm_model)
        idx = _select(ss, score)
        if idx is None:
            continue
        y = float(ss[idx]["y"])
        geo.append(_geometry(score))
        bp.append(np.asarray(context[t], dtype=np.float64))
        target.append(max(0.0, float(score[idx]) - y))
        used.append(t)
    if not target:
        return np.zeros((0, len(GEOMETRY_NAMES))), np.zeros((0, len(BASEPOINT_NAMES))), np.zeros((0,)), []
    return np.stack(geo), np.stack(bp), np.asarray(target, dtype=np.float64), used


def _evaluate_policy(
    ss: list[dict[str, Any]], rsm_model, reservation_model: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None,
    reservation_feature: np.ndarray | None,
) -> tuple[int | None, float, float, float]:
    score = _structured_scores(ss, rsm_model)
    idx = _select(ss, score)
    if idx is None:
        return None, 0.0, 0.0, float("nan")
    raw = float(score[idx])
    reservation = 0.0 if reservation_model is None else _reservation_value(np.asarray(reservation_feature), reservation_model)
    adjusted = raw - reservation
    chosen = idx if adjusted > 0.0 else None
    return chosen, raw, reservation, adjusted


def _metrics_from_lists(vals: list[float], captured: int, opp: int, noop_sel: int, opp_sel: int, noop_scenes: int) -> dict[str, Any]:
    return _extended_diag(vals, captured, opp, noop_sel, opp_sel, noop_scenes)


def _nested(groups: dict[str, list[dict[str, Any]]], audit_csv: Path) -> dict[str, Any]:
    scene, context = _build(groups)
    names = ["rsm", "basepoint", "geometry"]
    agg = {n: [] for n in names}
    caps = {n: 0 for n in names}
    noops = {n: 0 for n in names}
    oppsels = {n: 0 for n in names}
    folds: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    total_opp = 0
    total_noop = 0

    for k in range(FOLDS):
        test = [t for t in scene if _fold(t) == k]
        cal_fold = (k + 1) % FOLDS
        cal = [t for t in scene if _fold(t) == cal_fold]
        fit = [t for t in scene if _fold(t) not in {k, cal_fold}]
        rsm = _fit_regret_structured_margin(scene, fit)
        gx, bx, target, used = _cal_samples(scene, context, cal, rsm)
        if len(used) < MIN_RESERVATION_CAL_PROPOSALS:
            raise SystemExit(f"V64.3.36 nested reservation calibration fold {k} has {len(used)} proposals < {MIN_RESERVATION_CAL_PROPOSALS}")
        gmodel = _fit_reservation(gx, target, mode="selection_geometry")
        bmodel = _fit_reservation(bx, target, mode="incumbent_basepoint")

        fv = {n: [] for n in names}; fc = {n: 0 for n in names}; fn = {n: 0 for n in names}; fo = {n: 0 for n in names}
        opp = 0; noop_scenes = 0
        subset_ok = True; winner_identity_ok = True
        for t in test:
            ss = scene[t]
            y = np.asarray([float(a["y"]) for a in ss], dtype=np.float64)
            has_opp = bool(np.any(y > 0.0)); opp += int(has_opp); noop_scenes += int(not has_opp)
            rscore = _structured_scores(ss, rsm); ri = _select(ss, rscore)
            gi, raw_g, rg, adj_g = _evaluate_policy(ss, rsm, gmodel, _geometry(rscore))
            bi, raw_b, rb, adj_b = _evaluate_policy(ss, rsm, bmodel, context[t])
            subset_ok = subset_ok and (gi is None or ri is not None) and (bi is None or ri is not None)
            winner_identity_ok = winner_identity_ok and (gi is None or gi == ri) and (bi is None or bi == ri)
            chosen = {"rsm": ri, "basepoint": bi, "geometry": gi}
            for name, idx in chosen.items():
                if idx is None: continue
                yy = float(y[idx]); fv[name].append(yy); fc[name] += int(has_opp and yy > 0.0); fn[name] += int(not has_opp); fo[name] += int(has_opp)
            audits.append({
                "scenario_token": t, "outer_test_fold": k, "calibration_fold": cal_fold, "candidate_count": len(ss),
                "positive_opportunity": int(has_opp),
                "rsm_selected_action": -1 if ri is None else int(ss[ri]["action"]),
                "rsm_selected_score": float("nan") if ri is None else float(rscore[ri]),
                "rsm_selected_teacher_improvement": float("nan") if ri is None else float(y[ri]),
                "basepoint_selected_action": -1 if bi is None else int(ss[bi]["action"]),
                "basepoint_reservation": rb, "basepoint_adjusted_margin": adj_b,
                "basepoint_selected_teacher_improvement": float("nan") if bi is None else float(y[bi]),
                "geometry_selected_action": -1 if gi is None else int(ss[gi]["action"]),
                "geometry_reservation": rg, "geometry_adjusted_margin": adj_g,
                "geometry_selected_teacher_improvement": float("nan") if gi is None else float(y[gi]),
            })
        total_opp += opp; total_noop += noop_scenes
        fd = {}
        for n in names:
            fd[n] = _metrics_from_lists(fv[n], fc[n], opp, fn[n], fo[n], noop_scenes)
            agg[n].extend(fv[n]); caps[n] += fc[n]; noops[n] += fn[n]; oppsels[n] += fo[n]
        folds.append({
            "fold": k, "fit_scenes": len(fit), "calibration_scenes": len(cal), "test_scenes": len(test),
            "reservation_calibration_proposal_count": len(used),
            "selection_geometry_reservation_fit": gmodel[2], "basepoint_reservation_fit": bmodel[2],
            "rsmr_rank": fd["rsm"], "basepoint_frozen_order_reservation": fd["basepoint"], "selection_geometry_reservation": fd["geometry"],
            "monotone_subset_valid": subset_ok, "frozen_winner_identity_valid": winner_identity_ok,
        })

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(audits[0]) if audits else []
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(audits)
    A = {n: _metrics_from_lists(agg[n], caps[n], total_opp, noops[n], oppsels[n], total_noop) for n in names}
    r = A["rsm"]; b = A["basepoint"]; g = A["geometry"]
    noop_target = (1.0 - NOOP_REDUCTION_MIN) * float(r["no_positive_opportunity_false_intervention_count"])
    cat_target = (1.0 - CATASTROPHE_REDUCTION_MIN) * float(r["catastrophic_count"])
    basepoint_signal = bool(
        b["no_positive_opportunity_false_intervention_count"] <= noop_target + EPS
        and b["positive_capture_rate"] >= r["positive_capture_rate"] - CAPTURE_TOL - EPS
    )
    geometry_existence = bool(
        g["no_positive_opportunity_false_intervention_count"] <= noop_target + EPS
        and g["positive_capture_rate"] >= r["positive_capture_rate"] - CAPTURE_TOL - EPS
    )
    geometry_tail = bool(
        g["catastrophic_count"] <= cat_target + EPS
        and g["teacher_negative_rms"] <= r["teacher_negative_rms"] + EPS
        and g["teacher_improvement_sum"] >= -EPS
    )
    fold_direction = all(f["selection_geometry_reservation"]["teacher_improvement_sum"] >= -EPS for f in folds)
    contracts = all(f["monotone_subset_valid"] and f["frozen_winner_identity_valid"] and f["reservation_calibration_proposal_count"] >= MIN_RESERVATION_CAL_PROPOSALS for f in folds)
    train_pass = bool(geometry_existence and geometry_tail and fold_direction and contracts and g["selected_count"] >= 64 and g["selected_positive_count"] >= 32)
    if not geometry_existence:
        diagnosis = "selection_geometry_reservation_does_not_resolve_intervention_existence_without_destroying_v34_capture"
    elif not geometry_tail or not fold_direction:
        diagnosis = "selection_geometry_reservation_improves_existence_but_selected_tail_or_crossfold_direction_remains_unstable"
    elif not contracts:
        diagnosis = "reservation_contract_or_calibration_population_invalid"
    else:
        diagnosis = "full_nested_train_pass" if train_pass else "reservation_coverage_population_too_small"
    return {
        "folds": folds, "scene_audit_csv": str(audit_csv),
        "rsmr_rank_aggregate": r, "basepoint_frozen_order_reservation_aggregate": b, "selection_geometry_reservation_aggregate": g,
        "basepoint_clean_signal": basepoint_signal, "selection_geometry_existence_gain": geometry_existence,
        "selection_geometry_tail_gain": geometry_tail, "all_folds_selection_geometry_sum_nonnegative": fold_direction,
        "monotone_frozen_order_contract_valid": contracts, "train_gate_pass": train_pass, "failure_diagnosis": diagnosis,
        "noop_reduction_fraction_min": NOOP_REDUCTION_MIN, "capture_tolerance": CAPTURE_TOL,
        "catastrophe_reduction_fraction_min": CATASTROPHE_REDUCTION_MIN,
    }


def _base_cfg(path: str) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    ic = cfg.setdefault("runtime", {}).setdefault("decisive_frontier_value", {}).setdefault("incumbent_contrastive_extremal_recovery", {})
    ic["incumbent_retention_policy"] = "preserve_admissible_incumbent"
    ic["regret_risk_enabled"] = False; ic["retention_regret_risk_enabled"] = False; ic["replacement_regret_risk_enabled"] = False
    return cfg


def _write_rsmr(base: dict[str, Any], path: str, model) -> None:
    w, scale, _ = model
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    sc = ic.setdefault("selection_conditioned_intervention_recovery", {})
    sc.update({
        "enabled": True, "mode": "rank_only", "model_type": "v36_frozen_v34_rsmr_ordering",
        "base_feature_names": BASE_FEATURE_NAMES, "feature_names": FEATURE_NAMES,
        "feature_mean": [0.0] * len(FEATURE_NAMES), "feature_std": [float(x) for x in scale],
        "weights": [float(x) for x in w], "bias": 0.0, "ridge_lambda": RIDGE_LAMBDA,
        "leverage_inverse": [], "selection_scale_floor": 1.0, "require_positive_predicted_improvement": True,
        "no_fallback": True, "scene_reservation_enabled": False,
        "training_population": "TRAIN_only_incumbent_deployment_admissible_support_positive_direct_scenes",
        "training_target": "V34_teacher_best_vs_worst_cost_augmented_rival_structured_regret_margin",
        "proposal_operator": "frozen_RSMR_argmax_positive_score_with_incumbent_zero_pseudoitem",
    })
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.36-RSMR-FROZEN"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.36-RSMR-FROZEN"
    cfg.setdefault("experiment", {})["name"] = "v64_3_36_rsmr_frozen_order"
    cfg["experiment"]["algorithm"] = "V64.3.36 frozen V34 RSMR ordering for reservation calibration"
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="V64.3.36 nested frozen-order scene-reservation diagnostic")
    ap.add_argument("--train-frontier-edges", required=True); ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-preserve-config", required=True); ap.add_argument("--output-rsmr-config", required=True); ap.add_argument("--output-report", required=True); ap.add_argument("--output-scene-audit", required=True)
    a = ap.parse_args()
    _, groups = _read_edges(Path(a.train_frontier_edges))
    nested = _nested(groups, Path(a.output_scene_audit))
    report = {
        "audit": "v64_3_36_eaf_icer_sgrr_fit",
        "scientific_role": "TRAIN_only_frozen_RSMR_ordering_plus_clean_basepoint_vs_selection_geometry_reservation_disambiguation",
        "frozen_train_scenes": len(groups), "direct_support_positive_training_scenes": len(_build(groups)[0]),
        "ridge_lambda": RIDGE_LAMBDA, "geometry_feature_names": GEOMETRY_NAMES, "basepoint_feature_names": BASEPOINT_NAMES,
        "reservation_target": "selected_policy_overprediction=max(0, frozen_RSMR_winner_score-selected_teacher_improvement)",
        "mechanism_hypothesis": "V34 ordering contains useful regret signal; V35 basepoint context is weak and jointly confounded. Freeze winner identity and learn only a nonnegative scene-common reservation. If selection geometry succeeds where clean basepoint does not, the dominant existence error is selection-induced overestimation/set context rather than missing absolute incumbent state.",
        "nested_crossfit": nested, "train_gate_pass": nested["train_gate_pass"],
        "train_gate_contract": {
            "reservation_monotone_subset_of_RSMR": True, "winner_identity_must_equal_RSMR_when_accepted": True,
            "noop_false_intervention_reduction_fraction_min": NOOP_REDUCTION_MIN, "capture_tolerance": CAPTURE_TOL,
            "catastrophe_reduction_fraction_min": CATASTROPHE_REDUCTION_MIN, "all_test_folds_selected_sum_nonnegative": True,
            "selected_min": 64, "positive_min": 32, "no_runtime_threshold_or_lambda_sweep": True,
        },
    }
    rp = Path(a.output_report); rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["train_gate_pass"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(f"V64.3.36 SGRR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before CAL/fresh selection")
    scene, _ = _build(groups)
    base = _base_cfg(a.base_config)
    pcfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    pcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"] = {"enabled": False}
    pcfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.36-PRESERVE-CONTROL"
    pcfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.36-PRESERVE-CONTROL"
    Path(a.output_preserve_config).write_text(yaml.safe_dump(pcfg, sort_keys=False))
    full_rsm = _fit_regret_structured_margin(scene, list(scene))
    _write_rsmr(base, a.output_rsmr_config, full_rsm)
    print(json.dumps({"pass": True, "output_rsmr_config": a.output_rsmr_config}, sort_keys=True))


if __name__ == "__main__":
    main()
