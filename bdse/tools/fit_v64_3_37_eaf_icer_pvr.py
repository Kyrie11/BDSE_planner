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
    BASE_FEATURE_NAMES,
    CAT,
    FEATURE_NAMES,
    FOLDS,
    RIDGE_LAMBDA,
    _fold,
    _read_edges,
    _scene_samples,
    _select,
)
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import (
    _extended_diag,
    _fit_regret_structured_margin,
    _structured_scores,
)

EPS = 1.0e-12
MIN_VALUE_CAL_PROPOSALS = 64
NOOP_REDUCTION_MIN = 0.20
CAPTURE_TOL = 0.03
CATASTROPHE_REDUCTION_MIN = 0.25


def _build(groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    scene = {t: _scene_samples(g) for t, g in groups.items()}
    return {t: ss for t, ss in scene.items() if ss}


def _proposal_geometry(
    ss: list[dict[str, Any]],
    rsm_model: tuple[np.ndarray, np.ndarray, dict[str, Any]],
    idx: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return frozen RSMR scalar, standardized winner feature, and score-orthogonal feature.

    z is the exact 19-D standardized candidate-minus-incumbent evidence used by
    V34 RSMR.  z_perp removes only the one-dimensional RSMR score direction:
        z_perp = z - w (w^T z) / ||w||^2.
    Hence w^T z_perp == 0 (numerically), so the OPVR residual head cannot merely
    relearn a second copy of the scalar ranking score.
    """
    w, scale, _ = rsm_model
    x = np.asarray(ss[idx]["x"], dtype=np.float64)
    z = x / np.maximum(np.asarray(scale, dtype=np.float64), 1.0e-6)
    ww = float(np.dot(w, w))
    if ww <= EPS:
        raise RuntimeError("V64.3.37 RSMR score direction has zero norm")
    u_linear = float(np.dot(z, w))
    u = float(np.clip(u_linear, -40.0, 40.0))
    z_perp = z - np.asarray(w, dtype=np.float64) * (u_linear / ww)
    if abs(float(np.dot(z_perp, w))) > 1.0e-8 * max(1.0, float(np.linalg.norm(z)), float(np.linalg.norm(w))):
        raise RuntimeError("V64.3.37 orthogonal proposal decomposition lost score orthogonality")
    return u, z, z_perp


def _fit_affine_value(u: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    uu = np.asarray(u, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if uu.size != yy.size or yy.size < MIN_VALUE_CAL_PROPOSALS or np.any(~np.isfinite(uu)) or np.any(~np.isfinite(yy)):
        raise ValueError("V64.3.37 affine value population is malformed or too small")
    mean_u = float(np.mean(uu))
    scale_u = max(float(np.sqrt(np.mean((uu - mean_u) ** 2))), 1.0e-6)
    zu = (uu - mean_u) / scale_u
    mean_y = float(np.mean(yy))
    yc = yy - mean_y
    slope = float(np.dot(zu, yc) / (np.dot(zu, zu) + RIDGE_LAMBDA))
    pred = mean_y + slope * zu
    return {
        "score_mean": mean_u,
        "score_std": scale_u,
        "intercept": mean_y,
        "score_weight": slope,
        "ridge_lambda": RIDGE_LAMBDA,
        "sample_count": int(yy.size),
        "fit_mse": float(np.mean((pred - yy) ** 2)),
        "target_mean": mean_y,
        "target_rms": float(np.sqrt(np.mean(yy * yy))),
        "prediction_mean": float(np.mean(pred)),
        "training_target": "teacher_improvement_of_frozen_RSMR_selected_proposal",
        "operator": "post_selection_absolute_value_readout_no_rerank",
    }


def _affine_value(u: float, model: dict[str, Any]) -> float:
    z = (float(u) - float(model["score_mean"])) / max(float(model["score_std"]), 1.0e-6)
    return float(model["intercept"] + float(model["score_weight"]) * z)


def _fit_orthogonal_residual(z_perp: np.ndarray, residual: np.ndarray) -> dict[str, Any]:
    X = np.asarray(z_perp, dtype=np.float64)
    e = np.asarray(residual, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[0] != e.size or e.size < MIN_VALUE_CAL_PROPOSALS:
        raise ValueError("V64.3.37 orthogonal residual population is malformed or too small")
    if X.shape[1] != len(FEATURE_NAMES) or np.any(~np.isfinite(X)) or np.any(~np.isfinite(e)):
        raise ValueError("V64.3.37 orthogonal residual features/targets are non-finite or wrong-shaped")
    mean = np.mean(X, axis=0)
    xc = X - mean[None, :]
    std = np.maximum(np.sqrt(np.mean(xc * xc, axis=0)), 1.0e-6)
    Z = xc / std[None, :]
    gram = Z.T @ Z + RIDGE_LAMBDA * np.eye(Z.shape[1], dtype=np.float64)
    rhs = Z.T @ e
    w = np.linalg.solve(gram, rhs)
    pred = Z @ w
    return {
        "feature_mean": [float(v) for v in mean],
        "feature_std": [float(v) for v in std],
        "weights": [float(v) for v in w],
        "bias": 0.0,
        "ridge_lambda": RIDGE_LAMBDA,
        "sample_count": int(e.size),
        "residual_mean": float(np.mean(e)),
        "residual_rms": float(np.sqrt(np.mean(e * e))),
        "fit_mse": float(np.mean((pred - e) ** 2)),
        "weight_l2": float(np.linalg.norm(w)),
        "feature_definition": "selected_RSMR_standardized_19D_projected_orthogonal_to_frozen_RSMR_score_direction",
        "training_target": "teacher_improvement_minus_affine_score_value",
    }


def _orthogonal_value(u: float, z_perp: np.ndarray, affine: dict[str, Any], residual: dict[str, Any]) -> float:
    mean = np.asarray(residual["feature_mean"], dtype=np.float64)
    std = np.asarray(residual["feature_std"], dtype=np.float64)
    w = np.asarray(residual["weights"], dtype=np.float64)
    z = (np.asarray(z_perp, dtype=np.float64) - mean) / np.maximum(std, 1.0e-6)
    return float(_affine_value(u, affine) + np.dot(z, w) + float(residual.get("bias", 0.0)))


def _cal_samples(
    scene: dict[str, list[dict[str, Any]]],
    tokens: list[str],
    rsm_model: tuple[np.ndarray, np.ndarray, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    us: list[float] = []
    perps: list[np.ndarray] = []
    ys: list[float] = []
    used: list[str] = []
    for t in tokens:
        ss = scene.get(t, [])
        if not ss:
            continue
        score = _structured_scores(ss, rsm_model)
        idx = _select(ss, score)
        if idx is None:
            continue
        u, _, zperp = _proposal_geometry(ss, rsm_model, idx)
        if abs(u - float(score[idx])) > 1.0e-8:
            raise RuntimeError("V64.3.37 frozen RSMR score replay mismatch in value calibration")
        us.append(u)
        perps.append(zperp)
        ys.append(float(ss[idx]["y"]))
        used.append(t)
    if not ys:
        return np.zeros((0,)), np.zeros((0, len(FEATURE_NAMES))), np.zeros((0,)), []
    return np.asarray(us), np.stack(perps), np.asarray(ys), used


def _fit_value_models(
    scene: dict[str, list[dict[str, Any]]],
    tokens: list[str],
    rsm_model: tuple[np.ndarray, np.ndarray, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    u, zp, y, used = _cal_samples(scene, tokens, rsm_model)
    if len(used) < MIN_VALUE_CAL_PROPOSALS:
        raise ValueError(f"V64.3.37 selected-policy value calibration has {len(used)} proposals < {MIN_VALUE_CAL_PROPOSALS}")
    affine = _fit_affine_value(u, y)
    affine_pred = np.asarray([_affine_value(v, affine) for v in u], dtype=np.float64)
    residual = _fit_orthogonal_residual(zp, y - affine_pred)
    orth_pred = np.asarray([_orthogonal_value(v, q, affine, residual) for v, q in zip(u, zp)], dtype=np.float64)
    residual["combined_fit_mse"] = float(np.mean((orth_pred - y) ** 2))
    residual["combined_prediction_mean"] = float(np.mean(orth_pred))
    return affine, residual, used


def _evaluate(
    ss: list[dict[str, Any]],
    rsm_model: tuple[np.ndarray, np.ndarray, dict[str, Any]],
    affine: dict[str, Any] | None,
    residual: dict[str, Any] | None,
) -> tuple[int | None, float, float]:
    score = _structured_scores(ss, rsm_model)
    idx = _select(ss, score)
    if idx is None:
        return None, 0.0, float("nan")
    u, _, zperp = _proposal_geometry(ss, rsm_model, idx)
    if affine is None:
        return idx, u, u
    value = _affine_value(u, affine) if residual is None else _orthogonal_value(u, zperp, affine, residual)
    return (idx if value > 0.0 else None), u, float(value)


def _metrics(vals: list[float], captured: int, opp: int, noop_selected: int, opp_selected: int, noop_scenes: int) -> dict[str, Any]:
    return _extended_diag(vals, captured, opp, noop_selected, opp_selected, noop_scenes)


def _gate(m: dict[str, Any], r: dict[str, Any], folds: list[dict[str, Any]], key: str) -> dict[str, bool]:
    noop_target = (1.0 - NOOP_REDUCTION_MIN) * float(r["no_positive_opportunity_false_intervention_count"])
    cat_target = (1.0 - CATASTROPHE_REDUCTION_MIN) * float(r["catastrophic_count"])
    existence = bool(
        m["no_positive_opportunity_false_intervention_count"] <= noop_target + EPS
        and m["positive_capture_rate"] >= r["positive_capture_rate"] - CAPTURE_TOL - EPS
    )
    tail = bool(
        m["catastrophic_count"] <= cat_target + EPS
        and m["teacher_negative_rms"] <= r["teacher_negative_rms"] + EPS
        and m["teacher_improvement_sum"] >= -EPS
    )
    fold_direction = all(float(f[key]["teacher_improvement_sum"]) >= -EPS for f in folds)
    population = bool(m["selected_count"] >= 64 and m["selected_positive_count"] >= 32)
    return {"existence_and_capture": existence, "tail": tail, "all_folds_sum_nonnegative": fold_direction, "population": population, "pass": bool(existence and tail and fold_direction and population)}


def _nested(groups: dict[str, list[dict[str, Any]]], audit_csv: Path) -> dict[str, Any]:
    scene = _build(groups)
    names = ["rsmr", "affine", "orthogonal"]
    agg = {n: [] for n in names}; caps = {n: 0 for n in names}; noops = {n: 0 for n in names}; oppsels = {n: 0 for n in names}
    folds: list[dict[str, Any]] = []; audits: list[dict[str, Any]] = []
    total_opp = 0; total_noop = 0

    for k in range(FOLDS):
        test = [t for t in scene if _fold(t) == k]
        cal_fold = (k + 1) % FOLDS
        cal = [t for t in scene if _fold(t) == cal_fold]
        fit = [t for t in scene if _fold(t) not in {k, cal_fold}]
        rsm = _fit_regret_structured_margin(scene, fit)
        affine, residual, used = _fit_value_models(scene, cal, rsm)

        fv = {n: [] for n in names}; fc = {n: 0 for n in names}; fn = {n: 0 for n in names}; fo = {n: 0 for n in names}
        opp = 0; noop_scenes = 0; subset_ok = True; identity_ok = True
        for t in test:
            ss = scene[t]
            y = np.asarray([float(a["y"]) for a in ss], dtype=np.float64)
            has_opp = bool(np.any(y > 0.0)); opp += int(has_opp); noop_scenes += int(not has_opp)
            r_i, r_u, r_v = _evaluate(ss, rsm, None, None)
            a_i, _, a_v = _evaluate(ss, rsm, affine, None)
            o_i, _, o_v = _evaluate(ss, rsm, affine, residual)
            subset_ok = subset_ok and (a_i is None or r_i is not None) and (o_i is None or r_i is not None)
            identity_ok = identity_ok and (a_i is None or a_i == r_i) and (o_i is None or o_i == r_i)
            chosen = {"rsmr": r_i, "affine": a_i, "orthogonal": o_i}
            for n, idx in chosen.items():
                if idx is None:
                    continue
                yy = float(y[idx]); fv[n].append(yy); fc[n] += int(has_opp and yy > 0.0); fn[n] += int(not has_opp); fo[n] += int(has_opp)
            audits.append({
                "scenario_token": t, "outer_test_fold": k, "calibration_fold": cal_fold, "candidate_count": len(ss),
                "positive_opportunity": int(has_opp),
                "rsm_selected_action": -1 if r_i is None else int(ss[r_i]["action"]),
                "rsm_selected_score": float("nan") if r_i is None else r_u,
                "rsm_selected_teacher_improvement": float("nan") if r_i is None else float(y[r_i]),
                "affine_selected_action": -1 if a_i is None else int(ss[a_i]["action"]),
                "affine_value": a_v,
                "affine_selected_teacher_improvement": float("nan") if a_i is None else float(y[a_i]),
                "orthogonal_selected_action": -1 if o_i is None else int(ss[o_i]["action"]),
                "orthogonal_value": o_v,
                "orthogonal_selected_teacher_improvement": float("nan") if o_i is None else float(y[o_i]),
            })
        total_opp += opp; total_noop += noop_scenes
        fd: dict[str, Any] = {}
        for n in names:
            fd[n] = _metrics(fv[n], fc[n], opp, fn[n], fo[n], noop_scenes)
            agg[n].extend(fv[n]); caps[n] += fc[n]; noops[n] += fn[n]; oppsels[n] += fo[n]
        folds.append({
            "fold": k, "fit_scenes": len(fit), "value_calibration_scenes": len(cal), "test_scenes": len(test),
            "value_calibration_proposal_count": len(used), "affine_value_fit": affine, "orthogonal_residual_fit": residual,
            "rsmr_rank": fd["rsmr"], "affine_score_value": fd["affine"], "orthogonal_proposal_value": fd["orthogonal"],
            "monotone_subset_valid": subset_ok, "frozen_winner_identity_valid": identity_ok,
        })

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(audits[0]) if audits else []
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(audits)

    A = {n: _metrics(agg[n], caps[n], total_opp, noops[n], oppsels[n], total_noop) for n in names}
    r = A["rsmr"]; a = A["affine"]; o = A["orthogonal"]
    contracts = all(f["monotone_subset_valid"] and f["frozen_winner_identity_valid"] and f["value_calibration_proposal_count"] >= MIN_VALUE_CAL_PROPOSALS for f in folds)
    ag = _gate(a, r, folds, "affine_score_value"); og = _gate(o, r, folds, "orthogonal_proposal_value")
    train_pass = bool(contracts and og["pass"])
    if not contracts:
        diagnosis = "selected_policy_value_contract_or_calibration_population_invalid"
    elif not og["existence_and_capture"]:
        diagnosis = "proposal_conditioned_value_readout_does_not_resolve_intervention_existence_without_destroying_RSMR_capture"
    elif not og["tail"] or not og["all_folds_sum_nonnegative"]:
        diagnosis = "proposal_conditioned_value_readout_improves_existence_but_selected_tail_or_crossfold_direction_remains_unstable"
    elif not og["population"]:
        diagnosis = "proposal_value_population_too_small"
    else:
        diagnosis = "full_nested_train_pass" if train_pass else "proposal_value_gate_failed_unspecified"
    return {
        "folds": folds, "scene_audit_csv": str(audit_csv),
        "rsmr_rank_aggregate": r, "affine_score_value_aggregate": a, "orthogonal_proposal_value_aggregate": o,
        "affine_score_only_gate": ag, "orthogonal_proposal_value_gate": og,
        "monotone_frozen_winner_contract_valid": contracts, "train_gate_pass": train_pass, "failure_diagnosis": diagnosis,
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
        "enabled": True, "mode": "rank_only", "model_type": "v37_frozen_v34_rsmr_ordering",
        "base_feature_names": BASE_FEATURE_NAMES, "feature_names": FEATURE_NAMES,
        "feature_mean": [0.0] * len(FEATURE_NAMES), "feature_std": [float(x) for x in scale],
        "weights": [float(x) for x in w], "bias": 0.0, "ridge_lambda": RIDGE_LAMBDA,
        "leverage_inverse": [], "selection_scale_floor": 1.0, "require_positive_predicted_improvement": True,
        "no_fallback": True, "scene_reservation_enabled": False, "post_selection_value_enabled": False,
        "training_population": "TRAIN_only_incumbent_deployment_admissible_support_positive_direct_scenes",
        "training_target": "V34_teacher_best_vs_worst_cost_augmented_rival_structured_regret_margin",
        "proposal_operator": "frozen_RSMR_argmax_positive_score_with_incumbent_zero_pseudoitem",
    })
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.37-RSMR-FROZEN"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.37-RSMR-FROZEN"
    cfg.setdefault("experiment", {})["name"] = "v64_3_37_rsmr_frozen_order"
    cfg["experiment"]["algorithm"] = "V64.3.37 frozen V34 RSMR ordering for post-selection value calibration"
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="V64.3.37 nested frozen-order proposal-value diagnostic")
    ap.add_argument("--train-frontier-edges", required=True); ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-preserve-config", required=True); ap.add_argument("--output-rsmr-config", required=True)
    ap.add_argument("--output-report", required=True); ap.add_argument("--output-scene-audit", required=True)
    a = ap.parse_args()
    _, groups = _read_edges(Path(a.train_frontier_edges))
    nested = _nested(groups, Path(a.output_scene_audit))
    report = {
        "audit": "v64_3_37_eaf_icer_pvr_fit",
        "scientific_role": "TRAIN_only_frozen_RSMR_ordering_plus_score_affine_vs_score_orthogonal_selected_proposal_absolute_value_disambiguation",
        "frozen_train_scenes": len(groups), "direct_support_positive_training_scenes": len(_build(groups)),
        "ridge_lambda": RIDGE_LAMBDA, "value_calibration_proposal_min": MIN_VALUE_CAL_PROPOSALS,
        "mechanism_hypothesis": "V36 shows scene-common nonnegative reservations mainly suppress intervention frequency and are highly aliased with the RSMR top score. Freeze the V34 winner, reinterpret RSMR as a ranking estimand, and learn a distinct post-selection absolute teacher-improvement estimand. The affine arm tests scalar miscalibration; the orthogonal arm tests whether proposal-specific sign/value information survives in 19-D evidence directions discarded by scalar RSMR scoring.",
        "nested_crossfit": nested, "train_gate_pass": nested["train_gate_pass"],
        "train_gate_contract": {
            "post_selection_arm_monotone_subset_of_RSMR": True,
            "winner_identity_must_equal_RSMR_when_accepted": True,
            "noop_false_intervention_reduction_fraction_min": NOOP_REDUCTION_MIN,
            "capture_tolerance": CAPTURE_TOL,
            "catastrophe_reduction_fraction_min": CATASTROPHE_REDUCTION_MIN,
            "all_test_folds_selected_sum_nonnegative": True,
            "selected_min": 64, "positive_min": 32,
            "score_affine_arm_is_diagnostic_control": True,
            "orthogonal_residual_excludes_RSMR_score_direction": True,
            "no_runtime_threshold_lambda_or_feature_sweep": True,
        },
    }
    rp = Path(a.output_report); rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["train_gate_pass"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(f"V64.3.37 PVR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before CAL/fresh selection")

    scene = _build(groups); base = _base_cfg(a.base_config)
    pcfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    pcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"] = {"enabled": False}
    pcfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.37-PRESERVE-CONTROL"
    pcfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.37-PRESERVE-CONTROL"
    Path(a.output_preserve_config).write_text(yaml.safe_dump(pcfg, sort_keys=False))
    full_rsm = _fit_regret_structured_margin(scene, list(scene))
    _write_rsmr(base, a.output_rsmr_config, full_rsm)
    print(json.dumps({"pass": True, "output_rsmr_config": a.output_rsmr_config}, sort_keys=True))


if __name__ == "__main__":
    main()
