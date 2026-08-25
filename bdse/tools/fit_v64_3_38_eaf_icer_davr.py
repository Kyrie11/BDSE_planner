from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.tools.fit_v64_3_32_1_eaf_icer_ssir_weightfix import _fit_ridge as _fit_dense_value_ridge
from bdse.tools.fit_v64_3_32_1_eaf_icer_ssir_weightfix import _predict as _predict_dense_value
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import (
    BASE_FEATURE_NAMES,
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
CAT = -0.5


def _build(groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    scene = {t: _scene_samples(g) for t, g in groups.items()}
    return {t: ss for t, ss in scene.items() if ss}


def _fit_affine_scalar(x: np.ndarray, y: np.ndarray, source: str) -> dict[str, Any]:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if xx.size != yy.size or yy.size < MIN_VALUE_CAL_PROPOSALS or np.any(~np.isfinite(xx)) or np.any(~np.isfinite(yy)):
        raise ValueError("V64.3.38 affine selected-policy value population malformed or too small")
    mean_x = float(np.mean(xx))
    scale_x = max(float(np.sqrt(np.mean((xx - mean_x) ** 2))), 1.0e-6)
    z = (xx - mean_x) / scale_x
    mean_y = float(np.mean(yy))
    yc = yy - mean_y
    slope = float(np.dot(z, yc) / (np.dot(z, z) + RIDGE_LAMBDA))
    pred = mean_y + slope * z
    return {
        "input_mean": mean_x,
        "input_std": scale_x,
        "intercept": mean_y,
        "input_weight": slope,
        "ridge_lambda": RIDGE_LAMBDA,
        "sample_count": int(yy.size),
        "fit_mse": float(np.mean((pred - yy) ** 2)),
        "target_mean": mean_y,
        "target_rms": float(np.sqrt(np.mean(yy * yy))),
        "prediction_mean": float(np.mean(pred)),
        "input_source": source,
        "training_target": "teacher_improvement_of_frozen_RSMR_selected_proposal",
        "operator": "post_selection_absolute_value_readout_no_rerank",
    }


def _affine_scalar(x: float, model: dict[str, Any]) -> float:
    z = (float(x) - float(model["input_mean"])) / max(float(model["input_std"]), 1.0e-6)
    return float(model["intercept"] + float(model["input_weight"]) * z)


def _dense_value(ss: list[dict[str, Any]], dense_model, idx: int) -> float:
    mu, _ = _predict_dense_value(ss, dense_model)
    if not (0 <= idx < len(mu)):
        raise RuntimeError("V64.3.38 dense value selected index out of range")
    return float(mu[idx])


def _cal_samples(scene, tokens, rsm_model, dense_model):
    us: list[float] = []
    dvs: list[float] = []
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
        us.append(float(score[idx]))
        dvs.append(_dense_value(ss, dense_model, idx))
        ys.append(float(ss[idx]["y"]))
        used.append(t)
    if not ys:
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), []
    return np.asarray(us), np.asarray(dvs), np.asarray(ys), used


def _evaluate(ss, rsm_model, dense_model, score_affine=None, dense_affine=None):
    score = _structured_scores(ss, rsm_model)
    idx = _select(ss, score)
    if idx is None:
        return {"rsmr": (None, float("nan")), "avr": (None, float("nan")), "dense": (None, float("nan")), "davr": (None, float("nan"))}
    u = float(score[idx])
    dv = _dense_value(ss, dense_model, idx)
    av = u if score_affine is None else _affine_scalar(u, score_affine)
    cv = dv if dense_affine is None else _affine_scalar(dv, dense_affine)
    return {
        "rsmr": (idx, u),
        "avr": (idx if av > 0.0 else None, av),
        "dense": (idx if dv > 0.0 else None, dv),
        "davr": (idx if cv > 0.0 else None, cv),
    }


def _metrics(vals, captured, opp, noop_selected, opp_selected, noop_scenes):
    return _extended_diag(vals, captured, opp, noop_selected, opp_selected, noop_scenes)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    yy = np.asarray(y, dtype=bool).reshape(-1)
    ss = np.asarray(score, dtype=np.float64).reshape(-1)
    p = ss[yy]; n = ss[~yy]
    if p.size == 0 or n.size == 0:
        return float("nan")
    # Exact pairwise AUC; populations here are only frozen RSMR proposals.
    return float((np.sum(p[:, None] > n[None, :]) + 0.5 * np.sum(p[:, None] == n[None, :])) / (p.size * n.size))


def _value_diag(y: list[float], pred: list[float]) -> dict[str, Any]:
    yy = np.asarray(y, dtype=np.float64); pp = np.asarray(pred, dtype=np.float64)
    if yy.size == 0:
        return {"sample_count": 0}
    mse = float(np.mean((pp - yy) ** 2))
    corr = float(np.corrcoef(pp, yy)[0, 1]) if yy.size > 1 and np.std(pp) > 0 and np.std(yy) > 0 else float("nan")
    return {
        "sample_count": int(yy.size),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(pp - yy))),
        "pearson": corr,
        "positive_auc": _auc(yy > 0.0, pp),
        "noncatastrophe_auc": _auc(yy > CAT, pp),
        "zero_sign_accuracy": float(np.mean((pp > 0.0) == (yy > 0.0))),
    }


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
    names = ["rsmr", "avr", "dense", "davr"]
    agg = {n: [] for n in names}; caps = {n: 0 for n in names}; noops = {n: 0 for n in names}; oppsels = {n: 0 for n in names}
    value_y: list[float] = []; value_pred = {"rsmr": [], "avr": [], "dense": [], "davr": []}
    folds: list[dict[str, Any]] = []; audits: list[dict[str, Any]] = []
    total_opp = 0; total_noop = 0

    for k in range(FOLDS):
        test = [t for t in scene if _fold(t) == k]
        cal_fold = (k + 1) % FOLDS
        cal = [t for t in scene if _fold(t) == cal_fold]
        fit = [t for t in scene if _fold(t) not in {k, cal_fold}]
        rsm = _fit_regret_structured_margin(scene, fit)
        fit_samples = [a for t in fit for a in scene[t]]
        dense = _fit_dense_value_ridge(fit_samples)
        ucal, dcal, ycal, used = _cal_samples(scene, cal, rsm, dense)
        if len(used) < MIN_VALUE_CAL_PROPOSALS:
            raise ValueError(f"V64.3.38 selected-policy calibration has {len(used)} proposals < {MIN_VALUE_CAL_PROPOSALS}")
        score_affine = _fit_affine_scalar(ucal, ycal, "frozen_RSMR_scalar_score")
        dense_affine = _fit_affine_scalar(dcal, ycal, "dense_all_edge_absolute_value_prediction")

        fv = {n: [] for n in names}; fc = {n: 0 for n in names}; fn = {n: 0 for n in names}; fo = {n: 0 for n in names}
        fold_value_y: list[float] = []; fold_value_pred = {n: [] for n in names}
        opp = 0; noop_scenes = 0; subset_ok = True; identity_ok = True
        for t in test:
            ss = scene[t]
            yy = np.asarray([float(a["y"]) for a in ss], dtype=np.float64)
            has_opp = bool(np.any(yy > 0.0)); opp += int(has_opp); noop_scenes += int(not has_opp)
            ev = _evaluate(ss, rsm, dense, score_affine, dense_affine)
            r_idx, r_u = ev["rsmr"]
            if r_idx is not None:
                true = float(yy[r_idx])
                fold_value_y.append(true); value_y.append(true)
                for n in names:
                    pred = float(ev[n][1])
                    fold_value_pred[n].append(pred); value_pred[n].append(pred)
            chosen = {n: ev[n][0] for n in names}
            subset_ok = subset_ok and all(chosen[n] is None or r_idx is not None for n in ["avr", "dense", "davr"])
            identity_ok = identity_ok and all(chosen[n] is None or chosen[n] == r_idx for n in ["avr", "dense", "davr"])
            for n, idx in chosen.items():
                if idx is None:
                    continue
                val = float(yy[idx]); fv[n].append(val); fc[n] += int(has_opp and val > 0.0); fn[n] += int(not has_opp); fo[n] += int(has_opp)
            audits.append({
                "scenario_token": t, "outer_test_fold": k, "calibration_fold": cal_fold, "candidate_count": len(ss), "positive_opportunity": int(has_opp),
                "rsm_selected_action": -1 if r_idx is None else int(ss[r_idx]["action"]),
                "rsm_selected_score": float("nan") if r_idx is None else float(r_u),
                "rsm_selected_teacher_improvement": float("nan") if r_idx is None else float(yy[r_idx]),
                "avr_selected_action": -1 if chosen["avr"] is None else int(ss[chosen["avr"]]["action"]),
                "avr_value": float(ev["avr"][1]),
                "dense_selected_action": -1 if chosen["dense"] is None else int(ss[chosen["dense"]]["action"]),
                "dense_value": float(ev["dense"][1]),
                "davr_selected_action": -1 if chosen["davr"] is None else int(ss[chosen["davr"]]["action"]),
                "davr_value": float(ev["davr"][1]),
            })
        total_opp += opp; total_noop += noop_scenes
        fd: dict[str, Any] = {}
        for n in names:
            fd[n] = _metrics(fv[n], fc[n], opp, fn[n], fo[n], noop_scenes)
            agg[n].extend(fv[n]); caps[n] += fc[n]; noops[n] += fn[n]; oppsels[n] += fo[n]
        folds.append({
            "fold": k, "fit_scenes": len(fit), "dense_value_fit_candidate_count": len(fit_samples), "value_calibration_scenes": len(cal), "test_scenes": len(test),
            "value_calibration_proposal_count": len(used), "score_affine_fit": score_affine, "dense_selected_policy_affine_fit": dense_affine,
            "rsmr_rank": fd["rsmr"], "avr_score_value": fd["avr"], "dense_all_edge_value": fd["dense"], "davr_selected_calibrated_value": fd["davr"],
            "value_prediction_diagnostics": {n: _value_diag(fold_value_y, fold_value_pred[n]) for n in names},
            "monotone_subset_valid": subset_ok, "frozen_winner_identity_valid": identity_ok,
        })

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(audits[0]) if audits else []
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(audits)

    A = {n: _metrics(agg[n], caps[n], total_opp, noops[n], oppsels[n], total_noop) for n in names}
    r = A["rsmr"]; d = A["dense"]; m = A["davr"]
    contracts = all(f["monotone_subset_valid"] and f["frozen_winner_identity_valid"] and f["value_calibration_proposal_count"] >= MIN_VALUE_CAL_PROPOSALS for f in folds)
    dg = _gate(d, r, folds, "dense_all_edge_value")
    mg = _gate(m, r, folds, "davr_selected_calibrated_value")
    train_pass = bool(contracts and mg["pass"])
    value_diag = {n: _value_diag(value_y, value_pred[n]) for n in names}
    if not contracts:
        diagnosis = "rank_value_contract_or_calibration_population_invalid"
    elif not mg["existence_and_capture"]:
        if value_diag["dense"].get("mse", math.inf) >= value_diag["avr"].get("mse", math.inf) - EPS:
            diagnosis = "dense_all_edge_value_does_not_improve_selected_absolute_value_identification_or_capture_tradeoff"
        else:
            diagnosis = "dense_absolute_value_signal_exists_but_selected_zero_crossing_still_destroys_RSMR_capture"
    elif not mg["tail"] or not mg["all_folds_sum_nonnegative"]:
        diagnosis = "rank_value_factorization_improves_existence_but_selected_tail_or_crossfold_direction_remains_unstable"
    elif not mg["population"]:
        diagnosis = "rank_value_factorized_population_too_small"
    else:
        diagnosis = "full_nested_train_pass" if train_pass else "rank_value_factorized_gate_failed_unspecified"
    return {
        "folds": folds, "scene_audit_csv": str(audit_csv),
        "rsmr_rank_aggregate": r, "avr_score_value_aggregate": A["avr"], "dense_all_edge_value_aggregate": d, "davr_selected_calibrated_value_aggregate": m,
        "selected_proposal_value_prediction_diagnostics": value_diag,
        "dense_raw_gate": dg, "davr_main_gate": mg,
        "monotone_frozen_winner_contract_valid": contracts, "train_gate_pass": train_pass, "failure_diagnosis": diagnosis,
        "noop_reduction_fraction_min": NOOP_REDUCTION_MIN, "capture_tolerance": CAPTURE_TOL, "catastrophe_reduction_fraction_min": CATASTROPHE_REDUCTION_MIN,
    }


def _base_cfg(path: str) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    ic = cfg.setdefault("runtime", {}).setdefault("decisive_frontier_value", {}).setdefault("incumbent_contrastive_extremal_recovery", {})
    ic["incumbent_retention_policy"] = "preserve_admissible_incumbent"
    ic["regret_risk_enabled"] = False; ic["retention_regret_risk_enabled"] = False; ic["replacement_regret_risk_enabled"] = False
    return cfg


def _write_rsmr(base: dict[str, Any], path: str, model) -> dict[str, Any]:
    w, scale, _ = model
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    sc = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"].setdefault("selection_conditioned_intervention_recovery", {})
    sc.update({
        "enabled": True, "mode": "rank_only", "model_type": "v38_frozen_v34_rsmr_ordering",
        "base_feature_names": BASE_FEATURE_NAMES, "feature_names": FEATURE_NAMES,
        "feature_mean": [0.0] * len(FEATURE_NAMES), "feature_std": [float(x) for x in scale], "weights": [float(x) for x in w], "bias": 0.0,
        "ridge_lambda": RIDGE_LAMBDA, "leverage_inverse": [], "selection_scale_floor": 1.0, "require_positive_predicted_improvement": True,
        "no_fallback": True, "scene_reservation_enabled": False, "post_selection_value_enabled": False,
        "training_population": "TRAIN_only_incumbent_deployment_admissible_support_positive_direct_scenes",
        "training_target": "V34_teacher_best_vs_worst_cost_augmented_rival_structured_regret_margin",
        "proposal_operator": "frozen_RSMR_argmax_positive_score_with_incumbent_zero_pseudoitem",
    })
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.38-RSMR-FROZEN"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.38-RSMR-FROZEN"
    cfg.setdefault("experiment", {})["name"] = "v64_3_38_rsmr_frozen_order"
    cfg["experiment"]["algorithm"] = "V64.3.38 frozen V34 RSMR ordering for decoupled dense absolute value"
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def _write_dense(rsm_cfg: dict[str, Any], path: str, dense_model) -> None:
    w, b, mean, std, _ = dense_model
    cfg = yaml.safe_load(yaml.safe_dump(rsm_cfg, sort_keys=False))
    sc = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    sc.update({
        "post_selection_value_enabled": True,
        "post_selection_value_mode": "dense_edge_value",
        "post_selection_dense_feature_mean": [float(x) for x in mean],
        "post_selection_dense_feature_std": [float(x) for x in std],
        "post_selection_dense_weights": [float(x) for x in w],
        "post_selection_dense_bias": float(b),
        "post_selection_dense_ridge_lambda": RIDGE_LAMBDA,
        "post_selection_dense_training": "all_fit_candidate_edges_scene_equal_total_loss_mass_1_using_corrected_V32_1_objective",
        "post_selection_value_target": "absolute_teacher_improvement_candidate_minus_incumbent",
        "post_selection_operator": "freeze_RSMR_winner_then_evaluate_dense_absolute_value_on_same_winner_only_no_rerank_no_fallback",
        "post_selection_value_max_abs": 40.0,
    })
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.38-EAF-ICER-DENSE-VALUE"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.38-EAF-ICER-DENSE-VALUE"
    cfg.setdefault("experiment", {})["name"] = "v64_3_38_dense_all_edge_value"
    cfg["experiment"]["algorithm"] = "V64.3.38 frozen RSMR ranking + dense scene-equal all-edge absolute value readout"
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="V64.3.38 nested rank-value factorization diagnostic")
    ap.add_argument("--train-frontier-edges", required=True); ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-preserve-config", required=True); ap.add_argument("--output-rsmr-config", required=True); ap.add_argument("--output-dense-config", required=True)
    ap.add_argument("--output-report", required=True); ap.add_argument("--output-scene-audit", required=True)
    a = ap.parse_args()
    _, groups = _read_edges(Path(a.train_frontier_edges))
    nested = _nested(groups, Path(a.output_scene_audit))
    report = {
        "audit": "v64_3_38_eaf_icer_davr_fit",
        "scientific_role": "TRAIN_only_frozen_RSMR_ranking_plus_dense_all_edge_absolute_value_identification_plus_low_dim_selected_policy_recalibration",
        "frozen_train_scenes": len(groups), "direct_support_positive_training_scenes": len(_build(groups)), "ridge_lambda": RIDGE_LAMBDA,
        "value_calibration_proposal_min": MIN_VALUE_CAL_PROPOSALS,
        "mechanism_hypothesis": "V37 selected-only 18-D orthogonal residual reduces some tail errors but is statistically unstable and destroys capture. Reuse dense candidate-level teacher-improvement supervision with the corrected scene-equal V32.1 objective to identify cardinal value, while freezing V34 RSMR for ordinal challenger ranking. An independent selected-policy scalar recalibration then tests selection shift without high-dimensional selected-only fitting.",
        "nested_crossfit": nested, "train_gate_pass": nested["train_gate_pass"],
        "train_gate_contract": {
            "post_selection_arms_monotone_subset_of_RSMR": True, "winner_identity_must_equal_RSMR_when_accepted": True,
            "dense_value_head_never_reranks": True, "dense_value_uses_all_fit_candidate_labels_scene_equal": True,
            "selected_policy_recalibration_is_one_dimensional": True,
            "noop_false_intervention_reduction_fraction_min": NOOP_REDUCTION_MIN, "capture_tolerance": CAPTURE_TOL,
            "catastrophe_reduction_fraction_min": CATASTROPHE_REDUCTION_MIN, "all_test_folds_selected_sum_nonnegative": True,
            "selected_min": 64, "positive_min": 32, "no_runtime_threshold_lambda_or_feature_sweep": True,
        },
    }
    rp = Path(a.output_report); rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["train_gate_pass"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(f"V64.3.38 DAVR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before CAL/fresh selection")

    scene = _build(groups); base = _base_cfg(a.base_config)
    pcfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    pcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"] = {"enabled": False}
    pcfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.38-PRESERVE-CONTROL"; pcfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.38-PRESERVE-CONTROL"
    Path(a.output_preserve_config).write_text(yaml.safe_dump(pcfg, sort_keys=False))
    full_rsm = _fit_regret_structured_margin(scene, list(scene)); rsm_cfg = _write_rsmr(base, a.output_rsmr_config, full_rsm)
    full_samples = [x for ss in scene.values() for x in ss]; full_dense = _fit_dense_value_ridge(full_samples); _write_dense(rsm_cfg, a.output_dense_config, full_dense)
    print(json.dumps({"pass": True, "output_rsmr_config": a.output_rsmr_config, "output_dense_config": a.output_dense_config}, sort_keys=True))


if __name__ == "__main__":
    main()
