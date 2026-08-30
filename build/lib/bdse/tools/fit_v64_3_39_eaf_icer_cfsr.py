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
from bdse.tools.fit_v64_3_38_eaf_icer_davr import (
    CAPTURE_TOL,
    CAT,
    CATASTROPHE_REDUCTION_MIN,
    MIN_VALUE_CAL_PROPOSALS,
    NOOP_REDUCTION_MIN,
    _auc,
    _fit_affine_scalar,
    _value_diag,
    _write_dense,
    _write_rsmr,
)

EPS = 1.0e-12
# The mechanism claim is that inner cross-fitting must materially increase the
# honest selected-policy supervision population beyond the historical 64-sample
# minimum.  Three inner held-out fit folds therefore require >=3*64 examples.
MIN_INNER_OOF_SELECTED = 3 * MIN_VALUE_CAL_PROPOSALS


def _build(groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    scene = {t: _scene_samples(g) for t, g in groups.items()}
    return {t: ss for t, ss in scene.items() if ss}


def _dense_value(ss: list[dict[str, Any]], dense_model, idx: int) -> float:
    mu, _ = _predict_dense_value(ss, dense_model)
    if not (0 <= idx < len(mu)):
        raise RuntimeError("V64.3.39 dense value selected index out of range")
    return float(mu[idx])


def _fit_translation(pred: np.ndarray, y: np.ndarray, source: str) -> dict[str, Any]:
    pp = np.asarray(pred, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if pp.size != yy.size or yy.size < MIN_VALUE_CAL_PROPOSALS or np.any(~np.isfinite(pp)) or np.any(~np.isfinite(yy)):
        raise ValueError("V64.3.39 translation calibration population malformed or too small")
    bias = float(np.mean(yy - pp))
    out = pp + bias
    return {
        "selected_policy_bias": bias,
        "sample_count": int(yy.size),
        "fit_mse": float(np.mean((out - yy) ** 2)),
        "input_source": source,
        "training_target": "teacher_improvement_of_frozen_RSMR_selected_proposal",
        "operator": "unit_slope_translation_only_preserves_value_ordering",
    }


def _raw_linear_direction_rsmr(model) -> np.ndarray:
    w, scale, _ = model
    return np.asarray(w, dtype=np.float64) / np.maximum(np.asarray(scale, dtype=np.float64), 1.0e-6)


def _raw_linear_direction_dense(model) -> np.ndarray:
    w, _, _, std, _ = model
    return np.asarray(w, dtype=np.float64) / np.maximum(np.asarray(std, dtype=np.float64), 1.0e-6)


def _orthonormal_span(cols: list[np.ndarray]) -> np.ndarray:
    good: list[np.ndarray] = []
    for c in cols:
        v = np.asarray(c, dtype=np.float64).reshape(-1)
        if np.all(np.isfinite(v)) and float(np.linalg.norm(v)) > 1.0e-10:
            good.append(v)
    if not good:
        return np.zeros((len(FEATURE_NAMES), 0), dtype=np.float64)
    M = np.stack(good, axis=1)
    u, s, _ = np.linalg.svd(M, full_matrices=False)
    keep = s > max(float(s[0]) * 1.0e-10, 1.0e-12)
    return u[:, keep]


def _inner_oof_selected_residuals(
    scene: dict[str, list[dict[str, Any]]],
    fit_tokens: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_ids = sorted({_fold(t) for t in fit_tokens})
    if len(fold_ids) != 3:
        raise ValueError(f"V64.3.39 expected exactly 3 outer-fit folds, got {fold_ids}")
    rows: list[dict[str, Any]] = []
    per_fold: list[dict[str, Any]] = []
    for h in fold_ids:
        held = [t for t in fit_tokens if _fold(t) == h]
        train = [t for t in fit_tokens if _fold(t) != h]
        rsm = _fit_regret_structured_margin(scene, train)
        dense = _fit_dense_value_ridge([a for t in train for a in scene[t]])
        selected = 0
        for t in held:
            ss = scene[t]
            score = _structured_scores(ss, rsm)
            idx = _select(ss, score)
            if idx is None:
                continue
            dv = _dense_value(ss, dense, idx)
            y = float(ss[idx]["y"])
            rows.append({
                "token": t,
                "inner_holdout_fold": int(h),
                "x": np.asarray(ss[idx]["x"], dtype=np.float64),
                "y": y,
                "dense_oof": dv,
                "residual": y - dv,
            })
            selected += 1
        per_fold.append({"inner_holdout_fold": int(h), "train_scene_count": len(train), "heldout_scene_count": len(held), "selected_proposal_count": selected})
    if len({r["token"] for r in rows}) != len(rows):
        raise RuntimeError("V64.3.39 inner OOF selected-policy tokens are not unique")
    return rows, {"inner_fold_reports": per_fold, "selected_proposal_count": len(rows), "fit_fold_ids": fold_ids}


def _fit_selection_residual(
    rows: list[dict[str, Any]],
    final_rsm,
    final_dense,
) -> dict[str, Any]:
    if len(rows) < MIN_INNER_OOF_SELECTED:
        raise ValueError(f"V64.3.39 inner OOF selected residual population {len(rows)} < {MIN_INNER_OOF_SELECTED}")
    X = np.stack([np.asarray(r["x"], dtype=np.float64) for r in rows])
    e = np.asarray([float(r["residual"]) for r in rows], dtype=np.float64)
    if X.shape[1] != len(FEATURE_NAMES) or np.any(~np.isfinite(X)) or np.any(~np.isfinite(e)):
        raise ValueError("V64.3.39 inner OOF residual population malformed")
    mean = np.mean(X, axis=0)
    std = np.maximum(np.sqrt(np.mean((X - mean[None, :]) ** 2, axis=0)), 1.0e-6)
    Z = (X - mean[None, :]) / std[None, :]

    # Express final outer-fit RSMR and DENSE directions in this selected-residual
    # standardized coordinate system, then remove their span.  The correction is
    # therefore structurally unable to relearn either the ordinal ranking scalar
    # or the dense pointwise cardinal scalar.
    dr = _raw_linear_direction_rsmr(final_rsm) * std
    dd = _raw_linear_direction_dense(final_dense) * std
    Q = _orthonormal_span([dr, dd])
    if Q.shape[0] != Z.shape[1]:
        raise RuntimeError("V64.3.39 residual orthogonal span dimension mismatch")
    Zp = Z - (Z @ Q) @ Q.T if Q.shape[1] else Z
    yc = e - float(np.mean(e))
    A = Zp.T @ Zp + np.eye(Zp.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    coef = np.linalg.solve(A, Zp.T @ yc)
    if Q.shape[1]:
        coef = coef - Q @ (Q.T @ coef)
    pred = Z @ coef
    before = float(np.mean(e * e))
    after = float(np.mean((pred - e) ** 2))
    max_orth = 0.0
    for d in [dr, dd]:
        den = float(np.linalg.norm(coef) * np.linalg.norm(d))
        if den > EPS:
            max_orth = max(max_orth, abs(float(np.dot(coef, d))) / den)
    return {
        "feature_mean": mean,
        "feature_std": std,
        "weights": coef,
        "ridge_lambda": RIDGE_LAMBDA,
        "sample_count": int(len(rows)),
        "target_residual_mean": float(np.mean(e)),
        "target_residual_rms": float(np.sqrt(np.mean(e * e))),
        "oof_dense_residual_mse_before_feature_correction": before,
        "oof_dense_residual_mse_after_feature_correction": after,
        "removed_span_rank": int(Q.shape[1]),
        "max_abs_cosine_to_removed_rank_dense_span": float(max_orth),
        "operator": "dense_value_plus_cross_fitted_selected_policy_feature_residual_orthogonal_to_rank_and_dense_directions",
    }


def _residual_value(ss: list[dict[str, Any]], dense_model, residual_model: dict[str, Any], idx: int) -> float:
    dv = _dense_value(ss, dense_model, idx)
    x = np.asarray(ss[idx]["x"], dtype=np.float64)
    mean = np.asarray(residual_model["feature_mean"], dtype=np.float64)
    std = np.asarray(residual_model["feature_std"], dtype=np.float64)
    w = np.asarray(residual_model["weights"], dtype=np.float64)
    z = (x - mean) / np.maximum(std, 1.0e-6)
    return float(np.clip(dv + z @ w, -40.0, 40.0))


def _cal_population(scene, tokens, rsm, dense, residual_model):
    ys: list[float] = []
    dvs: list[float] = []
    cfs: list[float] = []
    used: list[str] = []
    for t in tokens:
        ss = scene.get(t, [])
        if not ss:
            continue
        score = _structured_scores(ss, rsm)
        idx = _select(ss, score)
        if idx is None:
            continue
        ys.append(float(ss[idx]["y"]))
        dvs.append(_dense_value(ss, dense, idx))
        cfs.append(_residual_value(ss, dense, residual_model, idx))
        used.append(t)
    if not ys:
        return np.zeros(0), np.zeros(0), np.zeros(0), []
    return np.asarray(ys), np.asarray(dvs), np.asarray(cfs), used


def _evaluate(ss, rsm, dense, residual_model, dense_bias: float, cfsr_bias: float):
    score = _structured_scores(ss, rsm)
    idx = _select(ss, score)
    if idx is None:
        return {n: (None, float("nan")) for n in ["rsmr", "dense", "dense_shift", "cfsr_raw", "cfsr_main"]}
    u = float(score[idx])
    dv = _dense_value(ss, dense, idx)
    cv = _residual_value(ss, dense, residual_model, idx)
    ds = dv + float(dense_bias)
    cm = cv + float(cfsr_bias)
    return {
        "rsmr": (idx, u),
        "dense": (idx if dv > 0.0 else None, dv),
        "dense_shift": (idx if ds > 0.0 else None, ds),
        "cfsr_raw": (idx if cv > 0.0 else None, cv),
        "cfsr_main": (idx if cm > 0.0 else None, cm),
    }


def _metrics(vals, captured, opp, noop_selected, opp_selected, noop_scenes):
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
    names = ["rsmr", "dense", "dense_shift", "cfsr_raw", "cfsr_main"]
    agg = {n: [] for n in names}; caps = {n: 0 for n in names}; noops = {n: 0 for n in names}; oppsels = {n: 0 for n in names}
    value_y: list[float] = []; value_pred = {n: [] for n in names}
    folds: list[dict[str, Any]] = []; audits: list[dict[str, Any]] = []
    total_opp = 0; total_noop = 0

    for k in range(FOLDS):
        test = [t for t in scene if _fold(t) == k]
        cal_fold = (k + 1) % FOLDS
        cal = [t for t in scene if _fold(t) == cal_fold]
        fit = [t for t in scene if _fold(t) not in {k, cal_fold}]
        rsm = _fit_regret_structured_margin(scene, fit)
        dense = _fit_dense_value_ridge([a for t in fit for a in scene[t]])
        oof_rows, oof_info = _inner_oof_selected_residuals(scene, fit)
        residual = _fit_selection_residual(oof_rows, rsm, dense)
        ycal, dcal, ccal, used = _cal_population(scene, cal, rsm, dense, residual)
        if len(used) < MIN_VALUE_CAL_PROPOSALS:
            raise ValueError(f"V64.3.39 selected-policy calibration has {len(used)} proposals < {MIN_VALUE_CAL_PROPOSALS}")
        dense_shift = _fit_translation(dcal, ycal, "dense_all_edge_absolute_value_prediction")
        cfsr_shift = _fit_translation(ccal, ycal, "cross_fitted_selection_residual_corrected_value")

        fv = {n: [] for n in names}; fc = {n: 0 for n in names}; fn = {n: 0 for n in names}; fo = {n: 0 for n in names}
        fold_value_y: list[float] = []; fold_value_pred = {n: [] for n in names}
        opp = 0; noop_scenes = 0; subset_ok = True; identity_ok = True
        for t in test:
            ss = scene[t]
            yy = np.asarray([float(a["y"]) for a in ss], dtype=np.float64)
            has_opp = bool(np.any(yy > 0.0)); opp += int(has_opp); noop_scenes += int(not has_opp)
            ev = _evaluate(ss, rsm, dense, residual, dense_shift["selected_policy_bias"], cfsr_shift["selected_policy_bias"])
            r_idx, _ = ev["rsmr"]
            if r_idx is not None:
                true = float(yy[r_idx]); fold_value_y.append(true); value_y.append(true)
                for n in names:
                    pred = float(ev[n][1]); fold_value_pred[n].append(pred); value_pred[n].append(pred)
            chosen = {n: ev[n][0] for n in names}
            subset_ok = subset_ok and all(chosen[n] is None or r_idx is not None for n in names if n != "rsmr")
            identity_ok = identity_ok and all(chosen[n] is None or chosen[n] == r_idx for n in names if n != "rsmr")
            for n, idx in chosen.items():
                if idx is None:
                    continue
                val = float(yy[idx]); fv[n].append(val); fc[n] += int(has_opp and val > 0.0); fn[n] += int(not has_opp); fo[n] += int(has_opp)
            audits.append({
                "scenario_token": t, "outer_test_fold": k, "calibration_fold": cal_fold, "candidate_count": len(ss), "positive_opportunity": int(has_opp),
                "rsm_selected_action": -1 if r_idx is None else int(ss[r_idx]["action"]),
                "rsm_selected_score": float(ev["rsmr"][1]),
                "rsm_selected_teacher_improvement": float("nan") if r_idx is None else float(yy[r_idx]),
                "dense_selected_action": -1 if chosen["dense"] is None else int(ss[chosen["dense"]]["action"]), "dense_value": float(ev["dense"][1]),
                "dense_shift_selected_action": -1 if chosen["dense_shift"] is None else int(ss[chosen["dense_shift"]]["action"]), "dense_shift_value": float(ev["dense_shift"][1]),
                "cfsr_raw_selected_action": -1 if chosen["cfsr_raw"] is None else int(ss[chosen["cfsr_raw"]]["action"]), "cfsr_raw_value": float(ev["cfsr_raw"][1]),
                "cfsr_main_selected_action": -1 if chosen["cfsr_main"] is None else int(ss[chosen["cfsr_main"]]["action"]), "cfsr_main_value": float(ev["cfsr_main"][1]),
            })
        total_opp += opp; total_noop += noop_scenes
        fd: dict[str, Any] = {}
        for n in names:
            fd[n] = _metrics(fv[n], fc[n], opp, fn[n], fo[n], noop_scenes)
            agg[n].extend(fv[n]); caps[n] += fc[n]; noops[n] += fn[n]; oppsels[n] += fo[n]
        folds.append({
            "fold": k, "fit_scenes": len(fit), "value_calibration_scenes": len(cal), "test_scenes": len(test),
            "value_calibration_proposal_count": len(used), "inner_oof_selected_residual_population": oof_info,
            "selection_residual_fit": {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in residual.items()},
            "dense_translation_fit": dense_shift, "cfsr_translation_fit": cfsr_shift,
            "rsmr_rank": fd["rsmr"], "dense_all_edge_value": fd["dense"], "dense_translation_control": fd["dense_shift"],
            "cfsr_raw": fd["cfsr_raw"], "cfsr_main": fd["cfsr_main"],
            "value_prediction_diagnostics": {n: _value_diag(fold_value_y, fold_value_pred[n]) for n in names},
            "monotone_subset_valid": subset_ok, "frozen_winner_identity_valid": identity_ok,
        })

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(audits[0]) if audits else []
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(audits)

    A = {n: _metrics(agg[n], caps[n], total_opp, noops[n], oppsels[n], total_noop) for n in names}
    r = A["rsmr"]
    contracts = all(
        f["monotone_subset_valid"] and f["frozen_winner_identity_valid"]
        and f["value_calibration_proposal_count"] >= MIN_VALUE_CAL_PROPOSALS
        and f["inner_oof_selected_residual_population"]["selected_proposal_count"] >= MIN_INNER_OOF_SELECTED
        and float(f["selection_residual_fit"]["max_abs_cosine_to_removed_rank_dense_span"]) <= 1.0e-8
        for f in folds
    )
    dg = _gate(A["dense"], r, folds, "dense_all_edge_value")
    sg = _gate(A["dense_shift"], r, folds, "dense_translation_control")
    rg = _gate(A["cfsr_raw"], r, folds, "cfsr_raw")
    mg = _gate(A["cfsr_main"], r, folds, "cfsr_main")
    value_diag = {n: _value_diag(value_y, value_pred[n]) for n in names}
    train_pass = bool(contracts and mg["pass"])

    if not contracts:
        diagnosis = "cross_fitted_selected_residual_contract_or_population_invalid"
    elif sg["pass"] and not mg["pass"]:
        diagnosis = "translation_only_selected_policy_shift_suffices_residual_mechanism_unnecessary_or_harmful"
    elif mg["pass"]:
        diagnosis = "full_nested_train_pass_cross_fitted_selection_residual_closes_dense_selected_tail_gap"
    elif value_diag["cfsr_raw"].get("noncatastrophe_auc", -math.inf) > value_diag["dense"].get("noncatastrophe_auc", -math.inf) + 0.05:
        diagnosis = "selection_residual_tail_signal_exists_but_zero_crossing_capture_tradeoff_remains"
    elif value_diag["dense"].get("positive_auc", -math.inf) > value_diag["rsmr"].get("positive_auc", -math.inf) + 0.03:
        diagnosis = "dense_cardinal_sign_signal_exists_but_cross_fitted_selection_residual_does_not_identify_tail_reliably"
    else:
        diagnosis = "current_19D_representation_does_not_support_stable_selected_absolute_value_or_tail_identification"

    return {
        "folds": folds,
        "scene_audit_csv": str(audit_csv),
        "rsmr_rank_aggregate": A["rsmr"],
        "dense_all_edge_value_aggregate": A["dense"],
        "dense_translation_control_aggregate": A["dense_shift"],
        "cfsr_raw_aggregate": A["cfsr_raw"],
        "cfsr_main_aggregate": A["cfsr_main"],
        "selected_proposal_value_prediction_diagnostics": value_diag,
        "dense_gate": dg, "translation_control_gate": sg, "cfsr_raw_gate": rg, "cfsr_main_gate": mg,
        "monotone_frozen_winner_contract_valid": contracts,
        "train_gate_pass": train_pass,
        "failure_diagnosis": diagnosis,
        "inner_oof_selected_min": MIN_INNER_OOF_SELECTED,
        "noop_reduction_fraction_min": NOOP_REDUCTION_MIN,
        "capture_tolerance": CAPTURE_TOL,
        "catastrophe_reduction_fraction_min": CATASTROPHE_REDUCTION_MIN,
    }


def _base_cfg(path: str) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    ic = cfg.setdefault("runtime", {}).setdefault("decisive_frontier_value", {}).setdefault("incumbent_contrastive_extremal_recovery", {})
    ic["incumbent_retention_policy"] = "preserve_admissible_incumbent"
    ic["regret_risk_enabled"] = False; ic["retention_regret_risk_enabled"] = False; ic["replacement_regret_risk_enabled"] = False
    return cfg


def _decorate_residual(dense_cfg: dict[str, Any], residual: dict[str, Any], path: str) -> None:
    cfg = yaml.safe_load(yaml.safe_dump(dense_cfg, sort_keys=False))
    sc = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    sc.update({
        "post_selection_value_enabled": True,
        "post_selection_value_mode": "dense_edge_cfsr",
        "post_selection_cfsr_feature_mean": [float(x) for x in residual["feature_mean"]],
        "post_selection_cfsr_feature_std": [float(x) for x in residual["feature_std"]],
        "post_selection_cfsr_weights": [float(x) for x in residual["weights"]],
        "post_selection_cfsr_bias": 0.0,
        "post_selection_cfsr_training": "cross_fitted_frozen_RSMR_selected_policy_residuals_from_TRAIN_only_orthogonal_to_rank_and_dense_directions",
        "post_selection_cfsr_inner_oof_selected_count": int(residual["sample_count"]),
        "post_selection_cfsr_max_abs_cosine_to_removed_rank_dense_span": float(residual["max_abs_cosine_to_removed_rank_dense_span"]),
        "post_selection_operator": "freeze_RSMR_winner_then_dense_cardinal_value_plus_cross_fitted_selection_residual_on_same_winner_only_no_rerank_no_fallback",
        "post_selection_value_max_abs": 40.0,
    })
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.39-EAF-ICER-CFSR-RAW"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.39-EAF-ICER-CFSR-RAW"
    cfg.setdefault("experiment", {})["name"] = "v64_3_39_cfsr_raw"
    cfg["experiment"]["algorithm"] = "V64.3.39 frozen RSMR + dense all-edge value + cross-fitted orthogonal selected-policy residual"
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


def _full_selection_residual(scene, full_rsm, full_dense) -> dict[str, Any]:
    # Full TRAIN uses five-way OOF policy outputs: each held-out historical fold is
    # selected by models trained on the other four folds.  This is the deployment
    # residual population used before the independent CAL500 translation shift.
    rows: list[dict[str, Any]] = []
    per: list[dict[str, Any]] = []
    toks = list(scene)
    for h in range(FOLDS):
        held = [t for t in toks if _fold(t) == h]
        train = [t for t in toks if _fold(t) != h]
        rsm = _fit_regret_structured_margin(scene, train)
        dense = _fit_dense_value_ridge([a for t in train for a in scene[t]])
        n = 0
        for t in held:
            ss = scene[t]; score = _structured_scores(ss, rsm); idx = _select(ss, score)
            if idx is None: continue
            dv = _dense_value(ss, dense, idx); y = float(ss[idx]["y"])
            rows.append({"token": t, "x": np.asarray(ss[idx]["x"], dtype=np.float64), "y": y, "dense_oof": dv, "residual": y-dv})
            n += 1
        per.append({"holdout_fold": h, "selected_proposal_count": n})
    model = _fit_selection_residual(rows, full_rsm, full_dense)
    model["full_train_oof_fold_reports"] = per
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="V64.3.39 cross-fitted selection-residual nested diagnostic")
    ap.add_argument("--train-frontier-edges", required=True); ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-preserve-config", required=True); ap.add_argument("--output-rsmr-config", required=True); ap.add_argument("--output-dense-config", required=True); ap.add_argument("--output-cfsr-config", required=True)
    ap.add_argument("--output-report", required=True); ap.add_argument("--output-scene-audit", required=True)
    a = ap.parse_args()
    _, groups = _read_edges(Path(a.train_frontier_edges))
    nested = _nested(groups, Path(a.output_scene_audit))
    report = {
        "audit": "v64_3_39_eaf_icer_cfsr_fit",
        "scientific_role": "TRAIN_only_frozen_RSMR_ranking_plus_dense_all_edge_cardinal_base_plus_cross_fitted_selected_policy_residual_adaptation",
        "frozen_train_scenes": len(groups), "direct_support_positive_training_scenes": len(_build(groups)), "ridge_lambda": RIDGE_LAMBDA,
        "value_calibration_proposal_min": MIN_VALUE_CAL_PROPOSALS, "inner_oof_selected_min": MIN_INNER_OOF_SELECTED,
        "mechanism_hypothesis": "V38 shows complementary estimands: dense all-edge value improves ordinary selected-proposal sign AUC but catastrophically misorders the selected tail, while V37 score-orthogonal selected residual contains tail signal but is unstable under only 86-110 selected samples. Generate honest frozen-policy residual labels by inner cross-fitting across all outer-fit scenes, fit only the residual component orthogonal to both final RSMR and dense linear directions, and use independent calibration only for a unit-slope translation so learned value ordering cannot be sign-flipped.",
        "nested_crossfit": nested, "train_gate_pass": nested["train_gate_pass"],
        "train_gate_contract": {
            "RSMR_is_sole_challenger_selector": True,
            "dense_and_CFSR_never_rerank_or_create_proposals": True,
            "inner_selected_residual_targets_are_out_of_fold": True,
            "selection_residual_is_orthogonal_to_final_rank_and_dense_directions": True,
            "selected_policy_calibration_is_translation_only_unit_slope": True,
            "inner_oof_selected_min": MIN_INNER_OOF_SELECTED,
            "noop_false_intervention_reduction_fraction_min": NOOP_REDUCTION_MIN,
            "capture_tolerance": CAPTURE_TOL,
            "catastrophe_reduction_fraction_min": CATASTROPHE_REDUCTION_MIN,
            "all_test_folds_selected_sum_nonnegative": True,
            "selected_min": 64, "positive_min": 32,
            "no_runtime_threshold_lambda_feature_or_temperature_sweep": True,
        },
    }
    rp = Path(a.output_report); rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["train_gate_pass"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(f"V64.3.39 CFSR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before CAL/fresh selection")

    scene = _build(groups); base = _base_cfg(a.base_config)
    pcfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    pcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"] = {"enabled": False}
    pcfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.39-PRESERVE-CONTROL"; pcfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.39-PRESERVE-CONTROL"
    Path(a.output_preserve_config).write_text(yaml.safe_dump(pcfg, sort_keys=False))
    full_rsm = _fit_regret_structured_margin(scene, list(scene)); rsm_cfg = _write_rsmr(base, a.output_rsmr_config, full_rsm)
    full_dense = _fit_dense_value_ridge([x for ss in scene.values() for x in ss]); _write_dense(rsm_cfg, a.output_dense_config, full_dense)
    dense_cfg = yaml.safe_load(Path(a.output_dense_config).read_text())
    full_residual = _full_selection_residual(scene, full_rsm, full_dense)
    _decorate_residual(dense_cfg, full_residual, a.output_cfsr_config)
    print(json.dumps({"pass": True, "output_rsmr_config": a.output_rsmr_config, "output_dense_config": a.output_dense_config, "output_cfsr_config": a.output_cfsr_config, "full_train_oof_selected": full_residual["sample_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
