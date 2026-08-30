from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.tools.fit_v64_3_32_1_eaf_icer_ssir_weightfix import _fit_ridge as _fit_scene_equal_ridge
from bdse.tools.fit_v64_3_32_1_eaf_icer_ssir_weightfix import _predict as _predict_scene_equal_ridge
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS, RIDGE_LAMBDA, _fold, _read_edges, _scene_samples, _select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _extended_diag, _fit_regret_structured_margin, _structured_scores
from bdse.tools.fit_v64_3_38_eaf_icer_davr import (
    CAT,
    CAPTURE_TOL,
    CATASTROPHE_REDUCTION_MIN,
    MIN_VALUE_CAL_PROPOSALS,
    NOOP_REDUCTION_MIN,
    _build,
    _fit_dense_value_ridge,
    _value_diag,
    _write_dense,
    _write_rsmr,
)
from bdse.tools.fit_v64_3_39_eaf_icer_cfsr import _fit_translation

EPS = 1.0e-12
MIN_INNER_SELECTED = 3 * MIN_VALUE_CAL_PROPOSALS
MIN_SELECTED_SIGN_CLASS = 64
PROB_EPS = 1.0e-4


def _retarget(samples: list[dict[str, Any]], target) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in samples:
        b = dict(a)
        b["y"] = float(target(float(a["y"])))
        out.append(b)
    return out


def _fit_hurdle_base(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("V64.3.40 hurdle base has no samples")
    sign = _fit_scene_equal_ridge(_retarget(samples, lambda y: 1.0 if y > 0.0 else 0.0))
    pos_src = [a for a in samples if float(a["y"]) > 0.0]
    neg_src = [a for a in samples if float(a["y"]) <= 0.0]
    if not pos_src or not neg_src:
        raise ValueError("V64.3.40 hurdle base needs both positive and non-positive training edges")
    pos = _fit_scene_equal_ridge(_retarget(pos_src, lambda y: y))
    neg = _fit_scene_equal_ridge(_retarget(neg_src, lambda y: -y))
    return {"sign": sign, "positive_magnitude": pos, "negative_magnitude": neg}


def _predict_one(ss: list[dict[str, Any]], model, idx: int) -> float:
    mu, _ = _predict_scene_equal_ridge(ss, model)
    if not (0 <= idx < len(mu)):
        raise RuntimeError("V64.3.40 selected index out of range")
    return float(mu[idx])


def _hurdle_components(ss: list[dict[str, Any]], model: dict[str, Any], idx: int) -> tuple[float, float, float]:
    p = float(np.clip(_predict_one(ss, model["sign"], idx), PROB_EPS, 1.0 - PROB_EPS))
    mp = float(np.clip(_predict_one(ss, model["positive_magnitude"], idx), 0.0, 40.0))
    mn = float(np.clip(_predict_one(ss, model["negative_magnitude"], idx), 0.0, 40.0))
    return p, mp, mn


def _hurdle_value(p: float, mp: float, mn: float) -> float:
    return float(np.clip(float(p) * float(mp) - (1.0 - float(p)) * float(mn), -40.0, 40.0))


def _logit(p: float) -> float:
    q = min(max(float(p), PROB_EPS), 1.0 - PROB_EPS)
    return math.log(q / (1.0 - q))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        e = math.exp(-min(x, 60.0)); return 1.0 / (1.0 + e)
    e = math.exp(max(x, -60.0)); return e / (1.0 + e)


def _fit_logit_offset(base_p: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(np.asarray(base_p, dtype=np.float64).reshape(-1), PROB_EPS, 1.0 - PROB_EPS)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.size != y.size or p.size < MIN_INNER_SELECTED or np.any(~np.isfinite(p)) or np.any(~np.isfinite(y)):
        raise ValueError("V64.3.40 selected sign calibration population malformed")
    off = np.log(p / (1.0 - p))
    c = 0.0
    for _ in range(80):
        z = np.clip(off + c, -60.0, 60.0)
        pr = 1.0 / (1.0 + np.exp(-z))
        g = float(np.sum(pr - y))
        h = float(np.sum(pr * (1.0 - pr)))
        step = g / max(h, 1.0e-9)
        c -= step
        if abs(step) < 1.0e-12:
            break
    return float(c)


def _fit_nonnegative_scale(pred: np.ndarray, target: np.ndarray, min_n: int) -> float:
    x = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < min_n or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("V64.3.40 selected magnitude calibration population malformed or too small")
    den = float(np.dot(x, x)) + EPS
    return float(max(0.0, np.dot(x, y) / den))


def _fit_selected_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < MIN_INNER_SELECTED:
        raise ValueError(f"V64.3.40 inner OOF selected population {len(rows)} < {MIN_INNER_SELECTED}")
    p = np.asarray([r["p"] for r in rows], dtype=np.float64)
    y = np.asarray([r["y"] for r in rows], dtype=np.float64)
    mp = np.asarray([r["mp"] for r in rows], dtype=np.float64)
    mn = np.asarray([r["mn"] for r in rows], dtype=np.float64)
    pos = y > 0.0
    neg = ~pos
    if int(pos.sum()) < MIN_SELECTED_SIGN_CLASS or int(neg.sum()) < MIN_SELECTED_SIGN_CLASS:
        raise ValueError(
            f"V64.3.40 selected sign classes too small pos={int(pos.sum())} neg={int(neg.sum())} < {MIN_SELECTED_SIGN_CLASS}"
        )
    logit_shift = _fit_logit_offset(p, pos.astype(np.float64))
    pos_scale = _fit_nonnegative_scale(mp[pos], y[pos], MIN_SELECTED_SIGN_CLASS)
    neg_scale = _fit_nonnegative_scale(mn[neg], -y[neg], MIN_SELECTED_SIGN_CLASS)
    ps = np.asarray([_sigmoid(_logit(v) + logit_shift) for v in p], dtype=np.float64)
    raw = p * mp - (1.0 - p) * mn
    sign_only = ps * mp - (1.0 - ps) * mn
    full = ps * (pos_scale * mp) - (1.0 - ps) * (neg_scale * mn)
    return {
        "sample_count": int(len(rows)),
        "positive_count": int(pos.sum()),
        "nonpositive_count": int(neg.sum()),
        "selected_logit_shift": float(logit_shift),
        "selected_positive_magnitude_scale": float(pos_scale),
        "selected_negative_magnitude_scale": float(neg_scale),
        "raw_hurdle_mse": float(np.mean((raw - y) ** 2)),
        "sign_shift_only_mse": float(np.mean((sign_only - y) ** 2)),
        "selected_distribution_mse": float(np.mean((full - y) ** 2)),
        "operator": "cross_fitted_selected_policy_hurdle_distribution_scalar_adaptation",
        "identity": "E[Y|selected,X]=P(Y>0|selected,X)E[Y|Y>0,selected,X]-P(Y<=0|selected,X)E[-Y|Y<=0,selected,X]",
    }


def _apply_selected_distribution(p: float, mp: float, mn: float, adapt: dict[str, Any], sign_only: bool = False) -> tuple[float, float, float, float]:
    ps = _sigmoid(_logit(p) + float(adapt["selected_logit_shift"]))
    if sign_only:
        pms, nms = mp, mn
    else:
        pms = float(adapt["selected_positive_magnitude_scale"]) * mp
        nms = float(adapt["selected_negative_magnitude_scale"]) * mn
    return _hurdle_value(ps, pms, nms), ps, float(pms), float(nms)


def _inner_oof_selected(scene: dict[str, list[dict[str, Any]]], fit_tokens: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_ids = sorted({_fold(t) for t in fit_tokens})
    if len(fold_ids) != 3:
        raise ValueError(f"V64.3.40 expected exactly 3 outer-fit folds, got {fold_ids}")
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for h in fold_ids:
        held = [t for t in fit_tokens if _fold(t) == h]
        train = [t for t in fit_tokens if _fold(t) != h]
        rsm = _fit_regret_structured_margin(scene, train)
        samples = [a for t in train for a in scene[t]]
        hurdle = _fit_hurdle_base(samples)
        n = 0
        for t in held:
            ss = scene[t]
            score = _structured_scores(ss, rsm)
            idx = _select(ss, score)
            if idx is None:
                continue
            p, mp, mn = _hurdle_components(ss, hurdle, idx)
            rows.append({"token": t, "inner_holdout_fold": int(h), "y": float(ss[idx]["y"]), "p": p, "mp": mp, "mn": mn})
            n += 1
        reports.append({"inner_holdout_fold": int(h), "train_scene_count": len(train), "heldout_scene_count": len(held), "selected_proposal_count": n})
    if len({r["token"] for r in rows}) != len(rows):
        raise RuntimeError("V64.3.40 inner OOF selected tokens are not unique")
    return rows, {"inner_fold_reports": reports, "selected_proposal_count": len(rows), "fit_fold_ids": fold_ids}


def _dense_value(ss: list[dict[str, Any]], dense_model, idx: int) -> float:
    return _predict_one(ss, dense_model, idx)


def _cal_population(scene, tokens, rsm, dense, hurdle, adapt):
    y: list[float] = []; raw: list[float] = []; used: list[str] = []
    for t in tokens:
        ss = scene.get(t, [])
        if not ss: continue
        score = _structured_scores(ss, rsm); idx = _select(ss, score)
        if idx is None: continue
        p, mp, mn = _hurdle_components(ss, hurdle, idx)
        sv, _, _, _ = _apply_selected_distribution(p, mp, mn, adapt, sign_only=False)
        y.append(float(ss[idx]["y"])); raw.append(sv); used.append(t)
    return np.asarray(y, dtype=np.float64), np.asarray(raw, dtype=np.float64), used


def _evaluate(ss, rsm, dense, hurdle, adapt, main_bias: float):
    score = _structured_scores(ss, rsm); idx = _select(ss, score)
    names = ["rsmr", "dense", "hurdle", "sign_shift", "sdfr_raw", "sdfr_main"]
    if idx is None:
        return {n: (None, float("nan")) for n in names}, {}
    u = float(score[idx]); dv = _dense_value(ss, dense, idx)
    p, mp, mn = _hurdle_components(ss, hurdle, idx)
    hv = _hurdle_value(p, mp, mn)
    sv, ps, _, _ = _apply_selected_distribution(p, mp, mn, adapt, sign_only=True)
    fv, ps2, mps, mns = _apply_selected_distribution(p, mp, mn, adapt, sign_only=False)
    mv = float(np.clip(fv + float(main_bias), -40.0, 40.0))
    ev = {
        "rsmr": (idx, u),
        "dense": (idx if dv > 0.0 else None, dv),
        "hurdle": (idx if hv > 0.0 else None, hv),
        "sign_shift": (idx if sv > 0.0 else None, sv),
        "sdfr_raw": (idx if fv > 0.0 else None, fv),
        "sdfr_main": (idx if mv > 0.0 else None, mv),
    }
    dist = {"base_positive_probability": p, "selected_positive_probability": ps2, "base_positive_magnitude": mp, "base_negative_magnitude": mn, "selected_positive_magnitude": mps, "selected_negative_magnitude": mns}
    return ev, dist


def _metrics(vals, captured, opp, noop_selected, opp_selected, noop_scenes):
    return _extended_diag(vals, captured, opp, noop_selected, opp_selected, noop_scenes)


def _gate(m: dict[str, Any], r: dict[str, Any], folds: list[dict[str, Any]], key: str) -> dict[str, bool]:
    noop_target = (1.0 - NOOP_REDUCTION_MIN) * float(r["no_positive_opportunity_false_intervention_count"])
    cat_target = (1.0 - CATASTROPHE_REDUCTION_MIN) * float(r["catastrophic_count"])
    existence = bool(m["no_positive_opportunity_false_intervention_count"] <= noop_target + EPS and m["positive_capture_rate"] >= r["positive_capture_rate"] - CAPTURE_TOL - EPS)
    tail = bool(m["catastrophic_count"] <= cat_target + EPS and m["teacher_negative_rms"] <= r["teacher_negative_rms"] + EPS and m["teacher_improvement_sum"] >= -EPS)
    fold_direction = all(float(f[key]["teacher_improvement_sum"]) >= -EPS for f in folds)
    population = bool(m["selected_count"] >= 64 and m["selected_positive_count"] >= 32)
    return {"existence_and_capture": existence, "tail": tail, "all_folds_sum_nonnegative": fold_direction, "population": population, "pass": bool(existence and tail and fold_direction and population)}


def _distribution_diag(y: list[float], prob: list[float], posmag: list[float], negmag: list[float]) -> dict[str, Any]:
    yy = np.asarray(y, dtype=np.float64); pp = np.asarray(prob, dtype=np.float64); pm = np.asarray(posmag, dtype=np.float64); nm = np.asarray(negmag, dtype=np.float64)
    if yy.size == 0: return {"sample_count": 0}
    pos = yy > 0.0; neg = ~pos
    brier = float(np.mean((pp - pos.astype(np.float64)) ** 2))
    def auc(labels, score):
        from bdse.tools.fit_v64_3_38_eaf_icer_davr import _auc
        return _auc(labels, score)
    out = {"sample_count": int(yy.size), "positive_probability_brier": brier, "positive_probability_auc": auc(pos, pp)}
    if np.any(pos): out["positive_magnitude_mse_on_positive"] = float(np.mean((pm[pos] - yy[pos]) ** 2))
    if np.any(neg): out["negative_magnitude_mse_on_nonpositive"] = float(np.mean((nm[neg] - (-yy[neg])) ** 2))
    if np.any(yy <= CAT): out["negative_magnitude_mean_on_catastrophe"] = float(np.mean(nm[yy <= CAT]))
    return out


def _nested(groups: dict[str, list[dict[str, Any]]], audit_csv: Path) -> dict[str, Any]:
    scene = _build(groups)
    names = ["rsmr", "dense", "hurdle", "sign_shift", "sdfr_raw", "sdfr_main"]
    agg = {n: [] for n in names}; caps = {n: 0 for n in names}; noops = {n: 0 for n in names}; oppsels = {n: 0 for n in names}
    value_y: list[float] = []; value_pred = {n: [] for n in names}
    dist_y: list[float] = []; dist_p: list[float] = []; dist_pm: list[float] = []; dist_nm: list[float] = []
    folds: list[dict[str, Any]] = []; audits: list[dict[str, Any]] = []
    total_opp = 0; total_noop = 0

    for k in range(FOLDS):
        test = [t for t in scene if _fold(t) == k]; cal_fold = (k + 1) % FOLDS
        cal = [t for t in scene if _fold(t) == cal_fold]; fit = [t for t in scene if _fold(t) not in {k, cal_fold}]
        rsm = _fit_regret_structured_margin(scene, fit)
        samples = [a for t in fit for a in scene[t]]
        dense = _fit_dense_value_ridge(samples)
        hurdle = _fit_hurdle_base(samples)
        oof_rows, oof_info = _inner_oof_selected(scene, fit)
        adapt = _fit_selected_distribution(oof_rows)
        ycal, rawcal, used = _cal_population(scene, cal, rsm, dense, hurdle, adapt)
        if len(used) < MIN_VALUE_CAL_PROPOSALS:
            raise ValueError(f"V64.3.40 selected-policy calibration has {len(used)} proposals < {MIN_VALUE_CAL_PROPOSALS}")
        shift = _fit_translation(rawcal, ycal, "cross_fitted_selected_distribution_value")

        fv = {n: [] for n in names}; fc = {n: 0 for n in names}; fn = {n: 0 for n in names}; fo = {n: 0 for n in names}
        fold_value_y: list[float] = []; fold_value_pred = {n: [] for n in names}
        fy: list[float] = []; fp: list[float] = []; fpm: list[float] = []; fnm: list[float] = []
        opp = 0; noop_scenes = 0; subset_ok = True; identity_ok = True
        for t in test:
            ss = scene[t]; yy = np.asarray([float(a["y"]) for a in ss], dtype=np.float64)
            has_opp = bool(np.any(yy > 0.0)); opp += int(has_opp); noop_scenes += int(not has_opp)
            ev, dd = _evaluate(ss, rsm, dense, hurdle, adapt, shift["selected_policy_bias"])
            r_idx, _ = ev["rsmr"]
            if r_idx is not None:
                true = float(yy[r_idx]); fold_value_y.append(true); value_y.append(true)
                for n in names: fold_value_pred[n].append(float(ev[n][1])); value_pred[n].append(float(ev[n][1]))
                fy.append(true); fp.append(float(dd["selected_positive_probability"])); fpm.append(float(dd["selected_positive_magnitude"])); fnm.append(float(dd["selected_negative_magnitude"]))
                dist_y.append(true); dist_p.append(float(dd["selected_positive_probability"])); dist_pm.append(float(dd["selected_positive_magnitude"])); dist_nm.append(float(dd["selected_negative_magnitude"]))
            chosen = {n: ev[n][0] for n in names}
            subset_ok = subset_ok and all(chosen[n] is None or r_idx is not None for n in names if n != "rsmr")
            identity_ok = identity_ok and all(chosen[n] is None or chosen[n] == r_idx for n in names if n != "rsmr")
            for n, idx in chosen.items():
                if idx is None: continue
                val = float(yy[idx]); fv[n].append(val); fc[n] += int(has_opp and val > 0.0); fn[n] += int(not has_opp); fo[n] += int(has_opp)
            audits.append({
                "scenario_token": t, "outer_test_fold": k, "calibration_fold": cal_fold, "candidate_count": len(ss), "positive_opportunity": int(has_opp),
                "rsm_selected_action": -1 if r_idx is None else int(ss[r_idx]["action"]), "rsm_selected_score": float(ev["rsmr"][1]), "rsm_selected_teacher_improvement": float("nan") if r_idx is None else float(yy[r_idx]),
                "dense_selected_action": -1 if chosen["dense"] is None else int(ss[chosen["dense"]]["action"]), "dense_value": float(ev["dense"][1]),
                "hurdle_selected_action": -1 if chosen["hurdle"] is None else int(ss[chosen["hurdle"]]["action"]), "hurdle_value": float(ev["hurdle"][1]),
                "sign_shift_selected_action": -1 if chosen["sign_shift"] is None else int(ss[chosen["sign_shift"]]["action"]), "sign_shift_value": float(ev["sign_shift"][1]),
                "sdfr_raw_selected_action": -1 if chosen["sdfr_raw"] is None else int(ss[chosen["sdfr_raw"]]["action"]), "sdfr_raw_value": float(ev["sdfr_raw"][1]),
                "sdfr_main_selected_action": -1 if chosen["sdfr_main"] is None else int(ss[chosen["sdfr_main"]]["action"]), "sdfr_main_value": float(ev["sdfr_main"][1]),
                "selected_positive_probability": float(dd.get("selected_positive_probability", float("nan"))), "selected_positive_magnitude": float(dd.get("selected_positive_magnitude", float("nan"))), "selected_negative_magnitude": float(dd.get("selected_negative_magnitude", float("nan"))),
            })
        total_opp += opp; total_noop += noop_scenes
        fd: dict[str, Any] = {}
        for n in names:
            fd[n] = _metrics(fv[n], fc[n], opp, fn[n], fo[n], noop_scenes); agg[n].extend(fv[n]); caps[n] += fc[n]; noops[n] += fn[n]; oppsels[n] += fo[n]
        folds.append({
            "fold": k, "fit_scenes": len(fit), "value_calibration_scenes": len(cal), "test_scenes": len(test), "value_calibration_proposal_count": len(used),
            "inner_oof_selected_distribution_population": oof_info, "selected_distribution_fit": adapt, "sdfr_translation_fit": shift,
            "rsmr_rank": fd["rsmr"], "dense_signed_mean": fd["dense"], "hurdle_dense_distribution": fd["hurdle"], "hurdle_sign_shift_only": fd["sign_shift"], "sdfr_raw": fd["sdfr_raw"], "sdfr_main": fd["sdfr_main"],
            "value_prediction_diagnostics": {n: _value_diag(fold_value_y, fold_value_pred[n]) for n in names}, "distribution_diagnostics": _distribution_diag(fy, fp, fpm, fnm),
            "monotone_subset_valid": subset_ok, "frozen_winner_identity_valid": identity_ok,
        })

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(audits[0]) if audits else []
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(audits)
    A = {n: _metrics(agg[n], caps[n], total_opp, noops[n], oppsels[n], total_noop) for n in names}; r = A["rsmr"]
    contracts = all(
        f["monotone_subset_valid"] and f["frozen_winner_identity_valid"] and f["value_calibration_proposal_count"] >= MIN_VALUE_CAL_PROPOSALS
        and f["inner_oof_selected_distribution_population"]["selected_proposal_count"] >= MIN_INNER_SELECTED
        and int(f["selected_distribution_fit"]["positive_count"]) >= MIN_SELECTED_SIGN_CLASS and int(f["selected_distribution_fit"]["nonpositive_count"]) >= MIN_SELECTED_SIGN_CLASS
        for f in folds
    )
    gates = {
        "dense": _gate(A["dense"], r, folds, "dense_signed_mean"),
        "hurdle": _gate(A["hurdle"], r, folds, "hurdle_dense_distribution"),
        "sign_shift": _gate(A["sign_shift"], r, folds, "hurdle_sign_shift_only"),
        "sdfr_raw": _gate(A["sdfr_raw"], r, folds, "sdfr_raw"),
        "sdfr_main": _gate(A["sdfr_main"], r, folds, "sdfr_main"),
    }
    value_diag = {n: _value_diag(value_y, value_pred[n]) for n in names}
    dist_diag = _distribution_diag(dist_y, dist_p, dist_pm, dist_nm)
    train_pass = bool(contracts and gates["sdfr_main"]["pass"])
    if not contracts:
        diagnosis = "selected_distribution_crossfit_contract_or_class_population_invalid"
    elif gates["hurdle"]["pass"] and not gates["sdfr_main"]["pass"]:
        diagnosis = "population_hurdle_factorization_suffices_selected_adaptation_unnecessary_or_harmful"
    elif gates["sdfr_main"]["pass"]:
        diagnosis = "full_nested_train_pass_selected_distribution_factorization_closes_zero_crossing_tail_tradeoff"
    elif gates["sign_shift"]["existence_and_capture"] and not gates["sdfr_raw"]["existence_and_capture"]:
        diagnosis = "selected_sign_frequency_is_primary_mediator_magnitude_adaptation_degrades_recovery"
    elif value_diag["hurdle"].get("positive_auc", -math.inf) > value_diag["dense"].get("positive_auc", -math.inf) + 0.03:
        diagnosis = "distributional_target_improves_selected_sign_identification_but_selected_policy_adaptation_or_tail_remains"
    elif value_diag["sdfr_raw"].get("noncatastrophe_auc", -math.inf) > value_diag["hurdle"].get("noncatastrophe_auc", -math.inf) + 0.05:
        diagnosis = "selected_distribution_adaptation_adds_tail_signal_but_zero_crossing_capture_tradeoff_remains"
    else:
        diagnosis = "current_19D_selected_value_representation_remains_insufficient_after_distribution_factorization"
    return {
        "folds": folds, "scene_audit_csv": str(audit_csv), "rsmr_rank_aggregate": A["rsmr"], "dense_signed_mean_aggregate": A["dense"], "hurdle_dense_distribution_aggregate": A["hurdle"],
        "hurdle_sign_shift_only_aggregate": A["sign_shift"], "sdfr_raw_aggregate": A["sdfr_raw"], "sdfr_main_aggregate": A["sdfr_main"],
        "selected_proposal_value_prediction_diagnostics": value_diag, "selected_distribution_diagnostics": dist_diag, "gates": gates,
        "monotone_frozen_winner_contract_valid": contracts, "train_gate_pass": train_pass, "failure_diagnosis": diagnosis,
        "inner_oof_selected_min": MIN_INNER_SELECTED, "selected_sign_class_min": MIN_SELECTED_SIGN_CLASS,
        "noop_reduction_fraction_min": NOOP_REDUCTION_MIN, "capture_tolerance": CAPTURE_TOL, "catastrophe_reduction_fraction_min": CATASTROPHE_REDUCTION_MIN,
    }


def _base_cfg(path: str) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    ic = cfg.setdefault("runtime", {}).setdefault("decisive_frontier_value", {}).setdefault("incumbent_contrastive_extremal_recovery", {})
    ic["incumbent_retention_policy"] = "preserve_admissible_incumbent"; ic["regret_risk_enabled"] = False; ic["retention_regret_risk_enabled"] = False; ic["replacement_regret_risk_enabled"] = False
    return cfg


def _model_fields(model) -> dict[str, Any]:
    w, b, mean, std, _ = model
    return {"feature_mean": [float(x) for x in mean], "feature_std": [float(x) for x in std], "weights": [float(x) for x in w], "bias": float(b)}


def _decorate_hurdle(rsm_cfg: dict[str, Any], hurdle: dict[str, Any], adapt: dict[str, Any] | None, mode: str, path: str, version: str) -> None:
    cfg = yaml.safe_load(yaml.safe_dump(rsm_cfg, sort_keys=False)); sc = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    sc.update({"post_selection_value_enabled": True, "post_selection_value_mode": mode, "post_selection_value_max_abs": 40.0, "post_selection_hurdle_probability_clip": PROB_EPS,
               "post_selection_hurdle_training": "scene_equal_all_edge_hurdle_distribution_fixed_lambda_1_frozen_RSMR_winner_only",
               "post_selection_operator": "freeze_RSMR_winner_then_reconstruct_selected_expected_intervention_value_from_sign_probability_and_conditional_positive_negative_magnitudes_no_rerank_no_fallback"})
    for prefix, key in [("sign", "sign"), ("positive_magnitude", "positive_magnitude"), ("negative_magnitude", "negative_magnitude")]:
        f = _model_fields(hurdle[key])
        for kk, vv in f.items(): sc[f"post_selection_hurdle_{prefix}_{kk}"] = vv
    if adapt is not None:
        sc.update({"post_selection_hurdle_selected_logit_shift": float(adapt["selected_logit_shift"]), "post_selection_hurdle_selected_positive_magnitude_scale": float(adapt["selected_positive_magnitude_scale"]), "post_selection_hurdle_selected_negative_magnitude_scale": float(adapt["selected_negative_magnitude_scale"]),
                   "post_selection_hurdle_selected_training": "cross_fitted_frozen_RSMR_policy_outputs_from_TRAIN_only_scalar_component_adaptation", "post_selection_hurdle_selected_sample_count": int(adapt["sample_count"])})
    cfg.setdefault("metadata", {})["algorithm_version"] = version; cfg.setdefault("provenance", {})["algorithm_version"] = version
    cfg.setdefault("experiment", {})["name"] = version.lower().replace(".", "_").replace("-", "_"); cfg["experiment"]["algorithm"] = version
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


def _full_selected_distribution(scene, full_hurdle) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []; per: list[dict[str, Any]] = []; toks = list(scene)
    for h in range(FOLDS):
        held = [t for t in toks if _fold(t) == h]; train = [t for t in toks if _fold(t) != h]
        rsm = _fit_regret_structured_margin(scene, train); hurdle = _fit_hurdle_base([a for t in train for a in scene[t]])
        n = 0
        for t in held:
            ss = scene[t]; score = _structured_scores(ss, rsm); idx = _select(ss, score)
            if idx is None: continue
            p, mp, mn = _hurdle_components(ss, hurdle, idx); rows.append({"token": t, "y": float(ss[idx]["y"]), "p": p, "mp": mp, "mn": mn}); n += 1
        per.append({"holdout_fold": h, "selected_proposal_count": n})
    model = _fit_selected_distribution(rows); model["full_train_oof_fold_reports"] = per
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="V64.3.40 selected-distribution factorized recovery")
    ap.add_argument("--train-frontier-edges", required=True); ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-preserve-config", required=True); ap.add_argument("--output-rsmr-config", required=True); ap.add_argument("--output-dense-config", required=True)
    ap.add_argument("--output-hurdle-config", required=True); ap.add_argument("--output-sign-shift-config", required=True); ap.add_argument("--output-sdfr-config", required=True)
    ap.add_argument("--output-report", required=True); ap.add_argument("--output-scene-audit", required=True)
    a = ap.parse_args(); _, groups = _read_edges(Path(a.train_frontier_edges)); nested = _nested(groups, Path(a.output_scene_audit))
    report = {
        "audit": "v64_3_40_eaf_icer_sdfr_fit", "scientific_role": "TRAIN_only_frozen_RSMR_plus_distribution_factorized_selected_value_recovery",
        "frozen_train_scenes": len(groups), "direct_support_positive_training_scenes": len(_build(groups)), "ridge_lambda": RIDGE_LAMBDA,
        "mechanism_hypothesis": "V39 proves cross-fitted selected residual contains real tail signal but signed-mean zero crossing loses many near-zero positives. Test the same 19-D representation with an exact hurdle distribution identity separating beneficial-event probability from conditional positive and negative magnitudes. Use dense all-edge scene-equal supervision for each component and only scalar cross-fitted selected-policy component adaptation; no nonlinear or higher-dimensional selected head is added.",
        "nested_crossfit": nested, "train_gate_pass": nested["train_gate_pass"],
        "train_gate_contract": {"RSMR_is_sole_challenger_selector": True, "all_value_arms_are_same_winner_subsets": True, "distribution_identity_has_no_tuned_tradeoff_weight": True, "dense_component_heads_use_fixed_lambda_1_scene_equal_objectives": True,
            "selected_policy_adaptation_is_scalar_per_distribution_component": True, "selected_probability_uses_logit_intercept_only": True, "selected_magnitude_adaptation_uses_nonnegative_scale_only": True,
            "inner_oof_selected_min": MIN_INNER_SELECTED, "selected_sign_class_min": MIN_SELECTED_SIGN_CLASS, "noop_false_intervention_reduction_fraction_min": NOOP_REDUCTION_MIN, "capture_tolerance": CAPTURE_TOL,
            "catastrophe_reduction_fraction_min": CATASTROPHE_REDUCTION_MIN, "all_test_folds_selected_sum_nonnegative": True, "selected_min": 64, "positive_min": 32,
            "no_threshold_lambda_alpha_feature_candidate_count_or_temperature_sweep": True},
    }
    rp = Path(a.output_report); rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["train_gate_pass"]:
        print(json.dumps(report, indent=2, sort_keys=True)); raise SystemExit(f"V64.3.40 SDFR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before CAL/fresh selection")
    scene = _build(groups); base = _base_cfg(a.base_config)
    pcfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False)); pcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"] = {"enabled": False}
    pcfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.40-PRESERVE-CONTROL"; pcfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.40-PRESERVE-CONTROL"; Path(a.output_preserve_config).write_text(yaml.safe_dump(pcfg, sort_keys=False))
    full_rsm = _fit_regret_structured_margin(scene, list(scene)); rsm_cfg = _write_rsmr(base, a.output_rsmr_config, full_rsm)
    samples = [a for ss in scene.values() for a in ss]; full_dense = _fit_dense_value_ridge(samples); _write_dense(rsm_cfg, a.output_dense_config, full_dense)
    full_hurdle = _fit_hurdle_base(samples); full_adapt = _full_selected_distribution(scene, full_hurdle)
    _decorate_hurdle(rsm_cfg, full_hurdle, None, "dense_edge_hurdle", a.output_hurdle_config, "V64.3.40-EAF-ICER-HURDLE-DENSE")
    _decorate_hurdle(rsm_cfg, full_hurdle, full_adapt, "dense_edge_hurdle_sign_shift", a.output_sign_shift_config, "V64.3.40-EAF-ICER-HURDLE-SIGN-SHIFT")
    _decorate_hurdle(rsm_cfg, full_hurdle, full_adapt, "dense_edge_hurdle_selected", a.output_sdfr_config, "V64.3.40-EAF-ICER-SDFR-RAW")
    print(json.dumps({"pass": True, "output_rsmr_config": a.output_rsmr_config, "output_hurdle_config": a.output_hurdle_config, "output_sdfr_config": a.output_sdfr_config, "full_train_oof_selected": full_adapt["sample_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
