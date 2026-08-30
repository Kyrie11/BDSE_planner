from __future__ import annotations

"""V64.3.49 SIIR nested TRAIN mechanism test.

V48.2 falsified the transportability of observed-multiplicity conditioning on
untouched A/B.  V49 therefore keeps the full deployment representation frozen
and changes only how selected-proposal risk is *identified*.

OBS-SIGN is an exact replay of V48 SIGN-NOMULT: the sign-risk ranker is fit on
observed full-set RSMR winners.

SIIR fits the identical zero-bias pairwise sign-risk objective (lambda=1), on
the identical consequence coordinates [Q, P-Q, E-P], but its TRAIN selected
population is produced by a fixed label-free intervention on the selection
operator: each scene's admissible challengers are hash-permuted and exactly one
uniformly indexed prefix is exposed to frozen RSMR.  No intervention parameter
is tuned and runtime always uses the full candidate set.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.planner.value_observables import QUALITY_NAMES
from bdse.planner.selection_interventional_risk_retention import (
    SIIR_STATE_NAMES,
    select_interventional_winner,
)
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS, _fold, _read_edges, _select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _fit_regret_structured_margin, _structured_scores
from bdse.tools.fit_v64_3_38_eaf_icer_davr import _auc
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import _fit_zero_ridge, _pred as _epv_pred
from bdse.tools.fit_v64_3_43_eaf_icer_cfrv import _fit_weighted_zero_ridge, _gate, _metrics, _quality_value, _quality_x
from bdse.tools.fit_v64_3_47_eaf_icer_fsfr import (
    ARM_FEATURES,
    _read_sidecar,
    _rval,
    _rx,
    _scene as _v47_scene,
)
from bdse.tools.fit_v64_3_48_eaf_icer_ocrr import (
    EPS,
    RIDGE_LAMBDA,
    _conformal_threshold,
    _extract_plan_and_ego_params,
    _fit_sign_ranker,
    _retention_alpha,
    _risk,
)

FIT_INTERVENTION_SEED = "v64.3.49-siir-train-intervention-v1"
DIAG_INTERVENTION_SEED = "v64.3.49-siir-heldout-intervention-v1"

V48_EXPECTED = {
    "rsmr_rank_aggregate": (502, 221, 107, 28, 43.29405361274824),
    "v45_plan_control_aggregate": (217, 121, 38, 9, 56.55117310290402),
    "v47_ego_reference_aggregate": (251, 136, 45, 9, 59.53269591505746),
    "sign_nomult_aggregate": (411, 187, 78, 18, 53.49557781828986),
    "sign_mult_aggregate": (439, 204, 74, 14, 62.63414076193869),
}
V48_NOMULT_AUC = 0.6139192605594113
V48_EGO_AUC = 0.6298288272330558


def _sig(d: dict[str, Any]) -> tuple[Any, Any, Any, Any, float]:
    return (
        d.get("selected_count"),
        d.get("selected_positive_count"),
        d.get("no_positive_opportunity_false_intervention_count"),
        d.get("catastrophic_count"),
        float(d.get("teacher_improvement_sum", float("nan"))),
    )


def _check_v48(v48_fit: Path, v48_screen: Path) -> dict[str, Any]:
    r = json.loads(v48_fit.read_text())
    n = r.get("nested_crossfit", {})
    for k, e in V48_EXPECTED.items():
        g = _sig(n.get(k, {}))
        if any(g[i] != e[i] for i in range(4)) or abs(g[4] - e[4]) > 1.0e-9:
            raise RuntimeError(f"V49 ENGINEERING STOP: V48 TRAIN signature changed {k}: {g}")
    ri = n.get("risk_identification", {})
    if abs(float(ri.get("sign_nomult", {}).get("aggregate_nonpositive_risk_auc", float("nan"))) - V48_NOMULT_AUC) > 1.0e-12:
        raise RuntimeError("V49 ENGINEERING STOP: V48 SIGN-NOMULT AUC changed")
    if abs(float(ri.get("sign_mult", {}).get("aggregate_baseline_neg_ego_ref_value_auc", float("nan"))) - V48_EGO_AUC) > 1.0e-12:
        raise RuntimeError("V49 ENGINEERING STOP: V48 EGO baseline AUC changed")
    if n.get("preferred_promotion_arm") != "sign_mult" or n.get("train_gate_pass") is not True:
        raise RuntimeError("V49 ENGINEERING STOP: V48 TRAIN preregistration replay changed")
    s = json.loads(v48_screen.read_text())
    if s.get("pass") is not False or s.get("split_A_pass") is not False or s.get("split_B_pass") is not False:
        raise RuntimeError("V49 ENGINEERING STOP: V48.2 fresh failure signature changed")
    if s.get("preferred_arm") != "sign_mult" or s.get("next_action") != "STOP_no_promotion_do_not_pool_A_B_or_tune":
        raise RuntimeError("V49 ENGINEERING STOP: V48.2 preregistered STOP signature changed")
    return r


def _read_v47_audit(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = str(r["scenario_token"])
            if t in out:
                raise RuntimeError(f"duplicate V47 audit token {t}")
            out[t] = {
                "outer_test_fold": int(float(r["outer_test_fold"])),
                "rsm_selected_action": int(float(r["rsm_selected_action"])),
                "quality_value": float(r["quality_value"]),
                "plan_control_value": float(r["plan_control_value"]),
                "ego_ref_value": float(r["ego_ref_value"]),
            }
    if len(out) != 782:
        raise RuntimeError(f"V49 ENGINEERING STOP: V47 scene audit must be 782 rows, got {len(out)}")
    return out


def _candidate_states(groups: dict[str, list[dict[str, Any]]], side: dict[str, np.ndarray], v47_audit: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    scene = _v47_scene(groups, side)
    if set(scene) != set(v47_audit):
        raise RuntimeError(f"V49 ENGINEERING STOP: V47 all-edge/scene-audit token mismatch {len(scene)} vs {len(v47_audit)}")
    out: dict[str, list[dict[str, Any]]] = {}
    for k in range(FOLDS):
        cf = (k + 1) % FOLDS
        fit = [t for t in scene if _fold(t) not in {k, cf}]
        test = [t for t in scene if _fold(t) == k]
        rsm = _fit_regret_structured_margin(scene, fit)
        epv = _fit_zero_ridge(scene, fit, "epv")
        q = _fit_weighted_zero_ridge(scene, fit, lambda a: float(a["y"]) - _epv_pred(a, epv), _quality_x, QUALITY_NAMES)
        plan = _fit_weighted_zero_ridge(
            scene,
            fit,
            lambda a: float(a["y"]) - _quality_value(a, epv, q),
            lambda a: _rx(a, ARM_FEATURES["plan_control"]),
            ARM_FEATURES["plan_control"],
        )
        ego = _fit_weighted_zero_ridge(
            scene,
            fit,
            lambda a: float(a["y"]) - _quality_value(a, epv, q),
            lambda a: _rx(a, ARM_FEATURES["ego_ref"]),
            ARM_FEATURES["ego_ref"],
        )
        for t in test:
            ss = scene[t]
            score = _structured_scores(ss, rsm)
            rr: list[dict[str, Any]] = []
            for j, a in enumerate(ss):
                qv = _quality_value(a, epv, q)
                pv = _rval(a, epv, q, plan, ARM_FEATURES["plan_control"])
                ev = _rval(a, epv, q, ego, ARM_FEATURES["ego_ref"])
                rr.append({
                    "scenario_token": t,
                    "outer_test_fold": k,
                    "action": int(a["action"]),
                    "y": float(a["y"]),
                    "rsmr_score": float(score[j]),
                    "support": float(a["support"]),
                    "margin": float(a["margin"]),
                    "utility_prior": int(a["utility_prior"]),
                    "quality_value": float(qv),
                    "plan_control_value": float(pv),
                    "ego_ref_value": float(ev),
                })
            idx = _select(ss, score)
            va = v47_audit[t]
            got_action = -1 if idx is None else int(ss[idx]["action"])
            if int(va["outer_test_fold"]) != k or got_action != int(va["rsm_selected_action"]):
                raise RuntimeError(f"V49 ENGINEERING STOP: V47 RSMR OOF replay failed {t}: {got_action} vs {va['rsm_selected_action']}")
            if idx is not None:
                got = rr[idx]
                for key in ["quality_value", "plan_control_value", "ego_ref_value"]:
                    if abs(float(got[key]) - float(va[key])) > 2.0e-8 * max(1.0, abs(float(got[key])), abs(float(va[key]))):
                        raise RuntimeError(f"V49 ENGINEERING STOP: V47 consequence OOF replay failed {t}/{key}: {got[key]} vs {va[key]}")
            out[t] = rr
    return out


def _winner_index(rr: list[dict[str, Any]]) -> int | None:
    cand = [j for j, a in enumerate(rr) if math.isfinite(float(a["rsmr_score"])) and float(a["rsmr_score"]) > 0.0]
    if not cand:
        return None
    return sorted(cand, key=lambda j: (
        -float(rr[j]["rsmr_score"]), -float(rr[j]["support"]), -float(rr[j]["margin"]), -int(rr[j]["utility_prior"]), int(rr[j]["action"])
    ))[0]


def _event(rr: list[dict[str, Any]], idx: int, *, prefix_size: int | None = None, intervention_seed: str | None = None) -> dict[str, Any]:
    a = rr[int(idx)]
    z = {
        "scenario_token": str(a["scenario_token"]),
        "outer_test_fold": int(a["outer_test_fold"]),
        "rsm_selected_action": int(a["action"]),
        "rsm_selected_teacher_improvement": float(a["y"]),
        "quality_value": float(a["quality_value"]),
        "plan_control_value": float(a["plan_control_value"]),
        "ego_ref_value": float(a["ego_ref_value"]),
        "candidate_count": int(len(rr)),
    }
    if prefix_size is not None:
        z["intervention_prefix_size"] = int(prefix_size)
        z["intervention_prefix_fraction"] = float(prefix_size / max(len(rr), 1))
        z["intervention_seed"] = str(intervention_seed or "")
    return z


def _full_event(rr: list[dict[str, Any]]) -> dict[str, Any] | None:
    idx = _winner_index(rr)
    return None if idx is None else _event(rr, idx)


def _intervention_event(rr: list[dict[str, Any]], seed: str) -> dict[str, Any] | None:
    if not rr:
        return None
    j, m = select_interventional_winner(
        str(rr[0]["scenario_token"]),
        [int(a["action"]) for a in rr],
        [float(a["rsmr_score"]) for a in rr],
        [float(a["support"]) for a in rr],
        [float(a["margin"]) for a in rr],
        [int(a["utility_prior"]) for a in rr],
        seed=seed,
    )
    return None if j is None else _event(rr, j, prefix_size=m, intervention_seed=seed)


def _compress_nomult_model(m: dict[str, Any]) -> dict[str, Any]:
    if bool(m.get("use_extremal_multiplicity", False)):
        raise ValueError("V49 SIIR cannot compress a multiplicity model")
    if len(m.get("weights", [])) != 4 or abs(float(m["weights"][3])) > 1.0e-10:
        raise ValueError("V49 SIIR expected exact zero multiplicity coefficient")
    out = dict(m)
    out["model"] = "zero_bias_pairwise_selection_interventional_sign_risk"
    out["feature_names"] = list(SIIR_STATE_NAMES)
    for key in ["feature_mean", "feature_std", "weights"]:
        out[key] = [float(x) for x in m[key][:3]]
    out.pop("use_extremal_multiplicity", None)
    out["selection_intervention"] = {
        "type": "one_label_free_uniform_hash_prefix_per_scene",
        "seed": FIT_INTERVENTION_SEED,
        "runtime_candidate_bank_changed": False,
    }
    return out


def _policy_metrics(states: dict[str, list[dict[str, Any]]], tokens: list[str], model: dict[str, Any], tau: float) -> tuple[dict[str, Any], list[float], list[float]]:
    vals: list[float] = []
    cap = noop = oppsel = opp = noopp = 0
    ys: list[float] = []
    risks: list[float] = []
    for t in tokens:
        rr = states[t]
        has = any(float(a["y"]) > 0.0 for a in rr)
        opp += int(has); noopp += int(not has)
        e = _full_event(rr)
        if e is None:
            continue
        y = float(e["rsm_selected_teacher_improvement"])
        risk = _risk(e, model)
        ys.append(y); risks.append(risk)
        if risk > tau:
            continue
        vals.append(y); cap += int(has and y > 0.0); noop += int(not has); oppsel += int(has)
    return _metrics(vals, cap, opp, noop, oppsel, noopp), ys, risks


def _direct_policy_metrics(states: dict[str, list[dict[str, Any]]], tokens: list[str], mode: str) -> dict[str, Any]:
    vals: list[float] = []
    cap = noop = oppsel = opp = noopp = 0
    for t in tokens:
        rr = states[t]; has = any(float(a["y"]) > 0.0 for a in rr); opp += int(has); noopp += int(not has)
        e = _full_event(rr)
        if e is None:
            continue
        y = float(e["rsm_selected_teacher_improvement"])
        keep = mode == "rsmr" or (mode == "ego_ref" and float(e["ego_ref_value"]) > 0.0)
        if not keep:
            continue
        vals.append(y); cap += int(has and y > 0.0); noop += int(not has); oppsel += int(has)
    return _metrics(vals, cap, opp, noop, oppsel, noopp)


def _nested(states: dict[str, list[dict[str, Any]]], v48: dict[str, Any], audit_csv: Path) -> dict[str, Any]:
    alpha = _retention_alpha(v48["nested_crossfit"]["rsmr_rank_aggregate"])
    toks = sorted(states)
    arms = ["rsmr", "ego_ref", "obs_sign", "siir"]
    all_vals = {a: [] for a in arms}; all_cap = {a: 0 for a in arms}; all_noop = {a: 0 for a in arms}; all_oppsel = {a: 0 for a in arms}
    total_opp = total_noopp = 0
    folds: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    full_y: list[float] = []; full_ego: list[float] = []; full_obs: list[float] = []; full_siir: list[float] = []
    obs_better_ego = siir_better_ego = siir_better_obs = 0
    diag_y: list[float] = []; diag_obs: list[float] = []; diag_siir: list[float] = []

    for k in range(FOLDS):
        cf = (k + 1) % FOLDS
        fit_t = [t for t in toks if _fold(t) not in {k, cf}]
        cal_t = [t for t in toks if _fold(t) == cf]
        test_t = [t for t in toks if _fold(t) == k]
        obs_fit = [e for t in fit_t if (e := _full_event(states[t])) is not None]
        int_fit = [e for t in fit_t if (e := _intervention_event(states[t], FIT_INTERVENTION_SEED)) is not None]
        obs_model = _fit_sign_ranker(obs_fit, False)
        siir_model = _fit_sign_ranker(int_fit, False)
        cal_full = [e for t in cal_t if (e := _full_event(states[t])) is not None]
        obs_tau, obs_cal = _conformal_threshold(cal_full, obs_model, alpha)
        siir_tau, siir_cal = _conformal_threshold(cal_full, siir_model, alpha)

        fv = {a: [] for a in arms}; fc = {a: 0 for a in arms}; fn = {a: 0 for a in arms}; fo = {a: 0 for a in arms}
        opp = noopp = 0
        fy: list[float] = []; fego: list[float] = []; fobs: list[float] = []; fsiir: list[float] = []
        fdy: list[float] = []; fdobs: list[float] = []; fdsiir: list[float] = []
        for t in test_t:
            rr = states[t]; has = any(float(a["y"]) > 0.0 for a in rr); opp += int(has); noopp += int(not has)
            e = _full_event(rr)
            chosen = {a: False for a in arms}
            y = float("nan"); ro = rs = float("nan")
            if e is not None:
                y = float(e["rsm_selected_teacher_improvement"])
                ro = _risk(e, obs_model); rs = _risk(e, siir_model)
                chosen["rsmr"] = True
                chosen["ego_ref"] = float(e["ego_ref_value"]) > 0.0
                chosen["obs_sign"] = ro <= obs_tau
                chosen["siir"] = rs <= siir_tau
                fy.append(y); fego.append(-float(e["ego_ref_value"])); fobs.append(ro); fsiir.append(rs)
                full_y.append(y); full_ego.append(-float(e["ego_ref_value"])); full_obs.append(ro); full_siir.append(rs)
            for a in arms:
                if not chosen[a]:
                    continue
                fv[a].append(y); fc[a] += int(has and y > 0.0); fn[a] += int(not has); fo[a] += int(has)
            ide = _intervention_event(rr, DIAG_INTERVENTION_SEED)
            if ide is not None:
                iy = float(ide["rsm_selected_teacher_improvement"]); iro = _risk(ide, obs_model); irs = _risk(ide, siir_model)
                fdy.append(iy); fdobs.append(iro); fdsiir.append(irs); diag_y.append(iy); diag_obs.append(iro); diag_siir.append(irs)
            audit_rows.append({
                "scenario_token": t, "outer_test_fold": k, "calibration_fold": cf,
                "candidate_count": len(rr), "positive_opportunity": int(has),
                "rsm_selected_action": -1 if e is None else int(e["rsm_selected_action"]),
                "rsm_selected_teacher_improvement": y,
                "ego_ref_value": float("nan") if e is None else float(e["ego_ref_value"]),
                "v49_obs_sign_risk": ro, "v49_siir_risk": rs,
                "v49_obs_sign_threshold": obs_tau, "v49_siir_threshold": siir_tau,
                "v49_obs_sign_selected_action": -1 if e is None or not chosen["obs_sign"] else int(e["rsm_selected_action"]),
                "v49_siir_selected_action": -1 if e is None or not chosen["siir"] else int(e["rsm_selected_action"]),
                "v49_heldout_intervention_selected_action": -1 if ide is None else int(ide["rsm_selected_action"]),
                "v49_heldout_intervention_teacher_improvement": float("nan") if ide is None else float(ide["rsm_selected_teacher_improvement"]),
                "v49_heldout_intervention_prefix_size": -1 if ide is None else int(ide["intervention_prefix_size"]),
            })
        total_opp += opp; total_noopp += noopp
        fd: dict[str, Any] = {}
        for a in arms:
            fd[a] = _metrics(fv[a], fc[a], opp, fn[a], fo[a], noopp)
            all_vals[a] += fv[a]; all_cap[a] += fc[a]; all_noop[a] += fn[a]; all_oppsel[a] += fo[a]
        ya = np.asarray(fy, dtype=np.float64); be = np.asarray(fego, dtype=np.float64); oo = np.asarray(fobs, dtype=np.float64); ss = np.asarray(fsiir, dtype=np.float64)
        ego_auc = _auc(ya <= 0.0, be); obs_auc = _auc(ya <= 0.0, oo); siir_auc = _auc(ya <= 0.0, ss)
        obs_better_ego += int(math.isfinite(obs_auc) and math.isfinite(ego_auc) and obs_auc > ego_auc + EPS)
        siir_better_ego += int(math.isfinite(siir_auc) and math.isfinite(ego_auc) and siir_auc > ego_auc + EPS)
        siir_better_obs += int(math.isfinite(siir_auc) and math.isfinite(obs_auc) and siir_auc > obs_auc + EPS)
        dya = np.asarray(fdy, dtype=np.float64); do = np.asarray(fdobs, dtype=np.float64); ds = np.asarray(fdsiir, dtype=np.float64)
        folds.append({
            "fold": k, "fit_scenes": len(fit_t), "value_calibration_scenes": len(cal_t), "test_scenes": len(test_t),
            **{a: fd[a] for a in arms},
            "risk_identification": {
                "ego_ref_auc": ego_auc, "obs_sign_auc": obs_auc, "siir_auc": siir_auc,
                "siir_better_ego": bool(siir_auc > ego_auc + EPS), "siir_better_obs": bool(siir_auc > obs_auc + EPS),
            },
            "heldout_intervention_diagnostic": {
                "selected_event_count": len(fdy),
                "obs_sign_auc": _auc(dya <= 0.0, do),
                "siir_auc": _auc(dya <= 0.0, ds),
            },
            "selection_intervention_fit": {
                "selected_event_count": len(int_fit),
                "positive_count": int(sum(float(e["rsm_selected_teacher_improvement"]) > 0.0 for e in int_fit)),
                "nonpositive_count": int(sum(float(e["rsm_selected_teacher_improvement"]) <= 0.0 for e in int_fit)),
                "seed": FIT_INTERVENTION_SEED,
            },
            "calibration": {"obs_sign": obs_cal, "siir": siir_cal},
            "same_winner_veto_only_contract": True,
        })

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys())); w.writeheader(); w.writerows(audit_rows)

    A = {a: _metrics(all_vals[a], all_cap[a], total_opp, all_noop[a], all_oppsel[a], total_noopp) for a in arms}
    gates = {a: _gate(A[a], A["rsmr"], folds, a) for a in ["ego_ref", "obs_sign", "siir"]}
    ya = np.asarray(full_y, dtype=np.float64); ego = np.asarray(full_ego, dtype=np.float64); obs = np.asarray(full_obs, dtype=np.float64); si = np.asarray(full_siir, dtype=np.float64)
    ego_auc = _auc(ya <= 0.0, ego); obs_auc = _auc(ya <= 0.0, obs); siir_auc = _auc(ya <= 0.0, si)
    dya = np.asarray(diag_y, dtype=np.float64); do = np.asarray(diag_obs, dtype=np.float64); ds = np.asarray(diag_siir, dtype=np.float64)
    identified = bool(
        math.isfinite(siir_auc) and siir_auc > max(ego_auc, obs_auc) + EPS
        and siir_better_ego >= 4 and siir_better_obs >= 4
    )

    # OBS-SIGN must exactly replay V48 SIGN-NOMULT.  This hard gate ensures the
    # only scientific difference is the TRAIN selection measure.
    exp = V48_EXPECTED["sign_nomult_aggregate"]
    got = _sig(A["obs_sign"])
    if any(got[i] != exp[i] for i in range(4)) or abs(got[4] - exp[4]) > 1.0e-9:
        raise RuntimeError(f"V49 ENGINEERING STOP: OBS-SIGN no longer replays V48 SIGN-NOMULT: {got}")
    if abs(obs_auc - V48_NOMULT_AUC) > 1.0e-12 or abs(ego_auc - V48_EGO_AUC) > 1.0e-12:
        raise RuntimeError(f"V49 ENGINEERING STOP: V48 risk-AUC replay changed obs={obs_auc} ego={ego_auc}")

    promotion = bool(gates["siir"]["pass"] and identified)
    if promotion:
        diagnosis = "selection_interventional_training_recovers_transportable_selected_policy_risk_without_multiplicity"
    elif identified:
        diagnosis = "selection_interventional_risk_is_identified_but_selected_zero_tail_deployment_gate_remains_open"
    else:
        diagnosis = "selection_interventional_risk_does_not_outperform_observational_selected_risk_close_current_offline_selected_risk_family"
    return {
        "folds": folds,
        "scene_audit_csv": str(audit_csv),
        "rsmr_rank_aggregate": A["rsmr"],
        "v47_ego_reference_aggregate": A["ego_ref"],
        "v48_observational_sign_aggregate": A["obs_sign"],
        "siir_aggregate": A["siir"],
        "gates": gates,
        "risk_identification": {
            "aggregate_ego_ref_auc": ego_auc,
            "aggregate_obs_sign_auc": obs_auc,
            "aggregate_siir_auc": siir_auc,
            "siir_better_ego_fold_count": int(siir_better_ego),
            "siir_better_obs_fold_count": int(siir_better_obs),
            "identified": identified,
            "heldout_intervention_obs_auc": _auc(dya <= 0.0, do),
            "heldout_intervention_siir_auc": _auc(dya <= 0.0, ds),
        },
        "retention_alpha": alpha,
        "retention_alpha_derivation": "unchanged_from_V48_CAPTURE_TOL_over_frozen_RSMR_capture",
        "train_gate_pass": promotion,
        "preferred_promotion_arm": "siir" if promotion else None,
        "failure_diagnosis": diagnosis,
        "frozen_contract": {
            "RSMR_selector_unchanged": True,
            "Q_P_E_consequence_coordinates_unchanged": True,
            "runtime_multiplicity_removed": True,
            "pairwise_loss_unchanged": True,
            "lambda": RIDGE_LAMBDA,
            "calibration_rule_unchanged": True,
            "runtime_full_candidate_set_unchanged": True,
            "same_winner_or_incumbent_only": True,
            "no_rerank_second_best_fallback": True,
            "selection_intervention_is_TRAIN_only_and_label_free": True,
        },
    }


def _decorate(base_cfg: dict[str, Any], plan: dict[str, Any], ego: dict[str, Any], risk4: dict[str, Any], tau: float, out: Path) -> None:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False))
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    sc = ic["selection_conditioned_intervention_recovery"]
    # Reuse the byte-locked V48 NOMULT runtime path exactly.  V49 changes only
    # the TRAIN identification population; the runtime selector/value path is
    # intentionally untouched.
    sc["post_selection_value_mode"] = "endpoint_potential_quality_operator_conditioned_risk_retention"
    sc["selected_policy_risk_plan_response_names"] = list(plan["names"])
    sc["selected_policy_risk_plan_response_scales"] = list(plan["scales"])
    sc["selected_policy_risk_plan_response_weights"] = list(plan["weights"])
    sc["selected_policy_risk_ego_reference_names"] = list(ego["names"])
    sc["selected_policy_risk_ego_reference_scales"] = list(ego["scales"])
    sc["selected_policy_risk_ego_reference_weights"] = list(ego["weights"])
    sc["operator_conditioned_risk_retention"] = {
        "feature_names": [
            "quality_value", "prospective_response_increment",
            "ego_reference_increment", "log_extremal_multiplicity",
        ],
        "aggregation": "sign_only",
        "use_extremal_multiplicity": False,
        "components": {"sign_risk": risk4},
        "retention_threshold": float(tau),
        "threshold_calibration": "unchanged_V48_TRAIN_full_set_selected_positive_split_conformal_from_frozen_capture_tolerance",
        "identification_distribution": "TRAIN_only_label_free_uniform_hash_prefix_selection_intervention",
        "intervention_seed": FIT_INTERVENTION_SEED,
        "runtime_candidate_set": "full_frozen_deployment_candidate_set",
    }
    sc["post_selection_selected_bias"] = 0.0
    sc["post_selection_value_training"] = "selection_interventional_selected_RSMR_pairwise_sign_risk_fixed_lambda_1_same_QPE_no_multiplicity"
    sc["post_selection_operator"] = "freeze_full_set_RSMR_winner_then_SIIR_veto_only_same_winner_or_incumbent_no_rerank_no_fallback"
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.49-EAF-ICER-SIIR"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.49-EAF-ICER-SIIR"
    cfg.setdefault("experiment", {})["name"] = "v64_3_49_eaf_icer_siir"
    cfg["experiment"]["algorithm"] = "V64.3.49-EAF-ICER-SIIR"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _write_candidate_audit(states: dict[str, list[dict[str, Any]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in sorted(states):
            rr = states[t]
            full = _full_event(rr); inte = _intervention_event(rr, FIT_INTERVENTION_SEED)
            f.write(json.dumps({
                "scenario_token": t,
                "outer_test_fold": _fold(t),
                "candidate_count": len(rr),
                "full_selected_action": -1 if full is None else int(full["rsm_selected_action"]),
                "intervention_selected_action": -1 if inte is None else int(inte["rsm_selected_action"]),
                "intervention_prefix_size": -1 if inte is None else int(inte["intervention_prefix_size"]),
                "candidates": rr,
            }, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v44-train-frontier-edges", required=True)
    ap.add_argument("--v47-fsfr-sidecar", required=True)
    ap.add_argument("--v47-scene-audit", required=True)
    ap.add_argument("--v47-plan-config", required=True)
    ap.add_argument("--v47-ego-ref-config", required=True)
    ap.add_argument("--v48-fit-report", required=True)
    ap.add_argument("--v48-screen-report", required=True)
    ap.add_argument("--output-siir-config", required=True)
    ap.add_argument("--output-report", required=True)
    ap.add_argument("--output-scene-audit", required=True)
    ap.add_argument("--output-candidate-audit", required=True)
    a = ap.parse_args()

    v48 = _check_v48(Path(a.v48_fit_report), Path(a.v48_screen_report))
    _, groups = _read_edges(Path(a.v44_train_frontier_edges))
    side = _read_sidecar(Path(a.v47_fsfr_sidecar))
    v47a = _read_v47_audit(Path(a.v47_scene_audit))
    states = _candidate_states(groups, side, v47a)
    _write_candidate_audit(states, Path(a.output_candidate_audit))
    nested = _nested(states, v48, Path(a.output_scene_audit))

    plan_cfg, plan, ego = _extract_plan_and_ego_params(Path(a.v47_plan_config), Path(a.v47_ego_ref_config))
    fit_t = [t for t in states if _fold(t) != 0]
    cal_t = [t for t in states if _fold(t) == 0]
    int_fit = [e for t in fit_t if (e := _intervention_event(states[t], FIT_INTERVENTION_SEED)) is not None]
    model = _fit_sign_ranker(int_fit, False)
    cal_full = [e for t in cal_t if (e := _full_event(states[t])) is not None]
    tau, cal_info = _conformal_threshold(cal_full, model, float(nested["retention_alpha"]))
    _decorate(plan_cfg, plan, ego, model, tau, Path(a.output_siir_config))

    report = {
        "audit": "v64_3_49_eaf_icer_siir_fit",
        "scientific_role": "operator_identification_intervention_after_V48_2_fresh_transport_failure",
        "mechanism_hypothesis": "Observed selected-policy risk is confounded by the realized selection regime. Randomize the TRAIN selection event with a fixed label-free candidate-prefix intervention, learn the same low-capacity sign-risk law on already validated Q/P/E consequence coordinates, and deploy it on the unchanged full-set RSMR winner without multiplicity conditioning.",
        "nested_crossfit": nested,
        "train_gate_pass": nested["train_gate_pass"],
        "final_runtime_fit": {
            "fit_intervention_seed": FIT_INTERVENTION_SEED,
            "fit_event_count": len(int_fit),
            "calibration": cal_info,
            "risk_model": _compress_nomult_model(model),
        },
        "preregistered_stop_if_failed": "If SIIR is not identified and deployment-sufficient on nested TRAIN, stop current offline selected-risk family. Do not tune intervention seed/prefix law, pairwise loss, lambda, calibration threshold, Q/P/E features, or add operator observables. If TRAIN passes but either untouched A500/B500 fails, the same STOP applies and fresh data may not tune V49.",
    }
    Path(a.output_report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "pass": nested["train_gate_pass"],
        "preferred_promotion_arm": nested["preferred_promotion_arm"],
        "failure_diagnosis": nested["failure_diagnosis"],
        "risk_identification": nested["risk_identification"],
    }, sort_keys=True))
    if not nested["train_gate_pass"]:
        raise SystemExit(f"V64.3.49 SIIR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before fresh selection")


if __name__ == "__main__":
    main()
