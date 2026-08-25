from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.planner.value_observables import QUALITY_NAMES, VALUE_OBSERVABLE_NAMES
from bdse.planner.response_value_observables import FUTURE_RESPONSE_OBSERVABLE_NAMES
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS, RIDGE_LAMBDA, _finite, _fold, _read_edges, _select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _extended_diag, _fit_regret_structured_margin, _structured_scores
from bdse.tools.fit_v64_3_38_eaf_icer_davr import (
    CAPTURE_TOL, CATASTROPHE_REDUCTION_MIN, MIN_VALUE_CAL_PROPOSALS,
    NOOP_REDUCTION_MIN, _value_diag, _write_rsmr,
)
from bdse.tools.fit_v64_3_39_eaf_icer_cfsr import _fit_translation
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import _base_cfg, _endpoint_scene, _fit_zero_ridge, _pred as _epv_pred

EPS = 1.0e-12
ALL_OBSERVABLE_NAMES = list(VALUE_OBSERVABLE_NAMES) + list(FUTURE_RESPONSE_OBSERVABLE_NAMES)
FUTURE_MEAN_NAME = "future_response_mean_agent_cost"
FUTURE_ROBUST_NAME = "future_response_robust_agent_cost"


def _scene(groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    base = _endpoint_scene(groups)
    keys = [f"icer_value_observable_{n}" for n in ALL_OBSERVABLE_NAMES]
    out: dict[str, list[dict[str, Any]]] = {}
    for tok, ss in base.items():
        rows = groups[tok]
        inc = int(rows[0].get("raw_top_action", -1))
        byact = {int(r.get("challenger_action", -2)): r for r in rows}
        ir = byact.get(inc)
        if ir is None:
            continue
        oi = np.asarray([_finite(ir, k) for k in keys], dtype=np.float64)
        if oi.size != len(keys) or not np.all(np.isfinite(oi)):
            raise RuntimeError(f"V43 incumbent observable schema invalid for scene {tok}")
        rr = []
        for a in ss:
            row = byact.get(int(a["action"]))
            if row is None:
                raise RuntimeError(f"V43 observable replay missing candidate row for scene {tok}")
            ob = np.asarray([_finite(row, k) for k in keys], dtype=np.float64)
            if ob.size != len(keys) or not np.all(np.isfinite(ob)):
                raise RuntimeError(f"V43 candidate observable schema invalid for scene {tok}")
            b = dict(a)
            b["observable_inc"] = oi.copy()
            b["observable_cand"] = ob
            b["observable_improvement"] = oi - ob
            rr.append(b)
        out[tok] = rr
    return out


def _fit_weighted_zero_ridge(scene, tokens, target_fn, feature_fn, names):
    X, y, w = [], [], []
    for tok in tokens:
        ss = scene[tok]
        n = len(ss)
        if n <= 0:
            continue
        for a in ss:
            X.append(np.asarray(feature_fn(a), dtype=np.float64).reshape(-1))
            y.append(float(target_fn(a)))
            w.append(1.0 / n)
    if not X:
        raise ValueError("V43 residual fit has no samples")
    X = np.stack(X)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    pm = w / max(float(w.sum()), EPS)
    scale = np.sqrt(np.sum((X * X) * pm[:, None], axis=0))
    scale = np.maximum(scale, 1.0e-6)
    Z = X / scale[None, :]
    root = np.sqrt(w)[:, None]
    Zw = Z * root
    yw = y * root[:, 0]
    coef = np.linalg.solve(Zw.T @ Zw + np.eye(Z.shape[1]) * RIDGE_LAMBDA, Zw.T @ yw)
    return {"names": list(names), "scale": scale, "weights": coef, "bias": 0.0, "sample_count": int(len(y)), "scene_weight_sum": float(w.sum())}


def _quality_x(a):
    x = np.asarray(a["observable_improvement"], dtype=np.float64)
    return x[:len(QUALITY_NAMES)]


def _future_x(a, name):
    x = np.asarray(a["observable_improvement"], dtype=np.float64)
    return np.asarray([x[ALL_OBSERVABLE_NAMES.index(name)]], dtype=np.float64)


def _lin(a, m, x):
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    return float((xx / np.maximum(np.asarray(m["scale"], dtype=np.float64), 1.0e-6)) @ np.asarray(m["weights"], dtype=np.float64))


def _quality_value(a, epv, q):
    return float(np.clip(_epv_pred(a, epv) + _lin(a, q, _quality_x(a)), -40.0, 40.0))


def _future_value(a, epv, q, r):
    name = str(r["names"][0])
    return float(np.clip(_quality_value(a, epv, q) + _lin(a, r, _future_x(a, name)), -40.0, 40.0))


def _metrics(vals, captured, opp, noop_selected, opp_selected, noop_scenes):
    return _extended_diag(vals, captured, opp, noop_selected, opp_selected, noop_scenes)


def _gate(m, r, folds, key):
    existence = (
        m["no_positive_opportunity_false_intervention_count"] <= (1.0 - NOOP_REDUCTION_MIN) * r["no_positive_opportunity_false_intervention_count"] + EPS
        and m["positive_capture_rate"] >= r["positive_capture_rate"] - CAPTURE_TOL - EPS
    )
    tail = (
        m["catastrophic_count"] <= (1.0 - CATASTROPHE_REDUCTION_MIN) * r["catastrophic_count"] + EPS
        and m["teacher_negative_rms"] <= r["teacher_negative_rms"] + EPS
        and m["teacher_improvement_sum"] >= -EPS
    )
    all_folds = all(float(f[key]["teacher_improvement_sum"]) >= -EPS for f in folds)
    population = m["selected_count"] >= 64 and m["selected_positive_count"] >= 32
    return {"existence_and_capture": bool(existence), "tail": bool(tail), "all_folds_sum_nonnegative": bool(all_folds), "population": bool(population), "pass": bool(existence and tail and all_folds and population)}


def _eval(ss, rsm, epv, q, mean, robust, shift):
    score = _structured_scores(ss, rsm)
    idx = _select(ss, score)
    names = ["rsmr", "epv_raw", "quality", "future_mean", "future_robust_raw", "future_robust_main"]
    if idx is None:
        return {n: (None, float("nan")) for n in names}
    a = ss[idx]
    ev = float(_epv_pred(a, epv))
    qv = _quality_value(a, epv, q)
    mv = _future_value(a, epv, q, mean)
    rv = _future_value(a, epv, q, robust)
    main = float(np.clip(rv + float(shift), -40.0, 40.0))
    return {
        "rsmr": (idx, float(score[idx])),
        "epv_raw": (idx if ev > 0 else None, ev),
        "quality": (idx if qv > 0 else None, qv),
        "future_mean": (idx if mv > 0 else None, mv),
        "future_robust_raw": (idx if rv > 0 else None, rv),
        "future_robust_main": (idx if main > 0 else None, main),
    }


def _nested(groups, audit_csv: Path):
    scene = _scene(groups)
    names = ["rsmr", "epv_raw", "quality", "future_mean", "future_robust_raw", "future_robust_main"]
    agg = {n: [] for n in names}; caps = {n: 0 for n in names}; noops = {n: 0 for n in names}; oppsels = {n: 0 for n in names}
    total_opp = total_noop = 0; folds = []; audits = []; vy = []; vp = {n: [] for n in names}
    for k in range(FOLDS):
        test = [t for t in scene if _fold(t) == k]
        cf = (k + 1) % FOLDS
        cal = [t for t in scene if _fold(t) == cf]
        fit = [t for t in scene if _fold(t) not in {k, cf}]
        rsm = _fit_regret_structured_margin(scene, fit)
        epv = _fit_zero_ridge(scene, fit, "epv")
        q = _fit_weighted_zero_ridge(scene, fit, lambda a: float(a["y"]) - _epv_pred(a, epv), _quality_x, QUALITY_NAMES)
        mean = _fit_weighted_zero_ridge(scene, fit, lambda a: float(a["y"]) - _quality_value(a, epv, q), lambda a: _future_x(a, FUTURE_MEAN_NAME), [FUTURE_MEAN_NAME])
        robust = _fit_weighted_zero_ridge(scene, fit, lambda a: float(a["y"]) - _quality_value(a, epv, q), lambda a: _future_x(a, FUTURE_ROBUST_NAME), [FUTURE_ROBUST_NAME])
        cy, cp, used = [], [], []
        for t in cal:
            ss = scene[t]; idx = _select(ss, _structured_scores(ss, rsm))
            if idx is None: continue
            cy.append(float(ss[idx]["y"])); cp.append(_future_value(ss[idx], epv, q, robust)); used.append(t)
        if len(used) < MIN_VALUE_CAL_PROPOSALS:
            raise ValueError(f"V43 calibration proposals {len(used)} < {MIN_VALUE_CAL_PROPOSALS}")
        shift_fit = _fit_translation(np.asarray(cp), np.asarray(cy), "quality_plus_runtime_future_response_robust")
        shift = float(shift_fit["selected_policy_bias"])

        fv = {n: [] for n in names}; fc = {n: 0 for n in names}; fn = {n: 0 for n in names}; fo = {n: 0 for n in names}
        opp = noopsc = 0; subset = identity = True
        for t in test:
            ss = scene[t]; yy = np.asarray([float(a["y"]) for a in ss]); has = bool(np.any(yy > 0.0)); opp += int(has); noopsc += int(not has)
            ev = _eval(ss, rsm, epv, q, mean, robust, shift); ridx = ev["rsmr"][0]
            if ridx is not None:
                vy.append(float(yy[ridx]))
                for n in names: vp[n].append(float(ev[n][1]))
            chosen = {n: ev[n][0] for n in names}
            subset = subset and all(chosen[n] is None or ridx is not None for n in names if n != "rsmr")
            identity = identity and all(chosen[n] is None or chosen[n] == ridx for n in names if n != "rsmr")
            for n, idx in chosen.items():
                if idx is None: continue
                val = float(yy[idx]); fv[n].append(val); fc[n] += int(has and val > 0); fn[n] += int(not has); fo[n] += int(has)
            audits.append({
                "scenario_token": t, "outer_test_fold": k, "calibration_fold": cf, "candidate_count": len(ss), "positive_opportunity": int(has),
                "rsm_selected_action": -1 if ridx is None else int(ss[ridx]["action"]), "rsm_selected_score": float(ev["rsmr"][1]),
                "rsm_selected_teacher_improvement": float("nan") if ridx is None else float(yy[ridx]),
                **{f"{n}_selected_action": -1 if chosen[n] is None else int(ss[chosen[n]]["action"]) for n in names if n != "rsmr"},
                **{f"{n}_value": float(ev[n][1]) for n in names if n != "rsmr"},
            })
        total_opp += opp; total_noop += noopsc; fd = {}
        for n in names:
            fd[n] = _metrics(fv[n], fc[n], opp, fn[n], fo[n], noopsc); agg[n] += fv[n]; caps[n] += fc[n]; noops[n] += fn[n]; oppsels[n] += fo[n]
        folds.append({
            "fold": k, "fit_scenes": len(fit), "value_calibration_scenes": len(cal), "test_scenes": len(test), "value_calibration_proposal_count": len(used),
            "selected_translation_fit": shift_fit,
            "quality_fit": {"sample_count": q["sample_count"], "scene_weight_sum": q["scene_weight_sum"]},
            "future_mean_fit": {"sample_count": mean["sample_count"], "scene_weight_sum": mean["scene_weight_sum"]},
            "future_robust_fit": {"sample_count": robust["sample_count"], "scene_weight_sum": robust["scene_weight_sum"]},
            **{n: fd[n] for n in names}, "monotone_subset_valid": subset, "frozen_winner_identity_valid": identity,
        })
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(audits[0])); w.writeheader(); w.writerows(audits)
    A = {n: _metrics(agg[n], caps[n], total_opp, noops[n], oppsels[n], total_noop) for n in names}
    vd = {n: _value_diag(vy, vp[n]) for n in names}
    gates = {n: _gate(A[n], A["rsmr"], folds, n) for n in names if n != "rsmr"}
    contracts = all(f["monotone_subset_valid"] and f["frozen_winner_identity_valid"] for f in folds)
    passed = bool(contracts and gates["future_robust_main"]["pass"])
    if passed:
        diag = "runtime_counterfactual_response_distribution_closes_selected_absolute_value_boundary"
    elif gates["future_mean"]["pass"] and not gates["future_robust_raw"]["pass"]:
        diag = "prospective_response_horizon_is_sufficient_but_teacher_style_tail_functional_is_not"
    elif vd["future_robust_raw"].get("noncatastrophe_auc", -9) > vd["quality"].get("noncatastrophe_auc", -9) + 0.03 and vd["future_robust_raw"].get("positive_auc", -9) >= vd["quality"].get("positive_auc", -9) - 0.02:
        diag = "runtime_response_distribution_adds_downside_signal_but_zero_boundary_or_behavior_model_remains"
    elif vd["future_mean"].get("positive_auc", -9) > vd["quality"].get("positive_auc", -9) + 0.03:
        diag = "future_response_horizon_adds_cardinal_signal_without_tail_closure"
    else:
        diag = "hand_constructed_runtime_response_modes_insufficient_require_plan_conditioned_behavior_or_occupancy_observable"
    return {
        "folds": folds, "scene_audit_csv": str(audit_csv), "rsmr_rank_aggregate": A["rsmr"], "endpoint_potential_raw_aggregate": A["epv_raw"],
        "quality_control_aggregate": A["quality"], "future_response_mean_aggregate": A["future_mean"], "future_response_robust_raw_aggregate": A["future_robust_raw"],
        "future_response_robust_main_aggregate": A["future_robust_main"], "selected_proposal_value_prediction_diagnostics": vd,
        "gates": gates, "monotone_frozen_winner_contract_valid": contracts, "train_gate_pass": passed, "failure_diagnosis": diag,
    }


def _decorate(rsm_cfg, epv, q, response, mode, path, version, selected_bias=0.0):
    cfg = yaml.safe_load(yaml.safe_dump(rsm_cfg, sort_keys=False))
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic["instrument_value_observables"] = True
    ic["instrument_future_response_observables"] = True
    sc = ic["selection_conditioned_intervention_recovery"]
    sc.update({
        "post_selection_value_enabled": True, "post_selection_value_mode": mode,
        "post_selection_endpoint_feature_names": list(epv["names"]), "post_selection_endpoint_feature_scale": [float(x) for x in epv["scale"]],
        "post_selection_endpoint_weights": [float(x) for x in epv["weights"]], "post_selection_endpoint_bias": 0.0,
        "post_selection_observable_names": list(ALL_OBSERVABLE_NAMES),
        "post_selection_quality_observable_names": list(QUALITY_NAMES), "post_selection_quality_observable_scale": [float(x) for x in q["scale"]],
        "post_selection_quality_observable_weights": [float(x) for x in q["weights"]],
        "post_selection_future_response_observable_name": str(response["names"][0]), "post_selection_future_response_scale": float(response["scale"][0]),
        "post_selection_future_response_weight": float(response["weights"][0]), "post_selection_selected_bias": float(selected_bias),
        "post_selection_value_training": "scene_equal_all_edge_EPV_plus_frozen_QUALITY_plus_runtime_future_response_residual_fixed_lambda_1",
        "post_selection_operator": "freeze_RSMR_winner_then_counterfactual_response_distribution_value_accept_same_winner_iff_positive_else_incumbent_no_rerank_no_fallback",
    })
    cfg.setdefault("metadata", {})["algorithm_version"] = version; cfg.setdefault("provenance", {})["algorithm_version"] = version
    cfg.setdefault("experiment", {})["name"] = version.lower().replace(".", "_").replace("-", "_"); cfg["experiment"]["algorithm"] = version
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False)); return cfg


def _write_quality_control(rsm_cfg, epv, q, path):
    # Reuse historical V42 runtime mode, but keep the observable schema exactly 9D.
    cfg = yaml.safe_load(yaml.safe_dump(rsm_cfg, sort_keys=False)); ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic["instrument_value_observables"] = True; ic["instrument_future_response_observables"] = False; sc = ic["selection_conditioned_intervention_recovery"]
    sc.update({"post_selection_value_enabled": True, "post_selection_value_mode": "endpoint_potential_quality_observable", "post_selection_endpoint_feature_names": list(epv["names"]),
               "post_selection_endpoint_feature_scale": [float(x) for x in epv["scale"]], "post_selection_endpoint_weights": [float(x) for x in epv["weights"]], "post_selection_endpoint_bias": 0.0,
               "post_selection_observable_names": list(VALUE_OBSERVABLE_NAMES), "post_selection_observable_quality_dim": len(QUALITY_NAMES), "post_selection_observable_scale": [float(x) for x in q["scale"]],
               "post_selection_observable_weights": [float(x) for x in q["weights"]], "post_selection_selected_bias": 0.0})
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.43-EAF-ICER-QUALITY-CONTROL"; Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False)); return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frontier-edges", required=True); ap.add_argument("--base-config", required=True)
    for name in ["preserve", "rsmr", "epv", "quality", "future_mean", "future_robust"]: ap.add_argument(f"--output-{name.replace('_','-')}-config", dest=f"output_{name}_config", required=True)
    ap.add_argument("--output-report", required=True); ap.add_argument("--output-scene-audit", required=True); a = ap.parse_args()
    _, groups = _read_edges(Path(a.train_frontier_edges)); nested = _nested(groups, Path(a.output_scene_audit))
    # V43 engineering gate: new prospective instrumentation must reproduce both
    # frozen RSMR and V42 QUALITY exactly before any scientific attribution.
    r = nested["rsmr_rank_aggregate"]; q = nested["quality_control_aggregate"]
    er = {"selected_count":502,"selected_positive_count":221,"no_positive_opportunity_false_intervention_count":107,"catastrophic_count":28}
    eq = {"selected_count":205,"selected_positive_count":129,"no_positive_opportunity_false_intervention_count":30,"catastrophic_count":13}
    bad = any(int(r[k]) != v for k,v in er.items()) or abs(float(r["teacher_improvement_sum"])-43.29405361274824)>1e-9 or abs(float(r["positive_capture_rate"])-0.38501742160278746)>1e-12
    bad = bad or any(int(q[k]) != v for k,v in eq.items()) or abs(float(q["teacher_improvement_sum"])-43.905547394411805)>1e-9 or abs(float(q["positive_capture_rate"])-0.22473867595818817)>1e-12 or abs(float(q["teacher_negative_rms"])-0.3126575113037135)>1e-12
    if bad: raise RuntimeError("V43 ENGINEERING STOP: future-response instrumentation changed frozen RSMR/V42 QUALITY TRAIN signatures")
    report = {
        "audit":"v64_3_43_eaf_icer_cfrv_fit", "scientific_role":"TRAIN_only_frozen_RSMR_plus_current_quality_plus_runtime_counterfactual_future_response_distribution",
        "frozen_train_scenes":len(groups), "direct_support_positive_training_scenes":len(_scene(groups)), "ridge_lambda":RIDGE_LAMBDA,
        "observable_names":ALL_OBSERVABLE_NAMES,
        "mechanism_hypothesis":"V42 proves current trajectory QUALITY is a real selective cardinal mediator but cannot close capture, while 17/28 catastrophes and 21/50 material positives have zero current RISK delta. V43 therefore keeps RSMR and the V42 QUALITY mechanism frozen, and adds a label-free prospective response distribution built from current-agent CV/CA/brake/yield/nonyield rollouts. Mean tests future horizon; frozen teacher-style mean+CVaR tests whether response-tail uncertainty is the missing deployment functional.",
        "nested_crossfit":nested, "train_gate_pass":nested["train_gate_pass"],
        "train_gate_contract":{"RSMR_is_sole_challenger_selector":True,"V42_QUALITY_replay_is_hard_engineering_gate":True,"future_modes_are_runtime_only_no_logged_future":True,"future_mean_vs_mean_CVaR_is_preregistered_ablation":True,"response_modes_and_CVaR_alpha_weight_inherited_from_frozen_teacher_config_no_sweep":True,"scene_equal_all_edge_fixed_lambda_1":True,"selected_policy_calibration_translation_only":True,"noop_false_intervention_reduction_fraction_min":NOOP_REDUCTION_MIN,"capture_tolerance":CAPTURE_TOL,"catastrophe_reduction_fraction_min":CATASTROPHE_REDUCTION_MIN,"all_test_folds_selected_sum_nonnegative":True,"selected_min":64,"positive_min":32,"no_threshold_lambda_alpha_mode_probability_topk_candidate_count_or_capacity_sweep":True}
    }
    Path(a.output_report).write_text(json.dumps(report, indent=2, sort_keys=True))
    # Always emit diagnostic configs; scientific promotion still stops on failed TRAIN gate.
    scene = _scene(groups); base = _base_cfg(a.base_config)
    base["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["instrument_value_observables"] = True
    base["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["instrument_future_response_observables"] = True
    pcfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False)); pcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"] = {"enabled":False}; Path(a.output_preserve_config).write_text(yaml.safe_dump(pcfg, sort_keys=False))
    rsm = _fit_regret_structured_margin(scene, list(scene)); rsmcfg = _write_rsmr(base, a.output_rsmr_config, rsm); rsmcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["instrument_future_response_observables"] = True; Path(a.output_rsmr_config).write_text(yaml.safe_dump(rsmcfg, sort_keys=False))
    epv = _fit_zero_ridge(scene, list(scene), "epv")
    # EPV config from V41 semantics, with future instrumentation kept off for exact control.
    epcfg = yaml.safe_load(yaml.safe_dump(rsmcfg, sort_keys=False)); eic=epcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]; eic["instrument_future_response_observables"]=False; esc=eic["selection_conditioned_intervention_recovery"]; esc.update({"post_selection_value_enabled":True,"post_selection_value_mode":"endpoint_potential_value","post_selection_endpoint_feature_names":list(epv["names"]),"post_selection_endpoint_feature_scale":[float(x) for x in epv["scale"]],"post_selection_endpoint_weights":[float(x) for x in epv["weights"]],"post_selection_endpoint_bias":0.0,"post_selection_selected_bias":0.0}); Path(a.output_epv_config).write_text(yaml.safe_dump(epcfg, sort_keys=False))
    qfit = _fit_weighted_zero_ridge(scene, list(scene), lambda x:float(x["y"])-_epv_pred(x,epv), _quality_x, QUALITY_NAMES); _write_quality_control(rsmcfg,epv,qfit,a.output_quality_config)
    mean = _fit_weighted_zero_ridge(scene, list(scene), lambda x:float(x["y"])-_quality_value(x,epv,qfit), lambda x:_future_x(x,FUTURE_MEAN_NAME), [FUTURE_MEAN_NAME])
    robust = _fit_weighted_zero_ridge(scene, list(scene), lambda x:float(x["y"])-_quality_value(x,epv,qfit), lambda x:_future_x(x,FUTURE_ROBUST_NAME), [FUTURE_ROBUST_NAME])
    _decorate(rsmcfg,epv,qfit,mean,"endpoint_potential_quality_future_response_mean",a.output_future_mean_config,"V64.3.43-EAF-ICER-CFRV-MEAN")
    _decorate(rsmcfg,epv,qfit,robust,"endpoint_potential_quality_future_response_robust",a.output_future_robust_config,"V64.3.43-EAF-ICER-CFRV-RAW")
    if not nested["train_gate_pass"]:
        print(json.dumps(report, indent=2, sort_keys=True)); raise SystemExit(f"V64.3.43 CFRV nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before CAL/fresh selection")
    print(json.dumps({"pass":True,"output_future_robust_config":a.output_future_robust_config}, sort_keys=True))

if __name__ == "__main__": main()
