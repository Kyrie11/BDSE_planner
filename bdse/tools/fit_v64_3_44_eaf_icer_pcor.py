from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.planner.value_observables import QUALITY_NAMES, VALUE_OBSERVABLE_NAMES
from bdse.planner.response_value_observables import (
    FUTURE_RESPONSE_OBSERVABLE_NAMES,
    PLAN_CONDITIONED_RESPONSE_ALL_NAMES,
    PLAN_RESPONSE_CONDITIONING_NAMES,
    PLAN_CONDITIONED_RESPONSE_OBSERVABLE_NAMES,
    RUNTIME_RESPONSE_MODE_NAMES,
)
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS, RIDGE_LAMBDA, _finite, _fold, _read_edges, _select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _fit_regret_structured_margin, _structured_scores
from bdse.tools.fit_v64_3_38_eaf_icer_davr import (
    CAPTURE_TOL, CATASTROPHE_REDUCTION_MIN, MIN_VALUE_CAL_PROPOSALS,
    NOOP_REDUCTION_MIN, _value_diag, _write_rsmr,
)
from bdse.tools.fit_v64_3_39_eaf_icer_cfsr import _fit_translation
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import _base_cfg, _fit_zero_ridge, _pred as _epv_pred
from bdse.tools.fit_v64_3_43_eaf_icer_cfrv import (
    ALL_OBSERVABLE_NAMES as V43_OBSERVABLE_NAMES,
    _scene as _v43_scene,
    _fit_weighted_zero_ridge,
    _quality_x,
    _quality_value,
    _future_x,
    _future_value,
    _metrics,
    _gate,
    _write_quality_control,
    _decorate as _decorate_v43,
)

EPS = 1.0e-12
M = len(RUNTIME_RESPONSE_MODE_NAMES)
V44_ALL_OBSERVABLE_NAMES = list(V43_OBSERVABLE_NAMES) + list(PLAN_CONDITIONED_RESPONSE_ALL_NAMES)
PC_RESPONSE_MEAN = PLAN_CONDITIONED_RESPONSE_OBSERVABLE_NAMES[0]
PC_RESPONSE_ROBUST = PLAN_CONDITIONED_RESPONSE_OBSERVABLE_NAMES[1]
PC_OCC_MEAN = PLAN_CONDITIONED_RESPONSE_OBSERVABLE_NAMES[2]
PC_OCC_ROBUST = PLAN_CONDITIONED_RESPONSE_OBSERVABLE_NAMES[3]


def _read_behavior(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tok = str(r["scenario_token"])
        if tok in out:
            raise ValueError(f"duplicate V44 behavior supervision scene {tok}")
        if [str(x) for x in r.get("conditioning_feature_names", [])] != PLAN_RESPONSE_CONDITIONING_NAMES:
            raise ValueError(f"V44 behavior feature schema mismatch for {tok}")
        x = np.asarray(r.get("conditioning_features", []), dtype=np.float64).reshape(-1)
        target = int(r.get("target_mode_index", -1))
        if x.size != len(PLAN_RESPONSE_CONDITIONING_NAMES) or not np.all(np.isfinite(x)) or not (0 <= target < M):
            raise ValueError(f"invalid V44 behavior row for {tok}")
        out[tok] = {**r, "x": x, "target": target}
    if len(out) < 512:
        raise ValueError(f"V44 behavior supervision only has {len(out)} scenes (<512)")
    return out


def _fit_behavior_posterior(behavior: dict[str, dict[str, Any]], tokens: list[str]) -> dict[str, Any]:
    rows = [behavior[t] for t in tokens if t in behavior]
    if len(rows) < 256:
        raise ValueError(f"V44 behavior-posterior fit only has {len(rows)} scenes (<256)")
    X = np.stack([r["x"] for r in rows]).astype(np.float64)
    y = np.asarray([int(r["target"]) for r in rows], dtype=np.int64)
    scale = np.sqrt(np.mean(X * X, axis=0))
    scale = np.maximum(scale, 1.0e-6)
    Z = X / scale[None, :]
    Y = np.eye(M, dtype=np.float64)[y]
    A = np.concatenate([Z, np.ones((len(Z), 1), dtype=np.float64)], axis=1)
    reg = np.eye(A.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    reg[-1, -1] = 0.0  # intercept is a class-prior term, not a behavior feature
    coef = np.linalg.solve(A.T @ A + reg, A.T @ Y)
    return {
        "enabled": True,
        "model": "scene_equal_multiclass_ridge_scores_then_softmax",
        "lambda": RIDGE_LAMBDA,
        "feature_names": list(PLAN_RESPONSE_CONDITIONING_NAMES),
        "feature_scale": scale,
        "weights": coef[:-1],
        "bias": coef[-1],
        "sample_count": int(len(rows)),
        "target_counts": [int(np.sum(y == i)) for i in range(M)],
    }


def _behavior_probs(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    one = xx.ndim == 1
    if one:
        xx = xx[None, :]
    z = xx / np.maximum(np.asarray(model["feature_scale"], dtype=np.float64)[None, :], 1.0e-6)
    logits = z @ np.asarray(model["weights"], dtype=np.float64) + np.asarray(model["bias"], dtype=np.float64)[None, :]
    logits -= np.max(logits, axis=1, keepdims=True)
    e = np.exp(np.clip(logits, -60.0, 60.0)); p = e / np.maximum(e.sum(axis=1, keepdims=True), 1.0e-12)
    return p[0] if one else p


def _behavior_diag(behavior: dict[str, dict[str, Any]], tokens: list[str], model: dict[str, Any]) -> dict[str, Any]:
    rows = [behavior[t] for t in tokens if t in behavior]
    if not rows:
        return {"sample_count": 0}
    X = np.stack([r["x"] for r in rows]); y = np.asarray([r["target"] for r in rows], dtype=np.int64)
    p = _behavior_probs(X, model)
    pred = np.argmax(p, axis=1)
    counts = np.bincount(y, minlength=M)
    prior_acc = float(counts.max() / max(len(y), 1))
    onehot = np.eye(M)[y]
    return {
        "sample_count": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "majority_baseline_accuracy": prior_acc,
        "multiclass_brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "target_counts": {RUNTIME_RESPONSE_MODE_NAMES[i]: int(counts[i]) for i in range(M)},
    }


def _candidate_specific_cvar(values: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    p = np.maximum(np.asarray(probs, dtype=np.float64).reshape(-1), 0.0)
    p = p / max(float(p.sum()), EPS)
    order = np.argsort(v, kind="mergesort"); v = v[order]; p = p[order]
    c = np.cumsum(p); q = float(np.clip(alpha, 0.0, 0.999)); s = int(np.searchsorted(c, q, side="left"))
    if s >= len(v): return float(v[-1])
    tv = v[s:]; tp = p[s:].copy(); tp[0] = max(float(c[s]) - q, 0.0)
    return float(np.dot(tv, tp) / max(float(tp.sum()), EPS))


def _pc_cost(raw: np.ndarray, model: dict[str, Any], kind: str, cfg: dict[str, Any]) -> float:
    x = np.asarray(raw, dtype=np.float64).reshape(-1)
    if x.size != 2 * M:
        raise ValueError("V44 PCOR raw response feature width mismatch")
    p = _behavior_probs(x, model)
    gated, occ = x[:M], x[M:]
    vals = gated if kind.startswith("response") else occ
    mean = float(np.dot(p, vals))
    if kind.endswith("mean"):
        return mean
    rcfg = cfg.get("teacher", {}).get("risk_aggregation", {}) or {}
    alpha = float(rcfg.get("cvar_alpha", (cfg.get("teacher", {}) or {}).get("cvar_alpha", 0.9)))
    mix = float(rcfg.get("cvar_weight", (cfg.get("teacher", {}) or {}).get("cvar_weight", 0.4)))
    cvar = _candidate_specific_cvar(vals, p, alpha)
    return float((1.0 - mix) * mean + mix * cvar)


def _pc_improvement(a: dict[str, Any], model: dict[str, Any], kind: str, cfg: dict[str, Any]) -> float:
    return float(_pc_cost(a["pc_raw_inc"], model, kind, cfg) - _pc_cost(a["pc_raw_cand"], model, kind, cfg))


def _scene_v44(groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    base = _v43_scene(groups)
    keys = [f"icer_value_observable_{n}" for n in PLAN_RESPONSE_CONDITIONING_NAMES]
    out: dict[str, list[dict[str, Any]]] = {}
    for tok, ss in base.items():
        rows = groups[tok]; inc = int(rows[0].get("raw_top_action", -1)); byact = {int(r.get("challenger_action", -2)): r for r in rows}
        ir = byact.get(inc)
        if ir is None: continue
        ri = np.asarray([_finite(ir, k) for k in keys], dtype=np.float64)
        if ri.size != len(keys) or not np.all(np.isfinite(ri)):
            raise RuntimeError(f"V44 incumbent raw response schema invalid for {tok}")
        rr = []
        for a in ss:
            row = byact.get(int(a["action"]))
            if row is None: raise RuntimeError(f"V44 missing candidate raw response row for {tok}")
            rb = np.asarray([_finite(row, k) for k in keys], dtype=np.float64)
            if rb.size != len(keys) or not np.all(np.isfinite(rb)):
                raise RuntimeError(f"V44 candidate raw response schema invalid for {tok}")
            z = dict(a); z["pc_raw_inc"] = ri.copy(); z["pc_raw_cand"] = rb; rr.append(z)
        out[tok] = rr
    return out


def _pc_value(a, epv, q, residual, model, kind, cfg):
    base = _quality_value(a, epv, q)
    imp = _pc_improvement(a, model, kind, cfg)
    r = float((np.asarray([imp]) / np.maximum(np.asarray(residual["scale"], dtype=np.float64), 1.0e-6)) @ np.asarray(residual["weights"], dtype=np.float64))
    return float(np.clip(base + r, -40.0, 40.0))


def _nested(groups, behavior, base_cfg, audit_csv: Path):
    scene = _scene_v44(groups)
    arms = ["rsmr","quality","v43_future_mean","v43_future_robust","pc_response_mean","pc_occupancy_mean","pc_occupancy_robust","pcor_main"]
    agg={a:[] for a in arms}; caps={a:0 for a in arms}; noops={a:0 for a in arms}; oppsels={a:0 for a in arms}; total_opp=total_noop=0
    folds=[]; audits=[]; vy=[]; vp={a:[] for a in arms}
    all_behavior_tokens=list(behavior)
    for k in range(FOLDS):
        test=[t for t in scene if _fold(t)==k]; cf=(k+1)%FOLDS; cal=[t for t in scene if _fold(t)==cf]; fit=[t for t in scene if _fold(t) not in {k,cf}]
        behavior_fit=[t for t in all_behavior_tokens if _fold(t) not in {k,cf}]
        behavior_test=[t for t in all_behavior_tokens if _fold(t)==k]
        bmodel=_fit_behavior_posterior(behavior,behavior_fit); bdiag=_behavior_diag(behavior,behavior_test,bmodel)
        rsm=_fit_regret_structured_margin(scene,fit); epv=_fit_zero_ridge(scene,fit,"epv")
        q=_fit_weighted_zero_ridge(scene,fit,lambda a:float(a["y"])-_epv_pred(a,epv),_quality_x,QUALITY_NAMES)
        v43m=_fit_weighted_zero_ridge(scene,fit,lambda a:float(a["y"])-_quality_value(a,epv,q),lambda a:_future_x(a,"future_response_mean_agent_cost"),["future_response_mean_agent_cost"])
        v43r=_fit_weighted_zero_ridge(scene,fit,lambda a:float(a["y"])-_quality_value(a,epv,q),lambda a:_future_x(a,"future_response_robust_agent_cost"),["future_response_robust_agent_cost"])
        def fit_pc(kind,name):
            return _fit_weighted_zero_ridge(scene,fit,lambda a:float(a["y"])-_quality_value(a,epv,q),lambda a:np.asarray([_pc_improvement(a,bmodel,kind,base_cfg)]),[name])
        pcr=fit_pc("response_mean",PC_RESPONSE_MEAN); pco=fit_pc("occupancy_mean",PC_OCC_MEAN); pcrob=fit_pc("occupancy_robust",PC_OCC_ROBUST)
        cy=[];cp=[];used=[]
        for t in cal:
            ss=scene[t]; idx=_select(ss,_structured_scores(ss,rsm))
            if idx is None: continue
            cy.append(float(ss[idx]["y"])); cp.append(_pc_value(ss[idx],epv,q,pcrob,bmodel,"occupancy_robust",base_cfg)); used.append(t)
        if len(used)<MIN_VALUE_CAL_PROPOSALS: raise ValueError(f"V44 calibration proposals {len(used)} < {MIN_VALUE_CAL_PROPOSALS}")
        shift_fit=_fit_translation(np.asarray(cp),np.asarray(cy),"quality_plus_plan_conditioned_occupancy_robust"); shift=float(shift_fit["selected_policy_bias"])
        fv={a:[] for a in arms};fc={a:0 for a in arms};fn={a:0 for a in arms};fo={a:0 for a in arms};opp=noopsc=0;subset=identity=True
        for t in test:
            ss=scene[t]; yy=np.asarray([float(a["y"]) for a in ss]); has=bool(np.any(yy>0));opp+=int(has);noopsc+=int(not has)
            score=_structured_scores(ss,rsm);idx=_select(ss,score)
            vals={a:float("nan") for a in arms}; chosen={a:None for a in arms}; chosen["rsmr"]=idx
            if idx is not None:
                a=ss[idx]; vals["rsmr"]=float(score[idx]); qv=_quality_value(a,epv,q); vals["quality"]=qv; chosen["quality"]=idx if qv>0 else None
                m=_future_value(a,epv,q,v43m); rr=_future_value(a,epv,q,v43r); vals["v43_future_mean"]=m; vals["v43_future_robust"]=rr; chosen["v43_future_mean"]=idx if m>0 else None; chosen["v43_future_robust"]=idx if rr>0 else None
                pr=_pc_value(a,epv,q,pcr,bmodel,"response_mean",base_cfg); po=_pc_value(a,epv,q,pco,bmodel,"occupancy_mean",base_cfg); prob=_pc_value(a,epv,q,pcrob,bmodel,"occupancy_robust",base_cfg); main=float(np.clip(prob+shift,-40,40))
                for nm,v in [("pc_response_mean",pr),("pc_occupancy_mean",po),("pc_occupancy_robust",prob),("pcor_main",main)]: vals[nm]=v; chosen[nm]=idx if v>0 else None
                vy.append(float(yy[idx])); [vp[n].append(float(vals[n])) for n in arms]
            subset=subset and all(chosen[n] is None or idx is not None for n in arms if n!="rsmr"); identity=identity and all(chosen[n] is None or chosen[n]==idx for n in arms if n!="rsmr")
            for n,ii in chosen.items():
                if ii is None: continue
                val=float(yy[ii]);fv[n].append(val);fc[n]+=int(has and val>0);fn[n]+=int(not has);fo[n]+=int(has)
            audits.append({"scenario_token":t,"outer_test_fold":k,"calibration_fold":cf,"candidate_count":len(ss),"positive_opportunity":int(has),"rsm_selected_action":-1 if idx is None else int(ss[idx]["action"]),"rsm_selected_teacher_improvement":float("nan") if idx is None else float(yy[idx]),**{f"{n}_selected_action":-1 if chosen[n] is None else int(ss[chosen[n]]["action"]) for n in arms if n!="rsmr"},**{f"{n}_value":float(vals[n]) for n in arms if n!="rsmr"}})
        total_opp+=opp;total_noop+=noopsc;fd={}
        for n in arms:
            fd[n]=_metrics(fv[n],fc[n],opp,fn[n],fo[n],noopsc);agg[n]+=fv[n];caps[n]+=fc[n];noops[n]+=fn[n];oppsels[n]+=fo[n]
        folds.append({"fold":k,"fit_scenes":len(fit),"value_calibration_scenes":len(cal),"test_scenes":len(test),"value_calibration_proposal_count":len(used),"behavior_fit":{k:v for k,v in bmodel.items() if k in {"sample_count","target_counts","lambda","model"}},"behavior_test":bdiag,"selected_translation_fit":shift_fit,**{n:fd[n] for n in arms},"monotone_subset_valid":subset,"frozen_winner_identity_valid":identity})
    audit_csv.parent.mkdir(parents=True,exist_ok=True)
    with audit_csv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(audits[0]));w.writeheader();w.writerows(audits)
    A={n:_metrics(agg[n],caps[n],total_opp,noops[n],oppsels[n],total_noop) for n in arms};vd={n:_value_diag(vy,vp[n]) for n in arms};gates={n:_gate(A[n],A["rsmr"],folds,n) for n in arms if n!="rsmr"}; contracts=all(f["monotone_subset_valid"] and f["frozen_winner_identity_valid"] for f in folds);passed=bool(contracts and gates["pcor_main"]["pass"])
    # Behavior metrics are kept separate from the value gate; they diagnose whether
    # the discrete response basis itself is learnable without post-hoc thresholding.
    bN=sum(f["behavior_test"].get("sample_count",0) for f in folds); bacc=sum(f["behavior_test"].get("accuracy",0)*f["behavior_test"].get("sample_count",0) for f in folds)/max(bN,1); bbase=sum(f["behavior_test"].get("majority_baseline_accuracy",0)*f["behavior_test"].get("sample_count",0) for f in folds)/max(bN,1)
    if passed: diag="plan_conditioned_occupancy_response_closes_selected_absolute_value_boundary"
    elif gates["pc_response_mean"]["pass"] and not gates["pc_occupancy_mean"]["pass"]: diag="candidate_conditioned_mode_posterior_is_sufficient_without_occupancy_support_extension"
    elif gates["pc_occupancy_mean"]["pass"] and not gates["pc_occupancy_robust"]["pass"]: diag="ungated_plan_conditioned_occupancy_support_is_sufficient_but_CVaR_not_required"
    elif bacc <= bbase + 0.01: diag="discrete_response_mode_basis_not_predictable_require_continuous_plan_conditioned_occupancy_model"
    else: diag="plan_conditioning_or_occupancy_adds_signal_but_discrete_mode_basis_or_absolute_zero_remains_insufficient"
    return {"folds":folds,"scene_audit_csv":str(audit_csv),"rsmr_rank_aggregate":A["rsmr"],"quality_control_aggregate":A["quality"],"v43_future_mean_control_aggregate":A["v43_future_mean"],"v43_future_robust_control_aggregate":A["v43_future_robust"],"pc_response_mean_aggregate":A["pc_response_mean"],"pc_occupancy_mean_aggregate":A["pc_occupancy_mean"],"pc_occupancy_robust_aggregate":A["pc_occupancy_robust"],"pcor_main_aggregate":A["pcor_main"],"selected_proposal_value_prediction_diagnostics":vd,"behavior_crossfit_summary":{"sample_count":bN,"accuracy":bacc,"majority_baseline_accuracy":bbase},"gates":gates,"monotone_frozen_winner_contract_valid":contracts,"train_gate_pass":passed,"failure_diagnosis":diag}


def _serialize_behavior(model):
    return {"enabled":True,"model":model["model"],"lambda":float(model["lambda"]),"feature_names":list(model["feature_names"]),"feature_scale":[float(x) for x in model["feature_scale"]],"weights":[[float(y) for y in row] for row in np.asarray(model["weights"])],"bias":[float(x) for x in model["bias"]],"sample_count":int(model["sample_count"]),"target_counts":[int(x) for x in model["target_counts"]]}


def _decorate_pcor(rsm_cfg,epv,q,residual,bmodel,observable_name,path,version,selected_bias=0.0,shift=False):
    cfg=yaml.safe_load(yaml.safe_dump(rsm_cfg,sort_keys=False));ic=cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"];ic["instrument_value_observables"]=True;ic["instrument_future_response_observables"]=True;ic["instrument_plan_conditioned_response_observables"]=True;sc=ic["selection_conditioned_intervention_recovery"]
    # Obtain endpoint names from the already fitted V41 EPV model.
    sc.update({"post_selection_value_enabled":True,"post_selection_value_mode":"endpoint_potential_quality_plan_conditioned_response_shift" if shift else "endpoint_potential_quality_plan_conditioned_response","post_selection_endpoint_feature_names":list(epv["names"]),"post_selection_endpoint_feature_scale":[float(x) for x in epv["scale"]],"post_selection_endpoint_weights":[float(x) for x in epv["weights"]],"post_selection_endpoint_bias":0.0,"post_selection_observable_names":list(V44_ALL_OBSERVABLE_NAMES),"post_selection_quality_observable_names":list(QUALITY_NAMES),"post_selection_quality_observable_scale":[float(x) for x in q["scale"]],"post_selection_quality_observable_weights":[float(x) for x in q["weights"]],"post_selection_future_response_observable_name":observable_name,"post_selection_future_response_scale":float(residual["scale"][0]),"post_selection_future_response_weight":float(residual["weights"][0]),"post_selection_selected_bias":float(selected_bias),"plan_conditioned_response_posterior":_serialize_behavior(bmodel),"post_selection_value_training":"scene_equal_all_edge_EPV_plus_frozen_QUALITY_plus_plan_conditioned_response_or_occupancy_residual_fixed_lambda_1","post_selection_operator":"freeze_RSMR_winner_then_plan_conditioned_response_occupancy_value_accept_same_winner_iff_positive_else_incumbent_no_rerank_no_fallback"})
    cfg.setdefault("metadata",{})["algorithm_version"]=version;cfg.setdefault("provenance",{})["algorithm_version"]=version;cfg.setdefault("experiment",{})["name"]=version.lower().replace(".","_").replace("-","_");cfg["experiment"]["algorithm"]=version;Path(path).write_text(yaml.safe_dump(cfg,sort_keys=False));return cfg


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--train-frontier-edges",required=True);ap.add_argument("--behavior-supervision",required=True);ap.add_argument("--base-config",required=True)
    for n in ["preserve","rsmr","quality","v43_future_mean","v43_future_robust","pc_response_mean","pc_occupancy_mean","pc_occupancy_robust"]: ap.add_argument(f"--output-{n.replace('_','-')}-config",dest=f"output_{n}_config",required=True)
    ap.add_argument("--output-report",required=True);ap.add_argument("--output-scene-audit",required=True);a=ap.parse_args()
    _,groups=_read_edges(Path(a.train_frontier_edges));behavior=_read_behavior(Path(a.behavior_supervision));base_cfg=_base_cfg(a.base_config);nested=_nested(groups,behavior,base_cfg,Path(a.output_scene_audit))
    # Hard V43 replay gate: V44 instrumentation may add raw mode/occupancy columns but
    # must not alter any previously registered decision path.
    expected={"rsmr_rank_aggregate":(502,221,107,28,43.29405361274824,.38501742160278746,.3556880321206127),"quality_control_aggregate":(205,129,30,13,43.905547394411805,.22473867595818817,.3126575113037135),"v43_future_mean_control_aggregate":(231,134,42,13,44.89147983003761,.23344947735191637,.29414012533643497),"v43_future_robust_control_aggregate":(228,131,42,13,45.91891311091216,.22822299651567945,.29606893309443927)}
    for key,(s,p,no,cat,sm,cap,neg) in expected.items():
        d=nested[key]
        bad=d["selected_count"]!=s or d["selected_positive_count"]!=p or d["no_positive_opportunity_false_intervention_count"]!=no or d["catastrophic_count"]!=cat or abs(float(d["teacher_improvement_sum"])-sm)>1e-9 or abs(float(d["positive_capture_rate"])-cap)>1e-12 or abs(float(d["teacher_negative_rms"])-neg)>1e-9
        if bad: raise RuntimeError(f"V44 ENGINEERING STOP: raw plan-conditioned instrumentation changed frozen V43 signature {key}")
    report={"audit":"v64_3_44_eaf_icer_pcor_fit","scientific_role":"TRAIN_only_frozen_RSMR_quality_then_plan_conditioned_behavior_posterior_and_ungated_occupancy_support","frozen_train_scenes":len(groups),"direct_support_positive_training_scenes":len(_scene_v44(groups)),"behavior_supervision_scenes":len(behavior),"ridge_lambda":RIDGE_LAMBDA,"mechanism_hypothesis":"V43 proves prospective horizon signal is real but fixed candidate-independent response priors and gated interaction severity leave both sign errors and exact-zero support. V44 learns a TRAIN-only plan-conditioned posterior over the frozen runtime response basis using logged agent future as behavior supervision only, and separately tests an ungated full-horizon occupancy potential before asking whether CVaR adds independent value.","nested_crossfit":nested,"train_gate_pass":nested["train_gate_pass"],"train_gate_contract":{"RSMR_is_sole_challenger_selector":True,"V43_RSMR_QUALITY_FUTURE_controls_are_exact_engineering_gate":True,"behavior_supervision_uses_no_teacher_improvement":True,"behavior_posterior_is_outer_fold_isolated":True,"deployment_uses_no_logged_future":True,"fixed_response_basis_no_mode_sweep":True,"ungated_occupancy_uses_existing_box_scale_and_full_candidate_horizon_no_new_radius_threshold":True,"scene_equal_all_edge_value_residual_fixed_lambda_1":True,"capture_tolerance":CAPTURE_TOL,"noop_reduction_min":NOOP_REDUCTION_MIN,"catastrophe_reduction_min":CATASTROPHE_REDUCTION_MIN,"selected_min":64,"positive_min":32,"no_threshold_lambda_CVaR_mode_probability_topk_candidate_count_or_capacity_sweep":True}}
    Path(a.output_report).write_text(json.dumps(report,indent=2,sort_keys=True))
    # all-TRAIN deployment configs; these are emitted even on scientific failure for reproducibility, but fresh selection remains launcher-gated.
    scene=_scene_v44(groups);all_direct=list(scene);all_beh=list(behavior);rsm=_fit_regret_structured_margin(scene,all_direct);epv=_fit_zero_ridge(scene,all_direct,"epv");q=_fit_weighted_zero_ridge(scene,all_direct,lambda x:float(x["y"])-_epv_pred(x,epv),_quality_x,QUALITY_NAMES);bmodel=_fit_behavior_posterior(behavior,all_beh)
    base=_base_cfg(a.base_config);base["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["instrument_value_observables"]=True;base["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["instrument_future_response_observables"]=True
    pcfg=yaml.safe_load(yaml.safe_dump(base,sort_keys=False));pcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]={"enabled":False};Path(a.output_preserve_config).write_text(yaml.safe_dump(pcfg,sort_keys=False));rsmcfg=_write_rsmr(base,a.output_rsmr_config,rsm);rsmcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["instrument_future_response_observables"]=True;rsmcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["instrument_plan_conditioned_response_observables"]=True;Path(a.output_rsmr_config).write_text(yaml.safe_dump(rsmcfg,sort_keys=False));_write_quality_control(rsmcfg,epv,q,a.output_quality_config)
    v43m=_fit_weighted_zero_ridge(scene,all_direct,lambda x:float(x["y"])-_quality_value(x,epv,q),lambda x:_future_x(x,"future_response_mean_agent_cost"),["future_response_mean_agent_cost"]);_decorate_v43(rsmcfg,epv,q,v43m,"endpoint_potential_quality_future_response_mean",a.output_v43_future_mean_config,"V64.3.44-EAF-ICER-V43-FUTURE-MEAN-CONTROL");v43r=_fit_weighted_zero_ridge(scene,all_direct,lambda x:float(x["y"])-_quality_value(x,epv,q),lambda x:_future_x(x,"future_response_robust_agent_cost"),["future_response_robust_agent_cost"]);_decorate_v43(rsmcfg,epv,q,v43r,"endpoint_potential_quality_future_response_robust",a.output_v43_future_robust_config,"V64.3.44-EAF-ICER-V43-FUTURE-ROBUST-CONTROL")
    def fit_pc(kind,name): return _fit_weighted_zero_ridge(scene,all_direct,lambda x:float(x["y"])-_quality_value(x,epv,q),lambda x:np.asarray([_pc_improvement(x,bmodel,kind,base_cfg)]),[name])
    pcr=fit_pc("response_mean",PC_RESPONSE_MEAN);pco=fit_pc("occupancy_mean",PC_OCC_MEAN);prob=fit_pc("occupancy_robust",PC_OCC_ROBUST)
    _decorate_pcor(rsmcfg,epv,q,pcr,bmodel,PC_RESPONSE_MEAN,a.output_pc_response_mean_config,"V64.3.44-EAF-ICER-PCOR-REWEIGHT")
    _decorate_pcor(rsmcfg,epv,q,pco,bmodel,PC_OCC_MEAN,a.output_pc_occupancy_mean_config,"V64.3.44-EAF-ICER-PCOR-OCC-MEAN")
    _decorate_pcor(rsmcfg,epv,q,prob,bmodel,PC_OCC_ROBUST,a.output_pc_occupancy_robust_config,"V64.3.44-EAF-ICER-PCOR-RAW")
    if not nested["train_gate_pass"]:
        print(json.dumps(report,indent=2,sort_keys=True));raise SystemExit(f"V64.3.44 PCOR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before CAL/fresh selection")
    print(json.dumps({"pass":True,"output_pc_occupancy_robust_config":a.output_pc_occupancy_robust_config},sort_keys=True))

if __name__=="__main__": main()
