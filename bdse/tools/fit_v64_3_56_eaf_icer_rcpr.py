from __future__ import annotations

"""Nested V64.3.56 RCPR fit.

Strict branch order:
  A REALIZED-CONSTRAINT-PROCESS (diagnostic oracle)
  B PREDICTED-CONSTRAINT-PROCESS (t0 deployable), only if A fully passes.

The outcome functional is unchanged from V55: effect-support hurdle plus the
same unweighted Pareto pairwise order and frozen conformal/deployment gates.
"""
import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np

from bdse.tools import fit_v64_3_50_eaf_icer_pior as v50
from bdse.tools import fit_v64_3_50_6_eaf_icer_pior as v506
from bdse.tools import fit_v64_3_52_eaf_icer_hodr as v52
from bdse.tools import fit_v64_3_53_eaf_icer_potr as v53
from bdse.tools import fit_v64_3_54_eaf_icer_pdrm as v54
from bdse.tools import fit_v64_3_55_eaf_icer_dmor as v55
from bdse.planner.paired_constraint_process_retention import (
    CONSTRAINT_PROFILE_SCHEMA, CONSTRAINT_PROCESS_NAMES, constraint_predictor_input,
    fit_zero_preserving_constraint_predictor, predict_constraint_process,
)
from bdse.planner.paired_dynamic_mediator_outcome_retention import (
    DMOR_REALIZED_STATE_NAMES, DMOR_PREDICTED_STATE_NAMES,
    fit_zero_preserving_mediator_ridge, predict_realized_endpoint,
)

FOLDS=5; RIDGE_LAMBDA=1.0; EPS=1e-12
EXPECTED_V55_CONCORDANCE=0.5677562326869806
EXPECTED_V54_ENDPOINT_AUC=0.6117518844791572
EXPECTED_V52_SUPPORT_AUC=0.6516244589283049
REALIZED_STATE_NAMES=list(DMOR_REALIZED_STATE_NAMES)+list(CONSTRAINT_PROCESS_NAMES)
PREDICTED_STATE_NAMES=list(DMOR_PREDICTED_STATE_NAMES)+[f"predicted_{x}" for x in CONSTRAINT_PROCESS_NAMES]


def _read_jsonl(path:Path)->list[dict[str,Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def _load_constraint(path:Path, expected:set[str])->dict[str,dict[str,Any]]:
    out={}
    for r in _read_jsonl(path):
        tok=str(r.get("scenario_token",""));
        if tok in out: raise RuntimeError(f"V56 duplicate constraint profile {tok}")
        if str(r.get("schema",""))!=CONSTRAINT_PROFILE_SCHEMA: raise RuntimeError(f"V56 profile schema mismatch {tok}")
        x=np.asarray(r.get("constraint_support_delta_process",[]),dtype=np.float64)
        c0=np.asarray(r.get("t0_constraint_risk",[]),dtype=np.float64)
        if x.shape!=(len(CONSTRAINT_PROCESS_NAMES),) or c0.shape!=(3,) or np.any(~np.isfinite(x)) or np.any(~np.isfinite(c0)): raise RuntimeError(f"V56 invalid profile {tok}")
        out[tok]=r
    if set(out)!=expected: raise RuntimeError(f"V56 constraint profile population mismatch {len(out)}/{len(expected)}")
    return out

def _constraint(r): return np.asarray(r["constraint_process_profile"]["constraint_support_delta_process"],dtype=np.float64)
def _t0risk(r): return np.asarray(r["constraint_process_profile"]["t0_constraint_risk"],dtype=np.float64)
def _real_med(r): return v55._realized_mediator(r)
def _planned(r): return v55._planned_input(r)

def _constraint_x(r): return constraint_predictor_input(dict(r["operator_trajectory_profile"]),_t0risk(r))

def _fit_endpoint(rows):
    X=np.stack([_planned(r) for r in rows]); Y=np.stack([_real_med(r) for r in rows]); return fit_zero_preserving_mediator_ridge(X,Y,ridge_lambda=1.0)
def _fit_process(rows):
    X=np.stack([_constraint_x(r) for r in rows]); Y=np.stack([_constraint(r) for r in rows]); return fit_zero_preserving_constraint_predictor(X,Y,ridge_lambda=1.0)
def _pred_maps(rows, em, pm):
    e={str(r["scenario_token"]):predict_realized_endpoint(_planned(r),em) for r in rows}
    p={str(r["scenario_token"]):predict_constraint_process(_constraint_x(r),pm) for r in rows}; return e,p

def _inner_oof(rows):
    ep={}; pp={}; folds=sorted({int(r["outer_test_fold"]) for r in rows})
    for f in folds:
        tr=[r for r in rows if int(r["outer_test_fold"])!=f]; te=[r for r in rows if int(r["outer_test_fold"])==f]
        em=_fit_endpoint(tr); pm=_fit_process(tr); e,p=_pred_maps(te,em,pm); ep.update(e); pp.update(p)
    if len(ep)!=len(rows) or len(pp)!=len(rows): raise RuntimeError("V56 inner OOF nuisance coverage mismatch")
    return ep,pp

def _state(r,*,predicted:bool,ep=None,pp=None):
    tok=str(r["scenario_token"]); q=float(r["quality_value"]); p=float(r["plan_control_value"]); e=float(r["ego_ref_value"]); d=float(r["operator_execution_contrast_linf"])
    med=ep[tok] if predicted else _real_med(r); proc=pp[tok] if predicted else _constraint(r)
    z=np.concatenate([np.asarray([q,p-q,e-p,d],dtype=np.float64),np.asarray(med,dtype=np.float64),np.asarray(proc,dtype=np.float64)])
    names=PREDICTED_STATE_NAMES if predicted else REALIZED_STATE_NAMES
    if z.shape!=(len(names),) or np.any(~np.isfinite(z)): raise ValueError("V56 invalid outcome state")
    return z

def _fit_outcome(rows,safety,*,predicted=False,ep=None,pp=None):
    pairs=v52._pareto_pairs(rows,safety)
    if len(rows)<64 or len(pairs)<64: raise ValueError(f"V56 insufficient Pareto support rows={len(rows)} pairs={len(pairs)}")
    X=np.stack([_state(r,predicted=predicted,ep=ep,pp=pp) for r in rows]); mean=X.mean(0); std=np.maximum(X.std(0),1e-6); Z=(X-mean)/std
    D=np.stack([Z[i]-Z[j] for i,j in pairs]); w=np.zeros(Z.shape[1],dtype=np.float64)
    for _ in range(80):
        s=D@w; q=1/(1+np.exp(np.clip(s,-60,60))); grad=-(D.T@q)+w; hw=q*(1-q); H=(D.T*hw)@D+np.eye(D.shape[1]); step=np.linalg.solve(H,grad); w-=step
        if float(np.linalg.norm(step))<1e-9: break
    raw=Z@w; ben=raw[np.asarray([bool(r["closed_loop_beneficial"]) for r in rows],dtype=bool)]
    if ben.size<24: raise ValueError("V56 insufficient beneficial normalization")
    return {"feature_mean":mean,"feature_std":std,"weights":w,"fit_beneficial_score_mean":float(ben.mean()),"fit_beneficial_score_std":max(float(ben.std()),1e-6),"pareto_pair_count":len(pairs)}
def _risk(r,m,*,predicted=False,ep=None,pp=None):
    z=_state(r,predicted=predicted,ep=ep,pp=pp); raw=float(((z-m["feature_mean"])/m["feature_std"])@m["weights"]); return (raw-m["fit_beneficial_score_mean"])/m["fit_beneficial_score_std"]
def _threshold(cal,support,outcome,alpha,*,predicted=False,ep=None,pp=None):
    vals=sorted(max(v52._component_risk(r,support),_risk(r,outcome,predicted=predicted,ep=ep,pp=pp)) for r in cal if bool(r["closed_loop_beneficial"]))
    n=len(vals); rank=int(math.ceil((n+1)*(1-alpha))) if n else 1
    if n==0 or rank>n: raise ValueError(f"V56 calibration insufficient n={n} rank={rank}")
    return float(vals[rank-1]),{"positive_calibration_count":n,"conformal_rank":rank,"alpha":alpha,"threshold":float(vals[rank-1]),"single_joint_threshold":True}

def _nuisance_metric(test,em,pm):
    Ye=np.stack([_real_med(r) for r in test]); Yp=np.stack([_constraint(r) for r in test]); Pe=np.stack([predict_realized_endpoint(_planned(r),em) for r in test]); Pp=np.stack([predict_constraint_process(_constraint_x(r),pm) for r in test])
    es=np.asarray(em["output_rms"],dtype=np.float64); ps=np.asarray(pm["output_rms"],dtype=np.float64)
    ee=((Pe-Ye)/es[None,:])**2; ez=(Ye/es[None,:])**2; pe=((Pp-Yp)/ps[None,:])**2; pz=(Yp/ps[None,:])**2
    return {"endpoint_mse":float(ee.mean()),"endpoint_zero_mse":float(ez.mean()),"endpoint_better_zero":bool(ee.mean()<ez.mean()-EPS),"endpoint_sse":float(ee.sum()),"endpoint_zero_sse":float(ez.sum()),"endpoint_elements":int(ee.size),
            "process_mse":float(pe.mean()),"process_zero_mse":float(pz.mean()),"process_better_zero":bool(pe.mean()<pz.mean()-EPS),"process_sse":float(pe.sum()),"process_zero_sse":float(pz.sum()),"process_elements":int(pe.size)}

def _evaluate(rows,*,predicted:bool,alpha:float,safety:list[str],v55_control:dict[str,Any]):
    control_by={int(f["fold"]):f for f in v55_control["folds"]}; cagg=float(v55_control["identification"]["pareto_concordance"])
    folds=[]; keep_by={}; all_dom=[]; above=better=0; ee=ez=pe=pz=0.0; een=pen=0; eb=pb=0
    for k in range(FOLDS):
        cf=(k+1)%FOLDS; fit=[r for r in rows if int(r["outer_test_fold"]) not in {k,cf}]; cal=[r for r in rows if int(r["outer_test_fold"])==cf]; test=[r for r in rows if int(r["outer_test_fold"])==k]
        support=v52._fit_models(fit,"hurdle_sign",safety)["effect_support_risk"]
        fep=fpp=cep=cpp=tep=tpp=None; nd=None
        if predicted:
            fep,fpp=_inner_oof(fit); em=_fit_endpoint(fit); pm=_fit_process(fit); cep,cpp=_pred_maps(cal,em,pm); tep,tpp=_pred_maps(test,em,pm); nd=_nuisance_metric(test,em,pm)
            ee+=nd["endpoint_sse"];ez+=nd["endpoint_zero_sse"];een+=nd["endpoint_elements"];pe+=nd["process_sse"];pz+=nd["process_zero_sse"];pen+=nd["process_elements"];eb+=int(nd["endpoint_better_zero"]);pb+=int(nd["process_better_zero"])
        outcome=_fit_outcome(fit,safety,predicted=predicted,ep=fep,pp=fpp); tau,ci=_threshold(cal,support,outcome,alpha,predicted=predicted,ep=cep,pp=cpp)
        rr=np.asarray([_risk(r,outcome,predicted=predicted,ep=tep,pp=tpp) for r in test]); sr=np.asarray([v52._component_risk(r,support) for r in test]); keep=[bool(x<=tau) for x in np.maximum(sr,rr)]
        for r,kp in zip(test,keep): keep_by[str(r["scenario_token"])]=kp
        dp=v52._pareto_pairs(test,safety); dom=v52._concordance(test,rr,dp); cd=float(control_by[k]["pareto_concordance"]); ar=bool(math.isfinite(dom) and dom>0.5+EPS); bc=bool(math.isfinite(dom) and dom>cd+EPS); above+=ar;better+=bc
        for bad,good in dp:
            d=float(rr[bad]-rr[good]); all_dom.append(1.0 if d>EPS else 0.5 if abs(d)<=EPS else 0.0)
        bm=v50._delta_metrics(test,[True]*len(test)); dm=v50._delta_metrics(test,keep)
        folds.append({"fold":k,"fit_events":len(fit),"calibration_events":len(cal),"test_events":len(test),"pareto_pair_count":len(dp),"pareto_concordance":dom,"v55_realized_dominance_concordance":cd,"above_random":ar,"better_v55":bc,"nuisance_prediction":nd,"rsmr":bm,"rcpr":dm,"calibration":ci})
    dom=float(np.mean(all_dom)) if all_dom else float("nan"); fid=bool(math.isfinite(dom) and dom>0.5+EPS and dom>cagg+EPS and above>=4 and better>=4)
    pdiag=None
    if predicted:
        pdiag={"endpoint_normalized_mse":ee/max(een,1),"endpoint_zero_baseline_mse":ez/max(een,1),"endpoint_better_zero_fold_count":eb,"endpoint_identified":bool(ee<ez-EPS and eb>=4),
               "process_normalized_mse":pe/max(pen,1),"process_zero_baseline_mse":pz/max(pen,1),"process_better_zero_fold_count":pb,"process_identified":bool(pe<pz-EPS and pb>=4)}
        pdiag["identified"]=bool(pdiag["endpoint_identified"] and pdiag["process_identified"])
    keep=v506._align_oof_keep_by_token(rows,keep_by); base=v50._delta_metrics(rows,[True]*len(rows)); chosen=v50._delta_metrics(rows,keep); dep=v50._gate(base,chosen,alpha,[{"rsmr":f["rsmr"],"pior":f["rcpr"]} for f in folds])
    return {"folds":folds,"identification":{"pareto_concordance":dom,"v55_realized_dominance_concordance":cagg,"folds_above_random":above,"better_v55_fold_count":better,"functional_identified":fid,"nuisance_prediction":pdiag},"aggregate":chosen,"deployment_gate":dep,"pass":False}

def _final_runtime(rows, alpha, safety):
    # Freeze fold 0 as the final TRAIN calibration block, matching V55.
    fit=[r for r in rows if int(r["outer_test_fold"])!=0]
    cal=[r for r in rows if int(r["outer_test_fold"])==0]
    endpoint_model=_fit_endpoint(fit)
    process_model=_fit_process(fit)
    fit_ep,fit_pp=_inner_oof(fit)
    cal_ep,cal_pp=_pred_maps(cal,endpoint_model,process_model)
    support=v52._fit_models(fit,"hurdle_sign",safety)["effect_support_risk"]
    outcome=_fit_outcome(fit,safety,predicted=True,ep=fit_ep,pp=fit_pp)
    tau,ci=_threshold(cal,support,outcome,alpha,predicted=True,ep=cal_ep,pp=cal_pp)
    return {
        "state_family":"predicted_realized_endpoint_plus_predicted_constraint_process",
        "endpoint_mediator_predictor":endpoint_model,
        "constraint_process_predictor":process_model,
        "effect_support_risk":support,
        "conditional_pareto_risk":outcome,
        "calibration":ci,
        "state_names":list(PREDICTED_STATE_NAMES),
        "runtime_inputs":"QPE+D, fixed V53 planned endpoint+DCT profile, and D-gated t0 current constraint risk only; no post-intervention state",
        "operator":"same frozen full-set RSMR winner or incumbent; veto-only; no rerank/second-best/fallback",
        "claim_boundary":"post-intervention V56 process is supervision only; deployment consumes only cross-fitted t0 predictors",
    }


def main():
    p=argparse.ArgumentParser();
    p.add_argument("--v49-candidate-audit",type=Path,required=True);p.add_argument("--v49-scene-audit",type=Path,required=True);p.add_argument("--paired-outcomes",type=Path,required=True);p.add_argument("--v50-5-root",type=Path,required=True);p.add_argument("--v52-fit-report",type=Path,required=True);p.add_argument("--v53-operator-profiles",type=Path,required=True);p.add_argument("--v54-fit-report",type=Path,required=True);p.add_argument("--v54-dynamic-profiles",type=Path,required=True);p.add_argument("--v55-fit-report",type=Path,required=True);p.add_argument("--constraint-profiles",type=Path,required=True);p.add_argument("--output-report",type=Path,required=True);p.add_argument("--output-runtime-artifact",type=Path,required=True)
    a=p.parse_args(); v55rep=json.loads(a.v55_fit_report.read_text()); ctrl=v55rep["nested_crossfit"]["arms"]["realized_dominance"]
    if ctrl["identification"]["functional_identified"] is not True or ctrl["deployment_gate"]["pass"] is not False or abs(float(ctrl["identification"]["pareto_concordance"])-EXPECTED_V55_CONCORDANCE)>1e-12: raise RuntimeError("V56 ENGINEERING STOP: V55 branch signature drift")
    v52rep=json.loads(a.v52_fit_report.read_text()); v54rep=json.loads(a.v54_fit_report.read_text())
    rows=v55._load_rows(v49_candidate_audit=a.v49_candidate_audit,v49_scene_audit=a.v49_scene_audit,paired_outcomes=a.paired_outcomes,v50_5_root=a.v50_5_root,v53_profiles=a.v53_operator_profiles,v54_dynamic=a.v54_dynamic_profiles)
    proc=_load_constraint(a.constraint_profiles,{str(r["scenario_token"]) for r in rows})
    for r in rows: r["constraint_process_profile"]=proc[str(r["scenario_token"])]
    safety=v52._safety_names(rows); alpha=float(v52rep["nested_crossfit"]["retention_alpha"])
    oracle=_evaluate(rows,predicted=False,alpha=alpha,safety=safety,v55_control=ctrl); opass=bool(oracle["identification"]["functional_identified"] and oracle["deployment_gate"]["pass"]);oracle["pass"]=opass
    arms={"realized_constraint_process":oracle}; final=None
    if not opass:
        arms["predicted_constraint_process"]={"status":"NOT_EVALUATED_BY_PREREGISTERED_BRANCH_ORDER","eligible_by_branch_order":False,"pass":False}
        diagnosis="final_realized_interaction_safety_process_oracle_does_not_close_static_deployment_gate_internal_search_converged_by_falsification"
        train=False
    else:
        pred=_evaluate(rows,predicted=True,alpha=alpha,safety=safety,v55_control=ctrl); nd=pred["identification"]["nuisance_prediction"]; ppass=bool(nd and nd["identified"] and pred["identification"]["functional_identified"] and pred["deployment_gate"]["pass"]);pred["pass"]=ppass;pred["eligible_by_branch_order"]=True;arms["predicted_constraint_process"]=pred;train=ppass
        diagnosis="deployable_predicted_constraint_process_closes_train_gate" if ppass else "final_oracle_sufficient_but_t0_constraint_process_bridge_fails_internal_search_converged"
        if ppass: final=_final_runtime(rows,alpha,safety)
    out={"algorithm_version":"V64.3.56-EAF-ICER-RCPR","audit":"v64_3_56_eaf_icer_rcpr_fit","preregistered_branch_order":["realized_constraint_process","predicted_constraint_process"],"nested_crossfit":{"arms":arms,"retention_alpha":alpha,"safety_delta_names":safety,"train_gate_pass":train,"failure_diagnosis":diagnosis,"internal_search_converged":bool((not opass) or (opass and not train)),"v52_effect_support_control":{"auc":EXPECTED_V52_SUPPORT_AUC,"frozen":True},"v54_realized_endpoint_control":{"auc":EXPECTED_V54_ENDPOINT_AUC,"frozen":True},"v55_realized_dominance_control":{"concordance":EXPECTED_V55_CONCORDANCE,"functional_identified":True,"deployment_pass":False}},"final_runtime_fit":final}
    a.output_report.parent.mkdir(parents=True,exist_ok=True);a.output_report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if train:
        a.output_runtime_artifact.write_text(json.dumps(final,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps({"pass":True,"next":"freeze; engineering integration then untouched validation"},sort_keys=True)); return
    print(f"V64.3.56 RCPR scientific STOP ({diagnosis})")
    raise SystemExit(3)
if __name__=="__main__": main()
