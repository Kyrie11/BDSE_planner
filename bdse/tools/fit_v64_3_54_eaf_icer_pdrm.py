from __future__ import annotations

"""Fit V64.3.54 PDRM realized-response mediator models.

V54 is intentionally a mediator-identification experiment, not a t=0 runtime
retention policy.  It reuses the exact V50.5 paired outcomes and V52 effect-
support hurdle, and asks whether the realized treatment-control ego transition
identifies effectful final outcome order better than the strongest frozen V53
pre-execution state.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bdse.planner.paired_dynamic_response_mediation import (
    DYNAMIC_PROFILE_SCHEMA,
    PDRM_ENDPOINT_STATE_NAMES,
    PDRM_TEMPORAL_STATE_NAMES,
    outcome_state_from_dynamic_profile,
)
from bdse.tools import fit_v64_3_50_eaf_icer_pior as v50
from bdse.tools import fit_v64_3_50_6_eaf_icer_pior as v506
from bdse.tools import fit_v64_3_51_eaf_icer_pocr as v51
from bdse.tools import fit_v64_3_52_eaf_icer_hodr as v52
from bdse.tools import fit_v64_3_53_eaf_icer_potr as v53

FOLDS=5
EPS=1.0e-12
RIDGE_LAMBDA=1.0
ARMS=("realized_endpoint","realized_temporal")
EXPECTED_V53_SHA="9174ffeac064a85bef6c1727915d93903271f9afe1770f5e5ba3e3e51efe1b6e"


def _sha256(path: Path) -> str:
    import hashlib
    h=hashlib.sha256()
    with path.open("rb") as f:
        for ch in iter(lambda:f.read(1024*1024),b""): h.update(ch)
    return h.hexdigest()


def _auc(y: np.ndarray, risk: np.ndarray) -> float:
    return v52._auc(y,risk)


def _check_v53(path: Path) -> dict[str,Any]:
    if _sha256(path)!=EXPECTED_V53_SHA:
        raise RuntimeError(f"V54 ENGINEERING STOP: V53 fit hash drift {_sha256(path)}")
    d=json.load(open(path))
    n=d.get("nested_crossfit",{})
    if d.get("train_gate_pass") is not False or n.get("failure_diagnosis")!="preexecution_operator_trajectory_contrast_does_not_identify_effectful_outcome_order":
        raise RuntimeError("V54 ENGINEERING STOP: wrong V53 preregistered branch")
    for a in ("endpoint","temporal"):
        if bool(n.get("arms",{}).get(a,{}).get("identification",{}).get("identified",True)):
            raise RuntimeError(f"V54 ENGINEERING STOP: V53 {a} identification signature drift")
    return d


def _load_dynamic(path: Path, expected: set[str]) -> dict[str,dict[str,Any]]:
    out={}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r=json.loads(line); tok=str(r.get("scenario_token",""))
        if not tok or tok in out: raise RuntimeError(f"V54 invalid/duplicate dynamic profile token={tok!r}")
        if str(r.get("schema",""))!=DYNAMIC_PROFILE_SCHEMA: raise RuntimeError(f"V54 dynamic profile schema mismatch token={tok}")
        if len(r.get("endpoint_signed",[]))!=4 or len(r.get("cosine_modes_1_2",[]))!=8: raise RuntimeError(f"V54 dynamic profile shape mismatch token={tok}")
        out[tok]=r
    if set(out)!=expected: raise RuntimeError(f"V54 dynamic profile population mismatch {len(out)}/{len(expected)}")
    return out


def _state(r: dict[str,Any], arm: str) -> np.ndarray:
    return outcome_state_from_dynamic_profile(float(r["quality_value"]),float(r["plan_control_value"]),float(r["ego_ref_value"]),dict(r["dynamic_response_profile"]),state_family=arm)


def _names(arm: str) -> list[str]:
    if arm=="realized_endpoint": return list(PDRM_ENDPOINT_STATE_NAMES)
    if arm=="realized_temporal": return list(PDRM_TEMPORAL_STATE_NAMES)
    raise ValueError(arm)


def _sign_pairs(rows: list[dict[str,Any]]) -> list[tuple[int,int]]:
    good=[i for i,r in enumerate(rows) if v52._effect_support(r) and bool(r["closed_loop_beneficial"])]
    bad=[i for i,r in enumerate(rows) if v52._effect_support(r) and not bool(r["closed_loop_beneficial"])]
    return [(i,j) for i in bad for j in good]


def _fit(rows: list[dict[str,Any]], arm: str) -> dict[str,Any]:
    pairs=_sign_pairs(rows)
    if len(rows)<64 or len(pairs)<64: raise ValueError(f"V54 {arm} insufficient support rows={len(rows)} pairs={len(pairs)}")
    X=np.stack([_state(r,arm) for r in rows]); mean=X.mean(0); std=np.maximum(X.std(0),1e-6); Z=(X-mean)/std
    D=np.stack([Z[i]-Z[j] for i,j in pairs]); w=np.zeros(Z.shape[1],dtype=np.float64)
    before=float(D.shape[0]*math.log(2.0))
    for _ in range(80):
        s=D@w; q=1.0/(1.0+np.exp(np.clip(s,-60,60))); grad=-(D.T@q)+RIDGE_LAMBDA*w
        hw=q*(1-q); H=(D.T*hw)@D+RIDGE_LAMBDA*np.eye(D.shape[1]); step=np.linalg.solve(H,grad); w-=step
        if float(np.linalg.norm(step))<1e-9: break
    raw=Z@w; ben=raw[np.asarray([bool(r["closed_loop_beneficial"]) for r in rows],dtype=bool)]
    if ben.size<24: raise ValueError(f"V54 {arm} insufficient beneficial normalization={ben.size}")
    after=float(np.sum(np.logaddexp(0.0,-D@w))+0.5*np.dot(w,w))
    return {"model":"zero_bias_pairwise_effectful_conditional_sign_risk","state_family":arm,"feature_names":_names(arm),
            "feature_mean":[float(x) for x in mean],"feature_std":[float(x) for x in std],"weights":[float(x) for x in w],
            "bias":0.0,"lambda":1.0,"fit_row_count":len(rows),"pair_count":len(pairs),"objective_at_zero":before,"objective_final":after,
            "fit_beneficial_score_mean":float(ben.mean()),"fit_beneficial_score_std":max(float(ben.std()),1e-6)}


def _risk(r: dict[str,Any], m: dict[str,Any]) -> float:
    z=_state(r,str(m["state_family"])); mean=np.asarray(m["feature_mean"]); std=np.asarray(m["feature_std"]); w=np.asarray(m["weights"])
    raw=float(((z-mean)/np.maximum(std,1e-6))@w)
    return float((raw-float(m["fit_beneficial_score_mean"]))/max(float(m["fit_beneficial_score_std"]),1e-6))


def _threshold(cal: list[dict[str,Any]], support_model: dict[str,Any], outcome_model: dict[str,Any], alpha: float) -> tuple[float,dict[str,Any]]:
    vals=sorted(max(v52._component_risk(r,support_model),_risk(r,outcome_model)) for r in cal if bool(r["closed_loop_beneficial"]))
    n=len(vals); rank=int(math.ceil((n+1)*(1-alpha))) if n else 1; min_n=int(math.ceil((1-alpha)/alpha))
    if n==0 or rank>n: raise ValueError(f"V54 calibration insufficient n={n} rank={rank}")
    return float(vals[rank-1]),{"positive_calibration_count":n,"conformal_rank":rank,"alpha":alpha,"threshold":float(vals[rank-1]),
                                 "single_joint_threshold":True,"diagnostic_only_post_intervention":True,"minimum_positive_count_for_finite_rank":min_n}


def _nested(rows: list[dict[str,Any]], alpha: float, v52_report: dict[str,Any], v53_report: dict[str,Any]) -> dict[str,Any]:
    safety=v52._safety_names(rows)
    v52_scalar=v52_report["nested_crossfit"]["arms"]["hurdle_sign"]
    v53_temporal=v53_report["nested_crossfit"]["arms"]["temporal"]
    arms={}; oof={a:{} for a in ARMS}
    for arm in ARMS:
        folds=[]; keep_by_token={}; all_y=[]; all_r=[]
        above=better_pre=better_v52=better_v51=0
        for k in range(FOLDS):
            cf=(k+1)%FOLDS; fit=[r for r in rows if int(r["outer_test_fold"]) not in {k,cf}]; cal=[r for r in rows if int(r["outer_test_fold"])==cf]; test=[r for r in rows if int(r["outer_test_fold"])==k]
            sm=v52._fit_models(fit,"hurdle_sign",safety)["effect_support_risk"]
            om=_fit(fit,arm); tau,ci=_threshold(cal,sm,om,alpha)
            sr=np.asarray([v52._component_risk(r,sm) for r in test]); rr=np.asarray([_risk(r,om) for r in test])
            # All frozen pre-execution controls are locked by the exact V53 fit hash.
            # Requiring each comparator separately prevents a high aggregate AUC from
            # hiding the same cross-fold reversal that falsified V53.
            pre_auc=float(v53_temporal["folds"][k]["conditional_sign_auc"])
            v52_auc=float(v53_temporal["folds"][k]["v52_scalar_conditional_sign_auc"])
            v51_auc=float(v53_temporal["folds"][k]["v51_conditional_sign_auc"])
            eff=[i for i,r in enumerate(test) if v52._effect_support(r)]; idx=np.asarray(eff,dtype=np.int64); y=np.asarray([not bool(test[i]["closed_loop_beneficial"]) for i in eff],dtype=bool)
            auc=_auc(y,rr[idx])
            ar=bool(math.isfinite(auc) and auc>0.5+EPS)
            bp=bool(math.isfinite(auc) and auc>pre_auc+EPS)
            b52=bool(math.isfinite(auc) and auc>v52_auc+EPS)
            b51=bool(math.isfinite(auc) and auc>v51_auc+EPS)
            above+=int(ar); better_pre+=int(bp); better_v52+=int(b52); better_v51+=int(b51)
            all_y.extend(y.tolist()); all_r.extend(rr[idx].tolist())
            joint=np.maximum(sr,rr); keep=[bool(x<=tau) for x in joint]
            for r,kk,rv in zip(test,keep,rr): keep_by_token[str(r["scenario_token"])]=kk; oof[arm][str(r["scenario_token"])]=float(rv)
            bm=v50._delta_metrics(test,[True]*len(test)); dm=v50._delta_metrics(test,keep)
            folds.append({"fold":k,"fit_events":len(fit),"calibration_events":len(cal),"test_events":len(test),"conditional_sign_auc":auc,
                          "v53_preexecution_temporal_auc":pre_auc,"v52_scalar_conditional_sign_auc":v52_auc,"v51_conditional_sign_auc":v51_auc,
                          "above_random":ar,"better_v53_temporal":bp,"better_v52_scalar":b52,"better_v51_scalar":b51,
                          "rsmr":bm,"diagnostic_retention":dm,"calibration":ci})
        auc=_auc(np.asarray(all_y,dtype=bool),np.asarray(all_r)); pre_auc=float(v53_temporal["identification"]["aggregate_auc"])
        pre_v52=float(v53_temporal["identification"]["v52_scalar_conditional_sign_auc"])
        pre_v51=float(v53_temporal["identification"]["v51_conditional_sign_auc"])
        identified=bool(
            math.isfinite(auc) and auc>0.5+EPS and auc>pre_auc+EPS and auc>pre_v52+EPS and auc>pre_v51+EPS
            and above>=4 and better_pre>=4 and better_v52>=4 and better_v51>=4
        )
        keep=v506._align_oof_keep_by_token(rows,keep_by_token); base=v50._delta_metrics(rows,[True]*len(rows)); chosen=v50._delta_metrics(rows,keep)
        oracle=v50._gate(base,chosen,alpha,[{"rsmr":f["rsmr"],"pior":f["diagnostic_retention"]} for f in folds])
        arms[arm]={"folds":folds,"identification":{"aggregate_auc":auc,"v53_preexecution_temporal_auc":pre_auc,
                    "v52_scalar_conditional_sign_auc":pre_v52,"v51_conditional_sign_auc":pre_v51,"folds_above_random":above,
                    "better_v53_temporal_fold_count":better_pre,"better_v52_scalar_fold_count":better_v52,
                    "better_v51_scalar_fold_count":better_v51,"identified":identified},"retrospective_oracle_aggregate":chosen,
                    "retrospective_oracle_gate":oracle,"scientific_use_of_oracle":"diagnostic_only_post_intervention_not_a_t0_deployable_retention_policy"}
    ep=arms["realized_endpoint"]; tp=arms["realized_temporal"]
    # temporal necessity within realized response family
    tokens=sorted(oof["realized_endpoint"]); effmap={str(r["scenario_token"]):r for r in rows if v52._effect_support(r)}; use=[t for t in tokens if t in effmap]
    y=np.asarray([not bool(effmap[t]["closed_loop_beneficial"]) for t in use],dtype=bool); er=np.asarray([oof["realized_endpoint"][t] for t in use]); tr=np.asarray([oof["realized_temporal"][t] for t in use])
    eauc=_auc(y,er); tauc=_auc(y,tr); better=sum(int(float(tp["folds"][k]["conditional_sign_auc"])>float(ep["folds"][k]["conditional_sign_auc"])+EPS) for k in range(FOLDS))
    tp["identification"]["realized_endpoint_control_auc"]=eauc; tp["identification"]["better_realized_endpoint_fold_count"]=better
    tp["identification"]["temporal_necessity_identified"]=bool(tp["identification"]["identified"] and tauc>eauc+EPS and better>=4)
    preferred="realized_endpoint" if ep["identification"]["identified"] else "realized_temporal" if tp["identification"]["temporal_necessity_identified"] else None
    if preferred=="realized_endpoint": diag="realized_one_replan_ego_transition_mediates_effectful_selected_outcome_order"
    elif preferred=="realized_temporal": diag="realized_temporal_ego_response_shape_required_beyond_realized_endpoint"
    else: diag="short_horizon_realized_ego_response_does_not_identify_effectful_outcome_order"
    return {"retention_alpha":alpha,"v52_effect_support_control":{"auc":float(v52_scalar["identification"]["support_auc"]),"frozen":True},
            "v53_preexecution_control":{"temporal_auc":float(v53_temporal["identification"]["aggregate_auc"]),
                "v52_scalar_auc":float(v53_temporal["identification"]["v52_scalar_conditional_sign_auc"]),
                "v51_scalar_auc":float(v53_temporal["identification"]["v51_conditional_sign_auc"]),"frozen":True},
            "arms":arms,"preferred_mediator_arm":preferred,"mediator_identification_pass":preferred is not None,"failure_diagnosis":diag}


def main() -> None:
    p=argparse.ArgumentParser();
    p.add_argument("--v49-candidate-audit",type=Path,required=True); p.add_argument("--v49-scene-audit",type=Path,required=True); p.add_argument("--paired-outcomes",type=Path,required=True)
    p.add_argument("--v50-5-root",type=Path,required=True); p.add_argument("--v51-fit-report",type=Path,required=True); p.add_argument("--v52-fit-report",type=Path,required=True)
    p.add_argument("--v53-fit-report",type=Path,required=True); p.add_argument("--dynamic-profiles",type=Path,required=True); p.add_argument("--output-report",type=Path,required=True)
    p.add_argument("--output-analysis-table",type=Path,required=True)
    a=p.parse_args(); v53_report=_check_v53(a.v53_fit_report); v52_report=v53._check_v52(a.v52_fit_report); _=v52._check_v51(a.v51_fit_report)
    states=v50._candidate_states(a.v49_candidate_audit); obs=v50._obs_risk(a.v49_scene_audit); rows=v50._join(states,a.paired_outcomes,obs)
    scalar=v51._load_treatment_contrast(a.v50_5_root,set(states)); dyn=_load_dynamic(a.dynamic_profiles,set(states))
    for r in rows:
        tok=str(r["scenario_token"]); p=dyn[tok]
        if abs(float(p["planned_execution_contrast_linf"])-float(scalar[tok]))>1e-9: raise RuntimeError(f"V54 dynamic profile planned D drift token={tok}")
        r["operator_execution_contrast_linf"]=float(scalar[tok]); r["dynamic_response_profile"]=p
    alpha=float(v52_report["nested_crossfit"]["retention_alpha"]); nested=_nested(rows,alpha,v52_report,v53_report)
    # Self-contained analysis snapshot: future result audits should not depend on
    # reconstructing the entire historical V49/V50.5 root. This file is analysis
    # provenance only and is never consumed by the fit above.
    a.output_analysis_table.parent.mkdir(parents=True,exist_ok=True)
    with a.output_analysis_table.open("w",encoding="utf-8") as f:
        for r in sorted(rows,key=lambda x:str(x["scenario_token"])):
            p=dict(r["dynamic_response_profile"]); row={
                "scenario_token":str(r["scenario_token"]),"outer_test_fold":int(r["outer_test_fold"]),
                "closed_loop_beneficial":bool(r["closed_loop_beneficial"]),"closed_loop_hard_harm":bool(r.get("closed_loop_hard_harm",False)),
                "closed_loop_score_delta":float(r["closed_loop_score_delta"]),
                "quality_value":float(r["quality_value"]),"plan_control_value":float(r["plan_control_value"]),"ego_ref_value":float(r["ego_ref_value"]),
                "planned_execution_contrast_linf":float(r["operator_execution_contrast_linf"]),
                "dynamic_endpoint_signed":[float(x) for x in p["endpoint_signed"]],
                "dynamic_cosine_modes_1_2":[float(x) for x in p["cosine_modes_1_2"]],
                "realized_response_linf":float(p["realized_response_linf"]),
            }
            for k,v in r.items():
                if str(k).startswith("hard_safety_delta_"):
                    row[str(k)]=float(v)
            f.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
    analysis_sha=_sha256(a.output_analysis_table)
    report={"audit":"v64_3_54_eaf_icer_pdrm_fit","algorithm_version":"V64.3.54-EAF-ICER-PDRM","scientific_role":"preregistered_V53_both_preexecution_state_families_failed_branch_post_intervention_dynamic_response_mediation",
            "mechanism_hypothesis":"Effect support is already identifiable, but planned operator geometry is not a fold-stable conditional outcome state. Test whether the realized paired ego state transition over the exact one-shot exposure window is the missing outcome mediator.",
            "runtime_claim_boundary":"post-intervention mediator is not available for the initial t0 veto; this version performs mechanism identification only and emits no runtime retention config",
            "frozen_contract":{"RSMR_selector_unchanged":True,"V52_effect_support_unchanged":True,"V50_5_paired_outcomes_unchanged":True,"QPE_and_planned_D_unchanged":True,"lambda":1.0,
                "no_new_full_horizon_outcome_collection":True,"no_new_offline_future_observable":True,"no_rerank_second_best_fallback":True},
            "analysis_population_snapshot":{"path":str(a.output_analysis_table),"sha256":analysis_sha,"row_count":len(rows),"role":"audit_only_not_fit_input"},
            "nested_crossfit":nested,"mediator_identification_pass":bool(nested["mediator_identification_pass"]),"runtime_config_emitted":False,
            "preregistered_branch_order":["realized_endpoint","realized_temporal"],
            "preregistered_next_branch":{"if_realized_endpoint_identified":"realized one-replan state transition is the minimal mediator; next version must make it deployable via pre-execution prediction or a continuation operator, not use future state at t0",
                "if_only_realized_temporal_identified":"realized response shape across the one-shot window is necessary; next version predicts/uses that process under an auditable temporal operator",
                "if_identified_but_oracle_gate_fails":"dynamic state carries outcome signal but static sign functional remains insufficient; next step is structured dynamic outcome functional before any state expansion",
                "if_both_fail":"close short-horizon ego-response mediator; next paired evidence must add realized interaction/safety consequence process, not more ego trajectory geometry"},
            "prohibited_tuning":["post-intervention state used directly as t0 runtime input","response horizon sweep","DCT mode/basis sweep","threshold/alpha/lambda/loss/class/focal/catastrophe weighting","bigger MLP/attention","new offline future observables","RSMR/B/M/topK/candidate-count change","rerank/second-best/fallback","untouched validation before a deployable mechanism exists"]}
    a.output_report.parent.mkdir(parents=True,exist_ok=True); a.output_report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not nested["mediator_identification_pass"]:
        print(json.dumps({"pass":False,"failure_diagnosis":nested["failure_diagnosis"],"runtime_config_emitted":False},sort_keys=True)); raise SystemExit(f"V64.3.54 PDRM mediator scientific STOP ({nested['failure_diagnosis']})")
    print(json.dumps({"pass":True,"preferred_mediator_arm":nested["preferred_mediator_arm"],"runtime_config_emitted":False,"next":"design deployable mediator prediction/continuation; do not run untouched validation"},sort_keys=True))

if __name__=="__main__": main()
