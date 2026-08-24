from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from bdse.tools.fit_v64_3_32_1_eaf_icer_ssir_weightfix import _fit_ridge as _fit_mean_ridge, _predict as _predict_mean
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import (
    ALPHA, BASE_FEATURE_NAMES, CAT, EXPECTED_FRONTIER_ROWS, EXPECTED_SCENES,
    FEATURE_NAMES, FOLDS, RIDGE_LAMBDA, _conformal_q, _diag, _finite,
    _fit_pair_gap, _fold, _pair_scores, _read_edges, _scene_samples, _select,
    _teacher_best_index,
)
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import (
    MIN_NESTED_CAL_PROPOSALS, _extended_diag, _fit_regret_structured_margin,
    _structured_scores,
)

EPS = 1.0e-12
LBFGS_MAX_ITER = 250
LBFGS_HISTORY = 50
CONTEXT_NAMES = [f"incumbent::{n}" for n in BASE_FEATURE_NAMES] + ["incumbent::support_logit"]
CONTEXT_GAIN_MIN_FRACTION = 0.20
CAPTURE_TOL = 0.03


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


def _build_scene_data(groups: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    scene_map: dict[str, list[dict[str, Any]]] = {}
    context_map: dict[str, np.ndarray] = {}
    for tok, g in groups.items():
        ss = _scene_samples(g)
        cc = _scene_context(g)
        if ss and cc is not None:
            scene_map[tok] = ss
            context_map[tok] = cc
    return scene_map, context_map


def _delta_scale(scene_map: dict[str, list[dict[str, Any]]], tokens: list[str]) -> np.ndarray:
    xs: list[np.ndarray] = []
    ws: list[float] = []
    for tok in tokens:
        ss = scene_map.get(tok, [])
        if not ss:
            continue
        m = len(ss)
        for a in ss:
            xs.append(np.asarray(a["x"], dtype=np.float64)); ws.append(1.0 / m)
    if not xs:
        raise ValueError("V35 factorized structured fit has no delta features")
    X = np.stack(xs); w = np.asarray(ws, dtype=np.float64); p = w / max(float(w.sum()), EPS)
    return np.maximum(np.sqrt(np.sum((X * X) * p[:, None], axis=0)), 1.0e-6)


def _context_scale(context_map: dict[str, np.ndarray], tokens: list[str]) -> np.ndarray:
    C = np.stack([np.asarray(context_map[t], dtype=np.float64) for t in tokens if t in context_map])
    if C.size == 0:
        raise ValueError("V35 base-point context fit has no incumbent contexts")
    # One equal moment mass per scene; zero-preserving RMS, no post-hoc centering/threshold.
    return np.maximum(np.sqrt(np.mean(C * C, axis=0)), 1.0e-6)


def _factorized_objective_numpy(
    theta: np.ndarray, scene_map: dict[str, list[dict[str, Any]]], context_map: dict[str, np.ndarray],
    tokens: list[str], dscale: np.ndarray, cscale: np.ndarray | None,
) -> float:
    d = len(FEATURE_NAMES); wd = theta[:d]; wc = theta[d:] if cscale is not None else None
    total = RIDGE_LAMBDA * float(np.dot(theta, theta))
    for tok in tokens:
        ss = scene_map.get(tok, [])
        if not ss: continue
        X = np.stack([np.asarray(a["x"], dtype=np.float64) for a in ss]) / dscale[None, :]
        shift = 0.0 if wc is None else float((np.asarray(context_map[tok]) / cscale) @ wc)
        s = X @ wd + shift
        y = np.asarray([float(a["y"]) for a in ss], dtype=np.float64)
        bi = _teacher_best_index(ss)
        if bi < 0:
            ve = max(float(np.max(-y + s)), 0.0)
            total += ve * ve
        else:
            ve = max(float(y[bi] - s[bi]), 0.0)
            rivals = [j for j in range(len(ss)) if j != bi]
            vo = 0.0
            if rivals:
                rr = np.asarray(rivals, dtype=np.int64)
                vo = max(float(np.max((y[bi] - y[rr]) - (s[bi] - s[rr]))), 0.0)
            total += ve * ve + vo * vo
    return float(total)


def _fit_factorized(
    scene_map: dict[str, list[dict[str, Any]]], context_map: dict[str, np.ndarray], tokens: list[str], *, use_context: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    dscale = _delta_scale(scene_map, tokens)
    cscale = _context_scale(context_map, tokens) if use_context else None
    dim = len(FEATURE_NAMES) + (len(CONTEXT_NAMES) if use_context else 0)
    tw = torch.zeros((dim,), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [tw], lr=1.0, max_iter=LBFGS_MAX_ITER, history_size=LBFGS_HISTORY,
        tolerance_grad=1.0e-10, tolerance_change=1.0e-12, line_search_fn="strong_wolfe",
    )
    packed=[]
    for tok in tokens:
        ss=scene_map.get(tok,[])
        if not ss: continue
        X=np.stack([np.asarray(a["x"],dtype=np.float64) for a in ss])/dscale[None,:]
        y=np.asarray([float(a["y"]) for a in ss],dtype=np.float64)
        bi=_teacher_best_index(ss)
        c=None if not use_context else np.asarray(context_map[tok],dtype=np.float64)/cscale
        packed.append((torch.from_numpy(X),torch.from_numpy(y),bi,None if c is None else torch.from_numpy(c)))
    if not packed: raise ValueError("V35 factorized structured fit has no scenes")
    eval_count=0
    def closure():
        nonlocal eval_count
        opt.zero_grad(); loss=RIDGE_LAMBDA*torch.dot(tw,tw)
        wd=tw[:len(FEATURE_NAMES)]; wc=tw[len(FEATURE_NAMES):] if use_context else None
        for X,y,bi,c in packed:
            score=X.mv(wd)
            if use_context:
                score=score+torch.dot(c,wc)
            if bi < 0:
                ve=torch.clamp(torch.max(-y+score),min=0.0)
                loss=loss+ve*ve
            else:
                ve=torch.clamp(y[bi]-score[bi],min=0.0)
                rivals=[j for j in range(len(y)) if j!=bi]
                if rivals:
                    rr=torch.tensor(rivals,dtype=torch.long)
                    vo=torch.clamp(torch.max((y[bi]-y[rr])-(score[bi]-score[rr])),min=0.0)
                    loss=loss+ve*ve+vo*vo
                else:
                    loss=loss+ve*ve
        loss.backward(); eval_count+=1; return loss
    zero=np.zeros((dim,),dtype=np.float64)
    before=_factorized_objective_numpy(zero,scene_map,context_map,tokens,dscale,cscale)
    opt.step(closure)
    theta=tw.detach().cpu().numpy().astype(np.float64)
    after=_factorized_objective_numpy(theta,scene_map,context_map,tokens,dscale,cscale)
    if not np.all(np.isfinite(theta)) or not math.isfinite(after): raise RuntimeError("V35 solver produced non-finite parameters")
    if after > before + 1e-7*max(1.0,abs(before)): raise RuntimeError(f"V35 solver increased objective {before}->{after}")
    info={"solver":"torch_cpu_float64_lbfgs_strong_wolfe","max_iter":LBFGS_MAX_ITER,"history_size":LBFGS_HISTORY,
          "closure_evaluations":int(eval_count),"objective_at_zero":float(before),"objective_final":float(after),
          "scene_count":len(packed),"use_incumbent_context":bool(use_context),"weight_l2":float(np.linalg.norm(theta))}
    return theta,dscale,cscale,info


def _factorized_scores(ss: list[dict[str, Any]], context: np.ndarray, model) -> np.ndarray:
    theta,dscale,cscale,_=model; d=len(FEATURE_NAMES); wd=theta[:d]
    X=np.stack([np.asarray(a["x"],dtype=np.float64) for a in ss])/dscale[None,:]
    score=X@wd
    if cscale is not None:
        wc=theta[d:]; score=score+float((np.asarray(context,dtype=np.float64)/cscale)@wc)
    return np.clip(score,-40.0,40.0)


def _nested(groups: dict[str,list[dict[str,Any]]], audit_csv: Path) -> dict[str,Any]:
    scene_map,context_map=_build_scene_data(groups)
    names=["mean","pair","rsm","factor","context","main"]
    agg={n:[] for n in names}; caps={n:0 for n in names}; noops={n:0 for n in names}; oppsels={n:0 for n in names}
    total_opp=0; total_noopp=0; folds=[]; audits=[]
    for k in range(FOLDS):
        test=[t for t in scene_map if _fold(t)==k]; cf=(k+1)%FOLDS
        cal=[t for t in scene_map if _fold(t)==cf]; fit=[t for t in scene_map if _fold(t) not in {k,cf}]
        fit_samples=[a for t in fit for a in scene_map[t]]
        mean_model=_fit_mean_ridge(fit_samples); pair_model=_fit_pair_gap(scene_map,fit); rsm_model=_fit_regret_structured_margin(scene_map,fit)
        fac_model=_fit_factorized(scene_map,context_map,fit,use_context=False)
        ctx_model=_fit_factorized(scene_map,context_map,fit,use_context=True)
        cres=[]
        for t in cal:
            ss=scene_map[t]; sc=_factorized_scores(ss,context_map[t],ctx_model); j=_select(ss,sc)
            if j is not None: cres.append(float(sc[j]-float(ss[j]["y"])))
        if len(cres)<MIN_NESTED_CAL_PROPOSALS: q=float("inf"); qidx=-1
        else: q,qidx=_conformal_q(cres)
        fv={n:[] for n in names}; fc={n:0 for n in names}; fn={n:0 for n in names}; fo={n:0 for n in names}; opp=0; noop=0
        for t in test:
            ss=scene_map[t]; y=np.asarray([float(a["y"]) for a in ss]); has=bool(np.any(y>0)); opp+=int(has); noop+=int(not has)
            mm,_=_predict_mean(ss,mean_model); mi=_select(ss,mm)
            ps=_pair_scores(ss,pair_model); pi=_select(ss,ps)
            rs=_structured_scores(ss,rsm_model); ri=_select(ss,rs)
            fs=_factorized_scores(ss,context_map[t],fac_model); fi=_select(ss,fs)
            cs=_factorized_scores(ss,context_map[t],ctx_model); ci=_select(ss,cs)
            main_i=ci if (ci is not None and math.isfinite(q) and float(cs[ci]-q)>0.0) else None
            chosen={"mean":mi,"pair":pi,"rsm":ri,"factor":fi,"context":ci,"main":main_i}
            for n,j in chosen.items():
                if j is None: continue
                yy=float(y[j]); fv[n].append(yy); fc[n]+=int(has and yy>0); fn[n]+=int(not has); fo[n]+=int(has)
            audits.append({
                "scenario_token":t,"outer_test_fold":k,"calibration_fold":cf,"candidate_count":len(ss),"positive_opportunity":int(has),
                "teacher_best_action":int(ss[_teacher_best_index(ss)]["action"]) if _teacher_best_index(ss)>=0 else -1,
                "teacher_best_improvement":max(0.0,float(np.max(y))),
                "mean_selected_action":-1 if mi is None else int(ss[mi]["action"]),"mean_selected_teacher_improvement":float("nan") if mi is None else float(y[mi]),
                "pair_selected_action":-1 if pi is None else int(ss[pi]["action"]),"pair_selected_teacher_improvement":float("nan") if pi is None else float(y[pi]),
                "rsm_selected_action":-1 if ri is None else int(ss[ri]["action"]),"rsm_selected_score":float("nan") if ri is None else float(rs[ri]),"rsm_selected_teacher_improvement":float("nan") if ri is None else float(y[ri]),
                "factor_selected_action":-1 if fi is None else int(ss[fi]["action"]),"factor_selected_score":float("nan") if fi is None else float(fs[fi]),"factor_selected_teacher_improvement":float("nan") if fi is None else float(y[fi]),
                "context_selected_action":-1 if ci is None else int(ss[ci]["action"]),"context_selected_score":float("nan") if ci is None else float(cs[ci]),"context_selected_teacher_improvement":float("nan") if ci is None else float(y[ci]),
                "selected_policy_conformal_q":q,"main_selected_action":-1 if main_i is None else int(ss[main_i]["action"]),
                "main_selected_lower_bound":float("nan") if ci is None or not math.isfinite(q) else float(cs[ci]-q),
                "main_selected_teacher_improvement":float("nan") if main_i is None else float(y[main_i]),
            })
        fd={n:_extended_diag(fv[n],fc[n],opp,fn[n],fo[n],noop) for n in names}
        folds.append({"fold":k,"fit_scenes":len(fit),"calibration_scenes":len(cal),"test_scenes":len(test),
                      "selected_policy_calibration_proposal_count":len(cres),"selected_policy_conformal_order_index_1based":qidx,"selected_policy_conformal_quantile":q,
                      "factorized_solver":fac_model[3],"context_solver":ctx_model[3],
                      "mean_rank":fd["mean"],"v33_pair_gap_rank":fd["pair"],"v34_rsmr_rank":fd["rsm"],
                      "factorized_delta_rank":fd["factor"],"basepoint_context_rank":fd["context"],"basepoint_context_policy_conformal":fd["main"]})
        total_opp+=opp; total_noopp+=noop
        for n in names: agg[n].extend(fv[n]); caps[n]+=fc[n]; noops[n]+=fn[n]; oppsels[n]+=fo[n]
    audit_csv.parent.mkdir(parents=True,exist_ok=True)
    with audit_csv.open("w",encoding="utf-8",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(audits[0].keys())); wr.writeheader(); wr.writerows(audits)
    A={n:_extended_diag(agg[n],caps[n],total_opp,noops[n],oppsels[n],total_noopp) for n in names}
    rsm=A["rsm"]; factor=A["factor"]; ctx=A["context"]; main=A["main"]
    factorization_gain=bool(factor["no_positive_opportunity_false_intervention_count"]<rsm["no_positive_opportunity_false_intervention_count"] and factor["positive_capture_rate"]>=rsm["positive_capture_rate"]-CAPTURE_TOL)
    ctx_noop_target=(1.0-CONTEXT_GAIN_MIN_FRACTION)*float(factor["no_positive_opportunity_false_intervention_count"])
    context_boundary_gain=bool(ctx["no_positive_opportunity_false_intervention_count"]<=ctx_noop_target+EPS)
    context_preserves_ordering=bool(ctx["positive_capture_rate"]>=factor["positive_capture_rate"]-CAPTURE_TOL and ctx["teacher_improvement_sum"]>=-EPS)
    context_tail_gain=bool(ctx["catastrophic_count"]<rsm["catastrophic_count"] and ctx["teacher_negative_rms"]<=rsm["teacher_negative_rms"]+EPS)
    fold_direction=all(fr["basepoint_context_rank"]["teacher_improvement_sum"]>=-EPS for fr in folds)
    rank_pass=bool(context_boundary_gain and context_preserves_ordering and context_tail_gain and fold_direction and ctx["selected_count"]>=64 and ctx["selected_positive_count"]>=32)
    cal_support=all(fr["selected_policy_calibration_proposal_count"]>=MIN_NESTED_CAL_PROPOSALS for fr in folds)
    main_tail=all(fr["basepoint_context_policy_conformal"]["path_nonharmful"] and fr["basepoint_context_policy_conformal"]["catastrophic_count"]==0 for fr in folds)
    train_pass=bool(rank_pass and cal_support and main_tail and main["selected_count"]>=64 and main["selected_positive_count"]>=32)
    if not factorization_gain:
        diagnosis="factorized_existence_ordering_loss_does_not_resolve_v34_tradeoff_under_delta_only_representation"
    elif not context_boundary_gain:
        diagnosis="incumbent_basepoint_context_does_not_materially_reduce_noop_false_interventions_vs_factorized_delta"
    elif not context_preserves_ordering:
        diagnosis="basepoint_context_reduces_boundary_errors_but_damages_opportunity_recovery_or_selected_sum"
    elif not context_tail_gain or not fold_direction:
        diagnosis="basepoint_context_improves_existence_but_selected_tail_or_crossfold_direction_remains_unstable"
    elif not cal_support:
        diagnosis="rank_mechanism_passes_but_policy_output_density_is_insufficient_for_nested_calibration"
    elif not main_tail or main["selected_count"]<64 or main["selected_positive_count"]<32:
        diagnosis="rank_mechanism_passes_but_selected_policy_marginal_certificate_does_not_meet_hard_tail_coverage_contract"
    else:
        diagnosis="full_nested_train_pass"
    return {"folds":folds,"scene_audit_csv":str(audit_csv),"mean_rank_aggregate":A["mean"],"v33_pair_gap_rank_aggregate":A["pair"],"v34_rsmr_rank_aggregate":rsm,
            "factorized_delta_rank_aggregate":factor,"basepoint_context_rank_aggregate":ctx,"basepoint_context_policy_conformal_aggregate":main,
            "factorization_gain":factorization_gain,"context_boundary_gain":context_boundary_gain,"context_preserves_ordering":context_preserves_ordering,
            "context_tail_gain":context_tail_gain,"all_folds_context_selected_sum_nonnegative":fold_direction,"rank_mechanism_pass":rank_pass,
            "nested_calibration_support_pass":cal_support,"train_gate_pass":train_pass,"failure_diagnosis":diagnosis,
            "context_gain_min_fraction":CONTEXT_GAIN_MIN_FRACTION,"capture_tolerance":CAPTURE_TOL}


def _base_cfg(path:str)->dict[str,Any]:
    cfg=yaml.safe_load(Path(path).read_text()); ic=cfg.setdefault("runtime",{}).setdefault("decisive_frontier_value",{}).setdefault("incumbent_contrastive_extremal_recovery",{})
    ic["incumbent_retention_policy"]="preserve_admissible_incumbent"; ic["regret_risk_enabled"]=False; ic["retention_regret_risk_enabled"]=False; ic["replacement_regret_risk_enabled"]=False
    return cfg


def _write_cfg(base,path,*,version,name,algorithm,model,mode="rank_only",model_type=None,training_target=None,training_weighting=None):
    if len(model)==3:
        theta,dscale,info=model; cscale=None
    else:
        theta,dscale,cscale,info=model
    d=len(FEATURE_NAMES)
    cfg=yaml.safe_load(yaml.safe_dump(base,sort_keys=False)); ic=cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    sc=ic.setdefault("selection_conditioned_intervention_recovery",{})
    sc.update({"enabled":True,"mode":mode,"model_type":model_type or ("v35_factorized_basepoint_context_structured_recovery" if cscale is not None else "v35_factorized_delta_structured_recovery"),
               "base_feature_names":BASE_FEATURE_NAMES,"feature_names":FEATURE_NAMES,"feature_mean":[0.0]*d,"feature_std":[float(x) for x in dscale],"weights":[float(x) for x in theta[:d]],"bias":0.0,
               "ridge_lambda":RIDGE_LAMBDA,"leverage_inverse":[],"selection_scale_floor":1.0,"require_positive_predicted_improvement":True,"no_fallback":True,
               "training_population":"TRAIN_only_incumbent_deployment_admissible_support_positive_direct_scenes",
               "training_target":training_target or "factorized_intervention_existence_plus_conditional_challenger_regret_margins",
               "training_weighting":training_weighting or "one_existence_constraint_per_scene_plus_one_ordering_constraint_per_positive_opportunity_scene_no_candidate_count_dilution",
               "proposal_operator":"argmax_positive_structured_candidate_score_with_incumbent_zero_pseudoitem","conformal_alpha":ALPHA,"conformal_overprediction_quantile":0.0,
               "calibration_status":"not_yet_selected_policy_calibrated"})
    if cscale is not None:
        sc.update({"incumbent_context_feature_names":CONTEXT_NAMES,"incumbent_context_feature_mean":[0.0]*len(CONTEXT_NAMES),
                   "incumbent_context_feature_std":[float(x) for x in cscale],"incumbent_context_weights":[float(x) for x in theta[d:]],"incumbent_context_bias":0.0,
                   "context_operator":"candidate_independent_common_shift_cannot_change_challenger_ordering"})
    cfg.setdefault("metadata",{})["algorithm_version"]=version; cfg.setdefault("provenance",{})["algorithm_version"]=version
    cfg.setdefault("experiment",{})["name"]=name; cfg["experiment"]["algorithm"]=algorithm
    Path(path).write_text(yaml.safe_dump(cfg,sort_keys=False))


def main():
    ap=argparse.ArgumentParser(description="Fit V64.3.35 factorized basepoint-conditioned structured recovery")
    ap.add_argument("--train-frontier-edges",required=True); ap.add_argument("--base-config",required=True)
    for n in ["preserve","mean","rsmr","factor","rank"]: ap.add_argument(f"--output-{n}-config",required=True)
    ap.add_argument("--output-report",required=True); ap.add_argument("--output-scene-audit",required=True)
    a=ap.parse_args(); _,groups=_read_edges(Path(a.train_frontier_edges)); nested=_nested(groups,Path(a.output_scene_audit)); scene_map,context_map=_build_scene_data(groups)
    report={"audit":"v64_3_35_eaf_icer_fbcsr_fit","scientific_role":"TRAIN_only_factorized_vs_basepoint_context_structured_intervention_disambiguation_plus_selected_policy_gate",
            "frozen_train_scenes":len(groups),"direct_support_positive_training_scenes":len(scene_map),"feature_names":FEATURE_NAMES,"incumbent_context_feature_names":CONTEXT_NAMES,
            "ridge_lambda":RIDGE_LAMBDA,"conformal_alpha":ALPHA,"fit_uses_validation":False,"fit_uses_test":False,"nested_crossfit":nested,"train_gate_pass":nested["train_gate_pass"],
            "mechanism_hypothesis":"challenger ordering is contrastive, but intervention existence may require incumbent base-point context; context enters only as a candidate-independent scene shift and therefore cannot re-rank challengers",
            "factorized_objective":"positive-opportunity scenes get one incumbent-boundary regret margin plus one challenger-ordering max regret margin; no-opportunity scenes get one incumbent-vs-most-dangerous-challenger regret margin",
            "train_gate_contract":{"context_noop_false_intervention_reduction_vs_factorized_delta_fraction_min":CONTEXT_GAIN_MIN_FRACTION,"context_capture_tolerance":CAPTURE_TOL,
                                   "all_context_test_folds_selected_sum_nonnegative":True,"context_catastrophes_below_v34_rsmr":True,"context_negative_rms_not_worse_than_v34_rsmr":True,
                                   "nested_calibration_proposals_min_per_fold":MIN_NESTED_CAL_PROPOSALS,"all_main_folds_nonharmful_and_zero_catastrophe":True,"aggregate_main_selected_min":64,"aggregate_main_positive_min":32,
                                   "no_lambda_alpha_feature_runtime_threshold_sweep":True}}
    rp=Path(a.output_report); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(report,indent=2,sort_keys=True))
    if not report["train_gate_pass"]:
        print(json.dumps(report,indent=2,sort_keys=True)); raise SystemExit(f"V64.3.35 FBCSR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before CAL/fresh selection")
    base=_base_cfg(a.base_config)
    # Preserve and historical controls are emitted only after TRAIN pass, so no server fresh population is spent on a failed mechanism.
    pcfg=yaml.safe_load(yaml.safe_dump(base,sort_keys=False)); pcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]={"enabled":False}; Path(a.output_preserve_config).write_text(yaml.safe_dump(pcfg,sort_keys=False))
    samples=[x for ss in scene_map.values() for x in ss]; mm=_fit_mean_ridge(samples); mw,mb,mmean,mstd,_=mm
    # write MEAN directly with historical schema
    mcfg=yaml.safe_load(yaml.safe_dump(base,sort_keys=False)); sc=mcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"].setdefault("selection_conditioned_intervention_recovery",{})
    sc.update({"enabled":True,"mode":"mean_rank","model_type":"v32_1_scene_equal_edge_mean_ridge_control","base_feature_names":BASE_FEATURE_NAMES,"feature_names":FEATURE_NAMES,"feature_mean":[float(x) for x in mmean],"feature_std":[float(x) for x in mstd],"weights":[float(x) for x in mw],"bias":float(mb),"ridge_lambda":RIDGE_LAMBDA,"leverage_inverse":[],"no_fallback":True}); Path(a.output_mean_config).write_text(yaml.safe_dump(mcfg,sort_keys=False))
    rsm=_fit_regret_structured_margin(scene_map,list(scene_map)); _write_cfg(base,a.output_rsmr_config,version="V64.3.35-RSMR-CONTROL",name="v64_3_35_rsmr_control",algorithm="exact V34 RSMR control",model=rsm,model_type="incumbent_augmented_scene_max_teacher_regret_structured_margin",training_target="teacher_best_vs_worst_cost_augmented_rival_structured_regret_margin",training_weighting="one_max_regret_violation_per_direct_scene_plus_fixed_l2")
    fac=_fit_factorized(scene_map,context_map,list(scene_map),use_context=False); _write_cfg(base,a.output_factor_config,version="V64.3.35-FDSR",name="v64_3_35_factorized_delta",algorithm="factorized existence+ordering structured recovery on frozen 19-D contrast only",model=fac)
    ctx=_fit_factorized(scene_map,context_map,list(scene_map),use_context=True); _write_cfg(base,a.output_rank_config,version="V64.3.35-EAF-ICER-FBCSR-RANK",name="v64_3_35_eaf_icer_fbcsr_rank",algorithm="factorized basepoint-conditioned structured recovery rank",model=ctx)
    report["full_context_model"]={"delta_weights":[float(x) for x in ctx[0][:len(FEATURE_NAMES)]],"context_weights":[float(x) for x in ctx[0][len(FEATURE_NAMES):]],"delta_scale":[float(x) for x in ctx[1]],"context_scale":[float(x) for x in ctx[2]],"solver":ctx[3]}
    rp.write_text(json.dumps(report,indent=2,sort_keys=True)); print(json.dumps({"pass":True,"output_rank_config":a.output_rank_config},sort_keys=True))

if __name__=="__main__": main()
