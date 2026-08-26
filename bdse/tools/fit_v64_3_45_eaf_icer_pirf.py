from __future__ import annotations

import argparse,csv,json
from pathlib import Path
from typing import Any
import numpy as np, yaml

from bdse.planner.value_observables import VALUE_OBSERVABLE_NAMES,QUALITY_NAMES
from bdse.planner.response_value_observables import FUTURE_RESPONSE_OBSERVABLE_NAMES
from bdse.planner.interaction_response_field import RESPONSE_FIELD_OBSERVABLE_NAMES
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS,RIDGE_LAMBDA,_fold,_read_edges,_select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _fit_regret_structured_margin,_structured_scores
from bdse.tools.fit_v64_3_38_eaf_icer_davr import CAPTURE_TOL,CATASTROPHE_REDUCTION_MIN,NOOP_REDUCTION_MIN,_value_diag,_write_rsmr
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import _base_cfg,_fit_zero_ridge,_pred as _epv_pred
from bdse.tools.fit_v64_3_43_eaf_icer_cfrv import _scene as _v43_scene,_fit_weighted_zero_ridge,_quality_x,_quality_value,_metrics,_gate,_write_quality_control

ALL_OBSERVABLE_NAMES=list(VALUE_OBSERVABLE_NAMES)+list(FUTURE_RESPONSE_OBSERVABLE_NAMES)+list(RESPONSE_FIELD_OBSERVABLE_NAMES)
ARM_COST={'cv_occ':'response_field_cv_occupancy_cost','local_rf':'response_field_local_occupancy_cost','plan_rf':'response_field_plan_occupancy_cost'}


def _read_sidecar(path:Path):
    out={}
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line); t=str(r['scenario_token']); names=[str(x) for x in r['observable_names']]
        if names!=RESPONSE_FIELD_OBSERVABLE_NAMES: raise ValueError(f'V45 sidecar schema mismatch {t}')
        c=np.asarray(r['costs'],dtype=np.float64)
        if c.ndim!=2 or c.shape[1]!=3 or not np.all(np.isfinite(c)): raise ValueError(f'V45 invalid costs {t}')
        if t in out: raise ValueError(f'duplicate V45 sidecar {t}')
        out[t]=c
    return out


def _scene(groups,side):
    base=_v43_scene(groups); out={}
    for t,ss in base.items():
        if t not in side: raise ValueError(f'V45 sidecar missing {t}')
        c=side[t]; inc=int(groups[t][0].get('raw_top_action',-1))
        if not (0<=inc<len(c)): raise ValueError(f'V45 incumbent out of sidecar range {t}')
        rr=[]
        for a in ss:
            act=int(a['action'])
            if not (0<=act<len(c)): raise ValueError(f'V45 action out of sidecar range {t}/{act}')
            z=dict(a)
            for j,n in enumerate(RESPONSE_FIELD_OBSERVABLE_NAMES): z[n+'_improvement']=float(c[inc,j]-c[act,j])
            rr.append(z)
        out[t]=rr
    return out


def _rx(a,name): return np.asarray([float(a[name+'_improvement'])],dtype=np.float64)
def _rval(a,epv,q,r,name):
    base=_quality_value(a,epv,q); x=_rx(a,name); return float(np.clip(base+(x/np.maximum(np.asarray(r['scale']),1e-6))@np.asarray(r['weights']),-40,40))


def _nested(groups,side,audit_csv:Path,response_report:dict[str,Any]):
    scene=_scene(groups,side); arms=['rsmr','quality','cv_occ','local_rf','plan_rf']; agg={a:[] for a in arms};caps={a:0 for a in arms};noops={a:0 for a in arms};oppsels={a:0 for a in arms};total_opp=total_noop=0;folds=[];aud=[];vy=[];vp={a:[] for a in arms}
    for k in range(FOLDS):
        test=[t for t in scene if _fold(t)==k];cf=(k+1)%FOLDS;cal=[t for t in scene if _fold(t)==cf];fit=[t for t in scene if _fold(t) not in {k,cf}]
        rsm=_fit_regret_structured_margin(scene,fit);epv=_fit_zero_ridge(scene,fit,'epv');q=_fit_weighted_zero_ridge(scene,fit,lambda a:float(a['y'])-_epv_pred(a,epv),_quality_x,QUALITY_NAMES)
        rs={arm:_fit_weighted_zero_ridge(scene,fit,lambda a:float(a['y'])-_quality_value(a,epv,q),lambda a,n=n:_rx(a,n),[n]) for arm,n in ARM_COST.items()}
        fv={a:[] for a in arms};fc={a:0 for a in arms};fn={a:0 for a in arms};fo={a:0 for a in arms};opp=noopsc=0;subset=identity=True
        for t in test:
            ss=scene[t];yy=np.asarray([float(a['y']) for a in ss]);has=bool(np.any(yy>0));opp+=int(has);noopsc+=int(not has);score=_structured_scores(ss,rsm);idx=_select(ss,score)
            vals={a:float('nan') for a in arms};chosen={a:None for a in arms};chosen['rsmr']=idx
            if idx is not None:
                a=ss[idx];vals['rsmr']=float(score[idx]);qv=_quality_value(a,epv,q);vals['quality']=qv;chosen['quality']=idx if qv>0 else None
                for arm,n in ARM_COST.items():
                    v=_rval(a,epv,q,rs[arm],n);vals[arm]=v;chosen[arm]=idx if v>0 else None
                vy.append(float(yy[idx]));[vp[n].append(float(vals[n])) for n in arms]
            subset=subset and all(chosen[n] is None or idx is not None for n in arms if n!='rsmr');identity=identity and all(chosen[n] is None or chosen[n]==idx for n in arms if n!='rsmr')
            for n,ii in chosen.items():
                if ii is None:continue
                v=float(yy[ii]);fv[n].append(v);fc[n]+=int(has and v>0);fn[n]+=int(not has);fo[n]+=int(has)
            aud.append({'scenario_token':t,'outer_test_fold':k,'calibration_fold':cf,'candidate_count':len(ss),'positive_opportunity':int(has),'rsm_selected_action':-1 if idx is None else int(ss[idx]['action']),'rsm_selected_teacher_improvement':float('nan') if idx is None else float(yy[idx]),**{f'{n}_selected_action':-1 if chosen[n] is None else int(ss[chosen[n]]['action']) for n in arms if n!='rsmr'},**{f'{n}_value':float(vals[n]) for n in arms if n!='rsmr'}})
        total_opp+=opp;total_noop+=noopsc;fd={}
        for n in arms: fd[n]=_metrics(fv[n],fc[n],opp,fn[n],fo[n],noopsc);agg[n]+=fv[n];caps[n]+=fc[n];noops[n]+=fn[n];oppsels[n]+=fo[n]
        folds.append({'fold':k,'fit_scenes':len(fit),'value_calibration_scenes':len(cal),'test_scenes':len(test),**{n:fd[n] for n in arms},'monotone_subset_valid':subset,'frozen_winner_identity_valid':identity})
    audit_csv.parent.mkdir(parents=True,exist_ok=True)
    with audit_csv.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(aud[0]));w.writeheader();w.writerows(aud)
    A={n:_metrics(agg[n],caps[n],total_opp,noops[n],oppsels[n],total_noop) for n in arms};vd={n:_value_diag(vy,vp[n]) for n in arms};g={n:_gate(A[n],A['rsmr'],folds,n) for n in arms if n!='rsmr'};contracts=all(f['monotone_subset_valid'] and f['frozen_winner_identity_valid'] for f in folds)
    local_ident=bool(response_report['aggregate']['local_mse']<response_report['aggregate']['cv_mse'] and response_report.get('local_better_than_cv_fold_count',0)>=4)
    plan_ident=bool(response_report.get('plan_response_identified',False))
    promotion={'cv_occ':bool(g['cv_occ']['pass']),'local_rf':bool(g['local_rf']['pass'] and local_ident),'plan_rf':bool(g['plan_rf']['pass'] and plan_ident)}
    preferred='cv_occ' if promotion['cv_occ'] else ('local_rf' if promotion['local_rf'] else ('plan_rf' if promotion['plan_rf'] else None))
    if preferred=='cv_occ': diag='continuous_ungated_occupancy_support_sufficient_without_response_learning'
    elif preferred=='local_rf': diag='agent_local_continuous_response_is_required_but_ego_plan_conditioning_not_required'
    elif preferred=='plan_rf': diag='agent_local_plan_conditioned_continuous_response_field_supported'
    elif g['plan_rf']['pass'] and not plan_ident: diag='value_gain_without_independent_plan_response_identification_fail_closed'
    elif plan_ident: diag='plan_conditioned_response_is_identifiable_but_absolute_zero_or_remaining_consequence_family_is_insufficient'
    else: diag='continuous_response_field_not_identified_or_not_value_sufficient_require_general_plan_conditioned_occupancy_response'
    return {'folds':folds,'scene_audit_csv':str(audit_csv),'rsmr_rank_aggregate':A['rsmr'],'quality_control_aggregate':A['quality'],'cv_occupancy_aggregate':A['cv_occ'],'local_response_field_aggregate':A['local_rf'],'plan_response_field_aggregate':A['plan_rf'],'selected_proposal_value_prediction_diagnostics':vd,'gates':g,'response_identification':{'local_identified':local_ident,'plan_identified':plan_ident,'crossfit_report':response_report},'promotion_eligible':promotion,'preferred_promotion_arm':preferred,'monotone_frozen_winner_contract_valid':contracts,'train_gate_pass':bool(contracts and preferred is not None),'failure_diagnosis':diag}


def _check_v44(path:Path):
    r=json.loads(path.read_text()); n=r.get('nested_crossfit',{}); exp={'rsmr_rank_aggregate':(502,221,107,28,43.29405361274824),'quality_control_aggregate':(205,129,30,13,43.905547394411805),'pc_occupancy_mean_aggregate':(218,124,41,9,60.375374572449246),'pc_occupancy_robust_aggregate':(222,125,44,8,61.61711750781815)}
    for k,e in exp.items():
        d=n.get(k,{})
        got=(d.get('selected_count'),d.get('selected_positive_count'),d.get('no_positive_opportunity_false_intervention_count'),d.get('catastrophic_count'),d.get('teacher_improvement_sum'))
        if any(got[i]!=e[i] for i in range(4)) or abs(float(got[4])-e[4])>1e-9: raise RuntimeError(f'V45 ENGINEERING STOP: V44 signature mismatch {k}: {got}')
    b=n.get('behavior_crossfit_summary',{})
    if abs(float(b.get('accuracy',-1))-0.8523333333333334)>1e-12 or abs(float(b.get('majority_baseline_accuracy',-2))-0.8523333333333334)>1e-12: raise RuntimeError('V45 ENGINEERING STOP: V44 behavior failure signature mismatch')


def _decorate(rsmcfg,epv,q,residual,model,obs,path,version,plan_enabled):
    cfg=yaml.safe_load(yaml.safe_dump(rsmcfg,sort_keys=False));ic=cfg['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery'];ic['instrument_value_observables']=True;ic['instrument_future_response_observables']=True;ic['instrument_plan_conditioned_response_observables']=False;ic['instrument_interaction_response_field_observables']=True;sc=ic['selection_conditioned_intervention_recovery']
    mm=dict(model);mm['enabled']=obs!='response_field_cv_occupancy_cost';mm['plan_enabled']=bool(plan_enabled and mm['enabled'])
    sc.update({'post_selection_value_enabled':True,'post_selection_value_mode':'endpoint_potential_quality_interaction_response_field','post_selection_endpoint_feature_names':list(epv['names']),'post_selection_endpoint_feature_scale':[float(x) for x in epv['scale']],'post_selection_endpoint_weights':[float(x) for x in epv['weights']],'post_selection_endpoint_bias':0.0,'post_selection_observable_names':list(ALL_OBSERVABLE_NAMES),'post_selection_quality_observable_names':list(QUALITY_NAMES),'post_selection_quality_observable_scale':[float(x) for x in q['scale']],'post_selection_quality_observable_weights':[float(x) for x in q['weights']],'post_selection_future_response_observable_name':obs,'post_selection_future_response_scale':float(residual['scale'][0]),'post_selection_future_response_weight':float(residual['weights'][0]),'post_selection_selected_bias':0.0,'interaction_response_field':mm,'post_selection_value_training':'scene_equal_all_edge_EPV_plus_QUALITY_plus_agent_local_continuous_response_field_occupancy_residual_fixed_lambda_1','post_selection_operator':'freeze_RSMR_winner_then_response_field_value_accept_same_winner_iff_positive_else_incumbent_no_rerank_no_fallback'})
    cfg.setdefault('metadata',{})['algorithm_version']=version;cfg.setdefault('provenance',{})['algorithm_version']=version;cfg.setdefault('experiment',{})['name']=version.lower().replace('.','_').replace('-','_');cfg['experiment']['algorithm']=version;Path(path).write_text(yaml.safe_dump(cfg,sort_keys=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-frontier-edges',required=True);ap.add_argument('--response-sidecar',required=True);ap.add_argument('--response-model',required=True);ap.add_argument('--response-report',required=True);ap.add_argument('--v44-fit-report',required=True);ap.add_argument('--base-config',required=True)
    for n in ['rsmr','quality','cv_occ','local_rf','plan_rf']: ap.add_argument(f'--output-{n.replace("_","-")}-config',dest=f'output_{n}_config',required=True)
    ap.add_argument('--output-report',required=True);ap.add_argument('--output-scene-audit',required=True);a=ap.parse_args();_check_v44(Path(a.v44_fit_report))
    _,groups=_read_edges(Path(a.train_frontier_edges));side=_read_sidecar(Path(a.response_sidecar));rr=json.loads(Path(a.response_report).read_text());nested=_nested(groups,side,Path(a.output_scene_audit),rr)
    r=nested['rsmr_rank_aggregate'];q0=nested['quality_control_aggregate']
    if r['selected_count']!=502 or r['selected_positive_count']!=221 or abs(r['teacher_improvement_sum']-43.29405361274824)>1e-9 or q0['selected_count']!=205 or q0['selected_positive_count']!=129 or abs(q0['teacher_improvement_sum']-43.905547394411805)>1e-9: raise RuntimeError('V45 ENGINEERING STOP: sidecar/value instrumentation changed frozen RSMR/QUALITY')
    report={'audit':'v64_3_45_eaf_icer_pirf_fit','scientific_role':'TRAIN_only_agent_local_continuous_plan_conditioned_response_field_after_frozen_RSMR_and_QUALITY','frozen_train_scenes':len(groups),'ridge_lambda':RIDGE_LAMBDA,'mechanism_hypothesis':'V44 proves ungated full-horizon occupancy support is a strong mediator but scene-global five-mode behavior supervision collapses to the yield majority. V45 replaces that target with agent-local continuous longitudinal response, separately testing no-response CV support, candidate-independent local response, and zero-at-zero-interaction ego-plan-conditioned response.','nested_crossfit':nested,'train_gate_pass':nested['train_gate_pass'],'train_gate_contract':{'V44_failure_and_success_signatures_are_exact_hard_gate':True,'RSMR_is_sole_challenger_selector':True,'response_supervision_is_agent_local_continuous_and_uses_no_teacher_value':True,'response_models_are_outer_fold_isolated':True,'deployment_uses_no_logged_future':True,'plan_response_correction_has_zero_bias_and_vanishes_with_interaction_exposure':True,'full_horizon_occupancy_support_reuses_V44_geometry_no_new_threshold':True,'no_selected_translation_or_CVaR_tuning':True,'capture_tolerance':CAPTURE_TOL,'noop_reduction_min':NOOP_REDUCTION_MIN,'catastrophe_reduction_min':CATASTROPHE_REDUCTION_MIN}}
    Path(a.output_report).write_text(json.dumps(report,indent=2,sort_keys=True))
    scene=_scene(groups,side);base=_base_cfg(a.base_config);base['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['instrument_value_observables']=True;base['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['instrument_future_response_observables']=True
    rsm=_fit_regret_structured_margin(scene,list(scene));rsmcfg=_write_rsmr(base,a.output_rsmr_config,rsm);Path(a.output_rsmr_config).write_text(yaml.safe_dump(rsmcfg,sort_keys=False));epv=_fit_zero_ridge(scene,list(scene),'epv');q=_fit_weighted_zero_ridge(scene,list(scene),lambda x:float(x['y'])-_epv_pred(x,epv),_quality_x,QUALITY_NAMES);_write_quality_control(rsmcfg,epv,q,a.output_quality_config)
    model=json.loads(Path(a.response_model).read_text())
    for arm,n in ARM_COST.items():
        res=_fit_weighted_zero_ridge(scene,list(scene),lambda x:float(x['y'])-_quality_value(x,epv,q),lambda x,nn=n:_rx(x,nn),[n]);_decorate(rsmcfg,epv,q,res,model,n,getattr(a,f'output_{arm}_config'),f'V64.3.45-EAF-ICER-PIRF-{arm.upper()}',arm=='plan_rf')
    print(json.dumps({'pass':nested['train_gate_pass'],'preferred_promotion_arm':nested['preferred_promotion_arm'],'failure_diagnosis':nested['failure_diagnosis']},sort_keys=True))
    if not nested['train_gate_pass']: raise SystemExit(f"V64.3.45 PIRF nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before fresh selection")

if __name__=='__main__': main()
