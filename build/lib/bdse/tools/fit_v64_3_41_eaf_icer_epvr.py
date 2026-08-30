from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from typing import Any
import numpy as np, yaml
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import BASE_FEATURE_NAMES, FOLDS, RIDGE_LAMBDA, _finite, _fold, _read_edges, _select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _fit_regret_structured_margin,_structured_scores,_extended_diag
from bdse.tools.fit_v64_3_38_eaf_icer_davr import CAT,CAPTURE_TOL,CATASTROPHE_REDUCTION_MIN,MIN_VALUE_CAL_PROPOSALS,NOOP_REDUCTION_MIN,_build,_fit_dense_value_ridge,_value_diag,_write_dense,_write_rsmr
from bdse.tools.fit_v64_3_39_eaf_icer_cfsr import _fit_translation
EPS=1e-12
ENDPOINT_NAMES=list(BASE_FEATURE_NAMES)+['support_logit']
ZDELTA_NAMES=[f'zdelta::{n}' for n in ENDPOINT_NAMES]
DNL_NAMES=ZDELTA_NAMES+[f'delta_signed_square::{n}' for n in ENDPOINT_NAMES]
EPV_NAMES=ZDELTA_NAMES+[f'midpoint_times_delta::{n}' for n in ENDPOINT_NAMES]

def _endpoint_scene(groups):
    scene=_build(groups)
    out={}
    for tok,ss in scene.items():
        g=groups[tok]; inc=int(g[0].get('raw_top_action',-1)); ir=next((r for r in g if int(r.get('challenger_action',-2))==inc),None)
        if ir is None: continue
        qi=np.asarray([_finite(ir,f'icer_feature_{n}') for n in BASE_FEATURE_NAMES]+[_finite(ir,'icer_support_logit')],dtype=np.float64)
        if not np.all(np.isfinite(qi)): continue
        byact={int(r.get('challenger_action',-2)):r for r in g}
        rr=[]
        for a in ss:
            row=byact.get(int(a['action']))
            if row is None: raise RuntimeError('V41 endpoint replay missing candidate row')
            qb=np.asarray([_finite(row,f'icer_feature_{n}') for n in BASE_FEATURE_NAMES]+[_finite(row,'icer_support_logit')],dtype=np.float64)
            if not np.all(np.isfinite(qb)): raise RuntimeError('V41 endpoint feature nonfinite')
            b=dict(a); b['q_inc']=qi.copy(); b['q_cand']=qb; b['delta_endpoint']=qb-qi
            if np.max(np.abs(b['delta_endpoint']-np.asarray(a['x'],dtype=np.float64)))>1e-9:
                raise RuntimeError('V41 endpoint delta does not replay frozen 19-D contrast')
            rr.append(b)
        out[tok]=rr
    return out

def _phi(a,mode):
    d=np.asarray(a['delta_endpoint'],dtype=np.float64); qi=np.asarray(a['q_inc']); qb=np.asarray(a['q_cand'])
    if mode=='zdelta': return d
    if mode=='dnl': return np.concatenate([d,d*np.abs(d)])
    if mode=='epv':
        m=0.5*(qi+qb); return np.concatenate([d,m*d])
    raise ValueError(mode)

def _fit_zero_ridge(scene,tokens,mode):
    X=[]; y=[]; w=[]
    for t in tokens:
        ss=scene[t]; n=len(ss)
        if n<=0: continue
        for a in ss:
            X.append(_phi(a,mode)); y.append(float(a['y'])); w.append(1.0/n)
    X=np.stack(X); y=np.asarray(y); w=np.asarray(w)
    pm=w/max(float(w.sum()),EPS); scale=np.sqrt(np.sum((X*X)*pm[:,None],axis=0)); scale=np.maximum(scale,1e-6)
    Z=X/scale[None,:]; root=np.sqrt(w)[:,None]; Zw=Z*root; yw=y*root[:,0]
    A=Zw.T@Zw+np.eye(Z.shape[1])*RIDGE_LAMBDA; rhs=Zw.T@yw; coef=np.linalg.solve(A,rhs)
    names={'zdelta':ZDELTA_NAMES,'dnl':DNL_NAMES,'epv':EPV_NAMES}[mode]
    return {'mode':mode,'names':names,'scale':scale,'weights':coef,'bias':0.0,'scene_weight_sum':float(w.sum()),'sample_count':len(y)}

def _pred(a,m): return float(np.clip((_phi(a,m['mode'])/m['scale'])@m['weights'],-40,40))

def _metrics(vals,captured,opp,noop_selected,opp_selected,noop_scenes): return _extended_diag(vals,captured,opp,noop_selected,opp_selected,noop_scenes)

def _gate(m,r,folds,key):
    exist=(m['no_positive_opportunity_false_intervention_count'] <= (1-NOOP_REDUCTION_MIN)*r['no_positive_opportunity_false_intervention_count']+EPS and m['positive_capture_rate'] >= r['positive_capture_rate']-CAPTURE_TOL-EPS)
    tail=(m['catastrophic_count'] <= (1-CATASTROPHE_REDUCTION_MIN)*r['catastrophic_count']+EPS and m['teacher_negative_rms'] <= r['teacher_negative_rms']+EPS and m['teacher_improvement_sum']>=-EPS)
    fd=all(float(f[key]['teacher_improvement_sum'])>=-EPS for f in folds); pop=m['selected_count']>=64 and m['selected_positive_count']>=32
    return {'existence_and_capture':bool(exist),'tail':bool(tail),'all_folds_sum_nonnegative':bool(fd),'population':bool(pop),'pass':bool(exist and tail and fd and pop)}

def _eval(ss,rsm,dense,zd,dnl,epv,bias):
    score=_structured_scores(ss,rsm); idx=_select(ss,score); names=['rsmr','dense','zdelta','dnl','epv_raw','epv_main']
    if idx is None: return {n:(None,float('nan')) for n in names}
    # historical dense helper inline
    from bdse.tools.fit_v64_3_32_1_eaf_icer_ssir_weightfix import _predict as _pr
    dv=float(_pr(ss,dense)[0][idx]); zv=_pred(ss[idx],zd); nv=_pred(ss[idx],dnl); pv=_pred(ss[idx],epv); mv=float(np.clip(pv+bias,-40,40))
    return {'rsmr':(idx,float(score[idx])),'dense':(idx if dv>0 else None,dv),'zdelta':(idx if zv>0 else None,zv),'dnl':(idx if nv>0 else None,nv),'epv_raw':(idx if pv>0 else None,pv),'epv_main':(idx if mv>0 else None,mv)}

def _nested(groups,audit_csv):
    scene=_endpoint_scene(groups); names=['rsmr','dense','zdelta','dnl','epv_raw','epv_main']
    agg={n:[] for n in names}; caps={n:0 for n in names}; noops={n:0 for n in names}; oppsels={n:0 for n in names}; total_opp=total_noop=0
    folds=[]; audits=[]; vy=[]; vp={n:[] for n in names}
    for k in range(FOLDS):
        test=[t for t in scene if _fold(t)==k]; cf=(k+1)%FOLDS; cal=[t for t in scene if _fold(t)==cf]; fit=[t for t in scene if _fold(t) not in {k,cf}]
        rsm=_fit_regret_structured_margin(scene,fit); samples=[a for t in fit for a in scene[t]]; dense=_fit_dense_value_ridge(samples)
        zd=_fit_zero_ridge(scene,fit,'zdelta'); dnl=_fit_zero_ridge(scene,fit,'dnl'); epv=_fit_zero_ridge(scene,fit,'epv')
        cy=[]; cp=[]; used=[]
        for t in cal:
            ss=scene[t]; score=_structured_scores(ss,rsm); idx=_select(ss,score)
            if idx is None: continue
            cy.append(float(ss[idx]['y'])); cp.append(_pred(ss[idx],epv)); used.append(t)
        if len(used)<MIN_VALUE_CAL_PROPOSALS: raise ValueError(f'V41 calibration proposals {len(used)} < {MIN_VALUE_CAL_PROPOSALS}')
        shift=_fit_translation(np.asarray(cp),np.asarray(cy),'endpoint_potential_value')
        fv={n:[] for n in names}; fc={n:0 for n in names}; fn={n:0 for n in names}; fo={n:0 for n in names}; opp=noopsc=0; subset=identity=True
        for t in test:
            ss=scene[t]; yy=np.asarray([float(a['y']) for a in ss]); has=bool(np.any(yy>0)); opp+=int(has); noopsc+=int(not has)
            ev=_eval(ss,rsm,dense,zd,dnl,epv,shift['selected_policy_bias']); ridx=ev['rsmr'][0]
            if ridx is not None:
                vy.append(float(yy[ridx])); [vp[n].append(float(ev[n][1])) for n in names]
            chosen={n:ev[n][0] for n in names}; subset=subset and all(chosen[n] is None or ridx is not None for n in names if n!='rsmr'); identity=identity and all(chosen[n] is None or chosen[n]==ridx for n in names if n!='rsmr')
            for n,idx in chosen.items():
                if idx is None: continue
                val=float(yy[idx]); fv[n].append(val); fc[n]+=int(has and val>0); fn[n]+=int(not has); fo[n]+=int(has)
            audits.append({'scenario_token':t,'outer_test_fold':k,'calibration_fold':cf,'candidate_count':len(ss),'positive_opportunity':int(has),'rsm_selected_action':-1 if ridx is None else int(ss[ridx]['action']),'rsm_selected_score':float(ev['rsmr'][1]),'rsm_selected_teacher_improvement':float('nan') if ridx is None else float(yy[ridx]),**{f'{n}_selected_action':-1 if chosen[n] is None else int(ss[chosen[n]]['action']) for n in names if n!='rsmr'},**{f'{n}_value':float(ev[n][1]) for n in names if n!='rsmr'}})
        total_opp+=opp; total_noop+=noopsc; fd={}
        for n in names:
            fd[n]=_metrics(fv[n],fc[n],opp,fn[n],fo[n],noopsc); agg[n]+=fv[n]; caps[n]+=fc[n]; noops[n]+=fn[n]; oppsels[n]+=fo[n]
        folds.append({'fold':k,'fit_scenes':len(fit),'value_calibration_scenes':len(cal),'test_scenes':len(test),'value_calibration_proposal_count':len(used),'epv_translation_fit':shift,'zdelta_fit':{x:zd[x] if x not in ['scale','weights'] else None for x in ['sample_count','scene_weight_sum']},'dnl_fit':{x:dnl[x] if x not in ['scale','weights'] else None for x in ['sample_count','scene_weight_sum']},'epv_fit':{x:epv[x] if x not in ['scale','weights'] else None for x in ['sample_count','scene_weight_sum']},**{n:fd[n] for n in names},'monotone_subset_valid':subset,'frozen_winner_identity_valid':identity})
    audit_csv.parent.mkdir(parents=True,exist_ok=True); fields=list(audits[0]);
    with audit_csv.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(audits)
    A={n:_metrics(agg[n],caps[n],total_opp,noops[n],oppsels[n],total_noop) for n in names}; vd={n:_value_diag(vy,vp[n]) for n in names}; gates={n:_gate(A[n],A['rsmr'],folds,n) for n in names if n!='rsmr'}
    contracts=all(f['monotone_subset_valid'] and f['frozen_winner_identity_valid'] for f in folds); passed=bool(contracts and gates['epv_main']['pass'])
    # diagnostic branch logic is preregistered and descriptive only.
    if gates['epv_main']['pass']: diag='endpoint_potential_value_closes_intervention_boundary_tradeoff'
    elif vd['epv_raw'].get('positive_auc',-9)>vd['zdelta'].get('positive_auc',-9)+0.03 and vd['epv_raw'].get('positive_auc',-9)>vd['dnl'].get('positive_auc',-9)+0.02: diag='basepoint_conditioned_endpoint_gradient_adds_cardinal_signal_but_selected_zero_or_tail_remains'
    elif vd['dnl'].get('positive_auc',-9)>vd['zdelta'].get('positive_auc',-9)+0.03 and vd['dnl'].get('positive_auc',-9)>=vd['epv_raw'].get('positive_auc',-9)-0.01: diag='generic_delta_nonlinearity_is_primary_representation_mediator_not_basepoint_interaction'
    elif vd['epv_raw'].get('noncatastrophe_auc',-9)>vd['zdelta'].get('noncatastrophe_auc',-9)+0.05: diag='endpoint_potential_adds_tail_signal_but_zero_crossing_still_fails'
    else: diag='existing_EAF_endpoint_observables_insufficient_for_selected_absolute_value_close_feature_head_route'
    return {'folds':folds,'scene_audit_csv':str(audit_csv),'rsmr_rank_aggregate':A['rsmr'],'dense_19d_aggregate':A['dense'],'zero_delta_aggregate':A['zdelta'],'delta_nonlinear_aggregate':A['dnl'],'endpoint_potential_raw_aggregate':A['epv_raw'],'endpoint_potential_main_aggregate':A['epv_main'],'selected_proposal_value_prediction_diagnostics':vd,'gates':gates,'monotone_frozen_winner_contract_valid':contracts,'train_gate_pass':passed,'failure_diagnosis':diag}

def _base_cfg(path):
    cfg=yaml.safe_load(Path(path).read_text()); ic=cfg.setdefault('runtime',{}).setdefault('decisive_frontier_value',{}).setdefault('incumbent_contrastive_extremal_recovery',{}); ic['incumbent_retention_policy']='preserve_admissible_incumbent'; ic['regret_risk_enabled']=ic['retention_regret_risk_enabled']=ic['replacement_regret_risk_enabled']=False; return cfg

def _decorate(rsm_cfg,m,mode,path,version,bias=0.0):
    cfg=yaml.safe_load(yaml.safe_dump(rsm_cfg,sort_keys=False)); sc=cfg['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['selection_conditioned_intervention_recovery']; sc.update({'post_selection_value_enabled':True,'post_selection_value_mode':mode,'post_selection_endpoint_feature_names':m['names'],'post_selection_endpoint_feature_scale':[float(x) for x in m['scale']],'post_selection_endpoint_weights':[float(x) for x in m['weights']],'post_selection_endpoint_bias':0.0,'post_selection_selected_bias':float(bias),'post_selection_value_training':'scene_equal_all_edge_zero_preserving_endpoint_value_fixed_lambda_1','post_selection_operator':'freeze_RSMR_winner_then_endpoint_value_accept_same_winner_iff_positive_else_incumbent_no_rerank_no_fallback'}); cfg.setdefault('metadata',{})['algorithm_version']=version; cfg.setdefault('provenance',{})['algorithm_version']=version; cfg.setdefault('experiment',{})['name']=version.lower().replace('.','_').replace('-','_'); cfg['experiment']['algorithm']=version; Path(path).write_text(yaml.safe_dump(cfg,sort_keys=False)); return cfg

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-frontier-edges',required=True); ap.add_argument('--base-config',required=True); ap.add_argument('--output-preserve-config',required=True); ap.add_argument('--output-rsmr-config',required=True); ap.add_argument('--output-dense-config',required=True); ap.add_argument('--output-zdelta-config',required=True); ap.add_argument('--output-dnl-config',required=True); ap.add_argument('--output-epv-config',required=True); ap.add_argument('--output-report',required=True); ap.add_argument('--output-scene-audit',required=True); a=ap.parse_args()
    _,groups=_read_edges(Path(a.train_frontier_edges)); nested=_nested(groups,Path(a.output_scene_audit)); report={'audit':'v64_3_41_eaf_icer_epvr_fit','scientific_role':'TRAIN_only_frozen_RSMR_plus_value_specific_endpoint_representation_ablation','frozen_train_scenes':len(groups),'direct_support_positive_training_scenes':len(_build(groups)),'ridge_lambda':RIDGE_LAMBDA,'mechanism_hypothesis':'V40 falsifies further target/head changes on the pure 19-D delta value route. Test whether selected absolute value requires endpoint-conditioned local utility geometry. Compare zero-preserving linear delta, generic nonlinear delta, and an antisymmetric endpoint-potential difference U(q_b)-U(q_i)=a^T delta+b^T(midpoint*delta), while RSMR alone freezes winner identity.','nested_crossfit':nested,'train_gate_pass':nested['train_gate_pass'],'train_gate_contract':{'RSMR_is_sole_challenger_selector':True,'all_value_arms_are_same_winner_subsets':True,'endpoint_potential_is_antisymmetric_and_zero_for_identical_endpoints':True,'no_naive_candidate_incumbent_concat':True,'dense_all_edge_scene_equal_fixed_lambda_1':True,'selected_policy_calibration_is_translation_only':True,'noop_false_intervention_reduction_fraction_min':NOOP_REDUCTION_MIN,'capture_tolerance':CAPTURE_TOL,'catastrophe_reduction_fraction_min':CATASTROPHE_REDUCTION_MIN,'all_test_folds_selected_sum_nonnegative':True,'selected_min':64,'positive_min':32,'no_threshold_lambda_alpha_feature_candidate_count_or_temperature_sweep':True}}
    Path(a.output_report).write_text(json.dumps(report,indent=2,sort_keys=True))
    if not nested['train_gate_pass']:
        print(json.dumps(report,indent=2,sort_keys=True)); raise SystemExit(f"V64.3.41 EPVR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before CAL/fresh selection")
    scene=_endpoint_scene(groups); base=_base_cfg(a.base_config); pcfg=yaml.safe_load(yaml.safe_dump(base,sort_keys=False)); pcfg['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['selection_conditioned_intervention_recovery']={'enabled':False}; Path(a.output_preserve_config).write_text(yaml.safe_dump(pcfg,sort_keys=False))
    rsm=_fit_regret_structured_margin(scene,list(scene)); rsmcfg=_write_rsmr(base,a.output_rsmr_config,rsm); _write_dense(rsmcfg,a.output_dense_config,_fit_dense_value_ridge([x for s in scene.values() for x in s])); zd=_fit_zero_ridge(scene,list(scene),'zdelta'); dn=_fit_zero_ridge(scene,list(scene),'dnl'); ep=_fit_zero_ridge(scene,list(scene),'epv'); _decorate(rsmcfg,zd,'endpoint_zero_delta',a.output_zdelta_config,'V64.3.41-EAF-ICER-ZDELTA'); _decorate(rsmcfg,dn,'endpoint_delta_nonlinear',a.output_dnl_config,'V64.3.41-EAF-ICER-DNLV'); _decorate(rsmcfg,ep,'endpoint_potential_value',a.output_epv_config,'V64.3.41-EAF-ICER-EPV-RAW'); print(json.dumps({'pass':True,'output_epv_config':a.output_epv_config},sort_keys=True))
if __name__=='__main__': main()
