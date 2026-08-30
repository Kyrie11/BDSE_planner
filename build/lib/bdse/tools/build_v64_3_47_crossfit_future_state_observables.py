from __future__ import annotations

"""Honest cross-fit V47 FSFR nuisance models and runtime-only observables."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.data.cache_schema import load_sample_npz
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.planner.future_state_factorization import (
    EGO_REFERENCE_FEATURE_NAMES,
    FSFR_OBSERVABLE_NAMES,
    runtime_future_state_factorization_observable_costs,
)
from bdse.planner.interaction_response_field import RESPONSE_FIELD_LOCAL_FEATURE_NAMES, RESPONSE_FIELD_PLAN_FEATURE_NAMES
from bdse.tools.build_v64_3_45_crossfit_response_observables import (
    _fit as _fit_mean,
    _read as _read_v45_supervision,
    _scene_weights,
    _weighted_scale,
)
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS, RIDGE_LAMBDA, _fold

EPS = 1.0e-12


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out=[]
    for line in path.read_text().splitlines():
        if line.strip(): out.append(json.loads(line))
    return out


def _fit_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, bias: bool=True) -> tuple[np.ndarray,np.ndarray,float]:
    scale=_weighted_scale(X,w); Z=X/scale[None,:]
    A=np.concatenate([Z,np.ones((len(Z),1),dtype=np.float64)],axis=1) if bias else Z
    sw=np.sqrt(np.maximum(w,0.0)); Aw=A*sw[:,None]; yw=y*sw
    reg=np.eye(A.shape[1],dtype=np.float64)*RIDGE_LAMBDA
    if bias: reg[-1,-1]=0.0
    coef=np.linalg.solve(Aw.T@Aw+reg,Aw.T@yw)
    return (scale,coef[:-1],float(coef[-1])) if bias else (scale,coef,0.0)


def _fit_lateral(rows: list[dict[str,Any]]) -> dict[str,Any]:
    Xl=np.asarray([r['local_features'] for r in rows],dtype=np.float64)
    Xp=np.asarray([r['plan_features_logged_ego'] for r in rows],dtype=np.float64)
    y=np.asarray([r['target_lateral_drift_mps'] for r in rows],dtype=np.float64)
    sw=_scene_weights(rows)
    ls,lw,lb=_fit_ridge(Xl,y,sw,True)
    local=Xl/np.maximum(ls[None,:],1e-6)@lw+lb
    resid=y-local
    exposure=np.maximum(np.asarray([r['logged_ego_interaction_exposure'] for r in rows],dtype=np.float64),0.0)
    toks=[str(r['scenario_token']) for r in rows]; sums={}
    for t,e in zip(toks,exposure): sums[t]=sums.get(t,0.0)+float(e)
    pw_scene=np.asarray([sw[i]*(exposure[i]/max(sums[toks[i]],EPS) if sums[toks[i]]>EPS else 0.0) for i in range(len(rows))],dtype=np.float64)
    if float(pw_scene.sum())<=EPS:
        ps=np.ones((Xp.shape[1],),dtype=np.float64); pw=np.zeros((Xp.shape[1],),dtype=np.float64)
    else:
        ps,pw,_=_fit_ridge(Xp,resid,pw_scene,False)
    return {
        'enabled':True,'model':'agent_local_plan_conditioned_lateral_drift_field','lambda':RIDGE_LAMBDA,
        'local_feature_names':list(RESPONSE_FIELD_LOCAL_FEATURE_NAMES),'local_feature_scale':ls,'local_weights':lw,'local_bias':lb,
        'plan_enabled':True,'plan_feature_names':list(RESPONSE_FIELD_PLAN_FEATURE_NAMES),'plan_feature_scale':ps,'plan_weights':pw,'plan_bias':0.0,
    }


def _predict_lateral_rows(rows: list[dict[str,Any]],m: dict[str,Any]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    Xl=np.asarray([r['local_features'] for r in rows],dtype=np.float64); Xp=np.asarray([r['plan_features_logged_ego'] for r in rows],dtype=np.float64)
    y=np.asarray([r['target_lateral_drift_mps'] for r in rows],dtype=np.float64)
    local=Xl/np.maximum(np.asarray(m['local_feature_scale'])[None,:],1e-6)@np.asarray(m['local_weights'])+float(m['local_bias'])
    plan=local+Xp/np.maximum(np.asarray(m['plan_feature_scale'])[None,:],1e-6)@np.asarray(m['plan_weights'])
    return y,local,plan


def _fit_ego(rows: list[dict[str,Any]]) -> dict[str,Any]:
    X=np.asarray([r['features'] for r in rows],dtype=np.float64); y=np.asarray([r['target_demo_component'] for r in rows],dtype=np.float64); w=_scene_weights(rows)
    s,coef,b=_fit_ridge(X,y,w,True)
    return {'enabled':True,'model':'runtime_predictable_ego_future_reference_cost','lambda':RIDGE_LAMBDA,'feature_names':list(EGO_REFERENCE_FEATURE_NAMES),'feature_scale':s,'weights':coef,'bias':b}


def _predict_ego_rows(rows: list[dict[str,Any]],m: dict[str,Any]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    X=np.asarray([r['features'] for r in rows],dtype=np.float64); y=np.asarray([r['target_demo_component'] for r in rows],dtype=np.float64)
    base=np.asarray([r['cv_demo_proxy'] for r in rows],dtype=np.float64)
    pred=np.maximum(0.0,X/np.maximum(np.asarray(m['feature_scale'])[None,:],1e-6)@np.asarray(m['weights'])+float(m['bias']))
    return y,base,pred


def _scene_equal_mse(rows: list[dict[str,Any]],y: np.ndarray,p: np.ndarray) -> float:
    by={}
    for r,z in zip(rows,(y-p)**2): by.setdefault(str(r['scenario_token']),[]).append(float(z))
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float('nan')


def _serial(x: Any) -> Any:
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,dict): return {k:_serial(v) for k,v in x.items()}
    if isinstance(x,list): return [_serial(v) for v in x]
    return x


def _cfg_with_models(base: dict[str,Any],mean: dict[str,Any],lat: dict[str,Any],ego: dict[str,Any]) -> dict[str,Any]:
    c=copy.deepcopy(base)
    ic=c.setdefault('runtime',{}).setdefault('decisive_frontier_value',{}).setdefault('incumbent_contrastive_extremal_recovery',{})
    ic['instrument_value_observables']=True; ic['instrument_interaction_response_field_observables']=True; ic['instrument_future_state_factorization_observables']=True
    sc=ic.setdefault('selection_conditioned_intervention_recovery',{})
    sc['interaction_response_field']=_serial(mean)
    sc['future_state_factorization']={'agent_lateral_response':_serial(lat),'ego_reference_model':_serial(ego)}
    return c


def _read_v45_plan(path: Path) -> dict[str,np.ndarray]:
    out={}
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line); c=np.asarray(r['costs'],dtype=np.float64)
        if c.ndim!=2 or c.shape[1]!=3: raise ValueError('V47 V45 sidecar schema mismatch')
        out[str(r['scenario_token'])]=c[:,2]
    return out


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--agent-supervision',required=True); ap.add_argument('--ego-supervision',required=True)
    ap.add_argument('--v45-supervision',required=True); ap.add_argument('--v45-sidecar',required=True); ap.add_argument('--v45-response-report',required=True)
    ap.add_argument('--preprocessed-dir',required=True); ap.add_argument('--split',default='train'); ap.add_argument('--scenario-token-file',required=True); ap.add_argument('--base-config',required=True)
    ap.add_argument('--output-sidecar',required=True); ap.add_argument('--output-model',required=True); ap.add_argument('--output-report',required=True)
    a=ap.parse_args()
    agent=_read_jsonl(Path(a.agent_supervision)); ego=_read_jsonl(Path(a.ego_supervision)); v45rows=_read_v45_supervision(Path(a.v45_supervision)); v45side=_read_v45_plan(Path(a.v45_sidecar))
    if len(agent)<4096 or len(ego)<4096: raise SystemExit('STOP V47 DATA: insufficient future-state supervision')
    for r in agent:
        if r.get('local_feature_names')!=RESPONSE_FIELD_LOCAL_FEATURE_NAMES or r.get('plan_feature_names')!=RESPONSE_FIELD_PLAN_FEATURE_NAMES: raise SystemExit('STOP V47 DATA: lateral schema mismatch')
    for r in ego:
        if r.get('feature_names')!=EGO_REFERENCE_FEATURE_NAMES: raise SystemExit('STOP V47 DATA: ego-reference schema mismatch')
    rr=json.loads(Path(a.v45_response_report).read_text()); ag=rr.get('aggregate',{}); exp=(.30137842796229286,.12486025654085724,.12385468573917016); got=(float(ag.get('cv_mse',-1)),float(ag.get('local_mse',-1)),float(ag.get('plan_mse',-1)))
    if any(abs(x-y)>1e-12 for x,y in zip(got,exp)) or not bool(rr.get('plan_response_identified',False)): raise SystemExit(f'STOP V47 prerequisite V45 response changed {got}')
    tokens=[x.strip() for x in Path(a.scenario_token_file).read_text().splitlines() if x.strip()]; want=set(tokens)
    if len(tokens)!=3000 or len(want)!=3000: raise SystemExit('STOP V47 DATA: frozen TRAIN changed')
    base=yaml.safe_load(Path(a.base_config).read_text())
    by_agent={k:[] for k in range(FOLDS)}; by_ego={k:[] for k in range(FOLDS)}
    for r in agent:
        if str(r['scenario_token']) in want: by_agent[_fold(str(r['scenario_token']))].append(r)
    for r in ego:
        if str(r['scenario_token']) in want: by_ego[_fold(str(r['scenario_token']))].append(r)
    mean_models={}; lat_models={}; ego_models={}; fd=[]
    for k in range(FOLDS):
        cf=(k+1)%FOLDS
        fitv=[r for r in v45rows if str(r['scenario_token']) in want and _fold(str(r['scenario_token'])) not in {k,cf}]
        fita=[r for r in agent if str(r['scenario_token']) in want and _fold(str(r['scenario_token'])) not in {k,cf}]
        fite=[r for r in ego if str(r['scenario_token']) in want and _fold(str(r['scenario_token'])) not in {k,cf}]
        mm=_fit_mean(fitv); lm=_fit_lateral(fita); em=_fit_ego(fite)
        mean_models[k]=mm; lat_models[k]=lm; ego_models[k]=em
        ya,z,lp=_predict_lateral_rows(by_agent[k],lm); ye,bp,ep=_predict_ego_rows(by_ego[k],em)
        z0=np.zeros_like(ya)
        z_m=_scene_equal_mse(by_agent[k],ya,z0); l_m=_scene_equal_mse(by_agent[k],ya,z); p_m=_scene_equal_mse(by_agent[k],ya,lp)
        b_m=_scene_equal_mse(by_ego[k],ye,bp); e_m=_scene_equal_mse(by_ego[k],ye,ep)
        fd.append({'fold':k,'fit_agent_rows':len(fita),'test_agent_rows':len(by_agent[k]),'fit_ego_rows':len(fite),'test_ego_rows':len(by_ego[k]),
                   'lateral_zero_mse':z_m,'lateral_local_mse':l_m,'lateral_plan_mse':p_m,'lateral_local_better_than_zero':bool(l_m<z_m),'lateral_plan_better_than_local':bool(p_m<l_m),
                   'ego_cv_proxy_mse':b_m,'ego_reference_mse':e_m,'ego_reference_better_than_cv_proxy':bool(e_m<b_m)})
    full_mean=_fit_mean([r for r in v45rows if str(r['scenario_token']) in want]); full_lat=_fit_lateral([r for r in agent if str(r['scenario_token']) in want]); full_ego=_fit_ego([r for r in ego if str(r['scenario_token']) in want])
    ds=PreprocessedBDSEDataset(a.preprocessed_dir,split=a.split,scenario_tokens=want); bytok={}
    for p in ds.build_index():
        try:
            with np.load(p,allow_pickle=True) as z: tok=str(z['scenario_token'].item() if z['scenario_token'].shape==() else z['scenario_token'].reshape(-1)[0])
        except Exception: continue
        if tok in want: bytok[tok]=Path(p)
    miss=want-set(bytok)
    if miss: raise SystemExit(f'STOP V47 DATA: current cache missing {len(miss)}')
    out=Path(a.output_sidecar); out.parent.mkdir(parents=True,exist_ok=True); replay=0.0
    with out.open('w') as f:
        for tok in tokens:
            s=load_sample_npz(bytok[tok],include_label_future=False,include_candidate_metadata=False,include_evidence_aux_metadata=False)
            cfg=_cfg_with_models(base,mean_models[_fold(tok)],lat_models[_fold(tok)],ego_models[_fold(tok)])
            cost,names=runtime_future_state_factorization_observable_costs(s.runtime,s.candidates,cfg)
            if names!=FSFR_OBSERVABLE_NAMES or cost.shape!=(s.candidates.K,len(FSFR_OBSERVABLE_NAMES)): raise SystemExit(f'STOP V47 DATA: runtime cost malformed {tok}')
            if tok not in v45side or v45side[tok].shape[0]!=s.candidates.K: raise SystemExit(f'STOP V47 prerequisite V45 sidecar mismatch {tok}')
            diff=float(np.max(np.abs(cost[:,0]-v45side[tok]))) if s.candidates.K else 0.0; replay=max(replay,diff)
            if diff>1e-10: raise SystemExit(f'STOP V47 ENGINEERING: V45 PLAN occupancy replay drift {tok}: {diff}')
            f.write(json.dumps({'scenario_token':tok,'outer_fold':_fold(tok),'observable_names':names,'costs':cost.tolist()},sort_keys=True)+'\n')
    agg={k:float(np.mean([d[k] for d in fd])) for k in ['lateral_zero_mse','lateral_local_mse','lateral_plan_mse','ego_cv_proxy_mse','ego_reference_mse']}
    lc=sum(d['lateral_local_better_than_zero'] for d in fd); pc=sum(d['lateral_plan_better_than_local'] for d in fd); ec=sum(d['ego_reference_better_than_cv_proxy'] for d in fd)
    report={'audit':'v64_3_47_crossfit_future_state_factorization','folds':fd,'aggregate':agg,
            'lateral_local_better_fold_count':lc,'lateral_plan_better_fold_count':pc,'ego_reference_better_fold_count':ec,
            'agent_2d_response_identified':bool(agg['lateral_local_mse']<agg['lateral_zero_mse'] and agg['lateral_plan_mse']<agg['lateral_local_mse'] and lc>=4 and pc>=4),
            'ego_reference_identified':bool(agg['ego_reference_mse']<agg['ego_cv_proxy_mse'] and ec>=4),
            'v45_plan_occupancy_exact_replay_max_abs':replay,'deployment_uses_logged_future':False,'uses_teacher_total_or_improvement':False,
            'full_train_mean_response_model':_serial(full_mean),'full_train_agent_lateral_model':_serial(full_lat),'full_train_ego_reference_model':_serial(full_ego)}
    Path(a.output_model).write_text(json.dumps({'mean_response_model':_serial(full_mean),'future_state_factorization':{'agent_lateral_response':_serial(full_lat),'ego_reference_model':_serial(full_ego)}},indent=2,sort_keys=True))
    Path(a.output_report).write_text(json.dumps(report,indent=2,sort_keys=True)); print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__': main()
