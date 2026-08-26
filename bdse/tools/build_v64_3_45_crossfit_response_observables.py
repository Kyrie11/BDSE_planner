from __future__ import annotations

"""Fit honest V45 response fields and materialize runtime-only cross-fit costs.

For scene fold k, the nuisance response model excludes k and (k+1)%5, matching
the nested value protocol.  Agent-response targets are TRAIN-only logged-future
quantities.  All sidecar costs are recomputed from current runtime + candidate
trajectories without label future.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.data.cache_schema import load_sample_npz
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.planner.interaction_response_field import (
    RESPONSE_FIELD_LOCAL_FEATURE_NAMES, RESPONSE_FIELD_PLAN_FEATURE_NAMES,
    RESPONSE_FIELD_OBSERVABLE_NAMES, runtime_interaction_response_field_observable_costs,
)
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS, RIDGE_LAMBDA, _fold

EPS=1e-12


def _read(path: Path):
    rows=[]
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        if r.get('local_feature_names') != RESPONSE_FIELD_LOCAL_FEATURE_NAMES or r.get('plan_feature_names') != RESPONSE_FIELD_PLAN_FEATURE_NAMES:
            raise ValueError('V45 response supervision schema mismatch')
        rows.append(r)
    if len(rows)<4096: raise ValueError(f'V45 response supervision only {len(rows)} rows')
    return rows


def _scene_weights(rows):
    counts={}
    for r in rows: counts[r['scenario_token']]=counts.get(r['scenario_token'],0)+1
    return np.asarray([1.0/max(counts[r['scenario_token']],1) for r in rows],dtype=np.float64)


def _weighted_scale(X,w):
    den=max(float(w.sum()),EPS)
    return np.maximum(np.sqrt(np.sum(w[:,None]*X*X,axis=0)/den),1e-6)


def _fit_local(rows):
    X=np.asarray([r['local_features'] for r in rows],dtype=np.float64); y=np.asarray([r['target_longitudinal_accel_mps2'] for r in rows],dtype=np.float64); w=_scene_weights(rows)
    scale=_weighted_scale(X,w); Z=X/scale[None,:]; A=np.concatenate([Z,np.ones((len(Z),1))],axis=1)
    sw=np.sqrt(w); Aw=A*sw[:,None]; yw=y*sw
    reg=np.eye(A.shape[1])*RIDGE_LAMBDA; reg[-1,-1]=0.0
    coef=np.linalg.solve(Aw.T@Aw+reg,Aw.T@yw)
    return scale,coef[:-1],float(coef[-1])


def _local_pred(rows,model):
    X=np.asarray([r['local_features'] for r in rows],dtype=np.float64)
    return np.clip(X/np.maximum(model['local_feature_scale'][None,:],1e-6)@model['local_weights']+model['local_bias'],-2.0,0.5)


def _fit_plan(rows, local_model):
    X=np.asarray([r['plan_features_logged_ego'] for r in rows],dtype=np.float64); y=np.asarray([r['target_longitudinal_accel_mps2'] for r in rows],dtype=np.float64)
    base=_local_pred(rows,local_model); resid=y-base; scene_w=_scene_weights(rows); exposure=np.maximum(np.asarray([r['logged_ego_interaction_exposure'] for r in rows],dtype=np.float64),0.0)
    # Within each scene, plan supervision mass is proportional to continuous interaction exposure.
    toks=[r['scenario_token'] for r in rows]; sums={}
    for t,e in zip(toks,exposure): sums[t]=sums.get(t,0.0)+float(e)
    w=np.asarray([scene_w[i]*(exposure[i]/max(sums[toks[i]],EPS) if sums[toks[i]]>EPS else 0.0) for i in range(len(rows))],dtype=np.float64)
    if float(w.sum())<=EPS:
        return np.ones(X.shape[1]),np.zeros(X.shape[1]),0
    scale=_weighted_scale(X,w); Z=X/scale[None,:]; sw=np.sqrt(w); Zw=Z*sw[:,None]; rw=resid*sw
    coef=np.linalg.solve(Zw.T@Zw+np.eye(Z.shape[1])*RIDGE_LAMBDA,Zw.T@rw)
    return scale,coef,int(np.sum(exposure>EPS))


def _fit(rows):
    ls,lw,lb=_fit_local(rows); lm={'local_feature_scale':ls,'local_weights':lw,'local_bias':lb}
    ps,pw,nexp=_fit_plan(rows,lm)
    return {'enabled':True,'model':'agent_local_continuous_longitudinal_response_field','lambda':RIDGE_LAMBDA,
            'local_feature_names':list(RESPONSE_FIELD_LOCAL_FEATURE_NAMES),'local_feature_scale':ls,'local_weights':lw,'local_bias':lb,
            'plan_enabled':True,'plan_feature_names':list(RESPONSE_FIELD_PLAN_FEATURE_NAMES),'plan_feature_scale':ps,'plan_weights':pw,'plan_bias':0.0,
            'plan_nonzero_exposure_rows':nexp,'response_accel_clip_mps2':[-2.0,0.5]}


def _serial(m):
    o={}
    for k,v in m.items(): o[k]=v.tolist() if isinstance(v,np.ndarray) else v
    return o


def _predict(rows,m):
    lp=_local_pred(rows,m)
    X=np.asarray([r['plan_features_logged_ego'] for r in rows],dtype=np.float64)
    corr=X/np.maximum(m['plan_feature_scale'][None,:],1e-6)@m['plan_weights']
    pp=np.clip(lp+corr,-2.0,0.5)
    y=np.asarray([r['target_longitudinal_accel_mps2'] for r in rows],dtype=np.float64)
    return y,lp,pp


def _scene_equal_mse(rows,y,p):
    d=(y-p)**2; by={}
    for r,z in zip(rows,d): by.setdefault(r['scenario_token'],[]).append(float(z))
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float('nan')


def _cfg_with_model(cfg,m):
    c=copy.deepcopy(cfg)
    ic=c.setdefault('runtime',{}).setdefault('decisive_frontier_value',{}).setdefault('incumbent_contrastive_extremal_recovery',{})
    ic['instrument_value_observables']=True; ic['instrument_interaction_response_field_observables']=True
    sc=ic.setdefault('selection_conditioned_intervention_recovery',{}); sc['interaction_response_field']=_serial(m)
    return c


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--supervision',required=True); ap.add_argument('--preprocessed-dir',required=True); ap.add_argument('--split',default='train'); ap.add_argument('--scenario-token-file',required=True); ap.add_argument('--base-config',required=True); ap.add_argument('--output-sidecar',required=True); ap.add_argument('--output-model',required=True); ap.add_argument('--output-report',required=True); args=ap.parse_args()
    rows=_read(Path(args.supervision)); base=yaml.safe_load(Path(args.base_config).read_text())
    tokens=[x.strip() for x in Path(args.scenario_token_file).read_text().splitlines() if x.strip()]; want=set(tokens)
    if len(tokens)!=len(want): raise SystemExit('STOP V45 DATA: duplicate frozen tokens')
    byfold={k:[] for k in range(FOLDS)}
    for r in rows:
        t=str(r['scenario_token'])
        if t in want: byfold[_fold(t)].append(r)
    models={}; fold_diag=[]; oof_all=[]
    for k in range(FOLDS):
        cf=(k+1)%FOLDS; fit=[r for r in rows if str(r['scenario_token']) in want and _fold(str(r['scenario_token'])) not in {k,cf}]; test=byfold[k]
        m=_fit(fit); models[k]=m; y,lp,pp=_predict(test,m); cv=np.zeros_like(y)
        d={'fold':k,'fit_agent_rows':len(fit),'test_agent_rows':len(test),'test_scenes':len(set(str(r['scenario_token']) for r in test)),
           'cv_mse':_scene_equal_mse(test,y,cv),'local_mse':_scene_equal_mse(test,y,lp),'plan_mse':_scene_equal_mse(test,y,pp),
           'local_better_than_cv':bool(_scene_equal_mse(test,y,lp)<_scene_equal_mse(test,y,cv)),
           'plan_better_than_local':bool(_scene_equal_mse(test,y,pp)<_scene_equal_mse(test,y,lp))}
        fold_diag.append(d); oof_all.append((test,y,lp,pp))
    full=_fit([r for r in rows if str(r['scenario_token']) in want])

    ds=PreprocessedBDSEDataset(args.preprocessed_dir,split=args.split,scenario_tokens=want); paths=ds.build_index(); bytok={}
    for p in paths:
        try:
            with np.load(p,allow_pickle=True) as z: tok=str(z['scenario_token'].item() if z['scenario_token'].shape==() else z['scenario_token'].reshape(-1)[0])
        except Exception: continue
        if tok in want: bytok[tok]=Path(p)
    miss=want-set(bytok)
    if miss: raise SystemExit(f'STOP V45 DATA: current-state cache missing {len(miss)} frozen tokens')
    out=Path(args.output_sidecar); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        for tok in tokens:
            s=load_sample_npz(bytok[tok],include_label_future=False,include_candidate_metadata=False,include_evidence_aux_metadata=False)
            cfg=_cfg_with_model(base,models[_fold(tok)]); cost,names=runtime_interaction_response_field_observable_costs(s.runtime,s.candidates,cfg)
            if names!=RESPONSE_FIELD_OBSERVABLE_NAMES or cost.shape[0]!=s.candidates.K: raise SystemExit(f'STOP V45 DATA: response costs malformed for {tok}')
            f.write(json.dumps({'scenario_token':tok,'outer_fold':_fold(tok),'observable_names':names,'costs':cost.tolist()},sort_keys=True)+'\n')
    Path(args.output_model).write_text(json.dumps(_serial(full),indent=2,sort_keys=True))
    agg={key:float(np.mean([d[key] for d in fold_diag])) for key in ('cv_mse','local_mse','plan_mse')}
    report={'audit':'v64_3_45_crossfit_response_field','folds':fold_diag,'aggregate':agg,
            'local_better_than_cv_fold_count':sum(d['local_better_than_cv'] for d in fold_diag),
            'plan_better_than_local_fold_count':sum(d['plan_better_than_local'] for d in fold_diag),
            'plan_response_identified':bool(agg['plan_mse']<agg['local_mse'] and sum(d['plan_better_than_local'] for d in fold_diag)>=4),
            'current_state_sidecar_scenes':len(tokens),'deployment_uses_logged_future':False,'full_train_model':_serial(full)}
    Path(args.output_report).write_text(json.dumps(report,indent=2,sort_keys=True)); print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__': main()
