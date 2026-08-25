from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.tools.fit_v64_3_37_eaf_icer_pvr import (
    _affine_value,
    _fit_affine_value,
    _fit_orthogonal_residual,
    _orthogonal_value,
)


def test_affine_value_is_absolute_signed_readout_not_nonnegative_reservation():
    u=np.linspace(0.05,1.0,80); y=0.4-0.9*u
    m=_fit_affine_value(u,y)
    assert m['ridge_lambda']==1.0 and m['sample_count']==80
    assert _affine_value(float(u[0]),m)>0.0
    assert _affine_value(float(u[-1]),m)<0.0


def test_orthogonal_residual_fit_uses_fixed_lambda_and_can_add_signed_correction():
    rng=np.random.default_rng(0); X=rng.normal(size=(80,19)); e=X[:,0]-0.3*X[:,1]
    m=_fit_orthogonal_residual(X,e)
    assert m['ridge_lambda']==1.0 and m['sample_count']==80
    a={'score_mean':0.0,'score_std':1.0,'intercept':0.0,'score_weight':0.0}
    vals=np.array([_orthogonal_value(0.0,x,a,m) for x in X])
    assert np.corrcoef(vals,e)[0,1]>.95
    assert np.any(vals>0) and np.any(vals<0)


def test_runtime_opvr_feature_is_orthogonal_to_frozen_rsmr_score_direction():
    names=[f'delta::{n}' for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]+['delta::support_logit']
    d=len(names); X=np.zeros((3,d)); X[1,0]=1.0; X[1,1]=2.0; X[2,0]=0.5
    w=np.zeros(d); w[0]=1.0; w[1]=0.5
    mu=np.clip(X@w,-40,40)
    cfg={'post_selection_value_enabled':True,'post_selection_value_mode':'orthogonal_proposal_value','feature_names':names,'feature_mean':[0.]*d,'feature_std':[1.]*d,'weights':w.tolist(),'bias':0.0,'scene_reservation_enabled':False,'post_selection_score_mean':0.0,'post_selection_score_std':1.0,'post_selection_affine_intercept':0.0,'post_selection_affine_score_weight':1.0,'post_selection_residual_feature_mean':[0.]*d,'post_selection_residual_feature_std':[1.]*d,'post_selection_residual_weights':[0.]*d,'post_selection_residual_bias':0.0,'post_selection_value_max_abs':40.0}
    v,f,fn=_icer_post_selection_value(1,mu,X,names,cfg)
    zp=f[1:]
    assert v==pytest.approx(mu[1])
    assert abs(float(zp@w))<1e-10
    assert len(fn)==d+1


def test_runtime_post_selection_value_schema_fails_closed_and_is_mutually_exclusive_with_reservation():
    names=[f'delta::{n}' for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]+['delta::support_logit']; d=len(names)
    X=np.zeros((2,d)); mu=np.zeros(2)
    with pytest.raises(ValueError):
        _icer_post_selection_value(1,mu,X,names,{'post_selection_value_enabled':True,'post_selection_value_mode':'score_affine','scene_reservation_enabled':True})


def test_calibrator_preserves_rsmr_and_writes_affine_and_orthogonal_views(tmp_path):
    names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES); fn=[f'delta::{x}' for x in names]+['delta::support_logit']; d=len(fn)
    rows=tmp_path/'rows.jsonl'; edges=tmp_path/'edges.jsonl'; rank=tmp_path/'rank.yaml'; ac=tmp_path/'a.yaml'; oc=tmp_path/'o.yaml'; rep=tmp_path/'rep.json'
    with rows.open('w') as f:
        for i in range(500): f.write(json.dumps({'scenario_token':f't{i:03d}'})+'\n')
    with edges.open('w') as f:
        for i in range(100):
            t=f't{i:03d}'
            inc={'scenario_token':t,'raw_top_action':0,'challenger_action':0,'icer_admissible':1.0,'icer_support_logit':1.0,'teacher_margin':0.0,'icer_scir_predicted_improvement':0.0,'icer_scir_raw_predicted_improvement':0.0}
            alt={'scenario_token':t,'raw_top_action':0,'challenger_action':1,'icer_admissible':1.0,'icer_support_logit':1.0,'teacher_margin':0.8 if i%2 else -0.4,'icer_scir_predicted_improvement':1.0,'icer_scir_raw_predicted_improvement':1.0,'raw_margin':0.1,'dacer_utility_prior':0.0}
            alt2={'scenario_token':t,'raw_top_action':0,'challenger_action':2,'icer_admissible':1.0,'icer_support_logit':1.0,'teacher_margin':0.1,'icer_scir_predicted_improvement':0.2,'icer_scir_raw_predicted_improvement':0.2,'raw_margin':0.05,'dacer_utility_prior':0.0}
            for n in names:
                inc[f'icer_feature_{n}']=0.0; alt[f'icer_feature_{n}']=0.0; alt2[f'icer_feature_{n}']=0.0
            alt[f'icer_feature_{names[0]}']=1.0; alt2[f'icer_feature_{names[0]}']=0.2
            for r in [inc,alt,alt2]: f.write(json.dumps(r)+'\n')
    w=[0.0]*d; w[0]=1.0
    cfg={'runtime':{'decisive_frontier_value':{'incumbent_contrastive_extremal_recovery':{'selection_conditioned_intervention_recovery':{'enabled':True,'mode':'rank_only','base_feature_names':names,'feature_names':fn,'feature_mean':[0.]*d,'feature_std':[1.]*d,'weights':w,'bias':0.0,'scene_reservation_enabled':False,'post_selection_value_enabled':False}}}},'metadata':{},'provenance':{},'experiment':{}}
    rank.write_text(yaml.safe_dump(cfg,sort_keys=False))
    subprocess.run([sys.executable,'-m','bdse.tools.calibrate_v64_3_37_eaf_icer_pvr','--calibration-rows',str(rows),'--calibration-edges',str(edges),'--rsmr-config',str(rank),'--output-affine-config',str(ac),'--output-orthogonal-config',str(oc),'--output-report',str(rep)],check=True,capture_output=True,text=True)
    aa=yaml.safe_load(ac.read_text())['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['selection_conditioned_intervention_recovery']
    oo=yaml.safe_load(oc.read_text())['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['selection_conditioned_intervention_recovery']
    assert aa['weights']==w and oo['weights']==w
    assert aa['post_selection_value_mode']=='score_affine'
    assert oo['post_selection_value_mode']=='orthogonal_proposal_value'
    assert aa['post_selection_operator']==oo['post_selection_operator']
    rr=json.loads(rep.read_text()); assert rr['selected_policy_proposal_count']==100 and rr['frozen_rsmr_score_replay_max_abs']<1e-12
