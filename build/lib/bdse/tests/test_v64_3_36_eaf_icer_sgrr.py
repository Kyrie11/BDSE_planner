from __future__ import annotations

import numpy as np
import pytest

from bdse.planner.tournament import (
    _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES,
    _icer_scene_reservation_value,
)
from bdse.tools.fit_v64_3_36_eaf_icer_sgrr import (
    BASEPOINT_NAMES, GEOMETRY_NAMES, _fit_reservation, _geometry, _reservation_value,
)


def test_selection_geometry_is_permutation_invariant_and_contains_extremal_geometry():
    a=np.array([0.2,-0.1,0.5,0.3]); b=a[[2,0,3,1]]
    ga=_geometry(a); gb=_geometry(b)
    assert np.allclose(ga,gb)
    assert ga[0]==pytest.approx(0.5)
    assert ga[1]==pytest.approx(0.2)
    assert ga[3]==pytest.approx(0.75)


def test_reservation_fit_is_nonnegative_at_runtime_and_fixed_lambda():
    X=np.array([[1.,1.,.5,.5,.2],[2.,.5,1.,.7,.4],[.5,.2,.2,.1,.1],[3.,1.,2.,.8,.6]]*20)
    y=np.array([.5,1.2,0.,2.0]*20)
    m=_fit_reservation(X,y,mode='selection_geometry')
    assert m[2]['ridge_lambda']==1.0
    assert m[2]['sample_count']==80
    assert _reservation_value(X[0],m)>=0.0


def test_runtime_selection_geometry_reservation_common_subtraction_cannot_rerank():
    mu=np.array([0.,0.4,0.2,-0.1])
    cand=np.array([1,2,3])
    feat=np.zeros((4,len(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)))
    sup=np.ones(4)
    cfg={
        'scene_reservation_feature_mode':'selection_geometry',
        'scene_reservation_feature_names':GEOMETRY_NAMES,
        'scene_reservation_feature_mean':[0.]*len(GEOMETRY_NAMES),
        'scene_reservation_feature_std':[1.]*len(GEOMETRY_NAMES),
        'scene_reservation_weights':[.2,0,0,0,0],
        'scene_reservation_bias':0.,'scene_reservation_max':40.,
    }
    r,_,_=_icer_scene_reservation_value(mu,cand,feat,list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES),sup,0,cfg)
    adj=mu.copy(); adj[cand]-=r
    assert r>=0.0
    assert int(cand[np.argmax(mu[cand])])==int(cand[np.argmax(adj[cand])])
    assert np.allclose((mu[cand,None]-mu[cand][None,:]),(adj[cand,None]-adj[cand][None,:]))


def test_runtime_basepoint_reservation_uses_only_incumbent_context_and_is_nonnegative():
    names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    feat=np.zeros((3,len(names))); feat[0,0]=2.0; feat[1,0]=100.; feat[2,0]=-100.
    sup=np.array([1.,5.,6.]); mu=np.array([0.,.5,.4]); cand=np.array([1,2])
    cfg={
        'base_feature_names':names,
        'scene_reservation_feature_mode':'incumbent_basepoint',
        'scene_reservation_feature_names':BASEPOINT_NAMES,
        'scene_reservation_feature_mean':[0.]*len(BASEPOINT_NAMES),
        'scene_reservation_feature_std':[1.]*len(BASEPOINT_NAMES),
        'scene_reservation_weights':[.5]+[0.]*(len(BASEPOINT_NAMES)-1),
        'scene_reservation_bias':0.,'scene_reservation_max':40.,
    }
    r,_,runtime_names=_icer_scene_reservation_value(mu,cand,feat,names,sup,0,cfg)
    assert runtime_names==BASEPOINT_NAMES
    assert r==pytest.approx(1.0)


def test_runtime_reservation_schema_fails_closed():
    names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    with pytest.raises(ValueError):
        _icer_scene_reservation_value(
            np.array([0.,1.]),np.array([1]),np.zeros((2,len(names))),names,np.ones(2),0,
            {'scene_reservation_feature_mode':'selection_geometry','scene_reservation_feature_names':['bad'],'scene_reservation_feature_mean':[0.],'scene_reservation_feature_std':[1.],'scene_reservation_weights':[1.]}
        )


def test_calibrator_preserves_frozen_ordering_and_writes_two_reservation_views(tmp_path):
    import json, subprocess, sys, yaml
    names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    rows=tmp_path/'rows.jsonl'; edges=tmp_path/'edges.jsonl'; rank=tmp_path/'rank.yaml'; bcfg=tmp_path/'b.yaml'; gcfg=tmp_path/'g.yaml'; rep=tmp_path/'rep.json'
    with rows.open('w') as f:
        for i in range(500): f.write(json.dumps({'scenario_token':f't{i:03d}'})+'\n')
    with edges.open('w') as f:
        for i in range(100):
            t=f't{i:03d}'
            inc={'scenario_token':t,'raw_top_action':0,'challenger_action':0,'icer_admissible':1.0,'icer_support_logit':1.0,'teacher_margin':0.0,'icer_scir_predicted_improvement':0.0}
            alt={'scenario_token':t,'raw_top_action':0,'challenger_action':1,'icer_admissible':1.0,'icer_support_logit':1.0,'teacher_margin':0.5 if i%2 else -0.5,'icer_scir_predicted_improvement':1.0,'raw_margin':0.1,'dacer_utility_prior':0.0}
            alt2={'scenario_token':t,'raw_top_action':0,'challenger_action':2,'icer_admissible':1.0,'icer_support_logit':1.0,'teacher_margin':0.1,'icer_scir_predicted_improvement':0.2,'raw_margin':0.05,'dacer_utility_prior':0.0}
            for k,n in enumerate(names): inc[f'icer_feature_{n}']=float(k+1)/10.; alt[f'icer_feature_{n}']=0.; alt2[f'icer_feature_{n}']=0.
            for r in [inc,alt,alt2]: f.write(json.dumps(r)+'\n')
    fn=[f'delta::{x}' for x in names]+['delta::support_logit']
    w=[0.1]*len(fn)
    cfg={'runtime':{'decisive_frontier_value':{'incumbent_contrastive_extremal_recovery':{'selection_conditioned_intervention_recovery':{'enabled':True,'mode':'rank_only','base_feature_names':names,'feature_names':fn,'feature_mean':[0.]*len(fn),'feature_std':[1.]*len(fn),'weights':w,'bias':0.0,'scene_reservation_enabled':False}}}},'metadata':{},'provenance':{},'experiment':{}}
    rank.write_text(yaml.safe_dump(cfg,sort_keys=False))
    subprocess.run([sys.executable,'-m','bdse.tools.calibrate_v64_3_36_eaf_icer_sgrr','--calibration-rows',str(rows),'--calibration-edges',str(edges),'--rsmr-config',str(rank),'--output-basepoint-config',str(bcfg),'--output-geometry-config',str(gcfg),'--output-report',str(rep)],check=True,capture_output=True,text=True)
    b=yaml.safe_load(bcfg.read_text())['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['selection_conditioned_intervention_recovery']
    g=yaml.safe_load(gcfg.read_text())['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['selection_conditioned_intervention_recovery']
    assert b['weights']==w and g['weights']==w
    assert b['scene_reservation_feature_mode']=='incumbent_basepoint'
    assert g['scene_reservation_feature_mode']=='selection_geometry'
    assert b['scene_reservation_operator']==g['scene_reservation_operator']=='nonnegative_common_subtraction_monotone_subset_no_rerank_no_fallback'
    rr=json.loads(rep.read_text()); assert rr['selected_policy_proposal_count']==100
