from __future__ import annotations

import numpy as np
import pytest

from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.tools.fit_v64_3_32_1_eaf_icer_ssir_weightfix import _fit_ridge, _predict
from bdse.tools.fit_v64_3_38_eaf_icer_davr import _affine_scalar, _fit_affine_scalar


def _names():
    return [f"delta::{n}" for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES] + ["delta::support_logit"]


def _base_cfg(d: int):
    w=np.zeros(d); w[0]=1.0
    return {"post_selection_value_enabled":True,"post_selection_value_mode":"dense_edge_value","scene_reservation_enabled":False,"feature_names":_names(),"feature_mean":[0.0]*d,"feature_std":[1.0]*d,"weights":w.tolist(),"bias":0.0,"post_selection_value_max_abs":40.0}


def test_dense_value_runtime_replays_separate_cardinal_head_after_frozen_rsmr():
    names=_names(); d=len(names); X=np.zeros((3,d)); X[1,0]=2.0; X[2,0]=1.0
    rank_w=np.zeros(d); rank_w[0]=1.0; mu=X@rank_w
    cfg=_base_cfg(d)
    # Dense value deliberately disagrees with ranking magnitude but cannot change proposal identity.
    dw=np.zeros(d); dw[0]=-0.5; dw[1]=2.0
    cfg.update({"post_selection_dense_feature_mean":[0.0]*d,"post_selection_dense_feature_std":[1.0]*d,"post_selection_dense_weights":dw.tolist(),"post_selection_dense_bias":0.25})
    v,feat,fn=_icer_post_selection_value(1,mu,X,names,cfg)
    assert v==pytest.approx(-0.75)
    assert feat.tolist()==pytest.approx([-0.75])
    assert fn==["post_value::dense_all_edge_absolute_value"]


def test_dense_affine_runtime_is_one_dimensional_selected_policy_recalibration():
    names=_names(); d=len(names); X=np.zeros((2,d)); X[1,0]=1.0
    mu=np.array([0.0,1.0]); cfg=_base_cfg(d); cfg["post_selection_value_mode"]="dense_edge_affine"
    dw=np.zeros(d); dw[0]=0.5
    cfg.update({"post_selection_dense_feature_mean":[0.0]*d,"post_selection_dense_feature_std":[1.0]*d,"post_selection_dense_weights":dw.tolist(),"post_selection_dense_bias":0.0,"post_selection_dense_cal_mean":0.5,"post_selection_dense_cal_std":0.25,"post_selection_dense_cal_intercept":-0.1,"post_selection_dense_cal_weight":0.2})
    v,feat,fn=_icer_post_selection_value(1,mu,X,names,cfg)
    assert v==pytest.approx(-0.1)
    assert feat[0]==pytest.approx(0.5) and feat[1]==pytest.approx(1.0)
    assert fn[0]=="post_value::dense_all_edge_absolute_value"


def test_affine_scalar_fixed_lambda_and_signed_zero_crossing():
    x=np.linspace(-2,2,100); y=0.3+0.7*x
    m=_fit_affine_scalar(x,y,"unit_test")
    assert m["ridge_lambda"]==1.0 and m["sample_count"]==100
    assert _affine_scalar(-2,m)<0.0 and _affine_scalar(2,m)>0.0


def test_corrected_dense_value_objective_uses_scene_equal_mass_not_edge_count():
    # Two scenes with very different candidate counts. Duplicating candidates inside
    # one scene should not multiply that scene's total squared-loss mass.
    def sample(tok,x,y): return {"token":tok,"x":np.asarray([x]+[0.0]*18,dtype=float),"y":float(y)}
    a=[sample("A",1.0,1.0)]
    b=[sample("B",2.0,-1.0) for _ in range(10)]
    m1=_fit_ridge(a+b)
    b2=[sample("B",2.0,-1.0) for _ in range(30)]
    m2=_fit_ridge(a+b2)
    q=[sample("Q",1.5,0.0)]
    p1=float(_predict(q,m1)[0][0]); p2=float(_predict(q,m2)[0][0])
    assert p1==pytest.approx(p2,abs=1e-10)


def test_dense_modes_fail_closed_on_malformed_schema_and_reservation_conflict():
    names=_names(); d=len(names); X=np.zeros((2,d)); mu=np.zeros(2); cfg=_base_cfg(d)
    cfg.update({"post_selection_dense_feature_mean":[0.0]*(d-1),"post_selection_dense_feature_std":[1.0]*d,"post_selection_dense_weights":[0.0]*d,"post_selection_dense_bias":0.0})
    with pytest.raises(ValueError): _icer_post_selection_value(1,mu,X,names,cfg)
    cfg=_base_cfg(d); cfg["scene_reservation_enabled"]=True
    with pytest.raises(ValueError): _icer_post_selection_value(1,mu,X,names,cfg)

def test_v38_calibrator_preserves_frozen_ranker_and_dense_head(tmp_path):
    import json, subprocess, sys, yaml
    names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES); fn=[f'delta::{x}' for x in names]+['delta::support_logit']; d=len(fn)
    rows=tmp_path/'rows.jsonl'; edges=tmp_path/'edges.jsonl'; dense=tmp_path/'dense.yaml'; ac=tmp_path/'a.yaml'; dc=tmp_path/'d.yaml'; rep=tmp_path/'rep.json'
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
    rw=[0.0]*d; rw[0]=1.0; dw=[0.0]*d; dw[0]=0.4
    sc={'enabled':True,'mode':'rank_only','base_feature_names':names,'feature_names':fn,'feature_mean':[0.0]*d,'feature_std':[1.0]*d,'weights':rw,'bias':0.0,'scene_reservation_enabled':False,'post_selection_value_enabled':True,'post_selection_value_mode':'dense_edge_value','post_selection_dense_feature_mean':[0.0]*d,'post_selection_dense_feature_std':[1.0]*d,'post_selection_dense_weights':dw,'post_selection_dense_bias':0.0}
    cfg={'runtime':{'decisive_frontier_value':{'incumbent_contrastive_extremal_recovery':{'selection_conditioned_intervention_recovery':sc}}},'metadata':{},'provenance':{},'experiment':{}}
    dense.write_text(yaml.safe_dump(cfg,sort_keys=False))
    subprocess.run([sys.executable,'-m','bdse.tools.calibrate_v64_3_38_eaf_icer_davr','--calibration-rows',str(rows),'--calibration-edges',str(edges),'--dense-config',str(dense),'--output-affine-config',str(ac),'--output-davr-config',str(dc),'--output-report',str(rep)],check=True,capture_output=True,text=True)
    aa=yaml.safe_load(ac.read_text())['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['selection_conditioned_intervention_recovery']
    dd=yaml.safe_load(dc.read_text())['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['selection_conditioned_intervention_recovery']
    assert aa['weights']==rw and dd['weights']==rw and dd['post_selection_dense_weights']==dw
    assert aa['post_selection_value_mode']=='score_affine' and dd['post_selection_value_mode']=='dense_edge_affine'
    rr=json.loads(rep.read_text()); assert rr['selected_policy_proposal_count']==100 and rr['frozen_rsmr_score_replay_max_abs']<1e-12
