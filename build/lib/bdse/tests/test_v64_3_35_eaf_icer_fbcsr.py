from __future__ import annotations
import numpy as np
import pytest

from bdse.tools.fit_v64_3_35_eaf_icer_fbcsr import (
    CONTEXT_NAMES, FEATURE_NAMES, _factorized_objective_numpy, _factorized_scores,
    _fit_factorized,
)


def _alt(tok,act,x0,y,support=1.0):
    x=np.zeros((len(FEATURE_NAMES),),dtype=np.float64); x[0]=x0
    return {'token':tok,'action':act,'x':x,'y':y,'support':support,'margin':0.0,'utility_prior':0}


def test_factorized_loss_learns_noop_and_positive_opportunity_without_pair_count_dilution():
    scene={'noop':[_alt('noop',1,1.0,-1.0),_alt('noop',2,2.0,-0.2)],'opp':[_alt('opp',1,-1.0,0.2),_alt('opp',2,-2.0,2.0),_alt('opp',3,1.0,-1.0)]}
    ctx={k:np.ones((len(CONTEXT_NAMES),),dtype=np.float64) for k in scene}
    m=_fit_factorized(scene,ctx,list(scene),use_context=False)
    ns=_factorized_scores(scene['noop'],ctx['noop'],m); os=_factorized_scores(scene['opp'],ctx['opp'],m)
    assert float(ns.max())<0.0
    assert int(np.argmax(os))==1 and float(os[1])>0.0


def test_factorized_loss_duplicate_easy_rival_does_not_change_objective_at_fixed_scale():
    base={'s':[_alt('s',1,1.0,2.0),_alt('s',2,-1.0,-3.0)]}
    dup={'s':[_alt('s',1,1.0,2.0),_alt('s',2,-1.0,-3.0),_alt('s',3,-1.0,-3.0)]}
    ctx={'s':np.ones((len(CONTEXT_NAMES),),dtype=np.float64)}; ds=np.ones((len(FEATURE_NAMES),)); theta=np.zeros((len(FEATURE_NAMES),)); theta[0]=.5
    a=_factorized_objective_numpy(theta,base,ctx,['s'],ds,None); b=_factorized_objective_numpy(theta,dup,ctx,['s'],ds,None)
    assert a==pytest.approx(b)


def test_basepoint_context_common_shift_cannot_change_challenger_ordering():
    ss=[_alt('s',1,1.0,1.0),_alt('s',2,2.0,0.5),_alt('s',3,-1.0,-1.0)]
    theta=np.zeros((len(FEATURE_NAMES)+len(CONTEXT_NAMES),)); theta[0]=1.0; theta[len(FEATURE_NAMES)]=2.0
    model=(theta,np.ones((len(FEATURE_NAMES),)),np.ones((len(CONTEXT_NAMES),)),{})
    c1=np.zeros((len(CONTEXT_NAMES),)); c2=np.zeros((len(CONTEXT_NAMES),)); c2[0]=3.0
    s1=_factorized_scores(ss,c1,model); s2=_factorized_scores(ss,c2,model)
    assert np.allclose((s1[:,None]-s1[None,:]),(s2[:,None]-s2[None,:]))
    assert np.allclose(s2-s1,6.0)


def test_runtime_incumbent_context_shift_is_candidate_independent_and_legacy_remains_zero():
    from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_selection_conditioned_intervention_scores
    n=4; names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES); feat=np.zeros((n,len(names)),dtype=float); feat[0,0]=2.; feat[1,0]=3.; feat[2,0]=4.; feat[3,0]=1.
    sup=np.array([1.,2.,3.,.5]); legacy=0
    fn=[f'delta::{x}' for x in names]+['delta::support_logit']; cn=[f'incumbent::{x}' for x in names]+['incumbent::support_logit']
    cfg={'base_feature_names':names,'feature_names':fn,'feature_mean':[0.]*len(fn),'feature_std':[1.]*len(fn),'weights':[0.]*len(fn),'bias':0.,
         'incumbent_context_feature_names':cn,'incumbent_context_feature_mean':[0.]*len(cn),'incumbent_context_feature_std':[1.]*len(cn),'incumbent_context_weights':[1.]+[0.]*(len(cn)-1),'incumbent_context_bias':0.}
    mu,_,_,_=_icer_selection_conditioned_intervention_scores(feat,names,sup,legacy,cfg)
    assert mu[legacy]==0.0
    assert mu[1]==pytest.approx(2.0) and mu[2]==pytest.approx(2.0) and mu[3]==pytest.approx(2.0)


def test_runtime_context_schema_fails_closed():
    from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_selection_conditioned_intervention_scores
    names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES); feat=np.zeros((2,len(names))); sup=np.zeros((2,)); fn=[f'delta::{x}' for x in names]+['delta::support_logit']
    cfg={'base_feature_names':names,'feature_names':fn,'feature_mean':[0.]*len(fn),'feature_std':[1.]*len(fn),'weights':[0.]*len(fn),'incumbent_context_feature_names':['bad']}
    with pytest.raises(ValueError): _icer_selection_conditioned_intervention_scores(feat,names,sup,0,cfg)


def test_factorized_context_solver_is_deterministic_and_context_only_changes_boundary():
    scene={'a':[_alt('a',1,1.,1.),_alt('a',2,-1.,-.5)],'b':[_alt('b',1,1.,-.2),_alt('b',2,2.,-.4)],'c':[_alt('c',1,-1.,2.),_alt('c',2,1.,.1)]}
    ctx={'a':np.ones((len(CONTEXT_NAMES),)), 'b':-np.ones((len(CONTEXT_NAMES),)), 'c':np.ones((len(CONTEXT_NAMES),))*0.5}
    m1=_fit_factorized(scene,ctx,list(scene),use_context=True); m2=_fit_factorized(scene,ctx,list(scene),use_context=True)
    assert np.allclose(m1[0],m2[0],atol=1e-9,rtol=1e-9)
    assert m1[3]['objective_final']<=m1[3]['objective_at_zero']+1e-9

def test_fbcsr_calibrator_preserves_context_selector_exactly(tmp_path):
    import json, subprocess, sys, yaml
    from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES
    rows=tmp_path/'rows.jsonl'; edges=tmp_path/'edges.jsonl'; rank=tmp_path/'rank.yaml'; main=tmp_path/'main.yaml'; rep=tmp_path/'rep.json'
    with rows.open('w') as f:
        for i in range(500): f.write(json.dumps({'scenario_token':f't{i:03d}'})+'\n')
    with edges.open('w') as f:
        for i in range(100):
            t=f't{i:03d}'
            f.write(json.dumps({'scenario_token':t,'raw_top_action':0,'challenger_action':0,'icer_admissible':1.0,'icer_support_logit':1.0,'teacher_margin':0.0,'icer_scir_predicted_improvement':0.0})+'\n')
            f.write(json.dumps({'scenario_token':t,'raw_top_action':0,'challenger_action':1,'icer_admissible':1.0,'icer_support_logit':1.0,'teacher_margin':1.0,'icer_scir_predicted_improvement':1.4,'raw_margin':0.1,'dacer_utility_prior':0})+'\n')
    base=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES); fn=[f'delta::{x}' for x in base]+['delta::support_logit']; cn=[f'incumbent::{x}' for x in base]+['incumbent::support_logit']
    sc={'enabled':True,'mode':'rank_only','no_fallback':True,'base_feature_names':base,'feature_names':fn,'feature_mean':[0.]*len(fn),'feature_std':[1.]*len(fn),'weights':[0.]*len(fn),'bias':0.0,
        'incumbent_context_feature_names':cn,'incumbent_context_feature_mean':[0.]*len(cn),'incumbent_context_feature_std':[1.]*len(cn),'incumbent_context_weights':[0.1]*len(cn),'incumbent_context_bias':0.0,'context_operator':'candidate_independent_common_shift_cannot_change_challenger_ordering'}
    cfg={'runtime':{'decisive_frontier_value':{'incumbent_contrastive_extremal_recovery':{'selection_conditioned_intervention_recovery':sc}}},'metadata':{},'provenance':{},'experiment':{}}
    rank.write_text(yaml.safe_dump(cfg,sort_keys=False))
    subprocess.run([sys.executable,'-m','bdse.tools.calibrate_v64_3_35_eaf_icer_fbcsr','--calibration-rows',str(rows),'--calibration-edges',str(edges),'--rank-config',str(rank),'--output-main-config',str(main),'--output-report',str(rep),'--alpha','0.05'],check=True,capture_output=True,text=True)
    out=yaml.safe_load(main.read_text())['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['selection_conditioned_intervention_recovery']
    assert out['mode']=='conformal_veto'
    assert out['incumbent_context_weights']==sc['incumbent_context_weights']
    assert out['context_operator']==sc['context_operator']
