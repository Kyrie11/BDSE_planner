import numpy as np, pytest
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES,_icer_post_selection_value
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import _phi

def _a(qi,qb):
    qi=np.asarray(qi,dtype=float); qb=np.asarray(qb,dtype=float)
    return {'q_inc':qi,'q_cand':qb,'delta_endpoint':qb-qi,'x':qb-qi,'y':0.0,'action':1,'support':1.0,'margin':1.0,'utility_prior':0}

def test_endpoint_potential_is_antisymmetric_and_zero_at_identity():
    qi=np.linspace(-1,1,19); qb=np.linspace(2,-2,19)
    ab=_a(qi,qb); ba=_a(qb,qi); aa=_a(qi,qi)
    assert np.max(np.abs(_phi(ab,'epv')+_phi(ba,'epv'))) < 1e-12
    assert np.max(np.abs(_phi(aa,'epv'))) < 1e-12

def test_delta_nonlinear_control_is_antisymmetric():
    qi=np.zeros(19); qb=np.linspace(-2,2,19)
    assert np.max(np.abs(_phi(_a(qi,qb),'dnl')+_phi(_a(qb,qi),'dnl'))) < 1e-12

def _cfg(mode,names):
    d=19
    return {'feature_names':[f'delta::{n}' for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]+['delta::support_logit'],'feature_mean':[0.0]*d,'feature_std':[1.0]*d,'weights':[0.0]*d,'bias':0.0,'post_selection_value_enabled':True,'post_selection_value_mode':mode,'post_selection_endpoint_feature_names':names,'post_selection_endpoint_feature_scale':[1.0]*len(names),'post_selection_endpoint_weights':[1.0]*len(names),'post_selection_endpoint_bias':0.0,'post_selection_selected_bias':0.25}

def test_runtime_endpoint_value_replays_absolute_pair_and_shift_only_changes_zero():
    base=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES); raw_names=base.copy(); feat=np.zeros((2,len(base))); feat[1,0]=2.0; sup=np.array([0.5,1.5]); X=np.zeros((2,19)); X[1,0]=2.0; X[1,-1]=1.0; mu=np.zeros(2)
    names=[f'zdelta::{n}' for n in base+['support_logit']]+[f'midpoint_times_delta::{n}' for n in base+['support_logit']]
    c=_cfg('endpoint_potential_value',names)
    v,phi,_=_icer_post_selection_value(1,mu,X,c['feature_names'],c,raw_feat=feat,raw_feature_names=raw_names,support_logits=sup,legacy_action=0)
    assert phi.shape==(38,); assert np.isfinite(v)
    c['post_selection_value_mode']='endpoint_potential_shift'
    v2,_,_=_icer_post_selection_value(1,mu,X,c['feature_names'],c,raw_feat=feat,raw_feature_names=raw_names,support_logits=sup,legacy_action=0)
    assert v2==pytest.approx(v+0.25)

def test_runtime_endpoint_schema_fails_closed_without_absolute_evidence():
    base=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES); names=[f'zdelta::{n}' for n in base+['support_logit']]; c=_cfg('endpoint_zero_delta',names)
    with pytest.raises(ValueError): _icer_post_selection_value(0,np.zeros(1),np.zeros((1,19)),c['feature_names'],c)
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import _fit_zero_ridge

def test_scene_equal_endpoint_ridge_is_invariant_to_uniform_duplicate_candidates_within_scene():
    qi=np.zeros(19)
    a1=_a(qi,np.ones(19)); a1['y']=1.0
    a2=_a(qi,-np.ones(19)); a2['y']=-1.0
    b1=_a(qi,np.linspace(-.5,.5,19)); b1['y']=0.4
    s1={'A':[a1,a2],'B':[b1]}
    s2={'A':[dict(a1),dict(a1),dict(a2),dict(a2)],'B':[b1]}
    m1=_fit_zero_ridge(s1,['A','B'],'epv'); m2=_fit_zero_ridge(s2,['A','B'],'epv')
    assert np.max(np.abs(m1['scale']-m2['scale'])) < 1e-12
    assert np.max(np.abs(m1['weights']-m2['weights'])) < 1e-10
