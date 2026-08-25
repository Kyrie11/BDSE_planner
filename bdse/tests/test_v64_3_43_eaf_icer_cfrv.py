import numpy as np
import pytest

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.response_value_observables import FUTURE_RESPONSE_OBSERVABLE_NAMES, runtime_future_response_observable_costs
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.planner.value_observables import QUALITY_NAMES, VALUE_OBSERVABLE_NAMES, runtime_value_observable_costs
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES


def _bank():
    T=41; t=np.linspace(0.1,4.1,T,dtype=np.float32)
    tr=np.zeros((2,T,5),dtype=np.float32)
    # candidate 0 remains on the interaction centerline; 1 moves laterally away.
    tr[0,:,0]=4.0*t; tr[0,:,3]=4.0
    tr[1,:,0]=4.0*t; tr[1,:,1]=3.5; tr[1,:,3]=4.0
    tr[:,:,4]=t
    return CandidateBank(trajectories=tr,valid_mask=np.ones(2,dtype=bool),maneuver_ids=np.zeros(2,dtype=np.int64),theta=[{},{}],dynamic_flags=[{},{}],metadata=[{},{}])


def _runtime():
    cur=np.array([[12.0,0.0,0.0,2.0,0.0,2.0,0.0,4.8,2.0,1.0]],dtype=np.float32)
    return RuntimeFeatures(ego_history=np.zeros((1,5),dtype=np.float32),agent_history=cur[:,None,:].copy(),agent_valid=np.ones(1,dtype=bool),current_agents=cur,traffic_lights=[],map_features={'route_centerline':np.array([[0.,0.],[80.,0.]],dtype=np.float32),'route_corridor_width':4.0,'stop_lines':[],'speed_limit_mps':20.0},route_roadblock_ids=[],mission_goal=None)


def _cfg(instrument=True):
    return {'candidate':{'step_s':0.1,'horizon_s':8.0},'teacher':{'robust_modes':{'logged':{'enabled':True,'prob':.35},'cv':{'enabled':True,'prob':.2},'ca':{'enabled':True,'prob':.1},'brake':{'enabled':True,'prob':.15},'yield':{'enabled':True,'prob':.1},'nonyield':{'enabled':True,'prob':.1}},'risk_aggregation':{'cvar_alpha':.9,'cvar_weight':.4}},'runtime_safety':{'flag_mode':'hard','hard_check_horizon_s':4.0,'soft_check_horizon_s':4.0},'runtime':{'decisive_frontier_value':{'incumbent_contrastive_extremal_recovery':{'instrument_future_response_observables':instrument}}}}


def test_future_response_observables_are_finite_runtime_only_and_tail_ge_mean():
    x,n=runtime_future_response_observable_costs(_runtime(),_bank(),_cfg())
    assert n==FUTURE_RESPONSE_OBSERVABLE_NAMES
    assert x.shape==(2,3) and np.all(np.isfinite(x))
    assert np.all(x[:,1] >= x[:,0]-1e-8)
    # laterally separated candidate must not be riskier under every response mode.
    assert x[1,2] <= x[0,2] + 1e-8


def test_v43_instrumentation_appends_without_changing_v42_prefix():
    c0=_cfg(False); c1=_cfg(True)
    x0,n0=runtime_value_observable_costs(_runtime(),_bank(),c0)
    x1,n1=runtime_value_observable_costs(_runtime(),_bank(),c1)
    assert n0==VALUE_OBSERVABLE_NAMES
    assert n1==VALUE_OBSERVABLE_NAMES+FUTURE_RESPONSE_OBSERVABLE_NAMES
    assert np.max(np.abs(x0-x1[:,:len(n0)])) < 1e-12


def _runtime_cfg(mode, response_name):
    base=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES); fn=[f'delta::{n}' for n in base]+['delta::support_logit']
    names=VALUE_OBSERVABLE_NAMES+FUTURE_RESPONSE_OBSERVABLE_NAMES
    return {'feature_names':fn,'feature_mean':[0.0]*19,'feature_std':[1.0]*19,'weights':[0.0]*19,'bias':0.0,'base_feature_names':base,
        'post_selection_value_enabled':True,'post_selection_value_mode':mode,'post_selection_endpoint_feature_names':EPV_NAMES,'post_selection_endpoint_feature_scale':[1.0]*38,'post_selection_endpoint_weights':[0.0]*38,'post_selection_endpoint_bias':0.0,
        'post_selection_observable_names':names,'post_selection_quality_observable_names':QUALITY_NAMES,'post_selection_quality_observable_scale':[1.0]*3,'post_selection_quality_observable_weights':[1.0]*3,
        'post_selection_future_response_observable_name':response_name,'post_selection_future_response_scale':1.0,'post_selection_future_response_weight':1.0,'post_selection_selected_bias':0.0}


def test_v43_runtime_value_uses_incumbent_minus_candidate_and_never_changes_proposal():
    raw_names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES); raw=np.zeros((2,len(raw_names))); sup=np.zeros(2); X=np.zeros((2,19)); mu=np.zeros(2)
    names=VALUE_OBSERVABLE_NAMES+FUTURE_RESPONSE_OBSERVABLE_NAMES; obs=np.zeros((2,len(names)))
    obs[0,:3]=2.0; obs[1,:3]=1.0
    ridx=names.index('future_response_robust_agent_cost'); obs[0,ridx]=3.0; obs[1,ridx]=1.0
    c=_runtime_cfg('endpoint_potential_quality_future_response_robust','future_response_robust_agent_cost')
    v,feat,vn=_icer_post_selection_value(1,mu,X,c['feature_names'],c,raw_feat=raw,raw_feature_names=raw_names,support_logits=sup,legacy_action=0,value_observable_matrix=obs,value_observable_names=names)
    assert v==pytest.approx(5.0)  # 3 quality improvements + 2 robust-response improvement
    assert feat.shape[0]==38+3+1
    assert any('future_response_improvement' in x for x in vn)
