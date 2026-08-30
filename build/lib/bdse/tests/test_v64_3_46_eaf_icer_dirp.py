import numpy as np
import pytest

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.distributional_interaction_response import (
    DIRP_OBSERVABLE_NAMES,
    SIGMA_POINT_OFFSETS,
    SIGMA_POINT_WEIGHTS,
    _profile_functionals,
    runtime_distributional_interaction_response_observable_costs,
)
from bdse.planner.interaction_response_field import (
    RESPONSE_FIELD_LOCAL_FEATURE_NAMES,
    RESPONSE_FIELD_OBSERVABLE_NAMES,
    RESPONSE_FIELD_PLAN_FEATURE_NAMES,
    runtime_interaction_response_field_observable_costs,
)
from bdse.planner.response_value_observables import FUTURE_RESPONSE_OBSERVABLE_NAMES
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.planner.value_observables import VALUE_OBSERVABLE_NAMES, runtime_value_observable_costs
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES


def _runtime():
    cur=np.array([[18,0,0,2,0,2,0,4.8,2.0,1.0]],dtype=np.float32)
    hist=np.stack([cur.copy(),cur.copy()],axis=1);hist[0,0,0]-=.4
    ego=np.array([[-1,0,0,5,0],[0,0,0,5,0]],dtype=np.float32)
    return RuntimeFeatures(ego_history=ego,agent_history=hist,agent_valid=np.ones(1,bool),current_agents=cur,traffic_lights=[],map_features={'route_centerline':np.array([[0,0],[80,0]],dtype=np.float32),'route_corridor_width':4.0,'stop_lines':[],'speed_limit_mps':20.0},route_roadblock_ids=[],mission_goal=None)


def _bank():
    T=41;t=np.arange(1,T+1,dtype=np.float32)*.2;tr=np.zeros((2,T,5),dtype=np.float32)
    tr[0,:,0]=5*t;tr[0,:,3]=5;tr[1,:,0]=5*t;tr[1,:,1]=8;tr[1,:,3]=5;tr[:,:,4]=t[None,:]
    return CandidateBank(trajectories=tr,valid_mask=np.ones(2,bool),maneuver_ids=np.zeros(2,np.int64),theta=[{},{}],dynamic_flags=[{},{}],metadata=[{},{}])


def _cfg(enable_dirp=True):
    mean={'enabled':True,'model':'agent_local_continuous_longitudinal_response_field','local_feature_names':list(RESPONSE_FIELD_LOCAL_FEATURE_NAMES),'local_feature_scale':[1.0]*6,'local_weights':[0.0]*6,'local_bias':-0.2,'plan_enabled':True,'plan_feature_names':list(RESPONSE_FIELD_PLAN_FEATURE_NAMES),'plan_feature_scale':[1.0]*6,'plan_weights':[0.0]*6,'plan_bias':0.0}
    m2={'enabled':True,'local_feature_names':list(RESPONSE_FIELD_LOCAL_FEATURE_NAMES),'local_feature_scale':[1.0]*6,'local_weights':[0.0]*6,'local_bias':0.20,'plan_enabled':True,'plan_feature_names':list(RESPONSE_FIELD_PLAN_FEATURE_NAMES),'plan_feature_scale':[1.0]*6,'plan_weights':[0.0]*6,'plan_bias':0.0}
    return {'candidate':{'step_s':.2,'horizon_s':8.0},'runtime_safety':{'use_box_agent_risk':True,'ego_length_m':4.8,'ego_width_m':2.0,'soft_agent_radius_m':1.5,'hard_longitudinal_clearance_m':.2,'soft_longitudinal_extra_m':1.0,'hard_lateral_clearance_m':.15,'soft_lateral_extra_m':.65},'runtime':{'decisive_frontier_value':{'incumbent_contrastive_extremal_recovery':{'instrument_value_observables':True,'instrument_future_response_observables':True,'instrument_interaction_response_field_observables':True,'instrument_distributional_interaction_response_observables':enable_dirp,'selection_conditioned_intervention_recovery':{'interaction_response_field':mean,'distributional_interaction_response_field':m2}}}}}


def test_v46_sigma_rule_matches_zero_mean_unit_variance():
    assert np.dot(SIGMA_POINT_WEIGHTS,SIGMA_POINT_OFFSETS)==pytest.approx(0.0,abs=1e-12)
    assert np.dot(SIGMA_POINT_WEIGHTS,SIGMA_POINT_OFFSETS**2)==pytest.approx(1.0,abs=1e-12)
    assert SIGMA_POINT_WEIGHTS.sum()==pytest.approx(1.0)


def test_v46_profile_functionals_are_bounded_and_early_sensitive():
    p=np.array([[1.0,0,0,0],[0,0,0,1.0]],dtype=float)
    m,peak,early,second=_profile_functionals(p)
    assert np.allclose(m,[.25,.25]);assert np.allclose(peak,[1,1]);assert early[0]>early[1];assert np.allclose(second,[.25,.25])


def test_v46_plan_mean_exactly_replays_v45_plan_occupancy():
    cfg=_cfg();rf,_=runtime_interaction_response_field_observable_costs(_runtime(),_bank(),cfg);dr,n=runtime_distributional_interaction_response_observable_costs(_runtime(),_bank(),cfg)
    assert n==DIRP_OBSERVABLE_NAMES
    assert np.allclose(dr[:,0],rf[:,2],atol=1e-12,rtol=0)


def test_v46_distributional_observables_are_runtime_only_and_finite():
    dr,n=runtime_distributional_interaction_response_observable_costs(_runtime(),_bank(),_cfg())
    assert n==DIRP_OBSERVABLE_NAMES and dr.shape==(2,len(n)) and np.all(np.isfinite(dr))
    assert np.all(dr>=0)


def test_v46_runtime_value_observable_schema_appends_after_v45():
    x,n=runtime_value_observable_costs(_runtime(),_bank(),_cfg())
    assert n==VALUE_OBSERVABLE_NAMES+FUTURE_RESPONSE_OBSERVABLE_NAMES+RESPONSE_FIELD_OBSERVABLE_NAMES+DIRP_OBSERVABLE_NAMES
    assert x.shape==(2,len(n))


def test_v46_tournament_consumes_vector_response_profile_without_rerank():
    raw_names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES);feature_names=[f'delta::{n}' for n in raw_names]+['delta::support_logit'];obs_names=VALUE_OBSERVABLE_NAMES+FUTURE_RESPONSE_OBSERVABLE_NAMES+RESPONSE_FIELD_OBSERVABLE_NAMES+DIRP_OBSERVABLE_NAMES
    rn=['dirp_distribution_mean_occupancy_cost','dirp_distribution_peak_occupancy_cost']
    sc={'feature_names':feature_names,'feature_mean':[0.0]*19,'feature_std':[1.0]*19,'weights':[0.0]*19,'bias':0.0,'base_feature_names':raw_names,'post_selection_value_enabled':True,'post_selection_value_mode':'endpoint_potential_quality_distributional_response_profile','post_selection_endpoint_feature_names':EPV_NAMES,'post_selection_endpoint_feature_scale':[1.0]*38,'post_selection_endpoint_weights':[0.0]*38,'post_selection_endpoint_bias':0.0,'post_selection_observable_names':obs_names,'post_selection_quality_observable_names':['route_deviation_cost','progress_deficit_cost','global_comfort_cost'],'post_selection_quality_observable_scale':[1.0]*3,'post_selection_quality_observable_weights':[0.0]*3,'post_selection_future_response_observable_names':rn,'post_selection_future_response_scales':[1.0,1.0],'post_selection_future_response_weights':[1.0,2.0]}
    raw=np.zeros((2,len(raw_names)));sup=np.zeros(2);X=np.zeros((2,19));mu=np.zeros(2);obs=np.zeros((2,len(obs_names)))
    obs[0,obs_names.index(rn[0])]=2.0;obs[1,obs_names.index(rn[0])]=.5;obs[0,obs_names.index(rn[1])]=1.0;obs[1,obs_names.index(rn[1])]=.75
    v,feat,names=_icer_post_selection_value(1,mu,X,feature_names,sc,raw_feat=raw,raw_feature_names=raw_names,support_logits=sup,legacy_action=0,value_observable_matrix=obs,value_observable_names=obs_names)
    assert v==pytest.approx(2.0);assert names[-2].endswith(rn[0]) and names[-1].endswith(rn[1]);assert feat.shape[0]==43
