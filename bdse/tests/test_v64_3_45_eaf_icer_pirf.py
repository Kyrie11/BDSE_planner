import numpy as np
import pytest

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.interaction_response_field import (
    RESPONSE_FIELD_OBSERVABLE_NAMES, RESPONSE_FIELD_PLAN_FEATURE_NAMES,
    _cv_agent_future, _predict_plan_accel, logged_longitudinal_response_target,
    response_field_local_agent_features, response_field_plan_agent_features,
    runtime_interaction_response_field_observable_costs,
)
from bdse.planner.response_value_observables import FUTURE_RESPONSE_OBSERVABLE_NAMES
from bdse.planner.value_observables import VALUE_OBSERVABLE_NAMES, runtime_value_observable_costs
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES


def _cfg(rf=False):
    return {
        'candidate':{'step_s':0.2,'horizon_s':8.0},
        'teacher':{'risk_aggregation':{'cvar_alpha':0.9,'cvar_weight':0.4}},
        'runtime_safety':{'use_box_agent_risk':True,'ego_length_m':4.8,'ego_width_m':2.0,'soft_agent_radius_m':1.5,'hard_longitudinal_clearance_m':.2,'soft_longitudinal_extra_m':1.0,'hard_lateral_clearance_m':.15,'soft_lateral_extra_m':.65},
        'runtime':{'decisive_frontier_value':{'incumbent_contrastive_extremal_recovery':{'instrument_value_observables':True,'instrument_future_response_observables':True,'instrument_interaction_response_field_observables':rf,'selection_conditioned_intervention_recovery':{}}}},
    }


def _runtime(agent_x=18.0, agent_v=1.0):
    cur=np.array([[agent_x,0,0,agent_v,0,agent_v,0,4.8,2.0,1.0]],dtype=np.float32)
    hist=np.stack([cur.copy(),cur.copy()],axis=1)
    hist[0,0,0]-=agent_v*.2
    ego=np.array([[-.2,0,0,5,0],[0,0,0,5,0]],dtype=np.float32)
    return RuntimeFeatures(ego_history=ego,agent_history=hist,agent_valid=np.ones(1,bool),current_agents=cur,traffic_lights=[],map_features={'route_centerline':np.array([[0,0],[80,0]],dtype=np.float32),'route_corridor_width':4.0,'stop_lines':[],'speed_limit_mps':20.0},route_roadblock_ids=[],mission_goal=None)


def _bank():
    T=41;t=np.arange(1,T+1,dtype=np.float32)*.2;tr=np.zeros((2,T,5),dtype=np.float32)
    tr[0,:,0]=5*t;tr[0,:,3]=5;tr[1,:,0]=5*t;tr[1,:,1]=8;tr[1,:,3]=5;tr[:,:,4]=t[None,:]
    return CandidateBank(trajectories=tr,valid_mask=np.ones(2,bool),maneuver_ids=np.zeros(2,np.int64),theta=[{},{}],dynamic_flags=[{},{}],metadata=[{},{}])


def test_v45_continuous_response_target_recovers_cv_and_deceleration():
    rt=_runtime(agent_v=4.0);cfg=_cfg();T=20;dt=.2
    cv=_cv_agent_future(rt.current_agents[0],T,dt,0.0)[None,:,:]
    assert logged_longitudinal_response_target(rt,cv,0,cfg)==pytest.approx(0.0,abs=1e-6)
    dec=_cv_agent_future(rt.current_agents[0],T,dt,-1.0)[None,:,:]
    assert logged_longitudinal_response_target(rt,dec,0,cfg)==pytest.approx(-1.0,abs=.08)


def test_v45_plan_features_are_exposure_conditioned_and_candidate_specific():
    feat,exp=response_field_plan_agent_features(_runtime(),_bank(),_cfg())
    assert feat.shape==(2,1,len(RESPONSE_FIELD_PLAN_FEATURE_NAMES));assert exp.shape==(2,1)
    assert exp[0,0]>exp[1,0]
    assert np.linalg.norm(feat[0,0])>np.linalg.norm(feat[1,0])


def test_v45_plan_correction_has_exact_zero_at_zero_interaction_features():
    local=np.array([-0.4,0.1]);plan=np.zeros((3,2,len(RESPONSE_FIELD_PLAN_FEATURE_NAMES)))
    model={'enabled':True,'plan_enabled':True,'plan_feature_names':list(RESPONSE_FIELD_PLAN_FEATURE_NAMES),'plan_feature_scale':[1.0]*len(RESPONSE_FIELD_PLAN_FEATURE_NAMES),'plan_weights':[2.0]*len(RESPONSE_FIELD_PLAN_FEATURE_NAMES)}
    pred=_predict_plan_accel(local,plan,model)
    assert np.allclose(pred,np.broadcast_to(local[None,:],pred.shape))


def test_v45_runtime_response_field_is_label_free_and_appends_after_v43_prefix():
    cfg=_cfg(rf=True);x,n=runtime_value_observable_costs(_runtime(),_bank(),cfg)
    assert n==VALUE_OBSERVABLE_NAMES+FUTURE_RESPONSE_OBSERVABLE_NAMES+RESPONSE_FIELD_OBSERVABLE_NAMES
    rf,rn=runtime_interaction_response_field_observable_costs(_runtime(),_bank(),cfg)
    assert rn==RESPONSE_FIELD_OBSERVABLE_NAMES and rf.shape==(2,3) and np.all(np.isfinite(rf))
    assert rf[0,0]>rf[1,0]


def test_v45_local_features_do_not_consume_future():
    x=response_field_local_agent_features(_runtime(),_cfg())
    assert x.shape==(1,6);assert np.all(np.isfinite(x));assert x[0,0]>=0


def test_v45_tournament_accepts_response_field_observable_without_rerank():
    raw_names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES);feature_names=[f'delta::{n}' for n in raw_names]+['delta::support_logit'];obs_names=VALUE_OBSERVABLE_NAMES+FUTURE_RESPONSE_OBSERVABLE_NAMES+RESPONSE_FIELD_OBSERVABLE_NAMES
    sc={'feature_names':feature_names,'feature_mean':[0.0]*19,'feature_std':[1.0]*19,'weights':[0.0]*19,'bias':0.0,'base_feature_names':raw_names,'post_selection_value_enabled':True,'post_selection_value_mode':'endpoint_potential_quality_interaction_response_field','post_selection_endpoint_feature_names':EPV_NAMES,'post_selection_endpoint_feature_scale':[1.0]*38,'post_selection_endpoint_weights':[0.0]*38,'post_selection_endpoint_bias':0.0,'post_selection_observable_names':obs_names,'post_selection_quality_observable_names':['route_deviation_cost','progress_deficit_cost','global_comfort_cost'],'post_selection_quality_observable_scale':[1.0]*3,'post_selection_quality_observable_weights':[0.0]*3,'post_selection_future_response_observable_name':'response_field_plan_occupancy_cost','post_selection_future_response_scale':1.0,'post_selection_future_response_weight':1.0,'post_selection_selected_bias':0.0}
    raw=np.zeros((2,len(raw_names)));sup=np.zeros(2);X=np.zeros((2,19));mu=np.zeros(2);obs=np.zeros((2,len(obs_names)));i=obs_names.index('response_field_plan_occupancy_cost');obs[0,i]=2.0;obs[1,i]=.5
    v,feat,names=_icer_post_selection_value(1,mu,X,feature_names,sc,raw_feat=raw,raw_feature_names=raw_names,support_logits=sup,legacy_action=0,value_observable_matrix=obs,value_observable_names=obs_names)
    assert v==pytest.approx(1.5);assert names[-1].endswith('response_field_plan_occupancy_cost');assert feat.shape[0]==42
