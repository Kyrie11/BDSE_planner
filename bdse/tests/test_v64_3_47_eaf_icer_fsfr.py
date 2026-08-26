import numpy as np
import pytest

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.future_state_factorization import (
    EGO_REFERENCE_FEATURE_NAMES, FSFR_OBSERVABLE_NAMES, _agent_future_2d,
    ego_reference_candidate_features, logged_lateral_drift_target,
    runtime_future_state_factorization_observable_costs,
)
from bdse.planner.interaction_response_field import RESPONSE_FIELD_LOCAL_FEATURE_NAMES, RESPONSE_FIELD_PLAN_FEATURE_NAMES
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.planner.value_observables import VALUE_OBSERVABLE_NAMES, runtime_value_observable_costs
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES
from bdse.tools.fit_v64_3_47_eaf_icer_fsfr import _check_v46


def _runtime():
    cur=np.array([[18,0,0,2,0,2,0,4.8,2.0,1.0]],dtype=np.float32)
    hist=np.stack([cur.copy(),cur.copy()],axis=1); hist[0,0,0]-=.4
    ego=np.array([[-1,0,0,5,0],[0,0,0,5,0]],dtype=np.float32)
    return RuntimeFeatures(ego_history=ego,agent_history=hist,agent_valid=np.ones(1,bool),current_agents=cur,traffic_lights=[],map_features={'route_centerline':np.array([[0,0],[80,0]],dtype=np.float32),'route_corridor_width':4.0,'stop_lines':[],'speed_limit_mps':20.0},route_roadblock_ids=[],mission_goal=None)


def _bank():
    T=41;t=np.arange(1,T+1,dtype=np.float32)*.2;tr=np.zeros((2,T,5),dtype=np.float32)
    tr[0,:,0]=5*t;tr[0,:,3]=5;tr[1,:,0]=5*t;tr[1,:,1]=8;tr[1,:,3]=5;tr[:,:,4]=t[None,:]
    return CandidateBank(trajectories=tr,valid_mask=np.ones(2,bool),maneuver_ids=np.zeros(2,np.int64),theta=[{},{}],dynamic_flags=[{},{}],metadata=[{},{}])


def _cfg():
    mean={'enabled':True,'local_feature_names':list(RESPONSE_FIELD_LOCAL_FEATURE_NAMES),'local_feature_scale':[1.0]*6,'local_weights':[0.0]*6,'local_bias':-0.2,'plan_enabled':True,'plan_feature_names':list(RESPONSE_FIELD_PLAN_FEATURE_NAMES),'plan_feature_scale':[1.0]*6,'plan_weights':[0.0]*6,'plan_bias':0.0}
    lat={'enabled':True,'local_feature_names':list(RESPONSE_FIELD_LOCAL_FEATURE_NAMES),'local_feature_scale':[1.0]*6,'local_weights':[0.0]*6,'local_bias':0.0,'plan_enabled':True,'plan_feature_names':list(RESPONSE_FIELD_PLAN_FEATURE_NAMES),'plan_feature_scale':[1.0]*6,'plan_weights':[0.0]*6,'plan_bias':0.0}
    ego={'enabled':True,'feature_names':list(EGO_REFERENCE_FEATURE_NAMES),'feature_scale':[1.0]*len(EGO_REFERENCE_FEATURE_NAMES),'weights':[0.0]*len(EGO_REFERENCE_FEATURE_NAMES),'bias':0.2}
    return {'candidate':{'step_s':.2,'horizon_s':8.0},'teacher':{'demo_weight':1.0,'demo_scale':120.0,'route_weight':50.0,'route_scale':1.0,'progress_weight':5.0,'progress_scale':10.0,'comfort_global_weight':1.0,'comfort_scale':80.0},'runtime_safety':{'use_box_agent_risk':True,'ego_length_m':4.8,'ego_width_m':2.0,'soft_agent_radius_m':1.5,'hard_longitudinal_clearance_m':.2,'soft_longitudinal_extra_m':1.0,'hard_lateral_clearance_m':.15,'soft_lateral_extra_m':.65},'runtime':{'decisive_frontier_value':{'incumbent_contrastive_extremal_recovery':{'instrument_value_observables':True,'instrument_future_state_factorization_observables':True,'selection_conditioned_intervention_recovery':{'interaction_response_field':mean,'future_state_factorization':{'agent_lateral_response':lat,'ego_reference_model':ego}}}}}}


def test_v47_lateral_target_is_zero_for_cv_and_recovers_constant_drift():
    rt=_runtime();cfg=_cfg();T=20;dt=.2;st=rt.current_agents[0];t=np.arange(1,T+1)*dt
    gt=np.zeros((1,T,5),dtype=float);gt[0,:,0]=st[0]+2*t;gt[0,:,1]=0
    assert logged_lateral_drift_target(rt,gt,0,cfg)==pytest.approx(0.0,abs=1e-8)
    gt[0,:,1]=.5*t
    assert logged_lateral_drift_target(rt,gt,0,cfg)==pytest.approx(.5,abs=1e-8)


def test_v47_2d_agent_rollout_adds_lateral_motion_without_changing_longitudinal_basis():
    st=_runtime().current_agents[0];f=_agent_future_2d(st,10,.2,-.5,.4)
    assert f.shape==(10,5) and np.all(np.isfinite(f))
    assert f[-1,1]>f[0,1]
    z=_agent_future_2d(st,10,.2,-.5,0.0)
    assert np.allclose(z[:,1],0.0,atol=1e-12)


def test_v47_ego_reference_features_are_current_only_finite_and_candidate_specific():
    x=ego_reference_candidate_features(_runtime(),_bank(),_cfg())
    assert x.shape==(2,len(EGO_REFERENCE_FEATURE_NAMES)) and np.all(np.isfinite(x))
    assert not np.allclose(x[0],x[1])


def test_v47_runtime_fsfr_replays_v45_plan_as_first_column():
    c,n=runtime_future_state_factorization_observable_costs(_runtime(),_bank(),_cfg())
    assert n==FSFR_OBSERVABLE_NAMES and c.shape==(2,len(n)) and np.all(np.isfinite(c))
    # With zero lateral model, 1-D and 2-D interaction costs are exactly the same.
    assert np.allclose(c[:,0],c[:,1],atol=1e-12,rtol=0)


def test_v47_runtime_value_schema_can_be_base_plus_fsfr_only():
    x,n=runtime_value_observable_costs(_runtime(),_bank(),_cfg())
    assert n==VALUE_OBSERVABLE_NAMES+FSFR_OBSERVABLE_NAMES
    assert x.shape==(2,len(n))


def test_v47_tournament_consumes_vector_fsfr_without_rerank():
    raw_names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES);feature_names=[f'delta::{n}' for n in raw_names]+['delta::support_logit'];obs_names=VALUE_OBSERVABLE_NAMES+FSFR_OBSERVABLE_NAMES
    rn=['fsfr_plan_2d_occupancy_cost','fsfr_predicted_demo_cost']
    sc={'feature_names':feature_names,'feature_mean':[0.0]*19,'feature_std':[1.0]*19,'weights':[0.0]*19,'bias':0.0,'base_feature_names':raw_names,'post_selection_value_enabled':True,'post_selection_value_mode':'endpoint_potential_quality_future_state_factorization','post_selection_endpoint_feature_names':EPV_NAMES,'post_selection_endpoint_feature_scale':[1.0]*38,'post_selection_endpoint_weights':[0.0]*38,'post_selection_endpoint_bias':0.0,'post_selection_observable_names':obs_names,'post_selection_quality_observable_names':['route_deviation_cost','progress_deficit_cost','global_comfort_cost'],'post_selection_quality_observable_scale':[1.0]*3,'post_selection_quality_observable_weights':[0.0]*3,'post_selection_future_response_observable_names':rn,'post_selection_future_response_scales':[1.0,1.0],'post_selection_future_response_weights':[1.0,2.0]}
    raw=np.zeros((2,len(raw_names)));sup=np.zeros(2);X=np.zeros((2,19));mu=np.zeros(2);obs=np.zeros((2,len(obs_names)))
    obs[0,obs_names.index(rn[0])]=2.;obs[1,obs_names.index(rn[0])]=.5;obs[0,obs_names.index(rn[1])]=1.;obs[1,obs_names.index(rn[1])]=.75
    v,feat,names=_icer_post_selection_value(1,mu,X,feature_names,sc,raw_feat=raw,raw_feature_names=raw_names,support_logits=sup,legacy_action=0,value_observable_matrix=obs,value_observable_names=obs_names)
    assert v==pytest.approx(2.0);assert feat.shape[0]==43;assert names[-2].endswith(rn[0]) and names[-1].endswith(rn[1])


def _authoritative_v46_nested_signature():
    return {
        "rsmr_rank_aggregate": {"selected_count":502,"selected_positive_count":221,"no_positive_opportunity_false_intervention_count":107,"catastrophic_count":28,"teacher_improvement_sum":43.29405361274824},
        "quality_control_aggregate": {"selected_count":205,"selected_positive_count":129,"no_positive_opportunity_false_intervention_count":30,"catastrophic_count":13,"teacher_improvement_sum":43.905547394411805},
        "v45_plan_control_aggregate": {"selected_count":217,"selected_positive_count":121,"no_positive_opportunity_false_intervention_count":38,"catastrophic_count":9,"teacher_improvement_sum":56.55117310290402},
        "distribution_mean_aggregate": {"selected_count":216,"selected_positive_count":118,"no_positive_opportunity_false_intervention_count":39,"catastrophic_count":9,"teacher_improvement_sum":57.52556590728618},
        "temporal_profile_aggregate": {"selected_count":217,"selected_positive_count":119,"no_positive_opportunity_false_intervention_count":41,"catastrophic_count":14,"teacher_improvement_sum":51.263247843232456},
        "dirp_joint_aggregate": {"selected_count":207,"selected_positive_count":115,"no_positive_opportunity_false_intervention_count":39,"catastrophic_count":12,"teacher_improvement_sum":55.303074132712666},
        "failure_diagnosis":"response_second_moment_is_identifiable_but_acceleration_distribution_or_interaction_profile_still_not_absolute_value_sufficient",
        "distribution_identification":{"identified":True},
    }


def test_v47_historical_guard_accepts_authoritative_v46_signature(tmp_path):
    import json
    p=tmp_path/'v46.json'
    p.write_text(json.dumps({"train_gate_pass":False,"nested_crossfit":_authoritative_v46_nested_signature()}))
    _check_v46(p)


def test_v47_historical_guard_still_fails_closed_on_real_v46_drift(tmp_path):
    import json
    n=_authoritative_v46_nested_signature()
    n["distribution_mean_aggregate"]["teacher_improvement_sum"] += 1e-4
    p=tmp_path/'v46_bad.json'
    p.write_text(json.dumps({"train_gate_pass":False,"nested_crossfit":n}))
    with pytest.raises(RuntimeError, match="V46 signature mismatch distribution_mean_aggregate"):
        _check_v46(p)
