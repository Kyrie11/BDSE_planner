import numpy as np
import pytest

from bdse.planner.paired_operator_contrast_retention import (
    POCR_ADDITIVE_STATE_NAMES, POCR_INTERACTION_STATE_NAMES,
    execution_contrast_linf, operator_state, runtime_certificate,
)
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.planner.future_state_factorization import FSFR_OBSERVABLE_NAMES
from bdse.planner.value_observables import VALUE_OBSERVABLE_NAMES
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES
from bdse.tools.fit_v64_3_51_eaf_icer_pocr import _arm_state, _fit_ranker, _risk


def _model(names, w=None):
    n=len(names); w=[0.0]*n if w is None else list(w)
    return {"model":"zero_bias_pairwise_selected_sign_risk","feature_names":list(names),"feature_mean":[0.0]*n,"feature_std":[1.0]*n,"weights":w,"bias":0.0,"lambda":1.0,"fit_positive_score_mean":0.0,"fit_positive_score_std":1.0}


def test_v51_execution_contrast_exactly_matches_bounded_tensor_linf():
    a=np.array([[0.,0.,0.,1.,.1],[2.,1.,.2,3.,.2]])
    b=np.array([[0.,0.,0.,1.,.1],[1.,-2.,.1,2.,.2]])
    assert execution_contrast_linf(a,b)==pytest.approx(3.0)
    assert execution_contrast_linf(a,a)==0.0


def test_v51_operator_state_additive_and_interaction_are_fixed_factorization():
    a=np.zeros((2,5)); b=np.zeros((2,5)); a[1,0]=2.0
    z=operator_state(1.0,2.5,2.0,a,b,include_dose_interactions=False)
    assert z.tolist()==pytest.approx([1.0,1.5,-0.5,2.0])
    zi=operator_state(1.0,2.5,2.0,a,b,include_dose_interactions=True)
    assert zi.tolist()==pytest.approx([1.0,1.5,-0.5,2.0,2.0,3.0,-1.0])


def test_v51_pairwise_ranker_can_use_operator_contrast_without_changing_loss_family():
    rows=[]
    for i in range(40):
        rows.append({"scenario_token":f"g{i}","quality_value":0.,"plan_control_value":0.,"ego_ref_value":0.,"operator_execution_contrast_linf":1.+.01*i,"rsm_selected_teacher_improvement":1.})
    for i in range(40):
        rows.append({"scenario_token":f"b{i}","quality_value":0.,"plan_control_value":0.,"ego_ref_value":0.,"operator_execution_contrast_linf":10.+.01*i,"rsm_selected_teacher_improvement":-1.})
    m=_fit_ranker(rows,"qpe_dose")
    assert m["lambda"]==1.0 and m["bias"]==0.0
    assert m["feature_names"]==POCR_ADDITIVE_STATE_NAMES
    assert m["objective_final"]<m["objective_at_zero"]
    assert _risk(rows[-1],m)>_risk(rows[0],m)


def test_v51_runtime_certificate_uses_only_stored_sign_risk_and_frozen_threshold():
    cfg={"feature_names":POCR_ADDITIVE_STATE_NAMES,"aggregation":"sign_only","include_dose_interactions":False,"components":{"sign_risk":_model(POCR_ADDITIVE_STATE_NAMES,[0,0,0,1])},"retention_threshold":2.0}
    z=np.asarray([0.,0.,0.,3.])
    cert,risk,parts=runtime_certificate(z,cfg)
    assert risk==pytest.approx(3.0); assert cert==pytest.approx(-1.0); assert parts["execution_contrast_linf"]==3.0


def test_v51_tournament_computes_runtime_treatment_control_contrast_without_reranking():
    raw_names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    feature_names=[f"delta::{n}" for n in raw_names]+["delta::support_logit"]
    obs_names=VALUE_OBSERVABLE_NAMES+FSFR_OBSERVABLE_NAMES
    pnames=["fsfr_plan_1d_occupancy_cost"]; enames=["fsfr_plan_1d_occupancy_cost","fsfr_predicted_demo_cost"]
    risk=_model(POCR_ADDITIVE_STATE_NAMES,[0.,0.,0.,1.])
    sc={"feature_names":feature_names,"feature_mean":[0.]*19,"feature_std":[1.]*19,"weights":[0.]*19,"bias":0.,"base_feature_names":raw_names,"post_selection_value_enabled":True,"post_selection_value_mode":"endpoint_potential_quality_paired_operator_contrast_retention","post_selection_endpoint_feature_names":EPV_NAMES,"post_selection_endpoint_feature_scale":[1.]*38,"post_selection_endpoint_weights":[0.]*38,"post_selection_endpoint_bias":0.,"post_selection_observable_names":obs_names,"post_selection_quality_observable_names":["route_deviation_cost","progress_deficit_cost","global_comfort_cost"],"post_selection_quality_observable_scale":[1.]*3,"post_selection_quality_observable_weights":[0.]*3,"selected_policy_risk_plan_response_names":pnames,"selected_policy_risk_plan_response_scales":[1.],"selected_policy_risk_plan_response_weights":[0.],"selected_policy_risk_ego_reference_names":enames,"selected_policy_risk_ego_reference_scales":[1.,1.],"selected_policy_risk_ego_reference_weights":[0.,0.],"paired_operator_contrast_retention":{"feature_names":POCR_ADDITIVE_STATE_NAMES,"aggregation":"sign_only","include_dose_interactions":False,"components":{"sign_risk":risk},"retention_threshold":2.0}}
    raw=np.zeros((2,len(raw_names))); sup=np.zeros(2); X=np.zeros((2,19)); mu=np.zeros(2); obs=np.zeros((2,len(obs_names)))
    traj=np.zeros((2,2,5)); traj[1,1,0]=3.0
    v,_,names=_icer_post_selection_value(1,mu,X,feature_names,sc,raw_feat=raw,raw_feature_names=raw_names,support_logits=sup,legacy_action=0,value_observable_matrix=obs,value_observable_names=obs_names,selection_multiplicity=2,candidate_trajectories=traj)
    assert v==pytest.approx(-1.0)
    assert "pocr_risk" in names


def test_v51_dose_interaction_schema_is_not_the_closed_v48_multiplicity_state():
    row={"scenario_token":"x","quality_value":1.,"plan_control_value":2.,"ego_ref_value":1.5,"operator_execution_contrast_linf":4.}
    z=_arm_state(row,"qpe_dose_interaction")
    assert len(z)==7 and POCR_INTERACTION_STATE_NAMES[-1]=="contrast_x_ego_reference_increment"
    assert "log_extremal_multiplicity" not in POCR_INTERACTION_STATE_NAMES
