from __future__ import annotations

import numpy as np
import pytest

from bdse.planner.paired_operator_trajectory_retention import (
    PROFILE_SCHEMA, POTR_SUPPORT_STATE_NAMES, POTR_ENDPOINT_STATE_NAMES, POTR_TEMPORAL_STATE_NAMES,
    trajectory_contrast_profile, outcome_state_from_profile, runtime_states, runtime_certificate,
)
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.planner.future_state_factorization import FSFR_OBSERVABLE_NAMES
from bdse.planner.value_observables import VALUE_OBSERVABLE_NAMES
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES
from bdse.tools.fit_v64_3_53_eaf_icer_potr import _fit_outcome, _outcome_risk


def _component(names, w):
    return {"model":"unit","feature_names":list(names),"feature_mean":[0.0]*len(names),"feature_std":[1.0]*len(names),"weights":list(w),"bias":0.0,"lambda":1.0,"fit_beneficial_score_mean":0.0,"fit_beneficial_score_std":1.0}


def test_v53_profile_replays_scalar_D_and_zero_is_exact_null():
    inc=np.zeros((5,5)); prop=inc.copy(); prop[-1,0]=3.; prop[-1,3]=1.5
    p=trajectory_contrast_profile(prop,inc)
    assert p["schema"]==PROFILE_SCHEMA
    assert p["execution_contrast_linf"]==pytest.approx(3.0)
    assert p["endpoint_signed"]==pytest.approx([3.,0.,0.,1.5])
    z=trajectory_contrast_profile(inc,inc)
    assert z["execution_contrast_linf"]==0.0
    assert max(abs(x) for x in z["endpoint_signed"]+z["cosine_modes_1_2"])<1e-12


def test_v53_signed_yaw_uses_wrapped_operator_contrast():
    inc=np.zeros((4,5)); prop=np.zeros((4,5))
    inc[-1,2]=np.pi-0.1; prop[-1,2]=-np.pi+0.1
    p=trajectory_contrast_profile(prop,inc)
    assert p["endpoint_signed"][2]==pytest.approx(0.2,abs=1e-10)


def test_v53_endpoint_and_temporal_states_are_nested_fixed_families():
    inc=np.zeros((8,5)); prop=inc.copy(); prop[:,0]=np.linspace(0,2,8); prop[:,1]=np.linspace(0,-1,8)
    profile=trajectory_contrast_profile(prop,inc)
    ze=outcome_state_from_profile(1.,2.5,2.,profile,state_family="endpoint")
    zt=outcome_state_from_profile(1.,2.5,2.,profile,state_family="temporal")
    assert ze.shape==(len(POTR_ENDPOINT_STATE_NAMES),)
    assert zt.shape==(len(POTR_TEMPORAL_STATE_NAMES),)
    assert zt[:len(ze)].tolist()==pytest.approx(ze.tolist())
    assert POTR_ENDPOINT_STATE_NAMES[:4]==POTR_SUPPORT_STATE_NAMES
    assert len(POTR_TEMPORAL_STATE_NAMES)-len(POTR_ENDPOINT_STATE_NAMES)==8


def test_v53_pairwise_outcome_ranker_can_use_direction_after_scalar_D_is_matched():
    rows=[]
    for i in range(40):
        p=np.zeros((6,5)); q=np.zeros((6,5)); p[-1,0]=2.; p[-1,1]=+1.+.01*i
        prof=trajectory_contrast_profile(p,q)
        rows.append({"scenario_token":f"g{i}","quality_value":0.,"plan_control_value":0.,"ego_ref_value":0.,"operator_execution_contrast_linf":prof["execution_contrast_linf"],"operator_trajectory_profile":prof,"closed_loop_beneficial":True,"closed_loop_score_delta":1.,"safety_delta":{"collision":0.},"closed_loop_hard_harm":False})
    for i in range(40):
        p=np.zeros((6,5)); q=np.zeros((6,5)); p[-1,0]=2.; p[-1,1]=-1.-.01*i
        prof=trajectory_contrast_profile(p,q)
        rows.append({"scenario_token":f"b{i}","quality_value":0.,"plan_control_value":0.,"ego_ref_value":0.,"operator_execution_contrast_linf":prof["execution_contrast_linf"],"operator_trajectory_profile":prof,"closed_loop_beneficial":False,"closed_loop_score_delta":-1.,"safety_delta":{"collision":0.},"closed_loop_hard_harm":False})
    m=_fit_outcome(rows,"endpoint")
    assert m["lambda"]==1.0 and m["bias"]==0.0
    assert m["objective_final"]<m["objective_at_zero"]
    assert _outcome_risk(rows[-1],m)>_outcome_risk(rows[0],m)


def test_v53_runtime_keeps_frozen_support_and_combines_one_threshold():
    inc=np.zeros((6,5)); prop=inc.copy(); prop[-1,0]=2.; prop[-1,1]=-1.
    sz,oz,_=runtime_states(0.,0.,0.,prop,inc,state_family="endpoint")
    support=_component(POTR_SUPPORT_STATE_NAMES,[0,0,0,1])
    ow=[0.0]*len(POTR_ENDPOINT_STATE_NAMES); ow[5]=1.0
    outcome=_component(POTR_ENDPOINT_STATE_NAMES,ow)
    cfg={"state_family":"endpoint","aggregation":"max_support_outcome","components":{"effect_support_risk":support,"conditional_outcome_risk":outcome},"retention_threshold":1.5}
    cert,risk,parts=runtime_certificate(sz,oz,cfg)
    assert parts["effect_support_risk"]==pytest.approx(2.0)
    assert risk>=2.0 and cert==pytest.approx(1.5-risk)


def test_v53_tournament_only_changes_retention_certificate_not_winner():
    raw_names=list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    feature_names=[f"delta::{n}" for n in raw_names]+["delta::support_logit"]
    obs_names=VALUE_OBSERVABLE_NAMES+FSFR_OBSERVABLE_NAMES
    pnames=["fsfr_plan_1d_occupancy_cost"]; enames=["fsfr_plan_1d_occupancy_cost","fsfr_predicted_demo_cost"]
    support=_component(POTR_SUPPORT_STATE_NAMES,[0.,0.,0.,1.])
    ow=[0.]*len(POTR_ENDPOINT_STATE_NAMES); ow[4]=1.
    outcome=_component(POTR_ENDPOINT_STATE_NAMES,ow)
    sc={"feature_names":feature_names,"feature_mean":[0.]*19,"feature_std":[1.]*19,"weights":[0.]*19,"bias":0.,"base_feature_names":raw_names,"post_selection_value_enabled":True,"post_selection_value_mode":"endpoint_potential_quality_paired_operator_trajectory_retention","post_selection_endpoint_feature_names":EPV_NAMES,"post_selection_endpoint_feature_scale":[1.]*38,"post_selection_endpoint_weights":[0.]*38,"post_selection_endpoint_bias":0.,"post_selection_observable_names":obs_names,"post_selection_quality_observable_names":["route_deviation_cost","progress_deficit_cost","global_comfort_cost"],"post_selection_quality_observable_scale":[1.]*3,"post_selection_quality_observable_weights":[0.]*3,"selected_policy_risk_plan_response_names":pnames,"selected_policy_risk_plan_response_scales":[1.],"selected_policy_risk_plan_response_weights":[0.],"selected_policy_risk_ego_reference_names":enames,"selected_policy_risk_ego_reference_scales":[1.,1.],"selected_policy_risk_ego_reference_weights":[0.,0.],"paired_operator_trajectory_retention":{"state_family":"endpoint","aggregation":"max_support_outcome","components":{"effect_support_risk":support,"conditional_outcome_risk":outcome},"retention_threshold":1.0}}
    raw=np.zeros((2,len(raw_names))); sup=np.zeros(2); X=np.zeros((2,19)); mu=np.zeros(2); obs=np.zeros((2,len(obs_names)))
    traj=np.zeros((2,6,5)); traj[1,-1,0]=3.
    v,_,names=_icer_post_selection_value(1,mu,X,feature_names,sc,raw_feat=raw,raw_feature_names=raw_names,support_logits=sup,legacy_action=0,value_observable_matrix=obs,value_observable_names=obs_names,candidate_trajectories=traj)
    assert v<0.0
    assert "potr_risk" in names


def test_v53_nuplan_profile_hook_is_process_local_and_historical_planner_unchanged():
    import hashlib, inspect
    from pathlib import Path
    from bdse.tools import nuplan_v53_operator_profile_run_simulation as hook
    src=inspect.getsource(hook.install_operator_profile_sidecar)
    assert 'trajectory_contrast_profile(expected, incumbent)' in src
    assert 'return chosen, diag' in src
    root=Path(__file__).resolve().parents[2]
    assert hashlib.sha256((root/'bdse/planner/nuplan_planner.py').read_bytes()).hexdigest() == 'c3a6e37901349408b7c8e6ab7b3811f905f3a81b0c441e6aa7ddf4dde92131ef'
