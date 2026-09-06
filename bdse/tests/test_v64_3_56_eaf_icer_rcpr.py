from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from bdse.data.cache_schema import RuntimeFeatures
from bdse.planner.paired_constraint_process_retention import (
    CONSTRAINT_PROCESS_NAMES, CONSTRAINT_PROFILE_SCHEMA,
    instantaneous_constraint_risk, paired_constraint_profile,
    constraint_predictor_input, fit_zero_preserving_constraint_predictor,
    predict_constraint_process,
)


def _runtime(agent_x=10.0, agent_vx=0.0, route_y=0.0):
    ego=np.array([[0,0,0,5,0]],dtype=np.float32)
    cur=np.zeros((1,10),dtype=np.float32); cur[0]=[agent_x,0,0,abs(agent_vx),0,agent_vx,0,4.8,2.0,0]
    route=np.array([[0,route_y],[40,route_y]],dtype=np.float32)
    return RuntimeFeatures(ego_history=ego,agent_history=cur[:,None,:].copy(),agent_valid=np.ones(1,bool),current_agents=cur,
        traffic_lights=[],map_features={'route_centerline':route,'route_corridor_width':4.0},route_roadblock_ids=[],mission_goal=None)

def _cfg():
    return {'candidate':{'route_width_m':4.0},'runtime_safety':{'use_box_agent_risk':True,'ego_length_m':4.8,'ego_width_m':2.0,
        'hard_longitudinal_clearance_m':.2,'soft_longitudinal_extra_m':1.0,'hard_lateral_clearance_m':.15,'soft_lateral_extra_m':.65,
        'agent_ttc_safe_s':3.0,'hard_off_route_margin_m':3.0}}

def test_v56_constraint_risk_is_physical_and_finite():
    near=instantaneous_constraint_risk(_runtime(agent_x=4.0,agent_vx=0.0),_cfg())
    far=instantaneous_constraint_risk(_runtime(agent_x=30.0,agent_vx=0.0),_cfg())
    closing=instantaneous_constraint_risk(_runtime(agent_x=6.0,agent_vx=0.0),_cfg())
    off=instantaneous_constraint_risk(_runtime(agent_x=30.0,agent_vx=5.0,route_y=10.0),_cfg())
    assert near.shape==(3,) and np.all(np.isfinite(near))
    assert near[0] > far[0]
    assert closing[1] > 0.0
    assert off[2] > 0.0

def test_v56_paired_profile_sign_and_initial_identity():
    c=np.zeros((6,3),dtype=float); t=c.copy(); t[1:,0]=0.2; t[1:,1]=0.1
    p=paired_constraint_profile(t,c,iteration_indices=list(range(6)),timestamps_us=[i*100000 for i in range(6)])
    assert p['schema']==CONSTRAINT_PROFILE_SCHEMA
    x=np.asarray(p['constraint_support_delta_process'])
    assert x.shape==(len(CONSTRAINT_PROCESS_NAMES),)
    assert np.all(x.reshape(5,3)[:,0] < 0) and np.all(x.reshape(5,3)[:,1] < 0)

def test_v56_constraint_predictor_is_zero_preserving():
    rng=np.random.default_rng(4); X=rng.normal(size=(96,15)); Y=rng.normal(size=(96,len(CONSTRAINT_PROCESS_NAMES)))
    m=fit_zero_preserving_constraint_predictor(X,Y)
    z=predict_constraint_process(np.zeros(15),m)
    assert np.max(np.abs(z))==0.0

def test_v56_predictor_input_dose_gates_context():
    p={'endpoint_signed':[0,0,0,0],'cosine_modes_1_2':[0]*8,'execution_contrast_linf':0.0}
    x=constraint_predictor_input(p,np.array([.9,.8,.7]))
    assert np.max(np.abs(x))==0.0

def test_v56_preregistration_is_final_family_and_sequential():
    root=Path(__file__).resolve().parents[2]
    d=json.loads((root/'V64_3_56_PREREGISTRATION.json').read_text())
    assert d['branch_order']==['realized_constraint_process','predicted_constraint_process']
    assert d['internal_search_stop']['if_realized_constraint_process_fails'].startswith('STOP internal algorithm search')
    assert d['frozen']['V55_pareto_functional_and_deployment_gate'] is True

def test_v56_constraint_sidecar_installs_on_actual_adapter():
    import bdse.tools.nuplan_v56_constraint_process_run_simulation as wrapper
    cls = wrapper.v54._resolve_nuplan_planner_class()
    original = cls.compute_planner_trajectory
    marker = wrapper._MARK
    had_marker = hasattr(cls, marker)
    old_marker = getattr(cls, marker, None)
    try:
        if had_marker:
            delattr(cls, marker)
        wrapper.install_constraint_sidecar()
        assert cls.compute_planner_trajectory is not original
        assert getattr(cls, marker, False) is True
    finally:
        cls.compute_planner_trajectory = original
        if had_marker:
            setattr(cls, marker, old_marker)
        elif hasattr(cls, marker):
            delattr(cls, marker)


def test_v56_launcher_locks_planned_zero_and_does_not_recollect_outcome():
    root=Path(__file__).resolve().parents[2]
    launcher=(root/'RUN_V64_3_56_EAF_ICER_RCPR_TRAIN.sh').read_text(encoding='utf-8')
    collector=(root/'bdse/tools/run_v64_3_56_constraint_process_probe.py').read_text(encoding='utf-8')
    assert '--v53-operator-profiles "$V53_PROFILES"' in launcher
    assert 'planned_equal != 38' in collector
    assert '"paired_outcome_labels_recollected":False' in collector
    assert 'run_metric=false' in collector


def test_v56_fit_is_strictly_sequential_and_final_stop():
    root=Path(__file__).resolve().parents[2]
    fit=(root/'bdse/tools/fit_v64_3_56_eaf_icer_rcpr.py').read_text(encoding='utf-8')
    assert 'if not opass:' in fit
    assert 'NOT_EVALUATED_BY_PREREGISTERED_BRANCH_ORDER' in fit
    assert fit.index('if not opass:') < fit.index('pred=_evaluate(rows,predicted=True')
    assert 'internal_search_converged' in fit
