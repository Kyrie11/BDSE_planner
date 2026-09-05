from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bdse.planner.paired_dynamic_response_mediation import (
    DYNAMIC_PROFILE_SCHEMA,
    PDRM_ENDPOINT_STATE_NAMES,
    PDRM_TEMPORAL_STATE_NAMES,
    outcome_state_from_dynamic_profile,
    paired_realized_profile,
)
from bdse.tools.nuplan_v54_dynamic_response_run_simulation import _ego_row


def _trace(n=6):
    c=np.zeros((n,4),dtype=float); c[:,0]=np.arange(n,dtype=float); c[:,2]=math.pi/2; c[:,3]=5.0
    t=c.copy(); t[:,1]+=np.linspace(0,1,n); t[:,3]+=np.linspace(0,0.5,n)
    return t,c


def test_profile_zero_identity():
    _,c=_trace(); p=paired_realized_profile(c,c,iteration_indices=list(range(6)),timestamps_us=[i*50000 for i in range(6)],planned_execution_contrast_linf=0.0)
    assert p["schema"]==DYNAMIC_PROFILE_SCHEMA
    assert p["realized_response_linf"]==0.0
    assert max(abs(x) for x in p["endpoint_signed"]+p["cosine_modes_1_2"])==0.0


def test_profile_uses_common_initial_local_frame():
    t,c=_trace(); p=paired_realized_profile(t,c,iteration_indices=list(range(6)),timestamps_us=[i*50000 for i in range(6)],planned_execution_contrast_linf=3.0)
    # yaw0=pi/2, a +world-y treatment displacement is +local-x.
    assert p["endpoint_signed"][0] > 0.99
    assert abs(p["endpoint_signed"][1]) < 1e-9
    assert p["endpoint_signed"][3] > 0.49


def test_profile_rejects_unsynchronized_indices():
    t,c=_trace()
    try:
        paired_realized_profile(t,c,iteration_indices=[0,1,2,4,5,6],timestamps_us=[i*50000 for i in range(6)],planned_execution_contrast_linf=1.0)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("expected fail-closed index check")


def test_endpoint_and_temporal_state_shapes():
    t,c=_trace(); p=paired_realized_profile(t,c,iteration_indices=list(range(6)),timestamps_us=[i*50000 for i in range(6)],planned_execution_contrast_linf=2.5)
    e=outcome_state_from_dynamic_profile(1.0,1.2,1.5,p,state_family="realized_endpoint")
    z=outcome_state_from_dynamic_profile(1.0,1.2,1.5,p,state_family="realized_temporal")
    assert e.shape==(len(PDRM_ENDPOINT_STATE_NAMES),)
    assert z.shape==(len(PDRM_TEMPORAL_STATE_NAMES),)
    assert np.allclose(e[:4],[1.0,0.2,0.3,2.5])
    assert np.allclose(z[:len(e)],e)


def test_ego_row_reads_current_simulated_state_only():
    rear=SimpleNamespace(x=1.0,y=2.0,heading=0.3)
    dyn=SimpleNamespace(speed=4.5)
    ego=SimpleNamespace(rear_axle=rear,dynamic_car_state=dyn)
    obs=object(); hist=SimpleNamespace(current_state=(ego,obs))
    it=SimpleNamespace(index=5,time_point=SimpleNamespace(time_us=123456))
    row=_ego_row(SimpleNamespace(history=hist,iteration=it))
    assert row=={"iteration_index":5,"time_us":123456,"ego_world":[1.0,2.0,0.3,4.5]}


def test_v54_preregistration_closes_v53_preexecution_geometry():
    p=Path(__file__).resolve().parents[2]/"V64_3_54_PREREGISTRATION.json"
    d=json.loads(p.read_text(encoding="utf-8"))
    assert d["trigger"]["v53_failure_diagnosis"]=="preexecution_operator_trajectory_contrast_does_not_identify_effectful_outcome_order"
    assert d["branch_order"]==["realized_endpoint","realized_temporal"]
    assert d["runtime_claim_boundary"]["post_intervention_state_not_t0_available"] is True


def test_v54_gate_keeps_all_frozen_conditional_controls():
    p=Path(__file__).resolve().parents[2]/"V64_3_54_PREREGISTRATION.json"
    d=json.loads(p.read_text(encoding="utf-8"))
    g=d["arms"]["realized_endpoint"]["identification_gate"]
    assert g["aggregate_auc_gt_exact_V53_temporal"] is True and g["folds_gt_exact_V53_temporal_min"]==4
    assert g["aggregate_auc_gt_exact_V52_scalar"] is True and g["folds_gt_exact_V52_scalar_min"]==4
    assert g["aggregate_auc_gt_exact_V51_scalar"] is True and g["folds_gt_exact_V51_scalar_min"]==4


def test_v54_short_horizon_wrapper_bypasses_terminal_replan_and_metrics():
    root=Path(__file__).resolve().parents[2]
    wrapper=(root/"bdse/tools/nuplan_v54_dynamic_response_run_simulation.py").read_text(encoding="utf-8")
    runner=(root/"bdse/tools/run_v64_3_54_dynamic_response_probe.py").read_text(encoding="utf-8")
    assert "bypass_terminal_replan = bool(idx == exposure)" in wrapper
    assert "_cached_local_trajectory" in wrapper
    assert '"run_metric=false"' in runner
    assert '"outcome_labels_recollected": False' in runner
