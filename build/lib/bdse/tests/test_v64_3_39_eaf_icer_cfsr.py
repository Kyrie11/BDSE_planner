from __future__ import annotations

import numpy as np
import pytest

from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.tools.fit_v64_3_39_eaf_icer_cfsr import (
    _fit_selection_residual,
    _fit_translation,
)


def _names():
    return [f"delta::{n}" for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES] + ["delta::support_logit"]


def _base_cfg(d: int):
    rw=np.zeros(d); rw[0]=1.0
    dw=np.zeros(d); dw[1]=0.5
    return {
        "post_selection_value_enabled":True,
        "post_selection_value_mode":"dense_edge_cfsr",
        "scene_reservation_enabled":False,
        "feature_names":_names(),"feature_mean":[0.0]*d,"feature_std":[1.0]*d,"weights":rw.tolist(),"bias":0.0,
        "post_selection_dense_feature_mean":[0.0]*d,"post_selection_dense_feature_std":[1.0]*d,"post_selection_dense_weights":dw.tolist(),"post_selection_dense_bias":0.0,
        "post_selection_cfsr_feature_mean":[0.0]*d,"post_selection_cfsr_feature_std":[1.0]*d,
        "post_selection_cfsr_weights":([0.0,0.0,1.0]+[0.0]*(d-3)),"post_selection_cfsr_bias":0.0,
        "post_selection_value_max_abs":40.0,
    }


def test_cfsr_runtime_adds_dense_and_residual_only_after_frozen_proposal():
    names=_names(); d=len(names); X=np.zeros((3,d)); X[1,0]=2.0; X[1,1]=4.0; X[1,2]=-1.0
    mu=np.array([0.0,2.0,1.0]); cfg=_base_cfg(d)
    v,feat,fn=_icer_post_selection_value(1,mu,X,names,cfg)
    # dense=2, residual=-1 => corrected selected-proposal value=1. No other candidate is scored here.
    assert v==pytest.approx(1.0)
    assert feat.tolist()==pytest.approx([2.0,-1.0])
    assert fn==["post_value::dense_all_edge_absolute_value","post_value::cross_fitted_selection_residual"]


def test_cfsr_translation_is_unit_slope_and_preserves_value_order():
    x=np.linspace(-2.0,3.0,100); y=x+0.75
    m=_fit_translation(x,y,"unit")
    assert m["selected_policy_bias"]==pytest.approx(0.75)
    assert m["operator"]=="unit_slope_translation_only_preserves_value_ordering"
    shifted=x+m["selected_policy_bias"]
    assert np.all(np.diff(shifted)>0.0)


def test_cfsr_shift_runtime_preserves_corrected_value_ordering():
    names=_names(); d=len(names); X=np.zeros((3,d)); X[1,0]=2.0; X[1,1]=2.0; X[1,2]=0.5
    mu=np.array([0.0,2.0,1.0]); cfg=_base_cfg(d); cfg["post_selection_value_mode"]="dense_edge_cfsr_shift"; cfg["post_selection_selected_bias"]=-0.25
    v,_,_=_icer_post_selection_value(1,mu,X,names,cfg)
    # dense=1, residual=.5, translation=-.25
    assert v==pytest.approx(1.25)


def test_selection_residual_is_orthogonal_to_rank_and_dense_directions():
    d=len(_names()); rng=np.random.default_rng(7)
    # final RSMR raw direction e0, final dense raw direction e1.
    rw=np.zeros(d); rw[0]=1.0; rsm=(rw,np.ones(d),{})
    dw=np.zeros(d); dw[1]=1.0; dense=(dw,0.0,np.zeros(d),np.ones(d),np.eye(d))
    rows=[]
    for i in range(240):
        x=rng.normal(size=d)
        # residual signal lives in e2 plus noise, not in e0/e1.
        e=1.3*x[2]+0.05*rng.normal()
        rows.append({"token":f"t{i}","x":x,"residual":e,"dense_oof":0.0,"y":e})
    m=_fit_selection_residual(rows,rsm,dense)
    w=np.asarray(m["weights"]); std=np.asarray(m["feature_std"])
    dr=(rw/np.ones(d))*std; dd=(dw/np.ones(d))*std
    assert abs(float(w@dr))<1e-8
    assert abs(float(w@dd))<1e-8
    assert m["oof_dense_residual_mse_after_feature_correction"] < 0.1*m["oof_dense_residual_mse_before_feature_correction"]
    assert abs(w[2])>0.5


def test_cfsr_runtime_fails_closed_on_bad_residual_schema_or_reservation_conflict():
    names=_names(); d=len(names); X=np.zeros((2,d)); mu=np.array([0.0,1.0]); cfg=_base_cfg(d)
    cfg["post_selection_cfsr_weights"]=[0.0]*(d-1)
    with pytest.raises(ValueError): _icer_post_selection_value(1,mu,X,names,cfg)
    cfg=_base_cfg(d); cfg["scene_reservation_enabled"]=True
    with pytest.raises(ValueError): _icer_post_selection_value(1,mu,X,names,cfg)


def test_dense_shift_runtime_uses_translation_not_affine_slope():
    names=_names(); d=len(names); X=np.zeros((2,d)); X[1,0]=1.0; X[1,1]=4.0; mu=np.array([0.0,1.0]); cfg=_base_cfg(d)
    cfg["post_selection_value_mode"]="dense_edge_shift"; cfg["post_selection_selected_bias"]=-0.4
    v,feat,fn=_icer_post_selection_value(1,mu,X,names,cfg)
    assert v==pytest.approx(1.6)  # dense=2, translation=-.4
    assert feat.tolist()==pytest.approx([2.0])
    assert fn==["post_value::dense_all_edge_absolute_value"]
