from __future__ import annotations

import numpy as np
import pytest

from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.tools.fit_v64_3_40_eaf_icer_sdfr import _fit_selected_distribution, _hurdle_value


def _names():
    return [f"delta::{n}" for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES] + ["delta::support_logit"]


def _base_cfg(d: int, mode: str):
    rw=np.zeros(d); rw[0]=1.0
    cfg={"post_selection_value_enabled":True,"post_selection_value_mode":mode,"scene_reservation_enabled":False,
         "feature_names":_names(),"feature_mean":[0.0]*d,"feature_std":[1.0]*d,"weights":rw.tolist(),"bias":0.0,
         "post_selection_hurdle_probability_clip":1e-4,"post_selection_value_max_abs":40.0}
    # sign p = x1 + .5; positive magnitude = x2; negative magnitude = x3
    specs={"sign":(1,1.0,0.5),"positive_magnitude":(2,1.0,0.0),"negative_magnitude":(3,1.0,0.0)}
    for name,(j,w,b) in specs.items():
        ww=np.zeros(d); ww[j]=w
        cfg[f"post_selection_hurdle_{name}_feature_mean"]=[0.0]*d
        cfg[f"post_selection_hurdle_{name}_feature_std"]=[1.0]*d
        cfg[f"post_selection_hurdle_{name}_weights"]=ww.tolist()
        cfg[f"post_selection_hurdle_{name}_bias"]=b
    return cfg


def test_hurdle_identity_reconstructs_signed_expected_value():
    assert _hurdle_value(0.75,2.0,1.0)==pytest.approx(1.25)
    assert _hurdle_value(0.25,2.0,1.0)==pytest.approx(-0.25)


def test_selected_distribution_fit_recovers_sign_and_magnitude_shifts_without_high_dimensional_head():
    rows=[]
    # balanced enough for the preregistered per-sign minimum. Base p=.5. Selected policy is 2:1 positive.
    for i in range(240):
        pos=(i%3)!=0
        rows.append({"p":0.5,"mp":1.0,"mn":1.0,"y":2.0 if pos else -3.0})
    m=_fit_selected_distribution(rows)
    assert m["sample_count"]==240 and m["positive_count"]==160 and m["nonpositive_count"]==80
    assert m["selected_logit_shift"]==pytest.approx(np.log(2.0),abs=1e-8)
    assert m["selected_positive_magnitude_scale"]==pytest.approx(2.0)
    assert m["selected_negative_magnitude_scale"]==pytest.approx(3.0)


def test_v40_runtime_distribution_modes_only_value_the_frozen_proposal():
    names=_names(); d=len(names); X=np.zeros((3,d)); X[1,0]=2.0; X[1,1]=0.25; X[1,2]=2.0; X[1,3]=1.0
    mu=np.array([0.0,2.0,1.0])
    cfg=_base_cfg(d,"dense_edge_hurdle")
    v,feat,fn=_icer_post_selection_value(1,mu,X,names,cfg)
    # p=.75, m+=2, m-=1 => 1.25
    assert v==pytest.approx(1.25)
    assert feat.tolist()==pytest.approx([0.75,2.0,1.0])
    assert fn==["post_value::selected_positive_probability","post_value::selected_positive_magnitude","post_value::selected_negative_magnitude"]


def test_v40_selected_component_adaptation_is_scalar_and_translation_preserves_final_order():
    names=_names(); d=len(names); X=np.zeros((2,d)); X[1,0]=1.0; X[1,1]=0.0; X[1,2]=2.0; X[1,3]=1.0
    mu=np.array([0.0,1.0]); cfg=_base_cfg(d,"dense_edge_hurdle_selected")
    cfg.update({"post_selection_hurdle_selected_logit_shift":0.0,"post_selection_hurdle_selected_positive_magnitude_scale":2.0,"post_selection_hurdle_selected_negative_magnitude_scale":0.5})
    v,_,_=_icer_post_selection_value(1,mu,X,names,cfg)
    # p=.5, scaled m+=4, m-=.5 => 1.75
    assert v==pytest.approx(1.75)
    cfg["post_selection_value_mode"]="dense_edge_hurdle_selected_shift"; cfg["post_selection_selected_bias"]=-0.25
    v2,_,_=_icer_post_selection_value(1,mu,X,names,cfg)
    assert v2==pytest.approx(1.5)


def test_v40_sign_shift_ablation_changes_only_positive_probability_component():
    names=_names(); d=len(names); X=np.zeros((2,d)); X[1,0]=1.0; X[1,1]=0.0; X[1,2]=2.0; X[1,3]=1.0
    mu=np.array([0.0,1.0]); cfg=_base_cfg(d,"dense_edge_hurdle_sign_shift")
    cfg.update({"post_selection_hurdle_selected_logit_shift":np.log(3.0),"post_selection_hurdle_selected_positive_magnitude_scale":9.0,"post_selection_hurdle_selected_negative_magnitude_scale":9.0})
    v,feat,_=_icer_post_selection_value(1,mu,X,names,cfg)
    # p=.5 -> .75 after odds*3. Sign-only mode must ignore stored magnitude scales.
    assert feat.tolist()==pytest.approx([0.75,2.0,1.0])
    assert v==pytest.approx(1.25)


def test_v40_runtime_fails_closed_on_missing_distribution_parameters():
    names=_names(); d=len(names); X=np.zeros((2,d)); X[1,0]=1.0; mu=np.array([0.0,1.0]); cfg=_base_cfg(d,"dense_edge_hurdle_selected")
    with pytest.raises(ValueError):
        _icer_post_selection_value(1,mu,X,names,cfg)
