from __future__ import annotations

import copy
import numpy as np
import pytest

from bdse.planner.tournament import (
    _apply_decisive_frontier_icer,
    _icer_transition_feature_matrix,
    _ICER_TRANSITION_FEATURE_NAMES,
    _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES,
)
from bdse.tests.test_v64_3_19_eaf_icer import _cfg, _diag, _matrix
from bdse.tools.fit_v64_3_22_eaf_icer_tcr import _fit_weighted_logistic, _predict, _risk_metrics


def _traj() -> np.ndarray:
    # [x,y,yaw,v,t]; action 1 is a slow straight incumbent, action 3 is a lateral/progressive alternative.
    t=np.asarray([1.,2.,3.])
    out=np.zeros((4,3,5),dtype=np.float32)
    for k in range(4):
        out[k,:,0]=np.asarray([1.,2.,3.])*(1+.25*k)
        out[k,:,1]=0.0
        out[k,:,3]=1+.2*k
        out[k,:,4]=t
    out[3,:,1]=np.asarray([0.,.5,1.5]); out[3,:,2]=np.asarray([0.,.1,.2])
    return out


def _risk_names(mode: str) -> list[str]:
    n=[f"evidence::{x}" for x in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]
    if mode=="transition_conditioned": n += [f"transition::{x}" for x in _ICER_TRANSITION_FEATURE_NAMES]
    return n


def _tcr_cfg(*, rep_bias: float, ret_bias: float, mode: str="evidence_only", dominance: str="dual_equal_mean") -> dict:
    c=_cfg(support_bias=2.0,dominance_bias=2.0,raw_margin_weight=0.0,policy=dominance)
    ic=c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic["all_flagged_policy"]="preserve_legacy_for_structural_guard"
    ic["incumbent_retention_policy"]="preserve_admissible_incumbent"
    ic["regret_risk_enabled"]=True; ic["regret_risk_feature_mode"]=mode
    names=_risk_names(mode)
    for prefix,bias in [("replacement_regret_risk",rep_bias),("retention_regret_risk",ret_bias)]:
        ic[f"{prefix}_feature_names"]=names
        ic[f"{prefix}_feature_mean"]=[0.0]*len(names)
        ic[f"{prefix}_feature_std"]=[1.0]*len(names)
        ic[f"{prefix}_weights"]=[0.0]*len(names)
        ic[f"{prefix}_bias"]=bias
    return c


def _run(c: dict, safety: np.ndarray|None=None, trajectories: np.ndarray|None=None):
    return _apply_decisive_frontier_icer(
        2,0,_matrix(),np.asarray([0.,.10,.05,.08]),np.zeros((3,4),np.float32),np.asarray([0.,.20,.40,.30]),
        np.ones(4,bool),np.zeros(4,bool) if safety is None else safety,_diag(),1.0,None,c,
        candidate_trajectories=trajectories, maneuver_ids=np.asarray([0,1,0,4],dtype=np.int64),
    )


def test_transition_features_are_reference_conditioned_and_runtime_only() -> None:
    x,n=_icer_transition_feature_matrix(_traj(),np.asarray([0,1,0,4]),np.ones(4,bool),1)
    assert n==list(_ICER_TRANSITION_FEATURE_NAMES) and x.shape==(4,len(n))
    p={k:i for i,k in enumerate(n)}
    assert x[1,p["same_maneuver"]]==1.0 and x[3,p["same_maneuver"]]==0.0
    assert x[3,p["delta_terminal_y_norm"]]>0.0
    assert x[3,p["mean_path_separation_norm"]]>0.0
    assert abs(x[1,p["endpoint_separation_norm"]])<1e-8


def test_regret_risk_veto_blocks_binary_positive_alternative_without_tuning_threshold() -> None:
    blocked=_tcr_cfg(rep_bias=-1.0,ret_bias=1.0)
    s,d=_run(blocked); assert s==2 and d["decisive_frontier_icer_regret_risk_enabled"]==1.0
    allow=_tcr_cfg(rep_bias=1.0,ret_bias=1.0)
    s2,_=_run(allow); assert s2!=2


def test_regret_sensitive_retention_is_separate_from_generic_support() -> None:
    c=_tcr_cfg(rep_bias=-1.0,ret_bias=-1.0)
    s,d=_run(c); assert s==0
    assert float(d["decisive_frontier_icer_legacy_retention_regret_risk_logit"])<0.0


def test_transition_conditioned_head_requires_geometry_and_all_flagged_still_delegates() -> None:
    c=_tcr_cfg(rep_bias=1.0,ret_bias=1.0,mode="transition_conditioned")
    with pytest.raises(ValueError,match="candidate trajectory bank"):
        _run(c,trajectories=None)
    s,d=_run(c,safety=np.ones(4,bool),trajectories=None)
    assert s==2 and float(d["decisive_frontier_icer_structural_domain_delegated"])==1.0


def test_magnitude_weighted_logistic_zero_boundary_tracks_net_improvement() -> None:
    rng=np.random.default_rng(322)
    X=rng.normal(size=(800,3)); delta=2.0*X[:,0]-0.3*X[:,1]+0.1*rng.normal(size=800)
    fit=np.ones(800,bool); w,b,m,s=_fit_weighted_logistic(X,delta,fit); score=_predict(X,w,b,m,s)
    met=_risk_metrics(delta,score)
    assert met["auc_positive_teacher_improvement"]>.95
    assert met["teacher_improvement_sum_on_predicted_positive"]>0.0
    assert met["teacher_improvement_sum_on_predicted_negative"]<0.0


def test_v22_design_exclusion_adds_exact_v21_double_fresh() -> None:
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/"configs"
    old={x.strip() for x in (root/"v64_3_21_design_exclude_v64_3_20_screen_tokens.txt").read_text().splitlines() if x.strip()}
    new={x.strip() for x in (root/"v64_3_22_design_exclude_v64_3_21_screen_tokens.txt").read_text().splitlines() if x.strip()}
    assert len(old)==3700 and len(new)==4700 and old<=new and len(new-old)==1000


def test_signed_profile_is_ranking_only_not_binary_eligibility_gate() -> None:
    c=_tcr_cfg(rep_bias=1.0,ret_bias=1.0,dominance="scalar_positive_dual_equal_mean")
    ic=c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    # Scalar view supports replacement, signed-profile view opposes it strongly.
    # V22 keeps scalar eligibility fixed; the profile view may affect rank only.
    ic["scalar_dominance_bias"]=2.0
    ic["profile_dominance_bias"]=-4.0
    s,d=_run(c)
    assert s != 2
    assert float(d["decisive_frontier_icer_dominance_policy_scalar_positive_dual_equal_mean"]) == 1.0
