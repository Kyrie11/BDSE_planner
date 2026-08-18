from __future__ import annotations

import copy, json
from pathlib import Path
import numpy as np

from bdse.planner.tournament import _apply_decisive_frontier_icer, _ICER_DOMINANCE_PROFILE_BASE_NAMES
from bdse.tests.test_v64_3_19_eaf_icer import _cfg, _diag, _matrix
from bdse.tools.fit_v64_3_21_eaf_icer_mcr import _fit_ridge, _predict
from bdse.tools.check_v64_3_19_eaf_icer_screen import _icer_edge_diag


def _run(cfg: dict, safety: np.ndarray|None=None):
    return _apply_decisive_frontier_icer(2,0,_matrix(),np.asarray([0.,.10,.05,.08]),np.zeros((3,4),np.float32),np.asarray([0.,.20,.40,.30]),np.ones(4,bool),np.zeros(4,bool) if safety is None else safety,_diag(),1.0,None,cfg)


def _mcr_cfg(*, retention_bias: float, consensus: bool=False) -> dict:
    c=_cfg(support_bias=1.0,dominance_bias=1.0,raw_margin_weight=0.0,policy="dual_equal_mean")
    ic=c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic["all_flagged_policy"]="preserve_legacy_for_structural_guard"
    ic["incumbent_retention_policy"]="selected_incumbent_profile_margin_mse"
    names=list(_ICER_DOMINANCE_PROFILE_BASE_NAMES)
    ic["retention_feature_names"]=names; ic["retention_feature_mean"]=[0.0]*len(names); ic["retention_feature_std"]=[1.0]*len(names); ic["retention_weights"]=[0.0]*len(names); ic["retention_bias"]=retention_bias
    if consensus: ic["dominance_policy"]="dual_positive_consensus_mean"
    return c


def test_mcr_retention_margin_not_generic_support_controls_admissible_incumbent_fallback() -> None:
    keep=_mcr_cfg(retention_bias=1.0); ic=keep["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    # No positive dominance => baseline is the incumbent when retention margin is positive.
    ic["scalar_dominance_bias"]=-10.0; ic["profile_dominance_bias"]=-10.0
    s,d=_run(keep); assert s==2; assert float(d["decisive_frontier_icer_legacy_retention_margin"])>0
    drop=copy.deepcopy(keep); di=drop["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]; di["retention_bias"]=-1.0
    s2,d2=_run(drop); assert s2==0; assert float(d2["decisive_frontier_icer_legacy_retention_margin"])<0


def test_mcr_consensus_requires_both_dominance_views_positive() -> None:
    c=_mcr_cfg(retention_bias=1.0,consensus=True); ic=c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    # Scalar is positive, signed-profile is negative: mean could be positive but consensus must reject.
    ic["scalar_dominance_bias"]=3.0; ic["profile_dominance_bias"]=-1.0
    s,_=_run(c); assert s==2
    ic["profile_dominance_bias"]=1.0
    s2,_=_run(c); assert s2!=2


def test_mcr_all_flagged_still_delegates_exact_legacy() -> None:
    c=_mcr_cfg(retention_bias=-10.0,consensus=True)
    s,d=_run(c,safety=np.ones(4,bool)); assert s==2; assert float(d["decisive_frontier_icer_structural_domain_delegated"])==1.0


def test_retention_ridge_preserves_semantic_zero_sign() -> None:
    rng=np.random.default_rng(21); X=rng.normal(size=(500,4)); y=2.0*X[:,0]-0.5*X[:,1]
    mask=np.ones(len(y),bool); w,b,m,s,scale=_fit_ridge(X,y,mask); pred=_predict(X,w,b,m,s)
    assert scale>0 and np.mean((pred>=0)==(y>=0))>.95


def test_edge_diag_can_exclude_delegated_domain(tmp_path: Path) -> None:
    rows=[
      {"scenario_token":"safe","anchor_action":0,"raw_top_action":1,"challenger_action":1,"teacher_margin":.2,"icer_admissible":1,"icer_selected_action":1,"icer_support_logit":1,"icer_dominance_logit":0,"icer_scalar_dominance_logit":0,"icer_profile_dominance_logit":0},
      {"scenario_token":"flagged","anchor_action":0,"raw_top_action":1,"challenger_action":1,"teacher_margin":-.2,"icer_admissible":0,"icer_selected_action":1,"icer_support_logit":0,"icer_dominance_logit":0,"icer_scalar_dominance_logit":0,"icer_profile_dominance_logit":0},
    ]
    p=tmp_path/"e.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    d=_icer_edge_diag(p,{"safe"}); assert d["scene_count"]==1.0 and d["selected_nonanchor_teacher_better_rate"]==1.0


def test_v21_design_exclusion_adds_exact_v20_fresh_block() -> None:
    root=Path(__file__).resolve().parents[1]/"configs"
    old={x.strip() for x in (root/"v64_3_20_design_exclude_v64_3_19_screen_tokens.txt").read_text().splitlines() if x.strip()}
    new={x.strip() for x in (root/"v64_3_21_design_exclude_v64_3_20_screen_tokens.txt").read_text().splitlines() if x.strip()}
    assert len(old)==3200 and len(new)==3700 and old<=new and len(new-old)==500
