from __future__ import annotations

import numpy as np

from bdse.planner.tournament import (
    _DALER_FEATURE_NAMES,
    _apply_certificate_utility_refinement,
    _apply_decisive_frontier_daler,
    _decisive_frontier_daler_executable_mask,
    _decisive_frontier_daler_features,
)
from bdse.tools.fit_v64_3_17_eaf_daler import _fit, _logits


def _base_cfg(*, enabled: bool, attr_weight: float = 0.0, bias: float = 0.0, require_utility: bool = False) -> dict:
    weights=[0.0]*len(_DALER_FEATURE_NAMES)
    weights[_DALER_FEATURE_NAMES.index('attribution_scale')]=attr_weight
    return {
        'runtime': {
            'pair_action_anchor_guard': {'enabled': True, 'flip_margin': 0.015, 'score_margin': 0.0},
            'dual_certificate': {
                'enabled': True,
                'require_evidence_certificate_before_residual_flip': True,
                'min_evidence_certificate_fraction_for_residual_flip': 1.0,
            },
            'decisive_frontier_value': {
                'deployment_aligned_listwise_extremal_reliability': {
                    'enabled': enabled,
                    'instrument_features': True,
                    'feature_names': list(_DALER_FEATURE_NAMES),
                    'feature_mean': [0.0]*len(_DALER_FEATURE_NAMES),
                    'feature_std': [1.0]*len(_DALER_FEATURE_NAMES),
                    'weights': weights,
                    'bias': bias,
                    'anchor_logit': 0.0,
                    'ratio_floor': 1e-3,
                    'valid_action_normalizer': 32.0,
                    'require_guard_executable': True,
                    'require_utility_equivalence': require_utility,
                }
            },
        },
        'tournament': {'utility_refinement': {'enabled': False}},
    }


def _matrix() -> np.ndarray:
    M=np.zeros((3,3),dtype=np.float32)
    M[1,0]=0.30; M[0,1]=-0.30
    M[2,0]=0.50; M[0,2]=-0.50
    return M


def _diag() -> dict:
    return {
        'decisive_frontier_value_active':1.0,
        'decisive_frontier_value_residual_rms':0.2,
        'decisive_frontier_value_residual_abs_mean':0.15,
        'decisive_frontier_value_attribution_scale_rms':0.12,
        'decisive_frontier_value_attribution_scale_mean':0.10,
    }


def test_daler_listwise_extremal_operator_can_recover_reliable_runner_up() -> None:
    M=_matrix(); attr=np.asarray([0.0,0.20,0.01],dtype=np.float32)
    scores=np.asarray([0.0,0.25,0.50],dtype=np.float32)
    selected,d=_apply_decisive_frontier_daler(
        2,0,M,attr,scores,np.ones(3,bool),np.zeros(3,bool),_diag(),1.0,None,
        _base_cfg(enabled=True,attr_weight=50.0,bias=-2.0,require_utility=False),
    )
    assert selected==1
    assert d['decisive_frontier_daler_proposal_changed']==1.0
    assert d['decisive_frontier_daler_anchor_fallback']==0.0


def test_daler_anchor_is_explicit_fixed_logit_abstention_option() -> None:
    selected,d=_apply_decisive_frontier_daler(
        2,0,_matrix(),np.asarray([0.0,0.2,0.1]),np.asarray([0.0,0.2,0.5]),
        np.ones(3,bool),np.zeros(3,bool),_diag(),1.0,None,
        _base_cfg(enabled=True,attr_weight=0.0,bias=-3.0,require_utility=False),
    )
    assert selected==0
    assert d['decisive_frontier_daler_anchor_fallback']==1.0


def test_daler_cannot_select_margin_below_frozen_flip_guard() -> None:
    M=_matrix(); M[1,0]=0.01; M[0,1]=-0.01; M[2,0]=0.05; M[0,2]=-0.05
    attr=np.asarray([0.0,1.0,0.1]); scores=np.asarray([0.0,0.3,0.4])
    selected,d=_apply_decisive_frontier_daler(
        2,0,M,attr,scores,np.ones(3,bool),np.zeros(3,bool),_diag(),1.0,None,
        _base_cfg(enabled=True,attr_weight=20.0,bias=1.0,require_utility=False),
    )
    assert not bool(d['_decisive_frontier_daler_guard_mask'][1])
    assert selected==2


def test_daler_reuses_exact_utility_equivalence_mask() -> None:
    M=_matrix(); attr=np.asarray([0.0,1.0,0.1]); scores=np.asarray([0.0,0.3,0.5])
    utility_diag={
        '_utility_refinement_eligible_mask':np.asarray([False,False,True]),
        '_utility_refinement_cost':np.asarray([0.0,0.1,0.2],dtype=np.float32),
    }
    selected,d=_apply_decisive_frontier_daler(
        2,0,M,attr,scores,np.ones(3,bool),np.zeros(3,bool),_diag(),1.0,utility_diag,
        _base_cfg(enabled=True,attr_weight=20.0,bias=1.0,require_utility=True),
    )
    assert not bool(d['_decisive_frontier_daler_utility_equivalence_mask'][1])
    assert bool(d['_decisive_frontier_daler_executable_mask'][2])
    assert selected==2


def test_daler_evidence_certificate_fail_closes_to_anchor() -> None:
    cfg=_base_cfg(enabled=True,attr_weight=20.0,bias=1.0,require_utility=False)
    selected,d=_apply_decisive_frontier_daler(
        2,0,_matrix(),np.asarray([0.0,1.0,0.1]),np.asarray([0.0,0.3,0.5]),
        np.ones(3,bool),np.zeros(3,bool),_diag(),0.5,None,cfg,
    )
    assert int(np.asarray(d['_decisive_frontier_daler_executable_mask']).sum())==0
    assert selected==0


def test_daler_features_are_finite_runtime_only_statistics() -> None:
    cfg=_base_cfg(enabled=False,require_utility=False)
    M=_matrix(); margins=M[:,0]; scores=np.asarray([0.0,0.3,0.5]); valid=np.ones(3,bool)
    executable,_,_=_decisive_frontier_daler_executable_mask(margins,scores,valid,np.zeros(3,bool),0,1.0,None,cfg)
    mat,names=_decisive_frontier_daler_features(
        margins,np.asarray([0.0,0.2,0.01]),scores,valid,0,2,_diag(),1.0,None,executable,cfg
    )
    assert names==_DALER_FEATURE_NAMES
    assert mat.shape==(3,len(_DALER_FEATURE_NAMES))
    assert np.all(np.isfinite(mat))


def test_utility_refinement_refactor_preserves_legacy_selection() -> None:
    cfg={
        'candidate': {'step_s':0.1},
        'tournament': {'utility_refinement': {
            'enabled':True,'score_slack':0.4,'pair_certificate_enabled':True,
            'pair_margin_tolerance':0.04,'top_k':4,'require_unflagged':True,
            'min_utility_improvement':0.0,'progress_weight':1.0,'path_length_weight':0.0,
            'lateral_mean_weight':0.0,'lateral_final_weight':0.0,'comfort_weight':0.0,
            'curvature_weight':0.0,'speed_weight':0.0,'low_speed_threshold':-1.0,
            'low_speed_penalty':0.0,'unsafe_penalty':1000.0,
        }}
    }
    scores=np.asarray([1.0,0.9],dtype=np.float32); valid=np.ones(2,bool); flags=np.zeros(2,bool)
    # Candidate 1 has more progress, so the legacy refinement should switch to it.
    tr=np.zeros((2,3,3),dtype=np.float32); tr[0,:,0]=[0,0.1,0.2]; tr[1,:,0]=[0,0.5,1.0]
    M=np.asarray([[0.0,0.0],[0.0,0.0]],dtype=np.float32)
    chosen,d=_apply_certificate_utility_refinement(scores,0,valid,flags,cfg,candidate_trajectories=tr,margins=M)
    assert chosen==1
    assert bool(d['_utility_refinement_eligible_mask'][0]) and bool(d['_utility_refinement_eligible_mask'][1])


def test_listwise_fitter_learns_anchor_and_within_scene_ordering() -> None:
    X=[]; y=[]; tm=[]; tokens=[]
    rng=np.random.default_rng(17)
    for s in range(120):
        anchor_scene=(s%3)==0
        for b in range(2):
            # Feature 0 is reliability; feature 1 distinguishes the two challengers.
            if anchor_scene:
                x0=-2.0+0.05*rng.normal(); margin=-1.0-0.1*b
            else:
                x0=2.0+0.05*rng.normal(); margin=0.5+(0.5 if b==1 else 0.0)
            X.append([x0,float(b)]); y.append(float(margin>0)); tm.append(margin); tokens.append(f's{s}')
    X=np.asarray(X,float); y=np.asarray(y,float); tm=np.asarray(tm,float)
    w,b,mean,std=_fit(X,y,tm,tokens,steps=350,lr=.05,l2=1e-3,aux_edge_bce_weight=1.0,seed=17)
    logit=_logits(X,w,b,mean,std)
    correct=[]
    for s in range(120):
        idx=np.asarray([2*s,2*s+1]); best=int(idx[np.argmax(logit[idx])]); sel=best if logit[best]>0 else -1
        target=-1 if s%3==0 else 2*s+1
        correct.append(sel==target)
    assert float(np.mean(correct))>.95


def test_daler_abstains_on_all_flagged_bank_and_leaves_structural_guard_frozen() -> None:
    cfg=_base_cfg(enabled=True,attr_weight=20.0,bias=2.0,require_utility=False)
    cfg['runtime']['decisive_frontier_value']['deployment_aligned_listwise_extremal_reliability'][
        'require_safe_available_for_learned_intervention'
    ]=True
    selected,d=_apply_decisive_frontier_daler(
        2,0,_matrix(),np.asarray([0.0,1.0,0.5]),np.asarray([0.0,0.3,0.5]),
        np.ones(3,bool),np.ones(3,bool),_diag(),1.0,None,cfg,
    )
    assert int(np.asarray(d['_decisive_frontier_daler_executable_mask']).sum())==0
    assert selected==0
