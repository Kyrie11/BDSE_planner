from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import yaml

from bdse.planner.tournament import run_pair_conditioned_tournament
from bdse.tools.fit_v64_3_15_eaf_eair import _load_rows, _fit, _predict, _auc, FEATURE_NAMES


def _cfg(enabled: bool, weight: float) -> dict:
    return {
        "runtime": {
            "pair_tournament_anchor_mode": "selected_local",
            "pair_tournament_pair_delta_includes_local": True,
            "pair_tournament_aggregation_mode": "decisive_anchor_margin",
            "pair_action_anchor_guard": {"enabled": True, "flip_margin": 0.01, "score_margin": 0.0},
            "dual_certificate": {"enabled": False},
            "decisive_frontier_value": {
                "enabled": True,
                "scale": 1.0,
                "one_sided_intervention": {"enabled": False},
                "learned_intervention_reliability": {
                    "enabled": enabled,
                    "instrument_features": True,
                    "model_type": "standardized_logistic_teacher_better_edge",
                    "feature_names": ["raw_margin"],
                    "feature_mean": [0.0],
                    "feature_std": [1.0],
                    "weights": [weight],
                    "bias": 0.0,
                    "min_probability": 0.5,
                    "ratio_floor": 0.001,
                    "valid_action_normalizer": 32.0,
                    "require_frontier_active": True,
                },
            },
        },
        "model": {"pair_margin_normalized": False},
        "tournament": {"epsilon_cal": 0.0, "beta_uncertainty": 0.0, "use_softmin": True, "softmin_tau": 1.0},
        "selector": {"pair_screen_top_l": 3, "pair_screen_near_eta": 10.0},
    }


def _run(cfg: dict, *, frontier: bool = True):
    J0=np.array([0.0,2.0,5.0],np.float32); g=np.zeros((1,3),np.float32)
    pairs=np.array([[0,1]],np.int64); delta=np.zeros((1,1),np.float32)
    valid=np.ones(3,bool); safety=np.zeros(3,bool)
    kwargs={}
    if frontier:
        kwargs.update(
            frontier_value_atom_factors=np.ones((1,1),np.float32)*3.0,
            frontier_value_action_signed_factors=np.array([[0.0],[0.0],[-10.0]],np.float32),
            frontier_value_action_context_factors=np.ones((3,1),np.float32),
        )
    return run_pair_conditioned_tournament(J0,delta,pairs,[0],valid,safety,cfg,predicted_atom_costs=g,**kwargs)


def test_eair_can_selectively_block_or_allow_same_frozen_eaf_value() -> None:
    raw=_run(_cfg(False,0.0))
    allow=_run(_cfg(True,100.0))
    block=_run(_cfg(True,-100.0))
    assert raw.action_index==allow.action_index==2
    assert block.action_index==0
    np.testing.assert_allclose(raw.margins,allow.margins,rtol=0,atol=0)
    np.testing.assert_allclose(raw.margins,block.margins,rtol=0,atol=0)
    assert allow.diagnostics['decisive_frontier_eair_pass']==1.0
    assert block.diagnostics['decisive_frontier_eair_pass']==0.0


def test_eair_is_noop_when_eaf_absent_from_pairfull_ceiling() -> None:
    raw=_run(_cfg(False,0.0),frontier=False)
    gated=_run(_cfg(True,-100.0),frontier=False)
    assert raw.action_index==gated.action_index
    np.testing.assert_allclose(raw.margins,gated.margins,rtol=0,atol=0)
    assert gated.diagnostics['decisive_frontier_eair_active']==0.0


def test_eair_instruments_runtime_feature_contract_when_disabled() -> None:
    out=_run(_cfg(False,0.0))
    d=out.diagnostics
    assert 'decisive_frontier_eair_feature_raw_margin' in d
    assert 'decisive_frontier_eair_feature_proposed_attribution_scale' in d
    assert np.isfinite(d['decisive_frontier_eair_feature_margin_over_attribution'])


def test_eair_linear_fitter_learns_separable_teacher_better_edges(tmp_path: Path) -> None:
    rows=[]
    for i in range(300):
        pos=(i%2)==0
        raw=0.25 if pos else 0.03
        attr=0.08 if pos else 0.01
        rows.append({
            'scenario_token':f's{i}',
            'raw_frontier_anchor_action':0,
            'raw_frontier_proposed_action':1,
            'pair_action_anchor_raw_margin':raw,
            'decisive_frontier_ocfi_proposed_attribution_scale':attr,
            'decisive_frontier_value_residual_rms':attr*3,
            'decisive_frontier_value_residual_abs_mean':attr*2,
            'decisive_frontier_value_attribution_scale_rms':attr,
            'decisive_frontier_value_attribution_scale_mean':attr*0.8,
            'evidence_certificate_fraction':1.0,
            'valid_action_count':28,
            'decisive_frontier_value_teacher_proposed_vs_anchor_margin':0.5 if pos else -0.5,
        })
    p=tmp_path/'rows.jsonl'; p.write_text('\n'.join(json.dumps(r) for r in rows))
    X,y,_,_=_load_rows(p)
    assert X.shape==(300,len(FEATURE_NAMES))
    w,b,m,s=_fit(X,y,steps=500,lr=0.05,l2=1e-3,seed=1)
    prob=_predict(X,w,b,m,s)
    assert _auc(y,prob)>0.99
