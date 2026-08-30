from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from bdse.planner.tournament import (
    _RAER_FEATURE_NAMES,
    _apply_decisive_frontier_raer,
    _decisive_frontier_raer_features,
)
from bdse.tools.fit_v64_3_16_eaf_raer import _load, _fit, _predict, _auc, FEATURE_NAMES


def _cfg(enabled: bool, *, attr_weight: float = 0.0, bias: float = 0.0) -> dict:
    weights=[0.0]*len(_RAER_FEATURE_NAMES)
    weights[_RAER_FEATURE_NAMES.index('attribution_scale')]=attr_weight
    return {
        'runtime': {
            'decisive_frontier_value': {
                'reliability_aware_extremal_reranking': {
                    'enabled': enabled,
                    'instrument_features': True,
                    'feature_names': list(_RAER_FEATURE_NAMES),
                    'feature_mean': [0.0]*len(_RAER_FEATURE_NAMES),
                    'feature_std': [1.0]*len(_RAER_FEATURE_NAMES),
                    'weights': weights,
                    'bias': bias,
                    'min_probability': 0.5,
                    'ratio_floor': 1e-3,
                    'valid_action_normalizer': 32.0,
                    'require_positive_raw_margin': True,
                }
            }
        }
    }


def _matrix() -> np.ndarray:
    # M[b, a] > 0 means challenger b is preferred to anchor a.
    M=np.zeros((3,3),dtype=np.float32)
    M[1,0]=0.30; M[0,1]=-0.30
    M[2,0]=0.50; M[0,2]=-0.50
    return M


def test_raer_moves_reliability_before_extremal_argmax_and_can_recover_runner_up() -> None:
    M=_matrix(); attr=np.asarray([0.0,0.20,0.01],dtype=np.float32)
    diag={'decisive_frontier_value_active':1.0,'decisive_frontier_value_residual_rms':0.2,
          'decisive_frontier_value_residual_abs_mean':0.15,'decisive_frontier_value_attribution_scale_rms':0.12,
          'decisive_frontier_value_attribution_scale_mean':0.10}
    selected,d=_apply_decisive_frontier_raer(2,0,M,attr,np.ones(3,bool),np.zeros(3,bool),diag,1.0,_cfg(True,attr_weight=50.0,bias=-2.0))
    assert selected==1
    assert d['decisive_frontier_raer_proposal_changed']==1.0
    assert d['decisive_frontier_raer_selected_probability']>=0.5


def test_raer_disabled_is_exact_raw_action_noop_but_still_instruments_all_edges() -> None:
    M=_matrix(); attr=np.asarray([0.0,0.2,0.01],dtype=np.float32)
    diag={'decisive_frontier_value_active':1.0}
    selected,d=_apply_decisive_frontier_raer(2,0,M,attr,np.ones(3,bool),np.zeros(3,bool),diag,1.0,_cfg(False))
    assert selected==2
    np.testing.assert_allclose(d['_decisive_frontier_raer_raw_margin_star'],M[:,0])
    assert d['_decisive_frontier_raer_feature_matrix'].shape==(3,len(_RAER_FEATURE_NAMES))


def test_raer_is_noop_when_frontier_inactive() -> None:
    selected,d=_apply_decisive_frontier_raer(2,0,_matrix(),None,np.ones(3,bool),np.zeros(3,bool),{'decisive_frontier_value_active':0.0},1.0,_cfg(True,attr_weight=-100.0))
    assert selected==2
    assert d['decisive_frontier_raer_active']==0.0


def test_raer_feature_schema_is_runtime_only_and_finite() -> None:
    mat,names=_decisive_frontier_raer_features(_matrix()[:,0],np.asarray([0,.2,.01]),np.ones(3,bool),0,
        {'decisive_frontier_value_residual_rms':.2,'decisive_frontier_value_residual_abs_mean':.1,
         'decisive_frontier_value_attribution_scale_rms':.12,'decisive_frontier_value_attribution_scale_mean':.1},1.0,_cfg(False))
    assert names==FEATURE_NAMES
    assert np.all(np.isfinite(mat))


def test_raer_fitter_learns_separable_all_frontier_edges(tmp_path: Path) -> None:
    rows=[]
    for s in range(180):
        for b in (1,2,3):
            pos=((s+b)%2)==0
            feat={n:0.0 for n in FEATURE_NAMES}
            feat['raw_margin']=0.4 if b==3 else 0.2
            feat['attribution_scale']=0.2 if pos else 0.01
            feat['frontier_attribution_scale_rms']=0.1
            feat['attribution_over_frontier_rms']=feat['attribution_scale']/0.1
            rows.append({'scenario_token':f's{s}','challenger_action':b,'teacher_margin':0.5 if pos else -0.5,
                         'teacher_better':float(pos),'raw_margin':feat['raw_margin'],'is_raw_top':float(b==3),
                         **{f'feature_{k}':v for k,v in feat.items()}})
    p=tmp_path/'edges.jsonl'; p.write_text('\n'.join(json.dumps(r) for r in rows))
    X,y,_,_,_,_,_=_load(p)
    w,b,m,sd=_fit(X,y,steps=500,lr=.05,l2=1e-3,seed=16)
    prob=_predict(X,w,b,m,sd)
    assert X.shape[1]==len(FEATURE_NAMES)
    assert _auc(y,prob)>.99


def test_raer_never_resurrects_flagged_challenger_when_anchor_is_only_safe_action() -> None:
    M=_matrix(); attr=np.asarray([0.0,0.2,0.3],dtype=np.float32)
    diag={'decisive_frontier_value_active':1.0}
    flags=np.asarray([False,True,True])
    selected,d=_apply_decisive_frontier_raer(0,0,M,attr,np.ones(3,bool),flags,diag,1.0,_cfg(True,attr_weight=50.0,bias=-2.0))
    assert selected==0
