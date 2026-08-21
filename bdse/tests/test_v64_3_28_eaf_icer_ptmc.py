from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

import bdse.tools.fit_v64_3_28_eaf_icer_ptmc as ptmc
from bdse.planner import tournament as tour


def test_tail_mode_constants_are_frozen_pre_fresh():
    assert ptmc.CATASTROPHIC_DELTA_THRESHOLD == -0.5
    assert ptmc.POSITIVE_PROPOSAL_COVERAGE == 0.95
    assert tuple(ptmc.KS) == (32, 64)


def test_higher_quantile_is_deterministic_order_statistic():
    x=np.asarray([4.0,1.0,3.0,2.0])
    assert ptmc._higher_quantile(x,0.50)==2.0
    assert ptmc._higher_quantile(x,0.95)==4.0


def test_global_tail_model_separates_synthetic_catastrophic_mode():
    rng=np.random.default_rng(7)
    benign=rng.normal(+1.0,0.1,size=(160,4))
    cat=rng.normal(-1.0,0.1,size=(40,4))
    X=np.vstack([benign,cat])
    y=np.r_[np.ones(len(benign)), -np.ones(len(cat))]
    m=ptmc._fit_diag_tail_model(X,y)
    risk=ptmc._tail_risk(m,np.vstack([np.full((1,4),-1.0),np.full((1,4),+1.0)]))
    assert risk[0] > risk[1]
    assert m['catastrophic_count']==40


def test_runtime_global_tail_score_matches_offline_and_sha(tmp_path: Path):
    rng=np.random.default_rng(3)
    benign=rng.normal(+0.7,0.2,size=(160,3)); cat=rng.normal(-0.7,0.2,size=(40,3))
    X=np.vstack([benign,cat]); y=np.r_[np.ones(160),-np.ones(40)]
    m=ptmc._fit_diag_tail_model(X,y)
    names=['semantic_type::a','semantic_type::b','semantic_type::c']
    p=tmp_path/'tail.npz'
    np.savez_compressed(p,
        feature_mean=m['feature_mean'], feature_std=m['feature_std'], feature_names=np.asarray(names),
        catastrophic_mean=m['catastrophic_mean'], catastrophic_var=m['catastrophic_var'],
        benign_mean=m['benign_mean'], benign_var=m['benign_var'], risk_threshold=np.asarray([0.3]),
        catastrophic_delta_threshold=np.asarray([-0.5]), positive_proposal_coverage=np.asarray([0.95]),
    )
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    q=np.asarray([[-0.8,-0.8,-0.8],[0.8,0.8,0.8]])
    got=tour._icer_global_tail_mode_confirmation_score(q,names,str(p),sha)
    want=0.3-ptmc._tail_risk(m,q)
    assert np.allclose(got,want)
    assert got[0] < got[1]  # catastrophic-looking point is more likely vetoed.


def test_v28_main_config_freezes_proposal_and_uses_global_type_confirmation(tmp_path: Path):
    base=yaml.safe_load(Path('bdse/configs/v64_3_20_icer_dc_dual.yaml').read_text(encoding='utf-8'))
    am={'path':str(tmp_path/'agg.npz'),'sha256':'a'*64}
    tm={'path':str(tmp_path/'tail.npz'),'sha256':'b'*64}
    cfg=ptmc._cfg_main(base,am,tm)
    ic=cfg['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']
    assert ic['regret_risk_feature_mode']=='evidence_only'
    assert ic['replacement_local_regret_neighbor_k_values']==[32,64]
    assert ic['regret_risk_model_type']=='local_multiscale_downside_regret_with_global_type_tail_confirmation'
    assert ic['replacement_confirmation_regret_risk_feature_mode']=='semantic_type_only'
    assert ic['replacement_confirmation_tail_mode_label_threshold']==-0.5
    assert ic['replacement_confirmation_positive_proposal_coverage']==0.95
    assert 'NO fallback/reselection' in ic['replacement_operator']
    assert 'subset_of' in ic['replacement_selection_monotonicity']


def test_no_fallback_helper_still_cannot_reselect_second_candidate():
    cand=np.asarray([2,3],dtype=np.int64)
    dominance=np.asarray([0.0,0.0,2.0,1.0])
    aggregate=np.asarray([0.0,0.0,0.4,0.8])
    support=np.asarray([0.0,0.0,1.0,1.0])
    margin=np.asarray([0.0,0.0,0.3,0.2])
    utility=np.asarray([0,0,1,1])
    confirmation=np.asarray([np.nan,np.nan,-0.1,+10.0])
    assert tour._icer_select_extremal_candidate_with_optional_confirmation(
        cand,dominance,aggregate,support,margin,utility,confirmation_logits=confirmation
    ) is None


def test_launcher_keeps_v27_regression_and_6700_exclusion():
    s=Path('RUN_V64_3_28_EAF_ICER_PTMC_SCREEN_2GPU.sh').read_text(encoding='utf-8')
    assert 'test_v64_3_28_eaf_icer_ptmc.py' in s
    assert 'test_v64_3_27_eaf_icer_trcc.py' in s
    assert 'len(ex)!=6700' in s
    assert 'v64.3.28-eaf-icer-ptmc-double-fresh-v1' in s
    assert 'V28_TRAIN_EDGES' in s
