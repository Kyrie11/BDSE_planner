from __future__ import annotations

import numpy as np
import pytest

from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import (
    FEATURE_NAMES,
    RIDGE_LAMBDA,
    _fit_regret_structured_margin,
    _scene_equal_zero_rms_scale,
    _scene_margin_blocks,
    _structured_objective_numpy,
    _structured_scores,
)
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import _select


def _alt(token: str, action: int, x0: float, y: float, support: float = 1.0):
    x=np.zeros((len(FEATURE_NAMES),),dtype=np.float64); x[0]=x0
    return {'token':token,'action':action,'x':x,'y':y,'support':support,'margin':0.0,'utility_prior':0}


def test_rsmr_incumbent_augmented_margin_learns_noop_and_opportunity_choice():
    scene_map={
        'noop':[_alt('noop',1,1.0,-2.0),_alt('noop',2,2.0,-0.2)],
        'opp':[_alt('opp',1,-1.0,0.4),_alt('opp',2,-2.0,2.0),_alt('opp',3,1.0,-1.0)],
    }
    model=_fit_regret_structured_margin(scene_map,list(scene_map))
    ns=_structured_scores(scene_map['noop'],model)
    assert float(ns.max()) < 0.0
    os=_structured_scores(scene_map['opp'],model)
    oi=_select(scene_map['opp'],os)
    assert oi == 1
    assert float(os[oi]) > 0.0


def test_rsmr_scene_max_loss_is_invariant_to_duplicate_easy_rivals():
    base={'s':[_alt('s',1,1.0,2.0),_alt('s',2,-1.0,-3.0)]}
    dup={'s':[_alt('s',1,1.0,2.0),_alt('s',2,-1.0,-3.0),_alt('s',3,-1.0,-3.0)]}
    # Freeze a common external scale so this test isolates loss aggregation.
    scale=np.ones((len(FEATURE_NAMES),),dtype=np.float64)
    b1=_scene_margin_blocks(base,['s'],scale)
    b2=_scene_margin_blocks(dup,['s'],scale)
    w=np.zeros((len(FEATURE_NAMES),),dtype=np.float64); w[0]=0.5
    assert _structured_objective_numpy(w,b1) == pytest.approx(_structured_objective_numpy(w,b2))


def test_rsmr_structured_hinge_upper_bounds_selected_teacher_regret_gap():
    scene={'s':[_alt('s',1,1.0,2.0),_alt('s',2,3.0,0.5),_alt('s',3,-2.0,-1.0)]}
    scale=np.ones((len(FEATURE_NAMES),),dtype=np.float64)
    blocks=_scene_margin_blocks(scene,['s'],scale)
    # Choose w so runtime argmax is action 2 although teacher-best is action 1.
    w=np.zeros((len(FEATURE_NAMES),),dtype=np.float64); w[0]=1.0
    _,D,g=blocks[0]
    root_loss=max(float(np.max(g-D@w)),0.0)
    scores=np.array([a['x'][0] for a in scene['s']],dtype=float)
    pred=int(np.argmax(scores)); teacher=int(np.argmax([a['y'] for a in scene['s']]))
    regret=float(scene['s'][teacher]['y']-scene['s'][pred]['y'])
    assert root_loss + 1e-12 >= regret


def test_rsmr_scale_is_zero_preserving_scene_equal_candidate_rms():
    scene_map={'a':[_alt('a',1,1.0,1.0),_alt('a',2,3.0,-1.0)],'b':[_alt('b',1,2.0,1.0)]}
    scale=_scene_equal_zero_rms_scale(scene_map,['a','b'])
    # Scene a total moment mass 1 => .5*(1^2+3^2); scene b total mass 1 => 2^2.
    # Moment normalization across two scenes gives average of those scene moments.
    expected=np.sqrt((0.5*(1.0+9.0)+4.0)/2.0)
    assert scale[0] == pytest.approx(expected)
    assert np.all(scale[1:] >= 1e-6)


def test_rsmr_optimizer_is_deterministic_and_decreases_convex_objective():
    scene_map={
        'a':[_alt('a',1,1.0,1.0),_alt('a',2,-1.0,-1.0)],
        'b':[_alt('b',1,2.0,-0.2),_alt('b',2,1.0,-0.5)],
        'c':[_alt('c',1,-2.0,2.0),_alt('c',2,1.0,0.1)],
    }
    m1=_fit_regret_structured_margin(scene_map,list(scene_map)); m2=_fit_regret_structured_margin(scene_map,list(scene_map))
    assert np.allclose(m1[0],m2[0],atol=1e-10,rtol=1e-10)
    assert np.allclose(m1[1],m2[1],atol=0,rtol=0)
    assert m1[2]['objective_final'] <= m1[2]['objective_at_zero'] + 1e-10
    assert RIDGE_LAMBDA == 1.0


def test_rsmr_reuses_monotone_rank_main_runtime_without_fallback():
    from bdse.tests.test_v64_3_33_eaf_icer_spcr import _runtime_cfg, _run
    rank_action, rank=_run(_runtime_cfg(mode='rank_only'))
    main_action, main=_run(_runtime_cfg(mode='conformal_veto',q=10.0))
    assert int(main['decisive_frontier_icer_scir_proposal_action']) == int(rank['decisive_frontier_icer_scir_proposal_action'])
    assert rank_action == int(rank['decisive_frontier_icer_scir_proposal_action'])
    assert float(main['decisive_frontier_icer_scir_certificate_accepted']) == 0.0
    assert main_action == 2


def test_rsmr_selected_policy_calibrator_uses_one_frozen_proposal_per_scene(tmp_path):
    import json, subprocess, sys, yaml
    from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES
    rows=tmp_path/'rows.jsonl'; edges=tmp_path/'edges.jsonl'; rank_cfg=tmp_path/'rank.yaml'; main_cfg=tmp_path/'main.yaml'; report=tmp_path/'report.json'
    with rows.open('w') as f:
        for i in range(500): f.write(json.dumps({'scenario_token':f't{i:03d}'})+'\n')
    with edges.open('w') as f:
        for i in range(100):
            t=f't{i:03d}'
            f.write(json.dumps({'scenario_token':t,'raw_top_action':0,'challenger_action':0,'icer_admissible':1.0,'icer_support_logit':1.0,'teacher_margin':0.0,'icer_scir_predicted_improvement':0.0})+'\n')
            f.write(json.dumps({'scenario_token':t,'raw_top_action':0,'challenger_action':1,'icer_admissible':1.0,'icer_support_logit':1.0,'teacher_margin':1.0,'icer_scir_predicted_improvement':1.4,'raw_margin':0.1,'dacer_utility_prior':0})+'\n')
            f.write(json.dumps({'scenario_token':t,'raw_top_action':0,'challenger_action':2,'icer_admissible':1.0,'icer_support_logit':1.0,'teacher_margin':0.2,'icer_scir_predicted_improvement':0.3,'raw_margin':0.1,'dacer_utility_prior':0})+'\n')
    cfg={'runtime':{'decisive_frontier_value':{'incumbent_contrastive_extremal_recovery':{'selection_conditioned_intervention_recovery':{
        'enabled':True,'mode':'rank_only','no_fallback':True,'base_feature_names':list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES),'feature_names':FEATURE_NAMES,
        'feature_mean':[0.0]*len(FEATURE_NAMES),'feature_std':[1.0]*len(FEATURE_NAMES),'weights':[0.0]*len(FEATURE_NAMES),'bias':0.0,
    }}}},'metadata':{},'provenance':{},'experiment':{}}
    rank_cfg.write_text(yaml.safe_dump(cfg,sort_keys=False))
    cp=subprocess.run([sys.executable,'-m','bdse.tools.calibrate_v64_3_34_eaf_icer_rsmr','--calibration-rows',str(rows),'--calibration-edges',str(edges),'--rank-config',str(rank_cfg),'--output-main-config',str(main_cfg),'--output-report',str(report),'--alpha','0.05'],check=True,text=True,capture_output=True)
    assert '"pass": true' in cp.stdout.lower()
    rep=json.loads(report.read_text()); assert rep['selected_policy_proposal_count']==100; assert rep['selected_policy_conformal_quantile']==pytest.approx(0.4)
    main=yaml.safe_load(main_cfg.read_text()); sc=main['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['selection_conditioned_intervention_recovery']
    assert sc['mode']=='conformal_veto'; assert sc['conformal_overprediction_quantile']==pytest.approx(0.4)


def test_rsmr_contract_checker_requires_exact_pair_ablation_and_same_rank_selector(tmp_path):
    import json, subprocess, sys, yaml
    base_ic={'all_flagged_policy':'preserve_legacy_for_structural_guard','incumbent_retention_policy':'preserve_admissible_incumbent','regret_risk_enabled':False,'replacement_regret_risk_enabled':False,'retention_regret_risk_enabled':False}
    def cfg(sc):
        ic=dict(base_ic); ic['selection_conditioned_intervention_recovery']=sc
        return {'evidence':{'budget':16},'runtime':{'decisive_frontier_value':{'incumbent_contrastive_extremal_recovery':ic}}}
    common={'enabled':True,'feature_names':FEATURE_NAMES,'base_feature_names':[],'feature_mean':[0.0]*len(FEATURE_NAMES),'feature_std':[1.0]*len(FEATURE_NAMES),'weights':[0.0]*len(FEATURE_NAMES),'bias':0.0,'ridge_lambda':1.0,'training_target':'x','training_weighting':'scene','leverage_inverse':[],'no_fallback':True}
    mean=dict(common,mode='mean_rank',model_type='v32_1_scene_equal_edge_mean_ridge_control')
    pair=dict(common,mode='rank_only',model_type='v33_scene_equal_incumbent_augmented_teacher_best_pair_gap_control')
    rank=dict(common,mode='rank_only',model_type='incumbent_augmented_scene_max_teacher_regret_structured_margin')
    main=dict(rank,mode='conformal_veto',conformal_overprediction_quantile=0.4)
    files={'v20':{'evidence':{'budget':16},'runtime':{'decisive_frontier_value':{'incumbent_contrastive_extremal_recovery':{}}}},'preserve':cfg({'enabled':False}),'mean':cfg(mean),'pair':cfg(pair),'rank':cfg(rank),'main':cfg(main)}
    paths={}
    for k,v in files.items(): p=tmp_path/f'{k}.yaml'; p.write_text(yaml.safe_dump(v,sort_keys=False)); paths[k]=p
    cal=tmp_path/'cal.json'; cal.write_text(json.dumps({'calibration_total_scene_count':500,'selected_policy_proposal_count':100,'alpha':0.05,'selected_policy_conformal_quantile':0.4,'calibration_uses_promotion_labels':False}))
    out=tmp_path/'contract.json'
    subprocess.run([sys.executable,'-m','bdse.tools.check_v64_3_34_eaf_icer_rsmr_contract','--v20-config',str(paths['v20']),'--preserve-config',str(paths['preserve']),'--mean-config',str(paths['mean']),'--pair-config',str(paths['pair']),'--rank-config',str(paths['rank']),'--main-config',str(paths['main']),'--calibration-report',str(cal),'--output',str(out)],check=True,text=True,capture_output=True)
    assert json.loads(out.read_text())['pass'] is True
