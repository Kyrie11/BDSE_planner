from __future__ import annotations
import argparse, json, math
from pathlib import Path
import yaml


def _ic(c):
    return c.get('runtime',{}).get('decisive_frontier_value',{}).get('incumbent_contrastive_extremal_recovery',{}) or {}


def main():
    ap=argparse.ArgumentParser()
    for x in ['v20-config','preserve-config','mean-config','pair-config','rank-config','main-config','calibration-report','output']:
        ap.add_argument('--'+x,required=True)
    a=ap.parse_args()
    cfgs={}
    for n in ['v20','preserve','mean','pair','rank','main']:
        cfgs[n]=yaml.safe_load(Path(getattr(a,n+'_config')).read_text())
    C=json.load(open(a.calibration_report))
    ic={k:_ic(v) for k,v in cfgs.items()}
    sc={k:(ic[k].get('selection_conditioned_intervention_recovery',{}) or {}) for k in ['mean','pair','rank','main']}
    keys=['feature_names','base_feature_names','feature_mean','feature_std','weights','bias','ridge_lambda','training_target','training_weighting']
    checks={}
    checks['preserve_incumbent_default']=ic['preserve'].get('incumbent_retention_policy')=='preserve_admissible_incumbent' and not bool((ic['preserve'].get('selection_conditioned_intervention_recovery',{}) or {}).get('enabled',False))
    checks['all_learned_arms_preserve_incumbent_default']=all(ic[x].get('incumbent_retention_policy')=='preserve_admissible_incumbent' for x in ['mean','pair','rank','main'])
    checks['mean_is_corrected_control']=bool(sc['mean'].get('enabled')) and sc['mean'].get('mode')=='mean_rank'
    checks['pair_is_exact_v33_control']=bool(sc['pair'].get('enabled')) and sc['pair'].get('mode')=='rank_only' and sc['pair'].get('model_type')=='v33_scene_equal_incumbent_augmented_teacher_best_pair_gap_control'
    checks['rank_is_regret_structured_margin']=bool(sc['rank'].get('enabled')) and sc['rank'].get('mode')=='rank_only' and sc['rank'].get('model_type')=='incumbent_augmented_scene_max_teacher_regret_structured_margin'
    checks['main_selected_policy_veto']=bool(sc['main'].get('enabled')) and sc['main'].get('mode')=='conformal_veto' and bool(sc['main'].get('no_fallback'))
    checks['rank_main_same_selector']=all(sc['rank'].get(k)==sc['main'].get(k) for k in keys)
    q=float(sc['main'].get('conformal_overprediction_quantile',float('nan'))); cq=float(C.get('selected_policy_conformal_quantile',float('nan')))
    checks['q_exact']=math.isfinite(q) and q>=0 and abs(q-cq)<=1e-12
    checks['calibration_valid']=C.get('calibration_total_scene_count')==500 and int(C.get('selected_policy_proposal_count',0))>=64 and abs(float(C.get('alpha',0))-.05)<=1e-12 and C.get('calibration_uses_promotion_labels') is False
    checks['no_leverage_or_old_risk_head']=sc['rank'].get('leverage_inverse',[])==[] and all(not bool(ic[x].get('regret_risk_enabled',False)) and not bool(ic[x].get('replacement_regret_risk_enabled',False)) and not bool(ic[x].get('retention_regret_risk_enabled',False)) for x in ['preserve','mean','pair','rank','main'])
    checks['same_budget']=len({cfgs[x].get('evidence',{}).get('budget') for x in cfgs})==1
    checks['structural_delegation']=all(ic[x].get('all_flagged_policy')=='preserve_legacy_for_structural_guard' for x in ['preserve','mean','pair','rank','main'])
    ok=all(checks.values())
    rep={'audit':'v64_3_34_eaf_icer_rsmr_contract','pass':ok,'checks':checks,'interpretation':'RSMR replaces V33 all-rivals average pair regression with one cost-sensitive max regret violation per scene while retaining the incumbent as exact zero-score null action. PAIR is the frozen V33 selector ablation. Independent calibration attaches only to the frozen RSMR proposal; MAIN may only accept that proposal or return incumbent.'}
    Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
    print(json.dumps(rep,indent=2,sort_keys=True))
    if not ok: raise SystemExit('V64.3.34 RSMR contract failed')

if __name__=='__main__': main()
