from __future__ import annotations
import argparse,json,math
from pathlib import Path
import yaml
from bdse.tools.fit_v64_3_35_eaf_icer_fbcsr import CONTEXT_NAMES


def _ic(c): return c.get('runtime',{}).get('decisive_frontier_value',{}).get('incumbent_contrastive_extremal_recovery',{}) or {}

def main():
    ap=argparse.ArgumentParser()
    for x in ['v20-config','preserve-config','mean-config','rsmr-config','factor-config','rank-config','main-config','calibration-report','output']:
        ap.add_argument('--'+x,required=True)
    a=ap.parse_args(); cfgs={}
    for n in ['v20','preserve','mean','rsmr','factor','rank','main']: cfgs[n]=yaml.safe_load(Path(getattr(a,n+'_config')).read_text())
    C=json.load(open(a.calibration_report)); ic={k:_ic(v) for k,v in cfgs.items()}; sc={k:(ic[k].get('selection_conditioned_intervention_recovery',{}) or {}) for k in ['mean','rsmr','factor','rank','main']}
    selector_keys=['feature_names','base_feature_names','feature_mean','feature_std','weights','bias','ridge_lambda','training_target','training_weighting','incumbent_context_feature_names','incumbent_context_feature_mean','incumbent_context_feature_std','incumbent_context_weights','incumbent_context_bias']
    checks={}
    checks['preserve_incumbent_default']=ic['preserve'].get('incumbent_retention_policy')=='preserve_admissible_incumbent' and not bool((ic['preserve'].get('selection_conditioned_intervention_recovery',{}) or {}).get('enabled',False))
    checks['all_learned_arms_preserve_incumbent_default']=all(ic[x].get('incumbent_retention_policy')=='preserve_admissible_incumbent' for x in ['mean','rsmr','factor','rank','main'])
    checks['mean_control']=bool(sc['mean'].get('enabled')) and sc['mean'].get('mode')=='mean_rank'
    checks['rsmr_control']=sc['rsmr'].get('model_type')=='incumbent_augmented_scene_max_teacher_regret_structured_margin'
    checks['factor_delta_ablation']=sc['factor'].get('model_type')=='v35_factorized_delta_structured_recovery' and not sc['factor'].get('incumbent_context_feature_names',[])
    checks['rank_context_structured']=sc['rank'].get('model_type')=='v35_factorized_basepoint_context_structured_recovery' and sc['rank'].get('mode')=='rank_only' and sc['rank'].get('incumbent_context_feature_names')==CONTEXT_NAMES
    checks['context_is_common_shift_only']=sc['rank'].get('context_operator')=='candidate_independent_common_shift_cannot_change_challenger_ordering'
    checks['main_selected_policy_veto']=sc['main'].get('mode')=='conformal_veto' and bool(sc['main'].get('no_fallback'))
    checks['rank_main_same_selector']=all(sc['rank'].get(k)==sc['main'].get(k) for k in selector_keys)
    q=float(sc['main'].get('conformal_overprediction_quantile',float('nan'))); cq=float(C.get('selected_policy_conformal_quantile',float('nan')))
    checks['q_exact']=math.isfinite(q) and q>=0 and abs(q-cq)<=1e-12
    checks['calibration_valid']=C.get('calibration_total_scene_count')==500 and int(C.get('selected_policy_proposal_count',0))>=64 and abs(float(C.get('alpha',0))-.05)<=1e-12 and C.get('calibration_uses_promotion_labels') is False
    checks['same_budget']=len({cfgs[x].get('evidence',{}).get('budget') for x in cfgs})==1
    checks['no_old_risk_head']=all(not bool(ic[x].get('regret_risk_enabled',False)) and not bool(ic[x].get('replacement_regret_risk_enabled',False)) and not bool(ic[x].get('retention_regret_risk_enabled',False)) for x in ['preserve','mean','rsmr','factor','rank','main'])
    checks['structural_delegation']=all(ic[x].get('all_flagged_policy')=='preserve_legacy_for_structural_guard' for x in ['preserve','mean','rsmr','factor','rank','main'])
    ok=all(checks.values()); rep={'audit':'v64_3_35_eaf_icer_fbcsr_contract','pass':ok,'checks':checks,'interpretation':'FDSR factorizes existence and conditional challenger ordering on the frozen contrast representation. FBCSR adds only an absolute-incumbent context common shift shared by all challengers, so context can affect intervention existence but cannot alter challenger ordering. MAIN may only accept the exact frozen FBCSR proposal or return incumbent.'}
    Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n'); print(json.dumps(rep,indent=2,sort_keys=True))
    if not ok: raise SystemExit('V64.3.35 FBCSR contract failed')
if __name__=='__main__': main()
