from __future__ import annotations
import argparse, json, math
from pathlib import Path
import yaml


def _ic(cfg):
    return cfg.get('runtime',{}).get('decisive_frontier_value',{}).get('incumbent_contrastive_extremal_recovery',{}) or {}

def _same(a,b,k):
    return a.get(k)==b.get(k)

def main():
    ap=argparse.ArgumentParser()
    for x in ['v20-config','preserve-config','mean-config','main-config','calibration-report','output']:
        ap.add_argument('--'+x,required=True)
    a=ap.parse_args()
    V=yaml.safe_load(Path(a.v20_config).read_text()); P=yaml.safe_load(Path(a.preserve_config).read_text()); M=yaml.safe_load(Path(a.mean_config).read_text()); S=yaml.safe_load(Path(a.main_config).read_text()); C=json.load(open(a.calibration_report))
    vi,pi,mi,si=map(_ic,[V,P,M,S]); ms=mi.get('selection_conditioned_intervention_recovery',{}) or {}; ss=si.get('selection_conditioned_intervention_recovery',{}) or {}
    keys=['feature_names','base_feature_names','feature_mean','feature_std','weights','bias','ridge_lambda','leverage_inverse','selection_scale_floor','conformal_alpha']
    checks={}
    checks['preserve_only_changes_incumbent_default']=pi.get('incumbent_retention_policy')=='preserve_admissible_incumbent' and not bool((pi.get('selection_conditioned_intervention_recovery',{}) or {}).get('enabled',False))
    checks['mean_and_main_preserve_incumbent']=mi.get('incumbent_retention_policy')=='preserve_admissible_incumbent' and si.get('incumbent_retention_policy')=='preserve_admissible_incumbent'
    checks['mean_control_mode']=bool(ms.get('enabled')) and ms.get('mode') in {'mean_rank','rank_only'}
    checks['main_simultaneous_lcb_mode']=bool(ss.get('enabled')) and ss.get('mode')=='simultaneous_lcb' and bool(ss.get('no_fallback'))
    checks['model_and_scale_frozen_across_calibration']=all(_same(ms,ss,k) for k in keys)
    q=float(ss.get('simultaneous_conformal_quantile',float('nan'))); cq=float(C.get('scene_simultaneous_quantile',float('nan')))
    checks['calibration_quantile_exact']=math.isfinite(q) and q>=0 and math.isfinite(cq) and abs(q-cq)<=1e-12
    checks['calibration_protocol']=C.get('calibration_total_scene_count')==500 and int(C.get('direct_eligible_scene_count',0))>=64 and abs(float(C.get('alpha',0.0))-0.05)<=1e-12 and C.get('calibration_uses_promotion_labels') is False
    checks['risk_modules_disabled']=all(not bool(x.get('regret_risk_enabled',False)) and not bool(x.get('retention_regret_risk_enabled',False)) and not bool(x.get('replacement_regret_risk_enabled',False)) for x in [pi,mi,si])
    checks['same_upstream_budget']=V.get('evidence',{}).get('budget')==P.get('evidence',{}).get('budget')==M.get('evidence',{}).get('budget')==S.get('evidence',{}).get('budget')
    checks['all_flagged_delegation']=all(x.get('all_flagged_policy')=='preserve_legacy_for_structural_guard' for x in [pi,mi,si])
    ok=all(checks.values())
    rep={'audit':'v64_3_32_1_eaf_icer_ssir_contract','pass':ok,'checks':checks,'interpretation':'SSIR preserves the V31 same-scene continuous mean model but adds a frozen candidate-specific ridge-leverage normalization and one independent CAL500 direct-domain scene-simultaneous conformal quantile. Unlike V31 post-selection common-offset veto, the lower bound participates in extremal ordering before selection; no second-best fallback is introduced.'}
    Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True))
    print(json.dumps(rep,indent=2,sort_keys=True))
    if not ok: raise SystemExit('V64.3.32.1 SSIR contract failed')
if __name__=='__main__': main()
