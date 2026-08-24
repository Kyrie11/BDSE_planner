from __future__ import annotations
import argparse,json
from pathlib import Path
import yaml

ALPHA=0.05

def _ic(c): return c['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v20-config',required=True); ap.add_argument('--preserve-config',required=True); ap.add_argument('--rank-config',required=True); ap.add_argument('--main-config',required=True); ap.add_argument('--calibration-report',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    v=yaml.safe_load(open(a.v20_config)); p=yaml.safe_load(open(a.preserve_config)); r=yaml.safe_load(open(a.rank_config)); m=yaml.safe_load(open(a.main_config)); cr=json.load(open(a.calibration_report))
    vi,pi,ri,mi=_ic(v),_ic(p),_ic(r),_ic(m); ps=pi.get('selection_conditioned_intervention_recovery',{}) or {}; rs=ri.get('selection_conditioned_intervention_recovery',{}); ms=mi.get('selection_conditioned_intervention_recovery',{})
    checks={}
    checks['global_evidence_budget_B16']=all(int(c.get('evidence',{}).get('budget',-1))==16 for c in [p,r,m])
    sels=[c.get('selector',{}) or {} for c in [p,r,m]]
    checks['proposal_top_m_M24']=all(int(s.get('proposal_top_m',-1))==24 for s in sels)
    checks['no_fbic_capacity_probe']=all(not bool(s.get('full_bank_capacity_probe',{}).get('enabled',False)) for s in sels)
    checks['support_head_exactly_frozen']=all(ci.get(k)==vi.get(k) for ci in [pi,ri,mi] for k in vi if k.startswith('support_'))
    checks['preserve_control_old_direct_dominance_frozen']=pi.get('dominance_policy',vi.get('dominance_policy','dual_equal_mean'))==vi.get('dominance_policy','dual_equal_mean') and not bool(ps.get('enabled',False))
    checks['old_regret_risk_disabled']=all(not bool(ci.get('regret_risk_enabled',False)) for ci in [pi,ri,mi])
    checks['admissible_incumbent_default']=all(ci.get('incumbent_retention_policy')=='preserve_admissible_incumbent' for ci in [pi,ri,mi])
    checks['scir_rank_then_veto_only']=bool(rs.get('enabled')) and bool(ms.get('enabled')) and rs.get('mode')=='rank_only' and ms.get('mode')=='conformal_veto' and bool(ms.get('no_fallback'))
    keys=['base_feature_names','feature_names','feature_mean','feature_std','weights','bias','ridge_lambda','training_population','training_weighting','training_target','proposal_operator','require_positive_predicted_improvement','conformal_alpha']
    checks['rank_main_model_identical']=all(rs.get(k)==ms.get(k) for k in keys)
    checks['fixed_alpha_005']=abs(float(rs.get('conformal_alpha',-1))-ALPHA)<1e-12 and abs(float(ms.get('conformal_alpha',-1))-ALPHA)<1e-12 and abs(float(cr.get('alpha',-1))-ALPHA)<1e-12
    checks['q_matches_independent_calibration']=abs(float(ms.get('conformal_overprediction_quantile',-1))-float(cr.get('conformal_overprediction_quantile',-2)))<1e-12 and float(ms.get('conformal_overprediction_quantile',-1))>=0
    checks['calibration_population_sufficient']=int(cr.get('selected_proposal_count',0))>=64
    ok=all(checks.values())
    rep={'audit':'v64_3_31_eaf_icer_scir_contract','pass':ok,'checks':checks,'interpretation':'SCIR changes direct incumbent-replacement semantics/operator only. PRESERVE is the causal control that removes the previously falsified admissible-incumbent veto while retaining frozen V20 direct dominance; SCIR rank/main share that control and differ only by the new direct intervention model and then a veto-only conformal certificate.'}
    q=Path(a.output); q.parent.mkdir(parents=True,exist_ok=True); q.write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n'); print(json.dumps(rep,indent=2,sort_keys=True))
    if not ok: raise SystemExit('STOP V31 SCIR contract failure')
if __name__=='__main__': main()
