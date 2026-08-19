from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, yaml
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _ICER_TRANSITION_FEATURE_NAMES


def _icer(c): return c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--expect',choices=['evidence-risk-dual','transition-risk-scalar','transition-risk-dual'],required=True); ap.add_argument('--frozen-v20-dual-config',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    c=yaml.safe_load(Path(a.config).read_text()); b=yaml.safe_load(Path(a.frozen_v20_dual_config).read_text()); ic=_icer(c); bic=_icer(b)
    errs=[]
    if not bool(ic.get('enabled',False)): errs.append('ICER disabled')
    if not bool(ic.get('regret_risk_enabled',False)): errs.append('regret risk disabled')
    expected_mode='evidence_only' if a.expect=='evidence-risk-dual' else 'transition_conditioned'
    if str(ic.get('regret_risk_feature_mode',''))!=expected_mode: errs.append('regret risk mode mismatch')
    expected_dom='scalar_only' if a.expect=='transition-risk-scalar' else 'scalar_positive_dual_equal_mean'
    if str(ic.get('dominance_policy',''))!=expected_dom: errs.append('dominance policy mismatch')
    expected_names=[f'evidence::{n}' for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]+([] if expected_mode=='evidence_only' else [f'transition::{n}' for n in _ICER_TRANSITION_FEATURE_NAMES])
    for prefix in ['retention_regret_risk','replacement_regret_risk']:
        if list(ic.get(f'{prefix}_feature_names',[]))!=expected_names: errs.append(f'{prefix} schema mismatch')
        n=len(expected_names)
        for field in ['feature_mean','feature_std','weights']:
            if len(ic.get(f'{prefix}_{field}',[]))!=n: errs.append(f'{prefix}_{field} length mismatch')
        if not np.isfinite(float(ic.get(f'{prefix}_bias',np.nan))): errs.append(f'{prefix}_bias nonfinite')
    if str(ic.get('regret_risk_threshold_policy',''))!='fixed_zero_expected_improvement_boundary_no_validation_sweep': errs.append('zero-boundary contract missing')
    if str(ic.get('all_flagged_policy',''))!='preserve_legacy_for_structural_guard': errs.append('structural delegation changed')
    # V19/V20 support and dominance heads must remain byte-equivalent in numeric content.
    frozen_keys=['support_feature_names','support_feature_mean','support_feature_std','support_weights','support_bias',
                 'scalar_dominance_feature_names','scalar_dominance_base_feature_names','scalar_dominance_feature_mean','scalar_dominance_feature_std','scalar_dominance_weights','scalar_dominance_bias',
                 'profile_dominance_feature_names','profile_dominance_base_feature_names','profile_dominance_feature_mean','profile_dominance_feature_std','profile_dominance_weights','profile_dominance_bias']
    for k in frozen_keys:
        if ic.get(k)!=bic.get(k): errs.append(f'frozen head changed: {k}')
    rep={'pass':not errs,'errors':errs,'expect':a.expect,'feature_count':len(expected_names),'frozen_head_identity':not any(x.startswith('frozen head') for x in errs),'no_threshold_sweep':str(ic.get('regret_risk_threshold_policy','')).startswith('fixed_zero')}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True))
    if errs: raise SystemExit('STOP CONTRACT: '+'; '.join(errs))
    print(json.dumps(rep,sort_keys=True))
if __name__=='__main__': main()
