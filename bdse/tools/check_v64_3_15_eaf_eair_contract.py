from __future__ import annotations
import argparse, json, math
from pathlib import Path
import yaml

from bdse.tools.fit_v64_3_15_eaf_eair import FEATURE_NAMES


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--expect', choices=['raw','fitted'], required=True)
    ap.add_argument('--output')
    a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    runtime=cfg.get('runtime',{}) or {}
    frontier=runtime.get('decisive_frontier_value',{}) or {}
    eair=frontier.get('learned_intervention_reliability',{}) or {}
    ocfi=frontier.get('one_sided_intervention',{}) or {}
    selector=cfg.get('selector',{}) or {}
    evidence=cfg.get('evidence',{}) or {}
    metadata=cfg.get('metadata',{}) or {}
    meta=(cfg.get('metadata',{}) or {}).get('algorithm_version')
    prov=(cfg.get('provenance',{}) or {}).get('algorithm_version')
    trained=a.expect=='fitted'
    names=list(eair.get('feature_names',[]) or [])
    mean=list(eair.get('feature_mean',[]) or [])
    std=list(eair.get('feature_std',[]) or [])
    weights=list(eair.get('weights',[]) or [])
    checks={
      'version': meta=='V64.3.15-EAF-EAIR-DARM-DBR' and prov==meta,
      'frontier_enabled': bool(frontier.get('enabled',False)),
      'ocfi_disabled': not bool(ocfi.get('enabled',False)),
      'eair_state': bool(eair.get('enabled',False))==trained,
      'feature_instrumentation': bool(eair.get('instrument_features',False)),
      'model_type': str(eair.get('model_type',''))=='standardized_logistic_teacher_better_edge',
      'training_target': str(eair.get('training_target',''))=='teacher_proposed_vs_darm_anchor_margin_positive',
      'feature_schema': names==list(FEATURE_NAMES) if trained else names==[],
      'threshold_fixed': abs(float(eair.get('min_probability',-1))-0.5)<1e-12 and 'fixed_0.5' in str(eair.get('threshold_policy','')),
      'frontier_required': bool(eair.get('require_frontier_active',False)),
      'budget_B16': int(evidence.get('budget',-1))==16 and int(metadata.get('fixed_planner_interface_evidence_budget',-1))==16 and int(selector.get('min_selected_atoms',-1))==16,
      'topM24': int(selector.get('proposal_top_m',-1))==24 and int(metadata.get('fixed_proposal_top_m',-1))==24,
      'evaluation_only': bool((cfg.get('training',{}) or {}).get('evaluation_only',False)),
      'fitted_vector_contract': (len(names)==len(mean)==len(std)==len(weights) and len(names)>0 and all(math.isfinite(float(x)) for x in [*mean,*std,*weights,eair.get('bias',0.0)])) if trained else (len(names)==len(mean)==len(std)==len(weights)==0),
      'std_positive': all(float(x)>0 for x in std) if trained else True,
    }
    ok=all(checks.values())
    report={'audit':'v64_3_15_eaf_eair_contract','expect':a.expect,'passed':ok,'checks':checks}
    if a.output:
        Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps(report,indent=2,sort_keys=True))
    if not ok: raise SystemExit(2)
if __name__=='__main__': main()
