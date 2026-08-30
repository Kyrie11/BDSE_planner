from __future__ import annotations
import argparse,json,math
from pathlib import Path
import yaml
from bdse.tools.fit_v64_3_16_eaf_raer import FEATURE_NAMES

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--expect',choices=['raw','fitted'],required=True); ap.add_argument('--output'); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); runtime=cfg.get('runtime',{}) or {}; frontier=runtime.get('decisive_frontier_value',{}) or {}
    raer=frontier.get('reliability_aware_extremal_reranking',{}) or {}; eair=frontier.get('learned_intervention_reliability',{}) or {}; ocfi=frontier.get('one_sided_intervention',{}) or {}
    selector=cfg.get('selector',{}) or {}; evidence=cfg.get('evidence',{}) or {}; metadata=cfg.get('metadata',{}) or {}
    meta=metadata.get('algorithm_version'); prov=(cfg.get('provenance',{}) or {}).get('algorithm_version'); fitted=a.expect=='fitted'
    names=list(raer.get('feature_names',[]) or []); mean=list(raer.get('feature_mean',[]) or []); std=list(raer.get('feature_std',[]) or []); weights=list(raer.get('weights',[]) or [])
    checks={
      'version':meta=='V64.3.16-EAF-RAER-DARM-DBR' and prov==meta,
      'frontier_enabled':bool(frontier.get('enabled',False)), 'ocfi_disabled':not bool(ocfi.get('enabled',False)), 'scalar_eair_disabled':not bool(eair.get('enabled',False)),
      'raer_state':bool(raer.get('enabled',False))==fitted, 'feature_instrumentation':bool(raer.get('instrument_features',False)),
      'model_type':str(raer.get('model_type',''))=='standardized_logistic_all_frontier_teacher_better_edge',
      'training_target':str(raer.get('training_target',''))=='teacher_challenger_vs_darm_anchor_margin_positive',
      'selection_utility':str(raer.get('selection_utility',''))=='p_teacher_better_times_positive_frozen_eaf_margin',
      'feature_schema':names==FEATURE_NAMES if fitted else names==[],
      'threshold_fixed':abs(float(raer.get('min_probability',-1))-.5)<1e-12 and 'fixed_0.5' in str(raer.get('threshold_policy','')),
      'positive_raw_required':bool(raer.get('require_positive_raw_margin',False)),
      'budget_B16':int(evidence.get('budget',-1))==16 and int(metadata.get('fixed_planner_interface_evidence_budget',-1))==16 and int(selector.get('min_selected_atoms',-1))==16,
      'topM24':int(selector.get('proposal_top_m',-1))==24 and int(metadata.get('fixed_proposal_top_m',-1))==24,
      'evaluation_only':bool((cfg.get('training',{}) or {}).get('evaluation_only',False)),
      'vector_contract':(len(names)==len(mean)==len(std)==len(weights)>0 and all(math.isfinite(float(x)) for x in [*mean,*std,*weights,raer.get('bias',0.0)])) if fitted else len(names)==len(mean)==len(std)==len(weights)==0,
      'std_positive':all(float(x)>0 for x in std) if fitted else True,
    }
    report={'audit':'v64_3_16_eaf_raer_contract','expect':a.expect,'passed':all(checks.values()),'checks':checks}
    if a.output: Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps(report,indent=2,sort_keys=True));
    if not report['passed']: raise SystemExit(2)
if __name__=='__main__':main()
