from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from bdse.tools.fit_v64_3_17_eaf_daler import FEATURE_NAMES


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--expect',choices=['raw','fitted'],required=True); ap.add_argument('--output'); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    runtime=cfg.get('runtime',{}) or {}; frontier=runtime.get('decisive_frontier_value',{}) or {}
    daler=frontier.get('deployment_aligned_listwise_extremal_reliability',{}) or {}
    raer=frontier.get('reliability_aware_extremal_reranking',{}) or {}
    eair=frontier.get('learned_intervention_reliability',{}) or {}; ocfi=frontier.get('one_sided_intervention',{}) or {}
    selector=cfg.get('selector',{}) or {}; evidence=cfg.get('evidence',{}) or {}; metadata=cfg.get('metadata',{}) or {}
    utility=((cfg.get('tournament',{}) or {}).get('utility_refinement',{}) or {})
    guard=runtime.get('pair_action_anchor_guard',{}) or {}; dual=runtime.get('dual_certificate',{}) or {}
    meta=metadata.get('algorithm_version'); prov=(cfg.get('provenance',{}) or {}).get('algorithm_version'); fitted=a.expect=='fitted'
    names=list(daler.get('feature_names',[]) or []); mean=list(daler.get('feature_mean',[]) or []); std=list(daler.get('feature_std',[]) or []); weights=list(daler.get('weights',[]) or [])
    checks={
        'version':meta=='V64.3.17-EAF-DALER-DARM-DBR' and prov==meta,
        'frontier_enabled':bool(frontier.get('enabled',False)),
        'ocfi_disabled':not bool(ocfi.get('enabled',False)),
        'scalar_eair_disabled':not bool(eair.get('enabled',False)),
        'raer_disabled':not bool(raer.get('enabled',False)),
        'daler_state':bool(daler.get('enabled',False))==fitted,
        'feature_instrumentation':bool(daler.get('instrument_features',False)),
        'model_type':str(daler.get('model_type',''))=='standardized_linear_anchor_augmented_listwise_reliability',
        'training_target':str(daler.get('training_target',''))=='teacher_best_executable_challenger_or_anchor',
        'training_objective':str(daler.get('training_objective',''))=='anchor_augmented_listwise_ce_plus_class_balanced_edge_bce',
        'selection_operator':'fixed_anchor' in str(daler.get('threshold_policy','')) and 'argmax_shared_reliability_logit' in str(daler.get('selection_operator','')),
        'anchor_logit_fixed_zero':abs(float(daler.get('anchor_logit',99.0)))<1e-12,
        'aux_edge_bce_fixed_one':abs(float(daler.get('aux_edge_bce_weight',-1.0))-1.0)<1e-12,
        'deployment_alignment':bool(daler.get('require_guard_executable',False)) and bool(daler.get('require_utility_equivalence',False)) and bool(daler.get('require_safe_available_for_learned_intervention',False)),
        'feature_schema':names==FEATURE_NAMES if fitted else names==[],
        'vector_contract':(len(names)==len(mean)==len(std)==len(weights)>0 and all(math.isfinite(float(x)) for x in [*mean,*std,*weights,daler.get('bias',0.0)])) if fitted else len(names)==len(mean)==len(std)==len(weights)==0,
        'std_positive':all(float(x)>0 for x in std) if fitted else True,
        'utility_refinement_frozen_on':bool(utility.get('enabled',False)) and bool(utility.get('pair_certificate_enabled',False)),
        'one_sided_guard_frozen_on':bool(guard.get('enabled',False)),
        'evidence_certificate_frozen_on':bool(dual.get('enabled',False)) and bool(dual.get('require_evidence_certificate_before_residual_flip',False)),
        'budget_B16':int(evidence.get('budget',-1))==16 and int(metadata.get('fixed_planner_interface_evidence_budget',-1))==16 and int(selector.get('min_selected_atoms',-1))==16,
        'topM24':int(selector.get('proposal_top_m',-1))==24 and int(metadata.get('fixed_proposal_top_m',-1))==24,
        'evaluation_only':bool((cfg.get('training',{}) or {}).get('evaluation_only',False)),
    }
    report={'audit':'v64_3_17_eaf_daler_contract','expect':a.expect,'passed':all(checks.values()),'checks':checks}
    if a.output:
        Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,indent=2,sort_keys=True))
    if not report['passed']:
        raise SystemExit(2)


if __name__=='__main__': main()
