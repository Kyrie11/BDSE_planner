from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from bdse.planner.tournament import _DACER_FEATURE_NAMES


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--expect',choices=['raw','gdaler','dacer-scalar','dacer-profile'],required=True)
    ap.add_argument('--output')
    a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    runtime=cfg.get('runtime',{}) or {}; frontier=runtime.get('decisive_frontier_value',{}) or {}
    dacer=frontier.get('deployment_admissible_counterfactual_extremal_recovery',{}) or {}
    daler=frontier.get('deployment_aligned_listwise_extremal_reliability',{}) or {}
    raer=frontier.get('reliability_aware_extremal_reranking',{}) or {}
    eair=frontier.get('learned_intervention_reliability',{}) or {}; ocfi=frontier.get('one_sided_intervention',{}) or {}
    selector=cfg.get('selector',{}) or {}; evidence=cfg.get('evidence',{}) or {}; metadata=cfg.get('metadata',{}) or {}
    utility=((cfg.get('tournament',{}) or {}).get('utility_refinement',{}) or {})
    guard=runtime.get('pair_action_anchor_guard',{}) or {}; dual=runtime.get('dual_certificate',{}) or {}
    fitted=a.expect!='raw'; feature_mode='profile' if a.expect=='dacer-profile' else 'scalar'
    objective_mode='listwise' if a.expect=='gdaler' else 'counterfactual'
    names=list(dacer.get('feature_names',[]) or []); mean=list(dacer.get('feature_mean',[]) or []); std=list(dacer.get('feature_std',[]) or []); weights=list(dacer.get('weights',[]) or [])
    meta=metadata.get('algorithm_version'); prov=(cfg.get('provenance',{}) or {}).get('algorithm_version')
    checks={
        'version':meta=='V64.3.18-EAF-DACER-DARM-DBR' and prov==meta,
        'frontier_enabled':bool(frontier.get('enabled',False)),
        'legacy_learned_arms_disabled':not any(bool(x.get('enabled',False)) for x in [ocfi,eair,raer,daler]),
        'dacer_state':bool(dacer.get('enabled',False))==fitted,
        'feature_instrumentation':bool(dacer.get('instrument_features',False)),
        'model_type':str(dacer.get('model_type',''))=='standardized_linear_guard_admissible_counterfactual_score',
        'guard_admissible_contract':bool(dacer.get('require_guard_admissible',False)) and bool(dacer.get('require_safe_available_for_learned_intervention',False)),
        'utility_not_hard_gate':str(dacer.get('utility_equivalence_role',''))=='diagnostic_tiebreak_only_not_hard_mask',
        'anchor_logit_fixed_zero':abs(float(dacer.get('anchor_logit',99.0)))<1e-12,
        'threshold_policy':str(dacer.get('threshold_policy',''))=='fixed_anchor_logit_0_no_validation_threshold_sweep',
        'feature_mode':str(dacer.get('feature_mode','profile'))==(feature_mode if fitted else 'profile'),
        'feature_schema':names==list(_DACER_FEATURE_NAMES) if fitted else names==[],
        'vector_contract':(len(names)==len(mean)==len(std)==len(weights)==len(_DACER_FEATURE_NAMES) and all(math.isfinite(float(x)) for x in [*mean,*std,*weights,dacer.get('bias',0.0)])) if fitted else len(names)==len(mean)==len(std)==len(weights)==0,
        'std_positive':all(float(x)>0 for x in std) if fitted else True,
        'objective_mode':str(dacer.get('objective_mode',''))==objective_mode if fitted else True,
        'objective_weights':(
            abs(float(dacer.get('support_bce_weight',-1.0))-1.0)<1e-12
            and abs(float(dacer.get('incumbent_dominance_weight',-1.0))-(0.0 if objective_mode=='listwise' else 1.0))<1e-12
        ) if fitted else True,
        'selection_operator':'guard_admissible' in str(dacer.get('selection_operator','')) and 'argmax' in str(dacer.get('selection_operator','')),
        'utility_refinement_frozen_on':bool(utility.get('enabled',False)) and bool(utility.get('pair_certificate_enabled',False)),
        'one_sided_guard_frozen_on':bool(guard.get('enabled',False)) and abs(float(guard.get('flip_margin',99.0))-0.015)<1e-12 and abs(float(guard.get('score_margin',99.0)))<1e-12,
        'evidence_certificate_frozen_on':bool(dual.get('enabled',False)) and bool(dual.get('require_evidence_certificate_before_residual_flip',False)) and abs(float(dual.get('min_evidence_certificate_fraction_for_residual_flip',0.0))-1.0)<1e-12,
        # DACER calls its candidate set final-guard-admissible only under the frozen
        # V64.3.17 contract where the post-selection robust-margin corrections are zero.
        # Fail the contract if future configs silently introduce a beta/sigma or
        # residual-epsilon term that the pre-selection admissibility mask does not model.
        'robust_margin_corrections_frozen_zero':abs(float(dual.get('residual_beta_uncertainty',99.0)))<1e-12 and abs(float(dual.get('residual_epsilon_cal',dual.get('residual_epsilon',99.0))))<1e-12,
        'budget_cap_B16':int(evidence.get('budget',-1))==16 and int(metadata.get('fixed_planner_interface_evidence_budget',-1))==16,
        'topM24':int(selector.get('proposal_top_m',-1))==24 and int(metadata.get('fixed_proposal_top_m',-1))==24,
        'evaluation_only':bool((cfg.get('training',{}) or {}).get('evaluation_only',False)),
    }
    report={'audit':'v64_3_18_eaf_dacer_contract','expect':a.expect,'passed':all(checks.values()),'checks':checks}
    if a.output:
        Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,indent=2,sort_keys=True))
    if not report['passed']:
        raise SystemExit(2)


if __name__=='__main__': main()
