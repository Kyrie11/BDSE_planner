from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _f, _icer_edge_diag, _metric_pack
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _identity_rate, _load_rows
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _replacement_tail_diag
from bdse.tools.check_v64_3_30_eaf_icer_fbic_split import _query_diag
from bdse.tools.check_v64_3_33_eaf_icer_spcr_split import _containment, _policy_diag

CAT=-0.5
CAPTURE_GAIN_MIN=0.03
EPS=1e-12


def _structural(rows, raw, flagged):
    return {
        'all_flagged_scene_count':len(flagged),
        'final_identity_vs_raw':_identity_rate(rows,raw,flagged),
        'icer_structural_delegation_rate':float(np.mean([_f(rows[t],'decisive_frontier_icer_structural_domain_delegated',0.0) for t in flagged])) if flagged else 1.0,
    }


def main():
    ap=argparse.ArgumentParser(description='Audit one V64.3.34 RSMR fresh block.')
    ap.add_argument('--split-name',required=True)
    for tag in ['raw','v20','preserve','mean','pair','rank','main']:
        ap.add_argument(f'--{tag}-metrics',required=True)
        ap.add_argument(f'--{tag}-rows',required=True)
        if tag!='raw': ap.add_argument(f'--{tag}-edges',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()

    tags=['raw','v20','preserve','mean','pair','rank','main']
    metrics={t:json.load(open(getattr(a,t+'_metrics'))) for t in tags}
    rows={t:_load_rows(getattr(a,t+'_rows')) for t in tags}
    toks=set(rows['raw'])
    if len(toks)!=500 or any(set(rows[t])!=toks for t in tags[1:]):
        raise SystemExit('STOP DATA: V34 seven arms must contain exact paired 500 scenes')

    flagged={t for t in toks if _f(rows['raw'][t],'all_actions_safety_flagged_rate',0)>=.5}
    safe=toks-flagged
    q={t:_query_diag(rows['v20'],rows[t],toks) for t in ['preserve','mean','pair','rank','main']}
    query_ok=all(v['all_query_counts_exact_scene_parity'] for v in q.values())
    structural={t:_structural(rows[t],rows['raw'],flagged) for t in ['v20','preserve','mean','pair','rank','main']}
    struct_ok=all((not flagged) or (structural[t]['final_identity_vs_raw']==1.0 and structural[t]['icer_structural_delegation_rate']==1.0) for t in ['preserve','mean','pair','rank','main'])
    cont=_containment(rows['rank'],rows['main'],safe)
    edge={t:_icer_edge_diag(Path(getattr(a,t+'_edges')),safe) for t in ['v20','preserve','mean','pair','rank','main']}
    tail={t:_replacement_tail_diag(rows['raw'],rows[t],getattr(a,t+'_edges'),safe) for t in ['v20','preserve','mean','pair','rank','main']}
    policy={t:_policy_diag(getattr(a,t+'_edges'),safe) for t in ['mean','pair','rank','main']}
    M={t:_metric_pack(metrics[t]) for t in tags}

    existence=bool(policy['rank']['no_positive_opportunity_false_intervention_count'] < policy['mean']['no_positive_opportunity_false_intervention_count'])
    opportunity=bool(policy['rank']['positive_capture_rate'] > policy['pair']['positive_capture_rate'] + EPS)
    selected_direction=bool(policy['rank']['proposal_teacher_improvement_sum'] > policy['pair']['proposal_teacher_improvement_sum'] + EPS and policy['rank']['proposal_catastrophic_count'] < policy['mean']['proposal_catastrophic_count'])
    ordering=bool(existence and opportunity and selected_direction)

    prescap=float(edge['preserve']['direct_incumbent_opportunity_capture_rate'])
    maincap=float(edge['main']['direct_incumbent_opportunity_capture_rate'])
    coverage=bool(math.isfinite(maincap) and math.isfinite(prescap) and maincap>=prescap+CAPTURE_GAIN_MIN-EPS)
    main_tail=bool(
        tail['main']['count']>=8
        and tail['main']['teacher_improvement_sum']>=-EPS
        and math.isfinite(tail['main']['teacher_improvement_worst'])
        and tail['main']['teacher_improvement_worst']>CAT
        and tail['main']['teacher_negative_rms']<=tail['preserve']['teacher_negative_rms']+EPS
        and edge['main']['direct_incumbent_replacement_precision']>=edge['preserve']['direct_incumbent_replacement_precision']-EPS
    )
    cert=bool(
        cont['monotone_selected_policy_containment_valid']
        and tail['main']['teacher_negative_rms']<=tail['rank']['teacher_negative_rms']+EPS
        and (not math.isfinite(tail['rank']['teacher_improvement_worst']) or tail['main']['teacher_improvement_worst']>=tail['rank']['teacher_improvement_worst']-EPS)
        and edge['main']['direct_incumbent_replacement_precision']>=edge['rank']['direct_incumbent_replacement_precision']-EPS
    )
    endp=bool(
        M['main']['match']>=M['preserve']['match']-.002
        and M['main']['regret']<=M['preserve']['regret']*1.005
        and M['main']['match']>=M['v20']['match']-.002
        and M['main']['regret']<=M['v20']['regret']*1.005
    )
    eng=bool(query_ok and struct_ok and cont['monotone_selected_policy_containment_valid'])
    full=bool(eng and ordering and coverage and main_tail and cert and endp)

    if not eng:
        nxt='STOP_fix_RSMR_engineering_or_containment_before_scientific_interpretation'
    elif not existence:
        nxt='RSMR_does_not_improve_should_we_intervene_vs_corrected_mean_stop_do_not_add_classifier_or_threshold'
    elif not opportunity:
        nxt='RSMR_still_over_suppresses_positive_opportunity_recovery_vs_V33_pair_stop_revisit_scene_structured_objective_not_alpha'
    elif not selected_direction:
        nxt='RSMR_recovers_more_opportunities_but_which_intervention_ordering_or_selected_tail_remains_unreliable'
    elif not coverage:
        nxt='RSMR_rank_signal_exists_but_selected_policy_certificate_does_not_add_3pp_capture_over_preserve_stop_do_not_alpha_sweep'
    elif not main_tail:
        nxt='RSMR_coverage_exists_but_direct_selected_tail_fails_stop_do_not_threshold_rescue'
    elif not cert:
        nxt='selected_policy_certificate_not_incremental_over_RSMR_ranker_stop_revisit_calibration_target_not_features'
    elif not endp:
        nxt='direct_RSMR_passes_but_endpoint_does_not_convert_audit_runtime_path_composition'
    else:
        nxt='if_second_fresh_block_also_passes_freeze_RSMR_and_run_one_independent_full_validation_reproduction'

    rep={
        'audit':'v64_3_34_eaf_icer_rsmr_split',
        'split_name':a.split_name,
        'full_split_pass':full,
        'engineering_valid':eng,
        'intervention_existence_gain_vs_mean':existence,
        'opportunity_recovery_gain_vs_v33_pair':opportunity,
        'selected_path_direction_gain':selected_direction,
        'structured_regret_margin_mechanism_pass':ordering,
        'main_capture_gain_over_preserve':maincap-prescap if math.isfinite(maincap) and math.isfinite(prescap) else float('nan'),
        'main_meaningful_coverage':coverage,
        'main_tail_pass':main_tail,
        'certificate_incremental':cert,
        'endpoint_noninferior':endp,
        'next_action':nxt,
        'query_parity':q,
        'structural':structural,
        'monotone_selected_policy_containment':cont,
        'edge_diagnostics':edge,
        'policy_diagnostics':policy,
        'direct_selected_path_tail':tail,
        'metrics':M,
        'frozen_contract':{'catastrophic_threshold':CAT,'capture_gain_min':CAPTURE_GAIN_MIN,'main_selected_count_min':8,'no_AB_pooling':True,'no_alpha_lambda_feature_threshold_optimizer_sweep':True},
        'causal_note':'MEAN is corrected V32.1; PAIR is exact V33 all-rivals pair-gap control; RANK changes only the structured objective to one cost-sensitive max teacher-regret violation per scene; MAIN keeps exactly the same RANK proposal and may only accept it or return incumbent.',
    }
    Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(rep,indent=2,sort_keys=True))

if __name__=='__main__': main()
