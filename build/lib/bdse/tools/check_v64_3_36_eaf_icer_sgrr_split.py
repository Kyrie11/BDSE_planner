from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from bdse.tools.check_v64_3_19_eaf_icer_screen import _f,_icer_edge_diag,_metric_pack
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _identity_rate,_load_rows
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _replacement_tail_diag
from bdse.tools.check_v64_3_30_eaf_icer_fbic_split import _query_diag
from bdse.tools.check_v64_3_33_eaf_icer_spcr_split import _containment,_policy_diag
CAT=-0.5; NOOP_REDUCTION_MIN=.20; CAPTURE_TOL=.03; EPS=1e-12

def _structural(rows,raw,flagged):
    return {'all_flagged_scene_count':len(flagged),'final_identity_vs_raw':_identity_rate(rows,raw,flagged),'icer_structural_delegation_rate':float(np.mean([_f(rows[t],'decisive_frontier_icer_structural_domain_delegated',0.0) for t in flagged])) if flagged else 1.0}

def main():
    ap=argparse.ArgumentParser(description='Audit one V64.3.36 SGRR fresh block.')
    tags=['raw','v20','preserve','rsmr','basepoint','geometry']
    ap.add_argument('--split-name',required=True)
    for tag in tags:
        ap.add_argument(f'--{tag}-metrics',required=True); ap.add_argument(f'--{tag}-rows',required=True)
        if tag!='raw': ap.add_argument(f'--{tag}-edges',required=True)
    ap.add_argument('--output',required=True); a=ap.parse_args()
    metrics={t:json.load(open(getattr(a,t+'_metrics'))) for t in tags}; rows={t:_load_rows(getattr(a,t+'_rows')) for t in tags}; toks=set(rows['raw'])
    if len(toks)!=500 or any(set(rows[t])!=toks for t in tags[1:]): raise SystemExit('STOP DATA: V36 six arms must contain exact paired 500 scenes')
    flagged={t for t in toks if _f(rows['raw'][t],'all_actions_safety_flagged_rate',0)>=.5}; safe=toks-flagged
    q={t:_query_diag(rows['v20'],rows[t],toks) for t in ['preserve','rsmr','basepoint','geometry']}; query_ok=all(v['all_query_counts_exact_scene_parity'] for v in q.values())
    structural={t:_structural(rows[t],rows['raw'],flagged) for t in ['v20','preserve','rsmr','basepoint','geometry']}; struct_ok=all((not flagged) or (structural[t]['final_identity_vs_raw']==1.0 and structural[t]['icer_structural_delegation_rate']==1.0) for t in ['preserve','rsmr','basepoint','geometry'])
    cb=_containment(rows['rsmr'],rows['basepoint'],safe); cg=_containment(rows['rsmr'],rows['geometry'],safe)
    edge={t:_icer_edge_diag(Path(getattr(a,t+'_edges')),safe) for t in tags if t!='raw'}
    tail={t:_replacement_tail_diag(rows['raw'],rows[t],getattr(a,t+'_edges'),safe) for t in tags if t!='raw'}
    policy={t:_policy_diag(getattr(a,t+'_edges'),safe) for t in ['rsmr','basepoint','geometry']}; M={t:_metric_pack(metrics[t]) for t in tags}
    r=policy['rsmr']; b=policy['basepoint']; g=policy['geometry']
    basepoint_signal=bool(b['no_positive_opportunity_false_intervention_count'] <= (1-NOOP_REDUCTION_MIN)*r['no_positive_opportunity_false_intervention_count']+EPS and b['positive_capture_rate']>=r['positive_capture_rate']-CAPTURE_TOL-EPS)
    existence=bool(g['no_positive_opportunity_false_intervention_count'] <= (1-NOOP_REDUCTION_MIN)*r['no_positive_opportunity_false_intervention_count']+EPS and g['positive_capture_rate']>=r['positive_capture_rate']-CAPTURE_TOL-EPS)
    hard_tail=bool(g['proposal_count']>=8 and g['proposal_teacher_improvement_sum']>=-EPS and g['proposal_catastrophic_count']==0 and math.isfinite(g['proposal_worst']) and g['proposal_worst']>CAT and tail['geometry']['teacher_negative_rms']<=tail['rsmr']['teacher_negative_rms']+EPS)
    preserve_cap=float(edge['preserve']['direct_incumbent_opportunity_capture_rate']); geom_cap=float(edge['geometry']['direct_incumbent_opportunity_capture_rate'])
    coverage=bool(math.isfinite(geom_cap) and math.isfinite(preserve_cap) and geom_cap>=preserve_cap+0.03-EPS)
    endp=bool(M['geometry']['match']>=M['preserve']['match']-.002 and M['geometry']['regret']<=M['preserve']['regret']*1.005 and M['geometry']['match']>=M['v20']['match']-.002 and M['geometry']['regret']<=M['v20']['regret']*1.005)
    eng=bool(query_ok and struct_ok and cb['monotone_selected_policy_containment_valid'] and cg['monotone_selected_policy_containment_valid'])
    full=bool(eng and existence and hard_tail and coverage and endp)
    if not eng: nxt='STOP_fix_V36_engineering_or_frozen_order_containment_before_scientific_interpretation'
    elif not existence: nxt='selection_geometry_reservation_does_not_reproduce_intervention_existence_gain_without_capture_loss_stop_do_not_add_more_thresholds'
    elif not hard_tail: nxt='reservation_improves_existence_but_hard_selected_tail_still_fails_revisit_reservation_target_not_ordering_features'
    elif not coverage: nxt='reservation_tail_is_safe_but_useful_recovery_coverage_is_insufficient'
    elif not endp: nxt='direct_reservation_mechanism_passes_but_endpoint_does_not_convert_audit_runtime_path_composition'
    else: nxt='if_second_fresh_block_also_passes_freeze_SGRR_and_run_one_independent_full_validation_reproduction'
    rep={'audit':'v64_3_36_eaf_icer_sgrr_split','split_name':a.split_name,'full_split_pass':full,'engineering_valid':eng,'basepoint_clean_signal':basepoint_signal,'selection_geometry_existence_gain':existence,'geometry_hard_tail_pass':hard_tail,'geometry_capture_gain_over_preserve':geom_cap-preserve_cap if math.isfinite(geom_cap) and math.isfinite(preserve_cap) else float('nan'),'geometry_meaningful_coverage':coverage,'endpoint_noninferior':endp,'next_action':nxt,'query_parity':q,'structural':structural,'basepoint_containment':cb,'selection_geometry_containment':cg,'edge_diagnostics':edge,'policy_diagnostics':policy,'direct_selected_path_tail':tail,'metrics':M,'frozen_contract':{'catastrophic_threshold':CAT,'noop_reduction_fraction_min':NOOP_REDUCTION_MIN,'capture_tolerance':CAPTURE_TOL,'promotion_capture_gain_over_preserve_min':.03,'no_AB_pooling':True,'ordering_weights_frozen_across_reservation_arms':True,'reservation_nonnegative_common_subtraction':True,'no_runtime_threshold_or_lambda_sweep':True}}
    Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n'); print(json.dumps(rep,indent=2,sort_keys=True))
if __name__=='__main__': main()
