from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _icer_edge_diag, _metric_pack, _f
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _load_rows, _path_diag, _guard_block_rate, _identity_rate
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _replacement_tail_diag


def _lex_lt(after: tuple[float,...], before: tuple[float,...], eps: float=1e-8)->bool:
    if not all(math.isfinite(x) for x in (*after,*before)): return False
    for a,b in zip(after,before):
        if a < b-eps: return True
        if a > b+eps: return False
    return False


def _pcwer_diag(rows: dict[str,dict[str,Any]], all_flagged: set[str]) -> dict[str,Any]:
    vals=list(rows.values()); n=len(vals); p='selector_proposal_conditioned_witness_rebinding_'
    accepted=[]; contracts=[]; reductions=[]; lock_ok=[]; allflag_accepted=0; reasons={}
    for t,r in rows.items():
        en=_f(r,p+'enabled',0)>=.5; ac=_f(r,p+'accepted',0)>=.5
        code=str(int(round(_f(r,p+'reason_code',-1)))); reasons[code]=reasons.get(code,0)+1
        if ac:
            accepted.append(t); allflag_accepted += int(t in all_flagged)
            before=tuple(_f(r,p+'baseline_'+k+'_error') for k in ['margin_linf','attribution_linf','margin_rms','attribution_rms'])
            after=tuple(_f(r,p+'final_'+k+'_error') for k in ['margin_linf','attribution_linf','margin_rms','attribution_rms'])
            bq=int(round(_f(r,p+'baseline_proposal_action',-1))); cq=int(round(_f(r,p+'candidate_proposal_action',-2)))
            bi=int(round(_f(r,p+'baseline_incumbent_action',-1))); ci=int(round(_f(r,p+'candidate_incumbent_action',-2)))
            ba=int(round(_f(r,p+'baseline_anchor_action',-1))); ca=int(round(_f(r,p+'candidate_anchor_action',-2)))
            strict=_lex_lt(after,before)
            contract=bool(strict and _f(r,p+'cardinality_preserved',0)>=.5 and _f(r,p+'budget_preserved',0)>=.5 and bq==cq and bi==ci and ba==ca and _f(r,p+'candidate_incumbent_admissible',0)>=.5)
            contracts.append(contract)
            reductions.append(float(before[0]-after[0]) if math.isfinite(before[0]-after[0]) else float('nan'))
            target=int(round(_f(r,'decisive_frontier_icer_proposal_lock_target_action',-3)))
            selected=int(round(_f(r,'decisive_frontier_icer_selected_action',-4)))
            legacy=int(round(_f(r,'decisive_frontier_icer_legacy_selected_action',-5)))
            lock_enabled=_f(r,'decisive_frontier_icer_proposal_lock_enabled',0)>=.5
            lock_ok.append(bool(lock_enabled and target==bq and selected in {legacy,target}))
    enabled_rate=sum(_f(r,p+'enabled',0)>=.5 for r in vals)/max(n,1)
    attempted_rate=sum(_f(r,p+'attempted',0)>=.5 for r in vals)/max(n,1)
    return {
        'scene_count':n,'enabled_rate':float(enabled_rate),'attempted_rate':float(attempted_rate),
        'accepted_count':len(accepted),'accepted_rate':float(len(accepted)/max(n,1)),
        'accepted_contract_pass_rate':float(np.mean(contracts)) if contracts else float('nan'),
        'accepted_proposal_lock_integrity_rate':float(np.mean(lock_ok)) if lock_ok else float('nan'),
        'all_flagged_accepted_count':int(allflag_accepted),'reason_code_counts':reasons,
        'accepted_margin_linf_reduction_mean':float(np.nanmean(reductions)) if reductions else float('nan'),
    }


def _lock_control_diag(rows: dict[str,dict[str,Any]])->dict[str,Any]:
    p='selector_proposal_conditioned_witness_rebinding_'; vals=list(rows.values())
    locks=[]
    for r in vals:
        if _f(r,p+'proposal_lock',0)>=.5:
            q=int(round(_f(r,p+'baseline_proposal_action',-1)))
            target=int(round(_f(r,'decisive_frontier_icer_proposal_lock_target_action',-2)))
            selected=int(round(_f(r,'decisive_frontier_icer_selected_action',-3)))
            legacy=int(round(_f(r,'decisive_frontier_icer_legacy_selected_action',-4)))
            locks.append(bool(_f(r,p+'lock_only',0)>=.5 and _f(r,'decisive_frontier_icer_proposal_lock_enabled',0)>=.5 and target==q and selected in {legacy,target}))
    return {'lock_scene_count':len(locks),'lock_integrity_rate':float(np.mean(locks)) if locks else float('nan')}


def _positive_count(tail:dict[str,Any])->int:
    c=int(tail.get('count',0)); p=float(tail.get('teacher_positive_precision',float('nan')))
    return int(round(c*p)) if math.isfinite(p) else 0


def main()->None:
    ap=argparse.ArgumentParser(description='One untouched V64.3.30 PCWER mechanism block.')
    ap.add_argument('--split-name',required=True)
    for name in ['raw','v20','aggregate-downside','lock-downside','pcwer-v20','pcwer-downside']:
        ap.add_argument(f'--{name}-metrics',required=True); ap.add_argument(f'--{name}-rows',required=True)
        if name!='raw': ap.add_argument(f'--{name}-edges',required=True)
    ap.add_argument('--output',required=True); args=ap.parse_args()
    cli=['raw','v20','aggregate-downside','lock-downside','pcwer-v20','pcwer-downside']; tags=[x.replace('-','_') for x in cli]
    metrics={t:json.load(open(getattr(args,t+'_metrics'),encoding='utf-8')) for t in tags}
    rows={t:_load_rows(getattr(args,t+'_rows')) for t in tags}
    tokens=set(rows['raw'])
    if len(tokens)!=500 or any(set(rows[t])!=tokens for t in tags[1:]): raise SystemExit('STOP DATA: exact paired 500-scene identity required')
    all_flagged={t for t in tokens if _f(rows['raw'][t],'all_actions_safety_flagged_rate',0)>=.5}; safe=tokens-all_flagged
    edge_paths={t:getattr(args,t+'_edges') for t in tags[1:]}
    edge={t:_icer_edge_diag(Path(edge_paths[t]),safe) for t in tags[1:]}
    path={t:_path_diag(rows['raw'],rows[t],edge_paths[t],safe,all_flagged) for t in tags[1:]}
    tails={t:_replacement_tail_diag(rows['raw'],rows[t],edge_paths[t],safe) for t in ['aggregate_downside','lock_downside','pcwer_downside']}
    M={t:_metric_pack(metrics[t]) for t in tags}
    pc=_pcwer_diag(rows['pcwer_downside'],all_flagged); pc_v20=_pcwer_diag(rows['pcwer_v20'],all_flagged); lock=_lock_control_diag(rows['lock_downside'])
    pc_contract=bool(pc['enabled_rate']==1.0 and pc_v20['enabled_rate']==1.0 and pc['accepted_count']>=5 and pc_v20['accepted_count']>=5 and pc['accepted_contract_pass_rate']==1.0 and pc_v20['accepted_contract_pass_rate']==1.0 and pc['accepted_proposal_lock_integrity_rate']==1.0 and pc_v20['accepted_proposal_lock_integrity_rate']==1.0 and pc['all_flagged_accepted_count']==0 and pc_v20['all_flagged_accepted_count']==0)
    lock_contract=bool(lock['lock_scene_count']>=5 and lock['lock_integrity_rate']==1.0)
    structural={'all_flagged_scene_count':len(all_flagged),'main_all_flagged_final_identity_vs_raw':_identity_rate(rows['pcwer_downside'],rows['raw'],all_flagged),'main_all_flagged_delegation_rate':(sum(_f(rows['pcwer_downside'][t],'decisive_frontier_icer_structural_domain_delegated',0)>=.5 for t in all_flagged)/len(all_flagged) if all_flagged else float('nan')),'main_safe_guard_block_rate':_guard_block_rate(rows['pcwer_downside'],safe)}
    structural_ok=bool(len(all_flagged)>=3 and structural['main_all_flagged_final_identity_vs_raw']==1.0 and structural['main_all_flagged_delegation_rate']==1.0 and structural['main_safe_guard_block_rate']<=.001)
    asymmetric=bool(path['pcwer_downside']['admissible_incumbent_to_anchor']['count']==0 and abs(path['pcwer_downside']['admissible_incumbent_to_anchor']['regret_delta_sum'])<=1e-9)
    main_tail=tails['pcwer_downside']; v25_tail=tails['aggregate_downside']; lock_tail=tails['lock_downside']
    catastrophe_free=bool(main_tail['count']==0 or (math.isfinite(main_tail['teacher_improvement_worst']) and main_tail['teacher_improvement_worst']>-0.5))
    main_path=path['pcwer_downside']['direct_incumbent_to_alternative']
    selected_path_safe=bool(main_path['count']>=8 and main_path['regret_delta_sum']<=0.0 and catastrophe_free)
    main_edge=edge['pcwer_downside']; v25_edge=edge['aggregate_downside']; lock_edge=edge['lock_downside']
    main_pos=_positive_count(main_tail); v25_pos=_positive_count(v25_tail); lock_pos=_positive_count(lock_tail)
    # Primary causal test is PCWER vs same-proposal lock-only control. Also require
    # the main arm to clear the historical V25 safe-coverage level so operator
    # changes cannot manufacture an artificial win.
    coverage_gain_vs_lock=bool(main_edge['direct_incumbent_opportunity_capture_rate']>=lock_edge['direct_incumbent_opportunity_capture_rate']+.03 and main_pos>=lock_pos+5)
    coverage_not_below_v25=bool(main_edge['direct_incumbent_opportunity_capture_rate']>=v25_edge['direct_incumbent_opportunity_capture_rate']-1e-9 and main_pos>=v25_pos)
    def tail_noninferior(a,b):
        return bool(math.isfinite(a['regret_positive_rms']) and math.isfinite(b['regret_positive_rms']) and a['regret_positive_rms']<=b['regret_positive_rms']+1e-9 and a['worst_regret_increase']<=b['worst_regret_increase']+1e-9 and a['teacher_negative_rms']<=b['teacher_negative_rms']+1e-9 and a['teacher_improvement_worst']>=b['teacher_improvement_worst']-1e-9)
    tail_vs_lock=tail_noninferior(main_tail,lock_tail); tail_vs_v25=tail_noninferior(main_tail,v25_tail)
    preservation=bool(M['pcwer_downside']['harmful']<=M['raw']['harmful']+.005 and M['pcwer_downside']['flip']<=M['raw']['flip']+.01 and selected_path_safe and asymmetric)
    endpoint_noninferior=bool(M['pcwer_downside']['match']>=M['aggregate_downside']['match']-.002 and M['pcwer_downside']['regret']<=M['aggregate_downside']['regret']*1.005 and M['pcwer_downside']['regret']<=M['raw']['regret']*1.02)
    endpoint_signal=bool(M['pcwer_downside']['match']>=M['aggregate_downside']['match']+.002 or M['pcwer_downside']['regret']<M['aggregate_downside']['regret']-1e-6)
    instrumentation=bool(edge['pcwer_downside']['scene_count']>=450 and edge['pcwer_downside']['admissible_edge_count']>=1800 and _f(metrics['pcwer_downside'],'decisive_frontier_value_complete_star_coverage')>=.99 and pc_contract and lock_contract)
    full=bool(instrumentation and structural_ok and asymmetric and selected_path_safe and coverage_gain_vs_lock and coverage_not_below_v25 and tail_vs_lock and tail_vs_v25 and preservation and endpoint_noninferior)
    if not instrumentation: nxt='engineering_or_PCWER_inactive_stop_before_fresh_interpretation'
    elif not structural_ok or not asymmetric: nxt='preservation_contract_failure_stop_PCWER'
    elif not selected_path_safe or not tail_vs_lock or not tail_vs_v25: nxt='proposal_conditioned_evidence_did_not_stabilize_selected_tail_stop_no_DRC_threshold_tuning'
    elif not coverage_gain_vs_lock: nxt='PCWER_did_not_expand_safe_confirmation_coverage_over_same_proposal_control_stop_reassess_witness_sufficiency'
    elif not coverage_not_below_v25: nxt='operator_conditioning_gain_does_not_recover_historical_V25_coverage_stop'
    elif not endpoint_noninferior: nxt='mechanism_gain_without_endpoint_noninferiority_audit_final_guard_interaction'
    else: nxt='split_pass_freeze_PCWER_wait_for_second_independent_block'
    rep={'audit':'v64_3_30_eaf_icer_pcwer_split','split_name':args.split_name,'full_split_pass':full,'instrumentation_valid':instrumentation,'pcwer_contract':pc_contract,'proposal_lock_control_contract':lock_contract,'deployment_alignment':structural_ok,'incumbent_default_invariant':asymmetric,'selected_replacement_path_nonharmful':selected_path_safe,'selected_replacement_catastrophe_free':catastrophe_free,'safe_recovery_coverage_gain_over_lock_control':coverage_gain_vs_lock,'safe_recovery_coverage_not_below_V25_DRC':coverage_not_below_v25,'selected_tail_noninferior_to_lock_control':tail_vs_lock,'selected_tail_noninferior_to_V25_DRC':tail_vs_v25,'endpoint_noninferior_to_V25_DRC':endpoint_noninferior,'endpoint_strict_signal_over_V25_DRC':endpoint_signal,'next_action':nxt,'pcwer_diagnostics':{'pcwer_v20':pc_v20,'pcwer_downside':pc,'lock_downside':lock},'structural':structural,'edge_diagnostics':edge,'path_diagnostics':path,'selected_replacement_tail_diagnostics':tails,'positive_direct_replacements':{'aggregate_downside':v25_pos,'lock_downside':lock_pos,'pcwer_downside':main_pos},'metrics':M}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rep,indent=2,sort_keys=True),encoding='utf-8'); print(json.dumps(rep,indent=2,sort_keys=True))

if __name__=='__main__': main()
