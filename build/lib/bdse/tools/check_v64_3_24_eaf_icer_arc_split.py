from __future__ import annotations
import argparse,json,math
from pathlib import Path
from bdse.tools.check_v64_3_19_eaf_icer_screen import _icer_edge_diag,_metric_pack,_f
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _load_rows,_path_diag,_guard_block_rate,_identity_rate

def _increment(main,ctrl,M,ed,path):
 m=ed[main]; c=ed[ctrl]; mr=path[main]['direct_incumbent_to_alternative']['regret_delta_sum']; cr=path[ctrl]['direct_incumbent_to_alternative']['regret_delta_sum']
 return bool(M[main]['regret']<=M[ctrl]['regret']*1.02 and (mr < cr-1e-6 or m['direct_incumbent_replacement_precision']>=c['direct_incumbent_replacement_precision']+.01 or M[main]['regret']<=M[ctrl]['regret']*.99))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split-name',required=True)
 tags_cli=['raw','v20','aggregate-meanse','aggregate-downside','attribution-meanse','attribution-downside']
 for n in tags_cli:
  ap.add_argument(f'--{n}-metrics',required=True);ap.add_argument(f'--{n}-rows',required=True)
  if n!='raw':ap.add_argument(f'--{n}-edges',required=True)
 ap.add_argument('--output',required=True);a=ap.parse_args(); tags=[x.replace('-','_') for x in tags_cli]
 metrics={t:json.load(open(getattr(a,t+'_metrics'))) for t in tags};rows={t:_load_rows(getattr(a,t+'_rows')) for t in tags};tokens=set(rows['raw'])
 if len(tokens)<480 or any(set(rows[t])!=tokens for t in tags[1:]): raise SystemExit('STOP DATA: paired token identity mismatch')
 allflag={t for t in tokens if _f(rows['raw'][t],'all_actions_safety_flagged_rate',0)>=.5};safe=tokens-allflag; edge={t:getattr(a,t+'_edges') for t in tags[1:]};ed={t:_icer_edge_diag(Path(edge[t]),safe) for t in tags[1:]};path={t:_path_diag(rows['raw'],rows[t],edge[t],safe,allflag) for t in tags[1:]};M={t:_metric_pack(metrics[t]) for t in tags}; main='attribution_downside'; anchor={'match':_f(metrics['raw'],'selected_local_anchor_action_match'),'regret':_f(metrics['raw'],'selected_local_anchor_teacher_regret')}
 structural={'all_flagged_scene_count':len(allflag),'main_all_flagged_final_identity_vs_raw':_identity_rate(rows[main],rows['raw'],allflag),'main_all_flagged_delegation_rate':sum(_f(rows[main][t],'decisive_frontier_icer_structural_domain_delegated',0)>=.5 for t in allflag)/len(allflag) if allflag else float('nan'),'main_safe_guard_block_rate':_guard_block_rate(rows[main],safe)}
 frozen_keys=['selected_local_anchor_action_match','pair_full_interface_action_match','local_pair_full_interface_action_match','evidence_certificate_fraction','decision_budget_atom_count','proposal_candidate_atom_count','proposal_decisive_atom_recall','selected_decisive_atom_recall','effective_selected_decisive_atom_recall']; frozen={k:bool(math.isfinite(_f(metrics['raw'],k)) and math.isfinite(_f(metrics[main],k)) and abs(_f(metrics['raw'],k)-_f(metrics[main],k))<=1e-6) for k in frozen_keys}
 c=ed[main]; instrumentation=bool(c['scene_count']>=450 and c['admissible_edge_count']>=1800 and c['direct_counterfactual_dominance_edge_count']>=400 and _f(metrics[main],'decisive_frontier_value_complete_star_coverage')>=.99 and all(frozen.values())); structural_ok=bool(len(allflag)>=3 and structural['main_all_flagged_final_identity_vs_raw']==1 and structural['main_all_flagged_delegation_rate']==1 and structural['main_safe_guard_block_rate']<=.001); candidate=bool(c['multi_admissible_proposal_rate']>=.25 and c['admissible_candidates_per_proposal_mean']>=3); reliability=bool(c['support_auc']>=.65 and c['direct_counterfactual_dominance_auc']>=.70)
 ret=path[main]['admissible_incumbent_to_anchor']; rep=path[main]['direct_incumbent_to_alternative']; asymmetric=bool(ret['count']==0 and abs(ret['regret_delta_sum'])<=1e-9); path_safe=bool(rep['count']>=8 and rep['regret_delta_sum']<=0); recovery=bool(c['alternative_recovery_rate']>=.03 and c['alternative_recovery_precision']>=.80 and c['direct_incumbent_replacement_rate']>=.02 and c['direct_incumbent_replacement_precision']>=.60 and c['direct_incumbent_opportunity_capture_rate']>=.08 and c['alternative_teacher_margin_mean']>0)
 downside_inc=_increment('aggregate_downside','aggregate_meanse',M,ed,path); attribution_inc=_increment('attribution_downside','aggregate_downside',M,ed,path); attr_meanse_diag=_increment('attribution_meanse','aggregate_meanse',M,ed,path)
 # V24 asymmetric preservation contract: the learned operator may only replace an already-admissible incumbent.  Therefore old "harmful -5pp / flip<raw" abstention gates are structurally incompatible.  Pre-register non-degradation plus path-level non-harm instead; V20 remains the abstention-preservation control.
 preservation=bool(M[main]['harmful']<=M['raw']['harmful']+.005 and M[main]['flip']<=M['raw']['flip']+.01 and path_safe and asymmetric)
 endpoint=bool(M[main]['match']>=anchor['match']+.005 and M[main]['regret']<=M['raw']['regret']*1.02 and M[main]['regret']<=M['v20']['regret']*1.02)
 full=bool(instrumentation and structural_ok and candidate and reliability and asymmetric and path_safe and recovery and downside_inc and attribution_inc and preservation and endpoint)
 if full: nxt='split_pass_freeze_attribution_resolved_downside_certificate_wait_for_second_block'
 elif not instrumentation:nxt='engineering_or_frozen_interface_failure'
 elif not structural_ok:nxt='deployment_semantics_failure_do_not_change_regret_certificate'
 elif not candidate or not reliability:nxt='frozen_frontier_or_support_dominance_regression'
 elif not asymmetric:nxt='incumbent_default_contract_broken_engineering_failure'
 elif not path_safe:nxt='attribution_resolved_downside_certificate_failed_selected_tail_do_not_tune_K_or_multiplier_audit_attribution_aliasing'
 elif not recovery:nxt='path_safe_but_too_conservative_do_not_tune_zero_or_K_audit_support_coverage'
 elif not downside_inc:nxt='downside_certificate_not_incremental_over_meanSE_do_not_tune_multiplier'
 elif not attribution_inc:nxt='full_attribution_spectrum_not_incremental_keep_aggregate_downside_candidate_do_not_tune_metric_weights'
 elif not preservation:nxt='replacement_path_safe_but_raw_harm_non_degradation_failed_audit_action_changes'
 else:nxt='mechanism_pass_but_endpoint_fail_audit_remaining_teacher_regret_tail'
 repout={'audit':'v64_3_24_eaf_icer_arc_split','split_name':a.split_name,'main_algorithm_arm':main,'full_split_pass':full,'instrumentation_valid':instrumentation,'deployment_alignment':structural_ok,'candidate_support_valid':candidate,'frozen_support_dominance_signal':reliability,'incumbent_default_invariant':asymmetric,'selected_replacement_path_nonharmful':path_safe,'counterfactual_recovery_mechanism':recovery,'downside_certificate_incremental':downside_inc,'attribution_resolved_incremental':attribution_inc,'attribution_meanse_incremental_diagnostic':attr_meanse_diag,'asymmetric_preservation_non_degradation':preservation,'endpoint_gain':endpoint,'next_action':nxt,'structural':structural,'edge_diagnostics':ed,'path_diagnostics':path,'metrics':{'anchor':anchor,**M},'frozen_interface':frozen,'thresholds':{'support_auc_min':.65,'dominance_auc_min':.70,'incumbent_to_anchor_count_max':0,'replacement_path_regret_sum_max':0,'direct_precision_min':.60,'capture_min':.08,'raw_harmful_non_degradation_tolerance':.005,'raw_flip_non_degradation_tolerance':.01,'match_over_anchor_min':.005,'regret_tolerance':.02,'K_and_downside_multiplier':'fixed_train_only_no_validation_sweep'}}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(repout,indent=2,sort_keys=True));print(json.dumps(repout,indent=2,sort_keys=True))
if __name__=='__main__':main()
