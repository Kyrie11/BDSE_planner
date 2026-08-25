from __future__ import annotations
import argparse,json,math
from pathlib import Path
from bdse.tools.fit_v64_3_38_eaf_icer_davr import CAPTURE_TOL
from bdse.tools.check_v64_3_39_eaf_icer_cfsr_split import EPS,_containment,_f,_icer_edge_diag,_load_rows,_mechanism_gate,_metric_pack,_query_diag,_replacement_tail_diag,_selected_policy_diag,_structural

def main():
 ap=argparse.ArgumentParser(); tags=['raw','v20','preserve','rsmr','epv_raw','quality','risk','joint_raw','ovdr']; ap.add_argument('--split-name',required=True)
 for tag in tags:
  x=tag.replace('_','-'); ap.add_argument(f'--{x}-metrics',dest=f'{tag}_metrics',required=True); ap.add_argument(f'--{x}-rows',dest=f'{tag}_rows',required=True)
  if tag!='raw': ap.add_argument(f'--{x}-edges',dest=f'{tag}_edges',required=True)
 ap.add_argument('--output',required=True); a=ap.parse_args(); metrics={t:json.load(open(getattr(a,t+'_metrics'))) for t in tags}; rows={t:_load_rows(getattr(a,t+'_rows')) for t in tags}; toks=set(rows['raw'])
 if len(toks)!=500 or any(set(rows[t])!=toks for t in tags[1:]): raise SystemExit('STOP DATA: V42 arms must contain exact paired 500 scenes')
 flagged={t for t in toks if _f(rows['raw'][t],'all_actions_safety_flagged_rate',0)>=.5}; safe=toks-flagged; valtags=['preserve','rsmr','epv_raw','quality','risk','joint_raw','ovdr']
 q={t:_query_diag(rows['v20'],rows[t],toks) for t in valtags}; structural={t:_structural(rows[t],rows['raw'],flagged) for t in ['v20']+valtags}; cont={t:_containment(rows['rsmr'],rows[t],safe) for t in ['epv_raw','quality','risk','joint_raw','ovdr']}
 eng=all(v['all_query_counts_exact_scene_parity'] for v in q.values()) and all((not flagged) or (structural[t]['final_identity_vs_raw']==1.0 and structural[t]['icer_structural_delegation_rate']==1.0) for t in valtags) and all(x['monotone_selected_policy_containment_valid'] for x in cont.values())
 edge={t:_icer_edge_diag(Path(getattr(a,t+'_edges')),safe) for t in tags if t!='raw'}; policy={t:_selected_policy_diag(getattr(a,t+'_edges'),safe) for t in ['rsmr','epv_raw','quality','risk','joint_raw','ovdr']}; gates={t:_mechanism_gate(policy[t],policy['rsmr']) for t in ['epv_raw','quality','risk','joint_raw','ovdr']}; M={t:_metric_pack(metrics[t]) for t in tags}; tail={t:_replacement_tail_diag(rows['raw'],rows[t],getattr(a,t+'_edges'),safe) for t in tags if t!='raw'}
 pc=float(edge['preserve']['direct_incumbent_opportunity_capture_rate']); mc=float(edge['ovdr']['direct_incumbent_opportunity_capture_rate']); coverage=math.isfinite(pc) and math.isfinite(mc) and mc>=pc+.03-EPS; endp=M['ovdr']['match']>=M['preserve']['match']-.002 and M['ovdr']['regret']<=M['preserve']['regret']*1.005 and M['ovdr']['match']>=M['v20']['match']-.002 and M['ovdr']['regret']<=M['v20']['regret']*1.005; full=bool(eng and gates['ovdr']['pass'] and coverage and endp)
 if not eng:nxt='STOP_fix_V42_engineering_before_scientific_interpretation'
 elif gates['epv_raw']['pass'] and not gates['ovdr']['pass']:nxt='endpoint_potential_alone_suffices_discard_observable_expansion'
 elif gates['quality']['pass'] and not gates['risk']['pass'] and not gates['ovdr']['pass']:nxt='trajectory_quality_observable_is_primary_mediator_keep_quality_only'
 elif gates['risk']['pass'] and not gates['quality']['pass'] and not gates['ovdr']['pass']:nxt='physical_risk_observable_is_primary_mediator_keep_risk_only'
 elif not gates['ovdr']['existence_and_capture']:nxt='deployment_observables_do_not_close_zero_boundary_require_future_sensitive_value_observable'
 elif not gates['ovdr']['hard_tail']:nxt='observable_value_recovers_capture_but_selected_tail_requires_future_sensitive_downside_observable'
 elif not coverage:nxt='OVDR_direct_gate_passes_but_gain_over_preserve_insufficient'
 elif not endp:nxt='OVDR_direct_gate_passes_but_endpoint_noninferiority_fails'
 else:nxt='if_second_fresh_block_also_passes_freeze_OVDR_and_run_full_validation_and_closed_loop'
 rep={'audit':'v64_3_42_eaf_icer_ovdr_split','split_name':a.split_name,'full_split_pass':full,'engineering_valid':eng,'mechanism_gates':gates,'ovdr_capture_gain_over_preserve':mc-pc if math.isfinite(mc) and math.isfinite(pc) else float('nan'),'endpoint_noninferior':endp,'next_action':nxt,'query_parity':q,'structural':structural,'containment':cont,'edge_diagnostics':edge,'selected_policy_diagnostics':policy,'direct_selected_path_tail':tail,'metrics':M,'frozen_contract':{'RSMR_winner_frozen':True,'new_observables_label_free_deployment_only':True,'quality_vs_risk_ablation_pre_registered':True,'CAL500_translation_unit_slope':True,'no_second_best_fallback':True,'no_AB_pooling':True}}
 Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n'); print(json.dumps(rep,indent=2,sort_keys=True))
if __name__=='__main__':main()
