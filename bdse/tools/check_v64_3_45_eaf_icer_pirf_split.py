from __future__ import annotations
import argparse,json,math
from pathlib import Path
from bdse.tools.check_v64_3_39_eaf_icer_cfsr_split import EPS,_containment,_f,_icer_edge_diag,_load_rows,_mechanism_gate,_metric_pack,_query_diag,_replacement_tail_diag,_selected_policy_diag,_structural

TAGS=['raw','v20','preserve','rsmr','quality','cv_occ','local_rf','plan_rf']; VALUE_TAGS=['quality','cv_occ','local_rf','plan_rf']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split-name',required=True);ap.add_argument('--preferred-arm',required=True,choices=['cv_occ','local_rf','plan_rf'])
    for t in TAGS:
        x=t.replace('_','-');ap.add_argument(f'--{x}-metrics',dest=t+'_metrics',required=True);ap.add_argument(f'--{x}-rows',dest=t+'_rows',required=True)
        if t!='raw':ap.add_argument(f'--{x}-edges',dest=t+'_edges',required=True)
    ap.add_argument('--output',required=True);a=ap.parse_args();metrics={t:json.load(open(getattr(a,t+'_metrics'))) for t in TAGS};rows={t:_load_rows(getattr(a,t+'_rows')) for t in TAGS};toks=set(rows['raw'])
    if len(toks)!=500 or any(set(rows[t])!=toks for t in TAGS[1:]):raise SystemExit('STOP DATA: V45 arms must contain exact paired 500 scenes')
    flagged={t for t in toks if _f(rows['raw'][t],'all_actions_safety_flagged_rate',0)>=.5};safe=toks-flagged
    q={t:_query_diag(rows['v20'],rows[t],toks) for t in TAGS[2:]};struct={t:_structural(rows[t],rows['raw'],flagged) for t in ['v20']+TAGS[2:]};cont={t:_containment(rows['rsmr'],rows[t],safe) for t in VALUE_TAGS}
    eng=all(v['all_query_counts_exact_scene_parity'] for v in q.values()) and all((not flagged) or (struct[t]['final_identity_vs_raw']==1.0 and struct[t]['icer_structural_delegation_rate']==1.0) for t in TAGS[2:]) and all(x['monotone_selected_policy_containment_valid'] for x in cont.values())
    edge={t:_icer_edge_diag(Path(getattr(a,t+'_edges')),safe) for t in TAGS if t!='raw'};policy={t:_selected_policy_diag(getattr(a,t+'_edges'),safe) for t in ['rsmr']+VALUE_TAGS};g={t:_mechanism_gate(policy[t],policy['rsmr']) for t in VALUE_TAGS};M={t:_metric_pack(metrics[t]) for t in TAGS};tail={t:_replacement_tail_diag(rows['raw'],rows[t],getattr(a,t+'_edges'),safe) for t in TAGS if t!='raw'}
    arm=a.preferred_arm;pc=float(edge['preserve']['direct_incumbent_opportunity_capture_rate']);mc=float(edge[arm]['direct_incumbent_opportunity_capture_rate']);coverage=math.isfinite(pc) and math.isfinite(mc) and mc>=pc+.03-EPS;endpoint=M[arm]['match']>=M['preserve']['match']-.002 and M[arm]['regret']<=M['preserve']['regret']*1.005 and M[arm]['match']>=M['v20']['match']-.002 and M[arm]['regret']<=M['v20']['regret']*1.005;full=bool(eng and g[arm]['pass'] and coverage and endpoint)
    rep={'audit':'v64_3_45_eaf_icer_pirf_split','split_name':a.split_name,'preferred_arm':arm,'full_split_pass':full,'engineering_valid':eng,'mechanism_gates':g,'preferred_capture_gain_over_preserve':mc-pc if math.isfinite(mc) and math.isfinite(pc) else float('nan'),'endpoint_noninferior':endpoint,'query_parity':q,'structural':struct,'containment':cont,'edge_diagnostics':edge,'selected_policy_diagnostics':policy,'direct_selected_path_tail':tail,'metrics':M,'frozen_contract':{'RSMR_winner_frozen':True,'V42_QUALITY_frozen':True,'V44_ungated_occupancy_support_retained':True,'response_field_runtime_no_logged_future':True,'no_selected_translation':True,'no_second_best_fallback':True,'no_AB_pooling':True}}
    Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');print(json.dumps(rep,indent=2,sort_keys=True))
if __name__=='__main__':main()
