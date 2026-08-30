from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from bdse.tools.check_v64_3_19_eaf_icer_screen import _f,_icer_edge_diag,_metric_pack
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _identity_rate,_load_rows
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _replacement_tail_diag
from bdse.tools.check_v64_3_30_eaf_icer_fbic_split import _query_diag
from bdse.tools.check_v64_3_33_eaf_icer_spcr_split import _containment,_policy_diag
CAT=-0.5; CAPTURE_GAIN_MIN=0.03; CONTEXT_GAIN_MIN=0.20; EPS=1e-12

def _structural(rows,raw,flagged): return {'all_flagged_scene_count':len(flagged),'final_identity_vs_raw':_identity_rate(rows,raw,flagged),'icer_structural_delegation_rate':float(np.mean([_f(rows[t],'decisive_frontier_icer_structural_domain_delegated',0.0) for t in flagged])) if flagged else 1.0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split-name',required=True)
    tags=['raw','v20','preserve','mean','rsmr','factor','rank','main']
    for tag in tags:
        ap.add_argument(f'--{tag}-metrics',required=True); ap.add_argument(f'--{tag}-rows',required=True)
        if tag!='raw': ap.add_argument(f'--{tag}-edges',required=True)
    ap.add_argument('--output',required=True); a=ap.parse_args()
    metrics={t:json.load(open(getattr(a,t+'_metrics'))) for t in tags}; rows={t:_load_rows(getattr(a,t+'_rows')) for t in tags}; toks=set(rows['raw'])
    if len(toks)!=500 or any(set(rows[t])!=toks for t in tags[1:]): raise SystemExit('STOP DATA: V35 eight arms must contain exact paired 500 scenes')
    flagged={t for t in toks if _f(rows['raw'][t],'all_actions_safety_flagged_rate',0)>=.5}; safe=toks-flagged
    q={t:_query_diag(rows['v20'],rows[t],toks) for t in tags if t not in {'raw','v20'}}; query_ok=all(v['all_query_counts_exact_scene_parity'] for v in q.values())
    structural={t:_structural(rows[t],rows['raw'],flagged) for t in tags if t!='raw'}; struct_ok=all((not flagged) or (structural[t]['final_identity_vs_raw']==1.0 and structural[t]['icer_structural_delegation_rate']==1.0) for t in tags if t not in {'raw','v20'})
    cont=_containment(rows['rank'],rows['main'],safe)
    edge={t:_icer_edge_diag(Path(getattr(a,t+'_edges')),safe) for t in tags if t!='raw'}; tail={t:_replacement_tail_diag(rows['raw'],rows[t],getattr(a,t+'_edges'),safe) for t in tags if t!='raw'}
    policy={t:_policy_diag(getattr(a,t+'_edges'),safe) for t in ['mean','rsmr','factor','rank','main']}; M={t:_metric_pack(metrics[t]) for t in tags}
    f_no=float(policy['factor']['no_positive_opportunity_false_intervention_count']); r_no=float(policy['rank']['no_positive_opportunity_false_intervention_count'])
    context_boundary=bool(r_no <= (1.0-CONTEXT_GAIN_MIN)*f_no+EPS)
    context_ordering=bool(policy['rank']['positive_capture_rate']>=policy['factor']['positive_capture_rate']-0.03-EPS and policy['rank']['proposal_teacher_improvement_sum']>=-EPS)
    context_tail=bool(policy['rank']['proposal_catastrophic_count']<policy['rsmr']['proposal_catastrophic_count'] and tail['rank']['teacher_negative_rms']<=tail['rsmr']['teacher_negative_rms']+EPS)
    rank_pass=bool(context_boundary and context_ordering and context_tail)
    prescap=float(edge['preserve']['direct_incumbent_opportunity_capture_rate']); maincap=float(edge['main']['direct_incumbent_opportunity_capture_rate']); coverage=bool(math.isfinite(maincap) and math.isfinite(prescap) and maincap>=prescap+CAPTURE_GAIN_MIN-EPS)
    main_tail=bool(tail['main']['count']>=8 and tail['main']['teacher_improvement_sum']>=-EPS and math.isfinite(tail['main']['teacher_improvement_worst']) and tail['main']['teacher_improvement_worst']>CAT and tail['main']['teacher_negative_rms']<=tail['preserve']['teacher_negative_rms']+EPS and edge['main']['direct_incumbent_replacement_precision']>=edge['preserve']['direct_incumbent_replacement_precision']-EPS)
    cert=bool(cont['monotone_selected_policy_containment_valid'] and tail['main']['teacher_negative_rms']<=tail['rank']['teacher_negative_rms']+EPS and (not math.isfinite(tail['rank']['teacher_improvement_worst']) or tail['main']['teacher_improvement_worst']>=tail['rank']['teacher_improvement_worst']-EPS) and edge['main']['direct_incumbent_replacement_precision']>=edge['rank']['direct_incumbent_replacement_precision']-EPS)
    endp=bool(M['main']['match']>=M['preserve']['match']-.002 and M['main']['regret']<=M['preserve']['regret']*1.005 and M['main']['match']>=M['v20']['match']-.002 and M['main']['regret']<=M['v20']['regret']*1.005)
    eng=bool(query_ok and struct_ok and cont['monotone_selected_policy_containment_valid']); full=bool(eng and rank_pass and coverage and main_tail and cert and endp)
    rep={'audit':'v64_3_35_eaf_icer_fbcsr_split','split_name':a.split_name,'full_split_pass':full,'engineering_valid':eng,'context_boundary_gain_vs_factorized_delta':context_boundary,'context_preserves_opportunity_ordering':context_ordering,'context_tail_gain_vs_v34_rsmr':context_tail,'rank_mechanism_pass':rank_pass,'main_capture_gain_over_preserve':maincap-prescap if math.isfinite(maincap) and math.isfinite(prescap) else float('nan'),'main_meaningful_coverage':coverage,'main_tail_pass':main_tail,'certificate_incremental':cert,'endpoint_noninferior':endp,'query_parity':q,'structural':structural,'monotone_selected_policy_containment':cont,'edge_diagnostics':edge,'policy_diagnostics':policy,'direct_selected_path_tail':tail,'metrics':M,'frozen_contract':{'catastrophic_threshold':CAT,'capture_gain_min':CAPTURE_GAIN_MIN,'context_noop_reduction_fraction_min':CONTEXT_GAIN_MIN,'no_AB_pooling':True,'no_runtime_threshold_or_lambda_alpha_sweep':True}}
    Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n'); print(json.dumps(rep,indent=2,sort_keys=True))
if __name__=='__main__': main()
