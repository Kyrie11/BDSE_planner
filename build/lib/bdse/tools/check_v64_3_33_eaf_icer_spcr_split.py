from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _auc, _f, _icer_edge_diag, _metric_pack
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _identity_rate, _load_rows
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _replacement_tail_diag
from bdse.tools.check_v64_3_30_eaf_icer_fbic_split import _query_diag

CAT=-0.5; CAPTURE_GAIN_MIN=0.03; EPS=1e-12


def _structural(rows, raw, flagged):
    return {'all_flagged_scene_count':len(flagged),'final_identity_vs_raw':_identity_rate(rows,raw,flagged),'icer_structural_delegation_rate':float(np.mean([_f(rows[t],'decisive_frontier_icer_structural_domain_delegated',0.0) for t in flagged])) if flagged else 1.0}


def _containment(rank, main, tokens):
    mismatch=[]; accept_bad=[]; fallback=[]; rp=acc=0
    for t in sorted(tokens):
        rr, mr=rank[t],main[t]
        rex=_f(rr,'decisive_frontier_icer_scir_proposal_exists',0)>=.5; mex=_f(mr,'decisive_frontier_icer_scir_proposal_exists',0)>=.5
        ra=int(round(_f(rr,'decisive_frontier_icer_scir_proposal_action',-999))); ma=int(round(_f(mr,'decisive_frontier_icer_scir_proposal_action',-999)))
        if (rex,ra)!=(mex,ma): mismatch.append(t)
        if not mex: continue
        rp+=1; baseline=int(round(_f(mr,'decisive_frontier_icer_baseline_action',-999))); sel=int(round(_f(mr,'decisive_frontier_icer_selected_action',baseline))); ok=_f(mr,'decisive_frontier_icer_scir_certificate_accepted',0)>=.5
        if ok:
            acc+=1
            if sel!=ma: accept_bad.append(t)
        elif sel!=baseline: fallback.append(t)
    return {'rank_proposal_count':rp,'main_accepted_proposal_count':acc,'main_accept_rate_given_rank_proposal':float(acc/max(rp,1)),'proposal_identity_mismatch_count':len(mismatch),'accepted_same_proposal_violation_count':len(accept_bad),'veto_to_incumbent_default_no_fallback_violation_count':len(fallback),'monotone_selected_policy_containment_valid':not mismatch and not accept_bad and not fallback,'example_proposal_mismatch':mismatch[:10],'example_accept_violation':accept_bad[:10],'example_fallback':fallback[:10]}


def _policy_diag(edge_path:str, allowed:set[str])->dict[str,Any]:
    groups={}
    for line in Path(edge_path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r=json.loads(line); t=str(r.get('scenario_token',''))
        if t in allowed: groups.setdefault(t,[]).append(r)
    labels=[]; scores=[]; ys=[]; proposal_y=[]; opp=cap=noopp=0; cats=[]
    for t,rs in groups.items():
        if not rs: continue
        inc=int(rs[0].get('raw_top_action',-1)); by={int(r.get('challenger_action',-2)):r for r in rs}; ir=by.get(inc)
        if ir is None or _f(ir,'icer_admissible',0)<.5: continue
        itm=_f(ir,'teacher_margin'); alts=[]
        if not math.isfinite(itm): continue
        for r in rs:
            a=int(r.get('challenger_action',-2))
            if a==inc or _f(r,'icer_admissible',0)<.5 or _f(r,'icer_support_logit',-math.inf)<=0: continue
            y=_f(r,'teacher_margin')-itm; s=_f(r,'icer_scir_predicted_improvement')
            if math.isfinite(y) and math.isfinite(s): labels.append(int(y>0)); scores.append(s); ys.append(y); alts.append((a,y,s))
        has=any(y>0 for _,y,_ in alts); opp+=int(has)
        exists=_f(rs[0],'icer_scir_proposal_exists',0)>=.5; prop=int(rs[0].get('icer_scir_proposal_action',inc))
        if exists and prop in by:
            py=_f(by[prop],'teacher_margin')-itm
            if math.isfinite(py): proposal_y.append(py); cap+=int(has and py>0); noopp+=int(not has); cats.append(int(py<=CAT))
    a=np.asarray(proposal_y,float); n=np.minimum(a,0) if a.size else a
    corr=float(np.corrcoef(np.asarray(ys),np.asarray(scores))[0,1]) if len(ys)>=2 and np.std(ys)>0 and np.std(scores)>0 else float('nan')
    return {'edge_count':len(labels),'score_sign_auc':_auc(labels,scores),'score_vs_teacher_pearson':corr,'proposal_count':int(a.size),'proposal_positive_count':int((a>0).sum()) if a.size else 0,'proposal_precision':float((a>0).mean()) if a.size else float('nan'),'proposal_teacher_improvement_sum':float(a.sum()) if a.size else 0.0,'proposal_worst':float(a.min()) if a.size else float('nan'),'proposal_negative_rms':float(np.sqrt(np.mean(n*n))) if a.size else 0.0,'proposal_catastrophic_count':int(sum(cats)),'positive_opportunity_scene_count':int(opp),'positive_capture_rate':float(cap/max(opp,1)),'no_positive_opportunity_false_intervention_count':int(noopp)}


def main():
    ap=argparse.ArgumentParser(description='Audit one V64.3.33 SPCR fresh block.')
    ap.add_argument('--split-name',required=True)
    for tag in ['raw','v20','preserve','mean','rank','main']:
        ap.add_argument(f'--{tag}-metrics',required=True); ap.add_argument(f'--{tag}-rows',required=True)
        if tag!='raw': ap.add_argument(f'--{tag}-edges',required=True)
    ap.add_argument('--output',required=True); a=ap.parse_args()
    tags=['raw','v20','preserve','mean','rank','main']; metrics={t:json.load(open(getattr(a,t+'_metrics'))) for t in tags}; rows={t:_load_rows(getattr(a,t+'_rows')) for t in tags}; toks=set(rows['raw'])
    if len(toks)!=500 or any(set(rows[t])!=toks for t in tags[1:]): raise SystemExit('STOP DATA: V33 six arms must contain exact paired 500 scenes')
    flagged={t for t in toks if _f(rows['raw'][t],'all_actions_safety_flagged_rate',0)>=.5}; safe=toks-flagged
    q={t:_query_diag(rows['v20'],rows[t],toks) for t in ['preserve','mean','rank','main']}; query_ok=all(v['all_query_counts_exact_scene_parity'] for v in q.values())
    structural={t:_structural(rows[t],rows['raw'],flagged) for t in ['v20','preserve','mean','rank','main']}; struct_ok=all((not flagged) or (structural[t]['final_identity_vs_raw']==1.0 and structural[t]['icer_structural_delegation_rate']==1.0) for t in ['preserve','mean','rank','main'])
    cont=_containment(rows['rank'],rows['main'],safe)
    edge={t:_icer_edge_diag(Path(getattr(a,t+'_edges')),safe) for t in ['v20','preserve','mean','rank','main']}
    tail={t:_replacement_tail_diag(rows['raw'],rows[t],getattr(a,t+'_edges'),safe) for t in ['v20','preserve','mean','rank','main']}
    policy={t:_policy_diag(getattr(a,t+'_edges'),safe) for t in ['mean','rank','main']}; M={t:_metric_pack(metrics[t]) for t in tags}
    prescap=float(edge['preserve']['direct_incumbent_opportunity_capture_rate']); maincap=float(edge['main']['direct_incumbent_opportunity_capture_rate'])
    ordering=bool(policy['rank']['no_positive_opportunity_false_intervention_count'] < policy['mean']['no_positive_opportunity_false_intervention_count'] and policy['rank']['proposal_catastrophic_count'] <= policy['mean']['proposal_catastrophic_count'] and (policy['rank']['proposal_teacher_improvement_sum'] > policy['mean']['proposal_teacher_improvement_sum']+1e-9 or policy['rank']['proposal_catastrophic_count'] < policy['mean']['proposal_catastrophic_count']))
    coverage=bool(math.isfinite(maincap) and math.isfinite(prescap) and maincap>=prescap+CAPTURE_GAIN_MIN-1e-12)
    main_tail=bool(tail['main']['count']>=8 and tail['main']['teacher_improvement_sum']>=-1e-9 and math.isfinite(tail['main']['teacher_improvement_worst']) and tail['main']['teacher_improvement_worst']>CAT and tail['main']['teacher_negative_rms']<=tail['preserve']['teacher_negative_rms']+EPS and edge['main']['direct_incumbent_replacement_precision']>=edge['preserve']['direct_incumbent_replacement_precision']-EPS)
    cert=bool(cont['monotone_selected_policy_containment_valid'] and tail['main']['teacher_negative_rms']<=tail['rank']['teacher_negative_rms']+EPS and (not math.isfinite(tail['rank']['teacher_improvement_worst']) or tail['main']['teacher_improvement_worst']>=tail['rank']['teacher_improvement_worst']-EPS) and edge['main']['direct_incumbent_replacement_precision']>=edge['rank']['direct_incumbent_replacement_precision']-EPS)
    endp=bool(M['main']['match']>=M['preserve']['match']-.002 and M['main']['regret']<=M['preserve']['regret']*1.005 and M['main']['match']>=M['v20']['match']-.002 and M['main']['regret']<=M['v20']['regret']*1.005)
    eng=bool(query_ok and struct_ok and cont['monotone_selected_policy_containment_valid']); full=bool(eng and ordering and coverage and main_tail and cert and endp)
    if not eng: nxt='STOP_fix_SPCR_engineering_or_containment_before_scientific_interpretation'
    elif not ordering: nxt='structured_incumbent_augmented_ordering_does_not_reduce_no_opportunity_false_interventions_and_selected_tail_vs_corrected_mean_stop_do_not_add_more_heads'
    elif not coverage: nxt='structured_ordering_signal_exists_but_selected_policy_certificate_does_not_add_3pp_capture_over_preserve_stop_do_not_alpha_sweep'
    elif not main_tail: nxt='SPCR_coverage_exists_but_direct_selected_tail_fails_stop_do_not_threshold_rescue'
    elif not cert: nxt='selected_policy_certificate_not_incremental_over_structured_ranker_stop_revisit_certificate_target_not_features'
    elif not endp: nxt='direct_SPCR_passes_but_endpoint_does_not_convert_audit_runtime_path_composition'
    else: nxt='if_second_fresh_block_also_passes_freeze_SPCR_and_run_one_independent_full_validation_reproduction'
    rep={'audit':'v64_3_33_eaf_icer_spcr_split','split_name':a.split_name,'full_split_pass':full,'engineering_valid':eng,'structured_ordering_mechanism_pass':ordering,'main_capture_gain_over_preserve':maincap-prescap if math.isfinite(maincap) and math.isfinite(prescap) else float('nan'),'main_meaningful_coverage':coverage,'main_tail_pass':main_tail,'certificate_incremental':cert,'endpoint_noninferior':endp,'next_action':nxt,'query_parity':q,'structural':structural,'monotone_selected_policy_containment':cont,'edge_diagnostics':edge,'policy_diagnostics':policy,'direct_selected_path_tail':tail,'metrics':M,'frozen_contract':{'catastrophic_threshold':CAT,'capture_gain_min':CAPTURE_GAIN_MIN,'main_selected_count_min':8,'no_AB_pooling':True,'no_alpha_lambda_feature_threshold_sweep':True},'causal_note':'MEAN is the corrected V32.1 edge-mean control; RANK changes only the selector training objective to incumbent-augmented teacher-best-vs-rivals pair gaps; MAIN keeps exactly the same RANK proposal and may only accept it or return incumbent.'}
    Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(rep,indent=2,sort_keys=True))

if __name__=='__main__': main()
