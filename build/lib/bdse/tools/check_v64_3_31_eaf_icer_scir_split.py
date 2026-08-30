from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _auc, _f, _icer_edge_diag, _metric_pack
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _identity_rate, _load_rows
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _replacement_tail_diag
from bdse.tools.check_v64_3_30_eaf_icer_fbic_split import _query_diag

CAT = -0.5
RANK_CAPTURE_GAIN_MIN = 0.03


def _scir_subset_diag(rank: dict[str,dict[str,Any]], main: dict[str,dict[str,Any]], tokens: set[str]) -> dict[str,Any]:
    proposal_mismatch=[]; containment=[]; fallback=[]; main_accept=rank_proposals=0
    for t in sorted(tokens):
        rr=rank[t]; mr=main[t]
        rp=int(round(_f(rr,'decisive_frontier_icer_scir_proposal_action',-999)))
        mp=int(round(_f(mr,'decisive_frontier_icer_scir_proposal_action',-999)))
        rex=_f(rr,'decisive_frontier_icer_scir_proposal_exists',0)>=.5
        mex=_f(mr,'decisive_frontier_icer_scir_proposal_exists',0)>=.5
        if (rp, rex)!=(mp, mex): proposal_mismatch.append(t)
        if not mex: continue
        rank_proposals+=1
        baseline=int(round(_f(mr,'decisive_frontier_icer_baseline_action',-999)))
        msel=int(round(_f(mr,'decisive_frontier_icer_selected_action',-999)))
        accepted=_f(mr,'decisive_frontier_icer_scir_certificate_accepted',0)>=.5
        if accepted:
            main_accept+=1
            if msel!=mp: containment.append(t)
        else:
            if msel!=baseline: fallback.append(t)
    return {
        'rank_proposal_count':rank_proposals,
        'main_accepted_proposal_count':main_accept,
        'main_accept_rate_given_rank_proposal':float(main_accept/max(rank_proposals,1)),
        'proposal_identity_mismatch_count':len(proposal_mismatch),
        'accepted_same_proposal_violation_count':len(containment),
        'veto_to_incumbent_default_no_fallback_violation_count':len(fallback),
        'monotone_selected_proposal_containment_valid':not proposal_mismatch and not containment and not fallback,
        'example_proposal_mismatches':proposal_mismatch[:10],
        'example_containment_violations':containment[:10],
        'example_fallback_violations':fallback[:10],
    }


def _structural(rows: dict[str,dict[str,Any]], raw: dict[str,dict[str,Any]], flagged: set[str]) -> dict[str,Any]:
    return {
        'all_flagged_scene_count':len(flagged),
        'final_identity_vs_raw':_identity_rate(rows,raw,flagged),
        'icer_structural_delegation_rate':float(np.mean([_f(rows[t],'decisive_frontier_icer_structural_domain_delegated',0.0) for t in flagged])) if flagged else 1.0,
    }


def _scir_prediction_diag(edge_path: str, allowed: set[str]) -> dict[str,Any]:
    groups: dict[str,list[dict[str,Any]]]={}
    for line in Path(edge_path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r=json.loads(line); tok=str(r.get('scenario_token',''))
        if tok in allowed: groups.setdefault(tok,[]).append(r)
    yy=[]; mu=[]; proposal_y=[]; opp=cap=0
    for rs in groups.values():
        if not rs: continue
        inc=int(rs[0].get('raw_top_action',-1)); by={int(r.get('challenger_action',-2)):r for r in rs}; ir=by.get(inc)
        if ir is None or _f(ir,'icer_admissible',0.0)<.5: continue
        itm=_f(ir,'teacher_margin');
        if not math.isfinite(itm): continue
        alts=[]
        for r in rs:
            act=int(r.get('challenger_action',-2))
            if act==inc or _f(r,'icer_admissible',0.0)<.5 or _f(r,'icer_support_logit',-math.inf)<=0.0: continue
            y=_f(r,'teacher_margin')-itm; s=_f(r,'icer_scir_predicted_improvement')
            if math.isfinite(y) and math.isfinite(s):
                yy.append(int(y>0)); mu.append(s); alts.append((act,y,s))
        has_opp=any(y>0 for _,y,_ in alts); opp+=int(has_opp)
        exists=_f(rs[0],'icer_scir_proposal_exists',0.0)>=.5
        prop=int(rs[0].get('icer_scir_proposal_action',inc))
        if exists and prop in by:
            py=_f(by[prop],'teacher_margin')-itm
            if math.isfinite(py): proposal_y.append(py); cap+=int(has_opp and py>0)
    arr=np.asarray(proposal_y,dtype=np.float64); neg=np.minimum(arr,0.0) if arr.size else arr
    ycont=[]; mcont=[]
    # Reconstruct continuous vectors for correlation without a second parsing pass.
    for rs in groups.values():
        if not rs: continue
        inc=int(rs[0].get('raw_top_action',-1)); by={int(r.get('challenger_action',-2)):r for r in rs}; ir=by.get(inc)
        if ir is None or _f(ir,'icer_admissible',0.0)<.5: continue
        itm=_f(ir,'teacher_margin')
        for r in rs:
            act=int(r.get('challenger_action',-2)); y=_f(r,'teacher_margin')-itm; s=_f(r,'icer_scir_predicted_improvement')
            if act!=inc and _f(r,'icer_admissible',0.0)>=.5 and _f(r,'icer_support_logit',-math.inf)>0 and math.isfinite(y) and math.isfinite(s): ycont.append(y); mcont.append(s)
    corr=float(np.corrcoef(np.asarray(ycont),np.asarray(mcont))[0,1]) if len(ycont)>=2 and np.std(ycont)>0 and np.std(mcont)>0 else float('nan')
    return {
        'direct_support_positive_edge_count':len(yy),
        'predicted_improvement_sign_auc':_auc(yy,mu),
        'predicted_vs_teacher_improvement_pearson':corr,
        'rank_proposal_count':int(arr.size),
        'rank_proposal_precision':float((arr>0).mean()) if arr.size else float('nan'),
        'rank_proposal_teacher_improvement_sum':float(arr.sum()) if arr.size else 0.0,
        'rank_proposal_worst':float(arr.min()) if arr.size else float('nan'),
        'rank_proposal_negative_rms':float(np.sqrt(np.mean(neg*neg))) if arr.size else 0.0,
        'positive_opportunity_scene_count':int(opp),
        'rank_proposal_positive_capture_rate':float(cap/max(opp,1)),
    }


def main() -> None:
    ap=argparse.ArgumentParser(description='Audit one untouched V64.3.31 SCIR block: raw/V20/PRESERVE/SCIR-rank/SCIR-main.')
    ap.add_argument('--split-name',required=True)
    for tag in ['raw','v20','preserve','rank','main']:
        ap.add_argument(f'--{tag}-metrics',required=True); ap.add_argument(f'--{tag}-rows',required=True)
        if tag!='raw': ap.add_argument(f'--{tag}-edges',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    tags=['raw','v20','preserve','rank','main']
    metrics={t:json.load(open(getattr(a,t+'_metrics'),encoding='utf-8')) for t in tags}
    rows={t:_load_rows(getattr(a,t+'_rows')) for t in tags}
    tokens=set(rows['raw'])
    if len(tokens)!=500 or any(set(rows[t])!=tokens for t in tags[1:]):
        raise SystemExit('STOP DATA: V31 all five arms must contain exact paired 500-scene identity')
    flagged={t for t in tokens if _f(rows['raw'][t],'all_actions_safety_flagged_rate',0.0)>=.5}
    safe=tokens-flagged

    q={t:_query_diag(rows['v20'],rows[t],tokens) for t in ['preserve','rank','main']}
    query_ok=all(x['all_query_counts_exact_scene_parity'] for x in q.values())
    structural={t:_structural(rows[t],rows['raw'],flagged) for t in ['v20','preserve','rank','main']}
    structural_ok=all((not flagged) or (structural[t]['final_identity_vs_raw']==1.0 and structural[t]['icer_structural_delegation_rate']==1.0) for t in ['preserve','rank','main'])
    subset=_scir_subset_diag(rows['rank'],rows['main'],safe)

    edge={t:_icer_edge_diag(Path(getattr(a,t+'_edges')),safe) for t in ['v20','preserve','rank','main']}
    tail={t:_replacement_tail_diag(rows['raw'],rows[t],getattr(a,t+'_edges'),safe) for t in ['v20','preserve','rank','main']}
    scir_pred={t:_scir_prediction_diag(getattr(a,t+'_edges'),safe) for t in ['rank','main']}
    M={t:_metric_pack(metrics[t]) for t in tags}

    rank_capture=float(edge['rank']['direct_incumbent_opportunity_capture_rate']); pres_capture=float(edge['preserve']['direct_incumbent_opportunity_capture_rate']); main_capture=float(edge['main']['direct_incumbent_opportunity_capture_rate'])
    rank_ordering_gain=bool(math.isfinite(rank_capture) and math.isfinite(pres_capture) and rank_capture>=pres_capture+RANK_CAPTURE_GAIN_MIN-1e-12)
    main_coverage=bool(math.isfinite(main_capture) and math.isfinite(pres_capture) and main_capture>=pres_capture-1e-12)
    main_tail=bool(
        tail['main']['count']>=8
        and tail['main']['teacher_improvement_sum']>=-1e-9
        and math.isfinite(tail['main']['teacher_improvement_worst']) and tail['main']['teacher_improvement_worst']>CAT
        and tail['main']['teacher_negative_rms']<=tail['preserve']['teacher_negative_rms']+1e-12
        and edge['main']['direct_incumbent_replacement_precision']>=edge['preserve']['direct_incumbent_replacement_precision']-1e-12
    )
    cert_incremental=bool(
        subset['monotone_selected_proposal_containment_valid']
        and tail['main']['teacher_negative_rms']<=tail['rank']['teacher_negative_rms']+1e-12
        and tail['main']['teacher_improvement_worst']>=tail['rank']['teacher_improvement_worst']-1e-12
        and edge['main']['direct_incumbent_replacement_precision']>=edge['rank']['direct_incumbent_replacement_precision']-1e-12
    )
    endpoint_vs_pres=bool(M['main']['match']>=M['preserve']['match']-0.002 and M['main']['regret']<=M['preserve']['regret']*1.005)
    endpoint_vs_v20=bool(M['main']['match']>=M['v20']['match']-0.002 and M['main']['regret']<=M['v20']['regret']*1.005)
    endpoint=bool(endpoint_vs_pres and endpoint_vs_v20)
    engineering=bool(query_ok and structural_ok and subset['monotone_selected_proposal_containment_valid'])
    full=bool(engineering and rank_ordering_gain and main_coverage and main_tail and cert_incremental and endpoint)
    if not engineering: nxt='STOP_fix_SCIR_engineering_or_containment_before_scientific_interpretation'
    elif not rank_ordering_gain: nxt='SCIR_same_scene_continuous_ordering_does_not_add_3pp_direct_capture_over_preservation_control_stop_do_not_tune_conformal_alpha'
    elif not main_tail: nxt='rank_signal_exists_but_selected_path_certificate_does_not_close_tail_stop_before_full_validation_do_not_threshold_sweep'
    elif not main_coverage: nxt='certificate_is_too_conservative_for_useful_recovery_do_not_rescue_by_alpha_sweep_revisit_intervention_predictor_semantics'
    elif not endpoint: nxt='direct_mechanism_passes_but_endpoint_does_not_convert_audit_path_composition_before_new_representation'
    else: nxt='if_second_fresh_block_also_passes_freeze_SCIR_and_run_exactly_one_independent_full_validation_reproduction'
    report={
        'audit':'v64_3_31_eaf_icer_scir_split','split_name':a.split_name,'full_split_pass':full,'engineering_valid':engineering,
        'rank_direct_capture_gain_over_preservation_control':rank_capture-pres_capture if math.isfinite(rank_capture) and math.isfinite(pres_capture) else float('nan'),
        'rank_direct_capture_improves_over_preservation_control':rank_ordering_gain,'main_direct_capture_noninferior_to_preservation_control':main_coverage,
        'main_selected_path_tail_safe':main_tail,'conformal_certificate_incremental':cert_incremental,'endpoint_noninferior_to_preservation_and_v20':endpoint,
        'endpoint_noninferior_vs_preservation_control':endpoint_vs_pres,'endpoint_noninferior_vs_v20':endpoint_vs_v20,
        'next_action':nxt,'query_parity_vs_v20':q,'structural':structural,'scir_monotone_containment':subset,
        'edge_diagnostics':edge,'scir_prediction_diagnostics':scir_pred,'direct_selected_path_tail':tail,'metrics':M,
        'frozen_thresholds':{
            'catastrophic_teacher_improvement_threshold':CAT,'rank_capture_gain_over_preservation_control_min':RANK_CAPTURE_GAIN_MIN,
            'main_selected_count_min':8,'main_selected_teacher_improvement_sum_min':0.0,
            'main_capture_required_noninferior_to_preservation_control':True,'main_precision_required_noninferior_to_preservation_control':True,
            'main_negative_rms_required_nonworse_than_preservation_control':True,'main_worst_required_above_catastrophic_threshold':True,
            'main_match_tolerance_abs':0.002,'main_regret_tolerance_relative':0.005,'endpoint_must_pass_vs_both_preservation_control_and_v20':True,
            'no_pooled_AB_rescue':True,'no_conformal_alpha_sweep':True,
        },
        'causal_control_note':'PRESERVE differs from frozen V20 only by the already-supported admissible-incumbent default. RANK/MAIN share PRESERVE and therefore isolate the direct intervention ordering/certificate contribution.',
    }
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__': main()
