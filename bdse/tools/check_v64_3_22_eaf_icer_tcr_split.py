from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _icer_edge_diag, _metric_pack, _f
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _load_rows, _edge_groups, _path_diag, _guard_block_rate, _identity_rate
from bdse.planner.tournament import _ICER_TRANSITION_FEATURE_NAMES


def _transition_coverage(path: str) -> dict[str, float]:
    n=finite=nonzero=0
    key=[f"icer_transition_incumbent_{x}" for x in _ICER_TRANSITION_FEATURE_NAMES]
    for rs in _edge_groups(path).values():
        for r in rs:
            if _f(r,"icer_admissible",0.0) < 0.5:
                continue
            n+=1
            vals=[]; ok=True
            for k in key:
                try: v=float(r.get(k,np.nan))
                except Exception: ok=False; break
                if not np.isfinite(v): ok=False; break
                vals.append(v)
            finite+=int(ok)
            nonzero+=int(ok and np.max(np.abs(vals))>1e-8)
    return {"edge_count":float(n),"finite_schema_rate":float(finite/n) if n else float('nan'),"nonzero_transition_rate":float(nonzero/n) if n else float('nan')}


def _risk_diag(path: str, safe: set[str]) -> dict[str,float]:
    rep_y=[]; rep_s=[]; ret_y=[]; ret_s=[]
    for rs in _edge_groups(path,safe).values():
        if not rs: continue
        a=int(rs[0].get('anchor_action',-1)); lg=int(rs[0].get('raw_top_action',-1))
        lr=next((r for r in rs if int(r.get('challenger_action',-999))==lg),None)
        if lr is None or _f(lr,'icer_admissible',0)<.5: continue
        ltm=_f(lr,'teacher_margin'); rr=_f(lr,'icer_retention_regret_risk_logit')
        if math.isfinite(ltm) and math.isfinite(rr): ret_y.append(ltm); ret_s.append(rr)
        for r in rs:
            ch=int(r.get('challenger_action',-1))
            if ch in {a,lg} or _f(r,'icer_admissible',0)<.5: continue
            d=_f(r,'teacher_margin')-ltm; sc=_f(r,'icer_replacement_regret_risk_logit')
            if math.isfinite(d) and math.isfinite(sc): rep_y.append(d); rep_s.append(sc)
    def auc(y,s):
        from bdse.tools.check_v64_3_19_eaf_icer_screen import _auc
        return _auc([int(v>0) for v in y],s) if y else float('nan')
    ry=np.asarray(ret_y); rs=np.asarray(ret_s); py=np.asarray(rep_y); ps=np.asarray(rep_s)
    return {
        'retention_risk_auc':auc(ret_y,ret_s),'retention_negative_rate':float(np.mean(rs<0)) if rs.size else float('nan'),
        'retention_teacher_margin_sum_when_negative':float(ry[rs<0].sum()) if rs.size else float('nan'),
        'replacement_risk_auc':auc(rep_y,rep_s),'replacement_positive_rate':float(np.mean(ps>0)) if ps.size else float('nan'),
        'replacement_teacher_improvement_sum_when_positive':float(py[ps>0].sum()) if ps.size else float('nan'),
        'replacement_edge_count':float(len(py)),'retention_scene_count':float(len(ry)),
    }


def main()->None:
    ap=argparse.ArgumentParser(description='One independent V64.3.22 transition-conditioned regret-risk block checker.')
    ap.add_argument('--split-name',required=True)
    for n in ['raw','v21-control','evidence-risk','transition-scalar','transition-dual']:
        ap.add_argument(f'--{n}-metrics',required=True); ap.add_argument(f'--{n}-rows',required=True)
        if n!='raw': ap.add_argument(f'--{n}-edges',required=True)
    ap.add_argument('--output',required=True); a=ap.parse_args()
    tags=['raw','v21_control','evidence_risk','transition_scalar','transition_dual']
    metrics={t:json.load(open(getattr(a,t+'_metrics'))) for t in tags}
    rows={t:_load_rows(getattr(a,t+'_rows')) for t in tags}
    tokens=set(rows['raw'])
    if len(tokens)<480 or any(set(rows[t])!=tokens for t in tags[1:]): raise SystemExit('STOP DATA: paired fresh token identity mismatch')
    allflag={t for t in tokens if _f(rows['raw'][t],'all_actions_safety_flagged_rate',0)>=.5}; safe=tokens-allflag
    edge={t:getattr(a,t+'_edges') for t in tags[1:]}
    ed={t:_icer_edge_diag(Path(edge[t]),safe) for t in tags[1:]}
    path={t:_path_diag(rows['raw'],rows[t],edge[t],safe,allflag) for t in tags[1:]}
    risk={t:_risk_diag(edge[t],safe) for t in ['evidence_risk','transition_scalar','transition_dual']}
    trans={t:_transition_coverage(edge[t]) for t in ['transition_scalar','transition_dual']}
    M={t:_metric_pack(metrics[t]) for t in tags}
    anchor={'match':_f(metrics['raw'],'selected_local_anchor_action_match'),'regret':_f(metrics['raw'],'selected_local_anchor_teacher_regret')}
    structural={
        'all_flagged_scene_count':float(len(allflag)),
        'main_all_flagged_final_identity_vs_raw':_identity_rate(rows['transition_dual'],rows['raw'],allflag),
        'main_all_flagged_delegation_rate':float(np.mean([_f(rows['transition_dual'][t],'decisive_frontier_icer_structural_domain_delegated',0)>=.5 for t in allflag])) if allflag else float('nan'),
        'main_safe_guard_block_rate':_guard_block_rate(rows['transition_dual'],safe),
    }
    frozen_keys=['selected_local_anchor_action_match','pair_full_interface_action_match','local_pair_full_interface_action_match','evidence_certificate_fraction','decision_budget_atom_count','proposal_candidate_atom_count','proposal_decisive_atom_recall','selected_decisive_atom_recall','effective_selected_decisive_atom_recall']
    frozen={k:bool(math.isfinite(_f(metrics['raw'],k)) and math.isfinite(_f(metrics['transition_dual'],k)) and abs(_f(metrics['raw'],k)-_f(metrics['transition_dual'],k))<=1e-6) for k in frozen_keys}
    c=ed['transition_dual']; cs=ed['transition_scalar']; e=ed['evidence_risk']
    instrumentation=(c['scene_count']>=450 and c['admissible_edge_count']>=1800 and c['direct_counterfactual_dominance_edge_count']>=400 and trans['transition_dual']['finite_schema_rate']>=.99 and trans['transition_dual']['nonzero_transition_rate']>=.95 and _f(metrics['transition_dual'],'decisive_frontier_value_complete_star_coverage')>=.99 and all(frozen.values()))
    structural_ok=(len(allflag)>=3 and structural['main_all_flagged_final_identity_vs_raw']==1.0 and structural['main_all_flagged_delegation_rate']==1.0 and structural['main_safe_guard_block_rate']<=.001)
    candidate=c['multi_admissible_proposal_rate']>=.25 and c['admissible_candidates_per_proposal_mean']>=3.0
    signal=c['support_auc']>=.65 and c['direct_counterfactual_dominance_auc']>=.70 and risk['transition_dual']['replacement_risk_auc']>=.60 and risk['transition_dual']['retention_risk_auc']>=.60
    ret_path=path['transition_dual']['admissible_incumbent_to_anchor']['regret_delta_sum']
    rep_path=path['transition_dual']['direct_incumbent_to_alternative']['regret_delta_sum']
    path_safe=ret_path<=0.0 and rep_path<=0.0
    recovery=(c['alternative_recovery_rate']>=.03 and c['alternative_recovery_precision']>=.80 and c['direct_incumbent_replacement_rate']>=.02 and c['direct_incumbent_replacement_precision']>=.60 and c['direct_incumbent_opportunity_capture_rate']>=.08 and c['alternative_teacher_margin_mean']>0.0)
    transition_gain=(rep_path<=path['evidence_risk']['direct_incumbent_to_alternative']['regret_delta_sum']+1e-9 and M['transition_dual']['regret']<=M['evidence_risk']['regret']*1.01)
    signed_gain=((rep_path<path['transition_scalar']['direct_incumbent_to_alternative']['regret_delta_sum']-1e-6) or (c['direct_incumbent_replacement_precision']>=cs['direct_incumbent_replacement_precision']+.01) or (M['transition_dual']['regret']<=M['transition_scalar']['regret']*.99)) and M['transition_dual']['regret']<=M['transition_scalar']['regret']*1.02
    ret=M['transition_dual']['beneficial']/max(M['raw']['beneficial'],1e-12) if M['raw']['beneficial']>0 else float('nan')
    preservation=(M['raw']['harmful']-M['transition_dual']['harmful']>=.05 and ret>=.35 and M['transition_dual']['beneficial']>M['transition_dual']['harmful'] and M['transition_dual']['flip']>=.03 and M['transition_dual']['flip']<M['raw']['flip'])
    endpoint=(M['transition_dual']['match']>=anchor['match']+.005 and M['transition_dual']['regret']<=M['raw']['regret']*1.02)
    main_pass=bool(instrumentation and structural_ok and candidate and signal and path_safe and recovery and transition_gain and preservation and endpoint)
    scalar_pass=bool(instrumentation and structural_ok and candidate and cs['support_auc']>=.65 and cs['direct_counterfactual_dominance_auc']>=.70 and path['transition_scalar']['admissible_incumbent_to_anchor']['regret_delta_sum']<=0 and path['transition_scalar']['direct_incumbent_to_alternative']['regret_delta_sum']<=0 and cs['direct_incumbent_replacement_precision']>=.60 and M['transition_scalar']['regret']<=M['raw']['regret']*1.02)
    if main_pass: next_action='split_pass_freeze_do_not_tune_wait_for_second_independent_fresh_block'
    elif not instrumentation: next_action='engineering_or_transition_instrumentation_failure'
    elif not structural_ok: next_action='deployment_domain_semantics_failure_do_not_change_reliability'
    elif not candidate or not signal: next_action='frontier_or_regret_risk_signal_failure_audit_transition_representation_no_threshold_tuning'
    elif not path_safe: next_action='regret_tail_path_still_harmful_audit_transition_conditioning_or_expected_improvement_objective_do_not_tune_zero_boundary'
    elif not recovery: next_action='regret_safe_but_recovery_insufficient_audit_overconservative_risk_veto_no_threshold_sweep'
    elif not transition_gain: next_action='transition_semantics_not_incremental_keep_evidence_risk_control_and_audit_transition_features'
    elif not preservation: next_action='mechanism_regret_safe_but_preservation_failed_keep_guards_frozen_audit_retention_path'
    else: next_action='mechanism_paths_pass_but_endpoint_fail_audit_residual_teacher_improvement_ordering_before_any_new_representation'
    report={'audit':'v64_3_22_eaf_icer_tcr_split','split_name':a.split_name,'main_split_pass':main_pass,'transition_scalar_split_pass':scalar_pass,'instrumentation_valid':instrumentation,'deployment_complete_domain_alignment':structural_ok,'candidate_support_valid':candidate,'fresh_reliability_signal':signal,'regret_path_nonharmful':path_safe,'counterfactual_recovery_mechanism':recovery,'transition_conditioning_incremental':transition_gain,'signed_profile_incremental_diagnostic':signed_gain,'preservation_gain':preservation,'endpoint_gain':endpoint,'next_action':next_action,'structural':structural,'transition_instrumentation':trans,'risk_diagnostics':risk,'edge_diagnostics':ed,'path_diagnostics':path,'metrics':{'anchor':anchor,**M,'main_beneficial_retention_vs_raw':ret},'frozen_interface':frozen,'thresholds':{'support_auc_min':.65,'dominance_auc_min':.70,'risk_auc_min':.60,'retention_path_regret_delta_sum_max':0.0,'replacement_path_regret_delta_sum_max':0.0,'direct_replacement_precision_min':.60,'direct_capture_min':.08,'harmful_abs_reduction_min':.05,'beneficial_retention_min':.35,'match_over_anchor_min':.005,'regret_vs_raw_tolerance':.02}}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8'); print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__': main()
