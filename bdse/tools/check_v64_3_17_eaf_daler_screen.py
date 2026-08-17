from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _f(d,k,default=float('nan')):
    try: return float(d.get(k,default))
    except Exception: return float(default)


def _auc(y,s):
    y=np.asarray(y,dtype=int); s=np.asarray(s,dtype=float); pos=int((y==1).sum()); neg=int((y==0).sum())
    if not pos or not neg:return float('nan')
    order=np.argsort(s,kind='mergesort'); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(s)+1)
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True)
    for i,c in enumerate(cnt):
        if c>1:
            ix=np.flatnonzero(inv==i); ranks[ix]=ranks[ix].mean()
    return float((ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg))


def _edge_diag(path: Path, *, mode: str) -> dict:
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    groups={}; all_y=[]; all_score=[]; exec_y=[]; exec_score=[]
    score_key='raer_probability' if mode=='raer' else 'daler_logit'
    select_key='raer_selected_action' if mode=='raer' else 'daler_selected_action'
    for r in rows:
        tm=_f(r,'teacher_margin'); sc=_f(r,score_key)
        if math.isfinite(tm) and math.isfinite(sc):
            all_y.append(int(tm>0)); all_score.append(sc)
            if mode=='daler' and _f(r,'daler_executable',0)>=.5:
                exec_y.append(int(tm>0)); exec_score.append(sc)
        groups.setdefault(str(r.get('scenario_token','')),[]).append(r)
    legacy_good=[]; legacy_tm=[]; selected_good=[]; selected_tm=[]; changed=[]; fallback=[]
    alternative=[]; alt_good=[]; alt_tm=[]; proposal_scenes=0; selected_nonanchor=0
    for rs in groups.values():
        legacy=int(rs[0].get('raw_top_action',-1)); sel=int(rs[0].get(select_key,legacy)); anchor=int(rs[0].get('anchor_action',-1))
        if legacy==anchor:
            continue
        proposal_scenes+=1; by={int(r['challenger_action']):r for r in rs}
        lr=by.get(legacy); sr=by.get(sel)
        if lr is not None:
            t=_f(lr,'teacher_margin',0.0); legacy_good.append(float(t>0)); legacy_tm.append(t)
        if sel==anchor or sr is None:
            fallback.append(1.0); selected_tm.append(0.0)
        else:
            t=_f(sr,'teacher_margin',0.0); fallback.append(0.0); selected_tm.append(t)
            selected_good.append(float(t>0)); selected_nonanchor+=1
        changed.append(float(sel!=legacy))
        is_alt=bool(sel not in {legacy,anchor} and sr is not None)
        alternative.append(float(is_alt))
        if is_alt:
            t=_f(sr,'teacher_margin',0.0); alt_good.append(float(t>0)); alt_tm.append(t)
    def mean(x): return float(np.mean(x)) if x else float('nan')
    out={
        'scene_count':len(groups),'proposal_scene_count':proposal_scenes,
        'all_frontier_edge_count':len(all_y),'all_frontier_edge_auc':_auc(all_y,all_score),
        'all_frontier_positive_fraction':mean(all_y),
        'legacy_selected_teacher_better_rate':mean(legacy_good),
        'legacy_selected_teacher_margin_mean':mean(legacy_tm),
        'selected_nonanchor_count':selected_nonanchor,
        'selected_nonanchor_teacher_better_rate':mean(selected_good),
        'selected_teacher_margin_mean_including_anchor':mean(selected_tm),
        'proposal_changed_rate':mean(changed),'anchor_fallback_rate':mean(fallback),
        'alternative_recovery_rate':mean(alternative),
        'alternative_recovery_precision':mean(alt_good),
        'alternative_teacher_margin_mean':mean(alt_tm),
    }
    if mode=='daler':
        out.update({
            'executable_edge_count':len(exec_y),
            'executable_edge_auc':_auc(exec_y,exec_score),
            'executable_positive_fraction':mean(exec_y),
        })
    return out


def main() -> None:
    ap=argparse.ArgumentParser()
    for x in ['raw-metrics','eair-metrics','raer-metrics','daler-metrics','raer-edge-output','daler-edge-output','eair-fit-report','raer-fit-report','daler-fit-report','output']:
        ap.add_argument('--'+x,required=True)
    a=ap.parse_args()
    raw=json.load(open(a.raw_metrics)); eair=json.load(open(a.eair_metrics)); raer=json.load(open(a.raer_metrics)); daler=json.load(open(a.daler_metrics))
    ef=json.load(open(a.eair_fit_report)); rf=json.load(open(a.raer_fit_report)); df=json.load(open(a.daler_fit_report))
    red=_edge_diag(Path(a.raer_edge_output),mode='raer'); ded=_edge_diag(Path(a.daler_edge_output),mode='daler')

    frozen_keys=['selected_local_anchor_action_match','pair_full_interface_action_match','local_pair_full_interface_action_match','evidence_certificate_fraction','decision_budget_atom_count','proposal_candidate_atom_count','proposal_decisive_atom_recall','selected_decisive_atom_recall','effective_selected_decisive_atom_recall']
    frozen={}
    for k in frozen_keys:
        rv=_f(raw,k); dv=_f(daler,k)
        frozen[k]=bool(math.isfinite(rv) and math.isfinite(dv) and abs(rv-dv)<=1e-6)

    raw_match,raw_reg=_f(raw,'teacher_action_match'),_f(raw,'teacher_regret')
    eair_match,eair_reg=_f(eair,'teacher_action_match'),_f(eair,'teacher_regret')
    raer_match,raer_reg=_f(raer,'teacher_action_match'),_f(raer,'teacher_regret')
    dm,dr=_f(daler,'teacher_action_match'),_f(daler,'teacher_regret')
    anchor_match,anchor_reg=_f(raw,'selected_local_anchor_action_match'),_f(raw,'selected_local_anchor_teacher_regret')
    raw_harm,dh=_f(raw,'harmful_pair_potential_intervention_rate'),_f(daler,'harmful_pair_potential_intervention_rate')
    raw_ben,db=_f(raw,'beneficial_pair_potential_intervention_rate'),_f(daler,'beneficial_pair_potential_intervention_rate')
    raw_flip,dfli=_f(raw,'pair_potential_deployed_flip_rate'),_f(daler,'pair_potential_deployed_flip_rate')
    retention=db/max(raw_ben,1e-12) if raw_ben>0 else float('nan')
    guard_block=_f(daler,'pair_action_anchor_guard_blocked_flip',0.0)

    instrumentation=(
        ded['scene_count']>=480 and ded['all_frontier_edge_count']>=2048 and ded['executable_edge_count']>=512
        and math.isfinite(ded['executable_edge_auc']) and ded['executable_edge_auc']>=.65
        and _f(daler,'decisive_frontier_value_complete_star_coverage')>=.99 and all(frozen.values())
    )
    capacity=(
        float(df.get('internal_holdout_executable_edge_auc',0.0))>=.65
        and float(rf.get('internal_holdout_auc',0.0))>=.65
        and float(ef.get('internal_holdout_auc',0.0))>=.65
    )
    # Primary V64.3.17 causal test: listwise selection must recover alternative
    # executable challengers with positive teacher margin, not merely abstain more.
    mechanism=(
        ded['proposal_changed_rate']>=.03
        and ded['alternative_recovery_rate']>=.015
        and math.isfinite(ded['alternative_recovery_precision']) and ded['alternative_recovery_precision']>=.65
        and math.isfinite(ded['alternative_teacher_margin_mean']) and ded['alternative_teacher_margin_mean']>0.0
        and (
            not math.isfinite(red['alternative_recovery_precision'])
            or ded['alternative_recovery_precision']>=red['alternative_recovery_precision']+.10
        )
    )
    deployment_alignment=(guard_block<=.001)
    preservation=(raw_harm-dh>=.05 and retention>=.35 and db>dh and dfli>=.03 and dfli<raw_flip)
    anchor_endpoint=(dm>=anchor_match+.005 and dr<=raw_reg*1.02)
    paired_gain=(
        (dm>=raer_match+.005 and dr<=raer_reg*1.01)
        or (dr<=raer_reg*.99 and dm>=raer_match-.005)
    )
    endpoint=bool(anchor_endpoint and paired_gain)
    full=bool(instrumentation and capacity and mechanism and deployment_alignment and preservation and endpoint)
    if full:
        nxt='independent_full_val_reproduction_then_test_closed_loop_if_reproduced'
    elif not instrumentation or not capacity:
        nxt='structured_per_atom_or_query_conditioned_reliability_representation_keep_acquisition_frozen'
    elif not deployment_alignment:
        nxt='engineering_stop_fix_runtime_candidate_guard_alignment_before_any_algorithm_iteration'
    elif not mechanism:
        nxt='listwise_feature_or_objective_diagnosis_no_threshold_sweep_no_acquisition_reopen'
    elif preservation and not endpoint:
        nxt='same_frozen_frontier_add_robust_teacher_improvement_ordering_term_not_selector_changes'
    else:
        nxt='audit_preservation_failure_keep_B_M_acquisition_and_certificate_frozen'

    report={
        'audit':'v64_3_17_eaf_daler_screen','full_promotion':full,
        'instrumentation_valid':instrumentation,'capacity_signal':capacity,
        'deployment_alignment_invariant':deployment_alignment,
        'listwise_extremal_recovery_mechanism':mechanism,
        'preservation_gain':preservation,'endpoint_gain_vs_anchor_and_raer':endpoint,
        'frozen_interface':frozen,'raer_edge_diagnostics':red,'daler_edge_diagnostics':ded,
        'metrics':{
            'raw_match':raw_match,'eair_match':eair_match,'raer_match':raer_match,'daler_match':dm,'anchor_match':anchor_match,
            'raw_regret':raw_reg,'eair_regret':eair_reg,'raer_regret':raer_reg,'daler_regret':dr,'anchor_regret':anchor_reg,
            'raw_harmful':raw_harm,'daler_harmful':dh,'raw_beneficial':raw_ben,'daler_beneficial':db,
            'beneficial_retention':retention,'raw_flip':raw_flip,'daler_flip':dfli,
            'daler_post_selection_guard_block_rate':guard_block,
        },
        'thresholds':{
            'fit_executable_edge_auc_min':.65,'fresh_executable_edge_auc_min':.65,
            'proposal_changed_min':.03,'alternative_recovery_rate_min':.015,
            'alternative_recovery_precision_min':.65,'alternative_precision_gain_over_raer_min':.10,
            'harmful_abs_reduction_vs_raw_min':.05,'beneficial_retention_min':.35,
            'teacher_match_over_anchor':.005,'regret_vs_raw_tolerance':.02,
            'paired_match_gain_over_raer':.005,'paired_regret_gain_vs_raer':.01,
            'post_selection_guard_block_max':.001,
        },
        'next_action':nxt,
        'interpretation':'DALER is promoted only if a train-only anchor-augmented listwise readout generalizes on fresh executable edges, recovers genuinely teacher-better alternative challengers rather than winning by abstention, obeys the exact frozen deployment candidate set so the final guard is not doing hidden cleanup, preserves beneficial interventions, and gives a paired endpoint gain over frozen RAER. No validation threshold/objective-weight sweep is permitted.'
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__': main()
