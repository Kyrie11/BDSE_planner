from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _f(d: dict[str, Any], k: str, default: float = float('nan')) -> float:
    try:
        return float(d.get(k, default))
    except Exception:
        return float(default)


def _auc(y: list[int] | np.ndarray, s: list[float] | np.ndarray) -> float:
    y=np.asarray(y,dtype=np.int64); s=np.asarray(s,dtype=np.float64)
    good=np.isfinite(s); y=y[good]; s=s[good]
    pos=int((y==1).sum()); neg=int((y==0).sum())
    if not pos or not neg: return float('nan')
    order=np.argsort(s,kind='mergesort'); ranks=np.empty_like(order,dtype=np.float64); ranks[order]=np.arange(1,len(s)+1,dtype=np.float64)
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True)
    for i,c in enumerate(cnt):
        if c>1:
            idx=np.flatnonzero(inv==i); ranks[idx]=ranks[idx].mean()
    return float((ranks[y==1].sum()-pos*(pos+1)/2.0)/(pos*neg))


def _mean(x: list[float]) -> float:
    return float(np.mean(x)) if x else float('nan')


def _edge_diag(path: Path, *, mode: str) -> dict[str, float]:
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if mode=='raer': score_key='raer_probability'; selected_key='raer_selected_action'; admissible_key=None
    elif mode=='daler': score_key='daler_logit'; selected_key='daler_selected_action'; admissible_key='daler_executable'
    elif mode=='dacer': score_key='dacer_logit'; selected_key='dacer_selected_action'; admissible_key='dacer_admissible'
    else: raise ValueError(mode)
    groups: dict[str,list[dict[str,Any]]]={}; all_y=[]; all_s=[]; adm_y=[]; adm_s=[]
    for r in rows:
        tm=_f(r,'teacher_margin'); sc=_f(r,score_key)
        if math.isfinite(tm) and math.isfinite(sc):
            all_y.append(int(tm>0)); all_s.append(sc)
            if admissible_key and _f(r,admissible_key,0)>=.5:
                adm_y.append(int(tm>0)); adm_s.append(sc)
        groups.setdefault(str(r.get('scenario_token','')),[]).append(r)
    legacy_good=[]; legacy_tm=[]; selected_good=[]; selected_tm=[]; changed=[]; fallback=[]
    alt=[]; alt_good=[]; alt_tm=[]; cf_good=[]; opportunity=[]; capture=[]; target_correct=[]
    dominance_y=[]; dominance_score=[]; admissible_counts=[]; proposal_admissible_counts=[]
    proposal_scenes=0; selected_nonanchor=0; multi_all=0; multi_proposal=0
    for rs in groups.values():
        anchor=int(rs[0].get('anchor_action',-1)); legacy=int(rs[0].get('raw_top_action',-1)); sel=int(rs[0].get(selected_key,legacy))
        by={int(r.get('challenger_action',-1)):r for r in rs}
        adm=[] if admissible_key is None else [r for r in rs if _f(r,admissible_key,0)>=.5]
        admissible_counts.append(float(len(adm))); multi_all+=int(len(adm)>=2)
        lr=by.get(legacy); ltm=_f(lr or {},'teacher_margin',0.0); lscore=_f(lr or {},score_key)
        if admissible_key is not None and math.isfinite(lscore):
            for r in adm:
                if int(r.get('challenger_action',-1))==legacy: continue
                tm=_f(r,'teacher_margin'); sc=_f(r,score_key)
                d=tm-ltm
                if math.isfinite(tm) and math.isfinite(sc) and abs(d)>1e-12:
                    dominance_y.append(int(d>0)); dominance_score.append(sc-lscore)
        if legacy==anchor: continue
        proposal_scenes+=1; proposal_admissible_counts.append(float(len(adm))); multi_proposal+=int(len(adm)>=2)
        if lr is not None:
            legacy_good.append(float(ltm>0)); legacy_tm.append(ltm)
        sr=by.get(sel)
        if sel==anchor or sr is None:
            fallback.append(1.0); selected_tm.append(0.0)
        else:
            stm=_f(sr,'teacher_margin',0.0); fallback.append(0.0); selected_tm.append(stm); selected_good.append(float(stm>0)); selected_nonanchor+=1
        changed.append(float(sel!=legacy))
        is_alt=bool(sel not in {anchor,legacy} and sr is not None)
        alt.append(float(is_alt))
        if is_alt:
            stm=_f(sr,'teacher_margin',0.0); alt_good.append(float(stm>0)); alt_tm.append(stm); cf_good.append(float(stm>max(0.0,ltm)))
        if admissible_key is not None:
            opp=any(int(r.get('challenger_action',-1)) not in {anchor,legacy} and _f(r,'teacher_margin')>max(0.0,ltm) for r in adm)
            opportunity.append(float(opp))
            if opp: capture.append(float(is_alt and _f(sr or {},'teacher_margin')>max(0.0,ltm)))
            if adm:
                best=max(adm,key=lambda r:_f(r,'teacher_margin',-float('inf'))); target=int(best.get('challenger_action',-1)) if _f(best,'teacher_margin')>0 else anchor
            else: target=anchor
            target_correct.append(float(sel==target))
    out={
        'scene_count':float(len(groups)),'proposal_scene_count':float(proposal_scenes),
        'all_frontier_edge_count':float(len(all_y)),'all_frontier_edge_auc':_auc(all_y,all_s),'all_frontier_positive_fraction':_mean([float(x) for x in all_y]),
        'legacy_selected_teacher_better_rate':_mean(legacy_good),'legacy_selected_teacher_margin_mean':_mean(legacy_tm),
        'selected_nonanchor_count':float(selected_nonanchor),'selected_nonanchor_teacher_better_rate':_mean(selected_good),
        'selected_teacher_margin_mean_including_anchor':_mean(selected_tm),'proposal_changed_rate':_mean(changed),'anchor_fallback_rate':_mean(fallback),
        'alternative_recovery_rate':_mean(alt),'alternative_recovery_precision':_mean(alt_good),'alternative_teacher_margin_mean':_mean(alt_tm),
    }
    if admissible_key is not None:
        out.update({
            'admissible_edge_count':float(len(adm_y)),'admissible_edge_auc':_auc(adm_y,adm_s),'admissible_positive_fraction':_mean([float(x) for x in adm_y]),
            'admissible_candidates_per_scene_mean':_mean(admissible_counts),
            'multi_admissible_scene_count':float(multi_all),'multi_admissible_scene_rate':float(multi_all/max(len(groups),1)),
            'proposal_admissible_candidates_mean':_mean(proposal_admissible_counts),
            'multi_admissible_proposal_scene_count':float(multi_proposal),'multi_admissible_proposal_scene_rate':float(multi_proposal/max(proposal_scenes,1)),
            'incumbent_dominance_pair_count':float(len(dominance_y)),'incumbent_dominance_auc':_auc(dominance_y,dominance_score),
            'counterfactual_recovery_precision':_mean(cf_good),
            'counterfactual_recovery_opportunity_rate':_mean(opportunity),
            'counterfactual_opportunity_capture_rate':_mean(capture),
            'anchor_augmented_top1_accuracy':_mean(target_correct),
        })
    return out


def _metric_pack(m: dict[str,Any]) -> dict[str,float]:
    return {
        'match':_f(m,'teacher_action_match'),'regret':_f(m,'teacher_regret'),
        'harmful':_f(m,'harmful_pair_potential_intervention_rate'),'beneficial':_f(m,'beneficial_pair_potential_intervention_rate'),
        'flip':_f(m,'pair_potential_deployed_flip_rate'),'guard_block':_f(m,'pair_action_anchor_guard_blocked_flip',0.0),
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    for name in [
        'raw-metrics','raer-metrics','daler-metrics','gdaler-metrics','dacer-scalar-metrics','dacer-profile-metrics',
        'raer-edge-output','daler-edge-output','gdaler-edge-output','dacer-scalar-edge-output','dacer-profile-edge-output',
        'raer-fit-report','daler-fit-report','gdaler-fit-report','dacer-scalar-fit-report','dacer-profile-fit-report','output',
    ]: ap.add_argument('--'+name,required=True)
    a=ap.parse_args()
    raw=json.load(open(a.raw_metrics)); raer=json.load(open(a.raer_metrics)); daler=json.load(open(a.daler_metrics)); gdaler=json.load(open(a.gdaler_metrics)); ds=json.load(open(a.dacer_scalar_metrics)); dp=json.load(open(a.dacer_profile_metrics))
    rf=json.load(open(a.raer_fit_report)); olddf=json.load(open(a.daler_fit_report)); gf=json.load(open(a.gdaler_fit_report)); sf=json.load(open(a.dacer_scalar_fit_report)); pf=json.load(open(a.dacer_profile_fit_report))
    red=_edge_diag(Path(a.raer_edge_output),mode='raer'); oldded=_edge_diag(Path(a.daler_edge_output),mode='daler'); ged=_edge_diag(Path(a.gdaler_edge_output),mode='dacer'); sed=_edge_diag(Path(a.dacer_scalar_edge_output),mode='dacer'); ped=_edge_diag(Path(a.dacer_profile_edge_output),mode='dacer')

    frozen_keys=['selected_local_anchor_action_match','pair_full_interface_action_match','local_pair_full_interface_action_match','evidence_certificate_fraction','decision_budget_atom_count','proposal_candidate_atom_count','proposal_decisive_atom_recall','selected_decisive_atom_recall','effective_selected_decisive_atom_recall']
    frozen={}
    for k in frozen_keys:
        rv=_f(raw,k); pv=_f(dp,k); frozen[k]=bool(math.isfinite(rv) and math.isfinite(pv) and abs(rv-pv)<=1e-6)

    R=_metric_pack(raw); RR=_metric_pack(raer); OD=_metric_pack(daler); G=_metric_pack(gdaler); S=_metric_pack(ds); P=_metric_pack(dp)
    anchor_match=_f(raw,'selected_local_anchor_action_match'); anchor_regret=_f(raw,'selected_local_anchor_teacher_regret')
    benefit_retention=P['beneficial']/max(R['beneficial'],1e-12) if R['beneficial']>0 else float('nan')

    # A valid screen must first prove that V64.3.17's singleton pathology is gone.
    candidate_semantic_fix=(
        ped['admissible_edge_count']>=2048 and ped['multi_admissible_proposal_scene_rate']>=.25
        and ped['admissible_edge_count']>=max(5.0*oldded.get('admissible_edge_count',0.0),2048.0)
    )
    instrumentation=(
        ped['scene_count']>=480 and ped['all_frontier_edge_count']>=2048
        and ped['admissible_edge_count']>=2048 and ped['incumbent_dominance_pair_count']>=512
        and _f(dp,'decisive_frontier_value_complete_star_coverage')>=.99 and all(frozen.values())
    )
    train_capacity=(
        float(pf.get('internal_holdout_support_auc',0.0))>=.65
        and float(pf.get('internal_holdout_dominance_auc',0.0))>=.60
        and float(pf.get('internal_holdout_dominance_pairs',0.0))>=512
        and float(pf.get('holdout_multi_admissible_scene_rate',0.0))>=.25
        and float(sf.get('internal_holdout_support_auc',0.0))>=.65
        and float(gf.get('internal_holdout_support_auc',0.0))>=.65
        and float(rf.get('internal_holdout_auc',0.0))>=.65
    )
    fresh_capacity=(
        math.isfinite(ped['admissible_edge_auc']) and ped['admissible_edge_auc']>=.65
        and math.isfinite(ped['incumbent_dominance_auc']) and ped['incumbent_dominance_auc']>=.60
    )

    # Causal ablation 1: adding the incumbent-relative objective must improve a
    # recovery-specific mechanism over the candidate-set-only G-DALER control.
    objective_supported=(
        (math.isfinite(sed['incumbent_dominance_auc']) and math.isfinite(ged['incumbent_dominance_auc']) and sed['incumbent_dominance_auc']>=ged['incumbent_dominance_auc']+.02)
        or (math.isfinite(sed['counterfactual_opportunity_capture_rate']) and math.isfinite(ged['counterfactual_opportunity_capture_rate']) and sed['counterfactual_opportunity_capture_rate']>=ged['counterfactual_opportunity_capture_rate']+.02)
        or sed['alternative_recovery_rate']>=ged['alternative_recovery_rate']+.01
    )
    # Causal ablation 2: the exact signed selected-atom profile must add measurable
    # ordering/recovery signal over the same counterfactual objective with scalar attribution only.
    profile_mechanism_gain=(
        (math.isfinite(ped['incumbent_dominance_auc']) and math.isfinite(sed['incumbent_dominance_auc']) and ped['incumbent_dominance_auc']>=sed['incumbent_dominance_auc']+.01)
        or (math.isfinite(ped['counterfactual_recovery_precision']) and math.isfinite(sed['counterfactual_recovery_precision']) and ped['counterfactual_recovery_precision']>=sed['counterfactual_recovery_precision']+.05)
        or (math.isfinite(ped['counterfactual_opportunity_capture_rate']) and math.isfinite(sed['counterfactual_opportunity_capture_rate']) and ped['counterfactual_opportunity_capture_rate']>=sed['counterfactual_opportunity_capture_rate']+.01)
        or ped['alternative_recovery_rate']>=sed['alternative_recovery_rate']+.01
    )
    profile_endpoint_nonharm=(P['regret']<=S['regret']*1.02 and P['match']>=S['match']-.005 and P['harmful']<=S['harmful']+.01)
    profile_supported=bool(profile_mechanism_gain and profile_endpoint_nonharm)

    recovery_mechanism=(
        ped['proposal_changed_rate']>=.05
        and ped['alternative_recovery_rate']>=.03
        and math.isfinite(ped['alternative_recovery_precision']) and ped['alternative_recovery_precision']>=.70
        and math.isfinite(ped['alternative_teacher_margin_mean']) and ped['alternative_teacher_margin_mean']>0.0
        and math.isfinite(ped['counterfactual_recovery_precision']) and ped['counterfactual_recovery_precision']>=.60
        and math.isfinite(ped['counterfactual_opportunity_capture_rate']) and ped['counterfactual_opportunity_capture_rate']>=.05
        and ped['selected_nonanchor_teacher_better_rate']>=.75
    )
    deployment_alignment=P['guard_block']<=.001
    preservation=(R['harmful']-P['harmful']>=.05 and benefit_retention>=.35 and P['beneficial']>P['harmful'] and P['flip']>=.03 and P['flip']<R['flip'])
    anchor_endpoint=(P['match']>=anchor_match+.005 and P['regret']<=R['regret']*1.02)
    paired_raer=(
        (P['match']>=RR['match']+.005 and P['regret']<=RR['regret']*1.01)
        or (P['regret']<=RR['regret']*.99 and P['match']>=RR['match']-.005)
    )
    endpoint=bool(anchor_endpoint and paired_raer)
    full=bool(instrumentation and candidate_semantic_fix and train_capacity and fresh_capacity and objective_supported and profile_supported and recovery_mechanism and deployment_alignment and preservation and endpoint)

    if full:
        nxt='independent_full_val_reproduction_with_all_weights_objectives_and_candidate_contract_frozen_then_test_closed_loop_only_if_reproduced'
    elif not instrumentation:
        nxt='engineering_stop_fix_DACER_instrumentation_or_frozen_interface_mismatch_before_algorithm_iteration'
    elif not candidate_semantic_fix:
        nxt='engineering_algorithm_stop_guard_admissible_frontier_still_collapsed_audit_candidate_semantics_do_not_tune_thresholds'
    elif not train_capacity or not fresh_capacity:
        nxt='structured_counterfactual_attribution_representation_diagnosis_keep_B_M_acquisition_guard_and_certificate_frozen'
    elif not objective_supported:
        nxt='incumbent_dominance_objective_not_causally_supported_redesign_relative_ordering_target_no_threshold_sweep'
    elif not profile_supported:
        nxt='signed_selected_atom_profile_not_causally_supported_redesign_structured_attribution_encoder_keep_counterfactual_frontier_frozen'
    elif not deployment_alignment:
        nxt='engineering_stop_fix_guard_admissibility_equivalence_before_any_promotion'
    elif not recovery_mechanism:
        nxt='counterfactual_extremal_recovery_still_bottlenecked_improve_scene_relative_ordering_not_selector_acquisition_or_thresholds'
    elif preservation and not endpoint:
        nxt='same_guard_admissible_frontier_add_robust_teacher_improvement_magnitude_ordering_term_no_selector_changes'
    else:
        nxt='audit_preservation_failure_keep_B_M_acquisition_guard_certificate_frozen'

    report={
        'audit':'v64_3_18_eaf_dacer_screen','full_promotion':full,
        'instrumentation_valid':instrumentation,'candidate_semantic_fix_valid':candidate_semantic_fix,
        'train_capacity_signal':train_capacity,'fresh_capacity_signal':fresh_capacity,
        'counterfactual_objective_causal_support':objective_supported,'signed_atom_profile_causal_support':profile_supported,
        'counterfactual_extremal_recovery_mechanism':recovery_mechanism,'deployment_alignment_invariant':deployment_alignment,
        'preservation_gain':preservation,'endpoint_gain_vs_anchor_and_raer':endpoint,
        'frozen_interface':frozen,
        'edge_diagnostics':{'raer':red,'v64_3_17_daler':oldded,'guard_listwise_gdaler':ged,'dacer_scalar':sed,'dacer_profile':ped},
        'fit_diagnostics':{'raer':rf,'v64_3_17_daler':olddf,'guard_listwise_gdaler':gf,'dacer_scalar':sf,'dacer_profile':pf},
        'metrics':{
            'anchor':{'match':anchor_match,'regret':anchor_regret},'raw':R,'raer':RR,'v64_3_17_daler':OD,'guard_listwise_gdaler':G,'dacer_scalar':S,'dacer_profile':P,
            'profile_beneficial_retention_vs_raw':benefit_retention,
        },
        'thresholds':{
            'fresh_admissible_edges_min':2048,'fresh_multi_admissible_proposal_scene_rate_min':.25,'support_auc_min':.65,'dominance_auc_min':.60,
            'alternative_recovery_rate_min':.03,'alternative_precision_min':.70,'counterfactual_precision_min':.60,'counterfactual_capture_min':.05,
            'selected_nonanchor_teacher_better_min':.75,'harmful_abs_reduction_vs_raw_min':.05,'beneficial_retention_min':.35,
            'teacher_match_over_anchor_min':.005,'regret_vs_raw_tolerance':.02,'post_selection_guard_block_max':.001,
            'objective_ablation_dominance_auc_gain':.02,'profile_ablation_dominance_auc_gain':.01,
        },
        'next_action':nxt,
        'interpretation':'V64.3.18 is promotable only if correcting the V64.3.17 candidate-set semantics restores a genuinely multi-challenger final-guard-admissible frontier, train-only reliability and incumbent-relative dominance generalize, the counterfactual objective and exact signed selected-atom profile each receive causal ablation support, alternative recovery is precise and non-trivial rather than abstention-only, the unchanged final guard performs no hidden cleanup, and preservation/regret remain competitive. No validation threshold, B/M, acquisition, certificate, or objective-weight sweep is permitted.'
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__': main()
