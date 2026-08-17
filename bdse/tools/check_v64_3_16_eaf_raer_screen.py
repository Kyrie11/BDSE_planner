from __future__ import annotations
import argparse, json, math
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


def _edge_diag(path:Path):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    y=[];p=[]; groups={}
    for r in rows:
        tm=_f(r,'teacher_margin'); pr=_f(r,'raer_probability')
        if math.isfinite(tm) and math.isfinite(pr): y.append(int(tm>0)); p.append(pr)
        groups.setdefault(str(r.get('scenario_token','')),[]).append(r)
    raw_good=[]; sel_good=[]; raw_tm=[]; sel_tm=[]; changed=[]; alt_recovery=[]; fallback=[]; proposal_scenes=0
    for rs in groups.values():
        raw=int(rs[0].get('raw_top_action',-1)); sel=int(rs[0].get('raer_selected_action',raw)); anchor=int(rs[0].get('anchor_action',-1))
        if raw==anchor:
            continue
        proposal_scenes += 1
        by={int(r['challenger_action']):r for r in rs}
        rr=by.get(raw); sr=by.get(sel)
        if rr is not None:
            t=_f(rr,'teacher_margin',0); raw_good.append(float(t>0)); raw_tm.append(t)
        if sel==anchor or sr is None:
            sel_good.append(0.0); sel_tm.append(0.0); fallback.append(1.0)
        else:
            t=_f(sr,'teacher_margin',0); sel_good.append(float(t>0)); sel_tm.append(t); fallback.append(0.0)
        changed.append(float(sel!=raw))
        alt_recovery.append(float(sel not in {raw,anchor} and sr is not None and _f(sr,'teacher_margin',0)>0))
    return {
      'val_all_edge_auc':_auc(y,p),'val_edge_count':len(y),'val_positive_fraction':float(np.mean(y)) if y else float('nan'),
      'raw_top_teacher_better_rate':float(np.mean(raw_good)) if raw_good else float('nan'),
      'raer_selected_teacher_better_rate':float(np.mean(sel_good)) if sel_good else float('nan'),
      'raw_top_teacher_margin_mean':float(np.mean(raw_tm)) if raw_tm else float('nan'),
      'raer_selected_teacher_margin_mean':float(np.mean(sel_tm)) if sel_tm else float('nan'),
      'proposal_changed_rate':float(np.mean(changed)) if changed else float('nan'),
      'alternative_recovery_rate':float(np.mean(alt_recovery)) if alt_recovery else float('nan'),
      'anchor_fallback_rate':float(np.mean(fallback)) if fallback else float('nan'),
      'scene_count':len(groups),'proposal_scene_count':proposal_scenes,
    }


def main():
    ap=argparse.ArgumentParser();
    for x in ['raw-metrics','eair-metrics','raer-metrics','raer-edge-output','eair-fit-report','raer-fit-report','output']:
        ap.add_argument('--'+x,required=True)
    a=ap.parse_args(); raw=json.load(open(a.raw_metrics)); eair=json.load(open(a.eair_metrics)); raer=json.load(open(a.raer_metrics))
    ef=json.load(open(a.eair_fit_report)); rf=json.load(open(a.raer_fit_report)); ed=_edge_diag(Path(a.raer_edge_output))
    frozen_keys=['selected_local_anchor_action_match','pair_full_interface_action_match','local_pair_full_interface_action_match','evidence_certificate_fraction','decision_budget_atom_count','proposal_candidate_atom_count','proposal_decisive_atom_recall','selected_decisive_atom_recall','effective_selected_decisive_atom_recall']
    frozen={k:abs(_f(raw,k)-_f(raer,k))<=1e-6 for k in frozen_keys if math.isfinite(_f(raw,k)) and math.isfinite(_f(raer,k))}
    harm0,harm=_f(raw,'harmful_pair_potential_intervention_rate'),_f(raer,'harmful_pair_potential_intervention_rate')
    ben0,ben=_f(raw,'beneficial_pair_potential_intervention_rate'),_f(raer,'beneficial_pair_potential_intervention_rate')
    flip0,flip=_f(raw,'pair_potential_deployed_flip_rate'),_f(raer,'pair_potential_deployed_flip_rate')
    ret=ben/max(ben0,1e-12) if ben0>0 else float('nan')
    anchor_match=_f(raw,'selected_local_anchor_action_match'); anchor_reg=_f(raw,'selected_local_anchor_teacher_regret')
    raw_match,raw_reg=_f(raw,'teacher_action_match'),_f(raw,'teacher_regret')
    eair_match,eair_reg=_f(eair,'teacher_action_match'),_f(eair,'teacher_regret')
    rm,rr=_f(raer,'teacher_action_match'),_f(raer,'teacher_regret')
    instrumentation=(ed['scene_count']>=480 and ed['val_edge_count']>=2048 and math.isfinite(ed['val_all_edge_auc']) and ed['val_all_edge_auc']>=0.65 and _f(raer,'decisive_frontier_value_complete_star_coverage')>=.99 and all(frozen.values()))
    capacity=(float(rf.get('internal_holdout_auc',0))>=.65 and float(ef.get('internal_holdout_auc',0))>=.65)
    mechanism=(ed['raer_selected_teacher_better_rate']>=ed['raw_top_teacher_better_rate']+0.03 and ed['proposal_changed_rate']>=.03 and ed['alternative_recovery_rate']>=.01)
    preservation=(harm0-harm>=.05 and ret>=.35 and ben>harm and flip>=.03 and flip<flip0)
    endpoint=(rm>=anchor_match+.005 and rr<=raw_reg*1.02 and rr<=eair_reg*1.02)
    full=bool(instrumentation and capacity and mechanism and preservation and endpoint)
    if full: nxt='independent_full_val_reproduction_then_test_closed_loop_if_reproduced'
    elif not instrumentation or not capacity: nxt='structured_query_conditioned_per_atom_reliability_representation'
    elif not mechanism: nxt='frontier_reliability_features_or_training_objective_not_threshold_tuning'
    elif preservation and not endpoint: nxt='train_extremal_ordering_value_gain_head_keep_acquisition_frozen'
    else: nxt='audit_raer_preservation_failure_keep_B_M_acquisition_frozen'
    report={'audit':'v64_3_16_eaf_raer_screen','full_promotion':full,'instrumentation_valid':instrumentation,'capacity_signal':capacity,'extremal_reranking_mechanism':mechanism,'preservation_gain':preservation,'endpoint_gain':endpoint,'frozen_interface':frozen,'edge_diagnostics':ed,
      'metrics':{'raw_match':raw_match,'eair_match':eair_match,'raer_match':rm,'anchor_match':anchor_match,'raw_regret':raw_reg,'eair_regret':eair_reg,'raer_regret':rr,'anchor_regret':anchor_reg,'raw_harmful':harm0,'raer_harmful':harm,'raw_beneficial':ben0,'raer_beneficial':ben,'beneficial_retention':ret,'raw_flip':flip0,'raer_flip':flip},
      'thresholds':{'fit_auc_min':.65,'fresh_all_edge_auc_min':.65,'selected_teacher_better_gain_min':.03,'proposal_changed_min':.03,'alternative_recovery_min':.01,'harmful_abs_reduction_min':.05,'beneficial_retention_min':.35,'teacher_match_over_anchor':.005,'regret_vs_raw_and_scalar_eair_tolerance':.02},'next_action':nxt,
      'interpretation':'RAER is promoted only if all-frontier attribution-derived reliability generalizes to fresh validation, changes extremal selection in the intended runner-up-recovery way, improves preservation without collapsing flips, and preserves the raw/scalar-EAIR regret endpoint. No threshold sweep is permitted.'}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True)); print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
