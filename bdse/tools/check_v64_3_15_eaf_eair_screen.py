from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np


def _get(d,k,default=float('nan')):
    try: v=float(d.get(k,default))
    except Exception: return float(default)
    return v

def _auc(y,score):
    y=np.asarray(y,dtype=np.int64); score=np.asarray(score,dtype=np.float64)
    pos=int((y==1).sum()); neg=int((y==0).sum())
    if pos==0 or neg==0: return float('nan')
    order=np.argsort(score,kind='mergesort'); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(score)+1,dtype=float)
    vals,inv,cnt=np.unique(score,return_inverse=True,return_counts=True)
    for i,c in enumerate(cnt):
        if c>1:
            idx=np.flatnonzero(inv==i); ranks[idx]=ranks[idx].mean()
    return float((ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg))

def _val_auc(path: Path):
    y=[]; p=[]; n=0
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        a=r.get('raw_frontier_anchor_action',r.get('pair_action_anchor_raw_anchor_action'))
        b=r.get('raw_frontier_proposed_action',r.get('pair_action_anchor_raw_proposed_action'))
        m=r.get('decisive_frontier_value_teacher_proposed_vs_anchor_margin')
        prob=r.get('decisive_frontier_eair_probability')
        if a is None or b is None or int(a)==int(b) or m is None or prob is None: continue
        mv=_get({'x':m},'x'); pv=_get({'x':prob},'x')
        if not math.isfinite(mv) or not math.isfinite(pv): continue
        y.append(int(mv>0)); p.append(pv); n+=1
    return _auc(y,p), n, float(np.mean(y)) if y else float('nan')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-metrics',required=True)
    ap.add_argument('--eair-metrics',required=True)
    ap.add_argument('--eair-per-sample',required=True)
    ap.add_argument('--fit-report',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    raw=json.load(open(a.raw_metrics)); eair=json.load(open(a.eair_metrics)); fit=json.load(open(a.fit_report))
    val_auc,val_edges,val_pos=_val_auc(Path(a.eair_per_sample))
    frozen_keys=['selected_local_anchor_action_match','pair_full_interface_action_match','local_pair_full_interface_action_match','evidence_certificate_fraction','decision_budget_atom_count','proposal_candidate_atom_count','proposal_decisive_atom_recall','selected_decisive_atom_recall','effective_selected_decisive_atom_recall']
    frozen={k:abs(_get(raw,k)-_get(eair,k))<=1e-6 for k in frozen_keys if math.isfinite(_get(raw,k)) and math.isfinite(_get(eair,k))}
    frozen_ok=bool(frozen) and all(frozen.values())
    harm0,harm1=_get(raw,'harmful_pair_potential_intervention_rate'),_get(eair,'harmful_pair_potential_intervention_rate')
    ben0,ben1=_get(raw,'beneficial_pair_potential_intervention_rate'),_get(eair,'beneficial_pair_potential_intervention_rate')
    flip0,flip1=_get(raw,'pair_potential_deployed_flip_rate'),_get(eair,'pair_potential_deployed_flip_rate')
    anchor_match=_get(eair,'selected_local_anchor_action_match')
    anchor_regret=_get(eair,'selected_local_anchor_teacher_regret')
    teacher_match=_get(eair,'teacher_action_match'); teacher_regret=_get(eair,'teacher_regret')
    raw_regret=_get(raw,'teacher_regret')
    benefit_retention=ben1/max(ben0,1e-12) if ben0>0 else float('nan')
    preservation=(harm0-harm1>=0.05 and benefit_retention>=0.35 and ben1>harm1 and flip1>=0.03 and flip1<flip0)
    endpoint=(teacher_match>=anchor_match+0.005 and teacher_regret<=min(anchor_regret,raw_regret)*1.02)
    instrumentation=(
        _get(eair,'decisive_frontier_eair_active')>=0.95 and
        math.isfinite(val_auc) and val_auc>=0.65 and val_edges>=64 and
        _get(eair,'decisive_frontier_value_complete_star_coverage')>=0.99 and
        frozen_ok
    )
    fit_capacity=math.isfinite(float(fit.get('internal_holdout_auc',float('nan')))) and float(fit.get('internal_holdout_auc'))>=0.65
    full=bool(instrumentation and fit_capacity and preservation and endpoint)
    if full:
        next_action='full_val_reproduction_then_test_and_closed_loop_if_reproduced'
    elif not instrumentation or not fit_capacity:
        next_action='query_conditioned_action_evidence_representation_adapter'
    elif not preservation:
        next_action='query_conditioned_reliability_representation_not_more_scalar_gate_tuning'
    else:
        next_action='audit_value_ranking_top_challenger_after_reliable_gate'
    report={
      'audit':'v64_3_15_eaf_eair_screen',
      'full_promotion':full,
      'instrumentation_valid':instrumentation,
      'fit_capacity_signal':fit_capacity,
      'preservation_gain':preservation,
      'endpoint_gain':endpoint,
      'frozen_interface':frozen,
      'metrics':{
        'fit_internal_holdout_auc':fit.get('internal_holdout_auc'),
        'val_teacher_better_auc':val_auc,
        'val_proposal_edges':val_edges,
        'val_positive_fraction':val_pos,
        'raw_teacher_match':_get(raw,'teacher_action_match'),
        'eair_teacher_match':teacher_match,
        'anchor_teacher_match':anchor_match,
        'raw_teacher_regret':raw_regret,
        'eair_teacher_regret':teacher_regret,
        'anchor_teacher_regret':anchor_regret,
        'raw_harmful':harm0,'eair_harmful':harm1,
        'raw_beneficial':ben0,'eair_beneficial':ben1,
        'beneficial_retention':benefit_retention,
        'raw_flip_rate':flip0,'eair_flip_rate':flip1,
      },
      'thresholds':{
        'fit_auc_min':0.65,'val_auc_min':0.65,'harmful_absolute_reduction':0.05,
        'beneficial_retention_min':0.35,'teacher_match_over_anchor':0.005,
        'regret_vs_best_raw_or_anchor_tolerance':0.02,'min_deployed_flip':0.03,
      },
      'next_action':next_action,
      'interpretation':'EAIR tests whether frozen EAF evidence-attribution statistics contain enough one-sided reliability information to distinguish teacher-better from teacher-worse top challenger interventions. It is deliberately a small selective readout-capacity test; failure does not reopen acquisition or justify threshold sweeps.',
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__': main()
