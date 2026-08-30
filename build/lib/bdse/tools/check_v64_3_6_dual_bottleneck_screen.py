from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any


def _finite(x: Any) -> float | None:
    try:
        v=float(x)
    except (TypeError,ValueError): return None
    return v if math.isfinite(v) else None

def _delta(row:dict, anchor:dict, key:str)->float|None:
    a=_finite(anchor.get(key)); b=_finite(row.get(key))
    return None if a is None or b is None else b-a

def _max(rows,key):
    xs=[v for r in rows if (v:=_finite(r.get(key))) is not None]
    return max(xs) if xs else None

def load_rows(path:Path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def build(rows:list[dict], variant:str)->dict:
    anchor=next((r for r in rows if int(r.get('epoch',0))<0), rows[0])
    post=[r for r in rows if int(r.get('epoch',-1))>=0]
    if not post: post=[rows[-1]]
    keys={
      'topm':'val_teacher_exact_winner_flip_critical_recall_topm_micro',
      'selected':'val_teacher_exact_winner_flip_critical_recall_selected_micro',
      'proposal':'val_proposal_decisive_atom_recall','teacher':'val_teacher_action_match',
      'pairfull':'val_pair_full_interface_action_match','localpair':'val_local_pair_full_interface_action_match',
      'budgetpair':'val_budget_vs_pair_full_match','pairflip':'val_pair_full_to_budget_flip_rate',
      'frozen':'val_teacher_exact_winner_flip_frozen_family_slot_oracle_topm_recall',
      'global':'val_teacher_exact_winner_flip_global_oracle_topm_recall',
      'gap':'val_teacher_exact_winner_flip_family_slot_oracle_gap',
    }
    enriched=[]
    for r in post:
        d={k:_delta(r,anchor,key) for k,key in keys.items() if k not in {'frozen','global','gap'}}
        pair_adv=None
        pf=_finite(r.get(keys['pairfull'])); lp=_finite(r.get(keys['localpair']))
        if pf is not None and lp is not None: pair_adv=pf-lp
        acq=(d['topm'] is not None and d['topm']>=0.01 and d['selected'] is not None and d['selected']>=-0.005 and d['proposal'] is not None and d['proposal']>=-0.02 and d['teacher'] is not None and d['teacher']>=-0.005)
        val=(d['pairfull'] is not None and d['pairfull']>=0.01 and pair_adv is not None and pair_adv>=0.005 and d['teacher'] is not None and d['teacher']>=-0.005 and d['budgetpair'] is not None and d['budgetpair']>=-0.02)
        full=((d['teacher'] or -999)>=0.005 and (acq or val))
        enriched.append((r,d,pair_adv,acq,val,full))
    def score(t):
        r,d,adv,acq,val,full=t
        return (100*int(full)+20*int(acq)+20*int(val)+8*(d['teacher'] or -1)+4*(d['topm'] or -1)+4*(d['pairfull'] or -1)+(adv or -1))
    best=max(enriched,key=score)
    r,d,adv,acq,val,full=best
    vup=variant.upper()
    need_bcha='BCHA' in vup; need_lbpr='LBPR' in vup
    bcha_rms=_max(post,'critical_family_residual_rms') or 0.0
    lbpr_delta=_max(post,'literal_pair_adapter_parameter_delta_rms') or 0.0
    lbpr_rms=_max(post,'literal_boundary_pair_residual_rms') or 0.0
    lea_loss=_max(post,'L_critical_endpoint_attribution') or 0.0
    endpoint=_max(post,'critical_endpoint_representable_fraction') or 0.0
    valid=(endpoint>0.95 and lea_loss>0 and ((not need_bcha) or bcha_rms>1e-6) and ((not need_lbpr) or (lbpr_delta>1e-7 and lbpr_rms>1e-7)))
    frozen=_max(post,keys['frozen']); global_o=_max(post,keys['global']); gap=_max(post,keys['gap'])
    family_ceiling=(frozen is not None and global_o is not None and frozen<0.90 and (global_o-frozen)>=0.05)
    return {
      'audit':'v64_3_6_dual_bottleneck_screen','variant':variant,'valid':bool(valid),
      'anchor_epoch':anchor.get('epoch'),'selected_epoch':r.get('epoch'),
      'anchor':{k:_finite(anchor.get(key)) for k,key in keys.items()},
      'selected':{k:_finite(r.get(key)) for k,key in keys.items()},
      'deltas':d,'pair_full_advantage_over_local':adv,
      'meaningful_acquisition_gain':bool(acq),'meaningful_value_gain':bool(val),'full_promotion':bool(full and valid),
      'family_admission_ceiling_indicated':bool(family_ceiling),
      'frozen_family_slot_oracle_topm_recall_max':frozen,'global_oracle_topm_recall_max':global_o,'family_slot_oracle_gap_max':gap,
      'activation':{'bcha_family_residual_rms_max':bcha_rms,'lbpr_parameter_delta_rms_max':lbpr_delta,'lbpr_residual_rms_max':lbpr_rms,'lea_loss_max':lea_loss,'endpoint_representability_max':endpoint},
      'thresholds':{'critical_topm_gain':0.01,'selected_floor':-0.005,'proposal_floor':-0.02,'teacher_stability_floor':-0.005,'pair_full_gain':0.01,'pair_full_over_local':0.005,'budget_vs_pair_full_floor':-0.02,'full_teacher_gain':0.005,'family_oracle_ceiling':0.90,'family_oracle_gap':0.05},
    }

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--train-log',type=Path,required=True); ap.add_argument('--variant',required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    report=build(load_rows(a.train_log),a.variant); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8'); print(json.dumps(report,indent=2,sort_keys=True)); return 0 if report['valid'] else 3
if __name__=='__main__': raise SystemExit(main())
