from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

FEATURE_NAMES = [
    "raw_margin",
    "attribution_scale",
    "frontier_residual_rms",
    "frontier_residual_abs_mean",
    "frontier_attribution_scale_rms",
    "frontier_attribution_scale_mean",
    "evidence_certificate_fraction",
    "valid_action_count_norm",
    "margin_over_attribution",
    "attribution_over_frontier_rms",
    "raw_margin_z",
    "attribution_z",
    "raw_margin_rank",
    "attribution_rank",
    "margin_below_raw_top",
]


def _hash_holdout(token: str, seed: str, fraction: float) -> bool:
    h = hashlib.sha256((seed + "::" + token).encode()).digest()
    return int.from_bytes(h[:8], "big") / float(2**64) < fraction


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            idx = np.flatnonzero(inv == i); ranks[idx] = ranks[idx].mean()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    X=[]; y=[]; margins=[]; raw=[]; tokens=[]; challengers=[]; is_raw_top=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        vals=[]
        ok=True
        for n in FEATURE_NAMES:
            v=r.get(f"feature_{n}", None)
            try: v=float(v)
            except (TypeError,ValueError): ok=False; break
            if not np.isfinite(v): ok=False; break
            vals.append(v)
        if not ok: continue
        tm=float(r.get("teacher_margin", np.nan)); rm=float(r.get("raw_margin", np.nan))
        if not np.isfinite(tm) or not np.isfinite(rm): continue
        X.append(vals); y.append(float(tm>0)); margins.append(tm); raw.append(rm)
        tokens.append(str(r.get("scenario_token", len(tokens))))
        challengers.append(int(r.get("challenger_action", -1)))
        is_raw_top.append(float(r.get("is_raw_top",0.0)))
    if not X: raise SystemExit(f"no valid RAER edge rows in {path}")
    return (np.asarray(X,dtype=np.float64), np.asarray(y,dtype=np.float64),
            np.asarray(margins,dtype=np.float64), np.asarray(raw,dtype=np.float64),
            tokens, np.asarray(challengers,dtype=np.int64), np.asarray(is_raw_top,dtype=np.float64))


def _fit(X: np.ndarray, y: np.ndarray, *, steps:int, lr:float, l2:float, seed:int):
    mean=X.mean(0); std=np.maximum(X.std(0),1e-6); Z=(X-mean)/std
    torch.manual_seed(seed)
    z=torch.tensor(Z,dtype=torch.float32); t=torch.tensor(y,dtype=torch.float32)
    w=torch.zeros(Z.shape[1],dtype=torch.float32,requires_grad=True); b=torch.zeros((),dtype=torch.float32,requires_grad=True)
    opt=torch.optim.Adam([w,b],lr=lr)
    npos=max(float((y>0.5).sum()),1.0); nneg=max(float((y<=0.5).sum()),1.0)
    cw=torch.where(t>0.5, torch.tensor(len(y)/(2*npos)), torch.tensor(len(y)/(2*nneg)))
    for _ in range(steps):
        opt.zero_grad(set_to_none=True); logits=z@w+b
        lv=torch.nn.functional.binary_cross_entropy_with_logits(logits,t,reduction="none")
        loss=(lv*cw).mean()+l2*w.square().mean(); loss.backward(); opt.step()
    return w.detach().numpy().astype(float), float(b.detach()), mean.astype(float), std.astype(float)


def _predict(X,w,b,mean,std):
    z=(X-mean)/np.maximum(std,1e-6); l=z@w+b
    return 1/(1+np.exp(-np.clip(l,-40,40)))


def _selection_diag(tokens: list[str], raw_margin: np.ndarray, teacher_margin: np.ndarray, probs: np.ndarray, mask: np.ndarray) -> dict[str,float]:
    groups: dict[str,list[int]]={}
    for i,t in enumerate(tokens):
        if mask[i]: groups.setdefault(t,[]).append(i)
    raw_good=[]; raer_good=[]; raw_tm=[]; raer_tm=[]; changed=[]; abstain=[]
    for idxs in groups.values():
        idx=np.asarray(idxs,dtype=np.int64)
        raw_i=int(idx[np.argmax(raw_margin[idx])])
        eligible=idx[(raw_margin[idx]>0)&(probs[idx]>=0.5)]
        if eligible.size:
            util=probs[eligible]*np.maximum(raw_margin[eligible],0)
            sel_i=int(eligible[np.argmax(util)]); abstain.append(0.0)
            raer_good.append(float(teacher_margin[sel_i]>0)); raer_tm.append(float(teacher_margin[sel_i]))
            changed.append(float(sel_i!=raw_i))
        else:
            # Anchor fallback has zero teacher edge margin by definition.
            sel_i=-1; abstain.append(1.0); raer_good.append(0.0); raer_tm.append(0.0); changed.append(1.0)
        raw_good.append(float(teacher_margin[raw_i]>0)); raw_tm.append(float(teacher_margin[raw_i]))
    if not groups: return {}
    return {
        "holdout_scene_count": float(len(groups)),
        "holdout_raw_top_teacher_better_rate": float(np.mean(raw_good)),
        "holdout_raer_selected_teacher_better_rate": float(np.mean(raer_good)),
        "holdout_raw_top_teacher_margin_mean": float(np.mean(raw_tm)),
        "holdout_raer_selected_teacher_margin_mean": float(np.mean(raer_tm)),
        "holdout_raer_proposal_changed_rate": float(np.mean(changed)),
        "holdout_raer_anchor_fallback_rate": float(np.mean(abstain)),
    }


def main():
    ap=argparse.ArgumentParser(description="Fit V64.3.16 EAF-RAER train-only all-frontier reliability readout.")
    ap.add_argument("--train-frontier-edges",required=True); ap.add_argument("--base-config",required=True)
    ap.add_argument("--output-config",required=True); ap.add_argument("--output-report",required=True)
    ap.add_argument("--holdout-fraction",type=float,default=.2); ap.add_argument("--split-seed",default="v64.3.16-eaf-raer-v1")
    ap.add_argument("--steps",type=int,default=1200); ap.add_argument("--lr",type=float,default=.05); ap.add_argument("--l2",type=float,default=1e-3)
    ap.add_argument("--min-edges",type=int,default=2048); ap.add_argument("--min-class",type=int,default=128)
    args=ap.parse_args()
    X,y,tm,rm,tokens,challengers,is_raw_top=_load(Path(args.train_frontier_edges))
    if len(y)<args.min_edges or min(int(y.sum()),int((1-y).sum()))<args.min_class:
        raise SystemExit(f"insufficient RAER edges/classes: n={len(y)} pos={int(y.sum())} neg={int((1-y).sum())}")
    hold=np.asarray([_hash_holdout(t,args.split_seed,args.holdout_fraction) for t in tokens],dtype=bool)
    if hold.sum()<args.min_class or (~hold).sum()<args.min_class: raise SystemExit("RAER deterministic scene-group holdout too small")
    w0,b0,m0,s0=_fit(X[~hold],y[~hold],steps=args.steps,lr=args.lr,l2=args.l2,seed=1616)
    p0=_predict(X,w0,b0,m0,s0); auc=_auc(y[hold],p0[hold]); acc=float(((p0[hold]>=.5)==(y[hold]>.5)).mean())
    select_diag=_selection_diag(tokens,rm,tm,p0,hold)
    w,b,mean,std=_fit(X,y,steps=args.steps,lr=args.lr,l2=args.l2,seed=1616)

    cfg=yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    frontier=cfg.setdefault("runtime",{}).setdefault("decisive_frontier_value",{})
    frontier.setdefault("one_sided_intervention",{})["enabled"]=False
    frontier.setdefault("learned_intervention_reliability",{})["enabled"]=False
    raer=frontier.setdefault("reliability_aware_extremal_reranking",{})
    raer.update({
        "enabled":True,"instrument_features":True,"model_type":"standardized_logistic_all_frontier_teacher_better_edge",
        "feature_names":FEATURE_NAMES,"feature_mean":[float(x) for x in mean],"feature_std":[float(x) for x in std],
        "weights":[float(x) for x in w],"bias":float(b),"min_probability":0.5,"ratio_floor":1e-3,
        "valid_action_normalizer":32.0,"require_positive_raw_margin":True,
        "selection_utility":"p_teacher_better_times_positive_frozen_eaf_margin",
        "training_target":"teacher_challenger_vs_darm_anchor_margin_positive",
        "threshold_policy":"fixed_0.5_no_validation_threshold_sweep",
    })
    cfg.setdefault("metadata",{})["algorithm_version"]="V64.3.16-EAF-RAER-DARM-DBR"
    cfg.setdefault("provenance",{})["algorithm_version"]="V64.3.16-EAF-RAER-DARM-DBR"
    exp=cfg.setdefault("experiment",{}); exp["name"]="v64_3_16_eaf_raer_fitted"
    exp["algorithm"]="V64.3.16 EAF-RAER: Evidence-Attributed Reliability-Aware Extremal Re-ranking"
    exp["mechanism_chain"]="fixed B=16 selected evidence -> frozen EAF complete frontier value/attribution -> all-challenger train-only reliability -> pre-argmax reliability-aware extremal re-ranking -> unchanged one-sided/evidence certificate -> final decision"
    out=Path(args.output_config); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    report={
        "audit":"v64_3_16_eaf_raer_fit","train_frontier_edges":int(len(y)),"train_positive_fraction":float(y.mean()),
        "internal_holdout_edges":int(hold.sum()),"internal_holdout_auc":float(auc),"internal_holdout_accuracy_at_0_5":acc,
        **select_diag,"feature_names":FEATURE_NAMES,"weights":[float(x) for x in w],"bias":float(b),
        "min_probability":0.5,"fit_uses_nuplan_validation":False,"fit_uses_test":False,
        "interpretation":"Capacity/ordering diagnostic: whether already-computed frozen EAF evidence attribution can reliability-weight every complete-frontier challenger before extremal selection. Not a publication result by itself."
    }
    rp=Path(args.output_report); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__": main()
