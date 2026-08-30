from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from bdse.planner.tournament import _DALER_FEATURE_NAMES

FEATURE_NAMES = list(_DALER_FEATURE_NAMES)


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
            idx = np.flatnonzero(inv == i)
            ranks[idx] = ranks[idx].mean()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _load(path: Path) -> dict[str, Any]:
    X=[]; teacher_margin=[]; raw_margin=[]; tokens=[]; challengers=[]; anchors=[]; legacy=[]
    executable=[]; guard=[]; utility_eq=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r=json.loads(line)
        vals=[]; ok=True
        for n in FEATURE_NAMES:
            try:
                v=float(r.get(f"daler_feature_{n}", np.nan))
            except (TypeError, ValueError):
                ok=False; break
            if not np.isfinite(v):
                ok=False; break
            vals.append(v)
        if not ok:
            continue
        try:
            tm=float(r.get("teacher_margin", np.nan)); rm=float(r.get("raw_margin", np.nan))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(tm) or not np.isfinite(rm):
            continue
        X.append(vals); teacher_margin.append(tm); raw_margin.append(rm)
        tokens.append(str(r.get("scenario_token", len(tokens))))
        challengers.append(int(r.get("challenger_action", -1)))
        anchors.append(int(r.get("anchor_action", -1)))
        legacy.append(int(r.get("raw_top_action", -1)))
        executable.append(float(r.get("daler_executable", 0.0)) >= 0.5)
        guard.append(float(r.get("daler_guard_executable", 0.0)) >= 0.5)
        utility_eq.append(float(r.get("daler_utility_equivalent", 0.0)) >= 0.5)
    if not X:
        raise SystemExit(f"no valid DALER edge rows in {path}")
    return {
        "X": np.asarray(X, dtype=np.float64),
        "teacher_margin": np.asarray(teacher_margin, dtype=np.float64),
        "teacher_better": (np.asarray(teacher_margin, dtype=np.float64) > 0.0).astype(np.float64),
        "raw_margin": np.asarray(raw_margin, dtype=np.float64),
        "tokens": tokens,
        "challenger": np.asarray(challengers, dtype=np.int64),
        "anchor": np.asarray(anchors, dtype=np.int64),
        "legacy": np.asarray(legacy, dtype=np.int64),
        "executable": np.asarray(executable, dtype=bool),
        "guard": np.asarray(guard, dtype=bool),
        "utility_equivalent": np.asarray(utility_eq, dtype=bool),
    }


def _group_tensors(tokens: list[str], teacher_margin: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return group id per edge and target edge index per group (-1 = anchor)."""
    groups: dict[str, list[int]] = {}
    for i, token in enumerate(tokens):
        groups.setdefault(token, []).append(i)
    group_tokens = list(groups.keys())
    gid=np.empty((len(tokens),), dtype=np.int64)
    target=np.full((len(group_tokens),), -1, dtype=np.int64)
    for g,t in enumerate(group_tokens):
        idx=np.asarray(groups[t], dtype=np.int64); gid[idx]=g
        best=int(idx[np.argmax(teacher_margin[idx])])
        if float(teacher_margin[best]) > 0.0:
            target[g]=best
    return gid, target, group_tokens


def _fit(
    X: np.ndarray,
    y: np.ndarray,
    teacher_margin: np.ndarray,
    tokens: list[str],
    *,
    steps: int,
    lr: float,
    l2: float,
    aux_edge_bce_weight: float,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if len(X) == 0:
        raise ValueError("DALER fit received zero executable edges")
    mean=X.mean(0); std=np.maximum(X.std(0), 1e-6); Z=(X-mean)/std
    gid,target,_=_group_tensors(tokens,teacher_margin)
    torch.manual_seed(seed)
    z=torch.tensor(Z,dtype=torch.float32)
    t=torch.tensor(y,dtype=torch.float32)
    g=torch.tensor(gid,dtype=torch.long)
    target_t=torch.tensor(target,dtype=torch.long)
    n_groups=int(target.shape[0])
    w=torch.zeros(Z.shape[1],dtype=torch.float32,requires_grad=True)
    b=torch.zeros((),dtype=torch.float32,requires_grad=True)
    opt=torch.optim.Adam([w,b],lr=lr)
    npos=max(float((y>0.5).sum()),1.0); nneg=max(float((y<=0.5).sum()),1.0)
    pos_w=float(len(y)/(2*npos)); neg_w=float(len(y)/(2*nneg))
    class_w=torch.where(t>0.5, torch.tensor(pos_w), torch.tensor(neg_w))
    target_group=torch.nonzero(target_t>=0, as_tuple=False).reshape(-1)
    target_edge=target_t[target_group] if target_group.numel() else target_t.new_empty((0,))
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        logits=z@w+b
        # Anchor is an explicit pseudo-item with fixed logit zero in every scene.
        max_g=torch.zeros(n_groups,dtype=logits.dtype,device=logits.device).scatter_reduce(
            0,g,logits,reduce="amax",include_self=True
        )
        edge_sum=torch.zeros(n_groups,dtype=logits.dtype,device=logits.device).scatter_add(
            0,g,torch.exp(logits-max_g[g])
        )
        exp_sum=torch.exp(-max_g)+edge_sum
        lse=max_g+torch.log(torch.clamp(exp_sum,min=1e-12))
        target_logits=torch.zeros(n_groups,dtype=logits.dtype,device=logits.device)
        if target_group.numel():
            target_logits[target_group]=logits[target_edge]
        listwise=(lse-target_logits).mean()
        edge=torch.nn.functional.binary_cross_entropy_with_logits(logits,t,reduction="none")
        edge=(edge*class_w).mean()
        loss=listwise+float(aux_edge_bce_weight)*edge+float(l2)*w.square().mean()
        loss.backward(); opt.step()
    return w.detach().numpy().astype(float), float(b.detach()), mean.astype(float), std.astype(float)


def _logits(X: np.ndarray, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z=(X-mean)/np.maximum(std,1e-6)
    return np.clip(z@w+b,-40.0,40.0)


def _selection_diag(data: dict[str, Any], logits: np.ndarray, scene_mask: np.ndarray) -> dict[str,float]:
    tokens=data["tokens"]; tm=data["teacher_margin"]; ch=data["challenger"]; legacy=data["legacy"]
    exec_mask=data["executable"] & np.asarray(scene_mask,dtype=bool)
    groups: dict[str,list[int]]={}
    # Include only scenes with at least one executable challenger; no-model anchor-only
    # scenes are deterministic and provide no listwise gradient.
    for i,t in enumerate(tokens):
        if exec_mask[i]: groups.setdefault(t,[]).append(i)
    raw_nonanchor_good=[]; selected_nonanchor_good=[]; selected_tm=[]; fallback=[]; changed=[]
    alternative=[]; alt_good=[]; alt_tm=[]; target_correct=[]
    for idxs in groups.values():
        idx=np.asarray(idxs,dtype=np.int64); lg=int(legacy[idx[0]])
        raw_idx=idx[ch[idx]==lg]
        raw_good=float(tm[int(raw_idx[0])]>0) if raw_idx.size else float("nan")
        best=int(idx[np.argmax(logits[idx])])
        sel=int(ch[best]) if float(logits[best])>0.0 else -1
        if sel<0:
            fallback.append(1.0); changed.append(float(lg!=-1)); selected_tm.append(0.0)
        else:
            fallback.append(0.0); changed.append(float(sel!=lg)); selected_tm.append(float(tm[best]))
            selected_nonanchor_good.append(float(tm[best]>0))
        if np.isfinite(raw_good): raw_nonanchor_good.append(raw_good)
        is_alt=bool(sel>=0 and sel!=lg)
        alternative.append(float(is_alt))
        if is_alt:
            alt_good.append(float(tm[best]>0)); alt_tm.append(float(tm[best]))
        teacher_best=int(idx[np.argmax(tm[idx])])
        target_sel=int(ch[teacher_best]) if float(tm[teacher_best])>0 else -1
        target_correct.append(float(sel==target_sel))
    def mean(x): return float(np.mean(x)) if x else float("nan")
    return {
        "holdout_scene_count": float(len(groups)),
        "holdout_legacy_nonanchor_teacher_better_rate": mean(raw_nonanchor_good),
        "holdout_daler_selected_nonanchor_teacher_better_rate": mean(selected_nonanchor_good),
        "holdout_daler_selected_teacher_margin_mean_including_anchor": mean(selected_tm),
        "holdout_daler_anchor_fallback_rate": mean(fallback),
        "holdout_daler_proposal_changed_rate": mean(changed),
        "holdout_daler_alternative_recovery_rate": mean(alternative),
        "holdout_daler_alternative_recovery_precision": mean(alt_good),
        "holdout_daler_alternative_teacher_margin_mean": mean(alt_tm),
        "holdout_daler_anchor_augmented_top1_accuracy": mean(target_correct),
    }


def main() -> None:
    ap=argparse.ArgumentParser(description="Fit V64.3.17 EAF-DALER train-only deployment-aligned listwise reliability readout.")
    ap.add_argument("--train-frontier-edges",required=True); ap.add_argument("--base-config",required=True)
    ap.add_argument("--output-config",required=True); ap.add_argument("--output-report",required=True)
    ap.add_argument("--holdout-fraction",type=float,default=.2); ap.add_argument("--split-seed",default="v64.3.17-eaf-daler-v1")
    ap.add_argument("--steps",type=int,default=1200); ap.add_argument("--lr",type=float,default=.03); ap.add_argument("--l2",type=float,default=1e-3)
    ap.add_argument("--aux-edge-bce-weight",type=float,default=1.0)
    ap.add_argument("--min-executable-edges",type=int,default=1024); ap.add_argument("--min-class",type=int,default=128)
    args=ap.parse_args()
    if abs(float(args.aux_edge_bce_weight)-1.0)>1e-12:
        raise SystemExit("V64.3.17 protocol fixes aux-edge-BCE weight at 1.0; no validation tuning is permitted")
    data=_load(Path(args.train_frontier_edges)); X=data["X"]; y=data["teacher_better"]; tm=data["teacher_margin"]
    hold_scene=np.asarray([_hash_holdout(t,args.split_seed,args.holdout_fraction) for t in data["tokens"]],dtype=bool)
    executable=data["executable"]
    train=executable & ~hold_scene; hold=executable & hold_scene
    if int(executable.sum())<args.min_executable_edges or min(int(y[executable].sum()),int((1-y[executable]).sum()))<args.min_class:
        raise SystemExit(f"insufficient DALER executable edges/classes: n={int(executable.sum())} pos={int(y[executable].sum())} neg={int((1-y[executable]).sum())}")
    if int(train.sum())<args.min_class or int(hold.sum())<args.min_class:
        raise SystemExit("DALER deterministic scene-group holdout too small")
    train_tokens=[data["tokens"][i] for i in np.flatnonzero(train)]
    w0,b0,m0,s0=_fit(X[train],y[train],tm[train],train_tokens,steps=args.steps,lr=args.lr,l2=args.l2,aux_edge_bce_weight=1.0,seed=1717)
    all_logits=_logits(X,w0,b0,m0,s0); hold_auc=_auc(y[hold],all_logits[hold])
    hold_prob=1.0/(1.0+np.exp(-all_logits[hold])); hold_acc=float(((hold_prob>=.5)==(y[hold]>.5)).mean())
    select_diag=_selection_diag(data,all_logits,hold_scene)

    exec_tokens=[data["tokens"][i] for i in np.flatnonzero(executable)]
    w,b,mean,std=_fit(X[executable],y[executable],tm[executable],exec_tokens,steps=args.steps,lr=args.lr,l2=args.l2,aux_edge_bce_weight=1.0,seed=1717)

    cfg=yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    frontier=cfg.setdefault("runtime",{}).setdefault("decisive_frontier_value",{})
    frontier.setdefault("one_sided_intervention",{})["enabled"]=False
    frontier.setdefault("learned_intervention_reliability",{})["enabled"]=False
    frontier.setdefault("reliability_aware_extremal_reranking",{})["enabled"]=False
    daler=frontier.setdefault("deployment_aligned_listwise_extremal_reliability",{})
    daler.update({
        "enabled":True,"instrument_features":True,
        "model_type":"standardized_linear_anchor_augmented_listwise_reliability",
        "feature_names":FEATURE_NAMES,"feature_mean":[float(x) for x in mean],"feature_std":[float(x) for x in std],
        "weights":[float(x) for x in w],"bias":float(b),"anchor_logit":0.0,
        "ratio_floor":1e-3,"valid_action_normalizer":32.0,
        "require_guard_executable":True,"require_utility_equivalence":True,
        "require_safe_available_for_learned_intervention":True,
        "training_target":"teacher_best_executable_challenger_or_anchor",
        "training_objective":"anchor_augmented_listwise_ce_plus_class_balanced_edge_bce",
        "aux_edge_bce_weight":1.0,
        "selection_operator":"argmax_shared_reliability_logit_over_executable_challengers_and_fixed_anchor",
        "threshold_policy":"fixed_anchor_logit_0_no_validation_threshold_sweep",
    })
    cfg.setdefault("metadata",{})["algorithm_version"]="V64.3.17-EAF-DALER-DARM-DBR"
    cfg.setdefault("provenance",{})["algorithm_version"]="V64.3.17-EAF-DALER-DARM-DBR"
    exp=cfg.setdefault("experiment",{}); exp["name"]="v64_3_17_eaf_daler_fitted"
    exp["algorithm"]="V64.3.17 EAF-DALER: Evidence-Attributed Deployment-Aligned Listwise Extremal Reliability"
    exp["mechanism_chain"]="fixed planner-interface B<=16 -> frozen EAF complete frontier + exact attribution -> exact frozen deployment-equivalent executable challenger set -> anchor-augmented scene-listwise reliability -> extremal selection -> unchanged one-sided/evidence certificate"
    out=Path(args.output_config); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    report={
        "audit":"v64_3_17_eaf_daler_fit",
        "train_all_frontier_edges":int(len(y)),
        "train_executable_edges":int(executable.sum()),
        "train_executable_positive_fraction":float(y[executable].mean()),
        "internal_holdout_executable_edges":int(hold.sum()),
        "internal_holdout_executable_edge_auc":float(hold_auc),
        "internal_holdout_executable_edge_accuracy_at_anchor_logit_0":hold_acc,
        **select_diag,
        "feature_names":FEATURE_NAMES,"weights":[float(x) for x in w],"bias":float(b),"anchor_logit":0.0,
        "aux_edge_bce_weight":1.0,
        "fit_uses_nuplan_validation":False,"fit_uses_test":False,
        "interpretation":"Train-only capacity/mechanism diagnostic. The primary objective is scene-level anchor-augmented listwise extremal ordering over the exact frozen deployment-equivalent executable challenger set; edge BCE is a fixed auxiliary calibration term, not a validation-tuned gate."
    }
    rp=Path(args.output_report); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
