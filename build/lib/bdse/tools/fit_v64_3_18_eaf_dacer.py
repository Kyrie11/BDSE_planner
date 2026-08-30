from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from bdse.planner.tournament import _DACER_FEATURE_NAMES, _DALER_FEATURE_NAMES

FEATURE_NAMES = list(_DACER_FEATURE_NAMES)
SCALAR_FEATURE_COUNT = len(_DALER_FEATURE_NAMES)


def _hash_holdout(token: str, seed: str, fraction: float) -> bool:
    h = hashlib.sha256((seed + "::" + token).encode()).digest()
    return int.from_bytes(h[:8], "big") / float(2**64) < fraction


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=np.float64)
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            idx = np.flatnonzero(inv == i); ranks[idx] = ranks[idx].mean()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _load(path: Path) -> dict[str, Any]:
    X=[]; teacher_margin=[]; tokens=[]; challengers=[]; anchors=[]; legacy=[]; admissible=[]; utility_prior=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r=json.loads(line); vals=[]; ok=True
        for n in FEATURE_NAMES:
            try: v=float(r.get(f"dacer_feature_{n}", np.nan))
            except (TypeError, ValueError): ok=False; break
            if not np.isfinite(v): ok=False; break
            vals.append(v)
        if not ok:
            continue
        try: tm=float(r.get("teacher_margin", np.nan))
        except (TypeError, ValueError): continue
        if not np.isfinite(tm):
            continue
        X.append(vals); teacher_margin.append(tm); tokens.append(str(r.get("scenario_token", len(tokens))))
        challengers.append(int(r.get("challenger_action", -1))); anchors.append(int(r.get("anchor_action", -1)))
        legacy.append(int(r.get("raw_top_action", -1)))
        admissible.append(float(r.get("dacer_admissible", 0.0)) >= 0.5)
        utility_prior.append(float(r.get("dacer_utility_prior", 0.0)) >= 0.5)
    if not X:
        raise SystemExit(f"no valid DACER edge rows in {path}")
    tm=np.asarray(teacher_margin,dtype=np.float64)
    return {
        "X":np.asarray(X,dtype=np.float64), "teacher_margin":tm, "teacher_better":(tm>0).astype(np.float64),
        "tokens":tokens, "challenger":np.asarray(challengers,dtype=np.int64), "anchor":np.asarray(anchors,dtype=np.int64),
        "legacy":np.asarray(legacy,dtype=np.int64), "admissible":np.asarray(admissible,dtype=bool),
        "utility_prior":np.asarray(utility_prior,dtype=bool),
    }


def _group_index(tokens: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    groups: dict[str,list[int]]={}
    for i,t in enumerate(tokens): groups.setdefault(t,[]).append(i)
    return list(groups.keys()), {t:np.asarray(v,dtype=np.int64) for t,v in groups.items()}


def _fit(
    X_all: np.ndarray,
    y_all: np.ndarray,
    tm_all: np.ndarray,
    tokens: list[str],
    challenger: np.ndarray,
    legacy: np.ndarray,
    fit_mask: np.ndarray,
    *, feature_mode: str, objective_mode: str, steps: int, lr: float, l2: float, seed: int,
) -> tuple[np.ndarray,float,np.ndarray,np.ndarray]:
    fit_mask=np.asarray(fit_mask,dtype=bool)
    idx_fit=np.flatnonzero(fit_mask)
    if idx_fit.size == 0: raise ValueError("DACER fit received zero guard-admissible edges")
    mean=X_all[idx_fit].mean(0); std=np.maximum(X_all[idx_fit].std(0),1e-6)
    Z=(X_all-mean)/std
    active=np.ones((Z.shape[1],),dtype=bool)
    if feature_mode=="scalar": active[SCALAR_FEATURE_COUNT:]=False
    elif feature_mode!="profile": raise ValueError(f"unknown feature_mode={feature_mode}")
    Z[:,~active]=0.0

    group_tokens,groups=_group_index(tokens)
    fit_group_tokens=[]; fit_groups=[]; target_edge=[]
    for t in group_tokens:
        idx=groups[t]; cand=idx[fit_mask[idx]]
        if cand.size==0: continue
        fit_group_tokens.append(t); fit_groups.append(cand)
        best=int(cand[np.argmax(tm_all[cand])]); target_edge.append(best if float(tm_all[best])>0 else -1)
    if not fit_groups: raise ValueError("no DACER scene groups with admissible challengers")

    # Incumbent-dominance pairs use the raw EAF/utility-refined incumbent as
    # context even when that incumbent itself would fail the final guard.  The
    # candidate side must be guard-admissible; labels are TRAIN-only.
    pair_b=[]; pair_l=[]; pair_sign=[]
    for t,cand in zip(fit_group_tokens, fit_groups):
        idx=groups[t]; lg=int(legacy[idx[0]])
        li=idx[challenger[idx]==lg]
        if li.size==0: continue
        li=int(li[0]); ltm=float(tm_all[li])
        for b in cand.tolist():
            if int(challenger[b])==lg: continue
            d=float(tm_all[b])-ltm
            if abs(d)<=1e-12: continue
            pair_b.append(int(b)); pair_l.append(li); pair_sign.append(1.0 if d>0 else -1.0)

    torch.manual_seed(seed)
    z=torch.tensor(Z,dtype=torch.float32); y=torch.tensor(y_all,dtype=torch.float32)
    w=torch.zeros(Z.shape[1],dtype=torch.float32,requires_grad=True); b=torch.zeros((),dtype=torch.float32,requires_grad=True)
    opt=torch.optim.Adam([w,b],lr=lr)
    fit_t=torch.tensor(idx_fit,dtype=torch.long)
    # Group id aligned to fit_t for vectorized anchor-augmented listwise CE.
    global_to_group={}
    for gi,cand in enumerate(fit_groups):
        for ii in cand.tolist(): global_to_group[int(ii)]=gi
    gid_fit=torch.tensor([global_to_group[int(i)] for i in idx_fit.tolist()],dtype=torch.long)
    n_groups=len(fit_groups)
    target_edge_t=torch.tensor(target_edge,dtype=torch.long)
    target_group=torch.nonzero(target_edge_t>=0,as_tuple=False).reshape(-1)
    target_global=target_edge_t[target_group] if target_group.numel() else torch.empty((0,),dtype=torch.long)
    npos=max(float((y_all[idx_fit]>.5).sum()),1.0); nneg=max(float((y_all[idx_fit]<=.5).sum()),1.0)
    class_w=torch.where(y[fit_t]>.5,torch.tensor(len(idx_fit)/(2*npos)),torch.tensor(len(idx_fit)/(2*nneg)))
    pb=torch.tensor(pair_b,dtype=torch.long) if pair_b else torch.empty((0,),dtype=torch.long)
    pl=torch.tensor(pair_l,dtype=torch.long) if pair_l else torch.empty((0,),dtype=torch.long)
    ps=torch.tensor(pair_sign,dtype=torch.float32) if pair_sign else torch.empty((0,),dtype=torch.float32)

    if objective_mode not in {"listwise", "counterfactual"}:
        raise ValueError(f"unknown objective_mode={objective_mode}")
    dominance_weight = 1.0 if objective_mode == "counterfactual" else 0.0
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True); logits=z@w+b
        fit_logits=logits[fit_t]
        max_g=torch.zeros(n_groups,dtype=logits.dtype).scatter_reduce(0,gid_fit,fit_logits,reduce="amax",include_self=True)
        edge_sum=torch.zeros(n_groups,dtype=logits.dtype).scatter_add(0,gid_fit,torch.exp(fit_logits-max_g[gid_fit]))
        lse=max_g+torch.log(torch.clamp(torch.exp(-max_g)+edge_sum,min=1e-12))
        target_logits=torch.zeros(n_groups,dtype=logits.dtype)
        if target_group.numel(): target_logits[target_group]=logits[target_global]
        listwise=(lse-target_logits).mean()
        support=torch.nn.functional.binary_cross_entropy_with_logits(fit_logits,y[fit_t],reduction="none")
        support=(support*class_w).mean()
        if pb.numel():
            dominance=torch.nn.functional.softplus(-ps*(logits[pb]-logits[pl])).mean()
        else:
            dominance=torch.zeros((),dtype=logits.dtype)
        loss=listwise+support+dominance_weight*dominance+float(l2)*w.square().mean()
        loss.backward(); opt.step()
        with torch.no_grad():
            if feature_mode=="scalar": w[SCALAR_FEATURE_COUNT:]=0.0
    return w.detach().numpy().astype(float),float(b.detach()),mean.astype(float),std.astype(float)


def _logits(X: np.ndarray,w: np.ndarray,b:float,mean:np.ndarray,std:np.ndarray)->np.ndarray:
    return np.clip(((X-mean)/np.maximum(std,1e-6))@w+b,-40.0,40.0)


def _dominance_auc(data: dict[str,Any], logits: np.ndarray, scene_mask: np.ndarray) -> tuple[float,int]:
    _,groups=_group_index(data["tokens"]); yy=[]; ss=[]
    for idx in groups.values():
        if not bool(scene_mask[idx[0]]): continue
        lg=int(data["legacy"][idx[0]]); li=idx[data["challenger"][idx]==lg]
        if li.size==0: continue
        li=int(li[0]); ltm=float(data["teacher_margin"][li])
        for b in idx[data["admissible"][idx]].tolist():
            if int(data["challenger"][b])==lg: continue
            d=float(data["teacher_margin"][b])-ltm
            if abs(d)<=1e-12: continue
            yy.append(int(d>0)); ss.append(float(logits[b]-logits[li]))
    return _auc(np.asarray(yy),np.asarray(ss)),len(yy)


def _selection_diag(data: dict[str,Any], logits: np.ndarray, scene_mask: np.ndarray) -> dict[str,float]:
    _,groups=_group_index(data["tokens"])
    vals={k:[] for k in ["legacy_good","selected_good","selected_tm","fallback","changed","alternative","alt_good","alt_tm","counterfactual_good","opportunity","opportunity_capture","target_correct"]}
    multi=0; scenes=0
    for idx in groups.values():
        if not bool(scene_mask[idx[0]]): continue
        a=int(data["anchor"][idx[0]]); lg=int(data["legacy"][idx[0]])
        if lg==a: continue
        scenes+=1; cand=idx[data["admissible"][idx]]; multi+=int(cand.size>=2)
        li=idx[data["challenger"][idx]==lg]; ltm=float(data["teacher_margin"][int(li[0])]) if li.size else 0.0
        vals["legacy_good"].append(float(ltm>0))
        if cand.size:
            best=int(cand[np.argmax(logits[cand])]); sel=int(data["challenger"][best]) if float(logits[best])>0 else a
        else: best=-1; sel=a
        vals["fallback"].append(float(sel==a)); vals["changed"].append(float(sel!=lg))
        if sel!=a:
            tm=float(data["teacher_margin"][best]); vals["selected_good"].append(float(tm>0)); vals["selected_tm"].append(tm)
        else: vals["selected_tm"].append(0.0)
        is_alt=bool(sel not in {a,lg}); vals["alternative"].append(float(is_alt))
        if is_alt:
            tm=float(data["teacher_margin"][best]); vals["alt_good"].append(float(tm>0)); vals["alt_tm"].append(tm)
            vals["counterfactual_good"].append(float(tm>max(0.0,ltm)))
        opportunity=bool(any(int(data["challenger"][j]) not in {a,lg} and float(data["teacher_margin"][j])>max(0.0,ltm) for j in cand.tolist()))
        vals["opportunity"].append(float(opportunity))
        if opportunity: vals["opportunity_capture"].append(float(is_alt and float(data["teacher_margin"][best])>max(0.0,ltm)))
        if cand.size:
            tb=int(cand[np.argmax(data["teacher_margin"][cand])]); target=int(data["challenger"][tb]) if float(data["teacher_margin"][tb])>0 else a
        else: target=a
        vals["target_correct"].append(float(sel==target))
    def mean(x): return float(np.mean(x)) if x else float("nan")
    return {
        "scene_count":float(scenes),"multi_admissible_scene_rate":float(multi/max(scenes,1)),
        "legacy_teacher_better_rate":mean(vals["legacy_good"]),
        "selected_nonanchor_teacher_better_rate":mean(vals["selected_good"]),
        "selected_teacher_margin_mean_including_anchor":mean(vals["selected_tm"]),
        "anchor_fallback_rate":mean(vals["fallback"]),"proposal_changed_rate":mean(vals["changed"]),
        "alternative_recovery_rate":mean(vals["alternative"]),"alternative_recovery_precision":mean(vals["alt_good"]),
        "alternative_teacher_margin_mean":mean(vals["alt_tm"]),
        "counterfactual_recovery_precision":mean(vals["counterfactual_good"]),
        "counterfactual_recovery_opportunity_rate":mean(vals["opportunity"]),
        "counterfactual_opportunity_capture_rate":mean(vals["opportunity_capture"]),
        "anchor_augmented_top1_accuracy":mean(vals["target_correct"]),
    }


def main() -> None:
    ap=argparse.ArgumentParser(description="Fit V64.3.18 EAF-DACER train-only guard-admissible counterfactual extremal readout.")
    ap.add_argument("--train-frontier-edges",required=True); ap.add_argument("--base-config",required=True)
    ap.add_argument("--output-config",required=True); ap.add_argument("--output-report",required=True)
    ap.add_argument("--feature-mode",choices=["scalar","profile"],default="profile")
    ap.add_argument("--objective-mode",choices=["listwise","counterfactual"],default="counterfactual")
    ap.add_argument("--holdout-fraction",type=float,default=.2); ap.add_argument("--split-seed",default="v64.3.18-eaf-dacer-v1")
    ap.add_argument("--steps",type=int,default=1200); ap.add_argument("--lr",type=float,default=.03); ap.add_argument("--l2",type=float,default=1e-3)
    ap.add_argument("--support-bce-weight",type=float,default=1.0); ap.add_argument("--incumbent-dominance-weight",type=float,default=1.0)
    ap.add_argument("--min-admissible-edges",type=int,default=8192); ap.add_argument("--min-multi-scenes",type=int,default=512)
    args=ap.parse_args()
    if abs(args.support_bce_weight-1.0)>1e-12 or abs(args.incumbent_dominance_weight-1.0)>1e-12:
        raise SystemExit("V64.3.18 fixes declared support/dominance weights at 1.0; ablations use --objective-mode, never validation-tuned weights")
    if args.objective_mode == "listwise" and args.feature_mode != "scalar":
        raise SystemExit("V64.3.18 G-DALER control is pre-registered as scalar+listwise; profile+listwise is not an allowed screen arm")
    data=_load(Path(args.train_frontier_edges)); mask=data["admissible"]
    _,groups=_group_index(data["tokens"]); multi=sum(int(np.sum(mask[idx])>=2) for idx in groups.values())
    if int(mask.sum())<args.min_admissible_edges or multi<args.min_multi_scenes:
        raise SystemExit(f"insufficient DACER guard-admissible support: edges={int(mask.sum())}, multi_scenes={multi}")
    hold_scene=np.asarray([_hash_holdout(t,args.split_seed,args.holdout_fraction) for t in data["tokens"]],dtype=bool)
    train_mask=mask & ~hold_scene; hold_mask=mask & hold_scene
    if int(train_mask.sum())<4096 or int(hold_mask.sum())<1024:
        raise SystemExit("DACER deterministic scene-group holdout too small")

    w0,b0,m0,s0=_fit(data["X"],data["teacher_better"],data["teacher_margin"],data["tokens"],data["challenger"],data["legacy"],train_mask,
                     feature_mode=args.feature_mode,objective_mode=args.objective_mode,steps=args.steps,lr=args.lr,l2=args.l2,seed=1818)
    logits0=_logits(data["X"],w0,b0,m0,s0)
    hold_auc=_auc(data["teacher_better"][hold_mask],logits0[hold_mask]); dom_auc,dom_n=_dominance_auc(data,logits0,hold_scene)
    hold_diag=_selection_diag(data,logits0,hold_scene)

    w,b,mean,std=_fit(data["X"],data["teacher_better"],data["teacher_margin"],data["tokens"],data["challenger"],data["legacy"],mask,
                     feature_mode=args.feature_mode,objective_mode=args.objective_mode,steps=args.steps,lr=args.lr,l2=args.l2,seed=1818)
    cfg=yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8")); frontier=cfg.setdefault("runtime",{}).setdefault("decisive_frontier_value",{})
    frontier.setdefault("one_sided_intervention",{})["enabled"]=False
    frontier.setdefault("learned_intervention_reliability",{})["enabled"]=False
    frontier.setdefault("reliability_aware_extremal_reranking",{})["enabled"]=False
    frontier.setdefault("deployment_aligned_listwise_extremal_reliability",{})["enabled"]=False
    dacer=frontier.setdefault("deployment_admissible_counterfactual_extremal_recovery",{})
    dacer.update({
        "enabled":True,"instrument_features":True,"feature_mode":args.feature_mode,
        "model_type":"standardized_linear_guard_admissible_counterfactual_score",
        "feature_names":FEATURE_NAMES,"feature_mean":[float(x) for x in mean],"feature_std":[float(x) for x in std],
        "weights":[float(x) for x in w],"bias":float(b),"anchor_logit":0.0,
        "require_guard_admissible":True,"require_safe_available_for_learned_intervention":True,
        "utility_equivalence_role":"diagnostic_tiebreak_only_not_hard_mask",
        "training_target":"teacher_best_guard_admissible_challenger_or_anchor",
        "training_objective":("anchor_augmented_listwise_ce_plus_class_balanced_support_bce_plus_incumbent_dominance_rank" if args.objective_mode=="counterfactual" else "anchor_augmented_listwise_ce_plus_class_balanced_support_bce"),
        "objective_mode":args.objective_mode,
        "support_bce_weight":1.0,"incumbent_dominance_weight":(1.0 if args.objective_mode=="counterfactual" else 0.0),
        "selection_operator":"argmax_counterfactual_score_over_guard_admissible_challengers_and_fixed_anchor",
        "threshold_policy":"fixed_anchor_logit_0_no_validation_threshold_sweep",
    })
    cfg.setdefault("metadata",{})["algorithm_version"]="V64.3.18-EAF-DACER-DARM-DBR"
    cfg.setdefault("provenance",{})["algorithm_version"]="V64.3.18-EAF-DACER-DARM-DBR"
    exp=cfg.setdefault("experiment",{}); variant=("gdaler" if args.objective_mode=="listwise" else "dacer")
    exp["name"]=f"v64_3_18_eaf_{variant}_{args.feature_mode}_fitted"
    exp["algorithm"]="V64.3.18 EAF-DACER: Evidence-Attributed Deployment-Admissible Counterfactual Extremal Recovery"
    exp["mechanism_chain"]="fixed B<=16 -> frozen EAF complete frontier + exact signed selected-atom attribution -> frozen guard-admissible challenger frontier -> anchor/listwise support + incumbent counterfactual dominance -> extremal recovery/abstention -> unchanged final guard"
    out=Path(args.output_config); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    report={
        "audit":"v64_3_18_eaf_dacer_fit","feature_mode":args.feature_mode,"objective_mode":args.objective_mode,
        "train_all_frontier_edges":int(len(mask)),"train_admissible_edges":int(mask.sum()),
        "train_admissible_positive_fraction":float(data["teacher_better"][mask].mean()),"train_multi_admissible_scenes":int(multi),
        "internal_holdout_admissible_edges":int(hold_mask.sum()),"internal_holdout_support_auc":float(hold_auc),
        "internal_holdout_dominance_auc":float(dom_auc),"internal_holdout_dominance_pairs":int(dom_n),
        **{f"holdout_{k}":v for k,v in hold_diag.items()},
        "feature_names":FEATURE_NAMES,"scalar_feature_count":SCALAR_FEATURE_COUNT,"weights":[float(x) for x in w],"bias":float(b),"anchor_logit":0.0,
        "support_bce_weight":1.0,"incumbent_dominance_weight":(1.0 if args.objective_mode=="counterfactual" else 0.0),"fit_uses_nuplan_validation":False,"fit_uses_test":False,
        "interpretation":("Train-only causal capacity/mechanism diagnostic. Utility-equivalence is diagnostic / exact-tie-break context only (not a learned feature); learned competition is over the unchanged final-guard-admissible frontier. Counterfactual mode adds fixed incumbent-dominance supervision for alternative recovery." if args.objective_mode=="counterfactual" else "Train-only candidate-set correction control: guard-admissible frontier with the V64.3.17-style listwise+support objective and scalar representation; no incumbent-dominance term."),
    }
    rp=Path(args.output_report); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
