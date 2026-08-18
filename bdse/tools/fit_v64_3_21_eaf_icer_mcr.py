from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.planner.tournament import _ICER_DOMINANCE_PROFILE_BASE_NAMES

PROFILE_RETENTION_FEATURE_NAMES = list(_ICER_DOMINANCE_PROFILE_BASE_NAMES)
SCALAR_RETENTION_FEATURE_NAMES = list(_ICER_DOMINANCE_PROFILE_BASE_NAMES[:18])
_FIXED_L2 = 1.0e-3
_FIXED_HOLDOUT_FRACTION = 0.20
_FIXED_SPLIT_SEED = "v64.3.21-eaf-icer-mcr-retention-v1"


def _hash_holdout(token: str) -> bool:
    h = hashlib.sha256((_FIXED_SPLIT_SEED + "::" + token).encode()).digest()
    return int.from_bytes(h[:8], "big") / float(2**64) < _FIXED_HOLDOUT_FRACTION


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=np.float64)
    good = np.isfinite(score); y, score = y[good], score[good]
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if not pos or not neg: return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(score)+1, dtype=np.float64)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ii = np.flatnonzero(inv == i); ranks[ii] = ranks[ii].mean()
    return float((ranks[y == 1].sum() - pos*(pos+1)/2.0)/(pos*neg))


def _load_selected_incumbents(path: Path) -> dict[str, Any]:
    by_token: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r = json.loads(line); by_token.setdefault(str(r.get("scenario_token", "")), []).append(r)
    X: list[list[float]]=[]; y: list[float]=[]; tokens: list[str]=[]
    all_scenes=len(by_token); raw_proposal=0; admissible_inc=0
    for tok, rows in by_token.items():
        if not rows: continue
        try:
            legacy=int(rows[0].get("raw_top_action", -1)); anchor=int(rows[0].get("anchor_action", -1))
        except Exception: continue
        if legacy == anchor or legacy < 0: continue
        raw_proposal += 1
        inc = next((r for r in rows if int(r.get("challenger_action", -999)) == legacy), None)
        if inc is None or float(inc.get("dacer_admissible", 0.0)) < 0.5: continue
        vals=[]; ok=True
        for name in PROFILE_RETENTION_FEATURE_NAMES:
            try: v=float(inc.get(f"dacer_feature_{name}", np.nan))
            except Exception: ok=False; break
            if not np.isfinite(v): ok=False; break
            vals.append(v)
        try: tm=float(inc.get("teacher_margin", np.nan))
        except Exception: ok=False; tm=float("nan")
        if not ok or not np.isfinite(tm): continue
        X.append(vals); y.append(tm); tokens.append(tok); admissible_inc += 1
    if not X: raise SystemExit(f"no final-guard-admissible raw incumbents in {path}")
    return {"X":np.asarray(X,dtype=np.float64), "y":np.asarray(y,dtype=np.float64), "tokens":tokens,
            "all_scene_count":all_scenes, "raw_proposal_scene_count":raw_proposal, "admissible_incumbent_scene_count":admissible_inc}


def _fit_ridge(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray,float,np.ndarray,np.ndarray,float]:
    idx=np.flatnonzero(mask)
    if idx.size < 64: raise SystemExit(f"retention fit too small: {idx.size}")
    mean=X[idx].mean(axis=0); std=np.maximum(X[idx].std(axis=0),1e-6)
    Z=(X[idx]-mean[None,:])/std[None,:]
    # Positive scale only: zero in normalized target remains exactly the semantic teacher tie.
    y_scale=max(float(np.sqrt(np.mean(np.square(y[idx])))),1e-6)
    yn=y[idx]/y_scale
    # Intercept is unregularized; standardized features use fixed L2.
    A=np.concatenate([Z,np.ones((len(Z),1),dtype=np.float64)],axis=1)
    gram=(A.T@A)/len(A); rhs=(A.T@yn)/len(A)
    reg=np.eye(A.shape[1],dtype=np.float64)*_FIXED_L2; reg[-1,-1]=0.0
    coef=np.linalg.solve(gram+reg,rhs)
    return coef[:-1],float(coef[-1]),mean,std,y_scale


def _predict(X,w,b,mean,std):
    return ((X-mean[None,:])/np.maximum(std[None,:],1e-6))@w+b


def _metrics(y: np.ndarray, pred: np.ndarray, y_scale: float) -> dict[str, float]:
    yy=np.asarray(y,dtype=np.float64); pp=np.asarray(pred,dtype=np.float64)
    return {
        "auc_teacher_better_than_anchor": _auc((yy>0).astype(int),pp),
        "sign_accuracy": float(np.mean((pp>=0)==(yy>=0))),
        "mse_normalized_by_train_rms": float(np.mean(np.square(pp - yy/max(float(y_scale),1e-6)))),
        "mae_teacher_margin_units": float(np.mean(np.abs(pp*float(y_scale)-yy))),
        "predicted_negative_rate": float(np.mean(pp<0)),
        "teacher_positive_rate": float(np.mean(yy>0)),
        "fallback_regret_delta_teacher_margin_sum": float(np.sum(yy[pp<0])),
        "scene_count": float(len(yy)),
    }


def _icer(cfg: dict[str,Any]) -> dict[str,Any]:
    return cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]


def _make_cfg(base: dict[str,Any], *, w,b,mean,std,y_scale, policy: str, retention_features: list[str]) -> dict[str,Any]:
    # YAML round-trip makes a deep, serialization-faithful copy and preserves the frozen V20 heads.
    cfg=yaml.safe_load(yaml.safe_dump(base,sort_keys=False))
    ic=_icer(cfg)
    ic.update({
        "model_type":"decomposed_anchor_support_plus_dual_view_incumbent_contrastive_reliability_plus_selected_incumbent_profile_margin",
        "dominance_policy":policy,
        "incumbent_retention_policy":"selected_incumbent_profile_margin_mse" if len(retention_features)==len(PROFILE_RETENTION_FEATURE_NAMES) else "selected_incumbent_scalar_margin_mse",
        "retention_feature_names":list(retention_features),
        "retention_feature_mean":[float(x) for x in mean],
        "retention_feature_std":[float(x) for x in std],
        "retention_weights":[float(x) for x in w],
        "retention_bias":float(b),
        "retention_target_scale_teacher_margin_rms":float(y_scale),
        "retention_training_population":"train_only_raw_eaf_final_guard_admissible_selected_incumbents",
        "retention_training_target":"normalized_teacher_margin_JT_anchor_minus_JT_selected_incumbent",
        "retention_training_objective":"fixed_linear_mse_plus_l2_1e-3_zero_is_semantic_teacher_tie",
        "retention_threshold_policy":"fixed_zero_predicted_teacher_margin_no_validation_threshold_sweep",
        "dominance_corroboration_policy":"both_scalar_and_signed_profile_positive" if policy=="dual_positive_consensus_mean" else "none_equal_mean_control",
        "training_reuse":"exact_v64_3_20_support_and_v64_3_19_dominance_heads_frozen_retention_fit_train_only",
        "selection_operator":"deployment-complete asymmetric evidence burden: selected-incumbent retention margin decides incumbent-vs-anchor; alternatives require anchor support; consensus main additionally requires scalar and signed-profile dominance views both >0 before equal-mean ranking",
    })
    version="V64.3.21-EAF-ICER-MCR-DARM-DBR"
    cfg.setdefault("metadata",{})["algorithm_version"]=version
    cfg.setdefault("provenance",{})["algorithm_version"]=version
    exp=cfg.setdefault("experiment",{})
    ret_tag = "profile" if len(retention_features)==len(PROFILE_RETENTION_FEATURE_NAMES) else "scalar"
    exp["name"]=("v64_3_21_eaf_icer_mcr_"+ret_tag+"_consensus") if policy=="dual_positive_consensus_mean" else ("v64_3_21_eaf_icer_mcr_"+ret_tag+"_mean")
    exp["algorithm"]="V64.3.21 EAF-ICER-MCR: Evidence-Attributed Incumbent-Contrastive Extremal Recovery with Magnitude-aware Corroborated Reliability"
    exp["mechanism_chain"]="fixed B<=16 -> frozen EAF complete frontier + exact selected-evidence attribution -> deployment-complete admissible frontier -> selected-incumbent teacher-margin retention + corroborated incumbent dominance -> asymmetric preserve/replace/anchor -> unchanged final/structural guards"
    return cfg


def main()->None:
    ap=argparse.ArgumentParser(description="Fit V64.3.21 TRAIN-only selected-incumbent retention margin and emit mean/consensus configs.")
    ap.add_argument("--train-frontier-edges",required=True)
    ap.add_argument("--base-v20-dual-config",required=True)
    ap.add_argument("--output-scalar-retention-config",required=True)
    ap.add_argument("--output-mean-config",required=True)
    ap.add_argument("--output-consensus-config",required=True)
    ap.add_argument("--output-report",required=True)
    a=ap.parse_args()
    data=_load_selected_incumbents(Path(a.train_frontier_edges)); X,y=data["X"],data["y"]
    hold=np.asarray([_hash_holdout(t) for t in data["tokens"]],dtype=bool)
    if int(hold.sum())<128 or int((~hold).sum())<512: raise SystemExit(f"deterministic TRAIN holdout too small: train={(~hold).sum()} holdout={hold.sum()}")
    # Profile-retention fit uses all 30 registered incumbent-relative features.
    w0,b0,m0,s0,ys0=_fit_ridge(X,y,~hold); pred0=_predict(X[hold],w0,b0,m0,s0)
    hold_metrics=_metrics(y[hold],pred0,ys0)
    # Scalar-retention ablation is the exact first 18 non-atom features from the same registry.
    Xs=X[:, :len(SCALAR_RETENTION_FEATURE_NAMES)]
    sw0,sb0,sm0,ss0,sys0=_fit_ridge(Xs,y,~hold); spred0=_predict(Xs[hold],sw0,sb0,sm0,ss0)
    scalar_hold_metrics=_metrics(y[hold],spred0,sys0)
    # Promotion configs are refit on all TRAIN-only incumbents after the fixed internal diagnostic split.
    full=np.ones(len(y),dtype=bool); w,b,mean,std,y_scale=_fit_ridge(X,y,full)
    sw,sb,smean,sstd,sy_scale=_fit_ridge(Xs,y,full)
    base=yaml.safe_load(Path(a.base_v20_dual_config).read_text(encoding="utf-8"))
    base_ic=_icer(base)
    if not bool(base_ic.get("enabled",False)) or str(base_ic.get("dominance_policy",""))!="dual_equal_mean":
        raise SystemExit("base V20 dual config must contain frozen enabled ICER dual_equal_mean heads")
    if str(base_ic.get("all_flagged_policy",""))!="preserve_legacy_for_structural_guard":
        raise SystemExit("base V20 config is not deployment-complete in all-flagged domain")
    scalar_cfg=_make_cfg(base,w=sw,b=sb,mean=smean,std=sstd,y_scale=sy_scale,policy="dual_equal_mean",retention_features=SCALAR_RETENTION_FEATURE_NAMES)
    mean_cfg=_make_cfg(base,w=w,b=b,mean=mean,std=std,y_scale=y_scale,policy="dual_equal_mean",retention_features=PROFILE_RETENTION_FEATURE_NAMES)
    con_cfg=_make_cfg(base,w=w,b=b,mean=mean,std=std,y_scale=y_scale,policy="dual_positive_consensus_mean",retention_features=PROFILE_RETENTION_FEATURE_NAMES)
    for path,cfg in [(a.output_scalar_retention_config,scalar_cfg),(a.output_mean_config,mean_cfg),(a.output_consensus_config,con_cfg)]:
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    report={
        "audit":"v64_3_21_eaf_icer_mcr_train_only_retention_fit",
        **{k:v for k,v in data.items() if k.endswith("count")},
        "profile_feature_count":len(PROFILE_RETENTION_FEATURE_NAMES), "profile_retention_feature_names":PROFILE_RETENTION_FEATURE_NAMES,
        "scalar_feature_count":len(SCALAR_RETENTION_FEATURE_NAMES), "scalar_retention_feature_names":SCALAR_RETENTION_FEATURE_NAMES,
        "internal_holdout_scene_count":int(hold.sum()), "internal_fit_scene_count":int((~hold).sum()),
        "internal_holdout_profile_retention_metrics":hold_metrics,
        "internal_holdout_scalar_retention_metrics":scalar_hold_metrics,
        "full_train_teacher_margin_rms_scale":float(y_scale),
        "fixed_l2":_FIXED_L2, "holdout_fraction":_FIXED_HOLDOUT_FRACTION, "holdout_seed":_FIXED_SPLIT_SEED,
        "fit_uses_nuplan_validation":False, "fit_uses_test":False,
        "frozen_support_and_dominance_heads_reused":True,
        "zero_retention_boundary_is_semantic_teacher_tie":True,
        "interpretation":"TRAIN-only selected-incumbent magnitude readout. It is conditioned on the frozen raw-EAF incumbent actually being final-guard-admissible; it predicts anchor-minus-incumbent teacher margin with a zero-preserving target scale. No validation threshold/loss-weight/feature selection is performed.",
    }
    rp=Path(a.output_report); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__": main()
