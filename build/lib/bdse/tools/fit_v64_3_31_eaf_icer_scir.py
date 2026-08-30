from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES

BASE_FEATURE_NAMES = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
FEATURE_NAMES = [f"delta::{n}" for n in BASE_FEATURE_NAMES] + ["delta::support_logit"]
EPS = 1.0e-12
EXPECTED_FRONTIER_ROWS = 75133
EXPECTED_SCENES = 3000
FOLDS = 5
RIDGE_LAMBDA = 1.0
CONFORMAL_ALPHA = 0.05


def _finite(r: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        v = float(r.get(key, default))
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _fold(token: str) -> int:
    h = hashlib.sha256(("v64.3.31-scir-train-fold-v1::" + token).encode()).digest()
    return int.from_bytes(h[:8], "big") % FOLDS


def _read_edges(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            groups.setdefault(str(r.get("scenario_token", "")), []).append(r)
    if len(rows) != EXPECTED_FRONTIER_ROWS:
        raise SystemExit(f"V64.3.31 requires the frozen B16 TRAIN frontier: rows={len(rows)} expected={EXPECTED_FRONTIER_ROWS}")
    if len(groups) != EXPECTED_SCENES:
        raise SystemExit(f"V64.3.31 requires the frozen B16 TRAIN scenes: scenes={len(groups)} expected={EXPECTED_SCENES}")
    return rows, groups


def _scene_samples(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not group:
        return []
    inc = int(group[0].get("raw_top_action", -1))
    inc_rows = [r for r in group if int(r.get("challenger_action", -2)) == inc]
    if not inc_rows:
        return []
    inc_row = inc_rows[0]
    if _finite(inc_row, "icer_admissible", 0.0) < 0.5:
        return []
    inc_tm = _finite(inc_row, "teacher_margin")
    inc_support = _finite(inc_row, "icer_support_logit")
    if not math.isfinite(inc_tm) or not math.isfinite(inc_support):
        return []
    inc_base = np.asarray([_finite(inc_row, f"icer_feature_{n}") for n in BASE_FEATURE_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(inc_base)):
        return []
    out: list[dict[str, Any]] = []
    for r in group:
        act = int(r.get("challenger_action", -1))
        if act == inc:
            continue
        if _finite(r, "icer_admissible", 0.0) < 0.5 or _finite(r, "icer_support_logit") <= 0.0:
            continue
        cand_base = np.asarray([_finite(r, f"icer_feature_{n}") for n in BASE_FEATURE_NAMES], dtype=np.float64)
        support = _finite(r, "icer_support_logit")
        tm = _finite(r, "teacher_margin")
        if not np.all(np.isfinite(cand_base)) or not math.isfinite(support) or not math.isfinite(tm):
            continue
        x = np.concatenate([cand_base - inc_base, np.asarray([support - inc_support], dtype=np.float64)])
        out.append({
            "token": str(r.get("scenario_token", "")),
            "action": act,
            "x": x,
            "y": float(tm - inc_tm),
            "support": float(support),
            "margin": _finite(r, "raw_margin", -float("inf")),
            "utility_prior": int(_finite(r, "dacer_utility_prior", 0.0) >= 0.5),
        })
    return out


def _dataset(groups: dict[str, list[dict[str, Any]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[list[dict[str, Any]]]]:
    scene_samples: list[list[dict[str, Any]]] = []
    scene_tokens: list[str] = []
    X: list[np.ndarray] = []
    y: list[float] = []
    w: list[float] = []
    for tok, g in groups.items():
        ss = _scene_samples(g)
        if not ss:
            continue
        scene_tokens.append(tok)
        scene_samples.append(ss)
        sw = 1.0 / float(len(ss))
        for a in ss:
            X.append(a["x"])
            y.append(float(a["y"]))
            w.append(sw)
    if not X:
        raise SystemExit("V64.3.31 found no direct incumbent-admissible support-positive TRAIN alternatives")
    return np.stack(X), np.asarray(y, dtype=np.float64), np.asarray(w, dtype=np.float64), scene_tokens, scene_samples


def _fit_ridge(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, fit_mask: np.ndarray | None = None) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if fit_mask is None:
        fit_mask = np.ones((len(y),), dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    idx = np.flatnonzero(fit_mask)
    if idx.size == 0:
        raise ValueError("SCIR ridge fit has no rows")
    ww = np.asarray(sample_weight[idx], dtype=np.float64)
    ww = ww / max(float(ww.sum()), EPS)
    mean = np.sum(X[idx] * ww[:, None], axis=0)
    var = np.sum(((X[idx] - mean[None, :]) ** 2) * ww[:, None], axis=0)
    std = np.maximum(np.sqrt(var), 1.0e-6)
    Z = (X[idx] - mean[None, :]) / std[None, :]
    # Unregularized intercept + fixed ridge on the 19 auditable contrast features.
    A = np.concatenate([np.ones((len(idx), 1), dtype=np.float64), Z], axis=1)
    sw = np.sqrt(ww)[:, None]
    Aw = A * sw
    yw = y[idx] * sw[:, 0]
    reg = np.eye(A.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    reg[0, 0] = 0.0
    coef = np.linalg.solve(Aw.T @ Aw + reg, Aw.T @ yw)
    return coef[1:], float(coef[0]), mean, std


def _predict(X: np.ndarray, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip(((X - mean[None, :]) / np.maximum(std[None, :], 1.0e-6)) @ w + b, -40.0, 40.0)


def _fold_diagnostics(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    # Build per-scene samples once, then fit scene-disjoint folds with scene-equal weighting.
    token_samples = {tok: _scene_samples(g) for tok, g in groups.items()}
    token_samples = {tok: ss for tok, ss in token_samples.items() if ss}
    folds: list[dict[str, Any]] = []
    all_selected: list[float] = []
    all_opps = 0
    all_captured = 0
    for k in range(FOLDS):
        train_tokens = [t for t in token_samples if _fold(t) != k]
        test_tokens = [t for t in token_samples if _fold(t) == k]
        X=[]; y=[]; sw=[]
        for t in train_tokens:
            ss=token_samples[t]; q=1.0/len(ss)
            for a in ss: X.append(a["x"]); y.append(a["y"]); sw.append(q)
        w,b,m,s=_fit_ridge(np.stack(X),np.asarray(y),np.asarray(sw))
        selected: list[float] = []
        positive_opps=0; captured=0
        for t in test_tokens:
            ss=token_samples[t]
            xx=np.stack([a["x"] for a in ss]); mu=_predict(xx,w,b,m,s)
            positive_opps += int(any(float(a["y"])>0.0 for a in ss))
            cand=[j for j,v in enumerate(mu.tolist()) if math.isfinite(v) and v>0.0]
            if cand:
                best=sorted(cand,key=lambda j:(-float(mu[j]),-float(ss[j]["support"]),-float(ss[j]["margin"]),-int(ss[j]["utility_prior"]),int(ss[j]["action"])))[0]
                yy=float(ss[best]["y"]); selected.append(yy)
                captured += int(yy>0.0)
        arr=np.asarray(selected,dtype=np.float64)
        neg=np.minimum(arr,0.0)
        fold={
            "fold":k,
            "train_scenes":len(train_tokens),
            "holdout_scenes":len(test_tokens),
            "selected_count":int(arr.size),
            "selected_positive_count":int((arr>0).sum()) if arr.size else 0,
            "selected_precision":float((arr>0).mean()) if arr.size else float("nan"),
            "teacher_improvement_sum":float(arr.sum()) if arr.size else 0.0,
            "teacher_improvement_worst":float(arr.min()) if arr.size else float("nan"),
            "negative_rms":float(np.sqrt(np.mean(neg*neg))) if arr.size else 0.0,
            "positive_opportunity_scenes":int(positive_opps),
            "positive_capture_count":int(captured),
            "positive_capture_rate":float(captured/max(positive_opps,1)),
            "path_nonharmful":bool(arr.size>0 and float(arr.sum())>=-1e-9),
        }
        folds.append(fold); all_selected.extend(selected); all_opps+=positive_opps; all_captured+=captured
    aa=np.asarray(all_selected,dtype=np.float64)
    gate=bool(
        len(folds)==FOLDS
        and all(x["path_nonharmful"] for x in folds)
        and aa.size>=64
        and int((aa>0).sum())>=32
    )
    return {
        "folds":folds,
        "fold_pass_count":sum(int(x["path_nonharmful"]) for x in folds),
        "all_folds_selected_path_nonharmful":all(x["path_nonharmful"] for x in folds),
        "selected_count":int(aa.size),
        "selected_positive_count":int((aa>0).sum()) if aa.size else 0,
        "teacher_improvement_sum":float(aa.sum()) if aa.size else 0.0,
        "teacher_improvement_worst":float(aa.min()) if aa.size else float("nan"),
        "positive_opportunity_scenes":int(all_opps),
        "positive_capture_count":int(all_captured),
        "positive_capture_rate":float(all_captured/max(all_opps,1)),
        "train_gate_pass":gate,
    }


def main() -> None:
    ap=argparse.ArgumentParser(description="Fit V64.3.31 EAF-ICER-SCIR scene-equal incumbent-contrastive improvement regression.")
    ap.add_argument("--train-frontier-edges",required=True)
    ap.add_argument("--base-config",required=True)
    ap.add_argument("--output-preserve-config",required=True)
    ap.add_argument("--output-rank-config",required=True)
    ap.add_argument("--output-report",required=True)
    args=ap.parse_args()

    _,groups=_read_edges(Path(args.train_frontier_edges))
    X,y,sw,scene_tokens,scene_samples=_dataset(groups)
    cf=_fold_diagnostics(groups)
    report={
        "audit":"v64_3_31_eaf_icer_scir_fit",
        "scientific_role":"TRAIN_only_selection_aligned_intervention_utility_fit_before_independent_conformal_calibration",
        "frozen_train_scenes":len(groups),
        "direct_support_positive_training_scenes":len(scene_tokens),
        "direct_support_positive_training_edges":int(len(y)),
        "teacher_improvement_positive_fraction":float((y>0).mean()),
        "teacher_improvement_sum":float(y.sum()),
        "feature_names":FEATURE_NAMES,
        "base_feature_names":BASE_FEATURE_NAMES,
        "ridge_lambda":RIDGE_LAMBDA,
        "scene_equal_weighting":True,
        "target":"teacher_candidate_minus_same_scene_incumbent_improvement",
        "crossfit":cf,
        "train_gate_pass":bool(cf["train_gate_pass"]),
        "train_gate_contract":{
            "scene_folds":FOLDS,
            "all_folds_selected_path_teacher_improvement_sum_min":0.0,
            "aggregate_selected_count_min":64,
            "aggregate_selected_positive_count_min":32,
            "no_validation_threshold_or_weight_sweep":True,
        },
        "fit_uses_validation":False,
        "fit_uses_test":False,
    }
    rp=Path(args.output_report); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    if not report["train_gate_pass"]:
        print(json.dumps(report,indent=2,sort_keys=True))
        raise SystemExit("V64.3.31 SCIR TRAIN selected-path gate failed; STOP before spending calibration/fresh data")

    w,b,mean,std=_fit_ridge(X,y,sw)
    cfg=yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    ic=cfg.setdefault("runtime",{}).setdefault("decisive_frontier_value",{}).setdefault("incumbent_contrastive_extremal_recovery",{})
    # Keep frozen anchor support/admissibility; SCIR replaces only direct
    # incumbent-relative dominance/ranking semantics.  Preserve an admissible
    # incumbent by construction so the experiment is not confounded by a
    # learned incumbent->anchor path.
    # Keep the frozen V20 dominance head/policy bit-identical for diagnostics;
    # SCIR explicitly bypasses it for direct eligibility/ranking.
    ic["incumbent_retention_policy"]="preserve_admissible_incumbent"
    ic["regret_risk_enabled"]=False
    ic["retention_regret_risk_enabled"]=False
    ic["replacement_regret_risk_enabled"]=False
    # Causal preservation control: freeze the old V20 direct dominance semantics
    # while removing the already-falsified learned veto from an admissible incumbent.
    # SCIR rank/main are compared against this control so any direct-ordering gain
    # cannot be attributed to the preservation correction alone.
    preserve_cfg = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False))
    preserve_ic = preserve_cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    preserve_ic["selection_conditioned_intervention_recovery"] = {"enabled": False}
    preserve_cfg.setdefault("metadata",{})["algorithm_version"]="V64.3.31-EAF-ICER-PRESERVE-CONTROL"
    preserve_cfg.setdefault("provenance",{})["algorithm_version"]="V64.3.31-EAF-ICER-PRESERVE-CONTROL"
    preserve_exp=preserve_cfg.setdefault("experiment",{})
    preserve_exp["name"]="v64_3_31_eaf_icer_preserve_control"
    preserve_exp["algorithm"]="V64.3.31 preservation control: frozen V20 direct dominance with admissible-incumbent default"
    preserve_exp["mechanism_chain"]="frozen B16/V20 EAF + support/dominance -> preserve deployment-admissible incumbent by default -> unchanged direct V20 alternative operator -> unchanged final/structural guards"
    preserve_out=Path(args.output_preserve_config); preserve_out.parent.mkdir(parents=True,exist_ok=True); preserve_out.write_text(yaml.safe_dump(preserve_cfg,sort_keys=False),encoding="utf-8")

    scir=ic.setdefault("selection_conditioned_intervention_recovery",{})
    scir.update({
        "enabled":True,
        "mode":"rank_only",
        "model_type":"scene_equal_linear_ridge_same_scene_incumbent_contrastive_improvement",
        "base_feature_names":BASE_FEATURE_NAMES,
        "feature_names":FEATURE_NAMES,
        "feature_mean":[float(x) for x in mean],
        "feature_std":[float(x) for x in std],
        "weights":[float(x) for x in w],
        "bias":float(b),
        "ridge_lambda":RIDGE_LAMBDA,
        "training_population":"TRAIN_only_incumbent_deployment_admissible_support_positive_alternatives",
        "training_weighting":"each_scene_total_weight_1",
        "training_target":"continuous_teacher_candidate_minus_same_scene_incumbent_improvement",
        "proposal_operator":"support_positive_argmax_predicted_improvement_with_semantic_zero_incumbent_boundary",
        "require_positive_predicted_improvement":True,
        "conformal_alpha":CONFORMAL_ALPHA,
        "conformal_overprediction_quantile":0.0,
        "conformal_calibration_status":"not_yet_calibrated_rank_only",
        "no_fallback":True,
    })
    cfg.setdefault("metadata",{})["algorithm_version"]="V64.3.31-EAF-ICER-SCIR-RANK"
    cfg.setdefault("provenance",{})["algorithm_version"]="V64.3.31-EAF-ICER-SCIR-RANK"
    exp=cfg.setdefault("experiment",{})
    exp["name"]="v64_3_31_eaf_icer_scir_rank"
    exp["algorithm"]="V64.3.31 EAF-ICER-SCIR: Selection-Conditioned Intervention Recovery (rank calibration arm)"
    exp["mechanism_chain"]="bounded B16 interface -> exact EAF attribution -> deployment-admissible direct intervention -> same-scene incumbent-contrastive continuous improvement -> extremal proposal -> incumbent default -> unchanged final/structural guards"
    out=Path(args.output_rank_config); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    report["model"]={"weights":[float(x) for x in w],"bias":float(b),"feature_mean":[float(x) for x in mean],"feature_std":[float(x) for x in std]}
    rp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({"pass":True,"train_gate_pass":True,"selected_count":cf["selected_count"],"fold_pass_count":cf["fold_pass_count"],"output_preserve_config":str(preserve_out),"output_rank_config":str(out)},sort_keys=True))


if __name__=="__main__":
    main()
