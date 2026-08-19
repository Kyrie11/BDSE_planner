from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from bdse.planner.tournament import (
    _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES,
    _ICER_TRANSITION_FEATURE_NAMES,
)

_FIXED_HOLDOUT_FRACTION = 0.20
_FIXED_SPLIT_SEED = "v64.3.22-eaf-icer-tcr-regret-risk-v1"
_FIXED_L2 = 1.0e-3
_FIXED_MAX_ITER = 120


def _hash_holdout(token: str) -> bool:
    h = hashlib.sha256((_FIXED_SPLIT_SEED + "::" + str(token)).encode()).digest()
    return int.from_bytes(h[:8], "big") / float(2**64) < _FIXED_HOLDOUT_FRACTION


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=np.float64)
    good = np.isfinite(score); y, score = y[good], score[good]
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ii = np.flatnonzero(inv == i); ranks[ii] = ranks[ii].mean()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _icer(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]


def _runtime_feature_names(mode: str) -> list[str]:
    names = [f"evidence::{n}" for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]
    if mode == "transition_conditioned":
        names += [f"transition::{n}" for n in _ICER_TRANSITION_FEATURE_NAMES]
    elif mode != "evidence_only":
        raise ValueError(mode)
    return names


def _row_feature(r: dict[str, Any], mode: str, transition_prefix: str) -> list[float] | None:
    vals: list[float] = []
    for name in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES:
        try:
            v = float(r.get(f"icer_feature_{name}", np.nan))
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        vals.append(v)
    if mode == "transition_conditioned":
        for name in _ICER_TRANSITION_FEATURE_NAMES:
            try:
                v = float(r.get(f"icer_transition_{transition_prefix}_{name}", np.nan))
            except Exception:
                return None
            if not np.isfinite(v):
                return None
            vals.append(v)
    return vals


def _load_scenes(path: Path) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by.setdefault(str(r.get("scenario_token", "")), []).append(r)
    if not by:
        raise SystemExit(f"no frontier rows in {path}")
    return by


def _build_data(by: dict[str, list[dict[str, Any]]], mode: str) -> dict[str, Any]:
    ret_x: list[list[float]] = []; ret_delta: list[float] = []; ret_tok: list[str] = []
    rep_x: list[list[float]] = []; rep_delta: list[float] = []; rep_tok: list[str] = []
    rep_support: list[float] = []; rep_scalar_dom: list[float] = []; rep_dual_dom: list[float] = []
    rep_action: list[int] = []; rep_legacy: list[int] = []
    proposal_scenes = admissible_inc_scenes = multi_alt_scenes = 0
    transition_nonzero = transition_total = 0
    for tok, rows in by.items():
        if not rows:
            continue
        anchor = int(rows[0].get("anchor_action", -1)); legacy = int(rows[0].get("raw_top_action", -1))
        if legacy < 0 or legacy == anchor:
            continue
        proposal_scenes += 1
        inc = next((r for r in rows if int(r.get("challenger_action", -999)) == legacy), None)
        if inc is None or float(inc.get("icer_admissible", inc.get("dacer_admissible", 0.0))) < 0.5:
            continue
        try:
            inc_tm = float(inc.get("teacher_margin", np.nan))
        except Exception:
            continue
        if not np.isfinite(inc_tm):
            continue
        rv = _row_feature(inc, mode, "anchor")
        if rv is None:
            continue
        ret_x.append(rv); ret_delta.append(inc_tm); ret_tok.append(tok); admissible_inc_scenes += 1
        if mode == "transition_conditioned":
            t = np.asarray(rv[len(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES):], dtype=np.float64)
            transition_total += 1; transition_nonzero += int(float(np.max(np.abs(t))) > 1.0e-8)
        alt_count = 0
        for r in rows:
            ch = int(r.get("challenger_action", -1))
            if ch < 0 or ch in {anchor, legacy}:
                continue
            if float(r.get("icer_admissible", r.get("dacer_admissible", 0.0))) < 0.5:
                continue
            try:
                tm = float(r.get("teacher_margin", np.nan))
                sup = float(r.get("icer_support_logit", np.nan))
                sdom = float(r.get("icer_scalar_dominance_logit", np.nan))
                pdom = float(r.get("icer_profile_dominance_logit", np.nan))
            except Exception:
                continue
            if not all(np.isfinite(x) for x in [tm, sup, sdom, pdom]):
                continue
            dual_dom = 0.5 * (sdom + pdom)
            # Selection-conditioned replacement population.  Use only the frozen
            # scalar-dominance-positive extremal population so the risk head is
            # identical for scalar and signed-profile arms.  The signed selected-
            # evidence view is therefore a *ranking-only* ablation at runtime,
            # never a hidden TRAIN-sample gate.
            if not (sup > 0.0 and sdom > 0.0):
                continue
            xv = _row_feature(r, mode, "incumbent")
            if xv is None:
                continue
            rep_x.append(xv); rep_delta.append(tm - inc_tm); rep_tok.append(tok)
            rep_support.append(sup); rep_scalar_dom.append(sdom); rep_dual_dom.append(dual_dom)
            rep_action.append(ch); rep_legacy.append(legacy); alt_count += 1
            if mode == "transition_conditioned":
                t = np.asarray(xv[len(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES):], dtype=np.float64)
                transition_total += 1; transition_nonzero += int(float(np.max(np.abs(t))) > 1.0e-8)
        multi_alt_scenes += int(alt_count >= 2)
    if len(ret_x) < 256 or len(rep_x) < 1024:
        raise SystemExit(f"insufficient TRAIN support for V64.3.22: retention={len(ret_x)} replacement={len(rep_x)}")
    return {
        "ret_X": np.asarray(ret_x, dtype=np.float64), "ret_delta": np.asarray(ret_delta, dtype=np.float64), "ret_tok": ret_tok,
        "rep_X": np.asarray(rep_x, dtype=np.float64), "rep_delta": np.asarray(rep_delta, dtype=np.float64), "rep_tok": rep_tok,
        "rep_support": np.asarray(rep_support, dtype=np.float64), "rep_scalar_dom": np.asarray(rep_scalar_dom, dtype=np.float64),
        "rep_dual_dom": np.asarray(rep_dual_dom, dtype=np.float64), "rep_action": np.asarray(rep_action, dtype=np.int64),
        "rep_legacy": np.asarray(rep_legacy, dtype=np.int64),
        "proposal_scene_count": proposal_scenes, "admissible_incumbent_scene_count": admissible_inc_scenes,
        "multi_alternative_scene_count": multi_alt_scenes,
        "transition_nonzero_fraction": float(transition_nonzero / transition_total) if transition_total else float("nan"),
    }


def _fit_weighted_logistic(X: np.ndarray, delta: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    idx = np.flatnonzero(fit)
    if idx.size < 128:
        raise SystemExit(f"weighted logistic fit too small: {idx.size}")
    labels=(delta[idx] > 0.0).astype(np.float32)
    if int(labels.sum()) == 0 or int(labels.sum()) == int(len(labels)):
        raise SystemExit("weighted regret-risk fit requires both positive and negative TRAIN outcomes")
    mean = X[idx].mean(axis=0); std = np.maximum(X[idx].std(axis=0), 1.0e-6)
    z = torch.tensor((X[idx] - mean[None, :]) / std[None, :], dtype=torch.float32)
    y = torch.tensor(labels)
    mag = np.abs(delta[idx]).astype(np.float64)
    # Exact cost-sensitive semantics: magnitude weights are normalized only by a
    # positive scalar, so the zero-logit decision boundary is unchanged.
    mag = np.maximum(mag, 1.0e-4); mag /= max(float(np.mean(mag)), 1.0e-8)
    wt = torch.tensor(mag.astype(np.float32))
    w = torch.zeros((X.shape[1],), dtype=torch.float32, requires_grad=True)
    b = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=_FIXED_MAX_ITER, line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9)
    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        logit = z @ w + b
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * wt).mean()
        loss = loss + _FIXED_L2 * w.square().mean()
        loss.backward(); return loss
    opt.step(closure)
    return w.detach().numpy().astype(float), float(b.detach()), mean.astype(float), std.astype(float)


def _predict(X: np.ndarray, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip(((X - mean[None, :]) / np.maximum(std[None, :], 1.0e-6)) @ w + b, -40.0, 40.0)


def _risk_metrics(delta: np.ndarray, score: np.ndarray) -> dict[str, float]:
    neg = score < 0.0; pos = score > 0.0
    return {
        "auc_positive_teacher_improvement": _auc((delta > 0.0).astype(int), score),
        "sign_accuracy": float(np.mean((score >= 0.0) == (delta >= 0.0))),
        "positive_rate": float(np.mean(pos)),
        "true_positive_rate": float(np.mean(delta > 0.0)),
        "teacher_improvement_sum_on_predicted_positive": float(np.sum(delta[pos])),
        "teacher_improvement_sum_on_predicted_negative": float(np.sum(delta[neg])),
        "predicted_positive_precision": float(np.mean(delta[pos] > 0.0)) if np.any(pos) else float("nan"),
        "scene_or_edge_count": float(len(delta)),
    }


def _selection_metrics(data: dict[str, Any], score: np.ndarray, hold_tokens: set[str], dominance: str) -> dict[str, float]:
    toks = np.asarray(data["rep_tok"], dtype=object)
    use = np.asarray([str(t) in hold_tokens for t in toks], dtype=bool)
    support = data["rep_support"]
    scalar_dom = data["rep_scalar_dom"]
    rank_dom = scalar_dom if dominance == "scalar" else data["rep_dual_dom"]
    delta = data["rep_delta"]
    selected_delta: list[float] = []; opportunities = captures = scenes = 0
    for tok in sorted(hold_tokens):
        idx = np.flatnonzero(use & (toks == tok))
        if idx.size == 0:
            continue
        scenes += 1
        opportunity = bool(np.any(delta[idx] > 0.0)); opportunities += int(opportunity)
        eligible = idx[(support[idx] > 0.0) & (scalar_dom[idx] > 0.0) & (score[idx] > 0.0)]
        if eligible.size:
            # Scalar dominance fixes eligibility in every ablation.  Signed-profile
            # attribution may only change ranking, while regret risk is a veto/tie
            # view rather than a second uncalibrated extremal objective.
            order = sorted(eligible.tolist(), key=lambda j: (-float(rank_dom[j]), -float(score[j]), int(data["rep_action"][j])))
            j = int(order[0]); selected_delta.append(float(delta[j])); captures += int(float(delta[j]) > 0.0)
    arr = np.asarray(selected_delta, dtype=np.float64)
    return {
        "holdout_direct_replacement_count": float(len(arr)),
        "holdout_direct_replacement_precision": float(np.mean(arr > 0.0)) if arr.size else float("nan"),
        "holdout_direct_replacement_teacher_improvement_sum": float(arr.sum()) if arr.size else 0.0,
        "holdout_direct_replacement_teacher_improvement_mean": float(arr.mean()) if arr.size else float("nan"),
        "holdout_opportunity_count": float(opportunities),
        "holdout_opportunity_capture_rate": float(captures / opportunities) if opportunities else float("nan"),
        "holdout_scene_count": float(scenes),
    }


def _fit_mode(data: dict[str, Any], mode: str) -> dict[str, Any]:
    ret_hold = np.asarray([_hash_holdout(t) for t in data["ret_tok"]], dtype=bool)
    rep_hold = np.asarray([_hash_holdout(t) for t in data["rep_tok"]], dtype=bool)
    if int(ret_hold.sum()) < 128 or int(rep_hold.sum()) < 256:
        raise SystemExit(f"TRAIN internal holdout too small for {mode}: retention={ret_hold.sum()} replacement={rep_hold.sum()}")
    rw, rb, rm, rs = _fit_weighted_logistic(data["ret_X"], data["ret_delta"], ~ret_hold)
    pw, pb, pm, ps = _fit_weighted_logistic(data["rep_X"], data["rep_delta"], ~rep_hold)
    rscore = _predict(data["ret_X"][ret_hold], rw, rb, rm, rs)
    pscore_all = _predict(data["rep_X"], pw, pb, pm, ps)
    pscore = pscore_all[rep_hold]
    ret_metrics = _risk_metrics(data["ret_delta"][ret_hold], rscore)
    rep_metrics = _risk_metrics(data["rep_delta"][rep_hold], pscore)
    hold_tokens = {str(t) for t, h in zip(data["rep_tok"], rep_hold.tolist()) if h}
    sel_scalar = _selection_metrics(data, pscore_all, hold_tokens, "scalar")
    sel_dual = _selection_metrics(data, pscore_all, hold_tokens, "dual")
    # Hard TRAIN-only fail-close contracts that V64.3.21 lacked.
    retention_safe = bool(ret_metrics["teacher_improvement_sum_on_predicted_negative"] <= 1.0e-9)
    replacement_safe = bool(sel_dual["holdout_direct_replacement_count"] >= 8 and sel_dual["holdout_direct_replacement_teacher_improvement_sum"] >= -1.0e-9)
    # Refit only after the diagnostic split; fresh validation never participates.
    full_ret = np.ones(len(data["ret_delta"]), dtype=bool); full_rep = np.ones(len(data["rep_delta"]), dtype=bool)
    frw, frb, frm, frs = _fit_weighted_logistic(data["ret_X"], data["ret_delta"], full_ret)
    fpw, fpb, fpm, fps = _fit_weighted_logistic(data["rep_X"], data["rep_delta"], full_rep)
    return {
        "mode": mode,
        "feature_names": _runtime_feature_names(mode),
        "retention": {"weights": frw, "bias": frb, "mean": frm, "std": frs, "holdout": ret_metrics, "train_holdout_safe": retention_safe},
        "replacement": {"weights": fpw, "bias": fpb, "mean": fpm, "std": fps, "holdout": rep_metrics, "selection_scalar": sel_scalar, "selection_dual": sel_dual, "train_holdout_safe": replacement_safe},
        "train_holdout_safe": bool(retention_safe and replacement_safe),
    }


def _make_cfg(base: dict[str, Any], fit: dict[str, Any], dominance_policy: str, tag: str) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False)); ic = _icer(cfg)
    if not bool(ic.get("enabled", False)):
        raise SystemExit("base V20 ICER config must be enabled")
    mode = fit["mode"]; names = fit["feature_names"]
    r, p = fit["retention"], fit["replacement"]
    ic.update({
        "model_type": "frozen_anchor_support_plus_incumbent_contrastive_dominance_plus_transition_conditioned_regret_risk",
        "dominance_policy": dominance_policy,
        "incumbent_retention_policy": "preserve_admissible_incumbent",
        "regret_risk_enabled": True,
        "regret_risk_feature_mode": mode,
        "retention_regret_risk_feature_names": names,
        "retention_regret_risk_feature_mean": [float(x) for x in r["mean"]],
        "retention_regret_risk_feature_std": [float(x) for x in r["std"]],
        "retention_regret_risk_weights": [float(x) for x in r["weights"]],
        "retention_regret_risk_bias": float(r["bias"]),
        "replacement_regret_risk_feature_names": names,
        "replacement_regret_risk_feature_mean": [float(x) for x in p["mean"]],
        "replacement_regret_risk_feature_std": [float(x) for x in p["std"]],
        "replacement_regret_risk_weights": [float(x) for x in p["weights"]],
        "replacement_regret_risk_bias": float(p["bias"]),
        "regret_risk_objective": "magnitude_weighted_binary_logistic_expected_improvement_fixed_zero_boundary_l2_1e-3",
        "regret_risk_training_population": "TRAIN_only_final_guard_admissible_raw_incumbents_and_frozen_support_positive_scalar_dominance_positive_alternatives",
        "retention_regret_target": "teacher_margin_incumbent_vs_anchor",
        "replacement_regret_target": "teacher_margin_candidate_minus_teacher_margin_raw_incumbent",
        "regret_risk_threshold_policy": "fixed_zero_expected_improvement_boundary_no_validation_sweep",
        "replacement_operator": "anchor_support>0 AND frozen_scalar_dominance>0 AND regret_risk>0; scalar arm ranks by scalar dominance; signed-profile arm ranks by equal-mean scalar/profile dominance; risk logit only tie-breaks",
        "retention_operator": "selected_incumbent preserved unless TRAIN-only magnitude-weighted retention risk logit<0; all-flagged structural delegation unchanged",
        "train_holdout_retention_path_safe": bool(r["train_holdout_safe"]),
        "train_holdout_replacement_path_safe": bool(p["train_holdout_safe"]),
        "all_flagged_policy": "preserve_legacy_for_structural_guard",
    })
    version = "V64.3.22-EAF-ICER-TCR-DARM-DBR"
    cfg.setdefault("metadata", {})["algorithm_version"] = version
    cfg.setdefault("provenance", {})["algorithm_version"] = version
    exp = cfg.setdefault("experiment", {})
    exp["name"] = f"v64_3_22_eaf_icer_tcr_{tag}"
    exp["algorithm"] = "V64.3.22 EAF-ICER-TCR: Evidence-Attributed Incumbent-Contrastive Extremal Recovery with Transition-Conditioned Regret Reliability"
    exp["mechanism_chain"] = "fixed B<=16 -> frozen EAF complete frontier + exact selected-evidence attribution -> deployment-complete admissible frontier -> frozen support/dominance -> transition-conditioned magnitude-weighted regret risk -> conservative extremal replacement/retention -> unchanged final/structural guards"
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.22 TRAIN-only magnitude-weighted regret-risk heads.")
    ap.add_argument("--train-frontier-edges", required=True)
    ap.add_argument("--base-v20-dual-config", required=True)
    ap.add_argument("--output-evidence-risk-config", required=True)
    ap.add_argument("--output-transition-scalar-config", required=True)
    ap.add_argument("--output-transition-dual-config", required=True)
    ap.add_argument("--output-report", required=True)
    a = ap.parse_args()
    by = _load_scenes(Path(a.train_frontier_edges))
    evidence = _build_data(by, "evidence_only")
    transition = _build_data(by, "transition_conditioned")
    ef = _fit_mode(evidence, "evidence_only")
    tf = _fit_mode(transition, "transition_conditioned")
    if not np.isfinite(float(transition.get("transition_nonzero_fraction", np.nan))) or float(transition["transition_nonzero_fraction"]) < 0.95:
        raise SystemExit(f"STOP TRAIN INSTRUMENTATION: transition feature coverage too low: {transition.get('transition_nonzero_fraction')}")
    if not tf["train_holdout_safe"]:
        raise SystemExit("STOP TRAIN RISK: transition-conditioned retention/replacement path is not non-harmful on deterministic TRAIN holdout; do not spend fresh validation GPU")
    base = yaml.safe_load(Path(a.base_v20_dual_config).read_text(encoding="utf-8"))
    base_ic = _icer(base)
    if str(base_ic.get("dominance_policy", "")) != "dual_equal_mean" or str(base_ic.get("all_flagged_policy", "")) != "preserve_legacy_for_structural_guard":
        raise SystemExit("base config must be frozen V20 dual_equal_mean deployment-complete ICER")
    out_e = _make_cfg(base, ef, "scalar_positive_dual_equal_mean", "evidence_risk_dual")
    out_s = _make_cfg(base, tf, "scalar_only", "transition_risk_scalar")
    out_d = _make_cfg(base, tf, "scalar_positive_dual_equal_mean", "transition_risk_dual")
    for path, cfg in [(a.output_evidence_risk_config, out_e), (a.output_transition_scalar_config, out_s), (a.output_transition_dual_config, out_d)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    report = {
        "version": "V64.3.22",
        "train_frontier_edges": str(a.train_frontier_edges),
        "evidence_only": {**{k:v for k,v in evidence.items() if not isinstance(v, (np.ndarray, list))}, "fit": _jsonable(ef)},
        "transition_conditioned": {**{k:v for k,v in transition.items() if not isinstance(v, (np.ndarray, list))}, "fit": _jsonable(tf)},
        "main_train_holdout_safe": bool(tf["train_holdout_safe"]),
        "no_fresh_validation_used": True,
    }
    Path(a.output_report).parent.mkdir(parents=True, exist_ok=True); Path(a.output_report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"main_train_holdout_safe": tf["train_holdout_safe"], "transition_nonzero_fraction": transition["transition_nonzero_fraction"], "transition_dual_selection": tf["replacement"]["selection_dual"]}, sort_keys=True))


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict): return {k:_jsonable(v) for k,v in x.items()}
    if isinstance(x, np.ndarray): return [float(v) for v in x.reshape(-1)]
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, list): return [_jsonable(v) for v in x]
    return x


if __name__ == "__main__":
    main()
