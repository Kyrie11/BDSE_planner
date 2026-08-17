from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

FEATURE_NAMES = [
    "raw_margin",
    "proposed_attribution_scale",
    "frontier_residual_rms",
    "frontier_residual_abs_mean",
    "frontier_attribution_scale_rms",
    "frontier_attribution_scale_mean",
    "evidence_certificate_fraction",
    "valid_action_count_norm",
    "margin_over_attribution",
    "proposed_over_frontier_attribution",
]


def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if math.isfinite(v) else float(default)


def _row_features(row: dict[str, Any], *, ratio_floor: float = 1.0e-3) -> np.ndarray:
    # Prefer the V64.3.15 explicitly instrumented fields, but accept the V64.3.14
    # equivalents so the fitter can be regression-tested on the uploaded screen.
    raw_margin = _finite(row.get("decisive_frontier_eair_feature_raw_margin", row.get("pair_action_anchor_raw_margin")))
    proposed_attr = _finite(row.get("decisive_frontier_eair_feature_proposed_attribution_scale", row.get("decisive_frontier_ocfi_proposed_attribution_scale")))
    residual_rms = _finite(row.get("decisive_frontier_eair_feature_frontier_residual_rms", row.get("decisive_frontier_value_residual_rms")))
    residual_abs = _finite(row.get("decisive_frontier_eair_feature_frontier_residual_abs_mean", row.get("decisive_frontier_value_residual_abs_mean")))
    attr_rms = _finite(row.get("decisive_frontier_eair_feature_frontier_attribution_scale_rms", row.get("decisive_frontier_value_attribution_scale_rms")))
    attr_mean = _finite(row.get("decisive_frontier_eair_feature_frontier_attribution_scale_mean", row.get("decisive_frontier_value_attribution_scale_mean")))
    cert = _finite(row.get("decisive_frontier_eair_feature_evidence_certificate_fraction", row.get("evidence_certificate_fraction")))
    valid_norm = row.get("decisive_frontier_eair_feature_valid_action_count_norm", None)
    if valid_norm is None:
        valid_norm = _finite(row.get("valid_action_count"), 0.0) / 32.0
    else:
        valid_norm = _finite(valid_norm)
    margin_over_attr = row.get("decisive_frontier_eair_feature_margin_over_attribution", None)
    if margin_over_attr is None:
        margin_over_attr = raw_margin / max(proposed_attr, ratio_floor)
    proposed_over_frontier = row.get("decisive_frontier_eair_feature_proposed_over_frontier_attribution", None)
    if proposed_over_frontier is None:
        proposed_over_frontier = proposed_attr / max(attr_rms, ratio_floor)
    vals = np.asarray([
        raw_margin,
        proposed_attr,
        residual_rms,
        residual_abs,
        attr_rms,
        attr_mean,
        cert,
        valid_norm,
        _finite(margin_over_attr),
        _finite(proposed_over_frontier),
    ], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise ValueError("non-finite EAIR feature vector")
    return vals


def _load_rows(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    X: list[np.ndarray] = []
    y: list[int] = []
    tokens: list[str] = []
    teacher_margin: list[float] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        a = row.get("raw_frontier_anchor_action", row.get("pair_action_anchor_raw_anchor_action"))
        b = row.get("raw_frontier_proposed_action", row.get("pair_action_anchor_raw_proposed_action"))
        margin = row.get("decisive_frontier_value_teacher_proposed_vs_anchor_margin", None)
        if a is None or b is None or int(a) == int(b) or margin is None:
            continue
        m = _finite(margin, float("nan"))
        if not math.isfinite(m):
            continue
        X.append(_row_features(row))
        y.append(int(m > 0.0))
        tokens.append(str(row.get("scenario_token", len(tokens))))
        teacher_margin.append(m)
    if not X:
        raise SystemExit(f"no valid raw EAF proposal edges in {path}")
    return np.stack(X), np.asarray(y, dtype=np.float64), tokens, np.asarray(teacher_margin, dtype=np.float64)


def _hash_holdout(token: str, seed: str, holdout_fraction: float) -> bool:
    h = hashlib.sha256((seed + "::" + token).encode()).digest()
    u = int.from_bytes(h[:8], "big") / float(2**64)
    return u < holdout_fraction


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # Average ranks for ties.
    vals, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    if np.any(cnt > 1):
        for i, c in enumerate(cnt):
            if c > 1:
                idx = np.flatnonzero(inv == i)
                ranks[idx] = ranks[idx].mean()
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _fit(X: np.ndarray, y: np.ndarray, *, steps: int, lr: float, l2: float, seed: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.maximum(std, 1.0e-6)
    Z = (X - mean) / std
    torch.manual_seed(seed)
    z = torch.tensor(Z, dtype=torch.float32)
    t = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros((Z.shape[1],), dtype=torch.float32, requires_grad=True)
    b = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    npos = max(float((y > 0.5).sum()), 1.0)
    nneg = max(float((y <= 0.5).sum()), 1.0)
    class_weight = torch.where(t > 0.5, torch.tensor(len(y)/(2.0*npos)), torch.tensor(len(y)/(2.0*nneg)))
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        logits = z @ w + b
        loss_vec = torch.nn.functional.binary_cross_entropy_with_logits(logits, t, reduction="none")
        loss = (loss_vec * class_weight).mean() + float(l2) * w.square().mean()
        loss.backward()
        opt.step()
    return w.detach().cpu().numpy().astype(float), float(b.detach()), mean.astype(float), std.astype(float)


def _predict(X: np.ndarray, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (X - mean) / np.maximum(std, 1.0e-6)
    logit = z @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.15 EAF-EAIR learned one-sided intervention reliability readout.")
    ap.add_argument("--train-per-sample", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-config", required=True)
    ap.add_argument("--output-report", required=True)
    ap.add_argument("--holdout-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", default="v64.3.15-eaf-eair-v1")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--min-proposals", type=int, default=256)
    ap.add_argument("--min-class", type=int, default=32)
    args = ap.parse_args()

    X, y, tokens, margins = _load_rows(Path(args.train_per_sample))
    if len(y) < args.min_proposals or min(int(y.sum()), int((1-y).sum())) < args.min_class:
        raise SystemExit(f"insufficient EAIR train proposals/classes: n={len(y)} pos={int(y.sum())} neg={int((1-y).sum())}")
    hold = np.asarray([_hash_holdout(t, args.split_seed, args.holdout_fraction) for t in tokens], dtype=bool)
    if hold.sum() < args.min_class or (~hold).sum() < args.min_class:
        raise SystemExit("EAIR deterministic internal holdout is too small")
    w0, b0, m0, s0 = _fit(X[~hold], y[~hold], steps=args.steps, lr=args.lr, l2=args.l2, seed=1515)
    p_hold = _predict(X[hold], w0, b0, m0, s0)
    auc_hold = _auc(y[hold], p_hold)
    acc_hold = float(((p_hold >= 0.5) == (y[hold] > 0.5)).mean())
    # Refit on every train proposal after the capacity diagnostic.  The held-out
    # nuPlan validation set remains untouched by this fitting tool.
    w, b, mean, std = _fit(X, y, steps=args.steps, lr=args.lr, l2=args.l2, seed=1515)

    cfg = yaml.safe_load(Path(args.base_config).read_text())
    runtime = cfg.setdefault("runtime", {})
    frontier = runtime.setdefault("decisive_frontier_value", {})
    ocfi = frontier.setdefault("one_sided_intervention", {})
    ocfi["enabled"] = False
    eair = frontier.setdefault("learned_intervention_reliability", {})
    eair.update({
        "enabled": True,
        "instrument_features": True,
        "model_type": "standardized_logistic_teacher_better_edge",
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": [float(x) for x in mean],
        "feature_std": [float(x) for x in std],
        "weights": [float(x) for x in w],
        "bias": float(b),
        "min_probability": 0.5,
        "ratio_floor": 1.0e-3,
        "valid_action_normalizer": 32.0,
        "require_frontier_active": True,
        "training_target": "teacher_proposed_vs_darm_anchor_margin_positive",
        "threshold_policy": "fixed_0.5_no_validation_threshold_sweep",
    })
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.15-EAF-EAIR-DARM-DBR"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.15-EAF-EAIR-DARM-DBR"
    exp = cfg.setdefault("experiment", {})
    exp["name"] = "v64_3_15_eaf_eair_calibrated"
    exp["algorithm"] = "V64.3.15 EAF-EAIR: frozen EAF decisive-frontier value plus Evidence-Attributed Intervention Reliability"
    exp["mechanism_chain"] = "fixed B=16 selected evidence -> frozen EAF complete frontier value -> learned runtime-only evidence-attributed one-sided reliability -> legacy evidence certificate -> final decision preservation"
    Path(args.output_config).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_config).write_text(yaml.safe_dump(cfg, sort_keys=False))

    report = {
        "audit": "v64_3_15_eaf_eair_fit",
        "train_proposal_edges": int(len(y)),
        "train_positive_fraction": float(y.mean()),
        "internal_holdout_edges": int(hold.sum()),
        "internal_holdout_auc": float(auc_hold),
        "internal_holdout_accuracy_at_0_5": acc_hold,
        "teacher_margin_mean": float(margins.mean()),
        "teacher_margin_positive_mean": float(margins[y > 0.5].mean()),
        "teacher_margin_negative_mean": float(margins[y <= 0.5].mean()),
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": [float(x) for x in mean],
        "feature_std": [float(x) for x in std],
        "weights": [float(x) for x in w],
        "bias": float(b),
        "min_probability": 0.5,
        "fit_uses_nuplan_validation": False,
        "fit_uses_test": False,
        "interpretation": "AUC measures whether frozen EAF runtime statistics contain one-sided teacher-improvement information. It is a capacity/readout diagnostic, not a paper result by itself.",
    }
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_report).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
