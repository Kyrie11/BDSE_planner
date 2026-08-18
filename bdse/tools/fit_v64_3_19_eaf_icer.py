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
    _DACER_FEATURE_NAMES, _DALER_FEATURE_NAMES,
    _icer_quadratic_interaction_features,
)

PROFILE_FEATURE_NAMES = list(_DACER_FEATURE_NAMES)
SUPPORT_FEATURE_NAMES = list(_DALER_FEATURE_NAMES)
SCALAR_FEATURE_COUNT = len(SUPPORT_FEATURE_NAMES)


def _hash_holdout(token: str, seed: str, fraction: float) -> bool:
    h = hashlib.sha256((seed + "::" + token).encode()).digest()
    return int.from_bytes(h[:8], "big") / float(2**64) < fraction


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    good = np.isfinite(score)
    y, score = y[good], score[good]
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
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
    X: list[list[float]] = []
    tm: list[float] = []
    tokens: list[str] = []
    challenger: list[int] = []
    anchor: list[int] = []
    legacy: list[int] = []
    admissible: list[bool] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        vals: list[float] = []
        ok = True
        for name in PROFILE_FEATURE_NAMES:
            try:
                v = float(r.get(f"dacer_feature_{name}", np.nan))
            except (TypeError, ValueError):
                ok = False
                break
            if not np.isfinite(v):
                ok = False
                break
            vals.append(v)
        if not ok:
            continue
        try:
            teacher_margin = float(r.get("teacher_margin", np.nan))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(teacher_margin):
            continue
        X.append(vals)
        tm.append(teacher_margin)
        tokens.append(str(r.get("scenario_token", len(tokens))))
        challenger.append(int(r.get("challenger_action", -1)))
        anchor.append(int(r.get("anchor_action", -1)))
        legacy.append(int(r.get("raw_top_action", -1)))
        admissible.append(float(r.get("dacer_admissible", 0.0)) >= 0.5)
    if not X:
        raise SystemExit(f"no valid V64.3.18 DACER instrumentation rows in {path}")
    tm_a = np.asarray(tm, dtype=np.float64)
    return {
        "X": np.asarray(X, dtype=np.float64),
        "teacher_margin": tm_a,
        "teacher_better": (tm_a > 0.0).astype(np.float64),
        "tokens": tokens,
        "challenger": np.asarray(challenger, dtype=np.int64),
        "anchor": np.asarray(anchor, dtype=np.int64),
        "legacy": np.asarray(legacy, dtype=np.int64),
        "admissible": np.asarray(admissible, dtype=bool),
    }


def _groups(tokens: list[str]) -> dict[str, np.ndarray]:
    d: dict[str, list[int]] = {}
    for i, t in enumerate(tokens):
        d.setdefault(t, []).append(i)
    return {t: np.asarray(v, dtype=np.int64) for t, v in d.items()}


def _standardize(X: np.ndarray, fit_idx: np.ndarray, active_count: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X[fit_idx].mean(axis=0)
    std = np.maximum(X[fit_idx].std(axis=0), 1.0e-6)
    Z = (X - mean[None, :]) / std[None, :]
    if active_count is not None and active_count < Z.shape[1]:
        Z[:, active_count:] = 0.0
    return Z, mean, std


def _fit_support(
    data: dict[str, Any], fit_mask: np.ndarray, *, steps: int, lr: float, l2: float, seed: int
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    X = data["X"][:, :SCALAR_FEATURE_COUNT]
    y = data["teacher_better"]
    tm = data["teacher_margin"]
    idx_fit = np.flatnonzero(np.asarray(fit_mask, dtype=bool))
    if idx_fit.size == 0:
        raise ValueError("ICER support fit received zero admissible edges")
    Z, mean, std = _standardize(X, idx_fit)
    groups = _groups(data["tokens"])
    scene_groups: list[np.ndarray] = []
    targets: list[int] = []
    for idx in groups.values():
        cand = idx[fit_mask[idx]]
        if cand.size == 0:
            continue
        scene_groups.append(cand)
        best = int(cand[np.argmax(tm[cand])])
        targets.append(best if float(tm[best]) > 0.0 else -1)

    global_to_group: dict[int, int] = {}
    for gi, cand in enumerate(scene_groups):
        for j in cand.tolist():
            global_to_group[int(j)] = gi
    keep_fit = np.asarray([i for i in idx_fit.tolist() if int(i) in global_to_group], dtype=np.int64)
    if keep_fit.size == 0:
        raise ValueError("ICER support fit has no grouped admissible edges")

    torch.manual_seed(seed)
    z = torch.tensor(Z, dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(Z.shape[1], dtype=torch.float32, requires_grad=True)
    b = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    fit_t = torch.tensor(keep_fit, dtype=torch.long)
    gid = torch.tensor([global_to_group[int(i)] for i in keep_fit.tolist()], dtype=torch.long)
    target_t = torch.tensor(targets, dtype=torch.long)
    target_group = torch.nonzero(target_t >= 0, as_tuple=False).reshape(-1)
    target_global = target_t[target_group] if target_group.numel() else torch.empty((0,), dtype=torch.long)
    n_groups = len(scene_groups)
    npos = max(float((y[keep_fit] > 0.5).sum()), 1.0)
    nneg = max(float((y[keep_fit] <= 0.5).sum()), 1.0)
    cw = torch.where(
        yy[fit_t] > 0.5,
        torch.tensor(len(keep_fit) / (2.0 * npos)),
        torch.tensor(len(keep_fit) / (2.0 * nneg)),
    )
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        logits = z @ w + b
        fl = logits[fit_t]
        max_g = torch.zeros(n_groups, dtype=logits.dtype).scatter_reduce(0, gid, fl, reduce="amax", include_self=True)
        edge_sum = torch.zeros(n_groups, dtype=logits.dtype).scatter_add(0, gid, torch.exp(fl - max_g[gid]))
        lse = max_g + torch.log(torch.clamp(torch.exp(-max_g) + edge_sum, min=1.0e-12))
        target_logits = torch.zeros(n_groups, dtype=logits.dtype)
        if target_group.numel():
            target_logits[target_group] = logits[target_global]
        listwise = (lse - target_logits).mean()
        support = torch.nn.functional.binary_cross_entropy_with_logits(fl, yy[fit_t], reduction="none")
        support = (support * cw).mean()
        loss = listwise + support + float(l2) * w.square().mean()
        loss.backward()
        opt.step()
    return w.detach().numpy().astype(float), float(b.detach()), mean.astype(float), std.astype(float)


def _dominance_training_rows(data: dict[str, Any], scene_mask: np.ndarray, admissible_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[int], int]:
    """Return alternative-edge labels for direct incumbent-contrastive recovery.

    Positive means the admissible alternative has larger TRAIN-only teacher margin
    than both anchor (0) and the frozen raw-EAF incumbent.  The scene listwise
    target is the best such alternative; otherwise the fixed incumbent pseudo-item
    (logit 0) is the target.
    """
    tm = data["teacher_margin"]
    ch = data["challenger"]
    legacy = data["legacy"]
    groups = _groups(data["tokens"])
    alt_idx: list[int] = []
    alt_y: list[float] = []
    scene_alts: list[np.ndarray] = []
    scene_targets: list[int] = []
    opportunity_scenes = 0
    for idx in groups.values():
        if not bool(scene_mask[idx[0]]):
            continue
        lg = int(legacy[idx[0]])
        li = idx[ch[idx] == lg]
        if li.size == 0:
            continue
        li0 = int(li[0])
        # Dominance is only a deployment question when the frozen raw incumbent
        # itself survives the unchanged final-guard prerequisites.  Otherwise
        # the deployment incumbent is the anchor and the support head owns the
        # decision; training a dominance label against an inadmissible raw action
        # would recreate the V64.3.18 metric/operator mismatch.
        if not bool(admissible_mask[li0]):
            continue
        ltm = float(tm[li0])
        cand = idx[admissible_mask[idx] & (ch[idx] != lg)]
        if cand.size == 0:
            continue
        threshold = max(0.0, ltm)
        labels = (tm[cand] > threshold).astype(np.float64)
        alt_idx.extend(cand.tolist())
        alt_y.extend(labels.tolist())
        scene_alts.append(cand)
        pos = cand[labels > 0.5]
        if pos.size:
            opportunity_scenes += 1
            scene_targets.append(int(pos[np.argmax(tm[pos])]))
        else:
            scene_targets.append(-1)
    return (
        np.asarray(alt_idx, dtype=np.int64),
        np.asarray(alt_y, dtype=np.float64),
        scene_alts,
        scene_targets,
        opportunity_scenes,
    )


def _fit_dominance(
    data: dict[str, Any], scene_mask: np.ndarray, admissible_mask: np.ndarray, *, feature_mode: str, max_iter: int, l2: float, seed: int
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, list[str], list[str], dict[str, int]]:
    alt_idx, alt_y, scene_alts, scene_targets, opportunities = _dominance_training_rows(data, scene_mask, admissible_mask)
    if alt_idx.size == 0:
        raise ValueError("ICER dominance fit has zero incumbent alternatives")
    dom_x, expanded_names, base_names = _icer_quadratic_interaction_features(data["X"], PROFILE_FEATURE_NAMES, feature_mode)
    Z, mean, std = _standardize(dom_x, alt_idx)
    z = torch.tensor(Z[alt_idx], dtype=torch.float32)
    y = torch.tensor(alt_y, dtype=torch.float32)
    torch.manual_seed(seed)
    w = torch.zeros(Z.shape[1], dtype=torch.float32, requires_grad=True)
    b = torch.zeros((), dtype=torch.float32, requires_grad=True)
    # Full-batch LBFGS is deterministic and fast for this convex direct
    # counterfactual log-odds problem.  Unlike V64.3.18's shared objective, no
    # listwise pseudo-item is allowed to translate the absolute zero boundary.
    opt = torch.optim.LBFGS(
        [w, b], lr=1.0, max_iter=int(max_iter), line_search_fn="strong_wolfe",
        tolerance_grad=1.0e-7, tolerance_change=1.0e-9,
    )

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        logits = z @ w + b
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss = bce + float(l2) * w.square().mean()
        loss.backward()
        return loss

    opt.step(closure)
    stats = {
        "alternative_edges": int(alt_idx.size),
        "positive_alternative_edges": int((alt_y > 0.5).sum()),
        "scene_groups": int(len(scene_alts)),
        "opportunity_scenes": int(opportunities),
        "expanded_feature_count": int(len(expanded_names)),
        "base_feature_count": int(len(base_names)),
    }
    return (
        w.detach().numpy().astype(float), float(b.detach()), mean.astype(float), std.astype(float),
        expanded_names, base_names, stats,
    )


def _logits(X: np.ndarray, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip(((X - mean[None, :]) / np.maximum(std[None, :], 1.0e-6)) @ w + b, -40.0, 40.0)


def _icer_selection_diag(
    data: dict[str, Any], support_logits: np.ndarray, dominance_logits: np.ndarray, scene_mask: np.ndarray
) -> dict[str, float]:
    tm, ch, legacy, anchor, adm = (
        data["teacher_margin"], data["challenger"], data["legacy"], data["anchor"], data["admissible"]
    )
    groups = _groups(data["tokens"])
    proposal = alternatives = alt_good = cf_good = raw_cf_good = 0
    cf_opp = cf_capture = selected_nonanchor = selected_nonanchor_good = 0
    direct_incumbent_proposals = direct_incumbent_replacements = direct_incumbent_good = 0
    direct_incumbent_opp = direct_incumbent_capture = anchor_recoveries = anchor_recovery_good = 0
    fallback = changed = top1_correct = multi = 0
    selected_tm: list[float] = []; alt_tm: list[float] = []; cf_delta: list[float] = []
    direct_y: list[int] = []; direct_s: list[float] = []
    for idx in groups.values():
        if not bool(scene_mask[idx[0]]):
            continue
        a, lg = int(anchor[idx[0]]), int(legacy[idx[0]])
        if lg == a:
            continue
        proposal += 1
        cand = idx[adm[idx]]
        multi += int(cand.size >= 2)
        li = idx[ch[idx] == lg]
        legacy_row = int(li[0]) if li.size else -1
        ltm = float(tm[legacy_row]) if legacy_row >= 0 else 0.0
        legacy_admissible = bool(legacy_row >= 0 and adm[legacy_row])
        deployment_threshold = max(0.0, ltm) if legacy_admissible else 0.0
        raw_threshold = max(0.0, ltm)
        if legacy_admissible:
            for j in cand[ch[cand] != lg].tolist():
                direct_y.append(int(float(tm[j]) > deployment_threshold))
                direct_s.append(float(dominance_logits[j]))

        support_ok = adm & np.isfinite(support_logits) & (support_logits > 0.0)
        sel_action, sel_row = a, -1
        if legacy_admissible:
            direct_incumbent_proposals += 1
            legacy_supported = bool(support_ok[legacy_row])
            sel_action, sel_row = (lg, legacy_row) if legacy_supported else (a, -1)
            alt = cand[(ch[cand] != lg) & (support_logits[cand] > 0.0) & np.isfinite(dominance_logits[cand]) & (dominance_logits[cand] > 0.0)]
            if alt.size:
                best = int(alt[np.argmax(dominance_logits[alt])])
                sel_action, sel_row = int(ch[best]), best
        else:
            sup = cand[(support_logits[cand] > 0.0) & np.isfinite(support_logits[cand])]
            if sup.size:
                best = int(sup[np.argmax(support_logits[sup])])
                sel_action, sel_row = int(ch[best]), best

        fallback += int(sel_action == a); changed += int(sel_action != lg)
        if sel_action != a and sel_row >= 0:
            selected_nonanchor += 1; stm = float(tm[sel_row]); selected_nonanchor_good += int(stm > 0.0); selected_tm.append(stm)
        else:
            selected_tm.append(0.0)
        is_alt = bool(sel_action not in {a, lg}); alternatives += int(is_alt)
        if is_alt and sel_row >= 0:
            stm = float(tm[sel_row]); alt_tm.append(stm); alt_good += int(stm > 0.0)
            good = bool(stm > deployment_threshold); cf_good += int(good); raw_cf_good += int(stm > raw_threshold)
            cf_delta.append(stm - deployment_threshold)
            if legacy_admissible:
                direct_incumbent_replacements += 1
                direct_incumbent_good += int(good)
            else:
                anchor_recoveries += 1
                anchor_recovery_good += int(stm > 0.0)
        opportunity = any(
            int(ch[j]) not in {a, lg} and float(tm[j]) > deployment_threshold for j in cand.tolist()
        )
        cf_opp += int(opportunity)
        cf_capture += int(opportunity and is_alt and sel_row >= 0 and float(tm[sel_row]) > deployment_threshold)
        if legacy_admissible:
            direct_incumbent_opp += int(opportunity)
            direct_incumbent_capture += int(opportunity and is_alt and sel_row >= 0 and float(tm[sel_row]) > deployment_threshold)
        if cand.size:
            best_teacher = int(cand[np.argmax(tm[cand])]); target = int(ch[best_teacher]) if float(tm[best_teacher]) > 0.0 else a
        else:
            target = a
        top1_correct += int(sel_action == target)
    def div(x: float, y: float) -> float:
        return float(x / y) if y else float("nan")
    return {
        "proposal_scene_count": float(proposal), "multi_admissible_proposal_rate": div(multi, proposal),
        "alternative_recovery_rate": div(alternatives, proposal), "alternative_recovery_precision": div(alt_good, alternatives),
        "alternative_teacher_margin_mean": float(np.mean(alt_tm)) if alt_tm else float("nan"),
        "counterfactual_recovery_precision": div(cf_good, alternatives),
        "strict_raw_top_counterfactual_recovery_precision": div(raw_cf_good, alternatives),
        "counterfactual_opportunity_rate": div(cf_opp, proposal), "counterfactual_opportunity_capture_rate": div(cf_capture, cf_opp),
        "direct_incumbent_proposal_count": float(direct_incumbent_proposals),
        "direct_incumbent_replacement_rate": div(direct_incumbent_replacements, direct_incumbent_proposals),
        "direct_incumbent_replacement_precision": div(direct_incumbent_good, direct_incumbent_replacements),
        "direct_incumbent_opportunity_rate": div(direct_incumbent_opp, direct_incumbent_proposals),
        "direct_incumbent_opportunity_capture_rate": div(direct_incumbent_capture, direct_incumbent_opp),
        "anchor_recovery_rate_on_proposals": div(anchor_recoveries, proposal),
        "anchor_recovery_precision": div(anchor_recovery_good, anchor_recoveries),
        "counterfactual_delta_mean": float(np.mean(cf_delta)) if cf_delta else float("nan"),
        "selected_nonanchor_teacher_better_rate": div(selected_nonanchor_good, selected_nonanchor),
        "anchor_fallback_rate": div(fallback, proposal), "proposal_changed_rate": div(changed, proposal),
        "selected_teacher_margin_mean_including_anchor": float(np.mean(selected_tm)) if selected_tm else float("nan"),
        "anchor_augmented_top1_accuracy": div(top1_correct, proposal),
        "direct_counterfactual_dominance_auc": _auc(np.asarray(direct_y), np.asarray(direct_s)) if direct_y else float("nan"),
        "direct_counterfactual_dominance_edges": float(len(direct_y)),
    }



def _load_frozen_support_config(path: Path) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    frontier = ((cfg.get("runtime", {}) or {}).get("decisive_frontier_value", {}) or {})
    dacer = frontier.get("deployment_admissible_counterfactual_extremal_recovery", {}) or {}
    names = list(dacer.get("feature_names", []))
    mean = np.asarray(dacer.get("feature_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(dacer.get("feature_std", []), dtype=np.float64).reshape(-1)
    weights = np.asarray(dacer.get("weights", []), dtype=np.float64).reshape(-1)
    if names != PROFILE_FEATURE_NAMES or len(mean) != len(names) or len(std) != len(names) or len(weights) != len(names):
        raise SystemExit("frozen G-DALER support config does not expose the expected V64.3.18 DACER feature schema")
    if str(dacer.get("feature_mode", "")) != "scalar" or str(dacer.get("objective_mode", "")) != "listwise":
        raise SystemExit("ICER frozen support must be the V64.3.18 scalar G-DALER listwise+support control")
    if np.max(np.abs(weights[SCALAR_FEATURE_COUNT:])) > 1.0e-10:
        raise SystemExit("frozen G-DALER support config unexpectedly uses profile weights")
    return weights[:SCALAR_FEATURE_COUNT].astype(float), float(dacer.get("bias", 0.0)), mean[:SCALAR_FEATURE_COUNT].astype(float), std[:SCALAR_FEATURE_COUNT].astype(float)

def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.19 EAF-ICER train-only decomposed support and incumbent-contrastive heads.")
    ap.add_argument("--train-frontier-edges", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-config", required=True)
    ap.add_argument("--output-report", required=True)
    ap.add_argument("--frozen-support-config", default=None, help="V64.3.18 train-only scalar G-DALER fitted config; reused exactly for causal isolation and speed.")
    ap.add_argument("--dominance-policy", choices=["scalar_only", "dual_equal_mean"], default="dual_equal_mean")
    ap.add_argument("--companion-scalar-config", default=None, help="Optional scalar-only ablation config emitted from the same fitted heads; avoids a duplicate fit.")
    ap.add_argument("--companion-scalar-report", default=None, help="Report paired with --companion-scalar-config.")
    ap.add_argument("--holdout-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", default="v64.3.19-eaf-icer-v1")
    ap.add_argument("--support-steps", type=int, default=1200)
    ap.add_argument("--support-lr", type=float, default=0.03)
    ap.add_argument("--dominance-max-iter", type=int, default=80)
    ap.add_argument("--l2", type=float, default=1.0e-3)
    ap.add_argument("--support-loss-weight", type=float, default=1.0)
    ap.add_argument("--dominance-bce-weight", type=float, default=1.0)
    ap.add_argument("--min-admissible-edges", type=int, default=8192)
    ap.add_argument("--min-multi-scenes", type=int, default=512)
    args = ap.parse_args()
    if any(abs(v - 1.0) > 1.0e-12 for v in (args.support_loss_weight, args.dominance_bce_weight)):
        raise SystemExit("V64.3.19 fixes all declared ICER objective weights at 1.0; no validation loss-weight sweep is permitted")
    if bool(args.companion_scalar_config) != bool(args.companion_scalar_report):
        raise SystemExit("--companion-scalar-config and --companion-scalar-report must be provided together")
    if args.companion_scalar_config and args.dominance_policy != "dual_equal_mean":
        raise SystemExit("companion scalar artifacts are only emitted from the dual_equal_mean fit")

    data = _load(Path(args.train_frontier_edges))
    adm = data["admissible"]
    groups = _groups(data["tokens"])
    multi = sum(int(np.sum(adm[idx]) >= 2) for idx in groups.values())
    if int(adm.sum()) < args.min_admissible_edges or multi < args.min_multi_scenes:
        raise SystemExit(f"insufficient guard-admissible support for ICER: edges={int(adm.sum())}, multi_scenes={multi}")

    hold_scene = np.asarray([_hash_holdout(t, args.split_seed, args.holdout_fraction) for t in data["tokens"]], dtype=bool)
    train_adm = adm & ~hold_scene
    hold_adm = adm & hold_scene
    if int(train_adm.sum()) < 4096 or int(hold_adm.sum()) < 1024:
        raise SystemExit("ICER deterministic scene-group holdout too small")

    if args.frozen_support_config:
        sw0, sb0, sm0, ss0 = _load_frozen_support_config(Path(args.frozen_support_config))
        support_source = "reused_v64_3_18_train_only_scalar_gdaler"
    else:
        sw0, sb0, sm0, ss0 = _fit_support(data, train_adm, steps=args.support_steps, lr=args.support_lr, l2=args.l2, seed=1919)
        support_source = "refit_train_only_support_fallback"
    sdw0, sdb0, sdm0, sds0, sdom_names0, sdom_base0, sdom_train_stats = _fit_dominance(
        data, ~hold_scene, adm, feature_mode="scalar_interaction", max_iter=args.dominance_max_iter, l2=args.l2, seed=1920
    )
    pdw0, pdb0, pdm0, pds0, pdom_names0, pdom_base0, pdom_train_stats = _fit_dominance(
        data, ~hold_scene, adm, feature_mode="profile_interaction", max_iter=args.dominance_max_iter, l2=args.l2, seed=1921
    )
    support0 = _logits(data["X"][:, :SCALAR_FEATURE_COUNT], sw0, sb0, sm0, ss0)
    sx0, _, _ = _icer_quadratic_interaction_features(data["X"], PROFILE_FEATURE_NAMES, "scalar_interaction")
    px0, _, _ = _icer_quadratic_interaction_features(data["X"], PROFILE_FEATURE_NAMES, "profile_interaction")
    scalar_dom0 = _logits(sx0, sdw0, sdb0, sdm0, sds0)
    profile_dom0 = _logits(px0, pdw0, pdb0, pdm0, pds0)
    dominance0 = scalar_dom0 if args.dominance_policy == "scalar_only" else 0.5 * (scalar_dom0 + profile_dom0)
    hold_support_auc = _auc(data["teacher_better"][hold_adm], support0[hold_adm])
    hold_diag = _icer_selection_diag(data, support0, dominance0, hold_scene)
    hold_scalar_diag = _icer_selection_diag(data, support0, scalar_dom0, hold_scene)
    hold_profile_diag = _icer_selection_diag(data, support0, profile_dom0, hold_scene)

    if args.frozen_support_config:
        sw, sb, sm, ss = sw0, sb0, sm0, ss0
    else:
        sw, sb, sm, ss = _fit_support(data, adm, steps=args.support_steps, lr=args.support_lr, l2=args.l2, seed=1919)
    all_scene = np.ones(len(data["tokens"]), dtype=bool)
    sdw, sdb, sdm, sds, sdom_names, sdom_base, sdom_full_stats = _fit_dominance(
        data, all_scene, adm, feature_mode="scalar_interaction", max_iter=args.dominance_max_iter, l2=args.l2, seed=1920
    )
    pdw, pdb, pdm, pds, pdom_names, pdom_base, pdom_full_stats = _fit_dominance(
        data, all_scene, adm, feature_mode="profile_interaction", max_iter=args.dominance_max_iter, l2=args.l2, seed=1921
    )

    cfg = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    frontier = cfg.setdefault("runtime", {}).setdefault("decisive_frontier_value", {})
    frontier.setdefault("one_sided_intervention", {})["enabled"] = False
    frontier.setdefault("learned_intervention_reliability", {})["enabled"] = False
    frontier.setdefault("reliability_aware_extremal_reranking", {})["enabled"] = False
    frontier.setdefault("deployment_aligned_listwise_extremal_reliability", {})["enabled"] = False
    frontier.setdefault("deployment_admissible_counterfactual_extremal_recovery", {})["enabled"] = False
    icer = frontier.setdefault("incumbent_contrastive_extremal_recovery", {})
    icer.update({
        "enabled": True,
        "instrument_features": True,
        "model_type": "decomposed_anchor_support_plus_dual_view_quadratic_incumbent_contrastive_reliability",
        "dominance_policy": args.dominance_policy,
        "support_feature_names": SUPPORT_FEATURE_NAMES,
        "support_feature_mean": [float(x) for x in sm],
        "support_feature_std": [float(x) for x in ss],
        "support_weights": [float(x) for x in sw],
        "support_bias": float(sb),
        "scalar_dominance_base_feature_names": sdom_base,
        "scalar_dominance_feature_names": sdom_names,
        "scalar_dominance_feature_mean": [float(x) for x in sdm],
        "scalar_dominance_feature_std": [float(x) for x in sds],
        "scalar_dominance_weights": [float(x) for x in sdw],
        "scalar_dominance_bias": float(sdb),
        "profile_dominance_base_feature_names": pdom_base,
        "profile_dominance_feature_names": pdom_names,
        "profile_dominance_feature_mean": [float(x) for x in pdm],
        "profile_dominance_feature_std": [float(x) for x in pds],
        "profile_dominance_weights": [float(x) for x in pdw],
        "profile_dominance_bias": float(pdb),
        "anchor_logit": 0.0,
        "incumbent_logit": 0.0,
        "require_guard_admissible": True,
        "require_safe_available_for_learned_intervention": True,
        "training_support_target": "teacher_challenger_better_than_darm_anchor",
        "training_support_objective": "anchor_augmented_listwise_ce_plus_class_balanced_support_bce",
        "training_dominance_target": "teacher_alternative_better_than_max_anchor_and_frozen_incumbent",
        "training_dominance_objective": "unweighted_direct_counterfactual_bce_over_fixed_quadratic_evidence_interaction_map",
        "objective_weights": {"support": 1.0, "dominance_bce": 1.0},
        "selection_operator": "deployment_incumbent_conditioned: support_positive alternative must have positive pre_registered dominance aggregate; dual mode uses fixed equal mean of scalar and signed-profile counterfactual log-odds",
        "threshold_policy": "fixed_zero_direct_counterfactual_log_odds_no_validation_threshold_sweep",
        "utility_equivalence_role": "diagnostic_exact_tiebreak_only_not_hard_mask_not_learned_feature",
    })
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.19-EAF-ICER-DARM-DBR"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.19-EAF-ICER-DARM-DBR"
    exp = cfg.setdefault("experiment", {})
    exp["name"] = f"v64_3_19_eaf_icer_{args.dominance_policy}_fitted"
    exp["algorithm"] = "V64.3.19 EAF-ICER: Evidence-Attributed Incumbent-Contrastive Extremal Recovery"
    exp["mechanism_chain"] = "fixed B<=16 -> frozen EAF complete frontier + exact selected-evidence attribution -> final-guard-admissible frontier -> decomposed anchor support + direct incumbent-contrastive dominance -> alternative recovery/abstention -> unchanged final guard"
    out = Path(args.output_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    report = {
        "audit": "v64_3_19_eaf_icer_fit",
        "dominance_policy": args.dominance_policy,
        "train_all_frontier_edges": int(len(adm)),
        "train_admissible_edges": int(adm.sum()),
        "train_multi_admissible_scenes": int(multi),
        "internal_holdout_admissible_edges": int(hold_adm.sum()),
        "support_replay_auc_on_dominance_holdout_partition": float(hold_support_auc),
        "support_holdout_independent": bool(not args.frozen_support_config),
        "support_source": support_source,
        "frozen_support_config": str(args.frozen_support_config) if args.frozen_support_config else None,
        **{f"holdout_{k}": v for k, v in hold_diag.items()},
        "scalar_dominance_train_stats": sdom_train_stats,
        "profile_dominance_train_stats": pdom_train_stats,
        "scalar_dominance_full_stats": sdom_full_stats,
        "profile_dominance_full_stats": pdom_full_stats,
        "holdout_scalar_direct_counterfactual_dominance_auc": hold_scalar_diag.get("direct_counterfactual_dominance_auc"),
        "holdout_profile_direct_counterfactual_dominance_auc": hold_profile_diag.get("direct_counterfactual_dominance_auc"),
        "holdout_scalar_counterfactual_recovery_precision": hold_scalar_diag.get("counterfactual_recovery_precision"),
        "holdout_profile_counterfactual_recovery_precision": hold_profile_diag.get("counterfactual_recovery_precision"),
        "support_feature_names": SUPPORT_FEATURE_NAMES,
        "scalar_dominance_base_feature_names": sdom_base,
        "scalar_dominance_feature_names": sdom_names,
        "profile_dominance_base_feature_names": pdom_base,
        "profile_dominance_feature_names": pdom_names,
        "support_weights": [float(x) for x in sw],
        "support_bias": float(sb),
        "scalar_dominance_weights": [float(x) for x in sdw],
        "scalar_dominance_bias": float(sdb),
        "profile_dominance_weights": [float(x) for x in pdw],
        "profile_dominance_bias": float(pdb),
        "anchor_logit": 0.0,
        "incumbent_logit": 0.0,
        "objective_weights": {"support": 1.0, "dominance_bce": 1.0},
        "fit_uses_nuplan_validation": False,
        "fit_uses_test": False,
        "interpretation": "Train-only causal fit. Anchor support and incumbent dominance use independent heads over the same frozen final-guard-admissible frontier. The dominance head is trained as direct unweighted counterfactual log-odds over a fixed second-order interaction map; The dual_equal_mean policy symmetrically aggregates scalar and exact signed-profile counterfactual log-odds; scalar_only is the pre-registered attribution-structure ablation. No validation threshold or loss-weight tuning is allowed.",
    }
    rp = Path(args.output_report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    # Speed/causal-isolation path: scalar and dual use the exact same fitted
    # support/scalar/profile heads.  Emitting the scalar ablation from this one
    # train-only fit avoids fitting both dominance heads a second time.
    if args.companion_scalar_config:
        scalar_cfg = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False))
        scalar_icer = scalar_cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
        scalar_icer["dominance_policy"] = "scalar_only"
        scalar_cfg.setdefault("experiment", {})["name"] = "v64_3_19_eaf_icer_scalar_only_fitted"
        scp = Path(args.companion_scalar_config); scp.parent.mkdir(parents=True, exist_ok=True)
        scp.write_text(yaml.safe_dump(scalar_cfg, sort_keys=False), encoding="utf-8")

        scalar_report = dict(report)
        scalar_report["dominance_policy"] = "scalar_only"
        scalar_report.update({f"holdout_{k}": v for k, v in hold_scalar_diag.items()})
        scalar_report["companion_emitted_from_dual_fit"] = True
        scalar_report["shared_fitted_head_identity_with_dual"] = True
        srp = Path(args.companion_scalar_report); srp.parent.mkdir(parents=True, exist_ok=True)
        srp.write_text(json.dumps(scalar_report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
