from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _finite_cost(raw: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    cost = raw.float()
    valid = valid.bool()
    finite = torch.isfinite(cost) & valid
    large = torch.full_like(cost, 1e6)
    cost = torch.where(finite, cost, large)
    row_min = cost.min(dim=1, keepdim=True).values
    centered = cost - row_min
    centered = torch.where(finite, centered, large)
    # Quantile scaling keeps the candidate cost target well-conditioned while
    # preserving all within-scene rankings.  torch.quantile is relatively slow
    # on CUDA for these short per-scene vectors; an explicit sort + linear
    # interpolation is mathematically equivalent to q=0.75 with the default
    # interpolation mode and avoids that kernel bottleneck.
    values = centered.masked_fill(~finite, 0.0).abs()
    ordered = torch.sort(values, dim=1).values
    n = ordered.shape[1]
    if n <= 1:
        scale = ordered[:, :1]
    else:
        pos = 0.75 * float(n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - float(lo)
        scale = ordered[:, lo : lo + 1] * (1.0 - frac) + ordered[:, hi : hi + 1] * frac
    scale = scale.clamp_min(1.0)
    return centered / scale


def _planner_targets(batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, str]:
    ecfg = cfg.get("external_baseline", {}) or {}
    mode = str(ecfg.get("planner_supervision", "teacher_cost")).strip().lower()
    valid = batch["candidate_valid"].bool()
    if mode == "expert_imitation":
        if "expert_candidate_index" not in batch or "expert_candidate_cost" not in batch:
            raise KeyError(
                "planner_supervision=expert_imitation requires expert_candidate_index/expert_candidate_cost; "
                "train with caches containing label_logged_ego"
            )
        target = batch["expert_candidate_index"].long()
        cost = batch["expert_candidate_cost"].float()
    elif mode in {"teacher", "teacher_cost", "bdse_teacher"}:
        target = batch["teacher_a_star"].long()
        cost = batch["teacher_J_T"].float()
        mode = "teacher_cost"
    else:
        raise ValueError(f"unknown external_baseline.planner_supervision={mode!r}")
    target = target.clamp_min(0).clamp_max(valid.shape[1] - 1)
    return target, _finite_cost(cost, valid), mode


def compute_external_baseline_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    ecfg = cfg.get("external_baseline", {}) or {}
    weights = ecfg.get("loss_weights", {}) or {}
    J = out["J0"].float()
    valid = batch["candidate_valid"].bool()
    target, target_cost_norm, supervision_mode = _planner_targets(batch, cfg)
    logits = (-J).masked_fill(~valid, -1e9)
    ce = F.cross_entropy(logits, target)

    valid_f = valid.float()
    reg = F.smooth_l1_loss(
        torch.where(valid, J, torch.zeros_like(J)),
        torch.where(valid, target_cost_norm, torch.zeros_like(target_cost_norm)),
        reduction="none",
    )
    reg = (reg * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    # Candidate-pair ranking is optional.  Under expert_imitation the *indices*
    # can still come from the shared cache, but direction is recomputed from the
    # logged-expert ADE rather than from BDSE teacher margins.  Thus setting a
    # nonzero pair weight does not leak teacher preference labels.
    pair_idx = batch.get("pair_indices")
    pair_valid = batch.get("pair_valid")
    if pair_idx is not None and pair_valid is not None and pair_idx.numel() > 0:
        a = pair_idx[..., 0].long().clamp(0, J.shape[1] - 1)
        b = pair_idx[..., 1].long().clamp(0, J.shape[1] - 1)
        Ja = torch.gather(J, 1, a)
        Jb = torch.gather(J, 1, b)
        pv = pair_valid.bool()
        if supervision_mode == "expert_imitation":
            ca = torch.gather(target_cost_norm, 1, a)
            cb = torch.gather(target_cost_norm, 1, b)
            a_wins = ca <= cb
            Jwinner = torch.where(a_wins, Ja, Jb)
            Jloser = torch.where(a_wins, Jb, Ja)
            pv = pv & torch.isfinite(ca) & torch.isfinite(cb)
        else:
            # Cached teacher pairs follow winner->rival direction.
            Jwinner, Jloser = Ja, Jb
        margin = float(ecfg.get("rank_margin", 1.0))
        pr = F.relu(margin + Jwinner - Jloser)
        pair_loss = (pr * pv.float()).sum() / pv.float().sum().clamp_min(1.0)
    else:
        pair_loss = J.sum() * 0.0

    # This is the only intentionally BDSE-specific supervision retained for the
    # trainable baselines: it learns which evidence atoms to expose under the
    # common external fixed-budget interface.  It does not supervise planner cost.
    prop_loss = J.sum() * 0.0
    if "proposal_logits" in out and "oracle_selected_mask" in batch:
        prop_logits = out["proposal_logits"].float()
        active = batch.get("evidence_active", torch.ones_like(prop_logits, dtype=torch.bool)).bool()
        target_mask = batch["oracle_selected_mask"].float()
        bce = F.binary_cross_entropy_with_logits(prop_logits, target_mask, reduction="none")
        prop_loss = (bce * active.float()).sum() / active.float().sum().clamp_min(1.0)

    deep_action = J.sum() * 0.0
    aux = out.get("external_aux_costs")
    if isinstance(aux, torch.Tensor) and aux.ndim == 3 and aux.shape[1] > 1:
        aux_logits = (-aux.float()).masked_fill(~valid[:, None, :], -1e9)
        repeated_target = target[:, None].expand(-1, aux.shape[1]).reshape(-1)
        deep_action = F.cross_entropy(aux_logits.reshape(-1, aux.shape[-1]), repeated_target)

    total = (
        float(weights.get("action", 1.0)) * ce
        + float(weights.get("cost", 0.5)) * reg
        + float(weights.get("pair", 0.0)) * pair_loss
        + float(weights.get("proposal", 0.25)) * prop_loss
        + float(weights.get("deep_supervision", 0.0)) * deep_action
    )
    return {
        "loss": total,
        "action_ce": ce,
        "cost_reg": reg,
        "pair_rank": pair_loss,
        "proposal_bce": prop_loss,
        "deep_action_ce": deep_action,
    }
