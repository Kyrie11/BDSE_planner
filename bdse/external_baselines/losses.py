from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _finite_teacher_cost(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    jt = batch["teacher_J_T"].float()
    valid = batch["candidate_valid"].bool()
    finite = torch.isfinite(jt) & valid
    large = torch.full_like(jt, 1e6)
    jt = torch.where(finite, jt, large)
    row_min = jt.min(dim=1, keepdim=True).values
    centered = jt - row_min
    centered = torch.where(finite, centered, large)
    scale = torch.quantile(centered.masked_fill(~finite, 0.0).abs(), 0.75, dim=1, keepdim=True).clamp_min(1.0)
    return centered / scale


def compute_external_baseline_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    ecfg = cfg.get("external_baseline", {}) or {}
    weights = ecfg.get("loss_weights", {}) or {}
    J = out["J0"].float()
    valid = batch["candidate_valid"].bool()
    target = batch["teacher_a_star"].long().clamp_min(0).clamp_max(J.shape[1] - 1)
    logits = (-J).masked_fill(~valid, -1e9)
    ce = F.cross_entropy(logits, target)

    jt_norm = _finite_teacher_cost(batch)
    valid_f = valid.float()
    reg = F.smooth_l1_loss(torch.where(valid, J, torch.zeros_like(J)), torch.where(valid, jt_norm, torch.zeros_like(jt_norm)), reduction="none")
    reg = (reg * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    # Pairwise ranking: for supervised winner->rival pairs, teacher winner should
    # have lower predicted cost than the rival.  The cached pair direction follows
    # the BDSE positive-margin convention.
    pair_idx = batch.get("pair_indices")
    pair_valid = batch.get("pair_valid")
    if pair_idx is not None and pair_valid is not None and pair_idx.numel() > 0:
        a = pair_idx[..., 0].long().clamp(0, J.shape[1] - 1)
        b = pair_idx[..., 1].long().clamp(0, J.shape[1] - 1)
        Ja = torch.gather(J, 1, a)
        Jb = torch.gather(J, 1, b)
        pv = pair_valid.bool()
        margin = float(ecfg.get("rank_margin", 1.0))
        pr = F.relu(margin + Ja - Jb)
        pair_loss = (pr * pv.float()).sum() / pv.float().sum().clamp_min(1.0)
    else:
        pair_loss = J.sum() * 0.0

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
        + float(weights.get("pair", 0.5)) * pair_loss
        + float(weights.get("proposal", 0.25)) * prop_loss
        + float(weights.get("deep_supervision", 0.0)) * deep_action
    )
    return {"loss": total, "action_ce": ce, "cost_reg": reg, "pair_rank": pair_loss, "proposal_bce": prop_loss, "deep_action_ce": deep_action}
