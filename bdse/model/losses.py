from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def robust_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    return F.huber_loss(pred[mask], target[mask], delta=delta, reduction="mean")


def pair_gather(values: torch.Tensor, pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # values [B,K] or [B,E,K], pairs [B,P,2]
    B, P, _ = pairs.shape
    a = pairs[..., 0].long()
    b = pairs[..., 1].long()
    if values.dim() == 2:
        va = torch.gather(values, 1, a)
        vb = torch.gather(values, 1, b)
        return va, vb
    if values.dim() == 3:
        E = values.shape[1]
        idx_a = a[:, None, :].expand(B, E, P)
        idx_b = b[:, None, :].expand(B, E, P)
        va = torch.gather(values, 2, idx_a)
        vb = torch.gather(values, 2, idx_b)
        return va, vb
    raise ValueError("values must be [B,K] or [B,E,K]")


def full_interface_predicted_margin(J0: torch.Tensor, g: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    J_a, J_b = pair_gather(J0, pairs)
    g_a, g_b = pair_gather(g, pairs)
    return (J_b - J_a) + (g_b - g_a).sum(dim=1)


def selected_budget_margin(J0: torch.Tensor, g: torch.Tensor, pairs: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
    J_a, J_b = pair_gather(J0, pairs)
    g_a, g_b = pair_gather(g, pairs)
    support = ((g_b - g_a) * selected_mask[:, :, None].float()).sum(dim=1)
    return (J_b - J_a) + support


def compute_bdse_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    J0 = outputs["J0"]
    g = outputs["g"]
    valid = batch["candidate_valid"].bool()
    J_base_T = batch["teacher_J_base"].float()
    pairs = batch["pair_indices"].long()
    pair_valid = batch["pair_valid"].bool()
    pair_weights = batch["pair_weights"].float()
    residual_T = batch["pair_residuals"].float()
    target_action = batch["teacher_a_star"].long()
    lw = cfg.get("training", {}).get("loss_weights", {})

    finite_J0 = torch.where(torch.isfinite(J0), J0, torch.zeros_like(J0))
    L_base = robust_loss(finite_J0, J_base_T, valid)

    g_a, g_b = pair_gather(g, pairs)
    res_pred = (g_b - g_a).sum(dim=1)
    pair_mask = pair_valid & torch.isfinite(residual_T)
    if pair_mask.sum() > 0:
        L_res = F.huber_loss(res_pred[pair_mask], residual_T[pair_mask], delta=1.0, reduction="none")
        L_res = (L_res * pair_weights[pair_mask]).sum() / pair_weights[pair_mask].sum().clamp_min(1e-6)
    else:
        L_res = J0.new_tensor(0.0)

    M_hat_E = full_interface_predicted_margin(finite_J0, g, pairs)
    mu = float(cfg.get("training", {}).get("rank_margin", 1.0))
    if pair_mask.sum() > 0:
        L_rank = (pair_weights[pair_mask] * F.relu(mu - M_hat_E[pair_mask])).sum() / pair_weights[pair_mask].sum().clamp_min(1e-6)
    else:
        L_rank = J0.new_tensor(0.0)

    if "oracle_selected_mask" in batch:
        target_sel = batch["oracle_selected_mask"].float()
        L_sel = F.binary_cross_entropy_with_logits(outputs["selector_logits"], target_sel, reduction="none")
        e_mask = batch.get("evidence_active", torch.ones_like(target_sel).bool()).bool()
        L_sel = L_sel[e_mask].mean() if e_mask.sum() > 0 else J0.new_tensor(0.0)
    else:
        L_sel = J0.new_tensor(0.0)

    if "runtime_selected_mask" in batch and pairs.shape[1] > 0:
        # Use runtime-style selected evidence support in the action loss.
        selected_mask = batch["runtime_selected_mask"].bool()
        all_cost = finite_J0 + (g * selected_mask[:, :, None].float()).sum(dim=1)
    else:
        all_cost = finite_J0 + g.sum(dim=1)
    logits = -all_cost.masked_fill(~valid, 1e9)
    if "teacher_J_T" in batch and batch.get("use_regret_soft_target", False):
        p = torch.softmax(-batch["teacher_J_T"].float(), dim=1)
        L_act = -(p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    else:
        L_act = F.cross_entropy(logits, target_action)

    total = (
        float(lw.get("base", 1.0)) * L_base
        + float(lw.get("residual", 1.0)) * L_res
        + float(lw.get("rank", 1.0)) * L_rank
        + float(lw.get("selection", 0.5)) * L_sel
        + float(lw.get("action", 1.0)) * L_act
    )
    return {"loss": total, "L_base": L_base, "L_res": L_res, "L_rank": L_rank, "L_sel": L_sel, "L_act": L_act}
