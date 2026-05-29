from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from bdse.planner.pair_screen import build_runtime_pairs_from_base
from bdse.planner.selector import _greedy_cover_from_pair_support


def robust_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    return F.huber_loss(pred[mask], target[mask], delta=delta, reduction="mean")


def pair_gather(values: torch.Tensor, pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def _predicted_certificate_masks(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop-gradient predicted Top-M + base-screen sparse query masks for L_act.

    The dense offline ``g`` tensor is available during training for residual
    supervision, but the deployment action path may only use Top-M atoms and
    actions that appear in the runtime pair screen.  This helper returns both
    the selected atom mask and the atom-action query mask, preventing unqueried
    dense scores from leaking into L_act.
    """
    J0 = outputs["J0"].detach().cpu().numpy()
    g = outputs["g"].detach().cpu().numpy()
    logits = outputs["proposal_logits"].detach().cpu().numpy()
    valid = batch["candidate_valid"].detach().cpu().numpy().astype(bool)
    active = batch.get("evidence_active", torch.ones_like(outputs["proposal_logits"]).bool()).detach().cpu().numpy().astype(bool)
    costs = batch.get("evidence_budget_costs", torch.ones_like(outputs["proposal_logits"])).detach().cpu().numpy().astype(np.float32)
    B, E = logits.shape
    K = valid.shape[1]
    selected_mask = np.zeros((B, E), dtype=bool)
    query_mask = np.zeros((B, E, K), dtype=bool)
    budget = float(cfg.get("evidence", {}).get("budget", 16))
    M = int(cfg.get("selector", {}).get("proposal_top_m", max(int(2 * budget), int(budget) + 1)))
    L0 = int(cfg.get("tournament", {}).get("L_infer", 16))
    eta = float(cfg.get("selector", {}).get("eta_pred", 1.0))
    gamma = float(cfg.get("selector", {}).get("gamma_max_default", 100.0))
    for bidx in range(B):
        topm = np.argsort(-np.where(active[bidx], logits[bidx], -1e9))[: min(M, max(int(active[bidx].sum()), 1))]
        atom_active = np.zeros((E,), dtype=bool)
        atom_active[topm] = True
        atom_active &= active[bidx]
        flags = batch.get("runtime_safety_flags")
        flag_np = flags[bidx].detach().cpu().numpy().astype(bool) if flags is not None else np.zeros((K,), dtype=bool)
        pairs, weights = build_runtime_pairs_from_base(J0[bidx], valid[bidx], flag_np, L0=L0, eta0=eta)
        if len(pairs):
            action_ids = np.unique(pairs.reshape(-1))
            for ei in np.flatnonzero(atom_active):
                query_mask[bidx, ei, action_ids] = True
            a = pairs[:, 0]
            c = pairs[:, 1]
            base_delta = J0[bidx, c] - J0[bidx, a]
            base_support = np.maximum(base_delta, 0.0)
            g_sparse = np.where(query_mask[bidx], g[bidx], 0.0)
            atom_support = np.maximum(g_sparse[:, c] - g_sparse[:, a], 0.0)
            caps = np.minimum(np.maximum(np.abs(base_delta) + 0.25 * gamma, 1e-3), gamma).astype(np.float32)
        else:
            base_support = np.zeros((0,), dtype=np.float32)
            atom_support = np.zeros((E, 0), dtype=np.float32)
            caps = np.zeros((0,), dtype=np.float32)
            weights = np.zeros((0,), dtype=np.float32)
        selected, _, _ = _greedy_cover_from_pair_support(atom_support, base_support, caps, weights, costs[bidx], budget, atom_active)
        selected_mask[bidx, selected] = True
    device = outputs["J0"].device
    return torch.from_numpy(selected_mask).to(device), torch.from_numpy(query_mask).to(device)



def _oracle_certificate_masks(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor] | None:
    target_sel = batch.get("oracle_selected_mask")
    if target_sel is None:
        return None
    valid = batch["candidate_valid"].bool()
    e_mask = batch.get("evidence_active", torch.ones_like(outputs["proposal_logits"]).bool()).bool()
    selected = target_sel.bool() & e_mask
    # Teacher-forced certificates are an offline training stabilizer.  They may
    # query dense scores for oracle-selected atoms over valid actions, and are
    # annealed away so deployment remains sparse/predicted-only.
    query = selected[:, :, None] & valid[:, None, :]
    return selected, query


def _certificate_oracle_probability(cfg: dict[str, Any]) -> float:
    tc = cfg.get("training", {})
    sched = tc.get("certificate_schedule", {})
    if not bool(sched.get("enabled", False)):
        return 0.0
    epochs = max(int(tc.get("epochs", 1)), 1)
    epoch = int(tc.get("current_epoch", epochs))
    start = float(sched.get("oracle_start_prob", 1.0))
    end = float(sched.get("oracle_end_prob", 0.0))
    warmup = max(int(sched.get("warmup_epochs", 0)), 0)
    anneal = int(sched.get("anneal_epochs", max(epochs - warmup, 1)))
    if epoch < warmup:
        return float(np.clip(start, 0.0, 1.0))
    progress = min(max((epoch - warmup) / max(float(anneal), 1.0), 0.0), 1.0)
    return float(np.clip(start + (end - start) * progress, 0.0, 1.0))

def _softmin(vals: torch.Tensor, tau: float, dim: int) -> torch.Tensor:
    if tau <= 0:
        return vals.min(dim=dim).values
    return -float(tau) * torch.logsumexp(-vals / float(tau), dim=dim)


def _budgeted_tournament_scores(cost: torch.Tensor, valid: torch.Tensor, tau: float, epsilon_cal: float = 0.0) -> torch.Tensor:
    # M(a,b)=cost[b]-cost[a].  Higher score means larger lower margin against rivals.
    B, K = cost.shape
    margins = cost[:, None, :] - cost[:, :, None] - float(epsilon_cal)
    eye = torch.eye(K, dtype=torch.bool, device=cost.device)[None]
    rival_valid = valid[:, None, :].expand(B, K, K) & (~eye)
    vals = margins.masked_fill(~rival_valid, float("inf"))
    scores = _softmin(vals, tau, dim=2)
    no_rivals = ~rival_valid.any(dim=2)
    scores = scores.masked_fill(no_rivals, 0.0).masked_fill(~valid, -1e9)
    return scores


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
    tau_rank = float(cfg.get("training", {}).get("rank_tau", cfg.get("training", {}).get("rank_margin", 1.0)))
    if pair_mask.sum() > 0:
        rank_terms = F.softplus(-M_hat_E[pair_mask] / max(tau_rank, 1e-6))
        L_rank = (pair_weights[pair_mask] * rank_terms).sum() / pair_weights[pair_mask].sum().clamp_min(1e-6)
    else:
        L_rank = J0.new_tensor(0.0)

    target_sel = batch.get("oracle_selected_mask")
    gain = batch.get("proposal_target_gain")
    e_mask = batch.get("evidence_active", torch.ones_like(outputs["proposal_logits"]).bool()).bool()
    if target_sel is not None:
        bce = F.binary_cross_entropy_with_logits(outputs["proposal_logits"], target_sel.float(), reduction="none")
        L_prop_bce = bce[e_mask].mean() if e_mask.sum() > 0 else J0.new_tensor(0.0)
    else:
        L_prop_bce = J0.new_tensor(0.0)
    if gain is not None and e_mask.sum() > 0:
        target_dist = gain.float().masked_fill(~e_mask, 0.0)
        target_dist = target_dist / target_dist.sum(dim=1, keepdim=True).clamp_min(1e-6)
        logp = F.log_softmax(outputs["proposal_logits"].masked_fill(~e_mask, -1e9), dim=1)
        L_prop_rank = -(target_dist * logp).sum(dim=1).mean()
    else:
        L_prop_rank = J0.new_tensor(0.0)
    L_prop = L_prop_bce + 0.25 * L_prop_rank

    selected_mask, query_mask = _predicted_certificate_masks(outputs, batch, cfg)
    p_oracle = _certificate_oracle_probability(cfg)
    oracle_masks = _oracle_certificate_masks(outputs, batch) if p_oracle > 0.0 else None
    if oracle_masks is not None:
        oracle_selected, oracle_query = oracle_masks
        if p_oracle >= 1.0:
            selected_mask, query_mask = oracle_selected, oracle_query
        else:
            mix = (torch.rand((J0.shape[0],), device=J0.device) < p_oracle)
            selected_mask = torch.where(mix[:, None], oracle_selected, selected_mask)
            query_mask = torch.where(mix[:, None, None], oracle_query, query_mask)
    g_runtime = g * query_mask.float()
    budgeted_cost = finite_J0 + (g_runtime * selected_mask[:, :, None].float()).sum(dim=1)
    tau_q = float(cfg.get("tournament", {}).get("softmin_tau", 1.0))
    eps_cal = float(cfg.get("tournament", {}).get("epsilon_cal", cfg.get("calibration", {}).get("epsilon_cal", 0.0)))
    logits = _budgeted_tournament_scores(budgeted_cost, valid, tau_q, eps_cal)

    if "teacher_J_T" in batch:
        tau_T = float(cfg.get("training", {}).get("teacher_soft_target_tau", 1.0))
        teacher_cost = batch["teacher_J_T"].float().masked_fill(~valid, 1e9)
        p = torch.softmax(-teacher_cost / max(tau_T, 1e-6), dim=1)
        L_act = -(p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    else:
        L_act = F.cross_entropy(logits, target_action)

    # Optional post-hoc-style calibration surrogate: penalize pair margin residuals
    # above the configured epsilon_cal so the validation quantile has a training signal.
    eps_cal = float(cfg.get("tournament", {}).get("epsilon_cal", cfg.get("calibration", {}).get("epsilon_cal", 0.0)))
    if pair_mask.sum() > 0 and eps_cal > 0:
        cal_err = (M_hat_E[pair_mask] - batch["pair_margins"].float()[pair_mask]).abs()
        L_cal = F.relu(cal_err - eps_cal).mean()
    else:
        L_cal = J0.new_tensor(0.0)

    total = (
        float(lw.get("base", 1.0)) * L_base
        + float(lw.get("residual", 1.0)) * L_res
        + float(lw.get("full_interface_rank_aux", lw.get("rank", 0.1))) * L_rank
        + float(lw.get("proposal", lw.get("selection", 1.0))) * L_prop
        + float(lw.get("action", 1.0)) * L_act
        + float(lw.get("calibration", 0.0)) * L_cal
    )
    return {"loss": total, "L_base": L_base, "L_res": L_res, "L_rank": L_rank, "L_prop": L_prop, "L_sel": L_prop, "L_act": L_act, "L_cal": L_cal}
