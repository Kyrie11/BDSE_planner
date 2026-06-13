from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from bdse.planner.hab import select_topm_atoms_hab
from bdse.planner.pair_screen import build_runtime_pairs_from_base, build_rival_sets_from_base
from bdse.planner.selector import runtime_greedy_selector


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
    """Stop-gradient deployment certificate masks for L_act.

    Dense ``g`` is available during training, but the deployment path only sees
    scores for HAB Top-M atoms and actions contained in the base rival graph.
    This helper mirrors that path and returns the selected atom mask plus the
    sparse atom-action query mask, preventing dense-label leakage into action
    supervision.
    """
    J0 = outputs["J0"].detach().cpu().numpy().astype(np.float32)
    g = outputs["g"].detach().cpu().numpy().astype(np.float32)
    g_var = outputs.get("g_var")
    g_var_np = g_var.detach().cpu().numpy().astype(np.float32) if g_var is not None else None
    logits = outputs["proposal_logits"].detach().cpu().numpy().astype(np.float32)
    family_logits = outputs.get("family_logits")
    family_logits_np = family_logits.detach().cpu().numpy().astype(np.float32) if family_logits is not None else None
    valid = batch["candidate_valid"].detach().cpu().numpy().astype(bool)
    active = batch.get("evidence_active", torch.ones_like(outputs["proposal_logits"]).bool()).detach().cpu().numpy().astype(bool)
    costs = batch.get("evidence_budget_costs", torch.ones_like(outputs["proposal_logits"])).detach().cpu().numpy().astype(np.float32)
    fam_ids_t = batch.get("evidence_family_ids")
    fam_ids_np = fam_ids_t.detach().cpu().numpy().astype(np.int64) if fam_ids_t is not None else np.zeros_like(logits, dtype=np.int64)
    B, E = logits.shape
    K = valid.shape[1]
    selected_mask = np.zeros((B, E), dtype=bool)
    query_mask = np.zeros((B, E, K), dtype=bool)

    e_cfg = cfg.get("evidence", {})
    s_cfg = cfg.get("selector", {})
    t_cfg = cfg.get("tournament", {})
    c_cfg = cfg.get("calibration", {})
    budget = float(e_cfg.get("budget", 16))
    M = int(s_cfg.get("proposal_top_m", max(int(2 * budget), int(budget) + 1)))
    L0 = int(t_cfg.get("L_infer", 16))
    eta = float(s_cfg.get("eta_pred", 1.0))
    gamma = float(s_cfg.get("gamma_max_default", 100.0))
    lambda_near = float(s_cfg.get("lambda_near", 1.0))
    lambda_safety = float(s_cfg.get("lambda_safety", 2.0))
    beta_unc = float(t_cfg.get("beta_uncertainty", 0.0))
    eps_cal = float(t_cfg.get("epsilon_cal", c_cfg.get("epsilon_cal", 0.0)))
    lambda_info = float(s_cfg.get("lambda_info", 0.0))
    prior_var = s_cfg.get("unqueried_atom_variance", None)

    for bidx in range(B):
        topm, fam_budget, _ = select_topm_atoms_hab(
            logits[bidx],
            fam_ids_np[bidx],
            active[bidx],
            costs[bidx],
            budget,
            M,
            family_scores=family_logits_np[bidx] if family_logits_np is not None else None,
            free_budget=s_cfg.get("hab_free_budget", None),
            reserve_fraction=float(s_cfg.get("hab_reserve_fraction", 0.2)),
            enabled=bool(s_cfg.get("hab_enabled", True)),
        )
        atom_active = np.zeros((E,), dtype=bool)
        atom_active[topm] = True
        atom_active &= active[bidx]
        flags = batch.get("runtime_safety_flags")
        flag_np = flags[bidx].detach().cpu().numpy().astype(bool) if flags is not None else np.zeros((K,), dtype=bool)
        pairs, _ = build_runtime_pairs_from_base(
            J0[bidx], valid[bidx], flag_np, L0=L0, eta0=eta, lambda_near=lambda_near, lambda_safety=lambda_safety
        )
        rival_sets = build_rival_sets_from_base(J0[bidx], valid[bidx], flag_np, L_infer=L0, eta0=eta)
        action_set: set[int] = set()
        for a_idx, rivals in enumerate(rival_sets):
            if not bool(valid[bidx, a_idx]) or not rivals:
                continue
            action_set.add(int(a_idx))
            action_set.update(int(r) for r in rivals)
        if action_set:
            action_ids = np.asarray(sorted(action_set), dtype=np.int64)
            for ei in np.flatnonzero(atom_active):
                query_mask[bidx, ei, action_ids] = True
        elif len(pairs):
            action_ids = np.unique(pairs.reshape(-1))
            for ei in np.flatnonzero(atom_active):
                query_mask[bidx, ei, action_ids] = True
        g_sparse = np.where(query_mask[bidx], g[bidx], 0.0)
        var_sparse = np.where(query_mask[bidx], g_var_np[bidx], 0.0) if g_var_np is not None else None
        result = runtime_greedy_selector(
            J0[bidx],
            g_sparse,
            costs[bidx],
            valid[bidx],
            flag_np,
            budget,
            L_infer=L0,
            gamma_max=gamma,
            eta_pred=eta,
            lambda_near=lambda_near,
            lambda_safety=lambda_safety,
            atom_active_mask=atom_active,
            predicted_atom_variance=var_sparse,
            beta_uncertainty=beta_unc,
            epsilon_cal=eps_cal,
            lambda_info=lambda_info,
            prior_atom_variance=prior_var,
            family_ids=fam_ids_np[bidx],
            family_budget_caps=fam_budget.family_caps,
        )
        selected_mask[bidx, result.selected] = True
    device = outputs["J0"].device
    return torch.from_numpy(selected_mask).to(device), torch.from_numpy(query_mask).to(device)


def _softmin(vals: torch.Tensor, tau: float, dim: int) -> torch.Tensor:
    if tau <= 0:
        return vals.min(dim=dim).values
    return -float(tau) * torch.logsumexp(-vals / float(tau), dim=dim)


def _budgeted_tournament_scores(
    cost: torch.Tensor,
    valid: torch.Tensor,
    tau: float,
    epsilon_cal: float = 0.0,
    sigma: torch.Tensor | None = None,
    beta_uncertainty: float = 0.0,
) -> torch.Tensor:
    # M(a,b)=cost[b]-cost[a].  Higher score means larger lower margin against rivals.
    B, K = cost.shape
    margins = cost[:, None, :] - cost[:, :, None]
    if sigma is not None:
        margins = margins - float(beta_uncertainty) * sigma
    margins = margins - float(epsilon_cal)
    eye = torch.eye(K, dtype=torch.bool, device=cost.device)[None]
    rival_valid = valid[:, None, :].expand(B, K, K) & (~eye)
    vals = margins.masked_fill(~rival_valid, float("inf"))
    scores = _softmin(vals, tau, dim=2)
    no_rivals = ~rival_valid.any(dim=2)
    scores = scores.masked_fill(no_rivals, 0.0).masked_fill(~valid, -1e9)
    return scores


def _weighted_mean(loss: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return loss.new_tensor(0.0)
    w = weights.masked_fill(~mask, 0.0)
    return (loss * w).sum() / w.sum().clamp_min(1e-6)


def compute_bdse_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    J0 = outputs["J0"]
    g = outputs["g"]
    g_var = outputs.get("g_var")
    valid = batch["candidate_valid"].bool()
    J_base_T = batch["teacher_J_base"].float()
    pairs = batch["pair_indices"].long()
    pair_valid = batch["pair_valid"].bool()
    pair_weights = batch["pair_weights"].float()
    residual_T = batch["pair_residuals"].float()
    target_action = batch["teacher_a_star"].long()
    train_cfg = cfg.get("training", {})
    lw = train_cfg.get("loss_weights", {})

    finite_J0 = torch.where(torch.isfinite(J0), J0, torch.zeros_like(J0))
    L_base = robust_loss(finite_J0, J_base_T, valid)

    g_a, g_b = pair_gather(g, pairs)
    # Prefer the paper-faithful pair-conditioned scorer d_i(a,b) when present.
    # The factorized g_i(b)-g_i(a) path remains for ablations/backward-compatible
    # checkpoints and for dense action-cost diagnostics.
    pred_atom_delta = outputs.get("pair_atom_delta", g_b - g_a)
    res_pred = pred_atom_delta.sum(dim=1)
    pair_mask = pair_valid & torch.isfinite(residual_T)
    if pair_mask.sum() > 0:
        L_res_terms = F.huber_loss(res_pred[pair_mask], residual_T[pair_mask], delta=1.0, reduction="none")
        L_res = (L_res_terms * pair_weights[pair_mask]).sum() / pair_weights[pair_mask].sum().clamp_min(1e-6)
    else:
        L_res = J0.new_tensor(0.0)

    # Explicit atom-level pair-margin supervision from teacher d_i^T(a,b).
    teacher_g = batch.get("teacher_g_evid")
    e_mask = batch.get("evidence_active", torch.ones_like(outputs["proposal_logits"]).bool()).bool()
    if teacher_g is not None:
        tg_a, tg_b = pair_gather(teacher_g.float(), pairs)
        true_atom_delta = tg_b - tg_a
        atom_pair_mask = e_mask[:, :, None] & pair_mask[:, None, :]
        nonzero = true_atom_delta.abs() > 1e-6
        zero_w = float(train_cfg.get("pair_zero_weight", 0.1))
        atom_weights = pair_weights[:, None, :] * (zero_w + (1.0 - zero_w) * nonzero.float())
        L_pair_terms = F.huber_loss(pred_atom_delta, true_atom_delta, delta=1.0, reduction="none")
        L_pair = _weighted_mean(L_pair_terms, atom_weights, atom_pair_mask)
    else:
        true_atom_delta = None
        atom_pair_mask = None
        atom_weights = None
        L_pair = J0.new_tensor(0.0)

    # Heteroscedastic uncertainty: predict variance for each pair atom delta.
    pair_var_pred = outputs.get("pair_atom_var")
    if true_atom_delta is not None and atom_pair_mask is not None and atom_weights is not None and (pair_var_pred is not None or g_var is not None):
        if pair_var_pred is not None:
            pair_var = pair_var_pred.clamp_min(1e-6)
        else:
            v_a, v_b = pair_gather(g_var, pairs)
            pair_var = (v_a + v_b).clamp_min(1e-6)
        err2 = (pred_atom_delta - true_atom_delta).pow(2)
        nll = 0.5 * (err2 / pair_var + torch.log(pair_var))
        L_unc = _weighted_mean(nll, atom_weights, atom_pair_mask)
    else:
        L_unc = J0.new_tensor(0.0)

    J_a_pair, J_b_pair = pair_gather(finite_J0, pairs)
    M_hat_E = (J_b_pair - J_a_pair) + pred_atom_delta.sum(dim=1)
    tau_rank = float(train_cfg.get("rank_tau", train_cfg.get("rank_margin", 1.0)))
    if pair_mask.sum() > 0:
        rank_terms = F.softplus(-M_hat_E[pair_mask] / max(tau_rank, 1e-6))
        L_rank = (pair_weights[pair_mask] * rank_terms).sum() / pair_weights[pair_mask].sum().clamp_min(1e-6)
    else:
        L_rank = J0.new_tensor(0.0)

    target_sel = batch.get("oracle_selected_mask")
    gain = batch.get("proposal_target_gain")
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

    # Family-listwise loss for the HAB family gate.
    fam_gain = batch.get("family_target_gain")
    family_logits = outputs.get("family_logits")
    if fam_gain is not None and family_logits is not None:
        fam_active = batch.get("family_target_active", outputs.get("family_active", torch.ones_like(family_logits).bool())).bool()
        target = fam_gain.float().masked_fill(~fam_active, 0.0)
        mass = target.sum(dim=1, keepdim=True)
        uniform = fam_active.float() / fam_active.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        target_dist = torch.where(mass > 1e-6, target / mass.clamp_min(1e-6), uniform)
        logp = F.log_softmax(family_logits.masked_fill(~fam_active, -1e9), dim=1)
        L_fam = -(target_dist * logp).sum(dim=1).mean()
    else:
        L_fam = J0.new_tensor(0.0)

    selected_mask, query_mask = _predicted_certificate_masks(outputs, batch, cfg)
    g_runtime = g * query_mask.float()
    budgeted_cost = finite_J0 + (g_runtime * selected_mask[:, :, None].float()).sum(dim=1)
    tau_q = float(cfg.get("tournament", {}).get("softmin_tau", 1.0))
    eps_cal = float(cfg.get("tournament", {}).get("epsilon_cal", cfg.get("calibration", {}).get("epsilon_cal", 0.0)))
    beta_unc = float(cfg.get("tournament", {}).get("beta_uncertainty", 0.0))
    if g_var is not None:
        selected_var = (g_var * query_mask.float() * selected_mask[:, :, None].float()).sum(dim=1).clamp_min(0.0)
        sigma = torch.sqrt(selected_var[:, :, None] + selected_var[:, None, :] + 1e-12)
    else:
        sigma = None
    logits = _budgeted_tournament_scores(budgeted_cost, valid, tau_q, eps_cal, sigma=sigma, beta_uncertainty=beta_unc)

    if "teacher_J_T" in batch:
        tau_T = float(train_cfg.get("teacher_soft_target_tau", 1.0))
        teacher_cost = batch["teacher_J_T"].float().masked_fill(~valid, 1e9)
        p = torch.softmax(-teacher_cost / max(tau_T, 1e-6), dim=1)
        L_act = -(p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    else:
        L_act = F.cross_entropy(logits, target_action)

    # Optional post-hoc-style calibration surrogate: penalize pair margin residuals
    # above the configured epsilon_cal so the validation quantile has a training signal.
    if pair_mask.sum() > 0 and eps_cal > 0:
        cal_err = (M_hat_E[pair_mask] - batch["pair_margins"].float()[pair_mask]).abs()
        L_cal = F.relu(cal_err - eps_cal).mean()
    else:
        L_cal = J0.new_tensor(0.0)

    total = (
        float(lw.get("base", 1.0)) * L_base
        + float(lw.get("pair", 1.0)) * L_pair
        + float(lw.get("residual", 1.0)) * L_res
        + float(lw.get("uncertainty", 0.1)) * L_unc
        + float(lw.get("full_interface_rank_aux", lw.get("rank", 0.1))) * L_rank
        + float(lw.get("family", 0.5)) * L_fam
        + float(lw.get("proposal", lw.get("selection", 1.0))) * L_prop
        + float(lw.get("action", 1.0)) * L_act
        + float(lw.get("calibration", 0.0)) * L_cal
    )
    return {
        "loss": total,
        "L_base": L_base,
        "L_pair": L_pair,
        "L_res": L_res,
        "L_unc": L_unc,
        "L_rank": L_rank,
        "L_fam": L_fam,
        "L_prop": L_prop,
        "L_sel": L_prop,
        "L_act": L_act,
        "L_cal": L_cal,
    }
