from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from bdse.planner.hab import select_topm_atoms_hab
from bdse.planner.pair_screen import build_runtime_pairs_from_base, build_rival_sets_from_base
from bdse.planner.selector import runtime_greedy_selector, runtime_greedy_selector_pair_conditioned


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


def _neg_mask_value(x: torch.Tensor) -> float:
    if torch.is_floating_point(x):
        return float(torch.finfo(x.dtype).min / 2.0)
    return -1e9


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
    traj_np = batch.get("candidate_trajectories")
    traj_np = traj_np.detach().cpu().numpy().astype(np.float32) if traj_np is not None else None
    man_np = batch.get("candidate_maneuver_ids")
    man_np = man_np.detach().cpu().numpy().astype(np.int64) if man_np is not None else None
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
            min_family_slots=s_cfg.get("min_family_topm_slots", None),
        )
        # Mirror the hybrid selector used at deployment: hard/rule atoms that
        # support supervised margins must remain queryable even when proposal
        # logits are still immature.  This is not dense leakage into the final
        # certificate; it only expands the Top-M candidate pool before the same
        # budgeted greedy selector runs.
        if bool(s_cfg.get("force_decisive_hard_topm", True)) and "decisive_hard_mask" in batch:
            hard_train = batch["decisive_hard_mask"][bidx].detach().cpu().numpy().astype(bool) & active[bidx]
            forced = np.flatnonzero(hard_train)
            if forced.size:
                forced_cap = int(s_cfg.get("max_forced_hard_topm", max(1, M // 2)))
                forced = np.asarray(sorted(forced.tolist(), key=lambda i: (-float(logits[bidx, i]), int(i)))[:forced_cap], dtype=np.int64)
                non_forced = [int(i) for i in np.asarray(topm, dtype=np.int64).reshape(-1).tolist() if int(i) not in set(forced.tolist())]
                topm = np.asarray((forced.tolist() + non_forced)[:M], dtype=np.int64)
        atom_active = np.zeros((E,), dtype=bool)
        atom_active[topm] = True
        atom_active &= active[bidx]
        flags = batch.get("runtime_safety_flags")
        flag_np = flags[bidx].detach().cpu().numpy().astype(bool) if flags is not None else np.zeros((K,), dtype=bool)
        pairs, _ = build_runtime_pairs_from_base(
            J0[bidx], valid[bidx], flag_np, L0=L0, eta0=eta, lambda_near=lambda_near, lambda_safety=lambda_safety,
            candidate_trajectories=traj_np[bidx] if traj_np is not None else None,
            maneuver_ids=man_np[bidx] if man_np is not None else None,
            progress_pair_count=int(s_cfg.get("progress_pair_count", 8)),
            maneuver_pair_count=int(s_cfg.get("maneuver_pair_count", 8)),
        )
        rival_sets = build_rival_sets_from_base(
            J0[bidx], valid[bidx], flag_np, L_infer=L0, eta0=eta,
            candidate_trajectories=traj_np[bidx] if traj_np is not None else None,
            maneuver_ids=man_np[bidx] if man_np is not None else None,
            progress_rivals=int(s_cfg.get("progress_rivals", 4)),
            maneuver_rivals=int(s_cfg.get("maneuver_rivals", 4)),
        )
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
            bidirectional_pairs=bool(s_cfg.get("bidirectional_pairs", True)),
            reverse_pair_weight=float(s_cfg.get("reverse_pair_weight", 1.0)),
            pair_cap_multiplier=float(s_cfg.get("runtime_pair_cap_multiplier", 1.0)),
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




def _predicted_pair_certificate_masks(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> torch.Tensor:
    """Stop-gradient HAB masks for pair-conditioned action supervision.

    Runtime pair-conditioned BDSE selects atoms using signed atom-pair deltas
    d_i(a,b), not dense action costs g_i(a).  Training therefore needs a
    parallel stop-gradient selection path so L_act can send gradients through
    the same pair-margin head used at deployment.
    """
    if "pair_atom_delta" not in outputs or "pair_indices" not in batch:
        return outputs["J0"].new_zeros(outputs["proposal_logits"].shape, dtype=torch.bool)
    J0 = outputs["J0"].detach().cpu().numpy().astype(np.float32)
    delta = outputs["pair_atom_delta"].detach().cpu().numpy().astype(np.float32)
    pair_var_t = outputs.get("pair_atom_var")
    pair_var = pair_var_t.detach().cpu().numpy().astype(np.float32) if pair_var_t is not None else None
    pairs = batch["pair_indices"].detach().cpu().numpy().astype(np.int64)
    pair_valid = batch["pair_valid"].detach().cpu().numpy().astype(bool)
    pair_weights = batch.get("pair_weights", torch.ones_like(batch["pair_valid"], dtype=torch.float32)).detach().cpu().numpy().astype(np.float32)
    logits = outputs["proposal_logits"].detach().cpu().numpy().astype(np.float32)
    family_logits = outputs.get("family_logits")
    family_logits_np = family_logits.detach().cpu().numpy().astype(np.float32) if family_logits is not None else None
    valid = batch["candidate_valid"].detach().cpu().numpy().astype(bool)
    active = batch.get("evidence_active", torch.ones_like(outputs["proposal_logits"]).bool()).detach().cpu().numpy().astype(bool)
    costs = batch.get("evidence_budget_costs", torch.ones_like(outputs["proposal_logits"])).detach().cpu().numpy().astype(np.float32)
    fam_ids_t = batch.get("evidence_family_ids")
    fam_ids_np = fam_ids_t.detach().cpu().numpy().astype(np.int64) if fam_ids_t is not None else np.zeros_like(logits, dtype=np.int64)
    flags = batch.get("runtime_safety_flags")
    flags_np = flags.detach().cpu().numpy().astype(bool) if flags is not None else np.zeros_like(valid, dtype=bool)

    Bsz, E = logits.shape
    selected_mask = np.zeros((Bsz, E), dtype=bool)
    e_cfg = cfg.get("evidence", {})
    s_cfg = cfg.get("selector", {})
    t_cfg = cfg.get("tournament", {})
    c_cfg = cfg.get("calibration", {})
    budget = float(e_cfg.get("budget", 16))
    M = int(s_cfg.get("proposal_top_m", max(int(2 * budget), int(budget) + 1)))
    gamma = float(s_cfg.get("gamma_max_default", 100.0))
    eta = float(s_cfg.get("eta_pred", 1.0))
    beta_unc = float(t_cfg.get("beta_uncertainty", 0.0))
    eps_cal = float(t_cfg.get("epsilon_cal", c_cfg.get("epsilon_cal", 0.0)))
    lambda_info = float(s_cfg.get("lambda_info", 0.0))
    prior_var = s_cfg.get("unqueried_atom_variance", None)

    for bidx in range(Bsz):
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
            min_family_slots=s_cfg.get("min_family_topm_slots", None),
        )
        # Mirror the hybrid selector used at deployment: hard/rule atoms that
        # support supervised margins must remain queryable even when proposal
        # logits are still immature.  This is not dense leakage into the final
        # certificate; it only expands the Top-M candidate pool before the same
        # budgeted greedy selector runs.
        if bool(s_cfg.get("force_decisive_hard_topm", True)) and "decisive_hard_mask" in batch:
            hard_train = batch["decisive_hard_mask"][bidx].detach().cpu().numpy().astype(bool) & active[bidx]
            forced = np.flatnonzero(hard_train)
            if forced.size:
                forced_cap = int(s_cfg.get("max_forced_hard_topm", max(1, M // 2)))
                forced = np.asarray(sorted(forced.tolist(), key=lambda i: (-float(logits[bidx, i]), int(i)))[:forced_cap], dtype=np.int64)
                non_forced = [int(i) for i in np.asarray(topm, dtype=np.int64).reshape(-1).tolist() if int(i) not in set(forced.tolist())]
                topm = np.asarray((forced.tolist() + non_forced)[:M], dtype=np.int64)
        atom_active = np.zeros((E,), dtype=bool)
        atom_active[topm] = True
        atom_active &= active[bidx]
        pv = pair_valid[bidx]
        pair_arr = pairs[bidx][pv]
        weight_arr = pair_weights[bidx][pv]
        delta_arr = delta[bidx][:, pv]
        var_arr = pair_var[bidx][:, pv] if pair_var is not None else None
        # Remove out-of-range pairs defensively; padded pairs are already masked.
        if pair_arr.size:
            ok = (pair_arr[:, 0] >= 0) & (pair_arr[:, 0] < valid.shape[1]) & (pair_arr[:, 1] >= 0) & (pair_arr[:, 1] < valid.shape[1])
            ok &= valid[bidx, pair_arr[:, 0]] & valid[bidx, pair_arr[:, 1]]
            pair_arr = pair_arr[ok]
            weight_arr = weight_arr[ok]
            delta_arr = delta_arr[:, ok]
            if var_arr is not None:
                var_arr = var_arr[:, ok]
        result = runtime_greedy_selector_pair_conditioned(
            J0[bidx],
            delta_arr,
            pair_arr,
            weight_arr,
            costs[bidx],
            valid[bidx],
            flags_np[bidx],
            budget=budget,
            gamma_max=gamma,
            eta_pred=eta,
            atom_active_mask=atom_active,
            pair_atom_variance=var_arr,
            beta_uncertainty=beta_unc,
            epsilon_cal=eps_cal,
            lambda_info=lambda_info,
            prior_atom_variance=prior_var,
            family_ids=fam_ids_np[bidx],
            family_budget_caps=fam_budget.family_caps,
        )
        selected_mask[bidx, result.selected] = True
    return torch.from_numpy(selected_mask).to(outputs["J0"].device)


def _pair_conditioned_tournament_scores(
    J0: torch.Tensor,
    pair_delta: torch.Tensor,
    pairs: torch.Tensor,
    pair_valid: torch.Tensor,
    selected_mask: torch.Tensor,
    valid: torch.Tensor,
    tau: float,
    epsilon_cal: float = 0.0,
    pair_var: torch.Tensor | None = None,
    beta_uncertainty: float = 0.0,
) -> torch.Tensor:
    """Differentiable soft tournament from selected pair-conditioned deltas.

    Missing queried pairs fall back to the base margin, while present pairs update
    both directions to preserve antisymmetry: M(a,b)+=d and M(b,a)-=d.
    """
    B, K = J0.shape
    M = J0[:, None, :] - J0[:, :, None]
    support = (pair_delta * selected_mask[:, :, None].float()).sum(dim=1)
    pvalid = pair_valid.bool()
    a = pairs[..., 0].long().clamp(0, K - 1)
    b = pairs[..., 1].long().clamp(0, K - 1)
    pvalid = pvalid & valid.gather(1, a) & valid.gather(1, b)
    flat = torch.zeros((B, K * K), dtype=J0.dtype, device=J0.device)
    lin_ab = a * K + b
    lin_ba = b * K + a
    val = support.masked_fill(~pvalid, 0.0)
    flat.scatter_add_(1, lin_ab, val)
    flat.scatter_add_(1, lin_ba, -val)
    M = M + flat.view(B, K, K)
    sigma = None
    if pair_var is not None and beta_uncertainty > 0:
        var_support = (pair_var.clamp_min(0.0) * selected_mask[:, :, None].float()).sum(dim=1).masked_fill(~pvalid, 0.0)
        vflat = torch.zeros((B, K * K), dtype=J0.dtype, device=J0.device)
        vflat.scatter_add_(1, lin_ab, var_support)
        vflat.scatter_add_(1, lin_ba, var_support)
        sigma = torch.sqrt(vflat.view(B, K, K).clamp_min(0.0) + 1e-12)
    if sigma is not None:
        M = M - float(beta_uncertainty) * sigma
    M = M - float(epsilon_cal)
    eye = torch.eye(K, dtype=torch.bool, device=J0.device)[None]
    rival_valid = valid[:, None, :].expand(B, K, K) & (~eye)
    vals = M.masked_fill(~rival_valid, float("inf"))
    scores = _softmin(vals, tau, dim=2)
    no_rivals = ~rival_valid.any(dim=2)
    scores = scores.masked_fill(no_rivals, 0.0).masked_fill(~valid, _neg_mask_value(scores))
    return scores

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
    scores = scores.masked_fill(no_rivals, 0.0).masked_fill(~valid, _neg_mask_value(scores))
    return scores


def _weighted_mean(loss: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return loss.new_tensor(0.0)
    w = weights.masked_fill(~mask, 0.0)
    return (loss * w).sum() / w.sum().clamp_min(1e-6)


def _valid_row_scale(values: torch.Tensor, valid: torch.Tensor, min_scale: float = 1.0) -> torch.Tensor:
    """Per-sample robust scale for costs/margins with invalid entries masked out."""
    mask = valid.bool() & torch.isfinite(values)
    count = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    safe = torch.where(mask, values, torch.zeros_like(values))
    mean = safe.sum(dim=1, keepdim=True) / count
    var = (torch.where(mask, values - mean, torch.zeros_like(values)).pow(2).sum(dim=1, keepdim=True) / count).clamp_min(0.0)
    scale = torch.sqrt(var).clamp_min(float(min_scale))
    return scale.detach()


def _negative_cost_logits(cost: torch.Tensor, valid: torch.Tensor, min_scale: float = 1.0) -> torch.Tensor:
    """Convert arbitrary-scale costs to stable CE logits without changing argmin order."""
    finite_cost = torch.where(torch.isfinite(cost), cost, torch.zeros_like(cost))
    scale = _valid_row_scale(finite_cost, valid, min_scale=min_scale)
    center = torch.where(valid.bool(), finite_cost, torch.zeros_like(finite_cost)).sum(dim=1, keepdim=True) / valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    logits = -(finite_cost - center.detach()) / scale
    return logits.masked_fill(~valid.bool(), _neg_mask_value(logits))


def _pair_margin_scale(pair_margins: torch.Tensor, pair_mask: torch.Tensor, default: float = 100.0) -> torch.Tensor:
    vals = pair_margins[pair_mask].detach().abs()
    if vals.numel() == 0:
        return pair_margins.new_tensor(float(default))
    return vals.median().clamp_min(float(default))


def compute_bdse_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    # Losses include large teacher costs and masking sentinels.  Compute them in
    # float32 even when model forward uses CUDA AMP; otherwise fp16 masks such as
    # 1e9/-1e9 overflow.
    out = {
        k: (v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v)
        for k, v in outputs.items()
    }
    J0 = out["J0"]
    g = out["g"]
    g_var = out.get("g_var")
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
    pred_atom_delta = out.get("pair_atom_delta", g_b - g_a)
    res_pred = pred_atom_delta.sum(dim=1)
    pair_mask = pair_valid & torch.isfinite(residual_T)
    if pair_mask.sum() > 0:
        L_res_terms = F.huber_loss(res_pred[pair_mask], residual_T[pair_mask], delta=1.0, reduction="none")
        L_res = (L_res_terms * pair_weights[pair_mask]).sum() / pair_weights[pair_mask].sum().clamp_min(1e-6)
    else:
        L_res = J0.new_tensor(0.0)

    # Explicit action-conditioned local-cost supervision keeps the factorized
    # g_i(a) head usable for diagnostics/fallback and makes the teacher
    # partition J_T = J_base + sum_i g_i learnable, even when the main runtime
    # path uses the direct pair-conditioned scorer.
    teacher_g = batch.get("teacher_g_evid")
    e_mask = batch.get("evidence_active", torch.ones_like(out["proposal_logits"]).bool()).bool()
    if teacher_g is not None:
        atom_cost_mask = e_mask[:, :, None] & valid[:, None, :] & torch.isfinite(teacher_g.float())
        atom_cost_err = F.huber_loss(g, teacher_g.float(), delta=1.0, reduction="none")
        nonzero_g = teacher_g.float().abs() > 1e-6
        atom_zero_w = float(train_cfg.get("atom_zero_weight", train_cfg.get("pair_zero_weight", 0.1)))
        atom_cost_w = atom_zero_w + (1.0 - atom_zero_w) * nonzero_g.float()
        L_atom = _weighted_mean(atom_cost_err, atom_cost_w, atom_cost_mask)
    else:
        L_atom = J0.new_tensor(0.0)

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
    pair_var_pred = out.get("pair_atom_var")
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
    # Pair margins have scene-dependent magnitudes.  A fixed tau=1 makes the
    # rank loss explode numerically while contributing weak, unstable gradients
    # relative to the deployed decision objective.  Normalize by the supervised
    # margin scale and optionally regress the normalized margin value.
    tau_rank = float(train_cfg.get("rank_tau", train_cfg.get("rank_margin", 1.0)))
    if pair_mask.sum() > 0:
        margin_scale_cfg = float(train_cfg.get("pair_margin_scale", 0.0) or 0.0)
        margin_scale = J0.new_tensor(margin_scale_cfg) if margin_scale_cfg > 0 else _pair_margin_scale(batch["pair_margins"].float(), pair_mask, default=float(train_cfg.get("pair_margin_min_scale", 100.0)))
        rank_terms = F.softplus(-M_hat_E[pair_mask] / (margin_scale * max(tau_rank, 1e-6)))
        L_rank_cls = (pair_weights[pair_mask] * rank_terms).sum() / pair_weights[pair_mask].sum().clamp_min(1e-6)
        target_margin = batch["pair_margins"].float()
        reg_terms = F.huber_loss(M_hat_E[pair_mask] / margin_scale, target_margin[pair_mask] / margin_scale, delta=float(train_cfg.get("pair_margin_reg_delta", 1.0)), reduction="none")
        L_rank_reg = (pair_weights[pair_mask] * reg_terms).sum() / pair_weights[pair_mask].sum().clamp_min(1e-6)
        L_rank = L_rank_cls + float(train_cfg.get("pair_margin_reg_weight", 1.0)) * L_rank_reg
    else:
        L_rank = J0.new_tensor(0.0)

    target_sel = batch.get("oracle_selected_mask")
    decisive = batch.get("decisive_atom_mask")
    decisive_hard = batch.get("decisive_hard_mask")
    gain = batch.get("proposal_target_gain")
    if target_sel is not None:
        target_bool = target_sel.bool()
        if decisive is not None and bool(train_cfg.get("proposal_include_decisive_atoms", True)):
            target_bool = target_bool | decisive.bool()
        if decisive_hard is not None and bool(train_cfg.get("proposal_include_decisive_hard", True)):
            target_bool = target_bool | decisive_hard.bool()
        bce = F.binary_cross_entropy_with_logits(out["proposal_logits"], target_bool.float(), reduction="none")
        pos_w = float(train_cfg.get("proposal_positive_weight", 4.0))
        hard_w = float(train_cfg.get("proposal_decisive_hard_weight", 4.0))
        weights_prop = torch.ones_like(bce)
        weights_prop = torch.where(target_bool, weights_prop * pos_w, weights_prop)
        if decisive_hard is not None:
            weights_prop = torch.where(decisive_hard.bool(), weights_prop * hard_w, weights_prop)
        mask_prop = e_mask & torch.isfinite(bce)
        L_prop_bce = (bce[mask_prop] * weights_prop[mask_prop]).sum() / weights_prop[mask_prop].sum().clamp_min(1e-6) if mask_prop.sum() > 0 else J0.new_tensor(0.0)
    else:
        L_prop_bce = J0.new_tensor(0.0)
    if gain is not None and e_mask.sum() > 0:
        target_gain = gain.float().masked_fill(~e_mask, 0.0)
        if decisive_hard is not None:
            # Give safety/rule atoms nonzero listwise mass even when the greedy
            # trace spent its budget elsewhere; this improves hard-evidence recall.
            boost = float(train_cfg.get("proposal_decisive_hard_gain_boost", 1.0))
            target_gain = target_gain + boost * decisive_hard.float()
        target_dist = target_gain / target_gain.sum(dim=1, keepdim=True).clamp_min(1e-6)
        logp = F.log_softmax(out["proposal_logits"].masked_fill(~e_mask, _neg_mask_value(out["proposal_logits"])), dim=1)
        L_prop_rank = -(target_dist * logp).sum(dim=1).mean()
    else:
        L_prop_rank = J0.new_tensor(0.0)
    L_prop = L_prop_bce + float(train_cfg.get("proposal_rank_weight", 0.5)) * L_prop_rank

    # Family-listwise loss for the HAB family gate.
    fam_gain = batch.get("family_target_gain")
    family_logits = out.get("family_logits")
    if fam_gain is not None and family_logits is not None:
        fam_active = batch.get("family_target_active", out.get("family_active", torch.ones_like(family_logits).bool())).bool()
        target = fam_gain.float().masked_fill(~fam_active, 0.0)
        mass = target.sum(dim=1, keepdim=True)
        uniform = fam_active.float() / fam_active.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        target_dist = torch.where(mass > 1e-6, target / mass.clamp_min(1e-6), uniform)
        logp = F.log_softmax(family_logits.masked_fill(~fam_active, _neg_mask_value(family_logits)), dim=1)
        L_fam = -(target_dist * logp).sum(dim=1).mean()
    else:
        L_fam = J0.new_tensor(0.0)

    selected_mask, query_mask = _predicted_certificate_masks(out, batch, cfg)
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
    logits_action = _budgeted_tournament_scores(budgeted_cost, valid, tau_q, eps_cal, sigma=sigma, beta_uncertainty=beta_unc)

    use_pair_act = bool(cfg.get("runtime", {}).get("use_pair_conditioned_margins", cfg.get("model", {}).get("pair_conditioned", True)))
    if use_pair_act and "pair_atom_delta" in out:
        pair_selected_mask = _predicted_pair_certificate_masks(out, batch, cfg)
        logits_pair = _pair_conditioned_tournament_scores(
            finite_J0,
            out["pair_atom_delta"],
            pairs,
            pair_valid,
            pair_selected_mask,
            valid,
            tau_q,
            eps_cal,
            pair_var=out.get("pair_atom_var"),
            beta_uncertainty=beta_unc,
        )
    else:
        logits_pair = None

    full_pred_cost = finite_J0 + (g * e_mask[:, :, None].float()).sum(dim=1)
    full_pred_cost = full_pred_cost.masked_fill(~valid, J0.new_tensor(1e6))
    full_logits = _negative_cost_logits(full_pred_cost, valid, min_scale=float(train_cfg.get("full_action_min_scale", 1.0)))
    hard_action_targets = bool(train_cfg.get("hard_action_targets", True))
    if "teacher_J_T" in batch and not hard_action_targets:
        tau_T = float(train_cfg.get("teacher_soft_target_tau", 1.0))
        teacher_cost = batch["teacher_J_T"].float().masked_fill(~valid, J0.new_tensor(1e9))
        teacher_logits = _negative_cost_logits(teacher_cost, valid, min_scale=float(train_cfg.get("teacher_action_min_scale", 1.0))) / max(tau_T, 1e-6)
        p = torch.softmax(teacher_logits, dim=1)
        L_act_action = -(p * F.log_softmax(logits_action, dim=1)).sum(dim=1).mean()
        L_act_pair = -(p * F.log_softmax(logits_pair, dim=1)).sum(dim=1).mean() if logits_pair is not None else J0.new_tensor(0.0)
        L_full_action = -(p * F.log_softmax(full_logits, dim=1)).sum(dim=1).mean()
    else:
        L_act_action = F.cross_entropy(logits_action, target_action)
        L_act_pair = F.cross_entropy(logits_pair, target_action) if logits_pair is not None else J0.new_tensor(0.0)
        L_full_action = F.cross_entropy(full_logits, target_action)

    # Dense full-interface margin distillation.  Use a per-scene scale so this
    # term optimizes ordering rather than absolute cost units.
    Bsz, Ksz = full_pred_cost.shape
    tgt = target_action.clamp_min(0).clamp_max(Ksz - 1)
    c_star = torch.gather(full_pred_cost, 1, tgt[:, None]).expand(Bsz, Ksz)
    margin_mask = valid.clone()
    margin_mask.scatter_(1, tgt[:, None], False)
    if "teacher_J_T" in batch:
        full_scale = _valid_row_scale(batch["teacher_J_T"].float().masked_fill(~valid, 0.0), valid, min_scale=float(train_cfg.get("full_margin_min_scale", 100.0)))
    else:
        full_scale = _valid_row_scale(full_pred_cost, valid, min_scale=float(train_cfg.get("full_margin_min_scale", 100.0)))
    L_full_margin_terms = F.softplus((c_star - full_pred_cost) / (full_scale * max(float(train_cfg.get("full_margin_tau", 1.0)), 1e-6)))
    L_full_margin = L_full_margin_terms[margin_mask].mean() if margin_mask.sum() > 0 else J0.new_tensor(0.0)

    hard_mask = batch.get("teacher_hard_violation")
    if hard_mask is not None:
        hard_mask = hard_mask.bool() & valid
        safe_mask = (~hard_mask) & valid
        safe_cost = full_pred_cost.masked_fill(~safe_mask, J0.new_tensor(1e6)).min(dim=1, keepdim=True).values
        hard_cost = full_pred_cost.masked_fill(~hard_mask, J0.new_tensor(1e6))
        feasible_pair = safe_mask.any(dim=1, keepdim=True) & hard_mask
        hard_scale = _valid_row_scale(batch["teacher_J_T"].float().masked_fill(~valid, 0.0), valid, min_scale=float(train_cfg.get("hard_feasibility_min_scale", 100.0))) if "teacher_J_T" in batch else _valid_row_scale(full_pred_cost, valid, min_scale=float(train_cfg.get("hard_feasibility_min_scale", 100.0)))
        L_hard_feas = F.softplus((safe_cost - hard_cost + float(train_cfg.get("hard_feasibility_margin", 10.0))) / (hard_scale * max(float(train_cfg.get("hard_feasibility_tau", 1.0)), 1e-6)))
        L_hard_feas = L_hard_feas[feasible_pair].mean() if feasible_pair.sum() > 0 else J0.new_tensor(0.0)
    else:
        L_hard_feas = J0.new_tensor(0.0)
    pair_act_weight = float(train_cfg.get("pair_action_loss_weight", 1.0 if logits_pair is not None else 0.0))
    action_act_weight = float(train_cfg.get("action_conditioned_action_loss_weight", 0.25 if logits_pair is not None else 1.0))
    norm_act = max(pair_act_weight + action_act_weight, 1e-6)
    L_act = (pair_act_weight * L_act_pair + action_act_weight * L_act_action) / norm_act

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
        + float(lw.get("atom_cost", 0.25)) * L_atom
        + float(lw.get("uncertainty", 0.1)) * L_unc
        + float(lw.get("full_interface_rank_aux", lw.get("rank", 0.1))) * L_rank
        + float(lw.get("family", 0.5)) * L_fam
        + float(lw.get("proposal", lw.get("selection", 1.0))) * L_prop
        + float(lw.get("action", 1.0)) * L_act
        + float(lw.get("full_action", 1.0)) * L_full_action
        + float(lw.get("full_margin", 0.5)) * L_full_margin
        + float(lw.get("hard_feasibility", 0.5)) * L_hard_feas
        + float(lw.get("calibration", 0.0)) * L_cal
    )
    return {
        "loss": total,
        "L_base": L_base,
        "L_pair": L_pair,
        "L_res": L_res,
        "L_atom": L_atom,
        "L_unc": L_unc,
        "L_rank": L_rank,
        "L_fam": L_fam,
        "L_prop": L_prop,
        "L_sel": L_prop,
        "L_act": L_act,
        "L_act_pair": L_act_pair if 'L_act_pair' in locals() else J0.new_tensor(0.0),
        "L_act_action": L_act_action if 'L_act_action' in locals() else J0.new_tensor(0.0),
        "L_full_action": L_full_action if 'L_full_action' in locals() else J0.new_tensor(0.0),
        "L_full_margin": L_full_margin if 'L_full_margin' in locals() else J0.new_tensor(0.0),
        "L_hard_feas": L_hard_feas if 'L_hard_feas' in locals() else J0.new_tensor(0.0),
        "L_cal": L_cal,
    }
