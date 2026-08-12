"""Budgeted decisive-margin marginal utility (BDMU).

V64.3.8 trains acquisition on the same one-sided decision-margin object used by
DARM, instead of treating literal winner flips as the only positive target.
The utility is local to a *fixed-budget reference set* S_B:

* for an atom outside S_B, value is the best reduction in teacher decisive-margin
  deficit under a budget-feasible single exchange (or a direct add only when
  genuine budget slack exists);
* for an atom inside S_B, value is the increase in deficit caused by removing it.

Both terms are non-negative, cost-normalized and evaluated only on the teacher
winner versus its nearest valid teacher rivals.  The target therefore never
requires a B+1 interface.  Exact winner-flip atoms become a high-value limiting
case without collapsing supervision to a sparse binary event.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BDMUConfig:
    budget: float = 16.0
    rival_count: int = 4
    preserve_fraction: float = 0.60
    margin_floor: float = 0.02
    margin_cap: float = 0.75
    rival_temperature: float = 0.20
    min_action_scale: float = 100.0
    cost_power: float = 1.0
    min_atom_cost: float = 1.0e-3
    utility_epsilon: float = 1.0e-8


def _torch_row_scale(values: torch.Tensor, valid: torch.Tensor, min_scale: float) -> torch.Tensor:
    mask = valid.bool() & torch.isfinite(values)
    count = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    safe = torch.where(mask, values, torch.zeros_like(values))
    mean = safe.sum(dim=1, keepdim=True) / count
    var = (
        torch.where(mask, values - mean, torch.zeros_like(values)).square().sum(dim=1, keepdim=True)
        / count
    ).clamp_min(0.0)
    return torch.sqrt(var).clamp_min(float(min_scale))


@torch.no_grad()
def budgeted_decisive_margin_utility_torch(
    teacher_cost: torch.Tensor,
    teacher_g: torch.Tensor,
    active: torch.Tensor,
    valid: torch.Tensor,
    teacher_action: torch.Tensor,
    reference_mask: torch.Tensor,
    atom_costs: torch.Tensor,
    cfg: BDMUConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return per-atom BDMU targets and scene diagnostics.

    Shapes: teacher_cost [B,K], teacher_g [B,E,K], masks [B,E]/[B,K].
    Teacher costs are assumed to contain the complete evidence sum.  Scenes for
    which scalar teacher costs do not reproduce ``teacher_action`` are excluded.
    """
    tc = teacher_cost.detach().float()
    tg = teacher_g.detach().float()
    active = active.detach().bool()
    valid = valid.detach().bool()
    ref = reference_mask.detach().bool() & active
    costs = atom_costs.detach().float().clamp_min(float(cfg.min_atom_cost))
    target = teacher_action.detach().long()
    B, E, K = tg.shape
    if tc.shape != (B, K):
        raise ValueError(f"teacher_cost must be [B,K], got {tuple(tc.shape)} for teacher_g={tuple(tg.shape)}")
    if active.shape != (B, E) or ref.shape != (B, E) or valid.shape != (B, K):
        raise ValueError("BDMU masks have incompatible shapes")

    inf = torch.tensor(float("inf"), device=tc.device, dtype=tc.dtype)
    dense = tc.masked_fill(~valid, inf)
    scalar = dense.argmin(dim=1)
    in_range = (target >= 0) & (target < K)
    target_valid = valid.gather(1, target.clamp(0, K - 1)[:, None]).squeeze(1)
    aligned = in_range & target_valid & scalar.eq(target)

    active_f = active[:, :, None].float()
    teacher_base = tc - (tg * active_f).sum(dim=1)
    ref_cost = teacher_base + (tg * ref[:, :, None].float()).sum(dim=1)
    scale = _torch_row_scale(tc, valid, float(cfg.min_action_scale))
    winner_idx = target.clamp(0, K - 1)
    full_winner = tc.gather(1, winner_idx[:, None])
    ref_winner = ref_cost.gather(1, winner_idx[:, None])
    full_margin = (tc - full_winner) / scale
    ref_margin = (ref_cost - ref_winner) / scale

    rival_valid = valid.clone()
    rival_valid.scatter_(1, winner_idx[:, None], False)
    # Teacher winner must strictly dominate the rival under the scalar teacher.
    rival_valid &= full_margin > 0.0
    r = min(max(int(cfg.rival_count), 1), max(K - 1, 1))
    order_value = full_margin.masked_fill(~rival_valid, inf)
    rival_margin, rival_idx = torch.topk(order_value, k=r, dim=1, largest=False, sorted=True)
    rival_mask = torch.isfinite(rival_margin) & aligned[:, None]

    full_pos = rival_margin.clamp_min(0.0)
    desired = torch.maximum(
        full_pos * float(cfg.preserve_fraction),
        torch.full_like(full_pos, float(cfg.margin_floor)),
    )
    # Never demand more margin than the full teacher itself supplies.
    gamma = torch.minimum(full_pos, desired).clamp_max(float(cfg.margin_cap))
    rival_mask &= gamma > 0.0

    ref_rival_margin = ref_margin.gather(1, rival_idx)
    tau = max(float(cfg.rival_temperature), 1.0e-4)
    rival_logits = (-rival_margin / tau).masked_fill(~rival_mask, -1.0e9)
    rival_weight = torch.softmax(rival_logits, dim=1) * rival_mask.float()
    rival_weight = rival_weight / rival_weight.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

    # Pair contribution d_i(w,b) = g_i(b)-g_i(w), normalized per scene.
    winner_g = tg.gather(2, winner_idx[:, None, None].expand(B, E, 1)).squeeze(2)
    rival_g = tg.gather(2, rival_idx[:, None, :].expand(B, E, r))
    atom_delta = (rival_g - winner_g[:, :, None]) / scale[:, None, :]

    deficit = torch.relu(gamma - ref_rival_margin)  # [B,R]
    add_deficit = torch.relu(gamma[:, None, :] - (ref_rival_margin[:, None, :] + atom_delta))
    remove_deficit = torch.relu(gamma[:, None, :] - (ref_rival_margin[:, None, :] - atom_delta))
    add_gain = torch.relu(deficit[:, None, :] - add_deficit)
    removal_loss = torch.relu(remove_deficit - deficit[:, None, :])
    weighted_add = (add_gain * rival_weight[:, None, :]).sum(dim=2)
    weighted_removal = (removal_loss * rival_weight[:, None, :]).sum(dim=2)

    # A missed atom is valued only through a budget-feasible intervention.  If
    # there is true slack, S_B U {i} is feasible.  Otherwise use the best
    # one-for-one exchange S_B - {j} U {i}; no B+1 target is ever produced.
    # The [B,E,E,R] tensor is small in this interface (E is the auditable atom
    # bank, R<=4 by default) and replaces much heavier legacy pair objectives.
    finite_cost = torch.isfinite(costs) & (costs > 0.0)
    spent = torch.where(ref & finite_cost, costs, torch.zeros_like(costs)).sum(dim=1)
    budget = max(float(cfg.budget), 0.0)
    budget_eps = 1.0e-6
    outside = active & ~ref & finite_cost
    slack_feasible = outside & ((spent[:, None] + costs) <= budget + budget_eps)

    exchange_margin = (
        ref_rival_margin[:, None, None, :]
        + atom_delta[:, :, None, :]
        - atom_delta[:, None, :, :]
    )
    exchange_deficit = torch.relu(gamma[:, None, None, :] - exchange_margin)
    exchange_gain = torch.relu(deficit[:, None, None, :] - exchange_deficit)
    weighted_exchange = (exchange_gain * rival_weight[:, None, None, :]).sum(dim=3)
    exchange_feasible = (
        outside[:, :, None]
        & ref[:, None, :]
        & finite_cost[:, None, :]
        & ((spent[:, None, None] - costs[:, None, :] + costs[:, :, None]) <= budget + budget_eps)
    )
    neg_inf = torch.full_like(weighted_exchange, -1.0e9)
    best_exchange = torch.where(exchange_feasible, weighted_exchange, neg_inf).max(dim=2).values.clamp_min(0.0)
    outside_value = torch.where(slack_feasible, weighted_add, best_exchange)
    local_value = torch.where(ref, weighted_removal, outside_value)
    utility = local_value / costs.clamp_min(float(cfg.min_atom_cost)).pow(float(cfg.cost_power))
    utility = torch.where(active & aligned[:, None], utility.clamp_min(0.0), torch.zeros_like(utility))

    total_utility = utility.sum(dim=1)
    scene_has_utility = aligned & (total_utility > float(cfg.utility_epsilon))
    weighted_deficit = (deficit * rival_weight).sum(dim=1)
    selected_utility = (utility * ref.float()).sum(dim=1)
    missed_utility = (utility * (~ref).float()).sum(dim=1)
    positive_fraction = ((utility > float(cfg.utility_epsilon)) & active).float().sum(dim=1) / active.float().sum(dim=1).clamp_min(1.0)
    details = {
        "aligned": aligned,
        "scene_has_utility": scene_has_utility,
        "total_utility": total_utility,
        "selected_utility": selected_utility,
        "missed_utility": missed_utility,
        "weighted_deficit": weighted_deficit,
        "positive_fraction": positive_fraction,
        "reference_count": ref.float().sum(dim=1),
    }
    return utility, details


def budgeted_decisive_margin_utility_numpy(
    teacher_cost: np.ndarray,
    teacher_g: np.ndarray,
    active: np.ndarray,
    valid: np.ndarray,
    teacher_action: int,
    reference_mask: np.ndarray,
    atom_costs: np.ndarray,
    cfg: BDMUConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """NumPy mirror used by open-loop diagnostics."""
    tc = np.asarray(teacher_cost, dtype=np.float64).reshape(-1)
    tg = np.asarray(teacher_g, dtype=np.float64)
    if tg.ndim != 2:
        raise ValueError(f"teacher_g must be [E,K], got {tg.shape}")
    E, K = tg.shape
    if tc.shape[0] != K:
        raise ValueError("teacher_cost and teacher_g K mismatch")
    active = np.asarray(active, dtype=bool).reshape(-1)[:E]
    if active.shape[0] < E:
        active = np.pad(active, (0, E - active.shape[0]), constant_values=False)
    valid = np.asarray(valid, dtype=bool).reshape(-1)[:K]
    if valid.shape[0] < K:
        valid = np.pad(valid, (0, K - valid.shape[0]), constant_values=False)
    ref = np.asarray(reference_mask, dtype=bool).reshape(-1)[:E]
    if ref.shape[0] < E:
        ref = np.pad(ref, (0, E - ref.shape[0]), constant_values=False)
    ref &= active
    costs = np.asarray(atom_costs, dtype=np.float64).reshape(-1)[:E]
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)
    costs = np.maximum(costs, float(cfg.min_atom_cost))

    if not (0 <= int(teacher_action) < K and valid[int(teacher_action)]):
        return np.zeros((E,), dtype=np.float32), {"aligned": 0.0}
    dense = np.where(valid, tc, np.inf)
    aligned = bool(np.isfinite(dense).any() and int(np.argmin(dense)) == int(teacher_action))
    if not aligned:
        return np.zeros((E,), dtype=np.float32), {"aligned": 0.0}

    vals = tc[valid & np.isfinite(tc)]
    scale = max(float(np.std(vals)) if vals.size else 0.0, float(cfg.min_action_scale))
    base = tc - (tg * active[:, None]).sum(axis=0)
    ref_cost = base + (tg * ref[:, None]).sum(axis=0)
    w = int(teacher_action)
    full_margin = (tc - tc[w]) / scale
    ref_margin = (ref_cost - ref_cost[w]) / scale
    rivals = np.flatnonzero(valid & (np.arange(K) != w) & (full_margin > 0.0))
    if rivals.size == 0:
        return np.zeros((E,), dtype=np.float32), {"aligned": 1.0, "scene_has_utility": 0.0}
    rivals = rivals[np.argsort(full_margin[rivals], kind="stable")[: max(1, int(cfg.rival_count))]]
    fm = full_margin[rivals]
    gamma = np.minimum(
        fm,
        np.maximum(float(cfg.preserve_fraction) * fm, float(cfg.margin_floor)),
    )
    gamma = np.minimum(gamma, float(cfg.margin_cap))
    keep = gamma > 0.0
    rivals, fm, gamma = rivals[keep], fm[keep], gamma[keep]
    if rivals.size == 0:
        return np.zeros((E,), dtype=np.float32), {"aligned": 1.0, "scene_has_utility": 0.0}
    z = -fm / max(float(cfg.rival_temperature), 1.0e-4)
    z -= np.max(z)
    rw = np.exp(z)
    rw /= max(float(rw.sum()), 1.0e-12)
    rm = ref_margin[rivals]
    deficit = np.maximum(gamma - rm, 0.0)
    delta = (tg[:, rivals] - tg[:, [w]]) / scale
    add_deficit = np.maximum(gamma[None, :] - (rm[None, :] + delta), 0.0)
    remove_deficit = np.maximum(gamma[None, :] - (rm[None, :] - delta), 0.0)
    add_gain = np.maximum(deficit[None, :] - add_deficit, 0.0)
    removal_loss = np.maximum(remove_deficit - deficit[None, :], 0.0)
    weighted_add = (add_gain * rw[None, :]).sum(axis=1)
    weighted_removal = (removal_loss * rw[None, :]).sum(axis=1)

    finite_cost = np.isfinite(costs) & (costs > 0.0)
    spent = float(np.where(ref & finite_cost, costs, 0.0).sum())
    budget = max(float(cfg.budget), 0.0)
    outside_value = np.zeros((E,), dtype=np.float64)
    for i in np.flatnonzero(active & ~ref & finite_cost).tolist():
        if spent + float(costs[i]) <= budget + 1.0e-6:
            outside_value[i] = weighted_add[i]
            continue
        best = 0.0
        for j in np.flatnonzero(ref & finite_cost).tolist():
            if spent - float(costs[j]) + float(costs[i]) > budget + 1.0e-6:
                continue
            exchange_margin = rm + delta[i] - delta[j]
            exchange_deficit = np.maximum(gamma - exchange_margin, 0.0)
            exchange_gain = np.maximum(deficit - exchange_deficit, 0.0)
            best = max(best, float((exchange_gain * rw).sum()))
        outside_value[i] = best
    local = np.where(ref, weighted_removal, outside_value)
    utility = local / np.power(costs, float(cfg.cost_power))
    utility = np.where(active, np.maximum(utility, 0.0), 0.0)
    total = float(utility.sum())
    details = {
        "aligned": 1.0,
        "scene_has_utility": float(total > float(cfg.utility_epsilon)),
        "total_utility": total,
        "selected_utility": float(utility[ref].sum()),
        "missed_utility": float(utility[active & ~ref].sum()),
        "weighted_deficit": float(np.sum(deficit * rw)),
        "positive_fraction": float(np.mean(utility[active] > float(cfg.utility_epsilon))) if active.any() else 0.0,
        "reference_count": float(ref.sum()),
    }
    return utility.astype(np.float32), details
