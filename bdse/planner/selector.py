from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bdse.planner.pair_screen import build_runtime_pairs_from_base


@dataclass(slots=True)
class SelectionResult:
    selected: list[int]
    objective_value: float
    pair_indices: np.ndarray
    pair_weights: np.ndarray
    diagnostics: dict[str, Any]


def _finite_cost_for_margin(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float32).copy()
    finite = np.isfinite(cost)
    if finite.any():
        fill = float(np.nanmax(cost[finite]) + 1e6)
    else:
        fill = 1e6
    cost[~finite] = fill
    return cost


def full_interface_margin(J0: np.ndarray, g: np.ndarray) -> np.ndarray:
    J0 = np.asarray(J0, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    full_cost = _finite_cost_for_margin(J0 + g.sum(axis=0))
    return full_cost[None, :] - full_cost[:, None]


def budgeted_margin(J0: np.ndarray, g: np.ndarray, selected: list[int] | np.ndarray) -> np.ndarray:
    J0 = np.asarray(J0, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    if len(selected):
        cost = J0 + g[np.asarray(selected, dtype=np.int64)].sum(axis=0)
    else:
        cost = J0.copy()
    cost = _finite_cost_for_margin(cost)
    return cost[None, :] - cost[:, None]


def oracle_objective_value(
    selected: list[int] | np.ndarray,
    J_base: np.ndarray,
    g_true: np.ndarray,
    pairs: np.ndarray,
    margins: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Signed certificate objective for labeled teacher pairs.

    For pair (a, b), positive margin means action a should beat b.  The
    certificate score accumulates the signed margin contribution J(b)-J(a).
    Clipping every atom delta before summing can overestimate evidence because
    an atom that supports one pair may hurt another pair.
    """
    selected_arr = np.asarray(selected, dtype=np.int64)
    pair_arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pair_arr.size == 0:
        return 0.0
    caps = np.asarray(margins, dtype=np.float32).reshape(-1)[: pair_arr.shape[0]]
    w = np.asarray(weights, dtype=np.float32).reshape(-1)[: pair_arr.shape[0]]
    if caps.shape[0] != pair_arr.shape[0]:
        caps = np.zeros((pair_arr.shape[0],), dtype=np.float32)
    if w.shape[0] != pair_arr.shape[0]:
        w = np.ones((pair_arr.shape[0],), dtype=np.float32)
    a = pair_arr[:, 0]
    b = pair_arr[:, 1]
    margin = np.asarray(J_base, dtype=np.float32)[b] - np.asarray(J_base, dtype=np.float32)[a]
    if selected_arr.size:
        valid_sel = selected_arr[(selected_arr >= 0) & (selected_arr < np.asarray(g_true).shape[0])]
        if valid_sel.size:
            g = np.asarray(g_true, dtype=np.float32)
            margin = margin + (g[valid_sel[:, None], b[None, :]] - g[valid_sel[:, None], a[None, :]]).sum(axis=0)
    cert = np.minimum(caps, np.maximum(margin, 0.0))
    return float(np.sum(w * cert, dtype=np.float64))



def _greedy_cover_from_pair_delta(
    atom_delta: np.ndarray,
    base_margin: np.ndarray,
    caps: np.ndarray,
    pair_weights: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    atom_active_mask: np.ndarray | None = None,
) -> tuple[list[int], float, float]:
    """Greedy signed certificate coverage over pair margins.

    ``base_margin[p]`` and ``atom_delta[i, p]`` are signed contributions to
    J(b)-J(a) for pair p.  Value is sum_p w_p min(cap_p, max(margin_p, 0)).
    """
    margin = np.asarray(base_margin, dtype=np.float32).reshape(-1).copy()
    caps = np.asarray(caps, dtype=np.float32).reshape(-1)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    atom_delta = np.asarray(atom_delta, dtype=np.float32)
    E = int(atom_delta.shape[0]) if atom_delta.ndim == 2 else int(np.asarray(atom_budget_costs).shape[0])
    if atom_delta.ndim != 2 or atom_delta.shape[1] != margin.shape[0]:
        atom_delta = np.zeros((E, margin.shape[0]), dtype=np.float32)
    if caps.shape[0] != margin.shape[0]:
        caps = np.zeros((margin.shape[0],), dtype=np.float32)
    if weights.shape[0] != margin.shape[0]:
        weights = np.ones((margin.shape[0],), dtype=np.float32)
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else np.asarray(atom_active_mask, dtype=bool).reshape(-1).copy()
    if active.shape[0] < E:
        active = np.pad(active, (0, E - active.shape[0]), constant_values=False)
    active = active[:E]
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)

    def value(m: np.ndarray) -> float:
        return float(np.sum(weights * np.minimum(caps, np.maximum(m, 0.0)), dtype=np.float64))

    current = value(margin)
    selected: list[int] = []
    spent = 0.0
    while bool(active.any()):
        best: tuple[int, float, float] | None = None
        best_key = (-np.inf, -np.inf, np.inf)
        for i in np.flatnonzero(active):
            idx = int(i)
            c = float(costs[idx])
            if not np.isfinite(c) or spent + c > float(budget) + 1e-6:
                continue
            gain = float(value(margin + atom_delta[idx]) - current)
            key = (gain / max(c, 1e-6), gain, -idx)
            if key > best_key:
                best_key = key
                best = (idx, gain, c)
        if best is None or best_key[1] <= 1e-9:
            break
        idx, gain, c = best
        selected.append(idx)
        active[idx] = False
        spent += c
        margin += atom_delta[idx]
        current += float(gain)
    return selected, float(current), float(spent)



def _greedy_cover_from_pair_support(
    atom_support: np.ndarray,
    base_support: np.ndarray,
    caps: np.ndarray,
    pair_weights: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    atom_active_mask: np.ndarray | None = None,
) -> tuple[list[int], float, float]:
    """Greedy coverage used by oracle/runtime selectors, vectorized over pairs.

    ``atom_support[i, p]`` is the non-negative margin support contributed by atom
    ``i`` to pair ``p``.  This is algebraically identical to recomputing
    ``oracle_objective_value`` / ``runtime_objective_value`` for every candidate
    atom at every greedy step, but it avoids the Python loop over pairs inside the
    inner loop.  Tie-breaking intentionally mirrors the old implementation:
    maximize ``(gain / cost, gain, -atom_index)`` over sorted active atoms.
    """
    support = np.asarray(base_support, dtype=np.float32).reshape(-1).copy()
    caps = np.asarray(caps, dtype=np.float32).reshape(-1)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    atom_support = np.asarray(atom_support, dtype=np.float32)
    E = int(atom_support.shape[0]) if atom_support.ndim == 2 else int(np.asarray(atom_budget_costs).shape[0])
    if atom_support.ndim != 2 or atom_support.shape[1] != support.shape[0]:
        atom_support = np.zeros((E, support.shape[0]), dtype=np.float32)
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else np.asarray(atom_active_mask, dtype=bool).copy()
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)
    selected: list[int] = []
    spent = 0.0
    clipped = np.minimum(caps, support)
    current = float(np.sum(weights * clipped, dtype=np.float64))

    while bool(active.any()):
        best: tuple[int, float, float] | None = None
        best_key = (-np.inf, -np.inf, np.inf)
        baseline = np.minimum(caps, support)
        for i in np.flatnonzero(active):
            idx = int(i)
            c = float(costs[idx])
            if spent + c > float(budget) + 1e-6:
                continue
            trial = np.minimum(caps, support + atom_support[idx])
            gain = float(np.sum(weights * (trial - baseline), dtype=np.float64))
            key = (gain / max(c, 1e-6), gain, -idx)
            if key > best_key:
                best_key = key
                best = (idx, gain, c)
        if best is None or best_key[1] <= 1e-9:
            break
        idx, gain, c = best
        selected.append(idx)
        active[idx] = False
        spent += c
        support += atom_support[idx]
        current += float(gain)
    return selected, float(current), float(spent)


def _family_constraint_ok(
    idx: int,
    cost: float,
    spent_by_family: dict[int, float],
    family_ids: np.ndarray | None,
    family_budget_caps: np.ndarray | None,
) -> bool:
    if family_ids is None or family_budget_caps is None:
        return True
    fam = np.asarray(family_ids, dtype=np.int64).reshape(-1)
    if idx >= fam.shape[0]:
        return True
    f = int(fam[idx])
    caps = np.asarray(family_budget_caps, dtype=np.float32).reshape(-1)
    if f < 0 or f >= caps.shape[0] or not np.isfinite(caps[f]) or caps[f] <= 0.0:
        return True
    return spent_by_family.get(f, 0.0) + float(cost) <= float(caps[f]) + 1e-6


def _uncertainty_aware_greedy_from_pair_delta(
    atom_delta: np.ndarray,
    base_margin: np.ndarray,
    atom_pair_var: np.ndarray,
    base_pair_var: np.ndarray,
    caps: np.ndarray,
    pair_weights: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    atom_active_mask: np.ndarray | None = None,
    beta_uncertainty: float = 0.0,
    epsilon_cal: float = 0.0,
    lambda_info: float = 0.0,
    info_caps: np.ndarray | None = None,
    prior_atom_pair_var: np.ndarray | None = None,
    family_ids: np.ndarray | None = None,
    family_budget_caps: np.ndarray | None = None,
) -> tuple[list[int], float, float, dict[str, Any]]:
    """Greedy Eq. (HAB objective) over signed margins and uncertainty."""
    delta = np.asarray(atom_delta, dtype=np.float32)
    mu = np.asarray(base_margin, dtype=np.float32).reshape(-1).copy()
    var = np.asarray(base_pair_var, dtype=np.float32).reshape(-1).copy()
    if delta.ndim != 2 or delta.shape[1] != mu.shape[0]:
        E = int(np.asarray(atom_budget_costs).shape[0])
        delta = np.zeros((E, mu.shape[0]), dtype=np.float32)
    else:
        E = int(delta.shape[0])
    atom_var = np.asarray(atom_pair_var, dtype=np.float32)
    if atom_var.ndim != 2 or atom_var.shape != delta.shape:
        atom_var = np.zeros_like(delta, dtype=np.float32)
    caps = np.asarray(caps, dtype=np.float32).reshape(-1)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    if caps.shape[0] != mu.shape[0]:
        caps = np.zeros((mu.shape[0],), dtype=np.float32)
    if weights.shape[0] != mu.shape[0]:
        weights = np.ones((mu.shape[0],), dtype=np.float32)
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else np.asarray(atom_active_mask, dtype=bool).reshape(-1).copy()
    if active.shape[0] < E:
        active = np.pad(active, (0, E - active.shape[0]), constant_values=False)
    active = active[:E]
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)

    beta = max(float(beta_uncertainty), 0.0)
    eps_cal = max(float(epsilon_cal), 0.0)
    lam_info = max(float(lambda_info), 0.0)
    if info_caps is None:
        info_caps_arr = np.ones_like(mu, dtype=np.float32)
    else:
        info_caps_arr = np.asarray(info_caps, dtype=np.float32).reshape(-1)
        if info_caps_arr.shape[0] != mu.shape[0]:
            info_caps_arr = np.ones_like(mu, dtype=np.float32)
    prior_var = np.asarray(prior_atom_pair_var, dtype=np.float32) if prior_atom_pair_var is not None else None
    if prior_var is None or prior_var.shape != atom_var.shape:
        prior_var = (np.maximum(atom_var, 0.0) + 1.0).astype(np.float32)

    def objective(m: np.ndarray, u: np.ndarray, info_accum: np.ndarray) -> float:
        lcb = m - beta * np.sqrt(np.maximum(u, 0.0)) - eps_cal
        cert = np.minimum(caps, np.maximum(lcb, 0.0))
        if lam_info > 0:
            info = np.minimum(info_caps_arr, np.maximum(info_accum, 0.0))
            val = cert + lam_info * info
        else:
            val = cert
        return float(np.sum(weights * val, dtype=np.float64))

    info_state = np.zeros_like(mu, dtype=np.float32)
    current = objective(mu, var, info_state)
    selected: list[int] = []
    spent = 0.0
    spent_by_family: dict[int, float] = {}
    family_arr = np.asarray(family_ids, dtype=np.int64).reshape(-1) if family_ids is not None else None
    while bool(active.any()):
        best: tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
        best_key = (-np.inf, -np.inf, np.inf)
        for i in np.flatnonzero(active):
            idx = int(i)
            c = float(costs[idx])
            if not np.isfinite(c) or spent + c > float(budget) + 1e-6:
                continue
            if not _family_constraint_ok(idx, c, spent_by_family, family_arr, family_budget_caps):
                continue
            new_mu = mu + delta[idx]
            new_var = var + np.maximum(atom_var[idx], 0.0)
            new_info = info_state + np.maximum(0.0, prior_var[idx] - atom_var[idx])
            val = objective(new_mu, new_var, new_info)
            gain = float(val - current)
            key = (gain / max(c, 1e-6), gain, -idx)
            if key > best_key:
                best_key = key
                best = (idx, gain, c, new_mu, new_var, new_info)
        if best is None or best_key[1] <= 1e-9:
            break
        idx, gain, c, mu, var, info_state = best
        selected.append(idx)
        active[idx] = False
        spent += c
        if family_arr is not None and idx < family_arr.shape[0]:
            f = int(family_arr[idx])
            spent_by_family[f] = spent_by_family.get(f, 0.0) + c
        current += float(gain)
    diagnostics = {
        "spent_budget": float(spent),
        "family_spent": {int(k): float(v) for k, v in spent_by_family.items()},
        "beta_uncertainty": float(beta),
        "epsilon_cal": float(eps_cal),
        "lambda_info": float(lam_info),
    }
    return selected, float(current), float(spent), diagnostics


def oracle_greedy_selector(
    J_base: np.ndarray,
    g_true: np.ndarray,
    pairs: np.ndarray,
    margins: np.ndarray,
    weights: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    atom_active_mask: np.ndarray | None = None,
) -> SelectionResult:
    pair_arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    E = int(np.asarray(g_true).shape[0])
    if pair_arr.size == 0:
        selected, current, spent = _greedy_cover_from_pair_support(
            np.zeros((E, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            atom_budget_costs,
            budget,
            atom_active_mask,
        )
    else:
        a = pair_arr[:, 0]
        b = pair_arr[:, 1]
        base_delta = np.asarray(J_base, dtype=np.float32)[b] - np.asarray(J_base, dtype=np.float32)[a]
        atom_delta = np.asarray(g_true, dtype=np.float32)[:, b] - np.asarray(g_true, dtype=np.float32)[:, a]
        selected, current, spent = _greedy_cover_from_pair_delta(
            atom_delta,
            base_delta,
            np.asarray(margins, dtype=np.float32).reshape(-1)[: pair_arr.shape[0]],
            np.asarray(weights, dtype=np.float32).reshape(-1)[: pair_arr.shape[0]],
            atom_budget_costs,
            budget,
            atom_active_mask,
        )
    return SelectionResult(
        selected=selected,
        objective_value=current,
        pair_indices=pair_arr,
        pair_weights=np.asarray(weights, dtype=np.float32),
        diagnostics={"spent_budget": spent, "budget": float(budget), "mode": "oracle_greedy"},
    )


def build_predicted_pairs(
    predicted_full_margin: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    L_infer: int,
    eta_pred: float,
) -> tuple[np.ndarray, np.ndarray]:
    M = np.asarray(predicted_full_margin, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    K = len(valid)
    total_cost_proxy = -M.mean(axis=1)
    pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    for a in range(K):
        if not valid[a]:
            continue
        candidates = [b for b in range(K) if b != a and valid[b]]
        ranked = sorted(candidates, key=lambda b: (float(total_cost_proxy[b]), abs(float(M[a, b])), int(b)))[:L_infer]
        near = [b for b in candidates if abs(float(M[a, b])) < eta_pred]
        safety = [b for b in candidates if runtime_safety_flags[b]]
        rivals = sorted(set(ranked + near + safety), key=lambda b: (abs(float(M[a, b])), b))[: max(L_infer, len(safety))]
        for b in rivals:
            if M[a, b] > 0:
                w = 1.0
                if abs(float(M[a, b])) < eta_pred:
                    w += 1.0
                if runtime_safety_flags[b]:
                    w += 2.0
                pairs.append((a, b))
                weights.append(w)
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return np.asarray(pairs, dtype=np.int64), np.asarray(weights, dtype=np.float32)


def runtime_objective_value(
    selected: list[int] | np.ndarray,
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    pair_indices: np.ndarray,
    pair_weights: np.ndarray,
    gamma_max: float,
) -> float:
    selected_arr = np.asarray(selected, dtype=np.int64)
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    if pair_arr.size == 0:
        return 0.0
    M_full = full_interface_margin(predicted_base_cost, predicted_atom_costs)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)[: pair_arr.shape[0]]
    if weights.shape[0] != pair_arr.shape[0]:
        weights = np.ones((pair_arr.shape[0],), dtype=np.float32)
    a = pair_arr[:, 0]
    b = pair_arr[:, 1]
    gamma = np.minimum(np.maximum(M_full[a, b], 0.0), float(gamma_max)).astype(np.float32)
    margin = np.asarray(predicted_base_cost, dtype=np.float32)[b] - np.asarray(predicted_base_cost, dtype=np.float32)[a]
    if selected_arr.size:
        valid_sel = selected_arr[(selected_arr >= 0) & (selected_arr < np.asarray(predicted_atom_costs).shape[0])]
        if valid_sel.size:
            g = np.asarray(predicted_atom_costs, dtype=np.float32)
            margin = margin + (g[valid_sel[:, None], b[None, :]] - g[valid_sel[:, None], a[None, :]]).sum(axis=0)
    cert = np.minimum(gamma, np.maximum(margin, 0.0))
    return float(np.sum(weights * cert, dtype=np.float64))


def full_prescore_greedy_selector(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    atom_budget_costs: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    budget: float,
    L_infer: int = 16,
    gamma_max: float = 100.0,
    eta_pred: float = 1.0,
    lambda_near: float = 1.0,
    lambda_safety: float = 2.0,
    atom_active_mask: np.ndarray | None = None,
) -> SelectionResult:
    """Legacy full-interface selector kept only for ablations."""
    M = full_interface_margin(predicted_base_cost, predicted_atom_costs)
    pairs, base_weights = build_predicted_pairs(M, valid_mask, runtime_safety_flags, L_infer, eta_pred)
    if len(base_weights):
        pair_weights = np.ones_like(base_weights)
        for i, (a, b) in enumerate(pairs):
            pair_weights[i] += float(lambda_near) * float(abs(M[a, b]) < eta_pred)
            pair_weights[i] += float(lambda_safety) * float(runtime_safety_flags[b])
    else:
        pair_weights = base_weights

    E = int(predicted_atom_costs.shape[0])
    if len(pairs):
        a = pairs[:, 0]
        b = pairs[:, 1]
        base_delta = np.asarray(predicted_base_cost, dtype=np.float32)[b] - np.asarray(predicted_base_cost, dtype=np.float32)[a]
        atom_delta = np.asarray(predicted_atom_costs, dtype=np.float32)[:, b] - np.asarray(predicted_atom_costs, dtype=np.float32)[:, a]
        caps = np.minimum(np.maximum(M[a, b], 0.0), float(gamma_max)).astype(np.float32)
    else:
        base_delta = np.zeros((0,), dtype=np.float32)
        atom_delta = np.zeros((E, 0), dtype=np.float32)
        caps = np.zeros((0,), dtype=np.float32)
    selected, current, spent = _greedy_cover_from_pair_delta(
        atom_delta, base_delta, caps, pair_weights, atom_budget_costs, budget, atom_active_mask
    )
    return SelectionResult(
        selected=selected,
        objective_value=current,
        pair_indices=pairs,
        pair_weights=pair_weights,
        diagnostics={"spent_budget": spent, "budget": float(budget), "mode": "legacy_full_prescore", "pair_count": int(len(pairs))},
    )


def runtime_greedy_selector(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    atom_budget_costs: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    budget: float,
    L_infer: int = 16,
    gamma_max: float = 100.0,
    eta_pred: float = 1.0,
    lambda_near: float = 1.0,
    lambda_safety: float = 2.0,
    atom_active_mask: np.ndarray | None = None,
    predicted_atom_variance: np.ndarray | None = None,
    base_pair_variance: np.ndarray | None = None,
    beta_uncertainty: float = 0.0,
    epsilon_cal: float = 0.0,
    lambda_info: float = 0.0,
    prior_atom_variance: float | np.ndarray | None = None,
    family_ids: np.ndarray | None = None,
    family_budget_caps: np.ndarray | None = None,
    bidirectional_pairs: bool = True,
    reverse_pair_weight: float = 1.0,
    pair_cap_multiplier: float = 1.0,
) -> SelectionResult:
    """Two-stage runtime selector using base/cheap pair screening only.

    Unlike the legacy implementation, this function does not construct
    full_interface_margin before selecting evidence.  It assumes
    ``predicted_atom_costs`` contains only already-queried sparse values (zeros
    elsewhere) and restricts the active atoms through ``atom_active_mask``.
    """
    pairs, pair_weights = build_runtime_pairs_from_base(
        predicted_base_cost,
        valid_mask,
        runtime_safety_flags,
        L0=L_infer,
        eta0=eta_pred,
        lambda_near=lambda_near,
        lambda_safety=lambda_safety,
        bidirectional_pairs=bidirectional_pairs,
        reverse_pair_weight=reverse_pair_weight,
        pair_cap_multiplier=pair_cap_multiplier,
    )
    E = int(predicted_atom_costs.shape[0])
    if len(pairs):
        a = pairs[:, 0]
        b = pairs[:, 1]
        base_delta = np.asarray(predicted_base_cost, dtype=np.float32)[b] - np.asarray(predicted_base_cost, dtype=np.float32)[a]
        safety_b = np.asarray(runtime_safety_flags, dtype=bool)[b] if np.asarray(runtime_safety_flags).shape[0] > int(np.max(b, initial=0)) else np.zeros_like(base_delta, dtype=bool)
        caps = np.where(
            safety_b,
            float(gamma_max),
            np.minimum(np.maximum(np.abs(base_delta) + float(eta_pred), 1e-3), float(gamma_max)),
        ).astype(np.float32)
        atom_delta = np.asarray(predicted_atom_costs, dtype=np.float32)[:, b] - np.asarray(predicted_atom_costs, dtype=np.float32)[:, a]
        base_support = np.maximum(base_delta, 0.0)
        atom_support = np.maximum(atom_delta, 0.0)
        if predicted_atom_variance is not None:
            av = np.asarray(predicted_atom_variance, dtype=np.float32)
            if av.ndim == 2 and av.shape[1] >= max(int(np.max(b, initial=0)), int(np.max(a, initial=0))) + 1:
                atom_pair_var = np.maximum(av[:, a], 0.0) + np.maximum(av[:, b], 0.0)
            elif av.ndim == 2 and av.shape == atom_delta.shape:
                atom_pair_var = np.maximum(av, 0.0)
            else:
                atom_pair_var = np.zeros_like(atom_delta, dtype=np.float32)
        else:
            atom_pair_var = np.zeros_like(atom_delta, dtype=np.float32)
        if base_pair_variance is not None:
            bpv = np.asarray(base_pair_variance, dtype=np.float32)
            if bpv.ndim == 1 and bpv.shape[0] == len(pairs):
                base_var = np.maximum(bpv, 0.0)
            elif bpv.ndim == 2 and bpv.shape[0] >= len(pairs) and bpv.shape[1] >= 1:
                base_var = np.maximum(bpv[: len(pairs), 0], 0.0)
            else:
                base_var = np.zeros((len(pairs),), dtype=np.float32)
        else:
            base_var = np.zeros((len(pairs),), dtype=np.float32)
        info_caps = np.where(safety_b, float(gamma_max), np.maximum(float(eta_pred), np.abs(base_delta))).astype(np.float32)
    else:
        base_delta = np.zeros((0,), dtype=np.float32)
        base_support = np.zeros((0,), dtype=np.float32)
        atom_support = np.zeros((E, 0), dtype=np.float32)
        atom_delta = np.zeros((E, 0), dtype=np.float32)
        atom_pair_var = np.zeros((E, 0), dtype=np.float32)
        base_var = np.zeros((0,), dtype=np.float32)
        caps = np.zeros((0,), dtype=np.float32)
        info_caps = np.zeros((0,), dtype=np.float32)
    use_uncertainty_objective = (
        predicted_atom_variance is not None
        or abs(float(beta_uncertainty)) > 0.0
        or abs(float(epsilon_cal)) > 0.0
        or abs(float(lambda_info)) > 0.0
        or family_budget_caps is not None
    )
    if use_uncertainty_objective:
        if isinstance(prior_atom_variance, np.ndarray):
            prior = np.asarray(prior_atom_variance, dtype=np.float32)
            if len(pairs) and prior.ndim == 2 and prior.shape[1] >= max(int(np.max(pairs[:, 0], initial=0)), int(np.max(pairs[:, 1], initial=0))) + 1:
                prior_pair = np.maximum(prior[:, pairs[:, 0]], 0.0) + np.maximum(prior[:, pairs[:, 1]], 0.0)
            elif prior.shape == atom_pair_var.shape:
                prior_pair = prior
            else:
                prior_pair = None
        elif prior_atom_variance is not None:
            prior_pair = np.full_like(atom_pair_var, float(prior_atom_variance), dtype=np.float32)
        else:
            prior_pair = None
        selected, current, spent, extra_diag = _uncertainty_aware_greedy_from_pair_delta(
            atom_delta,
            base_delta,
            atom_pair_var,
            base_var,
            caps,
            pair_weights,
            atom_budget_costs,
            budget,
            atom_active_mask,
            beta_uncertainty=beta_uncertainty,
            epsilon_cal=epsilon_cal,
            lambda_info=lambda_info,
            info_caps=info_caps,
            prior_atom_pair_var=prior_pair,
            family_ids=family_ids,
            family_budget_caps=family_budget_caps,
        )
        mode = "runtime_hab_lcb_uncertainty"
    else:
        selected, current, spent = _greedy_cover_from_pair_delta(
            atom_delta,
            base_delta,
            caps,
            pair_weights,
            atom_budget_costs,
            budget,
            atom_active_mask,
        )
        extra_diag = {}
        mode = "runtime_base_screen_sparse_signed"
    return SelectionResult(
        selected=selected,
        objective_value=current,
        pair_indices=pairs,
        pair_weights=pair_weights,
        diagnostics={"spent_budget": spent, "budget": float(budget), "mode": mode, "pair_count": int(len(pairs)), **extra_diag},
    )


def runtime_greedy_selector_pair_conditioned(
    predicted_base_cost: np.ndarray,
    pair_atom_delta: np.ndarray,
    pair_indices: np.ndarray,
    pair_weights: np.ndarray,
    atom_budget_costs: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    budget: float,
    gamma_max: float = 100.0,
    eta_pred: float = 1.0,
    atom_active_mask: np.ndarray | None = None,
    pair_atom_variance: np.ndarray | None = None,
    beta_uncertainty: float = 0.0,
    epsilon_cal: float = 0.0,
    lambda_info: float = 0.0,
    prior_atom_variance: float | np.ndarray | None = None,
    family_ids: np.ndarray | None = None,
    family_budget_caps: np.ndarray | None = None,
) -> SelectionResult:
    """Runtime greedy selector over pair-conditioned atom deltas.

    ``pair_atom_delta[i,p]`` is the signed evidence margin contribution
    d_i(a,b) for ``pair_indices[p] = (a,b)``.  This implements the paper's
    pair-conditioned scorer while keeping the same greedy coverage objective.
    """
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    if weights.shape[0] < pair_arr.shape[0]:
        weights = np.pad(weights, (0, pair_arr.shape[0] - weights.shape[0]), constant_values=1.0)
    weights = weights[: pair_arr.shape[0]]
    E = int(np.asarray(pair_atom_delta).shape[0]) if np.asarray(pair_atom_delta).ndim == 2 else int(np.asarray(atom_budget_costs).shape[0])
    if pair_arr.size == 0:
        selected, current, spent = _greedy_cover_from_pair_support(
            np.zeros((E, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            atom_budget_costs,
            budget,
            atom_active_mask,
        )
        return SelectionResult(selected, current, pair_arr, weights, {"spent_budget": spent, "budget": float(budget), "mode": "runtime_pair_conditioned_empty", "pair_count": 0})
    a = pair_arr[:, 0]
    b = pair_arr[:, 1]
    J0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    base_delta = J0[b] - J0[a]
    flags = np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)
    safety_b = flags[b] if flags.shape[0] > int(np.max(b, initial=0)) else np.zeros_like(base_delta, dtype=bool)
    caps = np.where(
        safety_b,
        float(gamma_max),
        np.minimum(np.maximum(np.abs(base_delta) + float(eta_pred), 1e-3), float(gamma_max)),
    ).astype(np.float32)
    delta = np.asarray(pair_atom_delta, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[1] != pair_arr.shape[0]:
        delta = np.zeros((E, pair_arr.shape[0]), dtype=np.float32)
    base_support = np.maximum(base_delta, 0.0).astype(np.float32)
    atom_support = np.maximum(delta, 0.0).astype(np.float32)  # legacy/debug only

    use_uncertainty_objective = (
        pair_atom_variance is not None
        or abs(float(beta_uncertainty)) > 0.0
        or abs(float(epsilon_cal)) > 0.0
        or abs(float(lambda_info)) > 0.0
        or family_budget_caps is not None
    )
    if use_uncertainty_objective:
        pair_var = np.asarray(pair_atom_variance, dtype=np.float32) if pair_atom_variance is not None else np.zeros_like(delta, dtype=np.float32)
        if pair_var.shape != delta.shape:
            pair_var = np.zeros_like(delta, dtype=np.float32)
        base_var = np.zeros((pair_arr.shape[0],), dtype=np.float32)
        if isinstance(prior_atom_variance, np.ndarray):
            prior = np.asarray(prior_atom_variance, dtype=np.float32)
            prior_pair = prior if prior.shape == delta.shape else None
        elif prior_atom_variance is not None:
            prior_pair = np.full_like(delta, float(prior_atom_variance), dtype=np.float32)
        else:
            prior_pair = None
        selected, current, spent, extra_diag = _uncertainty_aware_greedy_from_pair_delta(
            delta,
            base_delta,
            pair_var,
            base_var,
            caps,
            weights,
            atom_budget_costs,
            budget,
            atom_active_mask,
            beta_uncertainty=beta_uncertainty,
            epsilon_cal=epsilon_cal,
            lambda_info=lambda_info,
            info_caps=np.where(safety_b, float(gamma_max), np.maximum(float(eta_pred), np.abs(base_delta))).astype(np.float32),
            prior_atom_pair_var=prior_pair,
            family_ids=family_ids,
            family_budget_caps=family_budget_caps,
        )
        mode = "runtime_pair_conditioned_lcb_uncertainty"
    else:
        selected, current, spent = _greedy_cover_from_pair_delta(delta, base_delta, caps, weights, atom_budget_costs, budget, atom_active_mask)
        extra_diag = {}
        mode = "runtime_pair_conditioned_signed"
    return SelectionResult(
        selected=selected,
        objective_value=current,
        pair_indices=pair_arr,
        pair_weights=weights,
        diagnostics={"spent_budget": float(spent), "budget": float(budget), "mode": mode, "pair_count": int(len(pair_arr)), **extra_diag},
    )


def select_by_mode(
    mode: str,
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    atom_budget_costs: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    budget: float,
    atom_families: list[str] | None = None,
    seed: int = 17,
    **kwargs,
) -> SelectionResult:
    E = int(predicted_atom_costs.shape[0])
    costs = np.asarray(atom_budget_costs, dtype=np.float32)
    max_count = int(budget) if np.allclose(costs, 1.0) else E
    if mode == "full_prescore_ablation":
        return full_prescore_greedy_selector(
            predicted_base_cost,
            predicted_atom_costs,
            atom_budget_costs,
            valid_mask,
            runtime_safety_flags,
            budget,
            **kwargs,
        )
    if mode == "runtime_predicted":
        return runtime_greedy_selector(
            predicted_base_cost,
            predicted_atom_costs,
            atom_budget_costs,
            valid_mask,
            runtime_safety_flags,
            budget,
            **kwargs,
        )
    if mode == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(E).tolist()
    elif mode == "top_magnitude":
        magnitude = np.abs(predicted_atom_costs[:, valid_mask]).mean(axis=1)
        order = sorted(range(E), key=lambda i: (-float(magnitude[i]), i))
    elif mode == "diversity":
        fams = atom_families or ["all"] * E
        order = []
        for fam in sorted(set(fams)):
            order.extend([i for i, f in enumerate(fams) if f == fam])
    elif mode in {"interaction_only", "rule_map_only", "risk_only"}:
        fams = atom_families or ["all"] * E
        want = "interaction" if mode == "interaction_only" else "rule_map"
        if mode == "risk_only":
            magnitude = np.abs(predicted_atom_costs[:, valid_mask]).max(axis=1)
            order = sorted(range(E), key=lambda i: (-float(magnitude[i]), i))
        else:
            order = [i for i, f in enumerate(fams) if f == want]
    else:
        raise ValueError(f"Unknown selector mode: {mode}")
    selected: list[int] = []
    spent = 0.0
    for i in order:
        c = float(costs[i])
        if spent + c <= budget + 1e-6:
            selected.append(int(i))
            spent += c
        if len(selected) >= max_count and np.allclose(costs, 1.0):
            break
    return SelectionResult(selected=selected, objective_value=0.0, pair_indices=np.zeros((0, 2), dtype=np.int64), pair_weights=np.zeros((0,), dtype=np.float32), diagnostics={"spent_budget": spent, "mode": mode})
