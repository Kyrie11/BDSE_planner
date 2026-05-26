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
    selected_arr = np.asarray(selected, dtype=np.int64)
    total = 0.0
    for pidx, (a, b) in enumerate(np.asarray(pairs, dtype=np.int64)):
        cap = float(margins[pidx])
        base_support = max(float(J_base[b] - J_base[a]), 0.0)
        if selected_arr.size:
            evid_support = np.maximum(g_true[selected_arr, b] - g_true[selected_arr, a], 0.0).sum()
        else:
            evid_support = 0.0
        total += float(weights[pidx]) * min(cap, base_support + float(evid_support))
    return float(total)




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
        base_support = np.maximum(np.asarray(J_base, dtype=np.float32)[b] - np.asarray(J_base, dtype=np.float32)[a], 0.0)
        atom_support = np.maximum(np.asarray(g_true, dtype=np.float32)[:, b] - np.asarray(g_true, dtype=np.float32)[:, a], 0.0)
        selected, current, spent = _greedy_cover_from_pair_support(
            atom_support,
            base_support,
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
    M_full = full_interface_margin(predicted_base_cost, predicted_atom_costs)
    total = 0.0
    for pidx, (a, b) in enumerate(np.asarray(pair_indices, dtype=np.int64)):
        gamma = min(max(float(M_full[a, b]), 0.0), float(gamma_max))
        base_support = max(float(predicted_base_cost[b] - predicted_base_cost[a]), 0.0)
        if selected_arr.size:
            evid_support = np.maximum(predicted_atom_costs[selected_arr, b] - predicted_atom_costs[selected_arr, a], 0.0).sum()
        else:
            evid_support = 0.0
        total += float(pair_weights[pidx]) * min(gamma, base_support + float(evid_support))
    return float(total)


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
        base_support = np.maximum(np.asarray(predicted_base_cost, dtype=np.float32)[b] - np.asarray(predicted_base_cost, dtype=np.float32)[a], 0.0)
        atom_support = np.maximum(np.asarray(predicted_atom_costs, dtype=np.float32)[:, b] - np.asarray(predicted_atom_costs, dtype=np.float32)[:, a], 0.0)
        caps = np.minimum(np.maximum(M[a, b], 0.0), float(gamma_max)).astype(np.float32)
    else:
        base_support = np.zeros((0,), dtype=np.float32)
        atom_support = np.zeros((E, 0), dtype=np.float32)
        caps = np.zeros((0,), dtype=np.float32)
    selected, current, spent = _greedy_cover_from_pair_support(
        atom_support, base_support, caps, pair_weights, atom_budget_costs, budget, atom_active_mask
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
    )
    E = int(predicted_atom_costs.shape[0])
    if len(pairs):
        a = pairs[:, 0]
        b = pairs[:, 1]
        base_delta = np.asarray(predicted_base_cost, dtype=np.float32)[b] - np.asarray(predicted_base_cost, dtype=np.float32)[a]
        base_support = np.maximum(base_delta, 0.0)
        atom_support = np.maximum(np.asarray(predicted_atom_costs, dtype=np.float32)[:, b] - np.asarray(predicted_atom_costs, dtype=np.float32)[:, a], 0.0)
        safety_b = np.asarray(runtime_safety_flags, dtype=bool)[b] if np.asarray(runtime_safety_flags).shape[0] > int(np.max(b, initial=0)) else np.zeros_like(base_delta, dtype=bool)
        caps = np.where(
            safety_b,
            float(gamma_max),
            np.minimum(np.maximum(np.abs(base_delta) + float(eta_pred), 1e-3), float(gamma_max)),
        ).astype(np.float32)
    else:
        base_support = np.zeros((0,), dtype=np.float32)
        atom_support = np.zeros((E, 0), dtype=np.float32)
        caps = np.zeros((0,), dtype=np.float32)
    selected, current, spent = _greedy_cover_from_pair_support(
        atom_support,
        base_support,
        caps,
        pair_weights,
        atom_budget_costs,
        budget,
        atom_active_mask,
    )
    return SelectionResult(
        selected=selected,
        objective_value=current,
        pair_indices=pairs,
        pair_weights=pair_weights,
        diagnostics={"spent_budget": spent, "budget": float(budget), "mode": "runtime_base_screen_sparse", "pair_count": int(len(pairs))},
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
