from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
    E = int(g_true.shape[0])
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else np.asarray(atom_active_mask, dtype=bool)
    selected: list[int] = []
    remaining = {int(i) for i in np.flatnonzero(active)}
    spent = 0.0
    current = oracle_objective_value(selected, J_base, g_true, pairs, margins, weights)
    costs = np.asarray(atom_budget_costs, dtype=np.float32)
    while remaining:
        best = None
        best_key = (-np.inf, np.inf, np.inf)
        for i in sorted(remaining):
            c = float(costs[i])
            if spent + c > budget + 1e-6:
                continue
            val = oracle_objective_value(selected + [i], J_base, g_true, pairs, margins, weights)
            gain = val - current
            key = (gain / max(c, 1e-6), gain, -i)
            if key > best_key:
                best_key = key
                best = (i, val, c)
        if best is None or best_key[1] <= 1e-9:
            break
        i, val, c = best
        selected.append(i)
        remaining.remove(i)
        spent += c
        current = float(val)
    return SelectionResult(
        selected=selected,
        objective_value=current,
        pair_indices=np.asarray(pairs, dtype=np.int64),
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
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else np.asarray(atom_active_mask, dtype=bool)
    costs = np.asarray(atom_budget_costs, dtype=np.float32)
    selected: list[int] = []
    remaining = {int(i) for i in np.flatnonzero(active)}
    spent = 0.0
    current = runtime_objective_value(selected, predicted_base_cost, predicted_atom_costs, pairs, pair_weights, gamma_max)
    while remaining:
        best = None
        best_key = (-np.inf, -np.inf, np.inf)
        for i in sorted(remaining):
            c = float(costs[i])
            if spent + c > budget + 1e-6:
                continue
            val = runtime_objective_value(selected + [i], predicted_base_cost, predicted_atom_costs, pairs, pair_weights, gamma_max)
            gain = val - current
            key = (gain / max(c, 1e-6), gain, -i)
            if key > best_key:
                best_key = key
                best = (i, val, c)
        if best is None or best_key[1] <= 1e-9:
            break
        i, val, c = best
        selected.append(i)
        remaining.remove(i)
        spent += c
        current = float(val)
    return SelectionResult(
        selected=selected,
        objective_value=current,
        pair_indices=pairs,
        pair_weights=pair_weights,
        diagnostics={"spent_budget": spent, "budget": float(budget), "mode": "runtime_predicted", "pair_count": int(len(pairs))},
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
