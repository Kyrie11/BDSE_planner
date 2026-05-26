from __future__ import annotations

from typing import Any

import numpy as np

from bdse.planner.selector import SelectionResult, _greedy_cover_from_pair_support


def runtime_certificate_selector_sparse(
    predicted_base_cost: np.ndarray,
    sparse_atom_costs: np.ndarray,
    atom_indices: np.ndarray,
    pair_indices: np.ndarray,
    pair_weights: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    gamma_max: float = 100.0,
    atom_active_mask: np.ndarray | None = None,
) -> SelectionResult:
    J0 = np.asarray(predicted_base_cost, dtype=np.float32)
    g = np.asarray(sparse_atom_costs, dtype=np.float32)
    atoms = np.asarray(atom_indices, dtype=np.int64).reshape(-1)
    pairs = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2) if np.asarray(pair_indices).size else np.zeros((0, 2), dtype=np.int64)
    active = np.zeros((g.shape[0],), dtype=bool)
    active[atoms[(atoms >= 0) & (atoms < g.shape[0])]] = True
    if atom_active_mask is not None:
        active &= np.asarray(atom_active_mask, dtype=bool)[: g.shape[0]]
    if len(pairs):
        a = pairs[:, 0]
        b = pairs[:, 1]
        base_delta = J0[b] - J0[a]
        base_support = np.maximum(base_delta, 0.0)
        atom_support = np.maximum(g[:, b] - g[:, a], 0.0)
        caps = np.minimum(np.maximum(np.abs(base_delta) + float(gamma_max) * 0.25, 1e-3), float(gamma_max)).astype(np.float32)
    else:
        base_support = np.zeros((0,), dtype=np.float32)
        atom_support = np.zeros((g.shape[0], 0), dtype=np.float32)
        caps = np.zeros((0,), dtype=np.float32)
    selected, current, spent = _greedy_cover_from_pair_support(
        atom_support,
        base_support,
        caps,
        np.asarray(pair_weights, dtype=np.float32)[: len(pairs)],
        atom_budget_costs,
        budget,
        active,
    )
    return SelectionResult(
        selected=selected,
        objective_value=current,
        pair_indices=pairs,
        pair_weights=np.asarray(pair_weights, dtype=np.float32)[: len(pairs)],
        diagnostics={"spent_budget": spent, "budget": float(budget), "mode": "runtime_sparse_certificate", "pair_count": int(len(pairs)), "proposal_atom_count": int(active.sum())},
    )
