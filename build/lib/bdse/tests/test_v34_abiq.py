from __future__ import annotations

import numpy as np

from bdse.planner.nuplan_planner import runtime_query_diagnostics
from bdse.planner.pair_screen import compact_runtime_pair_graph
from bdse.planner.selector import _complete_safety_aware_selection, runtime_greedy_selector_pair_conditioned


def test_compact_pair_graph_removes_reciprocals_and_respects_cap():
    pairs = np.asarray([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2]], dtype=np.int64)
    weights = np.asarray([1.0, 0.5, 2.0, 1.0, 0.2], dtype=np.float32)
    J0 = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    valid = np.ones((3,), dtype=bool)
    flags = np.asarray([False, True, False], dtype=bool)
    out, out_w, diag = compact_runtime_pair_graph(
        pairs, weights, J0, valid, flags, max_pairs=2, canonicalize_reciprocals=True
    )
    assert out.shape == (2, 2)
    assert out_w.shape == (2,)
    assert diag["reciprocal_pairs_removed"] >= 2
    # Safety-crossing pair is oriented unflagged -> flagged and preserved.
    assert (0, 1) in set(map(tuple, out.tolist()))


def test_interaction_quota_can_exchange_excess_hard_atoms_without_breaking_hard_floor():
    costs = np.ones((10,), dtype=np.float32)
    active = np.ones((10,), dtype=bool)
    mandatory = np.asarray([True] * 7 + [False] * 3, dtype=bool)
    family = np.asarray([1, 1, 1, 1, 1, 1, 1, 2, 3, 2], dtype=np.int64)
    utility = np.asarray([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32)
    selected, spent, diag = _complete_safety_aware_selection(
        list(range(7)),
        costs,
        budget=8,
        atom_active_mask=active,
        mandatory_atom_mask=mandatory,
        mandatory_quota=5,
        min_selected_atoms=8,
        force_fill_budget=True,
        utility=utility,
        family_ids=family,
        interaction_family_ids=[2, 3],
        interaction_family_quota=3,
    )
    assert spent == 8
    assert sum(bool(mandatory[i]) for i in selected) >= 5
    assert sum(int(family[i]) in {2, 3} for i in selected) >= 3
    assert diag["interaction_family_selected"] >= 3


def test_pair_conditioned_selector_collapses_reciprocal_objective():
    J0 = np.asarray([0.0, 0.2], dtype=np.float32)
    pairs = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    weights = np.ones((2,), dtype=np.float32)
    # Atom 0 strongly supports 0>1; atom 1 is weak. Reciprocal directions would
    # otherwise cancel under a signed coverage objective.
    delta = np.asarray([[1.0, -1.0], [0.1, -0.1]], dtype=np.float32)
    result = runtime_greedy_selector_pair_conditioned(
        J0,
        delta,
        pairs,
        weights,
        np.ones((2,), dtype=np.float32),
        np.ones((2,), dtype=bool),
        np.zeros((2,), dtype=bool),
        budget=1,
        atom_active_mask=np.ones((2,), dtype=bool),
        selector_cap_mode="legacy_abs",
        collapse_reciprocal_pairs=True,
    )
    assert result.selected == [0]
    assert result.pair_indices.shape[0] == 1


def test_query_diagnostics_reports_actual_unique_union():
    pred = {
        "top_m_atoms": np.arange(4),
        "queried_actions": np.arange(3),
        "runtime_pairs": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        "rival_pair_indices": np.asarray([[0, 1]], dtype=np.int64),
        "action_atom_query_count": 0,
        "selector_pair_atom_query_count": 8,
        "tournament_pair_atom_query_count": 4,
        "unique_pair_atom_query_count": 4,
        "actual_unique_pair_count": 1,
    }
    out = runtime_query_diagnostics(pred, selected_atoms=[0, 1])
    assert out["total_sparse_query_count"] == 4
    assert out["effective_query_count"] == 2
    assert out["selector_pair_atom_query_count"] == 8
    assert out["tournament_pair_atom_query_count"] == 4
