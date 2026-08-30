from __future__ import annotations

import numpy as np

from bdse.planner.nuplan_planner import runtime_query_diagnostics
from bdse.planner.pair_screen import restrict_pairs_to_viability_frontier
from bdse.planner.selector import restrict_topm_to_decision_evidence, structural_safety_mask


def test_structural_safety_mask_combines_hard_and_feasibility():
    hard = np.asarray([False, True, False, False, False], dtype=bool)
    family = np.asarray([1, 2, 3, 1, 4], dtype=np.int64)
    active = np.asarray([True, True, True, False, True], dtype=bool)
    out = structural_safety_mask(hard, family, active, include_feasibility=True)
    assert out.tolist() == [True, True, False, False, False]


def test_decision_topm_excludes_structural_and_preserves_fixed_size():
    # Initial Top-M contains structural atoms 0 and 1.  They must be replaced by
    # the highest-scoring decision atoms without increasing M.
    decision = np.asarray([False, False, True, True, True, True], dtype=bool)
    score = np.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0], dtype=np.float32)
    out, diag = restrict_topm_to_decision_evidence(
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        decision,
        score,
        max_size=4,
        family_ids=np.asarray([1, 2, 2, 3, 4, 5], dtype=np.int64),
        min_family_slots={2: 1, 3: 1},
    )
    assert len(out) == 4
    assert set(out.tolist()) == {2, 3, 4, 5}
    assert not np.any(~decision[out])
    assert diag["structural_atoms_excluded_from_topm"] == 2


def test_viability_frontier_uses_safe_safe_pairs_when_multiple_safe_actions():
    pairs = np.asarray([[0, 1], [0, 2], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    weights = np.ones((len(pairs),), dtype=np.float32)
    out, _, diag = restrict_pairs_to_viability_frontier(
        pairs,
        weights,
        valid_mask=np.ones((4,), dtype=bool),
        safety_flags=np.asarray([False, False, True, True], dtype=bool),
        predicted_base_cost=np.asarray([0.0, 0.2, 0.4, 0.6], dtype=np.float32),
    )
    assert set(map(tuple, out.tolist())) == {(0, 1)}
    assert diag["viability_scope_code"] == 3.0
    assert diag["viability_safe_action_count"] == 2.0


def test_viability_frontier_keeps_small_anchor_graph_for_single_safe_action():
    pairs = np.asarray([[0, 1], [0, 2], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    out, _, diag = restrict_pairs_to_viability_frontier(
        pairs,
        None,
        valid_mask=np.ones((4,), dtype=bool),
        safety_flags=np.asarray([False, True, True, True], dtype=bool),
        predicted_base_cost=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32),
        single_safe_rivals=2,
    )
    assert len(out) == 2
    assert all(0 in pair for pair in map(tuple, out.tolist()))
    assert diag["viability_scope_code"] == 2.0


def test_runtime_query_diagnostics_separates_structural_channel_from_budget():
    pred = {
        "top_m_atoms": np.arange(6),
        "runtime_pairs": np.asarray([[0, 1], [0, 2]], dtype=np.int64),
        "rival_pair_indices": np.asarray([[0, 1], [0, 2]], dtype=np.int64),
        "unique_pair_atom_query_count": 12,
        "actual_unique_pair_count": 2,
        "structural_safety_atom_count": 9,
        "structural_safety_bypass": True,
    }
    out = runtime_query_diagnostics(pred, selected_atoms=[2, 3, 4])
    assert out["decision_budget_atom_count"] == 3
    assert out["structural_safety_atom_count"] == 9
    assert out["decision_budget_excludes_structural_safety"] == 1
    assert out["effective_query_count"] == 6


def test_v36_gate_accepts_two_channel_coverage_and_rejects_unsafe_output():
    from bdse.tools.check_v36_runtime_gate import passes

    row = {
        "structural_hard_decisive_coverage": 1.0,
        "effective_hard_decisive_recall": 1.0,
        "selected_soft_interaction_decisive_recall": 0.35,
        "effective_interaction_decisive_recall": 0.40,
        "fallback_would_trigger_rate": 0.0,
        "selected_action_safety_flag_rate": 0.0,
        "teacher_action_match": 0.23,
        "effective_query_count": 6000.0,
        "total_sparse_query_count": 16000.0,
        "decision_budget_excludes_structural_safety": 1.0,
    }
    ok, failures = passes(row)
    assert ok, failures

    unsafe = dict(row)
    unsafe["selected_action_safety_flag_rate"] = 0.02
    ok, failures = passes(unsafe)
    assert not ok
    assert any("selected_action_safety_flag_rate" in x for x in failures)
