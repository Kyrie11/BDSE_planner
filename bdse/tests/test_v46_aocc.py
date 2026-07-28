from __future__ import annotations

import numpy as np

from bdse.planner.selector import runtime_greedy_selector_pair_conditioned


def _run(budget: float):
    # Full field predicts action 0 over rivals 1 and 2. Atom 0 protects both
    # margins, atom 1 only the first, atom 2 only the second.
    base_cost = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    pairs = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
    delta = np.asarray(
        [
            [0.60, 0.55],
            [0.50, 0.00],
            [0.00, 0.45],
            [-0.10, -0.10],
        ],
        dtype=np.float32,
    )
    return runtime_greedy_selector_pair_conditioned(
        base_cost,
        delta,
        pairs,
        np.ones((2,), dtype=np.float32),
        np.ones((4,), dtype=np.float32),
        np.ones((3,), dtype=bool),
        np.zeros((3,), dtype=bool),
        budget=budget,
        atom_active_mask=np.ones((4,), dtype=bool),
        selector_cap_mode="anytime_adverse_certificate",
        adverse_certificate_beta=0.0,
        adverse_certificate_epsilon=0.0,
        adverse_certificate_prior_radius=0.0,
        adverse_certificate_stop_when_certified=False,
        force_fill_budget=False,
        min_selected_atoms=0,
        soft_interaction_quota=0,
    )


def test_aocc_is_nested_across_budgets():
    b1 = _run(1.0)
    b2 = _run(2.0)
    assert b1.selected == b2.selected[: len(b1.selected)]
    assert b1.selected[0] == 0
    assert b2.diagnostics["aocc_final_deficit"] <= b1.diagnostics["aocc_final_deficit"] + 1e-8


def test_aocc_lower_bound_has_nonnegative_query_improvements():
    result = _run(3.0)
    assert result.diagnostics["aocc_max_lower_bound_violation"] <= 1e-8
    assert result.diagnostics["aocc_deficit_reduction"] >= -1e-8
    assert result.diagnostics["aocc_nested_order_length"] >= len(result.selected)
    assert result.diagnostics["aocc_selected_prefix_length"] == len(result.selected)


def _run_nonuniform(budget: float):
    base_cost = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    pairs = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
    delta = np.asarray(
        [
            [0.80, 0.80],  # high-value, higher-cost first item
            [0.45, 0.00],
            [0.00, 0.45],
            [-0.05, -0.05],
        ],
        dtype=np.float32,
    )
    costs = np.asarray([1.5, 1.0, 1.0, 0.5], dtype=np.float32)
    return runtime_greedy_selector_pair_conditioned(
        base_cost,
        delta,
        pairs,
        np.ones((2,), dtype=np.float32),
        costs,
        np.ones((3,), dtype=bool),
        np.zeros((3,), dtype=bool),
        budget=budget,
        atom_active_mask=np.ones((4,), dtype=bool),
        selector_cap_mode="anytime_adverse_certificate",
        adverse_certificate_beta=0.0,
        adverse_certificate_epsilon=0.0,
        adverse_certificate_prior_radius=0.0,
        adverse_certificate_stop_when_certified=False,
        force_fill_budget=False,
        min_selected_atoms=0,
        soft_interaction_quota=0,
    )


def test_aocc_nonuniform_cost_budgets_are_strict_prefixes():
    small = _run_nonuniform(1.0)
    medium = _run_nonuniform(2.0)
    large = _run_nonuniform(3.5)
    assert small.selected == medium.selected[: len(small.selected)]
    assert medium.selected == large.selected[: len(medium.selected)]
    assert small.diagnostics["aocc_nested_order_length"] == large.diagnostics["aocc_nested_order_length"]


def test_aocc_frontier_cap_reports_retained_weight() -> None:
    pairs = np.asarray([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.int64)
    delta = np.asarray(
        [
            [0.20, 0.18, 0.16, 0.14],
            [0.10, 0.08, 0.06, 0.04],
        ],
        dtype=np.float32,
    )
    result = runtime_greedy_selector_pair_conditioned(
        np.zeros((5,), dtype=np.float32),
        delta,
        pairs,
        np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        np.ones((2,), dtype=np.float32),
        np.ones((5,), dtype=bool),
        np.asarray([False, False, False, True, False], dtype=bool),
        budget=2.0,
        atom_active_mask=np.ones((2,), dtype=bool),
        selector_cap_mode="anytime_adverse_certificate",
        adverse_certificate_target_action=0,
        adverse_certificate_max_target_rivals=2,
        adverse_certificate_beta=0.0,
        adverse_certificate_epsilon=0.0,
        adverse_certificate_prior_radius=0.0,
        adverse_certificate_stop_when_certified=False,
    )
    assert result.diagnostics["aocc_frontier_original_pair_count"] == 4
    assert result.diagnostics["aocc_pair_count"] == 2
    retained = result.diagnostics["aocc_frontier_retained_weight_fraction"]
    assert 0.0 < retained <= 1.0
