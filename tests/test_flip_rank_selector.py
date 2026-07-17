import numpy as np

from bdse.planner.selector import runtime_greedy_selector_pair_conditioned


def test_flip_rank_selector_prefers_atom_that_crosses_pair_margin():
    # Pair (0,1): action 0 is not yet certified because base margin is negative.
    # Atom 0 crosses the pair margin; atom 1 only provides smaller positive support.
    J0 = np.array([0.0, -0.2], dtype=np.float32)  # base delta J0[1]-J0[0] = -0.2
    pair_delta = np.array([[0.35], [0.10]], dtype=np.float32)
    pairs = np.array([[0, 1]], dtype=np.int64)
    weights = np.array([1.0], dtype=np.float32)
    costs = np.ones((2,), dtype=np.float32)
    valid = np.array([True, True])
    flags = np.array([False, False])
    sel = runtime_greedy_selector_pair_conditioned(
        J0,
        pair_delta,
        pairs,
        weights,
        costs,
        valid,
        flags,
        budget=1,
        gamma_max=3.0,
        eta_pred=0.1,
        selector_cap_mode="flip_rank",
        boundary_certificate_cap=0.55,
        flip_bonus=1.0,
        flip_window=0.6,
    )
    assert sel.selected == [0]
    assert sel.diagnostics["mode"] == "runtime_pair_conditioned_flip_rank"


def test_flip_rank_selector_keeps_budget_and_valid_masks():
    J0 = np.array([0.0, -0.2], dtype=np.float32)
    pair_delta = np.array([[0.35], [0.50]], dtype=np.float32)
    pairs = np.array([[0, 1]], dtype=np.int64)
    weights = np.array([1.0], dtype=np.float32)
    costs = np.ones((2,), dtype=np.float32)
    valid = np.array([True, True])
    flags = np.array([False, False])
    active = np.array([True, False])
    sel = runtime_greedy_selector_pair_conditioned(
        J0,
        pair_delta,
        pairs,
        weights,
        costs,
        valid,
        flags,
        budget=1,
        atom_active_mask=active,
        gamma_max=3.0,
        eta_pred=0.1,
        selector_cap_mode="flip_rank",
        boundary_certificate_cap=0.55,
        flip_bonus=1.0,
        flip_window=0.6,
    )
    assert sel.selected == [0]


def test_action_rank_selector_is_not_bypassed_by_variance_or_family_caps():
    # Regression for v18: v15-v17 passed pair_atom_variance/family caps at runtime,
    # which accidentally forced the uncertainty objective and bypassed action_rank.
    J0 = np.array([0.0, -0.2, -0.3], dtype=np.float32)
    pairs = np.array([[0, 1], [1, 0], [0, 2], [2, 0]], dtype=np.int64)
    # Atom 0 changes the tournament ranking; atom 1 only gives weaker support.
    pair_delta = np.array([
        [0.45, -0.45, 0.40, -0.40],
        [0.10, -0.10, 0.10, -0.10],
    ], dtype=np.float32)
    weights = np.ones((4,), dtype=np.float32)
    costs = np.ones((2,), dtype=np.float32)
    valid = np.array([True, True, True])
    flags = np.array([False, False, False])
    sel = runtime_greedy_selector_pair_conditioned(
        J0,
        pair_delta,
        pairs,
        weights,
        costs,
        valid,
        flags,
        budget=1,
        gamma_max=3.0,
        eta_pred=0.1,
        selector_cap_mode="action_rank",
        boundary_certificate_cap=0.75,
        action_rank_certificate_weight=0.75,
        action_rank_score_weight=1.0,
        action_rank_gap_weight=0.75,
        action_rank_flip_weight=0.35,
        pair_atom_variance=np.ones_like(pair_delta),
        family_budget_caps=np.array([1, 1], dtype=np.int64),
    )
    assert sel.diagnostics["mode"] == "runtime_pair_conditioned_action_rank"


def test_force_uncertainty_objective_can_still_override_action_rank():
    J0 = np.array([0.0, -0.2], dtype=np.float32)
    pair_delta = np.array([[0.35], [0.10]], dtype=np.float32)
    pairs = np.array([[0, 1]], dtype=np.int64)
    weights = np.array([1.0], dtype=np.float32)
    costs = np.ones((2,), dtype=np.float32)
    valid = np.array([True, True])
    flags = np.array([False, False])
    sel = runtime_greedy_selector_pair_conditioned(
        J0,
        pair_delta,
        pairs,
        weights,
        costs,
        valid,
        flags,
        budget=1,
        selector_cap_mode="action_rank",
        pair_atom_variance=np.ones_like(pair_delta),
        force_uncertainty_objective=True,
    )
    assert sel.diagnostics["mode"] == "runtime_pair_conditioned_lcb_uncertainty"
