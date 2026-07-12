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
