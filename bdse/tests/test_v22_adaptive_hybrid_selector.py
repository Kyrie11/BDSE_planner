import numpy as np

from bdse.planner.selector import _adaptive_hybrid_lcb_fraction, runtime_greedy_selector_pair_conditioned


def test_adaptive_hybrid_increases_lcb_fraction_when_safety_risk_is_high():
    base = np.asarray([0.2, -0.1, 0.05, 0.3], dtype=np.float32)
    weights = np.ones((4,), dtype=np.float32)
    low, low_diag = _adaptive_hybrid_lcb_fraction(
        base,
        weights,
        np.zeros((4,), dtype=bool),
        None,
        base_frac=0.55,
        min_frac=0.35,
        max_frac=0.85,
    )
    high, high_diag = _adaptive_hybrid_lcb_fraction(
        base,
        weights,
        np.ones((4,), dtype=bool),
        None,
        base_frac=0.55,
        min_frac=0.35,
        max_frac=0.85,
    )
    assert high > low
    assert high_diag["adaptive_safety_density"] > low_diag["adaptive_safety_density"]


def test_adaptive_hybrid_reports_runtime_split_and_family_boost():
    j0 = np.asarray([0.0, 0.1], dtype=np.float32)
    pair_indices = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    pair_delta = np.asarray([[0.3, -0.3], [0.2, -0.2], [-0.1, 0.4]], dtype=np.float32)
    weights = np.ones((2,), dtype=np.float32)
    costs = np.ones((3,), dtype=np.float32)
    valid = np.asarray([True, True])
    flags = np.asarray([False, True])
    families = np.asarray([1, 2, 3], dtype=np.int64)
    sel = runtime_greedy_selector_pair_conditioned(
        j0,
        pair_delta,
        pair_indices,
        weights,
        costs,
        valid,
        flags,
        budget=2,
        family_ids=families,
        selector_cap_mode="adaptive_safety_gated_action_rank",
        action_rank_fast_greedy=True,
        adaptive_hybrid_lcb_budget=True,
        hybrid_lcb_budget_frac=0.5,
        adaptive_lcb_min_frac=0.25,
        adaptive_lcb_max_frac=0.85,
        decision_family_ids=[2, 3],
        decision_family_boost=0.2,
        decision_family_quota=1,
        force_fill_budget=True,
        min_selected_atoms=2,
    )
    assert sel.diagnostics["mode"] == "runtime_pair_conditioned_hybrid_lcb_action_rank"
    assert bool(sel.diagnostics["hybrid_adaptive_lcb_budget"])
    assert "adaptive_lcb_frac" in sel.diagnostics
    assert sel.diagnostics["decision_family_selected"] >= 1
