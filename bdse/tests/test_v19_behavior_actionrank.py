import numpy as np

from bdse.planner.selector import _selector_pair_caps, runtime_greedy_selector_pair_conditioned


def test_action_rank_uses_boundary_certificate_caps():
    base = np.asarray([0.0, 0.5, 4.0], dtype=np.float32)
    safety_b = np.asarray([False, False, False])
    caps = _selector_pair_caps(
        base,
        safety_b,
        gamma_max=3.0,
        eta_pred=0.08,
        cap_mode="action_rank",
        boundary_cap=0.75,
        base_margin_cap_multiplier=0.25,
    )
    assert np.all(caps <= 0.750001)
    assert caps[0] >= 0.079
    # The large easy-margin pair must not receive the legacy abs(base)+eta cap.
    assert caps[-1] < 1.0


def test_action_utility_weight_changes_action_rank_acquisition():
    # Two actions, two directed pairs.  Atom 0 certifies action 0 over 1;
    # atom 1 certifies action 1 over 0.  Certificate gains are symmetric, so
    # without utility the deterministic tie break selects atom 0.  With utility,
    # action 1 is cheaper/higher-progress, so the selector selects atom 1.
    J0 = np.asarray([0.0, 0.0], dtype=np.float32)
    pair_indices = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    pair_delta = np.asarray([[0.4, -0.4], [-0.4, 0.4]], dtype=np.float32)
    weights = np.ones((2,), dtype=np.float32)
    costs = np.ones((2,), dtype=np.float32)
    valid = np.asarray([True, True])
    flags = np.asarray([False, False])

    no_util = runtime_greedy_selector_pair_conditioned(
        J0,
        pair_delta,
        pair_indices,
        weights,
        costs,
        valid,
        flags,
        budget=1,
        selector_cap_mode="action_rank",
        action_rank_certificate_weight=1.0,
        action_rank_score_weight=1.0,
        action_rank_gap_weight=0.0,
        action_rank_flip_weight=0.0,
        action_rank_softmin_tau=0.2,
    )
    assert no_util.selected == [0]

    with_util = runtime_greedy_selector_pair_conditioned(
        J0,
        pair_delta,
        pair_indices,
        weights,
        costs,
        valid,
        flags,
        budget=1,
        selector_cap_mode="action_rank",
        action_rank_certificate_weight=1.0,
        action_rank_score_weight=1.0,
        action_rank_gap_weight=0.0,
        action_rank_flip_weight=0.0,
        action_rank_softmin_tau=0.2,
        action_utility_cost=np.asarray([1.0, 0.0], dtype=np.float32),
        action_utility_weight=2.0,
    )
    assert with_util.selected == [1]
