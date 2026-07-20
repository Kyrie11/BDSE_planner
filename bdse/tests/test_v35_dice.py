import numpy as np

from bdse.planner.selector import (
    _complete_safety_aware_selection,
    _direction_invariant_interaction_utility,
    reserve_topm_candidates,
    runtime_greedy_selector_pair_conditioned,
)


def test_direction_invariant_utility_rewards_both_margin_signs():
    delta = np.asarray([[0.8], [-0.8], [0.1]], dtype=np.float32)
    score = _direction_invariant_interaction_utility(
        delta,
        np.asarray([0.2], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        boundary_tau=0.5,
        flip_bonus=0.0,
    )
    assert np.isclose(score[0], score[1], atol=1e-6)
    assert score[0] > score[2]


def test_soft_interaction_floor_is_not_satisfied_by_hard_occupancy():
    selected, _, diag = _complete_safety_aware_selection(
        selected=[0, 1, 6, 7, 8, 9],
        atom_budget_costs=np.ones((10,), dtype=np.float32),
        budget=6,
        atom_active_mask=np.ones((10,), dtype=bool),
        mandatory_atom_mask=np.asarray([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
        mandatory_quota=2,
        utility=np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        family_ids=np.asarray([2, 2, 2, 3, 2, 3, 4, 4, 5, 5], dtype=np.int64),
        interaction_family_ids=[2, 3],
        soft_interaction_mask=np.asarray([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
        soft_interaction_quota=3,
        interaction_group_ids=np.asarray([0, 1, 0, 1, 2, 2, -1, -1, -1, -1], dtype=np.int64),
        interaction_utility=np.asarray([0.0, 0.0, 0.9, 0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert {0, 1}.issubset(set(selected))
    assert diag["mandatory_selected"] >= 2
    assert diag["soft_interaction_selected"] >= 3
    assert diag["soft_interaction_distinct_groups"] >= 3


def test_soft_interaction_topm_reservation_keeps_pool_size_and_protected_atoms():
    topm, diag = reserve_topm_candidates(
        np.asarray([0, 1, 6, 7, 8, 9], dtype=np.int64),
        candidate_mask=np.asarray([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
        scores=np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        max_size=6,
        min_slots=3,
        protected_mask=np.asarray([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
        group_ids=np.asarray([-1, -1, 0, 1, 2, 2, -1, -1, -1, -1], dtype=np.int64),
    )
    assert len(topm) == 6
    assert {0, 1}.issubset(set(topm.tolist()))
    assert diag["reserved_selected"] >= 3
    assert diag["reserved_distinct_groups"] >= 3


def test_pair_selector_uses_soft_interaction_floor_under_fixed_budget():
    result = runtime_greedy_selector_pair_conditioned(
        predicted_base_cost=np.asarray([0.0, 0.2], dtype=np.float32),
        pair_atom_delta=np.asarray(
            [
                [0.2], [0.15],  # hard interaction
                [-0.8], [0.7], [-0.6], [0.5],  # soft interaction, both signs
                [0.4], [0.3],  # other evidence
            ],
            dtype=np.float32,
        ),
        pair_indices=np.asarray([[0, 1]], dtype=np.int64),
        pair_weights=np.asarray([1.0], dtype=np.float32),
        atom_budget_costs=np.ones((8,), dtype=np.float32),
        valid_mask=np.asarray([1, 1], dtype=bool),
        runtime_safety_flags=np.asarray([0, 0], dtype=bool),
        budget=6,
        atom_active_mask=np.ones((8,), dtype=bool),
        mandatory_atom_mask=np.asarray([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
        mandatory_quota=2,
        family_ids=np.asarray([2, 2, 2, 3, 2, 3, 4, 5], dtype=np.int64),
        interaction_family_ids=[2, 3],
        soft_interaction_mask=np.asarray([0, 0, 1, 1, 1, 1, 0, 0], dtype=bool),
        soft_interaction_quota=3,
        interaction_group_ids=np.asarray([0, 1, 0, 1, 2, 2, -1, -1], dtype=np.int64),
        direction_invariant_interaction_weight=1.0,
        direction_invariant_boundary_tau=0.5,
        direction_invariant_flip_bonus=0.5,
        selector_cap_mode="action_rank",
        min_selected_atoms=6,
        force_fill_budget=True,
    )
    assert result.diagnostics["mandatory_selected"] >= 2
    assert result.diagnostics["soft_interaction_selected"] >= 3
    assert len(result.selected) == 6
