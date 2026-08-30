from __future__ import annotations

import numpy as np

from bdse.model.losses import _deployment_budget_entries_for_step
from bdse.planner.selector import _signed_margin_coreset_from_pair_delta


def test_weighted_round_robin_matches_budget_weights_across_two_ranks() -> None:
    budgets = [8.0, 16.0, 24.0]
    weights = [0.75, 1.5, 0.75]
    seen: list[float] = []
    for step in range(2):
        for rank in range(2):
            cfg = {
                "deployment_budget_strategy": "weighted_round_robin",
                "deployment_budget_schedule_slots": 4,
                "global_step": step,
                "global_rank": rank,
                "world_size": 2,
            }
            seen.append(_deployment_budget_entries_for_step(budgets, weights, cfg)[0][0])
    assert seen.count(8.0) == 1
    assert seen.count(16.0) == 2
    assert seen.count(24.0) == 1


def test_all_budget_strategy_is_backward_compatible() -> None:
    entries = _deployment_budget_entries_for_step(
        [8.0, 16.0, 24.0], [0.75, 1.5, 0.75], {"deployment_budget_strategy": "all"}
    )
    assert entries == [(8.0, 0.75), (16.0, 1.5), (24.0, 0.75)]


def test_vectorized_margin_coreset_returns_valid_budgeted_subset() -> None:
    rng = np.random.default_rng(7)
    evidence_count, pair_count, action_count = 32, 48, 12
    delta = rng.normal(0.0, 0.25, (evidence_count, pair_count)).astype(np.float32)
    base = rng.normal(0.0, 0.5, pair_count).astype(np.float32)
    pairs = np.stack(
        [rng.integers(0, action_count, pair_count), rng.integers(0, action_count, pair_count)], axis=1
    ).astype(np.int64)
    pairs[:, 1] = (pairs[:, 1] + (pairs[:, 1] == pairs[:, 0])) % action_count
    pair_weights = rng.uniform(0.1, 2.0, pair_count).astype(np.float32)
    costs = np.ones(evidence_count, dtype=np.float32)
    valid = np.ones(action_count, dtype=bool)
    flags = np.zeros(action_count, dtype=bool)
    active = np.ones(evidence_count, dtype=bool)

    selected, _, spent, diag = _signed_margin_coreset_from_pair_delta(
        delta,
        base,
        pair_weights,
        pairs,
        costs,
        budget=8.0,
        valid_mask=valid,
        runtime_safety_flags=flags,
        atom_active_mask=active,
        swap_passes=2,
    )
    assert len(selected) <= 8
    assert spent <= 8.0 + 1e-6
    assert len(set(selected)) == len(selected)
    assert diag["margin_coreset_active"] is True
