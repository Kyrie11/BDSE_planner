from __future__ import annotations

import numpy as np
import torch

from bdse.model.losses import _budgeted_tournament_scores, _pair_conditioned_tournament_scores
from bdse.planner.tournament import _pair_delta_margin_matrix, run_pair_conditioned_tournament
from bdse.tools.check_v54_ar_bfar_dbap_gate import _paired_action_mismatch


def test_anchor_relative_margin_zero_residual_recovers_selected_local_cost() -> None:
    j0 = np.asarray([0.0, 10.0, 20.0], dtype=np.float32)
    g = np.asarray([[10.0, -10.0, 0.0]], dtype=np.float32)
    pairs = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    local = np.asarray([[g[0, 1] - g[0, 0], g[0, 2] - g[0, 1]]], dtype=np.float32)
    margin = _pair_delta_margin_matrix(
        j0,
        pairs,
        local,
        [0],
        np.ones(3, dtype=bool),
        predicted_atom_costs=g,
        pair_delta_includes_local=True,
    )
    anchor = j0 + g[0]
    expected = anchor[None, :] - anchor[:, None]
    np.testing.assert_allclose(margin, expected, atol=1e-6)


def test_differentiable_anchor_relative_tournament_zero_residual_matches_anchor() -> None:
    j0 = torch.tensor([[0.0, 1.0, 2.0]])
    g = torch.tensor([[[0.2, -0.4, 0.1]]])
    pairs = torch.tensor([[[0, 1], [1, 2]]])
    pair_valid = torch.ones((1, 2), dtype=torch.bool)
    selected = torch.ones((1, 1), dtype=torch.bool)
    valid = torch.ones((1, 3), dtype=torch.bool)
    local = torch.tensor([[[-0.6, 0.5]]])
    anchor_cost = j0 + g.sum(dim=1)
    anchored = _pair_conditioned_tournament_scores(
        j0,
        local,
        pairs,
        pair_valid,
        selected,
        valid,
        tau=0.5,
        normalize_margins=False,
        anchor_cost=anchor_cost,
        local_pair_delta=local,
        pair_delta_includes_local=True,
    )
    reference = _budgeted_tournament_scores(anchor_cost, valid, tau=0.5)
    torch.testing.assert_close(anchored, reference)


def _cfg(flip_margin: float) -> dict:
    return {
        "model": {"pair_margin_normalized": False},
        "runtime": {
            "pair_tournament_anchor_mode": "selected_local",
            "pair_tournament_pair_delta_includes_local": False,
            "pair_action_anchor_guard": {"enabled": True, "flip_margin": flip_margin, "score_margin": 0.0},
        },
        "selector": {"eta_pred": 1.0, "progress_rivals": 0, "maneuver_rivals": 0},
        "tournament": {"L_infer": 2, "use_softmin": False, "softmin_tau": 1.0, "epsilon_cal": 0.0, "beta_uncertainty": 0.0},
    }


def test_action_anchor_guard_blocks_uncertified_flip_and_allows_certified_flip() -> None:
    j0 = np.asarray([0.0, 1.0], dtype=np.float32)
    g = np.zeros((1, 2), dtype=np.float32)
    pairs = np.asarray([[1, 0]], dtype=np.int64)
    valid = np.asarray([True, True])
    flags = np.asarray([False, False])
    blocked = run_pair_conditioned_tournament(
        j0, np.asarray([[1.02]], dtype=np.float32), pairs, [0], valid, flags,
        _cfg(0.06), predicted_atom_costs=g,
    )
    assert blocked.diagnostics["pair_action_anchor_guard_blocked_flip"]
    assert blocked.action_index == 0
    allowed = run_pair_conditioned_tournament(
        j0, np.asarray([[1.20]], dtype=np.float32), pairs, [0], valid, flags,
        _cfg(0.06), predicted_atom_costs=g,
    )
    assert allowed.diagnostics["pair_action_anchor_guard_allowed_flip"]
    assert allowed.action_index == 1


def test_gate_drift_compares_same_local_interface() -> None:
    cand = [
        {"scenario_token": "a", "timestamp_us": 1, "local_pair_full_action": 2, "pair_full_action": 1},
        {"scenario_token": "b", "timestamp_us": 2, "local_pair_full_action": 0, "pair_full_action": 3},
    ]
    local = [
        {"scenario_token": "a", "timestamp_us": 1, "local_pair_full_action": 2, "pair_full_action": 2},
        {"scenario_token": "b", "timestamp_us": 2, "local_pair_full_action": 0, "pair_full_action": 0},
    ]
    drift, n = _paired_action_mismatch(cand, local, "local_pair_full_action", "candidate", "local")
    assert n == 2
    assert drift == 0.0
