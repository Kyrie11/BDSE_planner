from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from bdse.model.losses import _action_potential_teacher_loss
from bdse.model.potential_projection import (
    project_pair_residual_to_action_potential_numpy,
    project_pair_residual_to_action_potential_torch,
)
from bdse.planner.tournament import run_pair_conditioned_tournament
from bdse.tools.check_v55_pc_bfar_dbap_gate import _paired_deployed_flip_stats, _training_health


def _runtime_cfg() -> dict:
    return {
        "model": {"pair_margin_normalized": False},
        "runtime": {
            "pair_tournament_anchor_mode": "selected_local",
            "pair_tournament_aggregation_mode": "integrable_potential",
            "pair_tournament_pair_delta_includes_local": True,
            "pair_potential_projection": {"ridge": 1.0e-4, "boundary_gain": 0.0},
            "pair_action_anchor_guard": {"enabled": True, "flip_margin": 0.0, "score_margin": 0.0},
        },
        "selector": {"progress_rivals": 0, "maneuver_rivals": 0},
        "tournament": {
            "L_infer": 1,
            "epsilon_cal": 0.0,
            "use_softmin": True,
            "softmin_tau": 1.0,
            "beta_uncertainty": 100.0,
            "hard_filter_unsafe_actions": False,
            "utility_refinement": {"enabled": False},
        },
    }


def test_numpy_projection_recovers_conservative_field() -> None:
    pairs = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    truth = np.asarray([-0.5, 0.75, -0.25], dtype=np.float32)
    residual = truth[pairs[:, 1]] - truth[pairs[:, 0]]
    potential, diag = project_pair_residual_to_action_potential_numpy(
        pairs, residual, np.ones((3,), dtype=bool), ridge=1.0e-6, boundary_gain=0.0
    )
    centered = truth - truth.mean()
    np.testing.assert_allclose(potential, centered, atol=2.0e-5)
    assert diag["pair_potential_cycle_fraction"] < 1.0e-8


def test_torch_projection_has_gradient_and_detects_cycle() -> None:
    pairs = torch.tensor([[[0, 1], [1, 2], [2, 0]]], dtype=torch.long)
    residual = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
    potential, reconstruction, cycle = project_pair_residual_to_action_potential_torch(
        pairs,
        residual,
        torch.ones((1, 3), dtype=torch.bool),
        torch.ones((1, 3), dtype=torch.bool),
        ridge=1.0e-3,
        boundary_gain=0.0,
    )
    assert potential.shape == (1, 3)
    assert float(cycle.detach()) > 0.9
    reconstruction.backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()


def test_zero_residual_runtime_exactly_matches_selected_local_anchor_even_with_variance() -> None:
    j0 = np.asarray([0.0, 0.2, 0.4], dtype=np.float32)
    g = np.asarray([[0.5, 0.1, -0.8]], dtype=np.float32)
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    local = g[:, pairs[:, 1]] - g[:, pairs[:, 0]]
    expected = int(np.argmin(j0 + g[0]))
    tour = run_pair_conditioned_tournament(
        j0,
        local,
        pairs,
        [0],
        np.ones((3,), dtype=bool),
        np.zeros((3,), dtype=bool),
        _runtime_cfg(),
        pair_atom_variance=np.full_like(local, 1.0e6),
        predicted_atom_costs=g,
    )
    assert tour.action_index == expected
    assert int(tour.diagnostics["pair_action_anchor_action"]) == expected
    assert not bool(tour.diagnostics["pair_action_anchor_deployed_flip"])
    assert tour.diagnostics["pair_tournament_aggregation_mode"] == "integrable_potential"


def test_potential_residual_can_flip_anchor_when_certified() -> None:
    cfg = _runtime_cfg()
    cfg["runtime"]["pair_tournament_pair_delta_includes_local"] = False
    cfg["tournament"]["beta_uncertainty"] = 0.0
    j0 = np.asarray([0.0, 0.2], dtype=np.float32)
    g = np.zeros((1, 2), dtype=np.float32)
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    residual = np.asarray([[-0.6]], dtype=np.float32)
    tour = run_pair_conditioned_tournament(
        j0, residual, pairs, [0], np.ones((2,), dtype=bool), np.zeros((2,), dtype=bool), cfg,
        pair_atom_variance=np.zeros_like(residual), predicted_atom_costs=g,
    )
    assert int(tour.diagnostics["pair_action_anchor_action"]) == 0
    assert tour.action_index == 1
    assert bool(tour.diagnostics["pair_action_anchor_guard_allowed_flip"])


def test_gate_exact_fraction_uses_configured_floor(tmp_path: Path) -> None:
    train_log = tmp_path / "train.jsonl"
    train_log.write_text(json.dumps({"epoch": 0, "loss": 1.0, "selector_exact_fraction": 0.025, "training_pair_fraction": 0.5}) + "\n")
    failures, stats = _training_health(train_log, 0.015)
    assert failures == []
    assert stats["last_exact_fraction"] == 0.025


def test_paired_flip_stats_are_causal() -> None:
    candidate = [
        {"scenario_token": "a", "timestamp_us": 1, "teacher_action": 1, "bdse_action": 1},
        {"scenario_token": "b", "timestamp_us": 2, "teacher_action": 0, "bdse_action": 2},
        {"scenario_token": "c", "timestamp_us": 3, "teacher_action": 0, "bdse_action": 0},
    ]
    local = [
        {"scenario_token": "a", "timestamp_us": 1, "teacher_action": 1, "bdse_action": 0},
        {"scenario_token": "b", "timestamp_us": 2, "teacher_action": 0, "bdse_action": 0},
        {"scenario_token": "c", "timestamp_us": 3, "teacher_action": 0, "bdse_action": 0},
    ]
    stats = _paired_deployed_flip_stats(candidate, local)
    assert stats["flip_rate"] == 2 / 3
    assert stats["beneficial_rate"] == 1 / 3
    assert stats["harmful_rate"] == 1 / 3


def test_action_potential_distillation_masks_invalid_costs_and_backpropagates() -> None:
    anchor = torch.tensor([[0.0, 1.0, float("inf")]], dtype=torch.float32)
    corrected = torch.tensor([[0.0, 0.4, float("inf")]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[0.2, 0.0, float("inf")]], dtype=torch.float32)
    valid = torch.tensor([[True, True, False]])
    cfg = {"training": {"action_potential_distillation": {"target_clip": 4.0}}}
    loss = _action_potential_teacher_loss(
        anchor, corrected, teacher, torch.tensor([1]), valid, torch.ones((1, 1)), cfg
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert corrected.grad is not None
    assert torch.isfinite(corrected.grad).all()
