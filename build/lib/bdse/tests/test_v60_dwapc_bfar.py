from __future__ import annotations

import numpy as np
import torch

from bdse.model.losses import _straight_through_topm_mask
from bdse.planner.tournament import run_pair_conditioned_tournament


def _cfg(residual_beta: float = 0.0) -> dict:
    return {
        "model": {"pair_margin_normalized": False},
        "selector": {"progress_rivals": 0, "maneuver_rivals": 0},
        "tournament": {"L_infer": 2, "epsilon_cal": 0.0, "beta_uncertainty": 9.0, "use_softmin": False},
        "runtime": {
            "pair_tournament_anchor_mode": "selected_local",
            "pair_tournament_aggregation_mode": "evidence_action_potential",
            "set_conditioned_residual_scale": 1.0,
            "pair_action_anchor_guard": {"enabled": True, "flip_margin": 0.05, "score_margin": -1e9},
            "dual_certificate": {
                "enabled": True,
                "require_evidence_certificate_before_residual_flip": True,
                "min_evidence_certificate_fraction_for_residual_flip": 1.0,
                "residual_epsilon_cal": 0.0,
                "residual_beta_uncertainty": residual_beta,
            },
        },
    }


def test_straight_through_topm_has_exact_hard_forward_and_proposal_gradient() -> None:
    logits = torch.tensor([[0.1, 2.0, 1.0, -3.0]], requires_grad=True)
    active = torch.tensor([[True, True, True, False]])
    st, hard = _straight_through_topm_mask(logits, active, top_m=2, tau=0.2)
    assert hard.tolist() == [[False, True, True, False]]
    assert torch.allclose(st.detach(), hard.float(), atol=1e-7, rtol=0.0)
    (st * torch.tensor([[1.0, -2.0, 3.0, 5.0]])).sum().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0
    assert float(logits.grad[0, 3]) == 0.0


def test_pair_full_set_factors_can_change_winner() -> None:
    result = run_pair_conditioned_tournament(
        predicted_base_cost=np.array([0.0, 0.2], dtype=np.float32),
        pair_atom_delta=np.zeros((1, 1), dtype=np.float32),
        pair_indices=np.array([[0, 1]], dtype=np.int64),
        selected_atoms=[0],
        valid_mask=np.array([True, True]),
        runtime_safety_flags=np.array([False, False]),
        cfg=_cfg(0.0),
        predicted_atom_costs=np.zeros((1, 2), dtype=np.float32),
        residual_action_potential=np.zeros((1, 2), dtype=np.float32),
        residual_action_variance=np.zeros((1, 2), dtype=np.float32),
        residual_set_atom_factors=np.array([[1.0]], dtype=np.float32),
        residual_set_action_factors=np.array([[1.0], [-1.0]], dtype=np.float32),
        evidence_certificate_fraction=1.0,
    )
    assert int(result.action_index) == 1
    assert float(result.diagnostics["set_conditioned_residual_active"]) == 1.0


def test_residual_specific_beta_overrides_tournament_beta() -> None:
    result = run_pair_conditioned_tournament(
        predicted_base_cost=np.array([0.0, 0.2], dtype=np.float32),
        pair_atom_delta=np.zeros((1, 1), dtype=np.float32),
        pair_indices=np.array([[0, 1]], dtype=np.int64),
        selected_atoms=[0],
        valid_mask=np.array([True, True]),
        runtime_safety_flags=np.array([False, False]),
        cfg=_cfg(0.0),
        predicted_atom_costs=np.zeros((1, 2), dtype=np.float32),
        residual_action_potential=np.array([[0.2, -0.2]], dtype=np.float32),
        residual_action_variance=np.full((1, 2), 100.0, dtype=np.float32),
        evidence_certificate_fraction=1.0,
    )
    assert int(result.action_index) == 1
    assert float(result.diagnostics["pair_action_anchor_residual_beta_uncertainty"]) == 0.0
