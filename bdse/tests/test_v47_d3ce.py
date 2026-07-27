from __future__ import annotations

import numpy as np
import torch

from bdse.model.losses import _counterfactual_critical_evidence_loss
from bdse.planner.selector import runtime_greedy_selector_pair_conditioned


def test_aocc_uses_explicit_deployment_target_and_calibration_provenance() -> None:
    result = runtime_greedy_selector_pair_conditioned(
        predicted_base_cost=np.zeros((3,), dtype=np.float32),
        pair_atom_delta=np.asarray(
            [
                [0.8, 0.7, -0.2],
                [0.2, 0.1, 0.6],
                [0.0, 0.0, 0.4],
            ],
            dtype=np.float32,
        ),
        pair_indices=np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64),
        pair_weights=np.ones((3,), dtype=np.float32),
        atom_budget_costs=np.ones((3,), dtype=np.float32),
        valid_mask=np.ones((3,), dtype=bool),
        runtime_safety_flags=np.zeros((3,), dtype=bool),
        budget=2.0,
        atom_active_mask=np.ones((3,), dtype=bool),
        selector_cap_mode="anytime_adverse_certificate",
        adverse_certificate_target_action=1,
        adverse_certificate_calibrated=True,
        adverse_certificate_beta=0.0,
        adverse_certificate_epsilon=0.0,
        adverse_certificate_prior_radius=0.0,
        adverse_certificate_stop_when_certified=False,
    )
    assert result.diagnostics["aocc_target_action"] == 1
    assert result.diagnostics["aocc_target_source"] == "deployment_tournament"
    assert result.diagnostics["aocc_bound_calibrated"] is True


def test_counterfactual_critical_evidence_loss_is_finite_and_differentiable() -> None:
    true_delta = torch.tensor(
        [[[0.8, 0.0], [0.1, 0.7], [-0.2, 0.0]]], dtype=torch.float32
    )
    pred_delta = torch.tensor(
        [[[0.2, 0.0], [0.1, 0.1], [0.0, 0.0]]], dtype=torch.float32, requires_grad=True
    )
    predicted_margin = pred_delta.sum(dim=1)
    proposal = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
    pair_loss, proposal_loss = _counterfactual_critical_evidence_loss(
        true_delta,
        pred_delta,
        predicted_margin,
        torch.tensor([[[0, 1], [0, 2]]], dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        torch.tensor([0], dtype=torch.long),
        torch.ones((1, 3), dtype=torch.bool),
        torch.tensor([[1.0, 2.0, 1.0]], dtype=torch.float32),
        proposal,
        {"training": {"counterfactual_critical_evidence": {"top_k_rivals": 2}}},
    )
    total = pair_loss + proposal_loss
    assert torch.isfinite(total)
    total.backward()
    assert pred_delta.grad is not None and torch.isfinite(pred_delta.grad).all()
    assert proposal.grad is not None and torch.isfinite(proposal.grad).all()
