from __future__ import annotations

import numpy as np
import torch

from bdse.data.cache_schema import CandidateBank, EvidenceAtom, EvidenceBank, PairLabels, TeacherLabels
from bdse.metrics.bdse_metrics import compute_bdse_diagnostics
from bdse.model.losses import _exact_winner_flip_critical_proposal_loss


def test_exact_winner_flip_criticality_uses_literal_loo_action_flip() -> None:
    # With atom 0 present, action 1 wins (0.2 < 1.0). Removing atom 0 makes
    # action 0 win (0.0 < 0.2), so atom 0 is exactly critical.
    J0 = torch.tensor([[0.0, 0.2]])
    g = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, True]])
    active = torch.tensor([[True, True]])
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    deployed = torch.tensor([[True, False]])
    target = torch.tensor([1])
    costs = torch.ones((1, 2))
    cfg = {
        "training": {
            "exact_winner_flip_criticality": {
                "enabled": True,
                "positive_weight": 8.0,
                "negative_weight": 0.25,
                "rank_weight": 1.0,
                "teacher_aligned_weight": 4.0,
                "min_action_scale": 1.0,
            }
        }
    }
    loss, recall, atom_fraction, scene_fraction, teacher_scene_fraction = (
        _exact_winner_flip_critical_proposal_loss(
            J0, g, valid, active, logits, deployed, target, costs, cfg
        )
    )
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
    assert float(recall) == 1.0
    assert float(atom_fraction) == 0.5
    assert float(scene_fraction) == 1.0
    assert float(teacher_scene_fraction) == 1.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] < logits.grad[0, 1]


def test_metric_export_keeps_set_residual_fields_and_names_signed_scalar_delta() -> None:
    candidates = CandidateBank(
        trajectories=np.zeros((2, 2, 5), dtype=np.float32),
        valid_mask=np.array([True, True]),
        maneuver_ids=np.zeros((2,), dtype=np.int64),
        theta=[{}, {}],
        dynamic_flags=[{}, {}],
        metadata=[{}, {}],
    )
    atom = EvidenceAtom(
        atom_id=0,
        type="yield",
        anchor={},
        budget_cost=1.0,
        is_hard=False,
        family="interaction",
        active_mask=True,
    )
    evidence = EvidenceBank(
        atoms=[atom],
        query_features=np.zeros((1, 2, 1), dtype=np.float32),
        active_mask=np.array([True]),
    )
    # Lexicographic teacher action 0 may have higher scalar J_T than action 1.
    teacher = TeacherLabels(
        J_base=np.array([10.0, 5.0], dtype=np.float32),
        g_evid=np.zeros((1, 2), dtype=np.float32),
        J_evid=np.zeros((2,), dtype=np.float32),
        J_T=np.array([10.0, 5.0], dtype=np.float32),
        a_star=0,
        hard_violation_mask=np.array([False, True]),
    )
    pairs = PairLabels(
        pairs=np.array([[0, 1]], dtype=np.int64),
        margins=np.array([1.0], dtype=np.float32),
        weights=np.array([1.0], dtype=np.float32),
        residuals=np.array([0.0], dtype=np.float32),
        valid_mask=np.array([True]),
    )
    result = compute_bdse_diagnostics(
        candidates,
        evidence,
        teacher,
        pairs,
        predicted_base=np.array([10.0, 5.0], dtype=np.float32),
        predicted_atom_costs=np.zeros((1, 2), dtype=np.float32),
        selected_atoms=[0],
        action_index=1,
        query_diagnostics={
            "set_conditioned_residual_active": 1.0,
            "set_conditioned_residual_rank": 8.0,
            "set_conditioned_residual_abs_mean": 0.1,
            "set_conditioned_residual_scale": 1.0,
            "action_query_mode_all_valid": 1.0,
            "valid_action_count": 2,
            "queried_action_count": 2,
            "queried_valid_action_fraction": 1.0,
        },
    )
    assert result.values["set_conditioned_residual_active"] == 1.0
    assert result.values["set_conditioned_residual_rank"] == 8.0
    assert result.values["teacher_scalar_cost_delta"] == -5.0
    assert result.values["teacher_nonnegative_scalar_regret"] == 0.0
    assert result.values["action_query_mode_all_valid"] == 1.0
    assert result.values["queried_valid_action_fraction"] == 1.0
