from __future__ import annotations

import torch
import pytest

from bdse.experiments.train import _validate_deployment_training_schedule
from bdse.model.losses import _certificate_action_gap_loss, _deployment_budget_entries_for_step


def _cfg() -> dict:
    return {
        "evidence": {"budget": 16},
        "tournament": {"softmin_tau": 1.0},
        "training": {
            "epochs": 12,
            "batch_size": 4,
            "action_loss_start_epoch": 0,
            "predicted_selector_start_epoch": 1,
            "pair_action_loss_weight": 1.0,
            "loss_weights": {"action": 1.0},
            "deployment_budgets": [8, 16, 24],
            "deployment_budget_weights": [0.75, 1.5, 0.75],
            "deployment_budget_strategy": "primary_plus_aux",
            "deployment_primary_budget": 16,
            "deployment_selector_scenes_per_rank": 0,
            "deployment_selector_every_n_steps": 1,
            "min_deployment_exact_fraction": 1.0,
            "certificate_gap": {
                "margin": 0.06,
                "safety_margin": 0.14,
                "safe_frontier_margin": 0.16,
                "tau": 0.07,
            },
        },
    }


def test_certificate_frontier_is_finite_without_safe_action() -> None:
    cfg = _cfg()
    logits = torch.tensor([[0.2, -0.1, 0.4], [0.1, 0.3, -0.2]], dtype=torch.float32)
    target = torch.tensor([0, 1])
    valid = torch.ones_like(logits, dtype=torch.bool)
    # First scene is all-hard (the v44 NaN trigger); second is all-safe.
    hard = torch.tensor([[True, True, True], [False, False, False]])
    losses = _certificate_action_gap_loss(logits, target, valid, hard, cfg)
    assert all(torch.isfinite(value) for value in losses)
    assert losses[2].item() == pytest.approx(0.0)


def test_primary_plus_aux_always_contains_primary_and_balances_aux() -> None:
    seen_aux: list[float] = []
    for step in range(2):
        for rank in range(2):
            cfg = {
                "deployment_budget_strategy": "primary_plus_aux",
                "deployment_primary_budget": 16,
                "deployment_budget_schedule_slots": 4,
                "global_step": step,
                "global_rank": rank,
                "world_size": 2,
            }
            entries = _deployment_budget_entries_for_step([8.0, 16.0, 24.0], [0.75, 1.5, 0.75], cfg)
            assert entries[0] == (16.0, 1.5)
            assert entries[1][1] == pytest.approx(1.5)
            seen_aux.append(entries[1][0])
    assert seen_aux.count(8.0) == 2
    assert seen_aux.count(24.0) == 2


def test_exact_selector_floor_rejects_v44_quarter_coverage() -> None:
    cfg = _cfg()
    cfg["training"]["deployment_selector_scenes_per_rank"] = 2
    cfg["training"]["deployment_selector_every_n_steps"] = 2
    with pytest.raises(ValueError, match="exact-supervision fraction"):
        _validate_deployment_training_schedule(cfg)


def test_exact_selector_floor_accepts_full_coverage() -> None:
    _validate_deployment_training_schedule(_cfg())


def test_fast_backend_schedule_uses_exact_distillation_fraction() -> None:
    cfg = _cfg()
    cfg["training"].update(
        {
            "deployment_selector_backend": "hybrid_fast",
            "deployment_exact_distill_scenes_per_rank": 1,
            "deployment_exact_distill_every_n_steps": 4,
            "min_deployment_exact_fraction": 0.05,
        }
    )
    _validate_deployment_training_schedule(cfg)
    cfg["training"]["min_deployment_exact_fraction"] = 0.1
    with pytest.raises(ValueError, match="exact-supervision fraction"):
        _validate_deployment_training_schedule(cfg)


def test_fast_margin_surrogate_is_nested_and_respects_budgets() -> None:
    from bdse.model.losses import _fast_pair_margin_surrogate_masks

    torch.manual_seed(7)
    B, K, E, P = 2, 8, 32, 24
    pairs = torch.randint(0, K, (B, P, 2))
    pairs[..., 1] = (pairs[..., 0] + torch.randint(1, K, (B, P))) % K
    outputs = {
        "J0": torch.randn(B, K) * 1000.0,
        "pair_atom_delta": torch.randn(B, E, P) * 0.1,
        "proposal_logits": torch.randn(B, E),
    }
    batch = {
        "pair_indices": pairs,
        "pair_valid": torch.ones(B, P, dtype=torch.bool),
        "pair_weights": torch.rand(B, P) + 0.1,
        "candidate_valid": torch.ones(B, K, dtype=torch.bool),
        "runtime_safety_flags": torch.zeros(B, K, dtype=torch.bool),
        "evidence_active": torch.ones(B, E, dtype=torch.bool),
        "evidence_budget_costs": torch.ones(B, E),
        "evidence_family_ids": torch.randint(2, 6, (B, E)),
        "evidence_features": torch.zeros(B, E, 18),
    }
    cfg = _cfg()
    cfg["model"] = {
        "pair_margin_normalized": True,
        "margin_normalization_min_scale": 100.0,
        "margin_normalization_quantile": 0.9,
    }
    cfg["selector"] = {
        "proposal_top_m": 24,
        "decision_budget_excludes_structural_safety": True,
        "structural_safety_include_feasibility": False,
        "interaction_family_ids": [2, 3],
        "min_soft_interaction_topm_slots": 8,
        "soft_interaction_quota": 2,
        "margin_coreset_boundary_tau": 0.3,
        "margin_coreset_target_clip": 3.0,
        "margin_coreset_huber_delta": 0.25,
        "margin_coreset_residual_weight": 1.0,
        "margin_coreset_sign_weight": 1.0,
        "margin_coreset_winner_weight": 2.2,
        "margin_coreset_action_weight": 0.8,
        "force_fill_budget": True,
    }
    masks = _fast_pair_margin_surrogate_masks(outputs, batch, cfg, [4.0, 8.0, 16.0])
    assert masks[4.0].sum(dim=1).tolist() == [4, 4]
    assert masks[8.0].sum(dim=1).tolist() == [8, 8]
    assert masks[16.0].sum(dim=1).tolist() == [16, 16]
    assert torch.all(masks[4.0] <= masks[8.0])
    assert torch.all(masks[8.0] <= masks[16.0])
