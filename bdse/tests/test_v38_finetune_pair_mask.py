from __future__ import annotations

import torch

from bdse.model.losses import _predicted_pair_certificate_masks


def _case(include_group_ids: bool) -> tuple[dict, dict, dict]:
    batch_size, atoms, actions, pairs = 1, 4, 3, 2
    outputs = {
        "J0": torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32),
        "proposal_logits": torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32),
        "pair_atom_delta": torch.tensor(
            [[[[0.2, 0.1], [0.1, -0.2], [0.3, 0.1], [-0.1, 0.2]][i] for i in range(atoms)]],
            dtype=torch.float32,
        ).reshape(batch_size, atoms, pairs),
    }
    batch = {
        "pair_indices": torch.tensor([[[0, 1], [0, 2]]], dtype=torch.long),
        "pair_valid": torch.tensor([[True, True]]),
        "pair_weights": torch.ones((batch_size, pairs), dtype=torch.float32),
        "candidate_valid": torch.ones((batch_size, actions), dtype=torch.bool),
        "evidence_active": torch.ones((batch_size, atoms), dtype=torch.bool),
        "evidence_budget_costs": torch.ones((batch_size, atoms), dtype=torch.float32),
        "evidence_family_ids": torch.tensor([[2, 2, 3, 4]], dtype=torch.long),
        "runtime_safety_flags": torch.zeros((batch_size, actions), dtype=torch.bool),
        "evidence_features": torch.zeros((batch_size, atoms, 2), dtype=torch.float32),
    }
    if include_group_ids:
        batch["evidence_agent_group_ids"] = torch.tensor([[10, 10, 11, -1]], dtype=torch.long)
    cfg = {
        "evidence": {"budget": 2},
        "model": {"pair_margin_normalized": True},
        "tournament": {},
        "selector": {
            "proposal_top_m": 4,
            "hab_enabled": False,
            "selector_cap_mode": "margin_coreset",
            "soft_interaction_quota": 1,
            "min_soft_interaction_topm_slots": 2,
            "interaction_family_ids": [2, 3],
        },
    }
    return outputs, batch, cfg


def test_pair_certificate_mask_initializes_agent_group_ids() -> None:
    outputs, batch, cfg = _case(include_group_ids=True)
    mask = _predicted_pair_certificate_masks(outputs, batch, cfg)
    assert mask.shape == outputs["proposal_logits"].shape
    assert int(mask.sum()) == 2


def test_pair_certificate_mask_has_safe_missing_group_fallback() -> None:
    outputs, batch, cfg = _case(include_group_ids=False)
    mask = _predicted_pair_certificate_masks(outputs, batch, cfg)
    assert mask.shape == outputs["proposal_logits"].shape
    assert int(mask.sum()) == 2


def test_pair_certificate_mask_honors_exact_topm_override() -> None:
    outputs, batch, cfg = _case(include_group_ids=True)
    override = torch.tensor([[False, False, True, True]], dtype=torch.bool)
    oracle_scores = torch.tensor([[0.1, 0.2, 2.0, 1.0]], dtype=torch.float32)
    mask = _predicted_pair_certificate_masks(
        outputs,
        batch,
        cfg,
        topm_mask_override=override,
        proposal_scores_override=oracle_scores,
    )
    assert int(mask.sum()) == 2
    assert torch.equal(mask, override)
