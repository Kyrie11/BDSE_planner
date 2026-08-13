from __future__ import annotations

import numpy as np
import torch

from bdse.config import load_config
from bdse.data.nuplan_dataset import _uniform_block_indices
from bdse.model import losses as losses_mod
from bdse.model.decisive_margin_utility import (
    BDMUConfig,
    budgeted_decisive_margin_utility_numpy,
    budgeted_decisive_margin_utility_torch,
)
from bdse.planner.hab import select_topm_atoms_hab
from bdse.planner.selector import finalize_runtime_topm_policy
from bdse.tools.validate_v64_pipeline_config import _check


def test_adaptive_frontier_matches_numpy_and_expands_only_near_rivals() -> None:
    teacher_cost = torch.tensor([[0.0, 0.10, 0.15, 0.50, 1.00]], dtype=torch.float32)
    teacher_g = torch.zeros((1, 2, 5), dtype=torch.float32)
    active = torch.tensor([[True, True]])
    valid = torch.ones((1, 5), dtype=torch.bool)
    ref = torch.tensor([[True, False]])
    costs = torch.ones((1, 2), dtype=torch.float32)
    cfg = BDMUConfig(
        budget=1.0,
        rival_mode="adaptive_frontier",
        rival_min_count=1,
        rival_max_count=4,
        frontier_margin_floor=0.12,
        frontier_margin_multiplier=2.0,
        min_action_scale=1.0,
        worst_rival_weight=0.35,
    )
    u_t, d_t = budgeted_decisive_margin_utility_torch(
        teacher_cost, teacher_g, active, valid, torch.tensor([0]), ref, costs, cfg
    )
    u_n, d_n = budgeted_decisive_margin_utility_numpy(
        teacher_cost[0].numpy(), teacher_g[0].numpy(), active[0].numpy(), valid[0].numpy(),
        0, ref[0].numpy(), costs[0].numpy(), cfg,
    )
    np.testing.assert_allclose(u_t[0].numpy(), u_n, rtol=1e-6, atol=1e-6)
    assert int(d_t["frontier_count"][0].item()) == 2
    assert int(d_n["frontier_count"]) == 2


def test_v64_3_9_config_is_strict_frozen_value_acquisition_isolation() -> None:
    path = 'bdse/configs/v64_3_9_cc_aocc_af_bdmu_daepc_train_2gpu.yaml'
    cfg = load_config(path)
    util = cfg['training']['budgeted_decisive_margin_utility']
    assert cfg['evidence']['budget'] == 16
    assert cfg['training']['trainable_modules'] == ['critical_proposal_adapter']
    assert cfg['training']['loss_weights']['budgeted_decisive_margin_utility'] == 1.0
    assert cfg['training']['loss_weights']['exact_winner_flip_critical_proposal'] == 0.0
    assert cfg['training']['exact_winner_flip_criticality']['enabled'] is False
    assert util['rival_mode'] == 'adaptive_frontier'
    assert util['rival_min_count'] == 4 and util['rival_max_count'] == 8
    assert util['worst_rival_weight'] > 0.0
    assert util['topm_swap_rank_weight'] > 0.0
    report = _check(__import__('pathlib').Path(path), 'train', 'v64.3.9')
    assert report['pass'], report['failures']


def test_uniform_blocks_covers_whole_validation_order_without_first_prefix_bias() -> None:
    idx = _uniform_block_indices(n=1000, cap=500, block_size=32)
    assert len(idx) == 500
    assert len(np.unique(idx)) == 500
    assert int(idx.min()) == 0
    assert int(idx.max()) > 900
    # The old prefix cap had every sample below 500. The replacement must draw
    # substantial support from both halves while preserving short local blocks.
    assert int((idx < 500).sum()) > 150
    assert int((idx >= 500).sum()) > 150
    assert int((np.abs(np.diff(idx)) == 1).sum()) > 300


def _af_bdmu_topm_parity_fixture() -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture where group-aware runtime reservation differs from the old fast surrogate."""
    cfg = {
        "evidence": {"budget": 1},
        "selector": {
            "proposal_top_m": 3,
            "hab_enabled": False,
            "interaction_family_ids": [2, 3],
            "min_soft_interaction_topm_slots": 2,
            "decision_budget_excludes_structural_safety": True,
            "structural_safety_include_feasibility": True,
            "force_hard_topm": True,
        },
        "training": {"loss_weights": {"budgeted_decisive_margin_utility": 1.0}},
    }
    # idx0 is structural feasibility. idx3/idx4 belong to the same agent group,
    # idx5 is a second interaction group. Runtime must reserve idx3+idx5, not
    # simply the two highest interaction logits idx3+idx4.
    logits = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.5, 6.0]], dtype=torch.float32)
    active = torch.ones_like(logits, dtype=torch.bool)
    costs = torch.ones_like(logits)
    family_ids = torch.tensor([[1, 4, 4, 2, 2, 2]], dtype=torch.long)
    features = torch.zeros((1, 6, 2), dtype=torch.float32)
    group_ids = torch.tensor([[-1, -1, -1, 10, 10, 20]], dtype=torch.long)
    return cfg, logits, active, costs, family_ids, features, group_ids


def test_runtime_topm_policy_is_structural_first_and_group_aware() -> None:
    cfg, logits, active, costs, family_ids, features, group_ids = _af_bdmu_topm_parity_fixture()
    initial, _, _ = select_topm_atoms_hab(
        logits[0].numpy(),
        family_ids[0].numpy(),
        active[0].numpy(),
        costs[0].numpy(),
        budget=1.0,
        proposal_top_m=3,
        enabled=False,
    )
    final, mandatory, soft, diag = finalize_runtime_topm_policy(
        initial,
        proposal_scores=logits[0].numpy(),
        family_ids=family_ids[0].numpy(),
        active_mask=active[0].numpy(),
        max_size=3,
        selector_cfg=cfg["selector"],
        raw_hard_mask=features[0, :, 0].numpy() > 0.5,
        interaction_group_ids=group_ids[0].numpy(),
    )
    assert set(final.tolist()) == {1, 3, 5}
    assert bool(mandatory[0])
    assert bool(soft[3]) and bool(soft[4]) and bool(soft[5])
    assert diag["soft_interaction_topm_reserved_distinct_groups"] == 2


def test_bdmu_only_loss_uses_exact_runtime_topm_for_every_training_scene(monkeypatch) -> None:
    cfg, logits, active, costs, family_ids, features, group_ids = _af_bdmu_topm_parity_fixture()
    captured: dict[str, torch.Tensor] = {}

    def fake_bdmu_loss(outputs, batch, cfg_in, deployment_topm):
        captured["topm"] = deployment_topm.detach().cpu().clone()
        return outputs["proposal_logits"].sum() * 0.0, {}

    monkeypatch.setattr(losses_mod, "_budgeted_decisive_margin_utility_loss", fake_bdmu_loss)
    outputs = {"proposal_logits": logits.clone().requires_grad_(True)}
    batch = {
        "evidence_active": active,
        "evidence_budget_costs": costs,
        "evidence_family_ids": family_ids,
        "evidence_features": features,
        "evidence_agent_group_ids": group_ids,
    }
    result = losses_mod._compute_bdmu_only_losses(outputs, batch, cfg)
    assert set(torch.nonzero(captured["topm"][0], as_tuple=False).squeeze(1).tolist()) == {1, 3, 5}
    assert float(result["bdmu_runtime_topm_exact_fraction"].item()) == 1.0
