from __future__ import annotations

from pathlib import Path

import torch

from bdse.config import load_config
from bdse.model import losses as losses_mod
from bdse.tools.validate_v64_pipeline_config import _check


def test_hap_bdmu_exact_utility_projection_uses_runtime_hab_policy() -> None:
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
    }
    # Utility strongly favors the structural atom at idx0, but the deployed
    # decision budget bypasses structural safety. Group-aware interaction reserve
    # must retain distinct agent groups 10 and 20 (idx3 and idx5).
    utility = torch.tensor([[100.0, 9.0, 8.0, 7.0, 6.5, 6.0]], dtype=torch.float32)
    family_ids = torch.tensor([[1, 4, 4, 2, 2, 2]], dtype=torch.long)
    batch = {
        "evidence_active": torch.ones_like(utility, dtype=torch.bool),
        "evidence_budget_costs": torch.ones_like(utility),
        "evidence_family_ids": family_ids,
        "evidence_features": torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        "evidence_agent_group_ids": torch.tensor([[-1, -1, -1, 10, 10, 20]], dtype=torch.long),
    }
    mask = losses_mod._runtime_hab_topm_mask_from_scores(utility, None, batch, cfg)
    selected = set(torch.nonzero(mask[0], as_tuple=False).squeeze(1).tolist())
    assert selected == {1, 3, 5}



def test_hap_bdmu_utility_projection_respects_frozen_family_slots() -> None:
    cfg = {
        "evidence": {"budget": 2},
        "selector": {
            "proposal_top_m": 3,
            "hab_enabled": True,
            "hab_reserve_fraction": 0.0,
            "min_soft_interaction_topm_slots": 0,
            "decision_budget_excludes_structural_safety": False,
            "force_hard_topm": False,
        },
    }
    # Family 2 has the globally largest utility atoms, but the frozen family
    # gate strongly prefers family 1. With M=3 the exact HAB slot allocation is
    # therefore two atoms from family 1 and one from family 2.  An unconstrained
    # utility Top-M would incorrectly pick all three family-2 atoms.
    utility = torch.tensor([[9.0, 8.0, 7.0, 100.0, 99.0, 98.0]], dtype=torch.float32)
    family_ids = torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.long)
    family_logits = torch.tensor([[0.0, 10.0, -10.0, -20.0, -20.0, -20.0]], dtype=torch.float32)
    batch = {
        "evidence_active": torch.ones_like(utility, dtype=torch.bool),
        "evidence_budget_costs": torch.ones_like(utility),
        "evidence_family_ids": family_ids,
        "evidence_features": torch.zeros((1, 6, 2), dtype=torch.float32),
        "evidence_agent_group_ids": torch.full((1, 6), -1, dtype=torch.long),
    }
    mask = losses_mod._runtime_hab_topm_mask_from_scores(utility, family_logits, batch, cfg)
    selected = set(torch.nonzero(mask[0], as_tuple=False).squeeze(1).tolist())
    assert selected == {0, 1, 3}
    assert selected != {3, 4, 5}

def test_v64_3_10_config_uses_feasible_projection_and_frozen_value() -> None:
    path = Path("bdse/configs/v64_3_10_cc_aocc_hap_bdmu_daepc_train_2gpu.yaml")
    cfg = load_config(str(path))
    util = cfg["training"]["budgeted_decisive_margin_utility"]
    assert cfg["evidence"]["budget"] == 16
    assert cfg["training"]["trainable_modules"] == ["critical_proposal_adapter"]
    assert util["admission_projection_mode"] == "exact_hab_utility"
    assert util["feasible_admission_rank_weight"] > 0.0
    assert util["feasible_admission_same_family"] is True
    assert util["topm_swap_rank_weight"] == 0.0
    assert util["listwise_weight"] < util["feasible_admission_rank_weight"]
    report = _check(path, "train", "v64.3.10")
    assert report["pass"], report["failures"]
