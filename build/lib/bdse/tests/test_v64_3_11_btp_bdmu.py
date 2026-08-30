from __future__ import annotations

import json
from pathlib import Path

import torch

from bdse.config import load_config
from bdse.model import losses as losses_mod
from bdse.tools.check_v64_3_11_btp_bdmu_contract import run_contract
from bdse.tools.check_v64_3_11_btp_bdmu_screen import main as screen_main
from bdse.tools.validate_v64_pipeline_config import _check


def test_v64_3_11_config_is_budget_transmission_only_and_frozen_value() -> None:
    path = Path("bdse/configs/v64_3_11_cc_aocc_btp_bdmu_daepc_train_2gpu.yaml")
    cfg = load_config(str(path))
    util = cfg["training"]["budgeted_decisive_margin_utility"]
    assert cfg["evidence"]["budget"] == 16
    assert cfg["training"]["trainable_modules"] == ["critical_proposal_adapter"]
    assert util["admission_projection_mode"] == "exact_hab_utility"
    assert util["budget_transmission_rank_weight"] > 0.0
    assert util["budget_transmission_selector_source"] == "frozen_pair_margin_surrogate"
    assert util["budget_transmission_exact_eval"] is True
    assert util["budget_transmission_same_family"] is True
    assert util["budget_transmission_cross_family_fallback"] is False
    assert util["budget_transmission_protect_current_budget"] is True
    assert util["listwise_weight"] == 0.0
    assert util["feasible_admission_rank_weight"] == 0.0
    assert util["topm_swap_rank_weight"] == 0.0
    report = _check(path, "train", "v64.3.11")
    assert report["pass"], report["failures"]
    contract = run_contract(str(path))
    assert contract["pass"], contract["checks"]


def _minimal_btp_cfg() -> dict:
    return {
        "evidence": {"budget": 1},
        "selector": {"proposal_top_m": 3},
        "model": {"critical_proposal_adapter": {"scale": 1.0}},
        "training": {
            "budgeted_decisive_margin_utility": {
                "enabled": True,
                "reference_source": "frozen_foundation_fast_budget",
                "topm_membership_source": "exact_runtime_hab",
                "reference_topm_pool_source": "exact_runtime_hab",
                "exchange_mode": "best_budget_feasible_single_exchange",
                "admission_projection_mode": "exact_hab_utility",
                "listwise_weight": 0.0,
                "topm_swap_rank_weight": 0.0,
                "feasible_admission_rank_weight": 0.0,
                "budget_transmission_rank_weight": 1.0,
                "budget_transmission_selector_source": "frozen_pair_margin_surrogate",
                "budget_transmission_same_family": True,
                "budget_transmission_cross_family_fallback": False,
                "budget_transmission_protect_current_budget": True,
                "budget_transmission_margin": 0.35,
                "budget_transmission_positive_k": 4,
                "budget_transmission_negative_k": 4,
                "min_action_scale": 1.0,
                "residual_l2_weight": 0.0,
            }
        },
    }


def test_btp_rank_updates_only_budget_transmitted_same_family_replacement(monkeypatch) -> None:
    cfg = _minimal_btp_cfg()
    logits = torch.tensor([[1.0, 0.8, 0.2, 0.7]], dtype=torch.float32, requires_grad=True)
    deployment_topm = torch.tensor([[True, True, False, True]])
    active = torch.ones_like(deployment_topm)
    outputs = {
        "J0": torch.zeros((1, 2)),
        "proposal_logits": logits,
        "critical_proposal_residual_logits": torch.zeros_like(logits),
        "pair_atom_delta": torch.zeros((1, 1, 1, 1)),
    }
    batch = {
        "teacher_J_T": torch.tensor([[0.0, 1.0]]),
        "teacher_g_evid": torch.zeros((1, 4, 2)),
        "teacher_a_star": torch.tensor([0]),
        "candidate_valid": torch.ones((1, 2), dtype=torch.bool),
        "evidence_active": active,
        "evidence_budget_costs": torch.ones((1, 4)),
        # idx2 (oracle-transmitted positive) can replace idx0 only; idx1/idx3
        # are different families and must not become broad ranking negatives.
        "evidence_family_ids": torch.tensor([[2, 3, 2, 4]]),
    }

    utility = torch.tensor([[0.10, 0.20, 1.00, 0.05]])
    target_diag = {
        "scene_has_utility": torch.tensor([True]),
        "positive_fraction": torch.tensor([0.75]),
        "weighted_deficit": torch.tensor([1.0]),
        "missed_utility": torch.tensor([1.0]),
        "total_utility": torch.tensor([1.35]),
        "frontier_count": torch.tensor([4.0]),
        "worst_deficit": torch.tensor([1.0]),
    }
    monkeypatch.setattr(
        losses_mod,
        "budgeted_decisive_margin_utility_torch",
        lambda *a, **k: (utility.clone(), target_diag),
    )
    monkeypatch.setattr(losses_mod, "_runtime_hab_topm_hard_mask", lambda *a, **k: deployment_topm.clone())
    # Utility oracle changes idx0 -> idx2 while keeping idx1/idx3.
    monkeypatch.setattr(
        losses_mod,
        "_runtime_hab_topm_mask_from_scores",
        lambda *a, **k: torch.tensor([[False, True, True, True]]),
    )
    calls = {"n": 0}

    def fake_budget_masks(*args, **kwargs):
        calls["n"] += 1
        budget = args[3][0]
        # 1: immutable reference B; 2: current learned Top-M B; 3: oracle Top-M B.
        masks = {
            1: torch.tensor([[False, True, False, False]]),
            2: torch.tensor([[False, True, False, False]]),
            3: torch.tensor([[False, False, True, False]]),
        }
        return {budget: masks[calls["n"]].clone()}

    monkeypatch.setattr(losses_mod, "_fast_pair_margin_surrogate_masks", fake_budget_masks)
    loss, diag = losses_mod._budgeted_decisive_margin_utility_loss(outputs, batch, cfg, deployment_topm)
    assert float(diag["bdmu_budget_transmission_pairs"].item()) == 1.0
    assert float(diag["bdmu_budget_transmission_scene_fraction"].item()) == 1.0
    assert float(diag["bdmu_oracle_budget_utility_capture"].item()) > float(diag["bdmu_current_budget_utility_capture"].item())
    loss.backward()
    # Gradient descent raises pos idx2 and lowers replaceable same-family neg idx0.
    assert float(logits.grad[0, 2]) < 0.0
    assert float(logits.grad[0, 0]) > 0.0
    assert abs(float(logits.grad[0, 1])) < 1e-8
    assert abs(float(logits.grad[0, 3])) < 1e-8



def test_btp_validation_metrics_require_exact_runtime_budget_projection(monkeypatch) -> None:
    cfg = _minimal_btp_cfg()
    cfg["training"]["budgeted_decisive_margin_utility"]["budget_transmission_exact_eval"] = True
    logits = torch.tensor([[1.0, 0.8, 0.2, 0.7]], dtype=torch.float32)
    deployment_topm = torch.tensor([[True, True, False, True]])
    outputs = {
        "J0": torch.zeros((1, 2)),
        "proposal_logits": logits,
        "critical_proposal_residual_logits": torch.zeros_like(logits),
        "pair_atom_delta": torch.zeros((1, 1, 1, 1)),
    }
    batch = {
        "teacher_J_T": torch.tensor([[0.0, 1.0]]),
        "teacher_g_evid": torch.zeros((1, 4, 2)),
        "teacher_a_star": torch.tensor([0]),
        "candidate_valid": torch.ones((1, 2), dtype=torch.bool),
        "evidence_active": torch.ones((1, 4), dtype=torch.bool),
        "evidence_budget_costs": torch.ones((1, 4)),
        "evidence_family_ids": torch.tensor([[2, 3, 2, 4]]),
    }
    utility = torch.tensor([[0.10, 0.20, 1.00, 0.05]])
    target_diag = {
        "scene_has_utility": torch.tensor([True]),
        "positive_fraction": torch.tensor([0.75]),
        "weighted_deficit": torch.tensor([1.0]),
        "missed_utility": torch.tensor([1.0]),
        "total_utility": torch.tensor([1.35]),
        "frontier_count": torch.tensor([4.0]),
        "worst_deficit": torch.tensor([1.0]),
    }
    monkeypatch.setattr(losses_mod, "budgeted_decisive_margin_utility_torch", lambda *a, **k: (utility.clone(), target_diag))
    monkeypatch.setattr(losses_mod, "_runtime_hab_topm_hard_mask", lambda *a, **k: deployment_topm.clone())
    monkeypatch.setattr(losses_mod, "_runtime_hab_topm_mask_from_scores", lambda *a, **k: torch.tensor([[False, True, True, True]]))
    # Fast surrogate is intentionally different from exact validation projection.
    calls = {"fast": 0, "exact": 0}
    def fake_fast(*args, **kwargs):
        calls["fast"] += 1
        budget = args[3][0]
        return {budget: torch.tensor([[True, False, False, False]])}
    def fake_exact(*args, **kwargs):
        calls["exact"] += 1
        return torch.tensor([[False, True, False, False]]) if calls["exact"] == 1 else torch.tensor([[False, False, True, False]])
    monkeypatch.setattr(losses_mod, "_fast_pair_margin_surrogate_masks", fake_fast)
    monkeypatch.setattr(losses_mod, "_predicted_pair_certificate_masks", fake_exact)
    with torch.no_grad():
        _, diag = losses_mod._budgeted_decisive_margin_utility_loss(outputs, batch, cfg, deployment_topm)
    assert calls["exact"] == 2
    assert float(diag["bdmu_budget_projection_exact_fraction"].item()) == 1.0
    assert float(diag["bdmu_budget_projection_topm_violation_fraction"].item()) == 0.0
    assert float(diag["bdmu_oracle_budget_utility_capture"].item()) > float(diag["bdmu_current_budget_utility_capture"].item())
    assert float(diag["bdmu_budget_selector_surrogate_jaccard_current"].item()) < 1.0

def test_btp_screen_pivots_to_value_when_budget_oracle_headroom_is_tiny(tmp_path, monkeypatch) -> None:
    anchor = {
        "epoch": -1,
        "val_teacher_action_match": 0.18,
        "val_teacher_regret": 100.0,
        "val_pair_full_interface_action_match": 0.18,
        "val_teacher_exact_winner_flip_critical_recall_topm_micro": 0.25,
        "val_teacher_exact_winner_flip_critical_recall_selected_micro": 0.15,
        "val_proposal_decisive_atom_recall": 0.75,
        "val_evidence_certificate_fraction": 0.93,
        "val_bdmu_current_topm_utility_capture": 0.48,
        "val_bdmu_hab_oracle_topm_utility_capture": 0.50,
        "val_bdmu_current_budget_utility_capture": 0.36,
        "val_bdmu_oracle_budget_utility_capture": 0.363,
        "val_bdmu_budget_transmission_gap": 0.003,
        "val_bdmu_reference_selected_utility_capture": 0.36,
        "val_bdmu_budget_transmission_rank_loss": 1.0,
        "val_bdmu_budget_transmission_pairs": 10.0,
        "val_bdmu_budget_transmission_scene_fraction": 0.5,
        "val_bdmu_budget_transmission_positive_fraction": 0.02,
        "val_bdmu_budget_protected_negative_fraction": 0.5,
        "val_bdmu_budget_projection_exact_fraction": 1.0,
        "val_bdmu_budget_projection_topm_violation_fraction": 0.0,
        "val_bdmu_budget_selector_surrogate_jaccard_current": 0.9,
        "val_bdmu_budget_selector_surrogate_jaccard_oracle": 0.9,
        "val_bdmu_runtime_topm_exact_fraction": 1.0,
        "critical_adapter_parameter_delta_rms": 0.0,
        "critical_proposal_residual_rms": 0.0,
    }
    trained = dict(anchor)
    trained.update({"epoch": 0, "critical_adapter_parameter_delta_rms": 0.01, "critical_proposal_residual_rms": 0.2})
    log = tmp_path / "train.jsonl"
    log.write_text(json.dumps(anchor) + "\n" + json.dumps(trained) + "\n")
    out = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["x", "--train-log", str(log), "--output", str(out)])
    assert screen_main() == 0
    report = json.loads(out.read_text())
    assert report["acquisition_capacity_not_binding"] is True
    assert report["pivot_to_value_frontier"] is True
    assert report["full_promotion"] is False
