from __future__ import annotations

import json
from pathlib import Path

import torch

from bdse.config import load_config
from bdse.model import losses as losses_mod
from bdse.tools.check_v64_3_12_cet_bdmu_contract import run_contract
from bdse.tools.check_v64_3_12_ret_cet_bdmu_screen import main as screen_main
from bdse.tools.validate_v64_pipeline_config import _check


def test_v64_3_12_configs_use_sampled_exact_runtime_training_projection() -> None:
    for variant in ("ret", "cet"):
        path = Path(f"bdse/configs/v64_3_12_cc_aocc_{variant}_bdmu_daepc_screen_2gpu.yaml")
        cfg = load_config(str(path))
        util = cfg["training"]["budgeted_decisive_margin_utility"]
        assert cfg["evidence"]["budget"] == 16
        assert cfg["training"]["trainable_modules"] == ["critical_proposal_adapter"]
        assert util["budget_transmission_selector_source"] == "exact_runtime_sampled"
        assert util["budget_transmission_exact_scenes_per_rank"] == 4
        assert util["budget_transmission_exact_every_n_steps"] == 1
        assert util["budget_transmission_exact_eval"] is True
        assert util["budget_transmission_same_family"] is True
        assert util["budget_transmission_cross_family_fallback"] is False
        assert util["listwise_weight"] == 0.0
        assert util["feasible_admission_rank_weight"] == 0.0
        assert util["topm_swap_rank_weight"] == 0.0
        assert util["budget_transmission_allow_controlled_budget_exchange"] is (variant == "cet")
        report = _check(path, "train", "v64.3.12")
        assert report["pass"], report["failures"]
        contract = run_contract(str(path))
        assert contract["pass"], contract["checks"]


def _minimal_exact_cfg(*, allow_exchange: bool) -> dict:
    return {
        "evidence": {"budget": 1},
        "selector": {"proposal_top_m": 3},
        "model": {"critical_proposal_adapter": {"scale": 1.0}},
        "training": {
            "global_step": 0,
            "global_rank": 0,
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
                "budget_transmission_selector_source": "exact_runtime_sampled",
                "budget_transmission_exact_scenes_per_rank": 1,
                "budget_transmission_exact_every_n_steps": 1,
                "budget_transmission_exact_eval": True,
                "budget_transmission_same_family": True,
                "budget_transmission_cross_family_fallback": False,
                "budget_transmission_protect_current_budget": True,
                "budget_transmission_allow_controlled_budget_exchange": allow_exchange,
                "budget_transmission_controlled_exchange_weight": 0.5,
                "budget_transmission_margin": 0.35,
                "budget_transmission_positive_k": 4,
                "budget_transmission_negative_k": 4,
                "min_action_scale": 1.0,
                "residual_l2_weight": 0.0,
            },
        },
    }


def _run_exchange_fixture(monkeypatch, *, allow_exchange: bool):
    cfg = _minimal_exact_cfg(allow_exchange=allow_exchange)
    logits = torch.tensor([[1.0, 0.8, 0.2, 0.7]], dtype=torch.float32, requires_grad=True)
    deployment_topm = torch.tensor([[True, True, False, True]])
    outputs = {
        "J0": torch.zeros((1, 2)),
        "proposal_logits": logits,
        "critical_proposal_residual_logits": torch.zeros_like(logits),
        "pair_atom_delta": torch.zeros((1, 4, 1)),
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

    # Surrogate masks intentionally disagree with the exact B intervention. They
    # may be used for diagnostics/reference, but never as the training target.
    def fake_fast(*args, **kwargs):
        budget = args[3][0]
        return {budget: torch.tensor([[False, True, False, False]])}

    calls = {"exact": 0}

    def fake_exact(*args, **kwargs):
        calls["exact"] += 1
        # Current exact B transmits idx0; utility-oracle exact B transmits idx2.
        return (
            torch.tensor([[True, False, False, False]])
            if calls["exact"] == 1
            else torch.tensor([[False, False, True, False]])
        )

    monkeypatch.setattr(losses_mod, "_fast_pair_margin_surrogate_masks", fake_fast)
    monkeypatch.setattr(losses_mod, "_predicted_pair_certificate_masks", fake_exact)
    loss, diag = losses_mod._budgeted_decisive_margin_utility_loss(outputs, batch, cfg, deployment_topm)
    return logits, loss, diag, calls


def test_ret_uses_exact_runtime_target_during_grad_and_keeps_blanket_B_protection(monkeypatch) -> None:
    logits, loss, diag, calls = _run_exchange_fixture(monkeypatch, allow_exchange=False)
    assert calls["exact"] == 2
    assert float(diag["bdmu_budget_projection_exact_fraction"].item()) == 1.0
    assert float(diag["bdmu_budget_selector_surrogate_jaccard_current"].item()) < 1.0
    # The only realizable negative is current-B idx0, so RET deliberately blocks it.
    assert float(diag["bdmu_budget_transmission_pairs"].item()) == 0.0
    assert float(diag["bdmu_budget_protected_negative_fraction"].item()) == 1.0
    assert float(diag["bdmu_budget_controlled_exchange_negative_fraction"].item()) == 1.0
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))


def test_cet_unlocks_only_exact_controlled_current_B_exchange(monkeypatch) -> None:
    logits, loss, diag, calls = _run_exchange_fixture(monkeypatch, allow_exchange=True)
    assert calls["exact"] == 2
    assert float(diag["bdmu_budget_transmission_pairs"].item()) == 1.0
    assert float(diag["bdmu_budget_controlled_exchange_pair_fraction"].item()) == 1.0
    assert float(diag["bdmu_budget_protected_negative_fraction"].item()) == 0.0
    loss.backward()
    # Gradient descent raises exact-oracle transmitted idx2 and lowers only the
    # exact-current-B atom idx0 that the oracle-B intervention explicitly drops.
    assert float(logits.grad[0, 2]) < 0.0
    assert float(logits.grad[0, 0]) > 0.0
    assert abs(float(logits.grad[0, 1])) < 1e-8
    assert abs(float(logits.grad[0, 3])) < 1e-8


def _screen_row(epoch: int, *, budget_capture: float, train_exact: float | None = None) -> dict:
    row = {
        "epoch": epoch,
        "val_teacher_action_match": 0.18,
        "val_teacher_regret": 100.0,
        "val_pair_full_interface_action_match": 0.18,
        "val_teacher_exact_winner_flip_critical_recall_topm_micro": 0.25,
        "val_teacher_exact_winner_flip_critical_recall_selected_micro": 0.15,
        "val_proposal_decisive_atom_recall": 0.75,
        "val_evidence_certificate_fraction": 0.93,
        "val_bdmu_current_topm_utility_capture": 0.48,
        "val_bdmu_hab_oracle_topm_utility_capture": 0.50,
        "val_bdmu_current_budget_utility_capture": budget_capture,
        "val_bdmu_oracle_budget_utility_capture": 0.40,
        "val_bdmu_budget_transmission_gap": 0.04,
        "val_bdmu_reference_selected_utility_capture": 0.35,
        "val_bdmu_budget_transmission_rank_loss": 0.5,
        "val_bdmu_budget_transmission_pairs": 2.0,
        "val_bdmu_budget_transmission_scene_fraction": 0.06,
        "val_bdmu_budget_transmission_positive_fraction": 0.04,
        "val_bdmu_budget_protected_negative_fraction": 0.7,
        "val_bdmu_budget_projection_exact_fraction": 1.0,
        "val_bdmu_budget_projection_topm_violation_fraction": 0.0,
        "val_bdmu_budget_selector_surrogate_jaccard_current": 0.77,
        "val_bdmu_budget_selector_surrogate_jaccard_oracle": 0.77,
        "val_bdmu_budget_exact_candidate_scene_fraction": 0.2,
        "val_bdmu_budget_current_oracle_jaccard": 0.6,
        "val_bdmu_budget_controlled_exchange_negative_fraction": 0.7,
        "val_bdmu_budget_controlled_exchange_pair_fraction": 0.4,
        "val_bdmu_runtime_topm_exact_fraction": 1.0,
        "critical_adapter_parameter_delta_rms": 0.01 if epoch >= 0 else 0.0,
        "critical_proposal_residual_rms": 0.2 if epoch >= 0 else 0.0,
    }
    if epoch >= 0 and train_exact is not None:
        row.update({
            "bdmu_budget_projection_exact_fraction": train_exact,
            "bdmu_budget_exact_candidate_scene_fraction": 0.25,
            "bdmu_budget_selector_surrogate_jaccard_current": 0.75,
            "bdmu_budget_selector_surrogate_jaccard_oracle": 0.74,
            "bdmu_budget_controlled_exchange_pair_fraction": 0.5,
        })
    return row


def test_screen_ret_failure_routes_to_cet_but_cet_failure_ends_acquisition(tmp_path, monkeypatch) -> None:
    anchor = _screen_row(-1, budget_capture=0.36)
    trained = _screen_row(0, budget_capture=0.359, train_exact=0.2)
    log = tmp_path / "train.jsonl"
    log.write_text(json.dumps(anchor) + "\n" + json.dumps(trained) + "\n")

    ret_out = tmp_path / "ret.json"
    monkeypatch.setattr("sys.argv", ["x", "--train-log", str(log), "--output", str(ret_out), "--variant", "ret"])
    assert screen_main() == 0
    ret = json.loads(ret_out.read_text())
    assert ret["instrumentation_valid"] is True
    assert ret["full_promotion"] is False
    assert ret["pivot_to_value_frontier"] is False
    assert "Run CET" in ret["diagnosis"]

    cet_out = tmp_path / "cet.json"
    monkeypatch.setattr("sys.argv", ["x", "--train-log", str(log), "--output", str(cet_out), "--variant", "cet"])
    assert screen_main() == 0
    cet = json.loads(cet_out.read_text())
    assert cet["instrumentation_valid"] is True
    assert cet["exact_acquisition_exhausted"] is True
    assert cet["pivot_to_value_frontier"] is True
    assert cet["full_promotion"] is False


def test_screen_rejects_missing_exact_training_instrumentation(tmp_path, monkeypatch) -> None:
    anchor = _screen_row(-1, budget_capture=0.36)
    trained = _screen_row(0, budget_capture=0.37, train_exact=None)
    log = tmp_path / "train.jsonl"
    log.write_text(json.dumps(anchor) + "\n" + json.dumps(trained) + "\n")
    out = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["x", "--train-log", str(log), "--output", str(out), "--variant", "cet"])
    # Checker writes a report but returns 2 when the selected epoch lacks proof
    # that exact runtime projection was used in the training objective.
    assert screen_main() == 2
    report = json.loads(out.read_text())
    assert report["instrumentation_valid"] is False
