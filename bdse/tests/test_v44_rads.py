from __future__ import annotations

import pytest

from bdse.experiments.train import _validate_deployment_training_schedule


def _base_cfg() -> dict:
    return {
        "training": {
            "epochs": 4,
            "action_loss_start_epoch": 0,
            "predicted_selector_start_epoch": 1,
            "pair_action_loss_weight": 1.0,
            "loss_weights": {"action": 1.0},
        }
    }


def test_rejects_oracle_only_schedule_by_default() -> None:
    cfg = _base_cfg()
    cfg["training"]["predicted_selector_start_epoch"] = 6
    with pytest.raises(ValueError, match="never action-supervised"):
        _validate_deployment_training_schedule(cfg)


def test_allows_explicit_oracle_only_ablation() -> None:
    cfg = _base_cfg()
    cfg["training"]["predicted_selector_start_epoch"] = 6
    cfg["training"]["allow_oracle_only_selector_training"] = True
    _validate_deployment_training_schedule(cfg)


def test_validates_any_budget_weights() -> None:
    cfg = _base_cfg()
    cfg["training"]["deployment_budgets"] = [8, 16, 24]
    cfg["training"]["deployment_budget_weights"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="must match"):
        _validate_deployment_training_schedule(cfg)


def test_accepts_deployment_aligned_any_budget_schedule() -> None:
    cfg = _base_cfg()
    cfg["training"]["epochs"] = 12
    cfg["training"]["deployment_budgets"] = [8, 16, 24]
    cfg["training"]["deployment_budget_weights"] = [0.75, 1.5, 0.75]
    cfg["training"]["deployment_regret_weight"] = 1.0
    _validate_deployment_training_schedule(cfg)
