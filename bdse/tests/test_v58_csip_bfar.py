from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from bdse.experiments.train import _optimizer_parameter_groups, _validation_competitive_score
from bdse.planner.tournament import run_pair_conditioned_tournament
from bdse.tools.check_v58_csip_bfar_dbap_gate import (
    _evaluation_config_health,
    _training_health,
)


def _runtime_cfg(residual_epsilon: float) -> dict:
    return {
        "model": {"pair_margin_normalized": False},
        "runtime": {
            "pair_tournament_anchor_mode": "selected_local",
            "pair_tournament_aggregation_mode": "evidence_action_potential",
            "pair_action_anchor_guard": {
                "enabled": True,
                "flip_margin": 0.015,
                "score_margin": 0.0,
            },
            "dual_certificate": {
                "enabled": True,
                "require_evidence_certificate_before_residual_flip": True,
                "min_evidence_certificate_fraction_for_residual_flip": 1.0,
                "residual_epsilon_cal": residual_epsilon,
            },
        },
        "selector": {"progress_rivals": 0, "maneuver_rivals": 0},
        "tournament": {
            "L_infer": 2,
            "epsilon_cal": 0.0,
            "use_softmin": True,
            "softmin_tau": 1.0,
            "beta_uncertainty": 1.0,
            "hard_filter_unsafe_actions": False,
            "utility_refinement": {"enabled": False},
        },
    }


def test_residual_conformal_epsilon_is_part_of_deployment_guard() -> None:
    j0 = np.asarray([0.0, 0.3], dtype=np.float32)
    local = np.zeros((1, 2), dtype=np.float32)
    residual = np.asarray([[0.0, -0.32]], dtype=np.float32)
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    common = dict(
        predicted_atom_costs=local,
        residual_action_potential=residual,
        residual_action_variance=np.zeros_like(residual),
        evidence_certificate_fraction=1.0,
    )
    uncalibrated = run_pair_conditioned_tournament(
        j0,
        np.zeros((1, 1), dtype=np.float32),
        pairs,
        [0],
        np.ones((2,), dtype=bool),
        np.zeros((2,), dtype=bool),
        _runtime_cfg(0.0),
        **common,
    )
    assert uncalibrated.action_index == 1
    assert bool(uncalibrated.diagnostics["pair_action_anchor_guard_allowed_flip"])

    calibrated = run_pair_conditioned_tournament(
        j0,
        np.zeros((1, 1), dtype=np.float32),
        pairs,
        [0],
        np.ones((2,), dtype=bool),
        np.zeros((2,), dtype=bool),
        _runtime_cfg(0.02),
        **common,
    )
    assert calibrated.action_index == 0
    assert float(calibrated.diagnostics["pair_action_anchor_residual_epsilon_cal"]) == 0.02
    assert not bool(calibrated.diagnostics["pair_action_anchor_guard_margin_certificate_pass"])


def test_v58_calibration_application_keeps_action_rule_separate(tmp_path: Path, monkeypatch) -> None:
    from bdse.tools.apply_v58_dual_calibration import main

    config = tmp_path / "in.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "selector": {"adverse_certificate_epsilon": 0.0},
                "tournament": {"epsilon_cal": 0.05, "beta_uncertainty": 1.0},
                "calibration": {"independent": False},
                "runtime": {
                    "dual_certificate": {"residual_epsilon_cal": 0.0},
                    "disable_pair_residual_intervention": False,
                },
            }
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "cal.json"
    calibration.write_text(
        json.dumps(
            {
                "recommended_adverse_certificate_epsilon": 0.03,
                "recommended_residual_flip_epsilon": 0.02,
                "independent_calibration": True,
            }
        ),
        encoding="utf-8",
    )

    candidate = tmp_path / "candidate.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["apply", "--config", str(config), "--calibration-json", str(calibration), "--output", str(candidate)],
    )
    main()
    cand = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    assert cand["selector"]["adverse_certificate_epsilon"] == 0.03
    assert cand["selector"]["adverse_certificate_calibrated"] is True
    assert cand["calibration"]["independent"] is True
    assert cand["tournament"]["epsilon_cal"] == 0.05
    assert cand["runtime"]["dual_certificate"]["residual_epsilon_cal"] == 0.02
    assert cand["runtime"]["disable_pair_residual_intervention"] is False

    control = tmp_path / "control.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply",
            "--config",
            str(config),
            "--calibration-json",
            str(calibration),
            "--output",
            str(control),
            "--control",
        ],
    )
    main()
    ctl = yaml.safe_load(control.read_text(encoding="utf-8"))
    assert ctl["selector"]["adverse_certificate_epsilon"] == 0.03
    assert ctl["runtime"]["dual_certificate"]["residual_epsilon_cal"] == 0.0
    assert ctl["runtime"]["disable_pair_residual_intervention"] is True


def test_evaluation_config_health_detects_beta_and_control_mismatch(tmp_path: Path) -> None:
    base = {
        "selector": {
            "adverse_certificate_epsilon": 0.03,
            "adverse_certificate_calibrated": True,
        },
        "calibration": {"independent": True},
        "tournament": {"epsilon_cal": 0.05, "beta_uncertainty": 1.0},
        "runtime": {
            "dual_certificate": {"residual_epsilon_cal": 0.02},
            "disable_pair_residual_intervention": False,
        },
    }
    paths = []
    for label in ("candidate", "local", "foundation"):
        cfg = json.loads(json.dumps(base))
        if label != "candidate":
            cfg["runtime"]["dual_certificate"]["residual_epsilon_cal"] = 0.0
            cfg["runtime"]["disable_pair_residual_intervention"] = True
        path = tmp_path / f"{label}.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        paths.append(path)
    train_cfg = {
        "training": {"certified_residual_winner": {"beta_uncertainty": 1.0}}
    }
    failures, stats = _evaluation_config_health(*paths, train_cfg)
    assert failures == []
    assert stats["checked"] is True

    bad = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    bad["runtime"]["disable_pair_residual_intervention"] = False
    paths[1].write_text(yaml.safe_dump(bad), encoding="utf-8")
    failures, _ = _evaluation_config_health(*paths, train_cfg)
    assert any("local control did not disable residual" in item for item in failures)


def test_gate_requires_certified_winner_loss_to_execute(tmp_path: Path) -> None:
    train_log = tmp_path / "train.jsonl"
    train_log.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss": 1.0,
                "selector_exact_fraction": 0.02,
                "training_pair_fraction": 0.5,
                "action_family_enabled": 1.0,
                "L_pair_full_action": 0.0,
                "L_certified_residual_winner": 0.0,
                "L_deploy_select": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = {
        "training": {
            "loss_weights": {
                "certified_residual_winner": 18.0,
                "deployment_selection": 6.0,
            }
        }
    }
    failures, _ = _training_health(train_log, 0.015, cfg)
    assert any("no non-zero winner-level supervision observed" in item for item in failures)


def test_competitive_checkpoint_score_rewards_actual_winner_gain() -> None:
    baseline = {
        "val_teacher_action_match": 0.14,
        "val_selected_local_anchor_action_match": 0.14,
        "val_pair_full_interface_action_match": 0.14,
        "val_beneficial_pair_potential_intervention_rate": 0.0,
        "val_harmful_pair_potential_intervention_rate": 0.0,
        "val_selected_decisive_atom_recall": 0.61,
        "val_selected_interaction_decisive_recall": 0.58,
        "val_fallback_would_trigger_rate": 0.11,
    }
    improved = dict(baseline)
    improved["val_teacher_action_match"] = 0.146
    improved["val_pair_full_interface_action_match"] = 0.146
    improved["val_beneficial_pair_potential_intervention_rate"] = 0.006
    assert _validation_competitive_score(improved) > _validation_competitive_score(baseline)


def test_residual_head_can_receive_higher_lr_without_changing_selector_lr() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.residual_action_head = torch.nn.Linear(2, 2)
            self.residual_action_var_head = torch.nn.Linear(2, 2)
            self.proposal_head = torch.nn.Linear(2, 1)

    model = Tiny()
    cfg = {
        "training": {
            "lr_multipliers": {
                "residual_action_head": 5.0,
                "residual_action_var_head": 2.0,
                "proposal_head": 1.0,
            }
        }
    }
    flat, groups = _optimizer_parameter_groups(model, cfg, base_lr=1e-5, is_main=False)
    assert len(flat) == len(list(model.parameters()))
    lrs = sorted(float(group["lr"]) for group in groups)
    assert lrs == [1e-5, 2e-5, 5e-5]


def test_v58_configs_align_training_and_deployment_certificate() -> None:
    root = Path(__file__).parents[1] / "configs"
    train = yaml.safe_load((root / "v58_csip_bfar_dbap_train_2gpu.yaml").read_text(encoding="utf-8"))
    candidate = yaml.safe_load((root / "v58_csip_bfar_dbap_cl.yaml").read_text(encoding="utf-8"))
    cert = train["training"]["certified_residual_winner"]
    assert train["training"]["loss_weights"]["certified_residual_winner"] > 0.0
    assert cert["residual_epsilon_reserve"] > 0.0
    assert cert["beta_uncertainty"] == candidate["tournament"]["beta_uncertainty"] == 1.0
    assert train["training"]["lr_multipliers"]["residual_action_head"] > 1.0
