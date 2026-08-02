from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from bdse.model.losses import (
    _action_family_supervision_requested,
    _action_potential_logits_dispatch,
)
from bdse.planner.tournament import run_pair_conditioned_tournament


def _cfg() -> dict:
    return {
        "model": {"pair_margin_normalized": False},
        "runtime": {
            "pair_tournament_anchor_mode": "selected_local",
            "pair_tournament_aggregation_mode": "evidence_action_potential",
            "pair_action_anchor_guard": {
                "enabled": True,
                "flip_margin": 0.0,
                "score_margin": 0.0,
            },
            "dual_certificate": {
                "enabled": True,
                "evidence_uncertainty_source": "none",
                "require_evidence_certificate_before_residual_flip": True,
                "min_evidence_certificate_fraction_for_residual_flip": 1.0,
            },
        },
        "selector": {"progress_rivals": 0, "maneuver_rivals": 0},
        "tournament": {
            "L_infer": 2,
            "epsilon_cal": 0.0,
            "use_softmin": True,
            "softmin_tau": 1.0,
            "beta_uncertainty": 0.0,
            "hard_filter_unsafe_actions": False,
            "utility_refinement": {"enabled": False},
        },
        "training": {
            "potential_action_min_scale": 1.0,
            "pair_action_aggregation_mode": "evidence_action_potential",
        },
    }


def test_action_family_is_not_disabled_by_legacy_action_weight() -> None:
    assert _action_family_supervision_requested({
        "action": 0.0,
        "pair_full_action": 12.0,
        "deployment_selection": 6.0,
    })
    assert not _action_family_supervision_requested({"action": 0.0})


def test_direct_budget_potential_changes_winner_and_backpropagates() -> None:
    anchor = torch.tensor([[0.0, 0.3]], dtype=torch.float32)
    residual = torch.tensor([[[0.0, -0.8]]], dtype=torch.float32, requires_grad=True)
    selected = torch.tensor([[True]])
    valid = torch.tensor([[True, True]])
    pairs = torch.tensor([[[0, 1]]], dtype=torch.long)
    pair_delta = torch.zeros((1, 1, 1), dtype=torch.float32)
    pair_valid = torch.ones((1, 1), dtype=torch.bool)
    logits, _, _, corrected = _action_potential_logits_dispatch(
        anchor,
        pair_delta,
        pairs,
        pair_valid,
        selected,
        valid,
        residual_action_potential=residual,
        normalize_margins=False,
        cfg=_cfg(),
    )
    assert int(logits.argmax(dim=1).item()) == 1
    assert float(corrected[0, 1].detach()) < float(corrected[0, 0].detach())
    (-logits[0, 1]).backward()
    assert residual.grad is not None
    assert float(residual.grad.abs().sum()) > 0.0


def test_dual_certificate_blocks_low_evidence_residual_flip() -> None:
    cfg = _cfg()
    j0 = np.asarray([0.0, 0.3], dtype=np.float32)
    g = np.zeros((1, 2), dtype=np.float32)
    residual = np.asarray([[0.0, -0.8]], dtype=np.float32)
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    low_cert = run_pair_conditioned_tournament(
        j0,
        np.zeros((1, 1), dtype=np.float32),
        pairs,
        [0],
        np.ones((2,), dtype=bool),
        np.zeros((2,), dtype=bool),
        cfg,
        predicted_atom_costs=g,
        residual_action_potential=residual,
        residual_action_variance=np.zeros_like(residual),
        evidence_certificate_fraction=0.2,
    )
    assert low_cert.action_index == 0
    assert bool(low_cert.diagnostics["pair_action_anchor_guard_blocked_by_evidence_certificate"])
    assert not bool(low_cert.diagnostics["pair_action_anchor_guard_allowed_flip"])

    full_cert = run_pair_conditioned_tournament(
        j0,
        np.zeros((1, 1), dtype=np.float32),
        pairs,
        [0],
        np.ones((2,), dtype=bool),
        np.zeros((2,), dtype=bool),
        cfg,
        predicted_atom_costs=g,
        residual_action_potential=residual,
        residual_action_variance=np.zeros_like(residual),
        evidence_certificate_fraction=1.0,
    )
    assert full_cert.action_index == 1
    assert bool(full_cert.diagnostics["pair_action_anchor_guard_allowed_flip"])


def test_gate_detects_silent_winner_supervision_shutdown(tmp_path: Path) -> None:
    from bdse.tools.check_v57_wcdcip_bfar_dbap_gate import _training_health

    train_log = tmp_path / "train.jsonl"
    train_log.write_text(
        json.dumps({
            "epoch": 0,
            "loss": 1.0,
            "selector_exact_fraction": 0.0,
            "training_pair_fraction": 0.5,
            "L_pair_full_action": 0.0,
            "L_pair_full_winner_margin": 0.0,
            "L_budget_preserve_pair_full": 0.0,
            "L_deploy_select": 0.0,
            "action_family_enabled": 0.0,
        })
        + "\n",
        encoding="utf-8",
    )
    cfg = {
        "training": {
            "loss_weights": {
                "action": 0.0,
                "pair_full_action": 12.0,
                "deployment_selection": 6.0,
            }
        }
    }
    failures, _ = _training_health(train_log, 0.015, cfg)
    assert any("winner/deployment action-family branch never activated" in item for item in failures)
    assert any("no non-zero winner-level supervision observed" in item for item in failures)
    assert any("no non-zero exact deployment-selection distillation observed" in item for item in failures)


def test_gate_separates_metric_minimum_from_protocol_minimum(tmp_path: Path, monkeypatch) -> None:
    from bdse.tools.check_v57_wcdcip_bfar_dbap_gate import main as gate_main

    candidate = {
        "teacher_action_match": 0.30,
        "pair_full_interface_action_match": 0.30,
        "local_pair_full_interface_action_match": 0.30,
        "evidence_certificate_fraction": 0.80,
        "selector_aocc_frontier_retained_weight_fraction": 0.70,
        "proposal_decisive_atom_recall": 0.82,
        "selected_decisive_atom_recall": 0.60,
        "effective_selected_decisive_atom_recall": 0.74,
        "selected_interaction_decisive_recall": 0.55,
        "fallback_would_trigger_rate": 0.10,
        "decision_budget_atom_count": 16.0,
        "configured_decision_budget_atom_count": 16.0,
        "selector_aocc_bound_calibrated": 1.0,
        "selector_aocc_exact_tournament_target_active": 1.0,
        "planner_latency_ms_p95": 800.0,
    }
    local = {
        "teacher_action_match": 0.30,
        "pair_full_interface_action_match": 0.30,
        "local_pair_full_interface_action_match": 0.30,
    }
    foundation = {"teacher_action_match": 0.30}
    paths: dict[str, Path] = {}
    for name, summary in (("candidate", candidate), ("local", local), ("foundation", foundation)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(summary), encoding="utf-8")
        paths[name] = p
        rows = tmp_path / f"{name}.jsonl"
        rows.write_text(
            json.dumps({
                "scenario_token": "s",
                "timestamp_us": 1,
                "teacher_action": 0,
                "bdse_action": 0,
                "local_pair_full_action": 0,
                "full_action": 0,
                "teacher_regret": 1.0,
            })
            + "\n",
            encoding="utf-8",
        )
        paths[f"{name}_rows"] = rows
    train = tmp_path / "train.jsonl"
    train.write_text(
        json.dumps({
            "epoch": 0,
            "loss": 1.0,
            "selector_exact_fraction": 0.0,
            "training_pair_fraction": 0.5,
            "action_family_enabled": 0.0,
            "L_pair_full_action": 0.0,
            "L_deploy_select": 0.0,
        })
        + "\n",
        encoding="utf-8",
    )
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        yaml.safe_dump({
            "training": {
                "min_deployment_exact_fraction": 0.015,
                "loss_weights": {
                    "pair_full_action": 12.0,
                    "deployment_selection": 6.0,
                },
            }
        }),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "gate",
        str(paths["candidate"]),
        str(paths["local"]),
        str(paths["foundation"]),
        "--candidate-jsonl",
        str(paths["candidate_rows"]),
        "--local-control-jsonl",
        str(paths["local_rows"]),
        "--foundation-control-jsonl",
        str(paths["foundation_rows"]),
        "--train-log",
        str(train),
        "--train-config",
        str(train_cfg),
        "--report-json",
        str(report),
    ])
    assert gate_main() == 3
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["protocol_pass"] is False
    assert result["minimum_metrics_pass"] is True
    assert result["minimum_pass"] is False


def test_v57_config_trains_winner_and_uncertainty_heads() -> None:
    cfg_path = Path(__file__).parents[1] / "configs" / "v57_wcdcip_bfar_dbap_train_2gpu.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    lw = cfg["training"]["loss_weights"]
    assert lw["action"] == 0.0
    assert lw["residual_winner_correction"] > 0.0
    assert lw["residual_action_uncertainty"] > 0.0
    assert "residual_action_head" in cfg["training"]["trainable_modules"]
    assert "residual_action_var_head" in cfg["training"]["trainable_modules"]
    dual = cfg["runtime"]["dual_certificate"]
    assert dual["require_evidence_certificate_before_residual_flip"] is True
    assert dual["min_evidence_certificate_fraction_for_residual_flip"] == 1.0
