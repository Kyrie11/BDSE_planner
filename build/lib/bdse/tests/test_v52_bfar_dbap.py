from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from bdse.experiments.train import (
    _boundary_focused_pair_subsample,
    _validate_deployment_training_schedule,
)
from bdse.tools.check_v52_anchor_quality import main as anchor_gate_main


def _batch() -> dict[str, torch.Tensor]:
    # Pair 0 contains the teacher winner; pair 1 crosses hard feasibility;
    # pair 2 is a near tie. These must outrank irrelevant far pairs.
    return {
        "teacher_a_star": torch.tensor([0]),
        "teacher_hard_violation": torch.tensor([[False, False, True, True, False]]),
        "pair_indices": torch.tensor([[[0, 4], [1, 2], [1, 4], [2, 3], [3, 4]]]),
        "pair_valid": torch.tensor([[True, True, True, True, True]]),
        "pair_margins": torch.tensor([[8.0, 5.0, 0.01, 30.0, 40.0]]),
        "pair_weights": torch.tensor([[1.0, 1.0, 1.0, 0.1, 0.1]]),
        "pair_residuals": torch.arange(5, dtype=torch.float32).reshape(1, 5),
    }


def test_boundary_sampler_preserves_flip_critical_pairs() -> None:
    cfg = {
        "training": {
            "global_step": 1,
            "current_epoch": 0,
            "epochs": 2,
            "steps_per_epoch": 10,
            "boundary_pair_sampler": {
                "enabled": True,
                "max_pairs": 3,
                "full_every_n_steps": 4,
                "full_last_n_steps": 0,
                "winner_bonus": 16.0,
                "hard_cross_bonus": 10.0,
                "near_tie_bonus": 6.0,
                "pair_weight_bonus": 3.0,
                "near_tie_tau": 0.5,
                "min_margin_scale": 1.0,
                "winner_quota": 1,
                "hard_cross_quota": 1,
                "near_tie_quota": 1,
            },
        }
    }
    out = _boundary_focused_pair_subsample(_batch(), cfg)
    kept = {tuple(x) for x in out["pair_indices"][0].tolist()}
    assert (0, 4) in kept  # teacher winner/rival
    assert (1, 2) in kept  # hard-feasibility crossing
    assert (1, 4) in kept  # near tie
    assert out["training_pair_selected_count"].item() == 3
    assert out["training_pair_original_count"].item() == 5
    assert out["training_pair_full_graph"].item() == 0


def test_boundary_sampler_keeps_full_graph_on_exact_cadence() -> None:
    cfg = {
        "training": {
            "global_step": 4,
            "current_epoch": 0,
            "epochs": 2,
            "steps_per_epoch": 10,
            "boundary_pair_sampler": {
                "enabled": True,
                "max_pairs": 2,
                "full_every_n_steps": 4,
                "full_last_n_steps": 0,
            },
        }
    }
    out = _boundary_focused_pair_subsample(_batch(), cfg)
    assert out["pair_indices"].shape[1] == 5
    assert out["training_pair_full_graph"].item() == 1


def test_v52_sparse_exact_schedule_is_valid() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    cfg = yaml.safe_load((root / "v52_bfar_dbap_train_2gpu.yaml").read_text(encoding="utf-8"))
    _validate_deployment_training_schedule(cfg)
    assert cfg["training"]["min_deployment_exact_fraction"] <= 0.0625


def test_anchor_gate_ignores_heads_reset_after_warm_start(tmp_path: Path, monkeypatch) -> None:
    # These are the actual v51 immutable-anchor diagnostics. Deliberately poor
    # direct pair/selector values must not block BFAR because those modules are reset.
    summary = {
        "full_interface_action_match": 0.359,
        "base_pair_sign_acc_winner_rival": 0.6711315,
        "dense_pair_sign_acc_winner_rival": 0.8029216,
        "dense_pair_sign_acc_near_tie": 0.7057019,
        "dense_pair_sign_acc_all": 0.7184714,
        "teacher_regret": 10808.2,
        "planner_latency_ms_p95": 1214.3,
        "pair_full_interface_action_match": 0.06,
        "evidence_sufficiency": 0.0499,
        "selector_aocc_certified_pair_fraction": 0.21,
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["check_v52_anchor_quality", str(path)])
    assert anchor_gate_main() == 0


def test_primary_budget_is_always_trained_while_aux_is_sparse() -> None:
    from bdse.model.losses import _deployment_budget_entries_for_step

    cfg = {
        "deployment_budget_strategy": "primary_plus_aux",
        "deployment_primary_budget": 16,
        "deployment_budget_schedule_slots": 4,
        "deployment_selector_every_n_steps": 4,
        "deployment_aux_every_n_exact_steps": 4,
        "global_rank": 0,
        "world_size": 2,
    }
    cfg["global_step"] = 4  # exact event 1: primary only
    assert _deployment_budget_entries_for_step([8, 16, 24], [0.75, 1.5, 0.75], cfg) == [(16.0, 1.5)]
    cfg["global_step"] = 16  # exact event 4: primary + one rotating auxiliary
    entries = _deployment_budget_entries_for_step([8, 16, 24], [0.75, 1.5, 0.75], cfg)
    assert entries[0] == (16.0, 1.5)
    assert len(entries) == 2 and entries[1][0] in {8.0, 24.0}
