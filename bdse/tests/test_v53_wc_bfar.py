from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

from bdse.config import load_config
from bdse.experiments.train import _reinitialize_modules_after_warm_start
from bdse.model.bdse_model import BDSEModel
from bdse.model.losses import _pair_cycle_consistency_loss
from bdse.tools.check_v53_anchor_quality import main as anchor_gate_main
from bdse.tools.check_v53_wc_bfar_dbap_gate import main as candidate_gate_main


def test_v53_anchor_gate_does_not_use_budgeted_regret(tmp_path: Path, monkeypatch) -> None:
    summary = {
        "full_interface_action_match": 0.359,
        "base_pair_sign_acc_winner_rival": 0.671,
        "dense_pair_sign_acc_winner_rival": 0.803,
        "dense_pair_sign_acc_near_tie": 0.706,
        "dense_pair_sign_acc_all": 0.718,
        "teacher_regret": 999999.0,  # budgeted runtime metric: intentionally irrelevant
        "planner_latency_ms_p95": 1100.0,
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        "".join(
            json.dumps({"scenario_token": f"s{i}", "timestamp_us": i}) + "\n"
            for i in range(1000)
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["gate", str(summary_path), "--jsonl", str(rows_path), "--report-json", str(report)],
    )
    assert anchor_gate_main() == 0
    assert json.loads(report.read_text())["passed"] is True


def test_v53_safe_residual_initialization_is_exact_noop() -> None:
    cfg_path = Path(__file__).parents[1] / "configs" / "v53_wc_bfar_dbap_train_2gpu.yaml"
    cfg = load_config(str(cfg_path))
    model = BDSEModel(cfg)
    base_before = [p.detach().clone() for p in model.base_head.parameters()]
    _reinitialize_modules_after_warm_start(model, cfg, is_main=False)
    last_pair = [m for m in model.pair_head.modules() if isinstance(m, torch.nn.Linear)][-1]
    assert torch.count_nonzero(last_pair.weight).item() == 0
    assert torch.count_nonzero(last_pair.bias).item() == 0
    assert all(torch.equal(a, b) for a, b in zip(base_before, model.base_head.parameters()))


def test_v53_training_objectives_are_gradient_effective_and_sparse() -> None:
    root = Path(__file__).parents[1] / "configs"
    cfg = yaml.safe_load((root / "v53_wc_bfar_dbap_train_2gpu.yaml").read_text())
    lw = cfg["training"]["loss_weights"]
    assert lw["full_action"] == 0.0
    assert lw["full_margin"] == 0.0
    assert lw["pair_full_action"] > 0.0
    assert lw["pair_full_winner_margin"] > 0.0
    assert lw["budget_preserve_pair_full"] > 0.0
    assert cfg["training"]["cycle_consistency"]["every_n_steps"] == 4
    assert cfg["training"]["boundary_pair_sampler"]["full_last_n_steps"] == 64


def test_cycle_consistency_skips_non_cadence_steps() -> None:
    pred = torch.randn(1, 3, requires_grad=True)
    pairs = torch.tensor([[[0, 1], [1, 2], [0, 2]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    target = torch.tensor([0])
    valid = torch.ones(1, 3, dtype=torch.bool)
    cfg = {
        "training": {
            "global_step": 1,
            "current_epoch": 0,
            "steps_per_epoch": 10,
            "epochs": 2,
            "cycle_consistency": {"every_n_steps": 4, "full_last_n_steps": 0},
        }
    }
    value = _pair_cycle_consistency_loss(pred, pairs, mask, target, valid, cfg)
    assert value.item() == 0.0


def test_two_tier_gate_allows_diagnostic_cl20_before_competitive_pass(tmp_path: Path, monkeypatch) -> None:
    candidate = {
        "teacher_action_match": 0.30,
        "pair_full_interface_action_match": 0.30,
        "local_pair_full_interface_action_match": 0.30,
        "harmful_residual_intervention_rate": 0.01,
        "beneficial_residual_intervention_rate": 0.01,
        "selector_aocc_certified_pair_fraction": 0.50,
        "selector_aocc_frontier_retained_weight_fraction": 0.50,
        "proposal_decisive_atom_recall": 0.75,
        "selected_decisive_atom_recall": 0.52,
        "effective_selected_decisive_atom_recall": 0.65,
        "selected_interaction_decisive_recall": 0.45,
        "fallback_would_trigger_rate": 0.50,
        "decision_budget_atom_count": 16.0,
        "configured_decision_budget_atom_count": 16.0,
        "selector_aocc_bound_calibrated": 1.0,
        "selector_aocc_exact_tournament_target_active": 1.0,
        "planner_latency_ms_p95": 800.0,
    }
    local = {"teacher_action_match": 0.30, "pair_full_interface_action_match": 0.30}
    foundation = {"teacher_action_match": 0.30}
    paths: dict[str, Path] = {}
    for name, data in (("candidate", candidate), ("local", local), ("foundation", foundation)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        paths[name] = p
        rows = tmp_path / f"{name}.jsonl"
        rows.write_text(json.dumps({"scenario_token": "s", "timestamp_us": 1, "teacher_regret": 100.0}) + "\n")
        paths[f"{name}_rows"] = rows
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"epoch": 0, "loss": 1.0, "selector_exact_fraction": 0.125, "training_pair_fraction": 0.4}) + "\n")
    report = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gate", str(paths["candidate"]), str(paths["local"]), str(paths["foundation"]),
            "--candidate-jsonl", str(paths["candidate_rows"]),
            "--local-control-jsonl", str(paths["local_rows"]),
            "--foundation-control-jsonl", str(paths["foundation_rows"]),
            "--train-log", str(train), "--report-json", str(report),
        ],
    )
    assert candidate_gate_main() == 0
    result = json.loads(report.read_text())
    assert result["minimum_pass"] is True
    assert result["competitive_pass"] is False
