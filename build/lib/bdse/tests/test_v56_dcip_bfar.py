from __future__ import annotations

import numpy as np
import torch

from bdse.model.losses import _evidence_action_potential_logits
from bdse.planner.tournament import run_pair_conditioned_tournament
from bdse.planner.nuplan_planner import _shared_model_cache_key


def _cfg() -> dict:
    return {
        "model": {"pair_margin_normalized": False},
        "runtime": {
            "pair_tournament_anchor_mode": "selected_local",
            "pair_tournament_aggregation_mode": "evidence_action_potential",
            "pair_action_anchor_guard": {"enabled": True, "flip_margin": 0.0, "score_margin": 0.0},
            "dual_certificate": {"enabled": True, "evidence_uncertainty_source": "none"},
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
        "training": {"potential_action_min_scale": 1.0},
    }


def test_evidence_action_potential_is_exactly_integrable_and_budget_attributable() -> None:
    anchor = torch.tensor([[0.2, 0.0, 0.5]], dtype=torch.float32)
    potential = torch.tensor(
        [[[0.3, -0.2, 0.1], [5.0, -5.0, 0.0]]], dtype=torch.float32, requires_grad=True
    )
    selected = torch.tensor([[True, False]])
    valid = torch.tensor([[True, True, True]])
    logits, reconstruction, cycle, corrected = _evidence_action_potential_logits(
        anchor, potential, selected, valid, normalize_margins=False,
        cfg={"training": {"potential_action_min_scale": 1.0}},
    )
    # Only the selected evidence atom can contribute. Gauge centering preserves
    # every pair difference and gives an exactly conservative action field.
    expected_phi = potential[0, 0] - potential[0, 0].mean()
    torch.testing.assert_close(corrected[0], anchor[0] + expected_phi)
    assert float(reconstruction) == 0.0
    assert float(cycle) == 0.0
    pair_cycle = ((corrected[0, 1] - corrected[0, 0]) +
                  (corrected[0, 2] - corrected[0, 1]) +
                  (corrected[0, 0] - corrected[0, 2]))
    torch.testing.assert_close(pair_cycle, torch.tensor(0.0))
    (-logits[0, 1]).backward()
    assert potential.grad is not None
    assert torch.count_nonzero(potential.grad[0, 1]) == 0


def test_zero_action_potential_matches_direct_selected_local_argmin_even_with_bad_pair_field() -> None:
    j0 = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    g = np.asarray([[0.7, 0.2, -0.4], [0.0, 0.1, 0.0]], dtype=np.float32)
    selected = [0]
    expected = int(np.argmin(j0 + g[selected].sum(axis=0)))
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    # Deliberately contradictory legacy pair field: DCIP must ignore it for the
    # final action and use the selected-local cost plus global action potential.
    bad_pair = np.asarray([[100.0, -100.0, 100.0], [-80.0, 80.0, -80.0]], dtype=np.float32)
    zero_potential = np.zeros_like(g)
    tour = run_pair_conditioned_tournament(
        j0, bad_pair, pairs, selected,
        np.ones((3,), dtype=bool), np.zeros((3,), dtype=bool), _cfg(),
        pair_atom_variance=np.full_like(bad_pair, 1.0e6),
        predicted_atom_costs=g,
        residual_action_potential=zero_potential,
        residual_action_variance=np.zeros_like(g),
    )
    assert tour.action_index == expected
    assert int(tour.diagnostics["pair_action_anchor_action"]) == expected
    assert not bool(tour.diagnostics["pair_action_anchor_deployed_flip"])
    assert float(tour.diagnostics["direct_evidence_action_potential_active"]) == 1.0


def test_evidence_action_potential_can_certifiably_correct_selected_local_winner() -> None:
    j0 = np.asarray([0.0, 0.3], dtype=np.float32)
    g = np.zeros((1, 2), dtype=np.float32)
    # Selected-local chooses action 0; evidence potential lowers action 1.
    residual = np.asarray([[0.0, -0.8]], dtype=np.float32)
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    tour = run_pair_conditioned_tournament(
        j0, np.zeros((1, 1), dtype=np.float32), pairs, [0],
        np.ones((2,), dtype=bool), np.zeros((2,), dtype=bool), _cfg(),
        predicted_atom_costs=g,
        residual_action_potential=residual,
        residual_action_variance=np.zeros_like(residual),
    )
    assert int(tour.diagnostics["pair_action_anchor_action"]) == 0
    assert tour.action_index == 1
    assert bool(tour.diagnostics["pair_action_anchor_guard_allowed_flip"])


def test_shared_model_cache_key_ignores_runtime_only_control_differences() -> None:
    cfg_a = {"model": {"hidden_dim": 32}, "runtime": {"disable_pair_residual_intervention": False}}
    cfg_b = {"model": {"hidden_dim": 32}, "runtime": {"disable_pair_residual_intervention": True}}
    assert _shared_model_cache_key("/tmp/model.pt", cfg_a, "cuda:0") == _shared_model_cache_key(
        "/tmp/model.pt", cfg_b, "cuda:0"
    )


def test_v56_config_activates_trainable_direct_evidence_potential() -> None:
    from pathlib import Path
    import yaml

    cfg_path = Path(__file__).parents[1] / "configs" / "v56_dcip_bfar_dbap_train_2gpu.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["model"]["evidence_action_residual"] is True
    assert cfg["training"]["skip_pair_head_forward"] is True
    assert cfg["training"]["loss_weights"]["residual_action_atom"] > 0.0
    assert "residual_action_head" in cfg["training"]["trainable_modules"]
    for name in (
        "v56_dcip_bfar_dbap_cl.yaml",
        "v56_dcip_bfar_dbap_local_control_cl.yaml",
        "v56_dcip_bfar_dbap_anchor_control_cl.yaml",
    ):
        runtime_cfg = yaml.safe_load((cfg_path.parent / name).read_text())
        # Keep architecture identical across the three causal controls so the
        # shared CUDA cache can reuse one checkpoint/model per process.
        assert runtime_cfg["model"]["evidence_action_residual"] is True


def test_v56_dense_forward_executes_residual_action_head() -> None:
    from pathlib import Path
    import yaml
    from bdse.model.bdse_model import BDSEModel

    cfg_path = Path(__file__).parents[1] / "configs" / "v56_dcip_bfar_dbap_train_2gpu.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    # Use a small hidden size for the unit test while preserving the V56 route.
    cfg["model"]["hidden_dim"] = 16
    cfg["model"]["scene_layers"] = 1
    cfg["model"]["scene_heads"] = 1
    model = BDSEModel(cfg)
    last = [m for m in model.residual_action_head.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(0.5)
    h = model.hidden_dim
    qdim = int(cfg["model"].get("query_feature_dim", 12))
    context = {
        "action_h": torch.zeros(1, 2, h),
        "evidence_h": torch.zeros(1, 1, h),
        "scene": torch.zeros(1, h),
        "J0": torch.zeros(1, 2),
        "evidence_valid": torch.ones(1, 1, dtype=torch.bool),
    }
    batch = {
        "candidate_valid": torch.ones(1, 2, dtype=torch.bool),
        "evidence_query_features": torch.zeros(1, 1, 2, qdim),
    }
    _, _, residual, _ = model._dense_local_from_batch(context, batch, compute_variance=False)
    torch.testing.assert_close(residual, torch.full_like(residual, 0.5))


def test_v56_gate_uses_evidence_certificate_not_residual_mixed_certificate(tmp_path, monkeypatch) -> None:
    import json
    import sys
    import yaml
    from bdse.tools.check_v56_dcip_bfar_dbap_gate import main as gate_main

    candidate = {
        "teacher_action_match": 0.30,
        "pair_full_interface_action_match": 0.30,
        "local_pair_full_interface_action_match": 0.30,
        # The legacy mixed certificate is deliberately bad.
        "selector_aocc_certified_pair_fraction": 0.10,
        # The V56 evidence-only certificate is healthy and must drive the gate.
        "evidence_certificate_fraction": 0.80,
        "residual_flip_certificate_pass": 0.0,
        "dual_certificate_deployment_certified": 1.0,
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
    paths = {}
    for name, summary in (("candidate", candidate), ("local", local), ("foundation", foundation)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(summary))
        paths[name] = p
        rows = tmp_path / f"{name}.jsonl"
        rows.write_text(json.dumps({
            "scenario_token": "s", "timestamp_us": 1,
            "teacher_action": 0, "bdse_action": 0,
            "local_pair_full_action": 0, "full_action": 0,
            "teacher_regret": 1.0,
        }) + "\n")
        paths[f"{name}_rows"] = rows
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({
        "epoch": 0, "loss": 1.0, "selector_exact_fraction": 0.02,
        "training_pair_fraction": 0.5,
    }) + "\n")
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(yaml.safe_dump({"training": {"min_deployment_exact_fraction": 0.015}}))
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "gate", str(paths["candidate"]), str(paths["local"]), str(paths["foundation"]),
        "--candidate-jsonl", str(paths["candidate_rows"]),
        "--local-control-jsonl", str(paths["local_rows"]),
        "--foundation-control-jsonl", str(paths["foundation_rows"]),
        "--train-log", str(train), "--train-config", str(train_cfg),
        "--report-json", str(report),
    ])
    assert gate_main() == 0
    result = json.loads(report.read_text())
    assert result["protocol_pass"] is True
    assert result["minimum_pass"] is True
    assert result["metrics"]["evidence_certified_fraction"] == 0.80


def test_v56_exact_selector_training_derives_local_pair_deltas_without_legacy_pair_head() -> None:
    from bdse.model.losses import _build_predicted_pair_numpy_cache, _predicted_pair_certificate_masks

    outputs = {
        "J0": torch.tensor([[0.0, 0.2]], dtype=torch.float32),
        # Evidence 0 supports action 0; evidence 1 is neutral.
        "g": torch.tensor([[[0.0, 0.5], [0.0, 0.0]]], dtype=torch.float32),
        "g_var": torch.zeros((1, 2, 2), dtype=torch.float32),
        "proposal_logits": torch.tensor([[2.0, 1.0]], dtype=torch.float32),
    }
    batch = {
        "pair_indices": torch.tensor([[[0, 1]]], dtype=torch.long),
        "pair_valid": torch.tensor([[True]]),
        "pair_weights": torch.ones((1, 1), dtype=torch.float32),
        "candidate_valid": torch.tensor([[True, True]]),
        "evidence_active": torch.tensor([[True, True]]),
        "evidence_budget_costs": torch.ones((1, 2), dtype=torch.float32),
        "evidence_family_ids": torch.zeros((1, 2), dtype=torch.long),
        "evidence_agent_group_ids": torch.full((1, 2), -1, dtype=torch.long),
        "runtime_safety_flags": torch.tensor([[False, False]]),
        "evidence_features": torch.zeros((1, 2, 4), dtype=torch.float32),
        "teacher_a_star": torch.tensor([0], dtype=torch.long),
    }
    cfg = {
        "model": {
            "pair_margin_normalized": False,
            "pair_head_residual_over_local": True,
            "evidence_action_residual": True,
        },
        "runtime": {
            "pair_delta_hybrid_local_weight": 0.0,
            "dual_certificate": {"enabled": True, "evidence_uncertainty_source": "none"},
        },
        "evidence": {"budget": 1, "max_atoms": 2},
        "selector": {
            "proposal_top_m": 2,
            "hab_enabled": False,
            "selector_cap_mode": "anytime_adverse_certificate",
            "force_fill_budget": True,
            "min_selected_atoms": 1,
            "normalized_gamma_max": 3.0,
            "normalized_eta_pred": 0.08,
            "adverse_certificate_stop_when_certified": False,
            "adverse_certificate_fill_to_budget_after_certified": True,
        },
        "tournament": {"beta_uncertainty": 0.0, "epsilon_cal": 0.0},
        "calibration": {"epsilon_cal": 0.0},
        "training": {"aocc_integrable_target_training": True},
    }
    cache = _build_predicted_pair_numpy_cache(outputs, batch, cfg)
    # The exact selector receives local evidence deltas even though the legacy
    # pair head was skipped.  For pair (0,1), evidence 0 contributes +0.5.
    np.testing.assert_allclose(cache["delta"][0, :, 0], np.asarray([0.5, 0.0], dtype=np.float32))
    mask = _predicted_pair_certificate_masks(outputs, batch, cfg, _numpy_cache=cache)
    assert mask.shape == (1, 2)
    assert int(mask.sum().item()) == 1
    assert bool(mask[0, 0])
