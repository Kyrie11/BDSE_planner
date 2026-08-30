from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from bdse.config import load_config
from bdse.experiments.train import _configure_trainable_modules, _reinitialize_modules_after_warm_start
from bdse.model.bdse_model import BDSEModel
from bdse.model.residual_gate import confidence_shrunk_residual_pair_delta_numpy
from bdse.planner.tournament import _pair_delta_margin_matrix


FAR_CFG = {
    "max_residual_weight": 0.50,
    "min_residual_weight": 0.0,
    "variance_tau": 0.15,
    "boundary_tau": 0.35,
    "min_boundary_trust": 0.03,
    "disagreement_penalty": 0.0,
    "magnitude_ratio_tau": 1.5,
    "aggregate_max_correction_ratio": 0.8,
    "aggregate_abs_cap": 0.04,
    "aggregate_preserve_sign_ratio": 0.95,
    "flip_confidence_beta": 1.75,
    "flip_margin": 0.04,
    "allow_confident_flips": True,
    "confident_flip_cap_ratio": 1.75,
}


def test_far_foundation_margin_cannot_be_flipped_by_evidence_only_disagreement() -> None:
    local = np.asarray([[0.01]], dtype=np.float32)
    residual = np.asarray([[-10.0]], dtype=np.float32)
    variance = np.zeros_like(residual)
    base = np.asarray([2.0], dtype=np.float32)
    combined, diag = confidence_shrunk_residual_pair_delta_numpy(
        local, residual, variance, FAR_CFG, base_margin=base
    )
    final_margin = base + combined.sum(axis=0)
    assert float(final_margin[0]) > 0.0
    assert diag["residual_pair_flip_allowed_rate"] == 0.0


def test_certified_near_boundary_correction_can_flip_full_anchor() -> None:
    local = np.asarray([[0.10]], dtype=np.float32)
    residual = np.asarray([[-2.0]], dtype=np.float32)
    variance = np.zeros_like(residual)
    base = np.asarray([0.0], dtype=np.float32)
    combined, diag = confidence_shrunk_residual_pair_delta_numpy(
        local, residual, variance, FAR_CFG, base_margin=base
    )
    final_margin = base + combined.sum(axis=0)
    assert float(final_margin[0]) < 0.0
    assert diag["residual_pair_confident_flip_rate"] == 1.0
    assert diag["residual_pair_flip_allowed_rate"] == 1.0


def test_nonfinite_invalid_candidates_do_not_create_nan_pair_margins() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matrix = _pair_delta_margin_matrix(
            predicted_base_cost=np.asarray([0.0, np.inf, 2.0], dtype=np.float32),
            pair_indices=np.asarray([[0, 2], [0, 1]], dtype=np.int64),
            pair_atom_delta=np.asarray([[0.2, 5.0]], dtype=np.float32),
            selected_atoms=[0],
            valid_mask=np.asarray([True, False, True]),
            normalize_margins=True,
            norm_min_scale=1.0,
        )
    assert np.isfinite(matrix).all()
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert matrix[1, 0] <= -1e8 and matrix[0, 1] <= -1e8
    np.testing.assert_allclose(matrix[0, 2], -matrix[2, 0], rtol=0, atol=1e-6)


def test_warm_start_reset_and_freeze_preserve_foundation_modules() -> None:
    cfg_path = Path(__file__).parents[1] / "configs" / "v51_far_dbap_train_2gpu.yaml"
    cfg = load_config(str(cfg_path))
    torch.manual_seed(3)
    model = BDSEModel(cfg)
    base_before = [p.detach().clone() for p in model.base_head.parameters()]
    pair_before = [p.detach().clone() for p in model.pair_head.parameters()]
    reset = _reinitialize_modules_after_warm_start(model, cfg, is_main=False)
    assert reset == ["pair_head", "pair_var_head"]
    assert all(torch.equal(a, b) for a, b in zip(base_before, model.base_head.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(pair_before, model.pair_head.parameters()))

    _configure_trainable_modules(model, cfg, is_main=False)
    assert all(not p.requires_grad for p in model.base_head.parameters())
    assert all(not p.requires_grad for p in model.local_head.parameters())
    assert all(p.requires_grad for p in model.pair_head.parameters())
    assert all(p.requires_grad for p in model.proposal_head.parameters())


def test_latency_is_separate_from_algorithm_gate_unless_enforced(tmp_path: Path, monkeypatch) -> None:
    import json
    import sys

    from bdse.tools.check_v51_far_dbap_gate import main as gate_main

    candidate = {
        "teacher_action_match": 0.30,
        "evidence_sufficiency": 0.08,
        "pair_sign_acc_winner_rival": 0.70,
        "pair_sign_acc_near_tie": 0.60,
        "local_pair_full_interface_action_match": 0.28,
        "pair_full_interface_action_match": 0.30,
        "harmful_residual_intervention_rate": 0.01,
        "beneficial_residual_intervention_rate": 0.02,
        "selector_aocc_certified_pair_fraction": 0.70,
        "selector_aocc_frontier_retained_weight_fraction": 0.70,
        "decision_budget_atom_count": 16.0,
        "configured_decision_budget_atom_count": 16.0,
        "selector_interaction_family_selected": 10.0,
        "fallback_would_trigger_rate": 0.10,
        "selector_aocc_bound_calibrated": 1.0,
        "selector_aocc_exact_tournament_target_active": 1.0,
        "planner_latency_ms_p95": 1000.0,
    }
    local = {"teacher_action_match": 0.29, "pair_full_interface_action_match": 0.28}
    foundation = {
        "teacher_action_match": 0.25,
        "evidence_sufficiency": 0.06,
        "pair_sign_acc_winner_rival": 0.65,
        "pair_sign_acc_near_tie": 0.55,
    }
    paths = {}
    for name, data in [("candidate", candidate), ("local", local), ("foundation", foundation)]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        paths[name] = path
        rows = tmp_path / f"{name}.jsonl"
        regret = 1.0 if name == "candidate" else 2.0
        rows.write_text(json.dumps({"scenario_token": "s", "timestamp_us": 1, "teacher_regret": regret}) + "\n", encoding="utf-8")
        paths[f"{name}_rows"] = rows
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"epoch": 0, "selector_exact_fraction": 1.0}) + "\n", encoding="utf-8")

    base_argv = [
        "gate",
        str(paths["candidate"]),
        str(paths["local"]),
        str(paths["foundation"]),
        "--candidate-jsonl", str(paths["candidate_rows"]),
        "--local-control-jsonl", str(paths["local_rows"]),
        "--foundation-control-jsonl", str(paths["foundation_rows"]),
        "--train-log", str(train),
        "--latency-target-ms", "500",
    ]
    monkeypatch.setattr(sys, "argv", base_argv)
    assert gate_main() == 0
    monkeypatch.setattr(sys, "argv", base_argv + ["--enforce-latency"])
    assert gate_main() == 3
