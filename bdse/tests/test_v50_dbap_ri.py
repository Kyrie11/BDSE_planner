from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bdse.experiments.train import _validation_fixed_budget_critical_score
from bdse.model.residual_gate import (
    confidence_shrunk_residual_pair_delta_numpy,
    confidence_shrunk_residual_pair_delta_torch,
)


CFG = {
    "max_residual_weight": 0.25,
    "min_residual_weight": 0.0,
    "variance_tau": 0.15,
    "boundary_tau": 0.30,
    "min_boundary_trust": 0.05,
    "disagreement_penalty": 1.0,
    "aggregate_max_correction_ratio": 0.50,
    "aggregate_abs_cap": 0.05,
    "aggregate_preserve_sign_ratio": 0.80,
    "flip_confidence_beta": 2.0,
    "flip_margin": 0.05,
    "allow_confident_flips": True,
}


def test_many_untrusted_atom_residuals_cannot_flip_local_pair() -> None:
    local = np.full((24, 1), 0.01, dtype=np.float32)
    residual = np.full((24, 1), -2.0, dtype=np.float32)
    variance = np.ones_like(residual)
    combined, diag = confidence_shrunk_residual_pair_delta_numpy(local, residual, variance, CFG)
    assert float(combined.sum()) > 0.0
    assert diag["residual_pair_flip_allowed_rate"] == 0.0


def test_numpy_and_torch_deployment_gates_match() -> None:
    rng = np.random.default_rng(5)
    local = rng.normal(size=(3, 12, 7)).astype(np.float32)
    residual = rng.normal(scale=0.4, size=(3, 12, 7)).astype(np.float32)
    variance = np.abs(rng.normal(scale=0.1, size=(3, 12, 7))).astype(np.float32)
    # NumPy runtime uses [E, P]; compare one scene at a time.
    np_rows = []
    for b in range(local.shape[0]):
        row, _ = confidence_shrunk_residual_pair_delta_numpy(local[b], residual[b], variance[b], CFG)
        np_rows.append(row)
    np_result = np.stack(np_rows, axis=0)
    torch_result = confidence_shrunk_residual_pair_delta_torch(
        torch.from_numpy(local), torch.from_numpy(residual), torch.from_numpy(variance), CFG
    ).detach().numpy()
    np.testing.assert_allclose(torch_result, np_result, rtol=2e-5, atol=2e-5)


def _metrics(pair: float, local_pair: float, harmful: float, beneficial: float) -> dict[str, float]:
    return {
        "val_teacher_action_match": 0.30,
        "val_full_interface_action_match": 0.38,
        "val_pair_full_interface_action_match": pair,
        "val_local_pair_full_interface_action_match": local_pair,
        "val_harmful_residual_intervention_rate": harmful,
        "val_beneficial_residual_intervention_rate": beneficial,
        "val_budget_vs_pair_full_match": 0.95,
        "val_pair_sign_acc_near_tie": 0.60,
        "val_pair_sign_acc_winner_rival": 0.70,
        "val_evidence_sufficiency": 0.09,
        "val_selected_hard_decisive_recall": 0.70,
        "val_selected_decisive_atom_recall": 0.60,
        "val_fallback_would_trigger_rate": 0.10,
        "val_teacher_regret": 10.0,
        "val_planner_latency_ms_p95": 450.0,
        "val_selector_interaction_family_selected": 11.0,
        "val_decision_budget_atom_count": 16.0,
        "val_configured_decision_budget_atom_count": 16.0,
    }


def test_checkpoint_score_penalizes_harmful_residual_interface() -> None:
    safe = _validation_fixed_budget_critical_score(_metrics(0.34, 0.34, 0.01, 0.02))
    harmful = _validation_fixed_budget_critical_score(_metrics(0.26, 0.34, 0.09, 0.01))
    assert safe > harmful
