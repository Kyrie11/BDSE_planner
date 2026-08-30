from __future__ import annotations

import numpy as np

from bdse.experiments.train import _validation_fixed_budget_critical_score
from bdse.model.bdse_model import _confidence_shrunk_residual_pair_delta_np
from bdse.planner.selector import runtime_greedy_selector_pair_conditioned


def test_residual_trust_is_lower_at_uncertain_conflicting_boundary() -> None:
    cfg = {
        "max_residual_weight": 0.35,
        "min_residual_weight": 0.02,
        "variance_tau": 0.15,
        "boundary_tau": 0.30,
        "min_boundary_trust": 0.10,
        "disagreement_penalty": 0.85,
        "magnitude_ratio_tau": 1.5,
    }
    local = np.asarray([[1.0, 0.02]], dtype=np.float32)
    residual = np.asarray([[0.2, -1.0]], dtype=np.float32)
    variance = np.asarray([[0.0, 1.0]], dtype=np.float32)
    combined, diag = _confidence_shrunk_residual_pair_delta_np(local, residual, variance, cfg)
    trust = (combined - local) / residual
    assert trust[0, 0] > trust[0, 1]
    assert trust[0, 1] <= 0.05
    assert diag["residual_sign_disagreement_rate"] == 0.5


def test_aocc_materializes_exact_budget_with_cross_family_prefix() -> None:
    result = runtime_greedy_selector_pair_conditioned(
        predicted_base_cost=np.zeros((3,), dtype=np.float32),
        pair_atom_delta=np.asarray(
            [
                [0.8, 0.8],
                [0.7, 0.7],
                [0.6, 0.6],
                [0.2, 0.2],
                [0.1, 0.1],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        pair_indices=np.asarray([[0, 1], [0, 2]], dtype=np.int64),
        pair_weights=np.ones((2,), dtype=np.float32),
        atom_budget_costs=np.ones((6,), dtype=np.float32),
        valid_mask=np.ones((3,), dtype=bool),
        runtime_safety_flags=np.zeros((3,), dtype=bool),
        budget=4.0,
        atom_active_mask=np.ones((6,), dtype=bool),
        family_ids=np.asarray([2, 2, 2, 0, 0, 0], dtype=np.int64),
        interaction_family_ids=[2, 3],
        selector_cap_mode="anytime_adverse_certificate",
        adverse_certificate_target_action=0,
        adverse_certificate_beta=0.0,
        adverse_certificate_epsilon=0.0,
        adverse_certificate_prior_radius=0.0,
        adverse_certificate_stop_when_certified=True,
        adverse_certificate_fill_to_budget_after_certified=True,
        adverse_certificate_max_interaction_prefix_fraction=0.5,
        force_fill_budget=False,
        min_selected_atoms=0,
        soft_interaction_quota=0,
    )
    assert len(result.selected) == 4
    assert sum(int(i < 3) for i in result.selected) <= 2
    assert result.diagnostics["aocc_fill_to_budget_after_certified"] is True
    assert result.diagnostics["aocc_selected_prefix_length"] == 4


def test_checkpoint_score_requires_real_pair_and_family_diagnostics() -> None:
    complete = {
        "val_teacher_action_match": 0.30,
        "val_full_interface_action_match": 0.36,
        "val_pair_full_interface_action_match": 0.33,
        "val_budget_vs_pair_full_match": 0.95,
        "val_pair_sign_acc_near_tie": 0.60,
        "val_pair_sign_acc_winner_rival": 0.72,
        "val_evidence_sufficiency": 0.10,
        "val_selected_hard_decisive_recall": 0.65,
        "val_selected_decisive_atom_recall": 0.40,
        "val_fallback_would_trigger_rate": 0.15,
        "val_teacher_regret": 30000.0,
        "val_planner_latency_ms_p95": 480.0,
        "val_selector_interaction_family_selected": 12.0,
        "val_decision_budget_atom_count": 16.0,
        "val_configured_decision_budget_atom_count": 16.0,
    }
    missing = dict(complete)
    missing.pop("val_pair_full_interface_action_match")
    missing.pop("val_selector_interaction_family_selected")
    assert _validation_fixed_budget_critical_score(complete) > _validation_fixed_budget_critical_score(missing)
