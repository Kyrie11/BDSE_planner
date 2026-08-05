from __future__ import annotations

import numpy as np
import torch

from bdse.data.cache_schema import CandidateBank, EvidenceAtom, EvidenceBank, PairLabels, TeacherLabels
from bdse.experiments.evaluate_open_loop import _align_bool_mask
from bdse.metrics.bdse_metrics import BDSEMetricResult, OnlineMetricMean, aggregate_metric_results, compute_bdse_diagnostics
from bdse.model.bdse_model import BDSEModel
from bdse.model.losses import _exact_winner_flip_critical_proposal_loss
from bdse.planner.nuplan_planner import runtime_query_diagnostics
from bdse.utils import deterministic_order


def test_exact_winner_flip_criticality_uses_literal_loo_action_flip() -> None:
    # With atom 0 present, action 1 wins (0.2 < 1.0). Removing atom 0 makes
    # action 0 win (0.0 < 0.2), so atom 0 is exactly critical.
    J0 = torch.tensor([[0.0, 0.2]])
    g = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, True]])
    active = torch.tensor([[True, True]])
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    deployed = torch.tensor([[True, False]])
    target = torch.tensor([1])
    costs = torch.ones((1, 2))
    cfg = {
        "training": {
            "exact_winner_flip_criticality": {
                "enabled": True,
                "positive_weight": 8.0,
                "negative_weight": 0.25,
                "rank_weight": 1.0,
                "teacher_aligned_weight": 4.0,
                "min_action_scale": 1.0,
            }
        }
    }
    loss, recall, atom_fraction, scene_fraction, teacher_scene_fraction = (
        _exact_winner_flip_critical_proposal_loss(
            J0, g, valid, active, logits, deployed, target, costs, cfg
        )
    )
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
    assert float(recall) == 1.0
    assert float(atom_fraction) == 0.5
    assert float(scene_fraction) == 1.0
    assert float(teacher_scene_fraction) == 1.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] < logits.grad[0, 1]


def test_metric_export_keeps_set_residual_fields_and_names_signed_scalar_delta() -> None:
    candidates = CandidateBank(
        trajectories=np.zeros((2, 2, 5), dtype=np.float32),
        valid_mask=np.array([True, True]),
        maneuver_ids=np.zeros((2,), dtype=np.int64),
        theta=[{}, {}],
        dynamic_flags=[{}, {}],
        metadata=[{}, {}],
    )
    atom = EvidenceAtom(
        atom_id=0,
        type="yield",
        anchor={},
        budget_cost=1.0,
        is_hard=False,
        family="interaction",
        active_mask=True,
    )
    evidence = EvidenceBank(
        atoms=[atom],
        query_features=np.zeros((1, 2, 1), dtype=np.float32),
        active_mask=np.array([True]),
    )
    # Lexicographic teacher action 0 may have higher scalar J_T than action 1.
    teacher = TeacherLabels(
        J_base=np.array([10.0, 5.0], dtype=np.float32),
        g_evid=np.zeros((1, 2), dtype=np.float32),
        J_evid=np.zeros((2,), dtype=np.float32),
        J_T=np.array([10.0, 5.0], dtype=np.float32),
        a_star=0,
        hard_violation_mask=np.array([False, True]),
    )
    pairs = PairLabels(
        pairs=np.array([[0, 1]], dtype=np.int64),
        margins=np.array([1.0], dtype=np.float32),
        weights=np.array([1.0], dtype=np.float32),
        residuals=np.array([0.0], dtype=np.float32),
        valid_mask=np.array([True]),
    )
    result = compute_bdse_diagnostics(
        candidates,
        evidence,
        teacher,
        pairs,
        predicted_base=np.array([10.0, 5.0], dtype=np.float32),
        predicted_atom_costs=np.zeros((1, 2), dtype=np.float32),
        selected_atoms=[0],
        action_index=1,
        query_diagnostics={
            "set_conditioned_residual_active": 1.0,
            "set_conditioned_residual_rank": 8.0,
            "set_conditioned_residual_abs_mean": 0.1,
            "set_conditioned_residual_scale": 1.0,
            "action_query_mode_all_valid": 1.0,
            "valid_action_count": 2,
            "queried_action_count": 2,
            "queried_valid_action_fraction": 1.0,
        },
    )
    assert result.values["set_conditioned_residual_active"] == 1.0
    assert result.values["set_conditioned_residual_rank"] == 8.0
    assert result.values["teacher_scalar_cost_delta"] == -5.0
    assert result.values["teacher_nonnegative_scalar_regret"] == 0.0
    assert result.values["action_query_mode_all_valid"] == 1.0
    assert result.values["queried_valid_action_fraction"] == 1.0


def test_dense_diagnostic_mask_padding_matches_configured_atom_axis() -> None:
    raw = np.array([True, False, True], dtype=bool)
    aligned = _align_bool_mask(raw, 8)
    assert aligned.shape == (8,)
    assert aligned.tolist() == [True, False, True, False, False, False, False, False]
    assert _align_bool_mask(np.ones((10,), dtype=bool), 4).tolist() == [True] * 4


def test_all_valid_action_bridge_reports_fixed_b_times_k_certificate_queries() -> None:
    pred = {
        "top_m_atoms": np.arange(48, dtype=np.int64),
        "queried_actions": np.arange(32, dtype=np.int64),
        "rival_pair_indices": np.arange(20, dtype=np.int64).reshape(10, 2),
        "action_query_mode_all_valid": True,
        "action_atom_query_count": 48 * 32,
        "unique_pair_atom_query_count": 48 * 10,
    }
    diag = runtime_query_diagnostics(pred, selected_atoms=np.arange(16, dtype=np.int64))
    assert diag["selected_certificate_query_count"] == 16 * 32
    assert diag["effective_query_count"] == 16 * 32


def test_optimized_dense_prediction_matches_full_forward(synthetic_sample, cfg) -> None:
    model = BDSEModel(cfg).eval()
    batch = model._make_batch(
        synthetic_sample.runtime,
        synthetic_sample.candidates,
        synthetic_sample.evidence_bank,
        include_dense_query=True,
    )
    with torch.inference_mode():
        reference = model.forward(batch)
    ref_j0 = reference["J0"][0].detach().cpu().numpy().astype(np.float32)
    ref_g = reference["g"][0].detach().cpu().numpy().astype(np.float32)
    ref_g_var = reference["g_var"][0].detach().cpu().numpy().astype(np.float32)
    active = _align_bool_mask(synthetic_sample.evidence_bank.active_mask, ref_g.shape[0])
    valid = _align_bool_mask(synthetic_sample.candidates.valid_mask, ref_g.shape[1])
    ref_g[~active, :] = 0.0
    ref_g[:, ~valid] = 0.0
    ref_g_var[~active, :] = 0.0
    ref_g_var[:, ~valid] = 0.0

    dense = model.predict_dense_numpy(
        synthetic_sample.runtime,
        synthetic_sample.candidates,
        synthetic_sample.evidence_bank,
        cfg,
    )
    np.testing.assert_allclose(dense["J0"], ref_j0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dense["g"], ref_g, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dense["g_var"], ref_g_var, rtol=1e-6, atol=1e-6)
    assert dense["g"].shape[0] == int(cfg["evidence"]["max_atoms"])


def test_deterministic_order_supports_one_shot_iterables() -> None:
    keys = (value for value in ["z", "a", "m"])
    assert deterministic_order(keys) == [1, 2, 0]


def test_online_metric_mean_matches_batch_aggregation() -> None:
    rows = [
        BDSEMetricResult(values={"a": 1.0, "b": float("nan")}, details={}),
        BDSEMetricResult(values={"a": 3.0, "b": 4.0}, details={}),
    ]
    online = OnlineMetricMean()
    for row in rows:
        online.update(row)
    expected = aggregate_metric_results(rows)
    actual = online.result()
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=0.0, equal_nan=True)
