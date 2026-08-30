from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bdse.model.bdse_model import _apply_structural_safety_residual_prior_np
from bdse.planner.pair_screen import reweight_pairs_by_viability_scope
from bdse.planner.selector import structural_safety_mask
from bdse.tools.check_v37_runtime_gate import passes


def test_structural_mask_can_exclude_soft_feasibility_family():
    hard = np.asarray([False, True, False, False], dtype=bool)
    family = np.asarray([1, 2, 1, 3], dtype=np.int64)
    active = np.ones((4,), dtype=bool)
    strict = structural_safety_mask(hard, family, active, include_feasibility=False)
    assert strict.tolist() == [False, True, False, False]


def test_viability_soft_weighting_keeps_graph_and_prefers_safe_safe():
    pairs = np.asarray([[0, 1], [0, 2], [2, 3]], dtype=np.int64)
    out, weights, diag = reweight_pairs_by_viability_scope(
        pairs,
        np.ones((3,), dtype=np.float32),
        valid_mask=np.ones((4,), dtype=bool),
        safety_flags=np.asarray([False, False, True, True], dtype=bool),
        predicted_base_cost=np.arange(4, dtype=np.float32),
        safe_safe_weight=1.0,
        cross_safety_weight=0.3,
        unsafe_unsafe_weight=0.1,
    )
    assert np.array_equal(out, pairs)
    assert weights[0] > weights[1] > weights[2]
    assert diag["pair_count_after_viability"] == 3.0


def test_structural_residual_prior_changes_only_valid_ranking(monkeypatch):
    candidates = SimpleNamespace(
        valid_mask=np.asarray([True, True, False], dtype=bool),
        K=3,
        trajectories=np.zeros((3, 2, 5), dtype=np.float32),
    )
    runtime = SimpleNamespace()
    risks = {
        "hard_agent": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "agent_ttc": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "hard_off_route": np.zeros((3,), dtype=np.float32),
        "soft_agent": np.zeros((3,), dtype=np.float32),
        "soft_off_route": np.zeros((3,), dtype=np.float32),
        "red_light": np.zeros((3,), dtype=np.float32),
    }
    monkeypatch.setattr("bdse.model.bdse_model.runtime_risk_scores", lambda *args, **kwargs: risks)
    cfg = {
        "model": {"margin_normalization_min_scale": 100.0},
        "runtime": {"structural_safety_residual": {"enabled": True, "weight": 0.5}},
    }
    out, diag = _apply_structural_safety_residual_prior_np(
        np.asarray([0.0, 0.0, np.inf], dtype=np.float32), runtime, candidates, cfg
    )
    assert out[0] < out[1]
    assert np.isinf(out[2])
    assert diag["structural_residual_enabled"] is True


def test_v37_gate_does_not_punish_unavoidable_all_flagged_rate():
    row = {
        "structural_hard_decisive_coverage": 1.0,
        "effective_hard_decisive_recall": 1.0,
        "selected_soft_interaction_decisive_recall": 0.40,
        "effective_interaction_decisive_recall": 0.50,
        "fallback_would_trigger_rate": 0.0,
        "avoidable_selected_action_safety_flag_rate": 0.0,
        "selected_action_safety_flag_rate": 0.035,
        "all_actions_safety_flagged_rate": 0.035,
        "all_flagged_risk_guard_applied_rate": 0.035,
        "teacher_action_match": 0.23,
        "effective_query_count": 6000.0,
        "total_sparse_query_count": 16000.0,
        "decision_budget_excludes_structural_safety": 1.0,
        "structural_residual_enabled": 1.0,
        "structural_safety_include_feasibility": 0.0,
    }
    ok, failures = passes(row)
    assert ok, failures
    bad = dict(row)
    bad["avoidable_selected_action_safety_flag_rate"] = 0.02
    ok, failures = passes(bad)
    assert not ok
    assert any("avoidable_selected_action_safety_flag_rate" in f for f in failures)
