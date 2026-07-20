from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bdse.config import load_config
from bdse.planner.nuplan_planner import runtime_query_diagnostics
from bdse.planner.selector import (
    _signed_margin_coreset_from_pair_delta,
    runtime_greedy_selector_pair_conditioned,
)
from bdse.tools.check_v38_runtime_gate import passes


def test_signed_margin_coreset_preserves_target_action_and_signs() -> None:
    delta = np.asarray(
        [
            [1.0, -1.0],
            [0.0, 2.0],
            [0.8, 0.8],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )
    selected, _, spent, diag = _signed_margin_coreset_from_pair_delta(
        delta,
        np.zeros((2,), dtype=np.float32),
        np.ones((2,), dtype=np.float32),
        np.asarray([[0, 1], [0, 2]], dtype=np.int64),
        np.ones((4,), dtype=np.float32),
        2.0,
        np.asarray([True, True, True]),
        np.asarray([False, False, False]),
        np.ones((4,), dtype=bool),
    )
    assert len(selected) == 2
    assert spent == 2.0
    assert diag["margin_coreset_target_action_preserved"] == 1.0
    assert diag["margin_coreset_target_sign_agreement"] == 1.0


def test_margin_coreset_selector_respects_soft_interaction_floor() -> None:
    rng = np.random.default_rng(4)
    delta = rng.normal(size=(8, 4)).astype(np.float32)
    soft = np.asarray([True, True, True, True, False, False, False, False])
    result = runtime_greedy_selector_pair_conditioned(
        np.zeros((4,), dtype=np.float32),
        delta,
        np.asarray([[0, 1], [0, 2], [0, 3], [1, 2]], dtype=np.int64),
        np.ones((4,), dtype=np.float32),
        np.ones((8,), dtype=np.float32),
        np.ones((4,), dtype=bool),
        np.zeros((4,), dtype=bool),
        4.0,
        atom_active_mask=np.ones((8,), dtype=bool),
        selector_cap_mode="margin_coreset",
        soft_interaction_mask=soft,
        soft_interaction_quota=2,
        min_selected_atoms=4,
        force_fill_budget=True,
    )
    assert result.diagnostics["mode"] == "runtime_pair_conditioned_margin_coreset"
    assert result.diagnostics["margin_coreset_active"]
    assert sum(bool(soft[i]) for i in result.selected) >= 2


def test_runtime_query_diagnostics_exposes_v38_activation_fields() -> None:
    pred = {
        "top_m_atoms": np.arange(3),
        "queried_actions": np.arange(2),
        "runtime_pairs": np.asarray([[0, 1]]),
        "rival_pair_indices": np.asarray([[0, 1]]),
        "unique_pair_atom_query_count": 3,
        "structural_safety_bypass": True,
        "structural_safety_include_feasibility": False,
        "structural_residual_enabled": True,
        "structural_residual_weight": 0.22,
        "pair_delta_calibration_enabled": True,
        "pair_delta_selector_local_weight_mean": 0.4,
    }
    diag = runtime_query_diagnostics(pred, [0, 1])
    assert diag["structural_residual_enabled"] == 1
    assert diag["structural_safety_include_feasibility"] == 0
    assert diag["pair_delta_calibration_enabled"] == 1
    assert diag["pair_delta_selector_local_weight_mean"] == 0.4


def test_v38_gate_accepts_complete_non_degenerate_row(tmp_path: Path) -> None:
    row = {
        "structural_hard_decisive_coverage": 1.0,
        "effective_hard_decisive_recall": 1.0,
        "selected_soft_interaction_decisive_recall": 0.5,
        "effective_interaction_decisive_recall": 0.6,
        "fallback_would_trigger_rate": 0.0,
        "avoidable_selected_action_safety_flag_rate": 0.0,
        "teacher_action_match": 0.23,
        "budget_vs_full_match": 0.20,
        "effective_query_count": 6000.0,
        "total_sparse_query_count": 12000.0,
        "decision_budget_excludes_structural_safety": 1.0,
        "structural_residual_enabled": 1.0,
        "structural_safety_include_feasibility": 0.0,
        "selector_margin_coreset_active": 1.0,
        "selector_margin_coreset_target_action_preserved": 0.95,
        "selector_margin_coreset_target_sign_agreement": 0.95,
        "pair_delta_calibration_enabled": 1.0,
        "selected_action_safety_flag_rate": 0.03,
        "all_actions_safety_flagged_rate": 0.03,
        "all_flagged_risk_guard_applied_rate": 0.03,
    }
    ok, failures = passes(row, path=tmp_path / "open_loop_v38_mars_balanced.json")
    assert ok, failures


def test_v38_configs_load() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in (
        "v38_bdse_mars_balanced_fast_cl.yaml",
        "v38_bdse_mars_winner_strong_fast_cl.yaml",
        "v38_bdse_mars_pair_only_fast_cl.yaml",
        "v38_bdse_mars_actionrank_control_fast_cl.yaml",
    ):
        cfg = load_config(str(root / "bdse" / "configs" / name))
        assert int(cfg["evidence"]["budget"]) == 16
        assert int(cfg["selector"]["proposal_top_m"]) == 64
