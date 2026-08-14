from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bdse.config import load_config
from bdse.model.bdse_model import BDSEModel, _DecisiveAnchorFrontierValueResidual
from bdse.model.checkpoint_contract import load_bdse_state_with_contract
from bdse.planner.tournament import (
    _decisive_frontier_value_star_residual_numpy,
    run_pair_conditioned_tournament,
)
from bdse.tools.check_v64_3_13_eaf_dmvr_contract import run_contract
from bdse.tools.validate_v64_pipeline_config import _check


def _runtime_cfg(frontier_enabled: bool) -> dict:
    return {
        "runtime": {
            "pair_tournament_anchor_mode": "selected_local",
            "pair_tournament_pair_delta_includes_local": True,
            "pair_tournament_aggregation_mode": "decisive_anchor_margin",
            "pair_action_anchor_guard": {"enabled": True, "flip_margin": 0.01, "score_margin": 0.0},
            "dual_certificate": {"enabled": False},
            "decisive_frontier_value": {"enabled": frontier_enabled, "scale": 1.0},
        },
        "model": {"pair_margin_normalized": False},
        "tournament": {"epsilon_cal": 0.0, "beta_uncertainty": 0.0, "use_softmin": True, "softmin_tau": 1.0},
        "selector": {"pair_screen_top_l": 3, "pair_screen_near_eta": 10.0},
    }


def test_eaf_dmvr_module_is_exact_zero_init_and_receives_gradient() -> None:
    torch.manual_seed(1)
    m = _DecisiveAnchorFrontierValueResidual(16, 7, zero_init=True)
    scene = torch.randn(2, 16)
    action = torch.randn(2, 4, 16)
    evidence = torch.randn(2, 5, 16)
    av = torch.ones(2, 4, dtype=torch.bool)
    ev = torch.ones(2, 5, dtype=torch.bool)
    atom, signed, context = m.factors(scene, action, evidence, av, ev)
    assert torch.count_nonzero(atom).item() == 0
    selected = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 1, 0, 0]], dtype=torch.float32)
    pooled = (torch.tanh(atom) * selected[:, :, None]).sum(dim=1) / torch.sqrt(selected.sum(dim=1, keepdim=True))
    residual = (pooled[:, None, :] * torch.tanh(context[:, :1] + context + context[:, :1] * context) * (signed - signed[:, :1])).sum()
    residual.backward()
    assert m.atom_head[-1].weight.grad is not None
    assert float(m.atom_head[-1].weight.grad.abs().sum()) > 0.0


def test_eaf_dmvr_star_is_exactly_antisymmetric_and_atom_additive() -> None:
    rng = np.random.default_rng(3)
    atom = rng.normal(size=(4, 6)).astype(np.float32)
    signed = rng.normal(size=(3, 6)).astype(np.float32)
    context = rng.normal(size=(3, 6)).astype(np.float32)
    valid = np.ones(3, dtype=bool)
    ab, _ = _decisive_frontier_value_star_residual_numpy([0, 2], valid, 0, atom, signed, context)
    ba, _ = _decisive_frontier_value_star_residual_numpy([0, 2], valid, 1, atom, signed, context)
    np.testing.assert_allclose(ab[1], -ba[0], rtol=1e-6, atol=1e-6)
    all_atoms, _ = _decisive_frontier_value_star_residual_numpy([0, 2], valid, 0, atom, signed, context)
    # The 1/sqrt(|S|) normalization means the two single-atom contributions sum
    # to sqrt(2) times the two-atom result.
    one0, _ = _decisive_frontier_value_star_residual_numpy([0], valid, 0, atom, signed, context)
    one2, _ = _decisive_frontier_value_star_residual_numpy([2], valid, 0, atom, signed, context)
    np.testing.assert_allclose(one0 + one2, np.sqrt(2.0) * all_atoms, rtol=1e-6, atol=1e-6)


def test_eaf_dmvr_zero_factors_are_exact_runtime_noop() -> None:
    cfg_off = _runtime_cfg(False)
    cfg_on = _runtime_cfg(True)
    J0 = np.array([0.0, 2.0, 5.0], dtype=np.float32)
    g = np.zeros((2, 3), dtype=np.float32)
    pairs = np.array([[0, 1]], dtype=np.int64)
    pair_delta = np.zeros((2, 1), dtype=np.float32)
    valid = np.ones(3, dtype=bool)
    safety = np.zeros(3, dtype=bool)
    kwargs = dict(predicted_atom_costs=g)
    base = run_pair_conditioned_tournament(J0, pair_delta, pairs, [0, 1], valid, safety, cfg_off, **kwargs)
    zero = run_pair_conditioned_tournament(
        J0, pair_delta, pairs, [0, 1], valid, safety, cfg_on,
        frontier_value_atom_factors=np.zeros((2, 4), np.float32),
        frontier_value_action_signed_factors=np.ones((3, 4), np.float32),
        frontier_value_action_context_factors=np.ones((3, 4), np.float32),
        **kwargs,
    )
    assert base.action_index == zero.action_index
    np.testing.assert_allclose(base.scores, zero.scores, rtol=0.0, atol=0.0)
    assert zero.diagnostics["decisive_frontier_value_active"] == 1.0
    assert zero.diagnostics["decisive_frontier_value_complete_star_coverage"] == 1.0


def test_eaf_dmvr_can_flip_anchor_through_missing_sparse_edge() -> None:
    cfg = _runtime_cfg(True)
    J0 = np.array([0.0, 2.0, 5.0], dtype=np.float32)
    g = np.zeros((1, 3), dtype=np.float32)
    # Sparse graph contains only 0<->1; the new complete star must still be able
    # to correct challenger 2 without changing the graph or B-set.
    pairs = np.array([[0, 1]], dtype=np.int64)
    pair_delta = np.zeros((1, 1), dtype=np.float32)
    valid = np.ones(3, dtype=bool)
    safety = np.zeros(3, dtype=bool)
    atom = np.ones((1, 1), dtype=np.float32) * 3.0
    signed = np.array([[0.0], [0.0], [-10.0]], dtype=np.float32)
    context = np.ones((3, 1), dtype=np.float32)
    out = run_pair_conditioned_tournament(
        J0, pair_delta, pairs, [0], valid, safety, cfg,
        predicted_atom_costs=g,
        frontier_value_atom_factors=atom,
        frontier_value_action_signed_factors=signed,
        frontier_value_action_context_factors=context,
    )
    assert out.action_index == 2
    assert out.diagnostics["decisive_frontier_value_complete_star_coverage"] == 1.0


def test_v64_3_13_configs_freeze_acquisition_and_only_train_value_head() -> None:
    for name, role in [
        ("bdse/configs/v64_3_13_eaf_dmvr_daepc_screen_2gpu.yaml", "train"),
        ("bdse/configs/v64_3_13_eaf_dmvr_daepc_train_2gpu.yaml", "train"),
        ("bdse/configs/v64_3_13_eaf_dmvr_cl.yaml", "eval"),
    ]:
        report = _check(Path(name), role, "v64.3.13")
        assert report["pass"], report["failures"]
    contract = run_contract("bdse/configs/v64_3_13_eaf_dmvr_daepc_screen_2gpu.yaml")
    assert contract["pass"], [k for k, v in contract["checks"].items() if not v]
    cfg = load_config("bdse/configs/v64_3_13_eaf_dmvr_daepc_screen_2gpu.yaml")
    assert cfg["evidence"]["budget"] == 16
    assert cfg["selector"]["proposal_top_m"] == 24
    assert cfg["training"]["trainable_modules"] == ["decisive_anchor_frontier_value_adapter"]
    assert cfg["training"]["budgeted_decisive_margin_utility"]["enabled"] is False
    model = BDSEModel(cfg)
    assert isinstance(model.decisive_anchor_frontier_value_adapter, _DecisiveAnchorFrontierValueResidual)


def test_v64_3_13_warm_start_may_omit_only_new_value_head() -> None:
    cfg = load_config("bdse/configs/v64_3_13_eaf_dmvr_daepc_screen_2gpu.yaml")
    model = BDSEModel(cfg)
    state = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("decisive_anchor_frontier_value_adapter.")}
    report = load_bdse_state_with_contract(model, state, cfg, context="v64.3.13 synthetic warm start")
    assert report["core_contract_pass"]
    assert any(k.startswith("decisive_anchor_frontier_value_adapter.") for k in report["missing"])


def _frontier_screen_row(epoch: int, *, sign: float, action: float, teacher: float, regret: float, delta: float) -> dict:
    return {
        "epoch": epoch,
        "val_teacher_action_match": teacher,
        "val_teacher_regret": regret,
        "val_pair_full_interface_action_match": 0.18,
        "val_local_pair_full_interface_action_match": 0.17,
        "val_proposal_decisive_atom_recall": 0.75,
        "val_teacher_exact_winner_flip_critical_recall_topm_micro": 0.237,
        "val_teacher_exact_winner_flip_critical_recall_selected_micro": 0.149,
        "val_evidence_certificate_fraction": 0.928,
        "val_base_pair_sign_acc_winner_rival": 0.628,
        "val_dense_pair_sign_acc_winner_rival": 0.626,
        "val_pair_sign_acc_winner_rival": 0.061,
        "val_frontier_value_pair_sign_acc": sign,
        "val_frontier_value_action_match": action,
        "val_frontier_value_anchor_wrong_fraction": 0.82,
        "val_frontier_value_wrong_anchor_corrected_fraction": max(action - 0.17, 0.0),
        "val_frontier_value_correct_anchor_preserved_fraction": 1.0,
        "val_frontier_value_residual_rms": 0.0 if epoch < 0 else 0.05,
        "val_frontier_value_exact_scene_fraction": 1.0,
        "val_frontier_value_complete_star_coverage": 1.0,
        "val_decisive_frontier_value_active": 1.0,
        "val_decisive_frontier_value_complete_star_coverage": 1.0,
        "val_decisive_frontier_value_residual_rms": 0.0 if epoch < 0 else 0.05,
        "frontier_value_adapter_parameter_delta_rms": delta,
        "frontier_value_adapter_parameter_delta_max_abs": delta * 4.0,
        "critical_adapter_parameter_delta_rms": 0.0,
        "decisive_pair_adapter_parameter_delta_rms": 0.0,
    }


def test_v64_3_13_screen_promotes_real_value_gain_without_acquisition_drift() -> None:
    from bdse.tools.check_v64_3_13_eaf_dmvr_screen import build
    rows = [
        _frontier_screen_row(-1, sign=0.20, action=0.18, teacher=0.178, regret=20000.0, delta=0.0),
        _frontier_screen_row(1, sign=0.24, action=0.20, teacher=0.190, regret=19700.0, delta=0.003),
    ]
    r=build(rows)
    assert r["instrumentation_valid"]
    assert r["acquisition_frozen"]
    assert r["mechanism_gain"]
    assert r["deployment_gain"]
    assert r["full_promotion"]


def test_v64_3_13_screen_failure_pivots_to_representation_not_acquisition() -> None:
    from bdse.tools.check_v64_3_13_eaf_dmvr_screen import build
    rows = [
        _frontier_screen_row(-1, sign=0.20, action=0.18, teacher=0.178, regret=20000.0, delta=0.0),
        _frontier_screen_row(1, sign=0.205, action=0.181, teacher=0.180, regret=19950.0, delta=0.003),
    ]
    r=build(rows)
    assert r["instrumentation_valid"]
    assert not r["mechanism_gain"]
    assert not r["full_promotion"]
    assert r["next_action"] == "selective_action_evidence_representation_capacity_test"
