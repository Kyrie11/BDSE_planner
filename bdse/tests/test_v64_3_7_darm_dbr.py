from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bdse.config import load_config
from bdse.model.bdse_model import BDSEModel, _DecisiveBoundaryPairResidual
from bdse.model.losses import _decisive_anchor_margin_scores
from bdse.model.checkpoint_contract import load_bdse_state_with_contract
from bdse.planner.tournament import (
    _decisive_anchor_margin_scores as _decisive_anchor_margin_scores_np,
    run_pair_conditioned_tournament,
)
from bdse.tools.validate_v64_pipeline_config import _check
from bdse.tools.check_v64_3_7_darm_dbr_screen import build as build_darm_screen


def test_dbr_zero_init_and_exact_antisymmetry() -> None:
    h, r = 16, 7
    module = _DecisiveBoundaryPairResidual(h, r, zero_init=True, query_gain=0.8)
    B, Q = 2, 5
    e = torch.randn(B, Q, h)
    s = torch.randn(B, Q, h)
    a = torch.randn(B, Q, h)
    b = torch.randn(B, Q, h)
    qa = torch.randn(B, Q, h)
    qb = torch.randn(B, Q, h)
    assert torch.count_nonzero(module.from_embeddings(e, s, a, b, qa, qb)).item() == 0
    with torch.no_grad():
        torch.nn.init.normal_(module.output.weight, std=0.1)
    ab = module.from_embeddings(e, s, a, b, qa, qb)
    ba = module.from_embeddings(e, s, b, a, qb, qa)
    torch.testing.assert_close(ab, -ba, rtol=1e-6, atol=1e-6)


def test_numpy_darm_zero_residual_is_exact_selected_local_noop() -> None:
    cost = np.array([0.0, 2.0, 5.0], dtype=np.float32)
    M = cost[None, :] - cost[:, None]
    valid = np.ones(3, dtype=bool)
    scores, anchor = _decisive_anchor_margin_scores_np(cost, M, valid, 1.0)
    assert anchor == 0
    assert int(np.argmax(scores)) == 0
    # Centering cancels; score differences must exactly equal direct cost differences.
    np.testing.assert_allclose(scores - scores[0], -(cost - cost[0]), rtol=1e-6, atol=1e-6)


def test_numpy_darm_uses_only_anchor_challenger_star() -> None:
    cost = np.array([0.0, 2.0, 5.0], dtype=np.float32)
    M = cost[None, :] - cost[:, None]
    # A correction on a non-anchor edge must not alter the action.
    M_non_anchor = M.copy()
    M_non_anchor[1, 2] = -100.0
    M_non_anchor[2, 1] = 100.0
    scores, _ = _decisive_anchor_margin_scores_np(cost, M_non_anchor, np.ones(3, bool), 1.0)
    assert int(np.argmax(scores)) == 0
    # A certified anchor->challenger margin crossing can alter the action.
    M_anchor = M.copy()
    M_anchor[0, 1] = -1.0
    M_anchor[1, 0] = 1.0
    scores, _ = _decisive_anchor_margin_scores_np(cost, M_anchor, np.ones(3, bool), 1.0)
    assert int(np.argmax(scores)) == 1


def test_torch_darm_training_scores_match_runtime_semantics() -> None:
    anchor_cost = torch.tensor([[0.0, 2.0, 5.0]])
    pairs = torch.tensor([[[0, 1]]], dtype=torch.long)
    pair_valid = torch.ones((1, 1), dtype=torch.bool)
    selected = torch.ones((1, 1), dtype=torch.bool)
    valid = torch.ones((1, 3), dtype=torch.bool)
    local = torch.zeros((1, 1, 1))
    residual = torch.zeros((1, 1, 1))
    scores, cov = _decisive_anchor_margin_scores(
        anchor_cost, residual, pairs, pair_valid, selected, valid,
        local_pair_delta=local, pair_delta_includes_local=True,
        pair_scale=torch.ones((1, 1)), normalize_margins=True,
    )
    assert int(scores.argmax(dim=1).item()) == 0
    assert abs(float(cov.item()) - 0.5) < 1e-6
    residual = torch.tensor([[[-3.0]]])
    scores, _ = _decisive_anchor_margin_scores(
        anchor_cost, residual, pairs, pair_valid, selected, valid,
        local_pair_delta=local, pair_delta_includes_local=True,
        pair_scale=torch.ones((1, 1)), normalize_margins=True,
    )
    assert int(scores.argmax(dim=1).item()) == 1


def test_runtime_darm_branch_can_correct_anchor_without_global_tournament() -> None:
    cfg = {
        'runtime': {
            'pair_tournament_anchor_mode': 'selected_local',
            'pair_tournament_pair_delta_includes_local': True,
            'pair_tournament_aggregation_mode': 'decisive_anchor_margin',
            'pair_action_anchor_guard': {'enabled': True, 'flip_margin': 0.01, 'score_margin': 0.0},
            'dual_certificate': {'enabled': False},
        },
        'model': {'pair_margin_normalized': False},
        'tournament': {'epsilon_cal': 0.0, 'beta_uncertainty': 0.0, 'use_softmin': True, 'softmin_tau': 1.0},
        'selector': {'pair_screen_top_l': 3, 'pair_screen_near_eta': 10.0},
    }
    J0 = np.array([0.0, 2.0, 5.0], dtype=np.float32)
    g = np.zeros((1, 3), dtype=np.float32)
    pairs = np.array([[0, 1]], dtype=np.int64)
    valid = np.ones(3, dtype=bool)
    safety = np.zeros(3, dtype=bool)
    zero = run_pair_conditioned_tournament(
        J0, np.array([[0.0]], dtype=np.float32), pairs, [0], valid, safety, cfg,
        predicted_atom_costs=g,
    )
    assert zero.action_index == 0
    assert zero.diagnostics['decisive_anchor_margin_active'] == 1.0
    corrected = run_pair_conditioned_tournament(
        J0, np.array([[-3.0]], dtype=np.float32), pairs, [0], valid, safety, cfg,
        predicted_atom_costs=g,
    )
    assert corrected.action_index == 1


def test_v64_3_7_configs_are_value_isolated_and_fixed_budget() -> None:
    for tag in ('broad', 'literal'):
        train = Path(f'bdse/configs/v64_3_7_cc_aocc_darm_dbr_{tag}_daepc_screen_2gpu.yaml')
        evalp = Path(f'bdse/configs/v64_3_7_cc_aocc_darm_dbr_{tag}_cl.yaml')
        rt = _check(train, 'train', 'v64.3.7')
        re = _check(evalp, 'eval', 'v64.3.7')
        assert rt['pass'], rt['failures']
        assert re['pass'], re['failures']
        cfg = load_config(train)
        assert cfg['evidence']['budget'] == 16
        assert cfg['model']['critical_proposal_adapter']['enabled'] is False
        assert cfg['model']['literal_boundary_pair_adapter']['enabled'] is False
        assert cfg['model']['decisive_boundary_pair_adapter']['enabled'] is True
        assert cfg['runtime']['pair_tournament_aggregation_mode'] == 'decisive_anchor_margin'
        assert cfg['training']['trainable_modules'] == ['decisive_boundary_pair_adapter']
        # DARM is a train/deploy pair-margin path: hard/safety atoms must not be
        # silently removed from its regression or action loss while remaining
        # active at runtime.
        assert cfg['training']['exclude_safety_atoms_from_pair_regression'] is False
        assert cfg['training']['exclude_safety_atoms_from_pair_action_loss'] is False
        model = BDSEModel(cfg)
        assert model.critical_proposal_adapter is None
        assert model.literal_boundary_pair_adapter is None
        assert isinstance(model.decisive_boundary_pair_adapter, _DecisiveBoundaryPairResidual)
        assert sum(p.numel() for p in model.decisive_boundary_pair_adapter.parameters()) < 40000


def test_v64_3_7_v62_warm_start_may_omit_only_new_dbr_head() -> None:
    cfg = load_config('bdse/configs/v64_3_7_cc_aocc_darm_dbr_broad_daepc_screen_2gpu.yaml')
    model = BDSEModel(cfg)
    state = {
        k: v.clone() for k, v in model.state_dict().items()
        if not k.startswith('decisive_boundary_pair_adapter.')
    }
    report = load_bdse_state_with_contract(model, state, cfg, context='v64.3.7 synthetic V62 warm start')
    assert report['core_contract_pass']
    assert any(k.startswith('decisive_boundary_pair_adapter.') for k in report['missing'])
    # A missing foundation tensor must still fail; optionality is restricted to the new head.
    bad = dict(state)
    core_key = next(k for k in bad if not k.startswith(('critical_proposal_adapter.', 'query_extension_proj.', 'literal_boundary_pair_adapter.')))
    bad.pop(core_key)
    try:
        load_bdse_state_with_contract(BDSEModel(cfg), bad, cfg, context='v64.3.7 bad warm start')
    except ValueError:
        pass
    else:
        raise AssertionError('missing core foundation tensor must be rejected')


def _screen_row(
    epoch: int,
    teacher: float,
    pairfull: float,
    localpair: float,
    *,
    dbr: float = 0.0,
    teacher_regret: float = 100.0,
    pairfull_regret: float = 100.0,
    budgetpair: float = 0.95,
    beneficial_compression: float = 0.0,
    harmful_compression: float = 0.0,
    coverage: float = 0.5,
) -> dict:
    return {
        'epoch': epoch,
        'val_teacher_action_match': teacher,
        'val_pair_full_interface_action_match': pairfull,
        'val_local_pair_full_interface_action_match': localpair,
        'val_budget_vs_pair_full_match': budgetpair,
        'val_pair_full_to_budget_flip_rate': 1.0 - budgetpair,
        'val_teacher_exact_winner_flip_critical_recall_topm_micro': 0.36,
        'val_teacher_exact_winner_flip_critical_recall_selected_micro': 0.25,
        'val_proposal_decisive_atom_recall': 0.78,
        'val_beneficial_residual_intervention_rate': 0.02 if epoch >= 0 else 0.0,
        'val_harmful_residual_intervention_rate': 0.005 if epoch >= 0 else 0.0,
        'val_teacher_regret': teacher_regret,
        'val_pair_full_teacher_regret': pairfull_regret,
        'val_local_pair_full_teacher_regret': 100.0,
        'val_beneficial_pair_compression_rate': beneficial_compression,
        'val_harmful_pair_compression_rate': harmful_compression,
        'decisive_pair_adapter_parameter_delta_rms': dbr,
        'decisive_boundary_pair_residual_rms': dbr,
        'decisive_anchor_full_pair_coverage': coverage,
        'decisive_anchor_budget_pair_coverage': coverage,
        'training_pair_full_graph_fraction': 0.125,
        'val_decisive_anchor_margin_active': 1.0,
    }


def test_v64_3_7_screen_gate_rejects_weak_anchor_even_if_pairfull_moves() -> None:
    rows = [
        _screen_row(-1, 0.18, 0.17, 0.17, teacher_regret=120.0, pairfull_regret=110.0),
        _screen_row(0, 0.19, 0.20, 0.18, dbr=0.01, teacher_regret=100.0, pairfull_regret=90.0),
    ]
    report = build_darm_screen(rows, 'synthetic')
    assert report['meaningful_value_gain']
    assert report['strong_selected_local_anchor_restored'] is False
    assert report['full_promotion'] is False
    assert report['valid'] is False


def test_v64_3_7_1_screen_accepts_teacher_aligned_budget_divergence_and_discrete_coverage() -> None:
    # Mirrors the uploaded BROAD pattern: pair/full and deployed teacher improve,
    # budget-vs-pair-full agreement falls, but that divergence is net beneficial.
    rows = [
        _screen_row(
            -1, 0.264, 0.264, 0.264,
            teacher_regret=14484.0, pairfull_regret=14080.0,
            budgetpair=0.962, coverage=0.19909,
        ),
        _screen_row(
            3, 0.282, 0.274, 0.264, dbr=0.005,
            teacher_regret=13367.0, pairfull_regret=13674.0,
            budgetpair=0.886, beneficial_compression=0.016,
            harmful_compression=0.008, coverage=0.19909,
        ),
    ]
    report = build_darm_screen(rows, 'synthetic-broad')
    assert report['valid']
    assert report['meaningful_value_gain']
    assert report['deployment_gain']
    assert report['full_promotion']
    assert report['budget_compression_net'] > 0
    assert report['activation']['decisive_anchor_full_pair_coverage_max'] < 0.20
    assert report['thresholds']['all_challenger_pair_coverage_is_gate'] is False
    assert report['thresholds']['budget_vs_pair_full_agreement_is_gate'] is False
