from __future__ import annotations

from pathlib import Path

import torch

from bdse.config import load_config
from bdse.experiments.train import _boundary_focused_pair_subsample
from bdse.model.bdse_model import BDSEModel, _LiteralBoundaryPairResidual
from bdse.tools.validate_v64_pipeline_config import _check


def test_lbpr_zero_init_and_exact_antisymmetry_after_training_like_weights() -> None:
    h, r = 16, 7
    m = _LiteralBoundaryPairResidual(h, r, zero_init=True, query_gain=0.7)
    B, Q = 2, 5
    e = torch.randn(B, Q, h)
    s = torch.randn(B, Q, h)
    a = torch.randn(B, Q, h)
    b = torch.randn(B, Q, h)
    qa = torch.randn(B, Q, h)
    qb = torch.randn(B, Q, h)
    z = m.from_embeddings(e, s, a, b, qa, qb)
    assert torch.count_nonzero(z).item() == 0
    with torch.no_grad():
        torch.nn.init.normal_(m.output.weight, std=0.1)
    ab = m.from_embeddings(e, s, a, b, qa, qb)
    ba = m.from_embeddings(e, s, b, a, qb, qa)
    torch.testing.assert_close(ab, -ba, rtol=1e-6, atol=1e-6)


def test_literal_boundary_pair_quota_reserves_exact_flip_edge() -> None:
    pairs = torch.tensor([[[0, 2], [1, 2], [0, 1]]], dtype=torch.long)
    batch = {
        'pair_indices': pairs,
        'pair_valid': torch.ones((1, 3), dtype=torch.bool),
        'pair_margins': torch.tensor([[0.01, 0.01, 9.0]]),
        'pair_weights': torch.tensor([[9.0, 8.0, 0.01]]),
        'pair_residuals': torch.zeros((1, 3)),
        'teacher_a_star': torch.tensor([0]),
        'teacher_hard_violation': torch.zeros((1, 3), dtype=torch.bool),
        'teacher_J_T': torch.tensor([[0.0, 0.2, 0.3]]),
        # Removing atom 0: [0, .2, .3] - [-.5, 0, 0] -> [.5,.2,.3], winner 1.
        'teacher_g_evid': torch.tensor([[[-0.5, 0.0, 0.0]]]),
        'evidence_active': torch.tensor([[True]]),
        'candidate_valid': torch.ones((1, 3), dtype=torch.bool),
    }
    cfg = {'training': {
        'global_step': 1, 'current_epoch': 0, 'epochs': 4, 'steps_per_epoch': 10,
        'boundary_pair_sampler': {
            'enabled': True, 'full_every_n_steps': 99, 'full_last_n_steps': 0,
            'max_pairs': 1, 'literal_boundary_quota': 1, 'literal_boundary_bonus': 32.0,
            'winner_quota': 0, 'hard_cross_quota': 0, 'near_tie_quota': 0,
            'winner_bonus': 0.0, 'hard_cross_bonus': 0.0, 'near_tie_bonus': 0.0,
            'pair_weight_bonus': 0.0, 'near_tie_tau': 0.5, 'min_margin_scale': 1.0,
        }
    }}
    out = _boundary_focused_pair_subsample(batch, cfg)
    torch.testing.assert_close(out['pair_indices'][0, 0], torch.tensor([0, 1]))
    assert bool(out['pair_valid'][0, 0])


def test_v64_3_6_configs_and_trainable_contracts() -> None:
    for tag in ('local', 'bcha', 'lbpr', 'bcha_lbpr'):
        train = Path(f'bdse/configs/v64_3_6_cc_aocc_ccbr_{tag}_lea_daepc_screen_2gpu.yaml')
        evalp = Path(f'bdse/configs/v64_3_6_cc_aocc_ccbr_{tag}_lea_cl.yaml')
        rt = _check(train, 'train', 'v64.3.6')
        re = _check(evalp, 'eval', 'v64.3.6')
        assert rt['pass'], rt['failures']
        assert re['pass'], re['failures']
        cfg = load_config(train)
        trainable = set(cfg['training']['trainable_modules'])
        assert 'critical_proposal_adapter' in trainable
        if 'lbpr' in tag:
            assert cfg['model']['literal_boundary_pair_adapter']['enabled'] is True
            assert 'literal_boundary_pair_adapter' in trainable
        else:
            assert cfg['model']['literal_boundary_pair_adapter']['enabled'] is False
            assert 'literal_boundary_pair_adapter' not in trainable
        assert cfg['evidence']['budget'] == 16
        assert cfg['runtime']['pair_tournament_aggregation_mode'] == 'legacy_tournament'


def test_bcha_and_lbpr_step_zero_preserve_v62_style_anchor() -> None:
    cfg = load_config('bdse/configs/v64_3_6_cc_aocc_ccbr_bcha_lbpr_lea_daepc_screen_2gpu.yaml')
    model = BDSEModel(cfg)
    assert model.critical_family_coupling_enabled
    assert model.literal_boundary_pair_adapter is not None
    assert torch.count_nonzero(model.critical_proposal_adapter.residual_head[-1].weight).item() == 0
    assert torch.count_nonzero(model.literal_boundary_pair_adapter.output.weight).item() == 0


def test_training_validation_oracles_are_emitted_and_separate_family_ceiling() -> None:
    import numpy as np
    from types import SimpleNamespace
    from bdse.experiments.train import _teacher_literal_criticality_full_support

    evidence_bank = SimpleNamespace(
        active_mask=np.array([True, True]),
        budget_costs=lambda: np.array([1.0, 1.0], dtype=np.float32),
    )
    sample = SimpleNamespace(
        evidence_bank=evidence_bank,
        candidates=SimpleNamespace(valid_mask=np.array([True, True])),
        teacher=SimpleNamespace(
            g_evid=np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            J_T=np.array([0.5, 1.0], dtype=np.float32),
            a_star=0,
        ),
    )
    pred = {
        'g': np.zeros((2, 2), dtype=np.float32),
        'top_m_atoms': np.array([0], dtype=np.int64),
        'family_ids': np.array([0, 1], dtype=np.int64),
        'family_logits': np.array([10.0, -10.0], dtype=np.float32),
        'J0': np.array([0.0, 1.0], dtype=np.float32),
    }
    cfg = {
        'evidence': {'budget': 1},
        'selector': {'proposal_top_m': 1, 'hab_enabled': True, 'hab_reserve_fraction': 0.0, 'hab_free_budget': 0},
    }
    values, _ = _teacher_literal_criticality_full_support(sample, pred, [0], cfg)
    assert np.isfinite(values['teacher_exact_winner_flip_frozen_family_slot_oracle_topm_recall'])
    assert np.isfinite(values['teacher_exact_winner_flip_global_oracle_topm_recall'])
    assert values['teacher_exact_winner_flip_global_oracle_topm_recall'] >= values['teacher_exact_winner_flip_frozen_family_slot_oracle_topm_recall']
