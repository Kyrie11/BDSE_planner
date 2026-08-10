from __future__ import annotations

from pathlib import Path

import torch

from bdse.config import load_config
from bdse.model.bdse_model import BDSEModel, _FrontierPairCriticalProposalAdapter
from bdse.model.losses import _exact_winner_flip_critical_proposal_loss
from bdse.tools.validate_v64_pipeline_config import _check


def _model() -> BDSEModel:
    cfg = load_config('bdse/configs/v64_3_4_cc_aocc_fpcca_lba_daepc_screen_2gpu.yaml')
    return BDSEModel(cfg)


def test_fpcca_is_exact_step_zero_noop() -> None:
    model = _model()
    adapter = model.critical_proposal_adapter
    assert isinstance(adapter, _FrontierPairCriticalProposalAdapter)
    assert torch.count_nonzero(adapter.residual_head[-1].weight).item() == 0
    assert torch.count_nonzero(adapter.residual_head[-1].bias).item() == 0
    h = model.hidden_dim
    residual, attention = adapter(
        torch.randn(2, 7, 5 * h),
        torch.randn(2, 6, h),
        torch.randn(2, 6, h),
        torch.rand(2, 6),
        torch.ones((2, 6), dtype=torch.bool),
    )
    assert residual.shape == (2, 7)
    assert attention.shape == (2, 7, 6)
    assert torch.count_nonzero(residual).item() == 0


def test_fpcca_pair_pooling_is_permutation_invariant() -> None:
    model = _model()
    adapter = model.critical_proposal_adapter
    assert isinstance(adapter, _FrontierPairCriticalProposalAdapter)
    with torch.no_grad():
        torch.nn.init.normal_(adapter.residual_head[-1].weight, std=0.01)
    h = model.hidden_dim
    atom = torch.randn(2, 5, 5 * h)
    a_h = torch.randn(2, 6, h)
    b_h = torch.randn(2, 6, h)
    gap = torch.rand(2, 6)
    valid = torch.tensor([[True] * 6, [True, True, False, True, True, False]])
    order = torch.tensor([4, 0, 5, 2, 1, 3])
    out_a, attn_a = adapter(atom, a_h, b_h, gap, valid)
    out_b, attn_b = adapter(atom, a_h[:, order], b_h[:, order], gap[:, order], valid[:, order])
    torch.testing.assert_close(out_a, out_b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(attn_a[:, :, order], attn_b, rtol=1e-5, atol=1e-6)


def test_fpcca_all_masked_pair_set_stays_finite() -> None:
    model = _model()
    adapter = model.critical_proposal_adapter
    assert isinstance(adapter, _FrontierPairCriticalProposalAdapter)
    with torch.no_grad():
        torch.nn.init.normal_(adapter.residual_head[-1].weight, std=0.01)
    h = model.hidden_dim
    residual, attention = adapter(
        torch.randn(1, 3, 5 * h),
        torch.randn(1, 4, h),
        torch.randn(1, 4, h),
        torch.zeros(1, 4),
        torch.zeros((1, 4), dtype=torch.bool),
    )
    assert torch.isfinite(residual).all()
    # Attention is intentionally a large finite negative sentinel, not -inf,
    # so downstream diagnostics and CE remain numerically safe.
    assert torch.isfinite(attention).all()


def test_literal_boundary_attribution_uses_only_exact_flip_pair() -> None:
    # Full teacher winner is action 0. Removing atom 0 changes the winner to
    # action 2, so the literal boundary target is the pair {0,2}. Atom 1 is not
    # critical and receives no boundary-attribution target.
    J0 = torch.zeros((1, 3), dtype=torch.float32)
    g = torch.zeros((1, 2, 3), dtype=torch.float32)
    valid = torch.ones((1, 3), dtype=torch.bool)
    active = torch.ones((1, 2), dtype=torch.bool)
    proposal = torch.zeros((1, 2), dtype=torch.float32)
    deployment_hard = torch.tensor([[True, False]])
    target = torch.tensor([0], dtype=torch.long)
    atom_costs = torch.ones((1, 2), dtype=torch.float32)
    teacher_cost = torch.tensor([[0.0, 0.5, 0.4]], dtype=torch.float32)
    teacher_g = torch.tensor([[[-0.6, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    # Pair order: {0,1}, {0,2}, {1,2}. Uniform logits make CE finite/nonzero.
    pair_indices = torch.tensor([[[0, 1], [0, 2], [1, 2]]], dtype=torch.long)
    attention_logits = torch.zeros((1, 2, 3), dtype=torch.float32)
    cfg = {
        'training': {
            'exact_winner_flip_criticality': {
                'enabled': True,
                'target_source': 'teacher_interface',
                'positive_weight': 1.0,
                'negative_weight': 1.0,
                'rank_weight': 0.0,
                'pairwise_rank_weight': 0.0,
                'coverage_weight': 0.0,
                'exchange_rank_weight': 0.0,
                'adapter_residual_alignment_weight': 0.0,
                'boundary_attribution_weight': 1.0,
                'boundary_attribution_severity_weight': 0.0,
            }
        }
    }
    result = _exact_winner_flip_critical_proposal_loss(
        J0,
        g,
        valid,
        active,
        proposal,
        deployment_hard,
        target,
        atom_costs,
        cfg,
        teacher_cost=teacher_cost,
        teacher_g=teacher_g,
        critical_boundary_attention_logits=attention_logits,
        critical_boundary_pair_indices=pair_indices,
        return_adapter_diagnostic=True,
    )
    loss, recall, critical_fraction, scene_fraction, aligned_fraction, acra, lba, representable = result
    assert torch.isfinite(loss)
    assert float(recall) == 1.0
    assert float(critical_fraction) == 0.5
    assert float(scene_fraction) == 1.0
    assert float(aligned_fraction) == 1.0
    assert float(acra) == 0.0
    assert float(lba) > 0.0
    assert float(representable) == 1.0


def test_v64_3_4_fpcca_train_and_eval_contracts_pass() -> None:
    train = _check(
        Path('bdse/configs/v64_3_4_cc_aocc_fpcca_lba_daepc_train_2gpu.yaml'),
        'train',
        'v64.3.4',
    )
    evaluate = _check(
        Path('bdse/configs/v64_3_4_cc_aocc_fpcca_cl.yaml'),
        'eval',
        'v64.3.4',
    )
    assert train['pass'], train['failures']
    assert evaluate['pass'], evaluate['failures']
