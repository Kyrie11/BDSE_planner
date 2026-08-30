from __future__ import annotations

import inspect

import numpy as np
import torch

from bdse.config import load_config
from bdse.experiments.evaluate_open_loop import _criticality_metrics
from bdse.model.bdse_model import BDSEModel, _MultiRivalBoundaryCriticalProposalAdapter
from bdse.model.losses import compute_bdse_losses
from bdse.tools.validate_v64_pipeline_config import _check


def _adapter() -> _MultiRivalBoundaryCriticalProposalAdapter:
    cfg = load_config('bdse/configs/v64_3_4_cc_aocc_mrbcca_daepc_screen_2gpu.yaml')
    model = BDSEModel(cfg)
    assert isinstance(model.critical_proposal_adapter, _MultiRivalBoundaryCriticalProposalAdapter)
    return model.critical_proposal_adapter


def test_mrbcca_is_exact_step_zero_noop() -> None:
    cfg = load_config('bdse/configs/v64_3_4_cc_aocc_mrbcca_daepc_screen_2gpu.yaml')
    model = BDSEModel(cfg)
    adapter = model.critical_proposal_adapter
    assert isinstance(adapter, _MultiRivalBoundaryCriticalProposalAdapter)
    assert torch.count_nonzero(adapter.residual_head[-1].weight).item() == 0
    assert torch.count_nonzero(adapter.residual_head[-1].bias).item() == 0
    h = model.hidden_dim
    out = adapter(
        torch.randn(2, 7, 6 * h),
        torch.randn(2, h),
        torch.randn(2, 4, h),
        torch.rand(2, 4),
        torch.ones((2, 4), dtype=torch.bool),
    )
    assert out.shape == (2, 7)
    assert torch.count_nonzero(out).item() == 0


def test_mrbcca_frontier_pooling_is_permutation_invariant() -> None:
    adapter = _adapter()
    with torch.no_grad():
        torch.nn.init.normal_(adapter.residual_head[-1].weight, std=0.01)
    h = adapter.atom_proj[1].in_features // 6
    atom = torch.randn(2, 5, 6 * h)
    winner = torch.randn(2, h)
    rivals = torch.randn(2, 4, h)
    gap = torch.rand(2, 4)
    valid = torch.tensor([[True, True, True, True], [True, True, False, True]])
    order = torch.tensor([2, 0, 3, 1])
    a = adapter(atom, winner, rivals, gap, valid)
    b = adapter(atom, winner, rivals[:, order], gap[:, order], valid[:, order])
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_mrbcca_all_masked_frontier_stays_finite() -> None:
    adapter = _adapter()
    with torch.no_grad():
        torch.nn.init.normal_(adapter.residual_head[-1].weight, std=0.01)
    h = adapter.atom_proj[1].in_features // 6
    out = adapter(
        torch.randn(1, 3, 6 * h),
        torch.randn(1, h),
        torch.randn(1, 4, h),
        torch.zeros(1, 4),
        torch.zeros((1, 4), dtype=torch.bool),
    )
    assert torch.isfinite(out).all()


def test_literal_critical_boundary_frontier_diagnostic() -> None:
    # Dense teacher winner is action 0. Removing the only active atom changes
    # the winner to action 2. The reference base ranks actions [0, 1, 2], so a
    # one-rival context (top2) cannot represent the boundary, while two rivals
    # (top3) can.
    base = np.array([0.3, 0.4, 0.2], dtype=np.float32)
    atom = np.array([[-0.2, 0.0, 0.0]], dtype=np.float32)
    values, _ = _criticality_metrics(
        base,
        atom,
        np.array([True]),
        np.array([True, True, True]),
        np.array([0]),
        np.array([0]),
        prefix='unit',
        forced_winner=0,
        reference_action_cost=np.array([0.0, 0.1, 0.2], dtype=np.float32),
        reference_frontier_rivals=(1, 2),
    )
    assert values['unit_critical_count'] == 1.0
    assert values['unit_critical_boundary_in_base_top2_fraction'] == 0.0
    assert values['unit_critical_boundary_in_base_top3_fraction'] == 1.0


def test_acra_standalone_diagnostic_is_exported_by_compute_losses() -> None:
    # Regression for V64.3.3: ACRA affected the exact-critical loss but the
    # standalone diagnostic key was omitted, causing the screen to call a real
    # optimization path "unwired".
    source = inspect.getsource(compute_bdse_losses)
    assert '"L_critical_adapter_residual_alignment": L_critical_adapter_residual_alignment' in source


def test_v64_3_4_train_and_eval_contracts_pass() -> None:
    train = _check(
        __import__('pathlib').Path('bdse/configs/v64_3_4_cc_aocc_mrbcca_daepc_train_2gpu.yaml'),
        'train',
        'v64.3.4',
    )
    evaluate = _check(
        __import__('pathlib').Path('bdse/configs/v64_3_4_cc_aocc_mrbcca_cl.yaml'),
        'eval',
        'v64.3.4',
    )
    assert train['pass'], train['failures']
    assert evaluate['pass'], evaluate['failures']
