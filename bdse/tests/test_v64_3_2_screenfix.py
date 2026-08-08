from __future__ import annotations

import torch

from bdse.config import load_config
from bdse.experiments.train import (
    _adapter_parameter_delta_metrics,
    _adapter_parameter_snapshot,
    _set_frozen_top_level_modules_eval,
)
from bdse.model.bdse_model import BDSEModel
from bdse.model.losses import _exact_winner_flip_critical_proposal_loss


def test_adapter_parameter_delta_detects_real_update() -> None:
    cfg = load_config('bdse/configs/v64_3_2_cc_aocc_apwcca_daepc_screen_2gpu.yaml')
    model = BDSEModel(cfg)
    ref = _adapter_parameter_snapshot(model)
    assert ref
    with torch.no_grad():
        model.critical_proposal_adapter[-1].bias.add_(0.01)
    diag = _adapter_parameter_delta_metrics(model, ref)
    assert diag['critical_adapter_parameter_delta_rms'] > 0.0
    assert diag['critical_adapter_parameter_delta_max_abs'] >= 0.0099


def test_frozen_foundation_modules_stay_eval_during_head_finetune() -> None:
    cfg = load_config('bdse/configs/v64_3_2_cc_aocc_apwcca_daepc_screen_2gpu.yaml')
    model = BDSEModel(cfg)
    trainable = set(cfg['training']['trainable_modules'])
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if any(name == x or name.startswith(x + '.') for x in trainable):
            p.requires_grad_(True)
    model.train()
    frozen = _set_frozen_top_level_modules_eval(model, cfg)
    assert 'scene' in frozen and model.scene.training is False
    assert 'action' in frozen and model.action.training is False
    assert model.critical_proposal_adapter.training is True


def test_acra_gives_zero_init_adapter_a_direct_gradient() -> None:
    J0 = torch.tensor([[0.0, 0.2]])
    g = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, True]])
    active = torch.tensor([[True, True]])
    proposal_logits = torch.zeros((1, 2), requires_grad=True)
    residual_logits = torch.zeros((1, 2), requires_grad=True)
    deployed = torch.tensor([[False, True]])
    cfg = {
        'training': {
            'exact_winner_flip_criticality': {
                'enabled': True,
                'target_source': 'model_dense',
                'positive_weight': 1.0,
                'negative_weight': 1.0,
                'rank_weight': 0.0,
                'pairwise_rank_weight': 0.0,
                'coverage_weight': 0.0,
                'exchange_rank_weight': 0.0,
                'adapter_residual_alignment_weight': 1.0,
                'adapter_residual_target_scale': 1.0,
                'adapter_residual_huber_delta': 0.25,
                'adapter_residual_positive_weight': 8.0,
                'min_action_scale': 1.0,
            }
        }
    }
    loss, *_ = _exact_winner_flip_critical_proposal_loss(
        J0, g, valid, active, proposal_logits, deployed,
        torch.tensor([1]), torch.ones((1, 2)), cfg,
        critical_residual_logits=residual_logits,
    )
    loss.backward()
    assert residual_logits.grad is not None
    assert float(residual_logits.grad.abs().sum()) > 0.0


def test_winner_rival_adapter_is_zero_output_at_step_zero() -> None:
    cfg = load_config('bdse/configs/v64_3_2_cc_aocc_apwrcca_daepc_screen_2gpu.yaml')
    model = BDSEModel(cfg)
    assert model.critical_proposal_conditioning == 'frozen_base_winner_rival_actions'
    last = model.critical_proposal_adapter[-1]
    assert torch.count_nonzero(last.weight).item() == 0
    assert torch.count_nonzero(last.bias).item() == 0


def test_checkpoint_contract_rejects_present_adapter_shape_mismatch() -> None:
    from bdse.model.checkpoint_contract import load_bdse_state_with_contract
    cfg_a = load_config('bdse/configs/v64_3_2_cc_aocc_apwcca_daepc_screen_2gpu.yaml')
    cfg_b = load_config('bdse/configs/v64_3_2_cc_aocc_apwrcca_daepc_screen_2gpu.yaml')
    model_a = BDSEModel(cfg_a)
    model_b = BDSEModel(cfg_b)
    state = model_b.state_dict()
    try:
        load_bdse_state_with_contract(model_a, state, cfg_a, context='shape-mismatch-unit')
    except ValueError as exc:
        assert 'shape_mismatch' in str(exc)
    else:
        raise AssertionError('present AP-WRCCA adapter must not be silently loaded into AP-WCCA architecture')
