from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bdse.config import load_config
from bdse.model.bdse_model import BDSEModel, _CompleteCandidateBoundaryRouter, _DecisiveBoundaryPairResidual
from bdse.model.decisive_margin_utility import (
    BDMUConfig,
    budgeted_decisive_margin_utility_numpy,
    budgeted_decisive_margin_utility_torch,
)


def test_bdmu_matches_numpy_and_values_add_remove_utility() -> None:
    # Full teacher margin is 10.  The frozen B-set keeps only the weaker atom1
    # (margin contribution 4), leaving a decisive-margin deficit.  Atom0 should
    # therefore have positive add-value; atom1 has positive removal-value.
    teacher_cost = torch.tensor([[0.0, 10.0]])
    teacher_g = torch.tensor([[[0.0, 6.0], [0.0, 4.0]]])
    active = torch.tensor([[True, True]])
    valid = torch.tensor([[True, True]])
    action = torch.tensor([0])
    ref = torch.tensor([[False, True]])
    costs = torch.tensor([[1.0, 1.0]])
    cfg = BDMUConfig(
        budget=1.0,
        rival_count=1,
        preserve_fraction=0.60,
        margin_floor=0.0,
        margin_cap=10.0,
        rival_temperature=0.2,
        min_action_scale=1.0,
        cost_power=1.0,
    )
    u_t, d_t = budgeted_decisive_margin_utility_torch(
        teacher_cost, teacher_g, active, valid, action, ref, costs, cfg
    )
    u_n, d_n = budgeted_decisive_margin_utility_numpy(
        teacher_cost[0].numpy(), teacher_g[0].numpy(), active[0].numpy(), valid[0].numpy(),
        0, ref[0].numpy(), costs[0].numpy(), cfg
    )
    np.testing.assert_allclose(u_t[0].numpy(), u_n, rtol=1e-6, atol=1e-6)
    assert float(u_t[0, 0]) > 0.0  # add-gain for missed decisive support
    assert float(u_t[0, 1]) > 0.0  # removal-loss for retained support
    assert bool(d_t['scene_has_utility'][0])
    assert d_n['scene_has_utility'] == 1.0



def test_bdmu_outside_value_never_uses_infeasible_b_plus_one_addition() -> None:
    # Full teacher margin=5 from atoms (4,1).  At B=1 the reference already
    # spends the full budget on the stronger atom.  Directly adding atom1 would
    # close the deficit, but that is a forbidden B+1 intervention; the only
    # feasible swap replaces contribution 4 by contribution 1 and has zero gain.
    teacher_cost = torch.tensor([[0.0, 5.0]])
    teacher_g = torch.tensor([[[0.0, 4.0], [0.0, 1.0]]])
    active = torch.tensor([[True, True]])
    valid = torch.tensor([[True, True]])
    ref = torch.tensor([[True, False]])
    costs = torch.tensor([[1.0, 1.0]])
    cfg = BDMUConfig(
        budget=1.0, rival_count=1, preserve_fraction=1.0, margin_floor=0.0,
        margin_cap=10.0, min_action_scale=1.0, cost_power=1.0,
    )
    utility, _ = budgeted_decisive_margin_utility_torch(
        teacher_cost, teacher_g, active, valid, torch.tensor([0]), ref, costs, cfg
    )
    assert float(utility[0, 1]) == 0.0


def test_bdmu_cost_normalization_prefers_equal_margin_support_with_lower_query_cost() -> None:
    teacher_cost = torch.tensor([[0.0, 8.0]])
    teacher_g = torch.tensor([[[0.0, 4.0], [0.0, 4.0]]])
    active = torch.tensor([[True, True]])
    valid = torch.tensor([[True, True]])
    # Empty reference produces equal add-value; atom0 is cheaper.
    ref = torch.tensor([[False, False]])
    costs = torch.tensor([[1.0, 2.0]])
    cfg = BDMUConfig(
        rival_count=1, preserve_fraction=0.5, margin_floor=0.0, margin_cap=10.0,
        min_action_scale=1.0, cost_power=1.0,
    )
    utility, _ = budgeted_decisive_margin_utility_torch(
        teacher_cost, teacher_g, active, valid, torch.tensor([0]), ref, costs, cfg
    )
    assert float(utility[0, 0]) > float(utility[0, 1])
    torch.testing.assert_close(utility[0, 0], 2.0 * utility[0, 1], rtol=1e-5, atol=1e-6)


def test_bdmu_excludes_scalar_teacher_misalignment() -> None:
    teacher_cost = torch.tensor([[2.0, 0.0]])
    teacher_g = torch.zeros((1, 2, 2))
    utility, diag = budgeted_decisive_margin_utility_torch(
        teacher_cost,
        teacher_g,
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones((1, 2), dtype=torch.bool),
        torch.tensor([0]),
        torch.tensor([[True, False]]),
        torch.ones((1, 2)),
        BDMUConfig(),
    )
    assert torch.count_nonzero(utility).item() == 0
    assert not bool(diag['aligned'][0])


def test_v64_3_8_config_is_strict_acquisition_isolation() -> None:
    cfg = load_config('bdse/configs/v64_3_8_cc_aocc_bdmu_daepc_screen_2gpu.yaml')
    assert int(cfg['evidence']['budget']) == 16
    assert cfg['runtime']['pair_tournament_aggregation_mode'] == 'decisive_anchor_margin'
    assert cfg['model']['decisive_boundary_pair_adapter']['enabled'] is True
    assert cfg['model']['critical_proposal_adapter']['enabled'] is True
    assert cfg['model']['critical_proposal_adapter']['conditioning'] == 'complete_candidate_boundary_routing'
    assert cfg['model']['critical_proposal_adapter']['family_coupling']['enabled'] is False
    assert cfg['training']['trainable_modules'] == ['critical_proposal_adapter']
    assert cfg['training']['loss_weights']['budgeted_decisive_margin_utility'] == 1.0
    assert cfg['training']['loss_weights']['exact_winner_flip_critical_proposal'] == 0.0
    assert cfg['training']['exact_winner_flip_criticality']['enabled'] is False
    assert cfg['training']['budgeted_decisive_margin_utility']['reference_source'] == 'frozen_foundation_fast_budget'
    assert cfg['training']['budgeted_decisive_margin_utility']['exchange_mode'] == 'best_budget_feasible_single_exchange'


def test_v64_3_8_zero_init_preserves_foundation_proposal_and_keeps_dbr_loaded() -> None:
    cfg = load_config('bdse/configs/v64_3_8_cc_aocc_bdmu_daepc_screen_2gpu.yaml')
    model = BDSEModel(cfg)
    assert isinstance(model.critical_proposal_adapter, _CompleteCandidateBoundaryRouter)
    assert isinstance(model.decisive_boundary_pair_adapter, _DecisiveBoundaryPairResidual)
    # CCBR residual is an exact no-op at initialization, so warm-starting the
    # promoted DARM+DBR checkpoint preserves its acquisition ranking at step 0.
    assert torch.count_nonzero(model.critical_proposal_adapter.residual_head[-1].weight).item() == 0


def test_v64_3_8_ablation_configs_change_only_theory_knob() -> None:
    main = load_config('bdse/configs/v64_3_8_cc_aocc_bdmu_daepc_screen_2gpu.yaml')
    r1 = load_config('bdse/configs/v64_3_8_cc_aocc_bdmu_r1_daepc_screen_2gpu.yaml')
    nocost = load_config('bdse/configs/v64_3_8_cc_aocc_bdmu_nocost_daepc_screen_2gpu.yaml')
    m = main['training']['budgeted_decisive_margin_utility']
    assert r1['training']['budgeted_decisive_margin_utility']['rival_count'] == 1
    assert nocost['training']['budgeted_decisive_margin_utility']['cost_power'] == 0.0
    assert m['rival_count'] == 4 and m['cost_power'] == 1.0
    assert r1['evidence']['budget'] == nocost['evidence']['budget'] == main['evidence']['budget']


def test_v64_3_8_bdmu_only_fast_path_is_strictly_opt_in() -> None:
    from bdse.model.losses import _bdmu_only_loss_mode

    cfg = load_config('bdse/configs/v64_3_8_cc_aocc_bdmu_daepc_screen_2gpu.yaml')
    tr = cfg['training']
    assert _bdmu_only_loss_mode(tr)
    changed = dict(tr)
    changed['loss_weights'] = dict(tr['loss_weights'])
    changed['loss_weights']['pair'] = 1e-3
    assert not _bdmu_only_loss_mode(changed)
    disabled = dict(tr)
    disabled['budgeted_decisive_margin_utility'] = dict(tr['budgeted_decisive_margin_utility'])
    disabled['budgeted_decisive_margin_utility']['enabled'] = False
    assert not _bdmu_only_loss_mode(disabled)


def test_v64_3_8_checkpoint_contract_requires_pretrained_dbr() -> None:
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(Path('bdse/configs/v64_3_8_cc_aocc_bdmu_daepc_screen_2gpu.yaml').read_text())
    missing = set(cfg['checkpoint_loading']['allowed_missing_prefixes'])
    assert 'critical_proposal_adapter.' in missing
    assert 'decisive_boundary_pair_adapter.' not in missing


def test_bdmu_diagnostics_are_opt_in_for_legacy_configs() -> None:
    from bdse.experiments.evaluate_open_loop import _bdmu_metrics_enabled

    old = load_config('bdse/configs/v64_3_7_cc_aocc_darm_dbr_literal_daepc_screen_2gpu.yaml')
    new = load_config('bdse/configs/v64_3_8_cc_aocc_bdmu_daepc_screen_2gpu.yaml')
    assert not _bdmu_metrics_enabled(old)
    assert _bdmu_metrics_enabled(new)
