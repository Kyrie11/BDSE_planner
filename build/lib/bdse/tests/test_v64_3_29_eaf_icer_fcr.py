from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from bdse.planner.frontier_contrast_rebinding import frontier_contrast_rebind


def _synthetic_problem():
    # Three already-queried evidence atoms, retain exactly two. Atom 0 is
    # deliberately redundant for the complete frontier, so [1,2] is the exact
    # full-M contrast-preserving subset while the baseline [0,1] is not.
    j0 = np.asarray([0.0, 10.0, 20.0], dtype=np.float32)
    g = np.asarray([
        [0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
    ], dtype=np.float32)
    valid = np.ones((3,), dtype=bool)
    pair_indices = np.zeros((0, 2), dtype=np.int64)
    pair_delta = np.zeros((3, 0), dtype=np.float32)
    atom_factors = np.zeros((3, 2), dtype=np.float32)
    signed = np.zeros((3, 2), dtype=np.float32)
    context = np.zeros((3, 2), dtype=np.float32)
    return j0, g, valid, pair_indices, pair_delta, atom_factors, signed, context


def test_fcr_accepts_only_same_budget_strict_frontier_improvement():
    j0, g, valid, pairs, delta, atom_f, signed, context = _synthetic_problem()

    def exact_eval(selected):
        # Synthetic exact downstream action equals selected-local anchor.
        total = j0 + g[np.asarray(selected, dtype=np.int64)].sum(axis=0)
        return int(np.argmin(total)), np.asarray([]), np.asarray([]), {}

    r = frontier_contrast_rebind(
        baseline_selected=[0, 1],
        reference_atoms=[0, 1, 2],
        predicted_base_cost=j0,
        predicted_atom_costs=g,
        pair_indices=pairs,
        pair_atom_delta=delta,
        valid_mask=valid,
        atom_budget_costs=np.ones((3,), dtype=np.float32),
        budget=2.0,
        normalize_margins=False,
        margin_scale=1.0,
        pair_delta_includes_local=True,
        frontier_value_atom_factors=atom_f,
        frontier_value_action_signed_factors=signed,
        frontier_value_action_context_factors=context,
        frontier_value_scale=1.0,
        deployment_evaluator=exact_eval,
        full_target_action=0,
    )
    assert r.diagnostics['frontier_contrast_rebinding_accepted'] == 1.0
    assert r.selected == [1, 2]
    assert len(r.selected) == 2
    assert r.diagnostics['frontier_contrast_rebinding_candidate_linf_error'] < r.diagnostics['frontier_contrast_rebinding_baseline_linf_error']
    assert r.diagnostics['frontier_contrast_rebinding_local_anchor_preserved'] == 1.0
    assert r.diagnostics['frontier_contrast_rebinding_candidate_exact_certificate'] == 1.0


def test_fcr_fails_closed_when_exact_target_would_change():
    j0, g, valid, pairs, delta, atom_f, signed, context = _synthetic_problem()

    def exact_eval(selected):
        # Make the frontier-optimal rebind fail the exact downstream contract.
        action = 1 if set(selected) == {1, 2} else 0
        return action, np.asarray([]), np.asarray([]), {}

    r = frontier_contrast_rebind(
        baseline_selected=[0, 1], reference_atoms=[0, 1, 2],
        predicted_base_cost=j0, predicted_atom_costs=g,
        pair_indices=pairs, pair_atom_delta=delta, valid_mask=valid,
        atom_budget_costs=np.ones((3,), dtype=np.float32), budget=2.0,
        normalize_margins=False, margin_scale=1.0, pair_delta_includes_local=True,
        frontier_value_atom_factors=atom_f,
        frontier_value_action_signed_factors=signed,
        frontier_value_action_context_factors=context,
        frontier_value_scale=1.0, deployment_evaluator=exact_eval, full_target_action=0,
    )
    assert r.selected == [0, 1]
    assert r.diagnostics['frontier_contrast_rebinding_accepted'] == 0.0
    assert r.diagnostics['frontier_contrast_rebinding_reason_code'] == 9.0
    assert r.diagnostics['frontier_contrast_rebinding_final_linf_error'] == r.diagnostics['frontier_contrast_rebinding_baseline_linf_error']


def test_v29_config_keeps_literal_budget_and_forbids_old_search_branches():
    cfg = yaml.safe_load(Path('bdse/configs/v64_3_29_eaf_icer_fcr_v20.yaml').read_text(encoding='utf-8'))
    fcr = cfg['selector']['frontier_contrast_rebinding']
    assert cfg['evidence']['budget'] == 16
    assert cfg['selector']['proposal_top_m'] == 24
    assert fcr['enabled'] is True
    assert fcr['teacher_labels'] is False
    assert fcr['additional_evidence_queries'] == 0
    assert fcr['beam_swap_repair'] is False
    assert 'complete_darm_eaf_anchor_star' in fcr['objective']
    assert cfg['metadata']['algorithm_version'] == 'V64.3.29-EAF-ICER-FCR'


def test_v29_exclusion_contains_7700_unique_inspected_tokens():
    p = Path('bdse/configs/v64_3_29_design_exclude_v64_3_28_screen_tokens.txt')
    tokens = [x.strip() for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    assert len(tokens) == 7700
    assert len(set(tokens)) == 7700


def test_v29_launcher_preregisters_interface_screen_and_no_ptmc_arm():
    s = Path('RUN_V64_3_29_EAF_ICER_FCR_SCREEN_2GPU.sh').read_text(encoding='utf-8')
    assert 'v64_3_29_design_exclude_v64_3_28_screen_tokens.txt' in s
    assert "len(ex)!=7700" in s
    assert 'b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4' in s
    assert 'test_v64_3_29_eaf_icer_fcr.py' in s
    assert 'fcr_downside' in s
    # Failed V28 PTMC must not be evaluated as a V29 arm.
    assert 'tail_mode_confirmed' not in s
    assert 'v64.3.29-eaf-icer-fcr-double-fresh-v1' in s
