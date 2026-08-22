from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import yaml

from bdse.planner.proposal_conditioned_witness_rebinding import proposal_conditioned_witness_rebind
from bdse.planner.tournament import run_pair_conditioned_tournament
from bdse.tools.fit_v64_3_30_eaf_icer_pcwer import _proposal_map


def _problem():
    j0=np.asarray([0.0,10.0,20.0],dtype=np.float32)
    g=np.asarray([[0,0,0],[0,5,0],[0,0,5]],dtype=np.float32)
    valid=np.ones(3,dtype=bool)
    pairs=np.zeros((0,2),dtype=np.int64)
    delta=np.zeros((3,0),dtype=np.float32)
    atom_f=np.zeros((3,2),dtype=np.float32)
    signed=np.zeros((3,2),dtype=np.float32)
    context=np.zeros((3,2),dtype=np.float32)
    return j0,g,valid,pairs,delta,atom_f,signed,context


def _eval_factory(fail_candidate=False):
    def ev(selected):
        q=1 if fail_candidate and set(selected)=={0,2} else 2
        return {'proposal_action':q,'incumbent_action':1,'anchor_action':0,'incumbent_admissible':True}
    return ev


def test_pcwer_accepts_same_budget_strict_operator_witness_improvement():
    j0,g,valid,pairs,delta,af,signed,context=_problem()
    r=proposal_conditioned_witness_rebind(
        baseline_selected=[0,1],reference_atoms=[0,1,2],predicted_base_cost=j0,predicted_atom_costs=g,
        pair_indices=pairs,pair_atom_delta=delta,valid_mask=valid,atom_budget_costs=np.ones(3,dtype=np.float32),budget=2.0,
        normalize_margins=False,margin_scale=1.0,pair_delta_includes_local=True,
        frontier_value_atom_factors=af,frontier_value_action_signed_factors=signed,
        frontier_value_action_context_factors=context,frontier_value_scale=1.0,proposal_evaluator=_eval_factory(),
    )
    assert r.diagnostics['proposal_conditioned_witness_rebinding_accepted']==1.0
    assert r.proposal_lock and r.proposal_action==2
    assert r.selected==[0,2]
    assert len(r.selected)==2
    before=(r.diagnostics['proposal_conditioned_witness_rebinding_baseline_margin_linf_error'],r.diagnostics['proposal_conditioned_witness_rebinding_baseline_attribution_linf_error'],r.diagnostics['proposal_conditioned_witness_rebinding_baseline_margin_rms_error'],r.diagnostics['proposal_conditioned_witness_rebinding_baseline_attribution_rms_error'])
    after=(r.diagnostics['proposal_conditioned_witness_rebinding_candidate_margin_linf_error'],r.diagnostics['proposal_conditioned_witness_rebinding_candidate_attribution_linf_error'],r.diagnostics['proposal_conditioned_witness_rebinding_candidate_margin_rms_error'],r.diagnostics['proposal_conditioned_witness_rebinding_candidate_attribution_rms_error'])
    assert after < before


def test_pcwer_fails_closed_if_exact_proposal_changes():
    j0,g,valid,pairs,delta,af,signed,context=_problem()
    r=proposal_conditioned_witness_rebind(
        baseline_selected=[0,1],reference_atoms=[0,1,2],predicted_base_cost=j0,predicted_atom_costs=g,
        pair_indices=pairs,pair_atom_delta=delta,valid_mask=valid,atom_budget_costs=np.ones(3,dtype=np.float32),budget=2.0,
        normalize_margins=False,margin_scale=1.0,pair_delta_includes_local=True,
        frontier_value_atom_factors=af,frontier_value_action_signed_factors=signed,
        frontier_value_action_context_factors=context,frontier_value_scale=1.0,proposal_evaluator=_eval_factory(fail_candidate=True),
    )
    assert r.selected==[0,1]
    assert r.proposal_lock and r.proposal_action==2
    assert r.diagnostics['proposal_conditioned_witness_rebinding_accepted']==0.0
    assert r.diagnostics['proposal_conditioned_witness_rebinding_reason_code']==8.0



def test_pcwer_no_improvement_fails_closed_on_evidence_but_keeps_same_proposal_lock():
    j0,g,valid,pairs,delta,af,signed,context=_problem()
    r=proposal_conditioned_witness_rebind(
        baseline_selected=[0,1],reference_atoms=[0,1],predicted_base_cost=j0,predicted_atom_costs=g,
        pair_indices=pairs,pair_atom_delta=delta,valid_mask=valid,atom_budget_costs=np.ones(3,dtype=np.float32),budget=2.0,
        normalize_margins=False,margin_scale=1.0,pair_delta_includes_local=True,
        frontier_value_atom_factors=af,frontier_value_action_signed_factors=signed,
        frontier_value_action_context_factors=context,frontier_value_scale=1.0,proposal_evaluator=_eval_factory(),
    )
    assert r.selected==[0,1]
    assert r.proposal_lock and r.proposal_action==2
    assert r.diagnostics['proposal_conditioned_witness_rebinding_accepted']==0.0
    assert r.diagnostics['proposal_conditioned_witness_rebinding_reason_code']==6.0


def test_v30_fitter_reads_proposal_from_selector_lock_diagnostics_not_frontier_selected_action():
    p='selector_proposal_conditioned_witness_rebinding_'
    rows=[{
        'scenario_token':'scene-a',
        p+'proposal_lock':1.0,
        p+'baseline_proposal_action':7.0,
        p+'baseline_incumbent_action':3.0,
        p+'baseline_anchor_action':1.0,
        'icer_selected_action':9.0,
    },{
        'scenario_token':'scene-b',
        p+'proposal_lock':0.0,
        p+'baseline_proposal_action':5.0,
        p+'baseline_incumbent_action':2.0,
        p+'baseline_anchor_action':1.0,
        'icer_selected_action':5.0,
    }]
    assert _proposal_map(rows)=={'scene-a':7}


def test_proposal_lock_only_control_keeps_original_evidence_and_locks_proposal():
    j0,g,valid,pairs,delta,af,signed,context=_problem()
    r=proposal_conditioned_witness_rebind(
        baseline_selected=[0,1],reference_atoms=[0,1,2],predicted_base_cost=j0,predicted_atom_costs=g,
        pair_indices=pairs,pair_atom_delta=delta,valid_mask=valid,atom_budget_costs=np.ones(3,dtype=np.float32),budget=2.0,
        normalize_margins=False,margin_scale=1.0,pair_delta_includes_local=True,
        frontier_value_atom_factors=af,frontier_value_action_signed_factors=signed,
        frontier_value_action_context_factors=context,frontier_value_scale=1.0,proposal_evaluator=_eval_factory(),rebind_enabled=False,
    )
    assert r.selected==[0,1]
    assert r.proposal_lock and r.proposal_action==2
    assert r.diagnostics['proposal_conditioned_witness_rebinding_lock_only']==1.0
    assert r.diagnostics['proposal_conditioned_witness_rebinding_reason_code']==10.0


def test_pcwer_structural_domain_is_identity_and_has_no_proposal_lock():
    j0,g,valid,pairs,delta,af,signed,context=_problem()
    r=proposal_conditioned_witness_rebind(
        baseline_selected=[0,1],reference_atoms=[0,1,2],predicted_base_cost=j0,predicted_atom_costs=g,
        pair_indices=pairs,pair_atom_delta=delta,valid_mask=valid,atom_budget_costs=np.ones(3,dtype=np.float32),budget=2.0,
        normalize_margins=False,margin_scale=1.0,pair_delta_includes_local=True,
        frontier_value_atom_factors=af,frontier_value_action_signed_factors=signed,
        frontier_value_action_context_factors=context,frontier_value_scale=1.0,proposal_evaluator=_eval_factory(),structural_bypass=True,
    )
    assert r.selected==[0,1] and not r.proposal_lock
    assert r.diagnostics['proposal_conditioned_witness_rebinding_structural_bypass']==1.0


def test_v30_configs_keep_B16_M24_and_separate_rebinding_from_lock_control():
    main=yaml.safe_load(Path('bdse/configs/v64_3_30_eaf_icer_pcwer_v20.yaml').read_text())
    ctrl=yaml.safe_load(Path('bdse/configs/v64_3_30_eaf_icer_proposal_lock_v20.yaml').read_text())
    for cfg in [main,ctrl]:
        assert cfg['evidence']['budget']==16
        assert cfg['selector']['proposal_top_m']==24
        pc=cfg['selector']['proposal_conditioned_witness_rebinding']
        assert pc['enabled'] is True and pc['teacher_labels'] is False
        assert pc['additional_evidence_queries']==0 and pc['beam_swap_repair'] is False
        assert cfg['selector']['frontier_contrast_rebinding']['enabled'] is False
    assert main['selector']['proposal_conditioned_witness_rebinding']['rebind_enabled'] is True
    assert ctrl['selector']['proposal_conditioned_witness_rebinding']['rebind_enabled'] is False


def test_v30_tournament_api_has_same_proposal_lock_and_no_second_best_contract():
    sig=inspect.signature(run_pair_conditioned_tournament)
    assert 'recovery_proposal_action' in sig.parameters
    src=inspect.getsource(run_pair_conditioned_tournament)
    assert 'forced_proposal_action=recovery_proposal_action' in src
    planner_src=Path('bdse/planner/nuplan_planner.py').read_text()
    assert 'recovery_proposal_action = -1' in planner_src
    assert 'explicit locked abstention sentinel' in planner_src


def test_v30_exclusion_contains_8700_unique_inspected_tokens():
    toks=[x.strip() for x in Path('bdse/configs/v64_3_30_design_exclude_v64_3_29_screen_tokens.txt').read_text().splitlines() if x.strip()]
    assert len(toks)==8700 and len(set(toks))==8700
