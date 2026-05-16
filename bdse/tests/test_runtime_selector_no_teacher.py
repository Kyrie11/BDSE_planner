from __future__ import annotations

import inspect

import numpy as np

from bdse.planner.selector import runtime_greedy_selector, runtime_objective_value


def test_runtime_selector_no_teacher_inputs(synthetic_sample, cfg):
    sig = inspect.signature(runtime_greedy_selector)
    forbidden = {"J_T", "M_T", "a_T_star", "P_t_plus", "future_agents"}
    assert forbidden.isdisjoint(sig.parameters.keys())
    J0 = np.nan_to_num(synthetic_sample.teacher.J_base, posinf=1e6)
    g = synthetic_sample.teacher.g_evid
    safety = synthetic_sample.teacher.hard_violation_mask
    r1 = runtime_greedy_selector(J0, g, synthetic_sample.evidence_bank.budget_costs(), synthetic_sample.candidates.valid_mask, safety, 4, atom_active_mask=synthetic_sample.evidence_bank.active_mask)
    r2 = runtime_greedy_selector(J0, g, synthetic_sample.evidence_bank.budget_costs(), synthetic_sample.candidates.valid_mask, safety, 4, atom_active_mask=synthetic_sample.evidence_bank.active_mask)
    assert r1.selected == r2.selected
    assert sum(synthetic_sample.evidence_bank.budget_costs()[r1.selected]) <= 4 + 1e-6
