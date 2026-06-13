from __future__ import annotations

import numpy as np

from bdse.planner.selector import build_predicted_pairs, runtime_objective_value, full_interface_margin, runtime_greedy_selector


def test_greedy_selector_improves_signed_objective(synthetic_sample):
    J0 = np.nan_to_num(synthetic_sample.teacher.J_base, posinf=1e6)
    g = synthetic_sample.teacher.g_evid
    safety = synthetic_sample.teacher.hard_violation_mask
    M = full_interface_margin(J0, g)
    pairs, weights = build_predicted_pairs(M, synthetic_sample.candidates.valid_mask, safety, 16, 1.0)
    base_value = runtime_objective_value([], J0, g, pairs, weights, 100.0)
    sel = runtime_greedy_selector(
        J0, g, synthetic_sample.evidence_bank.budget_costs(), synthetic_sample.candidates.valid_mask, safety,
        budget=8.0, L_infer=16, gamma_max=100.0, eta_pred=1.0, atom_active_mask=synthetic_sample.evidence_bank.active_mask
    )
    greedy_value = runtime_objective_value(sel.selected, J0, g, pairs, weights, 100.0)
    assert greedy_value + 1e-6 >= base_value


def test_signed_objective_accounts_for_harmful_atom_deltas():
    J0 = np.asarray([0.0, 0.0, 6.0], dtype=np.float32)
    # Atom 0 supports pair (0,1) but erases an existing base margin for (0,2).
    # The signed objective must account for that harm.
    g = np.asarray([
        [0.0, 10.0, -10.0],
        [0.0, 0.0, 8.0],
    ], dtype=np.float32)
    pairs = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
    weights = np.ones((2,), dtype=np.float32)
    unsigned_like_wrong = 10.0 + 6.0
    signed_value = runtime_objective_value([0], J0, g, pairs, weights, gamma_max=100.0)
    assert signed_value < unsigned_like_wrong
    assert np.isclose(signed_value, 10.0)
