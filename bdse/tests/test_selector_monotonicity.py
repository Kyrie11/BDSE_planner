from __future__ import annotations

import numpy as np

from bdse.planner.selector import build_predicted_pairs, runtime_objective_value, full_interface_margin


def test_selector_monotonicity(synthetic_sample):
    J0 = np.nan_to_num(synthetic_sample.teacher.J_base, posinf=1e6)
    g = synthetic_sample.teacher.g_evid
    safety = synthetic_sample.teacher.hard_violation_mask
    M = full_interface_margin(J0, g)
    pairs, weights = build_predicted_pairs(M, synthetic_sample.candidates.valid_mask, safety, 16, 1.0)
    selected = []
    last = runtime_objective_value(selected, J0, g, pairs, weights, 100.0)
    for i in range(min(5, g.shape[0])):
        selected.append(i)
        val = runtime_objective_value(selected, J0, g, pairs, weights, 100.0)
        assert val + 1e-6 >= last
        last = val
