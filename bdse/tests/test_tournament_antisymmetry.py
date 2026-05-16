from __future__ import annotations

import numpy as np

from bdse.planner.selector import budgeted_margin
from bdse.planner.tournament import assert_antisymmetric


def test_tournament_antisymmetry(synthetic_sample):
    J0 = np.nan_to_num(synthetic_sample.teacher.J_base, posinf=1e6)
    g = synthetic_sample.teacher.g_evid
    M = budgeted_margin(J0, g, list(range(min(4, g.shape[0]))))
    assert_antisymmetric(M, synthetic_sample.candidates.valid_mask)
