from __future__ import annotations

import numpy as np

from bdse.planner.pair_builder import margin_matrix
from bdse.planner.teacher_cost import teacher_margin


def test_margin_sign(synthetic_sample):
    teacher = synthetic_sample.teacher
    valid = synthetic_sample.candidates.valid_mask
    M = margin_matrix(teacher.J_T)
    for a in np.flatnonzero(valid):
        for b in np.flatnonzero(valid):
            assert np.isclose(M[a, b], -M[b, a])
            assert np.isclose(M[a, b], teacher_margin(teacher.J_T, int(a), int(b)))
    for (a, b), m in zip(synthetic_sample.pairs.pairs, synthetic_sample.pairs.margins):
        assert m > 0
        assert teacher.J_T[a] < teacher.J_T[b]
