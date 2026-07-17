from __future__ import annotations

import numpy as np


def test_cost_partition(synthetic_sample):
    teacher = synthetic_sample.teacher
    valid = synthetic_sample.candidates.valid_mask
    teacher.validate_partition(valid)
    assert np.allclose(teacher.J_T[valid], teacher.J_base[valid] + teacher.J_evid[valid])
    assert "J_hard" not in teacher.diagnostics
