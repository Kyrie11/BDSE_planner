from __future__ import annotations

from bdse.planner.teacher_cost import validate_residual_closure


def test_residual_closure(synthetic_sample):
    validate_residual_closure(synthetic_sample.teacher, synthetic_sample.pairs.pairs)
