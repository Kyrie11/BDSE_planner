from __future__ import annotations

import numpy as np


def test_padding_masks(synthetic_sample):
    valid = synthetic_sample.candidates.valid_mask
    invalid = ~valid
    if invalid.any():
        assert np.all(np.isinf(synthetic_sample.teacher.J_T[invalid]))
    for a, b in synthetic_sample.pairs.pairs:
        assert valid[a]
        assert valid[b]
    assert valid[synthetic_sample.teacher.a_star]
