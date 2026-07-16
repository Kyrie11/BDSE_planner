from __future__ import annotations

import numpy as np

from bdse.planner.selector import budgeted_margin
from bdse.planner.tournament import assert_antisymmetric


def test_tournament_antisymmetry(synthetic_sample):
    J0 = np.nan_to_num(synthetic_sample.teacher.J_base, posinf=1e6)
    g = synthetic_sample.teacher.g_evid
    M = budgeted_margin(J0, g, list(range(min(4, g.shape[0]))))
    assert_antisymmetric(M, synthetic_sample.candidates.valid_mask)


def test_pair_delta_margin_matrix_projects_directed_predictions_to_antisymmetry():
    from bdse.planner.tournament import _pair_delta_margin_matrix, _pair_sigma_matrix

    J0 = np.asarray([0.0, 2.0, 5.0], dtype=np.float32)
    # Two reciprocal predictions are intentionally inconsistent.  The runtime
    # matrix should project them into a single antisymmetric game rather than
    # allowing the last directed pair to overwrite the first one.
    pair_indices = np.asarray([[0, 1], [1, 0], [1, 2]], dtype=np.int64)
    pair_delta = np.asarray([[3.0, -1.0, 4.0]], dtype=np.float32)
    M = _pair_delta_margin_matrix(J0, pair_indices, pair_delta, [0], np.asarray([True, True, True]))
    assert_antisymmetric(M, np.asarray([True, True, True]))
    assert np.isclose(M[0, 1], 4.0)
    assert np.isclose(M[1, 0], -4.0)
    sigma = _pair_sigma_matrix(pair_indices, np.ones_like(pair_delta), [0], 3)
    assert sigma is not None
    assert np.isclose(sigma[0, 1], sigma[1, 0])
