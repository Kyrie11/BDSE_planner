import numpy as np

from bdse.model.bdse_model import _apply_runtime_base_prior_np, _robust_rank_cost_np


class _Candidates:
    def __init__(self):
        self.valid_mask = np.array([True, True, True], dtype=bool)
        self.K = 3
        self.trajectories = np.zeros((3, 4, 3), dtype=np.float32)
        self.trajectories[0, :, 0] = [0, 1, 2, 3]
        self.trajectories[1, :, 0] = [0, 2, 4, 8]
        self.trajectories[2, :, 0] = [0, 1, 1, 1]
        self.trajectories[2, :, 1] = [0, 3, 4, 5]


def test_robust_rank_cost_preserves_argmin():
    c = np.array([10.0, 2.0, 5.0], dtype=np.float32)
    z = _robust_rank_cost_np(c, np.array([True, True, True]))
    assert int(np.argmin(z)) == 1
    assert np.isfinite(z).all()


def test_runtime_base_prior_can_replace_learned_best():
    cfg = {
        "model": {"margin_normalization_min_scale": 20000.0},
        "runtime": {"base_prior": {"enabled": True, "mode": "prior_only", "progress_weight": 1.0, "lateral_mean_weight": 0.0, "lateral_final_weight": 0.0, "comfort_weight": 0.0, "curvature_weight": 0.0, "path_length_weight": 0.0, "unsafe_penalty": 1000.0}},
    }
    J0 = np.array([0.0, 100.0, 50.0], dtype=np.float32)
    out, diag = _apply_runtime_base_prior_np(J0, _Candidates(), np.array([False, False, False]), cfg)
    assert diag["base_prior_enabled"] is True
    assert diag["base_prior_best_action"] == 1
    assert int(np.argmin(out)) == 1
