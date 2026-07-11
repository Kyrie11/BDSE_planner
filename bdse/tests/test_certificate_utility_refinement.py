import numpy as np

from bdse.planner.tournament import _apply_certificate_utility_refinement


def test_utility_refinement_switches_only_inside_certificate_band():
    scores = np.array([1.0, 0.86, 0.40], dtype=np.float32)
    valid = np.array([True, True, True])
    flags = np.array([False, False, False])
    # Candidate 1 has much larger forward progress but is still within 0.2 of
    # the best certificate score. Candidate 2 is too far outside the band.
    traj = np.zeros((3, 3, 3), dtype=np.float32)
    traj[0, :, 0] = [0.0, 1.0, 2.0]
    traj[1, :, 0] = [0.0, 3.0, 6.0]
    traj[2, :, 0] = [0.0, 6.0, 12.0]
    cfg = {
        "tournament": {
            "utility_refinement": {
                "enabled": True,
                "score_slack": 0.2,
                "progress_weight": 1.0,
                "lateral_mean_weight": 0.0,
                "lateral_final_weight": 0.0,
                "comfort_weight": 0.0,
                "curvature_weight": 0.0,
                "path_length_weight": 0.0,
                "speed_weight": 0.0,
                "low_speed_threshold": 0.0,
                "unsafe_penalty": 1000.0,
            }
        }
    }
    action, diag = _apply_certificate_utility_refinement(scores, 0, valid, flags, cfg, traj)
    assert action == 1
    assert diag["utility_refinement_applied"] is True
    assert diag["utility_band_size"] == 2


def test_utility_refinement_respects_unflagged_constraint():
    scores = np.array([1.0, 0.95], dtype=np.float32)
    valid = np.array([True, True])
    flags = np.array([False, True])
    traj = np.zeros((2, 3, 3), dtype=np.float32)
    traj[0, :, 0] = [0.0, 1.0, 2.0]
    traj[1, :, 0] = [0.0, 5.0, 10.0]
    cfg = {"tournament": {"utility_refinement": {"enabled": True, "score_slack": 0.2, "progress_weight": 1.0, "require_unflagged": True}}}
    action, diag = _apply_certificate_utility_refinement(scores, 0, valid, flags, cfg, traj)
    assert action == 0
    assert diag["utility_refinement_applied"] is False
