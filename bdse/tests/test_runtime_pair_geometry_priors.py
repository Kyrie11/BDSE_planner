from __future__ import annotations

import numpy as np

from bdse.planner.selector import runtime_greedy_selector
from bdse.planner.tournament import run_tournament


def test_runtime_selector_passes_stop_go_geometry_pairs():
    J0 = np.asarray([100.0, 0.0], dtype=np.float32)
    g = np.zeros((1, 2), dtype=np.float32)
    # Atom 0 strongly supports progressive action 0 over stop/yield action 1,
    # but the cheap base score alone only sees action 1 when L=1.
    g[0, 0] = 0.0
    g[0, 1] = 220.0
    valid = np.ones((2,), dtype=bool)
    safety = np.zeros((2,), dtype=bool)
    traj = np.zeros((2, 3, 5), dtype=np.float32)
    traj[0, -1, 0] = 80.0  # progressive
    traj[1, -1, 0] = 5.0   # stop/yield
    maneuver = np.asarray([0, 1], dtype=np.int64)
    # Without geometry priors and L=1, the cheap base graph cannot see action 1.
    no_geo = runtime_greedy_selector(J0, g, np.ones(1, dtype=np.float32), valid, safety, budget=1, L_infer=1)
    assert no_geo.selected == []
    # With stop/go geometry priors, the selector can query the decisive pair.
    geo = runtime_greedy_selector(
        J0, g, np.ones(1, dtype=np.float32), valid, safety, budget=1, L_infer=1,
        candidate_trajectories=traj, maneuver_ids=maneuver, maneuver_pair_count=1, progress_pair_count=1,
    )
    assert geo.selected == [0]


def test_tournament_accepts_geometry_rival_arguments():
    J0 = np.asarray([0.0, 10.0], dtype=np.float32)
    g = np.zeros((1, 2), dtype=np.float32)
    valid = np.ones((2,), dtype=bool)
    safety = np.zeros((2,), dtype=bool)
    traj = np.zeros((2, 3, 5), dtype=np.float32)
    maneuver = np.asarray([0, 3], dtype=np.int64)
    result = run_tournament(J0, g, [0], valid, safety, {}, candidate_trajectories=traj, maneuver_ids=maneuver)
    assert result.action_index in {0, 1}
