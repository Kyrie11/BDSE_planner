import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.fallback import conservative_fallback_action, runtime_safety_flags_from_runtime, runtime_risk_scores


def _runtime():
    return RuntimeFeatures(
        ego_history=np.zeros((1, 7), dtype=np.float32),
        agent_history=np.zeros((0, 1, 7), dtype=np.float32),
        agent_valid=np.zeros((0,), dtype=bool),
        current_agents=np.zeros((0, 7), dtype=np.float32),
        traffic_lights=[],
        map_features={
            "route_centerline": np.array([[0.0, 0.0], [30.0, 0.0]], dtype=np.float32),
            "route_corridor_width": 1.0,
            "stop_lines": [],
            "speed_limit_mps": 15.0,
        },
        route_roadblock_ids=[],
        mission_goal=None,
    )


def _candidate_bank():
    # All candidates are just outside the hard route corridor, but candidate 1
    # has much better forward progress while remaining in a near-minimum
    # violation band.  v30 should not collapse to the absolute minimum-violation
    # low-progress trajectory in this all-flagged regime.
    traj = np.zeros((3, 4, 5), dtype=np.float32)
    xs = [4.0, 10.0, 12.0]
    ys = [1.12, 1.16, 1.75]
    for k, (x, y) in enumerate(zip(xs, ys)):
        traj[k, :, 0] = np.linspace(0.0, x, 4)
        traj[k, :, 1] = y
        traj[k, :, 3] = 2.0
        traj[k, :, 4] = np.linspace(0.0, 1.0, 4)
    return CandidateBank(
        trajectories=traj,
        valid_mask=np.ones((3,), dtype=bool),
        maneuver_ids=np.zeros((3,), dtype=np.int64),
        theta=[{} for _ in range(3)],
        dynamic_flags=[{} for _ in range(3)],
        metadata=[{} for _ in range(3)],
    )


def test_pareto_min_violation_recovery_preserves_progress_inside_risk_band():
    runtime = _runtime()
    candidates = _candidate_bank()
    cfg = {
        "runtime_safety": {
            "flag_mode": "hard",
            "hard_off_route_margin_m": 0.0,
            "soft_off_route_margin_m": 0.0,
            "risk_hard_offroute_weight": 1.0,
            "risk_hard_agent_weight": 1.0,
        },
        "fallback": {
            "safe_progress_recovery": {
                "pareto_min_violation": True,
                "min_pareto_pool": 2,
                "max_pareto_pool": 3,
                "hard_risk_abs_margin": 0.10,
                "hard_risk_rel_margin": 0.0,
                "soft_risk_abs_margin": 10.0,
                "progress_quantile_floor": 0.0,
                "progress_weight": 1.0,
                "path_length_weight": 0.0,
                "lateral_weight": 0.0,
                "lateral_final_weight": 0.0,
                "hard_risk_weight": 1.0,
                "soft_risk_weight": 0.0,
                "unsafe_penalty": 0.0,
            }
        },
    }
    flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
    assert flags.all()
    risks = runtime_risk_scores(runtime, candidates, cfg)["hard"]
    assert int(np.argmin(risks)) == 0
    action = conservative_fallback_action(candidates, safety_flags=flags, cfg=cfg, runtime=runtime)
    assert action == 1
