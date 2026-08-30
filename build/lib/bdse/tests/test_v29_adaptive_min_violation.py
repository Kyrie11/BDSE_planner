import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.fallback import (
    runtime_safety_flags_from_runtime,
    runtime_safety_diagnostics,
    runtime_risk_scores,
    conservative_fallback_action,
)


def _runtime(agent_x=5.0, agent_y=0.0):
    return RuntimeFeatures(
        ego_history=np.zeros((1, 7), dtype=np.float32),
        agent_history=np.zeros((1, 1, 7), dtype=np.float32),
        agent_valid=np.array([True]),
        current_agents=np.array([[agent_x, agent_y, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        traffic_lights=[],
        map_features={
            "route_centerline": np.array([[0.0, 0.0], [20.0, 0.0]], dtype=np.float32),
            "route_corridor_width": 1.0,
            "stop_lines": [],
            "speed_limit_mps": 15.0,
        },
        route_roadblock_ids=[],
        mission_goal=None,
    )


def _candidates(ys):
    K = len(ys)
    T = 4
    traj = np.zeros((K, T, 5), dtype=np.float32)
    for k, y in enumerate(ys):
        traj[k, :, 0] = np.linspace(0.0, 8.0, T)
        traj[k, :, 1] = float(y)
        traj[k, :, 3] = 2.0
        traj[k, :, 4] = np.linspace(0.0, 1.0, T)
    return CandidateBank(
        trajectories=traj,
        valid_mask=np.ones((K,), dtype=bool),
        maneuver_ids=np.zeros((K,), dtype=np.int64),
        theta=[{} for _ in range(K)],
        dynamic_flags=[{} for _ in range(K)],
        metadata=[{} for _ in range(K)],
    )


def test_adaptive_dual_tier_uses_soft_when_pool_is_sufficient():
    runtime = _runtime(agent_x=5.0, agent_y=0.0)
    candidates = _candidates([0.0, 0.5, 1.2, 1.4, 1.55, 1.65, 1.75, 1.85])
    cfg = {"runtime_safety": {"flag_mode": "adaptive_dual_tier", "soft_agent_radius_m": 1.75, "hard_agent_radius_m": 0.85, "adaptive_min_soft_safe_actions": 2, "adaptive_min_soft_safe_ratio": 0.2, "adaptive_max_extra_soft_flags": 8, "soft_off_route_margin_m": 10.0, "hard_off_route_margin_m": 10.0}}
    flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
    diag = runtime_safety_diagnostics(runtime, candidates, cfg)
    assert diag["active_flag_tier"] == "soft"
    assert int(flags.sum()) >= int(diag["hard_flagged_count"])


def test_min_violation_recovery_prefers_lower_continuous_risk_when_all_flagged():
    runtime = _runtime(agent_x=5.0, agent_y=0.0)
    candidates = _candidates([0.0, 0.7, 1.4])
    cfg = {"runtime_safety": {"flag_mode": "hard", "hard_agent_radius_m": 2.0, "soft_agent_radius_m": 2.5, "hard_off_route_margin_m": 10.0, "soft_off_route_margin_m": 10.0}, "fallback": {"safe_progress_recovery": {"progress_weight": 0.0, "path_length_weight": 0.0, "lateral_weight": 0.0, "lateral_final_weight": 0.0, "unsafe_penalty": 0.0, "hard_risk_weight": 100.0}}}
    flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
    assert flags.all()
    risks = runtime_risk_scores(runtime, candidates, cfg)["hard"]
    action = conservative_fallback_action(candidates, safety_flags=flags, cfg=cfg, runtime=runtime)
    assert action == int(np.argmin(risks))
