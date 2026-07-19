import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.fallback import (
    runtime_safety_flag_components,
    runtime_safety_flags_from_runtime,
    viability_frontier_recovery_action,
)


def _runtime() -> RuntimeFeatures:
    return RuntimeFeatures(
        ego_history=np.zeros((1, 7), dtype=np.float32),
        agent_history=np.zeros((0, 1, 7), dtype=np.float32),
        agent_valid=np.zeros((0,), dtype=bool),
        current_agents=np.zeros((0, 7), dtype=np.float32),
        traffic_lights=[],
        map_features={
            "route_centerline": np.array([[0.0, 0.0], [40.0, 0.0]], dtype=np.float32),
            "route_corridor_width": 1.0,
            "stop_lines": [],
            "speed_limit_mps": 15.0,
        },
        route_roadblock_ids=[],
        mission_goal=None,
    )


def _bank(xs=(4.0, 10.0, 12.0), ys=(1.12, 1.16, 2.5), horizon=4.0) -> CandidateBank:
    traj = np.zeros((len(xs), 9, 5), dtype=np.float32)
    for k, (x, y) in enumerate(zip(xs, ys)):
        traj[k, :, 0] = np.linspace(0.0, x, traj.shape[1])
        traj[k, :, 1] = y
        traj[k, :, 3] = 2.0
        traj[k, :, 4] = np.linspace(0.0, horizon, traj.shape[1])
    return CandidateBank(
        trajectories=traj,
        valid_mask=np.ones((len(xs),), dtype=bool),
        maneuver_ids=np.zeros((len(xs),), dtype=np.int64),
        theta=[{} for _ in xs],
        dynamic_flags=[{} for _ in xs],
        metadata=[{} for _ in xs],
    )


def _cfg() -> dict:
    return {
        "runtime_safety": {
            "flag_mode": "hard",
            "hard_off_route_margin_m": 0.0,
            "soft_off_route_margin_m": 0.0,
            "risk_hard_offroute_weight": 1.0,
            "risk_hard_agent_weight": 1.0,
            "hard_check_horizon_s": 4.0,
            "soft_check_horizon_s": 6.0,
        },
        "fallback": {
            "safe_progress_recovery": {
                "viability_frontier": {
                    "enabled": True,
                    "min_pool": 2,
                    "max_pool": 3,
                    "scale_quantile": 0.9,
                    "agent_epsilon_norm": 0.1,
                    "offroute_epsilon_norm": 0.2,
                    "certificate_epsilon_norm": 1.0,
                    "pareto_agent_epsilon": 0.01,
                    "pareto_offroute_epsilon": 0.02,
                    "pareto_soft_epsilon": 0.02,
                    "pareto_certificate_epsilon": 0.02,
                    "pareto_progress_epsilon": 0.02,
                    "progress_weight": 1.0,
                    "path_length_weight": 0.0,
                    "certificate_weight": 0.1,
                    "lateral_weight": 0.0,
                    "agent_risk_weight": 0.2,
                    "offroute_risk_weight": 0.2,
                    "soft_risk_weight": 0.0,
                    "hard_risk_weight": 0.0,
                    "reference_action_bonus": 0.0,
                    "low_speed_penalty": 0.0,
                }
            }
        },
    }


def test_vcdsr_preserves_progress_on_a_true_epsilon_frontier():
    runtime = _runtime()
    candidates = _bank()
    cfg = _cfg()
    flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
    assert flags.all()

    decision = viability_frontier_recovery_action(
        candidates,
        safety_flags=flags,
        cfg=cfg,
        runtime=runtime,
        tournament_scores=np.array([0.0, 0.05, -1.0], dtype=np.float32),
        reference_action=0,
    )
    assert decision.action_index == 1
    assert decision.diagnostics["frontier_size"] >= 2
    assert decision.diagnostics["progress_gain_over_min_risk"] > 0.0
    assert decision.diagnostics["score_conditioned"] is True


def test_vcdsr_can_keep_the_bdse_certificate_inside_the_viability_frontier():
    runtime = _runtime()
    candidates = _bank(xs=(9.0, 10.0), ys=(1.15, 1.15))
    cfg = _cfg()
    vc = cfg["fallback"]["safe_progress_recovery"]["viability_frontier"]
    vc.update({"progress_weight": 0.1, "certificate_weight": 2.0, "certificate_epsilon_norm": 2.0})
    flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
    decision = viability_frontier_recovery_action(
        candidates,
        safety_flags=flags,
        cfg=cfg,
        runtime=runtime,
        tournament_scores=np.array([1.0, 0.0], dtype=np.float32),
    )
    assert decision.action_index == 0
    assert decision.diagnostics["selected_score_loss_norm"] == 0.0


def test_receding_horizon_guard_ignores_only_far_future_route_excess():
    runtime = _runtime()
    traj = np.zeros((1, 9, 5), dtype=np.float32)
    traj[0, :, 0] = np.linspace(0.0, 12.0, 9)
    traj[0, :, 1] = np.array([0, 0, 0, 0, 0, 0, 4, 4, 4], dtype=np.float32)
    traj[0, :, 3] = 2.0
    traj[0, :, 4] = np.linspace(0.0, 8.0, 9)
    candidates = CandidateBank(
        trajectories=traj,
        valid_mask=np.ones((1,), dtype=bool),
        maneuver_ids=np.zeros((1,), dtype=np.int64),
        theta=[{}],
        dynamic_flags=[{}],
        metadata=[{}],
    )
    cfg = _cfg()
    cfg["runtime_safety"]["hard_off_route_margin_m"] = 0.0
    comp = runtime_safety_flag_components(runtime, candidates, cfg)
    assert not bool(comp["off_route_hard"][0])
    assert bool(comp["off_route_soft"][0])  # soft horizon extends to six seconds
