import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.fallback import (
    _candidate_safety_time_masks,
    runtime_risk_scores,
    runtime_safety_flags_from_runtime,
    viability_frontier_recovery_action,
)


def _bank() -> CandidateBank:
    T = 81
    t = np.linspace(0.0, 8.0, T, dtype=np.float32)
    traj = np.zeros((2, T, 5), dtype=np.float32)
    # Fast candidate passes through a stopped vehicle at x=20.
    traj[0, :, 0] = 20.0 * t
    traj[0, :, 3] = 20.0
    # Slower candidate brakes/stops before the vehicle.
    traj[1, :, 0] = np.minimum(5.0 * t, 14.0)
    traj[1, :, 3] = np.where(t < 2.8, 5.0, 0.0)
    traj[:, :, 4] = t
    return CandidateBank(
        trajectories=traj,
        valid_mask=np.ones((2,), dtype=bool),
        maneuver_ids=np.zeros((2,), dtype=np.int64),
        theta=[{}, {}],
        dynamic_flags=[{}, {}],
        metadata=[{}, {}],
    )


def _runtime() -> RuntimeFeatures:
    cur = np.array([[20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.8, 2.0, 1.0]], dtype=np.float32)
    return RuntimeFeatures(
        ego_history=np.zeros((1, 5), dtype=np.float32),
        agent_history=cur[:, None, :].copy(),
        agent_valid=np.ones((1,), dtype=bool),
        current_agents=cur,
        traffic_lights=[],
        map_features={
            "route_centerline": np.array([[0.0, 0.0], [180.0, 0.0]], dtype=np.float32),
            "route_corridor_width": 4.0,
            "stop_lines": [],
            "speed_limit_mps": 25.0,
        },
        route_roadblock_ids=[],
        mission_goal=None,
    )


def _cfg() -> dict:
    return {
        "candidate": {"step_s": 0.1},
        "runtime_safety": {
            "flag_mode": "hard",
            "hard_check_horizon_s": 4.0,
            "soft_check_horizon_s": 6.0,
            "speed_adaptive_horizon": True,
            "min_hard_horizon_s": 4.0,
            "max_hard_horizon_s": 6.5,
            "reaction_time_s": 0.7,
            "comfortable_emergency_decel_mps2": 5.0,
            "stopping_horizon_margin_s": 0.35,
            "soft_horizon_extra_s": 1.25,
            "max_soft_horizon_s": 7.5,
            "use_box_agent_risk": True,
            "ego_length_m": 4.8,
            "ego_width_m": 2.0,
            "closing_speed_buffer_s": 0.18,
            "hard_longitudinal_clearance_m": 0.15,
            "hard_lateral_clearance_m": 0.10,
            "soft_longitudinal_extra_m": 0.8,
            "soft_lateral_extra_m": 0.5,
            "agent_ttc_safe_s": 3.0,
            "hard_ttc_flag_threshold": 0.92,
            "soft_ttc_flag_threshold": 0.5,
            "risk_hard_agent_weight": 8.0,
            "risk_hard_ttc_weight": 4.0,
            "risk_hard_offroute_weight": 1.0,
            "risk_soft_agent_weight": 1.0,
            "risk_soft_ttc_weight": 1.0,
            "risk_soft_offroute_weight": 1.0,
        },
        "fallback": {
            "safe_progress_recovery": {
                "viability_frontier": {
                    "enabled": True,
                    "min_pool": 1,
                    "max_pool": 4,
                    "joint_viability_epsilon_norm": 0.18,
                    "pareto_agent_epsilon": 0.04,
                    "pareto_ttc_epsilon": 0.04,
                    "pareto_offroute_epsilon": 0.06,
                    "pareto_soft_epsilon": 0.08,
                    "pareto_certificate_epsilon": 0.08,
                    "pareto_progress_epsilon": 0.04,
                    "progress_weight": 1.0,
                    "path_length_weight": 0.0,
                    "certificate_weight": 0.2,
                    "lateral_weight": 0.0,
                    "agent_risk_weight": 0.6,
                    "ttc_risk_weight": 0.5,
                    "offroute_risk_weight": 0.2,
                    "soft_risk_weight": 0.1,
                    "hard_risk_weight": 0.1,
                    "low_speed_penalty": 0.0,
                }
            }
        },
    }


def test_stopping_distance_horizon_expands_only_for_fast_candidate():
    bank = _bank()
    rsc = _cfg()["runtime_safety"]
    _, _, _, hard_h, _ = _candidate_safety_time_masks(bank.trajectories, rsc, 0.1)
    assert hard_h[0] > hard_h[1]
    assert hard_h[0] >= 5.0
    assert hard_h[1] == 4.0


def test_box_and_ttc_risk_reject_high_speed_collision_without_extra_evidence():
    runtime, bank, cfg = _runtime(), _bank(), _cfg()
    risks = runtime_risk_scores(runtime, bank, cfg)
    assert risks["hard_agent"][0] > risks["hard_agent"][1]
    assert risks["agent_ttc"][0] > risks["agent_ttc"][1]
    flags = runtime_safety_flags_from_runtime(runtime, bank, cfg)
    assert bool(flags[0])
    assert not bool(flags[1])
    decision = viability_frontier_recovery_action(
        bank,
        safety_flags=flags,
        cfg=cfg,
        runtime=runtime,
        tournament_scores=np.array([1.0, 0.0], dtype=np.float32),
        reference_action=0,
    )
    assert decision.action_index == 1
    assert decision.diagnostics["selected_agent_ttc_risk"] <= risks["agent_ttc"][1] + 1e-6
