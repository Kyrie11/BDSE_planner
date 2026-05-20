import numpy as np

from bdse.data.cache_schema import RuntimeFeatures
from bdse.planner.candidate_generator import generate_candidate_bank


def _runtime_with_sharp_connector() -> RuntimeFeatures:
    route = np.asarray([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [5.0, 10.0], [10.0, 10.0]], dtype=np.float32)
    return RuntimeFeatures(
        ego_history=np.asarray([[0.0, 0.0, 0.0, 8.0, 0.0]] * 21, dtype=np.float32),
        agent_history=np.zeros((32, 21, 10), dtype=np.float32),
        agent_valid=np.zeros((32,), dtype=bool),
        current_agents=np.zeros((32, 10), dtype=np.float32),
        traffic_lights=[],
        map_features={"route_centerline": route, "route_corridor_width": 4.0, "map_valid": True},
        route_roadblock_ids=["r0"],
        mission_goal=np.asarray([10.0, 10.0], dtype=np.float32),
    )


def test_candidate_repair_prevents_strict_dynamic_bank_collapse():
    cfg = {
        "candidate": {
            "K": 48,
            "horizon_s": 8.0,
            "step_s": 0.1,
            "valid_requires_dynamic": True,
            "dynamic_eval_quantile": 0.98,
            "dynamic_low_speed_mps": 0.3,
            "repair_low_valid_count": True,
            "min_valid_candidates": 24,
            "max_accel": 3.0,
            "max_decel": -5.0,
            "max_jerk": 5.0,
            "max_curvature": 0.25,
            "max_curvature_rate": 0.20,
            "max_lateral_accel": 3.0,
            "counts": {
                "keep_follow": 10,
                "decelerate_stop": 12,
                "yield_creep": 6,
                "lane_change_left": 4,
                "lane_change_right": 4,
                "route_turn_connector": 4,
                "safe_fallback": 8,
            },
        }
    }
    bank = generate_candidate_bank(_runtime_with_sharp_connector(), cfg)
    assert int(bank.valid_mask.sum()) >= 24
    assert any(meta.get("recovery_candidate") for meta in bank.metadata)
