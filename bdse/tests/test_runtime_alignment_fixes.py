from __future__ import annotations

import numpy as np
import torch

from bdse.model.scene_encoder import SceneEncoder
from bdse.planner.nuplan_planner import BDSEPlannerCore


def test_scene_encoder_ignores_invalid_padding_token_values():
    torch.manual_seed(0)
    enc = SceneEncoder(hidden_dim=16, layers=1, heads=4, dropout=0.0, map_feature_dim=8, route_feature_dim=8, traffic_feature_dim=12, goal_feature_dim=4)
    enc.eval()
    base = {
        "ego_history": torch.zeros(1, 3, 5),
        "agent_history": torch.zeros(1, 2, 3, 10),
        "agent_valid": torch.tensor([[True, False]]),
        "map_polylines": torch.zeros(1, 2, 4, 8),
        "map_polyline_valid": torch.tensor([[True, False]]),
        "route_polylines": torch.zeros(1, 1, 4, 8),
        "route_token_valid": torch.tensor([[True]]),
        "traffic_control_tokens": torch.zeros(1, 2, 12),
        "traffic_token_valid": torch.tensor([[False, False]]),
        "mission_goal": torch.zeros(1, 4),
        "mission_goal_valid": torch.tensor(False),
    }
    noisy = {k: v.clone() if torch.is_tensor(v) else v for k, v in base.items()}
    noisy["agent_history"][0, 1] = 999.0
    noisy["map_polylines"][0, 1] = -999.0
    noisy["traffic_control_tokens"][0] = 123.0
    with torch.no_grad():
        a = enc(base)
        b = enc(noisy)
    assert torch.allclose(a, b, atol=1e-5)


def test_teacher_hard_priority_prefers_safe_candidate_when_available(synthetic_sample):
    teacher = synthetic_sample.teacher
    valid = synthetic_sample.candidates.valid_mask.astype(bool)
    hard = teacher.hard_violation_mask.astype(bool) & valid
    safe = (~teacher.hard_violation_mask.astype(bool)) & valid
    if hard.any() and safe.any():
        assert not hard[int(teacher.a_star)]
        assert float(teacher.J_T[np.flatnonzero(safe)].min()) < float(teacher.J_T[np.flatnonzero(hard)].min())


def test_fallback_expands_budget_and_requeries_when_confidence_low(synthetic_sample, cfg):
    local_cfg = dict(cfg)
    local_cfg["selector"] = {**cfg.get("selector", {}), "proposal_top_m": 1}
    local_cfg["evidence"] = {**cfg.get("evidence", {}), "budget": 1}
    local_cfg["fallback"] = {
        **cfg.get("fallback", {}),
        "enabled": True,
        "tau_delta": 1e9,
        "rival_stages": [2, 4],
        "budget_stages": [1, 4],
        "proposal_multiplier": 2.0,
        "rule_rerank_top_k": 0,
    }
    core = BDSEPlannerCore(model=None, cfg=local_cfg)
    _, _, diag = core.plan_from_runtime(synthetic_sample.runtime)
    records = diag["fallback_stage_records"]
    assert diag["fallback_triggered"]
    assert len(records) >= 2
    assert max(len(r["top_m_atoms"]) for r in records) >= len(records[0]["top_m_atoms"])
    assert max(r["sparse_query_count"] for r in records) >= records[0]["sparse_query_count"]


def test_static_map_cache_uses_stable_map_name_across_wrappers():
    from bdse.data import feature_builder as fb

    class MapA:
        map_name = "us-ma-boston"

    class MapB:
        map_name = "us-ma-boston"

    class Obj:
        id = "lane-1"
        baseline_path = [(0.0, 0.0), (10.0, 0.0)]

    a = MapA()
    b = MapB()
    obj = Obj()
    # Type is part of the key, so same map name but different wrapper class should
    # not collide.  Same class/name wrappers should share static geometry cache.
    assert fb._map_cache_identity(a)[1] == fb._map_cache_identity(b)[1]
    before = len(fb._MAP_GEOMETRY_CACHE)
    fb._cached_baseline_points(a, obj)
    fb._cached_baseline_points(a, obj)
    after_same = len(fb._MAP_GEOMETRY_CACHE)
    assert after_same == before + 1


def test_label_future_cv_mode_skips_future_tracked_window():
    from bdse.data.label_builder import build_label_future_from_scenario
    from bdse.data.feature_builder import build_runtime_features_from_arrays

    class Ego:
        x = 0.0
        y = 0.0
        heading = 0.0
        velocity = 0.0

    class Box:
        length = 4.0
        width = 2.0
        velocity = None
        center = type("Center", (), {"x": 10.0, "y": 0.0, "heading": 0.0})()

    class Obj:
        track_token = "agent-1"
        box = Box()
        velocity = type("Vel", (), {"x": 1.0, "y": 0.0})()

    class Scenario:
        token = "s"
        start_time = type("T", (), {"time_us": 0})()
        database_interval = 0.1

        def get_ego_state_at_iteration(self, iteration):
            e = Ego()
            e.x = float(iteration) * 0.1
            return e

        def get_tracked_objects_at_iteration(self, iteration):
            return [Obj()]

        def get_ego_future_trajectory(self, iteration, time_horizon, num_samples):
            return [self.get_ego_state_at_iteration(iteration + k + 1) for k in range(num_samples)]

        def get_future_tracked_objects(self, *args, **kwargs):
            raise AssertionError("cv label mode must not fetch logged future tracked objects")

    cfg = {
        "preprocess": {"label_agent_future_mode": "cv", "temporal_frame_cache": False},
        "runtime": {"max_agents": 1, "history_s": 0.1, "history_hz": 10},
        "candidate": {"horizon_s": 0.3, "step_s": 0.1},
    }
    runtime = build_runtime_features_from_arrays(
        ego_history=np.zeros((2, 5), dtype=np.float32),
        agent_history=np.zeros((1, 2, 10), dtype=np.float32),
        current_agents=np.asarray([[10.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 4.0, 2.0, 0.0]], dtype=np.float32),
        cfg=cfg,
    )
    runtime.metadata["selected_agent_tokens"] = ["agent-1"]
    lf = build_label_future_from_scenario(Scenario(), 0, cfg, runtime=runtime)
    assert lf.metadata["agent_future_mode"] == "cv"
    assert lf.metadata["agent_future_logged_count"] == 0
    assert lf.metadata["agent_future_cv_fallback_count"] == 1
    assert lf.logged_agents[0, -1, 0] > lf.logged_agents[0, 0, 0]


def test_runtime_current_repeat_history_mode_skips_past_tracked_window():
    from bdse.data.feature_builder import build_runtime_features_from_scenario

    class Ego:
        x = 0.0
        y = 0.0
        heading = 0.0
        velocity = 0.0

    class Box:
        length = 4.0
        width = 2.0
        velocity = None
        center = type("Center", (), {"x": 5.0, "y": 0.0, "heading": 0.0})()

    class Obj:
        track_token = "agent-1"
        box = Box()
        velocity = type("Vel", (), {"x": 0.0, "y": 0.0})()

    class Scenario:
        token = "s"
        map_api = None
        start_time = type("T", (), {"time_us": 0})()
        database_interval = 0.1

        def get_ego_state_at_iteration(self, iteration):
            return Ego()

        def get_ego_past_trajectory(self, *args, **kwargs):
            return [Ego()]

        def get_tracked_objects_at_iteration(self, iteration):
            return [Obj()]

        def get_past_tracked_objects(self, *args, **kwargs):
            raise AssertionError("current_repeat mode must not fetch logged past tracked objects")

        def get_traffic_light_status_at_iteration(self, iteration):
            return []

        def get_route_roadblock_ids(self):
            return []

    cfg = {
        "preprocess": {"runtime_agent_history_mode": "current_repeat", "temporal_frame_cache": False, "profile": True},
        "runtime": {"max_agents": 1, "history_s": 0.2, "history_hz": 10},
        "candidate": {"horizon_s": 0.3, "step_s": 0.1},
    }
    rt = build_runtime_features_from_scenario(Scenario(), 0, cfg)
    assert rt.metadata["agent_history_mode"] == "current_repeat"
    assert rt.metadata["profile_runtime"]["runtime_agent_history_current_repeat"] == 1.0
    assert np.allclose(rt.agent_history[0, 0], rt.current_agents[0])
