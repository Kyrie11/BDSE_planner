from __future__ import annotations

import numpy as np
import pytest

from bdse.config import load_config
from bdse.data.cache_schema import LabelOnlyFuture
from bdse.data.feature_builder import build_runtime_features_from_arrays, make_default_route_centerline
from bdse.data.label_builder import build_training_sample_from_runtime_and_future


@pytest.fixture()
def cfg():
    return load_config(overrides={"runtime": {"max_agents": 4}, "evidence": {"max_atoms": 32, "max_interaction_atoms": 8, "budget": 4}, "pairs": {"target_min": 4, "target_max": 32}, "candidate": {"K": 32}})


@pytest.fixture()
def synthetic_sample(cfg):
    h = int(round(cfg["runtime"]["history_s"] * cfg["runtime"]["history_hz"])) + 1
    ego = np.zeros((h, 5), dtype=np.float32)
    ego[:, 4] = np.linspace(-2.0, 0.0, h)
    ego[:, 3] = 5.0
    agent_hist = np.zeros((1, h, 10), dtype=np.float32)
    current_agents = np.zeros((1, 10), dtype=np.float32)
    current_agents[0, :10] = np.array([15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.8, 2.0, 1.0], dtype=np.float32)
    agent_hist[0, :, :] = current_agents[0]
    map_features = {"route_centerline": make_default_route_centerline(), "route_corridor_width": 4.0, "speed_limit_mps": 13.4}
    runtime = build_runtime_features_from_arrays(ego, agent_hist, current_agents, traffic_lights=[], map_features=map_features, cfg=cfg)
    T = int(round(cfg["candidate"]["horizon_s"] / cfg["candidate"]["step_s"]))
    times = np.arange(1, T + 1, dtype=np.float32) * cfg["candidate"]["step_s"]
    logged_ego = np.stack([5.0 * times, np.zeros_like(times), np.zeros_like(times), 5.0 * np.ones_like(times), times], axis=1).astype(np.float32)
    logged_agents = np.zeros((cfg["runtime"]["max_agents"], T, 5), dtype=np.float32)
    logged_agents[0, :, 0] = 15.0
    logged_agents[0, :, 1] = 0.0
    logged_agents[0, :, 2] = 0.0
    logged_agents[0, :, 3] = 0.0
    logged_agents[0, :, 4] = times
    agent_valid = np.zeros((cfg["runtime"]["max_agents"],), dtype=bool)
    agent_valid[0] = True
    future = LabelOnlyFuture(logged_ego=logged_ego, logged_agents=logged_agents, agent_valid=agent_valid)
    return build_training_sample_from_runtime_and_future(runtime, future, cfg)
