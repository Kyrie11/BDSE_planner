from __future__ import annotations

from typing import Any

import numpy as np

from bdse.data.cache_schema import RuntimeFeatures
from bdse.data.feature_builder import build_runtime_features_from_arrays, make_default_route_centerline
from bdse.utils import transform_states_to_local


def _state_to_array(state: Any) -> np.ndarray:
    if state is None:
        return np.zeros((5,), dtype=np.float32)
    if isinstance(state, np.ndarray):
        out = np.zeros((5,), dtype=np.float32)
        flat = state.astype(np.float32).reshape(-1)
        out[: min(5, flat.size)] = flat[: min(5, flat.size)]
        return out
    rear = getattr(state, "rear_axle", state)
    center = getattr(state, "center", rear)
    x = float(getattr(rear, "x", getattr(center, "x", 0.0)))
    y = float(getattr(rear, "y", getattr(center, "y", 0.0)))
    yaw = float(getattr(rear, "heading", getattr(center, "heading", getattr(state, "heading", 0.0))))
    dyn = getattr(state, "dynamic_car_state", None)
    speed = 0.0
    if dyn is not None:
        speed = float(getattr(dyn, "speed", 0.0) or 0.0)
        vel = getattr(dyn, "rear_axle_velocity_2d", getattr(dyn, "center_velocity_2d", None))
        if vel is not None and speed == 0.0:
            speed = float(np.hypot(float(getattr(vel, "x", 0.0)), float(getattr(vel, "y", 0.0))))
    else:
        speed = float(getattr(state, "speed", getattr(state, "velocity", 0.0)) or 0.0)
    t = getattr(getattr(state, "time_point", None), "time_s", 0.0)
    return np.asarray([x, y, yaw, speed, float(t or 0.0)], dtype=np.float32)


def _object_to_array(obj: Any) -> np.ndarray:
    box = getattr(obj, "box", obj)
    center = getattr(box, "center", box)
    x = float(getattr(center, "x", getattr(obj, "x", 0.0)))
    y = float(getattr(center, "y", getattr(obj, "y", 0.0)))
    yaw = float(getattr(center, "heading", getattr(box, "heading", getattr(obj, "heading", 0.0))))
    vel = getattr(obj, "velocity", getattr(box, "velocity", None))
    vx = float(getattr(vel, "x", 0.0)) if vel is not None else float(getattr(obj, "vx", 0.0))
    vy = float(getattr(vel, "y", 0.0)) if vel is not None else float(getattr(obj, "vy", 0.0))
    length = float(getattr(box, "length", getattr(obj, "length", 4.8)))
    width = float(getattr(box, "width", getattr(obj, "width", 2.0)))
    return np.asarray([x, y, yaw, float(np.hypot(vx, vy)), 0.0, vx, vy, length, width, 0.0], dtype=np.float32)


def _iter_objects(obs: Any) -> list[Any]:
    if obs is None:
        return []
    if isinstance(obs, (list, tuple)):
        return list(obs)
    tracked = getattr(obs, "tracked_objects", None)
    if tracked is not None:
        concrete = getattr(tracked, "tracked_objects", None)
        if concrete is not None:
            return list(concrete)
        try:
            return list(tracked)
        except TypeError:
            return []
    try:
        return list(obs)
    except TypeError:
        return []


def _history_lists(history: Any) -> tuple[list[Any], list[Any]]:
    ego_states = list(getattr(history, "ego_states", []) or [])
    observations = list(getattr(history, "observations", []) or [])
    if not ego_states and hasattr(history, "current_state"):
        try:
            ego, obs = history.current_state
            ego_states = [ego]
            observations = [obs]
        except Exception:
            pass
    return ego_states, observations


def _traffic_lights_from_input(current_input: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    data = getattr(current_input, "traffic_light_data", None)
    if data is None:
        return out
    try:
        iterator = list(data)
    except TypeError:
        iterator = [data]
    for item in iterator:
        status = str(getattr(item, "status", getattr(item, "status_name", item))).lower()
        lane_id = str(getattr(item, "lane_connector_id", getattr(item, "lane_id", "")))
        out.append({"status": status, "lane_connector_id": lane_id})
    return out


def _map_features_from_initialization(initialization: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    # Runtime-safe minimal map context.  Rich offline map extraction remains in
    # feature_builder for preprocessing; compute_trajectory never reads scenario DB.
    route_ids = list(getattr(initialization, "route_roadblock_ids", []) or []) if initialization is not None else []
    goal = getattr(initialization, "mission_goal", None) if initialization is not None else None
    route = make_default_route_centerline(float(cfg.get("runtime", {}).get("route_horizon_m", 160.0)))
    return {
        "route_centerline": route,
        "route_corridor_width": float(cfg.get("candidate", {}).get("route_width_m", 4.0)),
        "route_roadblock_ids": route_ids,
        "mission_goal_raw": str(goal) if goal is not None else "",
        "map_valid": False,
        "runtime_adapter": "planner_input_only_minimal_map",
        "speed_limit_mps": float(cfg.get("runtime", {}).get("default_speed_limit_mps", 13.4)),
    }


def build_runtime_features_from_planner_input(current_input: Any, initialization: Any, cfg: dict[str, Any]) -> RuntimeFeatures:
    history = getattr(current_input, "history", None)
    ego_states, observations = _history_lists(history)
    if not ego_states:
        raise TypeError("PlannerInput history does not expose ego_states/current_state; cannot build BDSE runtime features")
    h_steps = int(round(float(cfg.get("runtime", {}).get("history_s", 2.0)) * int(cfg.get("runtime", {}).get("history_hz", 10)))) + 1
    ego_global = np.asarray([_state_to_array(s) for s in ego_states[-h_steps:]], dtype=np.float32)
    # Fill/left-pad to expected length in global coordinates before transform.
    if len(ego_global) < h_steps:
        pad = np.repeat(ego_global[:1], h_steps - len(ego_global), axis=0)
        ego_global = np.concatenate([pad, ego_global], axis=0)
    current = ego_global[-1]
    ego_local = transform_states_to_local(ego_global, current)
    ego_local[:, 4] = np.linspace(-float(cfg.get("runtime", {}).get("history_s", 2.0)), 0.0, h_steps, dtype=np.float32)

    max_agents = int(cfg.get("runtime", {}).get("max_agents", 32))
    obs_tail = observations[-h_steps:] if observations else []
    current_objs = _iter_objects(obs_tail[-1]) if obs_tail else []
    current_agents_global = np.asarray([_object_to_array(o) for o in current_objs[:max_agents]], dtype=np.float32) if current_objs else np.zeros((0, 10), dtype=np.float32)
    agent_history = np.zeros((max_agents, h_steps, 10), dtype=np.float32)
    current_agents = np.zeros((max_agents, 10), dtype=np.float32)
    if len(current_agents_global):
        # Transform current detections to ego frame.  Earlier history uses current
        # snapshot repeated if object tracking alignment is unavailable in PlannerInput.
        xy = current_agents_global[:, :5].copy()
        local = transform_states_to_local(xy, current)
        current_agents[: len(local), :5] = local[:, :5]
        current_agents[: len(local), 5:10] = current_agents_global[:, 5:10]
        agent_history[: len(local), :, :] = current_agents[: len(local), None, :]
    map_features = _map_features_from_initialization(initialization, cfg)
    traffic = _traffic_lights_from_input(current_input)
    route_ids = list(getattr(initialization, "route_roadblock_ids", []) or []) if initialization is not None else []
    goal_obj = getattr(initialization, "mission_goal", None) if initialization is not None else None
    goal_arr = _state_to_array(goal_obj)[:4] if goal_obj is not None else None
    runtime = build_runtime_features_from_arrays(
        ego_local,
        agent_history=agent_history,
        current_agents=current_agents,
        traffic_lights=traffic,
        map_features=map_features,
        route_roadblock_ids=route_ids,
        mission_goal=goal_arr,
        cfg=cfg,
    )
    runtime.metadata["source"] = "PlannerInput.history_observations"
    runtime.metadata["scenario_db_replay_used"] = False
    return runtime
