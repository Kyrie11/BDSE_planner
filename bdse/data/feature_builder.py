from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

from bdse.data.cache_schema import RuntimeFeatures, pad_array
from bdse.utils import angle_wrap, transform_states_to_local


def _call(obj: Any, names: Sequence[str], *args, default=None, **kwargs):
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except TypeError:
                try:
                    return fn(*args)
                except TypeError:
                    continue
    return default


def _state_to_array(state: Any) -> np.ndarray:
    if state is None:
        return np.zeros((5,), dtype=np.float32)
    if isinstance(state, np.ndarray):
        arr = state.astype(np.float32).reshape(-1)
        out = np.zeros((5,), dtype=np.float32)
        out[: min(5, len(arr))] = arr[:5]
        return out
    rear = getattr(state, "rear_axle", None) or getattr(state, "center", None) or state
    x = float(getattr(rear, "x", getattr(state, "x", 0.0)))
    y = float(getattr(rear, "y", getattr(state, "y", 0.0)))
    yaw = float(getattr(rear, "heading", getattr(state, "heading", getattr(state, "yaw", 0.0))))
    vel_obj = getattr(state, "dynamic_car_state", None)
    speed = 0.0
    if vel_obj is not None:
        rear_vel = getattr(vel_obj, "rear_axle_velocity_2d", None) or getattr(vel_obj, "center_velocity_2d", None)
        if rear_vel is not None:
            vx = float(getattr(rear_vel, "x", 0.0))
            vy = float(getattr(rear_vel, "y", 0.0))
            speed = math.hypot(vx, vy)
        else:
            speed = float(getattr(vel_obj, "speed", 0.0))
    speed = float(getattr(state, "speed", speed))
    t = float(getattr(getattr(state, "time_point", None), "time_us", getattr(state, "time_us", 0.0))) / 1e6
    return np.asarray([x, y, yaw, speed, t], dtype=np.float32)


def _box_to_array(obj: Any) -> np.ndarray:
    center = getattr(obj, "center", obj)
    x = float(getattr(center, "x", getattr(obj, "x", 0.0)))
    y = float(getattr(center, "y", getattr(obj, "y", 0.0)))
    yaw = float(getattr(center, "heading", getattr(obj, "heading", getattr(obj, "yaw", 0.0))))
    vel = getattr(obj, "velocity", None)
    vx = float(getattr(vel, "x", 0.0)) if vel is not None else float(getattr(obj, "vx", 0.0))
    vy = float(getattr(vel, "y", 0.0)) if vel is not None else float(getattr(obj, "vy", 0.0))
    length = float(getattr(getattr(obj, "box", obj), "length", getattr(obj, "length", 4.8)))
    width = float(getattr(getattr(obj, "box", obj), "width", getattr(obj, "width", 2.0)))
    token = getattr(obj, "track_token", getattr(obj, "token", ""))
    token_hash = float(abs(hash(str(token))) % 1000000) / 1000000.0
    return np.asarray([x, y, yaw, math.hypot(vx, vy), 0.0, vx, vy, length, width, token_hash], dtype=np.float32)


def _iter_tracked_objects(container: Any) -> list[Any]:
    if container is None:
        return []
    if isinstance(container, (list, tuple)):
        return list(container)
    tracked = getattr(container, "tracked_objects", None)
    if tracked is not None:
        if hasattr(tracked, "get_tracked_objects_of_types"):
            try:
                return list(tracked.get_tracked_objects_of_types(list(getattr(tracked, "tracked_object_types", []))))
            except Exception:
                return []
        if hasattr(tracked, "tracked_objects"):
            return list(tracked.tracked_objects)
        try:
            return list(tracked)
        except TypeError:
            return []
    try:
        return list(container)
    except TypeError:
        return []


def _traffic_lights_to_list(statuses: Any) -> list[dict[str, Any]]:
    if statuses is None:
        return []
    out = []
    for st in statuses if isinstance(statuses, (list, tuple, set)) else [statuses]:
        out.append(
            {
                "lane_connector_id": str(getattr(st, "lane_connector_id", getattr(st, "connector_id", ""))),
                "status": str(getattr(st, "status", getattr(st, "traffic_light_status_type", "unknown"))).lower(),
                "timestamp_us": int(getattr(st, "timestamp", getattr(st, "timestamp_us", 0)) or 0),
            }
        )
    return out


def make_default_route_centerline(horizon_m: float = 160.0, step_m: float = 2.0) -> np.ndarray:
    x = np.arange(0.0, horizon_m + step_m, step_m, dtype=np.float32)
    y = np.zeros_like(x)
    return np.stack([x, y], axis=1)


def extract_map_features_from_api(map_api: Any, ego_state_local: np.ndarray, radius_m: float, route_ids: list[str]) -> dict[str, Any]:
    features: dict[str, Any] = {
        "route_centerline": make_default_route_centerline(),
        "stop_lines": [],
        "crosswalks": [],
        "speed_limits": [],
        "drivable_polygons": [],
        "route_corridor_width": 4.0,
    }
    if map_api is None:
        return features
    query_layers = [
        "LANE",
        "LANE_CONNECTOR",
        "STOP_LINE",
        "CROSSWALK",
        "INTERSECTION",
        "DRIVABLE_AREA",
        "ROADBLOCK",
        "ROADBLOCK_CONNECTOR",
    ]
    center = (0.0, 0.0)
    prox = _call(map_api, ["get_proximal_map_objects"], center, radius_m, query_layers, default={})
    features["raw_proximal_objects"] = prox
    route_polylines = []
    if isinstance(prox, dict):
        for values in prox.values():
            for obj in values if isinstance(values, (list, tuple, set)) else []:
                obj_id = str(getattr(obj, "id", getattr(obj, "token", "")))
                baseline = getattr(obj, "baseline_path", None)
                discrete = getattr(baseline, "discrete_path", None) if baseline is not None else getattr(obj, "discrete_path", None)
                pts = []
                for p in discrete or []:
                    pts.append([float(getattr(p, "x", 0.0)), float(getattr(p, "y", 0.0))])
                if pts and (not route_ids or obj_id in route_ids):
                    route_polylines.append(np.asarray(pts, dtype=np.float32))
                if "STOP" in obj.__class__.__name__.upper():
                    features["stop_lines"].append({"id": obj_id, "xy": np.asarray(pts, dtype=np.float32)})
    if route_polylines:
        features["route_centerline"] = max(route_polylines, key=lambda p: p.shape[0])
    return features


def select_agents_deterministic(
    raw_current_agents: np.ndarray,
    raw_agent_history: np.ndarray,
    ego_xy: np.ndarray,
    max_agents: int,
    radius_m: float,
    candidate_trajectories: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if raw_current_agents.size == 0:
        return (
            np.zeros((max_agents, raw_agent_history.shape[1] if raw_agent_history.ndim == 3 else 1, 10), dtype=np.float32),
            np.zeros((max_agents, 10), dtype=np.float32),
            np.zeros((max_agents,), dtype=bool),
        )
    cur = np.asarray(raw_current_agents, dtype=np.float32)
    hist = np.asarray(raw_agent_history, dtype=np.float32)
    dist = np.linalg.norm(cur[:, :2] - ego_xy[None, :], axis=1)
    keep = dist <= radius_m
    idxs = np.flatnonzero(keep)
    if idxs.size == 0:
        idxs = np.argsort(dist)[: min(max_agents, len(dist))]
    min_cand_dist = np.full(len(cur), 1e6, dtype=np.float32)
    if candidate_trajectories is not None and len(candidate_trajectories):
        cand_xy = candidate_trajectories.reshape(-1, candidate_trajectories.shape[-1])[:, :2]
        for i in range(len(cur)):
            min_cand_dist[i] = np.linalg.norm(cand_xy - cur[i, None, :2], axis=1).min()
    rel_speed = np.maximum(np.abs(cur[:, 3]) + 1e-3, 1e-3)
    ttc = dist / rel_speed
    order = sorted(idxs.tolist(), key=lambda i: (min_cand_dist[i], ttc[i], dist[i], i))
    order = order[:max_agents]
    h_steps = hist.shape[1] if hist.ndim == 3 else 1
    out_hist = np.zeros((max_agents, h_steps, 10), dtype=np.float32)
    out_cur = np.zeros((max_agents, 10), dtype=np.float32)
    valid = np.zeros((max_agents,), dtype=bool)
    for j, i in enumerate(order):
        out_cur[j] = cur[i]
        out_hist[j] = hist[i]
        valid[j] = True
    return out_hist, out_cur, valid


def build_runtime_features_from_scenario(scenario: Any, iteration: int, cfg: dict[str, Any], candidates=None) -> RuntimeFeatures:
    runtime_cfg = cfg.get("runtime", {})
    hist_s = float(runtime_cfg.get("history_s", 2.0))
    hist_hz = int(runtime_cfg.get("history_hz", 10))
    h_steps = int(round(hist_s * hist_hz)) + 1
    max_agents = int(runtime_cfg.get("max_agents", 32))
    radius = float(runtime_cfg.get("agent_radius_m", 80.0))
    map_radius = float(runtime_cfg.get("map_radius_m", 100.0))

    current_ego = _call(scenario, ["get_ego_state_at_iteration"], iteration, default=None)
    ego_arr_global = _state_to_array(current_ego)
    origin_xy = ego_arr_global[:2].copy()
    origin_yaw = float(ego_arr_global[2])

    past_ego = _call(
        scenario,
        ["get_ego_past_trajectory", "get_past_ego_trajectory"],
        iteration,
        time_horizon=hist_s,
        num_samples=h_steps - 1,
        default=[],
    )
    ego_states = [_state_to_array(s) for s in (list(past_ego) if past_ego is not None else [])] + [ego_arr_global]
    ego_history_global = pad_array(np.asarray(ego_states[-h_steps:], dtype=np.float32), (h_steps, 5))
    ego_history = transform_states_to_local(ego_history_global, origin_xy, origin_yaw)

    current_objects = _call(scenario, ["get_tracked_objects_at_iteration"], iteration, default=[])
    cur_objs = _iter_tracked_objects(current_objects)
    raw_current = np.asarray([_box_to_array(o) for o in cur_objs], dtype=np.float32) if cur_objs else np.zeros((0, 10), dtype=np.float32)
    if len(raw_current):
        xy_yaw_v = raw_current[:, [0, 1, 2, 3, 4]]
        local_main = transform_states_to_local(xy_yaw_v, origin_xy, origin_yaw)
        raw_current[:, 0:5] = local_main
        raw_current[:, 5:7] = raw_current[:, 5:7] @ np.array(
            [[math.cos(-origin_yaw), -math.sin(-origin_yaw)], [math.sin(-origin_yaw), math.cos(-origin_yaw)]],
            dtype=np.float32,
        ).T

    past_objects = _call(
        scenario,
        ["get_past_tracked_objects", "get_tracked_objects_past_trajectory"],
        iteration,
        time_horizon=hist_s,
        num_samples=h_steps - 1,
        default=None,
    )
    raw_hist = np.zeros((len(raw_current), h_steps, 10), dtype=np.float32)
    if len(raw_current):
        raw_hist[:, -1, :] = raw_current
        if past_objects is not None:
            frames = list(past_objects) if isinstance(past_objects, Iterable) else []
            frames = frames[-(h_steps - 1) :]
            token_to_idx = {int(raw_current[i, 9] * 1000000): i for i in range(len(raw_current))}
            start = h_steps - 1 - len(frames)
            for fi, frame in enumerate(frames):
                for obj in _iter_tracked_objects(frame):
                    arr = _box_to_array(obj)
                    key = int(arr[9] * 1000000)
                    if key in token_to_idx:
                        main = transform_states_to_local(arr[[0, 1, 2, 3, 4]], origin_xy, origin_yaw)
                        arr[0:5] = main
                        raw_hist[token_to_idx[key], start + fi] = arr
    cand_traj = None if candidates is None else getattr(candidates, "trajectories", candidates)
    agent_hist, current_agents, agent_valid = select_agents_deterministic(
        raw_current, raw_hist, np.zeros(2, dtype=np.float32), max_agents, radius, cand_traj
    )

    traffic_lights = _traffic_lights_to_list(
        _call(scenario, ["get_traffic_light_status_at_iteration"], iteration, default=[])
    )
    route_ids = list(_call(scenario, ["get_route_roadblock_ids", "route_roadblock_ids"], default=[]) or [])
    mission_goal_state = _call(scenario, ["get_mission_goal", "mission_goal"], default=None)
    mission_goal = None if mission_goal_state is None else transform_states_to_local(_state_to_array(mission_goal_state)[None], origin_xy, origin_yaw)[0]
    map_features = extract_map_features_from_api(getattr(scenario, "map_api", None), ego_history[-1], map_radius, route_ids)
    return RuntimeFeatures(
        ego_history=ego_history.astype(np.float32),
        agent_history=agent_hist.astype(np.float32),
        agent_valid=agent_valid,
        current_agents=current_agents.astype(np.float32),
        traffic_lights=traffic_lights,
        map_features=map_features,
        route_roadblock_ids=[str(r) for r in route_ids],
        mission_goal=mission_goal,
        metadata={
            "scenario_token": str(getattr(scenario, "token", getattr(scenario, "scenario_name", ""))),
            "iteration": int(iteration),
            "origin_xy": origin_xy,
            "origin_yaw": origin_yaw,
        },
    )


def build_runtime_features_from_arrays(
    ego_history: np.ndarray,
    agent_history: np.ndarray | None = None,
    current_agents: np.ndarray | None = None,
    traffic_lights: list[dict[str, Any]] | None = None,
    map_features: dict[str, Any] | None = None,
    route_roadblock_ids: list[str] | None = None,
    mission_goal: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
) -> RuntimeFeatures:
    cfg = cfg or {}
    max_agents = int(cfg.get("runtime", {}).get("max_agents", 32))
    h_steps = int(round(float(cfg.get("runtime", {}).get("history_s", 2.0)) * int(cfg.get("runtime", {}).get("history_hz", 10)))) + 1
    hist = np.zeros((max_agents, h_steps, 10), dtype=np.float32)
    cur = np.zeros((max_agents, 10), dtype=np.float32)
    valid = np.zeros((max_agents,), dtype=bool)
    if agent_history is not None and current_agents is not None:
        n = min(max_agents, len(current_agents))
        hist[:n, : min(h_steps, agent_history.shape[1]), : min(10, agent_history.shape[-1])] = agent_history[:n, :h_steps, :10]
        cur[:n, : min(10, current_agents.shape[-1])] = current_agents[:n, :10]
        valid[:n] = True
    return RuntimeFeatures(
        ego_history=pad_array(ego_history, (h_steps, 5)),
        agent_history=hist,
        agent_valid=valid,
        current_agents=cur,
        traffic_lights=traffic_lights or [],
        map_features=map_features or {"route_centerline": make_default_route_centerline(), "route_corridor_width": 4.0},
        route_roadblock_ids=route_roadblock_ids or [],
        mission_goal=mission_goal,
        metadata={},
    )
