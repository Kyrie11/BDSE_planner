from __future__ import annotations

from typing import Any

import numpy as np

from bdse.data.cache_schema import RuntimeFeatures
from bdse.data.feature_builder import (
    build_runtime_features_from_arrays,
    extract_map_features_from_api,
    make_default_route_centerline,
)
from bdse.utils import nearest_polyline_distance, transform_states_to_local


def _status_to_str(value: Any) -> str:
    if value is None:
        return "unknown"
    if hasattr(value, "name"):
        return str(getattr(value, "name")).lower()
    return str(value).lower()


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
    tp = getattr(state, "time_point", None)
    t = getattr(tp, "time_s", None)
    if t is None:
        t = float(getattr(tp, "time_us", 0.0) or 0.0) / 1e6 if tp is not None else 0.0
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


def _track_token(obj: Any, fallback: str = "") -> str:
    for name in ("track_token", "tracked_object_id", "token", "id", "track_id"):
        val = getattr(obj, name, None)
        if val is not None and str(val) != "":
            return str(val)
    metadata = getattr(obj, "metadata", None)
    if metadata is not None:
        for name in ("track_token", "token", "track_id"):
            val = getattr(metadata, name, None)
            if val is not None and str(val) != "":
                return str(val)
    return fallback


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


def _maybe_xy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32).reshape(-1)
        return arr[:2] if arr.size >= 2 else None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return np.asarray([float(value[0]), float(value[1])], dtype=np.float32)
    if hasattr(value, "x") and hasattr(value, "y"):
        return np.asarray([float(getattr(value, "x")), float(getattr(value, "y"))], dtype=np.float32)
    return None


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
        status = _status_to_str(getattr(item, "status", getattr(item, "status_name", getattr(item, "traffic_light_status_type", item))))
        lane_id = str(getattr(item, "lane_connector_id", getattr(item, "lane_id", getattr(item, "connector_id", ""))))
        rec: dict[str, Any] = {"status": status, "lane_connector_id": lane_id}
        for attr in ("xy", "stop_line_center", "center", "point"):
            xy = _maybe_xy(getattr(item, attr, None))
            if xy is not None:
                rec["xy"] = xy.astype(np.float32)
                rec["stop_line_center"] = xy.astype(np.float32)
                break
        ts = getattr(item, "timestamp", getattr(item, "timestamp_us", None))
        if ts is not None:
            rec["timestamp_us"] = int(ts)
        out.append(rec)
    return out


def _map_api_from_initialization(initialization: Any) -> Any | None:
    if initialization is None:
        return None
    for attr in ("map_api", "map", "_map_api"):
        val = getattr(initialization, attr, None)
        if val is not None:
            return val
    return None


def _fallback_map_features(initialization: Any, cfg: dict[str, Any], reason: str) -> dict[str, Any]:
    route_ids = list(getattr(initialization, "route_roadblock_ids", []) or []) if initialization is not None else []
    goal = getattr(initialization, "mission_goal", None) if initialization is not None else None
    runtime_cfg = cfg.get("runtime", {})
    route = make_default_route_centerline(float(runtime_cfg.get("route_horizon_m", 160.0)))
    return {
        "route_centerline": route,
        "route_source": "fallback_straight",
        "route_corridor_width": float(cfg.get("candidate", {}).get("route_width_m", 4.0)),
        "route_roadblock_ids": route_ids,
        "mission_goal_raw": str(goal) if goal is not None else "",
        "map_valid": False,
        "runtime_adapter": reason,
        "speed_limit_mps": float(runtime_cfg.get("default_speed_limit_mps", 13.4)),
        "stop_lines": [],
        "red_lane_connectors": [],
        "red_lane_connector_ids": [],
        "drivable_polygons": [],
        "lane_change": {"left": False, "right": False},
    }


def _map_features_from_initialization(initialization: Any, cfg: dict[str, Any], current_global: np.ndarray, traffic_lights: list[dict[str, Any]]) -> dict[str, Any]:
    route_ids = list(getattr(initialization, "route_roadblock_ids", []) or []) if initialization is not None else []
    map_api = _map_api_from_initialization(initialization)
    if map_api is None:
        return _fallback_map_features(initialization, cfg, "planner_input_no_map_api")
    try:
        features = extract_map_features_from_api(
            map_api,
            np.asarray(current_global, dtype=np.float32),
            float(cfg.get("runtime", {}).get("map_radius_m", 100.0)),
            route_ids,
            traffic_lights=traffic_lights,
            cfg=cfg,
        )
        features["route_roadblock_ids"] = route_ids
        features["runtime_adapter"] = "planner_input_map_api"
        return features
    except Exception as exc:
        features = _fallback_map_features(initialization, cfg, "planner_input_map_api_failed")
        features["map_error"] = repr(exc)
        return features


def _rotate_velocity_to_local(vxy: np.ndarray, origin_yaw: float) -> np.ndarray:
    arr = np.asarray(vxy, dtype=np.float32).reshape(-1, 2)
    c = float(np.cos(origin_yaw))
    s = float(np.sin(origin_yaw))
    out = np.empty_like(arr)
    out[:, 0] = c * arr[:, 0] + s * arr[:, 1]
    out[:, 1] = -s * arr[:, 0] + c * arr[:, 1]
    return out.reshape(np.asarray(vxy).shape)


def _agent_global_to_local(agent: np.ndarray, current_global: np.ndarray, t_rel: float = 0.0) -> np.ndarray:
    arr = np.asarray(agent, dtype=np.float32).reshape(-1).copy()
    out = np.zeros((10,), dtype=np.float32)
    out[: min(10, arr.size)] = arr[: min(10, arr.size)]
    pose = transform_states_to_local(out[None, :5], current_global[:2], float(current_global[2]))[0]
    out[:5] = pose
    out[4] = float(t_rel)
    out[5:7] = _rotate_velocity_to_local(out[5:7][None, :], float(current_global[2]))[0]
    out[3] = float(np.hypot(out[5], out[6])) if np.isfinite(out[5:7]).all() else float(out[3])
    return out


def _object_maps_by_token(observations: list[Any]) -> list[dict[str, np.ndarray]]:
    frames: list[dict[str, np.ndarray]] = []
    for obs in observations:
        frame: dict[str, np.ndarray] = {}
        for idx, obj in enumerate(_iter_objects(obs)):
            token = _track_token(obj, fallback=f"det_{idx:04d}")
            if token in frame:
                token = f"{token}#{idx}"
            frame[token] = _object_to_array(obj)
        frames.append(frame)
    return frames


def _select_current_agent_tokens(current_frame: dict[str, np.ndarray], current_global: np.ndarray, route_local: np.ndarray, cfg: dict[str, Any]) -> list[str]:
    runtime_cfg = cfg.get("runtime", {})
    max_agents = int(runtime_cfg.get("max_agents", 32))
    radius = float(runtime_cfg.get("agent_radius_m", 80.0))
    route_radius = float(cfg.get("evidence", {}).get("interaction_route_radius_m", cfg.get("candidate", {}).get("route_width_m", 4.0) * 2.0))
    rows = []
    for token, agent in current_frame.items():
        local = _agent_global_to_local(agent, current_global, 0.0)
        xy = local[:2]
        dist = float(np.linalg.norm(xy))
        if dist > radius:
            continue
        route_d = 1e6
        if route_local is not None and np.asarray(route_local).size >= 4:
            try:
                route_d = float(nearest_polyline_distance(xy[None, :], route_local)[0])
            except Exception:
                route_d = 1e6
        closing = max(float(local[5]), 0.0)
        ttc_proxy = float(local[0] / max(closing, 0.1)) if local[0] > 0.0 else 1e3
        rows.append((0 if route_d <= route_radius else 1, min(ttc_proxy, 1e3), dist, route_d, token))
    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    return [str(r[-1]) for r in rows[:max_agents]]


def _pad_tail(seq: list[Any], n: int) -> list[Any]:
    tail = list(seq[-n:])
    if not tail:
        return []
    if len(tail) < n:
        tail = [tail[0]] * (n - len(tail)) + tail
    return tail


def _build_agent_history_from_observations(
    observations: list[Any],
    ego_local: np.ndarray,
    current_global: np.ndarray,
    map_features: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    h_steps = int(ego_local.shape[0])
    max_agents = int(cfg.get("runtime", {}).get("max_agents", 32))
    if not observations:
        return np.zeros((0, h_steps, 10), dtype=np.float32), np.zeros((0, 10), dtype=np.float32), [], {"agent_history_mode": "none"}
    obs_tail = _pad_tail(observations, h_steps)
    frames = _object_maps_by_token(obs_tail)
    current_frame = frames[-1] if frames else {}
    route = np.asarray(map_features.get("route_centerline", make_default_route_centerline()), dtype=np.float32).reshape(-1, 2)
    selected_tokens = _select_current_agent_tokens(current_frame, current_global, route, cfg)
    n = min(len(selected_tokens), max_agents)
    hist = np.zeros((n, h_steps, 10), dtype=np.float32)
    cur = np.zeros((n, 10), dtype=np.float32)
    missing = 0
    exact = 0
    for j, token in enumerate(selected_tokens[:n]):
        fallback_global = current_frame.get(token, np.zeros((10,), dtype=np.float32))
        last_global = fallback_global
        for k, frame in enumerate(frames):
            arr = frame.get(token)
            if arr is None:
                missing += 1
                arr = last_global
            else:
                exact += 1
                last_global = arr
            hist[j, k] = _agent_global_to_local(arr, current_global, float(ego_local[k, 4]))
        cur[j] = hist[j, -1]
    meta = {
        "agent_history_mode": "tracked_id_alignment" if exact > n else "current_repeat_fallback",
        "selected_agent_count": int(n),
        "selected_agent_tokens": selected_tokens[:n],
        "agent_history_exact_frames": int(exact),
        "agent_history_missing_filled": int(missing),
    }
    return hist, cur, selected_tokens[:n], meta


def _attach_traffic_light_geometry(traffic: list[dict[str, Any]], map_features: dict[str, Any]) -> list[dict[str, Any]]:
    stop_lines = list(map_features.get("stop_lines", []) or [])
    red_connectors = list(map_features.get("red_lane_connectors", []) or [])
    by_connector = {str(x.get("id", "")): np.asarray(x.get("xy", []), dtype=np.float32).reshape(-1, 2) for x in red_connectors}
    out: list[dict[str, Any]] = []
    for item in traffic:
        rec = dict(item)
        if "xy" not in rec and "stop_line_center" not in rec:
            lane_id = str(rec.get("lane_connector_id", ""))
            xy: np.ndarray | None = None
            for sl in stop_lines:
                sid = str(sl.get("id", ""))
                if lane_id and sid.startswith(lane_id):
                    pts = np.asarray(sl.get("xy", []), dtype=np.float32).reshape(-1, 2)
                    if len(pts):
                        xy = pts.mean(axis=0)
                        break
            if xy is None and lane_id in by_connector and len(by_connector[lane_id]):
                xy = by_connector[lane_id][0]
            if xy is None:
                red_stops = [np.asarray(sl.get("xy", []), dtype=np.float32).reshape(-1, 2) for sl in stop_lines if bool(sl.get("red", False))]
                red_stops = [x for x in red_stops if len(x)]
                if red_stops:
                    xy = red_stops[0].mean(axis=0)
            if xy is not None:
                rec["xy"] = np.asarray(xy, dtype=np.float32)
                rec["stop_line_center"] = np.asarray(xy, dtype=np.float32)
        out.append(rec)
    map_features["traffic_lights"] = out
    return out


def build_runtime_features_from_planner_input(current_input: Any, initialization: Any, cfg: dict[str, Any]) -> RuntimeFeatures:
    history = getattr(current_input, "history", None)
    ego_states, observations = _history_lists(history)
    if not ego_states:
        raise TypeError("PlannerInput history does not expose ego_states/current_state; cannot build BDSE runtime features")
    h_steps = int(round(float(cfg.get("runtime", {}).get("history_s", 2.0)) * int(cfg.get("runtime", {}).get("history_hz", 10)))) + 1
    ego_tail = _pad_tail(ego_states, h_steps)
    ego_global = np.asarray([_state_to_array(s) for s in ego_tail], dtype=np.float32)
    current_global = ego_global[-1].copy()
    ego_local = transform_states_to_local(ego_global, current_global[:2], float(current_global[2]))
    ego_local[:, 4] = np.linspace(-float(cfg.get("runtime", {}).get("history_s", 2.0)), 0.0, h_steps, dtype=np.float32)

    traffic = _traffic_lights_from_input(current_input)
    map_features = _map_features_from_initialization(initialization, cfg, current_global, traffic)
    traffic = _attach_traffic_light_geometry(traffic, map_features)

    agent_history, current_agents, selected_tokens, agent_meta = _build_agent_history_from_observations(
        observations,
        ego_local,
        current_global,
        map_features,
        cfg,
    )

    route_ids = list(getattr(initialization, "route_roadblock_ids", []) or []) if initialization is not None else []
    goal_obj = getattr(initialization, "mission_goal", None) if initialization is not None else None
    goal_arr = None
    if goal_obj is not None:
        goal_global = _state_to_array(goal_obj)
        goal_arr = transform_states_to_local(goal_global[None, :], current_global[:2], float(current_global[2]))[0, :4].astype(np.float32)

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
    runtime.metadata.update(agent_meta)
    runtime.metadata["source"] = "PlannerInput.history_observations"
    runtime.metadata["scenario_db_replay_used"] = False
    runtime.metadata["map_valid"] = bool(map_features.get("map_valid", False))
    runtime.metadata["map_runtime_adapter"] = str(map_features.get("runtime_adapter", "unknown"))
    runtime.metadata["history_states"] = int(h_steps)
    runtime.metadata["history_window_s"] = float(cfg.get("runtime", {}).get("history_s", 2.0))
    runtime.metadata["selected_agent_tokens"] = selected_tokens
    return runtime
