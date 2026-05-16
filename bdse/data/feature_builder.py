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
            except Exception:
                continue
        elif fn is not None and not args and not kwargs:
            return fn
    return default


def _state_to_array(state: Any) -> np.ndarray:
    if state is None:
        return np.zeros((5,), dtype=np.float32)
    if isinstance(state, np.ndarray):
        arr = state.astype(np.float32).reshape(-1)
        out = np.zeros((5,), dtype=np.float32)
        out[: min(5, arr.size)] = arr[: min(5, arr.size)]
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
        if speed == 0.0:
            vel = getattr(dyn, "rear_axle_velocity_2d", getattr(dyn, "center_velocity_2d", None))
            if vel is not None:
                speed = math.hypot(float(getattr(vel, "x", 0.0)), float(getattr(vel, "y", 0.0)))
    else:
        speed = float(getattr(state, "velocity", getattr(state, "speed", 0.0)) or 0.0)
    return np.asarray([x, y, yaw, speed, 0.0], dtype=np.float32)


def _box_to_array(obj: Any) -> np.ndarray:
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
    token = getattr(obj, "track_token", getattr(obj, "token", getattr(obj, "tracked_object_id", "")))
    token_hash = float(abs(hash(str(token))) % 1000000) / 1000000.0
    return np.asarray([x, y, yaw, math.hypot(vx, vy), 0.0, vx, vy, length, width, token_hash], dtype=np.float32)


def _object_token(obj: Any, fallback: int | str = "") -> str:
    return str(getattr(obj, "track_token", getattr(obj, "token", getattr(obj, "tracked_object_id", fallback))))


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
    out: list[dict[str, Any]] = []
    items = statuses if isinstance(statuses, (list, tuple, set)) else list(statuses) if hasattr(statuses, "__iter__") else [statuses]
    for st in items:
        lane_connector_id = str(getattr(st, "lane_connector_id", getattr(st, "connector_id", "")))
        status = getattr(st, "status", getattr(st, "traffic_light_status_type", "unknown"))
        out.append({"lane_connector_id": lane_connector_id, "status": str(status).lower(), "timestamp_us": int(getattr(st, "timestamp", getattr(st, "timestamp_us", 0)) or 0)})
    return out


def make_default_route_centerline(horizon_m: float = 160.0, step_m: float = 2.0) -> np.ndarray:
    x = np.arange(0.0, horizon_m + step_m, step_m, dtype=np.float32)
    return np.stack([x, np.zeros_like(x)], axis=1)


def _layer(name: str) -> Any:
    try:
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        return getattr(SemanticMapLayer, name)
    except Exception:
        return name


def _point2d(x: float, y: float) -> Any:
    try:
        from nuplan.common.actor_state.state_representation import Point2D
        return Point2D(float(x), float(y))
    except Exception:
        return (float(x), float(y))


def _xy_from_point(p: Any) -> tuple[float, float] | None:
    if p is None:
        return None
    if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
        return float(p[0]), float(p[1])
    if hasattr(p, "x") and hasattr(p, "y"):
        return float(p.x), float(p.y)
    return None


def _polyline_from_path(path: Any) -> np.ndarray:
    if path is None:
        return np.zeros((0, 2), dtype=np.float32)
    pts = getattr(path, "discrete_path", path)
    out = []
    try:
        iterator = list(pts)
    except TypeError:
        iterator = []
    for p in iterator:
        xy = _xy_from_point(p)
        if xy is not None:
            out.append(xy)
    return np.asarray(out, dtype=np.float32).reshape(-1, 2) if out else np.zeros((0, 2), dtype=np.float32)


def _geometry_points(obj: Any) -> np.ndarray:
    candidates = [getattr(obj, "baseline_path", None), getattr(obj, "polygon", None), getattr(obj, "linestring", None), getattr(obj, "geometry", None), getattr(obj, "exterior", None)]
    for geom in candidates:
        if geom is None:
            continue
        if hasattr(geom, "discrete_path"):
            arr = _polyline_from_path(geom)
            if len(arr):
                return arr
        if hasattr(geom, "exterior"):
            geom = geom.exterior
        if hasattr(geom, "coords"):
            try:
                arr = np.asarray(list(geom.coords), dtype=np.float32)[:, :2]
                if len(arr):
                    return arr
            except Exception:
                pass
        arr = _polyline_from_path(geom)
        if len(arr):
            return arr
    return np.zeros((0, 2), dtype=np.float32)


def _baseline_points(obj: Any) -> np.ndarray:
    arr = _polyline_from_path(getattr(obj, "baseline_path", None))
    if len(arr):
        return arr
    return _geometry_points(obj)


def _to_local_xy(points_global: np.ndarray, origin_xy: np.ndarray, origin_yaw: float) -> np.ndarray:
    arr = np.asarray(points_global, dtype=np.float32).reshape(-1, 2)
    if len(arr) == 0:
        return arr
    c = math.cos(-origin_yaw)
    s = math.sin(-origin_yaw)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return ((arr - origin_xy[None, :]) @ rot.T).astype(np.float32)


def _extract_edges(obj: Any) -> list[Any]:
    edges = []
    for attr in ["interior_edges", "incoming_edges", "outgoing_edges"]:
        val = getattr(obj, attr, None)
        if val is None:
            continue
        try:
            edges.extend(list(val))
        except TypeError:
            pass
    return edges


def _obj_id(obj: Any) -> str:
    return str(getattr(obj, "id", getattr(obj, "token", getattr(obj, "lane_connector_id", ""))))


def _get_map_object(map_api: Any, obj_id: str, layers: Sequence[str]) -> Any | None:
    for layer_name in layers:
        layer = _layer(layer_name)
        for args in [(obj_id, layer), (layer, obj_id), (obj_id,)]:
            got = _call(map_api, ["get_map_object", "get_map_object_by_id"], *args, default=None)
            if got is not None:
                return got
    return None


def _proximal_objects(map_api: Any, center_global: np.ndarray, radius_m: float) -> dict[Any, list[Any]]:
    layer_names = ["LANE", "LANE_CONNECTOR", "STOP_LINE", "CROSSWALK", "INTERSECTION", "DRIVABLE_AREA", "ROADBLOCK", "ROADBLOCK_CONNECTOR"]
    layers = [_layer(n) for n in layer_names]
    center = _point2d(float(center_global[0]), float(center_global[1]))
    prox = _call(map_api, ["get_proximal_map_objects"], center, radius_m, layers, default=None)
    if prox is None:
        prox = _call(map_api, ["get_proximal_map_objects"], center, radius_m, layer_names, default={})
    return prox if isinstance(prox, dict) else {}


def _flatten_by_layer(prox: dict[Any, list[Any]], layer_name: str) -> list[Any]:
    out = []
    for key, values in prox.items():
        name = str(getattr(key, "name", key)).upper()
        if layer_name.upper() not in name:
            continue
        try:
            out.extend(list(values))
        except TypeError:
            pass
    return out


def _concat_route_polylines(polys_local: list[np.ndarray]) -> np.ndarray:
    polys = [p for p in polys_local if p.ndim == 2 and p.shape[0] >= 2]
    if not polys:
        return make_default_route_centerline()
    # Keep route pieces that are near/ahead of ego and sort by local longitudinal position.
    polys = sorted(polys, key=lambda p: float(np.nanmin(p[:, 0])))
    out: list[np.ndarray] = []
    for p in polys:
        if len(out) and np.linalg.norm(out[-1][-1] - p[0]) < 2.0:
            out[-1] = np.concatenate([out[-1], p[1:]], axis=0)
        else:
            out.append(p)
    merged = np.concatenate(out, axis=0)
    if merged.shape[0] < 2 or np.nanmax(merged[:, 0]) < 5.0:
        return max(polys, key=lambda p: p.shape[0])
    return merged.astype(np.float32)


def _speed_limit_from_obj(obj: Any) -> float | None:
    for attr in ["speed_limit_mps", "speed_limit", "speed_limit_meters_per_second"]:
        val = getattr(obj, attr, None)
        if val is not None:
            try:
                v = float(val)
                if v > 0:
                    return v
            except Exception:
                pass
    return None


def extract_map_features_from_api(map_api: Any, ego_state_global: np.ndarray, radius_m: float, route_ids: list[str], traffic_lights: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    origin_xy = np.asarray(ego_state_global[:2], dtype=np.float32)
    origin_yaw = float(ego_state_global[2])
    features: dict[str, Any] = {
        "route_centerline": make_default_route_centerline(),
        "route_source": "fallback_straight",
        "stop_lines": [],
        "crosswalks": [],
        "speed_limits": [],
        "speed_limit_mps": 13.4,
        "drivable_polygons": [],
        "route_corridor_width": 4.0,
        "lane_change": {"left": False, "right": False},
        "red_lane_connector_ids": [],
        "map_valid": False,
    }
    if map_api is None:
        return features

    red_connector_ids = {str(t.get("lane_connector_id", "")) for t in (traffic_lights or []) if "red" in str(t.get("status", "")).lower() and str(t.get("lane_connector_id", ""))}
    features["red_lane_connector_ids"] = sorted(red_connector_ids)
    prox = _proximal_objects(map_api, origin_xy, radius_m)

    route_polys: list[np.ndarray] = []
    route_objs: list[Any] = []
    for rid in route_ids:
        obj = _get_map_object(map_api, str(rid), ["ROADBLOCK", "ROADBLOCK_CONNECTOR", "LANE", "LANE_CONNECTOR"])
        if obj is None:
            continue
        route_objs.append(obj)
        edges = _extract_edges(obj) or [obj]
        for edge in edges:
            pts = _baseline_points(edge)
            if len(pts) >= 2:
                route_polys.append(_to_local_xy(pts, origin_xy, origin_yaw))
            v = _speed_limit_from_obj(edge)
            if v is not None:
                features["speed_limits"].append({"id": _obj_id(edge), "speed_limit_mps": v})
    if not route_polys:
        lanes = _flatten_by_layer(prox, "LANE") + _flatten_by_layer(prox, "LANE_CONNECTOR")
        scored = []
        for obj in lanes:
            pts = _baseline_points(obj)
            if len(pts) < 2:
                continue
            local = _to_local_xy(pts, origin_xy, origin_yaw)
            if np.nanmax(local[:, 0]) < -5.0:
                continue
            score = float(np.nanmin(np.linalg.norm(local, axis=1))) + 0.05 * abs(float(local[-1, 1]))
            scored.append((score, local, obj))
        for _, local, obj in sorted(scored, key=lambda x: x[0])[:6]:
            route_polys.append(local)
            route_objs.append(obj)
            v = _speed_limit_from_obj(obj)
            if v is not None:
                features["speed_limits"].append({"id": _obj_id(obj), "speed_limit_mps": v})
    if route_polys:
        features["route_centerline"] = _concat_route_polylines(route_polys)
        features["route_source"] = "route_roadblocks" if route_ids else "proximal_lane"
        features["map_valid"] = True

    if features["speed_limits"]:
        features["speed_limit_mps"] = float(np.median([x["speed_limit_mps"] for x in features["speed_limits"]]))

    for obj in _flatten_by_layer(prox, "DRIVABLE_AREA"):
        pts = _geometry_points(obj)
        if len(pts) >= 3:
            local = _to_local_xy(pts, origin_xy, origin_yaw)
            features["drivable_polygons"].append({"id": _obj_id(obj), "xy": local})
    for obj in _flatten_by_layer(prox, "STOP_LINE"):
        pts = _geometry_points(obj)
        if len(pts) >= 2:
            local = _to_local_xy(pts, origin_xy, origin_yaw)
            centroid = local.mean(axis=0)
            if -20.0 <= float(centroid[0]) <= radius_m + 20.0:
                features["stop_lines"].append({"id": _obj_id(obj), "xy": local, "red": bool(red_connector_ids)})
    for obj in _flatten_by_layer(prox, "CROSSWALK"):
        pts = _geometry_points(obj)
        if len(pts) >= 3:
            features["crosswalks"].append({"id": _obj_id(obj), "xy": _to_local_xy(pts, origin_xy, origin_yaw)})

    if route_objs:
        # Use explicit adjacent edge metadata when available. Conservative default is False.
        left = any(getattr(o, "left_neighbor", None) is not None or getattr(o, "adjacent_edges", None) is not None for o in route_objs)
        right = any(getattr(o, "right_neighbor", None) is not None or getattr(o, "adjacent_edges", None) is not None for o in route_objs)
        features["lane_change"] = {"left": bool(left), "right": bool(right)}
    return features


def _agent_selection_order(raw_current_agents: np.ndarray, ego_xy: np.ndarray, max_agents: int, radius_m: float, candidate_trajectories: np.ndarray | None = None) -> list[int]:
    if raw_current_agents.size == 0:
        return []
    cur = np.asarray(raw_current_agents, dtype=np.float32)
    dist = np.linalg.norm(cur[:, :2] - ego_xy[None, :], axis=1)
    idxs = np.flatnonzero(dist <= radius_m)
    if idxs.size == 0:
        idxs = np.argsort(dist)[: min(max_agents, len(dist))]
    min_cand_dist = np.full(len(cur), 1e6, dtype=np.float32)
    if candidate_trajectories is not None and len(candidate_trajectories):
        cand_xy = np.asarray(candidate_trajectories, dtype=np.float32)[..., :2].reshape(-1, 2)
        for i in range(len(cur)):
            min_cand_dist[i] = np.linalg.norm(cand_xy - cur[i, None, :2], axis=1).min()
    rel_speed = np.maximum(np.abs(cur[:, 3]) + 1e-3, 1e-3)
    ttc = dist / rel_speed
    return sorted(idxs.tolist(), key=lambda i: (min_cand_dist[i], ttc[i], dist[i], i))[:max_agents]


def select_agents_deterministic(raw_current_agents: np.ndarray, raw_agent_history: np.ndarray, ego_xy: np.ndarray, max_agents: int, radius_m: float, candidate_trajectories: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_steps = raw_agent_history.shape[1] if raw_agent_history.ndim == 3 else 1
    out_hist = np.zeros((max_agents, h_steps, 10), dtype=np.float32)
    out_cur = np.zeros((max_agents, 10), dtype=np.float32)
    valid = np.zeros((max_agents,), dtype=bool)
    order = _agent_selection_order(raw_current_agents, ego_xy, max_agents, radius_m, candidate_trajectories)
    hist = np.asarray(raw_agent_history, dtype=np.float32)
    cur = np.asarray(raw_current_agents, dtype=np.float32)
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

    past_ego = _call(scenario, ["get_ego_past_trajectory", "get_past_ego_trajectory"], iteration, time_horizon=hist_s, num_samples=h_steps - 1, default=[])
    ego_states = [_state_to_array(s) for s in (list(past_ego) if past_ego is not None else [])] + [ego_arr_global]
    ego_history_global = pad_array(np.asarray(ego_states[-h_steps:], dtype=np.float32), (h_steps, 5))
    ego_history = transform_states_to_local(ego_history_global, origin_xy, origin_yaw)

    current_objects = _call(scenario, ["get_tracked_objects_at_iteration"], iteration, default=[])
    cur_objs = _iter_tracked_objects(current_objects)
    raw_tokens = [_object_token(o, i) for i, o in enumerate(cur_objs)]
    raw_current = np.asarray([_box_to_array(o) for o in cur_objs], dtype=np.float32) if cur_objs else np.zeros((0, 10), dtype=np.float32)
    if len(raw_current):
        local_main = transform_states_to_local(raw_current[:, [0, 1, 2, 3, 4]], origin_xy, origin_yaw)
        raw_current[:, 0:5] = local_main
        c = math.cos(-origin_yaw)
        s = math.sin(-origin_yaw)
        rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        raw_current[:, 5:7] = raw_current[:, 5:7] @ rot.T

    past_objects = _call(scenario, ["get_past_tracked_objects", "get_tracked_objects_past_trajectory"], iteration, time_horizon=hist_s, num_samples=h_steps - 1, default=None)
    raw_hist = np.zeros((len(raw_current), h_steps, 10), dtype=np.float32)
    if len(raw_current):
        raw_hist[:, -1, :] = raw_current
        if past_objects is not None:
            frames = list(past_objects) if isinstance(past_objects, Iterable) else []
            frames = frames[-(h_steps - 1) :]
            token_to_idx = {raw_tokens[i]: i for i in range(len(raw_tokens))}
            start = h_steps - 1 - len(frames)
            for fi, frame in enumerate(frames):
                for obj in _iter_tracked_objects(frame):
                    token = _object_token(obj)
                    if token in token_to_idx:
                        arr = _box_to_array(obj)
                        arr[0:5] = transform_states_to_local(arr[[0, 1, 2, 3, 4]], origin_xy, origin_yaw)
                        raw_hist[token_to_idx[token], start + fi] = arr
    cand_traj = None if candidates is None else getattr(candidates, "trajectories", candidates)
    order = _agent_selection_order(raw_current, np.zeros(2, dtype=np.float32), max_agents, radius, cand_traj)
    agent_hist, current_agents, agent_valid = select_agents_deterministic(raw_current, raw_hist, np.zeros(2, dtype=np.float32), max_agents, radius, cand_traj)
    selected_tokens = [raw_tokens[i] for i in order]

    traffic_lights = _traffic_lights_to_list(_call(scenario, ["get_traffic_light_status_at_iteration"], iteration, default=[]))
    route_ids = list(_call(scenario, ["get_route_roadblock_ids", "route_roadblock_ids"], default=[]) or [])
    mission_goal_state = _call(scenario, ["get_mission_goal", "mission_goal"], default=None)
    mission_goal = None if mission_goal_state is None else transform_states_to_local(_state_to_array(mission_goal_state)[None], origin_xy, origin_yaw)[0]
    map_features = extract_map_features_from_api(getattr(scenario, "map_api", None), ego_arr_global, map_radius, [str(r) for r in route_ids], traffic_lights)
    return RuntimeFeatures(
        ego_history=ego_history.astype(np.float32),
        agent_history=agent_hist.astype(np.float32),
        agent_valid=agent_valid,
        current_agents=current_agents.astype(np.float32),
        traffic_lights=traffic_lights,
        map_features=map_features,
        route_roadblock_ids=[str(r) for r in route_ids],
        mission_goal=mission_goal,
        metadata={"scenario_token": str(getattr(scenario, "token", getattr(scenario, "scenario_name", ""))), "iteration": int(iteration), "origin_xy": origin_xy, "origin_yaw": origin_yaw, "selected_agent_tokens": selected_tokens, "map_valid": bool(map_features.get("map_valid", False)), "route_source": str(map_features.get("route_source", "unknown"))},
    )


def build_runtime_features_from_arrays(ego_history: np.ndarray, agent_history: np.ndarray | None = None, current_agents: np.ndarray | None = None, traffic_lights: list[dict[str, Any]] | None = None, map_features: dict[str, Any] | None = None, route_roadblock_ids: list[str] | None = None, mission_goal: np.ndarray | None = None, cfg: dict[str, Any] | None = None) -> RuntimeFeatures:
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
        map_features=map_features or {"route_centerline": make_default_route_centerline(), "route_corridor_width": 4.0, "map_valid": False},
        route_roadblock_ids=route_roadblock_ids or [],
        mission_goal=mission_goal,
        metadata={},
    )
