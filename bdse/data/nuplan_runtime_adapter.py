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
    tp = getattr(state, "time_point", None)
    if tp is not None and hasattr(tp, "time_s"):
        t = float(getattr(tp, "time_s") or 0.0)
    elif tp is not None and hasattr(tp, "time_us"):
        t = float(getattr(tp, "time_us") or 0.0) / 1e6
    else:
        t = float(getattr(state, "time_s", getattr(state, "timestamp", 0.0)) or 0.0)
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


def _semantic_map_layer(name: str) -> Any | None:
    try:
        mod = __import__("nuplan.common.maps.maps_datatypes", fromlist=["SemanticMapLayer"])
        return getattr(getattr(mod, "SemanticMapLayer"), name)
    except Exception:
        return None


def _xy_array_from_geometry(obj: Any) -> np.ndarray:
    """Best-effort conversion of nuPlan/shapely map geometries to [N,2]."""
    if obj is None:
        return np.zeros((0, 2), dtype=np.float32)
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj, dtype=np.float32).reshape(-1, obj.shape[-1] if obj.ndim else 1)
        return arr[:, :2] if arr.shape[1] >= 2 else np.zeros((0, 2), dtype=np.float32)
    # Common nuPlan path containers.
    for attr in ("discrete_path", "poses", "points"):
        val = getattr(obj, attr, None)
        if val is not None and val is not obj:
            arr = _xy_array_from_geometry(val)
            if len(arr) >= 2:
                return arr
    if isinstance(obj, (list, tuple)):
        pts = []
        for item in obj:
            x = getattr(item, "x", None)
            y = getattr(item, "y", None)
            if x is not None and y is not None:
                pts.append((float(x), float(y)))
            elif isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                pts.append((float(item[0]), float(item[1])))
        return np.asarray(pts, dtype=np.float32).reshape(-1, 2) if pts else np.zeros((0, 2), dtype=np.float32)
    # Shapely-like geometry.
    for attr in ("linestring", "polygon", "geometry"):
        val = getattr(obj, attr, None)
        if val is not None and val is not obj:
            arr = _xy_array_from_geometry(val)
            if len(arr) >= 2:
                return arr
    ext = getattr(obj, "exterior", None)
    if ext is not None:
        coords = getattr(ext, "coords", None)
        if coords is not None:
            return np.asarray(list(coords), dtype=np.float32).reshape(-1, 2)[:, :2]
    coords = getattr(obj, "coords", None)
    if coords is not None:
        return np.asarray(list(coords), dtype=np.float32).reshape(-1, 2)[:, :2]
    x = getattr(obj, "x", None)
    y = getattr(obj, "y", None)
    if x is not None and y is not None:
        return np.asarray([[float(x), float(y)]], dtype=np.float32)
    return np.zeros((0, 2), dtype=np.float32)


def _object_baseline_polylines(obj: Any) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for attr in ("baseline_path", "path"):
        val = getattr(obj, attr, None)
        if val is not None:
            arr = _xy_array_from_geometry(val)
            if len(arr) >= 2:
                out.append(arr)
    for attr in ("interior_edges", "incoming_edges", "outgoing_edges"):
        edges = getattr(obj, attr, None)
        if edges is None:
            continue
        try:
            iterator = list(edges)
        except TypeError:
            iterator = []
        for edge in iterator:
            out.extend(_object_baseline_polylines(edge))
    arr = _xy_array_from_geometry(obj)
    if len(arr) >= 2:
        out.append(arr)
    return out


def _to_local_xy(points_xy: np.ndarray, current_state: np.ndarray | None) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if current_state is None or len(pts) == 0:
        return pts
    st = np.zeros((len(pts), 5), dtype=np.float32)
    st[:, :2] = pts
    local = transform_states_to_local(st, current_state[:2], float(current_state[2]))
    return local[:, :2].astype(np.float32)


def _get_map_object_any_layer(map_api: Any, object_id: str) -> Any | None:
    if map_api is None or not object_id:
        return None
    for name in ("ROADBLOCK", "ROADBLOCK_CONNECTOR", "LANE", "LANE_CONNECTOR"):
        layer = _semantic_map_layer(name)
        if layer is None:
            continue
        try:
            obj = map_api.get_map_object(str(object_id), layer)
            if obj is not None:
                return obj
        except Exception:
            continue
    return None


def _route_from_map_api(initialization: Any, current_state: np.ndarray | None, cfg: dict[str, Any]) -> np.ndarray:
    route_ids = list(getattr(initialization, "route_roadblock_ids", []) or []) if initialization is not None else []
    map_api = getattr(initialization, "map_api", None) if initialization is not None else None
    pieces: list[np.ndarray] = []
    for rid in route_ids:
        obj = _get_map_object_any_layer(map_api, str(rid))
        if obj is None:
            continue
        baselines = _object_baseline_polylines(obj)
        if not baselines:
            continue
        # Prefer the first baseline exposed by nuPlan.  This is not a route search;
        # it simply makes closed-loop runtime use real map geometry rather than a
        # synthetic straight route.
        pieces.append(np.asarray(baselines[0], dtype=np.float32).reshape(-1, 2))
    if pieces:
        route_global = []
        for pts in pieces:
            if len(pts) < 2:
                continue
            if route_global and np.linalg.norm(route_global[-1] - pts[0]) < 1e-3:
                route_global.extend(pts[1:])
            else:
                route_global.extend(pts)
        if len(route_global) >= 2:
            route_local = _to_local_xy(np.asarray(route_global, dtype=np.float32), current_state)
            max_pts = int(cfg.get("runtime", {}).get("max_route_points", 512))
            return route_local[:max_pts]
    return make_default_route_centerline(float(cfg.get("runtime", {}).get("route_horizon_m", 160.0)))


def _stop_lines_from_map_api(initialization: Any, current_state: np.ndarray | None, traffic_lights: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    map_api = getattr(initialization, "map_api", None) if initialization is not None else None
    if map_api is None or current_state is None:
        return []
    layer = _semantic_map_layer("STOP_LINE")
    if layer is None:
        return []
    try:
        point_mod = __import__("nuplan.common.actor_state.state_representation", fromlist=["Point2D"])
        Point2D = getattr(point_mod, "Point2D")
        center = Point2D(float(current_state[0]), float(current_state[1]))
    except Exception:
        center = current_state[:2]
    radius = float(cfg.get("runtime", {}).get("map_radius_m", 100.0))
    try:
        prox = map_api.get_proximal_map_objects(center, radius, [layer])
        objs = prox.get(layer, []) if isinstance(prox, dict) else []
    except Exception:
        objs = []
    red_ids = {str(t.get("lane_connector_id", "")) for t in traffic_lights if "red" in str(t.get("status", "")).lower()}
    out: list[dict[str, Any]] = []
    for obj in list(objs):
        xy = _xy_array_from_geometry(obj)
        if len(xy) < 2:
            continue
        local = _to_local_xy(xy, current_state)
        obj_id = str(getattr(obj, "id", getattr(obj, "token", "")))
        conn_ids = []
        for attr in ("lane_connector_id", "lane_connector_ids", "lane_connector_fid", "lane_connector_fids"):
            val = getattr(obj, attr, None)
            if val is None:
                continue
            if isinstance(val, (list, tuple, set)):
                conn_ids.extend(str(v) for v in val)
            else:
                conn_ids.append(str(val))
        is_red = bool(red_ids and (obj_id in red_ids or any(cid in red_ids for cid in conn_ids)))
        out.append({"xy": local.astype(np.float32), "red": is_red, "status": "red" if is_red else "unknown", "id": obj_id})
    return out


def _map_features_from_initialization(initialization: Any, cfg: dict[str, Any], current_state: np.ndarray | None = None, traffic_lights: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    # Runtime-safe map context from nuPlan initialization.  This must not read
    # offline scenario labels, but it should use the route and map API available
    # to the deployed planner; otherwise closed-loop would plan on a synthetic
    # straight route with no stop-line geometry.
    traffic_lights = list(traffic_lights or [])
    route_ids = list(getattr(initialization, "route_roadblock_ids", []) or []) if initialization is not None else []
    goal = getattr(initialization, "mission_goal", None) if initialization is not None else None
    route = _route_from_map_api(initialization, current_state, cfg)
    stop_lines = _stop_lines_from_map_api(initialization, current_state, traffic_lights, cfg)
    return {
        "route_centerline": route,
        "route_corridor_width": float(cfg.get("candidate", {}).get("route_width_m", 4.0)),
        "route_roadblock_ids": route_ids,
        "mission_goal_raw": str(goal) if goal is not None else "",
        "stop_lines": stop_lines,
        "traffic_lights": traffic_lights,
        "map_valid": bool(len(route) >= 2 and not np.allclose(route[:, 1], 0.0)) or bool(stop_lines),
        "runtime_adapter": "planner_input_map_api_best_effort",
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
    ego_local = transform_states_to_local(ego_global, current[:2], float(current[2]))
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
        local = transform_states_to_local(xy, current[:2], float(current[2]))
        current_agents[: len(local), :5] = local[:, :5]
        current_agents[: len(local), 5:10] = current_agents_global[:, 5:10]
        agent_history[: len(local), :, :] = current_agents[: len(local), None, :]
    traffic = _traffic_lights_from_input(current_input)
    map_features = _map_features_from_initialization(initialization, cfg, current_state=current, traffic_lights=traffic)
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
