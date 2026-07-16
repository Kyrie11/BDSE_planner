from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from bdse.data.cache_schema import RuntimeFeatures
from bdse.data.feature_builder import (
    build_runtime_features_from_arrays,
    make_default_route_centerline,
    select_agents_deterministic,
    _agent_selection_order,
)
from bdse.utils import transform_states_to_local


_ROUTE_GLOBAL_CACHE: dict[tuple[int, tuple[str, ...]], np.ndarray] = {}
_ROUTE_GLOBAL_CACHE_MAX = 4096

# Closed-loop optimization: nuPlan map queries are expensive and this adapter is
# called at every simulation tick. Stop-line geometry is static over a local map
# patch, while only the red/unknown status changes with traffic-light data.
_STOP_LINE_GLOBAL_CACHE: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
_STOP_LINE_GLOBAL_CACHE_MAX = 4096


def _route_cache_key(initialization: Any) -> tuple[int, tuple[str, ...]]:
    route_ids = tuple(str(x) for x in (list(_safe_getattr(initialization, "route_roadblock_ids", []) or []) if initialization is not None else []))
    map_api = _safe_getattr(initialization, "map_api", None) if initialization is not None else None
    return (id(map_api), route_ids)


def _cached_route_global(initialization: Any) -> np.ndarray | None:
    key = _route_cache_key(initialization)
    arr = _ROUTE_GLOBAL_CACHE.get(key)
    if arr is None or arr.size < 4:
        return None
    return arr


def _store_route_global(initialization: Any, route_global: np.ndarray) -> None:
    arr = np.asarray(route_global, dtype=np.float32).reshape(-1, 2)
    if len(arr) < 2:
        return
    key = _route_cache_key(initialization)
    if len(_ROUTE_GLOBAL_CACHE) >= _ROUTE_GLOBAL_CACHE_MAX:
        _ROUTE_GLOBAL_CACHE.pop(next(iter(_ROUTE_GLOBAL_CACHE)))
    _ROUTE_GLOBAL_CACHE[key] = arr


def _traffic_has_red_light(traffic_lights: list[dict[str, Any]]) -> bool:
    return any("red" in str(t.get("status", "")).lower() for t in traffic_lights)


def _red_lane_connector_ids(traffic_lights: list[dict[str, Any]]) -> set[str]:
    return {str(t.get("lane_connector_id", "")) for t in traffic_lights if "red" in str(t.get("status", "")).lower()}


def _stop_line_cache_key(map_api: Any, current_state: np.ndarray, radius: float, tile_m: float) -> tuple[int, int, int, int]:
    tile = max(float(tile_m), 1.0)
    return (
        id(map_api),
        int(np.floor(float(current_state[0]) / tile)),
        int(np.floor(float(current_state[1]) / tile)),
        int(round(float(radius))),
    )


def _store_stop_lines_global(key: tuple[int, int, int, int], records: list[dict[str, Any]]) -> None:
    if len(_STOP_LINE_GLOBAL_CACHE) >= _STOP_LINE_GLOBAL_CACHE_MAX:
        _STOP_LINE_GLOBAL_CACHE.pop(next(iter(_STOP_LINE_GLOBAL_CACHE)))
    _STOP_LINE_GLOBAL_CACHE[key] = records


def _stop_line_records_to_local(records: list[dict[str, Any]], current_state: np.ndarray, red_ids: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        xy_global = np.asarray(rec.get("xy_global", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
        if len(xy_global) < 2:
            continue
        obj_id = str(rec.get("id", ""))
        conn_ids = tuple(str(x) for x in rec.get("conn_ids", ()))
        is_red = bool(red_ids and (obj_id in red_ids or any(cid in red_ids for cid in conn_ids)))
        out.append({"xy": _to_local_xy(xy_global, current_state), "red": is_red, "status": "red" if is_red else "unknown", "id": obj_id})
    return out


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




def _object_token(obj: Any, fallback_index: int) -> str:
    for attr in ("track_token", "token", "id", "track_id"):
        val = getattr(obj, attr, None)
        if val not in (None, ""):
            return str(val)
    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("track_token", "token", "id", "track_id"):
            val = metadata.get(key)
            if val not in (None, ""):
                return str(val)
    return f"idx:{int(fallback_index)}"


def _boxes_global_to_local(boxes: np.ndarray, current_state: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 10)
    if len(boxes) == 0:
        return boxes.copy()
    out = boxes.copy()
    local_pose = transform_states_to_local(boxes[:, :5], current_state[:2], float(current_state[2]))
    out[:, :5] = local_pose[:, :5]
    c = float(np.cos(-float(current_state[2])))
    s = float(np.sin(-float(current_state[2])))
    vx = boxes[:, 5].copy()
    vy = boxes[:, 6].copy()
    out[:, 5] = c * vx - s * vy
    out[:, 6] = s * vx + c * vy
    out[:, 3] = np.hypot(out[:, 5], out[:, 6]).astype(np.float32)
    return out.astype(np.float32)


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


@lru_cache(maxsize=None)
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


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Read potentially lazy nuPlan map attributes without crashing runtime feature construction."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _map_object_id(obj: Any) -> str:
    for attr in ("id", "token", "fid"):
        val = _safe_getattr(obj, attr, None)
        if val not in (None, ""):
            return str(val)
    return f"pyobj:{id(obj)}"


def _map_edge_roadblock_id(edge: Any) -> str:
    getter = _safe_getattr(edge, "get_roadblock_id", None)
    if callable(getter):
        try:
            rid = getter()
            if rid not in (None, ""):
                return str(rid)
        except Exception:
            pass
    parent = _safe_getattr(edge, "parent", None)
    if parent is not None:
        pid = _map_object_id(parent)
        if not pid.startswith("pyobj:"):
            return pid
    return _map_object_id(edge)


def _safe_iter_map_objects(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return []
    try:
        return list(value)
    except Exception:
        return []


def _edge_baseline_polyline(edge: Any) -> np.ndarray:
    for attr in ("baseline_path", "path"):
        val = _safe_getattr(edge, attr, None)
        if val is None:
            continue
        arr = _xy_array_from_geometry(val)
        if len(arr) >= 2:
            return np.asarray(arr, dtype=np.float32).reshape(-1, 2)
    arr = _xy_array_from_geometry(edge)
    if len(arr) >= 2:
        return np.asarray(arr, dtype=np.float32).reshape(-1, 2)
    return np.zeros((0, 2), dtype=np.float32)


def _object_interior_edges(obj: Any) -> list[Any]:
    """Return only container interior lane edges, not graph-neighbor edges.

    ``incoming_edges`` and ``outgoing_edges`` are nuPlan lane-graph links. They
    are intentionally excluded here because the road/lane graph is cyclic; a
    recursive traversal through those attributes can loop indefinitely.
    """
    edges = _safe_iter_map_objects(_safe_getattr(obj, "interior_edges", None))
    return [edge for edge in edges if edge is not None]


def _object_baseline_polylines(obj: Any) -> list[np.ndarray]:
    """Extract baseline polylines from a nuPlan map object without graph recursion."""
    out: list[np.ndarray] = []
    direct = _edge_baseline_polyline(obj)
    if len(direct) >= 2:
        out.append(direct)
    for edge in _object_interior_edges(obj):
        arr = _edge_baseline_polyline(edge)
        if len(arr) >= 2:
            out.append(arr)
    return out


def _nearest_polyline_distance(polyline: np.ndarray, point_xy: np.ndarray) -> float:
    pts = np.asarray(polyline, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(pts - point_xy.reshape(1, 2), axis=1)))


def _edge_start_distance(edge: Any, point_xy: np.ndarray) -> float:
    arr = _edge_baseline_polyline(edge)
    if len(arr) < 2:
        return float("inf")
    return float(np.linalg.norm(arr[0] - point_xy.reshape(2)))


def _select_initial_route_edge(edges: list[Any], current_state: np.ndarray | None) -> Any | None:
    if not edges:
        return None
    if current_state is None:
        return edges[0]
    ego_xy = np.asarray(current_state[:2], dtype=np.float32).reshape(2)
    return min(edges, key=lambda e: _nearest_polyline_distance(_edge_baseline_polyline(e), ego_xy))


def _choose_next_route_edge(prev_edge: Any | None, candidates: list[Any], prev_endpoint: np.ndarray | None) -> Any | None:
    if not candidates:
        return None
    if prev_edge is not None:
        outgoing_ids = {_map_object_id(edge) for edge in _safe_iter_map_objects(_safe_getattr(prev_edge, "outgoing_edges", None))}
        connected = [edge for edge in candidates if _map_object_id(edge) in outgoing_ids]
        if connected:
            candidates = connected
    if prev_endpoint is None:
        return candidates[0]
    return min(candidates, key=lambda e: _edge_start_distance(e, prev_endpoint))


def _route_edges_by_roadblock(initialization: Any) -> list[tuple[str, list[Any]]]:
    route_ids = list(_safe_getattr(initialization, "route_roadblock_ids", []) or []) if initialization is not None else []
    map_api = _safe_getattr(initialization, "map_api", None) if initialization is not None else None
    grouped: list[tuple[str, list[Any]]] = []
    for rid in route_ids:
        obj = _get_map_object_any_layer(map_api, str(rid))
        if obj is None:
            continue
        edges = _object_interior_edges(obj)
        if not edges:
            edges = [obj]
        # Drop objects that do not expose a usable baseline.  This avoids passing
        # lazy map objects downstream where a baseline lookup can fail on a single
        # malformed row and stop the whole simulation.
        usable = [edge for edge in edges if len(_edge_baseline_polyline(edge)) >= 2]
        if usable:
            grouped.append((str(rid), usable))
    return grouped


def _stitch_route_edge_sequence(edge_groups: list[tuple[str, list[Any]]], current_state: np.ndarray | None) -> list[np.ndarray]:
    pieces: list[np.ndarray] = []
    prev_edge: Any | None = None
    prev_endpoint: np.ndarray | None = None
    for _rid, candidates in edge_groups:
        if prev_edge is None:
            edge = _select_initial_route_edge(candidates, current_state)
        else:
            edge = _choose_next_route_edge(prev_edge, candidates, prev_endpoint)
        if edge is None:
            continue
        arr = _edge_baseline_polyline(edge)
        if len(arr) < 2:
            continue
        pieces.append(arr)
        prev_edge = edge
        prev_endpoint = arr[-1]
    return pieces


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
    # Route IDs supplied by nuPlan initialization are roadblock or
    # roadblock-connector IDs.  Keep lane layers as a fallback for tests/custom
    # data, but try container layers first to preserve route semantics.
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
    max_pts = int(cfg.get("runtime", {}).get("max_route_points", 512))

    # Route roadblock IDs and map baselines are static for a nuPlan scenario, but
    # this adapter is called at every closed-loop simulation tick.  Avoid repeated
    # map_api.get_map_object / baseline extraction; only redo the cheap global->local
    # transform for the current ego pose.
    cached_global = _cached_route_global(initialization)
    if cached_global is not None:
        return _to_local_xy(cached_global, current_state)[:max_pts]

    edge_groups = _route_edges_by_roadblock(initialization)
    pieces = _stitch_route_edge_sequence(edge_groups, current_state)
    if pieces:
        route_global: list[np.ndarray] = []
        for pts in pieces:
            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 2:
                continue
            if route_global and np.linalg.norm(route_global[-1] - pts[0]) < 1e-3:
                route_global.extend(list(pts[1:]))
            else:
                route_global.extend(list(pts))
        if len(route_global) >= 2:
            route_global_arr = np.asarray(route_global, dtype=np.float32)
            _store_route_global(initialization, route_global_arr)
            route_local = _to_local_xy(route_global_arr, current_state)
            return route_local[:max_pts]
    return make_default_route_centerline(float(cfg.get("runtime", {}).get("route_horizon_m", 160.0)))


def _stop_lines_from_map_api(initialization: Any, current_state: np.ndarray | None, traffic_lights: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    map_api = _safe_getattr(initialization, "map_api", None) if initialization is not None else None
    if map_api is None or current_state is None:
        return []

    runtime_cfg = cfg.get("runtime", {})
    # Most ticks have no red-light constraint. In that case querying nearby
    # STOP_LINE objects does not change BDSE runtime evidence, but it can
    # dominate closed-loop latency in nuPlan. Set this false for visualization.
    if bool(runtime_cfg.get("skip_stop_line_query_without_red_light", True)) and not _traffic_has_red_light(traffic_lights):
        return []

    layer = _semantic_map_layer("STOP_LINE")
    if layer is None:
        return []
    radius = float(runtime_cfg.get("map_radius_m", 100.0))
    tile_m = float(runtime_cfg.get("stop_line_cache_tile_m", 25.0))
    key = _stop_line_cache_key(map_api, current_state, radius, tile_m)
    red_ids = _red_lane_connector_ids(traffic_lights)
    cached = _STOP_LINE_GLOBAL_CACHE.get(key)
    if cached is not None:
        return _stop_line_records_to_local(cached, current_state, red_ids)

    try:
        point_mod = __import__("nuplan.common.actor_state.state_representation", fromlist=["Point2D"])
        Point2D = getattr(point_mod, "Point2D")
        center = Point2D(float(current_state[0]), float(current_state[1]))
    except Exception:
        center = current_state[:2]
    try:
        prox = map_api.get_proximal_map_objects(center, radius, [layer])
        objs = prox.get(layer, []) if isinstance(prox, dict) else []
    except Exception:
        objs = []

    records: list[dict[str, Any]] = []
    for obj in list(objs):
        xy = _xy_array_from_geometry(obj)
        if len(xy) < 2:
            continue
        obj_id = str(getattr(obj, "id", getattr(obj, "token", "")))
        conn_ids: list[str] = []
        for attr in ("lane_connector_id", "lane_connector_ids", "lane_connector_fid", "lane_connector_fids"):
            val = getattr(obj, attr, None)
            if val is None:
                continue
            if isinstance(val, (list, tuple, set)):
                conn_ids.extend(str(v) for v in val)
            else:
                conn_ids.append(str(val))
        records.append({"xy_global": np.asarray(xy, dtype=np.float32).reshape(-1, 2), "id": obj_id, "conn_ids": tuple(conn_ids)})
    _store_stop_lines_global(key, records)
    return _stop_line_records_to_local(records, current_state, red_ids)

def _map_features_from_initialization(initialization: Any, cfg: dict[str, Any], current_state: np.ndarray | None = None, traffic_lights: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    # Runtime-safe map context from nuPlan initialization.  This must not read
    # offline scenario labels, but it should use the route and map API available
    # to the deployed planner; otherwise closed-loop would plan on a synthetic
    # straight route with no stop-line geometry.
    traffic_lights = list(traffic_lights or [])
    route_ids = list(_safe_getattr(initialization, "route_roadblock_ids", []) or []) if initialization is not None else []
    goal = _safe_getattr(initialization, "mission_goal", None) if initialization is not None else None
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
    radius = float(cfg.get("runtime", {}).get("agent_radius_m", 80.0))
    obs_tail = observations[-h_steps:] if observations else []
    current_objs = _iter_objects(obs_tail[-1]) if obs_tail else []
    raw_current_global = np.asarray([_object_to_array(o) for o in current_objs], dtype=np.float32) if current_objs else np.zeros((0, 10), dtype=np.float32)
    raw_current = _boxes_global_to_local(raw_current_global, current) if len(raw_current_global) else np.zeros((0, 10), dtype=np.float32)
    raw_tokens = [_object_token(o, i) for i, o in enumerate(current_objs)]
    raw_hist = np.zeros((len(raw_current), h_steps, 10), dtype=np.float32)
    if len(raw_current):
        raw_hist[:, -1, :] = raw_current
        token_to_idx = {tok: i for i, tok in enumerate(raw_tokens)}
        frames = obs_tail[-h_steps:]
        start = h_steps - len(frames)
        for fi, obs in enumerate(frames):
            objs = _iter_objects(obs)
            if not objs:
                continue
            boxes_global = np.asarray([_object_to_array(o) for o in objs], dtype=np.float32)
            boxes_local = _boxes_global_to_local(boxes_global, current)
            for j, obj in enumerate(objs):
                idx = token_to_idx.get(_object_token(obj, j))
                if idx is not None and j < len(boxes_local):
                    raw_hist[idx, start + fi, :] = boxes_local[j]
    agent_order = _agent_selection_order(raw_current, np.zeros((2,), dtype=np.float32), max_agents, radius, candidate_trajectories=None)
    agent_history, current_agents, agent_valid = select_agents_deterministic(
        raw_current, raw_hist, np.zeros((2,), dtype=np.float32), max_agents, radius, candidate_trajectories=None
    )
    traffic = _traffic_lights_from_input(current_input)
    map_features = _map_features_from_initialization(initialization, cfg, current_state=current, traffic_lights=traffic)
    route_ids = list(_safe_getattr(initialization, "route_roadblock_ids", []) or []) if initialization is not None else []
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
    runtime.metadata["selected_agent_tokens"] = [raw_tokens[i] for i in agent_order if i < len(raw_tokens)]
    runtime.metadata["raw_agent_count"] = int(len(raw_current))
    runtime.metadata["agent_history_mode"] = "planner_input_track_token_match"
    if bool(cfg.get("preprocess", {}).get("candidate_aware_agent_selection", False)):
        runtime.metadata["_raw_agent_tokens"] = list(raw_tokens)
        runtime.metadata["_raw_current_agents"] = raw_current.astype(np.float32)
        runtime.metadata["_raw_agent_history"] = raw_hist.astype(np.float32)
    return runtime
