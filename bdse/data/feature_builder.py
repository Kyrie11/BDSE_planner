from __future__ import annotations

import hashlib
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
    token_hash = int(hashlib.blake2b(str(token).encode("utf-8"), digest_size=4).hexdigest(), 16) / float(0xFFFFFFFF)
    return np.asarray([x, y, yaw, math.hypot(vx, vy), 0.0, vx, vy, length, width, token_hash], dtype=np.float32)


def _object_token(obj: Any, fallback: int | str = "") -> str:
    for name in ["track_token", "token", "tracked_object_id"]:
        val = getattr(obj, name, None)
        if val is not None and str(val) != "":
            return str(val)
    return str(fallback)


def _iter_tracked_objects(container: Any) -> list[Any]:
    if container is None:
        return []
    if isinstance(container, (list, tuple)):
        return list(container)
    tracked = getattr(container, "tracked_objects", None)
    if tracked is not None:
        # nuPlan's DetectionsTracks.tracked_objects is usually a TrackedObjects
        # container.  Prefer its concrete .tracked_objects list.  The previous
        # implementation called get_tracked_objects_of_types([]) when the
        # container did not expose tracked_object_types, which returns an empty
        # list and silently removes every surrounding agent from preprocessing.
        concrete = getattr(tracked, "tracked_objects", None)
        if concrete is not None:
            try:
                return list(concrete)
            except TypeError:
                pass
        types = getattr(tracked, "tracked_object_types", None)
        if hasattr(tracked, "get_tracked_objects_of_types") and types:
            try:
                return list(tracked.get_tracked_objects_of_types(list(types)))
            except Exception:
                pass
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


def _proximal_objects(map_api: Any, center_global: np.ndarray, radius_m: float, layer_names: list[str] | None = None) -> dict[Any, list[Any]]:
    layer_names = layer_names or ["LANE", "LANE_CONNECTOR", "STOP_LINE", "CROSSWALK", "INTERSECTION", "DRIVABLE_AREA",
                                  "ROADBLOCK", "ROADBLOCK_CONNECTOR"]
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


def _dedup_polyline(polyline: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    arr = np.asarray(polyline, dtype=np.float32).reshape(-1, 2)
    if len(arr) < 2:
        return arr
    keep = [0]
    for i in range(1, len(arr)):
        if np.linalg.norm(arr[i] - arr[keep[-1]]) > eps:
            keep.append(i)
    return arr[keep]


def _polyline_heading(polyline: np.ndarray, at_end: bool = False) -> float:
    arr = _dedup_polyline(polyline)
    if len(arr) < 2:
        return 0.0
    if at_end:
        d = arr[-1] - arr[-2]
    else:
        d = arr[1] - arr[0]
    return float(math.atan2(float(d[1]), float(d[0])))


def _project_origin_tail(polyline: np.ndarray) -> np.ndarray:
    """Trim one lane/connector baseline so s=0 is the ego projection."""
    arr = _dedup_polyline(polyline)
    if len(arr) < 2:
        return arr
    origin = np.zeros(2, dtype=np.float32)
    best_i, best_t, best_score = 0, 0.0, float("inf")
    best_proj = arr[0].copy()
    for i, (a, b) in enumerate(zip(arr[:-1], arr[1:])):
        ab = b - a
        denom = float(ab @ ab)
        if denom <= 1e-9:
            continue
        t = float(np.clip(((origin - a) @ ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        d = float(np.linalg.norm(proj - origin))
        heading = math.atan2(float(ab[1]), float(ab[0]))
        heading_penalty = 2.0 * abs(float(angle_wrap(heading)))
        behind_penalty = 8.0 if max(float(a[0]), float(b[0])) < -2.0 else 0.0
        score = d + heading_penalty + behind_penalty
        if score < best_score:
            best_i, best_t, best_score = i, t, score
            best_proj = proj.astype(np.float32)
    return _dedup_polyline(np.asarray([best_proj, *arr[best_i + 1 :]], dtype=np.float32))


def _route_length(polyline: np.ndarray) -> float:
    arr = np.asarray(polyline, dtype=np.float32).reshape(-1, 2)
    if len(arr) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def _resample_polyline_by_arclength(polyline: np.ndarray, step_m: float = 2.0, max_length_m: float | None = None) -> np.ndarray:
    arr = _dedup_polyline(polyline)
    if len(arr) < 2:
        return arr
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    total = float(s[-1])
    if total <= 1e-6:
        return arr[:1]
    end = total if max_length_m is None else min(total, max(float(max_length_m), 0.0))
    step = max(float(step_m), 0.25)
    query = np.arange(0.0, end + 0.5 * step, step, dtype=np.float32)
    if query.size == 0 or float(query[-1]) < end:
        query = np.concatenate([query, np.asarray([end], dtype=np.float32)])
    x = np.interp(query, s, arr[:, 0]).astype(np.float32)
    y = np.interp(query, s, arr[:, 1]).astype(np.float32)
    return _dedup_polyline(np.stack([x, y], axis=1))


def _sanitize_route_for_runtime(polyline: np.ndarray, runtime_cfg: dict[str, Any], max_gap_m: float, min_length_m: float = 40.0) -> tuple[np.ndarray, dict[str, float]]:
    """Return a short, dense, gap-free local planning route.

    The extracted nuPlan roadblock route can be hundreds of metres long and may
    contain city-dependent connector discontinuities.  Candidate generation only
    needs the next few seconds, so truncate at the first large discontinuity and
    resample by arclength.  This prevents a single bad far-away roadblock from
    introducing route jumps/off-drivable labels into the local candidate set.
    """
    arr = _dedup_polyline(polyline)
    diag: dict[str, float] = {"route_sanitized": 0.0, "route_truncated_at_gap": 0.0}
    if len(arr) < 2:
        out = make_default_route_centerline()
        diag.update(_route_geometry_stats(out))
        diag["route_sanitized"] = 1.0
        return out, diag

    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    bad = np.flatnonzero(seg > max(float(max_gap_m), 1.0))
    if bad.size:
        # Keep the component starting at the ego projection.  If it is too short,
        # the guard below uses a straight local route instead of joining across the gap.
        arr = arr[: int(bad[0]) + 1]
        diag["route_truncated_at_gap"] = 1.0

    horizon_m = float(runtime_cfg.get("route_horizon_m", 220.0))
    step_m = float(runtime_cfg.get("route_resample_step_m", 2.0))
    arr = _resample_polyline_by_arclength(arr, step_m=step_m, max_length_m=horizon_m)
    if len(arr) < 2 or _route_length(arr) < min_length_m or float(np.nanmax(arr[:, 0])) < 5.0:
        arr = make_default_route_centerline(horizon_m=min(horizon_m, 160.0), step_m=step_m)
        diag["route_sanitized"] = 1.0
    diag.update(_route_geometry_stats(arr))
    return arr.astype(np.float32), diag


def _route_geometry_stats(polyline: np.ndarray) -> dict[str, float]:
    arr = np.asarray(polyline, dtype=np.float32).reshape(-1, 2)
    if len(arr) < 2:
        return {"route_length_m": 0.0, "route_max_segment_m": 0.0, "route_jump_count": 0.0, "route_backtrack_frac": 0.0}
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    dx = np.diff(arr[:, 0])
    return {
        "route_length_m": float(seg.sum()),
        "route_max_segment_m": float(seg.max(initial=0.0)),
        "route_jump_count": float((seg > 10.0).sum()),
        "route_backtrack_frac": float((dx < -0.5).mean()) if dx.size else 0.0,
    }


def _connect_candidate(prev_end: np.ndarray, prev_heading: float, polyline: np.ndarray, reverse_penalty: float = 5.0) -> tuple[float, np.ndarray, float, float]:
    arr = _dedup_polyline(polyline)
    best = (float("inf"), arr, 1e6, 0.0)
    for rev in (False, True):
        cand = arr[::-1].copy() if rev else arr
        if len(cand) < 2:
            continue
        gap = float(np.linalg.norm(cand[0] - prev_end))
        hdg = _polyline_heading(cand, at_end=False)
        hdg_err = abs(float(angle_wrap(hdg - prev_heading)))
        score = gap + 3.0 * hdg_err + (reverse_penalty if rev else 0.0)
        if score < best[0]:
            best = (score, cand, gap, hdg_err)
    return best


def _build_continuous_route_from_groups(route_groups: list[list[np.ndarray]], max_gap_m: float = 12.0, min_length_m: float = 40.0) -> tuple[np.ndarray, dict[str, float]]:
    """Pick one continuous lane/connector chain instead of concatenating all lanes.

    A nuPlan route roadblock contains multiple candidate lanes/connectors.  Concatenating
    every edge by local x creates artificial jumps and self-crossing centerlines, which then
    corrupts candidate rollouts and teacher hard labels.  This helper chooses a single
    ego-aligned lane in the current route group and greedily connects one compatible edge
    from subsequent route groups.
    """
    groups = [[_dedup_polyline(p) for p in g if np.asarray(p).ndim == 2 and len(p) >= 2] for g in route_groups]
    groups = [g for g in groups if g]
    if not groups:
        return make_default_route_centerline(), {"route_selected_groups": 0.0, "route_connection_gap_max": 0.0}

    starts: list[tuple[float, int, np.ndarray]] = []
    origin = np.zeros(2, dtype=np.float32)
    for gi, group in enumerate(groups):
        for p in group:
            tail = _project_origin_tail(p)
            if len(tail) < 2:
                continue
            dist = float(np.linalg.norm(tail[0] - origin))
            heading = _polyline_heading(tail, at_end=False)
            heading_penalty = 3.0 * abs(float(angle_wrap(heading)))
            behind_penalty = 10.0 if float(np.nanmax(tail[:, 0])) < 5.0 else 0.0
            starts.append((dist + heading_penalty + behind_penalty, gi, tail))
    if not starts:
        return make_default_route_centerline(), {"route_selected_groups": 0.0, "route_connection_gap_max": 0.0}

    _, start_gi, start = min(starts, key=lambda x: x[0])
    pieces = [start]
    prev_end = start[-1]
    prev_heading = _polyline_heading(start, at_end=True)
    gaps: list[float] = []
    selected_groups = 1
    for group in groups[start_gi + 1 :]:
        candidates = [_connect_candidate(prev_end, prev_heading, p) for p in group]
        candidates = [c for c in candidates if np.isfinite(c[0])]
        if not candidates:
            continue
        _, best_poly, gap, hdg_err = min(candidates, key=lambda x: x[0])
        # Do not join disconnected roadblocks with a straight chord.  The previous
        # version allowed a >max_gap join when the current piece was short, which
        # created rare 50m+ route jumps and noisy route-conditioned rollouts.  Skip
        # that route group instead; if the chain remains too short, the final guard
        # falls back to a straight ego-local route rather than inventing a shortcut.
        if gap > max_gap_m or hdg_err > np.deg2rad(120.0):
            if _route_length(np.concatenate(pieces, axis=0)) >= min_length_m:
                break
            continue
        pieces.append(best_poly[1:] if np.linalg.norm(best_poly[0] - prev_end) < 1e-3 else best_poly)
        prev_end = best_poly[-1]
        prev_heading = _polyline_heading(best_poly, at_end=True)
        gaps.append(gap)
        selected_groups += 1

    merged = _dedup_polyline(np.concatenate(pieces, axis=0))
    if len(merged) < 2 or _route_length(merged) < min_length_m or float(np.nanmax(merged[:, 0])) < 5.0:
        # Last resort: use a straight route rather than a disconnected polyline that
        # would create impossible candidates and noisy hard events.
        merged = make_default_route_centerline()
    diag = {"route_selected_groups": float(selected_groups), "route_connection_gap_max": float(max(gaps) if gaps else 0.0)}
    diag.update(_route_geometry_stats(merged))
    return merged.astype(np.float32), diag


def _concat_route_polylines(polys_local: list[np.ndarray]) -> np.ndarray:
    # Backwards-compatible wrapper for tests/synthetic callers.
    route, _ = _build_continuous_route_from_groups([[p] for p in polys_local])
    return route


def _trim_route_from_ego_projection(polyline: np.ndarray, min_length_m: float = 40.0) -> np.ndarray:
    """Return a local route whose arclength origin is the ego projection.

    nuPlan route roadblock ids usually describe the whole route, not just the
    current local segment.  If candidate rollout starts at the first route point,
    trajectories can begin hundreds of metres away from the ego.  Project the
    local ego origin (0, 0) onto the closest route segment and keep the route tail
    from that point onward.  The trajectory sampler then interprets s=0 as the
    current ego progress along the route, which is the intended planner frame.
    """
    arr = np.asarray(polyline, dtype=np.float32).reshape(-1, 2)
    if len(arr) < 2:
        return arr
    origin = np.zeros(2, dtype=np.float32)
    best_i = 0
    best_t = 0.0
    best_d = float("inf")
    best_proj = arr[0].copy()
    for i, (a, b) in enumerate(zip(arr[:-1], arr[1:])):
        ab = b - a
        denom = float(ab @ ab)
        if denom <= 1e-9:
            continue
        t = float(np.clip(((origin - a) @ ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        d = float(np.linalg.norm(proj - origin))
        # Prefer segments that are not entirely behind ego in local x when close.
        behind_penalty = 5.0 if max(float(a[0]), float(b[0])) < -2.0 else 0.0
        score = d + behind_penalty
        if score < best_d:
            best_d = score
            best_i = i
            best_t = t
            best_proj = proj.astype(np.float32)
    tail = [best_proj]
    tail.extend(arr[best_i + 1 :])
    out = np.asarray(tail, dtype=np.float32)
    if len(out) >= 2:
        keep = [0]
        for i in range(1, len(out)):
            if np.linalg.norm(out[i] - out[keep[-1]]) > 1e-3:
                keep.append(i)
        out = out[keep]
    # If the tail is too short or points mostly behind ego, keep a safe straight
    # fallback rather than constructing candidates on a degenerate route fragment.
    if len(out) < 2 or _route_length(out) < min_length_m or float(np.nanmax(out[:, 0])) < 5.0:
        return make_default_route_centerline()
    return _simplify_polyline(out, 512)


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

def _simplify_polyline(points: np.ndarray, max_points: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if max_points <= 0 or len(arr) <= max_points:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_points).round().astype(np.int64)
    return arr[idx]

def extract_map_features_from_api(map_api: Any, ego_state_global: np.ndarray, radius_m: float, route_ids: list[str],
                                  traffic_lights: list[dict[str, Any]] | None = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
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
        "route_quality": {},
    }
    if map_api is None:
        return features

    runtime_cfg = (cfg or {}).get("runtime", {})
    include_drivable = bool(runtime_cfg.get("include_drivable_polygons", False))
    include_crosswalks = bool(runtime_cfg.get("include_crosswalks", False))
    max_drivable = int(runtime_cfg.get("max_drivable_polygons", 12))
    max_poly_points = int(runtime_cfg.get("max_polygon_points", 64))
    max_route_points = int(runtime_cfg.get("max_route_points", 256))

    red_connector_ids = {str(t.get("lane_connector_id", "")) for t in (traffic_lights or []) if "red" in str(t.get("status", "")).lower() and str(t.get("lane_connector_id", ""))}
    features["red_lane_connector_ids"] = sorted(red_connector_ids)
    features["red_lane_connectors"] = []

    # Avoid a full 100m proximal map scan for every sample when the scenario already
    # provides route roadblock ids.  Full DRIVABLE_AREA polygons are especially slow
    # on nuPlan and are disabled by default; the hard off-road atom falls back to a
    # route-corridor approximation unless --include-drivable-polygons is requested.
    prox_layers: list[str] = []
    if not route_ids:
        prox_layers += ["LANE", "LANE_CONNECTOR"]
    if red_connector_ids:
        prox_layers += ["STOP_LINE"]
    if include_drivable:
        prox_layers += ["DRIVABLE_AREA"]
    if include_crosswalks:
        prox_layers += ["CROSSWALK"]
    prox = _proximal_objects(map_api, origin_xy, radius_m, sorted(set(prox_layers))) if prox_layers else {}

    for cid in sorted(red_connector_ids):
        conn = _get_map_object(map_api, cid, ["LANE_CONNECTOR"])
        pts = _baseline_points(conn) if conn is not None else np.zeros((0, 2), dtype=np.float32)
        if len(pts) >= 2:
            conn_xy = _to_local_xy(pts, origin_xy, origin_yaw)
            features["red_lane_connectors"].append({"id": cid, "xy": conn_xy})
            # If STOP_LINE objects are unavailable in this nuPlan/devkit build, keep
            # a runtime-only proxy in map_features.  This makes diagnostics honest
            # (stop_line_count no longer stays at zero) and gives the red-light atom
            # a stable anchor without using any future labels.
            if len(conn_xy) >= 2:
                p0, p1 = conn_xy[0], conn_xy[1]
                heading = p1 - p0
                norm = np.linalg.norm(heading)
                if norm > 1e-6:
                    normal = np.asarray([-heading[1], heading[0]], dtype=np.float32) / float(norm)
                    proxy = np.stack([p0 - 2.0 * normal, p0 + 2.0 * normal], axis=0).astype(np.float32)
                    features["stop_lines"].append({"id": f"{cid}:proxy_stop", "xy": proxy, "red": True, "proxy": True})

    route_groups: list[list[np.ndarray]] = []
    route_objs: list[Any] = []
    for rid in route_ids:
        obj = _get_map_object(map_api, str(rid), ["ROADBLOCK", "ROADBLOCK_CONNECTOR", "LANE", "LANE_CONNECTOR"])
        if obj is None:
            continue
        route_objs.append(obj)
        group: list[np.ndarray] = []
        edges = _extract_edges(obj) or [obj]
        for edge in edges:
            pts = _baseline_points(edge)
            if len(pts) >= 2:
                group.append(_to_local_xy(pts, origin_xy, origin_yaw))
            v = _speed_limit_from_obj(edge)
            if v is not None:
                features["speed_limits"].append({"id": _obj_id(edge), "speed_limit_mps": v})
        if group:
            route_groups.append(group)
    if not route_groups:
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
        # Proximal fallback is a single group: choose one ego-aligned lane, do not
        # concatenate all nearby lanes into a fake route.
        group = []
        for _, local, obj in sorted(scored, key=lambda x: x[0])[:8]:
            group.append(local)
            route_objs.append(obj)
            v = _speed_limit_from_obj(obj)
            if v is not None:
                features["speed_limits"].append({"id": _obj_id(obj), "speed_limit_mps": v})
        if group:
            route_groups.append(group)
    if route_groups:
        max_gap = float(runtime_cfg.get("max_route_connection_gap_m", 12.0))
        route, route_diag = _build_continuous_route_from_groups(route_groups, max_gap_m=max_gap)
        route = _trim_route_from_ego_projection(route)
        route, sanitize_diag = _sanitize_route_for_runtime(route, runtime_cfg, max_gap_m=max_gap)
        route_diag.update(sanitize_diag)
        route = _simplify_polyline(route, max_route_points)
        route_diag.update(_route_geometry_stats(route))
        features["route_centerline"] = route
        features["route_quality"] = route_diag
        features["route_source"] = "route_roadblocks_continuous" if route_ids else "proximal_lane_single"
        features["map_valid"] = bool(route_diag.get("route_jump_count", 0.0) == 0.0 and route_diag.get("route_max_segment_m", 0.0) <= max_gap)

    if features["speed_limits"]:
        features["speed_limit_mps"] = float(np.median([x["speed_limit_mps"] for x in features["speed_limits"]]))

    if include_drivable:
        scored_polys: list[tuple[float, Any, np.ndarray]] = []
        for obj in _flatten_by_layer(prox, "DRIVABLE_AREA"):
            pts = _geometry_points(obj)
            if len(pts) >= 3:
                local = _to_local_xy(pts, origin_xy, origin_yaw)
                # Keep only polygons that are near the ego/candidate corridor; far polygons
                # do not change the teacher label but dominate Python geometry cost.
                score = float(np.linalg.norm(local, axis=1).min())
                scored_polys.append((score, obj, local))

        # Some nuPlan map/devkit combinations do not expose a populated
        # DRIVABLE_AREA layer through get_proximal_map_objects, while lane and
        # lane-connector objects still carry polygons.  Falling back to those
        # route/local lane polygons prevents the paper-faithful configuration
        # from degenerating into a route-corridor-only off-drivable atom.
        if not scored_polys:
            fallback_objs: list[Any] = []
            for obj in route_objs:
                fallback_objs.extend(_extract_edges(obj) or [obj])
            fallback_objs.extend(_flatten_by_layer(prox, "LANE"))
            fallback_objs.extend(_flatten_by_layer(prox, "LANE_CONNECTOR"))
            seen: set[str] = set()
            for obj in fallback_objs:
                oid = _obj_id(obj)
                if oid in seen:
                    continue
                seen.add(oid)
                pts = _geometry_points(obj)
                if len(pts) >= 3:
                    local = _to_local_xy(pts, origin_xy, origin_yaw)
                    score = float(np.linalg.norm(local, axis=1).min())
                    scored_polys.append((score, obj, local))

        for _, obj, local in sorted(scored_polys, key=lambda x: x[0])[:max_drivable]:
            features["drivable_polygons"].append(
                {"id": _obj_id(obj), "xy": _simplify_polyline(local, max_poly_points)})
    red_connector_polys = [np.asarray(x.get("xy"), dtype=np.float32).reshape(-1, 2) for x in features.get("red_lane_connectors", []) if np.asarray(x.get("xy", [])).size >= 4]
    for obj in _flatten_by_layer(prox, "STOP_LINE"):
        pts = _geometry_points(obj)
        if len(pts) >= 2:
            local = _to_local_xy(pts, origin_xy, origin_yaw)
            centroid = local.mean(axis=0)
            if -20.0 <= float(centroid[0]) <= radius_m + 20.0:
                stop_id = _obj_id(obj)
                explicit_ids = set()
                for attr in ["lane_connector_ids", "lane_connectors", "traffic_light_lane_connector_ids"]:
                    val = getattr(obj, attr, None)
                    if val is None:
                        continue
                    try:
                        explicit_ids.update(str(getattr(v, "id", v)) for v in val)
                    except TypeError:
                        explicit_ids.add(str(getattr(val, "id", val)))
                is_red = bool(explicit_ids & red_connector_ids)
                if not is_red and red_connector_polys:
                    # In nuPlan maps, the stop line should be spatially near the beginning of
                    # the traffic-light-controlled lane connector.  Use this as a conservative
                    # fallback when connector membership is not exposed by the map object.
                    for conn_xy in red_connector_polys:
                        d = float(np.linalg.norm(local[:, None, :] - conn_xy[None, :, :], axis=2).min())
                        if d <= 8.0:
                            is_red = True
                            break
                features["stop_lines"].append({"id": stop_id, "xy": local, "red": is_red})
    if include_crosswalks:
        for obj in _flatten_by_layer(prox, "CROSSWALK"):
            pts = _geometry_points(obj)
            if len(pts) >= 3:
                features["crosswalks"].append({"id": _obj_id(obj),
                                               "xy": _simplify_polyline(_to_local_xy(pts, origin_xy, origin_yaw),
                                                                        max_poly_points)})
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
                        c = math.cos(-origin_yaw)
                        s = math.sin(-origin_yaw)
                        rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
                        arr[5:7] = arr[5:7] @ rot.T
                        raw_hist[token_to_idx[token], start + fi] = arr
    cand_traj = None if candidates is None else getattr(candidates, "trajectories", candidates)
    order = _agent_selection_order(raw_current, np.zeros(2, dtype=np.float32), max_agents, radius, cand_traj)
    agent_hist, current_agents, agent_valid = select_agents_deterministic(raw_current, raw_hist, np.zeros(2, dtype=np.float32), max_agents, radius, cand_traj)
    selected_tokens = [raw_tokens[i] for i in order]

    traffic_lights = _traffic_lights_to_list(_call(scenario, ["get_traffic_light_status_at_iteration"], iteration, default=[]))
    route_ids = list(_call(scenario, ["get_route_roadblock_ids", "route_roadblock_ids"], default=[]) or [])
    mission_goal_state = _call(scenario, ["get_mission_goal", "mission_goal"], default=None)
    mission_goal = None if mission_goal_state is None else transform_states_to_local(_state_to_array(mission_goal_state)[None], origin_xy, origin_yaw)[0]
    map_features = extract_map_features_from_api(getattr(scenario, "map_api", None), ego_arr_global, map_radius,
                                                 [str(r) for r in route_ids], traffic_lights, cfg)
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
