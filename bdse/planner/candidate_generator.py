from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.utils import angle_wrap, compute_curvature, finite_difference, nearest_polyline_distance, route_progress_along_polyline, sample_polyline

MANEUVER_IDS = {
    "keep_follow": 0,
    "decelerate_stop": 1,
    "yield_creep": 2,
    "lane_change_left": 3,
    "lane_change_right": 4,
    "route_turn_connector": 5,
    "safe_fallback": 6,
    "padding": -1,
}


@dataclass(frozen=True, slots=True)
class DynamicFeasibility:
    max_accel: float
    max_decel: float
    max_jerk: float
    max_curvature: float
    max_curvature_rate: float
    max_lateral_accel: float
    eval_quantile: float = 1.0
    low_speed_mps: float = 0.3


def _cfg_dyn(cfg: dict[str, Any]) -> DynamicFeasibility:
    c = cfg.get("candidate", {})
    return DynamicFeasibility(
        max_accel=float(c.get("max_accel", 3.0)),
        max_decel=float(c.get("max_decel", -5.0)),
        max_jerk=float(c.get("max_jerk", 5.0)),
        max_curvature=float(c.get("max_curvature", 0.25)),
        max_curvature_rate=float(c.get("max_curvature_rate", 0.20)),
        max_lateral_accel=float(c.get("max_lateral_accel", 3.0)),
        eval_quantile=float(c.get("dynamic_eval_quantile", 0.98)),
        low_speed_mps=float(c.get("dynamic_low_speed_mps", 0.3)),
    )


def _times(cfg: dict[str, Any]) -> np.ndarray:
    c = cfg.get("candidate", {})
    horizon = float(c.get("horizon_s", 8.0))
    dt = float(c.get("step_s", 0.1))
    return np.arange(dt, horizon + 0.5 * dt, dt, dtype=np.float32)


def _route_centerline(runtime: RuntimeFeatures, cfg: dict[str, Any]) -> np.ndarray:
    route = runtime.map_features.get("route_centerline") if runtime.map_features else None
    if route is not None:
        arr = np.asarray(route, dtype=np.float32).reshape(-1, 2)
        if arr.size:
            arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr) >= 2:
            return arr.astype(np.float32)
    x = np.linspace(0.0, 160.0, 81, dtype=np.float32)
    return np.stack([x, np.zeros_like(x)], axis=1)


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _speed_profile(v0: float, target_v: float, times: np.ndarray, stop_distance: float | None, accel_bias: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    T = float(times[-1])
    if stop_distance is not None:
        stop_distance = max(float(stop_distance), 0.1)
        a = -max(v0 * v0 / (2.0 * stop_distance), 0.1)
        v = np.maximum(v0 + a * times, 0.0)
        s = np.minimum(v0 * times + 0.5 * a * times * times, stop_distance)
        return s.astype(np.float32), v.astype(np.float32)
    target_v = max(float(target_v), 0.0)
    a = (target_v - v0) / max(T, 1e-3) + accel_bias
    v = np.maximum(v0 + a * times, 0.0)
    s = np.cumsum(v) * float(times[0])
    return s.astype(np.float32), v.astype(np.float32)


def _rollout_on_route(
    route: np.ndarray,
    v0: float,
    times: np.ndarray,
    target_speed: float,
    lateral_offset: float = 0.0,
    stop_distance: float | None = None,
    accel_bias: float = 0.0,
) -> np.ndarray:
    s, v = _speed_profile(v0, target_speed, times, stop_distance, accel_bias)
    xy, yaw = sample_polyline(route, s)
    if abs(lateral_offset) > 1e-4:
        T = float(times[-1])
        profile = _smoothstep(times / max(0.6 * T, 1e-3)) * lateral_offset
        normals = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
        xy = xy + normals * profile[:, None]
        dxy = np.gradient(xy, axis=0)
        yaw = np.arctan2(dxy[:, 1], dxy[:, 0]).astype(np.float32)
    return np.stack([xy[:, 0], xy[:, 1], yaw, v, times], axis=1).astype(np.float32)


def _rollout_straight(v0: float, times: np.ndarray, target_speed: float, stop_distance: float | None = None) -> np.ndarray:
    """Ego-frame straight fallback rollout using only runtime current speed.

    This is used as a last-resort recovery candidate when a map connector is so
    sharp/noisy that route-centered rollouts are dynamically masked.  It stays
    runtime-only and will still receive route-deviation cost from the teacher,
    but it prevents the candidate bank from degenerating to a handful of valid
    actions.
    """
    s, v = _speed_profile(v0, target_speed, times, stop_distance)
    x = s.astype(np.float32)
    y = np.zeros_like(x, dtype=np.float32)
    yaw = np.zeros_like(x, dtype=np.float32)
    return np.stack([x, y, yaw, v.astype(np.float32), times], axis=1).astype(np.float32)


def _robust_max(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if q >= 1.0 or arr.size < 4:
        return float(np.nanmax(arr))
    return float(np.nanquantile(arr, q))


def _robust_min(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if q >= 1.0 or arr.size < 4:
        return float(np.nanmin(arr))
    return float(np.nanquantile(arr, 1.0 - q))


def _dynamic_flags(traj: np.ndarray, dt: float, dyn: DynamicFeasibility) -> dict[str, Any]:
    v = traj[:, 3]
    acc = finite_difference(v, dt)
    jerk = finite_difference(acc, dt)
    curv = compute_curvature(traj[:, :2])
    curv_rate = finite_difference(curv, dt)
    lat_acc = v * v * np.abs(curv)
    q = float(np.clip(dyn.eval_quantile, 0.5, 1.0))

    # Finite-difference maxima are brittle for nuPlan centerlines: a single
    # duplicate point, map-connector kink, or stop-profile clamp can create a
    # one-frame jerk/curvature spike and collapse the valid candidate bank.  The
    # teacher should reject physically impossible rollouts, not route-map
    # discretization noise, so use high-percentile dynamic envelopes and ignore
    # steering/jerk spikes after the rollout has effectively stopped.
    moving = np.asarray(v > max(float(dyn.low_speed_mps), 0.0), dtype=bool)
    if moving.any():
        jerk_eval = jerk[moving]
        curv_eval = curv[moving]
        curv_rate_eval = curv_rate[moving]
        lat_eval = lat_acc[moving]
    else:
        jerk_eval = jerk[:1] * 0.0
        curv_eval = curv[:1] * 0.0
        curv_rate_eval = curv_rate[:1] * 0.0
        lat_eval = lat_acc[:1] * 0.0

    max_accel = _robust_max(acc, q)
    min_accel = _robust_min(acc, q)
    max_jerk = _robust_max(np.abs(jerk_eval), q)
    max_curv = _robust_max(np.abs(curv_eval), q)
    max_curv_rate = _robust_max(np.abs(curv_rate_eval), q)
    max_lat = _robust_max(lat_eval, q)
    flags = {
        "accel_ok": bool(max_accel <= dyn.max_accel + 1e-4),
        "decel_ok": bool(min_accel >= dyn.max_decel - 1e-4),
        "jerk_ok": bool(max_jerk <= dyn.max_jerk + 1e-4),
        "curvature_ok": bool(max_curv <= dyn.max_curvature + 1e-4),
        "curvature_rate_ok": bool(max_curv_rate <= dyn.max_curvature_rate + 1e-4),
        "lateral_accel_ok": bool(max_lat <= dyn.max_lateral_accel + 1e-4),
        "max_accel": float(max_accel),
        "min_accel": float(min_accel),
        "max_jerk_abs": float(max_jerk),
        "max_curvature_abs": float(max_curv),
        "max_curvature_rate_abs": float(max_curv_rate),
        "max_lateral_accel_abs": float(max_lat),
        "dynamic_eval_quantile": float(q),
    }
    flags["dynamically_feasible"] = all(
        bool(flags[k])
        for k in ["accel_ok", "decel_ok", "jerk_ok", "curvature_ok", "curvature_rate_ok", "lateral_accel_ok"]
    )
    return flags


def _append_candidate(
    trajectories: list[np.ndarray],
    valid: list[bool],
    maneuver_ids: list[int],
    theta: list[dict[str, Any]],
    flags: list[dict[str, bool]],
    metadata: list[dict[str, Any]],
    traj: np.ndarray,
    maneuver: str,
    params: dict[str, Any],
    cfg: dict[str, Any],
    force_valid: bool = True,
) -> None:
    dyn = _cfg_dyn(cfg)
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    dflags = _dynamic_flags(traj, dt, dyn)
    strict = bool(cfg.get("candidate", {}).get("valid_requires_dynamic", False))
    valid_flag = bool(force_valid and (dflags["dynamically_feasible"] or not strict))
    trajectories.append(traj.astype(np.float32))
    valid.append(valid_flag)
    maneuver_ids.append(MANEUVER_IDS[maneuver])
    theta.append(dict(params))
    flags.append(dflags)
    metadata.append({"maneuver": maneuver, "filled_by": params.get("filled_by", maneuver)})


def _legal_lane_change(runtime: RuntimeFeatures, side: str) -> bool:
    lane_change = runtime.map_features.get("lane_change", {}) if runtime.map_features else {}
    key = "left" if side == "left" else "right"
    if key in lane_change:
        return bool(lane_change[key])
    return True



def _history_motion_priors(runtime: RuntimeFeatures, times: np.ndarray, cfg: dict[str, Any]) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Runtime-only kinematic priors from the past ego state.

    These do not use logged future. They improve candidate-set coverage in cases
    where the route centerline is locally noisy, incomplete, or the vehicle is
    still aligning to the route. The teacher still decides using the same
    J_base + evidence partition.
    """
    if not bool(cfg.get("candidate", {}).get("include_history_priors", True)):
        return []
    hist = np.asarray(runtime.ego_history, dtype=np.float32)
    if hist.ndim != 2 or hist.shape[0] < 2:
        return []
    step = float(cfg.get("candidate", {}).get("step_s", 0.1))
    cur = hist[-1]
    v0 = float(max(cur[3] if cur.shape[0] > 3 else 0.0, 0.0))
    n_back = min(6, hist.shape[0])
    past = hist[-n_back:]
    dt_hist = max(step * (n_back - 1), 1e-3)
    yaw0 = float(cur[2]) if cur.shape[0] > 2 else 0.0
    yaw_prev = float(past[0, 2]) if past.shape[1] > 2 else yaw0
    yaw_rate_hist = float(angle_wrap(yaw0 - yaw_prev) / dt_hist)
    v_prev = float(max(past[0, 3] if past.shape[1] > 3 else v0, 0.0))
    accel_hist = float(np.clip((v0 - v_prev) / dt_hist, -2.0, 2.0))

    variants = [
        ("history_cv", 0.0, 0.0),
        ("history_ca", accel_hist, 0.0),
        ("history_ctrv", 0.0, float(np.clip(yaw_rate_hist, -0.35, 0.35))),
        ("history_slow", -min(max(v0 / max(float(times[-1]), 1e-3), 0.2), 2.5), float(np.clip(yaw_rate_hist, -0.25, 0.25))),
    ]
    out: list[tuple[np.ndarray, dict[str, Any]]] = []
    for name, acc, yaw_rate in variants:
        x = np.zeros((len(times),), dtype=np.float32)
        y = np.zeros((len(times),), dtype=np.float32)
        yaw = np.zeros((len(times),), dtype=np.float32)
        v = np.zeros((len(times),), dtype=np.float32)
        px = py = 0.0
        pyaw = 0.0
        pv = v0
        last_t = 0.0
        for k, t in enumerate(times):
            dt = float(t - last_t)
            last_t = float(t)
            pv = max(0.0, pv + float(acc) * dt)
            pyaw = float(angle_wrap(pyaw + float(yaw_rate) * dt))
            px += pv * np.cos(pyaw) * dt
            py += pv * np.sin(pyaw) * dt
            x[k], y[k], yaw[k], v[k] = px, py, pyaw, pv
        traj = np.stack([x, y, yaw, v, times], axis=1).astype(np.float32)
        out.append((traj, {"target_speed": float(v[-1]), "history_prior": name, "accel": float(acc), "yaw_rate": float(yaw_rate)}))
    return out


def _inject_history_priors(
    runtime: RuntimeFeatures,
    times: np.ndarray,
    trajectories: list[np.ndarray],
    valid: list[bool],
    maneuver_ids: list[int],
    theta: list[dict[str, Any]],
    flags: list[dict[str, bool]],
    metadata: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> None:
    priors = _history_motion_priors(runtime, times, cfg)
    if not priors or not trajectories:
        return
    # Keep safe_fallback entries. Replace route-turn/filler/invalid entries first.
    replace_order = [
        i for i, m in enumerate(metadata)
        if m.get("maneuver") in {"route_turn_connector", "keep_follow"} and not m.get("filled_by", "").startswith("safe")
    ]
    replace_order = replace_order[-len(priors):]
    if len(replace_order) < len(priors):
        used = set(replace_order)
        for i in range(len(trajectories)):
            if i not in used and metadata[i].get("maneuver") != "safe_fallback":
                replace_order.append(i)
            if len(replace_order) >= len(priors):
                break
    dyn = _cfg_dyn(cfg)
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    strict = bool(cfg.get("candidate", {}).get("valid_requires_dynamic", False))
    for idx, (traj, params) in zip(replace_order, priors):
        dflags = _dynamic_flags(traj, dt, dyn)
        trajectories[idx] = traj.astype(np.float32)
        valid[idx] = bool(dflags["dynamically_feasible"] or not strict)
        maneuver_ids[idx] = MANEUVER_IDS["keep_follow"]
        theta[idx] = dict(params)
        flags[idx] = dflags
        metadata[idx] = {"maneuver": "keep_follow", "filled_by": params.get("history_prior", "history_prior"), "history_prior": True}

def _unique_distances(values: list[float], min_sep_m: float = 2.5) -> list[float]:
    out: list[float] = []
    for d in sorted(float(x) for x in values if np.isfinite(x) and float(x) > 0.0):
        if all(abs(d - prev) >= min_sep_m for prev in out):
            out.append(d)
    return out


def _min_dynamic_stop_distance(v0: float, cfg: dict[str, Any], comfort_fraction: float = 0.85) -> float:
    dyn = _cfg_dyn(cfg)
    return max(2.0, float(v0 * v0 / max(2.0 * abs(dyn.max_decel) * max(comfort_fraction, 1e-3), 1e-3)))


def _candidate_stop_grid(v0: float, times: np.ndarray, cfg: dict[str, Any]) -> list[float]:
    ccfg = cfg.get("candidate", {})
    horizon_dist = max(float(v0) * float(times[-1]), 10.0)
    min_stop = _min_dynamic_stop_distance(v0, cfg)
    configured = [float(x) for x in ccfg.get("stop_distance_grid_m", [5, 10, 15, 20, 30, 40, 55, 70])]
    dynamic = [
        min_stop,
        min_stop + 5.0,
        min_stop + 12.0,
        max(min_stop + 20.0, 0.35 * horizon_dist),
        max(min_stop + 30.0, 0.55 * horizon_dist),
        max(min_stop + 45.0, 0.75 * horizon_dist),
    ]
    return _unique_distances(configured + dynamic, min_sep_m=3.0)


def _conflict_stop_distances(runtime: RuntimeFeatures, route: np.ndarray, times: np.ndarray, v0: float, cfg: dict[str, Any]) -> list[float]:
    """Runtime-only braking targets before route conflicts.

    Use current red stop-lines plus constant-velocity predictions of nearby agents.
    This remains runtime-only but covers train scenes where the blocker/crossing
    actor is not yet exactly on the route at the current frame.
    """
    ccfg = cfg.get("candidate", {})
    max_count = int(ccfg.get("conflict_stop_count", 6))
    if max_count <= 0 or len(route) < 2:
        return []
    route_width = float(runtime.map_features.get("route_corridor_width", ccfg.get("route_width_m", 4.0)))
    radius = float(ccfg.get("conflict_route_radius_m", route_width + 3.0))
    clearance = float(ccfg.get("conflict_stop_clearance_m", 8.0))
    min_stop = _min_dynamic_stop_distance(v0, cfg)
    horizon_dist = max(float(v0) * float(times[-1]) + 35.0, 50.0)
    stop_ds: list[float] = []

    # 1) Red-light stop lines are a map/runtime signal. Add a stop-before-line
    # candidate even when no currently selected agent suggests stopping.
    for sl in runtime.map_features.get("stop_lines", []):
        if not bool(sl.get("red", False)):
            continue
        xy = np.asarray(sl.get("xy", []), dtype=np.float32).reshape(-1, 2)
        if len(xy) < 2:
            continue
        center = xy.mean(axis=0, keepdims=True)
        route_d = float(nearest_polyline_distance(center, route)[0])
        if route_d > max(radius + 6.0, 12.0):
            continue
        p = float(route_progress_along_polyline(center, route)[0])
        if 2.0 < p < horizon_dist:
            stop_ds.append(max(min_stop, p - clearance))

    if not runtime.agent_valid.any():
        return _unique_distances(stop_ds)[:max_count]

    # 2) Current and constant-velocity future agent-route conflicts.
    cur = np.asarray(runtime.current_agents, dtype=np.float32)
    valid = np.asarray(runtime.agent_valid, dtype=bool)
    pred_times = times[:: max(1, int(ccfg.get("conflict_agent_time_stride", 5)))]
    for j in np.flatnonzero(valid):
        st = cur[j]
        xy0 = st[:2]
        vx = float(st[5]) if st.shape[0] > 5 else float(st[3]) * np.cos(float(st[2]))
        vy = float(st[6]) if st.shape[0] > 6 else float(st[3]) * np.sin(float(st[2]))
        speed = float(np.hypot(vx, vy))
        pts = xy0[None, :] + pred_times[:, None] * np.asarray([vx, vy], dtype=np.float32)[None, :]
        # Include the current position for static/slow obstacles.
        pts = np.concatenate([xy0[None, :], pts], axis=0).astype(np.float32)
        d_route = nearest_polyline_distance(pts, route)
        k = int(np.argmin(d_route))
        agent_len = float(st[7]) if st.shape[0] > 7 else 4.8
        agent_radius = 0.5 * agent_len + 1.0
        if float(d_route[k]) > radius + agent_radius:
            continue
        p = float(route_progress_along_polyline(pts[k : k + 1], route)[0])
        if p <= 2.0 or p > horizon_dist:
            continue
        # Ignore objects that are behind ego and moving away from the route region.
        if float(xy0[0]) < -8.0 and speed < 0.5:
            continue
        stop_ds.append(max(min_stop, p - clearance - 0.5 * agent_len))

    return _unique_distances(stop_ds)[:max_count]


def _inject_conflict_stop_priors(
    runtime: RuntimeFeatures,
    route: np.ndarray,
    times: np.ndarray,
    v0: float,
    trajectories: list[np.ndarray],
    valid: list[bool],
    maneuver_ids: list[int],
    theta: list[dict[str, Any]],
    flags: list[dict[str, bool]],
    metadata: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> None:
    stop_distances = _conflict_stop_distances(runtime, route, times, v0, cfg)
    if not stop_distances or not trajectories:
        return
    replace_order = [
        i for i, m in enumerate(metadata)
        if m.get("maneuver") in {"route_turn_connector", "keep_follow", "yield_creep"}
        and not m.get("history_prior", False)
        and m.get("maneuver") != "safe_fallback"
    ]
    # Replace low-priority/filler route-following samples first, preserving the
    # tabulated decelerate_stop and safe_fallback families.
    replace_order = replace_order[-len(stop_distances):]
    dyn = _cfg_dyn(cfg)
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    strict = bool(cfg.get("candidate", {}).get("valid_requires_dynamic", False))
    for idx, stop_d in zip(replace_order, stop_distances):
        traj = _rollout_on_route(route, v0, times, 0.0, stop_distance=float(stop_d))
        dflags = _dynamic_flags(traj, dt, dyn)
        trajectories[idx] = traj.astype(np.float32)
        valid[idx] = bool(dflags["dynamically_feasible"] or not strict)
        maneuver_ids[idx] = MANEUVER_IDS["safe_fallback"]
        theta[idx] = {"stop_distance": float(stop_d), "conflict_stop": True}
        flags[idx] = dflags
        metadata[idx] = {"maneuver": "safe_fallback", "filled_by": "conflict_stop_prior", "conflict_stop": True}


def _replace_slot_with_recovery_candidate(
    idx: int,
    traj: np.ndarray,
    maneuver: str,
    params: dict[str, Any],
    trajectories: list[np.ndarray],
    valid: list[bool],
    maneuver_ids: list[int],
    theta: list[dict[str, Any]],
    flags: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> bool:
    dyn = _cfg_dyn(cfg)
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    dflags = _dynamic_flags(traj, dt, dyn)
    ok = bool(dflags["dynamically_feasible"])
    trajectories[idx] = traj.astype(np.float32)
    valid[idx] = ok
    maneuver_ids[idx] = MANEUVER_IDS[maneuver]
    theta[idx] = dict(params)
    flags[idx] = dflags
    metadata[idx] = {"maneuver": maneuver, "filled_by": "valid_count_recovery", "recovery_candidate": True, **params}
    return ok


def _repair_low_valid_count(
    runtime: RuntimeFeatures,
    route: np.ndarray,
    times: np.ndarray,
    v0: float,
    v_ref: float,
    trajectories: list[np.ndarray],
    valid: list[bool],
    maneuver_ids: list[int],
    theta: list[dict[str, Any]],
    flags: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> None:
    """Keep the finite candidate bank from degenerating into a tiny action set.

    In final preprocessing we use strict dynamic masking, but the initial
    lattice can occasionally lose most actions because a route connector is
    numerically sharp or a braking profile has a one-frame finite-difference
    spike.  That violates the paper's candidate-set assumption: the generator
    should fill the bank with legal route-following and fallback candidates
    before masked padding is used.  This repair pass only replaces already
    invalid slots with conservative, freshly generated low-speed/stop rollouts
    that pass the same dynamic checker; it never marks a failed rollout valid.
    """
    ccfg = cfg.get("candidate", {})
    if not bool(ccfg.get("repair_low_valid_count", True)):
        return
    K = int(ccfg.get("K", 32))
    target = int(ccfg.get("min_valid_candidates", min(K, max(16, K // 3))))
    target = max(1, min(target, K))
    if sum(bool(x) for x in valid[:K]) >= target:
        return
    invalid_slots = [i for i in range(min(K, len(valid))) if not bool(valid[i])]
    if not invalid_slots:
        return

    min_stop = _min_dynamic_stop_distance(v0, cfg, comfort_fraction=0.9)
    horizon_dist = max(float(v0) * float(times[-1]), 12.0)
    candidate_specs: list[tuple[np.ndarray, str, dict[str, Any]]] = []

    for stop_d in _unique_distances(
        [
            min_stop,
            min_stop + 4.0,
            min_stop + 8.0,
            min_stop + 14.0,
            min_stop + 22.0,
            max(min_stop + 28.0, 0.35 * horizon_dist),
            max(min_stop + 40.0, 0.55 * horizon_dist),
            max(min_stop + 55.0, 0.75 * horizon_dist),
        ],
        min_sep_m=2.0,
    ):
        candidate_specs.append(
            (_rollout_on_route(route, v0, times, 0.0, stop_distance=float(stop_d)), "safe_fallback", {"stop_distance": float(stop_d)})
        )
        candidate_specs.append(
            (
                _rollout_straight(v0, times, 0.0, stop_distance=float(stop_d)),
                "safe_fallback",
                {"stop_distance": float(stop_d), "straight_fallback": True},
            )
        )

    for frac in [0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80, 0.95, 1.10]:
        target_v = max(0.2, float(frac) * float(v_ref))
        candidate_specs.append(
            (_rollout_on_route(route, v0, times, target_v), "keep_follow", {"target_speed": float(target_v), "low_speed_recovery": True})
        )
        candidate_specs.append(
            (
                _rollout_straight(v0, times, target_v),
                "keep_follow",
                {"target_speed": float(target_v), "low_speed_recovery": True, "straight_fallback": True},
            )
        )

    # If the route connector itself is the numerical source of invalidity,
    # runtime-only history priors can still provide non-degenerate low-speed
    # actions for selector training without using logged futures.
    for traj, params in _history_motion_priors(runtime, times, cfg):
        p = dict(params)
        p["history_recovery"] = True
        candidate_specs.append((traj, "keep_follow", p))

    slot_iter = iter(invalid_slots)
    for traj, maneuver, params in candidate_specs:
        if sum(bool(x) for x in valid[:K]) >= target:
            break
        try:
            idx = next(slot_iter)
        except StopIteration:
            break
        _replace_slot_with_recovery_candidate(idx, traj, maneuver, params, trajectories, valid, maneuver_ids, theta, flags, metadata, cfg)


def generate_candidate_bank(runtime: RuntimeFeatures, cfg: dict[str, Any]) -> CandidateBank:
    cand_cfg = cfg.get("candidate", {})
    K = int(cand_cfg.get("K", 32))
    counts = cand_cfg.get("counts", {})
    times = _times(cfg)
    route = _route_centerline(runtime, cfg)
    v0 = float(max(runtime.ego_history[-1, 3], 0.0)) if runtime.ego_history.size else 0.0
    speed_limit = float(runtime.map_features.get("speed_limit_mps", 13.4)) if runtime.map_features else 13.4
    v_ref = max(min(speed_limit, max(v0, 1.0)), 1.0)
    lane_width = float(cand_cfg.get("lane_width_m", 3.5))

    trajectories: list[np.ndarray] = []
    valid: list[bool] = []
    maneuver_ids: list[int] = []
    theta: list[dict[str, Any]] = []
    flags: list[dict[str, bool]] = []
    metadata: list[dict[str, Any]] = []

    keep_params = [
        (0.5 * v_ref, -0.1),
        (0.8 * v_ref, -0.05),
        (1.0 * v_ref, 0.0),
        (1.2 * v_ref, 0.0),
        (0.6 * v_ref, -0.2),
        (0.9 * v_ref, 0.05),
        (1.1 * v_ref, 0.05),
        (min(speed_limit, 1.3 * v_ref), 0.0),
        (0.35 * v_ref, -0.25),
        (min(speed_limit, 1.45 * v_ref), 0.05),
    ]
    for target, accel_bias in keep_params[: int(counts.get("keep_follow", 8))]:
        traj = _rollout_on_route(route, v0, times, target, accel_bias=accel_bias)
        _append_candidate(trajectories, valid, maneuver_ids, theta, flags, metadata, traj, "keep_follow", {"target_speed": target, "accel_bias": accel_bias}, cfg)

    for sd in _candidate_stop_grid(v0, times, cfg)[: int(counts.get("decelerate_stop", 6))]:
        traj = _rollout_on_route(route, v0, times, 0.0, stop_distance=float(sd))
        _append_candidate(trajectories, valid, maneuver_ids, theta, flags, metadata, traj, "decelerate_stop", {"stop_distance": float(sd)}, cfg)

    yield_profiles = [(0.8, 0.0), (0.5, 0.0), (0.3, 5.0), (0.2, 10.0), (0.15, 15.0), (0.1, 20.0)]
    for frac, delay in yield_profiles[: int(counts.get("yield_creep", 4))]:
        stop_dist = None if delay <= 0 else delay
        traj = _rollout_on_route(route, v0, times, frac * v_ref, stop_distance=stop_dist)
        _append_candidate(trajectories, valid, maneuver_ids, theta, flags, metadata, traj, "yield_creep", {"target_speed": frac * v_ref, "delay_stop_distance": delay}, cfg)

    for side, sign, count_key, maneuver in [
        ("left", 1.0, "lane_change_left", "lane_change_left"),
        ("right", -1.0, "lane_change_right", "lane_change_right"),
    ]:
        legal = _legal_lane_change(runtime, side)
        offsets = [sign * lane_width, sign * lane_width, sign * 0.5 * lane_width, sign * lane_width]
        speeds = [0.8 * v_ref, 1.0 * v_ref, 0.6 * v_ref, 1.1 * v_ref]
        for off, target in list(zip(offsets, speeds))[: int(counts.get(count_key, 4))]:
            traj = _rollout_on_route(route, v0, times, target, lateral_offset=off)
            _append_candidate(
                trajectories,
                valid,
                maneuver_ids,
                theta,
                flags,
                metadata,
                traj,
                maneuver,
                {"target_speed": target, "terminal_lateral_offset": off},
                cfg,
                force_valid=legal,
            )

    for target in [0.5 * v_ref, 0.7 * v_ref, 0.9 * v_ref, 1.0 * v_ref][: int(counts.get("route_turn_connector", 4))]:
        traj = _rollout_on_route(route, v0, times, target)
        _append_candidate(trajectories, valid, maneuver_ids, theta, flags, metadata, traj, "route_turn_connector", {"target_speed": target}, cfg)

    safe_count = int(counts.get("safe_fallback", 2))
    min_safe_stop = _min_dynamic_stop_distance(v0, cfg)
    horizon_dist = max(float(v0) * float(times[-1]), 10.0)
    safe_stop_grid = _unique_distances([
        min_safe_stop,
        min_safe_stop + 5.0,
        min_safe_stop + 12.0,
        min_safe_stop + 20.0,
        min_safe_stop + 30.0,
        max(min_safe_stop + 35.0, 0.45 * horizon_dist),
        max(min_safe_stop + 45.0, 0.60 * horizon_dist),
        max(min_safe_stop + 60.0, 0.80 * horizon_dist),
    ], min_sep_m=3.0)
    for stop_dist in safe_stop_grid[:safe_count]:
        traj = _rollout_on_route(route, v0, times, 0.0, stop_distance=float(stop_dist))
        _append_candidate(trajectories, valid, maneuver_ids, theta, flags, metadata, traj, "safe_fallback", {"stop_distance": float(stop_dist)}, cfg, force_valid=True)

    _inject_conflict_stop_priors(runtime, route, times, v0, trajectories, valid, maneuver_ids, theta, flags, metadata, cfg)

    filler_i = 0
    while len(trajectories) < K:
        target = v_ref * [0.4, 0.6, 0.8, 1.0, 1.1][filler_i % 5]
        traj = _rollout_on_route(route, v0, times, target)
        _append_candidate(
            trajectories,
            valid,
            maneuver_ids,
            theta,
            flags,
            metadata,
            traj,
            "keep_follow",
            {"target_speed": target, "filled_by": "legal_route_following_speed_filler"},
            cfg,
        )
        filler_i += 1

    _inject_history_priors(runtime, times, trajectories, valid, maneuver_ids, theta, flags, metadata, cfg)

    _repair_low_valid_count(
        runtime,
        route,
        times,
        v0,
        v_ref,
        trajectories,
        valid,
        maneuver_ids,
        theta,
        flags,
        metadata,
        cfg,
    )

    if len(trajectories) > K:
        trajectories = trajectories[:K]
        valid = valid[:K]
        maneuver_ids = maneuver_ids[:K]
        theta = theta[:K]
        flags = flags[:K]
        metadata = metadata[:K]

    if not any(valid):
        safe_traj = _rollout_on_route(route, v0, times, 0.0, stop_distance=max(2.0, v0))
        trajectories[0] = safe_traj
        valid[0] = True
        maneuver_ids[0] = MANEUVER_IDS["safe_fallback"]
        theta[0] = {"stop_distance": max(2.0, v0), "forced_safe": True}
        flags[0] = _dynamic_flags(safe_traj, float(cand_cfg.get("step_s", 0.1)), _cfg_dyn(cfg))
        metadata[0] = {"maneuver": "safe_fallback", "filled_by": "forced_safe"}

    traj_arr = np.stack(trajectories, axis=0).astype(np.float32)
    finite_traj = np.isfinite(traj_arr).all(axis=(1, 2))
    traj_arr = np.nan_to_num(traj_arr, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    valid_arr = np.asarray(valid, dtype=bool) & finite_traj
    if not valid_arr.any():
        safe_traj = _rollout_straight(v0, times, 0.0, stop_distance=max(2.0, v0))
        traj_arr[0] = np.nan_to_num(safe_traj, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        valid_arr[0] = True
        maneuver_ids[0] = MANEUVER_IDS["safe_fallback"]
        theta[0] = {"stop_distance": max(2.0, v0), "forced_safe_after_nan": True}
        flags[0] = _dynamic_flags(traj_arr[0], float(cand_cfg.get("step_s", 0.1)), _cfg_dyn(cfg))
        metadata[0] = {"maneuver": "safe_fallback", "filled_by": "forced_safe_after_nan"}
    for i in range(K):
        if not valid_arr[i]:
            src = int(np.flatnonzero(valid_arr)[0])
            traj_arr[i] = traj_arr[src]
            metadata[i]["duplicate_padding_from"] = src
            maneuver_ids[i] = MANEUVER_IDS["padding"]
    progress = route_progress_along_polyline(traj_arr[:, -1, :2], route)
    for i, p in enumerate(progress):
        metadata[i]["route_progress"] = float(p)
    return CandidateBank(
        trajectories=traj_arr,
        valid_mask=valid_arr,
        maneuver_ids=np.asarray(maneuver_ids, dtype=np.int64),
        theta=theta,
        dynamic_flags=flags,
        metadata=metadata,
    )


def interpolate_to_10hz(bank: CandidateBank, cfg: dict[str, Any]) -> CandidateBank:
    step = float(cfg.get("candidate", {}).get("step_s", 0.1))
    if abs(step - 0.1) < 1e-6:
        return bank
    horizon = float(cfg.get("candidate", {}).get("horizon_s", 8.0))
    new_t = np.arange(0.1, horizon + 0.05, 0.1, dtype=np.float32)
    trajs = []
    for traj in bank.trajectories:
        out = np.zeros((len(new_t), traj.shape[-1]), dtype=np.float32)
        for d in range(traj.shape[-1]):
            out[:, d] = np.interp(new_t, traj[:, 4], traj[:, d])
        out[:, 4] = new_t
        trajs.append(out)
    return CandidateBank(np.stack(trajs), bank.valid_mask.copy(), bank.maneuver_ids.copy(), list(bank.theta), list(bank.dynamic_flags), list(bank.metadata))
