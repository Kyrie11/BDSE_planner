from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.utils import angle_wrap, compute_curvature, finite_difference, route_progress_along_polyline, sample_polyline

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


def _cfg_dyn(cfg: dict[str, Any]) -> DynamicFeasibility:
    c = cfg.get("candidate", {})
    return DynamicFeasibility(
        max_accel=float(c.get("max_accel", 3.0)),
        max_decel=float(c.get("max_decel", -5.0)),
        max_jerk=float(c.get("max_jerk", 5.0)),
        max_curvature=float(c.get("max_curvature", 0.25)),
        max_curvature_rate=float(c.get("max_curvature_rate", 0.20)),
        max_lateral_accel=float(c.get("max_lateral_accel", 3.0)),
    )


def _times(cfg: dict[str, Any]) -> np.ndarray:
    c = cfg.get("candidate", {})
    horizon = float(c.get("horizon_s", 8.0))
    dt = float(c.get("step_s", 0.1))
    return np.arange(dt, horizon + 0.5 * dt, dt, dtype=np.float32)


def _route_centerline(runtime: RuntimeFeatures, cfg: dict[str, Any]) -> np.ndarray:
    route = runtime.map_features.get("route_centerline") if runtime.map_features else None
    if route is None or len(route) < 2:
        x = np.linspace(0.0, 160.0, 81, dtype=np.float32)
        return np.stack([x, np.zeros_like(x)], axis=1)
    return np.asarray(route, dtype=np.float32).reshape(-1, 2)


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


def _dynamic_flags(traj: np.ndarray, dt: float, dyn: DynamicFeasibility) -> dict[str, bool]:
    v = traj[:, 3]
    acc = finite_difference(v, dt)
    jerk = finite_difference(acc, dt)
    curv = compute_curvature(traj[:, :2])
    curv_rate = finite_difference(curv, dt)
    lat_acc = v * v * np.abs(curv)
    flags = {
        "accel_ok": bool(np.nanmax(acc) <= dyn.max_accel + 1e-4),
        "decel_ok": bool(np.nanmin(acc) >= dyn.max_decel - 1e-4),
        "jerk_ok": bool(np.nanmax(np.abs(jerk)) <= dyn.max_jerk + 1e-4),
        "curvature_ok": bool(np.nanmax(np.abs(curv)) <= dyn.max_curvature + 1e-4),
        "curvature_rate_ok": bool(np.nanmax(np.abs(curv_rate)) <= dyn.max_curvature_rate + 1e-4),
        "lateral_accel_ok": bool(np.nanmax(lat_acc) <= dyn.max_lateral_accel + 1e-4),
    }
    flags["dynamically_feasible"] = all(flags.values())
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
    ]
    for target, accel_bias in keep_params[: int(counts.get("keep_follow", 8))]:
        traj = _rollout_on_route(route, v0, times, target, accel_bias=accel_bias)
        _append_candidate(trajectories, valid, maneuver_ids, theta, flags, metadata, traj, "keep_follow", {"target_speed": target, "accel_bias": accel_bias}, cfg)

    for sd in [5, 10, 15, 20, 30, 40][: int(counts.get("decelerate_stop", 6))]:
        traj = _rollout_on_route(route, v0, times, 0.0, stop_distance=float(sd))
        _append_candidate(trajectories, valid, maneuver_ids, theta, flags, metadata, traj, "decelerate_stop", {"stop_distance": float(sd)}, cfg)

    yield_profiles = [(0.8, 0.0), (0.5, 0.0), (0.3, 5.0), (0.2, 10.0)]
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
    for stop_dist in [max(2.0, v0 * 0.8), max(4.0, v0 * 1.2)][:safe_count]:
        traj = _rollout_on_route(route, v0, times, 0.0, stop_distance=stop_dist)
        _append_candidate(trajectories, valid, maneuver_ids, theta, flags, metadata, traj, "safe_fallback", {"stop_distance": stop_dist}, cfg, force_valid=True)

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
    valid_arr = np.asarray(valid, dtype=bool)
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
