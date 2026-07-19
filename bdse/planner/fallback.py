from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.data.state_schema import DEFAULT_VEHICLE_LENGTH_M, DEFAULT_VEHICLE_WIDTH_M
from bdse.planner.selector import runtime_greedy_selector
from bdse.planner.tournament import TournamentResult, run_tournament
from bdse.utils import compute_curvature, finite_difference, nearest_polyline_distance


def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return bool((c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def _crosses_polyline(path_xy: np.ndarray, line_xy: np.ndarray) -> bool:
    path = np.asarray(path_xy, dtype=np.float32).reshape(-1, 2)
    line = np.asarray(line_xy, dtype=np.float32).reshape(-1, 2)
    if len(path) < 2 or len(line) < 2:
        return False
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        for j in range(len(line) - 1):
            if _segments_intersect(a, b, line[j], line[j + 1]):
                return True
    return False


@dataclass(slots=True)
class FallbackResult:
    action_index: int
    tournament: TournamentResult
    triggered: bool
    stage: str
    diagnostics: dict[str, Any]


def _trajectory_curvature_batch(xy: np.ndarray) -> np.ndarray:
    """Vectorized variant of ``compute_curvature`` for a K x T x 2 trajectory bank."""
    pts = np.nan_to_num(np.asarray(xy, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    if pts.ndim != 3 or pts.shape[1] < 3:
        return np.zeros(pts.shape[:2], dtype=np.float32)
    dx = np.gradient(pts[:, :, 0], axis=1)
    dy = np.gradient(pts[:, :, 1], axis=1)
    ddx = np.gradient(dx, axis=1)
    ddy = np.gradient(dy, axis=1)
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-6)
    return ((dx * ddy - dy * ddx) / denom).astype(np.float32)


def _candidate_safety_time_masks(
    traj: np.ndarray,
    rsc: dict[str, Any],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build receding-horizon masks with stopping-distance calibration.

    A fixed four-second hard horizon recovered progress in v31, but it was too
    short for high-speed scenes.  CAVR keeps the short horizon at urban speeds
    and expands it only when the candidate's stopping time requires more look-
    ahead.  The calculation is candidate-local and uses no future labels.
    """
    arr = np.asarray(traj, dtype=np.float32)
    K, T = int(arr.shape[0]), int(arr.shape[1])
    times = arr[:, :, 4] if arr.shape[2] > 4 else np.broadcast_to(
        np.arange(T, dtype=np.float32)[None, :] * float(dt), (K, T)
    )
    base_hard = float(rsc.get("hard_check_horizon_s", float("inf")))
    base_soft = float(rsc.get("soft_check_horizon_s", float("inf")))
    if bool(rsc.get("speed_adaptive_horizon", False)) and arr.shape[2] > 3:
        speed = np.maximum(np.nanmax(np.maximum(arr[:, :, 3], 0.0), axis=1), 0.0)
        reaction = max(float(rsc.get("reaction_time_s", 0.7)), 0.0)
        decel = max(float(rsc.get("comfortable_emergency_decel_mps2", 5.0)), 0.5)
        margin = max(float(rsc.get("stopping_horizon_margin_s", 0.35)), 0.0)
        min_h = float(rsc.get("min_hard_horizon_s", base_hard))
        max_h = float(rsc.get("max_hard_horizon_s", max(base_soft, base_hard)))
        hard_horizon = np.clip(reaction + speed / decel + margin, min_h, max_h).astype(np.float32)
        soft_extra = max(float(rsc.get("soft_horizon_extra_s", 1.5)), 0.0)
        max_soft = float(rsc.get("max_soft_horizon_s", max(base_soft, max_h)))
        soft_floor = np.full((K,), base_soft, dtype=np.float32)
        soft_horizon = np.minimum(np.maximum(soft_floor, hard_horizon + soft_extra), max_soft).astype(np.float32)
    else:
        hard_horizon = np.full((K,), base_hard, dtype=np.float32)
        soft_horizon = np.full((K,), base_soft, dtype=np.float32)
    hard_mask = np.isfinite(times) & (times <= hard_horizon[:, None] + 1e-6)
    soft_mask = np.isfinite(times) & (times <= soft_horizon[:, None] + 1e-6)
    if T:
        hard_mask[:, 0] = True
        soft_mask[:, 0] = True
    return times.astype(np.float32), hard_mask, soft_mask, hard_horizon, soft_horizon


def _agent_envelope_metrics(
    traj: np.ndarray,
    times: np.ndarray,
    hard_time_mask: np.ndarray,
    soft_time_mask: np.ndarray,
    current_agent: np.ndarray,
    rsc: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return box-aware hard/soft deficits and time-to-conflict risk.

    The v31 point-radius test underestimated risk for long vehicles and at high
    closing speed.  When agent dimensions are available, this routine evaluates
    an oriented elliptical proxy of the Minkowski sum of ego and agent boxes.
    The longitudinal buffer grows with closing speed.  A legacy circle fallback
    is retained for incomplete states and old configurations.
    """
    arr = np.asarray(traj, dtype=np.float32)
    cur = np.asarray(current_agent, dtype=np.float32).reshape(-1)
    K, T = int(arr.shape[0]), int(arr.shape[1])
    vx_a = float(cur[5]) if cur.size > 5 else float(cur[3]) * np.cos(float(cur[2]))
    vy_a = float(cur[6]) if cur.size > 6 else float(cur[3]) * np.sin(float(cur[2]))
    yaw_a = float(cur[2]) if cur.size > 2 else 0.0
    pred_x = float(cur[0]) + vx_a * times
    pred_y = float(cur[1]) + vy_a * times
    dx = arr[:, :, 0] - pred_x
    dy = arr[:, :, 1] - pred_y
    dist = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-8)).astype(np.float32)

    speed = arr[:, :, 3] if arr.shape[2] > 3 else np.zeros((K, T), dtype=np.float32)
    yaw = arr[:, :, 2] if arr.shape[2] > 2 else np.zeros((K, T), dtype=np.float32)
    ego_vx = speed * np.cos(yaw)
    ego_vy = speed * np.sin(yaw)
    rel_vx = ego_vx - vx_a
    rel_vy = ego_vy - vy_a
    closing = np.maximum(-(dx * rel_vx + dy * rel_vy) / np.maximum(dist, 1e-4), 0.0).astype(np.float32)
    ttc = np.where(closing > 1e-3, dist / np.maximum(closing, 1e-3), np.inf).astype(np.float32)

    has_dims = cur.size > 8 and float(cur[7]) > 0.0 and float(cur[8]) > 0.0
    use_box = bool(rsc.get("use_box_agent_risk", False)) and has_dims
    if use_box:
        ego_l = max(float(rsc.get("ego_length_m", DEFAULT_VEHICLE_LENGTH_M)), 0.5)
        ego_w = max(float(rsc.get("ego_width_m", DEFAULT_VEHICLE_WIDTH_M)), 0.3)
        agent_l = max(float(cur[7]), 0.3)
        agent_w = max(float(cur[8]), 0.2)
        c, sn = np.cos(yaw_a), np.sin(yaw_a)
        longitudinal = c * dx + sn * dy
        lateral = -sn * dx + c * dy
        closing_buffer = np.minimum(
            closing * max(float(rsc.get("closing_speed_buffer_s", 0.22)), 0.0),
            max(float(rsc.get("max_closing_buffer_m", 3.0)), 0.0),
        )
        hard_l = 0.5 * (ego_l + agent_l) + float(rsc.get("hard_longitudinal_clearance_m", 0.20)) + closing_buffer
        hard_w = 0.5 * (ego_w + agent_w) + float(rsc.get("hard_lateral_clearance_m", 0.15))
        soft_l = hard_l + float(rsc.get("soft_longitudinal_extra_m", 1.00))
        soft_w = hard_w + float(rsc.get("soft_lateral_extra_m", 0.65))
        hard_norm = np.sqrt((longitudinal / np.maximum(hard_l, 0.1)) ** 2 + (lateral / max(hard_w, 0.1)) ** 2)
        soft_norm = np.sqrt((longitudinal / np.maximum(soft_l, 0.1)) ** 2 + (lateral / max(soft_w, 0.1)) ** 2)
        hard_def = np.where(hard_time_mask, np.maximum(1.0 - hard_norm, 0.0), 0.0).max(axis=1)
        soft_def = np.where(soft_time_mask, np.maximum(1.0 - soft_norm, 0.0), 0.0).max(axis=1)
        ttc_gate = soft_norm <= float(rsc.get("ttc_envelope_gate", 1.35))
    else:
        hard_r = max(float(rsc.get("hard_agent_radius_m", 0.85)), 1e-3)
        soft_r = max(float(rsc.get("soft_agent_radius_m", 1.5)), hard_r)
        hard_def = np.where(hard_time_mask, np.maximum(hard_r - dist, 0.0) / hard_r, 0.0).max(axis=1)
        soft_def = np.where(soft_time_mask, np.maximum(soft_r - dist, 0.0) / soft_r, 0.0).max(axis=1)
        ttc_gate = dist <= soft_r * float(rsc.get("ttc_envelope_gate", 1.35))

    ttc_eval = np.where(soft_time_mask & ttc_gate, ttc, np.inf)
    min_ttc = np.min(ttc_eval, axis=1).astype(np.float32)
    safe_ttc = max(float(rsc.get("agent_ttc_safe_s", 3.0)), 1e-3)
    ttc_risk = np.maximum((safe_ttc - min_ttc) / safe_ttc, 0.0)
    ttc_risk = np.where(np.isfinite(ttc_risk), ttc_risk, 0.0).astype(np.float32)
    return hard_def.astype(np.float32), soft_def.astype(np.float32), ttc_risk, min_ttc


def runtime_safety_flag_components(runtime: RuntimeFeatures, candidates: CandidateBank, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return tiered runtime safety flags for candidate actions.

    v26 used one conservative boolean flag for several very different failure
    modes: off-route, red-light, close agent proximity, high speed, and dynamic
    discomfort.  In closed loop this often marked every valid candidate as
    unsafe, so the hard-mask could not actually select a safe alternative.  This
    helper separates hard violations from soft risk indicators.  The deployed
    hard mask can then be applied only to infeasible/high-risk candidates, while
    soft risk is handled by rule reranking and evidence/certificate scores.
    """
    K = int(candidates.K)
    zeros = np.zeros((K,), dtype=bool)
    out = {
        "valid": zeros.copy(),
        "off_route_soft": zeros.copy(),
        "off_route_hard": zeros.copy(),
        "speed_soft": zeros.copy(),
        "speed_hard": zeros.copy(),
        "dyn_soft": zeros.copy(),
        "dyn_hard": zeros.copy(),
        "agent_soft": zeros.copy(),
        "agent_hard": zeros.copy(),
        "red_light": zeros.copy(),
        "soft": zeros.copy(),
        "hard": zeros.copy(),
        "legacy": zeros.copy(),
    }
    valid_mask = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)[:K]
    if valid_mask.shape[0] < K:
        valid_mask = np.pad(valid_mask, (0, K - valid_mask.shape[0]), constant_values=False)
    out["valid"] = valid_mask.astype(bool)
    if K <= 0 or not bool(valid_mask.any()):
        return out

    traj = np.nan_to_num(np.asarray(candidates.trajectories, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    if traj.ndim != 3 or traj.shape[0] != K or traj.shape[2] < 4:
        return out
    T = int(traj.shape[1])
    xy_all = traj[:, :, :2]
    rsc = cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    times, hard_time_mask, soft_time_mask, hard_horizons, soft_horizons = _candidate_safety_time_masks(traj, rsc, dt)

    route = np.asarray(runtime.map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32)
    width = float(runtime.map_features.get("route_corridor_width", cfg.get("candidate", {}).get("route_width_m", 4.0)))
    route_dist = nearest_polyline_distance(xy_all.reshape(-1, 2), route).reshape(K, T)
    soft_off_margin = float(rsc.get("soft_off_route_margin_m", 1.0))
    hard_off_margin = float(rsc.get("hard_off_route_margin_m", 3.0))
    out["off_route_soft"] = ((route_dist > width + soft_off_margin) & soft_time_mask).any(axis=1)
    out["off_route_hard"] = ((route_dist > width + hard_off_margin) & hard_time_mask).any(axis=1)

    speed_limit = float(runtime.map_features.get("speed_limit_mps", 13.4))
    soft_speed_margin = float(rsc.get("soft_speed_margin_mps", 2.0))
    hard_speed_margin = float(rsc.get("hard_speed_margin_mps", 5.0))
    out["speed_soft"] = ((traj[:, :, 3] > speed_limit + soft_speed_margin) & soft_time_mask).any(axis=1)
    out["speed_hard"] = ((traj[:, :, 3] > speed_limit + hard_speed_margin) & hard_time_mask).any(axis=1)

    v = traj[:, :, 3]
    if T >= 2:
        acc = np.gradient(v, dt, axis=1).astype(np.float32)
        jerk = np.gradient(acc, dt, axis=1).astype(np.float32)
    else:
        acc = np.zeros_like(v, dtype=np.float32)
        jerk = np.zeros_like(v, dtype=np.float32)
    curv = _trajectory_curvature_batch(xy_all)
    soft_acc = float(rsc.get("soft_acc_abs", 4.0))
    soft_jerk = float(rsc.get("soft_jerk_abs", 8.0))
    soft_curv = float(rsc.get("soft_curvature_abs", 0.35))
    hard_acc = float(rsc.get("hard_acc_abs", 7.0))
    hard_jerk = float(rsc.get("hard_jerk_abs", 15.0))
    hard_curv = float(rsc.get("hard_curvature_abs", 0.55))
    out["dyn_soft"] = ((np.abs(acc) > soft_acc) & soft_time_mask).any(axis=1) | ((np.abs(jerk) > soft_jerk) & soft_time_mask).any(axis=1) | ((np.abs(curv) > soft_curv) & soft_time_mask).any(axis=1)
    out["dyn_hard"] = ((np.abs(acc) > hard_acc) & hard_time_mask).any(axis=1) | ((np.abs(jerk) > hard_jerk) & hard_time_mask).any(axis=1) | ((np.abs(curv) > hard_curv) & hard_time_mask).any(axis=1)

    red_light_bad = np.zeros((K,), dtype=bool)
    for sl in runtime.map_features.get("stop_lines", []) if runtime.map_features else []:
        status_red = bool(sl.get("red", False)) or ("red" in str(sl.get("status", "")).lower())
        if not status_red:
            continue
        line_xy = np.asarray(sl.get("xy", []), dtype=np.float32).reshape(-1, 2)
        if len(line_xy) < 2:
            continue
        for a in np.flatnonzero(valid_mask & ~red_light_bad):
            path_prefix = xy_all[int(a)][hard_time_mask[int(a)]]
            if _crosses_polyline(path_prefix, line_xy):
                red_light_bad[int(a)] = True
    out["red_light"] = red_light_bad

    agent_soft = np.zeros((K,), dtype=bool)
    agent_hard = np.zeros((K,), dtype=bool)
    agent_valid = np.asarray(runtime.agent_valid, dtype=bool).reshape(-1)
    if agent_valid.size and getattr(runtime, "current_agents", None) is not None:
        hard_ttc_flag = float(rsc.get("hard_ttc_flag_threshold", 0.92))
        soft_ttc_flag = float(rsc.get("soft_ttc_flag_threshold", 0.55))
        for j in np.flatnonzero(agent_valid):
            cur = np.asarray(runtime.current_agents[int(j)], dtype=np.float32).reshape(-1)
            if cur.size < 4:
                continue
            hard_def, soft_def, ttc_risk, _ = _agent_envelope_metrics(
                traj, times, hard_time_mask, soft_time_mask, cur, rsc
            )
            agent_soft |= (soft_def > 0.0) | (ttc_risk >= soft_ttc_flag)
            agent_hard |= (hard_def > 0.0) | (ttc_risk >= hard_ttc_flag)
    out["agent_soft"] = agent_soft
    out["agent_hard"] = agent_hard

    include_speed_soft = bool(rsc.get("include_speed_in_soft", True))
    include_dyn_soft = bool(rsc.get("include_dynamics_in_soft", True))
    include_speed_hard = bool(rsc.get("include_speed_in_hard", False))
    include_dyn_hard = bool(rsc.get("include_dynamics_in_hard", False))
    soft = out["off_route_soft"] | out["agent_soft"] | out["red_light"]
    hard = out["off_route_hard"] | out["agent_hard"] | out["red_light"]
    if include_speed_soft:
        soft = soft | out["speed_soft"]
    if include_dyn_soft:
        soft = soft | out["dyn_soft"]
    if include_speed_hard:
        hard = hard | out["speed_hard"]
    if include_dyn_hard:
        hard = hard | out["dyn_hard"]
    legacy = out["off_route_soft"] | out["speed_soft"] | out["dyn_soft"] | out["agent_soft"] | out["red_light"]
    out["soft"] = valid_mask & soft
    out["hard"] = valid_mask & hard
    out["legacy"] = valid_mask & legacy
    return out


def runtime_risk_scores(runtime: RuntimeFeatures, candidates: CandidateBank, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Continuous runtime risk scores for min-violation recovery.

    Boolean hard flags are necessary for constraint filtering, but they are not
    sufficient once every valid candidate violates at least one hard flag.  In
    that regime v28 treated all flagged actions almost equally, so recovery could
    choose a candidate with larger collision/TTC risk as long as it had better
    lateral/progress utility.  These scores keep the same fixed evidence budget
    and use only runtime geometry to rank *how severe* each violation is.

    Returns lower-is-better per-candidate arrays.  The main outputs are:
      - hard: severe off-route / hard agent / red-light violation severity;
      - soft: mild off-route / soft agent proximity severity;
      - agent: hard-agent proximity severity, useful for diagnostics.
    """
    K = int(candidates.K)
    z = np.zeros((K,), dtype=np.float32)
    out = {
        "hard": z.copy(),
        "soft": z.copy(),
        "agent": z.copy(),
        "hard_agent": z.copy(),
        "soft_agent": z.copy(),
        "off_route": z.copy(),
        "hard_off_route": z.copy(),
        "soft_off_route": z.copy(),
        "red_light": z.copy(),
        "agent_ttc": z.copy(),
        "min_ttc_s": np.full((K,), np.inf, dtype=np.float32),
        "hard_horizon_s": z.copy(),
        "soft_horizon_s": z.copy(),
    }
    if K <= 0:
        return out
    valid_mask = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)[:K]
    if valid_mask.shape[0] < K:
        valid_mask = np.pad(valid_mask, (0, K - valid_mask.shape[0]), constant_values=False)
    traj = np.nan_to_num(np.asarray(candidates.trajectories, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    if traj.ndim != 3 or traj.shape[0] != K or traj.shape[2] < 2:
        return out
    T = int(traj.shape[1])
    xy_all = traj[:, :, :2]
    rsc = cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1)) if isinstance(cfg, dict) else 0.1
    times, hard_time_mask, soft_time_mask, hard_horizons, soft_horizons = _candidate_safety_time_masks(traj, rsc, dt)
    out["hard_horizon_s"] = hard_horizons.astype(np.float32)
    out["soft_horizon_s"] = soft_horizons.astype(np.float32)

    route = np.asarray(runtime.map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32)
    width = float(runtime.map_features.get("route_corridor_width", cfg.get("candidate", {}).get("route_width_m", 4.0)))
    route_dist = nearest_polyline_distance(xy_all.reshape(-1, 2), route).reshape(K, T)
    soft_off_margin = float(rsc.get("soft_off_route_margin_m", 1.0))
    hard_off_margin = float(rsc.get("hard_off_route_margin_m", 3.0))
    soft_excess = np.maximum(route_dist - (width + soft_off_margin), 0.0)
    hard_excess = np.maximum(route_dist - (width + hard_off_margin), 0.0)
    off_soft = np.where(soft_time_mask, soft_excess, 0.0).max(axis=1).astype(np.float32)
    off_hard = np.where(hard_time_mask, hard_excess, 0.0).max(axis=1).astype(np.float32)
    out["off_route"] = off_hard
    out["hard_off_route"] = off_hard
    out["soft_off_route"] = off_soft

    agent_soft_def = np.zeros((K,), dtype=np.float32)
    agent_hard_def = np.zeros((K,), dtype=np.float32)
    agent_ttc_risk = np.zeros((K,), dtype=np.float32)
    min_ttc_s = np.full((K,), np.inf, dtype=np.float32)
    agent_valid = np.asarray(runtime.agent_valid, dtype=bool).reshape(-1)
    if agent_valid.size and getattr(runtime, "current_agents", None) is not None:
        for j in np.flatnonzero(agent_valid):
            cur = np.asarray(runtime.current_agents[int(j)], dtype=np.float32).reshape(-1)
            if cur.size < 4:
                continue
            hard_def, soft_def, ttc_risk, min_ttc = _agent_envelope_metrics(
                traj, times, hard_time_mask, soft_time_mask, cur, rsc
            )
            agent_hard_def = np.maximum(agent_hard_def, hard_def)
            agent_soft_def = np.maximum(agent_soft_def, soft_def)
            agent_ttc_risk = np.maximum(agent_ttc_risk, ttc_risk)
            min_ttc_s = np.minimum(min_ttc_s, min_ttc)
    out["agent"] = agent_hard_def
    out["hard_agent"] = agent_hard_def
    out["soft_agent"] = agent_soft_def
    out["agent_ttc"] = agent_ttc_risk
    out["min_ttc_s"] = min_ttc_s

    red = np.asarray(runtime_safety_flag_components(runtime, candidates, cfg).get("red_light", np.zeros((K,), dtype=bool)), dtype=bool).reshape(-1)[:K]
    red_risk = red.astype(np.float32) * float(rsc.get("red_light_risk", 10.0))
    out["red_light"] = red_risk

    hard_agent_w = float(rsc.get("risk_hard_agent_weight", 6.0))
    hard_off_w = float(rsc.get("risk_hard_offroute_weight", 3.0))
    red_w = float(rsc.get("risk_red_light_weight", 10.0))
    soft_agent_w = float(rsc.get("risk_soft_agent_weight", 1.2))
    soft_off_w = float(rsc.get("risk_soft_offroute_weight", 0.8))
    hard_ttc_w = float(rsc.get("risk_hard_ttc_weight", 2.5))
    soft_ttc_w = float(rsc.get("risk_soft_ttc_weight", 1.0))
    hard = hard_agent_w * agent_hard_def + hard_ttc_w * agent_ttc_risk + hard_off_w * off_hard + red_w * red.astype(np.float32)
    soft = soft_agent_w * agent_soft_def + soft_ttc_w * agent_ttc_risk + soft_off_w * off_soft
    hard = np.where(valid_mask, hard, np.inf).astype(np.float32)
    soft = np.where(valid_mask, soft, np.inf).astype(np.float32)
    out["hard"] = hard
    out["soft"] = soft
    return out


def runtime_safety_flags_from_runtime(runtime: RuntimeFeatures, candidates: CandidateBank, cfg: dict[str, Any]) -> np.ndarray:
    """Return per-candidate flags used by the tournament hard filter.

    Modes:
      - legacy: v26 conservative union;
      - soft: soft-risk union;
      - hard / dual_tier: v28 hard-only constraint;
      - tiered: v27 soft-if-available else hard;
      - adaptive_dual_tier: v29.  Use soft flags as constraints only when the
        soft-safe set is large enough; otherwise fall back to hard flags.  This
        recovers v27's collision/TTC benefit in easy scenes without collapsing
        route progress/drivable in dense scenes.
    """
    comp = runtime_safety_flag_components(runtime, candidates, cfg)
    valid = comp["valid"]
    rsc = cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}
    mode = str(rsc.get("flag_mode", "legacy")).lower().replace("-", "_")
    if mode in {"hard", "dual_tier", "dual", "hard_constraint_soft_price", "hard_constrained_soft_priced"}:
        flags = comp["hard"]
    elif mode == "soft":
        flags = comp["soft"]
    elif mode == "tiered":
        soft_safe_exists = bool((valid & ~comp["soft"]).any())
        flags = comp["soft"] if soft_safe_exists else comp["hard"]
    elif mode in {"adaptive_dual_tier", "adaptive_dual", "adaptive_hard_soft", "adaptive_soft_price"}:
        valid_n = max(int(valid.sum()), 1)
        soft_safe_count = int((valid & ~comp["soft"]).sum())
        hard_safe_count = int((valid & ~comp["hard"]).sum())
        extra_soft = int((valid & comp["soft"] & ~comp["hard"]).sum())
        min_soft_safe = int(rsc.get("adaptive_min_soft_safe_actions", 6))
        min_soft_ratio = float(rsc.get("adaptive_min_soft_safe_ratio", 0.20))
        max_extra_soft = int(rsc.get("adaptive_max_extra_soft_flags", max(valid_n, 1)))
        use_soft = (
            soft_safe_count >= min_soft_safe
            and (soft_safe_count / float(valid_n)) >= min_soft_ratio
            and extra_soft <= max_extra_soft
        )
        # If hard-safe and soft-safe sets are both empty, use hard so downstream
        # min-violation recovery can rank by continuous risk instead of declaring
        # everything equally unsafe.
        flags = comp["soft"] if use_soft else comp["hard"]
    else:
        flags = comp["legacy"]
    return (valid & np.asarray(flags, dtype=bool).reshape(-1)[: int(candidates.K)]).astype(bool)

def runtime_safety_diagnostics(runtime: RuntimeFeatures, candidates: CandidateBank, cfg: dict[str, Any]) -> dict[str, Any]:
    """Compact diagnostics for v28 dual-tier safety.

    These counters separate hard feasibility from soft interaction risk.  They
    make it possible to tell whether a selected flagged action is unavoidable
    because every candidate is hard-unsafe, or merely a soft-risk action that
    should be handled by pricing rather than fallback.
    """
    comp = runtime_safety_flag_components(runtime, candidates, cfg)
    valid = np.asarray(comp.get("valid", np.zeros((int(candidates.K),), dtype=bool)), dtype=bool).reshape(-1)[: int(candidates.K)]
    hard = np.asarray(comp.get("hard", np.zeros_like(valid)), dtype=bool).reshape(-1)[: int(candidates.K)]
    soft = np.asarray(comp.get("soft", np.zeros_like(valid)), dtype=bool).reshape(-1)[: int(candidates.K)]
    flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
    flags = np.asarray(flags, dtype=bool).reshape(-1)[: int(candidates.K)]
    valid_n = int(valid.sum())
    hard_safe = valid & ~hard
    soft_safe = valid & ~soft
    active_safe = valid & ~flags
    active_tier = "soft" if np.array_equal(flags, valid & soft) else ("hard" if np.array_equal(flags, valid & hard) else "custom")
    risks = runtime_risk_scores(runtime, candidates, cfg)
    out: dict[str, Any] = {
        "runtime_flag_mode": str((cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}).get("flag_mode", "legacy")),
        "active_flag_tier": active_tier,
        "valid_action_count": valid_n,
        "hard_flagged_count": int((valid & hard).sum()),
        "soft_flagged_count": int((valid & soft).sum()),
        "active_flagged_count": int((valid & flags).sum()),
        "hard_safe_action_available": bool(hard_safe.any()),
        "soft_safe_action_available": bool(soft_safe.any()),
        "active_safe_action_available": bool(active_safe.any()),
        "hard_safe_action_count": int(hard_safe.sum()),
        "soft_safe_action_count": int(soft_safe.sum()),
        "active_safe_action_count": int(active_safe.sum()),
        "min_hard_risk": float(np.nanmin(risks.get("hard", np.array([np.inf], dtype=np.float32)))) if valid_n else float("inf"),
        "min_soft_risk": float(np.nanmin(risks.get("soft", np.array([np.inf], dtype=np.float32)))) if valid_n else float("inf"),
        "min_hard_agent_risk": float(np.nanmin(risks.get("hard_agent", risks.get("agent", np.array([np.inf], dtype=np.float32))))) if valid_n else float("inf"),
        "min_hard_offroute_risk": float(np.nanmin(risks.get("hard_off_route", risks.get("off_route", np.array([np.inf], dtype=np.float32))))) if valid_n else float("inf"),
        "min_agent_ttc_risk": float(np.nanmin(risks.get("agent_ttc", np.array([np.inf], dtype=np.float32)))) if valid_n else float("inf"),
        "minimum_predicted_ttc_s": float(np.nanmin(risks.get("min_ttc_s", np.array([np.inf], dtype=np.float32)))) if valid_n else float("inf"),
        "mean_hard_horizon_s": float(np.nanmean(risks.get("hard_horizon_s", np.array([0.0], dtype=np.float32)))) if valid_n else 0.0,
        "max_hard_horizon_s": float(np.nanmax(risks.get("hard_horizon_s", np.array([0.0], dtype=np.float32)))) if valid_n else 0.0,
    }
    for name in ["off_route_hard", "off_route_soft", "agent_hard", "agent_soft", "red_light", "speed_hard", "speed_soft", "dyn_hard", "dyn_soft"]:
        arr = np.asarray(comp.get(name, np.zeros_like(valid)), dtype=bool).reshape(-1)[: int(candidates.K)]
        out[f"{name}_count"] = int((valid & arr).sum())
    return out


def rule_based_runtime_scores(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
    safety_flags: np.ndarray | None = None,
) -> np.ndarray:
    """Fast rule fallback scores.

    ``safety_flags`` may be supplied by the caller to avoid recomputing the same
    O(K * T * agents) safety check.  This fixes the main closed-loop hotspot in
    the fallback rule-rerank branch while preserving the previous scoring rule.
    """
    scores = np.full((candidates.K,), np.inf, dtype=np.float32)
    valid_mask = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)[: candidates.K]
    if not bool(valid_mask.any()):
        return scores
    route = np.asarray(runtime.map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32)
    traj = np.nan_to_num(np.asarray(candidates.trajectories, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    route_dist = nearest_polyline_distance(traj[:, :, :2].reshape(-1, 2), route).reshape(candidates.K, traj.shape[1])
    route_cost = np.square(route_dist).mean(axis=1)
    progress_reward = np.maximum(traj[:, -1, 0], 0.0)
    # Reranking can be more nuanced than the tournament hard mask: hard flags
    # receive a very large penalty, while soft risks receive a smaller penalty
    # so that the planner can still choose forward progress when every candidate
    # is near a soft interaction envelope.
    rsc = cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}
    comp = runtime_safety_flag_components(runtime, candidates, cfg)
    if safety_flags is None:
        safety_flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
    hard_flags = np.asarray(safety_flags, dtype=bool).reshape(-1)[: candidates.K]
    soft_flags = np.asarray(comp.get("soft", np.zeros((candidates.K,), dtype=bool)), dtype=bool).reshape(-1)[: candidates.K]
    hard_penalty = float(rsc.get("rule_hard_penalty", 1000.0))
    soft_penalty = float(rsc.get("rule_soft_penalty", 25.0))
    hard_risk_weight = float(rsc.get("rule_hard_risk_weight", 180.0))
    soft_risk_weight = float(rsc.get("rule_soft_risk_weight", 18.0))
    progress_weight = float(rsc.get("rule_progress_weight", 0.1))
    risks = runtime_risk_scores(runtime, candidates, cfg)
    hard_risk = np.nan_to_num(risks.get("hard", np.zeros((candidates.K,), dtype=np.float32)), nan=0.0, posinf=0.0, neginf=0.0)
    soft_risk = np.nan_to_num(risks.get("soft", np.zeros((candidates.K,), dtype=np.float32)), nan=0.0, posinf=0.0, neginf=0.0)
    scores[valid_mask] = (
        route_cost[valid_mask]
        - progress_weight * progress_reward[valid_mask]
        + hard_penalty * hard_flags[valid_mask].astype(np.float32)
        + soft_penalty * (soft_flags[valid_mask] & ~hard_flags[valid_mask]).astype(np.float32)
        + hard_risk_weight * hard_risk[valid_mask]
        + soft_risk_weight * soft_risk[valid_mask]
    ).astype(np.float32)
    return scores



@dataclass(slots=True)
class RecoveryDecision:
    """Final runtime recovery action and auditable selection diagnostics."""

    action_index: int
    diagnostics: dict[str, Any]


def _robust_min_scale(values: np.ndarray, ids: np.ndarray, quantile: float = 0.90, floor: float = 1e-3) -> tuple[float, float]:
    """Return a scene-adaptive minimum and robust upper spread.

    Fixed absolute PMV bands are brittle because agent, route and certificate
    scores have unrelated units and can change scale across scenes.  VCDSR uses
    a within-candidate robust scale so every epsilon is dimensionless.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    idx = np.asarray(ids, dtype=np.int64).reshape(-1)
    vals = arr[idx]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.min(vals))
    q = float(np.quantile(vals, min(max(float(quantile), 0.50), 1.0)))
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    scale = max(q - lo, mad, float(floor))
    return lo, scale


def _normalized_excess(values: np.ndarray, ids: np.ndarray, quantile: float, floor: float) -> tuple[np.ndarray, float, float]:
    lo, scale = _robust_min_scale(values, ids, quantile=quantile, floor=floor)
    out = np.maximum((np.asarray(values, dtype=np.float64) - lo) / scale, 0.0)
    out = np.nan_to_num(out, nan=1e6, posinf=1e6, neginf=0.0).astype(np.float32)
    return out, lo, scale


def _epsilon_pareto_frontier(ids: np.ndarray, objectives: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Return the epsilon-nondominated candidate ids (all objectives minimized)."""
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    obj = np.asarray(objectives, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64).reshape(1, -1)
    if ids.size <= 1:
        return ids.copy()
    keep = np.ones((ids.size,), dtype=bool)
    for i in range(ids.size):
        # j dominates i only when it is no worse outside the configured
        # tolerance in every objective and materially better in at least one.
        no_worse = np.all(obj <= obj[i : i + 1] + eps, axis=1)
        materially_better = np.any(obj < obj[i : i + 1] - eps, axis=1)
        dominated = no_worse & materially_better
        dominated[i] = False
        if bool(dominated.any()):
            keep[i] = False
    return ids[keep]


def viability_frontier_recovery_action(
    candidates: CandidateBank,
    safety_flags: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
    runtime: RuntimeFeatures | None = None,
    tournament_scores: np.ndarray | None = None,
    reference_action: int | None = None,
) -> RecoveryDecision:
    """Evidence-conditioned, viability-calibrated recovery for all-flagged scenes.

    VCDSR differs from v30 PMV-RBSR in three important ways:
      1. component risks are normalized with scene-adaptive robust scales rather
         than compared through fixed raw-unit bands;
      2. the candidate set is a true epsilon-Pareto frontier over agent risk,
         route risk, soft risk, BDSE certificate loss and progress loss;
      3. final progress recovery is conditioned on the evidence tournament score,
         so the fallback no longer becomes a disconnected rule-only controller.

    The function uses only runtime-observable geometry and already-computed BDSE
    tournament scores.  It does not query extra evidence or future labels.
    """
    valid = np.flatnonzero(np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)).astype(np.int64)
    if valid.size == 0:
        return RecoveryDecision(0, {"mode": "cavr", "reason": "no_valid_candidate"})

    traj = np.nan_to_num(np.asarray(candidates.trajectories, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    K = int(traj.shape[0])
    flags = np.zeros((K,), dtype=bool) if safety_flags is None else np.asarray(safety_flags, dtype=bool).reshape(-1)
    if flags.size < K:
        flags = np.pad(flags, (0, K - flags.size), constant_values=True)
    flags = flags[:K]
    safe = valid[~flags[valid]]
    all_flagged = safe.size == 0
    pool = valid.copy() if all_flagged else safe.copy()

    fc = (((cfg or {}).get("fallback", {}) or {}).get("safe_progress_recovery", {}) or {}) if isinstance(cfg, dict) else {}
    vc = (fc.get("viability_frontier", {}) or {}) if isinstance(fc, dict) else {}
    q = float(vc.get("scale_quantile", 0.90))
    scale_floor = float(vc.get("scale_floor", 1e-3))
    min_pool = max(1, int(vc.get("min_pool", 4)))
    max_pool = max(min_pool, int(vc.get("max_pool", 18)))

    xy = traj[:, :, :2]
    lateral_mean = np.mean(np.abs(xy[:, :, 1]), axis=1).astype(np.float32)
    lateral_final = np.abs(xy[:, -1, 1]).astype(np.float32)
    progress = xy[:, -1, 0].astype(np.float32)
    speed_final = traj[:, -1, 3].astype(np.float32) if traj.shape[-1] > 3 else np.zeros((K,), dtype=np.float32)
    path_len = np.linalg.norm(np.diff(xy, axis=1), axis=-1).sum(axis=1).astype(np.float32) if xy.shape[1] > 1 else np.zeros((K,), dtype=np.float32)

    if runtime is not None and isinstance(cfg, dict):
        risks = runtime_risk_scores(runtime, candidates, cfg)
    else:
        z = np.zeros((K,), dtype=np.float32)
        risks = {"hard": z, "soft": z, "hard_agent": z, "soft_agent": z, "hard_off_route": z, "soft_off_route": z, "red_light": z}
    hard = np.asarray(risks.get("hard", np.zeros((K,), dtype=np.float32)), dtype=np.float32)
    soft = np.asarray(risks.get("soft", np.zeros((K,), dtype=np.float32)), dtype=np.float32)
    agent_hard = np.asarray(risks.get("hard_agent", risks.get("agent", np.zeros((K,), dtype=np.float32))), dtype=np.float32)
    agent_soft = np.asarray(risks.get("soft_agent", np.zeros((K,), dtype=np.float32)), dtype=np.float32)
    agent_ttc = np.asarray(risks.get("agent_ttc", np.zeros((K,), dtype=np.float32)), dtype=np.float32)
    min_ttc_s = np.asarray(risks.get("min_ttc_s", np.full((K,), np.inf, dtype=np.float32)), dtype=np.float32)
    hard_horizon_s = np.asarray(risks.get("hard_horizon_s", np.zeros((K,), dtype=np.float32)), dtype=np.float32)
    off_hard = np.asarray(risks.get("hard_off_route", risks.get("off_route", np.zeros((K,), dtype=np.float32))), dtype=np.float32)
    off_soft = np.asarray(risks.get("soft_off_route", np.zeros((K,), dtype=np.float32)), dtype=np.float32)
    red = np.asarray(risks.get("red_light", np.zeros((K,), dtype=np.float32)), dtype=np.float32)

    # Preserve the hard rule hierarchy before any risk/progress trade-off.
    red_level = float(np.min(red[pool]))
    red_pool = pool[red[pool] <= red_level + float(vc.get("red_tolerance", 1e-6))]
    if red_pool.size:
        pool = red_pool

    # Every scale is estimated once from the same candidate population.  This
    # makes diagnostics comparable while avoiding cascading re-normalization.
    agent_n, agent_min, agent_scale = _normalized_excess(agent_hard, valid, q, scale_floor)
    off_n, off_min, off_scale = _normalized_excess(off_hard, valid, q, scale_floor)
    soft_n, soft_min, soft_scale = _normalized_excess(soft, valid, q, scale_floor)
    hard_n, hard_min, hard_scale = _normalized_excess(hard, valid, q, scale_floor)
    ttc_n, ttc_min, ttc_scale = _normalized_excess(agent_ttc, valid, q, scale_floor)

    scores = np.zeros((K,), dtype=np.float32)
    score_available = tournament_scores is not None
    if score_available:
        raw = np.asarray(tournament_scores, dtype=np.float32).reshape(-1)
        if raw.size < K:
            raw = np.pad(raw, (0, K - raw.size), constant_values=-np.inf)
        scores = raw[:K]
        finite_valid = valid[np.isfinite(scores[valid])]
        if finite_valid.size:
            best_score = float(np.max(scores[finite_valid]))
            score_loss = np.where(np.isfinite(scores), best_score - scores, np.inf).astype(np.float32)
        else:
            score_available = False
            score_loss = np.zeros((K,), dtype=np.float32)
    else:
        score_loss = np.zeros((K,), dtype=np.float32)
    score_n, score_min, score_scale = _normalized_excess(score_loss, valid, q, scale_floor)

    # Order-invariant joint viability guard.  V31 filtered agent, route and
    # certificate dimensions sequentially, so changing their order could change
    # the candidate set.  CAVR first forms a Chebyshev safety criticality over
    # normalized agent overlap, TTC and off-route risk, then admits a compact
    # near-minimum viability set.  The BDSE certificate is not used as a hard
    # gate; it remains an explicit Pareto objective so learned evidence cannot
    # discard a physically safer candidate.
    relaxations: list[str] = []
    agent_cap = max(float(vc.get("joint_agent_scale", 1.0)), 1e-3)
    ttc_cap = max(float(vc.get("joint_ttc_scale", 1.0)), 1e-3)
    off_cap = max(float(vc.get("joint_offroute_scale", 1.0)), 1e-3)
    joint_viability = np.maximum.reduce([agent_n / agent_cap, ttc_n / ttc_cap, off_n / off_cap]).astype(np.float32)
    pool_before_guard = int(pool.size)
    if pool.size > min_pool:
        joint_min = float(np.min(joint_viability[pool]))
        joint_eps = max(float(vc.get("joint_viability_epsilon_norm", 0.18)), 0.0)
        strict = pool[joint_viability[pool] <= joint_min + joint_eps]
        if strict.size >= min_pool:
            pool = strict
        else:
            relaxations.append("joint_viability")
            order = sorted(
                pool.tolist(),
                key=lambda a: (
                    float(joint_viability[int(a)]),
                    float(agent_n[int(a)]),
                    float(ttc_n[int(a)]),
                    float(off_n[int(a)]),
                    -float(progress[int(a)]),
                    int(a),
                ),
            )
            pool = np.asarray(order[:min_pool], dtype=np.int64)

    # Normalize utility objectives.  Progress is kept as an explicit Pareto
    # objective, so a low-risk stop trajectory cannot dominate a near-risk moving
    # trajectory merely because safety dimensions are slightly smaller.
    progress_cost = -progress
    progress_loss_n, _, progress_scale = _normalized_excess(progress_cost, valid, q, scale_floor)
    path_cost_n, _, _ = _normalized_excess(-path_len, valid, q, scale_floor)
    lateral_n, _, _ = _normalized_excess(lateral_mean + 0.5 * lateral_final, valid, q, scale_floor)

    obj = np.stack(
        [
            agent_n[pool],
            ttc_n[pool],
            off_n[pool],
            soft_n[pool],
            score_n[pool] if score_available else np.zeros((pool.size,), dtype=np.float32),
            progress_loss_n[pool],
        ],
        axis=1,
    )
    eps = np.asarray(
        [
            float(vc.get("pareto_agent_epsilon", 0.04)),
            float(vc.get("pareto_ttc_epsilon", 0.04)),
            float(vc.get("pareto_offroute_epsilon", 0.06)),
            float(vc.get("pareto_soft_epsilon", 0.08)),
            float(vc.get("pareto_certificate_epsilon", 0.08)),
            float(vc.get("pareto_progress_epsilon", 0.04)),
        ],
        dtype=np.float32,
    )
    frontier = _epsilon_pareto_frontier(pool, obj, eps)
    if frontier.size == 0:
        frontier = pool.copy()
        relaxations.append("empty_frontier")
    if frontier.size > max_pool:
        pre_cost = (
            float(vc.get("agent_risk_weight", 0.55)) * agent_n[frontier]
            + float(vc.get("ttc_risk_weight", 0.38)) * ttc_n[frontier]
            + float(vc.get("offroute_risk_weight", 0.45)) * off_n[frontier]
            + float(vc.get("soft_risk_weight", 0.12)) * soft_n[frontier]
            + float(vc.get("certificate_loss_weight", 0.20)) * score_n[frontier]
            + float(vc.get("progress_loss_weight", 0.18)) * progress_loss_n[frontier]
        )
        order = np.argsort(pre_cost, kind="stable")[:max_pool]
        frontier = frontier[order]

    low_speed_thr = float(vc.get("low_speed_threshold", fc.get("low_speed_threshold", 0.25)))
    utility = (
        float(vc.get("progress_weight", 1.00)) * (1.0 - progress_loss_n)
        + float(vc.get("path_length_weight", 0.10)) * (1.0 - path_cost_n)
        + float(vc.get("certificate_weight", 0.24)) * (1.0 - score_n if score_available else 0.0)
        - float(vc.get("lateral_weight", 0.18)) * lateral_n
        - float(vc.get("agent_risk_weight", 0.55)) * agent_n
        - float(vc.get("ttc_risk_weight", 0.38)) * ttc_n
        - float(vc.get("offroute_risk_weight", 0.45)) * off_n
        - float(vc.get("soft_risk_weight", 0.12)) * soft_n
        - float(vc.get("hard_risk_weight", 0.10)) * hard_n
        - float(vc.get("low_speed_penalty", 0.08)) * (speed_final < low_speed_thr).astype(np.float32)
    ).astype(np.float32)
    if reference_action is not None and 0 <= int(reference_action) < K:
        utility[int(reference_action)] += float(vc.get("reference_action_bonus", 0.03))

    chosen = max(
        frontier.tolist(),
        key=lambda a: (
            float(utility[int(a)]),
            -float(agent_n[int(a)]),
            -float(ttc_n[int(a)]),
            -float(off_n[int(a)]),
            float(progress[int(a)]),
            -int(a),
        ),
    )
    chosen = int(chosen)
    min_hard_id = int(valid[np.argmin(hard[valid])])
    diagnostics: dict[str, Any] = {
        "mode": "cavr",
        "all_flagged": bool(all_flagged),
        "score_conditioned": bool(score_available),
        "valid_pool_size": int(valid.size),
        "safe_pool_size": int(safe.size),
        "pool_before_guard": int(pool_before_guard),
        "guarded_pool_size": int(pool.size),
        "frontier_size": int(frontier.size),
        "frontier_actions": [int(a) for a in frontier.tolist()],
        "relaxations": list(relaxations),
        "selected_action": chosen,
        "selected_utility": float(utility[chosen]),
        "selected_progress": float(progress[chosen]),
        "selected_path_length": float(path_len[chosen]),
        "selected_hard_risk": float(hard[chosen]),
        "selected_soft_risk": float(soft[chosen]),
        "selected_agent_risk": float(agent_hard[chosen]),
        "selected_agent_ttc_risk": float(agent_ttc[chosen]),
        "selected_min_ttc_s": float(min_ttc_s[chosen]) if np.isfinite(min_ttc_s[chosen]) else None,
        "selected_hard_horizon_s": float(hard_horizon_s[chosen]),
        "selected_joint_viability": float(joint_viability[chosen]),
        "selected_offroute_risk": float(off_hard[chosen]),
        "selected_score": float(scores[chosen]) if score_available and np.isfinite(scores[chosen]) else None,
        "selected_score_loss_norm": float(score_n[chosen]) if score_available else 0.0,
        "selected_hard_risk_excess": float(hard[chosen] - hard[min_hard_id]),
        "minimum_hard_risk_action": min_hard_id,
        "minimum_hard_risk": float(hard[min_hard_id]),
        "progress_gain_over_min_risk": float(progress[chosen] - progress[min_hard_id]),
        "agent_risk_min": float(agent_min),
        "agent_risk_scale": float(agent_scale),
        "offroute_risk_min": float(off_min),
        "offroute_risk_scale": float(off_scale),
        "soft_risk_min": float(soft_min),
        "soft_risk_scale": float(soft_scale),
        "hard_risk_min": float(hard_min),
        "hard_risk_scale": float(hard_scale),
        "ttc_risk_min": float(ttc_min),
        "ttc_risk_scale": float(ttc_scale),
        "certificate_loss_min": float(score_min),
        "certificate_loss_scale": float(score_scale),
        "progress_scale": float(progress_scale),
    }
    return RecoveryDecision(chosen, diagnostics)


def conservative_fallback_action(
    candidates: CandidateBank,
    safety_flags: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
    runtime: RuntimeFeatures | None = None,
) -> int:
    """Choose a safe recovery action without collapsing to zero progress.

    The v25 conservative fallback sorted by low terminal speed and low progress,
    so many safety-triggered replans degenerated to action 0 / near-stop.  That
    improved neither fixed-budget evidence use nor closed-loop progress.  This
    recovery remains rule-only, but it is lexicographic: prefer unflagged valid
    candidates, then minimize route/lateral/comfort cost while rewarding forward
    progress.  It is only used when the certificate cannot provide an accepted
    action.
    """
    valid = np.flatnonzero(np.asarray(candidates.valid_mask, dtype=bool).reshape(-1))
    if len(valid) == 0:
        return 0
    traj = np.nan_to_num(np.asarray(candidates.trajectories, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    K = traj.shape[0]
    flags = np.zeros((K,), dtype=bool) if safety_flags is None else np.asarray(safety_flags, dtype=bool).reshape(-1)
    if flags.shape[0] < K:
        flags = np.pad(flags, (0, K - flags.shape[0]), constant_values=True)
    flags = flags[:K]
    safe_valid = [int(a) for a in valid.tolist() if not bool(flags[int(a)])]
    pool = safe_valid if safe_valid else [int(a) for a in valid.tolist()]
    xy = traj[:, :, :2]
    lateral_mean = np.mean(np.abs(xy[:, :, 1]), axis=1)
    lateral_final = np.abs(xy[:, -1, 1])
    progress = xy[:, -1, 0]
    speed_final = traj[:, -1, 3] if traj.shape[-1] > 3 else np.zeros((K,), dtype=np.float32)
    if xy.shape[1] > 1:
        step = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
        path_len = step.sum(axis=1)
    else:
        path_len = np.zeros((K,), dtype=np.float32)
    fc = ((cfg or {}).get("fallback", {}) or {}).get("safe_progress_recovery", {}) if isinstance(cfg, dict) else {}
    hard_risk = np.zeros((K,), dtype=np.float32)
    soft_risk = np.zeros((K,), dtype=np.float32)
    agent_risk = np.zeros((K,), dtype=np.float32)
    offroute_risk = np.zeros((K,), dtype=np.float32)
    if runtime is not None and isinstance(cfg, dict):
        risks = runtime_risk_scores(runtime, candidates, cfg)
        hard_risk = np.nan_to_num(risks.get("hard", hard_risk), nan=0.0, posinf=np.inf, neginf=0.0).astype(np.float32)
        soft_risk = np.nan_to_num(risks.get("soft", soft_risk), nan=0.0, posinf=np.inf, neginf=0.0).astype(np.float32)
        agent_risk = np.nan_to_num(risks.get("hard_agent", risks.get("agent", agent_risk)), nan=0.0, posinf=np.inf, neginf=0.0).astype(np.float32)
        offroute_risk = np.nan_to_num(risks.get("hard_off_route", risks.get("off_route", offroute_risk)), nan=0.0, posinf=np.inf, neginf=0.0).astype(np.float32)

    all_flagged = len(safe_valid) == 0
    if all_flagged and bool(fc.get("pareto_min_violation", True)) and runtime is not None and isinstance(cfg, dict):
        # v30 PMV-RBSR: when every valid candidate violates the active runtime
        # flag, do not rank unsafe actions by a single huge weighted sum.  First
        # keep a Pareto band around the minimum continuous violation, with agent
        # proximity protected before off-route/progress trade-offs.  Then choose
        # the best progress/drivable utility inside that near-min-violation set.
        pool0 = np.asarray(pool, dtype=np.int64)
        finite_pool = pool0[np.isfinite(hard_risk[pool0])]
        if finite_pool.size > 0:
            min_agent = float(np.min(agent_risk[finite_pool]))
            agent_margin = float(fc.get("agent_risk_abs_margin", 0.06))
            agent_pool = finite_pool[agent_risk[finite_pool] <= min_agent + agent_margin]
            if agent_pool.size >= int(fc.get("min_pareto_pool", 4)):
                finite_pool = agent_pool
            min_hard = float(np.min(hard_risk[finite_pool]))
            hard_margin = max(
                float(fc.get("hard_risk_abs_margin", 18.0)),
                float(fc.get("hard_risk_rel_margin", 0.08)) * max(abs(min_hard), 1.0),
            )
            risk_pool = finite_pool[hard_risk[finite_pool] <= min_hard + hard_margin]
            min_soft = float(np.min(soft_risk[risk_pool])) if risk_pool.size else float(np.min(soft_risk[finite_pool]))
            soft_margin = max(
                float(fc.get("soft_risk_abs_margin", 5.0)),
                float(fc.get("soft_risk_rel_margin", 0.12)) * max(abs(min_soft), 1.0),
            )
            soft_pool = risk_pool[soft_risk[risk_pool] <= min_soft + soft_margin] if risk_pool.size else finite_pool
            if soft_pool.size >= int(fc.get("min_pareto_pool", 4)):
                risk_pool = soft_pool
            if risk_pool.size > int(fc.get("max_pareto_pool", 16)):
                # Keep the lowest-risk actions but leave enough alternatives for
                # progress-aware utility to avoid over-conservative stopping.
                order = sorted(risk_pool.tolist(), key=lambda a: (float(agent_risk[a]), float(hard_risk[a]), float(soft_risk[a]), -float(progress[a]), int(a)))
                risk_pool = np.asarray(order[: int(fc.get("max_pareto_pool", 16))], dtype=np.int64)
            q = float(fc.get("progress_quantile_floor", 0.0))
            if q > 0.0 and risk_pool.size >= int(fc.get("min_pareto_pool", 4)):
                thr = float(np.quantile(progress[risk_pool], min(max(q, 0.0), 0.95)))
                progress_pool_np = risk_pool[progress[risk_pool] >= thr]
                if progress_pool_np.size >= int(fc.get("min_pareto_pool", 4)):
                    risk_pool = progress_pool_np
            if risk_pool.size > 0:
                pool = [int(a) for a in risk_pool.tolist()]

    min_progress = float(fc.get("min_progress", -1.0))
    progress_pool = [a for a in pool if float(progress[a]) >= min_progress]
    if progress_pool:
        pool = progress_pool
    progress_w = float(fc.get("progress_weight", 0.50))
    path_w = float(fc.get("path_length_weight", 0.04))
    lateral_w = float(fc.get("lateral_weight", 1.2))
    lateral_final_w = float(fc.get("lateral_final_weight", 0.6))
    low_speed_thr = float(fc.get("low_speed_threshold", 0.25))
    low_speed_penalty = float(fc.get("low_speed_penalty", 0.10))
    hard_risk_w = float(fc.get("hard_risk_weight", 260.0))
    soft_risk_w = float(fc.get("soft_risk_weight", 22.0))
    # Inside the all-flagged Pareto band, safety has already been handled
    # lexicographically, so cost only pays excess risk over the pool minimum.
    risk_ref_hard = float(np.min(hard_risk[np.asarray(pool, dtype=np.int64)])) if all_flagged and pool else 0.0
    risk_ref_soft = float(np.min(soft_risk[np.asarray(pool, dtype=np.int64)])) if all_flagged and pool else 0.0
    unsafe_penalty = float(fc.get("unsafe_penalty", 1000.0))
    def cost(a: int) -> tuple[float, int]:
        hard_term = max(float(hard_risk[a]) - risk_ref_hard, 0.0) if all_flagged else float(hard_risk[a])
        soft_term = max(float(soft_risk[a]) - risk_ref_soft, 0.0) if all_flagged else float(soft_risk[a])
        c = (
            lateral_w * float(lateral_mean[a])
            + lateral_final_w * float(lateral_final[a])
            - progress_w * float(progress[a])
            - path_w * float(path_len[a])
            + hard_risk_w * hard_term
            + soft_risk_w * soft_term
            + (low_speed_penalty if float(speed_final[a]) < low_speed_thr else 0.0)
        )
        if bool(flags[a]) and not all_flagged:
            # Preserve the lexicographic safe-first preference whenever a safe
            # candidate exists.  When all are flagged, avoid adding a constant
            # penalty that carries no decision information.
            c += unsafe_penalty
        return (float(c), int(a))
    return int(min(pool, key=cost))


def apply_fallback_if_needed(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    atom_budget_costs: np.ndarray,
    atom_active_mask: np.ndarray,
    initial_tournament: TournamentResult,
    cfg: dict[str, Any],
) -> FallbackResult:
    fcfg = cfg.get("fallback", {})
    if not bool(fcfg.get("enabled", True)):
        return FallbackResult(initial_tournament.action_index, initial_tournament, False, "disabled", dict(initial_tournament.diagnostics))
    tau_delta = float(fcfg.get("tau_delta", 0.1))
    delta = float(initial_tournament.diagnostics.get("delta_hat_B", 0.0))
    if delta >= tau_delta:
        return FallbackResult(initial_tournament.action_index, initial_tournament, False, "not_triggered", dict(initial_tournament.diagnostics))

    runtime_flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
    best_tournament = initial_tournament
    stage = "triggered"
    local_cfg = dict(cfg)
    local_cfg["tournament"] = dict(cfg.get("tournament", {}))
    local_cfg["evidence"] = dict(cfg.get("evidence", {}))

    if bool(fcfg.get("expand_rivals", True)):
        for L in [8, 16, min(31, candidates.K - 1)]:
            local_cfg["tournament"]["L_infer"] = int(L)
            trial = run_tournament(predicted_base_cost, predicted_atom_costs, best_tournament.diagnostics.get("selected_atoms", []), candidates.valid_mask, runtime_flags, local_cfg)
            best_tournament = trial
            stage = f"rival_expanded_{L}"
            if float(trial.diagnostics.get("delta_hat_B", 0.0)) >= tau_delta:
                return FallbackResult(trial.action_index, trial, True, stage, dict(trial.diagnostics))

    if bool(fcfg.get("expand_budget", True)):
        for B in [8, 16, 32]:
            if B < int(cfg.get("evidence", {}).get("budget", 16)):
                continue
            sel = runtime_greedy_selector(
                predicted_base_cost,
                predicted_atom_costs,
                atom_budget_costs,
                candidates.valid_mask,
                runtime_flags,
                budget=float(B),
                L_infer=int(local_cfg.get("tournament", {}).get("L_infer", 16)),
                gamma_max=float(cfg.get("selector", {}).get("gamma_max_default", 100.0)),
                eta_pred=float(cfg.get("selector", {}).get("eta_pred", 1.0)),
                atom_active_mask=atom_active_mask,
            )
            trial = run_tournament(predicted_base_cost, predicted_atom_costs, sel.selected, candidates.valid_mask, runtime_flags, local_cfg)
            best_tournament = trial
            stage = f"budget_expanded_{B}"
            if float(trial.diagnostics.get("delta_hat_B", 0.0)) >= tau_delta:
                return FallbackResult(trial.action_index, trial, True, stage, dict(trial.diagnostics))

    top_k = int(fcfg.get("rule_rerank_top_k", 5))
    if top_k > 0:
        valid_scores = best_tournament.scores.copy()
        top_actions = np.argsort(-valid_scores)[:top_k]
        rule_cost = rule_based_runtime_scores(runtime, candidates, cfg)
        best = min([int(a) for a in top_actions if candidates.valid_mask[a]], key=lambda a: (float(rule_cost[a]), a), default=best_tournament.action_index)
        if np.isfinite(rule_cost[best]) and not runtime_flags[best]:
            best_tournament.action_index = int(best)
            stage = "rule_rerank"
            return FallbackResult(int(best), best_tournament, True, stage, {**best_tournament.diagnostics, "rule_cost": float(rule_cost[best])})

    action = conservative_fallback_action(candidates, runtime=runtime, cfg=cfg)
    best_tournament.action_index = int(action)
    return FallbackResult(int(action), best_tournament, True, "conservative_fallback", dict(best_tournament.diagnostics))
