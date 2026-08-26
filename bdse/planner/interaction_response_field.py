from __future__ import annotations

"""V64.3.45 Plan-Conditioned Interaction Response Field (PIRF).

PIRF replaces V44's scene-global discrete response-mode posterior with an
agent-local continuous longitudinal response field.  Logged future is used only
for TRAIN supervision of response acceleration; deployment consumes only the
current runtime state, the already generated ego candidate and frozen model
parameters.

The field is factorized deliberately:

  a_local(j) = f_local(current agent/history, current ego-agent geometry)
  a_plan(j,a) = a_local(j) + g_plan(current state, ego candidate a)

The plan correction has zero bias and every plan feature is multiplied by the
full-horizon ungated interaction exposure, so a candidate with vanishing
interaction exposure cannot induce an arbitrary response correction.  Both
accelerations are clamped to the already-existing V43 response envelope
[-2.0, +0.5] m/s^2; these are inherited from the frozen brake/CA modes rather
than newly tuned V45 hyperparameters.
"""

from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.response_value_observables import _candidate_times, _ungated_future_agent_occupancy

RESPONSE_ACCEL_MIN = -2.0
RESPONSE_ACCEL_MAX = 0.5

RESPONSE_FIELD_LOCAL_FEATURE_NAMES = [
    "agent_speed",
    "agent_recent_longitudinal_accel",
    "current_rel_longitudinal",
    "current_rel_lateral",
    "agent_length",
    "agent_width",
]
RESPONSE_FIELD_PLAN_FEATURE_NAMES = [
    "exposure_mean",
    "exposure_times_peak",
    "exposure_times_earlyness",
    "exposure_times_rel_longitudinal_at_peak",
    "exposure_times_rel_lateral_at_peak",
    "exposure_times_relative_speed_at_peak",
]
RESPONSE_FIELD_OBSERVABLE_NAMES = [
    "response_field_cv_occupancy_cost",
    "response_field_local_occupancy_cost",
    "response_field_plan_occupancy_cost",
]

EPS = 1.0e-12


def _finite(x: np.ndarray | float, limit: float = 1.0e6) -> np.ndarray:
    return np.clip(np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=limit, neginf=-limit), -limit, limit)


def _agent_velocity(st: np.ndarray) -> tuple[float, float, float]:
    s = np.asarray(st, dtype=np.float64).reshape(-1)
    yaw = float(s[2]) if s.size > 2 and np.isfinite(s[2]) else 0.0
    speed = float(s[3]) if s.size > 3 and np.isfinite(s[3]) else 0.0
    vx = float(s[5]) if s.size > 5 and np.isfinite(s[5]) else speed * float(np.cos(yaw))
    vy = float(s[6]) if s.size > 6 and np.isfinite(s[6]) else speed * float(np.sin(yaw))
    vmag = float(np.hypot(vx, vy))
    if vmag <= 1.0e-4:
        vmag = max(speed, 0.0)
        hx, hy = float(np.cos(yaw)), float(np.sin(yaw))
    else:
        hx, hy = vx / vmag, vy / vmag
    return max(vmag, 0.0), float(hx), float(hy)


def _ego_current(runtime: RuntimeFeatures) -> np.ndarray:
    h = np.asarray(runtime.ego_history, dtype=np.float64)
    if h.ndim == 2 and h.shape[0] and h.shape[1] >= 2:
        return h[-1]
    return np.zeros((5,), dtype=np.float64)


def _recent_longitudinal_accel(runtime: RuntimeFeatures, j: int, dt: float) -> float:
    hist = np.asarray(runtime.agent_history, dtype=np.float64)
    if hist.ndim != 3 or j >= hist.shape[0] or hist.shape[1] < 2:
        return 0.0
    a = hist[j, -2]
    b = hist[j, -1]
    va = float(np.hypot(a[5], a[6])) if a.size > 6 and np.all(np.isfinite(a[5:7])) else (float(a[3]) if a.size > 3 else 0.0)
    vb = float(np.hypot(b[5], b[6])) if b.size > 6 and np.all(np.isfinite(b[5:7])) else (float(b[3]) if b.size > 3 else 0.0)
    return float(np.clip((vb - va) / max(float(dt), 1.0e-3), -10.0, 10.0))


def response_field_local_agent_features(runtime: RuntimeFeatures, cfg: dict[str, Any]) -> np.ndarray:
    cur = np.asarray(runtime.current_agents, dtype=np.float64)
    valid = np.asarray(runtime.agent_valid, dtype=bool).reshape(-1)
    N = min(len(cur), len(valid))
    out = np.zeros((N, len(RESPONSE_FIELD_LOCAL_FEATURE_NAMES)), dtype=np.float64)
    ego = _ego_current(runtime)
    ex, ey = (float(ego[0]), float(ego[1])) if ego.size >= 2 else (0.0, 0.0)
    dt = float((cfg.get("candidate", {}) or {}).get("step_s", 0.1))
    for j in range(N):
        if not valid[j]:
            continue
        st = cur[j].reshape(-1)
        v, hx, hy = _agent_velocity(st)
        nx, ny = -hy, hx
        dx = ex - float(st[0]); dy = ey - float(st[1])
        rel_long = hx * dx + hy * dy
        rel_lat = nx * dx + ny * dy
        length = float(st[7]) if st.size > 7 and np.isfinite(st[7]) and st[7] > 0 else 4.0
        width = float(st[8]) if st.size > 8 and np.isfinite(st[8]) and st[8] > 0 else 1.8
        out[j] = [v, _recent_longitudinal_accel(runtime, j, dt), rel_long, rel_lat, length, width]
    return _finite(out)


def _cv_agent_future(st: np.ndarray, T: int, dt: float, accel: float = 0.0) -> np.ndarray:
    s = np.asarray(st, dtype=np.float64).reshape(-1)
    x0 = float(s[0]) if s.size else 0.0
    y0 = float(s[1]) if s.size > 1 else 0.0
    yaw = float(s[2]) if s.size > 2 else 0.0
    v0, hx, hy = _agent_velocity(s)
    a = float(np.clip(accel, RESPONSE_ACCEL_MIN, RESPONSE_ACCEL_MAX))
    t = np.arange(1, T + 1, dtype=np.float64) * float(dt)
    if a < 0.0 and v0 > 0.0:
        tstop = v0 / max(-a, 1.0e-9)
        te = np.minimum(t, tstop)
        disp = v0 * te + 0.5 * a * te * te
        vel = np.maximum(v0 + a * t, 0.0)
    else:
        disp = v0 * t + 0.5 * a * t * t
        vel = np.maximum(v0 + a * t, 0.0)
    x = x0 + hx * disp
    y = y0 + hy * disp
    return np.stack([x, y, np.full_like(t, yaw), vel, t], axis=1)


def response_field_plan_agent_features(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return KxNxP zero-at-zero-interaction plan features and KxN exposure."""
    traj = np.asarray(candidates.trajectories, dtype=np.float64)
    K, T = int(traj.shape[0]), int(traj.shape[1])
    cur = np.asarray(runtime.current_agents, dtype=np.float64)
    valid = np.asarray(runtime.agent_valid, dtype=bool).reshape(-1)
    N = min(len(cur), len(valid))
    dt = float((cfg.get("candidate", {}) or {}).get("step_s", 0.1))
    times = _candidate_times(traj, dt)
    feats = np.zeros((K, N, len(RESPONSE_FIELD_PLAN_FEATURE_NAMES)), dtype=np.float64)
    exposure = np.zeros((K, N), dtype=np.float64)
    for j in range(N):
        if not valid[j]:
            continue
        st = cur[j].reshape(-1)
        cv = _cv_agent_future(st, T, dt, 0.0)
        pot = _ungated_future_agent_occupancy(traj, times, cv, st, cfg)
        if pot.size == 0:
            continue
        mean = np.mean(pot, axis=1)
        peak = np.max(pot, axis=1)
        tidx = np.argmax(pot, axis=1)
        vj, hx, hy = _agent_velocity(st); nx, ny = -hy, hx
        for k in range(K):
            q = int(tidx[k]); q = min(max(q, 0), T - 1)
            ax, ay = float(cv[q, 0]), float(cv[q, 1])
            dx = float(traj[k, q, 0]) - ax; dy = float(traj[k, q, 1]) - ay
            rel_long = hx * dx + hy * dy
            rel_lat = nx * dx + ny * dy
            ego_v = float(traj[k, q, 3]) if traj.shape[2] > 3 else 0.0
            e = float(mean[k])
            early = 1.0 - float(q) / max(float(T - 1), 1.0)
            feats[k, j] = [e, e * float(peak[k]), e * early, e * rel_long, e * rel_lat, e * (ego_v - vj)]
            exposure[k, j] = e
    return _finite(feats), _finite(exposure)


def logged_longitudinal_response_target(
    runtime: RuntimeFeatures,
    logged_agents: np.ndarray,
    agent_index: int,
    cfg: dict[str, Any],
) -> float:
    """Fit a physically bounded constant longitudinal acceleration to logged future."""
    cur = np.asarray(runtime.current_agents, dtype=np.float64)
    if agent_index >= len(cur):
        return 0.0
    gt = np.asarray(logged_agents, dtype=np.float64)
    if gt.ndim != 3 or agent_index >= gt.shape[0] or gt.shape[1] <= 0:
        return 0.0
    st = cur[agent_index].reshape(-1)
    v0, hx, hy = _agent_velocity(st)
    dt = float((cfg.get("candidate", {}) or {}).get("step_s", 0.1))
    T = int(gt.shape[1])
    t = np.arange(1, T + 1, dtype=np.float64) * dt
    cvx = float(st[0]) + hx * v0 * t
    cvy = float(st[1]) + hy * v0 * t
    rx = gt[agent_index, :, 0] - cvx
    ry = gt[agent_index, :, 1] - cvy
    rlong = hx * rx + hy * ry
    q = 0.5 * t * t
    good = np.isfinite(rlong) & np.isfinite(q)
    if not np.any(good):
        return 0.0
    den = float(np.dot(q[good], q[good]))
    if den <= EPS:
        return 0.0
    a = float(np.dot(q[good], rlong[good]) / den)
    return float(np.clip(a, RESPONSE_ACCEL_MIN, RESPONSE_ACCEL_MAX))


def _response_model_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    ic = (((cfg.get("runtime", {}) or {}).get("decisive_frontier_value", {}) or {}).get("incumbent_contrastive_extremal_recovery", {}) or {})
    sc = ic.get("selection_conditioned_intervention_recovery", {}) or {}
    return sc.get("interaction_response_field", {}) or {}


def _predict_local_accel(local: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    N = int(local.shape[0])
    if not bool(model.get("enabled", False)):
        return np.zeros((N,), dtype=np.float64)
    names = [str(x) for x in model.get("local_feature_names", [])]
    if names != RESPONSE_FIELD_LOCAL_FEATURE_NAMES:
        raise ValueError("V45 response-field local feature schema mismatch")
    scale = np.asarray(model.get("local_feature_scale", []), dtype=np.float64).reshape(-1)
    w = np.asarray(model.get("local_weights", []), dtype=np.float64).reshape(-1)
    if scale.size != local.shape[1] or w.size != local.shape[1]:
        raise ValueError("V45 response-field local model shape mismatch")
    b = float(model.get("local_bias", 0.0))
    z = local / np.maximum(scale[None, :], 1.0e-6)
    return np.clip(z @ w + b, RESPONSE_ACCEL_MIN, RESPONSE_ACCEL_MAX)


def _predict_plan_accel(local_a: np.ndarray, plan: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    K, N, P = plan.shape
    base = np.broadcast_to(local_a[None, :], (K, N)).copy()
    if not bool(model.get("enabled", False)) or not bool(model.get("plan_enabled", False)):
        return base
    names = [str(x) for x in model.get("plan_feature_names", [])]
    if names != RESPONSE_FIELD_PLAN_FEATURE_NAMES:
        raise ValueError("V45 response-field plan feature schema mismatch")
    scale = np.asarray(model.get("plan_feature_scale", []), dtype=np.float64).reshape(-1)
    w = np.asarray(model.get("plan_weights", []), dtype=np.float64).reshape(-1)
    if scale.size != P or w.size != P:
        raise ValueError("V45 response-field plan model shape mismatch")
    z = plan / np.maximum(scale[None, None, :], 1.0e-6)
    corr = np.tensordot(z, w, axes=([2], [0]))
    # No bias by construction.  Since every plan feature contains exposure, the
    # correction vanishes continuously as candidate-agent interaction vanishes.
    return np.clip(base + corr, RESPONSE_ACCEL_MIN, RESPONSE_ACCEL_MAX)


def _occupancy_cost_for_accels(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
    accels: np.ndarray,
) -> np.ndarray:
    traj = np.asarray(candidates.trajectories, dtype=np.float64)
    K, T = int(traj.shape[0]), int(traj.shape[1])
    cur = np.asarray(runtime.current_agents, dtype=np.float64)
    valid = np.asarray(runtime.agent_valid, dtype=bool).reshape(-1)
    N = min(len(cur), len(valid))
    dt = float((cfg.get("candidate", {}) or {}).get("step_s", 0.1))
    times = _candidate_times(traj, dt)
    A = np.asarray(accels, dtype=np.float64)
    if A.ndim == 1:
        A = np.broadcast_to(A[None, :], (K, A.size))
    if A.shape[0] != K or A.shape[1] < N:
        raise ValueError("V45 response-field acceleration shape mismatch")
    occ_time = np.zeros((K, T), dtype=np.float64)
    for k in range(K):
        one = traj[k : k + 1]
        one_t = times[:1] if times.ndim == 2 else times
        for j in range(N):
            if not valid[j]:
                continue
            fut = _cv_agent_future(cur[j], T, dt, float(A[k, j]))
            pot = _ungated_future_agent_occupancy(one, one_t, fut, cur[j], cfg)
            if pot.size:
                occ_time[k, : pot.shape[1]] = np.maximum(occ_time[k, : pot.shape[1]], pot[0])
    return _finite(np.mean(occ_time, axis=1) if T > 0 else np.zeros((K,), dtype=np.float64))


def runtime_interaction_response_field_observable_costs(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Return Kx3 lower-is-better CV/local/plan response-field occupancy costs."""
    K = int(candidates.K)
    if K <= 0:
        return np.zeros((0, len(RESPONSE_FIELD_OBSERVABLE_NAMES)), dtype=np.float64), list(RESPONSE_FIELD_OBSERVABLE_NAMES)
    local = response_field_local_agent_features(runtime, cfg)
    plan, _ = response_field_plan_agent_features(runtime, candidates, cfg)
    model = _response_model_cfg(cfg)
    a0 = np.zeros((local.shape[0],), dtype=np.float64)
    al = _predict_local_accel(local, model)
    ap = _predict_plan_accel(al, plan, model)
    cv = _occupancy_cost_for_accels(runtime, candidates, cfg, a0)
    lc = _occupancy_cost_for_accels(runtime, candidates, cfg, al)
    pc = _occupancy_cost_for_accels(runtime, candidates, cfg, ap)
    out = np.stack([cv, lc, pc], axis=1)
    if out.shape != (K, len(RESPONSE_FIELD_OBSERVABLE_NAMES)) or not np.all(np.isfinite(out)):
        raise ValueError("V45 response-field observable matrix malformed/non-finite")
    return out.astype(np.float64), list(RESPONSE_FIELD_OBSERVABLE_NAMES)
