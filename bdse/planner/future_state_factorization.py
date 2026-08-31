from __future__ import annotations

"""V64.3.47 Future-State Factorized Recovery (FSFR) observables.

V46 showed that a learnable longitudinal response second moment and hand-built
hazard-profile functionals are not deployment-sufficient.  V47 therefore does
not enlarge those scalar statistics.  It tests two distinct future-state
families that remain absent from the frozen V45 PLAN value path:

1. an agent-local, ego-plan-conditioned *2-D* response field.  V45's learned
   longitudinal acceleration is retained exactly and a separately supervised
   lateral drift component is added; and
2. a runtime-predictable ego-reference consequence approximating the teacher's
   label-only demonstration term.  Logged ego future is TRAIN-only nuisance
   supervision; deployment sees only current ego history, route and the already
   generated candidate trajectories.

All outputs are lower-is-better candidate costs.  This module never selects or
re-ranks actions and never consumes logged future at deployment.
"""

from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.evidence_atoms import _eval_traj
from bdse.planner.interaction_response_field import (
    EPS,
    RESPONSE_FIELD_LOCAL_FEATURE_NAMES,
    RESPONSE_FIELD_PLAN_FEATURE_NAMES,
    _agent_velocity,
    _candidate_times,
    _finite,
    _occupancy_cost_for_accels,
    _predict_local_accel,
    _predict_plan_accel,
    _response_model_cfg,
    response_field_local_agent_features,
    response_field_plan_agent_features,
)
from bdse.planner.response_value_observables import _ungated_future_agent_occupancy
from bdse.planner.teacher_cost import demo_cost, global_comfort_cost
from bdse.utils import nearest_polyline_distance, route_progress_along_polyline

FSFR_OBSERVABLE_NAMES = [
    "fsfr_plan_1d_occupancy_cost",
    "fsfr_plan_2d_occupancy_cost",
    "fsfr_predicted_demo_cost",
]

EGO_REFERENCE_FEATURE_NAMES = [
    "cv_position_mse",
    "cv_terminal_position_sq",
    "current_speed_mse",
    "terminal_speed_delta_sq",
    "current_heading_mse",
    "terminal_heading_delta_sq",
    "route_deviation_cost",
    "progress_deficit_cost",
    "global_comfort_cost",
]


def _wrap_angle(x: np.ndarray | float) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return np.arctan2(np.sin(a), np.cos(a))


def _fsfr_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    ic = (((cfg.get("runtime", {}) or {}).get("decisive_frontier_value", {}) or {}).get("incumbent_contrastive_extremal_recovery", {}) or {})
    sc = ic.get("selection_conditioned_intervention_recovery", {}) or {}
    return sc.get("future_state_factorization", {}) or {}


def logged_lateral_drift_target(
    runtime: RuntimeFeatures,
    logged_agents: np.ndarray,
    agent_index: int,
    cfg: dict[str, Any],
) -> float:
    """Least-squares lateral drift velocity of logged future relative to CV.

    The local normal is defined by the current measured velocity (falling back to
    yaw), so this target is orthogonal to V45's longitudinal acceleration target.
    No teacher value/action label is used.
    """
    cur = np.asarray(runtime.current_agents, dtype=np.float64)
    gt = np.asarray(logged_agents, dtype=np.float64)
    if agent_index >= len(cur) or gt.ndim != 3 or agent_index >= gt.shape[0] or gt.shape[1] <= 0:
        return 0.0
    st = cur[agent_index].reshape(-1)
    v0, hx, hy = _agent_velocity(st)
    nx, ny = -hy, hx
    dt = float((cfg.get("candidate", {}) or {}).get("step_s", 0.1))
    T = int(gt.shape[1])
    t = np.arange(1, T + 1, dtype=np.float64) * dt
    cvx = float(st[0]) + hx * v0 * t
    cvy = float(st[1]) + hy * v0 * t
    rx = gt[agent_index, :, 0] - cvx
    ry = gt[agent_index, :, 1] - cvy
    rlat = nx * rx + ny * ry
    good = np.isfinite(rlat) & np.isfinite(t)
    if not np.any(good):
        return 0.0
    den = float(np.dot(t[good], t[good]))
    if den <= EPS:
        return 0.0
    drift = float(np.dot(t[good], rlat[good]) / den)
    # Parameter-free physical containment: in the current local frame, the
    # learned lateral drift component cannot exceed the measured current
    # agent speed.  This uses no tuned V47 threshold and is applied to both
    # TRAIN target and runtime rollout.
    return float(np.clip(drift, -max(v0, 0.0), max(v0, 0.0)))


def _predict_lateral(local: np.ndarray, plan: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return local N and plan KxN lateral drift predictions."""
    model = (_fsfr_cfg(cfg).get("agent_lateral_response", {}) or {})
    N = int(local.shape[0])
    K = int(plan.shape[0])
    if not bool(model.get("enabled", False)):
        z = np.zeros((N,), dtype=np.float64)
        return z, np.zeros((K, N), dtype=np.float64)
    if [str(x) for x in model.get("local_feature_names", [])] != RESPONSE_FIELD_LOCAL_FEATURE_NAMES:
        raise ValueError("V47 FSFR lateral local feature schema mismatch")
    if [str(x) for x in model.get("plan_feature_names", [])] != RESPONSE_FIELD_PLAN_FEATURE_NAMES:
        raise ValueError("V47 FSFR lateral plan feature schema mismatch")
    ls = np.asarray(model.get("local_feature_scale", []), dtype=np.float64)
    lw = np.asarray(model.get("local_weights", []), dtype=np.float64)
    ps = np.asarray(model.get("plan_feature_scale", []), dtype=np.float64)
    pw = np.asarray(model.get("plan_weights", []), dtype=np.float64)
    if ls.size != local.shape[1] or lw.size != local.shape[1] or ps.size != plan.shape[2] or pw.size != plan.shape[2]:
        raise ValueError("V47 FSFR lateral model shape mismatch")
    local_v = local / np.maximum(ls[None, :], 1.0e-6) @ lw + float(model.get("local_bias", 0.0))
    base = np.broadcast_to(local_v[None, :], (K, N)).copy()
    if bool(model.get("plan_enabled", True)):
        base += np.tensordot(plan / np.maximum(ps[None, None, :], 1.0e-6), pw, axes=([2], [0]))
    return _finite(local_v), _finite(base)


def _agent_future_2d(st: np.ndarray, T: int, dt: float, accel: float, lateral_drift: float) -> np.ndarray:
    """Roll a continuous local-frame 2-D response with V45 longitudinal mean.

    Longitudinal motion is the frozen V45 constant-acceleration response.  The
    new component is a constant local-normal drift learned from TRAIN future.
    This is intentionally the smallest 2-D generalization after V46 closes
    second-moment/hand-profile rescue.
    """
    s = np.asarray(st, dtype=np.float64).reshape(-1)
    x0 = float(s[0]) if s.size else 0.0
    y0 = float(s[1]) if s.size > 1 else 0.0
    yaw0 = float(s[2]) if s.size > 2 and np.isfinite(s[2]) else 0.0
    v0, hx, hy = _agent_velocity(s)
    nx, ny = -hy, hx
    a = float(np.clip(accel, -2.0, 0.5))
    vl = float(np.nan_to_num(lateral_drift, nan=0.0, posinf=0.0, neginf=0.0))
    vl = float(np.clip(vl, -max(v0, 0.0), max(v0, 0.0)))
    t = np.arange(1, T + 1, dtype=np.float64) * float(dt)
    if a < 0.0 and v0 > 0.0:
        tstop = v0 / max(-a, 1.0e-9)
        te = np.minimum(t, tstop)
        along = v0 * te + 0.5 * a * te * te
        vlong = np.maximum(v0 + a * t, 0.0)
    else:
        along = v0 * t + 0.5 * a * t * t
        vlong = np.maximum(v0 + a * t, 0.0)
    lat = vl * t
    x = x0 + hx * along + nx * lat
    y = y0 + hy * along + ny * lat
    vx = hx * vlong + nx * vl
    vy = hy * vlong + ny * vl
    speed = np.hypot(vx, vy)
    yaw = np.where(speed > 1.0e-4, np.arctan2(vy, vx), yaw0)
    return np.stack([x, y, yaw, speed, t], axis=1)


def _occupancy_cost_2d(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
    long_accel: np.ndarray,
    lateral_drift: np.ndarray,
) -> np.ndarray:
    traj = np.asarray(candidates.trajectories, dtype=np.float64)
    K, T = int(traj.shape[0]), int(traj.shape[1])
    cur = np.asarray(runtime.current_agents, dtype=np.float64)
    valid = np.asarray(runtime.agent_valid, dtype=bool).reshape(-1)
    N = min(len(cur), len(valid))
    dt = float((cfg.get("candidate", {}) or {}).get("step_s", 0.1))
    times = _candidate_times(traj, dt)
    A = np.asarray(long_accel, dtype=np.float64)
    L = np.asarray(lateral_drift, dtype=np.float64)
    if A.shape != L.shape or A.shape[0] != K or A.shape[1] < N:
        raise ValueError("V47 FSFR 2-D response shape mismatch")
    occ = np.zeros((K, T), dtype=np.float64)
    for k in range(K):
        one = traj[k : k + 1]
        one_t = times[k : k + 1] if times.ndim == 2 else times
        for j in range(N):
            if not valid[j]:
                continue
            fut = _agent_future_2d(cur[j], T, dt, float(A[k, j]), float(L[k, j]))
            pot = _ungated_future_agent_occupancy(one, one_t, fut, cur[j], cfg)
            if pot.size:
                occ[k, : pot.shape[1]] = np.maximum(occ[k, : pot.shape[1]], pot[0])
    return _finite(np.mean(occ, axis=1) if T > 0 else np.zeros((K,), dtype=np.float64))


def _ego_current(runtime: RuntimeFeatures) -> np.ndarray:
    h = np.asarray(runtime.ego_history, dtype=np.float64)
    if h.ndim == 2 and h.shape[0] and h.shape[1] >= 4:
        return h[-1]
    return np.zeros((5,), dtype=np.float64)


def ego_reference_candidate_features(runtime: RuntimeFeatures, candidates: CandidateBank, cfg: dict[str, Any]) -> np.ndarray:
    """Kx9 current-only features for predicting the teacher demo component."""
    K = int(candidates.K)
    out = np.zeros((K, len(EGO_REFERENCE_FEATURE_NAMES)), dtype=np.float64)
    ego = _ego_current(runtime)
    x0, y0, yaw0, v0 = (float(ego[0]), float(ego[1]), float(ego[2]), float(ego[3]))
    tcfg = cfg.get("teacher", {}) if isinstance(cfg, dict) else {}
    base_dt = float((cfg.get("candidate", {}) or {}).get("step_s", 0.1))
    stride = max(1, int(tcfg.get("cost_eval_stride", 1)))
    dt = base_dt * stride
    route = np.asarray(runtime.map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float64).reshape(-1, 2)
    if route.size:
        route = route[np.isfinite(route).all(axis=1)]
    if len(route) < 2:
        route = np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float64)
    valid = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)
    terminal_progress = route_progress_along_polyline(np.asarray(candidates.trajectories[:, -1, :2], dtype=np.float64), route)
    best_progress = float(terminal_progress[valid].max()) if np.any(valid) else 0.0
    for a in range(K):
        tr = np.asarray(_eval_traj(candidates.trajectories[a], cfg), dtype=np.float64)
        n = len(tr)
        tt = np.arange(1, n + 1, dtype=np.float64) * dt
        cvx = x0 + v0 * np.cos(yaw0) * tt
        cvy = y0 + v0 * np.sin(yaw0) * tt
        d2 = (tr[:, 0] - cvx) ** 2 + (tr[:, 1] - cvy) ** 2
        speed = tr[:, 3] if tr.shape[1] > 3 else np.zeros((n,), dtype=np.float64)
        heading = tr[:, 2] if tr.shape[1] > 2 else np.full((n,), yaw0, dtype=np.float64)
        route_dev = float(np.square(nearest_polyline_distance(tr[:, :2], route)).mean()) if n else 0.0
        progress_def = float(max(0.0, best_progress - float(terminal_progress[a])))
        comfort = float(global_comfort_cost(tr, dt)) if n else 0.0
        out[a] = [
            float(np.mean(d2)) if n else 0.0,
            float(d2[-1]) if n else 0.0,
            float(np.mean((speed - v0) ** 2)) if n else 0.0,
            float((speed[-1] - v0) ** 2) if n else 0.0,
            float(np.mean(_wrap_angle(heading - yaw0) ** 2)) if n else 0.0,
            float(_wrap_angle(heading[-1] - yaw0) ** 2) if n else 0.0,
            route_dev,
            progress_def,
            comfort,
        ]
    return _finite(out)


def logged_demo_component_targets(runtime: RuntimeFeatures, logged_ego: np.ndarray, candidates: CandidateBank, cfg: dict[str, Any]) -> np.ndarray:
    """Exact TRAIN-only teacher demo component for each candidate."""
    tcfg = cfg.get("teacher", {}) if isinstance(cfg, dict) else {}
    w = float(tcfg.get("demo_weight", 1.0)) / max(float(tcfg.get("demo_scale", 120.0)), 1.0e-6)
    gt = _eval_traj(np.asarray(logged_ego, dtype=np.float64), cfg)
    out = np.zeros((int(candidates.K),), dtype=np.float64)
    for a in range(int(candidates.K)):
        out[a] = w * float(demo_cost(_eval_traj(candidates.trajectories[a], cfg), gt))
    return _finite(np.maximum(out, 0.0))


def _predict_demo(features: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    model = (_fsfr_cfg(cfg).get("ego_reference_model", {}) or {})
    K = int(features.shape[0])
    if not bool(model.get("enabled", False)):
        # Current-kinematic CV deviation is a deterministic runtime baseline.
        tcfg = cfg.get("teacher", {}) if isinstance(cfg, dict) else {}
        w = float(tcfg.get("demo_weight", 1.0)) / max(float(tcfg.get("demo_scale", 120.0)), 1.0e-6)
        return _finite(np.maximum(features[:, 0] * w, 0.0))
    if [str(x) for x in model.get("feature_names", [])] != EGO_REFERENCE_FEATURE_NAMES:
        raise ValueError("V47 FSFR ego-reference feature schema mismatch")
    scale = np.asarray(model.get("feature_scale", []), dtype=np.float64)
    weights = np.asarray(model.get("weights", []), dtype=np.float64)
    if scale.size != features.shape[1] or weights.size != features.shape[1]:
        raise ValueError("V47 FSFR ego-reference model shape mismatch")
    pred = features / np.maximum(scale[None, :], 1.0e-6) @ weights + float(model.get("bias", 0.0))
    return _finite(np.maximum(pred, 0.0))


def runtime_future_state_factorization_observable_costs(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Return Kx3 V47 current-only future-state costs."""
    K = int(candidates.K)
    if K <= 0:
        return np.zeros((0, len(FSFR_OBSERVABLE_NAMES)), dtype=np.float64), list(FSFR_OBSERVABLE_NAMES)
    local = response_field_local_agent_features(runtime, cfg)
    plan, _ = response_field_plan_agent_features(runtime, candidates, cfg)
    mean_model = _response_model_cfg(cfg)
    local_a = _predict_local_accel(local, mean_model)
    plan_a = _predict_plan_accel(local_a, plan, mean_model)
    plan_1d = _occupancy_cost_for_accels(runtime, candidates, cfg, plan_a)
    _, lateral_plan = _predict_lateral(local, plan, cfg)
    plan_2d = _occupancy_cost_2d(runtime, candidates, cfg, plan_a, lateral_plan)
    ego_feat = ego_reference_candidate_features(runtime, candidates, cfg)
    demo_hat = _predict_demo(ego_feat, cfg)
    out = np.stack([plan_1d, plan_2d, demo_hat], axis=1).astype(np.float64)
    if out.shape != (K, len(FSFR_OBSERVABLE_NAMES)) or not np.all(np.isfinite(out)):
        raise ValueError("V47 FSFR observable matrix malformed/non-finite")
    return out, list(FSFR_OBSERVABLE_NAMES)
