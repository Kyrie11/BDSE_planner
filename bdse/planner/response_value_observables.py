from __future__ import annotations

"""Runtime-only prospective response observables for V64.3.43 CFRV.

The V42 current-agent risk block extrapolates every tracked agent with a single
constant-velocity envelope.  V43 explicitly represents *future response
uncertainty* without using logged future labels: current agent states are rolled
under the frozen teacher response hypotheses {cv, ca, brake, yield, nonyield}.
For every candidate trajectory we evaluate the same continuous box-aware
interaction semantics under each response mode and expose three lower-is-better
functionals of the induced cost distribution:

  mean, CVaR_alpha, and the frozen teacher-style mean/CVaR mixture.

The logged response mode is impossible when label_future=None and is therefore
excluded by construction.  No neural evidence query, teacher label, or future
log is consumed at deployment.
"""

from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.response_modes import build_response_modes
from bdse.planner.robust_teacher import weighted_cvar

FUTURE_RESPONSE_OBSERVABLE_NAMES = [
    "future_response_mean_agent_cost",
    "future_response_cvar_agent_cost",
    "future_response_robust_agent_cost",
]


def _finite(x: np.ndarray, limit: float = 1.0e6) -> np.ndarray:
    return np.clip(np.nan_to_num(np.asarray(x, dtype=np.float64), nan=limit, posinf=limit, neginf=0.0), 0.0, limit)


def _candidate_times(traj: np.ndarray, dt: float) -> np.ndarray:
    K, T = int(traj.shape[0]), int(traj.shape[1])
    if traj.shape[2] > 4:
        t = np.asarray(traj[:, :, 4], dtype=np.float64)
        if np.all(np.isfinite(t)) and np.all(t >= -1.0e-6):
            return t
    return np.broadcast_to((np.arange(T, dtype=np.float64) + 1.0)[None, :] * dt, (K, T)).copy()


def _future_agent_envelope(
    traj: np.ndarray,
    times: np.ndarray,
    agent_future: np.ndarray,
    current_agent: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous K-vector hard/soft/TTC severities against one future agent.

    This is the future-trajectory analogue of fallback._agent_envelope_metrics:
    the geometry is box-aware when dimensions are available and uses the same
    frozen runtime-safety clearances/weights.  It does not discretize to a hard
    veto; the output remains a smooth lower-is-better consequence observable.
    """

    arr = np.asarray(traj, dtype=np.float64)
    fut = np.asarray(agent_future, dtype=np.float64)
    cur = np.asarray(current_agent, dtype=np.float64).reshape(-1)
    K, T = int(arr.shape[0]), int(arr.shape[1])
    Te = min(T, int(fut.shape[0]))
    if Te <= 0:
        z = np.zeros((K,), dtype=np.float64)
        return z, z.copy(), z.copy()
    arr = arr[:, :Te]
    fut = fut[:Te]
    tt = times[:, :Te]
    rsc = cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}

    ax = fut[:, 0][None, :]
    ay = fut[:, 1][None, :]
    ayaw = fut[:, 2][None, :] if fut.shape[1] > 2 else np.zeros((1, Te), dtype=np.float64)
    av = fut[:, 3][None, :] if fut.shape[1] > 3 else np.zeros((1, Te), dtype=np.float64)
    dx = arr[:, :, 0] - ax
    dy = arr[:, :, 1] - ay
    dist = np.sqrt(np.maximum(dx * dx + dy * dy, 1.0e-8))

    ev = arr[:, :, 3] if arr.shape[2] > 3 else np.zeros((K, Te), dtype=np.float64)
    eyaw = arr[:, :, 2] if arr.shape[2] > 2 else np.zeros((K, Te), dtype=np.float64)
    evx, evy = ev * np.cos(eyaw), ev * np.sin(eyaw)
    avx, avy = av * np.cos(ayaw), av * np.sin(ayaw)
    rel_vx, rel_vy = evx - avx, evy - avy
    closing = np.maximum(-(dx * rel_vx + dy * rel_vy) / np.maximum(dist, 1.0e-4), 0.0)
    ttc = np.where(closing > 1.0e-3, dist / np.maximum(closing, 1.0e-3), np.inf)

    # Keep the same horizon semantics as runtime risk, but derive the mask from
    # candidate clock time so the prospective modes do not add a new horizon knob.
    base_hard = float(rsc.get("hard_check_horizon_s", float("inf")))
    base_soft = float(rsc.get("soft_check_horizon_s", float("inf")))
    hard_mask = np.isfinite(tt) & (tt <= base_hard + 1.0e-6)
    soft_mask = np.isfinite(tt) & (tt <= base_soft + 1.0e-6)
    if bool(rsc.get("speed_adaptive_horizon", False)) and arr.shape[2] > 3:
        speed = np.maximum(np.nanmax(np.maximum(arr[:, :, 3], 0.0), axis=1), 0.0)
        reaction = max(float(rsc.get("reaction_time_s", 0.7)), 0.0)
        decel = max(float(rsc.get("comfortable_emergency_decel_mps2", 5.0)), 0.5)
        margin = max(float(rsc.get("stopping_horizon_margin_s", 0.35)), 0.0)
        min_h = float(rsc.get("min_hard_horizon_s", base_hard))
        max_h = float(rsc.get("max_hard_horizon_s", max(base_soft, base_hard)))
        hh = np.clip(reaction + speed / decel + margin, min_h, max_h)
        soft_extra = max(float(rsc.get("soft_horizon_extra_s", 1.5)), 0.0)
        max_soft = float(rsc.get("max_soft_horizon_s", max(base_soft, max_h)))
        sh = np.minimum(np.maximum(np.full((K,), base_soft), hh + soft_extra), max_soft)
        hard_mask = np.isfinite(tt) & (tt <= hh[:, None] + 1.0e-6)
        soft_mask = np.isfinite(tt) & (tt <= sh[:, None] + 1.0e-6)
    if Te:
        hard_mask[:, 0] = True
        soft_mask[:, 0] = True

    has_dims = cur.size > 8 and float(cur[7]) > 0.0 and float(cur[8]) > 0.0
    if bool(rsc.get("use_box_agent_risk", False)) and has_dims:
        ego_l = max(float(rsc.get("ego_length_m", 4.8)), 0.5)
        ego_w = max(float(rsc.get("ego_width_m", 2.0)), 0.3)
        agent_l = max(float(cur[7]), 0.3)
        agent_w = max(float(cur[8]), 0.2)
        c, s = np.cos(ayaw), np.sin(ayaw)
        longitudinal = c * dx + s * dy
        lateral = -s * dx + c * dy
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
        hard_def = np.where(hard_mask, np.maximum(1.0 - hard_norm, 0.0), 0.0).max(axis=1)
        soft_def = np.where(soft_mask, np.maximum(1.0 - soft_norm, 0.0), 0.0).max(axis=1)
        ttc_gate = soft_norm <= float(rsc.get("ttc_envelope_gate", 1.35))
    else:
        hard_r = max(float(rsc.get("hard_agent_radius_m", 0.85)), 1.0e-3)
        soft_r = max(float(rsc.get("soft_agent_radius_m", 1.5)), hard_r)
        hard_def = np.where(hard_mask, np.maximum(hard_r - dist, 0.0) / hard_r, 0.0).max(axis=1)
        soft_def = np.where(soft_mask, np.maximum(soft_r - dist, 0.0) / soft_r, 0.0).max(axis=1)
        ttc_gate = dist <= soft_r * float(rsc.get("ttc_envelope_gate", 1.35))

    ttc_eval = np.where(soft_mask & ttc_gate, ttc, np.inf)
    min_ttc = np.min(ttc_eval, axis=1)
    safe_ttc = max(float(rsc.get("agent_ttc_safe_s", 3.0)), 1.0e-3)
    ttc_risk = np.maximum((safe_ttc - min_ttc) / safe_ttc, 0.0)
    ttc_risk = np.where(np.isfinite(ttc_risk), ttc_risk, 0.0)
    return _finite(hard_def), _finite(soft_def), _finite(ttc_risk)


def runtime_future_response_observable_costs(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Return K x 3 label-free prospective response-distribution costs."""

    K = int(candidates.K)
    if K <= 0:
        return np.zeros((0, 3), dtype=np.float64), list(FUTURE_RESPONSE_OBSERVABLE_NAMES)
    traj = np.nan_to_num(np.asarray(candidates.trajectories, dtype=np.float64), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1)) if isinstance(cfg, dict) else 0.1
    times = _candidate_times(traj, dt)
    modes = build_response_modes(runtime, None, cfg)
    if not modes or any(bool(m.metadata.get("uses_label_future", False)) or str(m.name) == "logged" for m in modes):
        raise ValueError("V43 future-response observable must be runtime-only and exclude logged future mode")
    valid_agents = np.asarray(runtime.agent_valid, dtype=bool).reshape(-1)
    cur_agents = np.asarray(runtime.current_agents, dtype=np.float64)
    rsc = cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}
    wh = float(rsc.get("risk_hard_agent_weight", 6.0))
    ws = float(rsc.get("risk_soft_agent_weight", 1.2))
    wt = float(rsc.get("risk_hard_ttc_weight", 2.5)) + float(rsc.get("risk_soft_ttc_weight", 1.0))

    mode_costs = []
    probs = []
    for mode in modes:
        hard = np.zeros((K,), dtype=np.float64)
        soft = np.zeros((K,), dtype=np.float64)
        ttc = np.zeros((K,), dtype=np.float64)
        futures = np.asarray(mode.agent_futures, dtype=np.float64)
        n = min(len(valid_agents), len(cur_agents), int(futures.shape[0]) if futures.ndim >= 3 else 0)
        for j in range(n):
            if not bool(valid_agents[j]):
                continue
            hj, sj, tj = _future_agent_envelope(traj, times, futures[j], cur_agents[j], cfg)
            hard = np.maximum(hard, hj)
            soft = np.maximum(soft, sj)
            ttc = np.maximum(ttc, tj)
        mode_costs.append(_finite(wh * hard + ws * soft + wt * ttc))
        probs.append(float(mode.probability))

    stack = np.stack(mode_costs, axis=0) if mode_costs else np.zeros((1, K), dtype=np.float64)
    p = np.asarray(probs if probs else [1.0], dtype=np.float64)
    p = p / max(float(p.sum()), 1.0e-12)
    mean = np.tensordot(p, stack, axes=(0, 0)).astype(np.float64)
    rcfg = cfg.get("teacher", {}).get("risk_aggregation", {}) if isinstance(cfg, dict) else {}
    alpha = float(rcfg.get("cvar_alpha", cfg.get("teacher", {}).get("cvar_alpha", 0.9) if isinstance(cfg, dict) else 0.9))
    mix = float(rcfg.get("cvar_weight", cfg.get("teacher", {}).get("cvar_weight", 0.4) if isinstance(cfg, dict) else 0.4))
    cvar = np.asarray(weighted_cvar(stack.astype(np.float32), p.astype(np.float32), alpha), dtype=np.float64)
    robust = (1.0 - mix) * mean + mix * cvar
    out = np.stack([_finite(mean), _finite(cvar), _finite(robust)], axis=1)
    valid = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)[:K]
    # Invalid actions cannot be selected; keep finite values for instrumentation.
    out[~valid] = np.maximum(out[~valid], 0.0)
    if out.shape != (K, 3) or not np.all(np.isfinite(out)):
        raise ValueError("V43 future-response observable matrix malformed or non-finite")
    return out.astype(np.float64), list(FUTURE_RESPONSE_OBSERVABLE_NAMES)
