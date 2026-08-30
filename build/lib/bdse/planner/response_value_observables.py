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

# ---------------------------------------------------------------------------
# V64.3.44 PCOR: plan-conditioned response posterior + ungated occupancy support
# ---------------------------------------------------------------------------

RUNTIME_RESPONSE_MODE_NAMES = ["cv", "ca", "brake", "yield", "nonyield"]
PLAN_RESPONSE_RAW_MODE_COST_NAMES = [f"future_response_mode_cost_{m}" for m in RUNTIME_RESPONSE_MODE_NAMES]
PLAN_RESPONSE_RAW_OCCUPANCY_NAMES = [f"future_response_mode_occupancy_{m}" for m in RUNTIME_RESPONSE_MODE_NAMES]
PLAN_RESPONSE_CONDITIONING_NAMES = PLAN_RESPONSE_RAW_MODE_COST_NAMES + PLAN_RESPONSE_RAW_OCCUPANCY_NAMES
PLAN_CONDITIONED_RESPONSE_OBSERVABLE_NAMES = [
    "plan_conditioned_response_mean_agent_cost",
    "plan_conditioned_response_robust_agent_cost",
    "plan_conditioned_occupancy_mean_cost",
    "plan_conditioned_occupancy_robust_cost",
]
PLAN_CONDITIONED_RESPONSE_ALL_NAMES = PLAN_RESPONSE_CONDITIONING_NAMES + PLAN_CONDITIONED_RESPONSE_OBSERVABLE_NAMES


def _candidate_specific_weighted_cvar(values: np.ndarray, probs: np.ndarray, alpha: float) -> np.ndarray:
    """CVaR for MxK costs with candidate-specific MxK probabilities.

    This is the candidate-conditioned analogue of robust_teacher.weighted_cvar.
    M is tiny (five frozen response modes), so an explicit K loop is preferable
    to introducing another approximation or dependency.
    """
    vals = np.asarray(values, dtype=np.float64)
    pp = np.asarray(probs, dtype=np.float64)
    if vals.ndim != 2 or pp.shape != vals.shape:
        raise ValueError("candidate-specific CVaR expects matching MxK value/probability matrices")
    M, K = vals.shape
    out = np.zeros((K,), dtype=np.float64)
    q = float(np.clip(alpha, 0.0, 0.999))
    for k in range(K):
        p = np.maximum(pp[:, k], 0.0)
        p = p / max(float(p.sum()), 1.0e-12)
        order = np.argsort(vals[:, k], kind="mergesort")
        v = vals[order, k]
        p = p[order]
        c = np.cumsum(p)
        start = int(np.searchsorted(c, q, side="left"))
        if start >= M:
            out[k] = float(v[-1])
            continue
        tv = v[start:]
        tp = p[start:].copy()
        prev = float(c[start - 1]) if start > 0 else 0.0
        tp[0] = max(float(c[start]) - q, 0.0)
        denom = max(float(tp.sum()), 1.0e-12)
        out[k] = float(np.dot(tv, tp) / denom)
    return _finite(out)


def _ungated_future_agent_occupancy(
    traj: np.ndarray,
    times: np.ndarray,
    agent_future: np.ndarray,
    current_agent: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Long-range continuous interaction potential with no hard/soft cutoff.

    V43's hard/soft/TTC severities are exactly zero outside a finite envelope.
    V44 keeps those observables untouched, but additionally exposes a bounded
    occupancy potential over the *full candidate horizon*.  The normalization is
    inherited from the existing box/radius semantics, so there is no new distance
    threshold or kernel-bandwidth hyperparameter:

        potential = 1 / (1 + normalized_separation^2).

    This quantity is not a safety veto.  It is a smooth lower-is-better support
    statistic whose purpose is to retain weak prospective interaction evidence
    before the V43 gated risk turns on.
    """
    arr = np.asarray(traj, dtype=np.float64)
    fut = np.asarray(agent_future, dtype=np.float64)
    cur = np.asarray(current_agent, dtype=np.float64).reshape(-1)
    K, T = int(arr.shape[0]), int(arr.shape[1])
    Te = min(T, int(fut.shape[0]))
    if Te <= 0:
        return np.zeros((K, 0), dtype=np.float64)
    arr = arr[:, :Te]
    fut = fut[:Te]
    ax = fut[:, 0][None, :]
    ay = fut[:, 1][None, :]
    ayaw = fut[:, 2][None, :] if fut.shape[1] > 2 else np.zeros((1, Te), dtype=np.float64)
    dx = arr[:, :, 0] - ax
    dy = arr[:, :, 1] - ay
    rsc = cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}
    has_dims = cur.size > 8 and float(cur[7]) > 0.0 and float(cur[8]) > 0.0
    if bool(rsc.get("use_box_agent_risk", False)) and has_dims:
        ego_l = max(float(rsc.get("ego_length_m", 4.8)), 0.5)
        ego_w = max(float(rsc.get("ego_width_m", 2.0)), 0.3)
        agent_l = max(float(cur[7]), 0.3)
        agent_w = max(float(cur[8]), 0.2)
        c, s = np.cos(ayaw), np.sin(ayaw)
        longitudinal = c * dx + s * dy
        lateral = -s * dx + c * dy
        # Use the already-frozen soft envelope dimensions as the unit scale, but
        # do not truncate outside the envelope.
        soft_l = 0.5 * (ego_l + agent_l) + float(rsc.get("hard_longitudinal_clearance_m", 0.20)) + float(rsc.get("soft_longitudinal_extra_m", 1.00))
        soft_w = 0.5 * (ego_w + agent_w) + float(rsc.get("hard_lateral_clearance_m", 0.15)) + float(rsc.get("soft_lateral_extra_m", 0.65))
        norm2 = (longitudinal / max(float(soft_l), 0.1)) ** 2 + (lateral / max(float(soft_w), 0.1)) ** 2
    else:
        soft_r = max(float(rsc.get("soft_agent_radius_m", 1.5)), 1.0e-3)
        norm2 = (dx * dx + dy * dy) / (soft_r * soft_r)
    return _finite(1.0 / (1.0 + np.maximum(norm2, 0.0)))


def runtime_plan_response_mode_features(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Return Kx10 raw candidate-conditioned response features.

    The first five columns are the original V43 gated interaction costs evaluated
    separately for each frozen runtime-only response mode.  The last five are
    full-horizon ungated occupancy potentials for the same modes.  They are raw
    observables, not learned values, and can be emitted before a response-posterior
    model has been fitted.
    """
    from bdse.planner.response_modes import _roll_current_agents

    K = int(candidates.K)
    if K <= 0:
        return np.zeros((0, len(PLAN_RESPONSE_CONDITIONING_NAMES)), dtype=np.float64), list(PLAN_RESPONSE_CONDITIONING_NAMES)
    traj = np.nan_to_num(np.asarray(candidates.trajectories, dtype=np.float64), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1)) if isinstance(cfg, dict) else 0.1
    times = _candidate_times(traj, dt)
    valid_agents = np.asarray(runtime.agent_valid, dtype=bool).reshape(-1)
    cur_agents = np.asarray(runtime.current_agents, dtype=np.float64)
    rsc = cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}
    wh = float(rsc.get("risk_hard_agent_weight", 6.0))
    ws = float(rsc.get("risk_soft_agent_weight", 1.2))
    wt = float(rsc.get("risk_hard_ttc_weight", 2.5)) + float(rsc.get("risk_soft_ttc_weight", 1.0))
    T = int(traj.shape[1])

    gated_cols: list[np.ndarray] = []
    occ_cols: list[np.ndarray] = []
    for mode_name in RUNTIME_RESPONSE_MODE_NAMES:
        futures = np.asarray(_roll_current_agents(runtime, T, dt, mode_name), dtype=np.float64)
        hard = np.zeros((K,), dtype=np.float64)
        soft = np.zeros((K,), dtype=np.float64)
        ttc = np.zeros((K,), dtype=np.float64)
        occ_time = np.zeros((K, T), dtype=np.float64)
        n = min(len(valid_agents), len(cur_agents), int(futures.shape[0]) if futures.ndim >= 3 else 0)
        for j in range(n):
            if not bool(valid_agents[j]):
                continue
            hj, sj, tj = _future_agent_envelope(traj, times, futures[j], cur_agents[j], cfg)
            hard = np.maximum(hard, hj)
            soft = np.maximum(soft, sj)
            ttc = np.maximum(ttc, tj)
            pj = _ungated_future_agent_occupancy(traj, times, futures[j], cur_agents[j], cfg)
            if pj.size:
                occ_time[:, : pj.shape[1]] = np.maximum(occ_time[:, : pj.shape[1]], pj)
        gated_cols.append(_finite(wh * hard + ws * soft + wt * ttc))
        # Mean occupancy over the existing full candidate horizon.  Max over agents
        # preserves the original worst-interactor semantics without a new top-K.
        occ_cols.append(_finite(occ_time.mean(axis=1) if T > 0 else np.zeros((K,), dtype=np.float64)))
    out = np.stack(gated_cols + occ_cols, axis=1).astype(np.float64)
    if out.shape != (K, len(PLAN_RESPONSE_CONDITIONING_NAMES)) or not np.all(np.isfinite(out)):
        raise ValueError("V44 raw response-conditioning feature matrix malformed or non-finite")
    return out, list(PLAN_RESPONSE_CONDITIONING_NAMES)


def _fixed_runtime_mode_prior(cfg: dict[str, Any]) -> np.ndarray:
    from bdse.planner.response_modes import _mode_probs
    probs = _mode_probs(cfg)
    p = np.asarray([float(probs.get(m, 0.0)) for m in RUNTIME_RESPONSE_MODE_NAMES], dtype=np.float64)
    if float(p.sum()) <= 0.0:
        p[0] = 1.0
    return p / max(float(p.sum()), 1.0e-12)


def _plan_conditioned_mode_probabilities(raw_features: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Return Kx5 candidate-conditioned mode probabilities.

    If no trained posterior is present, this intentionally falls back to the V43
    fixed runtime prior.  This gives V44 a stable instrumentation schema before the
    TRAIN-only behavior model is fitted and makes exact V43 replay a hard gate.
    """
    X = np.asarray(raw_features, dtype=np.float64)
    K = int(X.shape[0])
    prior = _fixed_runtime_mode_prior(cfg)
    ic = (((cfg.get("runtime", {}) or {}).get("decisive_frontier_value", {}) or {}).get("incumbent_contrastive_extremal_recovery", {}) or {}) if isinstance(cfg, dict) else {}
    sc = ic.get("selection_conditioned_intervention_recovery", {}) or {}
    model = sc.get("plan_conditioned_response_posterior", {}) or {}
    if not bool(model.get("enabled", False)):
        return np.broadcast_to(prior[None, :], (K, len(prior))).copy()
    names = [str(x) for x in model.get("feature_names", [])]
    if names != PLAN_RESPONSE_CONDITIONING_NAMES:
        raise ValueError("V44 plan-conditioned response posterior feature schema mismatch")
    scale = np.asarray(model.get("feature_scale", []), dtype=np.float64).reshape(-1)
    weights = np.asarray(model.get("weights", []), dtype=np.float64)
    bias = np.asarray(model.get("bias", []), dtype=np.float64).reshape(-1)
    if scale.size != X.shape[1] or weights.shape != (X.shape[1], len(RUNTIME_RESPONSE_MODE_NAMES)) or bias.size != len(RUNTIME_RESPONSE_MODE_NAMES):
        raise ValueError("V44 plan-conditioned response posterior parameter shape mismatch")
    Z = X / np.maximum(scale[None, :], 1.0e-6)
    logits = Z @ weights + bias[None, :]
    logits = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(np.clip(logits, -60.0, 60.0))
    den = np.maximum(e.sum(axis=1, keepdims=True), 1.0e-12)
    return (e / den).astype(np.float64)


def runtime_plan_conditioned_response_observable_costs(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Return raw mode features plus four PCOR response/occupancy functionals."""
    raw, raw_names = runtime_plan_response_mode_features(runtime, candidates, cfg)
    K = int(raw.shape[0])
    M = len(RUNTIME_RESPONSE_MODE_NAMES)
    gated = raw[:, :M].T  # MxK
    occ = raw[:, M:].T
    probs_km = _plan_conditioned_mode_probabilities(raw, cfg)  # KxM
    probs = probs_km.T
    mean_g = np.sum(probs * gated, axis=0)
    mean_o = np.sum(probs * occ, axis=0)
    rcfg = cfg.get("teacher", {}).get("risk_aggregation", {}) if isinstance(cfg, dict) else {}
    alpha = float(rcfg.get("cvar_alpha", cfg.get("teacher", {}).get("cvar_alpha", 0.9) if isinstance(cfg, dict) else 0.9))
    mix = float(rcfg.get("cvar_weight", cfg.get("teacher", {}).get("cvar_weight", 0.4) if isinstance(cfg, dict) else 0.4))
    cvar_g = _candidate_specific_weighted_cvar(gated, probs, alpha)
    cvar_o = _candidate_specific_weighted_cvar(occ, probs, alpha)
    robust_g = (1.0 - mix) * mean_g + mix * cvar_g
    robust_o = (1.0 - mix) * mean_o + mix * cvar_o
    derived = np.stack([_finite(mean_g), _finite(robust_g), _finite(mean_o), _finite(robust_o)], axis=1)
    out = np.concatenate([raw, derived], axis=1)
    names = list(raw_names) + list(PLAN_CONDITIONED_RESPONSE_OBSERVABLE_NAMES)
    if out.shape != (K, len(names)) or not np.all(np.isfinite(out)):
        raise ValueError("V44 plan-conditioned response observable matrix malformed or non-finite")
    return out.astype(np.float64), names
