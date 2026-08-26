from __future__ import annotations

"""V64.3.46 Distributional Interaction Response Profile (DIRP).

V45 identifies an agent-local, ego-plan-conditioned *mean* longitudinal
response, but a point response still discards conditional response dispersion and
then collapses the future interaction trace to one time-average scalar.  DIRP
keeps the V45 mean field frozen in semantics and adds two orthogonal pieces:

1. a conditional second-moment field for response acceleration, and
2. a bounded temporal interaction-hazard profile.

The second moment is propagated through a fixed three-point moment-matching
quadrature (weights 1/6, 2/3, 1/6 at mu +/- sqrt(3) sigma).  These constants are
not tuned; the rule matches the first four moments of a Gaussian standard
variable through degree three and exactly preserves mean/variance.  Runtime
uses only current state, already generated candidates, and frozen TRAIN
parameters.  Logged future is never consumed at deployment.
"""

from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.interaction_response_field import (
    EPS,
    RESPONSE_ACCEL_MAX,
    RESPONSE_ACCEL_MIN,
    RESPONSE_FIELD_LOCAL_FEATURE_NAMES,
    RESPONSE_FIELD_PLAN_FEATURE_NAMES,
    _cv_agent_future,
    _finite,
    _occupancy_profile_for_accels,
    _predict_local_accel,
    _predict_plan_accel,
    _response_model_cfg,
    response_field_local_agent_features,
    response_field_plan_agent_features,
)
from bdse.planner.response_value_observables import _candidate_times, _ungated_future_agent_occupancy

DIRP_OBSERVABLE_NAMES = [
    "dirp_plan_mean_occupancy_cost",
    "dirp_distribution_mean_occupancy_cost",
    "dirp_plan_peak_occupancy_cost",
    "dirp_plan_early_occupancy_cost",
    "dirp_plan_second_moment_occupancy_cost",
    "dirp_distribution_peak_occupancy_cost",
    "dirp_distribution_early_occupancy_cost",
    "dirp_distribution_second_moment_occupancy_cost",
]

SIGMA_POINT_OFFSETS = np.asarray([-np.sqrt(3.0), 0.0, np.sqrt(3.0)], dtype=np.float64)
SIGMA_POINT_WEIGHTS = np.asarray([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=np.float64)


def _dirp_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    ic = (((cfg.get("runtime", {}) or {}).get("decisive_frontier_value", {}) or {}).get("incumbent_contrastive_extremal_recovery", {}) or {})
    sc = ic.get("selection_conditioned_intervention_recovery", {}) or {}
    return sc.get("distributional_interaction_response_field", {}) or {}


def _predict_second_moment(local: np.ndarray, plan: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Predict KxN conditional E[a^2 | current state, ego candidate]."""
    model = _dirp_cfg(cfg)
    K, N, P = plan.shape
    if not bool(model.get("enabled", False)):
        return np.zeros((K, N), dtype=np.float64)
    lnames = [str(x) for x in model.get("local_feature_names", [])]
    pnames = [str(x) for x in model.get("plan_feature_names", [])]
    if lnames != RESPONSE_FIELD_LOCAL_FEATURE_NAMES or pnames != RESPONSE_FIELD_PLAN_FEATURE_NAMES:
        raise ValueError("V46 DIRP second-moment feature schema mismatch")
    ls = np.asarray(model.get("local_feature_scale", []), dtype=np.float64).reshape(-1)
    lw = np.asarray(model.get("local_weights", []), dtype=np.float64).reshape(-1)
    lb = float(model.get("local_bias", 0.0))
    ps = np.asarray(model.get("plan_feature_scale", []), dtype=np.float64).reshape(-1)
    pw = np.asarray(model.get("plan_weights", []), dtype=np.float64).reshape(-1)
    if ls.size != local.shape[1] or lw.size != local.shape[1] or ps.size != P or pw.size != P:
        raise ValueError("V46 DIRP second-moment model shape mismatch")
    local_m2 = np.maximum(0.0, local / np.maximum(ls[None, :], 1.0e-6) @ lw + lb)
    base = np.broadcast_to(local_m2[None, :], (K, N)).copy()
    if bool(model.get("plan_enabled", True)):
        # Zero-bias plan correction.  V45 plan features all vanish continuously
        # with interaction exposure, so the second-moment correction inherits the
        # same zero-at-zero-interaction property.
        corr = np.tensordot(plan / np.maximum(ps[None, None, :], 1.0e-6), pw, axes=([2], [0]))
        base = base + corr
    return _finite(np.maximum(base, 0.0))


def _profile_functionals(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return bounded lower-is-better temporal hazard functionals.

    mean: total interaction mass (V45 statistic),
    peak: worst instantaneous interaction,
    early: first-horizon-weighted interaction mass, and
    second: squared interaction mass, retaining concentration without a new
    threshold or bandwidth.
    """
    H = np.asarray(profile, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError("V46 DIRP occupancy profile must be KxT")
    K, T = H.shape
    if T <= 0:
        z = np.zeros((K,), dtype=np.float64)
        return z, z.copy(), z.copy(), z.copy()
    tau = np.linspace(0.0, 1.0, T, dtype=np.float64)
    early_w = 1.0 - tau
    mean = np.mean(H, axis=1)
    peak = np.max(H, axis=1)
    early = np.mean(H * early_w[None, :], axis=1)
    second = np.mean(H * H, axis=1)
    return tuple(_finite(x) for x in (mean, peak, early, second))


def _distributional_profile(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
    mean_accel: np.ndarray,
    second_moment: np.ndarray,
) -> np.ndarray:
    """Expected per-agent occupancy, then worst interactor, over fixed sigma points."""
    traj = np.asarray(candidates.trajectories, dtype=np.float64)
    K, T = int(traj.shape[0]), int(traj.shape[1])
    cur = np.asarray(runtime.current_agents, dtype=np.float64)
    valid = np.asarray(runtime.agent_valid, dtype=bool).reshape(-1)
    N = min(len(cur), len(valid))
    dt = float((cfg.get("candidate", {}) or {}).get("step_s", 0.1))
    times = _candidate_times(traj, dt)
    mu = np.asarray(mean_accel, dtype=np.float64)
    m2 = np.asarray(second_moment, dtype=np.float64)
    if mu.shape[0] != K or m2.shape != mu.shape or mu.shape[1] < N:
        raise ValueError("V46 DIRP response moment shape mismatch")
    var = np.maximum(m2 - mu * mu, 0.0)
    sigma = np.sqrt(var)
    out = np.zeros((K, T), dtype=np.float64)
    for k in range(K):
        one = traj[k : k + 1]
        one_t = times[k : k + 1] if times.ndim == 2 else times
        for j in range(N):
            if not valid[j]:
                continue
            epot = np.zeros((T,), dtype=np.float64)
            for off, w in zip(SIGMA_POINT_OFFSETS, SIGMA_POINT_WEIGHTS):
                a = float(np.clip(mu[k, j] + off * sigma[k, j], RESPONSE_ACCEL_MIN, RESPONSE_ACCEL_MAX))
                fut = _cv_agent_future(cur[j], T, dt, a)
                pot = _ungated_future_agent_occupancy(one, one_t, fut, cur[j], cfg)
                if pot.size:
                    epot[: pot.shape[1]] += float(w) * pot[0]
            out[k] = np.maximum(out[k], epot)
    return _finite(out)


def runtime_distributional_interaction_response_observable_costs(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Return Kx8 V46 response-distribution/profile costs."""
    K = int(candidates.K)
    if K <= 0:
        return np.zeros((0, len(DIRP_OBSERVABLE_NAMES)), dtype=np.float64), list(DIRP_OBSERVABLE_NAMES)
    local = response_field_local_agent_features(runtime, cfg)
    plan, _ = response_field_plan_agent_features(runtime, candidates, cfg)
    mean_model = _response_model_cfg(cfg)
    local_a = _predict_local_accel(local, mean_model)
    plan_a = _predict_plan_accel(local_a, plan, mean_model)
    m2 = _predict_second_moment(local, plan, cfg)

    plan_profile = _occupancy_profile_for_accels(runtime, candidates, cfg, plan_a)
    dist_profile = _distributional_profile(runtime, candidates, cfg, plan_a, m2)
    pm, pp, pe, ps = _profile_functionals(plan_profile)
    dm, dp, de, ds = _profile_functionals(dist_profile)

    out = np.stack([pm, dm, pp, pe, ps, dp, de, ds], axis=1)
    if out.shape != (K, len(DIRP_OBSERVABLE_NAMES)) or not np.all(np.isfinite(out)):
        raise ValueError("V46 DIRP observable matrix malformed/non-finite")
    return out.astype(np.float64), list(DIRP_OBSERVABLE_NAMES)
