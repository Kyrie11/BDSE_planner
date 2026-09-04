from __future__ import annotations

"""V64.3.53 POTR: Paired Operator-Trajectory Retention.

V52 showed that QPE+D reliably identifies whether the one-shot proposal has any
closed-loop effect, but not the direction/order of the effect once structural
nulls are removed.  POTR therefore freezes the V52 scalar effect-support layer
and changes only the conditional-outcome state.

Two preregistered state families are supported:
  * endpoint: QPE+D plus the signed terminal proposal-incumbent contrast;
  * temporal: endpoint state plus two fixed cosine-basis coefficients per
    signed physical contrast channel.  No learned attention, horizon sweep,
    peak/early statistic, or offline future observable is introduced.
"""

from typing import Any
import math
import numpy as np

from bdse.planner.paired_operator_contrast_retention import (
    POCR_ADDITIVE_STATE_NAMES,
    execution_contrast_linf,
)

CHANNEL_NAMES = ["dx_m", "dy_m", "dyaw_rad", "dv_mps"]
ENDPOINT_EXTRA_NAMES = [f"operator_terminal_{n}" for n in CHANNEL_NAMES]
TEMPORAL_EXTRA_NAMES = [f"operator_cos{k}_{n}" for k in (1, 2) for n in CHANNEL_NAMES]
POTR_SUPPORT_STATE_NAMES = list(POCR_ADDITIVE_STATE_NAMES)
POTR_ENDPOINT_STATE_NAMES = POTR_SUPPORT_STATE_NAMES + ENDPOINT_EXTRA_NAMES
POTR_TEMPORAL_STATE_NAMES = POTR_ENDPOINT_STATE_NAMES + TEMPORAL_EXTRA_NAMES
POTR_STATE_FAMILIES = {"endpoint", "temporal"}
PROFILE_SCHEMA = "v64.3.53-operator-trajectory-contrast-v1"


def _signed_channels(proposal_trajectory: np.ndarray, incumbent_trajectory: np.ndarray) -> np.ndarray:
    p = np.asarray(proposal_trajectory, dtype=np.float64)
    i = np.asarray(incumbent_trajectory, dtype=np.float64)
    if p.ndim != 2 or i.ndim != 2 or p.shape != i.shape or p.shape[0] < 2 or p.shape[1] < 5:
        raise ValueError(f"V53 POTR trajectory shape mismatch {p.shape} vs {i.shape}; expected [T,5+] with T>=2")
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(i)):
        raise ValueError("V53 POTR trajectory contrast contains non-finite values")
    dx = p[:, 0] - i[:, 0]
    dy = p[:, 1] - i[:, 1]
    dyaw_raw = p[:, 2] - i[:, 2]
    dyaw = np.arctan2(np.sin(dyaw_raw), np.cos(dyaw_raw))
    dv = p[:, 3] - i[:, 3]
    return np.stack([dx, dy, dyaw, dv], axis=1)


def _cosine_coefficients(channels: np.ndarray) -> np.ndarray:
    """Fixed orthonormal DCT-II modes k=1,2 for each signed channel.

    k=0 is deliberately omitted: endpoint signed contrast already tests the
    static/directional hypothesis, while k=1,2 add the minimum fixed temporal
    shape basis (trend + curvature) without a bandwidth/horizon hyperparameter.
    """
    x = np.asarray(channels, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4 or x.shape[0] < 2:
        raise ValueError("V53 POTR cosine basis requires [T,4] signed channels")
    n = x.shape[0]
    t = np.arange(n, dtype=np.float64) + 0.5
    coeffs: list[float] = []
    for k in (1, 2):
        basis = math.sqrt(2.0 / n) * np.cos(math.pi * k * t / n)
        for c in range(4):
            coeffs.append(float(basis @ x[:, c]))
    out = np.asarray(coeffs, dtype=np.float64)
    if out.shape != (8,) or np.any(~np.isfinite(out)):
        raise ValueError("V53 POTR cosine coefficients invalid")
    return out


def trajectory_contrast_profile(proposal_trajectory: np.ndarray, incumbent_trajectory: np.ndarray) -> dict[str, Any]:
    ch = _signed_channels(proposal_trajectory, incumbent_trajectory)
    endpoint = np.asarray(ch[-1], dtype=np.float64)
    temporal = _cosine_coefficients(ch)
    d = execution_contrast_linf(proposal_trajectory, incumbent_trajectory)
    return {
        "schema": PROFILE_SCHEMA,
        "execution_contrast_linf": float(d),
        "endpoint_signed": [float(v) for v in endpoint],
        "cosine_modes_1_2": [float(v) for v in temporal],
        "trajectory_steps": int(ch.shape[0]),
    }


def support_state(quality_value: float, plan_control_value: float, ego_reference_value: float, d: float) -> np.ndarray:
    q = float(quality_value); p = float(plan_control_value); e = float(ego_reference_value); dd = float(d)
    z = np.asarray([q, p - q, e - p, dd], dtype=np.float64)
    if z.shape != (4,) or np.any(~np.isfinite(z)) or dd < 0.0:
        raise ValueError("V53 POTR support state invalid")
    return z


def outcome_state_from_profile(
    quality_value: float,
    plan_control_value: float,
    ego_reference_value: float,
    profile: dict[str, Any],
    *,
    state_family: str,
) -> np.ndarray:
    fam = str(state_family).strip().lower()
    if fam not in POTR_STATE_FAMILIES:
        raise ValueError(f"V53 POTR unknown state_family={state_family}")
    if str(profile.get("schema", "")) != PROFILE_SCHEMA:
        raise ValueError("V53 POTR profile schema mismatch")
    d = float(profile.get("execution_contrast_linf", float("nan")))
    base = support_state(quality_value, plan_control_value, ego_reference_value, d)
    endpoint = np.asarray(profile.get("endpoint_signed", []), dtype=np.float64).reshape(-1)
    temporal = np.asarray(profile.get("cosine_modes_1_2", []), dtype=np.float64).reshape(-1)
    if endpoint.size != 4 or temporal.size != 8 or np.any(~np.isfinite(endpoint)) or np.any(~np.isfinite(temporal)):
        raise ValueError("V53 POTR profile payload invalid")
    z = np.concatenate([base, endpoint])
    if fam == "temporal":
        z = np.concatenate([z, temporal])
    names = POTR_ENDPOINT_STATE_NAMES if fam == "endpoint" else POTR_TEMPORAL_STATE_NAMES
    if z.shape != (len(names),) or np.any(~np.isfinite(z)):
        raise ValueError("V53 POTR outcome state invalid")
    return z


def runtime_states(
    quality_value: float,
    plan_control_value: float,
    ego_reference_value: float,
    proposal_trajectory: np.ndarray,
    incumbent_trajectory: np.ndarray,
    *,
    state_family: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    profile = trajectory_contrast_profile(proposal_trajectory, incumbent_trajectory)
    sz = support_state(quality_value, plan_control_value, ego_reference_value, float(profile["execution_contrast_linf"]))
    oz = outcome_state_from_profile(quality_value, plan_control_value, ego_reference_value, profile, state_family=state_family)
    return sz, oz, profile


def _component_risk(z: np.ndarray, model: dict[str, Any], expected_names: list[str]) -> float:
    names = [str(x) for x in model.get("feature_names", [])]
    if names != expected_names:
        raise ValueError(f"V53 POTR feature schema mismatch: {names} != {expected_names}")
    mean = np.asarray(model.get("feature_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(model.get("feature_std", []), dtype=np.float64).reshape(-1)
    w = np.asarray(model.get("weights", []), dtype=np.float64).reshape(-1)
    d = len(expected_names)
    if mean.size != d or std.size != d or w.size != d:
        raise ValueError("V53 POTR component parameter size mismatch")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 0.0) or np.any(~np.isfinite(w)):
        raise ValueError("V53 POTR component parameters invalid")
    if abs(float(model.get("bias", 0.0))) > 1.0e-12 or abs(float(model.get("lambda", 1.0)) - 1.0) > 1.0e-12:
        raise ValueError("V53 POTR requires zero bias and fixed lambda=1")
    raw = float(((np.asarray(z, dtype=np.float64) - mean) / np.maximum(std, 1.0e-6)) @ w)
    bmean = float(model.get("fit_beneficial_score_mean", float("nan")))
    bstd = float(model.get("fit_beneficial_score_std", float("nan")))
    if not math.isfinite(bmean) or not math.isfinite(bstd) or bstd <= 0.0:
        raise ValueError("V53 POTR beneficial-score normalization invalid")
    return float((raw - bmean) / max(bstd, 1.0e-6))


def runtime_certificate(support_z: np.ndarray, outcome_z: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    fam = str(cfg.get("state_family", "")).strip().lower()
    if fam not in POTR_STATE_FAMILIES:
        raise ValueError(f"V53 POTR unknown runtime state_family={fam}")
    snames = POTR_SUPPORT_STATE_NAMES
    onames = POTR_ENDPOINT_STATE_NAMES if fam == "endpoint" else POTR_TEMPORAL_STATE_NAMES
    sz = np.asarray(support_z, dtype=np.float64).reshape(-1)
    oz = np.asarray(outcome_z, dtype=np.float64).reshape(-1)
    if sz.size != len(snames) or oz.size != len(onames) or np.any(~np.isfinite(sz)) or np.any(~np.isfinite(oz)):
        raise ValueError("V53 POTR runtime state shape invalid")
    if str(cfg.get("aggregation", "max_support_outcome")).strip().lower() != "max_support_outcome":
        raise ValueError("V53 POTR aggregation fixed to max_support_outcome")
    comp = cfg.get("components", {}) if isinstance(cfg, dict) else {}
    sr = _component_risk(sz, dict(comp.get("effect_support_risk", {})), snames)
    orisk = _component_risk(oz, dict(comp.get("conditional_outcome_risk", {})), onames)
    risk = float(max(sr, orisk))
    tau = float(cfg.get("retention_threshold", float("nan")))
    if not math.isfinite(tau):
        raise ValueError("V53 POTR retention threshold invalid")
    return float(tau - risk), risk, {
        "effect_support_risk": float(sr),
        "conditional_outcome_risk": float(orisk),
        "execution_contrast_linf": float(sz[3]),
    }
