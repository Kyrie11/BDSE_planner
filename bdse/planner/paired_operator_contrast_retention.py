from __future__ import annotations

"""V64.3.51 Paired Operator-Contrast Retention (POCR).

V50 showed that deployment-aligned paired outcome labels are not sufficient
when the selected-outcome state remains the compressed Q/P/E consequence
coordinates.  POCR keeps the frozen full-set RSMR proposal, Q/P/E consequence
views, pairwise sign-risk family, lambda, conformal retention budget, and
veto-only no-fallback operator unchanged.  The only new runtime information is
an operator-relative treatment/control contrast that is already available
before execution:

    D(b, i) = || trajectory_b - trajectory_i ||_infinity.

The additive arm uses [Q, P-Q, E-P, D].  The interaction arm additionally uses
D times each Q/P/E coordinate, allowing the consequence slopes to depend on
intervention dose without a new MLP or threshold sweep.
"""

from typing import Any
import math
import numpy as np

QPE_NAMES = [
    "quality_value",
    "prospective_response_increment",
    "ego_reference_increment",
]
POCR_ADDITIVE_STATE_NAMES = QPE_NAMES + ["operator_execution_contrast_linf"]
POCR_INTERACTION_STATE_NAMES = POCR_ADDITIVE_STATE_NAMES + [
    "contrast_x_quality_value",
    "contrast_x_prospective_response_increment",
    "contrast_x_ego_reference_increment",
]


def execution_contrast_linf(proposal_trajectory: np.ndarray, incumbent_trajectory: np.ndarray) -> float:
    """Exact bounded-interface treatment/control trajectory contrast.

    This deliberately reuses the physical-identity geometry already audited by
    V50.4/V50.5.  No learned scale or threshold is introduced.
    """
    p = np.asarray(proposal_trajectory, dtype=np.float64)
    i = np.asarray(incumbent_trajectory, dtype=np.float64)
    if p.ndim != 2 or i.ndim != 2 or p.shape != i.shape or p.size == 0:
        raise ValueError(f"V51 POCR treatment/control trajectory shape mismatch: {p.shape} vs {i.shape}")
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(i)):
        raise ValueError("V51 POCR treatment/control trajectory is non-finite")
    d = float(np.max(np.abs(p - i)))
    if not math.isfinite(d) or d < 0.0:
        raise ValueError("V51 POCR execution contrast is invalid")
    return d


def operator_state(
    quality_value: float,
    plan_control_value: float,
    ego_reference_value: float,
    proposal_trajectory: np.ndarray,
    incumbent_trajectory: np.ndarray,
    *,
    include_dose_interactions: bool,
) -> np.ndarray:
    q = float(quality_value)
    p = float(plan_control_value)
    e = float(ego_reference_value)
    base = np.asarray([q, p - q, e - p], dtype=np.float64)
    d = execution_contrast_linf(proposal_trajectory, incumbent_trajectory)
    z = np.concatenate([base, np.asarray([d], dtype=np.float64)])
    if include_dose_interactions:
        z = np.concatenate([z, d * base])
    expected = 7 if include_dose_interactions else 4
    if z.shape != (expected,) or np.any(~np.isfinite(z)):
        raise ValueError("V51 POCR operator state is invalid")
    return z


def _sign_risk_score(z: np.ndarray, model: dict[str, Any], expected_names: list[str]) -> float:
    names = [str(x) for x in model.get("feature_names", [])]
    if names != expected_names:
        raise ValueError(f"V51 POCR risk feature schema mismatch: {names} != {expected_names}")
    mean = np.asarray(model.get("feature_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(model.get("feature_std", []), dtype=np.float64).reshape(-1)
    w = np.asarray(model.get("weights", []), dtype=np.float64).reshape(-1)
    d = len(expected_names)
    if mean.size != d or std.size != d or w.size != d or np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 0.0) or np.any(~np.isfinite(w)):
        raise ValueError("V51 POCR risk parameters are invalid")
    if abs(float(model.get("bias", 0.0))) > 1.0e-12:
        raise ValueError("V51 POCR pairwise sign-risk ranker requires zero bias")
    raw = float(((np.asarray(z, dtype=np.float64) - mean) / np.maximum(std, 1.0e-6)) @ w)
    pmean = float(model.get("fit_positive_score_mean", float("nan")))
    pstd = float(model.get("fit_positive_score_std", float("nan")))
    if not math.isfinite(pmean) or not math.isfinite(pstd) or pstd <= 0.0:
        raise ValueError("V51 POCR positive-score normalization is invalid")
    return float((raw - pmean) / max(pstd, 1.0e-6))


def runtime_certificate(z: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    interaction = bool(cfg.get("include_dose_interactions", False))
    names = POCR_INTERACTION_STATE_NAMES if interaction else POCR_ADDITIVE_STATE_NAMES
    zz = np.asarray(z, dtype=np.float64).reshape(-1)
    if zz.size != len(names) or np.any(~np.isfinite(zz)):
        raise ValueError("V51 POCR runtime state has wrong shape or non-finite values")
    if str(cfg.get("aggregation", "sign_only")).strip().lower() != "sign_only":
        raise ValueError("V51 POCR only permits the preregistered sign_only risk functional")
    models = cfg.get("components", {}) if isinstance(cfg, dict) else {}
    risk = _sign_risk_score(zz, models.get("sign_risk", {}), names)
    tau = float(cfg.get("retention_threshold", float("nan")))
    if not math.isfinite(tau):
        raise ValueError("V51 POCR retention threshold is invalid")
    return float(tau - risk), float(risk), {"sign_risk": float(risk), "execution_contrast_linf": float(zz[3])}
