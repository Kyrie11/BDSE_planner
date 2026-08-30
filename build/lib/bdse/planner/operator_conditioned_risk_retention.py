from __future__ import annotations

"""V64.3.48 operator-conditioned selected-policy risk retention.

V48 never selects or re-ranks an action.  RSMR first freezes exactly one
proposal.  OCRR then evaluates only that proposal with a deliberately low-
capacity operator state:

    [QUALITY value,
     prospective-response increment,
     ego-reference increment,
     log(admissible-challenger multiplicity)].

The learned object is a zero-bias pairwise *sign-risk* ranker on the frozen
selected-policy population.  A stored TRAIN-only split-calibration threshold
turns risk into a veto-only certificate (threshold - risk).  Thus the runtime
operator can only retain the same RSMR proposal or return to the incumbent.
"""

from typing import Any
import math
import numpy as np

OCRR_STATE_NAMES = [
    "quality_value",
    "prospective_response_increment",
    "ego_reference_increment",
    "log_extremal_multiplicity",
]


def operator_state(
    quality_value: float,
    plan_control_value: float,
    ego_reference_value: float,
    candidate_count: int,
) -> np.ndarray:
    """Build the fixed four-dimensional state of the frozen proposal.

    ``candidate_count`` is the observed size of the *existing* admissible
    challenger set.  V48 does not alter the candidate bank, eligibility, or
    RSMR selector; multiplicity only conditions the post-selection risk law.
    """
    k = max(int(candidate_count), 1)
    z = np.asarray(
        [
            float(quality_value),
            float(plan_control_value) - float(quality_value),
            float(ego_reference_value) - float(plan_control_value),
            math.log(float(k)),
        ],
        dtype=np.float64,
    )
    if z.shape != (4,) or not np.all(np.isfinite(z)):
        raise ValueError("V48 OCRR operator state is non-finite")
    return z


def _sign_risk_score(z: np.ndarray, model: dict[str, Any]) -> float:
    names = [str(x) for x in model.get("feature_names", [])]
    if names != OCRR_STATE_NAMES:
        raise ValueError(f"V48 OCRR risk feature schema mismatch: {names}")
    mean = np.asarray(model.get("feature_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(model.get("feature_std", []), dtype=np.float64).reshape(-1)
    w = np.asarray(model.get("weights", []), dtype=np.float64).reshape(-1)
    if (
        mean.size != 4
        or std.size != 4
        or w.size != 4
        or np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(std))
        or np.any(std <= 0.0)
        or np.any(~np.isfinite(w))
    ):
        raise ValueError("V48 OCRR risk parameters are invalid")
    bias = float(model.get("bias", 0.0))
    if abs(bias) > 1.0e-12:
        raise ValueError("V48 OCRR pairwise sign-risk ranker requires zero bias")
    raw = float(((z - mean) / np.maximum(std, 1.0e-6)) @ w)
    pmean = float(model.get("fit_positive_score_mean", float("nan")))
    pstd = float(model.get("fit_positive_score_std", float("nan")))
    if not math.isfinite(pmean) or not math.isfinite(pstd) or pstd <= 0.0:
        raise ValueError("V48 OCRR positive-score normalization is invalid")
    return float((raw - pmean) / max(pstd, 1.0e-6))


def runtime_risk(z: np.ndarray, cfg: dict[str, Any]) -> tuple[float, dict[str, float]]:
    zz = np.asarray(z, dtype=np.float64).reshape(-1)
    if zz.size != 4 or not np.all(np.isfinite(zz)):
        raise ValueError("V48 OCRR runtime state must be finite 4-D")
    if str(cfg.get("aggregation", "sign_only")).strip().lower() != "sign_only":
        raise ValueError("V48 OCRR only permits the preregistered sign_only risk functional")
    models = cfg.get("components", {}) if isinstance(cfg, dict) else {}
    s = _sign_risk_score(zz, models.get("sign_risk", {}))
    return float(s), {"sign_risk": float(s)}


def runtime_certificate(z: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    risk, parts = runtime_risk(z, cfg)
    tau = float(cfg.get("retention_threshold", float("nan")))
    if not math.isfinite(tau):
        raise ValueError("V48 OCRR retention threshold is invalid")
    return float(tau - risk), float(risk), parts
