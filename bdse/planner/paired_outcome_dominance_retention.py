from __future__ import annotations

"""V64.3.52 HODR: Hurdle Outcome-Dominance Retention.

V51 established that the minimal operator-relative state
    z = [Q, P-Q, E-P, D]
is identifiable for paired selected outcomes, while the single sign-only
retention functional is not deployment-sufficient.  HODR therefore freezes
that state and changes only the selected-outcome functional.

The runtime certificate is the maximum of two low-capacity risk components:
  1) effect_support_risk: structural-null / no-effect risk;
  2) conditional_outcome_risk: bad outcome risk conditional on an effect.

The second component is trained either with the old binary sign ordering
(HURDLE-SIGN causal arm) or with a deployment-aligned Pareto ordering over the
paired closed-loop official-score delta and hard-safety deltas
(HURDLE-PARETO causal arm).  There is no learned safety scalarization, no
second threshold, and no standalone catastrophe veto.
"""

from typing import Any
import math
import numpy as np

from bdse.planner.paired_operator_contrast_retention import (
    POCR_ADDITIVE_STATE_NAMES,
    operator_state as _v51_operator_state,
)

HODR_STATE_NAMES = list(POCR_ADDITIVE_STATE_NAMES)
HODR_FUNCTIONALS = {"hurdle_sign", "hurdle_pareto"}


def operator_state(
    quality_value: float,
    plan_control_value: float,
    ego_reference_value: float,
    proposal_trajectory: np.ndarray,
    incumbent_trajectory: np.ndarray,
) -> np.ndarray:
    """Exact V51 additive operator-relative state; no new runtime observable."""
    z = _v51_operator_state(
        quality_value,
        plan_control_value,
        ego_reference_value,
        proposal_trajectory,
        incumbent_trajectory,
        include_dose_interactions=False,
    )
    if z.shape != (len(HODR_STATE_NAMES),) or np.any(~np.isfinite(z)):
        raise ValueError("V52 HODR operator state is invalid")
    return z


def _component_risk(z: np.ndarray, model: dict[str, Any]) -> float:
    names = [str(x) for x in model.get("feature_names", [])]
    if names != HODR_STATE_NAMES:
        raise ValueError(f"V52 HODR feature schema mismatch: {names} != {HODR_STATE_NAMES}")
    mean = np.asarray(model.get("feature_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(model.get("feature_std", []), dtype=np.float64).reshape(-1)
    w = np.asarray(model.get("weights", []), dtype=np.float64).reshape(-1)
    d = len(HODR_STATE_NAMES)
    if mean.size != d or std.size != d or w.size != d:
        raise ValueError("V52 HODR component parameter size mismatch")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 0.0) or np.any(~np.isfinite(w)):
        raise ValueError("V52 HODR component parameters are invalid")
    if abs(float(model.get("bias", 0.0))) > 1.0e-12 or abs(float(model.get("lambda", 1.0)) - 1.0) > 1.0e-12:
        raise ValueError("V52 HODR requires zero bias and fixed lambda=1")
    raw = float(((np.asarray(z, dtype=np.float64) - mean) / np.maximum(std, 1.0e-6)) @ w)
    bmean = float(model.get("fit_beneficial_score_mean", float("nan")))
    bstd = float(model.get("fit_beneficial_score_std", float("nan")))
    if not math.isfinite(bmean) or not math.isfinite(bstd) or bstd <= 0.0:
        raise ValueError("V52 HODR beneficial-score normalization is invalid")
    return float((raw - bmean) / max(bstd, 1.0e-6))


def runtime_certificate(z: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    """One-threshold structured retention certificate.

    `max` is fixed by preregistration: a proposal must look compatible with a
    beneficial intervention under both the effect-support and conditional
    outcome factors.  The single split-conformal threshold remains calibrated
    on paired beneficial outcomes, so no alpha split or threshold sweep exists.
    """
    zz = np.asarray(z, dtype=np.float64).reshape(-1)
    if zz.size != len(HODR_STATE_NAMES) or np.any(~np.isfinite(zz)):
        raise ValueError("V52 HODR runtime state has wrong shape or non-finite values")
    functional = str(cfg.get("functional", "")).strip().lower()
    if functional not in HODR_FUNCTIONALS:
        raise ValueError(f"V52 HODR unknown functional={functional}")
    if str(cfg.get("aggregation", "max_support_outcome")).strip().lower() != "max_support_outcome":
        raise ValueError("V52 HODR aggregation is fixed to max_support_outcome")
    components = cfg.get("components", {}) if isinstance(cfg, dict) else {}
    support = _component_risk(zz, dict(components.get("effect_support_risk", {})))
    outcome = _component_risk(zz, dict(components.get("conditional_outcome_risk", {})))
    risk = float(max(support, outcome))
    tau = float(cfg.get("retention_threshold", float("nan")))
    if not math.isfinite(tau):
        raise ValueError("V52 HODR retention threshold is invalid")
    return float(tau - risk), risk, {
        "effect_support_risk": float(support),
        "conditional_outcome_risk": float(outcome),
        "execution_contrast_linf": float(zz[3]),
    }
