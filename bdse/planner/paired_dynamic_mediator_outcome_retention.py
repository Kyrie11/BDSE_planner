from __future__ import annotations

"""V64.3.55 DMOR: Dynamic-Mediator Outcome Retention.

V54 identifies the realized treatment-control ego transition over the frozen
one-shot exposure window as a genuine mediator of effectful selected-outcome
order, but the original binary-sign retention functional still fails the
fold-wise deployment gate.  V55 therefore changes no evidence/state family
first: it tests an operator-aligned Pareto outcome order on the *identified*
realized endpoint mediator.  Only if that oracle structured functional closes
its TRAIN gate is the same mediator distilled from pre-execution V53 planned
operator geometry to form a t0-available candidate state.

The predictor is deliberately zero preserving: a physically identical planned
proposal/incumbent pair maps to a zero predicted realized response.  No bias,
horizon/basis sweep, attention or new offline future observable is introduced.
"""

from typing import Any
import math
import numpy as np

from bdse.planner.paired_operator_contrast_retention import POCR_ADDITIVE_STATE_NAMES
from bdse.planner.paired_operator_trajectory_retention import PROFILE_SCHEMA as V53_PROFILE_SCHEMA

PLANNED_MEDIATOR_INPUT_NAMES = [
    "planned_terminal_dx_m", "planned_terminal_dy_m", "planned_terminal_dyaw_rad", "planned_terminal_dv_mps",
    "planned_cos1_dx_m", "planned_cos1_dy_m", "planned_cos1_dyaw_rad", "planned_cos1_dv_mps",
    "planned_cos2_dx_m", "planned_cos2_dy_m", "planned_cos2_dyaw_rad", "planned_cos2_dv_mps",
]
REALIZED_MEDIATOR_NAMES = ["realized_dx_end", "realized_dy_end", "realized_dyaw_end", "realized_dv_end"]
DMOR_REALIZED_STATE_NAMES = list(POCR_ADDITIVE_STATE_NAMES) + list(REALIZED_MEDIATOR_NAMES)
DMOR_PREDICTED_STATE_NAMES = list(POCR_ADDITIVE_STATE_NAMES) + [f"predicted_{x}" for x in REALIZED_MEDIATOR_NAMES]
MEDIATOR_MODEL_SCHEMA = "v64.3.55-zero-preserving-planned-to-realized-endpoint-ridge-v1"


def planned_mediator_input(profile: dict[str, Any]) -> np.ndarray:
    if str(profile.get("schema", "")) != V53_PROFILE_SCHEMA:
        raise ValueError("V55 DMOR V53 operator-profile schema mismatch")
    endpoint = np.asarray(profile.get("endpoint_signed", []), dtype=np.float64).reshape(-1)
    temporal = np.asarray(profile.get("cosine_modes_1_2", []), dtype=np.float64).reshape(-1)
    if endpoint.size != 4 or temporal.size != 8 or np.any(~np.isfinite(endpoint)) or np.any(~np.isfinite(temporal)):
        raise ValueError("V55 DMOR invalid planned operator profile")
    x = np.concatenate([endpoint, temporal])
    if x.shape != (len(PLANNED_MEDIATOR_INPUT_NAMES),):
        raise ValueError("V55 DMOR planned mediator input shape mismatch")
    return x


def fit_zero_preserving_mediator_ridge(X: np.ndarray, Y: np.ndarray, *, ridge_lambda: float = 1.0) -> dict[str, Any]:
    """Fit a no-bias multi-output ridge map from planned contrast to realized endpoint.

    Inputs/targets are RMS-scaled but never centered, so x=0 => y_hat=0 exactly.
    """
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] != 12 or y.shape[1] != 4:
        raise ValueError(f"V55 DMOR mediator ridge shape mismatch X={x.shape} Y={y.shape}")
    if x.shape[0] < 64 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("V55 DMOR invalid mediator ridge training data")
    lam = float(ridge_lambda)
    if abs(lam - 1.0) > 1e-12:
        raise ValueError("V55 DMOR mediator ridge lambda is frozen to 1")
    xs = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-6)
    ys = np.maximum(np.sqrt(np.mean(y * y, axis=0)), 1e-6)
    zx = x / xs[None, :]
    zy = y / ys[None, :]
    A = zx.T @ zx + lam * np.eye(zx.shape[1], dtype=np.float64)
    W = np.linalg.solve(A, zx.T @ zy)
    pred = (zx @ W) * ys[None, :]
    norm_mse = float(np.mean(((pred - y) / ys[None, :]) ** 2))
    zero_norm_mse = float(np.mean((y / ys[None, :]) ** 2))
    return {
        "schema": MEDIATOR_MODEL_SCHEMA,
        "input_names": list(PLANNED_MEDIATOR_INPUT_NAMES),
        "output_names": list(REALIZED_MEDIATOR_NAMES),
        "lambda": 1.0,
        "bias": [0.0, 0.0, 0.0, 0.0],
        "input_rms": [float(v) for v in xs],
        "output_rms": [float(v) for v in ys],
        "weights": [[float(v) for v in row] for row in W],
        "fit_row_count": int(x.shape[0]),
        "fit_normalized_mse": norm_mse,
        "zero_baseline_normalized_mse": zero_norm_mse,
        "zero_preserving": True,
    }


def predict_realized_endpoint(planned_x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    if str(model.get("schema", "")) != MEDIATOR_MODEL_SCHEMA:
        raise ValueError("V55 DMOR mediator model schema mismatch")
    names = [str(v) for v in model.get("input_names", [])]
    onames = [str(v) for v in model.get("output_names", [])]
    if names != PLANNED_MEDIATOR_INPUT_NAMES or onames != REALIZED_MEDIATOR_NAMES:
        raise ValueError("V55 DMOR mediator model feature schema mismatch")
    if abs(float(model.get("lambda", float("nan"))) - 1.0) > 1e-12:
        raise ValueError("V55 DMOR mediator model lambda drift")
    b = np.asarray(model.get("bias", []), dtype=np.float64).reshape(-1)
    if b.size != 4 or np.max(np.abs(b)) > 1e-12:
        raise ValueError("V55 DMOR mediator model must be zero-bias")
    x = np.asarray(planned_x, dtype=np.float64).reshape(-1)
    xs = np.asarray(model.get("input_rms", []), dtype=np.float64).reshape(-1)
    ys = np.asarray(model.get("output_rms", []), dtype=np.float64).reshape(-1)
    W = np.asarray(model.get("weights", []), dtype=np.float64)
    if x.size != 12 or xs.size != 12 or ys.size != 4 or W.shape != (12, 4):
        raise ValueError("V55 DMOR mediator model parameter shape mismatch")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(xs)) or np.any(xs <= 0) or np.any(~np.isfinite(ys)) or np.any(ys <= 0) or np.any(~np.isfinite(W)):
        raise ValueError("V55 DMOR mediator model parameters invalid")
    y = ((x / xs) @ W) * ys
    if y.shape != (4,) or np.any(~np.isfinite(y)):
        raise ValueError("V55 DMOR predicted mediator invalid")
    return y


def outcome_state(q: float, p: float, e: float, d: float, mediator: np.ndarray, *, predicted: bool) -> np.ndarray:
    m = np.asarray(mediator, dtype=np.float64).reshape(-1)
    z = np.asarray([float(q), float(p) - float(q), float(e) - float(p), float(d)], dtype=np.float64)
    if m.size != 4 or np.any(~np.isfinite(m)) or np.any(~np.isfinite(z)) or float(d) < 0.0:
        raise ValueError("V55 DMOR outcome state invalid")
    out = np.concatenate([z, m])
    names = DMOR_PREDICTED_STATE_NAMES if predicted else DMOR_REALIZED_STATE_NAMES
    if out.shape != (len(names),):
        raise ValueError("V55 DMOR outcome state shape mismatch")
    return out
