from __future__ import annotations

"""V64.3.54 PDRM: Paired Dynamic Response Mediation.

V53 falsified pre-execution proposal-vs-incumbent trajectory geometry as a
sufficient, fold-stable description of effectful paired outcome order.  V54
therefore follows the preregistered branch and measures the *realized* paired
ego response caused by the exact one-shot intervention.

The state is deliberately diagnostic rather than a t=0 deployment input.  It is
collected only over the already frozen one-shot operator-exposure window (from
the intervention anchor through the first scheduled planner replan) and is used
to identify whether realized state transition, rather than planned treatment
geometry, mediates the final paired closed-loop outcome.
"""

from typing import Any
import math
import numpy as np

from bdse.planner.paired_operator_contrast_retention import POCR_ADDITIVE_STATE_NAMES

DYNAMIC_CHANNEL_NAMES = ["dx_local_m", "dy_local_m", "dyaw_rad", "dv_mps"]
DYNAMIC_ENDPOINT_EXTRA_NAMES = [f"realized_{x}" for x in DYNAMIC_CHANNEL_NAMES]
DYNAMIC_TEMPORAL_EXTRA_NAMES = [f"realized_cos{k}_{x}" for k in (1, 2) for x in DYNAMIC_CHANNEL_NAMES]
PDRM_BASE_STATE_NAMES = list(POCR_ADDITIVE_STATE_NAMES)
PDRM_ENDPOINT_STATE_NAMES = PDRM_BASE_STATE_NAMES + DYNAMIC_ENDPOINT_EXTRA_NAMES
PDRM_TEMPORAL_STATE_NAMES = PDRM_ENDPOINT_STATE_NAMES + DYNAMIC_TEMPORAL_EXTRA_NAMES
PDRM_STATE_FAMILIES = {"realized_endpoint", "realized_temporal"}
DYNAMIC_PROFILE_SCHEMA = "v64.3.54-paired-realized-ego-response-v1"


def _wrap(a: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(a), np.cos(a))


def _cosine_coefficients(channels: np.ndarray) -> np.ndarray:
    x = np.asarray(channels, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4 or x.shape[0] < 2:
        raise ValueError("V54 PDRM cosine basis requires [T,4] realized paired channels")
    n = int(x.shape[0])
    t = np.arange(n, dtype=np.float64) + 0.5
    out: list[float] = []
    for k in (1, 2):
        basis = math.sqrt(2.0 / n) * np.cos(math.pi * k * t / n)
        for c in range(4):
            out.append(float(basis @ x[:, c]))
    a = np.asarray(out, dtype=np.float64)
    if a.shape != (8,) or np.any(~np.isfinite(a)):
        raise ValueError("V54 PDRM invalid cosine coefficients")
    return a


def paired_realized_profile(
    treatment_trace: np.ndarray,
    control_trace: np.ndarray,
    *,
    iteration_indices: list[int] | np.ndarray,
    timestamps_us: list[int] | np.ndarray,
    planned_execution_contrast_linf: float,
) -> dict[str, Any]:
    """Build the paired realized ego-response profile in the common t0 frame.

    Trace rows are [x_world, y_world, yaw_world, speed].  The treatment-control
    position delta is rotated into the shared initial control ego frame, making
    the response invariant to global map orientation.  This is *realized*
    closed-loop state, not the frozen proposal trajectory used by V53.
    """
    tr = np.asarray(treatment_trace, dtype=np.float64)
    cr = np.asarray(control_trace, dtype=np.float64)
    idx = np.asarray(iteration_indices, dtype=np.int64).reshape(-1)
    ts = np.asarray(timestamps_us, dtype=np.int64).reshape(-1)
    if tr.ndim != 2 or cr.ndim != 2 or tr.shape != cr.shape or tr.shape[1] != 4 or tr.shape[0] < 2:
        raise ValueError(f"V54 PDRM trace shape mismatch treatment={tr.shape} control={cr.shape}")
    if idx.size != tr.shape[0] or ts.size != tr.shape[0]:
        raise ValueError("V54 PDRM iteration/timestamp length mismatch")
    if np.any(~np.isfinite(tr)) or np.any(~np.isfinite(cr)):
        raise ValueError("V54 PDRM non-finite realized trace")
    if not np.array_equal(idx, np.arange(idx.size, dtype=np.int64)):
        raise ValueError(f"V54 PDRM requires contiguous synchronized iterations 0..N, got {idx.tolist()}")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("V54 PDRM timestamps must be strictly increasing")

    dxy = tr[:, :2] - cr[:, :2]
    yaw0 = float(cr[0, 2])
    c, s = math.cos(yaw0), math.sin(yaw0)
    dx = c * dxy[:, 0] + s * dxy[:, 1]
    dy = -s * dxy[:, 0] + c * dxy[:, 1]
    dyaw = np.asarray(_wrap(tr[:, 2] - cr[:, 2]), dtype=np.float64)
    dv = tr[:, 3] - cr[:, 3]
    ch = np.stack([dx, dy, dyaw, dv], axis=1)
    initial_err = float(np.max(np.abs(ch[0])))
    if initial_err > 1.0e-8:
        raise ValueError(f"V54 PDRM paired initial-state mismatch max_abs={initial_err}")
    endpoint = ch[-1].copy()
    temporal = _cosine_coefficients(ch)
    realized_linf = float(np.max(np.abs(ch)))
    planned_d = float(planned_execution_contrast_linf)
    if not math.isfinite(planned_d) or planned_d < 0.0:
        raise ValueError("V54 PDRM invalid frozen planned D")
    return {
        "schema": DYNAMIC_PROFILE_SCHEMA,
        "planned_execution_contrast_linf": planned_d,
        "realized_response_linf": realized_linf,
        "endpoint_signed": [float(v) for v in endpoint],
        "cosine_modes_1_2": [float(v) for v in temporal],
        "iteration_indices": [int(v) for v in idx],
        "timestamps_us": [int(v) for v in ts],
        "exposure_ticks": int(idx[-1]),
        "sample_count": int(idx.size),
        "initial_pair_max_abs": initial_err,
    }


def outcome_state_from_dynamic_profile(
    quality_value: float,
    plan_control_value: float,
    ego_reference_value: float,
    profile: dict[str, Any],
    *,
    state_family: str,
) -> np.ndarray:
    fam = str(state_family).strip().lower()
    if fam not in PDRM_STATE_FAMILIES:
        raise ValueError(f"V54 PDRM unknown state family {state_family!r}")
    if str(profile.get("schema", "")) != DYNAMIC_PROFILE_SCHEMA:
        raise ValueError("V54 PDRM dynamic profile schema mismatch")
    q, p, e = float(quality_value), float(plan_control_value), float(ego_reference_value)
    d = float(profile.get("planned_execution_contrast_linf", float("nan")))
    endpoint = np.asarray(profile.get("endpoint_signed", []), dtype=np.float64).reshape(-1)
    temporal = np.asarray(profile.get("cosine_modes_1_2", []), dtype=np.float64).reshape(-1)
    if endpoint.size != 4 or temporal.size != 8 or not math.isfinite(d) or d < 0 or np.any(~np.isfinite(endpoint)) or np.any(~np.isfinite(temporal)):
        raise ValueError("V54 PDRM invalid dynamic profile payload")
    z = np.asarray([q, p - q, e - p, d], dtype=np.float64)
    z = np.concatenate([z, endpoint])
    names = PDRM_ENDPOINT_STATE_NAMES
    if fam == "realized_temporal":
        z = np.concatenate([z, temporal])
        names = PDRM_TEMPORAL_STATE_NAMES
    if z.shape != (len(names),) or np.any(~np.isfinite(z)):
        raise ValueError("V54 PDRM state invalid")
    return z
