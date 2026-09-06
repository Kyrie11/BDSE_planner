from __future__ import annotations

"""V64.3.56 RCPR: Realized Constraint-Process Retention.

V55 shows that a realized one-replan ego transition is a real mediator and that
an unweighted paired-outcome Pareto functional is itself identifiable, yet the
combination remains deployment-insufficient.  V56 therefore changes only the
state family, from ego motion to the realized *constraint consequences* induced
by the same frozen treatment.  The process channels are runtime-semantic and
share the already frozen safety geometry:

  1. ungated agent occupancy potential (V44 semantics),
  2. radial TTC risk using the frozen safe-TTC scale,
  3. hard route-corridor excess using the frozen off-route margin.

All channels are lower-is-safer.  The paired process stores control minus
treatment risk, so larger values always mean the treatment improved the
constraint state.  No learned safety weight or scalarization is introduced.
"""

from typing import Any
import math
import numpy as np

from bdse.data.state_schema import DEFAULT_VEHICLE_LENGTH_M, DEFAULT_VEHICLE_WIDTH_M
from bdse.planner.paired_dynamic_mediator_outcome_retention import (
    PLANNED_MEDIATOR_INPUT_NAMES,
    REALIZED_MEDIATOR_NAMES,
    fit_zero_preserving_mediator_ridge,
    predict_realized_endpoint,
)

CONSTRAINT_CHANNEL_NAMES = ["agent_occupancy_risk", "agent_ttc_risk", "hard_offroute_excess_m"]
POST_TICK_COUNT = 5
CONSTRAINT_PROCESS_NAMES = [f"delta_{name}_t{t}" for t in range(1, POST_TICK_COUNT + 1) for name in CONSTRAINT_CHANNEL_NAMES]
CONSTRAINT_PROFILE_SCHEMA = "v64.3.56-paired-realized-constraint-process-v1"
CONSTRAINT_PREDICTOR_SCHEMA = "v64.3.56-zero-preserving-constraint-process-ridge-v1"


def _point_to_polyline_distance_origin(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        return float("inf")
    if len(p) == 1:
        return float(np.linalg.norm(p[0]))
    a = p[:-1]
    b = p[1:]
    d = b - a
    den = np.sum(d * d, axis=1)
    q = np.zeros_like(den)
    ok = den > 1e-12
    q[ok] = np.clip(-np.sum(a[ok] * d[ok], axis=1) / den[ok], 0.0, 1.0)
    closest = a + q[:, None] * d
    return float(np.min(np.linalg.norm(closest, axis=1)))


def instantaneous_constraint_risk(runtime: Any, cfg: dict[str, Any]) -> np.ndarray:
    """Return the three frozen-semantics current constraint risks.

    Runtime agents are ego-local, so ego is at (0,0), heading 0.  This function
    consumes no future/logged state and is valid both during treatment/control
    replay and at t0.
    """
    rsc = cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}
    cur = np.asarray(getattr(runtime, "current_agents", np.zeros((0, 10))), dtype=np.float64).reshape(-1, 10)
    valid = np.asarray(getattr(runtime, "agent_valid", np.zeros((len(cur),), dtype=bool)), dtype=bool).reshape(-1)
    n = min(len(cur), len(valid))
    max_occ = 0.0
    min_ttc = float("inf")
    ego_speed = 0.0
    try:
        eh = np.asarray(runtime.ego_history, dtype=np.float64)
        if eh.ndim == 2 and len(eh):
            ego_speed = float(eh[-1, 3])
    except Exception:
        ego_speed = 0.0
    use_box = bool(rsc.get("use_box_agent_risk", False))
    ego_l = max(float(rsc.get("ego_length_m", DEFAULT_VEHICLE_LENGTH_M)), 0.5)
    ego_w = max(float(rsc.get("ego_width_m", DEFAULT_VEHICLE_WIDTH_M)), 0.3)
    soft_r = max(float(rsc.get("soft_agent_radius_m", 1.5)), 1e-3)
    for j in range(n):
        if not valid[j]:
            continue
        a = cur[j]
        if np.any(~np.isfinite(a[:9])):
            continue
        dx, dy = -float(a[0]), -float(a[1])  # same sign convention as V44 ego-agent displacement
        # Radial closing speed uses the same sign convention as the frozen
        # runtime-safety envelope.  It is also used for the longitudinal soft
        # box buffer, so the current realized process is semantically aligned
        # with the deployed constraint geometry rather than being a new radius.
        rx, ry = float(a[0]), float(a[1])
        dist = math.hypot(rx, ry)
        closing = 0.0
        if dist > 1e-6:
            rvx = float(a[5]) - ego_speed
            rvy = float(a[6])
            closing = max(-(rx * rvx + ry * rvy) / dist, 0.0)

        if use_box and float(a[7]) > 0.0 and float(a[8]) > 0.0:
            al = max(float(a[7]), 0.3); aw = max(float(a[8]), 0.2)
            c, s = math.cos(float(a[2])), math.sin(float(a[2]))
            lon = c * dx + s * dy
            lat = -s * dx + c * dy
            closing_buffer = min(
                closing * max(float(rsc.get("closing_speed_buffer_s", 0.22)), 0.0),
                max(float(rsc.get("max_closing_buffer_m", 3.0)), 0.0),
            )
            soft_l = (
                0.5 * (ego_l + al)
                + float(rsc.get("hard_longitudinal_clearance_m", 0.20))
                + closing_buffer
                + float(rsc.get("soft_longitudinal_extra_m", 1.00))
            )
            soft_w = 0.5 * (ego_w + aw) + float(rsc.get("hard_lateral_clearance_m", 0.15)) + float(rsc.get("soft_lateral_extra_m", 0.65))
            norm2 = (lon / max(soft_l, 0.1)) ** 2 + (lat / max(soft_w, 0.1)) ** 2
            ttc_gate = math.sqrt(max(norm2, 0.0)) <= float(rsc.get("ttc_envelope_gate", 1.35))
        else:
            norm2 = (dx * dx + dy * dy) / (soft_r * soft_r)
            ttc_gate = dist <= soft_r * float(rsc.get("ttc_envelope_gate", 1.35))
        max_occ = max(max_occ, 1.0 / (1.0 + max(norm2, 0.0)))

        # Radial constant-velocity TTC from the *realized current state*, gated
        # by the same frozen soft interaction envelope used by runtime safety.
        if dist > 1e-6 and closing > 1e-3 and ttc_gate:
            min_ttc = min(min_ttc, dist / closing)
    safe_ttc = max(float(rsc.get("agent_ttc_safe_s", 3.0)), 1e-3)
    ttc_risk = max((safe_ttc - min_ttc) / safe_ttc, 0.0) if math.isfinite(min_ttc) else 0.0

    mf = getattr(runtime, "map_features", {}) or {}
    route = np.asarray(mf.get("route_centerline", np.zeros((0, 2))), dtype=np.float64).reshape(-1, 2)
    route_dist = _point_to_polyline_distance_origin(route)
    if not math.isfinite(route_dist):
        raise ValueError("V56 RCPR requires a finite runtime route centerline")
    width = float(mf.get("route_corridor_width", cfg.get("candidate", {}).get("route_width_m", 4.0)))
    hard_margin = float(rsc.get("hard_off_route_margin_m", 3.0))
    off = max(route_dist - (width + hard_margin), 0.0)
    out = np.asarray([max_occ, ttc_risk, off], dtype=np.float64)
    if out.shape != (3,) or np.any(~np.isfinite(out)) or np.any(out < -1e-12):
        raise ValueError(f"V56 RCPR invalid constraint risk {out}")
    return out


def paired_constraint_profile(treatment: np.ndarray, control: np.ndarray, *, iteration_indices: list[int] | np.ndarray, timestamps_us: list[int] | np.ndarray) -> dict[str, Any]:
    tr = np.asarray(treatment, dtype=np.float64)
    cr = np.asarray(control, dtype=np.float64)
    idx = np.asarray(iteration_indices, dtype=np.int64).reshape(-1)
    ts = np.asarray(timestamps_us, dtype=np.int64).reshape(-1)
    if tr.shape != cr.shape or tr.ndim != 2 or tr.shape[1] != 3 or tr.shape[0] != POST_TICK_COUNT + 1:
        raise ValueError(f"V56 RCPR trace shape mismatch treatment={tr.shape} control={cr.shape}")
    if idx.size != len(tr) or ts.size != len(tr) or not np.array_equal(idx, np.arange(len(tr), dtype=np.int64)):
        raise ValueError("V56 RCPR requires synchronized contiguous iterations 0..5")
    if np.any(np.diff(ts) <= 0) or np.any(~np.isfinite(tr)) or np.any(~np.isfinite(cr)):
        raise ValueError("V56 RCPR invalid synchronized trace")
    init = float(np.max(np.abs(tr[0] - cr[0])))
    if init > 1e-8:
        raise ValueError(f"V56 RCPR initial constraint-state mismatch {init}")
    # Larger is better: positive means treatment reduced realized risk vs control.
    delta = cr[1:] - tr[1:]
    flat = delta.reshape(-1)
    if flat.shape != (len(CONSTRAINT_PROCESS_NAMES),):
        raise ValueError("V56 RCPR process dimension mismatch")
    return {
        "schema": CONSTRAINT_PROFILE_SCHEMA,
        "channel_names": list(CONSTRAINT_CHANNEL_NAMES),
        "t0_constraint_risk": [float(v) for v in cr[0]],
        "post_intervention_control_risk": [[float(v) for v in row] for row in cr[1:]],
        "post_intervention_treatment_risk": [[float(v) for v in row] for row in tr[1:]],
        "constraint_support_delta_process": [float(v) for v in flat],
        "iteration_indices": [int(v) for v in idx],
        "timestamps_us": [int(v) for v in ts],
        "sample_count": int(len(tr)),
        "initial_pair_max_abs": init,
    }


def constraint_predictor_input(planned_profile: dict[str, Any], t0_constraint_risk: np.ndarray) -> np.ndarray:
    endpoint = np.asarray(planned_profile.get("endpoint_signed", []), dtype=np.float64).reshape(-1)
    temporal = np.asarray(planned_profile.get("cosine_modes_1_2", []), dtype=np.float64).reshape(-1)
    d = float(planned_profile.get("execution_contrast_linf", float("nan")))
    c0 = np.asarray(t0_constraint_risk, dtype=np.float64).reshape(-1)
    if endpoint.size != 4 or temporal.size != 8 or c0.size != 3 or not math.isfinite(d) or d < 0 or np.any(~np.isfinite(endpoint)) or np.any(~np.isfinite(temporal)) or np.any(~np.isfinite(c0)):
        raise ValueError("V56 RCPR invalid predictor input")
    # Dose-gated context preserves the physical identity contract exactly.
    x = np.concatenate([endpoint, temporal, d * c0])
    if x.shape != (15,):
        raise ValueError("V56 RCPR predictor input dimension mismatch")
    return x


def fit_zero_preserving_constraint_predictor(X: np.ndarray, Y: np.ndarray, *, ridge_lambda: float = 1.0) -> dict[str, Any]:
    x = np.asarray(X, dtype=np.float64); y = np.asarray(Y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] != 15 or y.shape[1] != len(CONSTRAINT_PROCESS_NAMES):
        raise ValueError(f"V56 RCPR predictor shape mismatch X={x.shape} Y={y.shape}")
    if x.shape[0] < 64 or abs(float(ridge_lambda)-1.0)>1e-12 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("V56 RCPR invalid predictor training data")
    xs=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-6); ys=np.maximum(np.sqrt(np.mean(y*y,axis=0)),1e-6)
    zx=x/xs[None,:]; zy=y/ys[None,:]
    W=np.linalg.solve(zx.T@zx + np.eye(zx.shape[1]), zx.T@zy)
    pred=(zx@W)*ys[None,:]
    return {"schema":CONSTRAINT_PREDICTOR_SCHEMA,"lambda":1.0,"bias":[0.0]*len(CONSTRAINT_PROCESS_NAMES),
            "input_rms":[float(v) for v in xs],"output_rms":[float(v) for v in ys],"weights":[[float(v) for v in row] for row in W],
            "fit_row_count":int(len(x)),"fit_normalized_mse":float(np.mean(((pred-y)/ys[None,:])**2)),
            "zero_baseline_normalized_mse":float(np.mean((y/ys[None,:])**2)),"zero_preserving":True}


def predict_constraint_process(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    if str(model.get("schema", "")) != CONSTRAINT_PREDICTOR_SCHEMA or abs(float(model.get("lambda", float("nan")))-1.0)>1e-12:
        raise ValueError("V56 RCPR predictor schema/lambda drift")
    xx=np.asarray(x,dtype=np.float64).reshape(-1); xs=np.asarray(model.get("input_rms",[]),dtype=np.float64); ys=np.asarray(model.get("output_rms",[]),dtype=np.float64); W=np.asarray(model.get("weights",[]),dtype=np.float64)
    b=np.asarray(model.get("bias",[]),dtype=np.float64)
    if xx.size!=15 or xs.shape!=(15,) or ys.shape!=(len(CONSTRAINT_PROCESS_NAMES),) or W.shape!=(15,len(CONSTRAINT_PROCESS_NAMES)) or b.shape!=(len(CONSTRAINT_PROCESS_NAMES),) or np.max(np.abs(b))>1e-12:
        raise ValueError("V56 RCPR predictor parameter shape/bias mismatch")
    y=((xx/xs)@W)*ys
    if np.any(~np.isfinite(y)): raise ValueError("V56 RCPR non-finite prediction")
    return y
