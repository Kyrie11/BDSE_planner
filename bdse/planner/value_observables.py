from __future__ import annotations

"""Deployment-observable, value-specific trajectory consequences for V64.3.42.

These observables deliberately do *not* use label-only future information.  They
partition currently observable candidate consequences into two causal blocks:

* trajectory quality: the deployment-observable subset of the teacher base cost
  (route deviation, relative progress deficit, global comfort);
* physical runtime risk: continuous current-map/current-agent safety severities.

All outputs are lower-is-better candidate costs.  V42 converts them to
candidate-vs-incumbent improvements (incumbent cost - candidate cost) *after*
the frozen RSMR winner is chosen.  The module does not select or rank actions.
"""

from typing import Any

import os
import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.evidence_atoms import _eval_traj
from bdse.planner.fallback import runtime_risk_scores
from bdse.planner.teacher_cost import global_comfort_cost
from bdse.utils import nearest_polyline_distance, route_progress_along_polyline

QUALITY_NAMES = [
    "route_deviation_cost",
    "progress_deficit_cost",
    "global_comfort_cost",
]
RISK_NAMES = [
    "hard_agent_risk",
    "soft_agent_risk",
    "agent_ttc_risk",
    "hard_off_route_risk",
    "soft_off_route_risk",
    "red_light_risk",
]
VALUE_OBSERVABLE_NAMES = QUALITY_NAMES + RISK_NAMES
QUALITY_DIM = len(QUALITY_NAMES)


def _env_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _v50_elide_unused_fsfr_2d(cfg: dict[str, Any]) -> bool:
    """Whether V50 may skip the closed V47 AGENT-2D observable.

    The V48/V49/V50 OCRR/SIIR/SIOR operator consumes only QUALITY, the frozen
    V45 PLAN-1D occupancy coordinate, and the V47 EGO-REF coordinate.  The
    V47 ``fsfr_plan_2d_occupancy_cost`` column remains in the historical
    observable schema for provenance, but it is not indexed by the frozen
    post-selection functional.  Under the dedicated V50 execution flag we
    therefore keep the *full historical schema* while not evaluating that
    scientifically closed AGENT-2D branch.  Its in-memory, non-persisted slot
    is filled with zero; the frozen tournament byte-lock is untouched.

    The check is deliberately fail-closed: if a later config ever consumes
    PLAN-2D, this optimization refuses to activate rather than silently change
    a decision coordinate.
    """
    if not _env_true("BDSE_V50_ELIDE_UNUSED_FSFR_2D", default=False):
        return False
    ic = (((cfg.get("runtime", {}) or {}).get("decisive_frontier_value", {}) or {}).get("incumbent_contrastive_extremal_recovery", {}) or {})
    sc = ic.get("selection_conditioned_intervention_recovery", {}) or {}
    if str(sc.get("post_selection_value_mode", "")) != "endpoint_potential_quality_operator_conditioned_risk_retention":
        return False
    stored = [str(x) for x in sc.get("post_selection_observable_names", [])]
    qnames = [str(x) for x in sc.get("post_selection_quality_observable_names", [])]
    pnames = [str(x) for x in sc.get("selected_policy_risk_plan_response_names", [])]
    enames = [str(x) for x in sc.get("selected_policy_risk_ego_reference_names", [])]
    required = set(qnames + pnames + enames)
    target = "fsfr_plan_2d_occupancy_cost"
    if target not in stored:
        raise ValueError("V50 FSFR-2D elision requested but historical PLAN-2D observable is absent")
    if target in required:
        raise ValueError("V50 FSFR-2D elision refused because frozen Q/P/E consumes PLAN-2D")
    if not required or any(n not in stored for n in required):
        raise ValueError("V50 FSFR-2D elision found an incomplete frozen Q/P/E schema")
    return True

def _finite_clip(x: np.ndarray, limit: float = 1.0e6) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return np.clip(np.nan_to_num(a, nan=limit, posinf=limit, neginf=-limit), -limit, limit)


def runtime_value_observable_costs(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Return K x 9 lower-is-better, label-free value observables.

    The first three columns exactly mirror the route/progress/comfort terms used
    by ``evaluate_base_costs`` after applying the same configured weights/scales,
    while intentionally excluding the demonstration term because it requires the
    label-only logged future.  The remaining six columns are continuous runtime
    risk severities already available to the deployed planner.
    """

    K = int(candidates.K)
    if K <= 0:
        # Preserve the historical schema even when the V50 dead-observable
        # elision is enabled.  Tournament science-lock checks are schema-strict.
        names0 = list(VALUE_OBSERVABLE_NAMES)
        ic0 = (((cfg.get("runtime", {}) or {}).get("decisive_frontier_value", {}) or {}).get("incumbent_contrastive_extremal_recovery", {}) or {}) if isinstance(cfg, dict) else {}
        if bool(ic0.get("instrument_future_state_factorization_observables", False)):
            from bdse.planner.future_state_factorization import FSFR_OBSERVABLE_NAMES
            names0.extend(FSFR_OBSERVABLE_NAMES)
        return np.zeros((0, len(names0)), dtype=np.float64), names0

    tcfg = cfg.get("teacher", {}) if isinstance(cfg, dict) else {}
    route = np.asarray(
        runtime.map_features.get(
            "route_centerline",
            np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32),
        ),
        dtype=np.float32,
    ).reshape(-1, 2)
    if route.size:
        route = route[np.isfinite(route).all(axis=1)]
    if len(route) < 2:
        route = np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)

    base_dt = float(cfg.get("candidate", {}).get("step_s", 0.1)) if isinstance(cfg, dict) else 0.1
    stride = max(1, int(tcfg.get("cost_eval_stride", 1)))
    dt_eval = base_dt * stride

    route_dev = np.zeros((K,), dtype=np.float64)
    progress_def = np.zeros((K,), dtype=np.float64)
    comfort = np.zeros((K,), dtype=np.float64)

    terminal_progress = route_progress_along_polyline(candidates.trajectories[:, -1, :2], route)
    valid = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)
    best_progress = float(terminal_progress[valid].max()) if np.any(valid) else 0.0
    for a in range(K):
        traj = _eval_traj(candidates.trajectories[a], cfg)
        route_dev[a] = float(np.square(nearest_polyline_distance(traj[:, :2], route)).mean())
        progress_def[a] = float(max(0.0, best_progress - float(terminal_progress[a])))
        comfort[a] = float(global_comfort_cost(traj, dt_eval))

    quality = np.stack(
        [
            float(tcfg.get("route_weight", 50.0))
            * _finite_clip(route_dev)
            / max(float(tcfg.get("route_scale", 1.0)), 1.0e-6),
            float(tcfg.get("progress_weight", 5.0))
            * _finite_clip(progress_def)
            / max(float(tcfg.get("progress_scale", 10.0)), 1.0e-6),
            float(tcfg.get("comfort_global_weight", 1.0))
            * _finite_clip(comfort)
            / max(float(tcfg.get("comfort_scale", 80.0)), 1.0e-6),
        ],
        axis=1,
    )

    risks = runtime_risk_scores(runtime, candidates, cfg)
    risk = np.stack(
        [
            _finite_clip(risks.get("hard_agent", np.zeros((K,)))),
            _finite_clip(risks.get("soft_agent", np.zeros((K,)))),
            _finite_clip(risks.get("agent_ttc", np.zeros((K,)))),
            _finite_clip(risks.get("hard_off_route", np.zeros((K,)))),
            _finite_clip(risks.get("soft_off_route", np.zeros((K,)))),
            _finite_clip(risks.get("red_light", np.zeros((K,)))),
        ],
        axis=1,
    )
    out = np.concatenate([quality, risk], axis=1).astype(np.float64)
    names = list(VALUE_OBSERVABLE_NAMES)
    ic = (((cfg.get("runtime", {}) or {}).get("decisive_frontier_value", {}) or {}).get("incumbent_contrastive_extremal_recovery", {}) or {}) if isinstance(cfg, dict) else {}
    if bool(ic.get("instrument_future_response_observables", False)):
        from bdse.planner.response_value_observables import FUTURE_RESPONSE_OBSERVABLE_NAMES, runtime_future_response_observable_costs
        future, fn = runtime_future_response_observable_costs(runtime, candidates, cfg)
        if fn != FUTURE_RESPONSE_OBSERVABLE_NAMES or future.shape != (K, len(FUTURE_RESPONSE_OBSERVABLE_NAMES)):
            raise ValueError("V43 future-response observable schema mismatch")
        out = np.concatenate([out, future], axis=1)
        names.extend(fn)
    if bool(ic.get("instrument_plan_conditioned_response_observables", False)):
        from bdse.planner.response_value_observables import PLAN_CONDITIONED_RESPONSE_ALL_NAMES, runtime_plan_conditioned_response_observable_costs
        pc, pn = runtime_plan_conditioned_response_observable_costs(runtime, candidates, cfg)
        if pn != PLAN_CONDITIONED_RESPONSE_ALL_NAMES or pc.shape != (K, len(PLAN_CONDITIONED_RESPONSE_ALL_NAMES)):
            raise ValueError("V44 plan-conditioned response observable schema mismatch")
        out = np.concatenate([out, pc], axis=1)
        names.extend(pn)
    if bool(ic.get("instrument_interaction_response_field_observables", False)):
        from bdse.planner.interaction_response_field import RESPONSE_FIELD_OBSERVABLE_NAMES, runtime_interaction_response_field_observable_costs
        rf, rn = runtime_interaction_response_field_observable_costs(runtime, candidates, cfg)
        if rn != RESPONSE_FIELD_OBSERVABLE_NAMES or rf.shape != (K, len(RESPONSE_FIELD_OBSERVABLE_NAMES)):
            raise ValueError("V45 interaction-response-field observable schema mismatch")
        out = np.concatenate([out, rf], axis=1)
        names.extend(rn)
    if bool(ic.get("instrument_distributional_interaction_response_observables", False)):
        from bdse.planner.distributional_interaction_response import DIRP_OBSERVABLE_NAMES, runtime_distributional_interaction_response_observable_costs
        dr, dn = runtime_distributional_interaction_response_observable_costs(runtime, candidates, cfg)
        if dn != DIRP_OBSERVABLE_NAMES or dr.shape != (K, len(DIRP_OBSERVABLE_NAMES)):
            raise ValueError("V46 distributional-interaction-response observable schema mismatch")
        out = np.concatenate([out, dr], axis=1)
        names.extend(dn)
    if bool(ic.get("instrument_future_state_factorization_observables", False)):
        from bdse.planner.future_state_factorization import FSFR_OBSERVABLE_NAMES, runtime_future_state_factorization_observable_costs
        if _v50_elide_unused_fsfr_2d(cfg):
            requested = ["fsfr_plan_1d_occupancy_cost", "fsfr_predicted_demo_cost"]
            fs_req, fn_req = runtime_future_state_factorization_observable_costs(
                runtime, candidates, cfg, requested_names=requested
            )
            if fn_req != requested or fs_req.shape != (K, len(requested)):
                raise ValueError("V50 FSFR-2D elision required-observable schema mismatch")
            # Keep the exact V47 observable *schema* so the byte-locked V48
            # tournament sees the same interface.  PLAN-2D is scientifically
            # closed and is not indexed by the V48/V49/V50 Q/P/E functional.
            # V50 persists only selected_outcome_probe diagnostics, which contain
            # live Q/P/E and proposal identity, never this dead column.
            fs = np.zeros((K, len(FSFR_OBSERVABLE_NAMES)), dtype=np.float64)
            pos = {n: i for i, n in enumerate(FSFR_OBSERVABLE_NAMES)}
            for j, n in enumerate(fn_req):
                fs[:, pos[n]] = fs_req[:, j]
            fn = list(FSFR_OBSERVABLE_NAMES)
        else:
            fs, fn = runtime_future_state_factorization_observable_costs(runtime, candidates, cfg)
        if fn != FSFR_OBSERVABLE_NAMES or fs.shape != (K, len(FSFR_OBSERVABLE_NAMES)):
            raise ValueError("V47 future-state-factorization observable schema mismatch")
        out = np.concatenate([out, fs], axis=1)
        names.extend(fn)
    if out.shape != (K, len(names)) or not np.all(np.isfinite(out)):
        raise ValueError("deployment value-observable matrix is malformed or non-finite")
    return out, names
