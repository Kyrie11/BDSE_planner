from __future__ import annotations

from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, EvidenceBank, LabelOnlyFuture, RuntimeFeatures, TeacherLabels
from bdse.planner.evidence_atoms import _eval_traj, normalize_atom_costs, raw_local_costs_with_hard_events
from bdse.utils import compute_curvature, finite_difference, nearest_polyline_distance, route_progress_along_polyline


def global_comfort_cost(traj: np.ndarray, dt: float) -> float:
    v = traj[:, 3]
    acc = finite_difference(v, dt)
    jerk = finite_difference(acc, dt)
    curv = compute_curvature(traj[:, :2])
    cost = (
        np.maximum(0.0, np.abs(acc) - 3.0) ** 2
        + 0.25 * np.maximum(0.0, np.abs(jerk) - 5.0) ** 2
        + 4.0 * np.maximum(0.0, np.abs(curv) - 0.25) ** 2
    )
    return float(cost.mean())


def demo_cost(traj: np.ndarray, logged_ego: np.ndarray | None) -> float:
    if logged_ego is None or logged_ego.size == 0:
        return 0.0
    gt = np.asarray(logged_ego, dtype=np.float32)
    n = min(len(gt), len(traj))
    if n == 0:
        return 0.0
    return float(np.square(traj[:n, :2] - gt[:n, :2]).sum(axis=1).mean())


def evaluate_base_costs(runtime: RuntimeFeatures, label_future: LabelOnlyFuture | None, candidates: CandidateBank, cfg: dict[str, Any]) -> np.ndarray:
    tcfg = cfg.get("teacher", {})
    route = np.asarray(runtime.map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    if route.size:
        route = route[np.isfinite(route).all(axis=1)]
    if len(route) < 2:
        route = np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)
    base_dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    stride = max(1, int(cfg.get("teacher", {}).get("cost_eval_stride", 1)))
    dt_eval = base_dt * stride
    K = candidates.K
    route_dev = np.zeros((K,), dtype=np.float32)
    progress_def = np.zeros((K,), dtype=np.float32)
    comfort = np.zeros((K,), dtype=np.float32)
    demo = np.zeros((K,), dtype=np.float32)
    terminal_progress = route_progress_along_polyline(candidates.trajectories[:, -1, :2], route)
    best_progress = terminal_progress[candidates.valid_mask].max() if np.any(candidates.valid_mask) else 0.0
    logged_ego_eval = None if label_future is None else _eval_traj(label_future.logged_ego, cfg)
    for a in range(K):
        traj = _eval_traj(candidates.trajectories[a], cfg)
        route_dev[a] = float(np.square(nearest_polyline_distance(traj[:, :2], route)).mean())
        progress_def[a] = float(max(0.0, best_progress - terminal_progress[a]))
        comfort[a] = global_comfort_cost(traj, dt_eval)
        demo[a] = demo_cost(traj, logged_ego_eval)
    route_dev = np.nan_to_num(route_dev, nan=1e6, posinf=1e6, neginf=1e6)
    progress_def = np.nan_to_num(progress_def, nan=1e6, posinf=1e6, neginf=1e6)
    comfort = np.nan_to_num(comfort, nan=1e6, posinf=1e6, neginf=1e6)
    demo = np.nan_to_num(demo, nan=1e6, posinf=1e6, neginf=1e6)
    J = (
        float(tcfg.get("route_weight", 50.0)) * route_dev / max(float(tcfg.get("route_scale", 1.0)), 1e-6)
        + float(tcfg.get("progress_weight", 5.0)) * progress_def / max(float(tcfg.get("progress_scale", 10.0)), 1e-6)
        + float(tcfg.get("comfort_global_weight", 1.0)) * comfort / max(float(tcfg.get("comfort_scale", 80.0)), 1e-6)
        + float(tcfg.get("demo_weight", 1.0)) * demo / max(float(tcfg.get("demo_scale", 320.0)), 1e-6)
    ).astype(np.float32)
    J = np.nan_to_num(J, nan=1e9, posinf=1e9, neginf=1e9)
    J = J.astype(np.float64)
    J[~candidates.valid_mask] = np.inf
    return J


def evaluate_teacher_costs(
    runtime: RuntimeFeatures,
    label_future: LabelOnlyFuture | None,
    candidates: CandidateBank,
    evidence_bank: EvidenceBank,
    cfg: dict[str, Any],
) -> TeacherLabels:
    if cfg.get("teacher", {}).get("separate_hard_gate", False):
        raise ValueError("BDSE teacher cost has exactly J_base_T + sum_i g_i_T; no separate hard gate is allowed.")
    J_base = evaluate_base_costs(runtime, label_future, candidates, cfg)
    raw, hard_events = raw_local_costs_with_hard_events(evidence_bank.atoms, candidates, runtime, label_future, cfg)
    raw = np.nan_to_num(raw, nan=1e6, posinf=1e6, neginf=1e6)
    g = normalize_atom_costs(raw, evidence_bank.atoms, cfg)
    g = np.nan_to_num(g, nan=1e6, posinf=1e6, neginf=1e6)
    J_evid = g.sum(axis=0, dtype=np.float64)
    J_T = J_base.astype(np.float64) + J_evid
    J_T[~candidates.valid_mask] = np.inf
    if not np.any(candidates.valid_mask):
        raise ValueError("Teacher cost requires at least one valid candidate")
    a_star = int(np.argmin(J_T))
    hard_mask = evidence_bank.hard_mask()
    hard_violation = np.zeros((candidates.K,), dtype=bool)
    if hard_mask.size:
        hard_violation = hard_events[hard_mask].any(axis=0) & candidates.valid_mask
    labels = TeacherLabels(
        J_base=J_base.astype(np.float64),
        g_evid=g.astype(np.float32),
        J_evid=J_evid.astype(np.float64),
        J_T=J_T.astype(np.float64),
        a_star=a_star,
        hard_violation_mask=hard_violation,
        diagnostics={
            "valid_candidate_count": int(candidates.valid_mask.sum()),
            "teacher_cost_min": float(J_T[a_star]),
            "atom_count": len(evidence_bank.atoms),
            "hard_violation_rate": float(hard_violation[candidates.valid_mask].mean()) if np.any(candidates.valid_mask) else 0.0,
            "hard_event_atom_count": int(hard_events[hard_mask].any(axis=1).sum()) if hard_mask.size else 0,
        },
    )
    labels.validate_partition(candidates.valid_mask)
    return labels


def teacher_margin(J_T: np.ndarray, a: int, b: int) -> float:
    return float(J_T[b] - J_T[a])


def residual_margin(teacher: TeacherLabels, a: int, b: int) -> float:
    """Evidence-only margin delta for pair (a,b).

    Mathematically this is also
        (J_T[b] - J_T[a]) - (J_base[b] - J_base[a]).
    In real nuPlan samples J_base can be very large, so subtracting two large
    margins can lose enough floating-point precision to trip a strict closure
    assertion even though the stored teacher partition is valid.  Use the
    already-materialized evidence partition directly; validate_residual_closure
    below still checks it against the atom-level sum.
    """
    return float(np.asarray(teacher.J_evid, dtype=np.float64)[b] - np.asarray(teacher.J_evid, dtype=np.float64)[a])


def validate_residual_closure(teacher: TeacherLabels, pairs: np.ndarray, atol: float = 1e-4, rtol: float = 1e-6) -> None:
    g64 = np.asarray(teacher.g_evid, dtype=np.float64)
    evid64 = np.asarray(teacher.J_evid, dtype=np.float64)
    for a, b in np.asarray(pairs, dtype=np.int64):
        a_i, b_i = int(a), int(b)
        lhs = float(evid64[b_i] - evid64[a_i])
        rhs = float((g64[:, b_i] - g64[:, a_i]).sum(dtype=np.float64))
        if not np.isclose(lhs, rhs, atol=atol, rtol=rtol):
            raise AssertionError(
                "Residual margin must equal sum_i(g_i_T[b]-g_i_T[a]); "
                f"pair=({a_i},{b_i}) lhs={lhs:.9g} rhs={rhs:.9g} diff={lhs-rhs:.9g}"
            )
