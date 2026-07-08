from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.selector import runtime_greedy_selector
from bdse.planner.tournament import TournamentResult, run_tournament
from bdse.utils import compute_curvature, finite_difference, nearest_polyline_distance


def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return bool((c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def _crosses_polyline(path_xy: np.ndarray, line_xy: np.ndarray) -> bool:
    path = np.asarray(path_xy, dtype=np.float32).reshape(-1, 2)
    line = np.asarray(line_xy, dtype=np.float32).reshape(-1, 2)
    if len(path) < 2 or len(line) < 2:
        return False
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        for j in range(len(line) - 1):
            if _segments_intersect(a, b, line[j], line[j + 1]):
                return True
    return False


@dataclass(slots=True)
class FallbackResult:
    action_index: int
    tournament: TournamentResult
    triggered: bool
    stage: str
    diagnostics: dict[str, Any]


def _trajectory_curvature_batch(xy: np.ndarray) -> np.ndarray:
    """Vectorized variant of ``compute_curvature`` for a K x T x 2 trajectory bank."""
    pts = np.nan_to_num(np.asarray(xy, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    if pts.ndim != 3 or pts.shape[1] < 3:
        return np.zeros(pts.shape[:2], dtype=np.float32)
    dx = np.gradient(pts[:, :, 0], axis=1)
    dy = np.gradient(pts[:, :, 1], axis=1)
    ddx = np.gradient(dx, axis=1)
    ddy = np.gradient(dy, axis=1)
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-6)
    return ((dx * ddy - dy * ddx) / denom).astype(np.float32)


def runtime_safety_flags_from_runtime(runtime: RuntimeFeatures, candidates: CandidateBank, cfg: dict[str, Any]) -> np.ndarray:
    """Return conservative per-candidate safety flags.

    This function is called several times per closed-loop tick.  The previous
    implementation recomputed route distances, kinematic derivatives, and
    agent constant-velocity distances inside a Python loop for every candidate.
    In the fallback rule-rerank path it was accidentally called once per
    candidate again, creating an O(K^2) hotspot.  Keep the same conservative
    semantics but compute candidate-bank terms in batches.
    """
    K = int(candidates.K)
    flags = np.zeros((K,), dtype=bool)
    valid_mask = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)[:K]
    if K <= 0 or not bool(valid_mask.any()):
        return flags

    traj = np.nan_to_num(np.asarray(candidates.trajectories, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    if traj.ndim != 3 or traj.shape[0] != K or traj.shape[2] < 4:
        return flags
    T = int(traj.shape[1])
    xy_all = traj[:, :, :2]

    route = np.asarray(runtime.map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32)
    width = float(runtime.map_features.get("route_corridor_width", cfg.get("candidate", {}).get("route_width_m", 4.0)))
    route_dist = nearest_polyline_distance(xy_all.reshape(-1, 2), route).reshape(K, T)
    off_route = (route_dist > width + 1.0).any(axis=1)

    speed_limit = float(runtime.map_features.get("speed_limit_mps", 13.4))
    speed_bad = (traj[:, :, 3] > speed_limit + 2.0).any(axis=1)

    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    v = traj[:, :, 3]
    if T >= 2:
        acc = np.gradient(v, dt, axis=1).astype(np.float32)
        jerk = np.gradient(acc, dt, axis=1).astype(np.float32)
    else:
        acc = np.zeros_like(v, dtype=np.float32)
        jerk = np.zeros_like(v, dtype=np.float32)
    curv = _trajectory_curvature_batch(xy_all)
    dyn_bad = (np.abs(acc) > 4.0).any(axis=1) | (np.abs(jerk) > 8.0).any(axis=1) | (np.abs(curv) > 0.35).any(axis=1)

    red_light_bad = np.zeros((K,), dtype=bool)
    for sl in runtime.map_features.get("stop_lines", []) if runtime.map_features else []:
        status_red = bool(sl.get("red", False)) or ("red" in str(sl.get("status", "")).lower())
        if not status_red:
            continue
        line_xy = np.asarray(sl.get("xy", []), dtype=np.float32).reshape(-1, 2)
        if len(line_xy) < 2:
            continue
        for a in np.flatnonzero(valid_mask & ~red_light_bad):
            if _crosses_polyline(xy_all[int(a)], line_xy):
                red_light_bad[int(a)] = True

    agent_bad = np.zeros((K,), dtype=bool)
    agent_valid = np.asarray(runtime.agent_valid, dtype=bool).reshape(-1)
    if agent_valid.size and getattr(runtime, "current_agents", None) is not None:
        times = traj[:, :, 4] if traj.shape[2] > 4 else np.arange(T, dtype=np.float32)[None, :] * dt
        for j in np.flatnonzero(agent_valid):
            cur = np.asarray(runtime.current_agents[int(j)], dtype=np.float32).reshape(-1)
            if cur.size < 4:
                continue
            vx = float(cur[5]) if cur.size > 5 else float(cur[3]) * np.cos(float(cur[2]))
            vy = float(cur[6]) if cur.size > 6 else float(cur[3]) * np.sin(float(cur[2]))
            pred_x = float(cur[0]) + vx * times
            pred_y = float(cur[1]) + vy * times
            dx = xy_all[:, :, 0] - pred_x
            dy = xy_all[:, :, 1] - pred_y
            agent_bad |= ((dx * dx + dy * dy).min(axis=1) < 1.5 * 1.5)

    flags = valid_mask & (off_route | speed_bad | dyn_bad | agent_bad | red_light_bad)
    return flags.astype(bool)


def rule_based_runtime_scores(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    cfg: dict[str, Any],
    safety_flags: np.ndarray | None = None,
) -> np.ndarray:
    """Fast rule fallback scores.

    ``safety_flags`` may be supplied by the caller to avoid recomputing the same
    O(K * T * agents) safety check.  This fixes the main closed-loop hotspot in
    the fallback rule-rerank branch while preserving the previous scoring rule.
    """
    scores = np.full((candidates.K,), np.inf, dtype=np.float32)
    valid_mask = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)[: candidates.K]
    if not bool(valid_mask.any()):
        return scores
    route = np.asarray(runtime.map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32)
    traj = np.nan_to_num(np.asarray(candidates.trajectories, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    route_dist = nearest_polyline_distance(traj[:, :, :2].reshape(-1, 2), route).reshape(candidates.K, traj.shape[1])
    route_cost = np.square(route_dist).mean(axis=1)
    progress_reward = np.maximum(traj[:, -1, 0], 0.0)
    if safety_flags is None:
        safety_flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
    safety_flags = np.asarray(safety_flags, dtype=bool).reshape(-1)[: candidates.K]
    scores[valid_mask] = (route_cost[valid_mask] - 0.1 * progress_reward[valid_mask] + 100.0 * safety_flags[valid_mask].astype(np.float32)).astype(np.float32)
    return scores


def conservative_fallback_action(candidates: CandidateBank) -> int:
    valid = np.flatnonzero(candidates.valid_mask)
    if len(valid) == 0:
        return 0
    speeds = candidates.trajectories[valid, -1, 3]
    progress = candidates.trajectories[valid, -1, 0]
    idx = sorted(valid.tolist(), key=lambda a: (float(speeds[list(valid).index(a)]), float(progress[list(valid).index(a)]), a))[0]
    return int(idx)


def apply_fallback_if_needed(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    atom_budget_costs: np.ndarray,
    atom_active_mask: np.ndarray,
    initial_tournament: TournamentResult,
    cfg: dict[str, Any],
) -> FallbackResult:
    fcfg = cfg.get("fallback", {})
    if not bool(fcfg.get("enabled", True)):
        return FallbackResult(initial_tournament.action_index, initial_tournament, False, "disabled", dict(initial_tournament.diagnostics))
    tau_delta = float(fcfg.get("tau_delta", 0.1))
    delta = float(initial_tournament.diagnostics.get("delta_hat_B", 0.0))
    if delta >= tau_delta:
        return FallbackResult(initial_tournament.action_index, initial_tournament, False, "not_triggered", dict(initial_tournament.diagnostics))

    runtime_flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
    best_tournament = initial_tournament
    stage = "triggered"
    local_cfg = dict(cfg)
    local_cfg["tournament"] = dict(cfg.get("tournament", {}))
    local_cfg["evidence"] = dict(cfg.get("evidence", {}))

    if bool(fcfg.get("expand_rivals", True)):
        for L in [8, 16, min(31, candidates.K - 1)]:
            local_cfg["tournament"]["L_infer"] = int(L)
            trial = run_tournament(predicted_base_cost, predicted_atom_costs, best_tournament.diagnostics.get("selected_atoms", []), candidates.valid_mask, runtime_flags, local_cfg)
            best_tournament = trial
            stage = f"rival_expanded_{L}"
            if float(trial.diagnostics.get("delta_hat_B", 0.0)) >= tau_delta:
                return FallbackResult(trial.action_index, trial, True, stage, dict(trial.diagnostics))

    if bool(fcfg.get("expand_budget", True)):
        for B in [8, 16, 32]:
            if B < int(cfg.get("evidence", {}).get("budget", 16)):
                continue
            sel = runtime_greedy_selector(
                predicted_base_cost,
                predicted_atom_costs,
                atom_budget_costs,
                candidates.valid_mask,
                runtime_flags,
                budget=float(B),
                L_infer=int(local_cfg.get("tournament", {}).get("L_infer", 16)),
                gamma_max=float(cfg.get("selector", {}).get("gamma_max_default", 100.0)),
                eta_pred=float(cfg.get("selector", {}).get("eta_pred", 1.0)),
                atom_active_mask=atom_active_mask,
            )
            trial = run_tournament(predicted_base_cost, predicted_atom_costs, sel.selected, candidates.valid_mask, runtime_flags, local_cfg)
            best_tournament = trial
            stage = f"budget_expanded_{B}"
            if float(trial.diagnostics.get("delta_hat_B", 0.0)) >= tau_delta:
                return FallbackResult(trial.action_index, trial, True, stage, dict(trial.diagnostics))

    top_k = int(fcfg.get("rule_rerank_top_k", 5))
    if top_k > 0:
        valid_scores = best_tournament.scores.copy()
        top_actions = np.argsort(-valid_scores)[:top_k]
        rule_cost = rule_based_runtime_scores(runtime, candidates, cfg)
        best = min([int(a) for a in top_actions if candidates.valid_mask[a]], key=lambda a: (float(rule_cost[a]), a), default=best_tournament.action_index)
        if np.isfinite(rule_cost[best]) and not runtime_flags[best]:
            best_tournament.action_index = int(best)
            stage = "rule_rerank"
            return FallbackResult(int(best), best_tournament, True, stage, {**best_tournament.diagnostics, "rule_cost": float(rule_cost[best])})

    action = conservative_fallback_action(candidates)
    best_tournament.action_index = int(action)
    return FallbackResult(int(action), best_tournament, True, "conservative_fallback", dict(best_tournament.diagnostics))
