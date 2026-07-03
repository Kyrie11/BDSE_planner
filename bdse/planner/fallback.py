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


def runtime_safety_flags_from_runtime(runtime: RuntimeFeatures, candidates: CandidateBank, cfg: dict[str, Any]) -> np.ndarray:
    flags = np.zeros((candidates.K,), dtype=bool)
    route = np.asarray(runtime.map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32)
    width = float(runtime.map_features.get("route_corridor_width", cfg.get("candidate", {}).get("route_width_m", 4.0)))
    speed_limit = float(runtime.map_features.get("speed_limit_mps", 13.4))
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    for a in range(candidates.K):
        if not candidates.valid_mask[a]:
            continue
        traj = candidates.trajectories[a]
        off_route = bool((nearest_polyline_distance(traj[:, :2], route) > width + 1.0).any())
        speed_bad = bool((traj[:, 3] > speed_limit + 2.0).any())
        red_light_bad = False
        for sl in runtime.map_features.get("stop_lines", []) if runtime.map_features else []:
            status_red = bool(sl.get("red", False)) or ("red" in str(sl.get("status", "")).lower())
            if not status_red:
                continue
            xy = np.asarray(sl.get("xy", []), dtype=np.float32).reshape(-1, 2)
            if len(xy) >= 2 and _crosses_polyline(traj[:, :2], xy):
                red_light_bad = True
                break
        v = traj[:, 3]
        acc = finite_difference(v, dt)
        jerk = finite_difference(acc, dt)
        curv = compute_curvature(traj[:, :2])
        dyn_bad = bool((np.abs(acc) > 4.0).any() or (np.abs(jerk) > 8.0).any() or (np.abs(curv) > 0.35).any())
        agent_bad = False
        for j, valid in enumerate(runtime.agent_valid.astype(bool)):
            if not valid:
                continue
            cur = runtime.current_agents[j]
            vx = float(cur[5]) if cur.shape[0] > 5 else float(cur[3]) * np.cos(float(cur[2]))
            vy = float(cur[6]) if cur.shape[0] > 6 else float(cur[3]) * np.sin(float(cur[2]))
            times = traj[:, 4]
            pred = cur[:2][None, :] + np.stack([vx * times, vy * times], axis=1)
            if np.linalg.norm(traj[:, :2] - pred, axis=1).min() < 1.5:
                agent_bad = True
                break
        flags[a] = off_route or speed_bad or dyn_bad or agent_bad or red_light_bad
    return flags


def rule_based_runtime_scores(runtime: RuntimeFeatures, candidates: CandidateBank, cfg: dict[str, Any]) -> np.ndarray:
    scores = np.full((candidates.K,), np.inf, dtype=np.float32)
    route = np.asarray(runtime.map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32)
    for a in range(candidates.K):
        if not candidates.valid_mask[a]:
            continue
        traj = candidates.trajectories[a]
        route_cost = np.square(nearest_polyline_distance(traj[:, :2], route)).mean()
        progress_reward = max(float(traj[-1, 0]), 0.0)
        safety = runtime_safety_flags_from_runtime(runtime, candidates, cfg)[a]
        scores[a] = float(route_cost - 0.1 * progress_reward + 100.0 * float(safety))
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
