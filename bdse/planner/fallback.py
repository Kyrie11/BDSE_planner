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


def runtime_safety_flag_components(runtime: RuntimeFeatures, candidates: CandidateBank, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return tiered runtime safety flags for candidate actions.

    v26 used one conservative boolean flag for several very different failure
    modes: off-route, red-light, close agent proximity, high speed, and dynamic
    discomfort.  In closed loop this often marked every valid candidate as
    unsafe, so the hard-mask could not actually select a safe alternative.  This
    helper separates hard violations from soft risk indicators.  The deployed
    hard mask can then be applied only to infeasible/high-risk candidates, while
    soft risk is handled by rule reranking and evidence/certificate scores.
    """
    K = int(candidates.K)
    zeros = np.zeros((K,), dtype=bool)
    out = {
        "valid": zeros.copy(),
        "off_route_soft": zeros.copy(),
        "off_route_hard": zeros.copy(),
        "speed_soft": zeros.copy(),
        "speed_hard": zeros.copy(),
        "dyn_soft": zeros.copy(),
        "dyn_hard": zeros.copy(),
        "agent_soft": zeros.copy(),
        "agent_hard": zeros.copy(),
        "red_light": zeros.copy(),
        "soft": zeros.copy(),
        "hard": zeros.copy(),
        "legacy": zeros.copy(),
    }
    valid_mask = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)[:K]
    if valid_mask.shape[0] < K:
        valid_mask = np.pad(valid_mask, (0, K - valid_mask.shape[0]), constant_values=False)
    out["valid"] = valid_mask.astype(bool)
    if K <= 0 or not bool(valid_mask.any()):
        return out

    traj = np.nan_to_num(np.asarray(candidates.trajectories, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    if traj.ndim != 3 or traj.shape[0] != K or traj.shape[2] < 4:
        return out
    T = int(traj.shape[1])
    xy_all = traj[:, :, :2]
    rsc = cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}

    route = np.asarray(runtime.map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32)
    width = float(runtime.map_features.get("route_corridor_width", cfg.get("candidate", {}).get("route_width_m", 4.0)))
    route_dist = nearest_polyline_distance(xy_all.reshape(-1, 2), route).reshape(K, T)
    soft_off_margin = float(rsc.get("soft_off_route_margin_m", 1.0))
    hard_off_margin = float(rsc.get("hard_off_route_margin_m", 3.0))
    out["off_route_soft"] = (route_dist > width + soft_off_margin).any(axis=1)
    out["off_route_hard"] = (route_dist > width + hard_off_margin).any(axis=1)

    speed_limit = float(runtime.map_features.get("speed_limit_mps", 13.4))
    soft_speed_margin = float(rsc.get("soft_speed_margin_mps", 2.0))
    hard_speed_margin = float(rsc.get("hard_speed_margin_mps", 5.0))
    out["speed_soft"] = (traj[:, :, 3] > speed_limit + soft_speed_margin).any(axis=1)
    out["speed_hard"] = (traj[:, :, 3] > speed_limit + hard_speed_margin).any(axis=1)

    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    v = traj[:, :, 3]
    if T >= 2:
        acc = np.gradient(v, dt, axis=1).astype(np.float32)
        jerk = np.gradient(acc, dt, axis=1).astype(np.float32)
    else:
        acc = np.zeros_like(v, dtype=np.float32)
        jerk = np.zeros_like(v, dtype=np.float32)
    curv = _trajectory_curvature_batch(xy_all)
    soft_acc = float(rsc.get("soft_acc_abs", 4.0))
    soft_jerk = float(rsc.get("soft_jerk_abs", 8.0))
    soft_curv = float(rsc.get("soft_curvature_abs", 0.35))
    hard_acc = float(rsc.get("hard_acc_abs", 7.0))
    hard_jerk = float(rsc.get("hard_jerk_abs", 15.0))
    hard_curv = float(rsc.get("hard_curvature_abs", 0.55))
    out["dyn_soft"] = (np.abs(acc) > soft_acc).any(axis=1) | (np.abs(jerk) > soft_jerk).any(axis=1) | (np.abs(curv) > soft_curv).any(axis=1)
    out["dyn_hard"] = (np.abs(acc) > hard_acc).any(axis=1) | (np.abs(jerk) > hard_jerk).any(axis=1) | (np.abs(curv) > hard_curv).any(axis=1)

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
    out["red_light"] = red_light_bad

    agent_soft = np.zeros((K,), dtype=bool)
    agent_hard = np.zeros((K,), dtype=bool)
    agent_valid = np.asarray(runtime.agent_valid, dtype=bool).reshape(-1)
    if agent_valid.size and getattr(runtime, "current_agents", None) is not None:
        times = traj[:, :, 4] if traj.shape[2] > 4 else np.arange(T, dtype=np.float32)[None, :] * dt
        soft_r = float(rsc.get("soft_agent_radius_m", 1.5))
        hard_r = float(rsc.get("hard_agent_radius_m", 0.85))
        soft_r2 = soft_r * soft_r
        hard_r2 = hard_r * hard_r
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
            d2 = (dx * dx + dy * dy).min(axis=1)
            agent_soft |= d2 < soft_r2
            agent_hard |= d2 < hard_r2
    out["agent_soft"] = agent_soft
    out["agent_hard"] = agent_hard

    include_speed_soft = bool(rsc.get("include_speed_in_soft", True))
    include_dyn_soft = bool(rsc.get("include_dynamics_in_soft", True))
    include_speed_hard = bool(rsc.get("include_speed_in_hard", False))
    include_dyn_hard = bool(rsc.get("include_dynamics_in_hard", False))
    soft = out["off_route_soft"] | out["agent_soft"] | out["red_light"]
    hard = out["off_route_hard"] | out["agent_hard"] | out["red_light"]
    if include_speed_soft:
        soft = soft | out["speed_soft"]
    if include_dyn_soft:
        soft = soft | out["dyn_soft"]
    if include_speed_hard:
        hard = hard | out["speed_hard"]
    if include_dyn_hard:
        hard = hard | out["dyn_hard"]
    legacy = out["off_route_soft"] | out["speed_soft"] | out["dyn_soft"] | out["agent_soft"] | out["red_light"]
    out["soft"] = valid_mask & soft
    out["hard"] = valid_mask & hard
    out["legacy"] = valid_mask & legacy
    return out


def runtime_safety_flags_from_runtime(runtime: RuntimeFeatures, candidates: CandidateBank, cfg: dict[str, Any]) -> np.ndarray:
    """Return per-candidate flags used by the tournament hard filter.

    ``runtime_safety.flag_mode`` controls the semantics:
      - legacy: v26-compatible conservative union;
      - soft: soft risk union;
      - hard: only hard/infeasible runtime violations;
      - tiered: use soft flags if at least one valid soft-safe action exists,
        otherwise fall back to hard flags so the planner is not forced into an
        all-unsafe mask at dense interactions.
    """
    comp = runtime_safety_flag_components(runtime, candidates, cfg)
    valid = comp["valid"]
    rsc = cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}
    mode = str(rsc.get("flag_mode", "legacy")).lower()
    if mode == "hard":
        flags = comp["hard"]
    elif mode == "soft":
        flags = comp["soft"]
    elif mode == "tiered":
        soft_safe_exists = bool((valid & ~comp["soft"]).any())
        flags = comp["soft"] if soft_safe_exists else comp["hard"]
    else:
        flags = comp["legacy"]
    return (valid & np.asarray(flags, dtype=bool).reshape(-1)[: int(candidates.K)]).astype(bool)


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
    # Reranking can be more nuanced than the tournament hard mask: hard flags
    # receive a very large penalty, while soft risks receive a smaller penalty
    # so that the planner can still choose forward progress when every candidate
    # is near a soft interaction envelope.
    rsc = cfg.get("runtime_safety", {}) if isinstance(cfg, dict) else {}
    comp = runtime_safety_flag_components(runtime, candidates, cfg)
    if safety_flags is None:
        safety_flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
    hard_flags = np.asarray(safety_flags, dtype=bool).reshape(-1)[: candidates.K]
    soft_flags = np.asarray(comp.get("soft", np.zeros((candidates.K,), dtype=bool)), dtype=bool).reshape(-1)[: candidates.K]
    hard_penalty = float(rsc.get("rule_hard_penalty", 1000.0))
    soft_penalty = float(rsc.get("rule_soft_penalty", 25.0))
    progress_weight = float(rsc.get("rule_progress_weight", 0.1))
    scores[valid_mask] = (
        route_cost[valid_mask]
        - progress_weight * progress_reward[valid_mask]
        + hard_penalty * hard_flags[valid_mask].astype(np.float32)
        + soft_penalty * (soft_flags[valid_mask] & ~hard_flags[valid_mask]).astype(np.float32)
    ).astype(np.float32)
    return scores


def conservative_fallback_action(
    candidates: CandidateBank,
    safety_flags: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
) -> int:
    """Choose a safe recovery action without collapsing to zero progress.

    The v25 conservative fallback sorted by low terminal speed and low progress,
    so many safety-triggered replans degenerated to action 0 / near-stop.  That
    improved neither fixed-budget evidence use nor closed-loop progress.  This
    recovery remains rule-only, but it is lexicographic: prefer unflagged valid
    candidates, then minimize route/lateral/comfort cost while rewarding forward
    progress.  It is only used when the certificate cannot provide an accepted
    action.
    """
    valid = np.flatnonzero(np.asarray(candidates.valid_mask, dtype=bool).reshape(-1))
    if len(valid) == 0:
        return 0
    traj = np.nan_to_num(np.asarray(candidates.trajectories, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    K = traj.shape[0]
    flags = np.zeros((K,), dtype=bool) if safety_flags is None else np.asarray(safety_flags, dtype=bool).reshape(-1)
    if flags.shape[0] < K:
        flags = np.pad(flags, (0, K - flags.shape[0]), constant_values=True)
    flags = flags[:K]
    safe_valid = [int(a) for a in valid.tolist() if not bool(flags[int(a)])]
    pool = safe_valid if safe_valid else [int(a) for a in valid.tolist()]
    xy = traj[:, :, :2]
    lateral_mean = np.mean(np.abs(xy[:, :, 1]), axis=1)
    lateral_final = np.abs(xy[:, -1, 1])
    progress = xy[:, -1, 0]
    speed_final = traj[:, -1, 3] if traj.shape[-1] > 3 else np.zeros((K,), dtype=np.float32)
    if xy.shape[1] > 1:
        step = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
        path_len = step.sum(axis=1)
    else:
        path_len = np.zeros((K,), dtype=np.float32)
    fc = ((cfg or {}).get("fallback", {}) or {}).get("safe_progress_recovery", {}) if isinstance(cfg, dict) else {}
    min_progress = float(fc.get("min_progress", -1.0))
    progress_pool = [a for a in pool if float(progress[a]) >= min_progress]
    if progress_pool:
        pool = progress_pool
    progress_w = float(fc.get("progress_weight", 0.50))
    path_w = float(fc.get("path_length_weight", 0.04))
    lateral_w = float(fc.get("lateral_weight", 1.2))
    lateral_final_w = float(fc.get("lateral_final_weight", 0.6))
    low_speed_thr = float(fc.get("low_speed_threshold", 0.25))
    low_speed_penalty = float(fc.get("low_speed_penalty", 0.10))
    def cost(a: int) -> tuple[float, int]:
        c = (
            lateral_w * float(lateral_mean[a])
            + lateral_final_w * float(lateral_final[a])
            - progress_w * float(progress[a])
            - path_w * float(path_len[a])
            + (low_speed_penalty if float(speed_final[a]) < low_speed_thr else 0.0)
        )
        if bool(flags[a]):
            c += float(fc.get("unsafe_penalty", 1000.0))
        return (float(c), int(a))
    return int(min(pool, key=cost))


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
