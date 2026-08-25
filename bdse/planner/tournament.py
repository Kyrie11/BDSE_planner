from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import hashlib
import math

import numpy as np

from bdse.planner.pair_screen import build_rival_sets_from_base, SAFE_LIKE_MANEUVER_IDS, PROGRESSIVE_MANEUVER_IDS
from bdse.model.potential_projection import project_pair_residual_to_action_potential_numpy
from bdse.planner.selector import _finite_cost_for_margin, budgeted_margin, full_interface_margin, margin_normalization_scale
from bdse.utils import softmin_np




def selected_pair_sigma_from_action_variance(
    predicted_atom_variance: np.ndarray | None,
    selected_atoms: list[int] | np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build sigma(a,b) from selected per-atom action variances.

    The model predicts a variance for each queried g_i(a).  For a pair margin
    d_i(a,b)=g_i(b)-g_i(a), we use var_i(a)+var_i(b), summed over selected
    atoms.  Unqueried actions/atoms should be zero in the supplied variance
    matrix, matching the sparse deployment interface.
    """
    if predicted_atom_variance is None:
        return None
    var = np.asarray(predicted_atom_variance, dtype=np.float32)
    if var.ndim != 2:
        return None
    K = var.shape[1]
    selected = np.asarray(selected_atoms, dtype=np.int64).reshape(-1)
    selected = selected[(selected >= 0) & (selected < var.shape[0])]
    action_var = np.zeros((K,), dtype=np.float32)
    if selected.size:
        action_var = np.maximum(var[selected], 0.0).sum(axis=0).astype(np.float32)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.shape[0] == K:
            action_var = np.where(valid, action_var, 0.0)
    return np.sqrt(np.maximum(action_var[:, None] + action_var[None, :], 0.0)).astype(np.float32)


@dataclass(slots=True)
class TournamentResult:
    action_index: int
    scores: np.ndarray
    margins: np.ndarray
    rival_sets: list[list[int]]
    diagnostics: dict[str, Any]


def _pair_selector_eta(cfg: dict[str, Any]) -> float:
    """Eta in the same units as the pair-conditioned margin matrix.

    Pair-conditioned deployment may use normalized margins, while the legacy
    tournament rival builder previously read selector.eta_pred in raw cost units.
    Keeping this consistent with training prevents a large pair-screen mismatch.
    """
    sc = cfg.get("selector", {}) if isinstance(cfg, dict) else {}
    normalize = bool(cfg.get("model", {}).get("pair_margin_normalized", False)) if isinstance(cfg, dict) else False
    if normalize:
        return float(sc.get("normalized_eta_pred", sc.get("eta_pred", 0.1)))
    return float(sc.get("eta_pred", 1.0))


def build_rival_sets(
    predicted_full_margin: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    L_infer: int = 16,
    eta_pred: float = 1.0,
) -> list[list[int]]:
    M = np.asarray(predicted_full_margin, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    K = len(valid)
    full_cost_proxy = -M.mean(axis=1)
    rivals: list[list[int]] = []
    for a in range(K):
        if not valid[a]:
            rivals.append([])
            continue
        cand = [b for b in range(K) if b != a and valid[b]]
        top = sorted(cand, key=lambda b: (float(full_cost_proxy[b]), abs(float(M[a, b])), b))[:L_infer]
        near = [b for b in cand if abs(float(M[a, b])) < eta_pred]
        safety = [b for b in cand if runtime_safety_flags[b]]
        all_rivals = sorted(set(top + near + safety), key=lambda b: (abs(float(M[a, b])), b))
        if len(all_rivals) > max(L_infer, len(safety)):
            safety_set = set(safety)
            kept = [b for b in all_rivals if b in safety_set]
            for b in all_rivals:
                if b not in safety_set and len(kept) < L_infer:
                    kept.append(b)
            all_rivals = kept
        rivals.append(all_rivals)
    return rivals


def tournament_scores(
    margins: np.ndarray,
    valid_mask: np.ndarray,
    rival_sets: list[list[int]],
    use_softmin: bool = True,
    softmin_tau: float = 1.0,
    beta_uncertainty: float = 0.0,
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    K = margins.shape[0]
    scores = np.full((K,), -1e9, dtype=np.float32)
    for a in range(K):
        if not valid_mask[a]:
            continue
        rivals = rival_sets[a]
        if not rivals:
            scores[a] = 0.0
            continue
        vals = np.asarray([margins[a, b] for b in rivals], dtype=np.float32)
        if sigma is not None and beta_uncertainty > 0:
            vals = vals - float(beta_uncertainty) * np.asarray([sigma[a, b] for b in rivals], dtype=np.float32)
        scores[a] = float(softmin_np(vals, softmin_tau if use_softmin else 0.0))
    return scores


def _apply_safety_score_guard(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Optional final safety dominance guard for normalized tournament scores.

    ``runtime_safety_flags`` marks candidates that violate a cheap hard/runtime
    safety check.  The guard is disabled by default and activated only through
    tournament.unsafe_action_score_penalty and/or
    tournament.prefer_unflagged_action_margin.  This keeps legacy behavior while
    letting BDSE use the same safety dominance that the hard-safety-only baseline
    empirically exploited in closed loop.
    """
    guarded = np.asarray(scores, dtype=np.float32).copy()
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    flags = np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)
    n = guarded.shape[0]
    if valid.shape[0] < n:
        valid = np.pad(valid, (0, n - valid.shape[0]), constant_values=False)
    valid = valid[:n]
    if flags.shape[0] < n:
        flags = np.pad(flags, (0, n - flags.shape[0]), constant_values=False)
    flags = flags[:n]
    tc = cfg.get("tournament", {}) if isinstance(cfg, dict) else {}
    raw_action = int(np.argmax(guarded)) if guarded.size else 0
    unsafe = valid & flags
    safe_valid = valid & ~flags
    penalty = max(float(tc.get("unsafe_action_score_penalty", 0.0)), 0.0)
    hard_filter = bool(tc.get("hard_filter_unsafe_actions", False))
    hard_filter_applied = False
    if hard_filter and bool(safe_valid.any()):
        # v25 still selected runtime-flagged actions in ~32--49% of replans.
        # For closed-loop control, cheap hard/runtime flags are constraints, not
        # soft preferences.  Preserve fixed evidence budget but mask unsafe
        # candidates before tournament/utility tie-breaking whenever a feasible
        # unflagged candidate exists.
        guarded[unsafe] = -1e9
        hard_filter_applied = True
    elif penalty > 0.0 and bool(unsafe.any()):
        guarded[unsafe] = guarded[unsafe] - penalty
    action = int(np.argmax(guarded)) if guarded.size else 0
    prefer_margin_raw = tc.get("prefer_unflagged_action_margin", None)
    switched_to_unflagged = False
    if prefer_margin_raw is not None and 0 <= action < n and bool(flags[action]):
        if bool(safe_valid.any()):
            safe_idx = np.flatnonzero(safe_valid)
            best_safe = int(safe_idx[int(np.argmax(guarded[safe_idx]))])
            if hard_filter or float(guarded[best_safe]) >= float(guarded[action]) - float(prefer_margin_raw):
                action = best_safe
                switched_to_unflagged = True
    selected_flag = bool(flags[action]) if 0 <= action < n else True
    safe_available = bool(safe_valid.any())
    all_flagged = bool(valid.any() and not safe_available)
    diag = {
        "action_before_safety_guard": int(raw_action),
        "safety_guard_applied": bool(action != raw_action or penalty > 0.0 or switched_to_unflagged or hard_filter_applied),
        "unsafe_action_score_penalty": float(penalty),
        "hard_filter_unsafe_actions": bool(hard_filter),
        "hard_filter_applied": bool(hard_filter_applied),
        "safe_action_available": safe_available,
        "all_actions_safety_flagged": all_flagged,
        "avoidable_selected_action_safety_flag": bool(selected_flag and safe_available),
        "prefer_unflagged_action_margin": float(prefer_margin_raw) if prefer_margin_raw is not None else None,
        "switched_to_unflagged": bool(switched_to_unflagged),
    }
    return guarded, action, diag




def _trajectory_utility_cost_np(
    candidate_trajectories: np.ndarray | None,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Deployment-time utility cost for certificate-constrained tie breaking.

    Lower is better.  Unlike evidence atoms, this uses only the already generated
    candidate geometry.  It is intentionally used *after* the BDSE safety/evidence
    tournament as a utility selector among actions whose certificate scores are
    already indistinguishable under a configured slack.  This preserves the fixed
    evidence budget and avoids turning utility into an unconstrained override.
    """
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    K = int(valid.shape[0])
    out = np.full((K,), np.inf, dtype=np.float32)
    traj = None if candidate_trajectories is None else np.asarray(candidate_trajectories, dtype=np.float32)
    if traj is None or traj.ndim < 3 or traj.shape[0] < K or traj.shape[1] < 1:
        out[valid] = 0.0
        return out
    tr = np.nan_to_num(traj[:K], nan=0.0, posinf=1e6, neginf=-1e6)
    xy = tr[..., :2]
    if xy.shape[1] > 1:
        dxy = np.diff(xy, axis=1)
        step = np.linalg.norm(dxy, axis=-1)
    else:
        step = np.zeros((K, 1), dtype=np.float32)
    progress = xy[:, -1, 0].astype(np.float32)
    path_len = step.sum(axis=1).astype(np.float32)
    lateral = xy[..., 1]
    lateral_mean = np.mean(np.abs(lateral), axis=1).astype(np.float32)
    lateral_final = np.abs(lateral[:, -1]).astype(np.float32)
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1)) if isinstance(cfg, dict) else 0.1
    speed = step / max(dt, 1e-3)
    speed_mean = speed.mean(axis=1).astype(np.float32) if speed.size else np.zeros((K,), dtype=np.float32)
    speed_final = speed[:, -1].astype(np.float32) if speed.ndim == 2 and speed.shape[1] else np.zeros((K,), dtype=np.float32)
    acc = np.diff(speed, axis=1) if speed.ndim == 2 and speed.shape[1] > 1 else np.zeros((K, 1), dtype=np.float32)
    jerk = np.diff(acc, axis=1) if acc.ndim == 2 and acc.shape[1] > 1 else np.zeros((K, 1), dtype=np.float32)
    acc_rms = np.sqrt(np.mean(acc * acc, axis=1)).astype(np.float32) if acc.size else np.zeros((K,), dtype=np.float32)
    jerk_rms = np.sqrt(np.mean(jerk * jerk, axis=1)).astype(np.float32) if jerk.size else np.zeros((K,), dtype=np.float32)
    comfort = 0.25 * acc_rms + 0.10 * jerk_rms
    yaw = tr[..., 2] if tr.shape[-1] > 2 else np.zeros((K, tr.shape[1]), dtype=np.float32)
    yaw_delta = np.arctan2(np.sin(np.diff(yaw, axis=1)), np.cos(np.diff(yaw, axis=1))) if yaw.shape[1] > 1 else np.zeros((K, 1), dtype=np.float32)
    curvature = np.mean(np.abs(yaw_delta), axis=1).astype(np.float32) if yaw_delta.size else np.zeros((K,), dtype=np.float32)

    uc = ((cfg.get("tournament", {}) or {}).get("utility_refinement", {}) or {}) if isinstance(cfg, dict) else {}
    cost = (
        float(uc.get("lateral_mean_weight", 1.25)) * lateral_mean
        + float(uc.get("lateral_final_weight", 0.60)) * lateral_final
        + float(uc.get("comfort_weight", 0.35)) * comfort
        + float(uc.get("curvature_weight", 0.25)) * curvature
        - float(uc.get("progress_weight", 0.14)) * progress
        - float(uc.get("path_length_weight", 0.015)) * path_len
        - float(uc.get("speed_weight", 0.02)) * speed_mean
    ).astype(np.float32)
    cost += np.where(speed_mean < float(uc.get("low_speed_threshold", 0.35)), float(uc.get("low_speed_penalty", 0.20)), 0.0).astype(np.float32)
    cost += np.where(speed_final < float(uc.get("low_final_speed_threshold", -1.0)), float(uc.get("low_final_speed_penalty", 0.0)), 0.0).astype(np.float32)
    flags = np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)
    if flags.shape[0] < K:
        flags = np.pad(flags, (0, K - flags.shape[0]), constant_values=False)
    cost += flags[:K].astype(np.float32) * float(uc.get("unsafe_penalty", 1000.0))
    out[valid[:K]] = cost[valid[:K]]
    return out.astype(np.float32)


def _certificate_utility_refinement_context(
    scores: np.ndarray,
    action: int,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    cfg: dict[str, Any],
    candidate_trajectories: np.ndarray | None = None,
    margins: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the frozen deployment-equivalence set used by utility refinement.

    V64.3.17 reuses this exact set for DALER instead of reconstructing a looser
    challenger set after the legacy action has already been refined.  Keeping one
    implementation prevents a train/runtime semantic drift: the same score band,
    safety rule, top-k restriction, pair certificate and utility finiteness rule
    define both the legacy deployment refinement and DALER eligibility.
    """
    tc = cfg.get("tournament", {}) if isinstance(cfg, dict) else {}
    uc = tc.get("utility_refinement", {}) or {}
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n = int(scores.shape[0])
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    flags = np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)
    if valid.shape[0] < n:
        valid = np.pad(valid, (0, n - valid.shape[0]), constant_values=False)
    if flags.shape[0] < n:
        flags = np.pad(flags, (0, n - flags.shape[0]), constant_values=False)
    valid = valid[:n]
    flags = flags[:n]
    finite = valid & np.isfinite(scores)
    empty_mask = np.zeros((n,), dtype=bool)
    inf_cost = np.full((n,), np.inf, dtype=np.float32)
    out: dict[str, Any] = {
        "utility_refinement_enabled": bool(uc.get("enabled", False)),
        "_utility_refinement_eligible_mask": empty_mask,
        "_utility_refinement_cost": inf_cost,
    }
    if not bool(uc.get("enabled", False)):
        out.update({"utility_refinement_applied": False, "utility_refinement_reason": "disabled"})
        return out
    if not bool(finite.any()):
        out.update({"utility_refinement_applied": False, "utility_refinement_reason": "no_finite"})
        return out

    best_score = float(np.max(scores[finite]))
    slack = max(float(uc.get("score_slack", 0.25)), 0.0)
    eligible = finite & (scores >= best_score - slack)
    if bool(uc.get("require_unflagged", True)):
        safe_eligible = eligible & ~flags
        if bool(safe_eligible.any()):
            eligible = safe_eligible
    top_k = int(uc.get("top_k", 0))
    if top_k > 0 and int(eligible.sum()) > top_k:
        order = np.argsort(-np.where(eligible, scores, -np.inf))[:top_k]
        mask = np.zeros((n,), dtype=bool)
        mask[order] = True
        eligible = eligible & mask

    pair_cert_used = False
    pair_cert_kept = int(eligible.sum())
    current_for_cert = int(action) if 0 <= int(action) < n else int(np.argmax(np.where(finite, scores, -np.inf)))
    if bool(uc.get("pair_certificate_enabled", False)) and margins is not None and 0 <= current_for_cert < n:
        M = np.asarray(margins, dtype=np.float32)
        if M.ndim == 2 and M.shape[0] >= n and M.shape[1] >= n:
            tol = max(float(uc.get("pair_margin_tolerance", 0.05)), 0.0)
            cert_mask = np.zeros((n,), dtype=bool)
            cand0 = np.flatnonzero(eligible).astype(np.int64)
            for c in cand0.tolist():
                if int(c) == current_for_cert:
                    cert_mask[int(c)] = True
                else:
                    cert_mask[int(c)] = bool(
                        np.isfinite(M[int(c), current_for_cert])
                        and float(M[int(c), current_for_cert]) >= -tol
                    )
            if bool(cert_mask.any()):
                eligible = eligible & cert_mask
                pair_cert_used = True
                pair_cert_kept = int(eligible.sum())

    if int(eligible.sum()) <= 0:
        out.update({
            "utility_refinement_applied": False,
            "utility_refinement_reason": "empty_band",
            "utility_score_slack": float(slack),
            "utility_pair_certificate_used": bool(pair_cert_used),
            "utility_pair_certificate_enabled": bool(uc.get("pair_certificate_enabled", False)),
            "utility_pair_margin_tolerance": float(max(float(uc.get("pair_margin_tolerance", 0.05)), 0.0)),
            "utility_pair_certificate_kept": int(pair_cert_kept),
        })
        return out

    utility_cost = _trajectory_utility_cost_np(candidate_trajectories, valid, flags, cfg)
    candidate_mask = eligible & np.isfinite(utility_cost)
    cand = np.flatnonzero(candidate_mask).astype(np.int64)
    out.update({
        "_utility_refinement_eligible_mask": np.asarray(candidate_mask, dtype=bool),
        "_utility_refinement_cost": np.asarray(utility_cost, dtype=np.float32),
        "utility_score_slack": float(slack),
        "utility_pair_certificate_used": bool(pair_cert_used),
        "utility_pair_certificate_enabled": bool(uc.get("pair_certificate_enabled", False)),
        "utility_pair_margin_tolerance": float(max(float(uc.get("pair_margin_tolerance", 0.05)), 0.0)),
        "utility_pair_certificate_kept": int(pair_cert_kept),
    })
    if cand.size == 0:
        out.update({"utility_refinement_applied": False, "utility_refinement_reason": "no_utility"})
        return out

    best_util = int(sorted(cand.tolist(), key=lambda a: (float(utility_cost[a]), -float(scores[a]), int(a)))[0])
    current = int(action) if 0 <= int(action) < n else int(np.argmax(np.where(finite, scores, -np.inf)))
    min_improvement = float(uc.get("min_utility_improvement", 0.0))
    applied = bool(
        best_util != current
        and float(utility_cost[best_util]) <= float(utility_cost[current]) - min_improvement
    )
    chosen = int(best_util) if applied else int(current)
    out.update({
        "utility_refinement_applied": bool(applied),
        "utility_refinement_action_before": int(current),
        "utility_refinement_action_after": int(chosen),
        "utility_band_size": int(cand.size),
        "utility_best_score": float(scores[best_util]),
        "utility_current_score": float(scores[current]) if 0 <= current < n else float("nan"),
        "utility_best_cost": float(utility_cost[best_util]),
        "utility_current_cost": float(utility_cost[current]) if 0 <= current < n and np.isfinite(utility_cost[current]) else float("inf"),
        "_utility_refinement_chosen_action": int(chosen),
    })
    return out


def _apply_certificate_utility_refinement(
    scores: np.ndarray,
    action: int,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    cfg: dict[str, Any],
    candidate_trajectories: np.ndarray | None = None,
    margins: np.ndarray | None = None,
) -> tuple[int, dict[str, Any]]:
    """Lexicographic action refinement: certificate first, utility second.

    The selection arithmetic is intentionally delegated to
    :func:`_certificate_utility_refinement_context` so V64.3.17 DALER can consume
    exactly the same deployment-equivalence mask without changing legacy output.
    """
    ctx = _certificate_utility_refinement_context(
        scores,
        action,
        valid_mask,
        runtime_safety_flags,
        cfg,
        candidate_trajectories=candidate_trajectories,
        margins=margins,
    )
    if not bool(ctx.get("utility_refinement_enabled", False)):
        return int(action), ctx
    chosen = int(ctx.get("_utility_refinement_chosen_action", action))
    return chosen, ctx

def run_tournament(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    selected_atoms: list[int] | np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    cfg: dict[str, Any],
    sigma: np.ndarray | None = None,
    candidate_trajectories: np.ndarray | None = None,
    maneuver_ids: np.ndarray | None = None,
) -> TournamentResult:
    tc = cfg.get("tournament", {})
    sc = cfg.get("selector", {})
    rivals = build_rival_sets_from_base(
        predicted_base_cost,
        valid_mask,
        runtime_safety_flags,
        L_infer=int(tc.get("L_infer", 16)),
        eta0=float(sc.get("eta_pred", 1.0)),
        candidate_trajectories=candidate_trajectories,
        maneuver_ids=maneuver_ids,
        progress_rivals=int(sc.get("progress_rivals", 0)),
        maneuver_rivals=int(sc.get("maneuver_rivals", 0)),
    )
    M_B = budgeted_margin(predicted_base_cost, predicted_atom_costs, selected_atoms)
    epsilon_cal = float(tc.get("epsilon_cal", cfg.get("calibration", {}).get("epsilon_cal", 0.0)))
    M_eval = M_B - epsilon_cal
    scores = tournament_scores(
        M_eval,
        np.asarray(valid_mask, dtype=bool),
        rivals,
        use_softmin=bool(tc.get("use_softmin", True)),
        softmin_tau=float(tc.get("softmin_tau", 1.0)),
        beta_uncertainty=float(tc.get("beta_uncertainty", 0.0)),
        sigma=sigma,
    )
    scores, action, safety_guard_diag = _apply_safety_score_guard(scores, valid_mask, runtime_safety_flags, cfg)
    action, utility_refinement_diag = _apply_certificate_utility_refinement(
        scores,
        action,
        valid_mask,
        runtime_safety_flags,
        cfg,
        candidate_trajectories=candidate_trajectories,
        margins=M_eval,
    )
    sorted_scores = np.sort(scores[np.asarray(valid_mask, dtype=bool)])
    delta = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) >= 2 else float("inf")
    safety_idx = np.flatnonzero(np.asarray(runtime_safety_flags, dtype=bool) & np.asarray(valid_mask, dtype=bool))
    if safety_idx.size and action not in safety_idx:
        safety_lcb_min = float(np.min(M_eval[action, safety_idx]))
    elif action in safety_idx:
        safety_lcb_min = -float("inf")
    else:
        safety_lcb_min = float("inf")
    return TournamentResult(
        action_index=action,
        scores=scores,
        margins=M_eval,
        rival_sets=rivals,
        diagnostics={
            "delta_hat_B": delta,
            "selected_atoms": list(map(int, selected_atoms)),
            "valid_actions": int(np.asarray(valid_mask).sum()),
            "rival_source": "base_score_cheap_flags",
            "epsilon_cal": epsilon_cal,
            "beta_uncertainty": float(tc.get("beta_uncertainty", 0.0)),
            "sigma_used": bool(sigma is not None),
            "safety_lcb_min": safety_lcb_min,
            **safety_guard_diag,
            **utility_refinement_diag,
            "selected_action_safety_flag": bool(np.asarray(runtime_safety_flags, dtype=bool)[action]) if 0 <= action < len(runtime_safety_flags) else False,
            "avoidable_selected_action_safety_flag": bool(
                (bool(np.asarray(runtime_safety_flags, dtype=bool)[action]) if 0 <= action < len(runtime_safety_flags) else True)
                and bool(safety_guard_diag.get("safe_action_available", False))
            ),
        },
    )


def _pair_delta_margin_matrix(
    predicted_base_cost: np.ndarray,
    pair_indices: np.ndarray,
    pair_atom_delta: np.ndarray,
    selected_atoms: list[int] | np.ndarray,
    valid_mask: np.ndarray,
    normalize_margins: bool = False,
    margin_scale: float | None = None,
    norm_min_scale: float = 100.0,
    norm_quantile: float = 0.75,
    predicted_atom_costs: np.ndarray | None = None,
    pair_delta_includes_local: bool = False,
) -> np.ndarray:
    """Build a valid-valid antisymmetric sparse pair-margin matrix.

    ``pair_atom_delta[:, p]`` predicts directed evidence contribution for
    ``pair_indices[p] = (a,b)``.  The planning tournament, however, is defined on
    a pairwise margin game where M[a,b] = -M[b,a] for the same queried support.
    Therefore every queried directed margin is mirrored immediately.  If both
    directions are queried and the neural pair scorer is not exactly
    antisymmetric, we use the antisymmetric projection
    0.5 * (M_hat[a,b] - M_hat[b,a]) rather than letting the later direction
    silently overwrite the earlier one.
    """
    J0_raw = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    J0 = _finite_cost_for_margin(J0_raw)
    K = J0.shape[0]
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.shape[0] < K:
        valid = np.pad(valid, (0, K - valid.shape[0]), constant_values=False)
    valid = valid[:K]
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    scale = 1.0
    if bool(normalize_margins):
        if margin_scale is not None and np.isfinite(float(margin_scale)) and float(margin_scale) > 0:
            scale = float(margin_scale)
        elif pair_arr.size:
            ok_pairs = pair_arr[(pair_arr[:, 0] >= 0) & (pair_arr[:, 0] < K) & (pair_arr[:, 1] >= 0) & (pair_arr[:, 1] < K)]
            if ok_pairs.size:
                ok_pairs = ok_pairs[valid[ok_pairs[:, 0]] & valid[ok_pairs[:, 1]]]
            scale = margin_normalization_scale(J0[ok_pairs[:, 1]] - J0[ok_pairs[:, 0]], min_scale=float(norm_min_scale), quantile=float(norm_quantile)) if ok_pairs.size else float(norm_min_scale)
        else:
            scale = float(norm_min_scale)
    delta = np.asarray(pair_atom_delta, dtype=np.float32)
    selected = np.asarray(selected_atoms, dtype=np.int64).reshape(-1)
    selected = selected[(selected >= 0) & (selected < delta.shape[0])] if delta.ndim == 2 else np.zeros((0,), dtype=np.int64)

    # V54 AR-BFAR: use the integrable selected-evidence cost as the tournament
    # anchor, and let the pair head contribute only a residual correction.  The
    # legacy path reconstructed the entire margin game from J0 plus a sparse pair
    # graph; missing edges therefore fell back to J0 and destroyed the much
    # stronger action-conditioned local interface.  With this anchor-relative
    # construction, zero residual is exactly the selected-local planner.
    atom_costs = None
    anchor_cost = J0
    if predicted_atom_costs is not None:
        candidate = np.asarray(predicted_atom_costs, dtype=np.float32)
        if candidate.ndim == 2 and candidate.shape[1] == K:
            atom_costs = candidate
            selected_anchor = selected[(selected >= 0) & (selected < candidate.shape[0])]
            if selected_anchor.size:
                anchor_cost = _finite_cost_for_margin(J0 + candidate[selected_anchor].sum(axis=0))
    M = (anchor_cost[None, :] - anchor_cost[:, None]) / max(scale, 1e-6)
    np.fill_diagonal(M, 0.0)

    directed: dict[tuple[int, int], float] = {}
    if pair_arr.size and delta.ndim == 2 and delta.shape[1] >= pair_arr.shape[0] and selected.size:
        support = delta[selected, : pair_arr.shape[0]].sum(axis=0)
        for pidx, (a_raw, b_raw) in enumerate(pair_arr.tolist()):
            a, b = int(a_raw), int(b_raw)
            if 0 <= a < K and 0 <= b < K and a != b:
                correction = float(support[pidx])
                if atom_costs is not None and pair_delta_includes_local:
                    selected_local = selected[(selected >= 0) & (selected < atom_costs.shape[0])]
                    if selected_local.size:
                        local_raw = float((atom_costs[selected_local, b] - atom_costs[selected_local, a]).sum())
                        correction -= local_raw / max(scale, 1e-6) if normalize_margins else local_raw
                directed[(a, b)] = (anchor_cost[b] - anchor_cost[a]) / max(scale, 1e-6) + correction

    done: set[tuple[int, int]] = set()
    for (a, b), value_ab in directed.items():
        if (a, b) in done or (b, a) in done:
            continue
        value_ba = directed.get((b, a), None)
        if value_ba is None:
            m_ab = float(value_ab)
        else:
            # Antisymmetric projection of two independently predicted directions.
            m_ab = 0.5 * (float(value_ab) - float(value_ba))
        M[a, b] = m_ab
        M[b, a] = -m_ab
        done.add((a, b)); done.add((b, a))

    M[~valid, :] = -1e9
    M[:, ~valid] = -1e9
    M[np.diag_indices(K)] = 0.0
    return M.astype(np.float32)


def _pair_sigma_matrix(
    pair_indices: np.ndarray,
    pair_atom_variance: np.ndarray | None,
    selected_atoms: list[int] | np.ndarray,
    K: int,
) -> np.ndarray | None:
    """Build a symmetric sigma matrix for conservative pair tournaments."""
    if pair_atom_variance is None:
        return None
    var = np.asarray(pair_atom_variance, dtype=np.float32)
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    selected = np.asarray(selected_atoms, dtype=np.int64).reshape(-1)
    selected = selected[(selected >= 0) & (selected < var.shape[0])] if var.ndim == 2 else np.zeros((0,), dtype=np.int64)
    if var.ndim != 2 or not pair_arr.size or not selected.size:
        return None
    sigma = np.zeros((K, K), dtype=np.float32)
    support = np.maximum(var[selected, : pair_arr.shape[0]], 0.0).sum(axis=0)
    for pidx, (a_raw, b_raw) in enumerate(pair_arr.tolist()):
        a, b = int(a_raw), int(b_raw)
        if 0 <= a < K and 0 <= b < K and a != b:
            s = float(np.sqrt(max(float(support[pidx]), 0.0)))
            sigma[a, b] = max(float(sigma[a, b]), s)
            sigma[b, a] = max(float(sigma[b, a]), s)
    return sigma




def _selected_local_anchor_cost(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray | None,
    selected_atoms: list[int] | np.ndarray,
) -> np.ndarray:
    anchor = _finite_cost_for_margin(np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1))
    if predicted_atom_costs is None:
        return anchor
    atom_costs = np.asarray(predicted_atom_costs, dtype=np.float32)
    selected = np.asarray(selected_atoms, dtype=np.int64).reshape(-1)
    if atom_costs.ndim != 2 or atom_costs.shape[1] != anchor.shape[0]:
        return anchor
    selected = selected[(selected >= 0) & (selected < atom_costs.shape[0])]
    if selected.size:
        anchor = _finite_cost_for_margin(anchor + atom_costs[selected].sum(axis=0))
    return anchor


def _direct_action_scores_from_cost(cost: np.ndarray, valid_mask: np.ndarray, scale: float) -> np.ndarray:
    cost = _finite_cost_for_margin(np.asarray(cost, dtype=np.float32).reshape(-1))
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.shape[0] < cost.shape[0]:
        valid = np.pad(valid, (0, cost.shape[0] - valid.shape[0]), constant_values=False)
    valid = valid[: cost.shape[0]]
    finite = valid & np.isfinite(cost)
    scores = np.full_like(cost, -1e9, dtype=np.float32)
    if bool(finite.any()):
        center = float(np.median(cost[finite]))
        scores[finite] = -((cost[finite] - center) / max(float(scale), 1e-6)).astype(np.float32)
    return scores


def _decisive_anchor_margin_scores(
    anchor_cost: np.ndarray,
    margin_matrix: np.ndarray,
    valid_mask: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, int]:
    """Convert only the selected-local anchor star into direct action scores.

    V64.3.7 DARM never traverses a global learned pair tournament.  It keeps the
    direct selected-local argmin as the immutable anchor and lets queried pair
    residuals refine only anchor-vs-challenger margins.  Missing edges inherit
    the selected-local margin, so a zero residual is exactly a no-op.
    """
    scores = _direct_action_scores_from_cost(anchor_cost, valid_mask, scale)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)[: scores.shape[0]]
    if not bool(valid.any()):
        return scores, -1
    anchor = int(np.argmax(np.where(valid, scores, -1e30)))
    M = np.asarray(margin_matrix, dtype=np.float32)
    if M.shape != (scores.shape[0], scores.shape[0]):
        raise ValueError(f"margin_matrix shape mismatch: {M.shape} vs {(scores.shape[0], scores.shape[0])}")
    refined = scores.copy()
    idx = np.flatnonzero(valid)
    refined[idx] = scores[anchor] - M[anchor, idx]
    refined[anchor] = scores[anchor]
    return refined.astype(np.float32), anchor


def _integrable_potential_cost(
    predicted_base_cost: np.ndarray,
    pair_indices: np.ndarray,
    pair_atom_delta: np.ndarray,
    selected_atoms: list[int] | np.ndarray,
    valid_mask: np.ndarray,
    *,
    predicted_atom_costs: np.ndarray | None,
    pair_delta_includes_local: bool,
    normalize_margins: bool,
    margin_scale: float,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Selected-local anchor plus Hodge-projected pair residual potential."""
    anchor = _selected_local_anchor_cost(predicted_base_cost, predicted_atom_costs, selected_atoms)
    pairs = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    delta = np.asarray(pair_atom_delta, dtype=np.float32)
    selected = np.asarray(selected_atoms, dtype=np.int64).reshape(-1)
    selected = selected[(selected >= 0) & (selected < delta.shape[0])] if delta.ndim == 2 else np.zeros((0,), dtype=np.int64)
    pair_count = min(int(pairs.shape[0]), int(delta.shape[1]) if delta.ndim == 2 else 0)
    pairs = pairs[:pair_count]
    if pair_count and selected.size:
        residual = delta[selected, :pair_count].sum(axis=0).astype(np.float32)
    else:
        residual = np.zeros((pair_count,), dtype=np.float32)
    if pair_delta_includes_local and predicted_atom_costs is not None and pair_count and selected.size:
        atom_costs = np.asarray(predicted_atom_costs, dtype=np.float32)
        if atom_costs.ndim == 2 and atom_costs.shape[1] == anchor.shape[0]:
            a = pairs[:, 0].clip(0, atom_costs.shape[1] - 1)
            b = pairs[:, 1].clip(0, atom_costs.shape[1] - 1)
            local = (atom_costs[selected][:, b] - atom_costs[selected][:, a]).sum(axis=0)
            residual = residual - (local / max(float(margin_scale), 1e-6) if normalize_margins else local)
    anchor_margin = np.zeros((pair_count,), dtype=np.float32)
    if pair_count:
        a = pairs[:, 0].clip(0, anchor.shape[0] - 1)
        b = pairs[:, 1].clip(0, anchor.shape[0] - 1)
        anchor_margin = anchor[b] - anchor[a]
        if normalize_margins:
            anchor_margin = anchor_margin / max(float(margin_scale), 1e-6)
    pcfg = ((cfg.get("runtime", {}) or {}).get("pair_potential_projection", {}) or {})
    potential, diag = project_pair_residual_to_action_potential_numpy(
        pairs,
        residual,
        valid_mask,
        pair_weights=None,
        anchor_margin=anchor_margin,
        ridge=float(pcfg.get("ridge", 0.02)),
        boundary_tau=float(pcfg.get("boundary_tau", 0.35)),
        boundary_gain=float(pcfg.get("boundary_gain", 2.0)),
        weight_floor=float(pcfg.get("weight_floor", 0.05)),
    )
    cost_scale = max(float(margin_scale), 1e-6) if normalize_margins else 1.0
    corrected = _finite_cost_for_margin(anchor + potential * cost_scale)
    diag.update({
        "pair_potential_active": 1.0,
        "pair_potential_cost_correction_abs_mean": float(np.mean(np.abs(potential * cost_scale))) if potential.size else 0.0,
        "pair_potential_residual_edge_abs_mean": float(np.mean(np.abs(residual))) if residual.size else 0.0,
    })
    return anchor, corrected, diag

def _evidence_action_potential_cost(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray | None,
    residual_action_potential: np.ndarray | None,
    selected_atoms: list[int] | np.ndarray,
    valid_mask: np.ndarray,
    *,
    residual_action_variance: np.ndarray | None,
    residual_set_atom_factors: np.ndarray | None = None,
    residual_set_action_factors: np.ndarray | None = None,
    set_residual_scale: float = 1.0,
    normalize_margins: bool,
    margin_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, float]]:
    """Selected-local anchor plus direct evidence-attributable action potential."""
    anchor = _selected_local_anchor_cost(predicted_base_cost, predicted_atom_costs, selected_atoms)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    K = anchor.shape[0]
    selected = np.asarray(selected_atoms, dtype=np.int64).reshape(-1)
    pot = np.asarray(residual_action_potential, dtype=np.float32) if residual_action_potential is not None else np.zeros((0, K), dtype=np.float32)
    selected = selected[(selected >= 0) & (selected < pot.shape[0])] if pot.ndim == 2 else np.zeros((0,), dtype=np.int64)
    action_potential = pot[selected].sum(axis=0).astype(np.float32) if selected.size else np.zeros((K,), dtype=np.float32)
    set_action_potential = np.zeros((K,), dtype=np.float32)
    atom_factors = np.asarray(residual_set_atom_factors, dtype=np.float32) if residual_set_atom_factors is not None else np.zeros((0, 0), dtype=np.float32)
    action_factors = np.asarray(residual_set_action_factors, dtype=np.float32) if residual_set_action_factors is not None else np.zeros((0, 0), dtype=np.float32)
    set_selected = selected[(selected >= 0) & (selected < atom_factors.shape[0])] if atom_factors.ndim == 2 else np.zeros((0,), dtype=np.int64)
    if (
        set_selected.size
        and atom_factors.ndim == 2
        and action_factors.ndim == 2
        and atom_factors.shape[1] > 0
        and action_factors.shape[0] >= K
        and action_factors.shape[1] == atom_factors.shape[1]
    ):
        rank = int(atom_factors.shape[1])
        pooled = atom_factors[set_selected].sum(axis=0) / np.sqrt(max(float(set_selected.size), 1.0))
        pooled = np.tanh(pooled).astype(np.float32)
        set_action_potential = (action_factors[:K] @ pooled / np.sqrt(max(float(rank), 1.0))).astype(np.float32)
        action_potential = action_potential + float(set_residual_scale) * set_action_potential
    finite_valid = valid[:K] & np.isfinite(anchor)
    if bool(finite_valid.any()):
        action_potential = action_potential - float(np.mean(action_potential[finite_valid]))
    action_potential[~finite_valid] = 0.0
    scale = max(float(margin_scale), 1e-6) if normalize_margins else 1.0
    corrected = _finite_cost_for_margin(anchor + action_potential * scale)
    sigma = selected_pair_sigma_from_action_variance(residual_action_variance, selected, valid_mask)
    diag = {
        "pair_potential_active": 1.0,
        "direct_evidence_action_potential_active": 1.0,
        "pair_potential_cost_correction_abs_mean": float(np.mean(np.abs(action_potential * scale))) if action_potential.size else 0.0,
        "pair_potential_residual_edge_abs_mean": 0.0,
        "residual_action_potential_abs_mean": float(np.mean(np.abs(action_potential))) if action_potential.size else 0.0,
        "residual_action_potential_selected_atom_count": float(len(selected)),
        "set_conditioned_residual_active": float(bool(set_selected.size and atom_factors.ndim == 2 and action_factors.ndim == 2)),
        "set_conditioned_residual_rank": float(atom_factors.shape[1] if atom_factors.ndim == 2 else 0),
        "set_conditioned_residual_abs_mean": float(np.mean(np.abs(set_action_potential))) if set_action_potential.size else 0.0,
        "set_conditioned_residual_scale": float(set_residual_scale),
    }
    return anchor, corrected, sigma, diag


def _decisive_frontier_value_star_residual_numpy(
    selected_atoms: list[int] | np.ndarray,
    valid_mask: np.ndarray,
    anchor_action: int,
    atom_factors: np.ndarray | None,
    action_signed_factors: np.ndarray | None,
    action_context_factors: np.ndarray | None,
    *,
    scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return exact-antisymmetric selected-evidence value residual on one anchor star.

    The residual is pair-specific because the symmetric pair-context gate depends
    jointly on anchor and challenger.  It therefore does not collapse to the V59
    global selected-set action potential.  Only already-selected B evidence is
    pooled and no proposal/selector score is changed.
    """
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    K = int(valid.shape[0])
    out = np.zeros((K,), dtype=np.float32)
    atom = None if atom_factors is None else np.asarray(atom_factors, dtype=np.float32)
    signed = None if action_signed_factors is None else np.asarray(action_signed_factors, dtype=np.float32)
    context = None if action_context_factors is None else np.asarray(action_context_factors, dtype=np.float32)
    if (
        atom is None or signed is None or context is None or atom.ndim != 2 or signed.ndim != 2
        or context.ndim != 2 or signed.shape != context.shape or signed.shape[0] < K
        or atom.shape[1] <= 0 or signed.shape[1] != atom.shape[1]
        or not (0 <= int(anchor_action) < K)
    ):
        return out, {
            "decisive_frontier_value_active": 0.0,
            "decisive_frontier_value_complete_star_coverage": 0.0,
            "decisive_frontier_value_residual_abs_mean": 0.0,
            "decisive_frontier_value_residual_rms": 0.0,
        }
    selected = np.asarray(selected_atoms, dtype=np.int64).reshape(-1)
    selected = selected[(selected >= 0) & (selected < atom.shape[0])]
    if selected.size == 0:
        return out, {
            "decisive_frontier_value_active": 0.0,
            "decisive_frontier_value_complete_star_coverage": 0.0,
            "decisive_frontier_value_residual_abs_mean": 0.0,
            "decisive_frontier_value_residual_rms": 0.0,
        }
    rank = int(atom.shape[1])
    # tanh is applied per atom before summation, preserving an exact additive
    # decomposition into selected-evidence contributions.  V64.3.14 additionally
    # keeps the root-sum-square energy of those *same* atom contributions as a
    # heteroscedastic calibration scale.  This is not a new evidence query and it
    # is not presented as an epistemic variance estimate; split conformal
    # calibration below learns how much over-estimation is compatible with this
    # auditable attribution scale.
    bounded_selected = np.tanh(atom[selected]).astype(np.float32)
    a = int(anchor_action)
    ctx_a = context[a][None, :]
    pair_sym = np.tanh(ctx_a + context[:K] + ctx_a * context[:K]).astype(np.float32)
    signed_diff = signed[:K] - signed[a][None, :]
    pair_vec = pair_sym * signed_diff

    # Keep the V64.3.13 residual arithmetic itself unchanged.  The attribution
    # decomposition is diagnostic/calibration side information only; it must not
    # perturb the learned value through a different floating-point reduction
    # order before OCFI is even enabled.
    pooled = bounded_selected.sum(axis=0) / np.sqrt(max(float(selected.size), 1.0))
    pooled = pooled.astype(np.float32)
    out = (
        (pooled[None, :] * pair_sym * signed_diff).sum(axis=1)
        / np.sqrt(max(float(rank), 1.0))
    ).astype(np.float32) * float(scale)

    denom = np.sqrt(max(float(selected.size * rank), 1.0))
    atom_contrib = np.einsum("nr,kr->nk", bounded_selected, pair_vec, optimize=True).astype(np.float32)
    atom_contrib = atom_contrib * (float(scale) / denom)
    attribution_scale = np.sqrt(np.sum(atom_contrib * atom_contrib, axis=0)).astype(np.float32)
    out[~valid[:K]] = 0.0
    attribution_scale[~valid[:K]] = 0.0
    out[a] = 0.0
    attribution_scale[a] = 0.0
    challengers = valid[:K].copy(); challengers[a] = False
    vals = out[challengers]
    attr_vals = attribution_scale[challengers]
    return out, {
        "decisive_frontier_value_active": 1.0,
        "decisive_frontier_value_complete_star_coverage": 1.0 if bool(challengers.any()) else 0.0,
        "decisive_frontier_value_selected_atom_count": float(selected.size),
        "decisive_frontier_value_rank": float(rank),
        "decisive_frontier_value_residual_abs_mean": float(np.mean(np.abs(vals))) if vals.size else 0.0,
        "decisive_frontier_value_residual_rms": float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0,
        "decisive_frontier_value_attribution_scale_mean": float(np.mean(attr_vals)) if attr_vals.size else 0.0,
        "decisive_frontier_value_attribution_scale_rms": float(np.sqrt(np.mean(attr_vals * attr_vals))) if attr_vals.size else 0.0,
        "decisive_frontier_value_scale": float(scale),
        # Private runtime-only vectors consumed by reliability/recovery heads.
        # They are stripped from scalar diagnostics automatically.  The atom
        # matrix is the exact signed selected-evidence decomposition whose
        # column sum equals the frozen EAF residual for each challenger.
        "_decisive_frontier_value_attribution_scale_star": attribution_scale,
        "_decisive_frontier_value_atom_contrib_star": np.asarray(atom_contrib, dtype=np.float32),
    }




_RAER_FEATURE_NAMES = [
    "raw_margin",
    "attribution_scale",
    "frontier_residual_rms",
    "frontier_residual_abs_mean",
    "frontier_attribution_scale_rms",
    "frontier_attribution_scale_mean",
    "evidence_certificate_fraction",
    "valid_action_count_norm",
    "margin_over_attribution",
    "attribution_over_frontier_rms",
    "raw_margin_z",
    "attribution_z",
    "raw_margin_rank",
    "attribution_rank",
    "margin_below_raw_top",
]


def _rank01(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Deterministic [0,1] ascending rank among masked entries (ties average-free)."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)[: x.shape[0]]
    out = np.zeros_like(x, dtype=np.float64)
    idx = np.flatnonzero(m)
    if idx.size <= 1:
        if idx.size == 1:
            out[idx[0]] = 1.0
        return out
    order = idx[np.argsort(x[idx], kind="mergesort")]
    out[order] = np.arange(order.size, dtype=np.float64) / float(order.size - 1)
    return out


def _decisive_frontier_raer_features(
    margin_star: np.ndarray,
    attribution_star: np.ndarray | None,
    valid_mask: np.ndarray,
    anchor_action: int,
    potential_diag: dict[str, Any],
    evidence_certificate_fraction: float | None,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Runtime-only per-challenger features for V64.3.16 EAF-RAER.

    Every feature is computed from the already-selected B evidence, frozen EAF
    frontier value/attribution, the unchanged evidence certificate, and the
    candidate validity mask.  No teacher/future label is consumed here.
    """
    margins = np.asarray(margin_star, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)[: margins.shape[0]]
    K = int(margins.shape[0])
    a = int(anchor_action)
    challengers = valid.copy()
    if 0 <= a < K:
        challengers[a] = False
    attr = np.zeros((K,), dtype=np.float64)
    if attribution_star is not None:
        tmp = np.asarray(attribution_star, dtype=np.float64).reshape(-1)
        attr[: min(K, tmp.size)] = tmp[: min(K, tmp.size)]
    attrs = attr[challengers]
    vals = margins[challengers]
    margin_mean = float(np.mean(vals)) if vals.size else 0.0
    margin_std = max(float(np.std(vals)) if vals.size else 0.0, 1.0e-6)
    attr_mean_local = float(np.mean(attrs)) if attrs.size else 0.0
    attr_std = max(float(np.std(attrs)) if attrs.size else 0.0, 1.0e-6)
    top_margin = float(np.max(vals)) if vals.size else 0.0
    frontier_residual_rms = float(potential_diag.get("decisive_frontier_value_residual_rms", 0.0))
    frontier_residual_abs_mean = float(potential_diag.get("decisive_frontier_value_residual_abs_mean", 0.0))
    frontier_attr_rms = float(potential_diag.get("decisive_frontier_value_attribution_scale_rms", 0.0))
    frontier_attr_mean = float(potential_diag.get("decisive_frontier_value_attribution_scale_mean", 0.0))
    cert = float(evidence_certificate_fraction) if evidence_certificate_fraction is not None and np.isfinite(float(evidence_certificate_fraction)) else 0.0
    raer_cfg = (((cfg.get("runtime", {}) or {}).get("decisive_frontier_value", {}) or {}).get("reliability_aware_extremal_reranking", {}) or {})
    attr_eps = max(float(raer_cfg.get("ratio_floor", 1.0e-3)), 1.0e-9)
    valid_norm = float(valid.sum()) / max(float(raer_cfg.get("valid_action_normalizer", 32.0)), 1.0)
    margin_rank = _rank01(margins, challengers)
    attr_rank = _rank01(attr, challengers)
    mat = np.zeros((K, len(_RAER_FEATURE_NAMES)), dtype=np.float64)
    for b in np.flatnonzero(challengers).tolist():
        m = float(margins[b])
        at = float(attr[b])
        vals_b = [
            m,
            at,
            frontier_residual_rms,
            frontier_residual_abs_mean,
            frontier_attr_rms,
            frontier_attr_mean,
            cert,
            valid_norm,
            m / max(at, attr_eps),
            at / max(frontier_attr_rms, attr_eps),
            (m - margin_mean) / margin_std,
            (at - attr_mean_local) / attr_std,
            float(margin_rank[b]),
            float(attr_rank[b]),
            top_margin - m,
        ]
        mat[b] = np.asarray(vals_b, dtype=np.float64)
    return mat, list(_RAER_FEATURE_NAMES)


def _apply_decisive_frontier_raer(
    raw_action: int,
    anchor_action: int,
    margin_matrix: np.ndarray,
    attribution_star: np.ndarray | None,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    potential_diag: dict[str, Any],
    evidence_certificate_fraction: float | None,
    cfg: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Reliability-aware extremal re-ranking before the one-sided flip guard."""
    runtime_cfg = cfg.get("runtime", {}) or {}
    frontier_cfg = runtime_cfg.get("decisive_frontier_value", {}) or {}
    raer_cfg = frontier_cfg.get("reliability_aware_extremal_reranking", {}) or {}
    enabled = bool(raer_cfg.get("enabled", False))
    instrument = bool(raer_cfg.get("instrument_features", True))
    frontier_active = bool(float(potential_diag.get("decisive_frontier_value_active", 0.0)) >= 0.5)
    M = np.asarray(margin_matrix, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    a = int(anchor_action)
    margin_star = M[:, a].copy() if 0 <= a < M.shape[1] else np.zeros((len(valid),), dtype=np.float64)
    feat, names = _decisive_frontier_raer_features(
        margin_star, attribution_star, valid, a, potential_diag, evidence_certificate_fraction, cfg
    )
    probs = np.ones((len(valid),), dtype=np.float64)
    utility = np.maximum(margin_star, 0.0)
    selected = int(raw_action)
    applies = bool(enabled and frontier_active and 0 <= a < len(valid))
    if applies:
        cfg_names = list(raer_cfg.get("feature_names", []))
        mean = np.asarray(raer_cfg.get("feature_mean", []), dtype=np.float64).reshape(-1)
        std = np.asarray(raer_cfg.get("feature_std", []), dtype=np.float64).reshape(-1)
        weights = np.asarray(raer_cfg.get("weights", []), dtype=np.float64).reshape(-1)
        if cfg_names != names or len(names) != len(mean) or len(names) != len(std) or len(names) != len(weights):
            raise ValueError(
                "EAF-RAER enabled but feature schema/mean/std/weights are inconsistent; "
                "fit the V64.3.16 train-only readout before enabling re-ranking"
            )
        z = (feat - mean[None, :]) / np.maximum(std[None, :], 1.0e-6)
        logits = z @ weights + float(raer_cfg.get("bias", 0.0))
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        utility = probs * np.maximum(margin_star, 0.0)
        eligible = valid.copy()
        eligible[a] = False
        # Preserve the pre-existing structural safety contract: if any unflagged
        # valid action exists, RAER cannot resurrect a safety-flagged challenger.
        flags = np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)[: len(valid)]
        # Include the anchor in the safety-existence test.  If the anchor is the
        # only unflagged action, RAER must abstain rather than resurrect a flagged
        # challenger after the legacy safety guard has already protected it.
        safe_exists = bool(np.any(valid & ~flags))
        if safe_exists:
            eligible &= ~flags
        min_prob = float(raer_cfg.get("min_probability", 0.5))
        if bool(raer_cfg.get("require_positive_raw_margin", True)):
            eligible &= margin_star > 0.0
        eligible &= probs >= min_prob
        eligible &= np.isfinite(utility)
        if bool(np.any(eligible)):
            cand = np.flatnonzero(eligible)
            selected = int(cand[int(np.argmax(utility[cand]))])
        else:
            selected = a
    selected_prob = float(probs[selected]) if 0 <= selected < len(probs) else 1.0
    raw_prob = float(probs[int(raw_action)]) if 0 <= int(raw_action) < len(probs) else 1.0
    diag: dict[str, Any] = {
        "decisive_frontier_raer_enabled": float(enabled),
        "decisive_frontier_raer_instrument_features": float(instrument),
        "decisive_frontier_raer_active": float(applies),
        "decisive_frontier_raer_raw_top_action": float(raw_action),
        "decisive_frontier_raer_selected_action": float(selected),
        "decisive_frontier_raer_proposal_changed": float(int(selected) != int(raw_action)),
        "decisive_frontier_raer_selected_probability": selected_prob,
        "decisive_frontier_raer_raw_top_probability": raw_prob,
        "decisive_frontier_raer_selected_utility": float(utility[selected]) if 0 <= selected < len(utility) else 0.0,
        "decisive_frontier_raer_min_probability": float(raer_cfg.get("min_probability", 0.5)),
        # Private arrays are exported only when explicitly requested by the
        # train-split edge instrumentation path; scalar metric aggregation ignores them.
        "_decisive_frontier_raer_anchor_action": int(a),
        "_decisive_frontier_raer_raw_margin_star": np.asarray(margin_star, dtype=np.float32),
        "_decisive_frontier_raer_attribution_scale_star": np.zeros_like(margin_star, dtype=np.float32)
            if attribution_star is None else np.asarray(attribution_star, dtype=np.float32).reshape(-1)[: len(margin_star)],
        "_decisive_frontier_raer_raw_top_action": int(raw_action),
        "_decisive_frontier_raer_feature_matrix": np.asarray(feat, dtype=np.float32),
        "_decisive_frontier_raer_feature_names": names,
        "_decisive_frontier_raer_probability_star": np.asarray(probs, dtype=np.float32),
        "_decisive_frontier_raer_utility_star": np.asarray(utility, dtype=np.float32),
    }
    return selected, diag



_DALER_FEATURE_NAMES = [
    "raw_margin",
    "attribution_scale",
    "frontier_residual_rms",
    "frontier_residual_abs_mean",
    "frontier_attribution_scale_rms",
    "frontier_attribution_scale_mean",
    "evidence_certificate_fraction",
    "valid_action_count_norm",
    "margin_over_attribution",
    "attribution_over_frontier_rms",
    "raw_margin_z",
    "attribution_z",
    "raw_margin_rank",
    "attribution_rank",
    "margin_below_frontier_max",
    "is_legacy_selected",
    "margin_minus_legacy_selected",
    "attribution_minus_legacy_selected",
    "eaf_score_gain_vs_anchor",
    "eaf_score_minus_legacy_selected",
    "utility_cost_minus_legacy_selected",
    "guard_margin_excess",
    "eaf_score_rank",
    "utility_cost_rank",
    "executable_candidate_fraction",
]


def _decisive_frontier_daler_executable_mask(
    margin_star: np.ndarray,
    scores: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    anchor_action: int,
    evidence_certificate_fraction: float | None,
    utility_refinement_diag: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return DALER's exact frozen-deployment challenger masks.

    ``guard_mask`` implements the existing one-sided anchor guard prerequisites;
    ``utility_mask`` is the exact legacy utility-equivalence set.  ``executable``
    is their conjunction.  No learned score and no teacher label participates.
    """
    margins = np.asarray(margin_star, dtype=np.float64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)[: margins.shape[0]]
    flags = np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)
    if flags.shape[0] < len(valid):
        flags = np.pad(flags, (0, len(valid) - flags.shape[0]), constant_values=False)
    flags = flags[: len(valid)]
    a = int(anchor_action)
    runtime_cfg = cfg.get("runtime", {}) or {}
    frontier_cfg = runtime_cfg.get("decisive_frontier_value", {}) or {}
    daler_cfg = frontier_cfg.get("deployment_aligned_listwise_extremal_reliability", {}) or {}
    guard_cfg = runtime_cfg.get("pair_action_anchor_guard", {}) or {}
    dual_cfg = runtime_cfg.get("dual_certificate", {}) or {}

    guard_mask = valid.copy()
    if 0 <= a < len(guard_mask):
        guard_mask[a] = False
    safe_exists = bool(np.any(valid & ~flags))
    if safe_exists:
        guard_mask &= ~flags
    elif bool(daler_cfg.get("require_safe_available_for_learned_intervention", True)):
        # The planner has a separate frozen continuous structural-risk guard for
        # all-flagged candidate banks.  DALER has no access to that guard's risk
        # pool here, so learning a pre-structural re-ranking would create a hidden
        # train/deployment mismatch.  Abstain and leave these scenes entirely to
        # the existing structural guard instead.
        guard_mask &= False

    flip_margin = float(guard_cfg.get("flip_margin", runtime_cfg.get("pair_residual_trust", {}).get("flip_margin", 0.05)))
    score_margin = float(guard_cfg.get("score_margin", 0.0))
    guard_mask &= np.isfinite(margins) & (margins >= flip_margin)
    if 0 <= a < scores_arr.size and np.isfinite(scores_arr[a]):
        score_gain = scores_arr[: len(valid)] - float(scores_arr[a])
        guard_mask &= np.isfinite(score_gain) & (score_gain >= score_margin)
    else:
        guard_mask &= False

    require_evidence = bool(
        dual_cfg.get("enabled", False)
        and dual_cfg.get("require_evidence_certificate_before_residual_flip", False)
    )
    min_evidence = float(dual_cfg.get("min_evidence_certificate_fraction_for_residual_flip", 1.0))
    cert = float(evidence_certificate_fraction) if evidence_certificate_fraction is not None else float("nan")
    evidence_pass = bool(
        (not require_evidence)
        or (np.isfinite(cert) and cert + 1.0e-9 >= min_evidence)
    )
    if not evidence_pass:
        guard_mask &= False

    utility_mask = np.ones_like(guard_mask, dtype=bool)
    require_utility = bool(daler_cfg.get("require_utility_equivalence", True))
    if require_utility:
        raw = None if utility_refinement_diag is None else utility_refinement_diag.get("_utility_refinement_eligible_mask", None)
        if raw is None:
            utility_mask = np.zeros_like(guard_mask, dtype=bool)
        else:
            tmp = np.asarray(raw, dtype=bool).reshape(-1)
            utility_mask = np.zeros_like(guard_mask, dtype=bool)
            utility_mask[: min(len(utility_mask), len(tmp))] = tmp[: min(len(utility_mask), len(tmp))]
    if 0 <= a < len(utility_mask):
        utility_mask[a] = False
    executable = guard_mask & utility_mask
    return executable, guard_mask, utility_mask


def _decisive_frontier_daler_features(
    margin_star: np.ndarray,
    attribution_star: np.ndarray | None,
    scores: np.ndarray,
    valid_mask: np.ndarray,
    anchor_action: int,
    legacy_action: int,
    potential_diag: dict[str, Any],
    evidence_certificate_fraction: float | None,
    utility_refinement_diag: dict[str, Any] | None,
    executable_mask: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Runtime-only V64.3.17 deployment-aligned listwise reliability features."""
    margins = np.asarray(margin_star, dtype=np.float64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)[: margins.shape[0]]
    K = int(margins.shape[0]); a = int(anchor_action); legacy = int(legacy_action)
    challengers = valid.copy()
    if 0 <= a < K:
        challengers[a] = False
    attr = np.zeros((K,), dtype=np.float64)
    if attribution_star is not None:
        tmp = np.asarray(attribution_star, dtype=np.float64).reshape(-1)
        attr[: min(K, tmp.size)] = tmp[: min(K, tmp.size)]
    vals = margins[challengers]; attrs = attr[challengers]
    margin_mean = float(np.mean(vals)) if vals.size else 0.0
    margin_std = max(float(np.std(vals)) if vals.size else 0.0, 1.0e-6)
    attr_mean = float(np.mean(attrs)) if attrs.size else 0.0
    attr_std = max(float(np.std(attrs)) if attrs.size else 0.0, 1.0e-6)
    frontier_max = float(np.max(vals)) if vals.size else 0.0
    frontier_residual_rms = float(potential_diag.get("decisive_frontier_value_residual_rms", 0.0))
    frontier_residual_abs_mean = float(potential_diag.get("decisive_frontier_value_residual_abs_mean", 0.0))
    frontier_attr_rms = float(potential_diag.get("decisive_frontier_value_attribution_scale_rms", 0.0))
    frontier_attr_mean = float(potential_diag.get("decisive_frontier_value_attribution_scale_mean", 0.0))
    cert = float(evidence_certificate_fraction) if evidence_certificate_fraction is not None and np.isfinite(float(evidence_certificate_fraction)) else 0.0
    daler_cfg = (((cfg.get("runtime", {}) or {}).get("decisive_frontier_value", {}) or {}).get("deployment_aligned_listwise_extremal_reliability", {}) or {})
    attr_eps = max(float(daler_cfg.get("ratio_floor", 1.0e-3)), 1.0e-9)
    valid_norm = float(valid.sum()) / max(float(daler_cfg.get("valid_action_normalizer", 32.0)), 1.0)
    margin_rank = _rank01(margins, challengers)
    attr_rank = _rank01(attr, challengers)
    score_rank = _rank01(scores_arr[:K], challengers)

    util = np.full((K,), np.nan, dtype=np.float64)
    if utility_refinement_diag is not None:
        raw_u = utility_refinement_diag.get("_utility_refinement_cost", None)
        if raw_u is not None:
            tmp = np.asarray(raw_u, dtype=np.float64).reshape(-1)
            util[: min(K, tmp.size)] = tmp[: min(K, tmp.size)]
    util_rank_mask = challengers & np.isfinite(util)
    utility_rank = _rank01(util, util_rank_mask)

    legacy_margin = float(margins[legacy]) if 0 <= legacy < K and legacy != a and np.isfinite(margins[legacy]) else 0.0
    legacy_attr = float(attr[legacy]) if 0 <= legacy < K and legacy != a and np.isfinite(attr[legacy]) else 0.0
    legacy_score = float(scores_arr[legacy]) if 0 <= legacy < scores_arr.size and np.isfinite(scores_arr[legacy]) else (float(scores_arr[a]) if 0 <= a < scores_arr.size and np.isfinite(scores_arr[a]) else 0.0)
    legacy_util = float(util[legacy]) if 0 <= legacy < K and np.isfinite(util[legacy]) else 0.0
    anchor_score = float(scores_arr[a]) if 0 <= a < scores_arr.size and np.isfinite(scores_arr[a]) else 0.0
    flip_margin = float(((cfg.get("runtime", {}) or {}).get("pair_action_anchor_guard", {}) or {}).get("flip_margin", ((cfg.get("runtime", {}) or {}).get("pair_residual_trust", {}) or {}).get("flip_margin", 0.05)))
    exec_fraction = float(np.asarray(executable_mask, dtype=bool).sum()) / max(float(challengers.sum()), 1.0)

    mat = np.zeros((K, len(_DALER_FEATURE_NAMES)), dtype=np.float64)
    for b in np.flatnonzero(challengers).tolist():
        m = float(margins[b]); at = float(attr[b]); sc = float(scores_arr[b]) if b < scores_arr.size and np.isfinite(scores_arr[b]) else anchor_score
        udelta = float(util[b] - legacy_util) if np.isfinite(util[b]) else 0.0
        vals_b = [
            m, at, frontier_residual_rms, frontier_residual_abs_mean,
            frontier_attr_rms, frontier_attr_mean, cert, valid_norm,
            m / max(at, attr_eps), at / max(frontier_attr_rms, attr_eps),
            (m - margin_mean) / margin_std, (at - attr_mean) / attr_std,
            float(margin_rank[b]), float(attr_rank[b]), frontier_max - m,
            float(b == legacy), m - legacy_margin, at - legacy_attr,
            sc - anchor_score, sc - legacy_score, udelta, m - flip_margin,
            float(score_rank[b]), float(utility_rank[b]), exec_fraction,
        ]
        mat[b] = np.asarray(vals_b, dtype=np.float64)
    return mat, list(_DALER_FEATURE_NAMES)


def _apply_decisive_frontier_daler(
    legacy_action: int,
    anchor_action: int,
    margin_matrix: np.ndarray,
    attribution_star: np.ndarray | None,
    scores: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    potential_diag: dict[str, Any],
    evidence_certificate_fraction: float | None,
    utility_refinement_diag: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Anchor-augmented listwise extremal reliability on executable challengers."""
    runtime_cfg = cfg.get("runtime", {}) or {}
    frontier_cfg = runtime_cfg.get("decisive_frontier_value", {}) or {}
    daler_cfg = frontier_cfg.get("deployment_aligned_listwise_extremal_reliability", {}) or {}
    enabled = bool(daler_cfg.get("enabled", False))
    instrument = bool(daler_cfg.get("instrument_features", True))
    frontier_active = bool(float(potential_diag.get("decisive_frontier_value_active", 0.0)) >= 0.5)
    M = np.asarray(margin_matrix, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    a = int(anchor_action); legacy = int(legacy_action)
    margin_star = M[:, a].copy() if 0 <= a < M.shape[1] else np.zeros((len(valid),), dtype=np.float64)
    executable, guard_mask, utility_mask = _decisive_frontier_daler_executable_mask(
        margin_star, scores, valid, runtime_safety_flags, a,
        evidence_certificate_fraction, utility_refinement_diag, cfg,
    )
    feat, names = _decisive_frontier_daler_features(
        margin_star, attribution_star, scores, valid, a, legacy, potential_diag,
        evidence_certificate_fraction, utility_refinement_diag, executable, cfg,
    )
    logits = np.zeros((len(valid),), dtype=np.float64)
    probs = np.full((len(valid),), 0.5, dtype=np.float64)
    selected = legacy
    applies = bool(enabled and frontier_active and 0 <= a < len(valid))
    anchor_logit = float(daler_cfg.get("anchor_logit", 0.0))
    if applies:
        cfg_names = list(daler_cfg.get("feature_names", []))
        mean = np.asarray(daler_cfg.get("feature_mean", []), dtype=np.float64).reshape(-1)
        std = np.asarray(daler_cfg.get("feature_std", []), dtype=np.float64).reshape(-1)
        weights = np.asarray(daler_cfg.get("weights", []), dtype=np.float64).reshape(-1)
        if cfg_names != names or len(names) != len(mean) or len(names) != len(std) or len(names) != len(weights):
            raise ValueError(
                "EAF-DALER enabled but feature schema/mean/std/weights are inconsistent; "
                "fit the V64.3.17 train-only listwise readout before enabling selection"
            )
        z = (feat - mean[None, :]) / np.maximum(std[None, :], 1.0e-6)
        logits = z @ weights + float(daler_cfg.get("bias", 0.0))
        logits = np.clip(logits, -40.0, 40.0)
        probs = 1.0 / (1.0 + np.exp(-(logits - anchor_logit)))
        cand = np.flatnonzero(executable & np.isfinite(logits)).astype(np.int64)
        if cand.size:
            util_cost = np.full((len(valid),), np.inf, dtype=np.float64)
            if utility_refinement_diag is not None and utility_refinement_diag.get("_utility_refinement_cost", None) is not None:
                tmp = np.asarray(utility_refinement_diag["_utility_refinement_cost"], dtype=np.float64).reshape(-1)
                util_cost[: min(len(valid), len(tmp))] = tmp[: min(len(valid), len(tmp))]
            # Stable tie-break: reliability logit, preserve legacy if tied, stronger
            # frozen EAF margin, lower deployment utility cost, then index.
            best = sorted(
                cand.tolist(),
                key=lambda b: (
                    -float(logits[b]),
                    -int(b == legacy),
                    -float(margin_star[b]),
                    float(util_cost[b]) if np.isfinite(util_cost[b]) else float("inf"),
                    int(b),
                ),
            )[0]
            selected = int(best) if float(logits[best]) > anchor_logit + 1.0e-12 else a
        else:
            selected = a
    selected_logit = float(logits[selected]) if 0 <= selected < len(logits) and selected != a else anchor_logit
    selected_prob = float(probs[selected]) if 0 <= selected < len(probs) and selected != a else 0.5
    diag: dict[str, Any] = {
        "decisive_frontier_daler_enabled": float(enabled),
        "decisive_frontier_daler_instrument_features": float(instrument),
        "decisive_frontier_daler_active": float(applies),
        "decisive_frontier_daler_legacy_selected_action": float(legacy),
        "decisive_frontier_daler_selected_action": float(selected),
        "decisive_frontier_daler_proposal_changed": float(int(selected) != int(legacy)),
        "decisive_frontier_daler_anchor_fallback": float(int(selected) == int(a)),
        "decisive_frontier_daler_selected_logit": selected_logit,
        "decisive_frontier_daler_selected_probability_vs_anchor": selected_prob,
        "decisive_frontier_daler_anchor_logit": anchor_logit,
        "decisive_frontier_daler_executable_candidate_count": float(np.asarray(executable, dtype=bool).sum()),
        "_decisive_frontier_daler_anchor_action": int(a),
        "_decisive_frontier_daler_legacy_selected_action": int(legacy),
        "_decisive_frontier_daler_raw_margin_star": np.asarray(margin_star, dtype=np.float32),
        "_decisive_frontier_daler_attribution_scale_star": np.zeros_like(margin_star, dtype=np.float32)
            if attribution_star is None else np.asarray(attribution_star, dtype=np.float32).reshape(-1)[: len(margin_star)],
        "_decisive_frontier_daler_feature_matrix": np.asarray(feat, dtype=np.float32),
        "_decisive_frontier_daler_feature_names": names,
        "_decisive_frontier_daler_logit_star": np.asarray(logits, dtype=np.float32),
        "_decisive_frontier_daler_probability_star": np.asarray(probs, dtype=np.float32),
        "_decisive_frontier_daler_executable_mask": np.asarray(executable, dtype=bool),
        "_decisive_frontier_daler_guard_mask": np.asarray(guard_mask, dtype=bool),
        "_decisive_frontier_daler_utility_equivalence_mask": np.asarray(utility_mask, dtype=bool),
    }
    return selected, diag



# V64.3.18 EAF-DACER -------------------------------------------------------
# The V64.3.17 DALER screen showed that treating the upstream utility-refinement
# pool as a hard deployment admissibility set collapses almost every scene to a
# singleton.  DACER therefore learns over the *actual frozen guard-admissible*
# frontier.  The legacy utility pool is retained only as an auditable diagnostic / exact-tie-break
# prior; its membership bit is not a learned feature and it is not a safety or execution constraint.
_DACER_PROFILE_FEATURE_NAMES = [
    "selected_atom_count_norm",
    "atom_contrib_l1",
    "atom_contrib_positive_mass_fraction",
    "atom_contrib_top1_abs_fraction",
    "atom_contrib_effective_support_norm",
    "delta_atom_contrib_l1",
    "delta_atom_contrib_positive_mass_fraction",
    "delta_atom_contrib_top1_abs_fraction",
    "delta_atom_contrib_effective_support_norm",
    "atom_top1_signed_norm",
    "atom_top2_signed_norm",
    "atom_top3_signed_norm",
    "atom_top4_signed_norm",
    "delta_atom_top1_signed_norm",
    "delta_atom_top2_signed_norm",
    "delta_atom_top3_signed_norm",
    "delta_atom_top4_signed_norm",
]
_DACER_FEATURE_NAMES = list(_DALER_FEATURE_NAMES) + list(_DACER_PROFILE_FEATURE_NAMES)


def _decisive_frontier_guard_admissible_mask(
    margin_star: np.ndarray,
    scores: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    anchor_action: int,
    evidence_certificate_fraction: float | None,
    cfg: dict[str, Any],
    *,
    require_safe_available_for_learned_intervention: bool = True,
) -> np.ndarray:
    """Frozen one-sided deployment prerequisites, before any learned score.

    This is deliberately narrower than "all valid" but broader than the legacy
    utility-refinement candidate pool.  Every action in this mask can pass the
    unchanged V64.3.13 one-sided/evidence guard under the V64.3.18 raw contract;
    utility refinement is an upstream choice heuristic, not an execution guard.
    """
    margins = np.asarray(margin_star, dtype=np.float64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)[: margins.shape[0]]
    flags = np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)
    if flags.shape[0] < len(valid):
        flags = np.pad(flags, (0, len(valid) - flags.shape[0]), constant_values=False)
    flags = flags[: len(valid)]
    a = int(anchor_action)
    runtime_cfg = cfg.get("runtime", {}) or {}
    guard_cfg = runtime_cfg.get("pair_action_anchor_guard", {}) or {}
    dual_cfg = runtime_cfg.get("dual_certificate", {}) or {}

    mask = valid.copy()
    if 0 <= a < len(mask):
        mask[a] = False
    safe_exists = bool(np.any(valid & ~flags))
    if safe_exists:
        mask &= ~flags
    elif bool(require_safe_available_for_learned_intervention):
        mask &= False

    flip_margin = float(guard_cfg.get("flip_margin", runtime_cfg.get("pair_residual_trust", {}).get("flip_margin", 0.05)))
    score_margin = float(guard_cfg.get("score_margin", 0.0))
    mask &= np.isfinite(margins) & (margins >= flip_margin)
    if scores_arr.size >= len(valid) and 0 <= a < scores_arr.size and np.isfinite(scores_arr[a]):
        score_gain = scores_arr[: len(valid)] - float(scores_arr[a])
        mask &= np.isfinite(score_gain) & (score_gain >= score_margin)
    else:
        # Fail closed on malformed score vectors rather than accidentally making
        # a partially indexed challenger deployable.
        mask &= False

    require_evidence = bool(
        dual_cfg.get("enabled", False)
        and dual_cfg.get("require_evidence_certificate_before_residual_flip", False)
    )
    min_evidence = float(dual_cfg.get("min_evidence_certificate_fraction_for_residual_flip", 1.0))
    cert = float(evidence_certificate_fraction) if evidence_certificate_fraction is not None else float("nan")
    evidence_pass = bool(
        (not require_evidence)
        or (np.isfinite(cert) and cert + 1.0e-9 >= min_evidence)
    )
    if not evidence_pass:
        mask &= False
    return np.asarray(mask, dtype=bool)


def _signed_atom_profile(vec: np.ndarray, *, top_k: int = 4) -> list[float]:
    """Permutation-invariant signed attribution profile for one challenger."""
    x = np.asarray(vec, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [0.0, 0.0, 0.0, 0.0] + [0.0] * int(top_k)
    ax = np.abs(x)
    l1 = float(ax.sum())
    sq = float(np.dot(x, x))
    denom = max(l1, 1.0e-9)
    pos_fraction = float(np.maximum(x, 0.0).sum() / denom)
    top1_fraction = float(ax.max() / denom) if ax.size else 0.0
    effective_support = float((l1 * l1) / max(sq, 1.0e-12) / max(float(x.size), 1.0)) if l1 > 0.0 else 0.0
    order = np.argsort(-ax, kind="mergesort")[: int(top_k)]
    top = [float(x[i] / denom) for i in order.tolist()]
    top.extend([0.0] * (int(top_k) - len(top)))
    return [l1, pos_fraction, top1_fraction, effective_support, *top]


_ICER_ATTRIBUTION_SPECTRUM_BUDGET = 16


def _icer_attribution_resolved_feature_names(budget: int) -> list[str]:
    """Return the fixed signed-spectrum schema for one retained-interface ceiling.

    Historical V24--V29 arms use B<=16 and therefore retain the exact legacy
    32-dimensional schema.  V64.3.30 is a capacity-ceiling diagnostic whose
    post-selector retained interface may contain all 24 already-queried atoms;
    its instrumentation must therefore be able to represent 24 candidate and
    24 candidate-minus-incumbent contributions without truncation.
    """
    b = int(budget)
    if b <= 0:
        raise ValueError(f"attribution spectrum budget must be positive, got {budget}")
    return (
        [f"candidate_atom_signed_spectrum_{i:02d}" for i in range(b)]
        + [f"delta_atom_signed_spectrum_{i:02d}" for i in range(b)]
    )


_ICER_ATTRIBUTION_RESOLVED_FEATURE_NAMES = _icer_attribution_resolved_feature_names(
    _ICER_ATTRIBUTION_SPECTRUM_BUDGET
)


def _icer_runtime_attribution_spectrum_budget(cfg: dict[str, Any]) -> int:
    """Resolve the instrumentation width without changing historical B16 semantics.

    The global ``evidence.budget`` remains the frozen upstream AOCC budget in
    FBIC.  Only ``selector.full_bank_capacity_probe.interface_budget`` describes
    the post-selector retained-interface ceiling.  Reading that field here is an
    instrumentation compatibility fix, not a new selector or decision rule.
    """
    budget = int(_ICER_ATTRIBUTION_SPECTRUM_BUDGET)
    selector_cfg = (cfg.get("selector", {}) or {}) if isinstance(cfg, dict) else {}
    probe = selector_cfg.get("full_bank_capacity_probe", {}) or {}
    if bool(probe.get("enabled", False)):
        try:
            interface_budget = int(round(float(probe.get("interface_budget", budget))))
        except Exception as exc:
            raise ValueError("invalid FBIC retained-interface budget") from exc
        budget = max(budget, interface_budget)
    return int(budget)


def _signed_attribution_spectrum(vec: np.ndarray, *, budget: int = _ICER_ATTRIBUTION_SPECTRUM_BUDGET) -> np.ndarray:
    """Full fixed-budget signed attribution spectrum, normalized by L1 mass.

    Unlike the historical top-4 summary, this retains every selected-evidence
    contribution up to the planner-interface budget.  Sorting by absolute
    contribution makes the representation permutation-invariant while preserving
    the sign of each contribution.  Magnitude remains available through the
    existing audited ICER aggregate attribution features.
    """
    x = np.asarray(vec, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    out = np.zeros((int(budget),), dtype=np.float64)
    if x.size == 0:
        return out
    denom = float(np.abs(x).sum())
    if denom <= 1.0e-12:
        return out
    order = np.argsort(-np.abs(x), kind="mergesort")[: int(budget)]
    vals = x[order] / denom
    out[: len(vals)] = vals
    return out


def _icer_attribution_resolved_feature_matrix(
    atom_contrib_star: np.ndarray | None,
    valid_mask: np.ndarray,
    anchor_action: int,
    legacy_action: int,
    *,
    budget: int = _ICER_ATTRIBUTION_SPECTRUM_BUDGET,
) -> tuple[np.ndarray, list[str]]:
    """Candidate and candidate-minus-incumbent full signed evidence spectra.

    ``budget`` is the retained-interface instrumentation ceiling.  The default
    remains 16 for strict backward compatibility with V24--V29.
    """
    spectrum_budget = int(budget)
    names = _icer_attribution_resolved_feature_names(spectrum_budget)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    K = len(valid); a = int(anchor_action); legacy = int(legacy_action)
    out = np.zeros((K, len(names)), dtype=np.float64)
    contrib = np.zeros((0, K), dtype=np.float64)
    if atom_contrib_star is not None:
        raw = np.asarray(atom_contrib_star, dtype=np.float64)
        if raw.ndim == 2 and raw.shape[1] >= K:
            contrib = raw[:, :K]
    if contrib.shape[0] > spectrum_budget:
        raise ValueError(
            f"selected evidence count {contrib.shape[0]} exceeds attribution spectrum budget "
            f"{spectrum_budget}"
        )
    legacy_vec = (
        contrib[:, legacy]
        if contrib.size and 0 <= legacy < K and legacy != a
        else np.zeros((contrib.shape[0],), dtype=np.float64)
    )
    for b in np.flatnonzero(valid).tolist():
        if b == a:
            continue
        cand = contrib[:, b] if contrib.size else np.zeros((0,), dtype=np.float64)
        delta = cand - legacy_vec if contrib.size else np.zeros((0,), dtype=np.float64)
        out[b, :spectrum_budget] = _signed_attribution_spectrum(cand, budget=spectrum_budget)
        out[b, spectrum_budget:] = _signed_attribution_spectrum(delta, budget=spectrum_budget)
    return out, names




_ICER_SEMANTIC_FAMILY_IDS = (1, 2, 3, 4, 5)
_ICER_SEMANTIC_FAMILY_NAMES = {
    1: "feasibility",
    2: "reachability_interaction",
    3: "precedence",
    4: "decision_boundary",
    5: "dynamic_regularity",
}
_ICER_SEMANTIC_FAMILY_FEATURE_NAMES = (
    [f"candidate_family_{_ICER_SEMANTIC_FAMILY_NAMES[f]}_signed_sum" for f in _ICER_SEMANTIC_FAMILY_IDS]
    + [f"delta_family_{_ICER_SEMANTIC_FAMILY_NAMES[f]}_signed_sum" for f in _ICER_SEMANTIC_FAMILY_IDS]
)


def _icer_semantic_family_feature_matrix(
    atom_contrib_star: np.ndarray | None,
    selected_atom_family_ids: np.ndarray | list[int] | None,
    valid_mask: np.ndarray,
    anchor_action: int,
    legacy_action: int,
) -> tuple[np.ndarray, list[str]]:
    """Identity-preserving selected-evidence attribution grouped by fixed semantic family.

    V64.3.24 showed that absolute-magnitude sorting plus per-candidate L1
    normalization can destroy atom identity and distort the local regret geometry.
    V64.3.26 therefore keeps a fixed semantic coordinate system.  For each of the
    five frozen evidence families it exposes (1) the candidate's signed selected-
    evidence contribution sum and (2) the candidate-minus-incumbent signed sum on
    the *same selected atoms*.  There is no sorting, L1 normalization, learned
    embedding, validation-selected family weight, or family-specific threshold.
    """
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    K = len(valid); a = int(anchor_action); legacy = int(legacy_action)
    out = np.zeros((K, len(_ICER_SEMANTIC_FAMILY_FEATURE_NAMES)), dtype=np.float64)
    contrib = np.zeros((0, K), dtype=np.float64)
    if atom_contrib_star is not None:
        raw = np.asarray(atom_contrib_star, dtype=np.float64)
        if raw.ndim == 2 and raw.shape[1] >= K:
            contrib = raw[:, :K]
    if selected_atom_family_ids is None:
        return out, list(_ICER_SEMANTIC_FAMILY_FEATURE_NAMES)
    fam = np.asarray(selected_atom_family_ids, dtype=np.int64).reshape(-1)
    if contrib.shape[0] != fam.size:
        if contrib.shape[0] == 0 and fam.size == 0:
            return out, list(_ICER_SEMANTIC_FAMILY_FEATURE_NAMES)
        raise ValueError(
            f"EAF-ICER semantic-family attribution requires one family id per selected atom: "
            f"contrib_rows={contrib.shape[0]} family_ids={fam.size}"
        )
    unknown = sorted(set(int(x) for x in fam.tolist()) - {0, *_ICER_SEMANTIC_FAMILY_IDS})
    if unknown:
        raise ValueError(f"EAF-ICER semantic-family attribution received unknown family ids {unknown}")
    legacy_vec = (
        contrib[:, legacy]
        if contrib.size and 0 <= legacy < K and legacy != a
        else np.zeros((contrib.shape[0],), dtype=np.float64)
    )
    for b in np.flatnonzero(valid).tolist():
        if b == a:
            continue
        cand = contrib[:, b] if contrib.size else np.zeros((0,), dtype=np.float64)
        delta = cand - legacy_vec if contrib.size else np.zeros((0,), dtype=np.float64)
        for j, f in enumerate(_ICER_SEMANTIC_FAMILY_IDS):
            mask = fam == int(f)
            if np.any(mask):
                out[b, j] = float(np.sum(cand[mask]))
                out[b, len(_ICER_SEMANTIC_FAMILY_IDS) + j] = float(np.sum(delta[mask]))
    return out, list(_ICER_SEMANTIC_FAMILY_FEATURE_NAMES)

_ICER_SEMANTIC_TYPE_NAMES = (
    "occupancy",
    "ttc",
    "gap",
    "drivable_area",
    "wrong_way",
    "speed_limit",
    "red_light",
    "route_connector",
    "local_comfort_accel",
    "local_comfort_jerk",
    "local_comfort_curvature",
    "local_comfort_brake",
)
_ICER_SEMANTIC_TYPE_FEATURE_NAMES = (
    [f"candidate_type_{t}_signed_sum" for t in _ICER_SEMANTIC_TYPE_NAMES]
    + [f"delta_type_{t}_signed_sum" for t in _ICER_SEMANTIC_TYPE_NAMES]
)


def _icer_semantic_type_feature_matrix(
    atom_contrib_star: np.ndarray | None,
    selected_atom_type_names: list[str] | np.ndarray | None,
    valid_mask: np.ndarray,
    anchor_action: int,
    legacy_action: int,
) -> tuple[np.ndarray, list[str]]:
    """Type-resolved selected-evidence attribution on a fixed semantic axis.

    V64.3.26 showed that five coarse family sums are not outcome-sufficient and,
    when concatenated into one KNN geometry, can hide a catastrophic mode that
    the aggregate evidence view rejects.  V64.3.27 therefore exposes the finer
    *existing* atom-type identity without sorting or candidate-wise
    normalization.  This view is used only to confirm the already selected
    aggregate-DRC candidate; it is never allowed to re-rank or resurrect an
    alternative.
    """
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    K = len(valid); a = int(anchor_action); legacy = int(legacy_action)
    out = np.zeros((K, len(_ICER_SEMANTIC_TYPE_FEATURE_NAMES)), dtype=np.float64)
    contrib = np.zeros((0, K), dtype=np.float64)
    if atom_contrib_star is not None:
        raw = np.asarray(atom_contrib_star, dtype=np.float64)
        if raw.ndim == 2 and raw.shape[1] >= K:
            contrib = raw[:, :K]
    if selected_atom_type_names is None:
        return out, list(_ICER_SEMANTIC_TYPE_FEATURE_NAMES)
    types = np.asarray([str(x) for x in list(selected_atom_type_names)], dtype=object).reshape(-1)
    if contrib.shape[0] != types.size:
        if contrib.shape[0] == 0 and types.size == 0:
            return out, list(_ICER_SEMANTIC_TYPE_FEATURE_NAMES)
        raise ValueError(
            "EAF-ICER type-resolved attribution requires one atom type per selected attribution row: "
            f"contrib_rows={contrib.shape[0]} type_names={types.size}"
        )
    unknown = sorted(set(types.tolist()) - set(_ICER_SEMANTIC_TYPE_NAMES))
    if unknown:
        raise ValueError(f"EAF-ICER type-resolved attribution received unknown selected atom types {unknown}")
    legacy_vec = (
        contrib[:, legacy]
        if contrib.size and 0 <= legacy < K and legacy != a
        else np.zeros((contrib.shape[0],), dtype=np.float64)
    )
    T = len(_ICER_SEMANTIC_TYPE_NAMES)
    for b in np.flatnonzero(valid).tolist():
        if b == a:
            continue
        cand = contrib[:, b] if contrib.size else np.zeros((0,), dtype=np.float64)
        delta = cand - legacy_vec if contrib.size else np.zeros((0,), dtype=np.float64)
        for j, name in enumerate(_ICER_SEMANTIC_TYPE_NAMES):
            mask = types == name
            if np.any(mask):
                out[b, j] = float(np.sum(cand[mask]))
                out[b, T + j] = float(np.sum(delta[mask]))
    return out, list(_ICER_SEMANTIC_TYPE_FEATURE_NAMES)


def _decisive_frontier_dacer_features(
    margin_star: np.ndarray,
    attribution_star: np.ndarray | None,
    atom_contrib_star: np.ndarray | None,
    scores: np.ndarray,
    valid_mask: np.ndarray,
    anchor_action: int,
    legacy_action: int,
    potential_diag: dict[str, Any],
    evidence_certificate_fraction: float | None,
    utility_refinement_diag: dict[str, Any] | None,
    admissible_mask: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Guard-admissible scalar features + exact signed selected-atom profile."""
    # Reuse the audited V64.3.17 scalar representation, but compute its candidate
    # fraction on the guard-admissible frontier rather than the collapsed utility
    # intersection.  This preserves all previous diagnostics as a strict ablation.
    scalar, scalar_names = _decisive_frontier_daler_features(
        margin_star, attribution_star, scores, valid_mask, anchor_action, legacy_action,
        potential_diag, evidence_certificate_fraction, utility_refinement_diag,
        admissible_mask, cfg,
    )
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    K = len(valid); a = int(anchor_action); legacy = int(legacy_action)
    challengers = valid.copy()
    if 0 <= a < K:
        challengers[a] = False

    utility_prior = np.zeros((K,), dtype=bool)
    if utility_refinement_diag is not None:
        raw = utility_refinement_diag.get("_utility_refinement_eligible_mask", None)
        if raw is not None:
            tmp = np.asarray(raw, dtype=bool).reshape(-1)
            utility_prior[: min(K, len(tmp))] = tmp[: min(K, len(tmp))]
    if 0 <= a < K:
        utility_prior[a] = False

    contrib = np.zeros((0, K), dtype=np.float64)
    if atom_contrib_star is not None:
        tmp = np.asarray(atom_contrib_star, dtype=np.float64)
        if tmp.ndim == 2 and tmp.shape[1] >= K:
            contrib = tmp[:, :K]
    S = int(contrib.shape[0])
    legacy_vec = contrib[:, legacy] if S and 0 <= legacy < K and legacy != a else np.zeros((S,), dtype=np.float64)
    extra = np.zeros((K, len(_DACER_PROFILE_FEATURE_NAMES)), dtype=np.float64)
    for b in np.flatnonzero(challengers).tolist():
        cand = contrib[:, b] if S else np.zeros((0,), dtype=np.float64)
        delta = cand - legacy_vec if S else np.zeros((0,), dtype=np.float64)
        cp = _signed_atom_profile(cand, top_k=4)
        dp = _signed_atom_profile(delta, top_k=4)
        # cp/dp each: l1, positive-mass fraction, top1-abs fraction,
        # normalized effective support, then top-4 signed normalized atoms.
        budget_norm = max(float((cfg.get("evidence", {}) or {}).get("budget", 16)), 1.0)
        vals = [
            float(S) / budget_norm,
            cp[0], cp[1], cp[2], cp[3],
            dp[0], dp[1], dp[2], dp[3],
            cp[4], cp[5], cp[6], cp[7],
            dp[4], dp[5], dp[6], dp[7],
        ]
        extra[b] = np.asarray(vals, dtype=np.float64)
    return np.concatenate([scalar, extra], axis=1), list(_DACER_FEATURE_NAMES), utility_prior


def _apply_decisive_frontier_dacer(
    legacy_action: int,
    anchor_action: int,
    margin_matrix: np.ndarray,
    attribution_star: np.ndarray | None,
    atom_contrib_star: np.ndarray | None,
    scores: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    potential_diag: dict[str, Any],
    evidence_certificate_fraction: float | None,
    utility_refinement_diag: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """V64.3.18 guard-admissible counterfactual extremal recovery."""
    runtime_cfg = cfg.get("runtime", {}) or {}
    frontier_cfg = runtime_cfg.get("decisive_frontier_value", {}) or {}
    dacer_cfg = frontier_cfg.get("deployment_admissible_counterfactual_extremal_recovery", {}) or {}
    enabled = bool(dacer_cfg.get("enabled", False))
    instrument = bool(dacer_cfg.get("instrument_features", True))
    frontier_active = bool(float(potential_diag.get("decisive_frontier_value_active", 0.0)) >= 0.5)
    M = np.asarray(margin_matrix, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    a = int(anchor_action); legacy = int(legacy_action)
    margin_star = M[:, a].copy() if 0 <= a < M.shape[1] else np.zeros((len(valid),), dtype=np.float64)
    admissible = _decisive_frontier_guard_admissible_mask(
        margin_star, scores, valid, runtime_safety_flags, a,
        evidence_certificate_fraction, cfg,
        require_safe_available_for_learned_intervention=bool(
            dacer_cfg.get("require_safe_available_for_learned_intervention", True)
        ),
    )
    feat, names, utility_prior = _decisive_frontier_dacer_features(
        margin_star, attribution_star, atom_contrib_star, scores, valid, a, legacy,
        potential_diag, evidence_certificate_fraction, utility_refinement_diag,
        admissible, cfg,
    )
    logits = np.zeros((len(valid),), dtype=np.float64)
    selected = legacy
    applies = bool(enabled and frontier_active and 0 <= a < len(valid))
    anchor_logit = float(dacer_cfg.get("anchor_logit", 0.0))
    feature_mode = str(dacer_cfg.get("feature_mode", "profile")).strip().lower()
    if applies:
        cfg_names = list(dacer_cfg.get("feature_names", []))
        mean = np.asarray(dacer_cfg.get("feature_mean", []), dtype=np.float64).reshape(-1)
        std = np.asarray(dacer_cfg.get("feature_std", []), dtype=np.float64).reshape(-1)
        weights = np.asarray(dacer_cfg.get("weights", []), dtype=np.float64).reshape(-1)
        if cfg_names != names or len(names) != len(mean) or len(names) != len(std) or len(names) != len(weights):
            raise ValueError(
                "EAF-DACER enabled but feature schema/mean/std/weights are inconsistent; "
                "fit the V64.3.18 train-only counterfactual readout before enabling selection"
            )
        z = (feat - mean[None, :]) / np.maximum(std[None, :], 1.0e-6)
        logits = np.clip(z @ weights + float(dacer_cfg.get("bias", 0.0)), -40.0, 40.0)
        cand = np.flatnonzero(admissible & np.isfinite(logits)).astype(np.int64)
        if cand.size:
            util_cost = np.full((len(valid),), np.inf, dtype=np.float64)
            if utility_refinement_diag is not None and utility_refinement_diag.get("_utility_refinement_cost", None) is not None:
                tmp = np.asarray(utility_refinement_diag["_utility_refinement_cost"], dtype=np.float64).reshape(-1)
                util_cost[: min(len(valid), len(tmp))] = tmp[: min(len(valid), len(tmp))]
            # The learned score is the only new extremal operator.  Frozen EAF
            # margin/utility are deterministic tie-breakers, never hard gates
            # beyond the already-computed guard-admissible mask.
            best = sorted(
                cand.tolist(),
                key=lambda b: (
                    -float(logits[b]),
                    -int(b == legacy),
                    -float(margin_star[b]),
                    -int(utility_prior[b]),
                    float(util_cost[b]) if np.isfinite(util_cost[b]) else float("inf"),
                    int(b),
                ),
            )[0]
            selected = int(best) if float(logits[best]) > anchor_logit + 1.0e-12 else a
        else:
            selected = a
    selected_logit = float(logits[selected]) if 0 <= selected < len(logits) and selected != a else anchor_logit
    legacy_logit = float(logits[legacy]) if 0 <= legacy < len(logits) and legacy != a else anchor_logit
    diag: dict[str, Any] = {
        "decisive_frontier_dacer_enabled": float(enabled),
        "decisive_frontier_dacer_instrument_features": float(instrument),
        "decisive_frontier_dacer_active": float(applies),
        "decisive_frontier_dacer_feature_mode_profile": float(feature_mode == "profile"),
        "decisive_frontier_dacer_legacy_selected_action": float(legacy),
        "decisive_frontier_dacer_selected_action": float(selected),
        "decisive_frontier_dacer_proposal_changed": float(int(selected) != int(legacy)),
        "decisive_frontier_dacer_anchor_fallback": float(int(selected) == int(a)),
        "decisive_frontier_dacer_selected_logit": selected_logit,
        "decisive_frontier_dacer_legacy_logit": legacy_logit,
        "decisive_frontier_dacer_admissible_candidate_count": float(np.asarray(admissible, dtype=bool).sum()),
        "decisive_frontier_dacer_utility_prior_candidate_count": float(np.asarray(utility_prior, dtype=bool).sum()),
        "_decisive_frontier_dacer_anchor_action": int(a),
        "_decisive_frontier_dacer_legacy_selected_action": int(legacy),
        "_decisive_frontier_dacer_raw_margin_star": np.asarray(margin_star, dtype=np.float32),
        "_decisive_frontier_dacer_attribution_scale_star": np.zeros_like(margin_star, dtype=np.float32)
            if attribution_star is None else np.asarray(attribution_star, dtype=np.float32).reshape(-1)[: len(margin_star)],
        "_decisive_frontier_dacer_feature_matrix": np.asarray(feat, dtype=np.float32),
        "_decisive_frontier_dacer_feature_names": names,
        "_decisive_frontier_dacer_logit_star": np.asarray(logits, dtype=np.float32),
        "_decisive_frontier_dacer_admissible_mask": np.asarray(admissible, dtype=bool),
        "_decisive_frontier_dacer_utility_prior_mask": np.asarray(utility_prior, dtype=bool),
    }
    return selected, diag




_ICER_TRANSITION_MANEUVER_NAMES = [
    "keep_follow", "decelerate_stop", "yield_creep", "lane_change_left",
    "lane_change_right", "route_turn_connector", "safe_fallback",
]
_ICER_TRANSITION_FEATURE_NAMES = (
    [f"candidate_maneuver_{n}" for n in _ICER_TRANSITION_MANEUVER_NAMES]
    + [f"reference_maneuver_{n}" for n in _ICER_TRANSITION_MANEUVER_NAMES]
    + [
        "same_maneuver", "candidate_safe_like", "reference_safe_like",
        "candidate_progressive", "reference_progressive", "safe_to_progressive",
        "progressive_to_safe", "candidate_terminal_x_norm", "reference_terminal_x_norm",
        "delta_terminal_x_norm", "candidate_terminal_y_norm", "reference_terminal_y_norm",
        "delta_terminal_y_norm", "candidate_terminal_speed_norm", "reference_terminal_speed_norm",
        "delta_terminal_speed_norm", "candidate_path_length_norm", "reference_path_length_norm",
        "delta_path_length_norm", "candidate_max_abs_y_norm", "reference_max_abs_y_norm",
        "delta_max_abs_y_norm", "endpoint_separation_norm", "mean_path_separation_norm",
        "max_path_separation_norm", "sin_terminal_yaw_delta", "cos_terminal_yaw_delta",
    ]
)


def _icer_transition_feature_matrix(
    candidate_trajectories: np.ndarray | None,
    maneuver_ids: np.ndarray | None,
    valid_mask: np.ndarray,
    reference_action: int,
) -> tuple[np.ndarray, list[str]]:
    """Runtime-only planner-transition semantics for regret-tail reliability.

    These features deliberately describe *how the proposed planner action changes*
    relative to a frozen reference action.  They use only the already-generated
    candidate bank (trajectory and maneuver family), add no evidence query, and
    contain no teacher/future information.  V64.3.21 exposed repeated high-regret
    incumbent->candidate failures that were visible in TRAIN but indistinguishable
    to the evidence-only reliability heads.  The transition view makes those
    planner semantics explicit without memorizing candidate-slot indices.
    """
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    K = int(len(valid))
    out = np.zeros((K, len(_ICER_TRANSITION_FEATURE_NAMES)), dtype=np.float64)
    if K == 0:
        return out, list(_ICER_TRANSITION_FEATURE_NAMES)

    man = np.full((K,), -999, dtype=np.int64)
    if maneuver_ids is not None:
        src = np.asarray(maneuver_ids, dtype=np.int64).reshape(-1)
        man[: min(K, src.size)] = src[:K]
    ref = int(reference_action)
    if not (0 <= ref < K):
        ref = 0
    ref_man = int(man[ref])
    col = 0
    for mid in range(len(_ICER_TRANSITION_MANEUVER_NAMES)):
        out[:, col] = (man == mid).astype(np.float64); col += 1
    for mid in range(len(_ICER_TRANSITION_MANEUVER_NAMES)):
        out[:, col] = float(ref_man == mid); col += 1
    cand_safe = np.asarray([int(m) in SAFE_LIKE_MANEUVER_IDS for m in man], dtype=np.float64)
    cand_prog = np.asarray([int(m) in PROGRESSIVE_MANEUVER_IDS for m in man], dtype=np.float64)
    ref_safe = float(ref_man in SAFE_LIKE_MANEUVER_IDS)
    ref_prog = float(ref_man in PROGRESSIVE_MANEUVER_IDS)
    out[:, col] = (man == ref_man).astype(np.float64); col += 1
    out[:, col] = cand_safe; col += 1
    out[:, col] = ref_safe; col += 1
    out[:, col] = cand_prog; col += 1
    out[:, col] = ref_prog; col += 1
    out[:, col] = cand_safe * ref_prog; col += 1
    out[:, col] = cand_prog * ref_safe; col += 1

    tr = None
    if candidate_trajectories is not None:
        raw = np.asarray(candidate_trajectories, dtype=np.float64)
        if raw.ndim == 3 and raw.shape[0] >= K and raw.shape[1] >= 1 and raw.shape[2] >= 2:
            tr = np.nan_to_num(raw[:K], nan=0.0, posinf=0.0, neginf=0.0)
    if tr is None:
        # Keep semantic maneuver features available even if a legacy caller has
        # no trajectory bank.  The config contract prevents a transition-trained
        # head from silently consuming all-zero geometry at deployment.
        return out, list(_ICER_TRANSITION_FEATURE_NAMES)

    x = tr[:, :, 0]
    y = tr[:, :, 1]
    yaw = tr[:, :, 2] if tr.shape[2] > 2 else np.zeros_like(x)
    speed = tr[:, :, 3] if tr.shape[2] > 3 else np.zeros_like(x)
    end_x, end_y, end_yaw, end_speed = x[:, -1], y[:, -1], yaw[:, -1], speed[:, -1]
    dxy = np.diff(tr[:, :, :2], axis=1)
    path_len = np.linalg.norm(dxy, axis=2).sum(axis=1) if dxy.shape[1] else np.zeros((K,), dtype=np.float64)
    max_abs_y = np.max(np.abs(y), axis=1)
    scale_idx = np.flatnonzero(valid)
    if scale_idx.size == 0:
        scale_idx = np.arange(K)
    pos_scale = max(float(np.max(np.sqrt(end_x[scale_idx] ** 2 + end_y[scale_idx] ** 2))), 1.0)
    speed_scale = max(float(np.max(np.abs(end_speed[scale_idx]))), 1.0)
    length_scale = max(float(np.max(np.abs(path_len[scale_idx]))), 1.0)
    lateral_scale = max(float(np.max(np.abs(max_abs_y[scale_idx]))), 1.0)

    rx, ry, ryaw, rv = float(end_x[ref]), float(end_y[ref]), float(end_yaw[ref]), float(end_speed[ref])
    rlen, rlat = float(path_len[ref]), float(max_abs_y[ref])
    out[:, col] = end_x / pos_scale; col += 1
    out[:, col] = rx / pos_scale; col += 1
    out[:, col] = (end_x - rx) / pos_scale; col += 1
    out[:, col] = end_y / pos_scale; col += 1
    out[:, col] = ry / pos_scale; col += 1
    out[:, col] = (end_y - ry) / pos_scale; col += 1
    out[:, col] = end_speed / speed_scale; col += 1
    out[:, col] = rv / speed_scale; col += 1
    out[:, col] = (end_speed - rv) / speed_scale; col += 1
    out[:, col] = path_len / length_scale; col += 1
    out[:, col] = rlen / length_scale; col += 1
    out[:, col] = (path_len - rlen) / length_scale; col += 1
    out[:, col] = max_abs_y / lateral_scale; col += 1
    out[:, col] = rlat / lateral_scale; col += 1
    out[:, col] = (max_abs_y - rlat) / lateral_scale; col += 1

    ref_xy = tr[ref, :, :2]
    sep = np.linalg.norm(tr[:, :, :2] - ref_xy[None, :, :], axis=2)
    out[:, col] = np.linalg.norm(np.stack([end_x-rx, end_y-ry], axis=1), axis=1) / pos_scale; col += 1
    out[:, col] = np.mean(sep, axis=1) / pos_scale; col += 1
    out[:, col] = np.max(sep, axis=1) / pos_scale; col += 1
    dyaw = np.arctan2(np.sin(end_yaw - ryaw), np.cos(end_yaw - ryaw))
    out[:, col] = np.sin(dyaw); col += 1
    out[:, col] = np.cos(dyaw); col += 1
    if col != out.shape[1]:
        raise RuntimeError(f"ICER transition feature count mismatch: wrote {col}, expected {out.shape[1]}")
    return out, list(_ICER_TRANSITION_FEATURE_NAMES)



_ICER_DOMINANCE_PROFILE_BASE_NAMES = [
    "raw_margin", "attribution_scale", "margin_over_attribution", "attribution_over_frontier_rms",
    "raw_margin_z", "attribution_z", "raw_margin_rank", "attribution_rank", "margin_below_frontier_max",
    "margin_minus_legacy_selected", "attribution_minus_legacy_selected", "eaf_score_gain_vs_anchor",
    "eaf_score_minus_legacy_selected", "utility_cost_minus_legacy_selected", "guard_margin_excess",
    "eaf_score_rank", "utility_cost_rank", "executable_candidate_fraction",
    "atom_contrib_l1", "atom_contrib_positive_mass_fraction", "atom_contrib_top1_abs_fraction",
    "atom_contrib_effective_support_norm", "delta_atom_contrib_l1",
    "delta_atom_contrib_positive_mass_fraction", "delta_atom_contrib_top1_abs_fraction",
    "delta_atom_contrib_effective_support_norm", "delta_atom_top1_signed_norm",
    "delta_atom_top2_signed_norm", "delta_atom_top3_signed_norm", "delta_atom_top4_signed_norm",
]

_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES = list(_ICER_DOMINANCE_PROFILE_BASE_NAMES[:18])


@lru_cache(maxsize=16)
def _load_icer_local_regret_memory(
    path_str: str,
    expected_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, tuple[int, ...], float, str, float]:
    """Load one immutable TRAIN-only V64.3.23 regret-coherence memory.

    The memory is deliberately small and auditable.  It stores only frozen
    TRAIN runtime features and candidate-vs-incumbent teacher-improvement
    targets.  Runtime lookup uses a pre-registered group-balanced metric and a
    two-scale one-standard-error lower bound; validation/test labels are never
    stored or consulted.
    """
    path = Path(path_str)
    if not path.is_file():
        raise ValueError(f"missing EAF-ICER local regret memory: {path}")
    if expected_sha256:
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if h != expected_sha256:
            raise ValueError(f"EAF-ICER local regret memory SHA256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        memory = np.asarray(z["memory_metric_z"], dtype=np.float64)
        delta = np.asarray(z["teacher_improvement"], dtype=np.float64).reshape(-1)
        mean = np.asarray(z["feature_mean"], dtype=np.float64).reshape(-1)
        std = np.asarray(z["feature_std"], dtype=np.float64).reshape(-1)
        names = tuple(str(x) for x in np.asarray(z["feature_names"]).reshape(-1).tolist())
        metric_weight = np.asarray(z["feature_metric_weight"], dtype=np.float64).reshape(-1)
        ks = tuple(int(x) for x in np.asarray(z["neighbor_k_values"]).reshape(-1).tolist())
        se_multiplier = float(np.asarray(z["se_multiplier"]).reshape(-1)[0])
        certificate_kind = str(np.asarray(z["certificate_kind"]).reshape(-1)[0]) if "certificate_kind" in z.files else "mean_minus_standard_error"
        downside_multiplier = float(np.asarray(z["downside_multiplier"]).reshape(-1)[0]) if "downside_multiplier" in z.files else 1.0
    if (
        memory.ndim != 2
        or memory.shape[0] != len(delta)
        or memory.shape[1] != len(mean)
        or len(mean) != len(std)
        or len(names) != len(mean)
        or len(metric_weight) != len(mean)
    ):
        raise ValueError("EAF-ICER local regret memory schema is inconsistent")
    if not ks or any(k <= 1 for k in ks) or memory.shape[0] < max(ks):
        raise ValueError("EAF-ICER local regret memory has insufficient TRAIN support")
    if np.any(metric_weight <= 0.0) or not np.isfinite(se_multiplier) or se_multiplier < 0.0:
        raise ValueError("EAF-ICER local regret metric/lower-bound parameters invalid")
    if certificate_kind not in {"mean_minus_standard_error", "mean_minus_downside_rms"}:
        raise ValueError(f"unknown EAF-ICER local regret certificate_kind={certificate_kind}")
    if not np.isfinite(downside_multiplier) or downside_multiplier < 0.0:
        raise ValueError("EAF-ICER local downside multiplier invalid")
    if not (
        np.all(np.isfinite(memory))
        and np.all(np.isfinite(delta))
        and np.all(np.isfinite(mean))
        and np.all(np.isfinite(std))
        and np.all(np.isfinite(metric_weight))
    ):
        raise ValueError("EAF-ICER local regret memory contains non-finite values")
    return memory, delta, mean, np.maximum(std, 1.0e-6), names, metric_weight, ks, se_multiplier, certificate_kind, downside_multiplier


@lru_cache(maxsize=16)
def _load_icer_global_tail_mode_model(
    path_str: str,
    expected_sha256: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Load immutable TRAIN-only V64.3.28 global catastrophic-mode model."""
    path = Path(path_str)
    if not path.is_file():
        raise ValueError(f"missing EAF-ICER global tail-mode model: {path}")
    if expected_sha256:
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if h != expected_sha256:
            raise ValueError(f"EAF-ICER global tail-mode model SHA256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        mean = np.asarray(z["feature_mean"], dtype=np.float64).reshape(-1)
        std = np.asarray(z["feature_std"], dtype=np.float64).reshape(-1)
        names = tuple(str(x) for x in np.asarray(z["feature_names"]).reshape(-1).tolist())
        cat_mean = np.asarray(z["catastrophic_mean"], dtype=np.float64).reshape(-1)
        cat_var = np.asarray(z["catastrophic_var"], dtype=np.float64).reshape(-1)
        benign_mean = np.asarray(z["benign_mean"], dtype=np.float64).reshape(-1)
        benign_var = np.asarray(z["benign_var"], dtype=np.float64).reshape(-1)
        risk_threshold = float(np.asarray(z["risk_threshold"]).reshape(-1)[0])
        catastrophic_delta_threshold = float(np.asarray(z["catastrophic_delta_threshold"]).reshape(-1)[0])
        positive_proposal_coverage = float(np.asarray(z["positive_proposal_coverage"]).reshape(-1)[0])
    n = len(names)
    if not (n and len(mean) == len(std) == len(cat_mean) == len(cat_var) == len(benign_mean) == len(benign_var) == n):
        raise ValueError("EAF-ICER global tail-mode model schema is inconsistent")
    if not all(np.all(np.isfinite(a)) for a in (mean, std, cat_mean, cat_var, benign_mean, benign_var)):
        raise ValueError("EAF-ICER global tail-mode model contains non-finite arrays")
    if np.any(std <= 0.0) or np.any(cat_var <= 0.0) or np.any(benign_var <= 0.0):
        raise ValueError("EAF-ICER global tail-mode model has non-positive scale/variance")
    if not (np.isfinite(risk_threshold) and np.isfinite(catastrophic_delta_threshold) and 0.0 < positive_proposal_coverage < 1.0):
        raise ValueError("EAF-ICER global tail-mode model scalar contract invalid")
    return mean, std, names, cat_mean, cat_var, benign_mean, benign_var, risk_threshold, catastrophic_delta_threshold, positive_proposal_coverage


def _icer_global_tail_mode_confirmation_score(
    x: np.ndarray,
    runtime_names: list[str],
    model_path: str,
    model_sha256: str,
) -> np.ndarray:
    """Return a zero-boundary confirmation score for V64.3.28.

    Risk is the equal-prior diagonal-Gaussian log likelihood ratio
    ``log p(x|catastrophic) - log p(x|non-catastrophic)``.  Positive output
    means the proposal lies below the immutable TRAIN-calibrated risk threshold.
    """
    mean, std, names, cat_mean, cat_var, benign_mean, benign_var, risk_threshold, _, _ = _load_icer_global_tail_mode_model(
        model_path, model_sha256
    )
    if list(names) != list(runtime_names):
        raise ValueError("EAF-ICER global tail-mode feature schema mismatch")
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[1] != len(mean):
        raise ValueError("EAF-ICER global tail-mode runtime feature shape mismatch")
    z = (xx - mean[None, :]) / std[None, :]
    log_cat = -0.5 * np.sum(np.log(cat_var)[None, :] + ((z - cat_mean[None, :]) ** 2) / cat_var[None, :], axis=1)
    log_benign = -0.5 * np.sum(np.log(benign_var)[None, :] + ((z - benign_mean[None, :]) ** 2) / benign_var[None, :], axis=1)
    risk = log_cat - log_benign
    return risk_threshold - risk


def _icer_local_regret_lower_bound(
    x: np.ndarray,
    runtime_names: list[str],
    memory_path: str,
    memory_sha256: str,
) -> np.ndarray:
    """Two-scale local lower-bound estimate of candidate-vs-incumbent gain.

    V64.3.23 intentionally does not learn a larger network or a validation
    threshold.  For each runtime candidate it forms inverse-distance-weighted
    TRAIN neighborhoods at two fixed scales.  At each scale the risk score is
    ``local mean teacher improvement - 1 standard error``; the runtime score is
    the minimum across scales.  Positive therefore means both a local and a
    coarser neighborhood support non-negative replacement improvement under the
    same fixed evidence/transition metric.
    """
    memory, delta, mean, std, names, metric_weight, ks, se_multiplier, certificate_kind, downside_multiplier = _load_icer_local_regret_memory(
        memory_path, memory_sha256
    )
    if list(names) != list(runtime_names):
        raise ValueError("EAF-ICER local regret memory feature schema mismatch")
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[1] != len(mean):
        raise ValueError("EAF-ICER local regret runtime feature shape mismatch")
    z = ((xx - mean[None, :]) / std[None, :]) * np.sqrt(metric_weight[None, :])
    z2 = np.sum(z * z, axis=1, keepdims=True)
    m2 = np.sum(memory * memory, axis=1, keepdims=False)[None, :]
    d2 = np.maximum(z2 + m2 - 2.0 * (z @ memory.T), 0.0)
    rows = np.arange(len(z))[:, None]
    bounds: list[np.ndarray] = []
    for k in ks:
        kk = min(int(k), memory.shape[0])
        nbr = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        dist = np.sqrt(d2[rows, nbr])
        w = 1.0 / np.maximum(dist, 1.0e-6)
        w = w / np.maximum(np.sum(w, axis=1, keepdims=True), 1.0e-12)
        y = delta[nbr]
        local_mean = np.sum(w * y, axis=1)
        if certificate_kind == "mean_minus_standard_error":
            local_var = np.sum(w * (y - local_mean[:, None]) ** 2, axis=1)
            effective_n = 1.0 / np.maximum(np.sum(w * w, axis=1), 1.0e-12)
            local_se = np.sqrt(local_var / np.maximum(effective_n, 1.0))
            bound = local_mean - se_multiplier * local_se
        else:
            # V64.3.24 outcome/downside certificate.  The V23 failure showed
            # that confidence in a positive neighborhood *mean* does not bound
            # the downside of one extremally selected replacement.  Penalize
            # the weighted RMS magnitude of negative local outcomes directly.
            downside = np.minimum(y, 0.0)
            downside_rms = np.sqrt(np.sum(w * downside * downside, axis=1))
            bound = local_mean - downside_multiplier * downside_rms
        bounds.append(bound)
    return np.min(np.stack(bounds, axis=1), axis=1)

def _icer_select_extremal_candidate_with_optional_confirmation(
    candidate_indices: np.ndarray | list[int],
    dominance_logits: np.ndarray,
    replacement_regret_risk_logits: np.ndarray,
    support_logits: np.ndarray,
    margin_star: np.ndarray,
    utility_prior: np.ndarray,
    *,
    confirmation_logits: np.ndarray | None = None,
) -> int | None:
    """Select one aggregate-DRC extremal candidate, then optionally confirm only it.

    V64.3.27's scientific invariant is monotone refinement: the confirmation
    view cannot re-rank candidates or fall through to a second alternative.
    If the aggregate proposal fails confirmation, ``None`` is returned and the
    caller preserves the incumbent.
    """
    cand = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if cand.size == 0:
        return None
    best = sorted(
        cand.tolist(),
        key=lambda b: (
            -float(dominance_logits[b]),
            -float(replacement_regret_risk_logits[b]),
            -float(support_logits[b]),
            -float(margin_star[b]),
            -int(utility_prior[b]),
            int(b),
        ),
    )[0]
    if confirmation_logits is not None:
        c = np.asarray(confirmation_logits, dtype=np.float64).reshape(-1)
        if best >= len(c) or not (np.isfinite(c[best]) and c[best] > 0.0):
            return None
    return int(best)


def _icer_regret_risk_feature_matrix(
    feat: np.ndarray,
    feature_names: list[str],
    transition_feat: np.ndarray,
    transition_names: list[str],
    mode: str,
    attribution_resolved_feat: np.ndarray | None = None,
    attribution_resolved_names: list[str] | None = None,
    semantic_family_feat: np.ndarray | None = None,
    semantic_family_names: list[str] | None = None,
    semantic_type_feat: np.ndarray | None = None,
    semantic_type_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Auditable feature view for V64.3.22 regret-risk heads.

    The evidence-local main uses exactly the 18 scalar runtime EAF statistics
    already present in ICER.  The transition-conditioned controlled ablation
    appends planner-transition semantics from the candidate bank.  There is no
    learned hidden representation and no validation-selected feature subset.
    """
    pos = {str(n): i for i, n in enumerate(feature_names)}
    if any(n not in pos for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES):
        raise ValueError("EAF-ICER regret-risk evidence features missing from runtime schema")
    base = np.asarray(feat[:, [pos[n] for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]], dtype=np.float64)
    names = [f"evidence::{n}" for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]
    mode = str(mode).strip().lower()
    if mode == "evidence_only":
        return base, names
    if mode == "attribution_resolved":
        ar = np.asarray(attribution_resolved_feat, dtype=np.float64) if attribution_resolved_feat is not None else np.zeros((0, 0), dtype=np.float64)
        ar_names = list(attribution_resolved_names or [])
        ar_budget = ar.shape[1] // 2 if ar.ndim == 2 and ar.shape[1] % 2 == 0 else 0
        expected_ar_names = _icer_attribution_resolved_feature_names(ar_budget) if ar_budget > 0 else []
        if ar.ndim != 2 or ar.shape[0] != base.shape[0] or ar.shape[1] != len(ar_names) or ar_names != expected_ar_names:
            raise ValueError("EAF-ICER attribution-resolved feature matrix/schema mismatch")
        return np.concatenate([base, ar], axis=1), names + [f"attribution::{n}" for n in ar_names]
    if mode == "semantic_family_aligned":
        sf = np.asarray(semantic_family_feat, dtype=np.float64) if semantic_family_feat is not None else np.zeros((0, 0), dtype=np.float64)
        sf_names = list(semantic_family_names or [])
        if sf.ndim != 2 or sf.shape[0] != base.shape[0] or sf.shape[1] != len(sf_names) or sf_names != list(_ICER_SEMANTIC_FAMILY_FEATURE_NAMES):
            raise ValueError("EAF-ICER semantic-family feature matrix/schema mismatch")
        return np.concatenate([base, sf], axis=1), names + [f"semantic_family::{n}" for n in sf_names]
    if mode == "semantic_type_only":
        st = np.asarray(semantic_type_feat, dtype=np.float64) if semantic_type_feat is not None else np.zeros((0, 0), dtype=np.float64)
        st_names = list(semantic_type_names or [])
        if st.ndim != 2 or st.shape[0] != base.shape[0] or st.shape[1] != len(st_names) or st_names != list(_ICER_SEMANTIC_TYPE_FEATURE_NAMES):
            raise ValueError("EAF-ICER semantic-type feature matrix/schema mismatch")
        return st, [f"semantic_type::{n}" for n in st_names]
    if mode != "transition_conditioned":
        raise ValueError(f"unknown EAF-ICER regret_risk_feature_mode={mode}")
    tr = np.asarray(transition_feat, dtype=np.float64)
    if tr.ndim != 2 or tr.shape[0] != base.shape[0] or tr.shape[1] != len(transition_names):
        raise ValueError("EAF-ICER transition feature matrix/schema mismatch")
    return np.concatenate([base, tr], axis=1), names + [f"transition::{n}" for n in transition_names]


def _icer_quadratic_interaction_features(
    feat: np.ndarray, feature_names: list[str], mode: str
) -> tuple[np.ndarray, list[str], list[str]]:
    """Fixed, auditable second-order map for incumbent-contrastive reliability.

    V64.3.18 established signal in the exact signed selected-evidence profile but
    a linear shared score could not protect the extremal operator from high-score
    false-positive alternatives.  ICER keeps a linear *readout* while exposing
    fixed pairwise interactions among the pre-registered runtime-only evidence
    statistics.  This adds no evidence query and no learned hidden representation.
    """
    all_names = list(feature_names)
    if mode == "scalar_interaction":
        base_names = list(_DALER_FEATURE_NAMES)
    elif mode == "profile_interaction":
        base_names = list(_ICER_DOMINANCE_PROFILE_BASE_NAMES)
    else:
        raise ValueError(f"unknown EAF-ICER dominance_feature_mode={mode}")
    pos = {str(n): i for i, n in enumerate(all_names)}
    missing = [n for n in base_names if n not in pos]
    if missing:
        raise ValueError(f"EAF-ICER dominance base features missing from DACER schema: {missing}")
    X = np.asarray(feat, dtype=np.float64)[:, [pos[n] for n in base_names]]
    parts = [X]
    out_names = [f"lin::{n}" for n in base_names]
    for i, ni in enumerate(base_names):
        block = X[:, i : i + 1] * X[:, i:]
        parts.append(block)
        out_names.extend([f"quad::{ni}*{nj}" for nj in base_names[i:]])
    return np.concatenate(parts, axis=1), out_names, base_names



def _icer_selection_conditioned_intervention_scores(
    feat: np.ndarray,
    feature_names: list[str],
    support_logits: np.ndarray,
    legacy_action: int,
    scir_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """V64.3.31 same-scene incumbent-contrastive intervention utility.

    The V30.3 capacity closure shows that exposing more decisive evidence does
    not stabilize the historical independent dominance head.  SCIR therefore
    removes common-mode scene variation explicitly: the runtime representation
    is the difference between each candidate and the *same-scene incumbent* in
    the frozen 18-D evidence view, plus the difference of the frozen anchor-
    support logit.  A low-capacity TRAIN-only linear ridge readout predicts the
    continuous teacher improvement used only during fitting; runtime consumes
    no teacher/future value and makes no additional evidence query.
    """
    xx = np.asarray(feat, dtype=np.float64)
    sup = np.asarray(support_logits, dtype=np.float64).reshape(-1)
    legacy = int(legacy_action)
    if xx.ndim != 2 or sup.shape[0] != xx.shape[0] or not (0 <= legacy < xx.shape[0]):
        n = xx.shape[0] if xx.ndim == 2 else 0
        return np.zeros((n,), dtype=np.float64), np.zeros((0, 0), dtype=np.float64), [], np.ones((n,), dtype=np.float64)
    base_names = list(scir_cfg.get("base_feature_names", _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES))
    if base_names != list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES):
        raise ValueError("EAF-ICER SCIR base feature schema must equal the frozen 18-D evidence view")
    pos = {str(n): i for i, n in enumerate(feature_names)}
    missing = [n for n in base_names if n not in pos]
    if missing:
        raise ValueError(f"EAF-ICER SCIR base features missing from runtime schema: {missing}")
    base = xx[:, [pos[n] for n in base_names]]
    delta = base - base[legacy : legacy + 1]
    support_delta = (sup - float(sup[legacy]))[:, None]
    x = np.concatenate([delta, support_delta], axis=1)
    runtime_names = [f"delta::{n}" for n in base_names] + ["delta::support_logit"]
    stored_names = list(scir_cfg.get("feature_names", []))
    mean = np.asarray(scir_cfg.get("feature_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(scir_cfg.get("feature_std", []), dtype=np.float64).reshape(-1)
    weights = np.asarray(scir_cfg.get("weights", []), dtype=np.float64).reshape(-1)
    if stored_names != runtime_names or len(runtime_names) != len(mean) or len(runtime_names) != len(std) or len(runtime_names) != len(weights):
        raise ValueError("EAF-ICER SCIR feature schema/mean/std/weights are inconsistent")
    z = (x - mean[None, :]) / np.maximum(std[None, :], 1.0e-6)
    mu = z @ weights + float(scir_cfg.get("bias", 0.0))

    # V64.3.35 optional base-point context shift.  The context is the frozen
    # absolute incumbent evidence view plus incumbent support.  Its contribution
    # is identical for every challenger in the scene, so it can change only the
    # challenger-vs-incumbent intervention boundary; it cannot re-rank two
    # challengers.  Historical configs omit these fields and are bit-compatible.
    context_names = list(scir_cfg.get("incumbent_context_feature_names", []))
    if context_names:
        expected_context_names = [f"incumbent::{n}" for n in base_names] + ["incumbent::support_logit"]
        cmean = np.asarray(scir_cfg.get("incumbent_context_feature_mean", []), dtype=np.float64).reshape(-1)
        cstd = np.asarray(scir_cfg.get("incumbent_context_feature_std", []), dtype=np.float64).reshape(-1)
        cw = np.asarray(scir_cfg.get("incumbent_context_weights", []), dtype=np.float64).reshape(-1)
        if (
            context_names != expected_context_names
            or len(context_names) != len(cmean)
            or len(context_names) != len(cstd)
            or len(context_names) != len(cw)
        ):
            raise ValueError("EAF-ICER SCIR incumbent-context schema/mean/std/weights are inconsistent")
        c = np.concatenate([base[legacy], np.asarray([float(sup[legacy])], dtype=np.float64)])
        cz = (c - cmean) / np.maximum(cstd, 1.0e-6)
        shift = float(cz @ cw + float(scir_cfg.get("incumbent_context_bias", 0.0)))
        mu = mu + shift
        # Incumbent remains the exact zero-score pseudo-action.
        mu[legacy] = 0.0
    mu = np.clip(mu, -40.0, 40.0)

    # V64.3.32 selection-stable scale.  This is intentionally *not* claimed as
    # a probabilistic variance.  It is a frozen TRAIN-only ridge-leverage scale
    # used to normalize one-sided conformal nonconformity.  If a historical V31
    # artifact omits the matrix, the scale is exactly one and legacy behavior is
    # bit-compatible.
    leverage_inverse = np.asarray(scir_cfg.get("leverage_inverse", []), dtype=np.float64)
    if leverage_inverse.size == 0:
        scale = np.ones((x.shape[0],), dtype=np.float64)
    else:
        if leverage_inverse.shape != (len(runtime_names), len(runtime_names)):
            raise ValueError("EAF-ICER SCIR leverage_inverse shape is inconsistent with feature schema")
        h = np.einsum("bi,ij,bj->b", z, leverage_inverse, z)
        h = np.maximum(np.where(np.isfinite(h), h, 0.0), 0.0)
        scale = np.sqrt(1.0 + h)
        floor = max(float(scir_cfg.get("selection_scale_floor", 1.0)), 1.0e-6)
        scale = np.maximum(scale, floor)
    return mu, x, runtime_names, scale



def _icer_scene_reservation_value(
    predicted_improvement: np.ndarray,
    candidate_indices: np.ndarray | list[int],
    feat: np.ndarray,
    feature_names: list[str],
    support_logits: np.ndarray,
    legacy_action: int,
    scir_cfg: dict[str, Any],
) -> tuple[float, np.ndarray, list[str]]:
    """V64.3.36 monotone scene reservation for the frozen challenger ordering.

    V35 shows that a common incumbent-context shift has weak existence signal but
    its joint refit with the delta head does not isolate whether the gain comes
    from the boundary or from a changed challenger ordering.  V36 freezes the
    RSMR ordering and learns a *non-negative* reservation from an independent
    selected-policy calibration population.  The reservation is subtracted from
    every eligible challenger score, so it can only shrink the frozen proposal
    set and can never re-rank challengers or create a new intervention path.

    Supported views:
      - incumbent_basepoint: absolute frozen 18-D incumbent evidence + support;
      - selection_geometry: permutation-invariant geometry of the frozen RSMR
        score set (top score, top-vs-runner-up/incumbent gap, RMS, positive
        fraction, and log effective competitor mass).
    """
    cand = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    mode = str(scir_cfg.get("scene_reservation_feature_mode", "")).strip().lower()
    if cand.size == 0 or not mode:
        return 0.0, np.zeros((0,), dtype=np.float64), []
    mu = np.asarray(predicted_improvement, dtype=np.float64).reshape(-1)
    if np.any(cand < 0) or np.any(cand >= mu.size):
        raise ValueError("EAF-ICER V36 reservation candidate indices are invalid")

    if mode == "selection_geometry":
        sv = np.asarray(mu[cand], dtype=np.float64)
        if sv.size == 0 or not np.all(np.isfinite(sv)):
            raise ValueError("EAF-ICER V36 selection-geometry reservation requires finite frozen scores")
        order = np.sort(sv)[::-1]
        top = float(order[0])
        second = float(order[1]) if order.size >= 2 else float("-inf")
        runner = max(0.0, second) if math.isfinite(second) else 0.0
        gap = top - runner
        rms = float(np.sqrt(np.mean(sv * sv)))
        pos_frac = float(np.mean(sv > 0.0))
        # Stable log sum exp of score excesses.  This is a smooth effective
        # multiplicity statistic, not a hard candidate-count gate.
        log_mass = float(np.log(np.exp(np.clip(sv - top, -60.0, 0.0)).sum()))
        x = np.asarray([top, gap, rms, pos_frac, log_mass], dtype=np.float64)
        runtime_names = [
            "reservation::top_score",
            "reservation::top_gap_to_runnerup_or_incumbent",
            "reservation::score_rms",
            "reservation::positive_score_fraction",
            "reservation::log_effective_competitor_mass",
        ]
    elif mode == "incumbent_basepoint":
        xx = np.asarray(feat, dtype=np.float64)
        sup = np.asarray(support_logits, dtype=np.float64).reshape(-1)
        legacy = int(legacy_action)
        base_names = list(scir_cfg.get("base_feature_names", _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES))
        if base_names != list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES):
            raise ValueError("EAF-ICER V36 basepoint reservation requires frozen 18-D evidence schema")
        pos = {str(n): i for i, n in enumerate(feature_names)}
        missing = [n for n in base_names if n not in pos]
        if missing or not (0 <= legacy < xx.shape[0]) or sup.shape[0] != xx.shape[0]:
            raise ValueError("EAF-ICER V36 basepoint reservation runtime schema is incomplete")
        base = xx[:, [pos[n] for n in base_names]]
        x = np.concatenate([base[legacy], np.asarray([float(sup[legacy])], dtype=np.float64)])
        runtime_names = [f"reservation_incumbent::{n}" for n in base_names] + ["reservation_incumbent::support_logit"]
    else:
        raise ValueError(f"unknown EAF-ICER V36 scene_reservation_feature_mode={mode}")

    stored_names = list(scir_cfg.get("scene_reservation_feature_names", []))
    mean = np.asarray(scir_cfg.get("scene_reservation_feature_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(scir_cfg.get("scene_reservation_feature_std", []), dtype=np.float64).reshape(-1)
    weights = np.asarray(scir_cfg.get("scene_reservation_weights", []), dtype=np.float64).reshape(-1)
    if stored_names != runtime_names or len(runtime_names) != len(mean) or len(runtime_names) != len(std) or len(runtime_names) != len(weights):
        raise ValueError("EAF-ICER V36 reservation feature schema/mean/std/weights are inconsistent")
    z = (x - mean) / np.maximum(std, 1.0e-6)
    raw = float(z @ weights + float(scir_cfg.get("scene_reservation_bias", 0.0)))
    reservation = max(0.0, raw)
    max_res = max(float(scir_cfg.get("scene_reservation_max", 40.0)), 0.0)
    reservation = min(reservation, max_res)
    return float(reservation), x, runtime_names



def _icer_post_selection_value(
    proposal_action: int,
    raw_predicted_improvement: np.ndarray,
    scir_feature_matrix: np.ndarray,
    scir_feature_names: list[str],
    scir_cfg: dict[str, Any],
) -> tuple[float, np.ndarray, list[str]]:
    """Absolute value readout for an already frozen RSMR proposal.

    V37 modes remain supported:
      score_affine, orthogonal_proposal_value.

    V38 adds rank/value factorization modes:
      dense_edge_value:
        a corrected V32.1 scene-equal absolute teacher-improvement ridge is
        trained on *all fit candidate edges* but evaluated only after RSMR has
        frozen one proposal;
      dense_edge_affine:
        the same dense value followed by a one-dimensional selected-policy
        affine recalibration fitted on an independent calibration population.

    Every mode can only accept the exact frozen RSMR proposal or veto it to the
    incumbent.  None can score a second-best action after proposal freezing.
    """
    if not bool(scir_cfg.get("post_selection_value_enabled", False)):
        return 0.0, np.zeros((0,), dtype=np.float64), []
    if bool(scir_cfg.get("scene_reservation_enabled", False)):
        raise ValueError("EAF-ICER post-selection value is mutually exclusive with scene reservation")
    mode = str(scir_cfg.get("post_selection_value_mode", "")).strip().lower()
    allowed = {"score_affine", "orthogonal_proposal_value", "dense_edge_value", "dense_edge_affine"}
    if mode not in allowed:
        raise ValueError(f"unknown EAF-ICER post_selection_value_mode={mode}")
    b = int(proposal_action)
    mu = np.asarray(raw_predicted_improvement, dtype=np.float64).reshape(-1)
    X = np.asarray(scir_feature_matrix, dtype=np.float64)
    if X.ndim != 2 or not (0 <= b < X.shape[0]) or mu.size != X.shape[0]:
        raise ValueError("EAF-ICER proposal/value runtime shapes are inconsistent")
    stored_names = list(scir_cfg.get("feature_names", []))
    if list(scir_feature_names) != stored_names or len(stored_names) != X.shape[1]:
        raise ValueError("EAF-ICER selected-proposal feature schema does not match frozen RSMR")
    mean = np.asarray(scir_cfg.get("feature_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(scir_cfg.get("feature_std", []), dtype=np.float64).reshape(-1)
    rank_w = np.asarray(scir_cfg.get("weights", []), dtype=np.float64).reshape(-1)
    if len(mean) != X.shape[1] or len(std) != X.shape[1] or len(rank_w) != X.shape[1]:
        raise ValueError("EAF-ICER frozen RSMR parameter schema is inconsistent")
    rank_bias = float(scir_cfg.get("bias", 0.0))
    if abs(rank_bias) > 1.0e-12:
        raise ValueError("EAF-ICER post-selection value requires zero-bias frozen RSMR")
    z = (X[b] - mean) / np.maximum(std, 1.0e-6)
    u_linear = float(z @ rank_w)
    u = float(np.clip(u_linear + rank_bias, -40.0, 40.0))
    if not math.isfinite(u) or abs(u - float(mu[b])) > 1.0e-6 * max(1.0, abs(u), abs(float(mu[b]))):
        raise ValueError("EAF-ICER frozen RSMR score does not replay from selected-proposal evidence")

    if mode in {"dense_edge_value", "dense_edge_affine"}:
        dmean = np.asarray(scir_cfg.get("post_selection_dense_feature_mean", []), dtype=np.float64).reshape(-1)
        dstd = np.asarray(scir_cfg.get("post_selection_dense_feature_std", []), dtype=np.float64).reshape(-1)
        dw = np.asarray(scir_cfg.get("post_selection_dense_weights", []), dtype=np.float64).reshape(-1)
        db = float(scir_cfg.get("post_selection_dense_bias", float("nan")))
        if len(dmean) != X.shape[1] or len(dstd) != X.shape[1] or len(dw) != X.shape[1] or not math.isfinite(db):
            raise ValueError("EAF-ICER V38 dense value parameter schema is inconsistent")
        dz = (X[b] - dmean) / np.maximum(dstd, 1.0e-6)
        dense_value = float(np.clip(dz @ dw + db, -40.0, 40.0))
        if mode == "dense_edge_value":
            value = dense_value
            value_feature = np.asarray([dense_value], dtype=np.float64)
            value_names = ["post_value::dense_all_edge_absolute_value"]
        else:
            cm = float(scir_cfg.get("post_selection_dense_cal_mean", float("nan")))
            cs = float(scir_cfg.get("post_selection_dense_cal_std", float("nan")))
            ci = float(scir_cfg.get("post_selection_dense_cal_intercept", float("nan")))
            cw = float(scir_cfg.get("post_selection_dense_cal_weight", float("nan")))
            if not all(math.isfinite(v) for v in [cm, cs, ci, cw]) or cs <= 0.0:
                raise ValueError("EAF-ICER V38 dense selected-policy affine parameters are invalid")
            value = ci + cw * ((dense_value - cm) / max(cs, 1.0e-6))
            value_feature = np.asarray([dense_value, u], dtype=np.float64)
            value_names = ["post_value::dense_all_edge_absolute_value", "post_value::frozen_rsmr_score_diagnostic"]
    else:
        score_mean = float(scir_cfg.get("post_selection_score_mean", float("nan")))
        score_std = float(scir_cfg.get("post_selection_score_std", float("nan")))
        intercept = float(scir_cfg.get("post_selection_affine_intercept", float("nan")))
        score_weight = float(scir_cfg.get("post_selection_affine_score_weight", float("nan")))
        if not all(math.isfinite(v) for v in [score_mean, score_std, intercept, score_weight]) or score_std <= 0.0:
            raise ValueError("EAF-ICER V37 affine post-selection value parameters are invalid")
        value = intercept + score_weight * ((u - score_mean) / max(score_std, 1.0e-6))
        value_feature = np.asarray([u], dtype=np.float64)
        value_names = ["post_value::frozen_rsmr_score"]

        if mode == "orthogonal_proposal_value":
            ww = float(rank_w @ rank_w)
            if ww <= 1.0e-12:
                raise ValueError("EAF-ICER V37 frozen RSMR score direction has zero norm")
            zperp = z - rank_w * (u_linear / ww)
            if abs(float(zperp @ rank_w)) > 1.0e-8 * max(1.0, float(np.linalg.norm(zperp)), float(np.linalg.norm(rank_w))):
                raise ValueError("EAF-ICER V37 orthogonal proposal feature lost score orthogonality")
            rmean = np.asarray(scir_cfg.get("post_selection_residual_feature_mean", []), dtype=np.float64).reshape(-1)
            rstd = np.asarray(scir_cfg.get("post_selection_residual_feature_std", []), dtype=np.float64).reshape(-1)
            rw = np.asarray(scir_cfg.get("post_selection_residual_weights", []), dtype=np.float64).reshape(-1)
            if len(rmean) != X.shape[1] or len(rstd) != X.shape[1] or len(rw) != X.shape[1]:
                raise ValueError("EAF-ICER V37 orthogonal residual schema is inconsistent")
            rz = (zperp - rmean) / np.maximum(rstd, 1.0e-6)
            value += float(rz @ rw + float(scir_cfg.get("post_selection_residual_bias", 0.0)))
            value_feature = np.concatenate([np.asarray([u], dtype=np.float64), zperp])
            value_names = ["post_value::frozen_rsmr_score"] + [f"post_value_orthogonal::{n}" for n in stored_names]
    if not math.isfinite(value):
        raise ValueError("EAF-ICER post-selection value is non-finite")
    max_abs = max(float(scir_cfg.get("post_selection_value_max_abs", 40.0)), 1.0e-6)
    value = float(np.clip(value, -max_abs, max_abs))
    return value, value_feature, value_names

def _icer_select_scir_candidate(
    candidate_indices: np.ndarray | list[int],
    predicted_improvement: np.ndarray,
    support_logits: np.ndarray,
    margin_star: np.ndarray,
    utility_prior: np.ndarray,
) -> int | None:
    """Deterministic extremal proposal for V64.3.31 SCIR."""
    cand = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if cand.size == 0:
        return None
    best = sorted(
        cand.tolist(),
        key=lambda b: (
            -float(predicted_improvement[b]),
            -float(support_logits[b]),
            -float(margin_star[b]),
            -int(utility_prior[b]),
            int(b),
        ),
    )[0]
    return int(best)


def _apply_decisive_frontier_icer(
    legacy_action: int,
    anchor_action: int,
    margin_matrix: np.ndarray,
    attribution_star: np.ndarray | None,
    atom_contrib_star: np.ndarray | None,
    scores: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    potential_diag: dict[str, Any],
    evidence_certificate_fraction: float | None,
    utility_refinement_diag: dict[str, Any] | None,
    cfg: dict[str, Any],
    candidate_trajectories: np.ndarray | None = None,
    maneuver_ids: np.ndarray | None = None,
    selected_atom_family_ids: np.ndarray | list[int] | None = None,
    selected_atom_type_names: list[str] | np.ndarray | None = None,
) -> tuple[int, dict[str, Any]]:
    """V64.3.19+ incumbent-contrastive extremal recovery.

    V64.3.18 showed that one shared pointwise score can be a strong anchor-support
    reliability signal yet still replace a good incumbent with an inferior
    alternative.  ICER therefore decomposes the decision into two train-only
    readouts over the *same* frozen guard-admissible frontier and selected evidence:

    1) scalar anchor-support score: is a challenger supported over the DARM anchor?
    2) incumbent-contrastive score: should an alternative replace the frozen
       legacy incumbent under the same evidence interface?

    No alternative can replace a supported admissible incumbent unless both scores
    are positive.  Zero is the fixed pseudo-item logit for anchor/incumbent; there
    is no validation threshold sweep.  The unchanged final one-sided/evidence and
    structural-risk guards still execute after this operator.
    """
    runtime_cfg = cfg.get("runtime", {}) or {}
    frontier_cfg = runtime_cfg.get("decisive_frontier_value", {}) or {}
    icer_cfg = frontier_cfg.get("incumbent_contrastive_extremal_recovery", {}) or {}
    enabled = bool(icer_cfg.get("enabled", False))
    instrument = bool(icer_cfg.get("instrument_features", True))
    frontier_active = bool(float(potential_diag.get("decisive_frontier_value_active", 0.0)) >= 0.5)
    M = np.asarray(margin_matrix, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    a = int(anchor_action); legacy = int(legacy_action)
    margin_star = M[:, a].copy() if 0 <= a < M.shape[1] else np.zeros((len(valid),), dtype=np.float64)
    admissible = _decisive_frontier_guard_admissible_mask(
        margin_star, scores, valid, runtime_safety_flags, a,
        evidence_certificate_fraction, cfg,
        require_safe_available_for_learned_intervention=bool(
            icer_cfg.get("require_safe_available_for_learned_intervention", True)
        ),
    )
    feat, names, utility_prior = _decisive_frontier_dacer_features(
        margin_star, attribution_star, atom_contrib_star, scores, valid, a, legacy,
        potential_diag, evidence_certificate_fraction, utility_refinement_diag,
        admissible, cfg,
    )
    attribution_resolved, attribution_resolved_names = _icer_attribution_resolved_feature_matrix(
        atom_contrib_star, valid, a, legacy,
        budget=_icer_runtime_attribution_spectrum_budget(cfg),
    )
    semantic_family, semantic_family_names = _icer_semantic_family_feature_matrix(
        atom_contrib_star, selected_atom_family_ids, valid, a, legacy
    )
    semantic_type, semantic_type_names = _icer_semantic_type_feature_matrix(
        atom_contrib_star, selected_atom_type_names, valid, a, legacy
    )
    transition_vs_incumbent, transition_names = _icer_transition_feature_matrix(
        candidate_trajectories, maneuver_ids, valid, legacy
    )
    transition_vs_anchor, transition_anchor_names = _icer_transition_feature_matrix(
        candidate_trajectories, maneuver_ids, valid, a
    )
    if transition_names != transition_anchor_names:
        raise RuntimeError("ICER transition schemas differ between incumbent/anchor reference views")
    support_logits = np.zeros((len(valid),), dtype=np.float64)
    dominance_logits = np.zeros((len(valid),), dtype=np.float64)
    # Always initialize the component-head diagnostics.  V64.3.20 deliberately
    # skips both learned dominance heads in the all-flagged structural domain,
    # but diagnostic serialization still exposes the arrays for schema stability.
    scalar_dominance_logits = np.zeros((len(valid),), dtype=np.float64)
    profile_dominance_logits = np.zeros((len(valid),), dtype=np.float64)
    incumbent_retention_margin = np.zeros((len(valid),), dtype=np.float64)
    replacement_regret_risk_logits = np.zeros((len(valid),), dtype=np.float64)
    replacement_confirmation_regret_risk_logits = np.zeros((len(valid),), dtype=np.float64)
    retention_regret_risk_logits = np.zeros((len(valid),), dtype=np.float64)
    scir_predicted_improvement = np.zeros((len(valid),), dtype=np.float64)
    scir_raw_predicted_improvement = np.zeros((len(valid),), dtype=np.float64)
    scir_scene_reservation = 0.0
    scir_scene_reservation_feature = np.zeros((0,), dtype=np.float64)
    scir_scene_reservation_feature_names: list[str] = []
    scir_post_selection_value = 0.0
    scir_post_selection_value_feature = np.zeros((0,), dtype=np.float64)
    scir_post_selection_value_feature_names: list[str] = []
    scir_selection_scale = np.ones((len(valid),), dtype=np.float64)
    scir_lower_bound = np.zeros((len(valid),), dtype=np.float64)
    scir_feature_matrix = np.zeros((len(valid), 0), dtype=np.float64)
    scir_feature_names: list[str] = []
    scir_proposal_action = int(legacy)
    scir_proposal_exists = False
    scir_certificate_accepted = False
    scir_cfg = icer_cfg.get("selection_conditioned_intervention_recovery", {}) or {}
    scir_enabled = bool(scir_cfg.get("enabled", False))
    scir_mode = str(scir_cfg.get("mode", "rank_only")).strip().lower()
    selected = legacy
    baseline = legacy
    flags = np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)
    if flags.shape[0] < len(valid):
        flags = np.pad(flags, (0, len(valid) - flags.shape[0]), constant_values=False)
    flags = flags[: len(valid)]
    safe_available = bool(np.any(valid & ~flags))
    all_flagged_domain = bool(valid.any() and not safe_available)
    all_flagged_policy = str(icer_cfg.get("all_flagged_policy", "anchor_abstention")).strip().lower()
    structural_domain_delegated = bool(all_flagged_domain and all_flagged_policy == "preserve_legacy_for_structural_guard")
    applies = bool(enabled and frontier_active and 0 <= a < len(valid))
    if applies and structural_domain_delegated:
        # V64.3.20 deployment-complete delegation.  In an all-flagged bank the
        # learned ICER frontier is intentionally empty, but choosing the DARM
        # anchor is *not* a neutral abstention: it changes the proposal entering
        # the frozen continuous structural-risk guard and can therefore change
        # the final deployed action.  Preserve the frozen raw-EAF proposal
        # exactly and delegate the whole scene to the unchanged one-sided /
        # evidence / structural-risk stack.  No learned support/dominance score
        # is consumed in this domain.
        selected = int(legacy)
        baseline = int(legacy)
    elif applies:
        support_names = list(icer_cfg.get("support_feature_names", []))
        support_mean = np.asarray(icer_cfg.get("support_feature_mean", []), dtype=np.float64).reshape(-1)
        support_std = np.asarray(icer_cfg.get("support_feature_std", []), dtype=np.float64).reshape(-1)
        support_weights = np.asarray(icer_cfg.get("support_weights", []), dtype=np.float64).reshape(-1)
        expected_support_names = list(_DALER_FEATURE_NAMES)
        if (
            support_names != expected_support_names
            or len(support_names) != len(support_mean)
            or len(support_names) != len(support_std)
            or len(support_names) != len(support_weights)
        ):
            raise ValueError("EAF-ICER support head schema/mean/std/weights are inconsistent")
        sx = feat[:, : len(expected_support_names)]
        sz = (sx - support_mean[None, :]) / np.maximum(support_std[None, :], 1.0e-6)
        support_logits = np.clip(sz @ support_weights + float(icer_cfg.get("support_bias", 0.0)), -40.0, 40.0)

        dominance_policy = str(icer_cfg.get("dominance_policy", "dual_equal_mean"))

        def _read_dom_head(prefix: str, mode: str) -> np.ndarray:
            dom_x, runtime_names, runtime_base_names = _icer_quadratic_interaction_features(feat, names, mode)
            stored_names = list(icer_cfg.get(f"{prefix}_feature_names", []))
            stored_base_names = list(icer_cfg.get(f"{prefix}_base_feature_names", []))
            mean = np.asarray(icer_cfg.get(f"{prefix}_feature_mean", []), dtype=np.float64).reshape(-1)
            std = np.asarray(icer_cfg.get(f"{prefix}_feature_std", []), dtype=np.float64).reshape(-1)
            weights = np.asarray(icer_cfg.get(f"{prefix}_weights", []), dtype=np.float64).reshape(-1)
            if (
                stored_names != runtime_names
                or stored_base_names != runtime_base_names
                or len(runtime_names) != len(mean)
                or len(runtime_names) != len(std)
                or len(runtime_names) != len(weights)
            ):
                raise ValueError(f"EAF-ICER {prefix} interaction schema/mean/std/weights are inconsistent")
            z = (dom_x - mean[None, :]) / np.maximum(std[None, :], 1.0e-6)
            return np.clip(z @ weights + float(icer_cfg.get(f"{prefix}_bias", 0.0)), -40.0, 40.0)

        scalar_dominance_logits = _read_dom_head("scalar_dominance", "scalar_interaction")
        if dominance_policy == "scalar_only":
            profile_dominance_logits = np.zeros_like(scalar_dominance_logits)
            dominance_logits = scalar_dominance_logits.copy()
            dominance_positive = np.isfinite(scalar_dominance_logits) & (scalar_dominance_logits > 0.0)
        elif dominance_policy == "dual_equal_mean":
            profile_dominance_logits = _read_dom_head("profile_dominance", "profile_interaction")
            dominance_logits = 0.5 * (scalar_dominance_logits + profile_dominance_logits)
            dominance_positive = np.isfinite(dominance_logits) & (dominance_logits > 0.0)
        elif dominance_policy == "scalar_positive_dual_equal_mean":
            # V64.3.22 clean signed-attribution ablation: scalar dominance keeps
            # the incumbent-replacement eligibility boundary fixed, while the
            # exact signed selected-evidence profile is allowed to change only
            # the extremal ordering score.  This avoids the V64.3.21 consensus
            # failure mode and prevents the scalar arm from being contaminated
            # by profile-dependent TRAIN sample selection.
            profile_dominance_logits = _read_dom_head("profile_dominance", "profile_interaction")
            dominance_logits = 0.5 * (scalar_dominance_logits + profile_dominance_logits)
            dominance_positive = np.isfinite(scalar_dominance_logits) & (scalar_dominance_logits > 0.0)
        elif dominance_policy == "scalar_positive_dual_mean_positive":
            # V64.3.23 RCR: the signed-profile view remains a *combined* view,
            # not the V64.3.21 hard per-view consensus.  However, if the exact
            # evidence-attributed score that will be used to rank alternatives
            # is itself non-positive, it is semantically inconsistent to let it
            # trigger an incumbent replacement merely because the scalar view
            # and regret-risk head are positive.  Eligibility therefore requires
            # scalar dominance > 0 AND equal-mean scalar/profile dominance > 0.
            # No validation threshold or view-weight tuning is introduced.
            profile_dominance_logits = _read_dom_head("profile_dominance", "profile_interaction")
            dominance_logits = 0.5 * (scalar_dominance_logits + profile_dominance_logits)
            dominance_positive = (
                np.isfinite(scalar_dominance_logits)
                & np.isfinite(dominance_logits)
                & (scalar_dominance_logits > 0.0)
                & (dominance_logits > 0.0)
            )
        elif dominance_policy == "dual_positive_consensus_mean":
            # V64.3.21: a signed-attribution view is corroborating evidence, not
            # a compensating term.  Both independently trained TRAIN-only views
            # must cross their semantic zero-log-odds boundary; only then is
            # their equal mean used for extremal ranking.  This is a structural
            # consensus rule, not a validation-tuned threshold.
            profile_dominance_logits = _read_dom_head("profile_dominance", "profile_interaction")
            dominance_logits = 0.5 * (scalar_dominance_logits + profile_dominance_logits)
            dominance_positive = (
                np.isfinite(scalar_dominance_logits) & np.isfinite(profile_dominance_logits)
                & (scalar_dominance_logits > 0.0) & (profile_dominance_logits > 0.0)
            )
        else:
            raise ValueError(f"unknown EAF-ICER dominance_policy={dominance_policy}")

        if scir_enabled:
            if scir_mode not in {"rank_only", "mean_rank", "conformal_veto", "simultaneous_lcb"}:
                raise ValueError(f"unknown EAF-ICER SCIR mode={scir_mode}")
            scir_predicted_improvement, scir_feature_matrix, scir_feature_names, scir_selection_scale = _icer_selection_conditioned_intervention_scores(
                feat, names, support_logits, legacy, scir_cfg
            )
            scir_raw_predicted_improvement = scir_predicted_improvement.copy()
            if scir_mode == "conformal_veto":
                q = float(scir_cfg.get("conformal_overprediction_quantile", 0.0))
                scir_lower_bound = scir_predicted_improvement - q
            elif scir_mode == "simultaneous_lcb":
                q = float(scir_cfg.get("simultaneous_conformal_quantile", 0.0))
                scir_lower_bound = scir_predicted_improvement - q * scir_selection_scale
            else:
                q = 0.0
                scir_lower_bound = scir_predicted_improvement.copy()
            if not np.isfinite(q) or q < 0.0:
                raise ValueError("EAF-ICER SCIR conformal quantile must be finite and non-negative")

        # V64.3.21 selection-conditioned incumbent retention.  The historical
        # all-edge support head is still the absolute support gate for *new*
        # alternatives, but it is no longer asked to veto an already admissible
        # extremal incumbent.  A dedicated TRAIN-only linear margin readout is
        # fit only on frozen raw-EAF incumbents and predicts the normalized
        # teacher margin J(anchor)-J(incumbent).  Zero is therefore a semantic
        # preserve-vs-anchor boundary and requires no validation threshold.
        retention_policy = str(icer_cfg.get("incumbent_retention_policy", "generic_support")).strip().lower()
        if retention_policy in {"selected_incumbent_scalar_margin_mse", "selected_incumbent_profile_margin_mse"}:
            stored_base = list(icer_cfg.get("retention_feature_names", []))
            expected_base = list(_ICER_DOMINANCE_PROFILE_BASE_NAMES[:18] if retention_policy == "selected_incumbent_scalar_margin_mse" else _ICER_DOMINANCE_PROFILE_BASE_NAMES)
            rmean = np.asarray(icer_cfg.get("retention_feature_mean", []), dtype=np.float64).reshape(-1)
            rstd = np.asarray(icer_cfg.get("retention_feature_std", []), dtype=np.float64).reshape(-1)
            rw = np.asarray(icer_cfg.get("retention_weights", []), dtype=np.float64).reshape(-1)
            if stored_base != expected_base or len(rmean) != len(expected_base) or len(rstd) != len(expected_base) or len(rw) != len(expected_base):
                raise ValueError("EAF-ICER incumbent retention margin schema/mean/std/weights are inconsistent")
            fpos = {str(n): i for i, n in enumerate(names)}
            if any(n not in fpos for n in expected_base):
                raise ValueError("EAF-ICER incumbent retention profile features missing from runtime schema")
            rx = feat[:, [fpos[n] for n in expected_base]]
            rz = (rx - rmean[None, :]) / np.maximum(rstd[None, :], 1.0e-6)
            incumbent_retention_margin = np.clip(rz @ rw + float(icer_cfg.get("retention_bias", 0.0)), -40.0, 40.0)
        elif retention_policy == "preserve_admissible_incumbent":
            incumbent_retention_margin = np.ones((len(valid),), dtype=np.float64)
        elif retention_policy == "generic_support":
            incumbent_retention_margin = support_logits.copy()
        else:
            raise ValueError(f"unknown EAF-ICER incumbent_retention_policy={retention_policy}")

        # V64.3.22 transition-conditioned regret-risk.  Binary support and
        # dominance remain the frozen reliability views; this additional TRAIN-only
        # head is cost-sensitive to teacher-improvement magnitude and therefore
        # acts as a zero-boundary *risk veto*, not a tuned probability threshold.
        regret_risk_enabled = bool(icer_cfg.get("regret_risk_enabled", False))
        regret_risk_mode = str(icer_cfg.get("regret_risk_feature_mode", "evidence_only")).strip().lower()
        # V64.3.23 separates the two asymmetric action-change risks.  Historical
        # V22 configs omit these flags and therefore keep the old both-heads
        # behavior.  RCR disables learned incumbent->anchor veto while retaining
        # the TRAIN-only replacement regret-risk head.
        retention_regret_risk_enabled = bool(icer_cfg.get("retention_regret_risk_enabled", regret_risk_enabled))
        replacement_regret_risk_enabled = bool(icer_cfg.get("replacement_regret_risk_enabled", regret_risk_enabled))
        if regret_risk_enabled:
            geometry_available = False
            if candidate_trajectories is not None:
                _tr_chk = np.asarray(candidate_trajectories)
                geometry_available = bool(_tr_chk.ndim == 3 and _tr_chk.shape[0] >= len(valid) and _tr_chk.shape[1] >= 1 and _tr_chk.shape[2] >= 2)
            if regret_risk_mode == "transition_conditioned" and not geometry_available:
                raise ValueError("EAF-ICER transition-conditioned regret risk requires the runtime candidate trajectory bank")
            if regret_risk_mode == "semantic_family_aligned":
                fam_runtime = np.asarray(selected_atom_family_ids if selected_atom_family_ids is not None else [], dtype=np.int64).reshape(-1)
                contrib_rows = int(np.asarray(atom_contrib_star).shape[0]) if atom_contrib_star is not None and np.asarray(atom_contrib_star).ndim == 2 else 0
                if selected_atom_family_ids is None or fam_runtime.size != contrib_rows:
                    raise ValueError("EAF-ICER semantic-family regret risk requires exact selected-atom family ids aligned to selected attribution rows")
            rep_x, rep_names = _icer_regret_risk_feature_matrix(
                feat, names, transition_vs_incumbent, transition_names, regret_risk_mode,
                attribution_resolved, attribution_resolved_names,
                semantic_family, semantic_family_names,
                semantic_type, semantic_type_names,
            )
            ret_x, ret_names = _icer_regret_risk_feature_matrix(
                feat, names, transition_vs_anchor, transition_names, regret_risk_mode,
                attribution_resolved, attribution_resolved_names,
                semantic_family, semantic_family_names,
                semantic_type, semantic_type_names,
            )
            if rep_names != ret_names:
                raise RuntimeError("EAF-ICER replacement/retention regret-risk schemas diverged")

            def _read_regret_risk_head(prefix: str, x: np.ndarray, runtime_names: list[str]) -> np.ndarray:
                stored_names = list(icer_cfg.get(f"{prefix}_feature_names", []))
                mean = np.asarray(icer_cfg.get(f"{prefix}_feature_mean", []), dtype=np.float64).reshape(-1)
                std = np.asarray(icer_cfg.get(f"{prefix}_feature_std", []), dtype=np.float64).reshape(-1)
                weights = np.asarray(icer_cfg.get(f"{prefix}_weights", []), dtype=np.float64).reshape(-1)
                if stored_names != runtime_names or len(runtime_names) != len(mean) or len(runtime_names) != len(std) or len(runtime_names) != len(weights):
                    raise ValueError(f"EAF-ICER {prefix} regret-risk schema/mean/std/weights are inconsistent")
                z = (x - mean[None, :]) / np.maximum(std[None, :], 1.0e-6)
                return np.clip(z @ weights + float(icer_cfg.get(f"{prefix}_bias", 0.0)), -40.0, 40.0)

            risk_model_type = str(icer_cfg.get("regret_risk_model_type", "linear_weighted_logistic")).strip().lower()
            if replacement_regret_risk_enabled:
                if risk_model_type in {"local_multiscale_regret_lower_bound", "local_multiscale_downside_regret_certificate", "local_multiscale_downside_regret_with_type_confirmation", "local_multiscale_downside_regret_with_global_type_tail_confirmation"}:
                    replacement_regret_risk_logits = _icer_local_regret_lower_bound(
                        rep_x,
                        rep_names,
                        str(icer_cfg.get("replacement_local_regret_memory_path", "")),
                        str(icer_cfg.get("replacement_local_regret_memory_sha256", "")),
                    )
                    if risk_model_type in {"local_multiscale_downside_regret_with_type_confirmation", "local_multiscale_downside_regret_with_global_type_tail_confirmation"}:
                        if selected_atom_type_names is None:
                            raise ValueError("EAF-ICER type confirmation requires exact selected atom type names")
                        type_x, type_names = _icer_regret_risk_feature_matrix(
                            feat, names, transition_vs_incumbent, transition_names, "semantic_type_only",
                            attribution_resolved, attribution_resolved_names,
                            semantic_family, semantic_family_names,
                            semantic_type, semantic_type_names,
                        )
                        if risk_model_type == "local_multiscale_downside_regret_with_type_confirmation":
                            replacement_confirmation_regret_risk_logits = _icer_local_regret_lower_bound(
                                type_x,
                                type_names,
                                str(icer_cfg.get("replacement_confirmation_local_regret_memory_path", "")),
                                str(icer_cfg.get("replacement_confirmation_local_regret_memory_sha256", "")),
                            )
                        else:
                            replacement_confirmation_regret_risk_logits = _icer_global_tail_mode_confirmation_score(
                                type_x,
                                type_names,
                                str(icer_cfg.get("replacement_confirmation_tail_mode_model_path", "")),
                                str(icer_cfg.get("replacement_confirmation_tail_mode_model_sha256", "")),
                            )
                elif risk_model_type == "linear_weighted_logistic":
                    replacement_regret_risk_logits = _read_regret_risk_head("replacement_regret_risk", rep_x, rep_names)
                else:
                    raise ValueError(f"unknown EAF-ICER regret_risk_model_type={risk_model_type}")
            if retention_regret_risk_enabled:
                if risk_model_type != "linear_weighted_logistic":
                    raise ValueError("local regret memory is replacement-only; learned retention veto must remain disabled")
                retention_regret_risk_logits = _read_regret_risk_head("retention_regret_risk", ret_x, ret_names)

        support_ok = admissible & np.isfinite(support_logits) & (support_logits > 0.0)
        legacy_admissible = bool(0 <= legacy < len(valid) and admissible[legacy])
        legacy_supported = bool(legacy_admissible and support_ok[legacy])
        if regret_risk_enabled and retention_regret_risk_enabled:
            legacy_retained = bool(
                legacy_admissible
                and np.isfinite(retention_regret_risk_logits[legacy])
                and retention_regret_risk_logits[legacy] >= 0.0
            )
        else:
            legacy_retained = bool(
                legacy_admissible
                and np.isfinite(incumbent_retention_margin[legacy])
                and incumbent_retention_margin[legacy] >= 0.0
            )
        # When the frozen incumbent itself is deployment-admissible, every
        # alternative replacement remains incumbent-contrastive.  V64.3.21
        # makes the baseline asymmetric: only the selected-incumbent margin head
        # may demote an admissible incumbent to the anchor; generic edge support
        # continues to gate alternatives but cannot veto the incumbent itself.
        if legacy_admissible:
            baseline = int(legacy if legacy_retained else a)
            alternative = support_ok.copy()
            if 0 <= legacy < len(alternative):
                alternative[legacy] = False
            replacement_positive = np.ones((len(valid),), dtype=bool)
            if regret_risk_enabled and replacement_regret_risk_enabled:
                replacement_positive = np.isfinite(replacement_regret_risk_logits) & (replacement_regret_risk_logits > 0.0)
            if scir_enabled:
                reservation_enabled = bool(scir_cfg.get("scene_reservation_enabled", False))
                post_value_enabled = bool(scir_cfg.get("post_selection_value_enabled", False))
                if reservation_enabled and post_value_enabled:
                    raise ValueError("EAF-ICER cannot enable V36 scene reservation and V37 post-selection value simultaneously")
                if post_value_enabled:
                    # V64.3.37: freeze the exact RSMR winner first.  The value
                    # readout is evaluated only for that winner and may either
                    # keep it or veto it to the incumbent.  It never re-ranks.
                    raw_positive = np.isfinite(scir_raw_predicted_improvement) & (scir_raw_predicted_improvement > 0.0)
                    raw_cand = np.flatnonzero(alternative & raw_positive).astype(np.int64)
                    best = _icer_select_scir_candidate(
                        raw_cand, scir_raw_predicted_improvement, support_logits, margin_star, utility_prior
                    )
                    scir_proposal_exists = bool(best is not None)
                    scir_proposal_action = int(legacy if best is None else best)
                    scir_predicted_improvement = scir_raw_predicted_improvement.copy()
                    scir_lower_bound = scir_predicted_improvement.copy()
                    if best is None:
                        scir_certificate_accepted = False
                        selected = int(baseline)
                    else:
                        scir_post_selection_value, scir_post_selection_value_feature, scir_post_selection_value_feature_names = _icer_post_selection_value(
                            best, scir_raw_predicted_improvement, scir_feature_matrix, scir_feature_names, scir_cfg
                        )
                        # For diagnostics, replace only the frozen proposal's
                        # absolute intervention value.  All non-proposal scores
                        # remain raw RSMR ranking scores and are never consulted
                        # after proposal freezing.
                        scir_predicted_improvement[best] = scir_post_selection_value
                        scir_lower_bound[best] = scir_post_selection_value
                        scir_certificate_accepted = bool(scir_post_selection_value > 0.0)
                        selected = int(best if scir_certificate_accepted else baseline)
                elif reservation_enabled:
                    # V64.3.36: proposal identity is frozen *before* reservation.
                    # The reservation is a non-negative, scene-common subtraction
                    # and can only accept that exact RSMR proposal or return the
                    # incumbent.  It never re-ranks or falls through.
                    reservation_candidates = np.flatnonzero(alternative).astype(np.int64)
                    raw_positive = np.isfinite(scir_raw_predicted_improvement) & (scir_raw_predicted_improvement > 0.0)
                    raw_cand = np.flatnonzero(alternative & raw_positive).astype(np.int64)
                    best = _icer_select_scir_candidate(
                        raw_cand, scir_raw_predicted_improvement, support_logits, margin_star, utility_prior
                    )
                    scir_proposal_exists = bool(best is not None)
                    scir_proposal_action = int(legacy if best is None else best)
                    scir_scene_reservation, scir_scene_reservation_feature, scir_scene_reservation_feature_names = _icer_scene_reservation_value(
                        scir_raw_predicted_improvement, reservation_candidates, feat, names, support_logits, legacy, scir_cfg
                    )
                    scir_predicted_improvement = scir_raw_predicted_improvement.copy()
                    if reservation_candidates.size:
                        scir_predicted_improvement[reservation_candidates] = (
                            scir_raw_predicted_improvement[reservation_candidates] - scir_scene_reservation
                        )
                    if 0 <= legacy < len(scir_predicted_improvement):
                        scir_predicted_improvement[legacy] = 0.0
                    scir_lower_bound = scir_predicted_improvement.copy()
                    if best is None:
                        scir_certificate_accepted = False
                        selected = int(baseline)
                    else:
                        scir_certificate_accepted = bool(
                            np.isfinite(scir_predicted_improvement[best]) and scir_predicted_improvement[best] > 0.0
                        )
                        selected = int(best if scir_certificate_accepted else baseline)
                else:
                    # V64.3.31: the direct intervention score itself is the semantic
                    # candidate-vs-incumbent utility.  The old binary dominance head
                    # remains instrumented as a causal diagnostic but no longer gates
                    # or ranks the SCIR arm.
                    if scir_mode == "simultaneous_lcb":
                        scir_positive = np.isfinite(scir_lower_bound) & (scir_lower_bound > 0.0)
                        ordering_score = scir_lower_bound
                    else:
                        scir_positive = np.isfinite(scir_predicted_improvement) & (scir_predicted_improvement > 0.0)
                        ordering_score = scir_predicted_improvement
                    cand = np.flatnonzero(alternative & scir_positive).astype(np.int64)
                    best = _icer_select_scir_candidate(
                        cand, ordering_score, support_logits, margin_star, utility_prior
                    )
                    scir_proposal_exists = bool(best is not None)
                    scir_proposal_action = int(legacy if best is None else best)
                    if best is None:
                        scir_certificate_accepted = False
                        selected = int(baseline)
                    elif scir_mode == "conformal_veto":
                        scir_certificate_accepted = bool(np.isfinite(scir_lower_bound[best]) and scir_lower_bound[best] > 0.0)
                        selected = int(best if scir_certificate_accepted else baseline)
                    else:
                        scir_certificate_accepted = True
                        selected = int(best)
            else:
                cand = np.flatnonzero(alternative & dominance_positive & replacement_positive).astype(np.int64)
                confirmation_logits = None
                if (
                    regret_risk_enabled
                    and replacement_regret_risk_enabled
                    and risk_model_type in {"local_multiscale_downside_regret_with_type_confirmation", "local_multiscale_downside_regret_with_global_type_tail_confirmation"}
                ):
                    confirmation_logits = replacement_confirmation_regret_risk_logits
                # V64.3.27 monotone candidate confirmation: the aggregate-DRC
                # proposal is the only alternative the semantic view may inspect.
                # A failed confirmation preserves the incumbent; it never falls
                # through to a lower-ranked alternative.
                best = _icer_select_extremal_candidate_with_optional_confirmation(
                    cand,
                    dominance_logits,
                    replacement_regret_risk_logits if (regret_risk_enabled and replacement_regret_risk_enabled) else np.zeros_like(dominance_logits),
                    support_logits,
                    margin_star,
                    utility_prior,
                    confirmation_logits=confirmation_logits,
                )
                selected = int(baseline if best is None else best)
        else:
            # If raw top is not deployment-admissible, the actual deployment
            # incumbent is the anchor.  The learned recovery is therefore
            # anchor-relative and must not be forced to beat an action that the
            # frozen final guard would reject anyway.
            baseline = int(a)
            cand = np.flatnonzero(support_ok).astype(np.int64)
            if cand.size:
                best = sorted(
                    cand.tolist(),
                    key=lambda b: (-float(support_logits[b]), -float(margin_star[b]), -int(utility_prior[b]), int(b)),
                )[0]
                selected = int(best)
            else:
                selected = int(a)
    support_selected = float(support_logits[selected]) if 0 <= selected < len(valid) and selected != a else 0.0
    dominance_selected = float(dominance_logits[selected]) if 0 <= selected < len(valid) and selected not in {a, legacy} else 0.0
    diag: dict[str, Any] = {
        "decisive_frontier_icer_enabled": float(enabled),
        "decisive_frontier_icer_instrument_features": float(instrument),
        "decisive_frontier_icer_active": float(applies),
        "decisive_frontier_icer_legacy_selected_action": float(legacy),
        "decisive_frontier_icer_baseline_action": float(baseline),
        "decisive_frontier_icer_selected_action": float(selected),
        "decisive_frontier_icer_proposal_changed": float(int(selected) != int(legacy)),
        "decisive_frontier_icer_anchor_fallback": float(int(selected) == int(a)),
        "decisive_frontier_icer_selected_support_logit": support_selected,
        "decisive_frontier_icer_selected_dominance_logit": dominance_selected,
        "decisive_frontier_icer_scir_enabled": float(scir_enabled),
        "decisive_frontier_icer_scir_rank_only": float(scir_enabled and scir_mode in {"rank_only", "mean_rank"}),
        "decisive_frontier_icer_scir_mean_rank": float(scir_enabled and scir_mode in {"rank_only", "mean_rank"}),
        "decisive_frontier_icer_scir_conformal_veto": float(scir_enabled and scir_mode == "conformal_veto"),
        "decisive_frontier_icer_scir_simultaneous_lcb": float(scir_enabled and scir_mode == "simultaneous_lcb"),
        "decisive_frontier_icer_scir_proposal_exists": float(bool(scir_proposal_exists)),
        "decisive_frontier_icer_scir_proposal_action": float(scir_proposal_action),
        "decisive_frontier_icer_scir_certificate_accepted": float(bool(scir_certificate_accepted)),
        "decisive_frontier_icer_scir_selected_predicted_improvement": float(scir_predicted_improvement[selected]) if scir_enabled and 0 <= selected < len(valid) and selected != legacy else 0.0,
        "decisive_frontier_icer_scir_scene_reservation_enabled": float(bool(scir_enabled and scir_cfg.get("scene_reservation_enabled", False))),
        "decisive_frontier_icer_scir_scene_reservation_value": float(scir_scene_reservation) if scir_enabled else 0.0,
        "decisive_frontier_icer_scir_post_selection_value_enabled": float(bool(scir_enabled and scir_cfg.get("post_selection_value_enabled", False))),
        "decisive_frontier_icer_scir_post_selection_value": float(scir_post_selection_value) if scir_enabled else 0.0,
        "decisive_frontier_icer_scir_proposal_predicted_improvement": float(scir_predicted_improvement[scir_proposal_action]) if scir_enabled and scir_proposal_exists and 0 <= scir_proposal_action < len(valid) else 0.0,
        "decisive_frontier_icer_scir_proposal_lower_bound": float(scir_lower_bound[scir_proposal_action]) if scir_enabled and scir_proposal_exists and 0 <= scir_proposal_action < len(valid) else 0.0,
        "decisive_frontier_icer_scir_conformal_alpha": float(scir_cfg.get("conformal_alpha", 0.0)) if scir_enabled else 0.0,
        "decisive_frontier_icer_scir_conformal_overprediction_quantile": float(scir_cfg.get("conformal_overprediction_quantile", 0.0)) if scir_enabled and scir_mode == "conformal_veto" else 0.0,
        "decisive_frontier_icer_scir_simultaneous_conformal_quantile": float(scir_cfg.get("simultaneous_conformal_quantile", 0.0)) if scir_enabled and scir_mode == "simultaneous_lcb" else 0.0,
        "decisive_frontier_icer_scir_selected_scale": float(scir_selection_scale[selected]) if scir_enabled and 0 <= selected < len(valid) and selected != legacy else 1.0,
        "decisive_frontier_icer_legacy_support_logit": float(support_logits[legacy]) if 0 <= legacy < len(valid) and legacy != a else 0.0,
        "decisive_frontier_icer_legacy_retention_margin": float(incumbent_retention_margin[legacy]) if 0 <= legacy < len(valid) and legacy != a else 0.0,
        "decisive_frontier_icer_legacy_admissible": float(bool(0 <= legacy < len(valid) and admissible[legacy])),
        "decisive_frontier_icer_dominance_policy_dual_equal_mean": float(str(icer_cfg.get("dominance_policy", "dual_equal_mean")) == "dual_equal_mean"),
        "decisive_frontier_icer_dominance_policy_scalar_positive_dual_equal_mean": float(str(icer_cfg.get("dominance_policy", "dual_equal_mean")) == "scalar_positive_dual_equal_mean"),
        "decisive_frontier_icer_dominance_policy_scalar_positive_dual_mean_positive": float(str(icer_cfg.get("dominance_policy", "dual_equal_mean")) == "scalar_positive_dual_mean_positive"),
        "decisive_frontier_icer_dominance_policy_dual_positive_consensus": float(str(icer_cfg.get("dominance_policy", "dual_equal_mean")) == "dual_positive_consensus_mean"),
        "decisive_frontier_icer_incumbent_retention_profile_margin": float(str(icer_cfg.get("incumbent_retention_policy", "generic_support")) == "selected_incumbent_profile_margin_mse"),
        "decisive_frontier_icer_incumbent_retention_scalar_margin": float(str(icer_cfg.get("incumbent_retention_policy", "generic_support")) == "selected_incumbent_scalar_margin_mse"),
        "decisive_frontier_icer_regret_risk_enabled": float(bool(icer_cfg.get("regret_risk_enabled", False))),
        "decisive_frontier_icer_retention_regret_risk_enabled": float(bool(icer_cfg.get("retention_regret_risk_enabled", icer_cfg.get("regret_risk_enabled", False)))),
        "decisive_frontier_icer_replacement_regret_risk_enabled": float(bool(icer_cfg.get("replacement_regret_risk_enabled", icer_cfg.get("regret_risk_enabled", False)))),
        "decisive_frontier_icer_regret_risk_transition_conditioned": float(str(icer_cfg.get("regret_risk_feature_mode", "evidence_only")).strip().lower() == "transition_conditioned"),
        "decisive_frontier_icer_regret_risk_attribution_resolved": float(str(icer_cfg.get("regret_risk_feature_mode", "evidence_only")).strip().lower() == "attribution_resolved"),
        "decisive_frontier_icer_regret_risk_semantic_family_aligned": float(str(icer_cfg.get("regret_risk_feature_mode", "evidence_only")).strip().lower() == "semantic_family_aligned"),
        "decisive_frontier_icer_regret_risk_type_confirmation": float(str(icer_cfg.get("regret_risk_model_type", "linear_weighted_logistic")).strip().lower() in {"local_multiscale_downside_regret_with_type_confirmation", "local_multiscale_downside_regret_with_global_type_tail_confirmation"}),
        "decisive_frontier_icer_regret_risk_global_tail_mode_confirmation": float(str(icer_cfg.get("regret_risk_model_type", "linear_weighted_logistic")).strip().lower() == "local_multiscale_downside_regret_with_global_type_tail_confirmation"),
        "decisive_frontier_icer_regret_risk_local_memory": float(str(icer_cfg.get("regret_risk_model_type", "linear_weighted_logistic")).strip().lower() in {"local_multiscale_regret_lower_bound", "local_multiscale_downside_regret_certificate", "local_multiscale_downside_regret_with_type_confirmation", "local_multiscale_downside_regret_with_global_type_tail_confirmation"}),
        "decisive_frontier_icer_regret_risk_downside_certificate": float(str(icer_cfg.get("regret_risk_model_type", "linear_weighted_logistic")).strip().lower() in {"local_multiscale_downside_regret_certificate", "local_multiscale_downside_regret_with_type_confirmation"}),
        "decisive_frontier_icer_selected_replacement_regret_risk_logit": float(replacement_regret_risk_logits[selected]) if 0 <= selected < len(valid) and selected not in {a, legacy} else 0.0,
        "decisive_frontier_icer_selected_replacement_confirmation_regret_risk_logit": float(replacement_confirmation_regret_risk_logits[selected]) if 0 <= selected < len(valid) and selected not in {a, legacy} else 0.0,
        "decisive_frontier_icer_legacy_retention_regret_risk_logit": float(retention_regret_risk_logits[legacy]) if 0 <= legacy < len(valid) and legacy != a else 0.0,
        "decisive_frontier_icer_admissible_candidate_count": float(np.asarray(admissible, dtype=bool).sum()),
        "decisive_frontier_icer_safe_domain_active": float(bool(safe_available)),
        "decisive_frontier_icer_all_flagged_domain": float(bool(all_flagged_domain)),
        "decisive_frontier_icer_structural_domain_delegated": float(bool(structural_domain_delegated)),
        "decisive_frontier_icer_all_flagged_preserved_legacy": float(bool(structural_domain_delegated and int(selected) == int(legacy))),
        "_decisive_frontier_icer_anchor_action": int(a),
        "_decisive_frontier_icer_legacy_selected_action": int(legacy),
        "_decisive_frontier_icer_raw_margin_star": np.asarray(margin_star, dtype=np.float32),
        "_decisive_frontier_icer_attribution_scale_star": np.zeros_like(margin_star, dtype=np.float32)
        if attribution_star is None else np.asarray(attribution_star, dtype=np.float32),
        "_decisive_frontier_icer_feature_matrix": np.asarray(feat, dtype=np.float32),
        "_decisive_frontier_icer_feature_names": names,
        "_decisive_frontier_icer_attribution_resolved_feature_matrix": np.asarray(attribution_resolved, dtype=np.float32),
        "_decisive_frontier_icer_attribution_resolved_feature_names": list(attribution_resolved_names),
        "_decisive_frontier_icer_semantic_family_feature_matrix": np.asarray(semantic_family, dtype=np.float32),
        "_decisive_frontier_icer_semantic_family_feature_names": list(semantic_family_names),
        "_decisive_frontier_icer_semantic_type_feature_matrix": np.asarray(semantic_type, dtype=np.float32),
        "_decisive_frontier_icer_semantic_type_feature_names": list(semantic_type_names),
        "_decisive_frontier_icer_support_logit_star": np.asarray(support_logits, dtype=np.float32),
        "_decisive_frontier_icer_dominance_logit_star": np.asarray(dominance_logits, dtype=np.float32),
        "_decisive_frontier_icer_scalar_dominance_logit_star": np.asarray(scalar_dominance_logits, dtype=np.float32) if applies else np.zeros_like(margin_star, dtype=np.float32),
        "_decisive_frontier_icer_profile_dominance_logit_star": np.asarray(profile_dominance_logits, dtype=np.float32) if applies else np.zeros_like(margin_star, dtype=np.float32),
        "_decisive_frontier_icer_incumbent_retention_margin_star": np.asarray(incumbent_retention_margin, dtype=np.float32),
        "_decisive_frontier_icer_replacement_regret_risk_logit_star": np.asarray(replacement_regret_risk_logits, dtype=np.float32),
        "_decisive_frontier_icer_replacement_confirmation_regret_risk_logit_star": np.asarray(replacement_confirmation_regret_risk_logits, dtype=np.float32),
        "_decisive_frontier_icer_retention_regret_risk_logit_star": np.asarray(retention_regret_risk_logits, dtype=np.float32),
        "_decisive_frontier_icer_scir_predicted_improvement_star": np.asarray(scir_predicted_improvement, dtype=np.float32),
        "_decisive_frontier_icer_scir_raw_predicted_improvement_star": np.asarray(scir_raw_predicted_improvement, dtype=np.float32),
        "_decisive_frontier_icer_scir_scene_reservation_feature": np.asarray(scir_scene_reservation_feature, dtype=np.float32),
        "_decisive_frontier_icer_scir_scene_reservation_feature_names": list(scir_scene_reservation_feature_names),
        "_decisive_frontier_icer_scir_post_selection_value_feature": np.asarray(scir_post_selection_value_feature, dtype=np.float32),
        "_decisive_frontier_icer_scir_post_selection_value_feature_names": list(scir_post_selection_value_feature_names),
        "_decisive_frontier_icer_scir_selection_scale_star": np.asarray(scir_selection_scale, dtype=np.float32),
        "_decisive_frontier_icer_scir_lower_bound_star": np.asarray(scir_lower_bound, dtype=np.float32),
        "_decisive_frontier_icer_scir_feature_matrix": np.asarray(scir_feature_matrix, dtype=np.float32),
        "_decisive_frontier_icer_scir_feature_names": list(scir_feature_names),
        "_decisive_frontier_icer_transition_vs_incumbent_feature_matrix": np.asarray(transition_vs_incumbent, dtype=np.float32),
        "_decisive_frontier_icer_transition_vs_anchor_feature_matrix": np.asarray(transition_vs_anchor, dtype=np.float32),
        "_decisive_frontier_icer_transition_feature_names": list(transition_names),
        "_decisive_frontier_icer_admissible_mask": np.asarray(admissible, dtype=bool),
    }
    return int(selected), diag

def run_pair_conditioned_tournament(
    predicted_base_cost: np.ndarray,
    pair_atom_delta: np.ndarray,
    pair_indices: np.ndarray,
    selected_atoms: list[int] | np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    cfg: dict[str, Any],
    pair_atom_variance: np.ndarray | None = None,
    candidate_trajectories: np.ndarray | None = None,
    maneuver_ids: np.ndarray | None = None,
    predicted_atom_costs: np.ndarray | None = None,
    residual_action_potential: np.ndarray | None = None,
    residual_action_variance: np.ndarray | None = None,
    residual_set_atom_factors: np.ndarray | None = None,
    residual_set_action_factors: np.ndarray | None = None,
    frontier_value_atom_factors: np.ndarray | None = None,
    frontier_value_action_signed_factors: np.ndarray | None = None,
    frontier_value_action_context_factors: np.ndarray | None = None,
    frontier_value_scale: float = 1.0,
    evidence_certificate_fraction: float | None = None,
    selected_atom_family_ids: np.ndarray | list[int] | None = None,
    selected_atom_type_names: list[str] | np.ndarray | None = None,
) -> TournamentResult:
    tc = cfg.get("tournament", {})
    sc = cfg.get("selector", {})
    rivals = build_rival_sets_from_base(
        predicted_base_cost,
        valid_mask,
        runtime_safety_flags,
        L_infer=int(tc.get("L_infer", 16)),
        eta0=_pair_selector_eta(cfg),
        candidate_trajectories=candidate_trajectories,
        maneuver_ids=maneuver_ids,
        progress_rivals=int(sc.get("progress_rivals", 0)),
        maneuver_rivals=int(sc.get("maneuver_rivals", 0)),
    )
    normalize_margins = bool(cfg.get("model", {}).get("pair_margin_normalized", False))
    mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    tcfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    norm_min_scale = float(mcfg.get("margin_normalization_min_scale", tcfg.get("pair_margin_min_scale", 100.0)))
    norm_quantile = float(mcfg.get("margin_normalization_quantile", 0.75))
    pair_margin_scale = cfg.get("runtime_pair_margin_scale", None)
    if pair_margin_scale is None:
        pair_margin_scale = cfg.get("pair_margin_scale", None)
    if pair_margin_scale is None and bool(normalize_margins):
        pair_arr_scale = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2) if np.asarray(pair_indices).size else np.zeros((0, 2), dtype=np.int64)
        J0_scale = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
        ok = pair_arr_scale[(pair_arr_scale[:, 0] >= 0) & (pair_arr_scale[:, 0] < J0_scale.shape[0]) & (pair_arr_scale[:, 1] >= 0) & (pair_arr_scale[:, 1] < J0_scale.shape[0])] if pair_arr_scale.size else np.zeros((0, 2), dtype=np.int64)
        pair_margin_scale = margin_normalization_scale(J0_scale[ok[:, 1]] - J0_scale[ok[:, 0]], min_scale=norm_min_scale, quantile=norm_quantile) if ok.size else norm_min_scale
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
    anchor_mode = str(runtime_cfg.get("pair_tournament_anchor_mode", "base_only")).strip().lower()
    use_selected_local_anchor = anchor_mode in {"selected_local", "integrable_selected", "anchor_relative"} and predicted_atom_costs is not None
    aggregation_mode = str(runtime_cfg.get("pair_tournament_aggregation_mode", "legacy_tournament")).strip().lower()
    use_integrable_potential = aggregation_mode in {"integrable_potential", "potential", "hodge_potential"} and use_selected_local_anchor
    use_evidence_action_potential = aggregation_mode in {"evidence_action_potential", "direct_evidence_potential", "dcip"} and use_selected_local_anchor
    use_decisive_anchor_margin = aggregation_mode in {"decisive_anchor_margin", "darm", "anchor_challenger_margin"} and use_selected_local_anchor
    epsilon_cal = float(tc.get("epsilon_cal", cfg.get("calibration", {}).get("epsilon_cal", 0.0)))
    sigma = _pair_sigma_matrix(
        pair_indices,
        pair_atom_variance,
        selected_atoms,
        int(np.asarray(predicted_base_cost).reshape(-1).shape[0]),
    )
    potential_diag: dict[str, Any] = {
        "pair_potential_active": 0.0,
        "direct_evidence_action_potential_active": 0.0,
        "decisive_anchor_margin_active": 0.0,
        "decisive_anchor_margin_anchor_action": -1.0,
    }
    # Runtime-only per-challenger EAF attribution scale.  It is defined from the
    # exact same selected-B atom contributions as the frontier residual and is
    # consumed only by the V64.3.14 one-sided intervention certificate.
    frontier_attribution_scale_star: np.ndarray | None = None
    frontier_atom_contrib_star: np.ndarray | None = None
    if use_evidence_action_potential:
        J_anchor, J_corrected, residual_sigma, potential_diag = _evidence_action_potential_cost(
            predicted_base_cost,
            predicted_atom_costs,
            residual_action_potential,
            selected_atoms,
            valid_mask,
            residual_action_variance=residual_action_variance,
            residual_set_atom_factors=residual_set_atom_factors,
            residual_set_action_factors=residual_set_action_factors,
            set_residual_scale=float(runtime_cfg.get("set_conditioned_residual_scale", 1.0)),
            normalize_margins=normalize_margins,
            margin_scale=float(pair_margin_scale or 1.0),
        )
        scale = max(float(pair_margin_scale or 1.0), 1e-6) if normalize_margins else 1.0
        M_B = (J_corrected[None, :] - J_corrected[:, None]) / scale
        M_eval = M_B - epsilon_cal
        scores = _direct_action_scores_from_cost(J_corrected, valid_mask, scale)
        sigma = residual_sigma
    elif use_integrable_potential:
        J_anchor, J_corrected, potential_diag = _integrable_potential_cost(
            predicted_base_cost,
            pair_indices,
            pair_atom_delta,
            selected_atoms,
            valid_mask,
            predicted_atom_costs=predicted_atom_costs,
            pair_delta_includes_local=bool(runtime_cfg.get("pair_tournament_pair_delta_includes_local", True)),
            normalize_margins=normalize_margins,
            margin_scale=float(pair_margin_scale or 1.0),
            cfg=cfg,
        )
        scale = max(float(pair_margin_scale or 1.0), 1e-6) if normalize_margins else 1.0
        M_B = (J_corrected[None, :] - J_corrected[:, None]) / scale
        M_eval = M_B - epsilon_cal
        # Pair uncertainty must not perturb the selected-local action anchor.  It
        # is used only by the certified flip guard below.
        scores = _direct_action_scores_from_cost(J_corrected, valid_mask, scale)
    elif use_decisive_anchor_margin:
        J_anchor = _selected_local_anchor_cost(predicted_base_cost, predicted_atom_costs, selected_atoms)
        M_B = _pair_delta_margin_matrix(
            predicted_base_cost,
            pair_indices,
            pair_atom_delta,
            selected_atoms,
            valid_mask,
            normalize_margins=normalize_margins,
            margin_scale=pair_margin_scale,
            norm_min_scale=norm_min_scale,
            norm_quantile=norm_quantile,
            predicted_atom_costs=predicted_atom_costs,
            pair_delta_includes_local=bool(runtime_cfg.get("pair_tournament_pair_delta_includes_local", True)),
        )
        scale = max(float(pair_margin_scale or 1.0), 1e-6) if normalize_margins else 1.0
        valid_arr = np.asarray(valid_mask, dtype=bool).reshape(-1)
        finite_anchor = np.asarray(J_anchor, dtype=np.float32).copy()
        finite_anchor[~valid_arr[: finite_anchor.shape[0]]] = np.inf
        darm_anchor = int(np.argmin(finite_anchor)) if np.isfinite(finite_anchor).any() else -1
        frontier_runtime_cfg = runtime_cfg.get("decisive_frontier_value", {}) or {}
        if bool(frontier_runtime_cfg.get("enabled", False)) and darm_anchor >= 0:
            frontier_star, frontier_diag = _decisive_frontier_value_star_residual_numpy(
                selected_atoms,
                valid_mask,
                darm_anchor,
                frontier_value_atom_factors,
                frontier_value_action_signed_factors,
                frontier_value_action_context_factors,
                scale=float(frontier_runtime_cfg.get("scale", frontier_value_scale)),
            )
            # New value is additive to the frozen DARM+DBR baseline.  This makes
            # a zero-initialized V64.3.13 checkpoint exactly reproduce V64.3.7
            # while exposing every valid challenger to the new pair-specific
            # residual, including edges absent from the sparse runtime graph.
            for challenger in np.flatnonzero(valid_arr).tolist():
                if int(challenger) == darm_anchor:
                    continue
                delta = float(frontier_star[int(challenger)])
                M_B[darm_anchor, int(challenger)] = float(M_B[darm_anchor, int(challenger)]) + delta
                M_B[int(challenger), darm_anchor] = -float(M_B[darm_anchor, int(challenger)])
            frontier_attribution_scale_star = frontier_diag.pop(
                "_decisive_frontier_value_attribution_scale_star", None
            )
            frontier_atom_contrib_star = frontier_diag.pop(
                "_decisive_frontier_value_atom_contrib_star", None
            )
            potential_diag.update(frontier_diag)
        else:
            potential_diag.update({
                "decisive_frontier_value_active": 0.0,
                "decisive_frontier_value_complete_star_coverage": 0.0,
            })
        M_eval = M_B - epsilon_cal
        scores, darm_anchor = _decisive_anchor_margin_scores(J_anchor, M_B, valid_mask, scale)
        potential_diag["decisive_anchor_margin_active"] = 1.0
        potential_diag["decisive_anchor_margin_anchor_action"] = float(darm_anchor)
    else:
        M_B = _pair_delta_margin_matrix(
            predicted_base_cost,
            pair_indices,
            pair_atom_delta,
            selected_atoms,
            valid_mask,
            normalize_margins=normalize_margins,
            margin_scale=pair_margin_scale,
            norm_min_scale=norm_min_scale,
            norm_quantile=norm_quantile,
            predicted_atom_costs=predicted_atom_costs if use_selected_local_anchor else None,
            pair_delta_includes_local=bool(runtime_cfg.get("pair_tournament_pair_delta_includes_local", True)),
        )
        M_eval = M_B - epsilon_cal
        scores = tournament_scores(
            M_eval,
            np.asarray(valid_mask, dtype=bool),
            rivals,
            use_softmin=bool(tc.get("use_softmin", True)),
            softmin_tau=float(tc.get("softmin_tau", 1.0)),
            beta_uncertainty=float(tc.get("beta_uncertainty", 0.0)),
            sigma=sigma,
        )
    scores, action, safety_guard_diag = _apply_safety_score_guard(scores, valid_mask, runtime_safety_flags, cfg)
    action, utility_refinement_diag = _apply_certificate_utility_refinement(
        scores,
        action,
        valid_mask,
        runtime_safety_flags,
        cfg,
        candidate_trajectories=candidate_trajectories,
        margins=M_eval,
    )
    anchor_guard_diag: dict[str, Any] = {
        "pair_action_anchor_guard_active": False,
        "pair_action_anchor_guard_blocked_flip": False,
        "pair_action_anchor_guard_allowed_flip": False,
    }
    guard_cfg = runtime_cfg.get("pair_action_anchor_guard", {}) if isinstance(runtime_cfg, dict) else {}
    if use_selected_local_anchor and bool(guard_cfg.get("enabled", True)):
        if use_integrable_potential or use_evidence_action_potential or use_decisive_anchor_margin:
            J_anchor = _selected_local_anchor_cost(predicted_base_cost, predicted_atom_costs, selected_atoms)
            scale = max(float(pair_margin_scale or 1.0), 1e-6) if normalize_margins else 1.0
            anchor_M = (J_anchor[None, :] - J_anchor[:, None]) / scale - epsilon_cal
            anchor_scores = _direct_action_scores_from_cost(J_anchor, valid_mask, scale)
        else:
            J_anchor = _selected_local_anchor_cost(predicted_base_cost, predicted_atom_costs, selected_atoms)
            scale = max(float(pair_margin_scale or 1.0), 1e-6) if normalize_margins else 1.0
            anchor_M = (J_anchor[None, :] - J_anchor[:, None]) / scale - epsilon_cal
            anchor_scores = tournament_scores(
                anchor_M,
                np.asarray(valid_mask, dtype=bool),
                rivals,
                use_softmin=bool(tc.get("use_softmin", True)),
                softmin_tau=float(tc.get("softmin_tau", 1.0)),
                beta_uncertainty=0.0,
                sigma=None,
            )
        anchor_scores, anchor_action, _ = _apply_safety_score_guard(
            anchor_scores, valid_mask, runtime_safety_flags, cfg
        )
        anchor_action, _ = _apply_certificate_utility_refinement(
            anchor_scores,
            anchor_action,
            valid_mask,
            runtime_safety_flags,
            cfg,
            candidate_trajectories=candidate_trajectories,
            margins=anchor_M,
        )
        raw_eaf_action = int(action)
        raer_action, raer_diag = _apply_decisive_frontier_raer(
            raw_eaf_action,
            anchor_action,
            M_B,
            frontier_attribution_scale_star,
            valid_mask,
            runtime_safety_flags,
            potential_diag,
            evidence_certificate_fraction,
            cfg,
        )
        daler_action, daler_diag = _apply_decisive_frontier_daler(
            raw_eaf_action,
            anchor_action,
            M_B,
            frontier_attribution_scale_star,
            scores,
            valid_mask,
            runtime_safety_flags,
            potential_diag,
            evidence_certificate_fraction,
            utility_refinement_diag,
            cfg,
        )
        dacer_action, dacer_diag = _apply_decisive_frontier_dacer(
            raw_eaf_action,
            anchor_action,
            M_B,
            frontier_attribution_scale_star,
            frontier_atom_contrib_star,
            scores,
            valid_mask,
            runtime_safety_flags,
            potential_diag,
            evidence_certificate_fraction,
            utility_refinement_diag,
            cfg,
        )
        icer_action, icer_diag = _apply_decisive_frontier_icer(
            raw_eaf_action,
            anchor_action,
            M_B,
            frontier_attribution_scale_star,
            frontier_atom_contrib_star,
            scores,
            valid_mask,
            runtime_safety_flags,
            potential_diag,
            evidence_certificate_fraction,
            utility_refinement_diag,
            cfg,
            candidate_trajectories=candidate_trajectories,
            maneuver_ids=maneuver_ids,
            selected_atom_family_ids=selected_atom_family_ids,
            selected_atom_type_names=selected_atom_type_names,
        )
        frontier_runtime_cfg = runtime_cfg.get("decisive_frontier_value", {}) or {}
        raer_enabled_runtime = bool((frontier_runtime_cfg.get("reliability_aware_extremal_reranking", {}) or {}).get("enabled", False))
        daler_enabled_runtime = bool((frontier_runtime_cfg.get("deployment_aligned_listwise_extremal_reliability", {}) or {}).get("enabled", False))
        dacer_enabled_runtime = bool((frontier_runtime_cfg.get("deployment_admissible_counterfactual_extremal_recovery", {}) or {}).get("enabled", False))
        icer_enabled_runtime = bool((frontier_runtime_cfg.get("incumbent_contrastive_extremal_recovery", {}) or {}).get("enabled", False))
        if int(raer_enabled_runtime) + int(daler_enabled_runtime) + int(dacer_enabled_runtime) + int(icer_enabled_runtime) > 1:
            raise ValueError("RAER, DALER, DACER and ICER are mutually exclusive causal arms; enable at most one")
        action = int(icer_action if icer_enabled_runtime else (dacer_action if dacer_enabled_runtime else (daler_action if daler_enabled_runtime else raer_action)))
        proposed_action = int(action)
        raw_margin = float(M_B[proposed_action, anchor_action]) if proposed_action != anchor_action else float("inf")
        residual_sigma = (
            float(sigma[proposed_action, anchor_action])
            if sigma is not None and proposed_action != anchor_action
            else 0.0
        )
        dual_cfg = runtime_cfg.get("dual_certificate", {}) or {}
        residual_epsilon = float(
            dual_cfg.get(
                "residual_epsilon_cal",
                dual_cfg.get("residual_epsilon", 0.0),
            )
        )
        robust_margin = raw_margin
        residual_beta_uncertainty = float(
            dual_cfg.get("residual_beta_uncertainty", tc.get("beta_uncertainty", 0.0))
        )

        # V64.3.14 EAF-OCFI: calibrate the *intervention* rather than changing the
        # learned frontier value.  A challenger is allowed to replace the frozen
        # selected-local/DARM anchor only when a split-conformal lower bound on
        # its predicted margin stays positive.  The calibration scale is derived
        # from the already-computed per-atom EAF attribution energy, so this adds
        # no evidence query and preserves the B=16/M=24 interface contract.
        frontier_runtime_cfg = runtime_cfg.get("decisive_frontier_value", {}) or {}
        ocfi_cfg = frontier_runtime_cfg.get("one_sided_intervention", {}) or {}
        ocfi_enabled = bool(ocfi_cfg.get("enabled", False))
        ocfi_requires_frontier = bool(ocfi_cfg.get("require_frontier_active", True))
        frontier_is_active = bool(float(potential_diag.get("decisive_frontier_value_active", 0.0)) >= 0.5)
        ocfi_mode = str(ocfi_cfg.get("normalization", "attribution")).strip().lower()
        ocfi_floor = max(float(ocfi_cfg.get("attribution_scale_floor", 1.0e-3)), 1.0e-9)
        proposed_attr_scale = 0.0
        if (
            frontier_attribution_scale_star is not None
            and proposed_action != anchor_action
            and 0 <= proposed_action < len(frontier_attribution_scale_star)
        ):
            proposed_attr_scale = float(frontier_attribution_scale_star[proposed_action])
        if ocfi_mode in {"none", "constant", "unnormalized"}:
            ocfi_scale = 1.0
        else:
            ocfi_scale = max(proposed_attr_scale, ocfi_floor)
        ocfi_quantile_raw = ocfi_cfg.get("calibration_quantile", None)
        if ocfi_enabled:
            if ocfi_quantile_raw is None or not np.isfinite(float(ocfi_quantile_raw)):
                raise ValueError(
                    "runtime.decisive_frontier_value.one_sided_intervention is enabled but "
                    "calibration_quantile is missing/non-finite; run the V64.3.14 split-calibration tool first"
                )
            ocfi_quantile = max(float(ocfi_quantile_raw), 0.0)
        else:
            ocfi_quantile = 0.0
        # Pair-full/local-pair-full diagnostics intentionally omit the EAF head.
        # OCFI must therefore be an exact no-op on those frozen ceilings rather
        # than blocking their legacy DBR intervention simply because the frontier
        # is absent by design.
        ocfi_applies = bool(ocfi_enabled and frontier_is_active)
        ocfi_additive_radius = max(float(ocfi_cfg.get("additive_radius", 0.0)), 0.0)
        ocfi_radius = (ocfi_quantile * ocfi_scale + ocfi_additive_radius) if ocfi_applies else 0.0
        if proposed_action != anchor_action:
            robust_margin -= residual_beta_uncertainty * residual_sigma
            robust_margin -= residual_epsilon
            robust_margin -= ocfi_radius
        ocfi_frontier_pass = True
        if ocfi_applies and ocfi_requires_frontier:
            ocfi_frontier_pass = frontier_is_active
        score_gain = float(scores[proposed_action] - scores[anchor_action]) if proposed_action != anchor_action else float("inf")
        flip_margin = float(guard_cfg.get("flip_margin", runtime_cfg.get("pair_residual_trust", {}).get("flip_margin", 0.05)))
        score_margin = float(guard_cfg.get("score_margin", 0.0))
        require_evidence_certificate = bool(
            dual_cfg.get("enabled", False)
            and dual_cfg.get("require_evidence_certificate_before_residual_flip", False)
        )
        min_evidence_certificate = float(
            dual_cfg.get("min_evidence_certificate_fraction_for_residual_flip", 1.0)
        )
        evidence_certificate_value = (
            float(evidence_certificate_fraction)
            if evidence_certificate_fraction is not None and np.isfinite(float(evidence_certificate_fraction))
            else float("nan")
        )
        evidence_certificate_pass = (
            (not require_evidence_certificate)
            or (np.isfinite(evidence_certificate_value) and evidence_certificate_value + 1.0e-9 >= min_evidence_certificate)
        )

        # V64.3.15 EAF-EAIR: the V64.3.14 screen showed that a single global
        # over-prediction radius suppresses both beneficial and harmful EAF
        # interventions.  Treat attribution magnitude as *decision support*, not
        # as an uncertainty scale, and learn a tiny standardized logistic readout
        # that predicts whether the raw EAF challenger is teacher-better than the
        # frozen DARM anchor.  The readout consumes only runtime-available EAF
        # statistics and changes no evidence query, acquisition score, B/M budget,
        # pair-full diagnostic, or EAF value itself.
        eair_cfg = frontier_runtime_cfg.get("learned_intervention_reliability", {}) or {}
        eair_enabled = bool(eair_cfg.get("enabled", False))
        eair_instrument = bool(eair_cfg.get("instrument_features", True))
        valid_count_eair = float(np.asarray(valid_mask, dtype=bool).sum())
        frontier_residual_rms = float(potential_diag.get("decisive_frontier_value_residual_rms", 0.0))
        frontier_residual_abs_mean = float(potential_diag.get("decisive_frontier_value_residual_abs_mean", 0.0))
        frontier_attr_rms = float(potential_diag.get("decisive_frontier_value_attribution_scale_rms", 0.0))
        frontier_attr_mean = float(potential_diag.get("decisive_frontier_value_attribution_scale_mean", 0.0))
        attr_eps = max(float(eair_cfg.get("ratio_floor", 1.0e-3)), 1.0e-9)
        eair_features = {
            "raw_margin": float(raw_margin if np.isfinite(raw_margin) else 0.0),
            "proposed_attribution_scale": float(proposed_attr_scale),
            "frontier_residual_rms": frontier_residual_rms,
            "frontier_residual_abs_mean": frontier_residual_abs_mean,
            "frontier_attribution_scale_rms": frontier_attr_rms,
            "frontier_attribution_scale_mean": frontier_attr_mean,
            "evidence_certificate_fraction": float(evidence_certificate_value) if np.isfinite(evidence_certificate_value) else 0.0,
            "valid_action_count_norm": valid_count_eair / max(float(eair_cfg.get("valid_action_normalizer", 32.0)), 1.0),
            "margin_over_attribution": float(raw_margin if np.isfinite(raw_margin) else 0.0) / max(float(proposed_attr_scale), attr_eps),
            "proposed_over_frontier_attribution": float(proposed_attr_scale) / max(frontier_attr_rms, attr_eps),
        }
        eair_prob = 1.0
        eair_logit = float("inf")
        eair_pass = True
        eair_applies = bool(eair_enabled and frontier_is_active)
        if eair_applies and proposed_action != anchor_action:
            names = list(eair_cfg.get("feature_names", []))
            mean = np.asarray(eair_cfg.get("feature_mean", []), dtype=np.float64).reshape(-1)
            std = np.asarray(eair_cfg.get("feature_std", []), dtype=np.float64).reshape(-1)
            weights = np.asarray(eair_cfg.get("weights", []), dtype=np.float64).reshape(-1)
            if not names or len(names) != len(mean) or len(names) != len(std) or len(names) != len(weights):
                raise ValueError("EAF-EAIR enabled but feature_names/mean/std/weights have inconsistent lengths")
            try:
                x = np.asarray([float(eair_features[name]) for name in names], dtype=np.float64)
            except KeyError as exc:
                raise ValueError(f"EAF-EAIR unknown runtime feature: {exc.args[0]}") from exc
            if not np.all(np.isfinite(x)):
                raise ValueError("EAF-EAIR runtime feature vector contains non-finite values")
            z = (x - mean) / np.maximum(std, 1.0e-6)
            eair_logit = float(np.dot(weights, z) + float(eair_cfg.get("bias", 0.0)))
            eair_prob = float(1.0 / (1.0 + np.exp(-np.clip(eair_logit, -40.0, 40.0))))
            eair_pass = bool(eair_prob >= float(eair_cfg.get("min_probability", 0.5)))
        # Pair-full/local-pair-full intentionally omit EAF.  Like OCFI, EAIR
        # must be an exact no-op when the frontier is absent by design.
        eair_frontier_pass = True if not eair_applies else ((not bool(eair_cfg.get("require_frontier_active", True))) or frontier_is_active)
        eair_pass = bool(eair_pass and eair_frontier_pass)

        margin_certificate_pass = (
            robust_margin >= flip_margin and score_gain >= score_margin and ocfi_frontier_pass and eair_pass
        )
        allow_flip = proposed_action == anchor_action or (margin_certificate_pass and evidence_certificate_pass)
        if not allow_flip:
            action = int(anchor_action)
        anchor_guard_diag = {
            "pair_action_anchor_guard_active": True,
            "pair_action_anchor_action": int(anchor_action),
            # Raw residual proposal is captured before the all-flagged structural
            # guard.  Keeping it immutable prevents structural tie-breaking from
            # being misreported as a learned residual intervention.
            "pair_action_anchor_raw_anchor_action": int(anchor_action),
            # Keep the frozen V64.3.13 EAF argmax separate from the V64.3.16
            # pre-guard re-ranked proposal so causal comparisons remain auditable.
            "pair_action_anchor_raw_eaf_proposed_action": int(raw_eaf_action),
            "pair_action_anchor_raw_proposed_action": int(proposed_action),
            "pair_action_anchor_proposed_action": int(proposed_action),
            **raer_diag,
            **daler_diag,
            **dacer_diag,
            **icer_diag,
            "pair_action_anchor_raw_margin": float(raw_margin),
            "pair_action_anchor_residual_sigma": float(residual_sigma),
            "pair_action_anchor_residual_beta_uncertainty": float(residual_beta_uncertainty),
            "pair_action_anchor_residual_epsilon_cal": float(residual_epsilon),
            "decisive_frontier_ocfi_enabled": float(ocfi_enabled),
            "decisive_frontier_ocfi_active": float(ocfi_applies),
            "decisive_frontier_ocfi_frontier_active": float(frontier_is_active),
            "decisive_frontier_ocfi_frontier_requirement_pass": float(ocfi_frontier_pass),
            "decisive_frontier_ocfi_normalization_attribution": float(ocfi_mode not in {"none", "constant", "unnormalized"}),
            "decisive_frontier_ocfi_proposed_attribution_scale": float(proposed_attr_scale),
            "decisive_frontier_ocfi_effective_scale": float(ocfi_scale),
            "decisive_frontier_ocfi_attribution_scale_floor": float(ocfi_floor),
            "decisive_frontier_ocfi_calibration_quantile": float(ocfi_quantile),
            "decisive_frontier_ocfi_calibration_radius": float(ocfi_radius),
            "decisive_frontier_ocfi_frontier_only_lcb": float(raw_margin - ocfi_radius) if proposed_action != anchor_action else float("inf"),
            "decisive_frontier_ocfi_one_sided_lcb": float(robust_margin),
            "decisive_frontier_eair_enabled": float(eair_enabled),
            "decisive_frontier_eair_instrument_features": float(eair_instrument),
            "decisive_frontier_eair_active": float(eair_applies),
            "decisive_frontier_eair_probability": float(eair_prob),
            "decisive_frontier_eair_logit": float(eair_logit) if np.isfinite(eair_logit) else 40.0,
            "decisive_frontier_eair_pass": float(eair_pass),
            "decisive_frontier_eair_min_probability": float(eair_cfg.get("min_probability", 0.5)),
            "decisive_frontier_eair_feature_raw_margin": float(eair_features["raw_margin"]),
            "decisive_frontier_eair_feature_proposed_attribution_scale": float(eair_features["proposed_attribution_scale"]),
            "decisive_frontier_eair_feature_frontier_residual_rms": float(eair_features["frontier_residual_rms"]),
            "decisive_frontier_eair_feature_frontier_residual_abs_mean": float(eair_features["frontier_residual_abs_mean"]),
            "decisive_frontier_eair_feature_frontier_attribution_scale_rms": float(eair_features["frontier_attribution_scale_rms"]),
            "decisive_frontier_eair_feature_frontier_attribution_scale_mean": float(eair_features["frontier_attribution_scale_mean"]),
            "decisive_frontier_eair_feature_evidence_certificate_fraction": float(eair_features["evidence_certificate_fraction"]),
            "decisive_frontier_eair_feature_valid_action_count_norm": float(eair_features["valid_action_count_norm"]),
            "decisive_frontier_eair_feature_margin_over_attribution": float(eair_features["margin_over_attribution"]),
            "decisive_frontier_eair_feature_proposed_over_frontier_attribution": float(eair_features["proposed_over_frontier_attribution"]),
            "pair_action_anchor_robust_margin": float(robust_margin),
            "pair_action_anchor_score_gain": float(score_gain),
            "pair_action_anchor_guard_blocked_flip": bool(proposed_action != anchor_action and not allow_flip),
            "pair_action_anchor_guard_allowed_flip": bool(proposed_action != anchor_action and allow_flip),
            "pair_action_anchor_guard_margin_certificate_pass": bool(margin_certificate_pass),
            "pair_action_anchor_guard_evidence_certificate_required": bool(require_evidence_certificate),
            "pair_action_anchor_guard_evidence_certificate_fraction": float(evidence_certificate_value),
            "pair_action_anchor_guard_min_evidence_certificate_fraction": float(min_evidence_certificate),
            "pair_action_anchor_guard_evidence_certificate_pass": bool(evidence_certificate_pass),
            "pair_action_anchor_guard_blocked_by_evidence_certificate": bool(
                proposed_action != anchor_action and margin_certificate_pass and not evidence_certificate_pass
            ),
            "pair_action_anchor_deployed_flip": bool(int(action) != int(anchor_action)),
            # Private arrays are consumed by BDSEPlannerCore after the final
            # all-flagged structural guard so residual flip metrics compare two
            # paths with identical post-processing.
            "_pair_action_anchor_scores": np.asarray(anchor_scores, dtype=np.float32),
            "_pair_action_anchor_margins": np.asarray(anchor_M, dtype=np.float32),
            "_pair_action_anchor_pre_structural_action": int(anchor_action),
        }
    sorted_scores = np.sort(scores[np.asarray(valid_mask, dtype=bool)])
    delta = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) >= 2 else float("inf")
    safety_idx = np.flatnonzero(np.asarray(runtime_safety_flags, dtype=bool) & np.asarray(valid_mask, dtype=bool))
    if safety_idx.size and action not in safety_idx:
        safety_lcb_min = float(np.min(M_eval[action, safety_idx]))
    elif action in safety_idx:
        safety_lcb_min = -float("inf")
    else:
        safety_lcb_min = float("inf")
    return TournamentResult(
        action_index=action,
        scores=scores,
        margins=M_eval,
        rival_sets=rivals,
        diagnostics={
            "delta_hat_B": delta,
            "selected_atoms": list(map(int, selected_atoms)),
            "valid_actions": int(np.asarray(valid_mask).sum()),
            "rival_source": "base_score_cheap_flags_pair_conditioned",
            "epsilon_cal": epsilon_cal,
            "beta_uncertainty": float(tc.get("beta_uncertainty", 0.0)),
            "sigma_used": bool(sigma is not None),
            "safety_lcb_min": safety_lcb_min,
            **safety_guard_diag,
            **utility_refinement_diag,
            **anchor_guard_diag,
            **potential_diag,
            "pair_tournament_aggregation_mode": aggregation_mode,
            "selected_action_safety_flag": bool(np.asarray(runtime_safety_flags, dtype=bool)[action]) if 0 <= action < len(runtime_safety_flags) else False,
            "avoidable_selected_action_safety_flag": bool(
                (bool(np.asarray(runtime_safety_flags, dtype=bool)[action]) if 0 <= action < len(runtime_safety_flags) else True)
                and bool(safety_guard_diag.get("safe_action_available", False))
            ),
            "pair_conditioned": True,
            "selector_eta_used": float(_pair_selector_eta(cfg)),
            "normalized_margins": bool(normalize_margins),
            "margin_scale": float(pair_margin_scale) if pair_margin_scale is not None else 1.0,
        },
    )


def full_interface_cost(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Dense full-interface cost with invalid actions masked to +inf."""
    J0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    g = np.asarray(predicted_atom_costs, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if g.ndim != 2 or g.shape[1] != J0.shape[0]:
        raise ValueError(f"predicted_atom_costs must have shape [E,K] with K={J0.shape[0]}, got {g.shape}")
    cost = J0 + g.sum(axis=0)
    cost = np.asarray(cost, dtype=np.float32).copy()
    cost[~valid] = np.inf
    return cost


def full_interface_action(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    valid_mask: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> int:
    """Hard-argmin dense full-interface diagnostic action.

    This diagnostic should measure whether J_base + sum_i g_i reconstructs the
    teacher argmin.  It intentionally does not use softmin tournament smoothing,
    because softmin can change the argmin even when the dense cost partition is
    exact.
    """
    cost = full_interface_cost(predicted_base_cost, predicted_atom_costs, valid_mask)
    if not np.isfinite(cost).any():
        return 0
    return int(np.nanargmin(cost))


def full_interface_soft_tournament_action(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    valid_mask: np.ndarray,
    cfg: dict[str, Any],
) -> int:
    """Backward-compatible dense soft tournament action for ablations only."""
    M = full_interface_margin(predicted_base_cost, predicted_atom_costs)
    valid = np.asarray(valid_mask, dtype=bool)
    K = len(valid)
    rivals = [[b for b in range(K) if b != a and valid[b]] for a in range(K)]
    scores = tournament_scores(M, valid, rivals, True, float(cfg.get("tournament", {}).get("softmin_tau", 1.0)))
    return int(np.argmax(scores))


def assert_antisymmetric(M: np.ndarray, valid_mask: np.ndarray, atol: float = 1e-5) -> None:
    valid = np.asarray(valid_mask, dtype=bool)
    for a in np.flatnonzero(valid):
        for b in np.flatnonzero(valid):
            if abs(float(M[a, b] + M[b, a])) > atol:
                raise AssertionError("Pairwise margins must be antisymmetric")
