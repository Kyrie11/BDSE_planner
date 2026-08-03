from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bdse.planner.pair_screen import build_rival_sets_from_base
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

    BDSE's budgeted tournament score is treated as a safety/decision certificate.
    When several valid unflagged candidates are within ``score_slack`` of the best
    certificate score, their evidence distinction is too small to justify picking
    a low-progress action.  Within that certified equivalence set, choose the
    lowest deployment utility cost.  This is not a post-hoc rule override: actions
    outside the certificate band remain ineligible.
    """
    tc = cfg.get("tournament", {}) if isinstance(cfg, dict) else {}
    uc = tc.get("utility_refinement", {}) or {}
    if not bool(uc.get("enabled", False)):
        return int(action), {"utility_refinement_enabled": False}
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
    if not bool(finite.any()):
        return int(action), {"utility_refinement_enabled": True, "utility_refinement_applied": False, "utility_refinement_reason": "no_finite"}
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
                    # M[c,current] > 0 means candidate c is pair-certified better
                    # than the current certificate action.  A small negative
                    # tolerance permits near-tie utility refinement but prevents
                    # utility from overriding a decisive evidence margin.
                    cert_mask[int(c)] = bool(np.isfinite(M[int(c), current_for_cert]) and float(M[int(c), current_for_cert]) >= -tol)
            if bool(cert_mask.any()):
                eligible = eligible & cert_mask
                pair_cert_used = True
                pair_cert_kept = int(eligible.sum())
    if int(eligible.sum()) <= 0:
        return int(action), {"utility_refinement_enabled": True, "utility_refinement_applied": False, "utility_refinement_reason": "empty_band", "utility_score_slack": float(slack)}
    utility_cost = _trajectory_utility_cost_np(candidate_trajectories, valid, flags, cfg)
    cand = np.flatnonzero(eligible & np.isfinite(utility_cost)).astype(np.int64)
    if cand.size == 0:
        return int(action), {"utility_refinement_enabled": True, "utility_refinement_applied": False, "utility_refinement_reason": "no_utility"}
    # Stable deterministic tie-break: utility cost, then higher certificate score, then index.
    best_util = int(sorted(cand.tolist(), key=lambda a: (float(utility_cost[a]), -float(scores[a]), int(a)))[0])
    current = int(action) if 0 <= int(action) < n else int(np.argmax(np.where(finite, scores, -np.inf)))
    min_improvement = float(uc.get("min_utility_improvement", 0.0))
    applied = bool(best_util != current and float(utility_cost[best_util]) <= float(utility_cost[current]) - min_improvement)
    chosen = int(best_util) if applied else int(current)
    diag = {
        "utility_refinement_enabled": True,
        "utility_refinement_applied": bool(applied),
        "utility_refinement_action_before": int(current),
        "utility_refinement_action_after": int(chosen),
        "utility_score_slack": float(slack),
        "utility_band_size": int(cand.size),
        "utility_best_score": float(scores[best_util]),
        "utility_current_score": float(scores[current]) if 0 <= current < n else float("nan"),
        "utility_best_cost": float(utility_cost[best_util]),
        "utility_current_cost": float(utility_cost[current]) if 0 <= current < n and np.isfinite(utility_cost[current]) else float("inf"),
        "utility_pair_certificate_used": bool(pair_cert_used),
        "utility_pair_certificate_enabled": bool(uc.get("pair_certificate_enabled", False)),
        "utility_pair_margin_tolerance": float(max(float(uc.get("pair_margin_tolerance", 0.05)), 0.0)),
        "utility_pair_certificate_kept": int(pair_cert_kept),
    }
    return chosen, diag

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
    evidence_certificate_fraction: float | None = None,
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
    epsilon_cal = float(tc.get("epsilon_cal", cfg.get("calibration", {}).get("epsilon_cal", 0.0)))
    sigma = _pair_sigma_matrix(
        pair_indices,
        pair_atom_variance,
        selected_atoms,
        int(np.asarray(predicted_base_cost).reshape(-1).shape[0]),
    )
    potential_diag: dict[str, Any] = {"pair_potential_active": 0.0, "direct_evidence_action_potential_active": 0.0}
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
        if use_integrable_potential or use_evidence_action_potential:
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
        if proposed_action != anchor_action:
            robust_margin -= float(tc.get("beta_uncertainty", 0.0)) * residual_sigma
            robust_margin -= residual_epsilon
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
        margin_certificate_pass = robust_margin >= flip_margin and score_gain >= score_margin
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
            "pair_action_anchor_raw_proposed_action": int(proposed_action),
            "pair_action_anchor_proposed_action": int(proposed_action),
            "pair_action_anchor_raw_margin": float(raw_margin),
            "pair_action_anchor_residual_sigma": float(residual_sigma),
            "pair_action_anchor_residual_epsilon_cal": float(residual_epsilon),
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
