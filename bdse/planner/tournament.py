from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bdse.planner.pair_screen import build_rival_sets_from_base
from bdse.planner.selector import budgeted_margin, full_interface_margin, margin_normalization_scale
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
    action = int(np.argmax(scores))
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
            "selected_action_safety_flag": bool(np.asarray(runtime_safety_flags, dtype=bool)[action]) if 0 <= action < len(runtime_safety_flags) else False,
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
    J0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    K = J0.shape[0]
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    scale = 1.0
    if bool(normalize_margins):
        if margin_scale is not None and np.isfinite(float(margin_scale)) and float(margin_scale) > 0:
            scale = float(margin_scale)
        elif pair_arr.size:
            ok_pairs = pair_arr[(pair_arr[:, 0] >= 0) & (pair_arr[:, 0] < K) & (pair_arr[:, 1] >= 0) & (pair_arr[:, 1] < K)]
            scale = margin_normalization_scale(J0[ok_pairs[:, 1]] - J0[ok_pairs[:, 0]], min_scale=float(norm_min_scale), quantile=float(norm_quantile)) if ok_pairs.size else float(norm_min_scale)
        else:
            scale = float(norm_min_scale)
    M = (J0[None, :] - J0[:, None]) / max(scale, 1e-6)
    np.fill_diagonal(M, 0.0)
    delta = np.asarray(pair_atom_delta, dtype=np.float32)
    selected = np.asarray(selected_atoms, dtype=np.int64).reshape(-1)
    selected = selected[(selected >= 0) & (selected < delta.shape[0])] if delta.ndim == 2 else np.zeros((0,), dtype=np.int64)

    directed: dict[tuple[int, int], float] = {}
    if pair_arr.size and delta.ndim == 2 and delta.shape[1] >= pair_arr.shape[0] and selected.size:
        support = delta[selected, : pair_arr.shape[0]].sum(axis=0)
        for pidx, (a_raw, b_raw) in enumerate(pair_arr.tolist()):
            a, b = int(a_raw), int(b_raw)
            if 0 <= a < K and 0 <= b < K and a != b:
                directed[(a, b)] = (J0[b] - J0[a]) / max(scale, 1e-6) + float(support[pidx])

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

    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.shape[0] < K:
        valid = np.pad(valid, (0, K - valid.shape[0]), constant_values=False)
    valid = valid[:K]
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
    )
    epsilon_cal = float(tc.get("epsilon_cal", cfg.get("calibration", {}).get("epsilon_cal", 0.0)))
    M_eval = M_B - epsilon_cal
    sigma = _pair_sigma_matrix(pair_indices, pair_atom_variance, selected_atoms, M_B.shape[0])
    scores = tournament_scores(
        M_eval,
        np.asarray(valid_mask, dtype=bool),
        rivals,
        use_softmin=bool(tc.get("use_softmin", True)),
        softmin_tau=float(tc.get("softmin_tau", 1.0)),
        beta_uncertainty=float(tc.get("beta_uncertainty", 0.0)),
        sigma=sigma,
    )
    action = int(np.argmax(scores))
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
            "selected_action_safety_flag": bool(np.asarray(runtime_safety_flags, dtype=bool)[action]) if 0 <= action < len(runtime_safety_flags) else False,
            "pair_conditioned": True,
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
