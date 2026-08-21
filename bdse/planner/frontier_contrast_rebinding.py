from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from bdse.planner.tournament import (
    _decisive_frontier_value_star_residual_numpy,
    _pair_delta_margin_matrix,
)


@dataclass(frozen=True)
class FrontierContrastRebindingResult:
    """Deterministic fixed-budget post-EAF evidence rebinding result.

    The mechanism never queries new evidence and never changes the frozen Top-M
    candidate pool.  It may only replace the baseline selected-B atom set by a
    same-cardinality subset of that already queried pool when three invariants
    simultaneously hold:

    1. the full-M selected-local anchor is preserved;
    2. the exact downstream full-M target action is preserved;
    3. the complete DARM+EAF anchor-star compression error is strictly improved.

    Otherwise the baseline selection is returned unchanged.
    """

    selected: list[int]
    diagnostics: dict[str, Any]


def _finite_valid_cost(cost: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    x = np.asarray(cost, dtype=np.float64).reshape(-1).copy()
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.shape[0] < x.shape[0]:
        valid = np.pad(valid, (0, x.shape[0] - valid.shape[0]), constant_values=False)
    valid = valid[: x.shape[0]]
    x[~valid] = np.inf
    x[~np.isfinite(x)] = np.inf
    return x


def _selected_local_anchor(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    selected_atoms: list[int] | np.ndarray,
    valid_mask: np.ndarray,
) -> int:
    j0 = np.asarray(predicted_base_cost, dtype=np.float64).reshape(-1)
    g = np.asarray(predicted_atom_costs, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != j0.shape[0]:
        return -1
    selected = np.asarray(selected_atoms, dtype=np.int64).reshape(-1)
    selected = selected[(selected >= 0) & (selected < g.shape[0])]
    total = j0.copy()
    if selected.size:
        total = total + g[selected].sum(axis=0)
    finite = _finite_valid_cost(total, valid_mask)
    return int(np.argmin(finite)) if np.isfinite(finite).any() else -1


def _frontier_star(
    *,
    predicted_base_cost: np.ndarray,
    pair_indices: np.ndarray,
    pair_atom_delta: np.ndarray,
    selected_atoms: list[int] | np.ndarray,
    valid_mask: np.ndarray,
    anchor_action: int,
    normalize_margins: bool,
    margin_scale: float,
    predicted_atom_costs: np.ndarray,
    pair_delta_includes_local: bool,
    frontier_value_atom_factors: np.ndarray,
    frontier_value_action_signed_factors: np.ndarray,
    frontier_value_action_context_factors: np.ndarray,
    frontier_value_scale: float,
) -> np.ndarray:
    """Exact frozen DARM+EAF margin star around one fixed anchor."""
    matrix = _pair_delta_margin_matrix(
        predicted_base_cost,
        pair_indices,
        pair_atom_delta,
        selected_atoms,
        valid_mask,
        normalize_margins=bool(normalize_margins),
        margin_scale=float(margin_scale),
        predicted_atom_costs=predicted_atom_costs,
        pair_delta_includes_local=bool(pair_delta_includes_local),
    )
    star = np.asarray(matrix[int(anchor_action)], dtype=np.float64).copy()
    frontier, _ = _decisive_frontier_value_star_residual_numpy(
        selected_atoms,
        valid_mask,
        int(anchor_action),
        frontier_value_atom_factors,
        frontier_value_action_signed_factors,
        frontier_value_action_context_factors,
        scale=float(frontier_value_scale),
    )
    star = star + np.asarray(frontier, dtype=np.float64)
    return star


def _error_tuple(star: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(star, dtype=np.float64)[mask] - np.asarray(target, dtype=np.float64)[mask]
    if diff.size == 0 or not np.isfinite(diff).all():
        return float("inf"), float("inf")
    abs_diff = np.abs(diff)
    return float(np.max(abs_diff)), float(np.sqrt(np.mean(diff * diff)))


def _strict_lexicographic_improvement(
    candidate: tuple[float, float],
    baseline: tuple[float, float],
    eps: float,
) -> bool:
    c_inf, c_rms = candidate
    b_inf, b_rms = baseline
    if not all(np.isfinite(v) for v in (c_inf, c_rms, b_inf, b_rms)):
        return False
    if c_inf < b_inf - eps:
        return True
    return bool(abs(c_inf - b_inf) <= eps and c_rms < b_rms - eps)


def _future_budget_feasible(
    *,
    chosen_cost: float,
    remaining_costs: list[float],
    remaining_slots: int,
    budget: float,
    eps: float,
) -> bool:
    if remaining_slots <= 0:
        return chosen_cost <= budget + eps
    finite = sorted(float(x) for x in remaining_costs if np.isfinite(float(x)) and float(x) >= 0.0)
    if len(finite) < remaining_slots:
        return False
    return chosen_cost + float(sum(finite[:remaining_slots])) <= budget + eps


def frontier_contrast_rebind(
    *,
    baseline_selected: list[int] | np.ndarray,
    reference_atoms: list[int] | np.ndarray,
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    pair_indices: np.ndarray,
    pair_atom_delta: np.ndarray,
    valid_mask: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    normalize_margins: bool,
    margin_scale: float,
    pair_delta_includes_local: bool,
    frontier_value_atom_factors: np.ndarray | None,
    frontier_value_action_signed_factors: np.ndarray | None,
    frontier_value_action_context_factors: np.ndarray | None,
    frontier_value_scale: float,
    deployment_evaluator: Callable[[list[int]], tuple[Any, ...]] | None,
    full_target_action: int | None,
    improvement_epsilon: float = 1e-8,
) -> FrontierContrastRebindingResult:
    """Post-EAF Frontier-Contrast Rebinding (FCR).

    This is deliberately knob-free apart from numerical epsilon.  It performs a
    deterministic forward construction over the frozen Top-M evidence pool.  The
    objective is the lexicographic (L_inf, RMS) error between the retained-B and
    full-M *complete* DARM+EAF margin star around the full-M selected-local
    anchor.  It never sees teacher labels or endpoint regret.

    Critically, the constructed set is only deployed when it is a monotone
    refinement of the baseline interface: same cardinality/budget, exact full-M
    local-anchor preservation, exact full-M downstream winner preservation, and
    strictly lower frontier-star compression error.  Failure of any condition is
    a no-op fallback to the already-audited baseline AOCC selection.
    """

    eps = max(float(improvement_epsilon), 0.0)
    baseline = [int(x) for x in np.asarray(baseline_selected, dtype=np.int64).reshape(-1).tolist()]
    reference = [int(x) for x in np.asarray(reference_atoms, dtype=np.int64).reshape(-1).tolist()]
    # Preserve deterministic order while removing duplicates.
    baseline = list(dict.fromkeys(baseline))
    reference = list(dict.fromkeys(reference))

    diag: dict[str, Any] = {
        "frontier_contrast_rebinding_enabled": 1.0,
        "frontier_contrast_rebinding_accepted": 0.0,
        "frontier_contrast_rebinding_attempted": 0.0,
        "frontier_contrast_rebinding_baseline_count": float(len(baseline)),
        "frontier_contrast_rebinding_reference_count": float(len(reference)),
        "frontier_contrast_rebinding_budget": float(budget),
        "frontier_contrast_rebinding_no_new_query": 1.0,
        "frontier_contrast_rebinding_teacher_free": 1.0,
        "frontier_contrast_rebinding_same_cardinality_required": 1.0,
        "frontier_contrast_rebinding_full_m_local_anchor_required": 1.0,
        "frontier_contrast_rebinding_exact_target_required": 1.0,
        "frontier_contrast_rebinding_strict_error_improvement_required": 1.0,
        "frontier_contrast_rebinding_reason_code": 0.0,
    }

    # Numeric reason codes are intentionally stable so scalar-only evaluator
    # diagnostics retain the failure mode without needing string serialization.
    # 0=accepted/ready, 1=empty/invalid pool, 2=shape/factor contract,
    # 3=baseline outside reference/budget, 4=no valid full-M anchor,
    # 5=missing exact target evaluator, 6=greedy infeasible,
    # 7=no strict compression improvement, 8=local-anchor mismatch,
    # 9=exact-target mismatch.
    def fallback(reason: int, **extra: Any) -> FrontierContrastRebindingResult:
        diag["frontier_contrast_rebinding_reason_code"] = float(reason)
        diag.update(extra)
        return FrontierContrastRebindingResult(selected=list(baseline), diagnostics=dict(diag))

    if not reference or not baseline or len(baseline) > len(reference):
        return fallback(1)

    j0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    g = np.asarray(predicted_atom_costs, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    costs = np.asarray(atom_budget_costs, dtype=np.float64).reshape(-1)
    pair_delta = np.asarray(pair_atom_delta, dtype=np.float32)
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    atom_f = None if frontier_value_atom_factors is None else np.asarray(frontier_value_atom_factors, dtype=np.float32)
    signed_f = None if frontier_value_action_signed_factors is None else np.asarray(frontier_value_action_signed_factors, dtype=np.float32)
    context_f = None if frontier_value_action_context_factors is None else np.asarray(frontier_value_action_context_factors, dtype=np.float32)

    if (
        g.ndim != 2
        or g.shape[1] != j0.shape[0]
        or valid.shape[0] < j0.shape[0]
        or pair_delta.ndim != 2
        or pair_delta.shape[1] < pair_arr.shape[0]
        or atom_f is None
        or signed_f is None
        or context_f is None
        or atom_f.ndim != 2
        or signed_f.ndim != 2
        or context_f.ndim != 2
        or signed_f.shape != context_f.shape
        or signed_f.shape[0] < j0.shape[0]
        or atom_f.shape[1] <= 0
        or signed_f.shape[1] != atom_f.shape[1]
    ):
        return fallback(2)

    max_atom = min(g.shape[0], pair_delta.shape[0], atom_f.shape[0], costs.shape[0])
    if any(x < 0 or x >= max_atom for x in reference) or any(x not in set(reference) for x in baseline):
        return fallback(3)
    baseline_cost = float(np.sum(costs[np.asarray(baseline, dtype=np.int64)]))
    if not np.isfinite(baseline_cost) or baseline_cost > float(budget) + eps:
        return fallback(3, frontier_contrast_rebinding_baseline_cost=baseline_cost)

    full_anchor = _selected_local_anchor(j0, g, reference, valid)
    if full_anchor < 0:
        return fallback(4)
    diag["frontier_contrast_rebinding_full_m_local_anchor"] = float(full_anchor)

    if deployment_evaluator is None or full_target_action is None:
        return fallback(5)
    target_action = int(full_target_action)
    diag["frontier_contrast_rebinding_full_m_exact_target_action"] = float(target_action)

    target_star = _frontier_star(
        predicted_base_cost=j0,
        pair_indices=pair_arr,
        pair_atom_delta=pair_delta,
        selected_atoms=reference,
        valid_mask=valid,
        anchor_action=full_anchor,
        normalize_margins=normalize_margins,
        margin_scale=margin_scale,
        predicted_atom_costs=g,
        pair_delta_includes_local=pair_delta_includes_local,
        frontier_value_atom_factors=atom_f,
        frontier_value_action_signed_factors=signed_f,
        frontier_value_action_context_factors=context_f,
        frontier_value_scale=frontier_value_scale,
    )
    challenger_mask = valid[: j0.shape[0]].copy()
    challenger_mask[full_anchor] = False
    if not bool(challenger_mask.any()) or not np.isfinite(target_star[challenger_mask]).all():
        return fallback(2)

    baseline_star = _frontier_star(
        predicted_base_cost=j0,
        pair_indices=pair_arr,
        pair_atom_delta=pair_delta,
        selected_atoms=baseline,
        valid_mask=valid,
        anchor_action=full_anchor,
        normalize_margins=normalize_margins,
        margin_scale=margin_scale,
        predicted_atom_costs=g,
        pair_delta_includes_local=pair_delta_includes_local,
        frontier_value_atom_factors=atom_f,
        frontier_value_action_signed_factors=signed_f,
        frontier_value_action_context_factors=context_f,
        frontier_value_scale=frontier_value_scale,
    )
    baseline_error = _error_tuple(baseline_star, target_star, challenger_mask)
    diag["frontier_contrast_rebinding_baseline_linf_error"] = float(baseline_error[0])
    diag["frontier_contrast_rebinding_baseline_rms_error"] = float(baseline_error[1])
    # Until all acceptance guards pass the deployed set is the baseline AOCC
    # selection, so the final error is initialized to the baseline error and is
    # overwritten only on a successful monotone rebind.
    diag["frontier_contrast_rebinding_final_linf_error"] = float(baseline_error[0])
    diag["frontier_contrast_rebinding_final_rms_error"] = float(baseline_error[1])
    diag["frontier_contrast_rebinding_changed_atom_count"] = 0.0

    # Precompute an exact additive decomposition of the frozen DARM/DBR star for
    # the fixed full-M anchor.  The pair construction is linear in selected
    # evidence for a fixed normalization scale; singleton-minus-empty therefore
    # gives each atom's exact contribution, including local-cost fallback edges.
    empty_matrix = _pair_delta_margin_matrix(
        j0,
        pair_arr,
        pair_delta,
        [],
        valid,
        normalize_margins=bool(normalize_margins),
        margin_scale=float(margin_scale),
        predicted_atom_costs=g,
        pair_delta_includes_local=bool(pair_delta_includes_local),
    )
    darm_base = np.asarray(empty_matrix[full_anchor], dtype=np.float64)
    darm_contrib: dict[int, np.ndarray] = {}
    for atom_idx in reference:
        one = _pair_delta_margin_matrix(
            j0,
            pair_arr,
            pair_delta,
            [atom_idx],
            valid,
            normalize_margins=bool(normalize_margins),
            margin_scale=float(margin_scale),
            predicted_atom_costs=g,
            pair_delta_includes_local=bool(pair_delta_includes_local),
        )
        darm_contrib[atom_idx] = np.asarray(one[full_anchor], dtype=np.float64) - darm_base

    # Exact EAF decomposition at the *final retained cardinality*.  The learned
    # frontier residual is sum_e phi_e / sqrt(|S|); because FCR preserves the
    # baseline cardinality, every forward step can evaluate the exact final-set
    # contribution without introducing a cardinality-dependent heuristic.
    final_count = len(baseline)
    rank = int(atom_f.shape[1])
    a = int(full_anchor)
    bounded = np.tanh(atom_f[np.asarray(reference, dtype=np.int64)]).astype(np.float64)
    pair_sym = np.tanh(
        context_f[a][None, :] + context_f[: j0.shape[0]] + context_f[a][None, :] * context_f[: j0.shape[0]]
    ).astype(np.float64)
    signed_diff = np.asarray(signed_f[: j0.shape[0]] - signed_f[a][None, :], dtype=np.float64)
    pair_vec = pair_sym * signed_diff
    raw_eaf = np.einsum("nr,kr->nk", bounded, pair_vec, optimize=True)
    raw_eaf *= float(frontier_value_scale) / np.sqrt(max(float(final_count * rank), 1.0))
    eaf_contrib = {idx: raw_eaf[pos] for pos, idx in enumerate(reference)}

    diag["frontier_contrast_rebinding_attempted"] = 1.0
    chosen: list[int] = []
    sum_darm = np.zeros_like(darm_base, dtype=np.float64)
    sum_eaf = np.zeros_like(darm_base, dtype=np.float64)
    spent = 0.0
    reference_set = set(reference)

    for step in range(final_count):
        best_key: tuple[float, float, float, int] | None = None
        best_idx: int | None = None
        best_darm: np.ndarray | None = None
        best_eaf: np.ndarray | None = None
        remaining_slots = final_count - step - 1
        for atom_idx in reference:
            if atom_idx in chosen:
                continue
            c = float(costs[atom_idx])
            if not np.isfinite(c) or c < 0.0 or spent + c > float(budget) + eps:
                continue
            rest_costs = [
                float(costs[j])
                for j in reference
                if j != atom_idx and j not in chosen and j in reference_set
            ]
            if not _future_budget_feasible(
                chosen_cost=spent + c,
                remaining_costs=rest_costs,
                remaining_slots=remaining_slots,
                budget=float(budget),
                eps=eps,
            ):
                continue
            cand_darm = sum_darm + darm_contrib[atom_idx]
            cand_eaf = sum_eaf + eaf_contrib[atom_idx]
            cand_star = darm_base + cand_darm + cand_eaf
            err = _error_tuple(cand_star, target_star, challenger_mask)
            # Lower atom cost is only a deterministic feasibility tie-break; the
            # scientific objective remains the complete frontier error.
            key = (float(err[0]), float(err[1]), c, int(atom_idx))
            if best_key is None or key < best_key:
                best_key = key
                best_idx = int(atom_idx)
                best_darm = cand_darm
                best_eaf = cand_eaf
        if best_idx is None or best_darm is None or best_eaf is None:
            return fallback(6, frontier_contrast_rebinding_partial_count=float(len(chosen)))
        chosen.append(best_idx)
        spent += float(costs[best_idx])
        sum_darm = best_darm
        sum_eaf = best_eaf

    if len(chosen) != final_count or len(set(chosen)) != final_count:
        return fallback(6, frontier_contrast_rebinding_partial_count=float(len(chosen)))

    # Recompute the candidate with the exact production primitives rather than
    # trusting the incremental objective arithmetic.  This is both an engineering
    # consistency check and the value used for the monotone acceptance contract.
    candidate_star = _frontier_star(
        predicted_base_cost=j0,
        pair_indices=pair_arr,
        pair_atom_delta=pair_delta,
        selected_atoms=chosen,
        valid_mask=valid,
        anchor_action=full_anchor,
        normalize_margins=normalize_margins,
        margin_scale=margin_scale,
        predicted_atom_costs=g,
        pair_delta_includes_local=pair_delta_includes_local,
        frontier_value_atom_factors=atom_f,
        frontier_value_action_signed_factors=signed_f,
        frontier_value_action_context_factors=context_f,
        frontier_value_scale=frontier_value_scale,
    )
    candidate_error = _error_tuple(candidate_star, target_star, challenger_mask)
    candidate_cost = float(np.sum(costs[np.asarray(chosen, dtype=np.int64)]))
    diag.update({
        "frontier_contrast_rebinding_candidate_count": float(len(chosen)),
        "frontier_contrast_rebinding_candidate_cost": candidate_cost,
        "frontier_contrast_rebinding_candidate_linf_error": float(candidate_error[0]),
        "frontier_contrast_rebinding_candidate_rms_error": float(candidate_error[1]),
        "frontier_contrast_rebinding_cardinality_preserved": float(len(chosen) == len(baseline)),
        "frontier_contrast_rebinding_budget_preserved": float(candidate_cost <= float(budget) + eps),
    })

    if not _strict_lexicographic_improvement(candidate_error, baseline_error, eps):
        return fallback(7)

    candidate_local_anchor = _selected_local_anchor(j0, g, chosen, valid)
    diag["frontier_contrast_rebinding_candidate_local_anchor"] = float(candidate_local_anchor)
    diag["frontier_contrast_rebinding_local_anchor_preserved"] = float(candidate_local_anchor == full_anchor)
    if candidate_local_anchor != full_anchor:
        return fallback(8)

    try:
        baseline_exact_action = int(deployment_evaluator(list(baseline))[0])
        candidate_exact_action = int(deployment_evaluator(list(chosen))[0])
    except Exception:
        # Rebinding is a refinement, so evaluator failure must be fail-closed to
        # the already audited baseline rather than changing behavior.
        return fallback(5)
    baseline_cert = float(baseline_exact_action == target_action)
    candidate_cert = float(candidate_exact_action == target_action)
    diag.update({
        "frontier_contrast_rebinding_baseline_exact_action": float(baseline_exact_action),
        "frontier_contrast_rebinding_candidate_exact_action": float(candidate_exact_action),
        "frontier_contrast_rebinding_baseline_exact_certificate": baseline_cert,
        "frontier_contrast_rebinding_candidate_exact_certificate": candidate_cert,
        "frontier_contrast_rebinding_certificate_non_decreasing": float(candidate_cert >= baseline_cert),
    })
    if candidate_exact_action != target_action:
        return fallback(9)

    diag.update({
        "frontier_contrast_rebinding_accepted": 1.0,
        "frontier_contrast_rebinding_reason_code": 0.0,
        "frontier_contrast_rebinding_final_linf_error": float(candidate_error[0]),
        "frontier_contrast_rebinding_final_rms_error": float(candidate_error[1]),
        "frontier_contrast_rebinding_final_exact_certificate": candidate_cert,
        "frontier_contrast_rebinding_changed_atom_count": float(len(set(baseline) ^ set(chosen))),
    })
    return FrontierContrastRebindingResult(selected=list(chosen), diagnostics=dict(diag))
