from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from bdse.planner.frontier_contrast_rebinding import (
    _finite_valid_cost,
    _frontier_star,
    _future_budget_feasible,
)
from bdse.planner.tournament import _decisive_frontier_value_star_residual_numpy


@dataclass(frozen=True)
class ProposalConditionedWitnessRebindingResult:
    """Fixed-budget proposal-conditioned evidence rebinding.

    PCWER is intentionally *operator conditioned*: a unique direct-recovery
    proposal is generated first by the frozen risk-free ICER generator on the
    incumbent AOCC B-set.  Rebinding may then change only which already-queried
    Top-M evidence atoms are transmitted through the same B-sized interface.
    The proposal identity, incumbent identity and selected-local anchor are hard
    contracts; the new evidence view is not allowed to rerank or invent a second
    action.
    """

    selected: list[int]
    proposal_action: int | None
    proposal_lock: bool
    diagnostics: dict[str, Any]


def _witness_errors(
    star: np.ndarray,
    attr: np.ndarray,
    target_star: np.ndarray,
    target_attr: np.ndarray,
    proposal: int,
    incumbent: int,
) -> tuple[float, float, float, float]:
    """Lexicographic proposal/anchor + proposal/incumbent witness error.

    The first two coordinates are the exact DARM+EAF decision contrasts.  The
    second pair retain the EAF attribution-energy statistics that feed the
    frozen ICER/DRC evidence view.  No learned/validation-selected weights are
    introduced: margin witnesses are prioritized, then attribution witnesses,
    with RMS used only as a deterministic secondary criterion.
    """

    s = np.asarray(star, dtype=np.float64).reshape(-1)
    a = np.asarray(attr, dtype=np.float64).reshape(-1)
    ts = np.asarray(target_star, dtype=np.float64).reshape(-1)
    ta = np.asarray(target_attr, dtype=np.float64).reshape(-1)
    q = int(proposal); i = int(incumbent)
    if min(q, i) < 0 or max(q, i) >= min(len(s), len(a), len(ts), len(ta)):
        return (float("inf"),) * 4
    m = np.asarray([s[q], s[q] - s[i]], dtype=np.float64)
    tm = np.asarray([ts[q], ts[q] - ts[i]], dtype=np.float64)
    e = np.asarray([a[q], a[q] - a[i]], dtype=np.float64)
    te = np.asarray([ta[q], ta[q] - ta[i]], dtype=np.float64)
    if not (np.isfinite(m).all() and np.isfinite(tm).all() and np.isfinite(e).all() and np.isfinite(te).all()):
        return (float("inf"),) * 4
    dm = m - tm
    de = e - te
    return (
        float(np.max(np.abs(dm))),
        float(np.max(np.abs(de))),
        float(np.sqrt(np.mean(dm * dm))),
        float(np.sqrt(np.mean(de * de))),
    )


def _strict_lexicographic_improvement(
    candidate: tuple[float, float, float, float],
    baseline: tuple[float, float, float, float],
    eps: float,
) -> bool:
    if not all(np.isfinite(v) for v in (*candidate, *baseline)):
        return False
    for c, b in zip(candidate, baseline):
        if c < b - eps:
            return True
        if c > b + eps:
            return False
    return False


def _attribution_star(
    selected_atoms: list[int] | np.ndarray,
    valid_mask: np.ndarray,
    anchor_action: int,
    atom_factors: np.ndarray,
    action_signed_factors: np.ndarray,
    action_context_factors: np.ndarray,
    scale: float,
) -> np.ndarray:
    _, diag = _decisive_frontier_value_star_residual_numpy(
        selected_atoms,
        valid_mask,
        int(anchor_action),
        atom_factors,
        action_signed_factors,
        action_context_factors,
        scale=float(scale),
    )
    return np.asarray(
        diag.get("_decisive_frontier_value_attribution_scale_star", []),
        dtype=np.float64,
    ).reshape(-1)


def proposal_conditioned_witness_rebind(
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
    proposal_evaluator: Callable[[list[int]], dict[str, Any]] | None,
    rebind_enabled: bool = True,
    structural_bypass: bool = False,
    improvement_epsilon: float = 1e-8,
) -> ProposalConditionedWitnessRebindingResult:
    """Proposal-Conditioned Witness Evidence Rebinding (PCWER).

    Scientific contract:
      * literal selected cardinality/budget is unchanged;
      * candidates come only from the already queried Top-M evidence bank;
      * runtime teacher labels and endpoint regret are never read;
      * only a *single* risk-free ICER direct proposal may condition rebinding;
      * the exact proposal, incumbent and selected-local anchor must survive the
        rebind; otherwise the AOCC B-set is returned unchanged;
      * final ICER is told to confirm/veto that same proposal only, so the new
        evidence view cannot rerank into a second-best action.

    The objective compresses the two witnesses that the downstream recovery
    operator actually needs: proposal-vs-anchor support and proposal-vs-incumbent
    replacement contrast.  It additionally preserves the matching EAF
    attribution-energy witnesses because those statistics enter the frozen DRC
    evidence representation.  This is deliberately narrower than V29 FCR's
    all-challenger reconstruction objective.
    """

    eps = max(float(improvement_epsilon), 0.0)
    baseline = list(dict.fromkeys(int(x) for x in np.asarray(baseline_selected, dtype=np.int64).reshape(-1).tolist()))
    reference = list(dict.fromkeys(int(x) for x in np.asarray(reference_atoms, dtype=np.int64).reshape(-1).tolist()))
    diag: dict[str, Any] = {
        "proposal_conditioned_witness_rebinding_enabled": 1.0,
        "proposal_conditioned_witness_rebinding_attempted": 0.0,
        "proposal_conditioned_witness_rebinding_accepted": 0.0,
        "proposal_conditioned_witness_rebinding_proposal_lock": 0.0,
        "proposal_conditioned_witness_rebinding_teacher_free": 1.0,
        "proposal_conditioned_witness_rebinding_no_new_query": 1.0,
        "proposal_conditioned_witness_rebinding_same_cardinality_required": 1.0,
        "proposal_conditioned_witness_rebinding_exact_proposal_required": 1.0,
        "proposal_conditioned_witness_rebinding_exact_incumbent_required": 1.0,
        "proposal_conditioned_witness_rebinding_exact_anchor_required": 1.0,
        "proposal_conditioned_witness_rebinding_rebind_enabled": float(bool(rebind_enabled)),
        "proposal_conditioned_witness_rebinding_lock_only": float(not bool(rebind_enabled)),
        "proposal_conditioned_witness_rebinding_structural_bypass": float(bool(structural_bypass)),
        "proposal_conditioned_witness_rebinding_baseline_count": float(len(baseline)),
        "proposal_conditioned_witness_rebinding_reference_count": float(len(reference)),
        "proposal_conditioned_witness_rebinding_budget": float(budget),
        "proposal_conditioned_witness_rebinding_reason_code": 0.0,
    }

    # Stable reason codes: 0 accepted; 1 structural/no direct proposal;
    # 2 malformed shapes/factors; 3 baseline/reference/budget contract;
    # 4 proposal-evaluator failure; 5 greedy infeasible;
    # 6 no strict witness improvement; 7 reserved;
    # 8 proposal/incumbent/anchor preservation mismatch; 10 proposal-lock-only control.
    def fallback(reason: int, **extra: Any) -> ProposalConditionedWitnessRebindingResult:
        diag["proposal_conditioned_witness_rebinding_reason_code"] = float(reason)
        diag.update(extra)
        return ProposalConditionedWitnessRebindingResult(
            selected=list(baseline), proposal_action=None, proposal_lock=False, diagnostics=dict(diag)
        )

    def fallback_locked(reason: int, proposal_action: int, **extra: Any) -> ProposalConditionedWitnessRebindingResult:
        # Once the frozen risk-free generator has produced a valid direct proposal,
        # every later PCWER failure must fail closed on the *evidence rebind only*.
        # The proposal itself remains locked so downstream DRC can only confirm/veto
        # that same action.  Dropping the lock here would silently reopen candidate
        # generation and violate the V30 operator contract.
        diag["proposal_conditioned_witness_rebinding_reason_code"] = float(reason)
        diag["proposal_conditioned_witness_rebinding_proposal_lock"] = 1.0
        diag.update(extra)
        return ProposalConditionedWitnessRebindingResult(
            selected=list(baseline), proposal_action=int(proposal_action), proposal_lock=True, diagnostics=dict(diag)
        )

    if structural_bypass or not baseline or not reference or len(baseline) > len(reference):
        return fallback(1)
    if proposal_evaluator is None:
        return fallback(4)

    j0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    g = np.asarray(predicted_atom_costs, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    costs = np.asarray(atom_budget_costs, dtype=np.float64).reshape(-1)
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    pair_delta = np.asarray(pair_atom_delta, dtype=np.float32)
    atom_f = None if frontier_value_atom_factors is None else np.asarray(frontier_value_atom_factors, dtype=np.float32)
    signed_f = None if frontier_value_action_signed_factors is None else np.asarray(frontier_value_action_signed_factors, dtype=np.float32)
    context_f = None if frontier_value_action_context_factors is None else np.asarray(frontier_value_action_context_factors, dtype=np.float32)
    if (
        g.ndim != 2 or g.shape[1] != len(j0) or valid.shape[0] < len(j0)
        or pair_delta.ndim != 2 or pair_delta.shape[1] < pair_arr.shape[0]
        or atom_f is None or signed_f is None or context_f is None
        or atom_f.ndim != 2 or signed_f.ndim != 2 or context_f.ndim != 2
        or signed_f.shape != context_f.shape or signed_f.shape[0] < len(j0)
        or atom_f.shape[1] <= 0 or signed_f.shape[1] != atom_f.shape[1]
    ):
        return fallback(2)
    max_atom = min(g.shape[0], pair_delta.shape[0], atom_f.shape[0], costs.shape[0])
    ref_set = set(reference)
    if any(x < 0 or x >= max_atom for x in reference) or any(x not in ref_set for x in baseline):
        return fallback(3)
    baseline_cost = float(np.sum(costs[np.asarray(baseline, dtype=np.int64)]))
    if not np.isfinite(baseline_cost) or baseline_cost > float(budget) + eps:
        return fallback(3, proposal_conditioned_witness_rebinding_baseline_cost=baseline_cost)

    try:
        base_eval = dict(proposal_evaluator(list(baseline)))
    except Exception:
        return fallback(4)
    q = int(base_eval.get("proposal_action", -1))
    incumbent = int(base_eval.get("incumbent_action", -1))
    anchor = int(base_eval.get("anchor_action", -1))
    incumbent_admissible = bool(base_eval.get("incumbent_admissible", False))
    if (
        not incumbent_admissible or min(q, incumbent, anchor) < 0
        or max(q, incumbent, anchor) >= len(j0) or q in {incumbent, anchor}
    ):
        return fallback(
            1,
            proposal_conditioned_witness_rebinding_baseline_proposal_action=float(q),
            proposal_conditioned_witness_rebinding_baseline_incumbent_action=float(incumbent),
            proposal_conditioned_witness_rebinding_baseline_anchor_action=float(anchor),
        )
    diag.update({
        "proposal_conditioned_witness_rebinding_baseline_proposal_action": float(q),
        "proposal_conditioned_witness_rebinding_baseline_incumbent_action": float(incumbent),
        "proposal_conditioned_witness_rebinding_baseline_anchor_action": float(anchor),
    })

    if not bool(rebind_enabled):
        diag.update({
            "proposal_conditioned_witness_rebinding_reason_code": 10.0,
            "proposal_conditioned_witness_rebinding_proposal_lock": 1.0,
            "proposal_conditioned_witness_rebinding_candidate_count": float(len(baseline)),
            "proposal_conditioned_witness_rebinding_candidate_cost": float(baseline_cost),
            "proposal_conditioned_witness_rebinding_cardinality_preserved": 1.0,
            "proposal_conditioned_witness_rebinding_budget_preserved": 1.0,
        })
        return ProposalConditionedWitnessRebindingResult(
            selected=list(baseline), proposal_action=int(q), proposal_lock=True, diagnostics=dict(diag)
        )

    target_star = _frontier_star(
        predicted_base_cost=j0,
        pair_indices=pair_arr,
        pair_atom_delta=pair_delta,
        selected_atoms=reference,
        valid_mask=valid,
        anchor_action=anchor,
        normalize_margins=normalize_margins,
        margin_scale=margin_scale,
        predicted_atom_costs=g,
        pair_delta_includes_local=pair_delta_includes_local,
        frontier_value_atom_factors=atom_f,
        frontier_value_action_signed_factors=signed_f,
        frontier_value_action_context_factors=context_f,
        frontier_value_scale=frontier_value_scale,
    )
    target_attr = _attribution_star(reference, valid, anchor, atom_f, signed_f, context_f, frontier_value_scale)
    baseline_star = _frontier_star(
        predicted_base_cost=j0,
        pair_indices=pair_arr,
        pair_atom_delta=pair_delta,
        selected_atoms=baseline,
        valid_mask=valid,
        anchor_action=anchor,
        normalize_margins=normalize_margins,
        margin_scale=margin_scale,
        predicted_atom_costs=g,
        pair_delta_includes_local=pair_delta_includes_local,
        frontier_value_atom_factors=atom_f,
        frontier_value_action_signed_factors=signed_f,
        frontier_value_action_context_factors=context_f,
        frontier_value_scale=frontier_value_scale,
    )
    baseline_attr = _attribution_star(baseline, valid, anchor, atom_f, signed_f, context_f, frontier_value_scale)
    baseline_error = _witness_errors(baseline_star, baseline_attr, target_star, target_attr, q, incumbent)
    for name, val in zip(("margin_linf", "attribution_linf", "margin_rms", "attribution_rms"), baseline_error):
        diag[f"proposal_conditioned_witness_rebinding_baseline_{name}_error"] = float(val)
        diag[f"proposal_conditioned_witness_rebinding_final_{name}_error"] = float(val)

    # Additive DARM/EAF contributions for the two fixed witnesses.  As in V29,
    # EAF contributions are evaluated at the final retained cardinality, making
    # every forward-step candidate exact for a completed B-sized set.
    from bdse.planner.tournament import _pair_delta_margin_matrix

    empty = _pair_delta_margin_matrix(
        j0, pair_arr, pair_delta, [], valid,
        normalize_margins=bool(normalize_margins), margin_scale=float(margin_scale),
        predicted_atom_costs=g, pair_delta_includes_local=bool(pair_delta_includes_local),
    )
    darm_base = np.asarray(empty[anchor], dtype=np.float64)
    darm_contrib: dict[int, np.ndarray] = {}
    for atom_idx in reference:
        one = _pair_delta_margin_matrix(
            j0, pair_arr, pair_delta, [atom_idx], valid,
            normalize_margins=bool(normalize_margins), margin_scale=float(margin_scale),
            predicted_atom_costs=g, pair_delta_includes_local=bool(pair_delta_includes_local),
        )
        darm_contrib[atom_idx] = np.asarray(one[anchor], dtype=np.float64) - darm_base

    final_count = len(baseline)
    rank = int(atom_f.shape[1])
    bounded = np.tanh(atom_f[np.asarray(reference, dtype=np.int64)]).astype(np.float64)
    pair_sym = np.tanh(
        context_f[anchor][None, :] + context_f[: len(j0)]
        + context_f[anchor][None, :] * context_f[: len(j0)]
    ).astype(np.float64)
    signed_diff = np.asarray(signed_f[: len(j0)] - signed_f[anchor][None, :], dtype=np.float64)
    pair_vec = pair_sym * signed_diff
    raw_eaf = np.einsum("nr,kr->nk", bounded, pair_vec, optimize=True)
    raw_eaf *= float(frontier_value_scale) / np.sqrt(max(float(final_count * rank), 1.0))
    eaf_contrib = {idx: raw_eaf[pos] for pos, idx in enumerate(reference)}

    diag["proposal_conditioned_witness_rebinding_attempted"] = 1.0
    chosen: list[int] = []
    sum_darm = np.zeros_like(darm_base, dtype=np.float64)
    sum_eaf = np.zeros_like(darm_base, dtype=np.float64)
    sum_eaf_sq = np.zeros_like(darm_base, dtype=np.float64)
    spent = 0.0
    for step in range(final_count):
        best_key: tuple[float, float, float, float, float, int] | None = None
        best_idx: int | None = None
        best_darm = best_eaf = best_sq = None
        remaining_slots = final_count - step - 1
        for atom_idx in reference:
            if atom_idx in chosen:
                continue
            c = float(costs[atom_idx])
            if not np.isfinite(c) or c < 0.0 or spent + c > float(budget) + eps:
                continue
            rest = [float(costs[j]) for j in reference if j != atom_idx and j not in chosen]
            if not _future_budget_feasible(
                chosen_cost=spent + c,
                remaining_costs=rest,
                remaining_slots=remaining_slots,
                budget=float(budget),
                eps=eps,
            ):
                continue
            cd = sum_darm + darm_contrib[atom_idx]
            ce = sum_eaf + eaf_contrib[atom_idx]
            cs = sum_eaf_sq + eaf_contrib[atom_idx] * eaf_contrib[atom_idx]
            star = darm_base + cd + ce
            attr = np.sqrt(np.maximum(cs, 0.0))
            err = _witness_errors(star, attr, target_star, target_attr, q, incumbent)
            key = (*err, c, int(atom_idx))
            if best_key is None or key < best_key:
                best_key = key; best_idx = int(atom_idx)
                best_darm = cd; best_eaf = ce; best_sq = cs
        if best_idx is None or best_darm is None or best_eaf is None or best_sq is None:
            return fallback_locked(5, q, proposal_conditioned_witness_rebinding_partial_count=float(len(chosen)))
        chosen.append(best_idx); spent += float(costs[best_idx])
        sum_darm = best_darm; sum_eaf = best_eaf; sum_eaf_sq = best_sq

    candidate_star = _frontier_star(
        predicted_base_cost=j0, pair_indices=pair_arr, pair_atom_delta=pair_delta,
        selected_atoms=chosen, valid_mask=valid, anchor_action=anchor,
        normalize_margins=normalize_margins, margin_scale=margin_scale,
        predicted_atom_costs=g, pair_delta_includes_local=pair_delta_includes_local,
        frontier_value_atom_factors=atom_f,
        frontier_value_action_signed_factors=signed_f,
        frontier_value_action_context_factors=context_f,
        frontier_value_scale=frontier_value_scale,
    )
    candidate_attr = _attribution_star(chosen, valid, anchor, atom_f, signed_f, context_f, frontier_value_scale)
    candidate_error = _witness_errors(candidate_star, candidate_attr, target_star, target_attr, q, incumbent)
    candidate_cost = float(np.sum(costs[np.asarray(chosen, dtype=np.int64)]))
    for name, val in zip(("margin_linf", "attribution_linf", "margin_rms", "attribution_rms"), candidate_error):
        diag[f"proposal_conditioned_witness_rebinding_candidate_{name}_error"] = float(val)
    diag.update({
        "proposal_conditioned_witness_rebinding_candidate_count": float(len(chosen)),
        "proposal_conditioned_witness_rebinding_candidate_cost": candidate_cost,
        "proposal_conditioned_witness_rebinding_cardinality_preserved": float(len(chosen) == len(baseline)),
        "proposal_conditioned_witness_rebinding_budget_preserved": float(candidate_cost <= float(budget) + eps),
    })
    if not _strict_lexicographic_improvement(candidate_error, baseline_error, eps):
        return fallback_locked(6, q)

    try:
        cand_eval = dict(proposal_evaluator(list(chosen)))
    except Exception:
        return fallback_locked(4, q)
    cq = int(cand_eval.get("proposal_action", -1))
    ci = int(cand_eval.get("incumbent_action", -1))
    ca = int(cand_eval.get("anchor_action", -1))
    cadm = bool(cand_eval.get("incumbent_admissible", False))
    diag.update({
        "proposal_conditioned_witness_rebinding_candidate_proposal_action": float(cq),
        "proposal_conditioned_witness_rebinding_candidate_incumbent_action": float(ci),
        "proposal_conditioned_witness_rebinding_candidate_anchor_action": float(ca),
        "proposal_conditioned_witness_rebinding_candidate_incumbent_admissible": float(cadm),
    })
    if cq != q or ci != incumbent or ca != anchor or not cadm:
        return fallback_locked(8, q)

    changed = len(set(baseline) ^ set(chosen))
    diag.update({
        "proposal_conditioned_witness_rebinding_accepted": 1.0,
        "proposal_conditioned_witness_rebinding_proposal_lock": 1.0,
        "proposal_conditioned_witness_rebinding_reason_code": 0.0,
        "proposal_conditioned_witness_rebinding_changed_atom_count": float(changed),
    })
    for name, val in zip(("margin_linf", "attribution_linf", "margin_rms", "attribution_rms"), candidate_error):
        diag[f"proposal_conditioned_witness_rebinding_final_{name}_error"] = float(val)
    return ProposalConditionedWitnessRebindingResult(
        selected=list(chosen), proposal_action=int(q), proposal_lock=True, diagnostics=dict(diag)
    )
