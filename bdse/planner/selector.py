from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from bdse.planner.pair_screen import build_runtime_pairs_from_base


@dataclass(slots=True)
class SelectionResult:
    selected: list[int]
    objective_value: float
    pair_indices: np.ndarray
    pair_weights: np.ndarray
    diagnostics: dict[str, Any]


def _finite_cost_for_margin(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float32).copy()
    finite = np.isfinite(cost)
    if finite.any():
        fill = float(np.nanmax(cost[finite]) + 1e6)
    else:
        fill = 1e6
    cost[~finite] = fill
    return cost


def _as_bool_mask(mask: np.ndarray | None, E: int) -> np.ndarray:
    if mask is None:
        return np.zeros((E,), dtype=bool)
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.shape[0] < E:
        arr = np.pad(arr, (0, E - arr.shape[0]), constant_values=False)
    return arr[:E]


def _spent_for(selected: list[int] | np.ndarray, costs: np.ndarray) -> float:
    total = 0.0
    for i in np.asarray(selected, dtype=np.int64).reshape(-1):
        if 0 <= int(i) < costs.shape[0] and np.isfinite(costs[int(i)]):
            total += float(costs[int(i)])
    return float(total)


def margin_normalization_scale(values: np.ndarray, min_scale: float = 100.0, quantile: float = 0.75) -> float:
    """Robust scalar used to convert raw teacher/base costs into margin units.

    Runtime does not know teacher costs, so it uses the current base rival-margin
    spread.  Training uses the same minimum scale on supervised pair margins.
    """
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = np.abs(arr[np.isfinite(arr)])
    if arr.size == 0:
        return float(min_scale)
    q = float(np.clip(float(quantile), 0.5, 0.99))
    return float(max(float(np.quantile(arr, q)), float(min_scale)))


def structural_safety_mask(
    hard_mask: np.ndarray | None,
    family_ids: np.ndarray | None,
    active_mask: np.ndarray | None,
    include_feasibility: bool = True,
) -> np.ndarray:
    """Atoms that are kept by rule/guard rather than learned interaction ranking."""
    E = 0
    for arr in (hard_mask, family_ids, active_mask):
        if arr is not None:
            E = max(E, int(np.asarray(arr).reshape(-1).shape[0]))
    if E == 0:
        return np.zeros((0,), dtype=bool)
    out = _as_bool_mask(hard_mask, E)
    if include_feasibility and family_ids is not None:
        fam = np.asarray(family_ids, dtype=np.int64).reshape(-1)
        if fam.shape[0] < E:
            fam = np.pad(fam, (0, E - fam.shape[0]), constant_values=0)
        # FAMILY_NAMES['feasibility'] == 1.  Keep the numeric dependency local so
        # selector.py stays independent from evidence_queries.
        out |= fam[:E] == 1
    if active_mask is not None:
        out &= _as_bool_mask(active_mask, E)
    return out


def reserve_topm_candidates(
    topm: np.ndarray | list[int],
    candidate_mask: np.ndarray,
    scores: np.ndarray,
    max_size: int,
    min_slots: int,
    *,
    protected_mask: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Reserve complementary candidates in a fixed-size proposal pool.

    The function only exchanges entries inside Top-M; it never increases M or
    the number of neural pair queries.  Group-aware ordering prevents all slots
    from being consumed by multiple evidence types attached to one agent.
    """
    mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    E = int(mask.shape[0])
    score = np.asarray(scores, dtype=np.float32).reshape(-1)
    if score.shape[0] < E:
        score = np.pad(score, (0, E - score.shape[0]), constant_values=-np.inf)
    score = score[:E]
    protected = _as_bool_mask(protected_mask, E)
    groups = np.full((E,), -1, dtype=np.int64)
    if group_ids is not None:
        raw = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        groups[: min(E, raw.shape[0])] = raw[: min(E, raw.shape[0])]

    size = max(0, int(max_size))
    current: list[int] = []
    seen: set[int] = set()
    for raw_i in np.asarray(topm, dtype=np.int64).reshape(-1).tolist():
        i = int(raw_i)
        if 0 <= i < E and i not in seen:
            current.append(i)
            seen.add(i)
        if size and len(current) >= size:
            break

    def gid(i: int) -> int:
        g = int(groups[i])
        return g if g >= 0 else -(i + 1)

    available = np.flatnonzero(mask).tolist()
    by_group: dict[int, list[int]] = {}
    for i in available:
        by_group.setdefault(gid(int(i)), []).append(int(i))
    group_best = [max(ids, key=lambda i: (float(score[i]), -int(i))) for ids in by_group.values()]
    group_best.sort(key=lambda i: (-float(score[i]), int(i)))
    rest = sorted(available, key=lambda i: (-float(score[i]), int(i)))
    order = group_best + [i for i in rest if i not in set(group_best)]
    target = min(max(0, int(min_slots)), len(available), size if size > 0 else len(available))

    for i in order:
        if sum(1 for j in current if mask[j]) >= target:
            break
        if i in seen:
            continue
        removable = [j for j in current if not protected[j] and not mask[j]]
        if not removable:
            break
        rm = min(removable, key=lambda j: (float(score[j]), -int(j)))
        pos = current.index(rm)
        current[pos] = int(i)
        seen.remove(rm)
        seen.add(int(i))

    current.sort(key=lambda i: (-float(score[i]), int(i)))
    if size:
        current = current[:size]
    diag = {
        "reserved_available": int(mask.sum()),
        "reserved_selected": int(sum(1 for i in current if mask[i])),
        "reserved_target": int(target),
        "reserved_distinct_groups": int(len({gid(i) for i in current if mask[i]})),
    }
    return np.asarray(current, dtype=np.int64), diag


def restrict_topm_to_decision_evidence(
    topm: np.ndarray | list[int],
    decision_mask: np.ndarray,
    proposal_scores: np.ndarray,
    max_size: int,
    *,
    family_ids: np.ndarray | None = None,
    min_family_slots: dict[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove structural-safety atoms from Top-M and refill in fixed size.

    The safety channel is evaluated deterministically and therefore must not
    consume the learned decision-evidence proposal pool.  This helper preserves
    the original M and performs only within-pool replacement.
    """
    allowed = np.asarray(decision_mask, dtype=bool).reshape(-1)
    E = int(allowed.shape[0])
    score = np.asarray(proposal_scores, dtype=np.float32).reshape(-1)
    if score.shape[0] < E:
        score = np.pad(score, (0, E - score.shape[0]), constant_values=-np.inf)
    score = score[:E]
    size = max(0, int(max_size))
    current: list[int] = []
    seen: set[int] = set()
    for raw in np.asarray(topm, dtype=np.int64).reshape(-1).tolist():
        i = int(raw)
        if 0 <= i < E and allowed[i] and i not in seen:
            current.append(i); seen.add(i)
        if size and len(current) >= size:
            break

    fam = np.full((E,), -999, dtype=np.int64)
    if family_ids is not None:
        raw_f = np.asarray(family_ids, dtype=np.int64).reshape(-1)
        fam[: min(E, raw_f.shape[0])] = raw_f[: min(E, raw_f.shape[0])]
    slots = {int(k): max(0, int(v)) for k, v in (min_family_slots or {}).items()}
    for fid, target in slots.items():
        have = sum(1 for i in current if int(fam[i]) == fid)
        if have >= target:
            continue
        candidates = [i for i in np.flatnonzero(allowed & (fam == fid)).tolist() if i not in seen]
        candidates.sort(key=lambda i: (-float(score[i]), int(i)))
        for i in candidates[: max(0, target - have)]:
            if size and len(current) >= size:
                removable = [j for j in current if int(fam[j]) != fid]
                if not removable:
                    break
                rm = min(removable, key=lambda j: (float(score[j]), -int(j)))
                current.remove(rm); seen.remove(rm)
            current.append(int(i)); seen.add(int(i))

    fill = sorted(np.flatnonzero(allowed).tolist(), key=lambda i: (-float(score[i]), int(i)))
    for i in fill:
        if i in seen:
            continue
        current.append(int(i)); seen.add(int(i))
        if size and len(current) >= size:
            break
    current = current[:size] if size else current
    current.sort(key=lambda i: (-float(score[i]), int(i)))
    return np.asarray(current, dtype=np.int64), {
        "decision_topm_available": int(allowed.sum()),
        "decision_topm_selected": int(len(current)),
        "structural_atoms_excluded_from_topm": int(E - int(allowed.sum())),
    }


def _complete_safety_aware_selection(
    selected: list[int] | np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    atom_active_mask: np.ndarray | None = None,
    mandatory_atom_mask: np.ndarray | None = None,
    mandatory_quota: int = 0,
    min_selected_atoms: int = 0,
    force_fill_budget: bool = False,
    utility: np.ndarray | None = None,
    prioritize_mandatory_fill: bool = True,
    family_ids: np.ndarray | None = None,
    decision_family_ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    decision_family_quota: int = 0,
    interaction_family_ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    interaction_family_quota: int = 0,
    soft_interaction_mask: np.ndarray | None = None,
    soft_interaction_quota: int = 0,
    interaction_group_ids: np.ndarray | None = None,
    interaction_utility: np.ndarray | None = None,
) -> tuple[list[int], float, dict[str, Any]]:
    """Post-process greedy acquisition for BDSE-v5.

    The old selector stopped as soon as the LCB/certificate gain became non-positive.
    That is appropriate for certification, but not for evidence acquisition when the
    margin model is still being calibrated.  This routine keeps the selected set within
    budget while (i) forcing safety/hard atoms into the certificate support and (ii)
    filling a minimum/budget quota with the most useful available atoms.
    """
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    E = int(costs.shape[0])
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else _as_bool_mask(atom_active_mask, E)
    active &= np.isfinite(costs) & (costs > 0)
    mandatory = _as_bool_mask(mandatory_atom_mask, E) & active
    util = np.zeros((E,), dtype=np.float32) if utility is None else np.asarray(utility, dtype=np.float32).reshape(-1)
    if util.shape[0] < E:
        util = np.pad(util, (0, E - util.shape[0]), constant_values=0.0)
    util = util[:E]
    fam = None
    if family_ids is not None:
        fam = np.asarray(family_ids, dtype=np.int64).reshape(-1)
        if fam.shape[0] < E:
            fam = np.pad(fam, (0, E - fam.shape[0]), constant_values=-999)
        fam = fam[:E]
    decision_family_set = set(int(x) for x in np.asarray(decision_family_ids if decision_family_ids is not None else [], dtype=np.int64).reshape(-1).tolist())
    decision_family = np.zeros((E,), dtype=bool)
    if fam is not None and decision_family_set:
        decision_family = np.asarray([int(x) in decision_family_set for x in fam.tolist()], dtype=bool) & active
    interaction_family_set = set(int(x) for x in np.asarray(interaction_family_ids if interaction_family_ids is not None else [], dtype=np.int64).reshape(-1).tolist())
    interaction_family = np.zeros((E,), dtype=bool)
    if fam is not None and interaction_family_set:
        interaction_family = np.asarray([int(x) in interaction_family_set for x in fam.tolist()], dtype=bool) & active
    # v35 DICE: hard occupancy atoms belong to the interaction family, but they
    # are already protected by ``mandatory_quota``.  Counting them again toward
    # an interaction quota made v34 appear to reserve interaction evidence while
    # selecting almost no complementary TTC/gap evidence.  Keep a distinct soft
    # interaction mask so safety and interaction coverage are complementary.
    if soft_interaction_mask is None:
        soft_interaction = interaction_family & ~mandatory
    else:
        soft_interaction = _as_bool_mask(soft_interaction_mask, E) & active & ~mandatory
    group_ids = np.full((E,), -1, dtype=np.int64)
    if interaction_group_ids is not None:
        raw_groups = np.asarray(interaction_group_ids, dtype=np.int64).reshape(-1)
        group_ids[: min(E, raw_groups.shape[0])] = raw_groups[: min(E, raw_groups.shape[0])]
    inter_util = util.copy()
    if interaction_utility is not None:
        raw_inter_util = np.asarray(interaction_utility, dtype=np.float32).reshape(-1)
        inter_util[: min(E, raw_inter_util.shape[0])] = raw_inter_util[: min(E, raw_inter_util.shape[0])]

    out: list[int] = []
    seen: set[int] = set()
    for raw in np.asarray(selected, dtype=np.int64).reshape(-1).tolist():
        i = int(raw)
        if 0 <= i < E and active[i] and i not in seen:
            out.append(i)
            seen.add(i)

    def spent() -> float:
        return _spent_for(out, costs)

    def can_add(i: int) -> bool:
        return i not in seen and active[i] and spent() + float(costs[i]) <= float(budget) + 1e-6

    # 1) Mandatory hard/safety atoms first.  If the greedy trace already filled the
    # budget with non-mandatory atoms, replace the weakest tail atoms.  This makes
    # hard-evidence recall a structural property rather than a hope learned from BCE.
    mandatory_order = sorted(np.flatnonzero(mandatory).tolist(), key=lambda i: (-float(util[i]), float(costs[i]), int(i)))
    forced_added = 0
    quota = max(0, int(mandatory_quota))
    for i in mandatory_order:
        if i in seen:
            forced_added += 1
            if quota and forced_added >= quota:
                break
            continue
        while not can_add(i):
            removable = [j for j in out if not mandatory[j]]
            if not removable:
                break
            # Remove lowest-utility non-mandatory atom first.
            rm = min(removable, key=lambda j: (float(util[j]), -int(j)))
            out.remove(rm)
            seen.remove(rm)
        if can_add(i):
            out.append(int(i))
            seen.add(int(i))
            forced_added += 1
        if quota and forced_added >= quota:
            break

    # 2) Complementary soft-interaction floor with agent diversity.  The first
    # pass gives different interacting agents a chance before taking a second
    # TTC/gap atom for the same agent.  The ranking is deployment-only and uses
    # the direction-invariant predicted influence supplied by the runtime
    # selector; it never sees teacher costs or logged futures.
    soft_quota = max(0, int(soft_interaction_quota))
    soft_available = np.flatnonzero(soft_interaction).tolist()

    def _soft_group(i: int) -> int:
        gid = int(group_ids[i]) if 0 <= i < E else -1
        # Unknown groups remain distinct instead of collapsing all non-agent
        # interaction atoms into a single pseudo-agent.
        return gid if gid >= 0 else -(int(i) + 1)

    def _mandatory_count() -> int:
        return int(sum(1 for j in out if mandatory[j]))

    def _remove_for_soft(candidate: int, prefer_duplicate_group: bool) -> bool:
        selected_soft_groups = [_soft_group(j) for j in out if soft_interaction[j]]
        group_counts: dict[int, int] = {}
        for gid in selected_soft_groups:
            group_counts[gid] = group_counts.get(gid, 0) + 1
        removable: list[int] = []
        for j in out:
            if mandatory[j] and _mandatory_count() - 1 < quota:
                continue
            if soft_interaction[j]:
                if not prefer_duplicate_group or group_counts.get(_soft_group(j), 0) <= 1:
                    continue
            removable.append(int(j))
        if not removable:
            return False
        rm = min(removable, key=lambda j: (float(inter_util[j]), float(util[j]), -int(j)))
        out.remove(rm)
        seen.remove(rm)
        return True

    if soft_quota > 0 and soft_available:
        # Diversity pass: best candidate from each not-yet represented agent.
        by_group: dict[int, list[int]] = {}
        for i in soft_available:
            by_group.setdefault(_soft_group(i), []).append(int(i))
        group_best = [
            max(ids, key=lambda i: (float(inter_util[i]), float(util[i]), -float(costs[i]), -int(i)))
            for ids in by_group.values()
        ]
        group_best.sort(key=lambda i: (-float(inter_util[i]), -float(util[i]), float(costs[i]), int(i)))
        represented = {_soft_group(i) for i in out if soft_interaction[i]}
        diversity_target = min(soft_quota, len(group_best))
        for i in group_best:
            if len(represented) >= diversity_target:
                break
            gid = _soft_group(i)
            if gid in represented:
                continue
            while not can_add(i):
                if not _remove_for_soft(i, prefer_duplicate_group=True):
                    if not _remove_for_soft(i, prefer_duplicate_group=False):
                        break
            if can_add(i):
                out.append(int(i))
                seen.add(int(i))
                represented.add(gid)

        # Cardinality pass: fill the remaining soft interaction floor by
        # two-sided influence, regardless of sign in the retained orientation.
        soft_selected = int(sum(1 for i in out if soft_interaction[i]))
        soft_order = sorted(
            soft_available,
            key=lambda i: (-float(inter_util[i]), -float(util[i]), float(costs[i]), int(i)),
        )
        for i in soft_order:
            if soft_selected >= soft_quota:
                break
            if i in seen:
                continue
            while not can_add(i):
                if not _remove_for_soft(i, prefer_duplicate_group=False):
                    break
            if can_add(i):
                out.append(int(i))
                seen.add(int(i))
                soft_selected += 1

    # 3) Legacy all-interaction reservation.  Retained for backward-compatible
    # ablations; v35 configs set this to zero and use the complementary soft floor.
    # quota can be satisfied almost entirely by feasibility atoms, which is why
    # v33 selected ~14 decision-family atoms yet recovered only ~27% of decisive
    # interaction evidence.  Protect families 2/3 directly, while allowing hard
    # atoms above the mandatory floor to be exchanged rather than permanently
    # starving interaction evidence.
    interaction_quota = max(0, int(interaction_family_quota))
    interaction_added = int(sum(1 for i in out if interaction_family[i]))
    if interaction_quota > 0 and bool(interaction_family.any()):
        interaction_order = sorted(np.flatnonzero(interaction_family).tolist(), key=lambda i: (-float(util[i]), float(costs[i]), int(i)))
        for i in interaction_order:
            if interaction_added >= interaction_quota:
                break
            if i in seen:
                continue
            while not can_add(i):
                mandatory_count = int(sum(1 for j in out if mandatory[j]))
                removable = [
                    j for j in out
                    if not interaction_family[j]
                    and (not mandatory[j] or mandatory_count - 1 >= quota)
                ]
                if not removable:
                    break
                rm = min(removable, key=lambda j: (float(util[j]), -int(j)))
                out.remove(rm)
                seen.remove(rm)
            if can_add(i):
                out.append(int(i))
                seen.add(int(i))
                interaction_added += 1

    # 4) Decision-family reservation.  The hard/feasibility quota protects safety,
    # but v18 showed that it can starve interaction/precedence evidence without
    # improving closed-loop collision/TTC.  Reserve a small, utility-ranked slice
    # for interaction-like families using only deployment-time family ids and
    # proposal/action-rank utility.  This is still budgeted evidence selection; it
    # does not inspect teacher labels or logged futures.
    decision_quota = max(0, int(decision_family_quota))
    decision_added = int(sum(1 for i in out if decision_family[i]))
    if decision_quota > 0 and bool(decision_family.any()):
        decision_order = sorted(np.flatnonzero(decision_family).tolist(), key=lambda i: (-float(util[i]), float(costs[i]), int(i)))
        for i in decision_order:
            if decision_added >= decision_quota:
                break
            if i in seen:
                continue
            while not can_add(i):
                removable = [j for j in out if not mandatory[j] and not decision_family[j]]
                if not removable:
                    break
                rm = min(removable, key=lambda j: (float(util[j]), -int(j)))
                out.remove(rm)
                seen.remove(rm)
            if can_add(i):
                out.append(int(i))
                seen.add(int(i))
                decision_added += 1

    # 5) Acquisition fill.  Keep querying until a minimum support size is reached;
    # optionally fill the whole cost budget.  This prevents early stopping caused by
    # conservative LCBs before the calibration epsilon is reliable.
    fill_target = max(0, int(min_selected_atoms))
    fill_budget = bool(force_fill_budget)
    if bool(prioritize_mandatory_fill):
        filler_key = lambda i: (not bool(mandatory[i]), -float(util[i]), float(costs[i]), int(i))
    else:
        # Safety/hard evidence can be inserted structurally through mandatory_quota.
        # After that quota, do not let all hard atoms consume the residual budget;
        # use the learned interaction/proposal utility to fill the remaining slots.
        filler_key = lambda i: (-float(util[i]), float(costs[i]), int(i))
    # Avoid sorting every active atom when acquisition fill is disabled and the
    # already-selected support satisfies the requested minimum (the v46 path).
    if fill_budget or len(out) < fill_target:
        filler_order = sorted(np.flatnonzero(active).tolist(), key=filler_key)
        for i in filler_order:
            if i in seen:
                continue
            if not can_add(i):
                continue
            if len(out) < fill_target or fill_budget:
                out.append(int(i))
                seen.add(int(i))
            if not fill_budget and len(out) >= fill_target:
                break

    final_spent = spent()
    diag = {
        "mandatory_available": int(mandatory.sum()),
        "mandatory_selected": int(sum(1 for i in out if mandatory[i])),
        "mandatory_quota": int(quota),
        "min_selected_atoms": int(fill_target),
        "force_fill_budget": bool(fill_budget),
        "prioritize_mandatory_fill": bool(prioritize_mandatory_fill),
        "decision_family_quota": int(max(0, int(decision_family_quota))),
        "decision_family_available": int(decision_family.sum()),
        "decision_family_selected": int(sum(1 for i in out if decision_family[i])) if E else 0,
        "interaction_family_quota": int(interaction_quota),
        "interaction_family_available": int(interaction_family.sum()),
        "interaction_family_selected": int(sum(1 for i in out if interaction_family[i])) if E else 0,
        "soft_interaction_quota": int(soft_quota),
        "soft_interaction_available": int(soft_interaction.sum()),
        "soft_interaction_selected": int(sum(1 for i in out if soft_interaction[i])) if E else 0,
        "soft_interaction_distinct_groups": int(len({_soft_group(i) for i in out if soft_interaction[i]})) if E else 0,
        "postfill_selected_atoms": int(len(out)),
        "postfill_spent_budget": float(final_spent),
    }
    return out, float(final_spent), diag


def full_interface_margin(J0: np.ndarray, g: np.ndarray) -> np.ndarray:
    J0 = np.asarray(J0, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    full_cost = _finite_cost_for_margin(J0 + g.sum(axis=0))
    return full_cost[None, :] - full_cost[:, None]


def budgeted_margin(J0: np.ndarray, g: np.ndarray, selected: list[int] | np.ndarray) -> np.ndarray:
    J0 = np.asarray(J0, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    if len(selected):
        cost = J0 + g[np.asarray(selected, dtype=np.int64)].sum(axis=0)
    else:
        cost = J0.copy()
    cost = _finite_cost_for_margin(cost)
    return cost[None, :] - cost[:, None]


def oracle_objective_value(
    selected: list[int] | np.ndarray,
    J_base: np.ndarray,
    g_true: np.ndarray,
    pairs: np.ndarray,
    margins: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Signed certificate objective for labeled teacher pairs.

    For pair (a, b), positive margin means action a should beat b.  The
    certificate score accumulates the signed margin contribution J(b)-J(a).
    Clipping every atom delta before summing can overestimate evidence because
    an atom that supports one pair may hurt another pair.
    """
    selected_arr = np.asarray(selected, dtype=np.int64)
    pair_arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pair_arr.size == 0:
        return 0.0
    caps = np.asarray(margins, dtype=np.float32).reshape(-1)[: pair_arr.shape[0]]
    w = np.asarray(weights, dtype=np.float32).reshape(-1)[: pair_arr.shape[0]]
    if caps.shape[0] != pair_arr.shape[0]:
        caps = np.zeros((pair_arr.shape[0],), dtype=np.float32)
    if w.shape[0] != pair_arr.shape[0]:
        w = np.ones((pair_arr.shape[0],), dtype=np.float32)
    a = pair_arr[:, 0]
    b = pair_arr[:, 1]
    margin = np.asarray(J_base, dtype=np.float32)[b] - np.asarray(J_base, dtype=np.float32)[a]
    if selected_arr.size:
        valid_sel = selected_arr[(selected_arr >= 0) & (selected_arr < np.asarray(g_true).shape[0])]
        if valid_sel.size:
            g = np.asarray(g_true, dtype=np.float32)
            margin = margin + (g[valid_sel[:, None], b[None, :]] - g[valid_sel[:, None], a[None, :]]).sum(axis=0)
    cert = np.minimum(caps, np.maximum(margin, 0.0))
    return float(np.sum(w * cert, dtype=np.float64))



def _greedy_cover_from_pair_delta(
    atom_delta: np.ndarray,
    base_margin: np.ndarray,
    caps: np.ndarray,
    pair_weights: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    atom_active_mask: np.ndarray | None = None,
) -> tuple[list[int], float, float]:
    """Greedy signed certificate coverage over pair margins.

    ``base_margin[p]`` and ``atom_delta[i, p]`` are signed contributions to
    J(b)-J(a) for pair p.  Value is sum_p w_p min(cap_p, max(margin_p, 0)).
    """
    margin = np.asarray(base_margin, dtype=np.float32).reshape(-1).copy()
    caps = np.asarray(caps, dtype=np.float32).reshape(-1)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    atom_delta = np.asarray(atom_delta, dtype=np.float32)
    E = int(atom_delta.shape[0]) if atom_delta.ndim == 2 else int(np.asarray(atom_budget_costs).shape[0])
    if atom_delta.ndim != 2 or atom_delta.shape[1] != margin.shape[0]:
        atom_delta = np.zeros((E, margin.shape[0]), dtype=np.float32)
    if caps.shape[0] != margin.shape[0]:
        caps = np.zeros((margin.shape[0],), dtype=np.float32)
    if weights.shape[0] != margin.shape[0]:
        weights = np.ones((margin.shape[0],), dtype=np.float32)
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else np.asarray(atom_active_mask, dtype=bool).reshape(-1).copy()
    if active.shape[0] < E:
        active = np.pad(active, (0, E - active.shape[0]), constant_values=False)
    active = active[:E]
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)

    def value(m: np.ndarray) -> float:
        return float(np.sum(weights * np.minimum(caps, np.maximum(m, 0.0)), dtype=np.float64))

    current = value(margin)
    selected: list[int] = []
    spent = 0.0
    while bool(active.any()):
        best: tuple[int, float, float] | None = None
        best_key = (-np.inf, -np.inf, np.inf)
        for i in np.flatnonzero(active):
            idx = int(i)
            c = float(costs[idx])
            if not np.isfinite(c) or spent + c > float(budget) + 1e-6:
                continue
            gain = float(value(margin + atom_delta[idx]) - current)
            key = (gain / max(c, 1e-6), gain, -idx)
            if key > best_key:
                best_key = key
                best = (idx, gain, c)
        if best is None or best_key[1] <= 1e-9:
            break
        idx, gain, c = best
        selected.append(idx)
        active[idx] = False
        spent += c
        margin += atom_delta[idx]
        current += float(gain)
    return selected, float(current), float(spent)


def _selector_pair_caps(
    base_delta: np.ndarray,
    safety_b: np.ndarray,
    gamma_max: float,
    eta_pred: float,
    cap_mode: str = "legacy_abs",
    boundary_cap: float | None = None,
    base_margin_cap_multiplier: float = 1.0,
) -> np.ndarray:
    """Caps for evidence selection over pair-conditioned margins.

    The legacy cap ``abs(base)+eta`` can spend budget on already-easy pairs.
    For fixed-budget decision evidence, the useful target is usually the
    decision boundary: query atoms until a pair is certified just beyond zero,
    not until its already-large margin becomes even larger.  Safety pairs remain
    high-cap because hard feasibility evidence can legitimately dominate.
    """
    bd = np.asarray(base_delta, dtype=np.float32).reshape(-1)
    sb = np.asarray(safety_b, dtype=bool).reshape(-1)
    if sb.shape[0] != bd.shape[0]:
        sb = np.zeros_like(bd, dtype=bool)
    mode = str(cap_mode or "legacy_abs").lower()
    eta = max(float(eta_pred), 1e-3)
    gamma = max(float(gamma_max), eta)
    if boundary_cap is None:
        boundary = eta
    else:
        boundary = max(float(boundary_cap), 1e-3)
    hard_boundary_modes = {"boundary", "flip", "near_boundary", "decision_boundary"}
    soft_boundary_modes = {
        "soft_boundary",
        "hybrid_boundary",
        # v19: action-rank is an objective family, not a license to use the old
        # abs(base)+eta cap.  With legacy caps the selector spends budget making
        # already-easy base pairs even easier, which is exactly the v18 failure:
        # true ActionRank became active but closed-loop progress dropped.  Treat
        # action-rank/flip-rank modes as boundary-aware certificate modes unless
        # a config explicitly asks for legacy_abs.
        "action_rank",
        "action_flip_rank",
        "tournament_rank",
        "safety_gated_action_rank",
        "lcb_action_rank_hybrid",
        "hybrid_lcb_action_rank",
        "safe_action_rank",
        "flip_rank",
        "fliprank",
        "flip_boundary_rank",
    }
    if mode in hard_boundary_modes:
        non_safety = np.full_like(bd, boundary, dtype=np.float32)
    elif mode in soft_boundary_modes:
        non_safety = np.minimum(
            np.maximum(np.abs(bd) * max(float(base_margin_cap_multiplier), 0.0) + eta, eta),
            boundary,
        ).astype(np.float32)
    else:
        non_safety = np.minimum(np.maximum(np.abs(bd) + eta, 1e-3), gamma).astype(np.float32)
    return np.where(sb, gamma, non_safety).astype(np.float32)


def _normalized_action_utility_cost(action_utility_cost: np.ndarray | None) -> np.ndarray | None:
    """Robust [0,4] normalized per-action deployment utility cost.

    Lower values are better.  The selector uses this only as a secondary
    acquisition prior; certificate/boundary evidence still dominates.
    """
    if action_utility_cost is None:
        return None
    uc = np.asarray(action_utility_cost, dtype=np.float32).reshape(-1)
    finite = uc[np.isfinite(uc)]
    if finite.size == 0:
        return None
    lo = float(np.min(finite))
    hi = float(np.quantile(finite, 0.90)) if finite.size >= 4 else float(np.max(finite))
    scale = max(hi - lo, float(np.std(finite)), 1e-3)
    return np.clip((uc - lo) / scale, 0.0, 4.0).astype(np.float32)


def _pair_utility_advantage(pair_indices: np.ndarray, action_utility_cost: np.ndarray | None) -> np.ndarray | None:
    """Positive when a directed pair supports a lower-utility-cost action."""
    util = _normalized_action_utility_cost(action_utility_cost)
    if util is None:
        return None
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    if pair_arr.size == 0:
        return None
    a = pair_arr[:, 0]
    b = pair_arr[:, 1]
    out = np.zeros((pair_arr.shape[0],), dtype=np.float32)
    valid = (a >= 0) & (b >= 0) & (a < util.shape[0]) & (b < util.shape[0])
    if bool(valid.any()):
        # Positive means action a has lower deployment cost than action b, so
        # evidence that increases margin a>b is more valuable for closed loop.
        out[valid] = np.maximum(util[b[valid]] - util[a[valid]], 0.0).astype(np.float32)
    return out


def _adaptive_hybrid_lcb_fraction(
    base_margin: np.ndarray,
    pair_weights: np.ndarray,
    safety_pair_mask: np.ndarray | None,
    pair_atom_variance: np.ndarray | None,
    *,
    base_frac: float,
    min_frac: float = 0.45,
    max_frac: float = 0.80,
    safety_weight: float = 0.25,
    fallback_weight: float = 0.20,
    uncertainty_weight: float = 0.10,
    boundary_action_weight: float = 0.25,
    boundary_tau: float = 0.35,
) -> tuple[float, dict[str, float]]:
    """Risk-calibrated runtime split between LCB seed and ActionRank.

    v22 used the raw density of safety-flagged pairs. In closed-loop runs that
    density was often close to one, so the adaptive allocator saturated at its
    maximum and starved the ActionRank stage. This v23 allocator distinguishes
    safety presence from safety pressure: LCB budget grows only when unsafe pairs
    are also near the decision boundary or already have a wrong-frontier directed
    margin. Boundary mass that is not explained by such safety pressure is
    treated as interaction/action need and is explicitly reserved for ActionRank.
    The formula still uses only deployment-time quantities.
    """
    m = np.asarray(base_margin, dtype=np.float32).reshape(-1)
    w = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    if m.size == 0:
        out = float(np.clip(float(base_frac), float(min_frac), float(max_frac)))
        return out, {
            "adaptive_lcb_frac": out,
            "adaptive_lcb_raw_frac": out,
            "adaptive_safety_density": 0.0,
            "adaptive_safety_pressure": 0.0,
            "adaptive_unsafe_fallback_risk": 0.0,
            "adaptive_fallback_risk": 0.0,
            "adaptive_boundary_density": 0.0,
            "adaptive_uncertainty_density": 0.0,
            "adaptive_action_need": 0.0,
        }
    if w.shape[0] < m.shape[0]:
        w = np.pad(w, (0, m.shape[0] - w.shape[0]), constant_values=1.0)
    w = np.maximum(w[: m.shape[0]], 0.0).astype(np.float32)
    denom = float(np.sum(w, dtype=np.float64))
    if denom <= 1e-9:
        w = np.ones_like(m, dtype=np.float32)
        denom = float(w.size)
    sb = np.zeros_like(m, dtype=bool)
    if safety_pair_mask is not None:
        raw = np.asarray(safety_pair_mask, dtype=bool).reshape(-1)
        if raw.shape[0] < m.shape[0]:
            raw = np.pad(raw, (0, m.shape[0] - raw.shape[0]), constant_values=False)
        sb = raw[: m.shape[0]]

    tau = max(float(boundary_tau), 1e-3)
    near = np.exp(-np.minimum(np.abs(m) / tau, 30.0)).astype(np.float32)
    neg = (m <= 0.0).astype(np.float32)
    sb_f = sb.astype(np.float32)

    safety_density = float(np.sum(w * sb_f, dtype=np.float64) / max(denom, 1e-9))
    # v23: only near-boundary unsafe pairs create LCB pressure. This prevents
    # all safety-labeled rival pairs from monopolizing the seed budget.
    safety_pressure = float(np.sum(w * sb_f * near, dtype=np.float64) / max(denom, 1e-9))
    fallback_risk = float(np.sum(w * neg, dtype=np.float64) / max(denom, 1e-9))
    unsafe_fallback_risk = float(np.sum(w * sb_f * neg, dtype=np.float64) / max(denom, 1e-9))
    boundary_density = float(np.sum(w * near, dtype=np.float64) / max(denom, 1e-9))

    uncertainty_density = 0.0
    if pair_atom_variance is not None:
        pv = np.asarray(pair_atom_variance, dtype=np.float32)
        if pv.ndim == 2 and pv.shape[1] >= m.shape[0] and pv.shape[0] > 0:
            v = np.nanmean(np.maximum(pv[:, : m.shape[0]], 0.0), axis=0).astype(np.float32)
            finite = v[np.isfinite(v)]
            if finite.size:
                scale = max(float(np.quantile(finite, 0.90)), float(np.mean(finite)), 1e-6)
                vn = np.clip(v / scale, 0.0, 2.0) * 0.5
                uncertainty_density = float(np.sum(w * vn, dtype=np.float64) / max(denom, 1e-9))

    # Boundary mass that is not already unsafe-boundary pressure should remain
    # available to ActionRank because it is where interaction/precedence evidence
    # changes the action order.
    action_need = float(max(0.0, boundary_density - safety_pressure))
    raw_frac = (
        float(base_frac)
        + float(safety_weight) * safety_pressure
        + float(fallback_weight) * unsafe_fallback_risk
        + float(uncertainty_weight) * uncertainty_density
        - float(boundary_action_weight) * action_need
    )
    frac = float(np.clip(raw_frac, float(min_frac), float(max_frac)))
    return frac, {
        "adaptive_lcb_frac": frac,
        "adaptive_lcb_raw_frac": float(raw_frac),
        "adaptive_safety_density": safety_density,
        "adaptive_safety_pressure": safety_pressure,
        "adaptive_unsafe_fallback_risk": unsafe_fallback_risk,
        "adaptive_fallback_risk": fallback_risk,
        "adaptive_boundary_density": boundary_density,
        "adaptive_uncertainty_density": uncertainty_density,
        "adaptive_action_need": action_need,
    }

def _flip_rank_value(
    margin: np.ndarray,
    base_delta: np.ndarray,
    caps: np.ndarray,
    weights: np.ndarray,
    *,
    flip_bonus: float = 0.0,
    flip_window: float = 0.5,
    certify_margin: float = 0.0,
    flip_mode: str = "hard",
    flip_temperature: float = 0.08,
) -> float:
    """Capped pair certificate plus an explicit action-order flip bonus.

    The legacy certificate objective rewards positive margin support.  Even with
    a small cap, it can still spend budget on small positive increments that never
    change the pair order.  This objective adds one discrete reward when the
    selected evidence moves a near-boundary pair from not-certified to certified:

        base_margin <= certify_margin  and  selected_margin > certify_margin.

    This is the runtime counterpart of the paper claim that useful interaction
    evidence is evidence that can change pair-conditioned action order.
    """
    m = np.asarray(margin, dtype=np.float32).reshape(-1)
    base = np.asarray(base_delta, dtype=np.float32).reshape(-1)
    cap = np.asarray(caps, dtype=np.float32).reshape(-1)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if cap.shape[0] != m.shape[0]:
        cap = np.zeros_like(m, dtype=np.float32)
    if w.shape[0] != m.shape[0]:
        w = np.ones_like(m, dtype=np.float32)
    cover = np.minimum(np.maximum(cap, 0.0), np.maximum(m, 0.0))
    val = np.sum(np.maximum(w, 0.0) * cover, dtype=np.float64)
    bonus = max(float(flip_bonus), 0.0)
    if bonus > 0.0 and m.size:
        window = max(float(flip_window), 1e-6)
        cert = float(certify_margin)
        mode = str(flip_mode or "hard").lower()
        wpos = np.maximum(w, 0.0)
        if mode in {"smooth", "soft", "soft_crossing", "smooth_flip"}:
            # Continuous crossing utility: reward atoms that move a near-boundary
            # pair toward the certified side, without the brittle all-or-nothing
            # discontinuity of the hard crossing bonus.  This makes greedy
            # acquisition less sensitive to one atom barely crossing zero and
            # better aligned with noisy pair-margin predictions.
            temp = max(float(flip_temperature), 1e-4)
            x0 = np.clip((base - cert) / temp, -30.0, 30.0)
            x1 = np.clip((m - cert) / temp, -30.0, 30.0)
            s0 = 1.0 / (1.0 + np.exp(-x0))
            s1 = 1.0 / (1.0 + np.exp(-x1))
            near_weight = np.exp(-np.minimum(np.abs(base - cert) / window, 30.0))
            improve = np.maximum(s1 - s0, 0.0) * near_weight
            val += bonus * float(np.sum(wpos * improve.astype(np.float32), dtype=np.float64))
        else:
            near = (np.abs(base - cert) <= window) | ((base < cert) & (base > cert - 2.0 * window))
            crossed = near & (base <= cert) & (m > cert)
            val += bonus * float(np.sum(wpos * crossed.astype(np.float32), dtype=np.float64))
    return float(val)


def _greedy_flip_rank_from_pair_delta(
    atom_delta: np.ndarray,
    base_margin: np.ndarray,
    caps: np.ndarray,
    pair_weights: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    atom_active_mask: np.ndarray | None = None,
    *,
    flip_bonus: float = 0.75,
    flip_window: float = 0.5,
    certify_margin: float = 0.0,
    flip_mode: str = "hard",
    flip_temperature: float = 0.08,
) -> tuple[list[int], float, float]:
    """Greedy selector that explicitly values action-order boundary flips.

    It uses signed atom deltas.  A selected atom receives gain not only for capped
    positive support, but also for making a previously non-certified near-boundary
    pair become certified.  This avoids the v14 failure mode where recall improved
    slightly but atoms did not reliably affect final winner-vs-rival order.
    """
    margin = np.asarray(base_margin, dtype=np.float32).reshape(-1).copy()
    base = margin.copy()
    caps_arr = np.asarray(caps, dtype=np.float32).reshape(-1)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    atom_delta = np.asarray(atom_delta, dtype=np.float32)
    E = int(atom_delta.shape[0]) if atom_delta.ndim == 2 else int(np.asarray(atom_budget_costs).shape[0])
    if atom_delta.ndim != 2 or atom_delta.shape[1] != margin.shape[0]:
        atom_delta = np.zeros((E, margin.shape[0]), dtype=np.float32)
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else np.asarray(atom_active_mask, dtype=bool).reshape(-1).copy()
    if active.shape[0] < E:
        active = np.pad(active, (0, E - active.shape[0]), constant_values=False)
    active = active[:E]
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)

    def value(m: np.ndarray) -> float:
        return _flip_rank_value(
            m, base, caps_arr, weights,
            flip_bonus=float(flip_bonus),
            flip_window=float(flip_window),
            certify_margin=float(certify_margin),
            flip_mode=str(flip_mode),
            flip_temperature=float(flip_temperature),
        )

    current = value(margin)
    selected: list[int] = []
    spent = 0.0
    while bool(active.any()):
        best: tuple[int, float, float] | None = None
        best_key = (-np.inf, -np.inf, np.inf)
        for i in np.flatnonzero(active):
            idx = int(i)
            c = float(costs[idx])
            if not np.isfinite(c) or spent + c > float(budget) + 1e-6:
                continue
            gain = float(value(margin + atom_delta[idx]) - current)
            key = (gain / max(c, 1e-6), gain, -idx)
            if key > best_key:
                best_key = key
                best = (idx, gain, c)
        if best is None or best_key[1] <= 1e-9:
            break
        idx, gain, c = best
        selected.append(idx)
        active[idx] = False
        spent += c
        margin += atom_delta[idx]
        current += float(gain)
    return selected, float(current), float(spent)


def _flip_gain_atom_utility(
    delta: np.ndarray,
    base_delta: np.ndarray,
    caps: np.ndarray,
    weights: np.ndarray,
    *,
    flip_bonus: float = 0.0,
    flip_window: float = 0.5,
    certify_margin: float = 0.0,
    flip_mode: str = "hard",
    flip_temperature: float = 0.08,
) -> np.ndarray:
    """Single-step gain for post-fill atom ranking under the selector objective."""
    d = np.asarray(delta, dtype=np.float32)
    if d.ndim != 2 or d.size == 0:
        return np.zeros((d.shape[0] if d.ndim else 0,), dtype=np.float32)
    base = np.asarray(base_delta, dtype=np.float32).reshape(-1)
    if base.shape[0] != d.shape[1]:
        return np.maximum(d, 0.0).mean(axis=1).astype(np.float32)
    before = _flip_rank_value(base, base, caps, weights, flip_bonus=flip_bonus, flip_window=flip_window, certify_margin=certify_margin, flip_mode=flip_mode, flip_temperature=flip_temperature)
    gains = []
    for i in range(d.shape[0]):
        after = _flip_rank_value(base + d[i], base, caps, weights, flip_bonus=flip_bonus, flip_window=flip_window, certify_margin=certify_margin, flip_mode=flip_mode, flip_temperature=flip_temperature)
        gains.append(max(float(after - before), 0.0))
    denom = max(float(np.sum(np.maximum(np.asarray(weights, dtype=np.float32).reshape(-1), 0.0))), 1e-6)
    return (np.asarray(gains, dtype=np.float32) / denom).astype(np.float32)



def _softmin_np_local(vals: np.ndarray, tau: float) -> float:
    vals = np.asarray(vals, dtype=np.float32).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    if float(tau) <= 0.0:
        return float(np.min(vals))
    t = max(float(tau), 1e-6)
    m = float(np.min(vals))
    # stable -tau*log(sum(exp(-x/tau)))
    return float(m - t * np.log(np.sum(np.exp(-(vals - m) / t))))


def _action_rank_value(
    margin: np.ndarray,
    base_delta: np.ndarray,
    caps: np.ndarray,
    weights: np.ndarray,
    pair_indices: np.ndarray,
    *,
    certificate_weight: float = 1.0,
    action_score_weight: float = 1.0,
    action_gap_weight: float = 0.5,
    action_flip_weight: float = 0.5,
    action_softmin_tau: float = 0.2,
    certify_margin: float = 0.0,
    action_utility_cost: np.ndarray | None = None,
    action_utility_weight: float = 0.0,
    action_pair_utility_weight: float = 0.0,
) -> float:
    """Selector objective aligned with the final pair-conditioned tournament.

    FlipRank rewarded any near-boundary pair crossing.  In closed loop this can
    spread a fixed evidence budget over pairs that never affect the chosen action.
    This action-rank objective keeps the capped certificate term but adds a
    differentiable proxy for the final tournament score: build a soft-min score
    for each action from its outgoing queried pair margins, then reward the best
    action score and the best-vs-second gap.  A small action-flip term rewards
    cases where the selected evidence changes the best action under this pair
    graph.  The objective remains teacher-free and fixed-budget.
    """
    m = np.asarray(margin, dtype=np.float32).reshape(-1)
    base = np.asarray(base_delta, dtype=np.float32).reshape(-1)
    cap = np.asarray(caps, dtype=np.float32).reshape(-1)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    P = int(min(m.shape[0], base.shape[0], cap.shape[0], w.shape[0], pair_arr.shape[0]))
    if P <= 0:
        return 0.0
    m = m[:P]; base = base[:P]; cap = cap[:P]; w = w[:P]; pair_arr = pair_arr[:P]
    wpos = np.maximum(w, 0.0)

    # Match run_pair_conditioned_tournament(): if both directions are queried,
    # the final margin matrix uses an antisymmetric projection instead of trusting
    # two independent directed predictions.  The selector should value evidence
    # in the same geometry, otherwise it can optimize pair signs that will be
    # projected away before the tournament action is chosen.
    def antisym_project(x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float32).reshape(-1).copy()
        pos = {(int(a), int(b)): int(i) for i, (a, b) in enumerate(pair_arr.tolist())}
        seen: set[tuple[int, int]] = set()
        for (a, b), i in pos.items():
            if (a, b) in seen or (b, a) in seen:
                continue
            j = pos.get((b, a), None)
            if j is not None:
                v = 0.5 * (float(x[i]) - float(x[j]))
                y[i] = v
                y[j] = -v
                seen.add((a, b)); seen.add((b, a))
        return y

    m_eval = antisym_project(m)
    base_eval = antisym_project(base)

    pair_eff_w = wpos
    pair_util_w = max(float(action_pair_utility_weight), 0.0)
    if pair_util_w > 0.0 and action_utility_cost is not None:
        adv = _pair_utility_advantage(pair_arr, action_utility_cost)
        if adv is not None and adv.shape[0] >= P:
            # This is the main v20 change: utility is not only a per-action
            # tie-break after the tournament.  It also reweights directed
            # boundary certificates toward actions that the final closed-loop
            # utility refinement would prefer.
            pair_eff_w = pair_eff_w * (1.0 + pair_util_w * np.maximum(adv[:P], 0.0))

    cover = np.minimum(np.maximum(cap, 0.0), np.maximum(m_eval, 0.0))
    val = float(max(float(certificate_weight), 0.0)) * float(np.sum(pair_eff_w * cover, dtype=np.float64))

    actions = np.unique(pair_arr[:, 0])
    if actions.size == 0:
        return float(val)

    utility_weight = max(float(action_utility_weight), 0.0)
    utility_norm = _normalized_action_utility_cost(action_utility_cost) if utility_weight > 0.0 else None

    def scores_for(x: np.ndarray) -> dict[int, float]:
        out: dict[int, float] = {}
        for a_raw in actions.tolist():
            a = int(a_raw)
            mask = pair_arr[:, 0] == a
            if not bool(mask.any()):
                continue
            # pair weights should influence the score without changing margin units;
            # repeat high-weight pairs by subtracting log-weight inside softmin.
            vals = x[mask].astype(np.float32)
            wm = np.maximum(wpos[mask], 1e-3).astype(np.float32)
            vals_eff = vals - float(action_softmin_tau) * np.log(wm)
            score = _softmin_np_local(vals_eff, float(action_softmin_tau))
            if utility_norm is not None and 0 <= a < utility_norm.shape[0] and np.isfinite(float(utility_norm[a])):
                # Lower deployment utility cost is better.  This term makes the
                # selector value the same lexicographic object used by final
                # utility refinement: preserve certificates first, then avoid
                # no-progress/low-utility actions inside a certified band.
                score -= utility_weight * float(utility_norm[a])
            out[a] = float(score)
        return out

    s_cur = scores_for(m_eval)
    if s_cur:
        cur_vals = np.asarray(list(s_cur.values()), dtype=np.float32)
        order = np.sort(cur_vals)
        top = float(order[-1])
        second = float(order[-2]) if order.size >= 2 else 0.0
        val += max(float(action_score_weight), 0.0) * top
        val += max(float(action_gap_weight), 0.0) * max(top - second, 0.0)
        flip_w = max(float(action_flip_weight), 0.0)
        if flip_w > 0.0:
            s_base = scores_for(base_eval)
            if s_base:
                best_base = max(s_base, key=lambda a: (float(s_base[a]), -int(a)))
                best_cur = max(s_cur, key=lambda a: (float(s_cur[a]), -int(a)))
                if int(best_cur) != int(best_base):
                    # Reward action-level flips only when the new winner is at least
                    # certified against the old one by the queried pair graph when
                    # that directed pair is available.
                    ok = True
                    idx = np.flatnonzero((pair_arr[:, 0] == int(best_cur)) & (pair_arr[:, 1] == int(best_base)))
                    if idx.size:
                        ok = bool(np.max(m_eval[idx]) >= float(certify_margin))
                    if ok:
                        val += flip_w
    return float(val)


def _greedy_action_rank_from_pair_delta(
    atom_delta: np.ndarray,
    base_margin: np.ndarray,
    caps: np.ndarray,
    pair_weights: np.ndarray,
    pair_indices: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    atom_active_mask: np.ndarray | None = None,
    *,
    certificate_weight: float = 1.0,
    action_score_weight: float = 1.0,
    action_gap_weight: float = 0.5,
    action_flip_weight: float = 0.5,
    action_softmin_tau: float = 0.2,
    certify_margin: float = 0.0,
    action_utility_cost: np.ndarray | None = None,
    action_utility_weight: float = 0.0,
    action_pair_utility_weight: float = 0.0,
    action_rank_fast_greedy: bool = False,
    family_ids: np.ndarray | None = None,
    decision_family_ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    decision_family_boost: float = 0.0,
) -> tuple[list[int], float, float]:
    margin = np.asarray(base_margin, dtype=np.float32).reshape(-1).copy()
    base = margin.copy()
    caps_arr = np.asarray(caps, dtype=np.float32).reshape(-1)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    atom_delta = np.asarray(atom_delta, dtype=np.float32)
    E = int(atom_delta.shape[0]) if atom_delta.ndim == 2 else int(np.asarray(atom_budget_costs).shape[0])
    if atom_delta.ndim != 2 or atom_delta.shape[1] != margin.shape[0]:
        atom_delta = np.zeros((E, margin.shape[0]), dtype=np.float32)
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else np.asarray(atom_active_mask, dtype=bool).reshape(-1).copy()
    if active.shape[0] < E:
        active = np.pad(active, (0, E - active.shape[0]), constant_values=False)
    active = active[:E]
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)

    if bool(action_rank_fast_greedy):
        # Closed-loop diagnostics showed v19 ActionRank reduced query count but
        # made the nuPlan loop slow because every greedy step evaluated every
        # candidate through Python dictionaries and antisymmetry rebuilds.  This
        # vectorized frontier objective keeps the certificate/flip geometry and
        # adds utility-aware directed-pair weighting, while avoiding the inner
        # action-score loop during closed-loop planning.
        pair_eff_w = np.maximum(weights, 0.0).astype(np.float32)
        pair_util_w = max(float(action_pair_utility_weight), 0.0)
        if pair_util_w > 0.0:
            adv = _pair_utility_advantage(pair_arr, action_utility_cost)
            if adv is not None and adv.shape[0] == pair_eff_w.shape[0]:
                pair_eff_w = pair_eff_w * (1.0 + pair_util_w * np.maximum(adv, 0.0).astype(np.float32))
        cert_w = max(float(certificate_weight), 0.0)
        flip_w = max(float(action_flip_weight), 0.0)
        score_w = max(float(action_score_weight), 0.0)
        gap_w = max(float(action_gap_weight), 0.0)
        cert = float(certify_margin)
        selected: list[int] = []
        spent = 0.0
        current_cover = np.minimum(np.maximum(caps_arr, 0.0), np.maximum(margin, 0.0))
        current = cert_w * float(np.sum(pair_eff_w * current_cover, dtype=np.float64))
        family_gain_mult = np.ones((E,), dtype=np.float32)
        family_boost = max(float(decision_family_boost), 0.0)
        if family_boost > 0.0 and family_ids is not None and decision_family_ids is not None:
            fam = np.asarray(family_ids, dtype=np.int64).reshape(-1)
            if fam.shape[0] < E:
                fam = np.pad(fam, (0, E - fam.shape[0]), constant_values=-999999)
            decision_set = set(map(int, np.asarray(decision_family_ids, dtype=np.int64).reshape(-1).tolist()))
            if decision_set:
                mask = np.asarray([int(x) in decision_set for x in fam[:E]], dtype=bool)
                family_gain_mult[mask] = 1.0 + family_boost
        while bool(active.any()):
            active_idx = np.flatnonzero(active)
            if active_idx.size == 0:
                break
            feasible: list[int] = []
            for i in active_idx.tolist():
                c = float(costs[int(i)])
                if np.isfinite(c) and spent + c <= float(budget) + 1e-6:
                    feasible.append(int(i))
            if not feasible:
                break
            idx_arr = np.asarray(feasible, dtype=np.int64)
            trial_margin = margin[None, :] + atom_delta[idx_arr]
            trial_cover = np.minimum(np.maximum(caps_arr[None, :], 0.0), np.maximum(trial_margin, 0.0))
            gain_vec = cert_w * np.sum(pair_eff_w[None, :] * (trial_cover - current_cover[None, :]), axis=1, dtype=np.float64).astype(np.float64)
            if flip_w > 0.0:
                cap_win = np.maximum(caps_arr[None, :], 1e-3)
                crossed = (margin[None, :] <= cert) & (trial_margin > cert)
                near = (np.abs(margin[None, :] - cert) <= cap_win) | ((margin[None, :] < cert) & (margin[None, :] > cert - 2.0 * cap_win))
                gain_vec += flip_w * np.sum(pair_eff_w[None, :] * (crossed & near).astype(np.float32), axis=1, dtype=np.float64)
            if score_w > 0.0 or gap_w > 0.0:
                # Cheap action-rank surrogate: favor atoms that increase margins
                # on high-priority frontier pairs.  The exact soft-min action
                # score remains available when action_rank_fast_greedy=false.
                pos_improve = np.maximum(trial_margin - margin[None, :], 0.0)
                gain_vec += 0.25 * (score_w + gap_w) * np.sum(pair_eff_w[None, :] * pos_improve, axis=1, dtype=np.float64)
            gain_vec = gain_vec * family_gain_mult[idx_arr].astype(np.float64)
            ratios = gain_vec / np.maximum(costs[idx_arr].astype(np.float64), 1e-6)
            # Deterministic tie-breaking: ratio, gain, lower atom index.
            order = np.lexsort((idx_arr, -gain_vec, -ratios))
            best_pos = int(order[0]) if order.size else -1
            if best_pos < 0 or float(gain_vec[best_pos]) <= 1e-9:
                break
            idx = int(idx_arr[best_pos])
            selected.append(idx)
            active[idx] = False
            spent += float(costs[idx])
            margin += atom_delta[idx]
            current_cover = np.minimum(np.maximum(caps_arr, 0.0), np.maximum(margin, 0.0))
            current += float(gain_vec[best_pos])
        return selected, float(current), float(spent)

    def value(m: np.ndarray) -> float:
        return _action_rank_value(
            m,
            base,
            caps_arr,
            weights,
            pair_arr,
            certificate_weight=certificate_weight,
            action_score_weight=action_score_weight,
            action_gap_weight=action_gap_weight,
            action_flip_weight=action_flip_weight,
            action_softmin_tau=action_softmin_tau,
            certify_margin=certify_margin,
            action_utility_cost=action_utility_cost,
            action_utility_weight=action_utility_weight,
            action_pair_utility_weight=action_pair_utility_weight,
        )

    current = value(margin)
    selected: list[int] = []
    spent = 0.0
    while bool(active.any()):
        best: tuple[int, float, float] | None = None
        best_key = (-np.inf, -np.inf, np.inf)
        for i in np.flatnonzero(active):
            idx = int(i)
            c = float(costs[idx])
            if not np.isfinite(c) or spent + c > float(budget) + 1e-6:
                continue
            gain = float(value(margin + atom_delta[idx]) - current)
            key = (gain / max(c, 1e-6), gain, -idx)
            if key > best_key:
                best_key = key
                best = (idx, gain, c)
        if best is None or best_key[1] <= 1e-9:
            break
        idx, gain, c = best
        selected.append(idx)
        active[idx] = False
        spent += c
        margin += atom_delta[idx]
        current += float(gain)
    return selected, float(current), float(spent)


def _hybrid_lcb_action_rank_from_pair_delta(
    atom_delta: np.ndarray,
    base_margin: np.ndarray,
    action_caps: np.ndarray,
    lcb_caps: np.ndarray,
    pair_weights: np.ndarray,
    pair_indices: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    atom_active_mask: np.ndarray | None = None,
    *,
    atom_pair_variance: np.ndarray | None = None,
    beta_uncertainty: float = 0.0,
    epsilon_cal: float = 0.0,
    lambda_info: float = 0.0,
    info_caps: np.ndarray | None = None,
    prior_atom_variance: np.ndarray | None = None,
    family_ids: np.ndarray | None = None,
    family_budget_caps: np.ndarray | None = None,
    hybrid_lcb_budget_frac: float = 0.55,
    hybrid_protect_lcb_seed: bool = True,
    hybrid_min_action_budget_frac: float = 0.0,
    hybrid_max_lcb_seed_atoms: int = 0,
    adaptive_hybrid_lcb_budget: bool = False,
    adaptive_lcb_min_frac: float = 0.45,
    adaptive_lcb_max_frac: float = 0.80,
    adaptive_lcb_safety_weight: float = 0.25,
    adaptive_lcb_fallback_weight: float = 0.20,
    adaptive_lcb_uncertainty_weight: float = 0.10,
    adaptive_lcb_boundary_action_weight: float = 0.25,
    adaptive_lcb_boundary_tau: float = 0.35,
    safety_pair_mask: np.ndarray | None = None,
    decision_family_ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    decision_family_boost: float = 0.0,
    certificate_weight: float = 1.0,
    action_score_weight: float = 1.0,
    action_gap_weight: float = 0.5,
    action_flip_weight: float = 0.5,
    action_softmin_tau: float = 0.2,
    certify_margin: float = 0.0,
    action_utility_cost: np.ndarray | None = None,
    action_utility_weight: float = 0.0,
    action_pair_utility_weight: float = 0.0,
    action_rank_fast_greedy: bool = True,
) -> tuple[list[int], float, float, dict[str, Any]]:
    """Safety-gated hybrid selector: LCB seed first, ActionRank refinement second.

    v20 showed that pure Frontier-ActionRank can improve open-loop teacher match
    while hurting closed-loop safety/progress.  This hybrid keeps the reliable
    LCB/HAB seed that protects hard/feasibility evidence, then spends the residual
    budget on utility-calibrated boundary ActionRank evidence.  It remains a
    budgeted sparse evidence selector and uses only deployment-time predictions.
    """
    delta = np.asarray(atom_delta, dtype=np.float32)
    base = np.asarray(base_margin, dtype=np.float32).reshape(-1)
    E = int(delta.shape[0]) if delta.ndim == 2 else int(np.asarray(atom_budget_costs).reshape(-1).shape[0])
    if delta.ndim != 2 or delta.shape[1] != base.shape[0]:
        delta = np.zeros((E, base.shape[0]), dtype=np.float32)
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else np.asarray(atom_active_mask, dtype=bool).reshape(-1).copy()
    if active.shape[0] < E:
        active = np.pad(active, (0, E - active.shape[0]), constant_values=False)
    active = active[:E]
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)
    adaptive_diag: dict[str, float] = {}
    if bool(adaptive_hybrid_lcb_budget):
        frac, adaptive_diag = _adaptive_hybrid_lcb_fraction(
            base,
            pair_weights,
            safety_pair_mask,
            atom_pair_variance,
            base_frac=float(hybrid_lcb_budget_frac),
            min_frac=float(adaptive_lcb_min_frac),
            max_frac=float(adaptive_lcb_max_frac),
            safety_weight=float(adaptive_lcb_safety_weight),
            fallback_weight=float(adaptive_lcb_fallback_weight),
            uncertainty_weight=float(adaptive_lcb_uncertainty_weight),
            boundary_action_weight=float(adaptive_lcb_boundary_action_weight),
            boundary_tau=float(adaptive_lcb_boundary_tau),
        )
    else:
        frac = float(np.clip(float(hybrid_lcb_budget_frac), 0.0, 1.0))
        adaptive_diag = {"adaptive_lcb_frac": float(frac)}
    # v24: reserve a guaranteed residual ActionRank slice.  v23 still let the
    # adaptive seed consume most of the budget in closed loop, so the decisive
    # interaction/refinement stage had only ~5 atoms on average.  This reserve is
    # a deployment-time budget split, not an oracle label: it only constrains how
    # much budget the safety seed may spend before ActionRank sees the frontier.
    min_action_budget = max(0.0, min(float(budget), float(budget) * max(float(hybrid_min_action_budget_frac), 0.0)))
    seed_budget = min(float(budget) * frac, max(float(budget) - min_action_budget, 0.0))
    if seed_budget <= 1e-6:
        seed_sel: list[int] = []
        seed_value = 0.0
        seed_spent = 0.0
        seed_diag: dict[str, Any] = {}
    else:
        atom_var = np.asarray(atom_pair_variance, dtype=np.float32) if atom_pair_variance is not None else np.zeros_like(delta, dtype=np.float32)
        if atom_var.shape != delta.shape:
            atom_var = np.zeros_like(delta, dtype=np.float32)
        base_var = np.zeros((base.shape[0],), dtype=np.float32)
        if isinstance(prior_atom_variance, np.ndarray):
            prior_pair = prior_atom_variance if np.asarray(prior_atom_variance).shape == delta.shape else None
        elif prior_atom_variance is not None:
            prior_pair = np.full_like(delta, float(prior_atom_variance), dtype=np.float32)
        else:
            prior_pair = None
        seed_sel, seed_value, seed_spent, seed_diag = _uncertainty_aware_greedy_from_pair_delta(
            delta,
            base,
            atom_var,
            base_var,
            lcb_caps,
            pair_weights,
            costs,
            seed_budget,
            active,
            beta_uncertainty=beta_uncertainty,
            epsilon_cal=epsilon_cal,
            lambda_info=lambda_info,
            info_caps=info_caps,
            prior_atom_pair_var=prior_pair,
            family_ids=family_ids,
            family_budget_caps=family_budget_caps,
        )
    max_seed_atoms = max(0, int(hybrid_max_lcb_seed_atoms))
    if max_seed_atoms > 0 and len(seed_sel) > max_seed_atoms:
        seed_sel = list(map(int, seed_sel[:max_seed_atoms]))
    spent = _spent_for(seed_sel, costs)
    residual_budget = max(float(budget) - float(spent), 0.0)
    residual_active = active.copy()
    for i in seed_sel:
        if 0 <= int(i) < residual_active.shape[0]:
            residual_active[int(i)] = False
    margin_after_seed = base.copy()
    if seed_sel:
        idx = np.asarray([i for i in seed_sel if 0 <= int(i) < E], dtype=np.int64)
        if idx.size:
            margin_after_seed = margin_after_seed + delta[idx].sum(axis=0)
    action_sel, action_value, action_spent = _greedy_action_rank_from_pair_delta(
        delta,
        margin_after_seed,
        action_caps,
        pair_weights,
        pair_indices,
        costs,
        residual_budget,
        residual_active,
        certificate_weight=certificate_weight,
        action_score_weight=action_score_weight,
        action_gap_weight=action_gap_weight,
        action_flip_weight=action_flip_weight,
        action_softmin_tau=action_softmin_tau,
        certify_margin=certify_margin,
        action_utility_cost=action_utility_cost,
        action_utility_weight=action_utility_weight,
        action_pair_utility_weight=action_pair_utility_weight,
        action_rank_fast_greedy=action_rank_fast_greedy,
        family_ids=family_ids,
        decision_family_ids=decision_family_ids,
        decision_family_boost=decision_family_boost,
    )
    selected = list(map(int, seed_sel)) + [int(i) for i in action_sel if int(i) not in set(map(int, seed_sel))]
    diag = {
        "hybrid_lcb_budget_frac": float(frac),
        "hybrid_lcb_seed_budget": float(seed_budget),
        "hybrid_min_action_budget": float(min_action_budget),
        "hybrid_max_lcb_seed_atoms": int(max_seed_atoms),
        "hybrid_lcb_seed_spent": float(spent),
        "hybrid_lcb_seed_atoms": int(len(seed_sel)),
        "hybrid_action_spent": float(action_spent),
        "hybrid_action_atoms": int(len(action_sel)),
        "hybrid_protect_lcb_seed": bool(hybrid_protect_lcb_seed),
        "hybrid_adaptive_lcb_budget": bool(adaptive_hybrid_lcb_budget),
        "hybrid_decision_family_boost": float(max(float(decision_family_boost), 0.0)),
        **adaptive_diag,
    }
    for k, v in (seed_diag or {}).items():
        if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
            diag[f"hybrid_lcb_{k}"] = float(v) if not isinstance(v, (bool, np.bool_)) else bool(v)
    return selected, float(seed_value + action_value), float(spent + action_spent), diag


def _action_rank_atom_utility(
    delta: np.ndarray,
    base_delta: np.ndarray,
    caps: np.ndarray,
    weights: np.ndarray,
    pair_indices: np.ndarray,
    *,
    certificate_weight: float = 1.0,
    action_score_weight: float = 1.0,
    action_gap_weight: float = 0.5,
    action_flip_weight: float = 0.5,
    action_softmin_tau: float = 0.2,
    certify_margin: float = 0.0,
    action_utility_cost: np.ndarray | None = None,
    action_utility_weight: float = 0.0,
    action_pair_utility_weight: float = 0.0,
) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float32)
    if d.ndim != 2 or d.size == 0:
        return np.zeros((d.shape[0] if d.ndim else 0,), dtype=np.float32)
    base = np.asarray(base_delta, dtype=np.float32).reshape(-1)
    if base.shape[0] != d.shape[1]:
        return np.maximum(d, 0.0).mean(axis=1).astype(np.float32)
    before = _action_rank_value(base, base, caps, weights, pair_indices, certificate_weight=certificate_weight, action_score_weight=action_score_weight, action_gap_weight=action_gap_weight, action_flip_weight=action_flip_weight, action_softmin_tau=action_softmin_tau, certify_margin=certify_margin, action_utility_cost=action_utility_cost, action_utility_weight=action_utility_weight, action_pair_utility_weight=action_pair_utility_weight)
    gains = []
    for i in range(d.shape[0]):
        after = _action_rank_value(base + d[i], base, caps, weights, pair_indices, certificate_weight=certificate_weight, action_score_weight=action_score_weight, action_gap_weight=action_gap_weight, action_flip_weight=action_flip_weight, action_softmin_tau=action_softmin_tau, certify_margin=certify_margin, action_utility_cost=action_utility_cost, action_utility_weight=action_utility_weight, action_pair_utility_weight=action_pair_utility_weight)
        gains.append(max(float(after - before), 0.0))
    return np.asarray(gains, dtype=np.float32)


def _direction_invariant_interaction_utility(
    delta: np.ndarray,
    base_delta: np.ndarray,
    caps: np.ndarray,
    weights: np.ndarray,
    *,
    boundary_tau: float = 0.35,
    flip_bonus: float = 0.5,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Two-sided deployment influence for canonicalized pair queries.

    v34 queried only one orientation of each unordered action pair.  A negative
    contribution in that retained orientation is nevertheless a positive
    contribution in the reciprocal orientation, and therefore can be decisive
    for the paper's direction-agnostic interaction support set.  This utility
    scores ``abs(delta)`` near the current decision boundary and adds a small
    bonus when the predicted atom crosses that boundary.  It is used only for
    complementary interaction allocation; the final tournament remains signed.
    """
    d = np.asarray(delta, dtype=np.float32)
    if d.ndim != 2 or d.size == 0:
        return np.zeros((d.shape[0] if d.ndim else 0,), dtype=np.float32)
    base = np.asarray(base_delta, dtype=np.float32).reshape(-1)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    cp = np.asarray(caps, dtype=np.float32).reshape(-1)
    if base.shape[0] != d.shape[1]:
        return np.mean(np.abs(d), axis=1).astype(np.float32)
    if w.shape[0] != base.shape[0]:
        w = np.ones_like(base, dtype=np.float32)
    if cp.shape[0] != base.shape[0]:
        cp = np.ones_like(base, dtype=np.float32)
    tau = max(float(boundary_tau), 1e-3)
    boundary_w = np.maximum(w, 0.0) * np.exp(-np.abs(base) / tau)
    if float(boundary_w.sum()) <= 1e-9:
        boundary_w = np.maximum(w, 1e-3)
    cap_mag = np.maximum(np.abs(cp), tau)
    effect = np.minimum(np.abs(d), cap_mag[None, :])
    crossed = (base[None, :] * (base[None, :] + d) <= 0.0) & (np.abs(d) > 1e-6)
    raw = np.sum(
        boundary_w[None, :] * (effect + max(float(flip_bonus), 0.0) * crossed.astype(np.float32) * cap_mag[None, :]),
        axis=1,
        dtype=np.float64,
    ) / max(float(boundary_w.sum()), 1e-9)
    raw = np.asarray(raw, dtype=np.float32)
    active = np.ones((raw.shape[0],), dtype=bool) if active_mask is None else _as_bool_mask(active_mask, raw.shape[0])
    finite = raw[active & np.isfinite(raw)]
    if finite.size:
        scale = max(float(np.quantile(finite, 0.90)), 1e-6)
        raw = np.clip(raw / scale, 0.0, 4.0)
    return np.nan_to_num(raw, nan=0.0, posinf=4.0, neginf=0.0).astype(np.float32)


def _greedy_cover_from_pair_support(
    atom_support: np.ndarray,
    base_support: np.ndarray,
    caps: np.ndarray,
    pair_weights: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    atom_active_mask: np.ndarray | None = None,
) -> tuple[list[int], float, float]:
    """Greedy coverage used by oracle/runtime selectors, vectorized over pairs.

    ``atom_support[i, p]`` is the non-negative margin support contributed by atom
    ``i`` to pair ``p``.  This is algebraically identical to recomputing
    ``oracle_objective_value`` / ``runtime_objective_value`` for every candidate
    atom at every greedy step, but it avoids the Python loop over pairs inside the
    inner loop.  Tie-breaking intentionally mirrors the old implementation:
    maximize ``(gain / cost, gain, -atom_index)`` over sorted active atoms.
    """
    support = np.asarray(base_support, dtype=np.float32).reshape(-1).copy()
    caps = np.asarray(caps, dtype=np.float32).reshape(-1)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    atom_support = np.asarray(atom_support, dtype=np.float32)
    E = int(atom_support.shape[0]) if atom_support.ndim == 2 else int(np.asarray(atom_budget_costs).shape[0])
    if atom_support.ndim != 2 or atom_support.shape[1] != support.shape[0]:
        atom_support = np.zeros((E, support.shape[0]), dtype=np.float32)
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else np.asarray(atom_active_mask, dtype=bool).copy()
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)
    selected: list[int] = []
    spent = 0.0
    clipped = np.minimum(caps, support)
    current = float(np.sum(weights * clipped, dtype=np.float64))

    while bool(active.any()):
        best: tuple[int, float, float] | None = None
        best_key = (-np.inf, -np.inf, np.inf)
        baseline = np.minimum(caps, support)
        for i in np.flatnonzero(active):
            idx = int(i)
            c = float(costs[idx])
            if spent + c > float(budget) + 1e-6:
                continue
            trial = np.minimum(caps, support + atom_support[idx])
            gain = float(np.sum(weights * (trial - baseline), dtype=np.float64))
            key = (gain / max(c, 1e-6), gain, -idx)
            if key > best_key:
                best_key = key
                best = (idx, gain, c)
        if best is None or best_key[1] <= 1e-9:
            break
        idx, gain, c = best
        selected.append(idx)
        active[idx] = False
        spent += c
        support += atom_support[idx]
        current += float(gain)
    return selected, float(current), float(spent)


def _family_constraint_ok(
    idx: int,
    cost: float,
    spent_by_family: dict[int, float],
    family_ids: np.ndarray | None,
    family_budget_caps: np.ndarray | None,
) -> bool:
    if family_ids is None or family_budget_caps is None:
        return True
    fam = np.asarray(family_ids, dtype=np.int64).reshape(-1)
    if idx >= fam.shape[0]:
        return True
    f = int(fam[idx])
    caps = np.asarray(family_budget_caps, dtype=np.float32).reshape(-1)
    if f < 0 or f >= caps.shape[0] or not np.isfinite(caps[f]) or caps[f] <= 0.0:
        return True
    return spent_by_family.get(f, 0.0) + float(cost) <= float(caps[f]) + 1e-6


def _uncertainty_aware_greedy_from_pair_delta(
    atom_delta: np.ndarray,
    base_margin: np.ndarray,
    atom_pair_var: np.ndarray,
    base_pair_var: np.ndarray,
    caps: np.ndarray,
    pair_weights: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    atom_active_mask: np.ndarray | None = None,
    beta_uncertainty: float = 0.0,
    epsilon_cal: float = 0.0,
    lambda_info: float = 0.0,
    info_caps: np.ndarray | None = None,
    prior_atom_pair_var: np.ndarray | None = None,
    family_ids: np.ndarray | None = None,
    family_budget_caps: np.ndarray | None = None,
) -> tuple[list[int], float, float, dict[str, Any]]:
    """Greedy Eq. (HAB objective) over signed margins and uncertainty."""
    delta = np.asarray(atom_delta, dtype=np.float32)
    mu = np.asarray(base_margin, dtype=np.float32).reshape(-1).copy()
    var = np.asarray(base_pair_var, dtype=np.float32).reshape(-1).copy()
    if delta.ndim != 2 or delta.shape[1] != mu.shape[0]:
        E = int(np.asarray(atom_budget_costs).shape[0])
        delta = np.zeros((E, mu.shape[0]), dtype=np.float32)
    else:
        E = int(delta.shape[0])
    atom_var = np.asarray(atom_pair_var, dtype=np.float32)
    if atom_var.ndim != 2 or atom_var.shape != delta.shape:
        atom_var = np.zeros_like(delta, dtype=np.float32)
    caps = np.asarray(caps, dtype=np.float32).reshape(-1)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    if caps.shape[0] != mu.shape[0]:
        caps = np.zeros((mu.shape[0],), dtype=np.float32)
    if weights.shape[0] != mu.shape[0]:
        weights = np.ones((mu.shape[0],), dtype=np.float32)
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else np.asarray(atom_active_mask, dtype=bool).reshape(-1).copy()
    if active.shape[0] < E:
        active = np.pad(active, (0, E - active.shape[0]), constant_values=False)
    active = active[:E]
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)

    beta = max(float(beta_uncertainty), 0.0)
    eps_cal = max(float(epsilon_cal), 0.0)
    lam_info = max(float(lambda_info), 0.0)
    if info_caps is None:
        info_caps_arr = np.ones_like(mu, dtype=np.float32)
    else:
        info_caps_arr = np.asarray(info_caps, dtype=np.float32).reshape(-1)
        if info_caps_arr.shape[0] != mu.shape[0]:
            info_caps_arr = np.ones_like(mu, dtype=np.float32)
    prior_var = np.asarray(prior_atom_pair_var, dtype=np.float32) if prior_atom_pair_var is not None else None
    if prior_var is None or prior_var.shape != atom_var.shape:
        prior_var = (np.maximum(atom_var, 0.0) + 1.0).astype(np.float32)

    def objective(m: np.ndarray, u: np.ndarray, info_accum: np.ndarray) -> float:
        lcb = m - beta * np.sqrt(np.maximum(u, 0.0)) - eps_cal
        cert = np.minimum(caps, np.maximum(lcb, 0.0))
        if lam_info > 0:
            info = np.minimum(info_caps_arr, np.maximum(info_accum, 0.0))
            val = cert + lam_info * info
        else:
            val = cert
        return float(np.sum(weights * val, dtype=np.float64))

    info_state = np.zeros_like(mu, dtype=np.float32)
    current = objective(mu, var, info_state)
    selected: list[int] = []
    spent = 0.0
    spent_by_family: dict[int, float] = {}
    family_arr = np.asarray(family_ids, dtype=np.int64).reshape(-1) if family_ids is not None else None
    while bool(active.any()):
        best: tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
        best_key = (-np.inf, -np.inf, np.inf)
        for i in np.flatnonzero(active):
            idx = int(i)
            c = float(costs[idx])
            if not np.isfinite(c) or spent + c > float(budget) + 1e-6:
                continue
            if not _family_constraint_ok(idx, c, spent_by_family, family_arr, family_budget_caps):
                continue
            new_mu = mu + delta[idx]
            new_var = var + np.maximum(atom_var[idx], 0.0)
            new_info = info_state + np.maximum(0.0, prior_var[idx] - atom_var[idx])
            val = objective(new_mu, new_var, new_info)
            gain = float(val - current)
            key = (gain / max(c, 1e-6), gain, -idx)
            if key > best_key:
                best_key = key
                best = (idx, gain, c, new_mu, new_var, new_info)
        if best is None or best_key[1] <= 1e-9:
            break
        idx, gain, c, mu, var, info_state = best
        selected.append(idx)
        active[idx] = False
        spent += c
        if family_arr is not None and idx < family_arr.shape[0]:
            f = int(family_arr[idx])
            spent_by_family[f] = spent_by_family.get(f, 0.0) + c
        current += float(gain)
    diagnostics = {
        "spent_budget": float(spent),
        "family_spent": {int(k): float(v) for k, v in spent_by_family.items()},
        "beta_uncertainty": float(beta),
        "epsilon_cal": float(eps_cal),
        "lambda_info": float(lam_info),
    }
    return selected, float(current), float(spent), diagnostics


def oracle_greedy_selector(
    J_base: np.ndarray,
    g_true: np.ndarray,
    pairs: np.ndarray,
    margins: np.ndarray,
    weights: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    atom_active_mask: np.ndarray | None = None,
) -> SelectionResult:
    pair_arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    E = int(np.asarray(g_true).shape[0])
    if pair_arr.size == 0:
        selected, current, spent = _greedy_cover_from_pair_support(
            np.zeros((E, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            atom_budget_costs,
            budget,
            atom_active_mask,
        )
    else:
        a = pair_arr[:, 0]
        b = pair_arr[:, 1]
        base_delta = np.asarray(J_base, dtype=np.float32)[b] - np.asarray(J_base, dtype=np.float32)[a]
        atom_delta = np.asarray(g_true, dtype=np.float32)[:, b] - np.asarray(g_true, dtype=np.float32)[:, a]
        selected, current, spent = _greedy_cover_from_pair_delta(
            atom_delta,
            base_delta,
            np.asarray(margins, dtype=np.float32).reshape(-1)[: pair_arr.shape[0]],
            np.asarray(weights, dtype=np.float32).reshape(-1)[: pair_arr.shape[0]],
            atom_budget_costs,
            budget,
            atom_active_mask,
        )
    return SelectionResult(
        selected=selected,
        objective_value=current,
        pair_indices=pair_arr,
        pair_weights=np.asarray(weights, dtype=np.float32),
        diagnostics={"spent_budget": spent, "budget": float(budget), "mode": "oracle_greedy"},
    )


def build_predicted_pairs(
    predicted_full_margin: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    L_infer: int,
    eta_pred: float,
) -> tuple[np.ndarray, np.ndarray]:
    M = np.asarray(predicted_full_margin, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    K = len(valid)
    total_cost_proxy = -M.mean(axis=1)
    pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    for a in range(K):
        if not valid[a]:
            continue
        candidates = [b for b in range(K) if b != a and valid[b]]
        ranked = sorted(candidates, key=lambda b: (float(total_cost_proxy[b]), abs(float(M[a, b])), int(b)))[:L_infer]
        near = [b for b in candidates if abs(float(M[a, b])) < eta_pred]
        safety = [b for b in candidates if runtime_safety_flags[b]]
        rivals = sorted(set(ranked + near + safety), key=lambda b: (abs(float(M[a, b])), b))[: max(L_infer, len(safety))]
        for b in rivals:
            if M[a, b] > 0:
                w = 1.0
                if abs(float(M[a, b])) < eta_pred:
                    w += 1.0
                if runtime_safety_flags[b]:
                    w += 2.0
                pairs.append((a, b))
                weights.append(w)
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return np.asarray(pairs, dtype=np.int64), np.asarray(weights, dtype=np.float32)


def runtime_objective_value(
    selected: list[int] | np.ndarray,
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    pair_indices: np.ndarray,
    pair_weights: np.ndarray,
    gamma_max: float,
) -> float:
    selected_arr = np.asarray(selected, dtype=np.int64)
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    if pair_arr.size == 0:
        return 0.0
    M_full = full_interface_margin(predicted_base_cost, predicted_atom_costs)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)[: pair_arr.shape[0]]
    if weights.shape[0] != pair_arr.shape[0]:
        weights = np.ones((pair_arr.shape[0],), dtype=np.float32)
    a = pair_arr[:, 0]
    b = pair_arr[:, 1]
    gamma = np.minimum(np.maximum(M_full[a, b], 0.0), float(gamma_max)).astype(np.float32)
    margin = np.asarray(predicted_base_cost, dtype=np.float32)[b] - np.asarray(predicted_base_cost, dtype=np.float32)[a]
    if selected_arr.size:
        valid_sel = selected_arr[(selected_arr >= 0) & (selected_arr < np.asarray(predicted_atom_costs).shape[0])]
        if valid_sel.size:
            g = np.asarray(predicted_atom_costs, dtype=np.float32)
            margin = margin + (g[valid_sel[:, None], b[None, :]] - g[valid_sel[:, None], a[None, :]]).sum(axis=0)
    cert = np.minimum(gamma, np.maximum(margin, 0.0))
    return float(np.sum(weights * cert, dtype=np.float64))


def full_prescore_greedy_selector(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    atom_budget_costs: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    budget: float,
    L_infer: int = 16,
    gamma_max: float = 100.0,
    eta_pred: float = 1.0,
    lambda_near: float = 1.0,
    lambda_safety: float = 2.0,
    atom_active_mask: np.ndarray | None = None,
) -> SelectionResult:
    """Legacy full-interface selector kept only for ablations."""
    M = full_interface_margin(predicted_base_cost, predicted_atom_costs)
    pairs, base_weights = build_predicted_pairs(M, valid_mask, runtime_safety_flags, L_infer, eta_pred)
    if len(base_weights):
        pair_weights = np.ones_like(base_weights)
        for i, (a, b) in enumerate(pairs):
            pair_weights[i] += float(lambda_near) * float(abs(M[a, b]) < eta_pred)
            pair_weights[i] += float(lambda_safety) * float(runtime_safety_flags[b])
    else:
        pair_weights = base_weights

    E = int(predicted_atom_costs.shape[0])
    if len(pairs):
        a = pairs[:, 0]
        b = pairs[:, 1]
        base_delta = np.asarray(predicted_base_cost, dtype=np.float32)[b] - np.asarray(predicted_base_cost, dtype=np.float32)[a]
        atom_delta = np.asarray(predicted_atom_costs, dtype=np.float32)[:, b] - np.asarray(predicted_atom_costs, dtype=np.float32)[:, a]
        caps = np.minimum(np.maximum(M[a, b], 0.0), float(gamma_max)).astype(np.float32)
    else:
        base_delta = np.zeros((0,), dtype=np.float32)
        atom_delta = np.zeros((E, 0), dtype=np.float32)
        caps = np.zeros((0,), dtype=np.float32)
    selected, current, spent = _greedy_cover_from_pair_delta(
        atom_delta, base_delta, caps, pair_weights, atom_budget_costs, budget, atom_active_mask
    )
    return SelectionResult(
        selected=selected,
        objective_value=current,
        pair_indices=pairs,
        pair_weights=pair_weights,
        diagnostics={"spent_budget": spent, "budget": float(budget), "mode": "legacy_full_prescore", "pair_count": int(len(pairs))},
    )



def _collapse_reciprocal_runtime_pairs(
    pairs: np.ndarray,
    pair_weights: np.ndarray,
    base_delta: np.ndarray,
    caps: np.ndarray,
    atom_delta: np.ndarray,
    atom_pair_var: np.ndarray,
    base_var: np.ndarray,
    info_caps: np.ndarray,
    atom_active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse reciprocal runtime edges into one evidence-sensitive orientation.

    A signed certificate objective cannot reward both ``a>b`` and ``b>a`` for
    the same unordered action pair: antisymmetric atom margins make the two
    gains cancel.  This was especially harmful for stop-vs-go pairs, where the
    cheap base score preferred stopping but a critical interaction/rule atom
    could legitimately flip the decision.  We retain one edge per unordered
    pair and orient it toward the direction with the largest *available*
    positive certificate gain.  No teacher label or future state is used.
    """
    pp = np.asarray(pairs, dtype=np.int64)
    if pp.ndim != 2 or pp.shape[0] <= 1:
        return pp, pair_weights, base_delta, caps, atom_delta, atom_pair_var, base_var, info_caps

    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    bd = np.asarray(base_delta, dtype=np.float32).reshape(-1)
    cp = np.asarray(caps, dtype=np.float32).reshape(-1)
    ad = np.asarray(atom_delta, dtype=np.float32)
    av = np.asarray(atom_pair_var, dtype=np.float32)
    bv = np.asarray(base_var, dtype=np.float32).reshape(-1)
    ic = np.asarray(info_caps, dtype=np.float32).reshape(-1)
    active = np.ones((ad.shape[0],), dtype=bool)
    if atom_active_mask is not None:
        raw = np.asarray(atom_active_mask, dtype=bool).reshape(-1)
        active[:] = False
        active[: min(len(raw), len(active))] = raw[: min(len(raw), len(active))]

    groups: dict[tuple[int, int], list[int]] = {}
    order: list[tuple[int, int]] = []
    for idx, (a, b) in enumerate(pp.tolist()):
        key = (min(int(a), int(b)), max(int(a), int(b)))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(int(idx))

    keep: list[int] = []
    out_weights: list[float] = []
    for key in order:
        idxs = groups[key]
        if len(idxs) == 1:
            chosen = idxs[0]
        else:
            scored: list[tuple[float, float, float, int]] = []
            for j in idxs:
                cap_j = float(cp[j]) if j < len(cp) else 0.0
                w_j = float(weights[j]) if j < len(weights) else 1.0
                base_value = w_j * min(cap_j, max(float(bd[j]), 0.0))
                best_gain = 0.0
                if ad.ndim == 2 and j < ad.shape[1] and bool(active.any()):
                    margins = float(bd[j]) + ad[active, j]
                    after = w_j * np.minimum(cap_j, np.maximum(margins, 0.0))
                    if after.size:
                        best_gain = max(0.0, float(np.max(after) - base_value))
                # Prefer evidence-sensitive flips; then an already positive base
                # certificate; finally deterministic original order.
                scored.append((best_gain, max(float(bd[j]), 0.0), w_j, -j))
            chosen = max(idxs, key=lambda j: next(x for x in scored if x[3] == -j))
        keep.append(chosen)
        out_weights.append(max(float(weights[j]) for j in idxs))

    k = np.asarray(keep, dtype=np.int64)
    return (
        pp[k],
        np.asarray(out_weights, dtype=np.float32),
        bd[k],
        cp[k],
        ad[:, k] if ad.ndim == 2 else ad,
        av[:, k] if av.ndim == 2 else av,
        bv[k] if bv.shape[0] == pp.shape[0] else bv,
        ic[k] if ic.shape[0] == pp.shape[0] else ic,
    )


def runtime_greedy_selector(
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    atom_budget_costs: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    budget: float,
    L_infer: int = 16,
    gamma_max: float = 100.0,
    eta_pred: float = 1.0,
    lambda_near: float = 1.0,
    lambda_safety: float = 2.0,
    atom_active_mask: np.ndarray | None = None,
    predicted_atom_variance: np.ndarray | None = None,
    base_pair_variance: np.ndarray | None = None,
    beta_uncertainty: float = 0.0,
    epsilon_cal: float = 0.0,
    lambda_info: float = 0.0,
    prior_atom_variance: float | np.ndarray | None = None,
    family_ids: np.ndarray | None = None,
    family_budget_caps: np.ndarray | None = None,
    mandatory_atom_mask: np.ndarray | None = None,
    mandatory_quota: int = 0,
    min_selected_atoms: int = 0,
    force_fill_budget: bool = False,
    prioritize_mandatory_fill: bool = True,
    bidirectional_pairs: bool = True,
    reverse_pair_weight: float = 1.0,
    pair_cap_multiplier: float = 1.0,
    candidate_trajectories: np.ndarray | None = None,
    maneuver_ids: np.ndarray | None = None,
    progress_pair_count: int = 0,
    maneuver_pair_count: int = 0,
    decision_family_ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    decision_family_quota: int = 0,
    interaction_family_ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    interaction_family_quota: int = 0,
    collapse_reciprocal_pairs: bool = True,
) -> SelectionResult:
    """Two-stage runtime selector using base/cheap pair screening only.

    Unlike the legacy implementation, this function does not construct
    full_interface_margin before selecting evidence.  It assumes
    ``predicted_atom_costs`` contains only already-queried sparse values (zeros
    elsewhere) and restricts the active atoms through ``atom_active_mask``.
    """
    pairs, pair_weights = build_runtime_pairs_from_base(
        predicted_base_cost,
        valid_mask,
        runtime_safety_flags,
        L0=L_infer,
        eta0=eta_pred,
        lambda_near=lambda_near,
        lambda_safety=lambda_safety,
        bidirectional_pairs=bidirectional_pairs,
        reverse_pair_weight=reverse_pair_weight,
        pair_cap_multiplier=pair_cap_multiplier,
        candidate_trajectories=candidate_trajectories,
        maneuver_ids=maneuver_ids,
        progress_pair_count=int(progress_pair_count),
        maneuver_pair_count=int(maneuver_pair_count),
    )
    E = int(predicted_atom_costs.shape[0])
    if len(pairs):
        a = pairs[:, 0]
        b = pairs[:, 1]
        base_delta = np.asarray(predicted_base_cost, dtype=np.float32)[b] - np.asarray(predicted_base_cost, dtype=np.float32)[a]
        safety_b = np.asarray(runtime_safety_flags, dtype=bool)[b] if np.asarray(runtime_safety_flags).shape[0] > int(np.max(b, initial=0)) else np.zeros_like(base_delta, dtype=bool)
        caps = np.where(
            safety_b,
            float(gamma_max),
            np.minimum(np.maximum(np.abs(base_delta) + float(eta_pred), 1e-3), float(gamma_max)),
        ).astype(np.float32)
        atom_delta = np.asarray(predicted_atom_costs, dtype=np.float32)[:, b] - np.asarray(predicted_atom_costs, dtype=np.float32)[:, a]
        base_support = np.maximum(base_delta, 0.0)
        atom_support = np.maximum(atom_delta, 0.0)
        if predicted_atom_variance is not None:
            av = np.asarray(predicted_atom_variance, dtype=np.float32)
            if av.ndim == 2 and av.shape[1] >= max(int(np.max(b, initial=0)), int(np.max(a, initial=0))) + 1:
                atom_pair_var = np.maximum(av[:, a], 0.0) + np.maximum(av[:, b], 0.0)
            elif av.ndim == 2 and av.shape == atom_delta.shape:
                atom_pair_var = np.maximum(av, 0.0)
            else:
                atom_pair_var = np.zeros_like(atom_delta, dtype=np.float32)
        else:
            atom_pair_var = np.zeros_like(atom_delta, dtype=np.float32)
        if base_pair_variance is not None:
            bpv = np.asarray(base_pair_variance, dtype=np.float32)
            if bpv.ndim == 1 and bpv.shape[0] == len(pairs):
                base_var = np.maximum(bpv, 0.0)
            elif bpv.ndim == 2 and bpv.shape[0] >= len(pairs) and bpv.shape[1] >= 1:
                base_var = np.maximum(bpv[: len(pairs), 0], 0.0)
            else:
                base_var = np.zeros((len(pairs),), dtype=np.float32)
        else:
            base_var = np.zeros((len(pairs),), dtype=np.float32)
        info_caps = np.where(safety_b, float(gamma_max), np.maximum(float(eta_pred), np.abs(base_delta))).astype(np.float32)
    else:
        base_delta = np.zeros((0,), dtype=np.float32)
        base_support = np.zeros((0,), dtype=np.float32)
        atom_support = np.zeros((E, 0), dtype=np.float32)
        atom_delta = np.zeros((E, 0), dtype=np.float32)
        atom_pair_var = np.zeros((E, 0), dtype=np.float32)
        base_var = np.zeros((0,), dtype=np.float32)
        caps = np.zeros((0,), dtype=np.float32)
        info_caps = np.zeros((0,), dtype=np.float32)
    if bool(collapse_reciprocal_pairs) and len(pairs) > 1:
        (
            pairs,
            pair_weights,
            base_delta,
            caps,
            atom_delta,
            atom_pair_var,
            base_var,
            info_caps,
        ) = _collapse_reciprocal_runtime_pairs(
            pairs,
            pair_weights,
            base_delta,
            caps,
            atom_delta,
            atom_pair_var,
            base_var,
            info_caps,
            atom_active_mask=atom_active_mask,
        )
        base_support = np.maximum(base_delta, 0.0)
        atom_support = np.maximum(atom_delta, 0.0)

    use_uncertainty_objective = (
        predicted_atom_variance is not None
        or abs(float(beta_uncertainty)) > 0.0
        or abs(float(epsilon_cal)) > 0.0
        or abs(float(lambda_info)) > 0.0
        or family_budget_caps is not None
    )
    if use_uncertainty_objective:
        if isinstance(prior_atom_variance, np.ndarray):
            prior = np.asarray(prior_atom_variance, dtype=np.float32)
            if len(pairs) and prior.ndim == 2 and prior.shape[1] >= max(int(np.max(pairs[:, 0], initial=0)), int(np.max(pairs[:, 1], initial=0))) + 1:
                prior_pair = np.maximum(prior[:, pairs[:, 0]], 0.0) + np.maximum(prior[:, pairs[:, 1]], 0.0)
            elif prior.shape == atom_pair_var.shape:
                prior_pair = prior
            else:
                prior_pair = None
        elif prior_atom_variance is not None:
            prior_pair = np.full_like(atom_pair_var, float(prior_atom_variance), dtype=np.float32)
        else:
            prior_pair = None
        selected, current, spent, extra_diag = _uncertainty_aware_greedy_from_pair_delta(
            atom_delta,
            base_delta,
            atom_pair_var,
            base_var,
            caps,
            pair_weights,
            atom_budget_costs,
            budget,
            atom_active_mask,
            beta_uncertainty=beta_uncertainty,
            epsilon_cal=epsilon_cal,
            lambda_info=lambda_info,
            info_caps=info_caps,
            prior_atom_pair_var=prior_pair,
            family_ids=family_ids,
            family_budget_caps=family_budget_caps,
        )
        mode = "runtime_hab_lcb_uncertainty"
    else:
        selected, current, spent = _greedy_cover_from_pair_delta(
            atom_delta,
            base_delta,
            caps,
            pair_weights,
            atom_budget_costs,
            budget,
            atom_active_mask,
        )
        extra_diag = {}
        mode = "runtime_base_screen_sparse_signed"
    utility = np.zeros((int(np.asarray(atom_budget_costs).shape[0]),), dtype=np.float32)
    if np.asarray(atom_delta).ndim == 2 and np.asarray(atom_delta).size:
        utility = np.maximum(np.asarray(atom_delta, dtype=np.float32), 0.0).mean(axis=1)
    post_mandatory_atom_mask = mandatory_atom_mask
    if mode == "runtime_pair_conditioned_hybrid_lcb_action_rank" and bool(hybrid_protect_lcb_seed):
        E_mask = int(np.asarray(atom_budget_costs).reshape(-1).shape[0])
        seed_mask = _as_bool_mask(mandatory_atom_mask, E_mask)
        n_seed = int(hybrid_diag.get("hybrid_lcb_seed_atoms", 0)) if isinstance(hybrid_diag, dict) else 0
        for ii in list(map(int, selected[:max(0, n_seed)])):
            if 0 <= ii < E_mask:
                seed_mask[ii] = True
        post_mandatory_atom_mask = seed_mask
    selected, spent_post, post_diag = _complete_safety_aware_selection(
        selected,
        atom_budget_costs,
        budget,
        atom_active_mask=atom_active_mask,
        mandatory_atom_mask=post_mandatory_atom_mask,
        mandatory_quota=mandatory_quota,
        min_selected_atoms=min_selected_atoms,
        force_fill_budget=force_fill_budget,
        utility=utility,
        prioritize_mandatory_fill=prioritize_mandatory_fill,
        family_ids=family_ids,
        decision_family_ids=decision_family_ids,
        decision_family_quota=decision_family_quota,
        interaction_family_ids=interaction_family_ids,
        interaction_family_quota=interaction_family_quota,
    )
    return SelectionResult(
        selected=selected,
        objective_value=current,
        pair_indices=pairs,
        pair_weights=pair_weights,
        diagnostics={"spent_budget": float(spent_post), "pre_postfill_spent_budget": float(spent), "budget": float(budget), "mode": mode, "pair_count": int(len(pairs)), **extra_diag, **post_diag},
    )



def _pairwise_preference_action(
    margins: np.ndarray,
    pair_indices: np.ndarray,
    pair_weights: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray | None = None,
    tau: float = 0.25,
) -> tuple[int, float]:
    """Infer a winner and confidence from directed pair margins.

    Positive ``margin[p]`` means the first action in pair ``p`` beats the
    second.  The score is a weighted Bradley--Terry/Copeland surrogate.  This
    helper uses only runtime predictions and candidate validity/safety.
    """
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    m = np.asarray(margins, dtype=np.float32).reshape(-1)
    w = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    K = int(valid.shape[0])
    if pair_arr.shape[0] == 0 or K == 0:
        idx = int(np.flatnonzero(valid)[0]) if bool(valid.any()) else -1
        return idx, 0.0
    if w.shape[0] != pair_arr.shape[0]:
        w = np.ones((pair_arr.shape[0],), dtype=np.float32)
    m = m[: pair_arr.shape[0]]
    t = max(float(tau), 1e-3)
    z = np.clip(m / t, -20.0, 20.0)
    p = 1.0 / (1.0 + np.exp(-z))
    score = np.zeros((K,), dtype=np.float64)
    denom = np.zeros((K,), dtype=np.float64)
    for idx, (a_raw, b_raw) in enumerate(pair_arr.tolist()):
        a, b = int(a_raw), int(b_raw)
        if not (0 <= a < K and 0 <= b < K and valid[a] and valid[b]):
            continue
        ww = max(float(w[idx]), 0.0)
        score[a] += ww * float(p[idx]); denom[a] += ww
        score[b] += ww * float(1.0 - p[idx]); denom[b] += ww
    avg = np.divide(score, np.maximum(denom, 1e-9))
    eligible = valid.copy()
    if runtime_safety_flags is not None:
        flags = np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)
        if flags.shape[0] < K:
            flags = np.pad(flags, (0, K - flags.shape[0]), constant_values=False)
        safe = valid & ~flags[:K]
        if bool(safe.any()):
            eligible = safe
    avg[~eligible] = -np.inf
    if not bool(np.isfinite(avg).any()):
        return -1, 0.0
    order = np.argsort(-avg)
    best = int(order[0])
    second = float(avg[order[1]]) if order.size > 1 and np.isfinite(avg[order[1]]) else 0.0
    confidence = float(np.clip(float(avg[best]) - second, 0.0, 1.0))
    return best, confidence


def _margin_coreset_objective(
    trial_margin: np.ndarray,
    target_margin: np.ndarray,
    pair_indices: np.ndarray,
    pair_weights: np.ndarray,
    target_action: int,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    *,
    residual_weight: float = 1.0,
    sign_weight: float = 0.8,
    winner_weight: float = 1.5,
    action_weight: float = 0.5,
    boundary_tau: float = 0.35,
    huber_delta: float = 0.25,
    target_clip: float = 3.0,
    sign_floor: float = 0.05,
) -> float:
    """Loss for compressing the Top-M signed margin field into B atoms."""
    trial = np.asarray(trial_margin, dtype=np.float32).reshape(-1)
    target = np.asarray(target_margin, dtype=np.float32).reshape(-1)
    pairs = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    w = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    if trial.shape[0] != target.shape[0] or trial.shape[0] != pairs.shape[0]:
        return float('inf')
    if w.shape[0] != trial.shape[0]:
        w = np.ones_like(trial, dtype=np.float32)
    w = np.maximum(w, 0.0)
    tau = max(float(boundary_tau), 1e-3)
    clip = max(float(target_clip), tau)
    t = np.clip(target, -clip, clip)
    x = np.clip(trial, -clip, clip)
    # Both near-boundary and confidently decisive pairs matter: boundary pairs
    # protect action flips, while decisive pairs protect the winner certificate.
    boundary = np.exp(-np.abs(t) / tau)
    decisive = np.tanh(np.abs(t) / tau)
    pw = w * (1.0 + 1.5 * boundary + 0.5 * decisive)
    norm = max(float(pw.sum()), 1e-9)
    err = (x - t) / max(float(huber_delta), 1e-3)
    abs_err = np.abs(err)
    huber = np.where(abs_err <= 1.0, 0.5 * err * err, abs_err - 0.5)
    loss = max(float(residual_weight), 0.0) * float(np.sum(pw * huber) / norm)
    mismatch = (x * t < 0.0) & (np.abs(t) >= max(float(sign_floor), 0.0))
    if bool(mismatch.any()):
        loss += max(float(sign_weight), 0.0) * float(np.sum(pw * mismatch.astype(np.float32) * np.minimum(np.abs(t), 1.0)) / norm)
    if target_action >= 0:
        rel_trial = []
        rel_target = []
        rel_weight = []
        for pi, (a_raw, b_raw) in enumerate(pairs.tolist()):
            a, b = int(a_raw), int(b_raw)
            if a == target_action:
                rel_trial.append(float(x[pi])); rel_target.append(float(t[pi])); rel_weight.append(float(pw[pi]))
            elif b == target_action:
                rel_trial.append(float(-x[pi])); rel_target.append(float(-t[pi])); rel_weight.append(float(pw[pi]))
        if rel_trial:
            rt = np.asarray(rel_trial, dtype=np.float32)
            rtar = np.asarray(rel_target, dtype=np.float32)
            rw = np.asarray(rel_weight, dtype=np.float32)
            wanted = rtar > 0.0
            if bool(wanted.any()):
                # Smoothly penalize loss of a target winner--rival certificate.
                v = np.log1p(np.exp(np.clip(-rt[wanted] / tau, -20.0, 20.0)))
                loss += max(float(winner_weight), 0.0) * float(np.sum(rw[wanted] * v) / max(float(rw[wanted].sum()), 1e-9))
    inferred, _ = _pairwise_preference_action(trial, pairs, w, valid_mask, runtime_safety_flags, tau=tau)
    if target_action >= 0 and inferred >= 0 and inferred != target_action:
        loss += max(float(action_weight), 0.0)
    return float(loss)


def _signed_margin_coreset_from_pair_delta(
    atom_delta: np.ndarray,
    base_margin: np.ndarray,
    pair_weights: np.ndarray,
    pair_indices: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    atom_active_mask: np.ndarray | None = None,
    *,
    soft_interaction_mask: np.ndarray | None = None,
    soft_interaction_quota: int = 0,
    proposal_scores: np.ndarray | None = None,
    residual_weight: float = 1.0,
    sign_weight: float = 0.8,
    winner_weight: float = 1.5,
    action_weight: float = 0.5,
    boundary_tau: float = 0.35,
    huber_delta: float = 0.25,
    target_clip: float = 3.0,
    swap_passes: int = 2,
) -> tuple[list[int], float, float, dict[str, Any]]:
    """Backward-elimination signed margin coreset.

    The runtime already scores every Top-M atom on the logical pair graph.  The
    previous selector used those scores only as independent positive support and
    therefore discarded cancellation and negative-but-decisive evidence.  Here
    Top-M defines a full predicted signed margin field, and the B-atom certificate
    is the minimum-loss coreset of that field.  No teacher labels or extra neural
    queries are used.
    """
    d = np.asarray(atom_delta, dtype=np.float32)
    base = np.asarray(base_margin, dtype=np.float32).reshape(-1)
    pairs = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    E = int(d.shape[0]) if d.ndim == 2 else int(np.asarray(atom_budget_costs).shape[0])
    if d.ndim != 2 or d.shape[1] != base.shape[0]:
        d = np.zeros((E, base.shape[0]), dtype=np.float32)
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else _as_bool_mask(atom_active_mask, E)
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)
    costs = costs[:E]
    active &= np.isfinite(costs) & (costs > 0.0)
    ids = np.flatnonzero(active).astype(np.int64).tolist()
    if not ids:
        return [], 0.0, 0.0, {"margin_coreset_active": True, "margin_coreset_target_atom_count": 0}
    soft = _as_bool_mask(soft_interaction_mask, E) & active
    soft_floor = min(max(0, int(soft_interaction_quota)), int(soft.sum()))
    proposal = np.zeros((E,), dtype=np.float32)
    if proposal_scores is not None:
        raw = np.asarray(proposal_scores, dtype=np.float32).reshape(-1)
        proposal[: min(E, raw.shape[0])] = raw[: min(E, raw.shape[0])]
    target = base + np.sum(d[np.asarray(ids, dtype=np.int64)], axis=0)
    target_action, target_conf = _pairwise_preference_action(target, pairs, weights, valid_mask, runtime_safety_flags, tau=boundary_tau)

    def loss(margin: np.ndarray) -> float:
        return _margin_coreset_objective(
            margin, target, pairs, weights, target_action, valid_mask, runtime_safety_flags,
            residual_weight=residual_weight, sign_weight=sign_weight, winner_weight=winner_weight,
            action_weight=action_weight, boundary_tau=boundary_tau, huber_delta=huber_delta, target_clip=target_clip,
        )

    # The original implementation called ``loss`` once for every possible atom
    # removal and once for every possible swap.  With Top-M=64 this means
    # thousands of small Python/Numpy calls per scene, and deployment-aligned
    # training repeats that work on every DDP rank.  Precompute the invariant
    # target weights / pair-to-action incidence matrices and score all candidate
    # removals (or swaps) in one vectorized call.  This is algebraically identical
    # to _margin_coreset_objective; only the candidate dimension is batched.
    tau_vec = max(float(boundary_tau), 1e-3)
    clip_vec = max(float(target_clip), tau_vec)
    target_clip_vec = np.clip(target, -clip_vec, clip_vec).astype(np.float32, copy=False)
    weights_nonneg = np.maximum(weights, 0.0).astype(np.float32, copy=False)
    boundary_vec = np.exp(-np.abs(target_clip_vec) / tau_vec)
    decisive_vec = np.tanh(np.abs(target_clip_vec) / tau_vec)
    pair_objective_w = weights_nonneg * (1.0 + 1.5 * boundary_vec + 0.5 * decisive_vec)
    pair_objective_norm = max(float(pair_objective_w.sum()), 1e-9)
    huber_delta_vec = max(float(huber_delta), 1e-3)
    sign_floor_vec = 0.05

    valid_actions = np.asarray(valid_mask, dtype=bool).reshape(-1)
    action_count = int(valid_actions.shape[0])
    pair_count = int(pairs.shape[0])
    incidence_a = np.zeros((pair_count, action_count), dtype=np.float32)
    incidence_b = np.zeros((pair_count, action_count), dtype=np.float32)
    action_denom = np.zeros((action_count,), dtype=np.float32)
    if pair_count and action_count:
        a_all = pairs[:, 0].astype(np.int64, copy=False)
        b_all = pairs[:, 1].astype(np.int64, copy=False)
        pair_ok = (a_all >= 0) & (a_all < action_count) & (b_all >= 0) & (b_all < action_count)
        if bool(pair_ok.any()):
            ok_idx = np.flatnonzero(pair_ok)
            pair_ok[ok_idx] &= valid_actions[a_all[ok_idx]] & valid_actions[b_all[ok_idx]]
        ok_idx = np.flatnonzero(pair_ok)
        if ok_idx.size:
            incidence_a[ok_idx, a_all[ok_idx]] = weights_nonneg[ok_idx]
            incidence_b[ok_idx, b_all[ok_idx]] = weights_nonneg[ok_idx]
            np.add.at(action_denom, a_all[ok_idx], weights_nonneg[ok_idx])
            np.add.at(action_denom, b_all[ok_idx], weights_nonneg[ok_idx])
    eligible_actions = valid_actions.copy()
    flags_vec = np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)
    if flags_vec.shape[0] < action_count:
        flags_vec = np.pad(flags_vec, (0, action_count - flags_vec.shape[0]), constant_values=False)
    if action_count:
        safe_actions = valid_actions & ~flags_vec[:action_count]
        if bool(safe_actions.any()):
            eligible_actions = safe_actions

    target_orientation = np.zeros((pair_count,), dtype=np.float32)
    if target_action >= 0 and pair_count:
        target_orientation[pairs[:, 0] == target_action] = 1.0
        target_orientation[pairs[:, 1] == target_action] = -1.0
    oriented_target = target_clip_vec * target_orientation
    wanted_target = (target_orientation != 0.0) & (oriented_target > 0.0)
    winner_norm = max(float(pair_objective_w[wanted_target].sum()), 1e-9) if bool(wanted_target.any()) else 1.0

    def loss_many(margins: np.ndarray) -> np.ndarray:
        trial_matrix = np.asarray(margins, dtype=np.float32)
        if trial_matrix.ndim == 1:
            trial_matrix = trial_matrix[None, :]
        if trial_matrix.ndim != 2 or trial_matrix.shape[1] != pair_count:
            return np.full((trial_matrix.shape[0] if trial_matrix.ndim else 1,), np.inf, dtype=np.float64)
        x = np.clip(trial_matrix, -clip_vec, clip_vec)
        err = (x - target_clip_vec[None, :]) / huber_delta_vec
        abs_err = np.abs(err)
        huber = np.where(abs_err <= 1.0, 0.5 * err * err, abs_err - 0.5)
        values = max(float(residual_weight), 0.0) * (huber * pair_objective_w[None, :]).sum(axis=1) / pair_objective_norm

        if float(sign_weight) > 0.0:
            mismatch = (x * target_clip_vec[None, :] < 0.0) & (np.abs(target_clip_vec)[None, :] >= sign_floor_vec)
            sign_penalty = (
                pair_objective_w[None, :]
                * mismatch.astype(np.float32)
                * np.minimum(np.abs(target_clip_vec), 1.0)[None, :]
            ).sum(axis=1) / pair_objective_norm
            values = values + max(float(sign_weight), 0.0) * sign_penalty

        if target_action >= 0 and bool(wanted_target.any()) and float(winner_weight) > 0.0:
            oriented_trial = x[:, wanted_target] * target_orientation[wanted_target][None, :]
            soft_violation = np.log1p(np.exp(np.clip(-oriented_trial / tau_vec, -20.0, 20.0)))
            winner_penalty = (
                soft_violation * pair_objective_w[wanted_target][None, :]
            ).sum(axis=1) / winner_norm
            values = values + max(float(winner_weight), 0.0) * winner_penalty

        if target_action >= 0 and action_count and bool(eligible_actions.any()) and float(action_weight) > 0.0:
            prob = 1.0 / (1.0 + np.exp(-np.clip(trial_matrix / tau_vec, -20.0, 20.0)))
            score = prob @ incidence_a + (1.0 - prob) @ incidence_b
            avg = score / np.maximum(action_denom[None, :], 1e-9)
            avg[:, ~eligible_actions] = -np.inf
            inferred = np.argmax(avg, axis=1)
            finite_any = np.isfinite(avg).any(axis=1)
            mismatch_action = finite_any & (inferred != int(target_action))
            values = values + max(float(action_weight), 0.0) * mismatch_action.astype(np.float32)
        return np.asarray(values, dtype=np.float64)

    selected = list(ids)
    selected_set = set(selected)
    current = target.copy()
    spent = _spent_for(selected, costs)
    removed: list[int] = []
    while spent > float(budget) + 1e-6 and selected:
        soft_count = int(sum(1 for i in selected if soft[i]))
        candidates = [int(i) for i in selected if not (soft[i] and soft_count - 1 < soft_floor)]
        if not candidates:
            break
        candidate_arr = np.asarray(candidates, dtype=np.int64)
        trial_matrix = current[None, :] - d[candidate_arr]
        losses = loss_many(trial_matrix)
        ratios = losses / np.maximum(costs[candidate_arr].astype(np.float64), 1e-6)
        # np.lexsort uses the last key as primary: ratio, then raw loss, then
        # proposal score, matching the tuple comparison in the old loop.
        order = np.lexsort((proposal[candidate_arr], losses, ratios))
        best = int(candidate_arr[int(order[0])])
        selected.remove(best); selected_set.remove(best); removed.append(best)
        current = current - d[best]
        spent -= float(costs[best])

    swaps = 0
    max_passes = max(0, int(swap_passes))
    for _ in range(max_passes):
        current_loss = loss(current)
        soft_count = int(sum(1 for i in selected if soft[i]))
        swap_out: list[int] = []
        swap_in: list[int] = []
        swap_spent: list[float] = []
        for out_i in list(selected):
            for in_i in list(removed):
                new_spent = spent - float(costs[out_i]) + float(costs[in_i])
                if new_spent > float(budget) + 1e-6:
                    continue
                if soft[out_i] and not soft[in_i] and soft_count - 1 < soft_floor:
                    continue
                swap_out.append(int(out_i)); swap_in.append(int(in_i)); swap_spent.append(float(new_spent))
        if not swap_out:
            break
        out_arr = np.asarray(swap_out, dtype=np.int64)
        in_arr = np.asarray(swap_in, dtype=np.int64)
        trial_matrix = current[None, :] - d[out_arr] + d[in_arr]
        losses = loss_many(trial_matrix)
        best_idx = int(np.argmin(losses))
        if not (float(losses[best_idx]) + 1e-9 < float(current_loss)):
            break
        out_i = int(out_arr[best_idx]); in_i = int(in_arr[best_idx])
        current = trial_matrix[best_idx]
        spent = float(swap_spent[best_idx])
        selected.remove(out_i); selected.append(in_i)
        removed.remove(in_i); removed.append(out_i)
        selected_set.remove(out_i); selected_set.add(in_i)
        swaps += 1

    selected = sorted(selected)
    selected_action, _ = _pairwise_preference_action(current, pairs, weights, valid_mask, runtime_safety_flags, tau=boundary_tau)
    target_sign = np.sign(target)
    selected_sign = np.sign(current)
    informative = np.abs(target) >= 0.05
    pw = np.maximum(weights, 0.0)
    if bool(informative.any()):
        sign_agree = float(np.sum(pw[informative] * (target_sign[informative] == selected_sign[informative])) / max(float(pw[informative].sum()), 1e-9))
    else:
        sign_agree = 1.0
    rmse = float(np.sqrt(np.sum(pw * np.square(current - target)) / max(float(pw.sum()), 1e-9))) if current.size else 0.0
    final_loss = loss(current)
    diag = {
        "margin_coreset_active": True,
        "margin_coreset_target_atom_count": int(len(ids)),
        "margin_coreset_selected_atom_count": int(len(selected)),
        "margin_coreset_removed_atom_count": int(len(removed)),
        "margin_coreset_swap_count": int(swaps),
        "margin_coreset_target_action": int(target_action),
        "margin_coreset_selected_action": int(selected_action),
        "margin_coreset_target_action_preserved": float(target_action >= 0 and selected_action == target_action),
        "margin_coreset_target_confidence": float(target_conf),
        "margin_coreset_target_sign_agreement": float(sign_agree),
        "margin_coreset_target_margin_rmse": float(rmse),
        "margin_coreset_objective": float(final_loss),
        "margin_coreset_soft_floor": int(soft_floor),
    }
    return selected, float(-final_loss), float(spent), diag


def _deployment_aligned_coreset_from_pair_delta(
    atom_delta: np.ndarray,
    base_margin: np.ndarray,
    pair_weights: np.ndarray,
    pair_indices: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    atom_active_mask: np.ndarray | None,
    deployment_evaluator: Callable[[list[int]], Any],
    *,
    soft_interaction_mask: np.ndarray | None = None,
    soft_interaction_quota: int = 0,
    proposal_scores: np.ndarray | None = None,
    exact_candidates: int = 8,
    swap_passes: int = 1,
    score_weight: float = 1.0,
    action_weight: float = 4.0,
    gap_weight: float = 2.0,
    margin_weight: float = 1.0,
    huber_delta: float = 0.25,
    margin_floor: float = 0.05,
    lexicographic_action_preservation: bool = False,
    preservation_scan_candidates: int = 0,
    repair_one_swap: bool = True,
    repair_two_swap_candidates: int = 0,
    beam_width: int = 0,
    beam_branch: int = 0,
    beam_max_evaluations: int = 0,
    beam_mismatch_fraction: float = 0.35,
    budget_layer_width: int = 0,
    budget_layer_branch: int = 0,
    budget_layer_iterations: int = 0,
    budget_layer_max_evaluations: int = 0,
    budget_layer_exhaustive_first: bool = True,
    budget_layer_seed_count: int = 0,
    budget_layer_diversity_distance: int = 2,
) -> tuple[list[int], float, float, dict[str, Any]]:
    """Compress Top-M evidence against the actual deployed decision map.

    The v39 prototype used a finite action-change penalty inside an otherwise
    continuous reconstruction loss.  That permits a lower score-RMSE subset to
    win even when it changes the deployment action, which is exactly the event
    the runtime gate is intended to reject.  The optional v40 path therefore
    uses a lexicographic objective:

      1. preserve the full-Top-M deployment action whenever a feasible deletion
         or exchange exists;
      2. among action-preserving subsets, minimize score/gap/margin distortion;
      3. only accept an action-changing step when no preserving step is found;
      4. if the single-path search is trapped, run a bounded action-diverse beam
         over the deletion lattice.  The beam deliberately retains a small
         number of temporarily mismatching states, because a target action can
         disappear at an intermediate cardinality and reappear after a later
         evidence deletion.

    Candidate screening still uses the cheap signed-margin surrogate.  Exact
    decisions are evaluated with the same downstream tournament, safety guard,
    utility refinement and all-flagged guard used by deployment.  No additional
    neural query is issued; the extra work is deterministic CPU/Numpy search over
    already computed Top-M pair deltas.
    """
    d = np.asarray(atom_delta, dtype=np.float32)
    E = int(d.shape[0]) if d.ndim == 2 else int(np.asarray(atom_budget_costs).reshape(-1).shape[0])
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)
    costs = costs[:E]
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else _as_bool_mask(atom_active_mask, E)
    active &= np.isfinite(costs) & (costs > 0.0)
    ids = np.flatnonzero(active).astype(np.int64).tolist()
    if not ids:
        return [], 0.0, 0.0, {
            "deployment_coreset_active": True,
            "deployment_coreset_lexicographic_active": bool(lexicographic_action_preservation),
            "deployment_coreset_target_atom_count": 0,
        }

    soft = _as_bool_mask(soft_interaction_mask, E) & active
    soft_floor = min(max(0, int(soft_interaction_quota)), int(soft.sum()))
    proposal = np.zeros((E,), dtype=np.float32)
    if proposal_scores is not None:
        raw = np.asarray(proposal_scores, dtype=np.float32).reshape(-1)
        proposal[: min(E, raw.shape[0])] = raw[: min(E, raw.shape[0])]

    # The MARS margin loss is retained only as a computational screen.
    base = np.asarray(base_margin, dtype=np.float32).reshape(-1)
    pairs = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    if d.ndim != 2 or d.shape[1] != base.shape[0]:
        d = np.zeros((E, base.shape[0]), dtype=np.float32)
    target_margin_surrogate = base + d[np.asarray(ids, dtype=np.int64)].sum(axis=0)
    surrogate_target_action, _ = _pairwise_preference_action(
        target_margin_surrogate, pairs, weights, valid_mask, runtime_safety_flags
    )

    eval_cache: dict[tuple[int, ...], tuple[int, np.ndarray, np.ndarray]] = {}
    eval_diag_cache: dict[tuple[int, ...], dict[str, Any]] = {}
    component_cache: dict[tuple[int, ...], tuple[int, float]] = {}

    def evaluate(sel: list[int]) -> tuple[int, np.ndarray, np.ndarray]:
        key = tuple(sorted(map(int, sel)))
        if key not in eval_cache:
            result = deployment_evaluator(list(key))
            diag: dict[str, Any] = {}
            if isinstance(result, dict):
                action = result.get("action", result.get("action_index", -1))
                scores = result.get("scores", [])
                margins = result.get("margins", [])
                raw_diag = result.get("diagnostics", {})
                if isinstance(raw_diag, dict):
                    diag = dict(raw_diag)
            else:
                values = tuple(result)
                if len(values) < 3:
                    raise ValueError("deployment_evaluator must return action, scores, margins")
                action, scores, margins = values[:3]
                if len(values) >= 4 and isinstance(values[3], dict):
                    diag = dict(values[3])
            eval_cache[key] = (
                int(action),
                np.asarray(scores, dtype=np.float32).reshape(-1),
                np.asarray(margins, dtype=np.float32),
            )
            eval_diag_cache[key] = diag
        return eval_cache[key]

    def evaluation_diagnostics(sel: list[int] | tuple[int, ...]) -> dict[str, Any]:
        key = tuple(sorted(map(int, sel)))
        if key not in eval_cache:
            evaluate(list(key))
        return eval_diag_cache.get(key, {})

    target_action, target_scores, target_margins = evaluate(ids)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    n = min(valid.shape[0], target_scores.shape[0])
    eligible = valid[:n] & np.isfinite(target_scores[:n]) & (target_scores[:n] > -1e8)
    target_scale = max(float(np.std(target_scores[:n][eligible])) if bool(eligible.any()) else 0.0, 0.1)
    hdelta = max(float(huber_delta), 1e-3)

    def distortion_components(sel: list[int]) -> tuple[int, float]:
        """Return deployment action and continuous distortion (no action penalty)."""
        key = tuple(sorted(map(int, sel)))
        if key in component_cache:
            return component_cache[key]
        action, scores, margins = evaluate(sel)
        m = min(n, scores.shape[0])
        mask = eligible[:m] & np.isfinite(scores[:m]) & (scores[:m] > -1e8)
        loss = 0.0
        if bool(mask.any()):
            err = (scores[:m][mask] - target_scores[:m][mask]) / target_scale
            ae = np.abs(err) / hdelta
            huber = np.where(ae <= 1.0, 0.5 * ae * ae, ae - 0.5)
            loss += max(float(score_weight), 0.0) * float(np.mean(huber))
        if 0 <= target_action < m and bool(mask.any()):
            rivals = np.flatnonzero(mask & (np.arange(m) != target_action))
            if rivals.size:
                target_gap = float(target_scores[target_action] - np.max(target_scores[rivals]))
                trial_gap = float(scores[target_action] - np.max(scores[rivals]))
                loss += max(float(gap_weight), 0.0) * max(0.0, target_gap - trial_gap) / target_scale
        if (
            target_margins.ndim == 2 and margins.ndim == 2
            and 0 <= target_action < min(target_margins.shape[0], margins.shape[0])
        ):
            k = min(target_margins.shape[1], margins.shape[1], valid.shape[0])
            rivals = valid[:k].copy()
            if target_action < k:
                rivals[target_action] = False
            trow = target_margins[target_action, :k]
            xrow = margins[target_action, :k]
            informative = rivals & np.isfinite(trow) & np.isfinite(xrow) & (
                np.abs(trow) >= max(float(margin_floor), 0.0)
            )
            if bool(informative.any()):
                sign_err = np.mean((trow[informative] * xrow[informative] < 0.0).astype(np.float32))
                mag_err = np.mean(np.minimum(np.abs(xrow[informative] - trow[informative]), 1.0))
                loss += max(float(margin_weight), 0.0) * float(sign_err + 0.25 * mag_err)
        out = (int(action), float(loss))
        component_cache[key] = out
        return out

    def exact_loss(sel: list[int]) -> float:
        action, distortion = distortion_components(sel)
        mismatch = bool(target_action >= 0 and action != target_action)
        return float(distortion + (max(float(action_weight), 0.0) if mismatch else 0.0))

    def screen_loss(margin: np.ndarray) -> float:
        return _margin_coreset_objective(
            margin, target_margin_surrogate, pairs, weights, surrogate_target_action,
            valid_mask, runtime_safety_flags, residual_weight=1.0, sign_weight=0.8,
            winner_weight=1.5, action_weight=0.5,
        )

    def mismatch_rank(action: int) -> int:
        return int(target_action >= 0 and int(action) != int(target_action))

    selected = list(ids)
    current_surrogate = target_margin_surrogate.copy()
    spent = _spent_for(selected, costs)
    removed: list[int] = []
    exact_k = max(1, int(exact_candidates))
    scan_limit_cfg = max(0, int(preservation_scan_candidates))
    forced_action_flip_steps = 0
    preservation_scan_evals = 0

    while spent > float(budget) + 1e-6 and selected:
        soft_count = int(sum(1 for i in selected if soft[i]))
        screened: list[tuple[float, int]] = []
        for i in selected:
            if soft[i] and soft_count - 1 < soft_floor:
                continue
            trial_margin = current_surrogate - d[i]
            screened.append((screen_loss(trial_margin) / max(float(costs[i]), 1e-6), int(i)))
        if not screened:
            break
        screened.sort(key=lambda x: (x[0], float(proposal[x[1]]), x[1]))
        ordered = [int(i) for _, i in screened]
        first_count = min(exact_k, len(ordered))
        trial_rows: list[tuple[int, float, float, int]] = []

        def evaluate_removal(i: int) -> None:
            trial_sel = [j for j in selected if j != i]
            action, distortion = distortion_components(trial_sel)
            trial_rows.append((mismatch_rank(action), float(distortion), float(proposal[i]), int(i)))

        for i in ordered[:first_count]:
            evaluate_removal(i)

        if bool(lexicographic_action_preservation) and target_action >= 0:
            # If the cheap top-k contains no action-preserving deletion, expand
            # the exact scan.  Zero means scan every feasible deletion.
            if not any(row[0] == 0 for row in trial_rows):
                scan_count = len(ordered) if scan_limit_cfg <= 0 else min(len(ordered), max(first_count, scan_limit_cfg))
                for i in ordered[first_count:scan_count]:
                    evaluate_removal(i)
                    preservation_scan_evals += 1
            preserving = [row for row in trial_rows if row[0] == 0]
            pool = preserving if preserving else trial_rows
            if not preserving:
                forced_action_flip_steps += 1
            best_row = min(
                pool,
                key=lambda row: (
                    row[1] / max(float(costs[row[3]]), 1e-6),
                    row[1], row[2], row[3],
                ),
            )
        else:
            best_row = min(
                trial_rows,
                key=lambda row: (
                    (row[1] + (max(float(action_weight), 0.0) if row[0] else 0.0))
                    / max(float(costs[row[3]]), 1e-6),
                    row[1] + (max(float(action_weight), 0.0) if row[0] else 0.0),
                    row[2], row[3],
                ),
            )
        best = int(best_row[3])
        selected.remove(best)
        removed.append(best)
        current_surrogate -= d[best]
        spent -= float(costs[best])

    swaps = 0
    for _ in range(max(0, int(swap_passes))):
        current_action, current_distortion = distortion_components(selected)
        current_rank = mismatch_rank(current_action)
        soft_count = int(sum(1 for i in selected if soft[i]))
        screened_swaps: list[tuple[float, int, int, float]] = []
        for out_i in selected:
            for in_i in removed:
                new_spent = spent - float(costs[out_i]) + float(costs[in_i])
                if new_spent > float(budget) + 1e-6:
                    continue
                if soft[out_i] and not soft[in_i] and soft_count - 1 < soft_floor:
                    continue
                trial_margin = current_surrogate - d[out_i] + d[in_i]
                screened_swaps.append((screen_loss(trial_margin), int(out_i), int(in_i), float(new_spent)))
        screened_swaps.sort(key=lambda x: (x[0], x[1], x[2]))
        if not screened_swaps:
            break

        first_count = min(exact_k, len(screened_swaps))
        swap_rows: list[tuple[int, float, int, int, float]] = []

        def evaluate_swap(row: tuple[float, int, int, float]) -> None:
            _, out_i, in_i, new_spent = row
            trial_sel = [j for j in selected if j != out_i] + [in_i]
            action, distortion = distortion_components(trial_sel)
            swap_rows.append((mismatch_rank(action), float(distortion), int(out_i), int(in_i), float(new_spent)))

        for row in screened_swaps[:first_count]:
            evaluate_swap(row)
        if bool(lexicographic_action_preservation) and target_action >= 0:
            # When the current set has already flipped, exhaust the one-exchange
            # neighborhood to actively repair it.  When it is preserved, never
            # accept a swap that breaks the action.
            if current_rank > 0 and not any(row[0] == 0 for row in swap_rows):
                for row in screened_swaps[first_count:]:
                    evaluate_swap(row)
                    preservation_scan_evals += 1
            candidates = [row for row in swap_rows if row[0] <= current_rank]
            if current_rank == 0:
                candidates = [row for row in candidates if row[0] == 0]
            if not candidates:
                break
            best_row = min(candidates, key=lambda row: (row[0], row[1], row[2], row[3]))
            if (best_row[0], best_row[1]) >= (current_rank, current_distortion - 1e-9):
                break
        else:
            best_row = min(
                swap_rows,
                key=lambda row: (
                    row[1] + (max(float(action_weight), 0.0) if row[0] else 0.0),
                    row[2], row[3],
                ),
            )
            best_total = best_row[1] + (max(float(action_weight), 0.0) if best_row[0] else 0.0)
            if best_total + 1e-9 >= exact_loss(selected):
                break
        _, _, out_i, in_i, new_spent = best_row
        selected.remove(out_i)
        selected.append(in_i)
        removed.remove(in_i)
        removed.append(out_i)
        current_surrogate = current_surrogate - d[out_i] + d[in_i]
        spent = float(new_spent)
        swaps += 1

    repair_attempted = False
    repair_success = False
    repair_one_swap_evals = 0
    repair_two_swap_evals = 0
    beam_attempted = False
    beam_success = False
    beam_evals = 0
    beam_depth = 0
    beam_peak_width = 0
    beam_terminal_count = 0
    budget_layer_attempted = False
    budget_layer_success = False
    budget_layer_evals = 0
    budget_layer_iterations_done = 0
    budget_layer_peak_width = 0
    budget_layer_unique_states = 0
    budget_layer_seed_states = 0
    budget_layer_best_target_rank = 0
    budget_layer_best_action_deficit = 0.0
    budget_layer_best_margin_deficit = 0.0
    budget_layer_best_stage = 0
    budget_layer_best_stage_violation = 0.0
    budget_layer_best_raw_action = -1
    selected_action, _ = distortion_components(selected)
    if bool(lexicographic_action_preservation) and target_action >= 0 and selected_action != target_action:
        repair_attempted = True
        current_soft_count = int(sum(1 for i in selected if soft[i]))
        if bool(repair_one_swap):
            repairing: list[tuple[float, int, int, float]] = []
            for out_i in list(selected):
                for in_i in list(removed):
                    new_spent = spent - float(costs[out_i]) + float(costs[in_i])
                    if new_spent > float(budget) + 1e-6:
                        continue
                    if soft[out_i] and not soft[in_i] and current_soft_count - 1 < soft_floor:
                        continue
                    trial_sel = [j for j in selected if j != out_i] + [in_i]
                    action, distortion = distortion_components(trial_sel)
                    repair_one_swap_evals += 1
                    if action == target_action:
                        repairing.append((float(distortion), int(out_i), int(in_i), float(new_spent)))
            if repairing:
                _, out_i, in_i, new_spent = min(repairing, key=lambda row: (row[0], row[1], row[2]))
                selected.remove(out_i)
                selected.append(in_i)
                removed.remove(in_i)
                removed.append(out_i)
                current_surrogate = current_surrogate - d[out_i] + d[in_i]
                spent = float(new_spent)
                repair_success = True

        selected_action, _ = distortion_components(selected)
        max_two = max(0, int(repair_two_swap_candidates))
        if selected_action != target_action and max_two > 0 and len(selected) >= 2 and len(removed) >= 2:
            current_soft_count = int(sum(1 for i in selected if soft[i]))
            two_screen: list[tuple[float, int, int, int, int, float]] = []
            sel_sorted = sorted(selected)
            rem_sorted = sorted(removed)
            for oi, out_i in enumerate(sel_sorted[:-1]):
                for out_j in sel_sorted[oi + 1:]:
                    removed_soft = int(bool(soft[out_i])) + int(bool(soft[out_j]))
                    for ii, in_i in enumerate(rem_sorted[:-1]):
                        for in_j in rem_sorted[ii + 1:]:
                            added_soft = int(bool(soft[in_i])) + int(bool(soft[in_j]))
                            if current_soft_count - removed_soft + added_soft < soft_floor:
                                continue
                            new_spent = (
                                spent - float(costs[out_i]) - float(costs[out_j])
                                + float(costs[in_i]) + float(costs[in_j])
                            )
                            if new_spent > float(budget) + 1e-6:
                                continue
                            trial_margin = current_surrogate - d[out_i] - d[out_j] + d[in_i] + d[in_j]
                            two_screen.append((
                                screen_loss(trial_margin), int(out_i), int(out_j),
                                int(in_i), int(in_j), float(new_spent),
                            ))
            two_screen.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
            repairing2: list[tuple[float, int, int, int, int, float]] = []
            for _, out_i, out_j, in_i, in_j, new_spent in two_screen[:max_two]:
                trial_sel = [j for j in selected if j not in {out_i, out_j}] + [in_i, in_j]
                action, distortion = distortion_components(trial_sel)
                repair_two_swap_evals += 1
                if action == target_action:
                    repairing2.append((float(distortion), out_i, out_j, in_i, in_j, float(new_spent)))
            if repairing2:
                _, out_i, out_j, in_i, in_j, new_spent = min(
                    repairing2, key=lambda row: (row[0], row[1], row[2], row[3], row[4])
                )
                selected.remove(out_i)
                selected.remove(out_j)
                selected.extend([in_i, in_j])
                removed.remove(in_i)
                removed.remove(in_j)
                removed.extend([out_i, out_j])
                current_surrogate = current_surrogate - d[out_i] - d[out_j] + d[in_i] + d[in_j]
                spent = float(new_spent)
                repair_success = True

    # v41 PR-DACC: the hard lexicographic greedy path can still become
    # trapped.  In the v40 gate failures, a preserving deletion often existed
    # at early cardinalities, but the chosen preserving branch later reached a
    # level where every one-step deletion flipped the action.  A final one/two
    # exchange cannot reliably undo that path dependence.  When the local
    # repair fails, search a bounded beam over the full deletion lattice.  The
    # beam keeps both exact target-preserving states and a controlled set of
    # temporarily mismatching states, ranked by exact deployment distortion and
    # target-action deficit.  This is still query-free: every state reuses the
    # already-computed Top-M pair deltas and the deployment callback.
    selected_action, _ = distortion_components(selected)
    bw = max(0, int(beam_width))
    bb = max(0, int(beam_branch))
    beam_eval_cap = max(0, int(beam_max_evaluations))
    if (
        bool(lexicographic_action_preservation)
        and target_action >= 0
        and selected_action != target_action
        and bw > 0
        and bb > 0
        and len(ids) > len(selected)
    ):
        beam_attempted = True
        beam_eval_start = len(eval_cache)
        target_valid_idx = np.flatnonzero(eligible)
        target_rivals = target_valid_idx[target_valid_idx != int(target_action)]
        if 0 <= target_action < target_scores.shape[0] and target_rivals.size:
            target_raw_gap = float(target_scores[target_action] - np.max(target_scores[target_rivals]))
        else:
            target_raw_gap = 0.0

        # Cheap target-oriented atom statistics from the same pair graph used by
        # the deployment-aligned screen.  Positive support helps the target
        # action; strongly negative support is a useful recovery deletion.
        target_pair_mask = np.zeros((pairs.shape[0],), dtype=bool)
        target_orient = np.zeros((pairs.shape[0],), dtype=np.float32)
        if 0 <= target_action:
            for pidx, (a_raw, b_raw) in enumerate(pairs.tolist()):
                a_i, b_i = int(a_raw), int(b_raw)
                if a_i == target_action:
                    target_pair_mask[pidx] = True
                    target_orient[pidx] = 1.0
                elif b_i == target_action:
                    target_pair_mask[pidx] = True
                    target_orient[pidx] = -1.0
        if bool(target_pair_mask.any()):
            oriented = d[:, target_pair_mask] * target_orient[target_pair_mask][None, :]
            target_atom_support = np.mean(oriented, axis=1).astype(np.float32)
            target_atom_abs = np.mean(np.abs(oriented), axis=1).astype(np.float32)
        else:
            target_atom_support = np.zeros((E,), dtype=np.float32)
            target_atom_abs = np.zeros((E,), dtype=np.float32)

        def recovery_row(sel_tuple: tuple[int, ...]) -> tuple[tuple[int, ...], int, float, float, float]:
            action, scores, margins = evaluate(list(sel_tuple))
            _, distortion = distortion_components(list(sel_tuple))
            score_deficit = 0.0
            if 0 <= target_action < scores.shape[0] and target_rivals.size:
                rr = target_rivals[target_rivals < scores.shape[0]]
                if rr.size:
                    trial_gap = float(scores[target_action] - np.max(scores[rr]))
                    score_deficit = max(0.0, target_raw_gap - trial_gap) / max(target_scale, 1e-6)
            margin_deficit = 0.0
            if (
                target_margins.ndim == 2 and margins.ndim == 2
                and 0 <= target_action < min(target_margins.shape[0], margins.shape[0])
            ):
                kk = min(target_margins.shape[1], margins.shape[1], valid.shape[0])
                rr_mask = valid[:kk].copy()
                if target_action < kk:
                    rr_mask[target_action] = False
                tr = target_margins[target_action, :kk]
                xr = margins[target_action, :kk]
                informative = rr_mask & np.isfinite(tr) & np.isfinite(xr) & (
                    np.abs(tr) >= max(float(margin_floor), 0.0)
                )
                if bool(informative.any()):
                    margin_deficit = float(np.mean((tr[informative] * xr[informative] < 0.0).astype(np.float32)))
            potential = float(distortion + score_deficit + 0.5 * margin_deficit)
            return sel_tuple, int(action), float(distortion), potential, float(_spent_for(list(sel_tuple), costs))

        def prune_beam(rows: list[tuple[tuple[int, ...], int, float, float, float]]) -> list[tuple[int, ...]]:
            # Exact duplicates arise from different parent/deletion paths.
            unique: dict[tuple[int, ...], tuple[tuple[int, ...], int, float, float, float]] = {}
            for row in rows:
                old = unique.get(row[0])
                if old is None or (row[1] != target_action, row[3], row[2], row[0]) < (
                    old[1] != target_action, old[3], old[2], old[0]
                ):
                    unique[row[0]] = row
            vals = list(unique.values())
            preserving = sorted(
                (row for row in vals if row[1] == target_action),
                key=lambda row: (row[2], row[3], row[0]),
            )
            mismatching = sorted(
                (row for row in vals if row[1] != target_action),
                key=lambda row: (row[3], row[2], row[1], row[0]),
            )
            mismatch_slots = min(
                bw,
                max(1, int(round(float(bw) * float(np.clip(beam_mismatch_fraction, 0.0, 1.0)))))
                if mismatching else 0,
            )
            preserving_slots = max(0, bw - mismatch_slots)
            chosen: list[tuple[tuple[int, ...], int, float, float, float]] = preserving[:preserving_slots]

            # Action diversity prevents all exploratory slots from collapsing to
            # one wrong winner.  Keep the best state for each wrong action first.
            seen_actions: set[int] = set()
            for row in mismatching:
                if len(chosen) >= preserving_slots + mismatch_slots:
                    break
                if row[1] in seen_actions:
                    continue
                chosen.append(row)
                seen_actions.add(row[1])
            if len(chosen) < preserving_slots + mismatch_slots:
                chosen_keys = {row[0] for row in chosen}
                for row in mismatching:
                    if len(chosen) >= preserving_slots + mismatch_slots:
                        break
                    if row[0] not in chosen_keys:
                        chosen.append(row)
                        chosen_keys.add(row[0])

            # If either pool is smaller than its reservation, fill by the global
            # recovery ranking rather than leaving beam capacity unused.
            if len(chosen) < bw:
                chosen_keys = {row[0] for row in chosen}
                rest = sorted(
                    (row for row in vals if row[0] not in chosen_keys),
                    key=lambda row: (row[1] != target_action, row[3], row[2], row[1], row[0]),
                )
                chosen.extend(rest[: max(0, bw - len(chosen))])
            return [row[0] for row in chosen[:bw]]

        desired_removed = max(0, len(ids) - len(selected))
        beam: list[tuple[int, ...]] = [tuple(sorted(ids))]
        terminals: list[tuple[tuple[int, ...], int, float, float, float]] = []
        for depth in range(desired_removed):
            beam_depth = depth + 1
            expanded_rows: list[tuple[tuple[int, ...], int, float, float, float]] = []
            for state in beam:
                state_list = list(state)
                state_soft_count = int(sum(1 for i in state_list if soft[i]))
                state_margin = base + d[np.asarray(state_list, dtype=np.int64)].sum(axis=0)
                removable: list[tuple[float, float, float, float, int]] = []
                for i in state_list:
                    if soft[i] and state_soft_count - 1 < soft_floor:
                        continue
                    trial_margin = state_margin - d[i]
                    cheap = float(screen_loss(trial_margin) / max(float(costs[i]), 1e-6))
                    removable.append((cheap, float(proposal[i]), float(target_atom_abs[i]), float(target_atom_support[i]), int(i)))
                if not removable:
                    continue

                # Build a diverse branch: normal deployment distortion screen,
                # low target-impact deletions, low proposal-prior deletions, and
                # (for recovery) atoms whose oriented contribution hurts target.
                selected_removals: list[int] = []
                def add_ranked(seq, count: int) -> None:
                    for row in seq:
                        i = int(row[-1])
                        if i not in selected_removals:
                            selected_removals.append(i)
                            if len(selected_removals) >= count:
                                break

                n_screen = max(1, bb // 2)
                n_aux = max(1, bb // 4)
                by_screen = sorted(removable, key=lambda row: (row[0], row[1], row[4]))
                by_abs = sorted(removable, key=lambda row: (row[2], row[0], row[4]))
                by_proposal = sorted(removable, key=lambda row: (row[1], row[0], row[4]))
                by_harm = sorted(removable, key=lambda row: (row[3], row[0], row[4]))
                add_ranked(by_screen, n_screen)
                add_ranked(by_abs, n_screen + n_aux)
                add_ranked(by_harm, n_screen + 2 * n_aux)
                add_ranked(by_proposal, bb)
                if len(selected_removals) < bb:
                    add_ranked(by_screen, bb)

                for i in selected_removals[:bb]:
                    if beam_eval_cap > 0 and len(eval_cache) - beam_eval_start >= beam_eval_cap:
                        break
                    child = tuple(j for j in state if j != i)
                    row = recovery_row(child)
                    # A fixed-cardinality beam is used so successful repair is
                    # directly executable and cannot rely on later force-fill.
                    expanded_rows.append(row)
                if beam_eval_cap > 0 and len(eval_cache) - beam_eval_start >= beam_eval_cap:
                    break

            if not expanded_rows:
                break
            beam = prune_beam(expanded_rows)
            beam_peak_width = max(beam_peak_width, len(beam))
            if depth + 1 == desired_removed:
                terminals.extend(recovery_row(state) for state in beam)
            if beam_eval_cap > 0 and len(eval_cache) - beam_eval_start >= beam_eval_cap:
                break

        beam_evals = max(0, len(eval_cache) - beam_eval_start)
        beam_terminal_count = len(terminals)
        feasible = [
            row for row in terminals
            if row[1] == target_action and row[4] <= float(budget) + 1e-6
        ]
        if feasible:
            best_state, _, _, _, best_spent = min(feasible, key=lambda row: (row[2], row[3], row[0]))
            selected = list(best_state)
            spent = float(best_spent)
            current_surrogate = base + d[np.asarray(selected, dtype=np.int64)].sum(axis=0)
            selected_action = int(target_action)
            beam_success = True
            repair_success = True


    # v42 CBL-DACC: v41's deletion-lattice beam spent most of its exact
    # evaluations at cardinalities larger than the executable budget and kept
    # only ``beam_width`` terminal subsets.  Search the fixed-budget exchange
    # graph instead.  Each edge is a one-out/one-in counterfactual, so every
    # evaluated state is directly executable.  The search is rival-directed:
    # it ranks a mismatching subset by the exact target-action deficit against
    # the action currently selected by the deployment operator, not only by
    # reconstruction error to the full Top-M scores.  It may traverse several
    # mismatching subsets before recovering the target action.
    selected_action, _ = distortion_components(selected)
    layer_width = max(0, int(budget_layer_width))
    layer_branch = max(0, int(budget_layer_branch))
    layer_iters = max(0, int(budget_layer_iterations))
    layer_eval_cap = max(0, int(budget_layer_max_evaluations))
    layer_seed_cap = max(0, int(budget_layer_seed_count))
    layer_min_distance = max(0, int(budget_layer_diversity_distance))
    if (
        bool(lexicographic_action_preservation)
        and target_action >= 0
        and selected_action != target_action
        and layer_width > 0
        and layer_branch > 0
        and layer_iters > 0
        and len(selected) > 0
        and len(ids) > len(selected)
    ):
        budget_layer_attempted = True
        layer_eval_start = len(eval_cache)
        executable_size = int(len(selected))
        id_set = set(map(int, ids))

        # Map each rival to the pair columns oriented as target-minus-rival.
        target_pair_cols: dict[int, list[tuple[int, float, float]]] = {}
        for pidx, (a_raw, b_raw) in enumerate(pairs.tolist()):
            a_i, b_i = int(a_raw), int(b_raw)
            if a_i == target_action:
                target_pair_cols.setdefault(b_i, []).append((int(pidx), 1.0, float(max(weights[pidx], 0.0))))
            elif b_i == target_action:
                target_pair_cols.setdefault(a_i, []).append((int(pidx), -1.0, float(max(weights[pidx], 0.0))))

        def oriented_atom_support(rival: int) -> np.ndarray:
            cols = target_pair_cols.get(int(rival), [])
            if not cols:
                return np.zeros((E,), dtype=np.float32)
            out = np.zeros((E,), dtype=np.float32)
            denom = 0.0
            for pidx, orient, weight in cols:
                w = max(float(weight), 1e-3)
                out += float(orient) * w * d[:, int(pidx)]
                denom += w
            return out / max(denom, 1e-6)

        global_target_support = np.zeros((E,), dtype=np.float32)
        global_support_weight = 0.0
        for rival, cols in target_pair_cols.items():
            closeness = 1.0
            if (
                target_margins.ndim == 2
                and 0 <= target_action < target_margins.shape[0]
                and 0 <= int(rival) < target_margins.shape[1]
                and np.isfinite(target_margins[target_action, int(rival)])
            ):
                closeness = 1.0 / (0.10 + abs(float(target_margins[target_action, int(rival)])))
            global_target_support += float(closeness) * oriented_atom_support(int(rival))
            global_support_weight += float(closeness)
        if global_support_weight > 0.0:
            global_target_support /= float(global_support_weight)

        # A row contains the exact deployment state plus a stage-aware recovery
        # certificate.  v42 ranked raw tournament scores even when the final
        # action was changed later by utility refinement.  In that case the raw
        # target rank can already be one and the old deficit is exactly zero, so
        # the search has no useful gradient.  v43 carries the downstream stage
        # diagnostics back into the coreset search and measures the distance to
        # the actual decision boundary that changed the final action.
        def budget_layer_row(sel_tuple: tuple[int, ...]) -> tuple[
            tuple[int, ...], int, float, int, float, float, float, float, int, float, int
        ]:
            action, scores, margins = evaluate(list(sel_tuple))
            deploy_diag = evaluation_diagnostics(sel_tuple)
            _, distortion = distortion_components(list(sel_tuple))
            m = min(int(scores.shape[0]), int(eligible.shape[0]))
            target_rank = max(1, int(np.sum(eligible[:m])))
            max_gap_deficit = 0.0
            selected_gap_deficit = 0.0
            margin_deficit = 0.0
            raw_action = int(deploy_diag.get("utility_refinement_action_before", -1))
            if not (0 <= raw_action < m):
                raw_action = int(np.argmax(np.where(eligible[:m], scores[:m], -np.inf))) if m > 0 else -1
            stage = 0 if int(action) == int(target_action) else (2 if raw_action == int(target_action) else 1)
            stage_violation = 0.0
            if 0 <= target_action < m and bool(eligible[target_action]):
                rivals = np.flatnonzero(eligible[:m] & (np.arange(m) != int(target_action)))
                if rivals.size:
                    target_score = float(scores[target_action])
                    rival_scores = scores[rivals]
                    target_rank = 1 + int(np.sum(rival_scores > target_score + 1e-7))
                    max_gap_deficit = max(0.0, float(np.max(rival_scores)) - target_score) / max(target_scale, 1e-6)
                if 0 <= int(action) < m and int(action) != int(target_action):
                    selected_gap_deficit = max(
                        0.0, float(scores[int(action)]) - float(scores[target_action])
                    ) / max(target_scale, 1e-6)
                    if (
                        margins.ndim == 2
                        and target_action < margins.shape[0]
                        and int(action) < margins.shape[1]
                        and np.isfinite(margins[target_action, int(action)])
                    ):
                        margin_deficit = max(0.0, -float(margins[target_action, int(action)]))
                if margins.ndim == 2 and target_action < margins.shape[0]:
                    kk = min(margins.shape[1], valid.shape[0])
                    rr = valid[:kk].copy()
                    if target_action < kk:
                        rr[target_action] = False
                    row = margins[target_action, :kk]
                    finite = rr & np.isfinite(row)
                    if bool(finite.any()):
                        margin_deficit += 0.25 * float(np.mean(np.maximum(-row[finite], 0.0)))

                if stage == 2 and 0 <= int(action) < m:
                    # Utility refinement selected a cheaper rival even though the
                    # target is the post-safety score winner.  Evidence can recover
                    # the target by excluding that rival from either the score band
                    # or the pair-certificate band.  Use the smaller exact boundary
                    # distance; trajectory utility itself is fixed and cannot be
                    # changed by evidence selection.
                    slack = max(float(deploy_diag.get("utility_score_slack", 0.0) or 0.0), 0.0)
                    score_gap = float(scores[target_action] - scores[int(action)])
                    band_distance = max(0.0, slack - score_gap) / max(target_scale, 1e-6)
                    cert_distance = float("inf")
                    pair_cert_enabled = bool(deploy_diag.get("utility_pair_certificate_enabled", False))
                    tol = max(float(deploy_diag.get("utility_pair_margin_tolerance", 0.0) or 0.0), 0.0)
                    if (
                        pair_cert_enabled and margins.ndim == 2
                        and int(action) < margins.shape[0] and target_action < margins.shape[1]
                        and np.isfinite(margins[int(action), target_action])
                    ):
                        # Candidate action remains utility-eligible while
                        # M[action,target] >= -tol.  Crossing below -tol excludes it.
                        cert_distance = max(0.0, float(margins[int(action), target_action]) + tol) / max(tol, 0.05)
                    stage_violation = min(band_distance, cert_distance)
                    if not np.isfinite(stage_violation):
                        stage_violation = band_distance
                elif stage == 1:
                    stage_violation = float(max_gap_deficit + selected_gap_deficit + margin_deficit)
            action_deficit = float(max_gap_deficit + selected_gap_deficit)
            return (
                tuple(sel_tuple), int(action), float(distortion), int(target_rank),
                float(action_deficit), float(margin_deficit),
                float(_spent_for(list(sel_tuple), costs)),
                float(screen_loss(base + d[np.asarray(sel_tuple, dtype=np.int64)].sum(axis=0))),
                int(stage), float(stage_violation), int(raw_action),
            )

        def layer_key(row: tuple[tuple[int, ...], int, float, int, float, float, float, float, int, float, int]) -> tuple:
            return (
                int(row[1] != target_action), float(row[9]), int(row[3]),
                float(row[4]), float(row[5]), float(row[2]), float(row[7]), row[0],
            )

        def prune_budget_layer(rows: list[tuple]) -> list[tuple]:
            unique: dict[tuple[int, ...], tuple] = {}
            for row in rows:
                old = unique.get(row[0])
                if old is None or layer_key(row) < layer_key(old):
                    unique[row[0]] = row
            ranked = sorted(unique.values(), key=layer_key)
            if len(ranked) <= layer_width:
                return ranked

            chosen: list[tuple] = []
            chosen_keys: set[tuple[int, ...]] = set()
            seen_actions: set[int] = set()
            action_slots = max(1, layer_width // 4)
            for row in ranked:
                if len(chosen) >= action_slots:
                    break
                if int(row[1]) in seen_actions:
                    continue
                chosen.append(row)
                chosen_keys.add(row[0])
                seen_actions.add(int(row[1]))

            exploit_slots = max(action_slots, int(round(0.65 * layer_width)))
            for row in ranked:
                if len(chosen) >= exploit_slots:
                    break
                if row[0] not in chosen_keys:
                    chosen.append(row)
                    chosen_keys.add(row[0])

            # Fill exploration slots with set-diverse states from a bounded high-
            # quality pool.  This lets the search cross a multi-swap valley while
            # retaining deterministic behavior.
            pool = [row for row in ranked[: max(layer_width * 8, layer_width)] if row[0] not in chosen_keys]
            while len(chosen) < layer_width and pool:
                best_idx = 0
                best_div = -1
                best_rank = None
                for idx, row in enumerate(pool):
                    if chosen:
                        rset = set(row[0])
                        diversity = min(len(rset.symmetric_difference(set(old[0]))) for old in chosen)
                    else:
                        diversity = executable_size * 2
                    qualifies = int(diversity >= layer_min_distance)
                    cand = (-qualifies, -diversity, layer_key(row))
                    if best_rank is None or cand < best_rank:
                        best_idx = idx
                        best_div = diversity
                        best_rank = cand
                row = pool.pop(best_idx)
                chosen.append(row)
                chosen_keys.add(row[0])
            if len(chosen) < layer_width:
                for row in ranked:
                    if len(chosen) >= layer_width:
                        break
                    if row[0] not in chosen_keys:
                        chosen.append(row)
                        chosen_keys.add(row[0])
            return chosen[:layer_width]

        def make_seed(order: np.ndarray) -> tuple[int, ...] | None:
            picked: list[int] = []
            picked_set: set[int] = set()
            spent_seed = 0.0
            ordered_ids = [int(i) for i in order.tolist() if int(i) in id_set]
            # Satisfy the interaction floor first, then fill by the same order.
            for want_soft in (True, False):
                for i in ordered_ids:
                    if i in picked_set or (want_soft and not bool(soft[i])):
                        continue
                    if want_soft and sum(1 for j in picked if soft[j]) >= soft_floor:
                        break
                    c = float(costs[i])
                    if spent_seed + c <= float(budget) + 1e-6:
                        picked.append(i); picked_set.add(i); spent_seed += c
                        if len(picked) >= executable_size:
                            break
                if len(picked) >= executable_size:
                    break
            if len(picked) < executable_size:
                for i in ordered_ids:
                    if i in picked_set:
                        continue
                    c = float(costs[i])
                    if spent_seed + c <= float(budget) + 1e-6:
                        picked.append(i); picked_set.add(i); spent_seed += c
                        if len(picked) >= executable_size:
                            break
            if len(picked) != executable_size or sum(1 for i in picked if soft[i]) < soft_floor:
                return None
            return tuple(sorted(picked))

        initial_states: list[tuple[int, ...]] = [tuple(sorted(map(int, selected)))]
        if layer_seed_cap > 0:
            orders: list[np.ndarray] = [
                np.argsort(-global_target_support, kind="stable"),
                np.argsort(-proposal, kind="stable"),
                np.argsort(-(global_target_support + 0.10 * proposal), kind="stable"),
                np.argsort(np.abs(global_target_support), kind="stable"),
            ]
            for order in orders[:layer_seed_cap]:
                seed = make_seed(order)
                if seed is not None and seed not in initial_states:
                    initial_states.append(seed)
        budget_layer_seed_states = len(initial_states)

        frontier_rows = prune_budget_layer([budget_layer_row(state) for state in initial_states])
        budget_layer_peak_width = max(budget_layer_peak_width, len(frontier_rows))
        seen_layer_states: set[tuple[int, ...]] = {row[0] for row in frontier_rows}
        preserving_rows = [row for row in frontier_rows if row[1] == target_action and row[6] <= float(budget) + 1e-6]

        for iteration in range(layer_iters):
            if preserving_rows:
                break
            budget_layer_iterations_done = iteration + 1
            expanded: list[tuple[tuple[int, ...], int, float, int, float, float, float, float]] = []
            for state_idx, state_row in enumerate(frontier_rows):
                state = state_row[0]
                state_set = set(state)
                outside = sorted(id_set - state_set)
                state_soft_count = int(sum(1 for i in state if soft[i]))
                state_margin = base + d[np.asarray(state, dtype=np.int64)].sum(axis=0)
                rival = int(state_row[1])
                if rival == target_action or rival not in target_pair_cols:
                    _, scores, _ = evaluate(list(state))
                    rr = np.flatnonzero(eligible[: min(len(scores), len(eligible))] & (
                        np.arange(min(len(scores), len(eligible))) != int(target_action)
                    ))
                    if rr.size and 0 <= target_action < scores.shape[0]:
                        rival = int(rr[int(np.argmax(scores[rr]))])
                rival_support = oriented_atom_support(rival)

                swap_rows: list[tuple[float, float, float, float, int, int, float]] = []
                for out_i in state:
                    for in_i in outside:
                        if soft[out_i] and not soft[in_i] and state_soft_count - 1 < soft_floor:
                            continue
                        new_spent = float(state_row[6]) - float(costs[out_i]) + float(costs[in_i])
                        if new_spent > float(budget) + 1e-6:
                            continue
                        trial_margin = state_margin - d[out_i] + d[in_i]
                        cheap = float(screen_loss(trial_margin))
                        rival_gain = float(rival_support[in_i] - rival_support[out_i])
                        global_gain = float(global_target_support[in_i] - global_target_support[out_i])
                        proposal_gain = float(proposal[in_i] - proposal[out_i])
                        swap_rows.append((cheap, rival_gain, global_gain, proposal_gain, int(out_i), int(in_i), new_spent))
                if not swap_rows:
                    continue

                chosen_swaps: list[tuple[float, float, float, float, int, int, float]] = []
                exhaustive = bool(budget_layer_exhaustive_first) and iteration == 0 and state_idx == 0
                if exhaustive:
                    chosen_swaps = sorted(swap_rows, key=lambda row: (row[4], row[5]))
                else:
                    def add_swaps(seq, target_count: int) -> None:
                        keys = {(row[4], row[5]) for row in chosen_swaps}
                        for row in seq:
                            key = (row[4], row[5])
                            if key not in keys:
                                chosen_swaps.append(row); keys.add(key)
                                if len(chosen_swaps) >= target_count:
                                    break
                    q = max(1, layer_branch // 4)
                    add_swaps(sorted(swap_rows, key=lambda row: (row[0], -row[1], row[4], row[5])), q)
                    add_swaps(sorted(swap_rows, key=lambda row: (-row[1], row[0], row[4], row[5])), 2 * q)
                    add_swaps(sorted(swap_rows, key=lambda row: (-row[2], row[0], row[4], row[5])), 3 * q)
                    add_swaps(sorted(swap_rows, key=lambda row: (-row[3], row[0], row[4], row[5])), layer_branch)
                    if len(chosen_swaps) < layer_branch:
                        add_swaps(sorted(swap_rows, key=lambda row: (row[0], row[4], row[5])), layer_branch)

                for _, _, _, _, out_i, in_i, _ in chosen_swaps[: (len(chosen_swaps) if exhaustive else layer_branch)]:
                    if layer_eval_cap > 0 and len(eval_cache) - layer_eval_start >= layer_eval_cap:
                        break
                    child = tuple(sorted((state_set - {int(out_i)}) | {int(in_i)}))
                    if child in seen_layer_states:
                        continue
                    seen_layer_states.add(child)
                    row = budget_layer_row(child)
                    expanded.append(row)
                    if row[1] == target_action and row[6] <= float(budget) + 1e-6:
                        preserving_rows.append(row)
                if preserving_rows or (layer_eval_cap > 0 and len(eval_cache) - layer_eval_start >= layer_eval_cap):
                    break
            if preserving_rows or not expanded:
                break
            frontier_rows = prune_budget_layer(expanded)
            budget_layer_peak_width = max(budget_layer_peak_width, len(frontier_rows))
            if layer_eval_cap > 0 and len(eval_cache) - layer_eval_start >= layer_eval_cap:
                break

        budget_layer_evals = max(0, len(eval_cache) - layer_eval_start)
        budget_layer_unique_states = len(seen_layer_states)
        all_rows = list(frontier_rows) + list(preserving_rows)
        if all_rows:
            best_recovery = min(all_rows, key=layer_key)
            budget_layer_best_target_rank = int(best_recovery[3])
            budget_layer_best_action_deficit = float(best_recovery[4])
            budget_layer_best_margin_deficit = float(best_recovery[5])
            budget_layer_best_stage = int(best_recovery[8])
            budget_layer_best_stage_violation = float(best_recovery[9])
            budget_layer_best_raw_action = int(best_recovery[10])
        if preserving_rows:
            best = min(preserving_rows, key=lambda row: (row[2], row[4], row[5], row[0]))
            selected = list(best[0])
            spent = float(best[6])
            current_surrogate = base + d[np.asarray(selected, dtype=np.int64)].sum(axis=0)
            selected_action = int(target_action)
            budget_layer_success = True
            repair_success = True

    selected = sorted(selected)
    selected_action, selected_scores, _ = evaluate(selected)
    final_loss = exact_loss(selected)
    _, final_distortion = distortion_components(selected)
    score_rmse = 0.0
    m = min(target_scores.shape[0], selected_scores.shape[0], valid.shape[0])
    mask = (
        valid[:m] & np.isfinite(target_scores[:m]) & np.isfinite(selected_scores[:m])
        & (target_scores[:m] > -1e8) & (selected_scores[:m] > -1e8)
    )
    if bool(mask.any()):
        score_rmse = float(np.sqrt(np.mean(np.square(selected_scores[:m][mask] - target_scores[:m][mask]))))
    diag = {
        "deployment_coreset_active": True,
        "deployment_coreset_lexicographic_active": bool(lexicographic_action_preservation),
        "deployment_coreset_target_atom_count": int(len(ids)),
        "deployment_coreset_selected_atom_count": int(len(selected)),
        "deployment_coreset_removed_atom_count": int(len(removed)),
        "deployment_coreset_swap_count": int(swaps),
        "deployment_coreset_target_action": int(target_action),
        "deployment_coreset_selected_action": int(selected_action),
        "deployment_coreset_target_action_preserved": float(target_action >= 0 and selected_action == target_action),
        "deployment_coreset_score_rmse": float(score_rmse),
        "deployment_coreset_distortion_objective": float(final_distortion),
        "deployment_coreset_objective": float(final_loss),
        "deployment_coreset_soft_floor": int(soft_floor),
        "deployment_coreset_evaluations": int(len(eval_cache)),
        "deployment_coreset_preservation_scan_evaluations": int(preservation_scan_evals),
        "deployment_coreset_forced_action_flip_steps": int(forced_action_flip_steps),
        "deployment_coreset_repair_attempted": bool(repair_attempted),
        "deployment_coreset_repair_success": bool(repair_success),
        "deployment_coreset_repair_one_swap_evaluations": int(repair_one_swap_evals),
        "deployment_coreset_repair_two_swap_evaluations": int(repair_two_swap_evals),
        "deployment_coreset_beam_attempted": bool(beam_attempted),
        "deployment_coreset_beam_success": bool(beam_success),
        "deployment_coreset_beam_evaluations": int(beam_evals),
        "deployment_coreset_beam_depth": int(beam_depth),
        "deployment_coreset_beam_peak_width": int(beam_peak_width),
        "deployment_coreset_beam_terminal_count": int(beam_terminal_count),
        "deployment_coreset_budget_layer_attempted": bool(budget_layer_attempted),
        "deployment_coreset_budget_layer_success": bool(budget_layer_success),
        "deployment_coreset_budget_layer_evaluations": int(budget_layer_evals),
        "deployment_coreset_budget_layer_iterations": int(budget_layer_iterations_done),
        "deployment_coreset_budget_layer_iteration_limit": int(layer_iters),
        "deployment_coreset_budget_layer_peak_width": int(budget_layer_peak_width),
        "deployment_coreset_budget_layer_unique_states": int(budget_layer_unique_states),
        "deployment_coreset_budget_layer_seed_states": int(budget_layer_seed_states),
        "deployment_coreset_budget_layer_best_target_rank": int(budget_layer_best_target_rank),
        "deployment_coreset_budget_layer_best_action_deficit": float(budget_layer_best_action_deficit),
        "deployment_coreset_budget_layer_best_margin_deficit": float(budget_layer_best_margin_deficit),
        "deployment_coreset_budget_layer_best_stage": int(budget_layer_best_stage),
        "deployment_coreset_budget_layer_best_stage_violation": float(budget_layer_best_stage_violation),
        "deployment_coreset_budget_layer_best_raw_action": int(budget_layer_best_raw_action),
    }
    return selected, float(-final_loss), float(spent), diag


def _build_anytime_one_sided_adverse_certificate_state(
    atom_delta: np.ndarray,
    base_margin: np.ndarray,
    pair_weights: np.ndarray,
    pair_indices: np.ndarray,
    atom_budget_costs: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    atom_active_mask: np.ndarray | None = None,
    *,
    atom_pair_variance: np.ndarray | None = None,
    adverse_beta: float = 1.0,
    adverse_epsilon: float = 0.05,
    prior_radius: float = 0.10,
    certificate_margin: float = 0.0,
    boundary_tau: float = 0.25,
    stop_when_certified: bool = True,
    max_target_rivals: int = 0,
    target_action_hint: int | None = None,
    explicitly_calibrated: bool = False,
    family_ids: np.ndarray | None = None,
    interaction_family_ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    max_interaction_prefix_fraction: float = 1.0,
    fill_to_budget_after_certified: bool = False,
) -> dict[str, Any]:
    """Build the budget-independent AOCC greedy order once."""
    d = np.asarray(atom_delta, dtype=np.float32)
    base = np.asarray(base_margin, dtype=np.float32).reshape(-1)
    pairs = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    valid_arr = np.asarray(valid_mask, dtype=bool).reshape(-1)
    flags_arr = np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)
    E = int(d.shape[0]) if d.ndim == 2 else int(costs.shape[0])
    if d.ndim != 2 or d.shape[1] != base.shape[0]:
        d = np.zeros((E, base.shape[0]), dtype=np.float32)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)
    costs = costs[:E]
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else _as_bool_mask(atom_active_mask, E)
    active &= np.isfinite(costs) & (costs > 0.0)
    fam = np.full((E,), -999, dtype=np.int64)
    if family_ids is not None:
        raw_fam = np.asarray(family_ids, dtype=np.int64).reshape(-1)
        fam[: min(E, raw_fam.shape[0])] = raw_fam[: min(E, raw_fam.shape[0])]
    interaction_ids = set(int(x) for x in np.asarray(
        interaction_family_ids if interaction_family_ids is not None else [],
        dtype=np.int64,
    ).reshape(-1).tolist())
    interaction_atom = np.asarray([int(x) in interaction_ids for x in fam.tolist()], dtype=bool) & active
    max_interaction_prefix_fraction = float(np.clip(float(max_interaction_prefix_fraction), 0.0, 1.0))
    if weights.shape[0] != pairs.shape[0]:
        weights = np.ones((pairs.shape[0],), dtype=np.float32)
    weights = np.maximum(weights[: pairs.shape[0]], 0.0)
    var_input = None if atom_pair_variance is None else np.asarray(atom_pair_variance, dtype=np.float32)
    cache_inputs = {
        "d": d,
        "base": base,
        "pairs": pairs,
        "weights": weights,
        "costs": costs,
        "active": active,
        "valid": valid_arr,
        "flags": flags_arr,
        "var": var_input,
        "family": fam,
        "interaction_atom": interaction_atom,
        "params": (
            float(adverse_beta),
            float(adverse_epsilon),
            float(prior_radius),
            float(certificate_margin),
            float(boundary_tau),
            bool(stop_when_certified),
            int(max_target_rivals),
            None if target_action_hint is None else int(target_action_hint),
            bool(explicitly_calibrated),
            tuple(sorted(interaction_ids)),
            float(max_interaction_prefix_fraction),
            bool(fill_to_budget_after_certified),
        ),
    }
    if pairs.shape[0] == 0 or not bool(active.any()):
        return {
            "terminal": True,
            "cache_inputs": cache_inputs,
            "diag": {
                "aocc_target_action": -1,
                "aocc_target_confidence": 0.0,
                "aocc_pair_count": 0,
                "aocc_initial_deficit": 0.0,
                "aocc_final_deficit": 0.0,
                "aocc_certified_pair_fraction": 1.0,
                "aocc_nested_order_length": 0,
                "aocc_bound_calibrated": bool(explicitly_calibrated),
                "aocc_target_source": "none",
            },
        }

    active_idx = np.flatnonzero(active)
    full_margin = base + d[active_idx].sum(axis=0)
    hinted = -1 if target_action_hint is None else int(target_action_hint)
    hint_valid = 0 <= hinted < valid_arr.shape[0] and bool(valid_arr[hinted])
    if hint_valid:
        target_action = hinted
        target_conf = 1.0
        target_source = "deployment_tournament"
    else:
        target_action, target_conf = _pairwise_preference_action(
            full_margin, pairs, weights, valid_arr, flags_arr,
            tau=max(float(boundary_tau), 1e-3),
        )
        target_source = "pairwise_preference"
    if target_action < 0:
        return {
            "terminal": True,
            "cache_inputs": cache_inputs,
            "diag": {
                "aocc_target_action": -1,
                "aocc_target_confidence": 0.0,
                "aocc_pair_count": 0,
                "aocc_initial_deficit": 0.0,
                "aocc_final_deficit": 0.0,
                "aocc_certified_pair_fraction": 0.0,
                "aocc_nested_order_length": 0,
                "aocc_bound_calibrated": bool(explicitly_calibrated),
                "aocc_target_source": str(target_source),
            },
        }

    orient = np.zeros((pairs.shape[0],), dtype=np.float32)
    orient[pairs[:, 0] == int(target_action)] = 1.0
    orient[pairs[:, 1] == int(target_action)] = -1.0
    pair_keep = (orient != 0.0) & (weights > 0.0)
    frontier_original_count = int(pair_keep.sum())
    frontier_original_weight = float(weights[pair_keep].sum()) if frontier_original_count else 0.0
    # V48 tournament-active rival frontier.  Requiring every pair incident to the
    # target to be individually certified made the certificate far more
    # conservative than the soft-min tournament it was intended to protect.  Keep
    # the closest target/rival margins and all safety-crossing rivals, then cap the
    # frontier deterministically.  This is still a deployment-only construction:
    # it uses predicted full margins, pair weights, and runtime safety flags.
    if bool(pair_keep.any()) and int(max_target_rivals) > 0 and int(pair_keep.sum()) > int(max_target_rivals):
        ids = np.flatnonzero(pair_keep)
        oriented_full = full_margin[ids] * orient[ids]
        closest = float(np.min(oriented_full))
        frontier_tau = max(float(boundary_tau), 1e-3)
        flags_safe = np.asarray(flags_arr, dtype=bool).reshape(-1)
        a_ids = pairs[ids, 0].clip(0, max(flags_safe.shape[0] - 1, 0))
        b_ids = pairs[ids, 1].clip(0, max(flags_safe.shape[0] - 1, 0))
        safety_cross = flags_safe[a_ids] ^ flags_safe[b_ids] if flags_safe.size else np.zeros((ids.size,), dtype=bool)
        frontier_mask = (oriented_full <= closest + frontier_tau) | safety_cross
        frontier_ids = ids[frontier_mask]
        min_frontier = min(max(3, int(np.ceil(np.sqrt(max(ids.size, 1))))), int(max_target_rivals))
        priority = (
            weights[ids] / (np.maximum(oriented_full - closest, 0.0) + frontier_tau)
            + 2.0 * safety_cross.astype(np.float32)
        )
        ranked = ids[np.argsort(-priority, kind="stable")]
        chosen_list = list(map(int, frontier_ids.tolist()))
        for idx in ranked.tolist():
            if int(idx) not in chosen_list:
                chosen_list.append(int(idx))
            if len(chosen_list) >= max(min_frontier, int(max_target_rivals)):
                break
        # Keep the highest-priority members when the safety/near frontier itself
        # exceeds the requested cap.
        chosen_set = set(chosen_list)
        chosen = np.asarray([int(i) for i in ranked.tolist() if int(i) in chosen_set][: int(max_target_rivals)], dtype=np.int64)
        pair_keep[:] = False
        pair_keep[chosen] = True
    frontier_retained_weight = (
        float(weights[pair_keep].sum()) / max(frontier_original_weight, 1e-9)
        if frontier_original_count else 1.0
    )
    if not bool(pair_keep.any()):
        return {
            "terminal": True,
            "cache_inputs": cache_inputs,
            "diag": {
                "aocc_target_action": int(target_action),
                "aocc_target_confidence": float(target_conf),
                "aocc_pair_count": 0,
                "aocc_initial_deficit": 0.0,
                "aocc_final_deficit": 0.0,
                "aocc_certified_pair_fraction": 1.0,
                "aocc_nested_order_length": 0,
                "aocc_bound_calibrated": bool(explicitly_calibrated),
                "aocc_target_source": str(target_source),
            },
        }

    o = orient[pair_keep]
    w = weights[pair_keep].astype(np.float32)
    w = w / max(float(w.sum()), 1e-9)
    oriented_base = base[pair_keep] * o
    oriented_delta = d[:, pair_keep] * o[None, :]

    if var_input is not None:
        var = var_input
        if var.shape != d.shape:
            var = np.zeros_like(d, dtype=np.float32)
        sigma = np.sqrt(np.maximum(var[:, pair_keep], 0.0) + max(float(prior_radius), 0.0) ** 2)
        has_learned_variance = bool(np.any(var[:, pair_keep] > 0.0))
    else:
        sigma = np.full_like(oriented_delta, max(float(prior_radius), 0.0), dtype=np.float32)
        has_learned_variance = False
    radius = max(float(adverse_beta), 0.0) * sigma + max(float(adverse_epsilon), 0.0)
    lower = np.minimum(0.0, oriented_delta - radius).astype(np.float32)
    improvement = np.maximum(oriented_delta - lower, 0.0).astype(np.float32)
    c0 = oriented_base + lower[active_idx].sum(axis=0)
    gamma = float(certificate_margin)
    deficit0 = np.maximum(gamma - c0, 0.0).astype(np.float32)

    order_gain_state = np.zeros_like(deficit0)
    full_order: list[int] = []
    full_order_gain: list[float] = []
    remaining = active.copy()
    selected_interaction = 0
    first_certified_prefix_length = 0
    first_certified_cost = 0.0
    while bool(remaining.any()):
        ids = np.flatnonzero(remaining)
        # Prefix family capacity keeps every nested prefix genuinely
        # cross-family. It is relaxed only if no non-interaction atom remains.
        if max_interaction_prefix_fraction < 1.0 and bool(interaction_atom[ids].any()):
            next_len = len(full_order) + 1
            max_interactions = int(np.ceil(max_interaction_prefix_fraction * next_len - 1e-12))
            non_inter_ids = ids[~interaction_atom[ids]]
            if selected_interaction >= max_interactions and non_inter_ids.size:
                ids = non_inter_ids
        before = np.minimum(deficit0, order_gain_state)
        after = np.minimum(deficit0[None, :], order_gain_state[None, :] + improvement[ids])
        marginal = ((after - before[None, :]) * w[None, :]).sum(axis=1)
        ratio = marginal / np.maximum(costs[ids], 1e-6)
        # Once certified, tighten the same lower bound to form a genuine exact-B
        # nested prefix. Added atoms cannot reduce the one-sided certificate.
        secondary = (improvement[ids] * w[None, :]).sum(axis=1) / np.maximum(costs[ids], 1e-6)
        primary_positive = bool(np.max(marginal, initial=0.0) > 1e-12)
        ranking = ratio if primary_positive else secondary
        # If the certificate is already complete and every remaining atom has
        # zero lower-bound gain, keep a deterministic cost-efficient order so
        # materialization can still expose the exact fixed-budget prefix.
        if bool(fill_to_budget_after_certified) and not bool(np.max(ranking, initial=0.0) > 1e-12):
            ranking = 1.0 / np.maximum(costs[ids], 1e-6)
        best_pos = int(np.lexsort((ids, costs[ids], -marginal, -ranking))[0])
        best = int(ids[best_pos])
        best_gain = float(marginal[best_pos])
        if best_gain <= 1e-12 and not bool(fill_to_budget_after_certified):
            break
        full_order.append(best)
        full_order_gain.append(best_gain)
        selected_interaction += int(interaction_atom[best])
        order_gain_state = order_gain_state + improvement[best]
        remaining[best] = False
        order_deficit_vec = np.maximum(deficit0 - order_gain_state, 0.0)
        if bool(np.all(order_deficit_vec <= 1e-8)) and first_certified_prefix_length == 0:
            first_certified_prefix_length = int(len(full_order))
            first_certified_cost = float(sum(float(costs[i]) for i in full_order))
        if bool(stop_when_certified) and not bool(fill_to_budget_after_certified) and bool(np.all(order_deficit_vec <= 1e-8)):
            break

    full_oriented = oriented_base + oriented_delta[active_idx].sum(axis=0)
    lower_violation = np.maximum(lower - oriented_delta, 0.0)
    return {
        "terminal": False,
        "cache_inputs": cache_inputs,
        "costs": costs,
        "full_order": full_order,
        "full_order_gain": full_order_gain,
        "improvement": improvement,
        "deficit0": deficit0,
        "weights_kept": w,
        "c0": c0,
        "gamma": gamma,
        "static_diag": {
            "aocc_target_action": int(target_action),
            "aocc_target_confidence": float(target_conf),
            "aocc_pair_count": int(pair_keep.sum()),
            "aocc_frontier_original_pair_count": int(frontier_original_count),
            "aocc_frontier_retained_weight_fraction": float(frontier_retained_weight),
            "aocc_initial_deficit": float(np.sum(w * deficit0)),
            "aocc_full_target_certified_pair_fraction": float(np.mean(full_oriented >= gamma - 1e-8)),
            "aocc_initial_certified_pair_fraction": float(np.mean(c0 >= gamma - 1e-8)),
            "aocc_nested_order_length": int(len(full_order)),
            "aocc_full_order_cost": float(sum(float(costs[i]) for i in full_order)),
            "aocc_first_certified_prefix_length": int(first_certified_prefix_length),
            "aocc_first_certified_cost": float(first_certified_cost),
            "aocc_fill_to_budget_after_certified": bool(fill_to_budget_after_certified),
            "aocc_max_interaction_prefix_fraction": float(max_interaction_prefix_fraction),
            "aocc_max_lower_bound_violation": float(np.max(lower_violation)) if lower_violation.size else 0.0,
            # Learned variance plus non-zero beta/epsilon is not evidence of an
            # independent calibration split.  Only an explicit provenance-backed
            # flag may mark the bound as calibrated.
            "aocc_bound_calibrated": bool(explicitly_calibrated),
            "aocc_has_learned_variance": bool(has_learned_variance),
            "aocc_target_source": str(target_source),
            "aocc_adverse_beta": float(adverse_beta),
            "aocc_adverse_epsilon": float(adverse_epsilon),
            "aocc_prior_radius": float(prior_radius),
            "aocc_certificate_margin": float(certificate_margin),
            "aocc_stop_when_certified": bool(stop_when_certified),
        },
    }


def _anytime_aocc_state_matches(
    state: dict[str, Any],
    atom_delta: np.ndarray,
    base_margin: np.ndarray,
    pair_weights: np.ndarray,
    pair_indices: np.ndarray,
    atom_budget_costs: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    atom_active_mask: np.ndarray | None,
    atom_pair_variance: np.ndarray | None,
    *,
    adverse_beta: float,
    adverse_epsilon: float,
    prior_radius: float,
    certificate_margin: float,
    boundary_tau: float,
    stop_when_certified: bool,
    max_target_rivals: int,
    target_action_hint: int | None,
    explicitly_calibrated: bool,
    family_ids: np.ndarray | None,
    interaction_family_ids: list[int] | tuple[int, ...] | np.ndarray | None,
    max_interaction_prefix_fraction: float,
    fill_to_budget_after_certified: bool,
) -> bool:
    """Exact guard for reusing an AOCC order across budgets."""
    saved = state.get("cache_inputs")
    if not isinstance(saved, dict):
        return False
    params = (
        float(adverse_beta), float(adverse_epsilon), float(prior_radius),
        float(certificate_margin), float(boundary_tau), bool(stop_when_certified),
        int(max_target_rivals), None if target_action_hint is None else int(target_action_hint),
        bool(explicitly_calibrated),
        tuple(sorted(int(x) for x in np.asarray(
            interaction_family_ids if interaction_family_ids is not None else [],
            dtype=np.int64,
        ).reshape(-1).tolist())),
        float(np.clip(float(max_interaction_prefix_fraction), 0.0, 1.0)),
        bool(fill_to_budget_after_certified),
    )
    if saved.get("params") != params:
        return False
    d = np.asarray(atom_delta, dtype=np.float32)
    base = np.asarray(base_margin, dtype=np.float32).reshape(-1)
    pairs = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    E = int(d.shape[0]) if d.ndim == 2 else int(costs.shape[0])
    if d.ndim != 2 or d.shape[1] != base.shape[0]:
        d = np.zeros((E, base.shape[0]), dtype=np.float32)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)
    costs = costs[:E]
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else _as_bool_mask(atom_active_mask, E)
    active &= np.isfinite(costs) & (costs > 0.0)
    fam = np.full((E,), -999, dtype=np.int64)
    if family_ids is not None:
        raw_fam = np.asarray(family_ids, dtype=np.int64).reshape(-1)
        fam[: min(E, raw_fam.shape[0])] = raw_fam[: min(E, raw_fam.shape[0])]
    interaction_ids = set(int(x) for x in np.asarray(
        interaction_family_ids if interaction_family_ids is not None else [],
        dtype=np.int64,
    ).reshape(-1).tolist())
    interaction_atom = np.asarray([int(x) in interaction_ids for x in fam.tolist()], dtype=bool) & active
    if weights.shape[0] != pairs.shape[0]:
        weights = np.ones((pairs.shape[0],), dtype=np.float32)
    weights = np.maximum(weights[: pairs.shape[0]], 0.0)
    var = None if atom_pair_variance is None else np.asarray(atom_pair_variance, dtype=np.float32)
    checks = (
        np.array_equal(saved.get("d"), d),
        np.array_equal(saved.get("base"), base),
        np.array_equal(saved.get("pairs"), pairs),
        np.array_equal(saved.get("weights"), weights),
        np.array_equal(saved.get("costs"), costs),
        np.array_equal(saved.get("active"), active),
        np.array_equal(saved.get("valid"), np.asarray(valid_mask, dtype=bool).reshape(-1)),
        np.array_equal(saved.get("flags"), np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)),
        np.array_equal(saved.get("family"), fam),
        np.array_equal(saved.get("interaction_atom"), interaction_atom),
        (saved.get("var") is None and var is None)
        or (saved.get("var") is not None and var is not None and np.array_equal(saved.get("var"), var)),
    )
    return bool(all(checks))


def _materialize_anytime_one_sided_adverse_certificate(
    state: dict[str, Any],
    budget: float,
) -> tuple[list[int], float, float, dict[str, Any]]:
    if bool(state.get("terminal", False)):
        return [], 0.0, 0.0, dict(state.get("diag", {}))
    costs = np.asarray(state["costs"], dtype=np.float32)
    full_order = list(map(int, state["full_order"]))
    selected: list[int] = []
    spent = 0.0
    for atom_id in full_order:
        next_spent = spent + float(costs[atom_id])
        if next_spent > float(budget) + 1e-6:
            break
        selected.append(int(atom_id))
        spent = next_spent

    improvement = np.asarray(state["improvement"], dtype=np.float32)
    deficit0 = np.asarray(state["deficit0"], dtype=np.float32)
    w = np.asarray(state["weights_kept"], dtype=np.float32)
    c0 = np.asarray(state["c0"], dtype=np.float32)
    gamma = float(state["gamma"])
    if selected:
        current_gain = improvement[np.asarray(selected, dtype=np.int64)].sum(axis=0)
    else:
        current_gain = np.zeros_like(deficit0)
    objective = float(np.sum(w * np.minimum(deficit0, current_gain)))
    final_certificate = c0 + current_gain
    final_deficit = np.maximum(gamma - final_certificate, 0.0)
    full_order_gain = list(map(float, state["full_order_gain"]))
    selected_order_gain = full_order_gain[: len(selected)]
    diag = {
        **dict(state["static_diag"]),
        "aocc_final_deficit": float(np.sum(w * final_deficit)),
        "aocc_deficit_reduction": float(np.sum(w * (deficit0 - final_deficit))),
        "aocc_certified_pair_fraction": float(np.mean(final_certificate >= gamma - 1e-8)),
        "aocc_selected_prefix_length": int(len(selected)),
        "aocc_anytime_stop_budget": float(spent),
        "aocc_mean_selected_marginal_gain": float(np.mean(selected_order_gain)) if selected_order_gain else 0.0,
    }
    return selected, float(objective), float(spent), diag


def _anytime_one_sided_adverse_certificate_from_pair_delta(
    atom_delta: np.ndarray,
    base_margin: np.ndarray,
    pair_weights: np.ndarray,
    pair_indices: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    atom_active_mask: np.ndarray | None = None,
    *,
    atom_pair_variance: np.ndarray | None = None,
    adverse_beta: float = 1.0,
    adverse_epsilon: float = 0.05,
    prior_radius: float = 0.10,
    certificate_margin: float = 0.0,
    boundary_tau: float = 0.25,
    stop_when_certified: bool = True,
    max_target_rivals: int = 0,
    target_action_hint: int | None = None,
    explicitly_calibrated: bool = False,
    family_ids: np.ndarray | None = None,
    interaction_family_ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    max_interaction_prefix_fraction: float = 1.0,
    fill_to_budget_after_certified: bool = False,
    state_cache: dict[str, Any] | None = None,
) -> tuple[list[int], float, float, dict[str, Any]]:
    """Nested one-sided adverse-bound certificate coreset.

    The expensive greedy order is budget-independent.  ``state_cache`` is used
    only when a caller evaluates the exact same scene under several budgets;
    every input is compared exactly before reuse, so single-budget deployment
    behavior and numerical results remain unchanged.
    """
    state = state_cache.get("state") if isinstance(state_cache, dict) else None
    if not isinstance(state, dict) or not _anytime_aocc_state_matches(
        state,
        atom_delta,
        base_margin,
        pair_weights,
        pair_indices,
        atom_budget_costs,
        valid_mask,
        runtime_safety_flags,
        atom_active_mask,
        atom_pair_variance,
        adverse_beta=adverse_beta,
        adverse_epsilon=adverse_epsilon,
        prior_radius=prior_radius,
        certificate_margin=certificate_margin,
        boundary_tau=boundary_tau,
        stop_when_certified=stop_when_certified,
        max_target_rivals=max_target_rivals,
        target_action_hint=target_action_hint,
        explicitly_calibrated=explicitly_calibrated,
        family_ids=family_ids,
        interaction_family_ids=interaction_family_ids,
        max_interaction_prefix_fraction=max_interaction_prefix_fraction,
        fill_to_budget_after_certified=fill_to_budget_after_certified,
    ):
        state = _build_anytime_one_sided_adverse_certificate_state(
            atom_delta,
            base_margin,
            pair_weights,
            pair_indices,
            atom_budget_costs,
            valid_mask,
            runtime_safety_flags,
            atom_active_mask,
            atom_pair_variance=atom_pair_variance,
            adverse_beta=adverse_beta,
            adverse_epsilon=adverse_epsilon,
            prior_radius=prior_radius,
            certificate_margin=certificate_margin,
            boundary_tau=boundary_tau,
            stop_when_certified=stop_when_certified,
            max_target_rivals=max_target_rivals,
            target_action_hint=target_action_hint,
            explicitly_calibrated=explicitly_calibrated,
            family_ids=family_ids,
            interaction_family_ids=interaction_family_ids,
            max_interaction_prefix_fraction=max_interaction_prefix_fraction,
            fill_to_budget_after_certified=fill_to_budget_after_certified,
        )
        if isinstance(state_cache, dict):
            state_cache["state"] = state
    return _materialize_anytime_one_sided_adverse_certificate(state, budget)


def runtime_greedy_selector_pair_conditioned(
    predicted_base_cost: np.ndarray,
    pair_atom_delta: np.ndarray,
    pair_indices: np.ndarray,
    pair_weights: np.ndarray,
    atom_budget_costs: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    budget: float,
    gamma_max: float = 100.0,
    eta_pred: float = 1.0,
    atom_active_mask: np.ndarray | None = None,
    pair_atom_variance: np.ndarray | None = None,
    beta_uncertainty: float = 0.0,
    epsilon_cal: float = 0.0,
    lambda_info: float = 0.0,
    prior_atom_variance: float | np.ndarray | None = None,
    family_ids: np.ndarray | None = None,
    family_budget_caps: np.ndarray | None = None,
    mandatory_atom_mask: np.ndarray | None = None,
    mandatory_quota: int = 0,
    min_selected_atoms: int = 0,
    force_fill_budget: bool = False,
    normalize_margins: bool = False,
    margin_scale: float | None = None,
    proposal_scores: np.ndarray | None = None,
    proposal_fill_weight: float = 0.0,
    prioritize_mandatory_fill: bool = True,
    selector_cap_mode: str = "legacy_abs",
    boundary_certificate_cap: float | None = None,
    base_margin_cap_multiplier: float = 1.0,
    flip_bonus: float = 0.0,
    flip_window: float = 0.5,
    certify_margin: float = 0.0,
    flip_mode: str = "hard",
    flip_temperature: float = 0.08,
    action_rank_certificate_weight: float = 1.0,
    action_rank_score_weight: float = 0.0,
    action_rank_gap_weight: float = 0.0,
    action_rank_flip_weight: float = 0.0,
    action_rank_softmin_tau: float = 0.2,
    action_utility_cost: np.ndarray | None = None,
    action_utility_weight: float = 0.0,
    action_pair_utility_weight: float = 0.0,
    action_rank_fast_greedy: bool = False,
    hybrid_lcb_budget_frac: float = 0.55,
    hybrid_lcb_cap_mode: str = "legacy_abs",
    hybrid_protect_lcb_seed: bool = True,
    hybrid_min_action_budget_frac: float = 0.0,
    hybrid_max_lcb_seed_atoms: int = 0,
    adaptive_hybrid_lcb_budget: bool = False,
    adaptive_lcb_min_frac: float = 0.45,
    adaptive_lcb_max_frac: float = 0.80,
    adaptive_lcb_safety_weight: float = 0.25,
    adaptive_lcb_fallback_weight: float = 0.20,
    adaptive_lcb_uncertainty_weight: float = 0.10,
    adaptive_lcb_boundary_action_weight: float = 0.25,
    adaptive_lcb_boundary_tau: float = 0.35,
    decision_family_boost: float = 0.0,
    decision_family_ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    decision_family_quota: int = 0,
    interaction_family_ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    interaction_family_quota: int = 0,
    soft_interaction_mask: np.ndarray | None = None,
    soft_interaction_quota: int = 0,
    interaction_group_ids: np.ndarray | None = None,
    direction_invariant_interaction_weight: float = 0.0,
    direction_invariant_boundary_tau: float = 0.35,
    direction_invariant_flip_bonus: float = 0.5,
    collapse_reciprocal_pairs: bool = False,
    force_uncertainty_objective: bool = False,
    margin_coreset_residual_weight: float = 1.0,
    margin_coreset_sign_weight: float = 0.8,
    margin_coreset_winner_weight: float = 1.5,
    margin_coreset_action_weight: float = 0.5,
    margin_coreset_boundary_tau: float = 0.35,
    margin_coreset_huber_delta: float = 0.25,
    margin_coreset_target_clip: float = 3.0,
    margin_coreset_swap_passes: int = 2,
    deployment_evaluator: Callable[[list[int]], tuple[int, np.ndarray, np.ndarray]] | None = None,
    deployment_coreset_exact_candidates: int = 8,
    deployment_coreset_swap_passes: int = 1,
    deployment_coreset_score_weight: float = 1.0,
    deployment_coreset_action_weight: float = 4.0,
    deployment_coreset_gap_weight: float = 2.0,
    deployment_coreset_margin_weight: float = 1.0,
    deployment_coreset_lexicographic_action_preservation: bool = False,
    deployment_coreset_preservation_scan_candidates: int = 0,
    deployment_coreset_repair_one_swap: bool = True,
    deployment_coreset_repair_two_swap_candidates: int = 0,
    deployment_coreset_beam_width: int = 0,
    deployment_coreset_beam_branch: int = 0,
    deployment_coreset_beam_max_evaluations: int = 0,
    deployment_coreset_beam_mismatch_fraction: float = 0.35,
    deployment_coreset_budget_layer_width: int = 0,
    deployment_coreset_budget_layer_branch: int = 0,
    deployment_coreset_budget_layer_iterations: int = 0,
    deployment_coreset_budget_layer_max_evaluations: int = 0,
    deployment_coreset_budget_layer_exhaustive_first: bool = True,
    deployment_coreset_budget_layer_seed_count: int = 0,
    deployment_coreset_budget_layer_diversity_distance: int = 2,
    adverse_certificate_beta: float = 1.0,
    adverse_certificate_epsilon: float = 0.05,
    adverse_certificate_prior_radius: float = 0.10,
    adverse_certificate_margin: float = 0.0,
    adverse_certificate_stop_when_certified: bool = True,
    adverse_certificate_max_target_rivals: int = 0,
    adverse_certificate_target_action: int | None = None,
    adverse_certificate_calibrated: bool = False,
    adverse_certificate_fill_to_budget_after_certified: bool = False,
    adverse_certificate_max_interaction_prefix_fraction: float = 1.0,
    aocc_state_cache: dict[str, Any] | None = None,
) -> SelectionResult:
    """Runtime greedy selector over pair-conditioned atom deltas.

    ``pair_atom_delta[i,p]`` is the signed evidence margin contribution
    d_i(a,b) for ``pair_indices[p] = (a,b)``.  This implements the paper's
    pair-conditioned scorer while keeping the same greedy coverage objective.
    """
    pair_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
    if weights.shape[0] < pair_arr.shape[0]:
        weights = np.pad(weights, (0, pair_arr.shape[0] - weights.shape[0]), constant_values=1.0)
    weights = weights[: pair_arr.shape[0]]
    E = int(np.asarray(pair_atom_delta).shape[0]) if np.asarray(pair_atom_delta).ndim == 2 else int(np.asarray(atom_budget_costs).shape[0])
    if pair_arr.size == 0:
        selected, current, spent = _greedy_cover_from_pair_support(
            np.zeros((E, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            atom_budget_costs,
            budget,
            atom_active_mask,
        )
        selected, spent_post, post_diag = _complete_safety_aware_selection(
            selected, atom_budget_costs, budget, atom_active_mask=atom_active_mask,
            mandatory_atom_mask=mandatory_atom_mask, mandatory_quota=mandatory_quota,
            min_selected_atoms=min_selected_atoms, force_fill_budget=force_fill_budget,
            prioritize_mandatory_fill=prioritize_mandatory_fill,
            family_ids=family_ids,
            decision_family_ids=decision_family_ids,
            decision_family_quota=decision_family_quota,
            interaction_family_ids=interaction_family_ids,
            interaction_family_quota=interaction_family_quota,
            soft_interaction_mask=soft_interaction_mask,
            soft_interaction_quota=soft_interaction_quota,
            interaction_group_ids=interaction_group_ids,
        )
        return SelectionResult(selected, current, pair_arr, weights, {"spent_budget": float(spent_post), "pre_postfill_spent_budget": float(spent), "budget": float(budget), "mode": "runtime_pair_conditioned_empty", "pair_count": 0, **post_diag})
    a = pair_arr[:, 0]
    b = pair_arr[:, 1]
    J0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    base_delta_raw = J0[b] - J0[a]
    scale = 1.0
    if bool(normalize_margins):
        scale = float(margin_scale) if margin_scale is not None and np.isfinite(float(margin_scale)) and float(margin_scale) > 0 else margin_normalization_scale(base_delta_raw, min_scale=100.0)
    base_delta = (base_delta_raw / max(scale, 1e-6)).astype(np.float32)
    flags = np.asarray(runtime_safety_flags, dtype=bool).reshape(-1)
    safety_b = flags[b] if flags.shape[0] > int(np.max(b, initial=0)) else np.zeros_like(base_delta, dtype=bool)
    caps = _selector_pair_caps(
        base_delta,
        safety_b,
        gamma_max=float(gamma_max),
        eta_pred=float(eta_pred),
        cap_mode=selector_cap_mode,
        boundary_cap=boundary_certificate_cap,
        base_margin_cap_multiplier=base_margin_cap_multiplier,
    )
    delta = np.asarray(pair_atom_delta, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[1] != pair_arr.shape[0]:
        delta = np.zeros((E, pair_arr.shape[0]), dtype=np.float32)
    # v34 ABIQ: the pair-conditioned path previously skipped the reciprocal
    # collapse used by the legacy action-sparse selector.  That let d(a,b) and
    # d(b,a) cancel in the fixed-budget objective.  Collapse them here using the
    # already-scored evidence-sensitive orientation; no extra query is required.
    if bool(collapse_reciprocal_pairs) and pair_arr.shape[0] > 1:
        pair_var_c = np.asarray(pair_atom_variance, dtype=np.float32) if pair_atom_variance is not None else np.zeros_like(delta, dtype=np.float32)
        if pair_var_c.shape != delta.shape:
            pair_var_c = np.zeros_like(delta, dtype=np.float32)
        info_caps_c = np.where(safety_b, float(gamma_max), np.maximum(float(eta_pred), np.abs(base_delta))).astype(np.float32)
        (
            pair_arr,
            weights,
            base_delta,
            caps,
            delta,
            pair_var_c,
            _base_var_c,
            _info_caps_c,
        ) = _collapse_reciprocal_runtime_pairs(
            pair_arr,
            weights,
            base_delta,
            caps,
            delta,
            pair_var_c,
            np.zeros((pair_arr.shape[0],), dtype=np.float32),
            info_caps_c,
            atom_active_mask=atom_active_mask,
        )
        pair_atom_variance = pair_var_c
        a = pair_arr[:, 0]
        b = pair_arr[:, 1]
        safety_b = flags[b] if flags.shape[0] > int(np.max(b, initial=0)) else np.zeros_like(base_delta, dtype=bool)
    base_support = np.maximum(base_delta, 0.0).astype(np.float32)
    atom_support = np.maximum(delta, 0.0).astype(np.float32)  # legacy/debug only

    cap_mode_l = str(selector_cap_mode or "legacy_abs").lower()
    action_rank_modes = {"action_rank", "action_flip_rank", "tournament_rank"}
    hybrid_action_lcb_modes = {"safety_gated_action_rank", "lcb_action_rank_hybrid", "hybrid_lcb_action_rank", "safe_action_rank", "adaptive_safety_gated_action_rank", "adaptive_hybrid_lcb_action_rank"}
    flip_rank_modes = {"flip_rank", "fliprank", "flip_boundary_rank"}
    margin_coreset_modes = {"margin_coreset", "signed_margin_coreset", "mars", "margin_preserving"}
    adverse_certificate_modes = {"anytime_adverse_certificate", "one_sided_adverse_certificate", "aocc", "aobcc", "nested_certificate"}
    deployment_coreset_modes = {
        "deployment_coreset", "deployment_aligned_coreset", "dacc", "exact_tournament_coreset",
        "lexicographic_deployment_coreset", "lex_dacc", "lexdacc",
        "path_relaxed_deployment_coreset", "pr_dacc", "prdacc", "beam_dacc",
        "counterfactual_budget_layer_coreset", "cbl_dacc", "cbldacc", "budget_layer_dacc",
        "stage_aware_budget_layer_coreset", "sab_dacc", "sabdacc", "stage_aware_dacc",
    }
    # v18: selector_cap_mode must own the dispatch.  In v15-v17, merely passing
    # pair_atom_variance or family_budget_caps forced the LCB/uncertainty path,
    # silently bypassing flip_rank/action_rank objectives even when configs asked
    # for them.  Use the uncertainty objective only for legacy modes or when it
    # is explicitly forced.
    use_uncertainty_objective = bool(force_uncertainty_objective) or (
        cap_mode_l not in action_rank_modes
        and cap_mode_l not in hybrid_action_lcb_modes
        and cap_mode_l not in flip_rank_modes
        and cap_mode_l not in margin_coreset_modes
        and cap_mode_l not in adverse_certificate_modes
        and cap_mode_l not in deployment_coreset_modes
        and (
            pair_atom_variance is not None
            or abs(float(beta_uncertainty)) > 0.0
            or abs(float(epsilon_cal)) > 0.0
            or abs(float(lambda_info)) > 0.0
            or family_budget_caps is not None
        )
    )
    hybrid_diag: dict[str, Any] = {}
    coreset_diag: dict[str, Any] = {}
    if use_uncertainty_objective:
        pair_var = np.asarray(pair_atom_variance, dtype=np.float32) if pair_atom_variance is not None else np.zeros_like(delta, dtype=np.float32)
        if pair_var.shape != delta.shape:
            pair_var = np.zeros_like(delta, dtype=np.float32)
        base_var = np.zeros((pair_arr.shape[0],), dtype=np.float32)
        if isinstance(prior_atom_variance, np.ndarray):
            prior = np.asarray(prior_atom_variance, dtype=np.float32)
            prior_pair = prior if prior.shape == delta.shape else None
        elif prior_atom_variance is not None:
            prior_pair = np.full_like(delta, float(prior_atom_variance), dtype=np.float32)
        else:
            prior_pair = None
        selected, current, spent, extra_diag = _uncertainty_aware_greedy_from_pair_delta(
            delta,
            base_delta,
            pair_var,
            base_var,
            caps,
            weights,
            atom_budget_costs,
            budget,
            atom_active_mask,
            beta_uncertainty=beta_uncertainty,
            epsilon_cal=epsilon_cal,
            lambda_info=lambda_info,
            info_caps=np.where(safety_b, float(gamma_max), np.maximum(float(eta_pred), np.abs(base_delta))).astype(np.float32),
            prior_atom_pair_var=prior_pair,
            family_ids=family_ids,
            family_budget_caps=family_budget_caps,
        )
        mode = "runtime_pair_conditioned_lcb_uncertainty"
    else:
        if cap_mode_l in adverse_certificate_modes:
            selected, current, spent, coreset_diag = _anytime_one_sided_adverse_certificate_from_pair_delta(
                delta, base_delta, weights, pair_arr, atom_budget_costs, budget,
                valid_mask, runtime_safety_flags, atom_active_mask,
                atom_pair_variance=pair_atom_variance,
                adverse_beta=adverse_certificate_beta,
                adverse_epsilon=adverse_certificate_epsilon,
                prior_radius=adverse_certificate_prior_radius,
                certificate_margin=adverse_certificate_margin,
                boundary_tau=margin_coreset_boundary_tau,
                stop_when_certified=adverse_certificate_stop_when_certified,
                max_target_rivals=adverse_certificate_max_target_rivals,
                target_action_hint=adverse_certificate_target_action,
                explicitly_calibrated=adverse_certificate_calibrated,
                family_ids=family_ids,
                interaction_family_ids=interaction_family_ids,
                max_interaction_prefix_fraction=adverse_certificate_max_interaction_prefix_fraction,
                fill_to_budget_after_certified=adverse_certificate_fill_to_budget_after_certified,
                state_cache=aocc_state_cache,
            )
            mode = "runtime_pair_conditioned_anytime_adverse_certificate"
        elif cap_mode_l in deployment_coreset_modes and deployment_evaluator is not None:
            selected, current, spent, coreset_diag = _deployment_aligned_coreset_from_pair_delta(
                delta, base_delta, weights, pair_arr, atom_budget_costs, budget,
                valid_mask, runtime_safety_flags, atom_active_mask, deployment_evaluator,
                soft_interaction_mask=soft_interaction_mask,
                soft_interaction_quota=soft_interaction_quota,
                proposal_scores=proposal_scores,
                exact_candidates=deployment_coreset_exact_candidates,
                swap_passes=deployment_coreset_swap_passes,
                score_weight=deployment_coreset_score_weight,
                action_weight=deployment_coreset_action_weight,
                gap_weight=deployment_coreset_gap_weight,
                margin_weight=deployment_coreset_margin_weight,
                huber_delta=margin_coreset_huber_delta,
                lexicographic_action_preservation=bool(deployment_coreset_lexicographic_action_preservation),
                preservation_scan_candidates=int(deployment_coreset_preservation_scan_candidates),
                repair_one_swap=bool(deployment_coreset_repair_one_swap),
                repair_two_swap_candidates=int(deployment_coreset_repair_two_swap_candidates),
                beam_width=int(deployment_coreset_beam_width),
                beam_branch=int(deployment_coreset_beam_branch),
                beam_max_evaluations=int(deployment_coreset_beam_max_evaluations),
                beam_mismatch_fraction=float(deployment_coreset_beam_mismatch_fraction),
                budget_layer_width=int(deployment_coreset_budget_layer_width),
                budget_layer_branch=int(deployment_coreset_budget_layer_branch),
                budget_layer_iterations=int(deployment_coreset_budget_layer_iterations),
                budget_layer_max_evaluations=int(deployment_coreset_budget_layer_max_evaluations),
                budget_layer_exhaustive_first=bool(deployment_coreset_budget_layer_exhaustive_first),
                budget_layer_seed_count=int(deployment_coreset_budget_layer_seed_count),
                budget_layer_diversity_distance=int(deployment_coreset_budget_layer_diversity_distance),
            )
            mode = "runtime_pair_conditioned_deployment_coreset"
        elif cap_mode_l in deployment_coreset_modes:
            # Training currently has no full deployment callback; preserve a
            # well-defined stop-gradient fallback instead of silently crashing.
            selected, current, spent, coreset_diag = _signed_margin_coreset_from_pair_delta(
                delta, base_delta, weights, pair_arr, atom_budget_costs, budget,
                valid_mask, runtime_safety_flags, atom_active_mask,
                soft_interaction_mask=soft_interaction_mask,
                soft_interaction_quota=soft_interaction_quota,
                proposal_scores=proposal_scores,
                residual_weight=margin_coreset_residual_weight,
                sign_weight=margin_coreset_sign_weight,
                winner_weight=margin_coreset_winner_weight,
                action_weight=margin_coreset_action_weight,
                boundary_tau=margin_coreset_boundary_tau,
                huber_delta=margin_coreset_huber_delta,
                target_clip=margin_coreset_target_clip,
                swap_passes=margin_coreset_swap_passes,
            )
            coreset_diag["deployment_coreset_evaluator_missing"] = True
            mode = "runtime_pair_conditioned_deployment_coreset_fallback"
        elif cap_mode_l in margin_coreset_modes:
            selected, current, spent, coreset_diag = _signed_margin_coreset_from_pair_delta(
                delta, base_delta, weights, pair_arr, atom_budget_costs, budget,
                valid_mask, runtime_safety_flags, atom_active_mask,
                soft_interaction_mask=soft_interaction_mask,
                soft_interaction_quota=soft_interaction_quota,
                proposal_scores=proposal_scores,
                residual_weight=margin_coreset_residual_weight,
                sign_weight=margin_coreset_sign_weight,
                winner_weight=margin_coreset_winner_weight,
                action_weight=margin_coreset_action_weight,
                boundary_tau=margin_coreset_boundary_tau,
                huber_delta=margin_coreset_huber_delta,
                target_clip=margin_coreset_target_clip,
                swap_passes=margin_coreset_swap_passes,
            )
            mode = "runtime_pair_conditioned_margin_coreset"
        elif cap_mode_l in hybrid_action_lcb_modes:
            lcb_caps = _selector_pair_caps(
                base_delta,
                safety_b,
                gamma_max=float(gamma_max),
                eta_pred=float(eta_pred),
                cap_mode=str(hybrid_lcb_cap_mode or "legacy_abs"),
                boundary_cap=boundary_certificate_cap,
                base_margin_cap_multiplier=base_margin_cap_multiplier,
            )
            info_caps = np.where(safety_b, float(gamma_max), np.maximum(float(eta_pred), np.abs(base_delta))).astype(np.float32)
            selected, current, spent, hybrid_diag = _hybrid_lcb_action_rank_from_pair_delta(
                delta,
                base_delta,
                caps,
                lcb_caps,
                weights,
                pair_arr,
                atom_budget_costs,
                budget,
                atom_active_mask,
                atom_pair_variance=pair_atom_variance,
                beta_uncertainty=beta_uncertainty,
                epsilon_cal=epsilon_cal,
                lambda_info=lambda_info,
                info_caps=info_caps,
                prior_atom_variance=prior_atom_variance,
                family_ids=family_ids,
                family_budget_caps=family_budget_caps,
                hybrid_lcb_budget_frac=float(hybrid_lcb_budget_frac),
                hybrid_protect_lcb_seed=bool(hybrid_protect_lcb_seed),
                hybrid_min_action_budget_frac=float(hybrid_min_action_budget_frac),
                hybrid_max_lcb_seed_atoms=int(hybrid_max_lcb_seed_atoms),
                adaptive_hybrid_lcb_budget=bool(adaptive_hybrid_lcb_budget or cap_mode_l.startswith("adaptive_")),
                adaptive_lcb_min_frac=float(adaptive_lcb_min_frac),
                adaptive_lcb_max_frac=float(adaptive_lcb_max_frac),
                adaptive_lcb_safety_weight=float(adaptive_lcb_safety_weight),
                adaptive_lcb_fallback_weight=float(adaptive_lcb_fallback_weight),
                adaptive_lcb_uncertainty_weight=float(adaptive_lcb_uncertainty_weight),
                adaptive_lcb_boundary_action_weight=float(adaptive_lcb_boundary_action_weight),
                adaptive_lcb_boundary_tau=float(adaptive_lcb_boundary_tau),
                safety_pair_mask=safety_b,
                decision_family_ids=decision_family_ids,
                decision_family_boost=float(decision_family_boost),
                certificate_weight=float(action_rank_certificate_weight),
                action_score_weight=float(action_rank_score_weight),
                action_gap_weight=float(action_rank_gap_weight),
                action_flip_weight=float(action_rank_flip_weight),
                action_softmin_tau=float(action_rank_softmin_tau),
                certify_margin=float(certify_margin),
                action_utility_cost=action_utility_cost,
                action_utility_weight=float(action_utility_weight),
                action_pair_utility_weight=float(action_pair_utility_weight),
                action_rank_fast_greedy=bool(action_rank_fast_greedy),
            )
            mode = "runtime_pair_conditioned_hybrid_lcb_action_rank"
        elif cap_mode_l in action_rank_modes:
            selected, current, spent = _greedy_action_rank_from_pair_delta(
                delta,
                base_delta,
                caps,
                weights,
                pair_arr,
                atom_budget_costs,
                budget,
                atom_active_mask,
                certificate_weight=float(action_rank_certificate_weight),
                action_score_weight=float(action_rank_score_weight),
                action_gap_weight=float(action_rank_gap_weight),
                action_flip_weight=float(action_rank_flip_weight),
                action_softmin_tau=float(action_rank_softmin_tau),
                certify_margin=float(certify_margin),
                action_utility_cost=action_utility_cost,
                action_utility_weight=float(action_utility_weight),
                action_pair_utility_weight=float(action_pair_utility_weight),
                action_rank_fast_greedy=bool(action_rank_fast_greedy),
            )
            mode = "runtime_pair_conditioned_action_rank"
        elif cap_mode_l in flip_rank_modes:
            selected, current, spent = _greedy_flip_rank_from_pair_delta(
                delta,
                base_delta,
                caps,
                weights,
                atom_budget_costs,
                budget,
                atom_active_mask,
                flip_bonus=float(flip_bonus),
                flip_window=float(flip_window),
                certify_margin=float(certify_margin),
                flip_mode=str(flip_mode),
                flip_temperature=float(flip_temperature),
            )
            mode = "runtime_pair_conditioned_flip_rank"
        else:
            selected, current, spent = _greedy_cover_from_pair_delta(delta, base_delta, caps, weights, atom_budget_costs, budget, atom_active_mask)
            mode = "runtime_pair_conditioned_signed"
        extra_diag = {"flip_bonus": float(flip_bonus), "flip_window": float(flip_window), "certify_margin": float(certify_margin), "flip_mode": str(flip_mode), "flip_temperature": float(flip_temperature), "action_rank_certificate_weight": float(action_rank_certificate_weight), "action_rank_score_weight": float(action_rank_score_weight), "action_rank_gap_weight": float(action_rank_gap_weight), "action_rank_flip_weight": float(action_rank_flip_weight), "action_rank_softmin_tau": float(action_rank_softmin_tau), "action_utility_weight": float(action_utility_weight), "action_pair_utility_weight": float(action_pair_utility_weight), "action_rank_fast_greedy": bool(action_rank_fast_greedy), "hybrid_lcb_budget_frac": float(hybrid_lcb_budget_frac), "hybrid_lcb_cap_mode": str(hybrid_lcb_cap_mode), "hybrid_protect_lcb_seed": bool(hybrid_protect_lcb_seed), "hybrid_min_action_budget_frac": float(hybrid_min_action_budget_frac), "hybrid_max_lcb_seed_atoms": int(hybrid_max_lcb_seed_atoms), "adaptive_hybrid_lcb_budget": bool(adaptive_hybrid_lcb_budget or cap_mode_l.startswith("adaptive_")), "adaptive_lcb_min_frac": float(adaptive_lcb_min_frac), "adaptive_lcb_max_frac": float(adaptive_lcb_max_frac), "decision_family_boost": float(decision_family_boost), "decision_family_quota": int(decision_family_quota), "interaction_family_quota": int(interaction_family_quota), "soft_interaction_quota": int(soft_interaction_quota), "direction_invariant_interaction_weight": float(direction_invariant_interaction_weight), "direction_invariant_boundary_tau": float(direction_invariant_boundary_tau), "direction_invariant_flip_bonus": float(direction_invariant_flip_bonus), "collapse_reciprocal_pairs": bool(collapse_reciprocal_pairs), "force_uncertainty_objective": bool(force_uncertainty_objective), "margin_coreset_residual_weight": float(margin_coreset_residual_weight), "margin_coreset_sign_weight": float(margin_coreset_sign_weight), "margin_coreset_winner_weight": float(margin_coreset_winner_weight), "margin_coreset_action_weight": float(margin_coreset_action_weight), "margin_coreset_boundary_tau": float(margin_coreset_boundary_tau), "margin_coreset_huber_delta": float(margin_coreset_huber_delta), "margin_coreset_target_clip": float(margin_coreset_target_clip), "margin_coreset_swap_passes": int(margin_coreset_swap_passes), "deployment_coreset_exact_candidates": int(deployment_coreset_exact_candidates), "deployment_coreset_swap_passes": int(deployment_coreset_swap_passes), "deployment_coreset_score_weight": float(deployment_coreset_score_weight), "deployment_coreset_action_weight": float(deployment_coreset_action_weight), "deployment_coreset_gap_weight": float(deployment_coreset_gap_weight), "deployment_coreset_margin_weight": float(deployment_coreset_margin_weight), "deployment_coreset_lexicographic_action_preservation": bool(deployment_coreset_lexicographic_action_preservation), "deployment_coreset_preservation_scan_candidates": int(deployment_coreset_preservation_scan_candidates), "deployment_coreset_repair_one_swap": bool(deployment_coreset_repair_one_swap), "deployment_coreset_repair_two_swap_candidates": int(deployment_coreset_repair_two_swap_candidates), "deployment_coreset_beam_width": int(deployment_coreset_beam_width), "deployment_coreset_beam_branch": int(deployment_coreset_beam_branch), "deployment_coreset_beam_max_evaluations": int(deployment_coreset_beam_max_evaluations), "deployment_coreset_beam_mismatch_fraction": float(deployment_coreset_beam_mismatch_fraction), "deployment_coreset_budget_layer_width": int(deployment_coreset_budget_layer_width), "deployment_coreset_budget_layer_branch": int(deployment_coreset_budget_layer_branch), "deployment_coreset_budget_layer_iterations": int(deployment_coreset_budget_layer_iterations), "deployment_coreset_budget_layer_max_evaluations": int(deployment_coreset_budget_layer_max_evaluations), "deployment_coreset_budget_layer_exhaustive_first": bool(deployment_coreset_budget_layer_exhaustive_first), "deployment_coreset_budget_layer_seed_count": int(deployment_coreset_budget_layer_seed_count), "deployment_coreset_budget_layer_diversity_distance": int(deployment_coreset_budget_layer_diversity_distance), "adverse_certificate_beta": float(adverse_certificate_beta), "adverse_certificate_epsilon": float(adverse_certificate_epsilon), "adverse_certificate_prior_radius": float(adverse_certificate_prior_radius), "adverse_certificate_margin": float(adverse_certificate_margin), "adverse_certificate_stop_when_certified": bool(adverse_certificate_stop_when_certified), "adverse_certificate_max_target_rivals": int(adverse_certificate_max_target_rivals), "adverse_certificate_target_action": -1 if adverse_certificate_target_action is None else int(adverse_certificate_target_action), "adverse_certificate_calibrated": bool(adverse_certificate_calibrated), "adverse_certificate_fill_to_budget_after_certified": bool(adverse_certificate_fill_to_budget_after_certified), "adverse_certificate_max_interaction_prefix_fraction": float(adverse_certificate_max_interaction_prefix_fraction), **hybrid_diag, **coreset_diag}
    # The ranking utility below is only consumed by the generic post-fill stage.
    # AOCC v46 disables every post-fill mechanism (no mandatory mask/quotas,
    # no minimum support, and no force-fill), so evaluating action-rank utility
    # here was a pure duplicate O(E*P + E*A) CPU computation for every budget.
    # Keep the generic behavior bit-for-bit whenever post-fill can modify the set.
    _costs_for_postfill = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    _E_for_postfill = int(_costs_for_postfill.shape[0])
    _active_for_postfill = (
        np.ones((_E_for_postfill,), dtype=bool)
        if atom_active_mask is None
        else _as_bool_mask(atom_active_mask, _E_for_postfill)
    )
    _active_for_postfill &= np.isfinite(_costs_for_postfill) & (_costs_for_postfill > 0)
    _mandatory_for_postfill = _as_bool_mask(mandatory_atom_mask, _E_for_postfill) & _active_for_postfill
    _selected_valid_count = len({
        int(i) for i in np.asarray(selected, dtype=np.int64).reshape(-1).tolist()
        if 0 <= int(i) < _E_for_postfill and bool(_active_for_postfill[int(i)])
    })
    _postfill_utility_required = bool(
        _mandatory_for_postfill.any()
        or int(soft_interaction_quota) > 0
        or int(interaction_family_quota) > 0
        or int(decision_family_quota) > 0
        or bool(force_fill_budget)
        or int(min_selected_atoms) > int(_selected_valid_count)
    )

    utility = np.zeros((_E_for_postfill,), dtype=np.float32)
    if _postfill_utility_required and np.asarray(delta).ndim == 2 and np.asarray(delta).size:
        if str(selector_cap_mode or "legacy_abs").lower() in {"action_rank", "action_flip_rank", "tournament_rank", "safety_gated_action_rank", "lcb_action_rank_hybrid", "hybrid_lcb_action_rank", "safe_action_rank", "adaptive_safety_gated_action_rank", "adaptive_hybrid_lcb_action_rank", "margin_coreset", "signed_margin_coreset", "mars", "margin_preserving", "deployment_coreset", "deployment_aligned_coreset", "dacc", "exact_tournament_coreset", "lexicographic_deployment_coreset", "lex_dacc", "lexdacc", "path_relaxed_deployment_coreset", "pr_dacc", "prdacc", "beam_dacc", "counterfactual_budget_layer_coreset", "cbl_dacc", "cbldacc", "budget_layer_dacc", "stage_aware_budget_layer_coreset", "sab_dacc", "sabdacc", "stage_aware_dacc", "anytime_adverse_certificate", "one_sided_adverse_certificate", "aocc", "aobcc", "nested_certificate"}:
            utility = _action_rank_atom_utility(
                delta, base_delta, caps, weights, pair_arr,
                certificate_weight=float(action_rank_certificate_weight),
                action_score_weight=float(action_rank_score_weight),
                action_gap_weight=float(action_rank_gap_weight),
                action_flip_weight=float(action_rank_flip_weight),
                action_softmin_tau=float(action_rank_softmin_tau),
                certify_margin=float(certify_margin),
                action_utility_cost=action_utility_cost,
                action_utility_weight=float(action_utility_weight),
                action_pair_utility_weight=float(action_pair_utility_weight),
            )
        else:
            utility = _flip_gain_atom_utility(
                delta,
                base_delta,
                caps,
                weights,
                flip_bonus=float(flip_bonus),
                flip_window=float(flip_window),
                certify_margin=float(certify_margin),
                flip_mode=str(flip_mode),
                flip_temperature=float(flip_temperature),
            )
    if _postfill_utility_required and proposal_scores is not None and float(proposal_fill_weight) > 0.0:
        prop = np.asarray(proposal_scores, dtype=np.float32).reshape(-1)
        if prop.shape[0] < utility.shape[0]:
            prop = np.pad(prop, (0, utility.shape[0] - prop.shape[0]), constant_values=-60.0)
        prop = prop[: utility.shape[0]]
        # Convert logits/scores to a bounded acquisition prior. This is used only
        # for post-fill tie breaking and cannot override the hard budget/active masks.
        prop_prior = 1.0 / (1.0 + np.exp(-np.clip(prop, -20.0, 20.0)))
        utility = utility + float(proposal_fill_weight) * prop_prior.astype(np.float32)
    interaction_utility = utility.copy()
    if (
        float(direction_invariant_interaction_weight) > 0.0
        and np.asarray(delta).ndim == 2
        and np.asarray(delta).size
    ):
        two_sided = _direction_invariant_interaction_utility(
            delta,
            base_delta,
            caps,
            weights,
            boundary_tau=float(direction_invariant_boundary_tau),
            flip_bonus=float(direction_invariant_flip_bonus),
            active_mask=atom_active_mask,
        )
        interaction_utility = interaction_utility + float(direction_invariant_interaction_weight) * two_sided
        extra_diag["direction_invariant_interaction_mean"] = float(np.mean(two_sided)) if two_sided.size else 0.0
        extra_diag["direction_invariant_interaction_p90"] = float(np.quantile(two_sided, 0.90)) if two_sided.size else 0.0
    pre_postfill_selected = list(map(int, selected))
    selected, spent_post, post_diag = _complete_safety_aware_selection(
        selected,
        atom_budget_costs,
        budget,
        atom_active_mask=atom_active_mask,
        mandatory_atom_mask=mandatory_atom_mask,
        mandatory_quota=mandatory_quota,
        min_selected_atoms=min_selected_atoms,
        force_fill_budget=force_fill_budget,
        utility=utility,
        prioritize_mandatory_fill=prioritize_mandatory_fill,
        family_ids=family_ids,
        decision_family_ids=decision_family_ids,
        decision_family_quota=decision_family_quota,
        interaction_family_ids=interaction_family_ids,
        interaction_family_quota=interaction_family_quota,
        soft_interaction_mask=soft_interaction_mask,
        soft_interaction_quota=soft_interaction_quota,
        interaction_group_ids=interaction_group_ids,
        interaction_utility=interaction_utility,
    )
    # The generic safety-aware post-fill predates DACC and may replace atoms to
    # satisfy quotas.  Audit the *actual returned set* with the deployment
    # callback, otherwise selector diagnostics can claim target preservation for
    # a pre-postfill set while the planner executes a different action.  For the
    # lexicographic mode, revert a post-fill change that breaks an already
    # preserved deployment action; the DACC routine itself enforces the active
    # soft-interaction floor.
    if cap_mode_l in deployment_coreset_modes and deployment_evaluator is not None:
        target_action = int(extra_diag.get("deployment_coreset_target_action", -1))
        pre_action = int(deployment_evaluator(pre_postfill_selected)[0])
        post_action = int(deployment_evaluator(list(map(int, selected)))[0])
        post_changed = tuple(sorted(pre_postfill_selected)) != tuple(sorted(map(int, selected)))
        reverted = False
        if (
            bool(deployment_coreset_lexicographic_action_preservation)
            and target_action >= 0
            and pre_action == target_action
            and post_action != target_action
        ):
            selected = list(pre_postfill_selected)
            spent_post = _spent_for(selected, np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1))
            post_action = pre_action
            reverted = True
        extra_diag["deployment_coreset_postfill_changed"] = bool(post_changed)
        extra_diag["deployment_coreset_postfill_reverted"] = bool(reverted)
        extra_diag["deployment_coreset_selected_action"] = int(post_action)
        extra_diag["deployment_coreset_target_action_preserved"] = float(
            target_action >= 0 and post_action == target_action
        )
    return SelectionResult(
        selected=selected,
        objective_value=current,
        pair_indices=pair_arr,
        pair_weights=weights,
        diagnostics={"spent_budget": float(spent_post), "pre_postfill_spent_budget": float(spent), "budget": float(budget), "mode": mode, "pair_count": int(len(pair_arr)), "normalized_margins": bool(normalize_margins), "margin_scale": float(scale), **extra_diag, **post_diag},
    )


def select_by_mode(
    mode: str,
    predicted_base_cost: np.ndarray,
    predicted_atom_costs: np.ndarray,
    atom_budget_costs: np.ndarray,
    valid_mask: np.ndarray,
    runtime_safety_flags: np.ndarray,
    budget: float,
    atom_families: list[str] | None = None,
    seed: int = 17,
    atom_active_mask: np.ndarray | None = None,
    proposal_scores: np.ndarray | None = None,
    mandatory_atom_mask: np.ndarray | None = None,
    **kwargs,
) -> SelectionResult:
    E = int(predicted_atom_costs.shape[0])
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)
    costs = costs[:E]
    active = np.ones((E,), dtype=bool) if atom_active_mask is None else np.asarray(atom_active_mask, dtype=bool).reshape(-1)
    if active.shape[0] < E:
        active = np.pad(active, (0, E - active.shape[0]), constant_values=False)
    active = active[:E] & np.isfinite(costs) & (costs > 0)
    max_count = int(budget) if active.any() and np.allclose(costs[active], 1.0) else E
    if mode == "full_prescore_ablation":
        return full_prescore_greedy_selector(
            predicted_base_cost,
            predicted_atom_costs,
            atom_budget_costs,
            valid_mask,
            runtime_safety_flags,
            budget,
            **kwargs,
        )
    if mode == "runtime_predicted":
        return runtime_greedy_selector(
            predicted_base_cost,
            predicted_atom_costs,
            atom_budget_costs,
            valid_mask,
            runtime_safety_flags,
            budget,
            **kwargs,
        )
    if mode == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(np.flatnonzero(active)).astype(np.int64).tolist()
    elif mode == "top_magnitude":
        magnitude = np.abs(predicted_atom_costs[:, valid_mask]).mean(axis=1)
        order = sorted(np.flatnonzero(active).tolist(), key=lambda i: (-float(magnitude[i]), i))
    elif mode == "diversity":
        fams = (atom_families or ["all"] * E)[:E]
        order = []
        for fam in sorted(set(fams)):
            order.extend([i for i, f in enumerate(fams) if active[i] and f == fam])
    elif mode in {"proposal_top", "hard_safety_only", "interaction_only", "rule_map_only", "risk_only"}:
        fams = (atom_families or ["all"] * E)[:E]
        if mode == "proposal_top":
            if proposal_scores is not None:
                scores = np.asarray(proposal_scores, dtype=np.float32).reshape(-1)
                if scores.shape[0] < E:
                    scores = np.pad(scores, (0, E - scores.shape[0]), constant_values=-np.inf)
                scores = scores[:E]
                order = sorted(np.flatnonzero(active).tolist(), key=lambda i: (-float(scores[i]), i))
            else:
                magnitude = np.abs(predicted_atom_costs[:, valid_mask]).mean(axis=1)
                order = sorted(np.flatnonzero(active).tolist(), key=lambda i: (-float(magnitude[i]), i))
        elif mode == "hard_safety_only":
            fam_order = [i for i, f in enumerate(fams) if active[i] and f in {"hard", "safety", "feasibility", "rule_map"}]
            if mandatory_atom_mask is not None:
                mandatory = np.asarray(mandatory_atom_mask, dtype=bool).reshape(-1)
                if mandatory.shape[0] < E:
                    mandatory = np.pad(mandatory, (0, E - mandatory.shape[0]), constant_values=False)
                mandatory = mandatory[:E]
                mand_order = [i for i in np.flatnonzero(active & mandatory).tolist() if i not in set(fam_order)]
                order = np.flatnonzero(active & mandatory).tolist() + fam_order
            else:
                order = fam_order
        elif mode == "risk_only":
            magnitude = np.abs(predicted_atom_costs[:, valid_mask]).max(axis=1)
            order = sorted(np.flatnonzero(active).tolist(), key=lambda i: (-float(magnitude[i]), i))
        else:
            want_set = {"interaction", "reachability_interaction", "precedence"} if mode == "interaction_only" else {"rule_map", "feasibility"}
            order = [i for i, f in enumerate(fams) if active[i] and f in want_set]
    else:
        raise ValueError(f"Unknown selector mode: {mode}")
    selected: list[int] = []
    selected_set: set[int] = set()
    spent = 0.0
    for i in order:
        if i < 0 or i >= E or int(i) in selected_set or not bool(active[int(i)]):
            continue
        c = float(costs[int(i)])
        if np.isfinite(c) and spent + c <= budget + 1e-6:
            selected.append(int(i))
            selected_set.add(int(i))
            spent += c
        if len(selected) >= max_count and active.any() and np.allclose(costs[active], 1.0):
            break
    return SelectionResult(selected=selected, objective_value=0.0, pair_indices=np.zeros((0, 2), dtype=np.int64), pair_weights=np.zeros((0,), dtype=np.float32), diagnostics={"spent_budget": spent, "mode": mode})
