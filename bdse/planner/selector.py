from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    # v18: selector_cap_mode must own the dispatch.  In v15-v17, merely passing
    # pair_atom_variance or family_budget_caps forced the LCB/uncertainty path,
    # silently bypassing flip_rank/action_rank objectives even when configs asked
    # for them.  Use the uncertainty objective only for legacy modes or when it
    # is explicitly forced.
    use_uncertainty_objective = bool(force_uncertainty_objective) or (
        cap_mode_l not in action_rank_modes
        and cap_mode_l not in hybrid_action_lcb_modes
        and cap_mode_l not in flip_rank_modes
        and (
            pair_atom_variance is not None
            or abs(float(beta_uncertainty)) > 0.0
            or abs(float(epsilon_cal)) > 0.0
            or abs(float(lambda_info)) > 0.0
            or family_budget_caps is not None
        )
    )
    hybrid_diag: dict[str, Any] = {}
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
        if cap_mode_l in hybrid_action_lcb_modes:
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
        extra_diag = {"flip_bonus": float(flip_bonus), "flip_window": float(flip_window), "certify_margin": float(certify_margin), "flip_mode": str(flip_mode), "flip_temperature": float(flip_temperature), "action_rank_certificate_weight": float(action_rank_certificate_weight), "action_rank_score_weight": float(action_rank_score_weight), "action_rank_gap_weight": float(action_rank_gap_weight), "action_rank_flip_weight": float(action_rank_flip_weight), "action_rank_softmin_tau": float(action_rank_softmin_tau), "action_utility_weight": float(action_utility_weight), "action_pair_utility_weight": float(action_pair_utility_weight), "action_rank_fast_greedy": bool(action_rank_fast_greedy), "hybrid_lcb_budget_frac": float(hybrid_lcb_budget_frac), "hybrid_lcb_cap_mode": str(hybrid_lcb_cap_mode), "hybrid_protect_lcb_seed": bool(hybrid_protect_lcb_seed), "hybrid_min_action_budget_frac": float(hybrid_min_action_budget_frac), "hybrid_max_lcb_seed_atoms": int(hybrid_max_lcb_seed_atoms), "adaptive_hybrid_lcb_budget": bool(adaptive_hybrid_lcb_budget or cap_mode_l.startswith("adaptive_")), "adaptive_lcb_min_frac": float(adaptive_lcb_min_frac), "adaptive_lcb_max_frac": float(adaptive_lcb_max_frac), "decision_family_boost": float(decision_family_boost), "decision_family_quota": int(decision_family_quota), "interaction_family_quota": int(interaction_family_quota), "soft_interaction_quota": int(soft_interaction_quota), "direction_invariant_interaction_weight": float(direction_invariant_interaction_weight), "direction_invariant_boundary_tau": float(direction_invariant_boundary_tau), "direction_invariant_flip_bonus": float(direction_invariant_flip_bonus), "collapse_reciprocal_pairs": bool(collapse_reciprocal_pairs), "force_uncertainty_objective": bool(force_uncertainty_objective), **hybrid_diag}
    utility = np.zeros((int(np.asarray(atom_budget_costs).shape[0]),), dtype=np.float32)
    if np.asarray(delta).ndim == 2 and np.asarray(delta).size:
        if str(selector_cap_mode or "legacy_abs").lower() in {"action_rank", "action_flip_rank", "tournament_rank", "safety_gated_action_rank", "lcb_action_rank_hybrid", "hybrid_lcb_action_rank", "safe_action_rank", "adaptive_safety_gated_action_rank", "adaptive_hybrid_lcb_action_rank"}:
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
    if proposal_scores is not None and float(proposal_fill_weight) > 0.0:
        prop = np.asarray(proposal_scores, dtype=np.float32).reshape(-1)
        if prop.shape[0] < utility.shape[0]:
            prop = np.pad(prop, (0, utility.shape[0] - prop.shape[0]), constant_values=-60.0)
        prop = prop[: utility.shape[0]]
        # Convert logits/scores to a bounded acquisition prior. This is used only
        # for post-fill tie breaking and cannot override the hard budget/active masks.
        prop_prior = 1.0 / (1.0 + np.exp(-np.clip(prop, -20.0, 20.0)))
        utility = utility + float(proposal_fill_weight) * prop_prior.astype(np.float32)
    interaction_utility = utility.copy()
    if float(direction_invariant_interaction_weight) > 0.0 and np.asarray(delta).ndim == 2 and np.asarray(delta).size:
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
