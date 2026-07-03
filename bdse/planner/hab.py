from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from bdse.planner.evidence_queries import FAMILY_NAMES, certificate_family


@dataclass()
class FamilyBudgetResult:
    family_pi: np.ndarray
    family_budgets: np.ndarray
    family_caps: np.ndarray
    free_budget: float
    active_families: np.ndarray
    diagnostics: dict[str, Any]


def max_family_id() -> int:
    return max(int(v) for v in FAMILY_NAMES.values())


def family_ids_from_atoms(atoms: Iterable[Any], max_atoms: int | None = None) -> np.ndarray:
    ids: list[int] = []
    for atom in atoms:
        fam = certificate_family(str(getattr(atom, "type", "")), str(getattr(atom, "family", "")))
        ids.append(int(FAMILY_NAMES.get(fam, FAMILY_NAMES.get(str(getattr(atom, "family", "")), 0))))
    arr = np.asarray(ids, dtype=np.int64)
    if max_atoms is not None and arr.shape[0] < int(max_atoms):
        arr = np.pad(arr, (0, int(max_atoms) - arr.shape[0]), constant_values=0)
    if max_atoms is not None:
        arr = arr[: int(max_atoms)]
    return arr


def _pad_scores(scores: np.ndarray | None, size: int) -> np.ndarray:
    if scores is None:
        return np.zeros((size,), dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    if s.shape[0] < size:
        s = np.pad(s, (0, size - s.shape[0]), constant_values=-np.inf)
    return s[:size]


def family_distribution(
    family_scores: np.ndarray | None,
    family_ids: np.ndarray,
    active_mask: np.ndarray,
    min_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert family logits/scores into a distribution over active families.

    Index 0 is reserved for padding/unknown and receives zero mass unless it is the
    only available family.  When no logits are provided, active atom counts define a
    stable uniform/count-based prior.
    """
    fam = np.asarray(family_ids, dtype=np.int64).reshape(-1)
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    size = int(max(np.max(fam, initial=0) + 1, max_family_id() + 1, int(min_size or 0)))
    present = np.zeros((size,), dtype=bool)
    counts = np.zeros((size,), dtype=np.float32)
    for f in fam[active & (fam >= 0)]:
        if int(f) < size:
            present[int(f)] = True
            counts[int(f)] += 1.0
    if present[1:].any():
        present[0] = False
    if not present.any():
        pi = np.zeros((size,), dtype=np.float32)
        return pi, present

    if family_scores is None:
        raw = counts.copy()
        raw[~present] = 0.0
        if raw.sum() <= 0:
            raw[present] = 1.0
        pi = raw / max(float(raw.sum()), 1e-6)
        return pi.astype(np.float32), present

    logits = _pad_scores(family_scores, size)
    logits = np.where(np.isfinite(logits), logits, -np.inf).astype(np.float32)
    logits[~present] = -np.inf
    if not np.isfinite(logits[present]).any():
        raw = counts.copy()
        raw[~present] = 0.0
        if raw.sum() <= 0:
            raw[present] = 1.0
        pi = raw / max(float(raw.sum()), 1e-6)
        return pi.astype(np.float32), present
    m = float(np.max(logits[present]))
    exp = np.zeros_like(logits, dtype=np.float32)
    exp[present] = np.exp(np.clip(logits[present] - m, -60.0, 60.0))
    pi = exp / max(float(exp.sum()), 1e-6)
    return pi.astype(np.float32), present


def allocate_family_budgets(
    family_scores: np.ndarray | None,
    family_ids: np.ndarray,
    active_mask: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    free_budget: float | None = None,
    reserve_fraction: float = 0.2,
) -> FamilyBudgetResult:
    fam = np.asarray(family_ids, dtype=np.int64).reshape(-1)
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    size = int(max(np.max(fam, initial=0) + 1, max_family_id() + 1))
    B = max(float(budget), 0.0)
    pi, present = family_distribution(family_scores, fam, active, min_size=size)
    if free_budget is None:
        reserve = min(B, max(0.0, round(B * float(reserve_fraction))))
    else:
        reserve = min(B, max(0.0, float(free_budget)))
    spendable = max(B - reserve, 0.0)
    raw = spendable * pi
    family_budgets = np.floor(raw).astype(np.float32)
    # Allocate remaining integer-like budget to families with the largest fractions.
    remaining = max(0, int(round(spendable - float(family_budgets.sum()))))
    order = np.argsort(-(raw - family_budgets + 1e-6 * pi))
    for f in order:
        if remaining <= 0:
            break
        if present[int(f)]:
            family_budgets[int(f)] += 1.0
            remaining -= 1
    # No family can receive more than its available active atom cost in the strict budget.
    available = np.zeros_like(family_budgets)
    for f in np.flatnonzero(present):
        mask = active & (fam == int(f))
        available[int(f)] = float(np.sum(costs[mask])) if mask.any() else 0.0
    family_budgets = np.minimum(family_budgets, available)
    family_caps = family_budgets + reserve
    family_caps[~present] = 0.0
    diagnostics = {
        "family_pi": {int(i): float(pi[i]) for i in np.flatnonzero(present)},
        "family_budgets": {int(i): float(family_budgets[i]) for i in np.flatnonzero(present)},
        "family_caps": {int(i): float(family_caps[i]) for i in np.flatnonzero(present)},
        "family_free_budget": float(reserve),
    }
    return FamilyBudgetResult(
        family_pi=pi.astype(np.float32),
        family_budgets=family_budgets.astype(np.float32),
        family_caps=family_caps.astype(np.float32),
        free_budget=float(reserve),
        active_families=present,
        diagnostics=diagnostics,
    )


def _slot_allocation(pi: np.ndarray, present: np.ndarray, active_counts: np.ndarray, total_slots: int, min_family_slots: dict[int, int] | None = None) -> np.ndarray:
    slots = np.zeros_like(pi, dtype=np.int64)
    M = max(int(total_slots), 0)
    fams = [int(f) for f in np.flatnonzero(present) if active_counts[int(f)] > 0]
    if M <= 0 or not fams:
        return slots
    mins = {int(k): max(0, int(v)) for k, v in (min_family_slots or {}).items()}
    # Reserve a few Top-M slots for safety/rule/interaction families.  This is a
    # proposal-stage recall guard, not teacher leakage: it only uses atom family ids.
    remaining_slots = M
    for f in sorted(fams):
        want = min(int(active_counts[f]), int(mins.get(int(f), 0)), remaining_slots)
        if want > 0:
            slots[int(f)] = max(slots[int(f)], want)
            remaining_slots -= want
    if remaining_slots <= 0:
        return slots
    ordered = sorted(fams, key=lambda f: (-float(pi[f]), f))
    if remaining_slots < len([f for f in ordered if slots[f] == 0]):
        for f in ordered:
            if slots[f] == 0 and remaining_slots > 0:
                slots[f] = 1
                remaining_slots -= 1
        return slots
    for f in ordered:
        if slots[f] == 0 and remaining_slots > 0:
            slots[f] = 1
            remaining_slots -= 1
    remaining = remaining_slots
    raw = M * pi
    # Families with higher desired count and fractional remainder receive extra slots.
    while remaining > 0:
        best = None
        best_key = (-np.inf, -np.inf)
        for f in ordered:
            if slots[f] >= int(active_counts[f]):
                continue
            key = (float(raw[f] - slots[f]), float(pi[f]))
            if key > best_key:
                best_key = key
                best = f
        if best is None:
            break
        slots[int(best)] += 1
        remaining -= 1
    return slots


def select_topm_atoms_hab(
    proposal_logits: np.ndarray,
    family_ids: np.ndarray,
    active_mask: np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    proposal_top_m: int,
    family_scores: np.ndarray | None = None,
    free_budget: float | None = None,
    reserve_fraction: float = 0.2,
    enabled: bool = True,
    min_family_slots: dict[int, int] | None = None,
) -> tuple[np.ndarray, FamilyBudgetResult, dict[str, Any]]:
    """Family-gated Top-M proposal for the Hierarchical Atom Builder.

    The returned atom ids are the only atoms eligible for expensive action/pair
    evidence queries.  The helper also returns family caps for the final greedy
    selector; callers may pass ``family_caps`` to ``runtime_greedy_selector``.
    """
    logits = np.asarray(proposal_logits, dtype=np.float32).reshape(-1)
    fam = np.asarray(family_ids, dtype=np.int64).reshape(-1)
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    costs = np.asarray(atom_budget_costs, dtype=np.float32).reshape(-1)
    E = logits.shape[0]
    if fam.shape[0] < E:
        fam = np.pad(fam, (0, E - fam.shape[0]), constant_values=0)
    fam = fam[:E]
    if active.shape[0] < E:
        active = np.pad(active, (0, E - active.shape[0]), constant_values=False)
    active = active[:E]
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=np.inf)
    costs = costs[:E]
    M = min(max(int(proposal_top_m), 0), int(active.sum()) if active.any() else E)
    result = allocate_family_budgets(family_scores, fam, active, costs, budget, free_budget=free_budget, reserve_fraction=reserve_fraction)
    if M <= 0:
        return np.zeros((0,), dtype=np.int64), result, {**result.diagnostics, "hab_enabled": bool(enabled), "proposal_top_m": 0}
    if not enabled:
        masked = np.where(active, logits, -np.inf)
        top = np.argsort(-masked)[:M].astype(np.int64)
        top = top[np.isfinite(masked[top])]
        return top.astype(np.int64), result, {**result.diagnostics, "hab_enabled": False, "proposal_top_m": int(len(top))}

    size = result.family_pi.shape[0]
    counts = np.zeros((size,), dtype=np.int64)
    for f in np.flatnonzero(result.active_families):
        counts[int(f)] = int(np.sum(active & (fam == int(f))))
    slots = _slot_allocation(result.family_pi, result.active_families, counts, M, min_family_slots=min_family_slots)
    selected: list[int] = []
    selected_set: set[int] = set()
    for f in np.flatnonzero(slots > 0):
        ids = np.flatnonzero(active & (fam == int(f)))
        if ids.size == 0:
            continue
        order = sorted(ids.tolist(), key=lambda i: (-float(logits[i]), float(costs[i]), int(i)))
        for i in order[: int(slots[int(f)])]:
            selected.append(int(i)); selected_set.add(int(i))
    # Free/cross-family reserve: fill remaining slots globally by acquisition score.
    if len(selected) < M:
        global_order = sorted(np.flatnonzero(active).tolist(), key=lambda i: (-float(logits[i]), float(costs[i]), int(i)))
        for i in global_order:
            if i in selected_set:
                continue
            selected.append(int(i)); selected_set.add(int(i))
            if len(selected) >= M:
                break
    top = np.asarray(selected[:M], dtype=np.int64)
    diag = {
        **result.diagnostics,
        "hab_enabled": True,
        "proposal_top_m": int(len(top)),
        "proposal_slots": {int(i): int(slots[i]) for i in np.flatnonzero(slots)},
    }
    return top, result, diag
