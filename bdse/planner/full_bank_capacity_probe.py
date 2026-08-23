from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FullBankCapacityProbeResult:
    """Result of the V64.3.30 full-bank interface capacity intervention.

    This is intentionally a *capacity ceiling*, not a new acquisition rule.
    It can expose every already-queried decision atom in the frozen Top-M bank
    to the downstream planner when the configured interface budget can pay the
    complete bank.  It never adds a model/evidence query and it is disabled in
    the all-flagged structural domain so that the V23+ structural-preservation
    contract is not changed by this diagnostic.
    """

    selected: list[int]
    diagnostics: dict[str, Any]


def full_bank_capacity_probe(
    *,
    baseline_selected: list[int] | np.ndarray,
    reference_atoms: list[int] | np.ndarray,
    atom_budget_costs: np.ndarray,
    budget: float,
    structural_domain: bool,
    expected_top_m: int | None = 24,
    epsilon: float = 1e-8,
) -> FullBankCapacityProbeResult:
    """Expose the complete already-queried Top-M bank when budget-feasible.

    The function has no learned/tunable objective.  Its sole purpose is to
    answer a causal question left open by V29: if the B=16 compression itself
    is the bottleneck, does removing *only* that compression while keeping the
    queried M=24 bank fixed recover the missing downstream signal?

    Numeric reason codes are stable for scalar-only evaluation diagnostics:
      0 = applied;
      1 = structural-domain preservation (no-op);
      2 = empty/invalid reference bank (no-op);
      3 = reference contains invalid/duplicate atom (no-op);
      4 = complete bank is not budget-feasible (no-op);
      5 = reference exceeds expected Top-M cardinality (no-op);
      6 = no capacity expansion over baseline (no-op);
      7 = baseline is not a subset of the already-queried reference bank (no-op);
      8 = baseline itself contains an invalid/duplicate atom (no-op).
    """

    eps = max(float(epsilon), 0.0)
    baseline_raw = [int(x) for x in np.asarray(baseline_selected, dtype=np.int64).reshape(-1).tolist()]
    reference_raw = [int(x) for x in np.asarray(reference_atoms, dtype=np.int64).reshape(-1).tolist()]
    baseline = list(baseline_raw)
    reference = list(dict.fromkeys(reference_raw))
    costs = np.asarray(atom_budget_costs, dtype=np.float64).reshape(-1)

    diag: dict[str, Any] = {
        "full_bank_capacity_probe_enabled": 1.0,
        "full_bank_capacity_probe_attempted": 1.0,
        "full_bank_capacity_probe_applied": 0.0,
        "full_bank_capacity_probe_reason_code": 0.0,
        "full_bank_capacity_probe_budget": float(budget),
        "full_bank_capacity_probe_baseline_count": float(len(baseline)),
        "full_bank_capacity_probe_reference_count": float(len(reference)),
        "full_bank_capacity_probe_final_count": float(len(baseline)),
        "full_bank_capacity_probe_no_new_query": 1.0,
        "full_bank_capacity_probe_teacher_free": 1.0,
        "full_bank_capacity_probe_structural_domain": float(bool(structural_domain)),
        "full_bank_capacity_probe_structural_preservation": 1.0,
    }

    def fallback(reason: int, **extra: Any) -> FullBankCapacityProbeResult:
        diag["full_bank_capacity_probe_reason_code"] = float(reason)
        diag.update(extra)
        return FullBankCapacityProbeResult(selected=list(baseline), diagnostics=dict(diag))

    if structural_domain:
        return fallback(1)
    if not reference or costs.size == 0:
        return fallback(2)
    if len(baseline) != len(set(baseline)) or any(i < 0 or i >= costs.size for i in baseline):
        return fallback(8)
    if len(reference) != len(reference_raw) or any(i < 0 or i >= costs.size for i in reference):
        return fallback(3)
    # A capacity-only intervention may add already-queried atoms, but it must
    # never delete/reallocate an atom chosen by the frozen B=16 selector.
    # Any baseline atom outside the reference bank means the assumed causal
    # nesting is false for this scene, so fail closed rather than changing two
    # mechanisms at once.
    if not set(baseline).issubset(set(reference)):
        return fallback(7)
    if expected_top_m is not None and int(expected_top_m) > 0 and len(reference) > int(expected_top_m):
        return fallback(5)

    ref_costs = costs[np.asarray(reference, dtype=np.int64)]
    total_cost = float(np.sum(ref_costs))
    diag["full_bank_capacity_probe_reference_cost"] = total_cost
    if (not np.isfinite(ref_costs).all()) or np.any(ref_costs < 0.0) or total_cost > float(budget) + eps:
        return fallback(4)
    if len(reference) <= len(baseline):
        return fallback(6)

    # Keep the deterministic Top-M order supplied by the caller.  No ranking,
    # teacher label, classifier, threshold, fallback action, or extra query is
    # introduced here.
    diag["full_bank_capacity_probe_applied"] = 1.0
    diag["full_bank_capacity_probe_final_count"] = float(len(reference))
    diag["full_bank_capacity_probe_added_atom_count"] = float(len(set(reference) - set(baseline)))
    diag["full_bank_capacity_probe_removed_atom_count"] = float(len(set(baseline) - set(reference)))
    diag["full_bank_capacity_probe_budget_preserved"] = float(total_cost <= float(budget) + eps)
    return FullBankCapacityProbeResult(selected=list(reference), diagnostics=dict(diag))
