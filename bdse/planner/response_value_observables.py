from __future__ import annotations

"""Counterfactual response-envelope cost observables for V64.3.43.

The V42 current-state risk block evaluates a single kinematic projection and is
zero on a large fraction of frozen RSMR proposals.  V43 instead evaluates the
*already selected* evidence atoms against a fixed, label-free family of future
agent-response hypotheses.  No new neural query is made and no logged future is
consumed.

For each candidate trajectory, three lower-is-better costs are returned:

* selected_evidence_cv_cost: selected-evidence physical cost under the CV mode;
* selected_evidence_response_mean_cost: probability-weighted mean raw cost over
  all runtime-only response modes;
* selected_evidence_response_robust_cost: the same fixed mean/CVaR functional
  used by the robust teacher, but over runtime-only response modes.

The aggregation is performed in raw atom-cost space and normalized only after
aggregation, matching the robust-teacher algebra.  Costs are summed only over
selected evidence atoms, preserving the bounded auditable evidence interface.
"""

from typing import Any, Iterable

import numpy as np

from bdse.data.cache_schema import CandidateBank, EvidenceBank, RuntimeFeatures
from bdse.planner.evidence_atoms import normalize_atom_costs, raw_local_costs_with_hard_events
from bdse.planner.response_modes import build_response_modes, mode_to_label_future
from bdse.planner.robust_teacher import weighted_cvar

RESPONSE_VALUE_OBSERVABLE_NAMES = [
    "selected_evidence_cv_cost",
    "selected_evidence_response_mean_cost",
    "selected_evidence_response_robust_cost",
]


def _selected_atoms(evidence_bank: EvidenceBank, selected_atom_indices: Iterable[int]) -> list[Any]:
    idx = np.asarray(list(selected_atom_indices), dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return []
    if np.any(idx < 0) or np.any(idx >= evidence_bank.E):
        raise ValueError("V43 selected response observable received out-of-range evidence index")
    if len(np.unique(idx)) != idx.size:
        raise ValueError("V43 selected response observable requires unique selected evidence indices")
    return [evidence_bank.atoms[int(i)] for i in idx]


def runtime_selected_response_costs(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    evidence_bank: EvidenceBank,
    selected_atom_indices: Iterable[int],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Return K x 3 label-free response-envelope costs on selected evidence.

    All response modes are generated from the current runtime state.  The
    function explicitly rejects any mode carrying label-future metadata.  This
    is an instrumentation/value layer only; it does not select candidates.
    """

    K = int(candidates.K)
    atoms = _selected_atoms(evidence_bank, selected_atom_indices)
    if K <= 0:
        return np.zeros((0, len(RESPONSE_VALUE_OBSERVABLE_NAMES)), dtype=np.float64), list(RESPONSE_VALUE_OBSERVABLE_NAMES)
    if not atoms:
        return np.zeros((K, len(RESPONSE_VALUE_OBSERVABLE_NAMES)), dtype=np.float64), list(RESPONSE_VALUE_OBSERVABLE_NAMES)

    modes = build_response_modes(runtime, None, cfg)
    if not modes:
        raise ValueError("V43 response-envelope observable requires at least one runtime response mode")
    if any(bool((m.metadata or {}).get("uses_label_future", False)) or str(m.name).lower() == "logged" for m in modes):
        raise ValueError("V43 response-envelope observable must never consume logged/label future")

    raws: list[np.ndarray] = []
    probs: list[float] = []
    cv_raw: np.ndarray | None = None
    for mode in modes:
        lf = mode_to_label_future(mode, None, runtime)
        raw, _ = raw_local_costs_with_hard_events(atoms, candidates, runtime, lf, cfg)
        raw = np.nan_to_num(np.asarray(raw, dtype=np.float32), nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
        if raw.shape != (len(atoms), K):
            raise ValueError("V43 response-envelope raw selected-evidence cost shape mismatch")
        raws.append(raw)
        probs.append(float(mode.probability))
        if str(mode.name).lower() == "cv":
            cv_raw = raw

    if cv_raw is None:
        raise ValueError("V43 response-envelope causal control requires the fixed CV response mode")

    raw_stack = np.stack(raws, axis=0).astype(np.float32)
    p = np.asarray(probs, dtype=np.float32)
    p = p / max(float(p.sum()), 1.0e-6)
    mean_raw = np.tensordot(p, raw_stack, axes=(0, 0)).astype(np.float32)
    rcfg = cfg.get("teacher", {}).get("risk_aggregation", {}) if isinstance(cfg, dict) else {}
    alpha = float(rcfg.get("cvar_alpha", cfg.get("teacher", {}).get("cvar_alpha", 0.9)))
    beta = float(rcfg.get("cvar_weight", cfg.get("teacher", {}).get("cvar_weight", 0.4)))
    cvar_raw = weighted_cvar(raw_stack, p, alpha)
    robust_raw = ((1.0 - beta) * mean_raw + beta * cvar_raw).astype(np.float32)

    def _sum_normalized(raw: np.ndarray) -> np.ndarray:
        g = normalize_atom_costs(raw, atoms, cfg)
        out = np.asarray(g, dtype=np.float64).sum(axis=0, dtype=np.float64)
        out = np.nan_to_num(out, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
        return out

    cv_cost = _sum_normalized(cv_raw)
    mean_cost = _sum_normalized(mean_raw)
    robust_cost = _sum_normalized(robust_raw)
    out = np.stack([cv_cost, mean_cost, robust_cost], axis=1).astype(np.float64)
    if out.shape != (K, len(RESPONSE_VALUE_OBSERVABLE_NAMES)) or not np.all(np.isfinite(out)):
        raise ValueError("V43 response-envelope observable matrix is malformed or non-finite")
    return out, list(RESPONSE_VALUE_OBSERVABLE_NAMES)
