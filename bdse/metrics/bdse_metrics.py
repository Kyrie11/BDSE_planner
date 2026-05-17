from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, EvidenceBank, PairLabels, TeacherLabels
from bdse.planner.selector import _finite_cost_for_margin, budgeted_margin, oracle_objective_value
from bdse.planner.tournament import full_interface_action


@dataclass(slots=True)
class BDSEMetricResult:
    values: dict[str, float]
    details: dict[str, Any]


def evidence_sufficiency(M_teacher: np.ndarray, M_pred: np.ndarray, pairs: np.ndarray, weights: np.ndarray, eps: float = 1e-6) -> float:
    if len(pairs) == 0:
        return 1.0
    errs = []
    den = []
    for w, (a, b) in zip(weights, pairs):
        errs.append(float(w) * abs(float(M_teacher[a, b] - M_pred[a, b])))
        den.append(float(w) * abs(float(M_teacher[a, b])))
    return float(np.clip(1.0 - sum(errs) / (sum(den) + eps), 0.0, 1.0))


def compute_bdse_diagnostics(
    candidates: CandidateBank,
    evidence_bank: EvidenceBank,
    teacher: TeacherLabels,
    pairs: PairLabels,
    predicted_base: np.ndarray,
    predicted_atom_costs: np.ndarray,
    selected_atoms: list[int],
    action_index: int,
    runtime_selected_atoms_for_oracle_value: list[int] | None = None,
    oracle_selected_atoms: list[int] | None = None,
    cfg: dict[str, Any] | None = None,
) -> BDSEMetricResult:
    cfg = cfg or {}
    valid = candidates.valid_mask.astype(bool)
    J = teacher.J_T
    a_star = int(teacher.a_star)
    J_margin = _finite_cost_for_margin(J)
    teacher_M = J_margin[None, :] - J_margin[:, None]
    M_B = budgeted_margin(predicted_base, predicted_atom_costs, selected_atoms)
    full_action = full_interface_action(predicted_base, predicted_atom_costs, valid, cfg)
    budget_vs_full = int(action_index == full_action)
    teacher_regret = float(J[action_index] - J[a_star]) if valid[action_index] else float("inf")
    hard = evidence_bank.hard_mask() & evidence_bank.active_mask
    selected_set = set(map(int, selected_atoms))
    active_hard = set(map(int, np.flatnonzero(hard)))
    # Paper-relevant hard recall: hard atoms that actually support at least one
    # labeled positive pair, not every hard atom merely present in the scene.
    decisive_hard: set[int] = set()
    if len(pairs.pairs):
        for a, b in np.asarray(pairs.pairs[pairs.valid_mask], dtype=np.int64):
            delta = np.asarray(teacher.g_evid[:, b] - teacher.g_evid[:, a], dtype=np.float32)
            for i in np.flatnonzero(hard & (delta > 1e-6)):
                decisive_hard.add(int(i))
    denom = decisive_hard if decisive_hard else active_hard
    hard_recall = float(len(selected_set & denom) / max(len(denom), 1))
    suff = evidence_sufficiency(teacher_M, M_B, pairs.pairs[pairs.valid_mask], pairs.weights[pairs.valid_mask])
    selector_ratio = np.nan
    if runtime_selected_atoms_for_oracle_value is not None and oracle_selected_atoms is not None and len(pairs.pairs):
        F_run = oracle_objective_value(runtime_selected_atoms_for_oracle_value, teacher.J_base, teacher.g_evid, pairs.pairs, pairs.margins, pairs.weights)
        F_oracle = oracle_objective_value(oracle_selected_atoms, teacher.J_base, teacher.g_evid, pairs.pairs, pairs.margins, pairs.weights)
        selector_ratio = float(F_run / (F_oracle + 1e-6))
    decisive_err = []
    for b in np.flatnonzero(valid):
        if b != a_star:
            decisive_err.append(abs(float(M_B[a_star, b] - teacher_M[a_star, b])))
    values = {
        "teacher_regret": teacher_regret,
        "teacher_action_match": float(action_index == a_star),
        "full_interface_action_match": float(full_action == a_star),
        "budget_vs_full_match": float(budget_vs_full),
        "preserved_margin_error": float(np.mean(decisive_err)) if decisive_err else 0.0,
        "evidence_sufficiency": suff,
        "decision_sufficiency": float(action_index == a_star),
        "selector_value_ratio": selector_ratio,
        "hard_evidence_recall": hard_recall,
        "effective_query_count": float(len(selected_atoms)),
        "teacher_pair_count": float(len(pairs.pairs)),
    }
    return BDSEMetricResult(values=values, details={"full_action": full_action, "a_star": a_star, "selected_atoms": selected_atoms})


def aggregate_metric_results(results: list[BDSEMetricResult]) -> dict[str, float]:
    keys = sorted({k for r in results for k in r.values})
    out = {}
    for k in keys:
        vals = [r.values[k] for r in results if k in r.values and np.isfinite(r.values[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
