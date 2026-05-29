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


def _safe_list_int(x: Any) -> list[int]:
    if x is None:
        return []
    try:
        return [int(v) for v in list(x)]
    except Exception:
        return []


def _decisive_rivals(teacher_M: np.ndarray, valid: np.ndarray, a_star: int, pairs: PairLabels) -> set[int]:
    decisive: set[int] = set()
    if pairs is not None and len(pairs.pairs):
        pair_arr = np.asarray(pairs.pairs, dtype=np.int64).reshape(-1, 2)
        pvalid = np.asarray(pairs.valid_mask, dtype=bool).reshape(-1)[: len(pair_arr)]
        for a, b in pair_arr[pvalid]:
            if int(a) == int(a_star) and 0 <= int(b) < len(valid) and valid[int(b)] and teacher_M[int(a_star), int(b)] > 0:
                decisive.add(int(b))
    if not decisive:
        for b in np.flatnonzero(valid):
            if int(b) != int(a_star) and teacher_M[int(a_star), int(b)] > 0:
                decisive.add(int(b))
    return decisive


def _critical_atoms(evidence_bank: EvidenceBank, teacher: TeacherLabels, pairs: PairLabels, eps: float = 1e-6) -> set[int]:
    hard = evidence_bank.hard_mask() & evidence_bank.active_mask
    critical: set[int] = set()
    if pairs is None or not len(pairs.pairs):
        return critical
    pair_arr = np.asarray(pairs.pairs, dtype=np.int64).reshape(-1, 2)
    pvalid = np.asarray(pairs.valid_mask, dtype=bool).reshape(-1)[: len(pair_arr)]
    g = np.asarray(teacher.g_evid, dtype=np.float32)
    for a, b in pair_arr[pvalid]:
        if not (0 <= int(a) < g.shape[1] and 0 <= int(b) < g.shape[1]):
            continue
        support = g[:, int(b)] - g[:, int(a)]
        for i in np.flatnonzero((support > eps) & np.asarray(evidence_bank.active_mask, dtype=bool)):
            critical.add(int(i))
        # Always count hard atoms that support labeled positive pairs as critical.
        for i in np.flatnonzero((support > eps) & hard):
            critical.add(int(i))
    return critical


def _rival_recall(decisive: set[int], a_star: int, rival_sets: list[list[int]] | None) -> tuple[float, float]:
    if not decisive:
        return 1.0, 1.0
    if not rival_sets or a_star >= len(rival_sets):
        return float("nan"), float("nan")
    forward = set(int(x) for x in rival_sets[a_star])
    forward_hit = sum(1 for b in decisive if b in forward)
    symmetric_hit = 0
    for b in decisive:
        if b in forward and b < len(rival_sets) and a_star in set(int(x) for x in rival_sets[b]):
            symmetric_hit += 1
    denom = max(len(decisive), 1)
    return float(forward_hit / denom), float(symmetric_hit / denom)


def _query_counts(selected_atoms: list[int], candidates: CandidateBank, planner_diagnostics: dict[str, Any] | None) -> dict[str, float]:
    diag = planner_diagnostics or {}
    queried_actions = _safe_list_int(diag.get("queried_actions"))
    q_actions = len(queried_actions) if queried_actions else int(np.asarray(candidates.valid_mask, dtype=bool).sum())
    proposed = _safe_list_int(diag.get("proposal_top_m_atoms"))
    rival_sets = diag.get("rival_sets")
    if isinstance(rival_sets, list):
        tournament_pairs = int(sum(len(r) for r in rival_sets))
    else:
        tournament_pairs = int(diag.get("tournament_comparison_count", 0) or 0)
    sparse_query_count = int(diag.get("sparse_query_count", 0) or 0)
    if sparse_query_count <= 0:
        sparse_query_count = int(len(proposed) * max(q_actions, 1)) if proposed else int(len(selected_atoms) * max(q_actions, 1))
    selected_query_count = int(diag.get("selected_certificate_query_count", 0) or 0)
    if selected_query_count <= 0:
        selected_query_count = int(len(selected_atoms) * max(q_actions, 1))
    return {
        "proposed_atom_count": float(len(proposed)),
        "queried_action_count": float(q_actions),
        "sparse_query_count": float(sparse_query_count),
        "selected_certificate_query_count": float(selected_query_count),
        "effective_query_count": float(selected_query_count),
        "tournament_comparison_count": float(tournament_pairs),
    }


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
    planner_diagnostics: dict[str, Any] | None = None,
    full_predicted_atom_costs: np.ndarray | None = None,
) -> BDSEMetricResult:
    cfg = cfg or {}
    valid = candidates.valid_mask.astype(bool)
    J = teacher.J_T
    a_star = int(teacher.a_star)
    selected_atoms = [int(x) for x in selected_atoms]
    J_margin = _finite_cost_for_margin(J)
    teacher_M = J_margin[None, :] - J_margin[:, None]
    M_B = budgeted_margin(predicted_base, predicted_atom_costs, selected_atoms)
    full_g = np.asarray(full_predicted_atom_costs if full_predicted_atom_costs is not None else predicted_atom_costs, dtype=np.float32)
    full_source = "dense" if full_predicted_atom_costs is not None else "sparse_fallback"
    full_action = full_interface_action(predicted_base, full_g, valid, cfg)
    budget_vs_full = int(action_index == full_action)
    teacher_regret = float(J[action_index] - J[a_star]) if 0 <= action_index < len(valid) and valid[action_index] else float("inf")

    hard = evidence_bank.hard_mask() & evidence_bank.active_mask
    selected_set = set(selected_atoms)
    active_hard = set(map(int, np.flatnonzero(hard)))
    decisive_hard: set[int] = set()
    if len(pairs.pairs):
        for a, b in np.asarray(pairs.pairs[pairs.valid_mask], dtype=np.int64):
            delta = np.asarray(teacher.g_evid[:, b] - teacher.g_evid[:, a], dtype=np.float32)
            for i in np.flatnonzero(hard & (delta > 1e-6)):
                decisive_hard.add(int(i))
    denom = decisive_hard if decisive_hard else active_hard
    hard_recall = float(len(selected_set & denom) / max(len(denom), 1))

    pair_valid = np.asarray(pairs.valid_mask, dtype=bool)
    valid_pairs = np.asarray(pairs.pairs, dtype=np.int64).reshape(-1, 2)[pair_valid]
    valid_weights = np.asarray(pairs.weights, dtype=np.float32).reshape(-1)[pair_valid]
    suff = evidence_sufficiency(teacher_M, M_B, valid_pairs, valid_weights)

    selector_ratio = np.nan
    if runtime_selected_atoms_for_oracle_value is not None and oracle_selected_atoms is not None and len(pairs.pairs):
        F_run = oracle_objective_value(runtime_selected_atoms_for_oracle_value, teacher.J_base, teacher.g_evid, pairs.pairs, pairs.margins, pairs.weights)
        F_oracle = oracle_objective_value(oracle_selected_atoms, teacher.J_base, teacher.g_evid, pairs.pairs, pairs.margins, pairs.weights)
        selector_ratio = float(F_run / (F_oracle + 1e-6))

    decisive = _decisive_rivals(teacher_M, valid, a_star, pairs)
    rival_sets = None
    if planner_diagnostics is not None and isinstance(planner_diagnostics.get("rival_sets"), list):
        rival_sets = [[int(x) for x in r] for r in planner_diagnostics.get("rival_sets", [])]
    decisive_forward_recall, decisive_symmetric_recall = _rival_recall(decisive, a_star, rival_sets)

    decisive_err = [abs(float(M_B[a_star, b] - teacher_M[a_star, b])) for b in decisive]
    all_teacher_err = []
    for b in np.flatnonzero(valid):
        if b != a_star:
            all_teacher_err.append(abs(float(M_B[a_star, b] - teacher_M[a_star, b])))

    critical = _critical_atoms(evidence_bank, teacher, pairs)
    top_m = set(_safe_list_int((planner_diagnostics or {}).get("proposal_top_m_atoms")))
    proposal_recall = float(len(critical & top_m) / max(len(critical), 1)) if critical else 1.0
    selected_critical_recall = float(len(critical & selected_set) / max(len(critical), 1)) if critical else 1.0

    query_values = _query_counts(selected_atoms, candidates, planner_diagnostics)
    stage_records = (planner_diagnostics or {}).get("fallback_stage_records", []) or []
    expanded_budget = np.nan
    expanded_L = np.nan
    if stage_records:
        expanded_budget = float(max(float(r.get("budget", 0.0)) for r in stage_records))
        expanded_L = float(max(float(r.get("L_infer", 0.0)) for r in stage_records))

    values = {
        "teacher_regret": teacher_regret,
        "teacher_action_match": float(action_index == a_star),
        "full_interface_action_match": float(full_action == a_star),
        "budget_vs_full_match": float(budget_vs_full),
        "preserved_margin_error": float(np.mean(decisive_err)) if decisive_err else 0.0,
        "preserved_margin_error_all_teacher_rivals": float(np.mean(all_teacher_err)) if all_teacher_err else 0.0,
        "evidence_sufficiency": suff,
        "decision_sufficiency": float(action_index == a_star),
        "selector_value_ratio": selector_ratio,
        "hard_evidence_recall": hard_recall,
        "decisive_rival_recall": decisive_forward_recall,
        "symmetric_decisive_rival_recall": decisive_symmetric_recall,
        "proposal_recall": proposal_recall,
        "selected_critical_atom_recall": selected_critical_recall,
        "critical_atom_count": float(len(critical)),
        "decisive_rival_count": float(len(decisive)),
        "teacher_pair_count": float(len(pairs.pairs)),
        "fallback_rate": float(bool((planner_diagnostics or {}).get("fallback_triggered", False))),
        "rule_rerank_rate": float(bool((planner_diagnostics or {}).get("rule_rerank_used", False))),
        "conservative_fallback_rate": float(bool((planner_diagnostics or {}).get("conservative_fallback_used", False))),
        "latency_ms": float((planner_diagnostics or {}).get("latency_ms", np.nan)),
        "expanded_budget_max": expanded_budget,
        "expanded_L_infer_max": expanded_L,
        **query_values,
    }
    return BDSEMetricResult(
        values=values,
        details={
            "full_action": full_action,
            "full_interface_source": full_source,
            "a_star": a_star,
            "selected_atoms": selected_atoms,
            "decisive_rivals": sorted(decisive),
            "critical_atoms": sorted(critical),
        },
    )


def aggregate_metric_results(results: list[BDSEMetricResult]) -> dict[str, float]:
    keys = sorted({k for r in results for k in r.values})
    out = {}
    for k in keys:
        vals = [r.values[k] for r in results if k in r.values and np.isfinite(r.values[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
