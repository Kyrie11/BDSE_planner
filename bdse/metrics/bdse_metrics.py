from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, EvidenceBank, PairLabels, TeacherLabels
from bdse.planner.selector import _finite_cost_for_margin, budgeted_margin, oracle_objective_value, structural_safety_mask
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


def _cost_margin_matrix(base: np.ndarray, atom_costs: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    J0 = np.asarray(base, dtype=np.float64).reshape(-1)
    g = np.asarray(atom_costs, dtype=np.float64)
    K = J0.shape[0]
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.shape[0] < K:
        valid = np.pad(valid, (0, K - valid.shape[0]), constant_values=False)
    valid = valid[:K]
    if g.ndim != 2 or g.shape[1] != K:
        cost = J0.copy()
    else:
        cost = J0 + g.sum(axis=0)
    cost = np.asarray(cost, dtype=np.float64)
    cost[~valid] = np.inf
    with np.errstate(invalid="ignore"):
        M = cost[None, :] - cost[:, None]
    M[~np.isfinite(M)] = -1e9
    np.fill_diagonal(M, 0.0)
    return M.astype(np.float32)


def _pair_group_metrics(prefix: str, M_teacher: np.ndarray, M_pred: np.ndarray, pair_arr: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    pair_arr = np.asarray(pair_arr, dtype=np.int64).reshape(-1, 2)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if pair_arr.shape[0] != mask.shape[0]:
        mask = np.zeros((pair_arr.shape[0],), dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_sign_acc": float("nan"),
            f"{prefix}_margin_mae": float("nan"),
            f"{prefix}_margin_signed_error": float("nan"),
        }
    a = pair_arr[idx, 0]
    b = pair_arr[idx, 1]
    ok = (a >= 0) & (a < M_teacher.shape[0]) & (b >= 0) & (b < M_teacher.shape[1]) & (a < M_pred.shape[0]) & (b < M_pred.shape[1])
    if not np.any(ok):
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_sign_acc": float("nan"),
            f"{prefix}_margin_mae": float("nan"),
            f"{prefix}_margin_signed_error": float("nan"),
        }
    a = a[ok]
    b = b[ok]
    t = np.asarray(M_teacher[a, b], dtype=np.float64)
    p = np.asarray(M_pred[a, b], dtype=np.float64)
    finite = np.isfinite(t) & np.isfinite(p) & (np.abs(t) > 1e-9)
    if not np.any(finite):
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_sign_acc": float("nan"),
            f"{prefix}_margin_mae": float("nan"),
            f"{prefix}_margin_signed_error": float("nan"),
        }
    t = t[finite]
    p = p[finite]
    if weights is not None:
        w0 = np.asarray(weights, dtype=np.float64).reshape(-1)[idx][ok][finite]
        w = np.maximum(w0, 0.0)
        if float(w.sum()) <= 1e-12:
            w = np.ones_like(t, dtype=np.float64)
    else:
        w = np.ones_like(t, dtype=np.float64)
    correct = (np.sign(t) == np.sign(p)) & (np.sign(p) != 0)
    return {
        f"{prefix}_count": float(t.size),
        f"{prefix}_sign_acc": float(np.average(correct.astype(np.float64), weights=w)),
        f"{prefix}_margin_mae": float(np.average(np.abs(p - t), weights=w)),
        f"{prefix}_margin_signed_error": float(np.average(p - t, weights=w)),
    }


def _recall(num_set: set[int], den_set: set[int]) -> float:
    return float(len(num_set & den_set) / max(len(den_set), 1))


def _critical_atom_sets(evidence_bank: EvidenceBank, teacher: TeacherLabels, pairs: PairLabels, cfg: dict[str, Any]) -> dict[str, set[int]]:
    active = np.asarray(evidence_bank.active_mask, dtype=bool).reshape(-1)
    hard = np.asarray(evidence_bank.hard_mask(), dtype=bool).reshape(-1)
    n = min(active.shape[0], hard.shape[0], teacher.g_evid.shape[0], len(evidence_bank.atoms))
    active = active[:n]
    hard = hard[:n]
    eps = float(cfg.get("diagnostics", {}).get("critical_atom_delta_eps", 1e-6))
    families = [str(getattr(a, "family", "")) for a in evidence_bank.atoms[:n]]
    types = [str(getattr(a, "type", "")) for a in evidence_bank.atoms[:n]]
    interaction_mask = np.asarray([
        (families[i] in {"interaction", "reachability_interaction", "precedence"})
        or ("interaction" in families[i])
        or (types[i] in {"lead_vehicle", "yield", "precedence"})
        for i in range(n)
    ], dtype=bool)
    decisive: set[int] = set()
    interaction_decisive: set[int] = set()
    soft_interaction_decisive: set[int] = set()
    hard_decisive: set[int] = set()
    pair_arr = np.asarray(pairs.pairs[pairs.valid_mask], dtype=np.int64).reshape(-1, 2) if pairs is not None else np.zeros((0, 2), dtype=np.int64)
    for a, b in pair_arr:
        if not (0 <= a < teacher.g_evid.shape[1] and 0 <= b < teacher.g_evid.shape[1]):
            continue
        delta = np.asarray(teacher.g_evid[:n, int(b)] - teacher.g_evid[:n, int(a)], dtype=np.float32)
        support = active & (delta > eps)
        for ei in np.flatnonzero(support):
            decisive.add(int(ei))
        for ei in np.flatnonzero(support & interaction_mask):
            interaction_decisive.add(int(ei))
        for ei in np.flatnonzero(support & interaction_mask & ~hard):
            soft_interaction_decisive.add(int(ei))
        for ei in np.flatnonzero(support & hard):
            hard_decisive.add(int(ei))
    return {
        "active": set(map(int, np.flatnonzero(active).tolist())),
        "hard": set(map(int, np.flatnonzero(active & hard).tolist())),
        "interaction": set(map(int, np.flatnonzero(active & interaction_mask).tolist())),
        "decisive": decisive,
        "interaction_decisive": interaction_decisive,
        "soft_interaction_decisive": soft_interaction_decisive,
        "hard_decisive": hard_decisive,
    }


def _pair_group_masks(evidence_bank: EvidenceBank, teacher: TeacherLabels, pairs: PairLabels, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    pair_arr = np.asarray(pairs.pairs, dtype=np.int64).reshape(-1, 2)
    valid_pair = np.asarray(pairs.valid_mask, dtype=bool).reshape(-1)
    n_pairs = pair_arr.shape[0]
    if valid_pair.shape[0] != n_pairs:
        valid_pair = np.zeros((n_pairs,), dtype=bool)
    out: dict[str, np.ndarray] = {"pair_sign_acc_all": valid_pair.copy()}
    if n_pairs == 0:
        empty = np.zeros((0,), dtype=bool)
        return {
            "pair_sign_acc_all": empty,
            "pair_sign_acc_winner_rival": empty,
            "pair_sign_acc_hard": empty,
            "pair_sign_acc_interaction": empty,
            "pair_sign_acc_near_tie": empty,
        }
    a_star = int(teacher.a_star)
    a = pair_arr[:, 0]
    b = pair_arr[:, 1]
    out["pair_sign_acc_winner_rival"] = valid_pair & (a == a_star)
    hard_violation = np.asarray(teacher.hard_violation_mask, dtype=bool).reshape(-1)
    hard_pair = np.zeros((n_pairs,), dtype=bool)
    ok = (a >= 0) & (a < hard_violation.shape[0]) & (b >= 0) & (b < hard_violation.shape[0])
    hard_pair[ok] = (~hard_violation[a[ok]]) & hard_violation[b[ok]]
    out["pair_sign_acc_hard"] = valid_pair & hard_pair

    margins = np.asarray(pairs.margins, dtype=np.float32).reshape(-1)
    finite_m = margins[valid_pair & np.isfinite(margins) & (margins > 0)]
    if finite_m.size:
        thr = float(np.percentile(finite_m, float(cfg.get("diagnostics", {}).get("near_tie_pair_percentile", 25.0))))
        out["pair_sign_acc_near_tie"] = valid_pair & np.isfinite(margins) & (margins <= thr)
    else:
        out["pair_sign_acc_near_tie"] = np.zeros((n_pairs,), dtype=bool)

    active = np.asarray(evidence_bank.active_mask, dtype=bool).reshape(-1)
    n = min(active.shape[0], teacher.g_evid.shape[0], len(evidence_bank.atoms))
    families = [str(getattr(x, "family", "")) for x in evidence_bank.atoms[:n]]
    types = [str(getattr(x, "type", "")) for x in evidence_bank.atoms[:n]]
    interaction_mask = np.asarray([
        (families[i] in {"interaction", "reachability_interaction", "precedence"})
        or ("interaction" in families[i])
        or (types[i] in {"lead_vehicle", "yield", "precedence"})
        for i in range(n)
    ], dtype=bool)
    eps = float(cfg.get("diagnostics", {}).get("critical_atom_delta_eps", 1e-6))
    inter_pair = np.zeros((n_pairs,), dtype=bool)
    for idx, (aa, bb) in enumerate(pair_arr.tolist()):
        if not valid_pair[idx] or not (0 <= aa < teacher.g_evid.shape[1] and 0 <= bb < teacher.g_evid.shape[1]):
            continue
        delta = np.asarray(teacher.g_evid[:n, int(bb)] - teacher.g_evid[:n, int(aa)], dtype=np.float32)
        inter_pair[idx] = bool(np.any((active[:n] & interaction_mask & (delta > eps))))
    out["pair_sign_acc_interaction"] = valid_pair & inter_pair
    return out


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
    inference_pairs: np.ndarray | None = None,
    queried_atom_count: int | None = None,
    query_diagnostics: dict[str, Any] | None = None,
    dense_predicted_base: np.ndarray | None = None,
    dense_predicted_atom_costs: np.ndarray | None = None,
    certificate_margin_matrix: np.ndarray | None = None,
) -> BDSEMetricResult:
    cfg = cfg or {}
    valid = candidates.valid_mask.astype(bool)
    J = teacher.J_T
    a_star = int(teacher.a_star)
    J_margin = _finite_cost_for_margin(J)
    teacher_M = J_margin[None, :] - J_margin[:, None]
    M_B_from_g = budgeted_margin(predicted_base, predicted_atom_costs, selected_atoms)
    if certificate_margin_matrix is not None:
        M_B = np.asarray(certificate_margin_matrix, dtype=np.float32)
        if M_B.shape != M_B_from_g.shape:
            M_B = M_B_from_g
    else:
        M_B = M_B_from_g
    qdiag = query_diagnostics or {}
    cert_is_normalized = bool(qdiag.get("normalized_margins", False)) or bool(cfg.get("model", {}).get("pair_margin_normalized", False) and certificate_margin_matrix is not None)
    mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    tcfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    scale_default = float(mcfg.get("margin_normalization_min_scale", tcfg.get("pair_margin_min_scale", 100.0)))
    cert_scale = float(qdiag.get("margin_scale", scale_default)) if cert_is_normalized else 1.0
    teacher_M_cert = teacher_M / max(cert_scale, 1e-6) if cert_is_normalized else teacher_M
    if dense_predicted_base is not None and dense_predicted_atom_costs is not None:
        full_action = full_interface_action(dense_predicted_base, dense_predicted_atom_costs, valid, cfg)
        dense_M = _cost_margin_matrix(dense_predicted_base, dense_predicted_atom_costs, valid)
        sparse_full_action = full_interface_action(predicted_base, predicted_atom_costs, valid, cfg)
    else:
        full_action = full_interface_action(predicted_base, predicted_atom_costs, valid, cfg)
        dense_M = _cost_margin_matrix(predicted_base, predicted_atom_costs, valid)
        sparse_full_action = full_action
    finite_base = _finite_cost_for_margin(np.asarray(predicted_base, dtype=np.float32).reshape(-1))
    base_M = finite_base[None, :] - finite_base[:, None]
    valid_base = valid & np.isfinite(np.asarray(predicted_base, dtype=np.float32).reshape(-1))
    base_action = int(np.argmin(np.where(valid_base, finite_base, np.inf))) if bool(valid_base.any()) else -1

    def _teacher_regret_for(action: int) -> float:
        if action < 0 or action >= len(J) or not bool(valid[action]) or not np.isfinite(J[action]) or not np.isfinite(J[a_star]):
            return float("inf")
        return float(J[action] - J[a_star])

    budget_vs_full = int(action_index == full_action)
    teacher_regret = _teacher_regret_for(int(action_index))
    full_interface_teacher_regret = _teacher_regret_for(int(full_action))
    sparse_full_interface_teacher_regret = _teacher_regret_for(int(sparse_full_action))
    base_interface_teacher_regret = _teacher_regret_for(int(base_action))
    query_pairs = pairs.pairs[pairs.valid_mask] if inference_pairs is None else np.asarray(inference_pairs, dtype=np.int64).reshape(-1, 2)
    if query_pairs.size:
        queried_actions = np.unique(query_pairs.reshape(-1))
        queried_actions = queried_actions[(queried_actions >= 0) & (queried_actions < candidates.K) & valid[queried_actions]]
        query_action_count = int(queried_actions.size)
    else:
        query_action_count = int(valid.sum())
    query_atom_count = int(len(selected_atoms) if queried_atom_count is None else queried_atom_count)
    pair_conditioned_runtime = bool(cfg.get("runtime", {}).get("use_pair_conditioned_margins", cfg.get("model", {}).get("pair_conditioned", False)))
    if pair_conditioned_runtime and inference_pairs is not None:
        effective_query_count = float(query_atom_count * len(query_pairs))
    else:
        effective_query_count = float(query_atom_count * query_action_count)
    critical_sets = _critical_atom_sets(evidence_bank, teacher, pairs, cfg)
    selected_set = set(map(int, selected_atoms))
    selector_cfg = cfg.get("selector", {}) if isinstance(cfg, dict) else {}
    structural_set: set[int] = set()
    if bool(selector_cfg.get("decision_budget_excludes_structural_safety", False)):
        try:
            raw_hard = np.asarray(evidence_bank.hard_mask(), dtype=bool).reshape(-1)
        except Exception:
            raw_hard = np.zeros((evidence_bank.E,), dtype=bool)
        fam_ids = np.asarray([getattr(a, "family_id", 0) for a in evidence_bank.atoms], dtype=np.int64)
        structural = structural_safety_mask(
            raw_hard,
            fam_ids,
            np.asarray(evidence_bank.active_mask, dtype=bool),
            include_feasibility=bool(selector_cfg.get("structural_safety_include_feasibility", True)),
        )
        structural_set = set(map(int, np.flatnonzero(structural).tolist()))
    effective_selected_set = selected_set | structural_set
    topm_raw = qdiag.get("top_m_atoms", [])
    try:
        topm_set = set(map(int, np.asarray(topm_raw, dtype=np.int64).reshape(-1).tolist()))
    except Exception:
        topm_set = set()
    hard_denom = critical_sets["hard_decisive"] if critical_sets["hard_decisive"] else critical_sets["hard"]
    hard_recall = _recall(selected_set, hard_denom)
    suff = evidence_sufficiency(teacher_M_cert, M_B, pairs.pairs[pairs.valid_mask], pairs.weights[pairs.valid_mask])
    selector_ratio = np.nan
    if runtime_selected_atoms_for_oracle_value is not None and oracle_selected_atoms is not None and len(pairs.pairs):
        F_run = oracle_objective_value(runtime_selected_atoms_for_oracle_value, teacher.J_base, teacher.g_evid, pairs.pairs, pairs.margins, pairs.weights)
        F_oracle = oracle_objective_value(oracle_selected_atoms, teacher.J_base, teacher.g_evid, pairs.pairs, pairs.margins, pairs.weights)
        selector_ratio = float(F_run / (F_oracle + 1e-6))
    decisive_err = []
    for b in np.flatnonzero(valid):
        if b != a_star:
            decisive_err.append(abs(float(M_B[a_star, b] - teacher_M_cert[a_star, b])))

    pair_metrics: dict[str, float] = {}
    pair_masks = _pair_group_masks(evidence_bank, teacher, pairs, cfg)
    pair_arr = np.asarray(pairs.pairs, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray(pairs.weights, dtype=np.float32).reshape(-1) if pairs is not None else None
    # Paper-facing names use the selected certificate/tournament margin.  Dense and
    # base variants are included to diagnose whether failures come from the learned
    # interface itself or from the sparse selector/tournament stage.
    rename_groups = {
        "pair_sign_acc_all": "pair_sign_acc_all",
        "pair_sign_acc_winner_rival": "pair_sign_acc_winner_rival",
        "pair_sign_acc_hard": "pair_sign_acc_hard",
        "pair_sign_acc_interaction": "pair_sign_acc_interaction",
        "pair_sign_acc_near_tie": "pair_sign_acc_near_tie",
    }
    for raw_prefix, final_prefix in rename_groups.items():
        vals = _pair_group_metrics(final_prefix, teacher_M_cert, M_B, pair_arr, pair_masks.get(raw_prefix, np.zeros((len(pair_arr),), dtype=bool)), weights)
        pair_metrics.update(vals)
        dense_vals = _pair_group_metrics("dense_" + final_prefix, teacher_M, dense_M, pair_arr, pair_masks.get(raw_prefix, np.zeros((len(pair_arr),), dtype=bool)), weights)
        pair_metrics.update(dense_vals)
        base_vals = _pair_group_metrics("base_" + final_prefix, teacher_M, base_M, pair_arr, pair_masks.get(raw_prefix, np.zeros((len(pair_arr),), dtype=bool)), weights)
        pair_metrics.update(base_vals)

    for group_name in ("all", "winner_rival", "hard", "interaction", "near_tie"):
        sparse_key = f"pair_sign_acc_{group_name}_sign_acc"
        dense_key = f"dense_pair_sign_acc_{group_name}_sign_acc"
        base_key = f"base_pair_sign_acc_{group_name}_sign_acc"
        if sparse_key in pair_metrics:
            pair_metrics[f"pair_sign_acc_{group_name}"] = pair_metrics[sparse_key]
        if dense_key in pair_metrics:
            pair_metrics[f"dense_pair_sign_acc_{group_name}"] = pair_metrics[dense_key]
        if base_key in pair_metrics:
            pair_metrics[f"base_pair_sign_acc_{group_name}"] = pair_metrics[base_key]

    values = {
        "teacher_regret": teacher_regret,
        "full_interface_teacher_regret": full_interface_teacher_regret,
        "sparse_full_interface_teacher_regret": sparse_full_interface_teacher_regret,
        "base_interface_teacher_regret": base_interface_teacher_regret,
        "teacher_action_match": float(action_index == a_star),
        "full_interface_action_match": float(full_action == a_star),
        "budget_vs_full_match": float(budget_vs_full),
        "sparse_full_interface_action_match": float(sparse_full_action == a_star),
        "budget_vs_sparse_full_match": float(action_index == sparse_full_action),
        "preserved_margin_error": float(np.mean(decisive_err)) if decisive_err else 0.0,
        "evidence_sufficiency": suff,
        "decision_sufficiency": float(action_index == a_star),
        "selector_value_ratio": selector_ratio,
        "hard_evidence_recall": hard_recall,
        "selected_decisive_atom_recall": _recall(selected_set, critical_sets["decisive"]),
        "effective_selected_decisive_atom_recall": _recall(effective_selected_set, critical_sets["decisive"]),
        "selected_interaction_decisive_recall": _recall(selected_set, critical_sets["interaction_decisive"]),
        "selected_soft_interaction_decisive_recall": _recall(selected_set, critical_sets["soft_interaction_decisive"]),
        "effective_interaction_decisive_recall": _recall(effective_selected_set, critical_sets["interaction_decisive"]),
        "selected_hard_decisive_recall": _recall(selected_set, hard_denom),
        "structural_hard_decisive_coverage": _recall(structural_set, hard_denom),
        "effective_hard_decisive_recall": _recall(effective_selected_set, hard_denom),
        "proposal_decisive_atom_recall": _recall(topm_set, critical_sets["decisive"]) if topm_set else float("nan"),
        "proposal_interaction_decisive_recall": _recall(topm_set, critical_sets["interaction_decisive"]) if topm_set else float("nan"),
        "proposal_hard_recall": _recall(topm_set, hard_denom) if topm_set else float("nan"),
        "decisive_atom_count": float(len(critical_sets["decisive"])),
        "interaction_decisive_atom_count": float(len(critical_sets["interaction_decisive"])),
        "soft_interaction_decisive_atom_count": float(len(critical_sets["soft_interaction_decisive"])),
        "hard_decisive_atom_count": float(len(hard_denom)),
        "structural_safety_atom_count": float(len(structural_set)),
        "decision_budget_atom_count": float(len(selected_set)),
        "configured_decision_budget_atom_count": float(qdiag.get("configured_decision_budget_atom_count", max(len(selected_set), 1))),
        "fallback_would_trigger_rate": float(bool(qdiag.get("fallback_would_trigger", False))),
        "selected_action_safety_flag_rate": float(bool(qdiag.get("selected_action_safety_flag", False))),
        "avoidable_selected_action_safety_flag_rate": float(bool(qdiag.get("avoidable_selected_action_safety_flag", False))),
        "all_actions_safety_flagged_rate": float(bool(qdiag.get("all_actions_safety_flagged", False))),
        "all_flagged_risk_guard_applied_rate": float(bool(qdiag.get("all_flagged_risk_guard_applied", False))),
        "all_flagged_hard_risk_regret": float(qdiag.get("all_flagged_hard_risk_regret", 0.0)),
        "effective_query_count": float(qdiag.get("effective_query_count", effective_query_count)),
        "effective_query_atom_count": float(qdiag.get("selected_atom_count", query_atom_count)),
        "effective_query_action_count": float(qdiag.get("queried_action_count", query_action_count)),
        "effective_pair_count": float(qdiag.get("tournament_pair_count", len(query_pairs))),
        "teacher_pair_count": float(len(pairs.pairs)),
        "total_sparse_query_count": float(qdiag.get("total_sparse_query_count", qdiag.get("sparse_query_count", effective_query_count))),
        "stage_predict_ms": float(qdiag.get("stage_predict_ms", 0.0)),
        "stage_selector_ms": float(qdiag.get("stage_selector_ms", 0.0)),
        "stage_tournament_ms": float(qdiag.get("stage_tournament_ms", 0.0)),
        "stage_total_internal_ms": float(qdiag.get("stage_total_internal_ms", 0.0)),
        "decision_budget_excludes_structural_safety": float(qdiag.get("decision_budget_excludes_structural_safety", 0.0)),
        "structural_safety_include_feasibility": float(qdiag.get("structural_safety_include_feasibility", 1.0)),
        "structural_residual_enabled": float(qdiag.get("structural_residual_enabled", 0.0)),
        "structural_residual_weight": float(qdiag.get("structural_residual_weight", 0.0)),
        "pair_delta_calibration_enabled": float(qdiag.get("pair_delta_calibration_enabled", 0.0)),
        "pair_delta_selector_local_weight_mean": float(qdiag.get("pair_delta_selector_local_weight_mean", 0.0)),
        "pair_delta_selector_local_weight_p90": float(qdiag.get("pair_delta_selector_local_weight_p90", 0.0)),
        "pair_delta_selector_sign_disagreement_rate": float(qdiag.get("pair_delta_selector_sign_disagreement_rate", 0.0)),
        "pair_delta_tournament_local_weight_mean": float(qdiag.get("pair_delta_tournament_local_weight_mean", 0.0)),
        "pair_delta_tournament_sign_disagreement_rate": float(qdiag.get("pair_delta_tournament_sign_disagreement_rate", 0.0)),
        "pair_delta_selector_residual_trust_mean": float(qdiag.get("pair_delta_selector_residual_trust_mean", 0.0)),
        "pair_delta_selector_residual_trust_p90": float(qdiag.get("pair_delta_selector_residual_trust_p90", 0.0)),
        "pair_delta_selector_residual_sign_disagreement_rate": float(qdiag.get("pair_delta_selector_residual_sign_disagreement_rate", 0.0)),
        "pair_delta_tournament_residual_trust_mean": float(qdiag.get("pair_delta_tournament_residual_trust_mean", 0.0)),
        "pair_delta_tournament_residual_trust_p90": float(qdiag.get("pair_delta_tournament_residual_trust_p90", 0.0)),
        "pair_delta_tournament_residual_sign_disagreement_rate": float(qdiag.get("pair_delta_tournament_residual_sign_disagreement_rate", 0.0)),
        **{k: float(v) for k, v in qdiag.items() if str(k).startswith("pair_delta_") and isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) and np.isfinite(float(v))},
        "action_atom_query_count": float(qdiag.get("action_atom_query_count", query_atom_count * query_action_count)),
        "selector_pair_atom_query_count": float(qdiag.get("selector_pair_atom_query_count", 0.0)),
        "tournament_pair_atom_query_count": float(qdiag.get("tournament_pair_atom_query_count", 0.0)),
        "selected_certificate_query_count": float(qdiag.get("selected_certificate_query_count", effective_query_count)),
        "selector_action_rank_active": float(qdiag.get("selector_action_rank_active", 0.0)),
        "selector_flip_rank_active": float(qdiag.get("selector_flip_rank_active", 0.0)),
        "selector_lcb_active": float(qdiag.get("selector_lcb_active", 0.0)),
        **{k: float(v) for k, v in qdiag.items() if str(k).startswith("selector_") and isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) and np.isfinite(float(v))},
        **{
            k: float(v)
            for k, v in qdiag.items()
            if (
                str(k).startswith("evidence_certificate_")
                or str(k).startswith("residual_flip_")
                or str(k).startswith("dual_certificate_")
                or str(k).startswith("pair_action_anchor_")
            )
            and isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_))
            and np.isfinite(float(v))
        },
        **pair_metrics,
    }
    return BDSEMetricResult(values=values, details={"full_action": full_action, "sparse_full_action": sparse_full_action, "base_action": base_action, "a_star": a_star, "selected_atoms": selected_atoms, "query_action_count": query_action_count})


def aggregate_metric_results(results: list[BDSEMetricResult]) -> dict[str, float]:
    keys = sorted({k for r in results for k in r.values})
    out = {}
    for k in keys:
        vals = [r.values[k] for r in results if k in r.values and np.isfinite(r.values[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
