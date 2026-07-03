from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, LabelOnlyFuture, Sample
from bdse.utils import nearest_polyline_distance


@dataclass(frozen=True, slots=True)
class QualityDecision:
    keep: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float | bool | int]


def candidate_logged_ego_metrics(candidates: CandidateBank, label_future: LabelOnlyFuture | None) -> dict[str, float]:
    """Runtime-candidate coverage against the logged ego future.

    These metrics are diagnostics/filters only.  They must never be used by the
    runtime planner or candidate generator because logged future is label-only.
    """
    out = {
        "candidate_log_ade_min": float("nan"),
        "candidate_log_fde_min": float("nan"),
        "candidate_log_ade_teacher": float("nan"),
    }
    if label_future is None or label_future.logged_ego is None or np.asarray(label_future.logged_ego).size == 0:
        return out
    gt = np.asarray(label_future.logged_ego, dtype=np.float32)
    trajs = np.asarray(candidates.trajectories, dtype=np.float32)
    valid = np.asarray(candidates.valid_mask, dtype=bool)
    if gt.ndim != 2 or trajs.ndim != 3 or not valid.any():
        return out
    n = min(gt.shape[0], trajs.shape[1])
    if n <= 0:
        return out
    diff = trajs[:, :n, :2] - gt[None, :n, :2]
    ade = np.linalg.norm(diff, axis=-1).mean(axis=1)
    fde = np.linalg.norm(diff[:, -1, :], axis=-1)
    ade = np.where(valid, ade, np.inf)
    fde = np.where(valid, fde, np.inf)
    if np.isfinite(ade).any():
        out["candidate_log_ade_min"] = float(np.nanmin(ade))
    if np.isfinite(fde).any():
        out["candidate_log_fde_min"] = float(np.nanmin(fde))
    return out


def logged_route_metrics(sample: Sample) -> dict[str, float]:
    out = {
        "logged_ego_route_dist_mean": float("nan"),
        "logged_ego_route_dist_p95": float("nan"),
        "logged_ego_route_dist_final": float("nan"),
    }
    if sample.label_future is None or sample.label_future.logged_ego is None:
        return out
    route = np.asarray((sample.runtime.map_features or {}).get("route_centerline", []), dtype=np.float32).reshape(-1, 2)
    gt = np.asarray(sample.label_future.logged_ego, dtype=np.float32)
    if len(route) < 2 or gt.ndim != 2 or gt.shape[0] == 0:
        return out
    d = nearest_polyline_distance(gt[:, :2], route)
    if d.size:
        out["logged_ego_route_dist_mean"] = float(np.mean(d))
        out["logged_ego_route_dist_p95"] = float(np.percentile(d, 95))
        out["logged_ego_route_dist_final"] = float(d[-1])
    return out


def summarize_sample_quality(sample: Sample) -> dict[str, float | bool | int]:
    metrics: dict[str, float | bool | int] = {}
    metrics.update(candidate_logged_ego_metrics(sample.candidates, sample.label_future))
    metrics.update(logged_route_metrics(sample))
    valid = np.asarray(sample.candidates.valid_mask, dtype=bool)
    hard = np.asarray(sample.teacher.hard_violation_mask, dtype=bool) if sample.teacher is not None else np.zeros_like(valid)
    metrics["valid_candidate_count"] = int(valid.sum())
    metrics["safe_candidate_count"] = int((valid & ~hard).sum()) if hard.shape == valid.shape else 0
    metrics["safe_candidate_exists"] = bool(metrics["safe_candidate_count"] > 0)
    metrics["teacher_hard_violation"] = bool(hard[int(sample.teacher.a_star)]) if sample.teacher is not None and hard.size else False
    if sample.teacher is not None:
        a = int(sample.teacher.a_star)
        # Compute teacher-vs-log ADE separately; this catches candidate banks where
        # a logged-like candidate exists but the teacher selects a hard/route artifact.
        if sample.label_future is not None and np.asarray(sample.label_future.logged_ego).size and 0 <= a < sample.candidates.K:
            gt = np.asarray(sample.label_future.logged_ego, dtype=np.float32)
            tr = np.asarray(sample.candidates.trajectories[a], dtype=np.float32)
            n = min(len(gt), len(tr))
            if n:
                metrics["candidate_log_ade_teacher"] = float(np.linalg.norm(tr[:n, :2] - gt[:n, :2], axis=1).mean())
                nearest = float(metrics.get("candidate_log_ade_min", float("nan")))
                if np.isfinite(nearest):
                    metrics["teacher_to_nearest_log_ade_gap"] = float(metrics["candidate_log_ade_teacher"] - nearest)
            else:
                metrics["candidate_log_ade_teacher"] = float("nan")
                metrics["teacher_to_nearest_log_ade_gap"] = float("nan")
    return metrics


def runtime_interface_metrics(sample: Sample, cfg: dict[str, Any]) -> dict[str, float | bool | int]:
    """Offline materialization-only BDSE interface diagnostics.

    These metrics deliberately use teacher labels and must therefore remain a
    preprocessing / dataset-selection signal.  They are not runtime planner
    inputs.  The goal is to avoid materializing paper-support caches whose
    finite candidate/evidence interface cannot reproduce the teacher even with
    oracle atom costs under the configured runtime pair screen and budget.
    """
    out: dict[str, float | bool | int] = {}
    if sample.teacher is None or sample.pairs is None or sample.evidence_bank is None:
        return out
    try:
        from bdse.planner.fallback import runtime_safety_flags_from_runtime
        from bdse.planner.selector import oracle_greedy_selector, oracle_objective_value, runtime_greedy_selector
        from bdse.planner.tournament import run_tournament

        J0 = np.asarray(sample.teacher.J_base, dtype=np.float32)
        g = np.asarray(sample.teacher.g_evid, dtype=np.float32)
        valid = np.asarray(sample.candidates.valid_mask, dtype=bool)
        flags = runtime_safety_flags_from_runtime(sample.runtime, sample.candidates, cfg)
        sel_cfg = cfg.get("selector", {}) or {}
        budget = float(cfg.get("evidence", {}).get("budget", 16))
        runtime_sel = runtime_greedy_selector(
            J0,
            g,
            sample.evidence_bank.budget_costs(),
            valid,
            flags,
            budget,
            L_infer=int(cfg.get("tournament", {}).get("L_infer", 16)),
            gamma_max=float(sel_cfg.get("gamma_max_default", 100.0)),
            eta_pred=float(sel_cfg.get("eta_pred", 1.0)),
            lambda_near=float(sel_cfg.get("lambda_near", 1.0)),
            lambda_safety=float(sel_cfg.get("lambda_safety", 2.0)),
            atom_active_mask=sample.evidence_bank.active_mask,
            bidirectional_pairs=bool(sel_cfg.get("bidirectional_pairs", True)),
            reverse_pair_weight=float(sel_cfg.get("reverse_pair_weight", 1.0)),
            pair_cap_multiplier=float(sel_cfg.get("runtime_pair_cap_multiplier", 1.0)),
        )
        oracle_sel = oracle_greedy_selector(
            sample.teacher.J_base,
            sample.teacher.g_evid,
            sample.pairs.pairs,
            sample.pairs.margins,
            sample.pairs.weights,
            sample.evidence_bank.budget_costs(),
            budget,
            sample.evidence_bank.active_mask,
        )
        tour = run_tournament(J0, g, runtime_sel.selected, valid, flags, cfg)
        F_run = oracle_objective_value(
            runtime_sel.selected,
            sample.teacher.J_base,
            sample.teacher.g_evid,
            sample.pairs.pairs,
            sample.pairs.margins,
            sample.pairs.weights,
        )
        F_oracle = oracle_objective_value(
            oracle_sel.selected,
            sample.teacher.J_base,
            sample.teacher.g_evid,
            sample.pairs.pairs,
            sample.pairs.margins,
            sample.pairs.weights,
        )
        ratio = float(F_run / (F_oracle + 1e-6)) if np.isfinite(F_oracle) else float("nan")
        a_star = int(sample.teacher.a_star)
        action = int(tour.action_index)
        out.update(
            {
                "runtime_decision_sufficiency": bool(action == a_star),
                "runtime_teacher_action_match": bool(action == a_star),
                "runtime_teacher_regret": float(sample.teacher.J_T[action] - sample.teacher.J_T[a_star]) if 0 <= action < len(sample.teacher.J_T) and valid[action] else float("inf"),
                "selector_value_ratio": ratio,
                "runtime_selector_objective": float(F_run),
                "oracle_selector_objective": float(F_oracle),
                "runtime_selected_atom_count": int(len(runtime_sel.selected)),
                "oracle_selected_atom_count": int(len(oracle_sel.selected)),
                "runtime_pair_count": int(len(runtime_sel.pair_indices)),
            }
        )
    except Exception:
        # Quality metrics must never crash preprocessing; a missing metric can be
        # rejected by reject_missing_quality_metrics when a strict materialization
        # gate is requested.
        out.update(
            {
                "runtime_decision_sufficiency": False,
                "runtime_teacher_action_match": False,
                "selector_value_ratio": float("nan"),
            }
        )
    return out


def quality_decision(metrics: dict[str, Any], cfg: dict[str, Any]) -> QualityDecision:
    train_qcfg = cfg.get("training", {}).get("quality_filter", {}) or {}
    # During training, an enabled training.quality_filter should override the
    # materialization-time defaults.  During preprocessing, the training filter is
    # normally disabled and we fall back to preprocess.quality_filter.
    qcfg = train_qcfg if bool(train_qcfg.get("enabled", False)) else (cfg.get("preprocess", {}).get("quality_filter", {}) or {})
    reasons: list[str] = []
    reject_missing = bool(qcfg.get("reject_missing_quality_metrics", False))

    def _finite_metric(name: str) -> float:
        try:
            return float(metrics.get(name, float("nan")))
        except Exception:
            return float("nan")

    def _check_upper(name: str, threshold: Any, reason: str) -> None:
        if threshold is None:
            return
        val = _finite_metric(name)
        if (not np.isfinite(val) and reject_missing) or (np.isfinite(val) and val > float(threshold)):
            reasons.append(reason)

    def _check_lower(name: str, threshold: Any, reason: str) -> None:
        if threshold is None:
            return
        val = _finite_metric(name)
        if (not np.isfinite(val) and reject_missing) or (np.isfinite(val) and val < float(threshold)):
            reasons.append(reason)

    if bool(qcfg.get("require_safe_candidate", False)) and not bool(metrics.get("safe_candidate_exists", False)):
        reasons.append("no_safe_candidate")
    min_valid = qcfg.get("min_valid_candidates", None)
    if min_valid is not None and int(metrics.get("valid_candidate_count", 0)) < int(min_valid):
        reasons.append("too_few_valid_candidates")
    _check_upper("candidate_log_ade_min", qcfg.get("max_candidate_log_ade_min", None), "poor_candidate_log_ade")
    _check_upper("candidate_log_ade_teacher", qcfg.get("max_candidate_log_ade_teacher", None), "poor_teacher_log_ade")
    _check_upper("teacher_to_nearest_log_ade_gap", qcfg.get("max_teacher_to_nearest_log_ade_gap", None), "teacher_far_from_log_nearest")
    _check_upper("logged_ego_route_dist_p95", qcfg.get("max_logged_route_p95", None), "logged_ego_far_from_route")
    _check_lower("selector_value_ratio", qcfg.get("min_selector_value_ratio", None), "low_selector_value_ratio")
    _check_upper("runtime_teacher_regret", qcfg.get("max_runtime_teacher_regret", None), "high_runtime_teacher_regret")
    if bool(qcfg.get("require_runtime_decision_sufficiency", False)) and not bool(metrics.get("runtime_decision_sufficiency", False)):
        reasons.append("runtime_decision_insufficient")
    if bool(qcfg.get("exclude_teacher_hard", False)) and bool(metrics.get("teacher_hard_violation", False)):
        reasons.append("teacher_hard_violation")
    return QualityDecision(keep=not reasons, reasons=tuple(reasons), metrics=metrics)
