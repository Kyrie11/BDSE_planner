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
            metrics["candidate_log_ade_teacher"] = float(np.linalg.norm(tr[:n, :2] - gt[:n, :2], axis=1).mean()) if n else float("nan")
    return metrics


def quality_decision(metrics: dict[str, Any], cfg: dict[str, Any]) -> QualityDecision:
    train_qcfg = cfg.get("training", {}).get("quality_filter", {}) or {}
    # During training, an enabled training.quality_filter should override the
    # materialization-time defaults.  During preprocessing, the training filter is
    # normally disabled and we fall back to preprocess.quality_filter.
    qcfg = train_qcfg if bool(train_qcfg.get("enabled", False)) else (cfg.get("preprocess", {}).get("quality_filter", {}) or {})
    reasons: list[str] = []
    if bool(qcfg.get("require_safe_candidate", False)) and not bool(metrics.get("safe_candidate_exists", False)):
        reasons.append("no_safe_candidate")
    min_valid = qcfg.get("min_valid_candidates", None)
    if min_valid is not None and int(metrics.get("valid_candidate_count", 0)) < int(min_valid):
        reasons.append("too_few_valid_candidates")
    max_ade = qcfg.get("max_candidate_log_ade_min", None)
    if max_ade is not None:
        val = float(metrics.get("candidate_log_ade_min", float("nan")))
        if np.isfinite(val) and val > float(max_ade):
            reasons.append("poor_candidate_log_ade")
    max_route_p95 = qcfg.get("max_logged_route_p95", None)
    if max_route_p95 is not None:
        val = float(metrics.get("logged_ego_route_dist_p95", float("nan")))
        if np.isfinite(val) and val > float(max_route_p95):
            reasons.append("logged_ego_far_from_route")
    if bool(qcfg.get("exclude_teacher_hard", False)) and bool(metrics.get("teacher_hard_violation", False)):
        reasons.append("teacher_hard_violation")
    return QualityDecision(keep=not reasons, reasons=tuple(reasons), metrics=metrics)
