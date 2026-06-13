from __future__ import annotations

import argparse
import json
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, PreprocessedBDSEDataset
from bdse.metrics.bdse_metrics import aggregate_metric_results, compute_bdse_diagnostics
from bdse.planner.fallback import runtime_safety_flags_from_runtime
from bdse.planner.evidence_atoms import hard_event_matrix, atom_weight_scale_cap
from bdse.planner.evidence_queries import certificate_family
from bdse.utils import nearest_polyline_distance
from bdse.planner.selector import oracle_greedy_selector, runtime_greedy_selector
from bdse.planner.tournament import run_tournament


def _finite_mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if np.isfinite(x)]
    return float(np.mean(vals)) if vals else float("nan")


def _finite_pct(xs: list[float], q: float) -> float:
    vals = [float(x) for x in xs if np.isfinite(x)]
    return float(np.percentile(vals, q)) if vals else float("nan")


def _log_nearest(s) -> tuple[int, np.ndarray]:
    if s.label_future is None or s.label_future.logged_ego.size == 0:
        costs = np.full(s.candidates.K, np.inf, dtype=np.float32)
        return -1, costs
    costs = np.linalg.norm(s.candidates.trajectories[:, :, :2] - s.label_future.logged_ego[None, :, :2], axis=-1).mean(axis=1)
    costs = np.where(s.candidates.valid_mask, costs, np.inf)
    return int(np.argmin(costs)), costs


def _route_stats(route: np.ndarray) -> dict[str, float]:
    arr = np.asarray(route, dtype=np.float32).reshape(-1, 2)
    if len(arr) < 2:
        return {
            "route_length_m": 0.0,
            "route_max_segment_m": 0.0,
            "route_jump_count": 0.0,
            "route_backtrack_frac": 0.0,
        }
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    dx = np.diff(arr[:, 0])
    return {
        "route_length_m": float(seg.sum()),
        "route_max_segment_m": float(seg.max(initial=0.0)),
        "route_jump_count": float((seg > 10.0).sum()),
        "route_backtrack_frac": float((dx < -0.5).mean()) if dx.size else 0.0,
    }


def _dynamic_feasible_rate(s, valid: np.ndarray) -> float:
    vals = []
    for i, ok in enumerate(valid.astype(bool).tolist()):
        if not ok:
            continue
        flags = s.candidates.dynamic_flags[i] if i < len(s.candidates.dynamic_flags) else {}
        vals.append(float(bool(flags.get("dynamically_feasible", True))))
    return float(np.mean(vals)) if vals else float("nan")


def _hard_event_type_counts(s, hard_events: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    out = Counter()
    for ei, atom in enumerate(s.evidence_bank.atoms):
        if atom.is_hard and hard_events[ei, valid].any():
            out[atom.type] += 1
    return {f"hard_event_{k}_atom_count": float(v) for k, v in out.items()}


def _hard_event_type_flags(s, hard_events: np.ndarray, candidate_idx: int, prefix: str) -> dict[str, float]:
    out = Counter()
    if candidate_idx < 0 or candidate_idx >= s.candidates.K:
        return {}
    for ei, atom in enumerate(s.evidence_bank.atoms):
        if atom.is_hard and bool(hard_events[ei, candidate_idx]):
            out[atom.type] += 1
    return {f"{prefix}_{k}": float(v) for k, v in out.items()}


def _certificate_family_counts(s) -> dict[str, float]:
    """Count paper-aligned certificate families with legacy aliases preserved.

    Older caches may store families such as interaction/rule_map/kinematic, while
    current atoms use reachability_interaction/feasibility/dynamic_regularity.
    Diagnostics should report the semantic family rather than returning misleading
    zeros for the legacy names.
    """
    counts = Counter(certificate_family(a.type, a.family) for a in s.evidence_bank.atoms)
    out = {
        "feasibility_atom_count": float(counts.get("feasibility", 0)),
        "reachability_interaction_atom_count": float(counts.get("reachability_interaction", 0)),
        "precedence_atom_count": float(counts.get("precedence", 0)),
        "dynamic_regularity_atom_count": float(counts.get("dynamic_regularity", 0)),
        "decision_boundary_atom_count": float(counts.get("decision_boundary", 0)),
    }
    # Backward-compatible aliases used by existing tables/scripts.
    out["rule_map_atom_count"] = out["feasibility_atom_count"]
    out["interaction_atom_count"] = out["reachability_interaction_atom_count"]
    out["kinematic_atom_count"] = out["dynamic_regularity_atom_count"]
    return out


def _sample_identity(s) -> dict[str, Any]:
    return {
        "scenario_token": str(getattr(s, "scenario_token", "")),
        "timestamp_us": int(getattr(s, "timestamp_us", 0) or 0),
    }


def _identity_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [f"{x.get('scenario_token','')}@{int(x.get('timestamp_us', 0) or 0)}" for x in items]
    h = hashlib.sha1("\n".join(sorted(keys)).encode("utf-8")).hexdigest() if keys else ""
    return {
        "identity_count": int(len(keys)),
        "identity_unique_count": int(len(set(keys))),
        "identity_duplicate_count": int(len(keys) - len(set(keys))),
        "identity_sha1": h,
        "identity_first5": keys[:5],
    }


def _atom_saturation_metrics(s, valid: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    if s.teacher is None or s.teacher.g_evid.size == 0 or not valid.any():
        return out
    total_active = 0
    total_sat = 0
    by_type: dict[str, list[float]] = defaultdict(list)
    g = np.asarray(s.teacher.g_evid, dtype=np.float32)
    for ei, atom in enumerate(s.evidence_bank.atoms):
        if ei >= g.shape[0]:
            continue
        weight, _, cap = atom_weight_scale_cap(atom, cfg)
        g_cap = float(weight) * float(cap)
        vals = g[ei, valid]
        active = vals > 1e-6
        if not np.any(active) or g_cap <= 0.0:
            continue
        sat = vals[active] >= 0.999 * g_cap
        total_active += int(active.sum())
        total_sat += int(sat.sum())
        by_type[atom.type].append(float(sat.mean()))
    out["atom_saturation_active_rate"] = float(total_sat / max(total_active, 1)) if total_active else 0.0
    for typ, rates in by_type.items():
        out[f"atom_saturation_type_{typ}"] = float(np.mean(rates))
    return out


def _safe_absence_type_flags(s, hard_events: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """For no-safe scenes, report which hard types cover all valid candidates."""
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return {}
    hard_mask = np.zeros((valid_idx.size,), dtype=bool)
    out: dict[str, float] = {}
    for typ in sorted({a.type for a in s.evidence_bank.atoms if a.is_hard}):
        rows = [ei for ei, atom in enumerate(s.evidence_bank.atoms) if atom.is_hard and atom.type == typ]
        if not rows:
            continue
        type_mask = hard_events[np.asarray(rows), :][:, valid_idx].any(axis=0)
        out[f"safe_absent_all_candidates_{typ}"] = float(type_mask.all())
        out[f"safe_absent_any_candidate_{typ}"] = float(type_mask.any())
        hard_mask |= type_mask
    out["safe_absent_all_candidates_any_hard"] = float(hard_mask.all())
    return out


def _teacher_sample_metrics(s, cfg: dict[str, Any], recompute_hard_events: bool = False) -> dict[str, float]:
    log_nearest, log_costs = _log_nearest(s)
    valid = s.candidates.valid_mask.astype(bool)
    a_star = int(s.teacher.a_star)
    route = np.asarray(s.runtime.map_features.get("route_centerline", []), dtype=np.float32).reshape(-1, 2)
    map_valid = float(bool(s.runtime.map_features.get("map_valid", False)))
    route_source = str(s.runtime.map_features.get("route_source", ""))
    route_fallback = float(route_source.startswith("fallback"))
    hard_events = None
    hard_valid = s.teacher.hard_violation_mask[valid]
    safe_mask = valid & (~s.teacher.hard_violation_mask.astype(bool))
    if recompute_hard_events:
        hard_events = hard_event_matrix(s.evidence_bank.atoms, s.candidates, s.runtime, s.label_future, cfg)
        hard_candidate_mask_from_atoms = hard_events[s.evidence_bank.hard_mask()][:, valid].any(axis=0) if valid.any() and s.evidence_bank.hard_mask().any() else np.zeros((int(valid.sum()),), dtype=bool)
    else:
        # The exact per-type hard-event matrix is expensive to reconstruct from
        # trajectories and agents.  For fast diagnostics, use the already saved
        # teacher hard-violation label for aggregate rates, and omit only the
        # per-hard-type decomposition metrics.
        hard_candidate_mask_from_atoms = hard_valid
    teacher_vs_log = float(log_nearest >= 0 and log_nearest != a_star)
    teacher_regret_to_log = float(s.teacher.J_T[log_nearest] - s.teacher.J_T[a_star]) if log_nearest >= 0 and np.isfinite(s.teacher.J_T[log_nearest]) else float("nan")
    hard_counts = Counter(a.type for a in s.evidence_bank.atoms if a.is_hard)
    route_diag = dict(s.runtime.map_features.get("route_quality", {}) or {})
    route_diag.update(_route_stats(route))
    if s.label_future is not None and len(route) >= 2 and s.label_future.logged_ego.size:
        logged_route_dist = nearest_polyline_distance(s.label_future.logged_ego[:, :2], route)
        log_route_dist_mean = float(np.mean(logged_route_dist))
        log_route_dist_p95 = float(np.percentile(logged_route_dist, 95))
        log_route_dist_final = float(logged_route_dist[-1])
    else:
        log_route_dist_mean = log_route_dist_p95 = log_route_dist_final = float("nan")
    label_meta = getattr(s.label_future, "metadata", {}) if s.label_future is not None else {}
    logged_mask = np.asarray(label_meta.get("agent_future_logged_mask", []), dtype=bool)
    cv_mask = np.asarray(label_meta.get("agent_future_cv_fallback_mask", []), dtype=bool)
    selected_agent_count = int(label_meta.get("selected_agent_count", int(s.runtime.agent_valid.sum())))
    teacher_diag = dict(getattr(s.teacher, "diagnostics", {}) or {})
    quality_reasons = {str(x) for x in teacher_diag.get("quality_reasons", []) or []}
    known_quality_reasons = [
        "no_safe_candidate",
        "too_few_valid_candidates",
        "poor_candidate_log_ade",
        "poor_teacher_log_ade",
        "teacher_far_from_log_nearest",
        "logged_ego_far_from_route",
        "teacher_hard_violation",
    ]
    out = {
        "teacher_vs_log_disagreement": teacher_vs_log,
        "teacher_regret_to_log_nearest": teacher_regret_to_log,
        "safe_candidate_exists": float(safe_mask.any()),
        "safe_candidate_count": float(safe_mask.sum()),
        "teacher_hard_violation": float(s.teacher.hard_violation_mask[a_star]),
        "teacher_hard_when_safe_exists": float(bool(s.teacher.hard_violation_mask[a_star]) and bool(safe_mask.any())),
        "log_nearest_hard_violation": float(log_nearest >= 0 and bool(s.teacher.hard_violation_mask[log_nearest])),
        "candidate_hard_violation_rate": float(hard_valid.mean()) if hard_valid.size else float("nan"),
        "candidate_hard_violation_rate_from_atoms": float(hard_candidate_mask_from_atoms.mean()) if hard_candidate_mask_from_atoms.size else float("nan"),
        "valid_candidate_count": float(valid.sum()),
        "candidate_dynamic_feasible_rate": _dynamic_feasible_rate(s, valid),
        "candidate_log_ade_min": float(np.nanmin(log_costs)) if log_nearest >= 0 else float("nan"),
        "candidate_log_ade_teacher": float(log_costs[a_star]) if log_nearest >= 0 else float("nan"),
        "map_valid_rate": map_valid,
        "route_fallback_rate": route_fallback,
        "route_centerline_points": float(len(route)),
        "route_length_m": float(route_diag.get("route_length_m", float("nan"))),
        "route_max_segment_m": float(route_diag.get("route_max_segment_m", float("nan"))),
        "route_jump_count": float(route_diag.get("route_jump_count", float("nan"))),
        "route_backtrack_frac": float(route_diag.get("route_backtrack_frac", float("nan"))),
        "logged_ego_route_dist_mean": log_route_dist_mean,
        "logged_ego_route_dist_p95": log_route_dist_p95,
        "logged_ego_route_dist_final": log_route_dist_final,
        "route_ids_nonempty": float(len(s.runtime.route_roadblock_ids) > 0),
        "mission_goal_nonempty": float(s.runtime.mission_goal is not None and np.asarray(s.runtime.mission_goal).size > 0),
        "traffic_light_nonempty": float(len(s.runtime.traffic_lights) > 0),
        "red_light_atom_count": float(sum(1 for a in s.evidence_bank.atoms if a.type == "red_light")),
        "drivable_polygon_count": float(len(s.runtime.map_features.get("drivable_polygons", []))),
        "stop_line_count": float(len(s.runtime.map_features.get("stop_lines", []))),
        "agent_slot_valid_rate": float(s.label_future.agent_valid.mean()) if s.label_future is not None else float("nan"),
        "selected_agent_count": float(selected_agent_count),
        "agent_future_logged_rate": float(logged_mask.sum() / max(selected_agent_count, 1)) if logged_mask.size else float("nan"),
        "agent_future_cv_fallback_rate": float(cv_mask.sum() / max(selected_agent_count, 1)) if cv_mask.size else float("nan"),
        "atom_count": float(len(s.evidence_bank.atoms)),
        "pair_count": float(0 if s.pairs is None else len(s.pairs.pairs)),
        "pair_nonempty": float(s.pairs is not None and len(s.pairs.pairs) > 0),
        **_certificate_family_counts(s),
        "hard_atom_count": float(sum(hard_counts.values())),
        "teacher_J_base_min": float(np.nanmin(s.teacher.J_base[valid])) if valid.any() else float("nan"),
        "teacher_J_evid_min": float(np.nanmin(s.teacher.J_evid[valid])) if valid.any() else float("nan"),
        "teacher_J_T_min": float(s.teacher.J_T[a_star]),
        "teacher_evidence_share_at_star": float(s.teacher.J_evid[a_star] / max(abs(s.teacher.J_T[a_star]), 1e-6)),
        "partition_max_abs_error": float(np.nanmax(np.abs(s.teacher.J_T[valid] - (s.teacher.J_base[valid] + s.teacher.J_evid[valid])))) if valid.any() else float("nan"),
        "evidence_sum_max_abs_error": float(np.nanmax(np.abs(s.teacher.J_evid[valid] - s.teacher.g_evid[:, valid].sum(axis=0)))) if valid.any() else float("nan"),
        "quality_keep_rate": float(bool(teacher_diag.get("quality_keep", True))),
        "quality_candidate_log_ade_min": float(teacher_diag.get("quality_candidate_log_ade_min", log_costs[log_nearest] if log_nearest >= 0 else float("nan"))),
        "quality_candidate_log_ade_teacher": float(teacher_diag.get("quality_candidate_log_ade_teacher", log_costs[a_star] if log_nearest >= 0 else float("nan"))),
        "quality_teacher_to_nearest_log_ade_gap": float(teacher_diag.get("quality_teacher_to_nearest_log_ade_gap", (log_costs[a_star] - log_costs[log_nearest]) if log_nearest >= 0 else float("nan"))),
    }
    for reason in known_quality_reasons:
        out[f"quality_reject_{reason}_rate"] = float(reason in quality_reasons)
    if hard_events is not None:
        out.update(_hard_event_type_counts(s, hard_events, valid))
        out.update(_hard_event_type_flags(s, hard_events, a_star, "teacher_hard_type"))
        out.update(_hard_event_type_flags(s, hard_events, log_nearest, "log_nearest_hard_type"))
        if not bool(safe_mask.any()):
            out.update(_safe_absence_type_flags(s, hard_events, valid))
    out.update(_atom_saturation_metrics(s, valid, cfg))
    return out


def _update_lists(acc: dict[str, list[float]], vals: dict[str, float]) -> None:
    for k, v in vals.items():
        acc[k].append(float(v))


def _summarize_lists(acc: dict[str, list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, vals in sorted(acc.items()):
        out[k] = _finite_mean(vals)
        if (
            k.endswith("_count")
            or k.endswith("_rate")
            or "ade" in k
            or "route" in k
            or k in {"teacher_J_T_min", "teacher_J_base_min", "teacher_J_evid_min", "teacher_regret_to_log_nearest"}
        ):
            out[k + "_p50"] = _finite_pct(vals, 50)
            out[k + "_p90"] = _finite_pct(vals, 90)
    return out


def _diagnose_one_sample(dataset: Any, i: int, cfg: dict[str, Any], budgets: list[float], recompute_hard_events: bool):
    s = dataset[i]
    if s.teacher is None or s.label_future is None or s.pairs is None:
        return None
    teacher_metrics = _teacher_sample_metrics(s, cfg, recompute_hard_events=recompute_hard_events)
    J0 = s.teacher.J_base.copy()
    g = s.teacher.g_evid.copy()
    flags = runtime_safety_flags_from_runtime(s.runtime, s.candidates, cfg)
    runtime_sel = runtime_greedy_selector(J0, g, s.evidence_bank.budget_costs(), s.candidates.valid_mask, flags, float(cfg["evidence"]["budget"]), atom_active_mask=s.evidence_bank.active_mask)
    oracle_sel = oracle_greedy_selector(s.teacher.J_base, s.teacher.g_evid, s.pairs.pairs, s.pairs.margins, s.pairs.weights, s.evidence_bank.budget_costs(), float(cfg["evidence"]["budget"]), s.evidence_bank.active_mask)
    tour = run_tournament(J0, g, runtime_sel.selected, s.candidates.valid_mask, flags, cfg)
    bdse_result = compute_bdse_diagnostics(
        s.candidates,
        s.evidence_bank,
        s.teacher,
        s.pairs,
        J0,
        g,
        runtime_sel.selected,
        tour.action_index,
        runtime_sel.selected,
        oracle_sel.selected,
        cfg,
        inference_pairs=runtime_sel.pair_indices,
    )
    budget_metrics: dict[str, float] = {}
    hard_mask = s.evidence_bank.hard_mask() & s.evidence_bank.active_mask
    hard = set(np.flatnonzero(hard_mask).astype(int).tolist())
    decisive_hard: set[int] = set()
    if len(s.pairs.pairs):
        for a_pair, b_pair in np.asarray(s.pairs.pairs[s.pairs.valid_mask], dtype=np.int64):
            delta = np.asarray(s.teacher.g_evid[:, b_pair] - s.teacher.g_evid[:, a_pair], dtype=np.float32)
            for ei in np.flatnonzero(hard_mask & (delta > 1e-6)):
                decisive_hard.add(int(ei))
    hard_denom = decisive_hard if decisive_hard else hard
    for B in budgets:
        sel = oracle_greedy_selector(s.teacher.J_base, s.teacher.g_evid, s.pairs.pairs, s.pairs.margins, s.pairs.weights, s.evidence_bank.budget_costs(), float(B), s.evidence_bank.active_mask)
        t = run_tournament(J0, g, sel.selected, s.candidates.valid_mask, flags, cfg)
        selected_set = set(sel.selected)
        budget_metrics[f"B{int(B)}_decision_sufficiency"] = float(t.action_index == s.teacher.a_star)
        budget_metrics[f"B{int(B)}_hard_recall"] = float(len(hard & selected_set) / max(len(hard), 1))
        budget_metrics[f"B{int(B)}_decisive_hard_recall"] = float(len(hard_denom & selected_set) / max(len(hard_denom), 1))
    return teacher_metrics, bdse_result, budget_metrics, _sample_identity(s)


def run_diagnostics(
    cfg: dict[str, Any],
    split: str,
    folders: list[str] | None,
    max_files: int | None,
    max_scenarios: int | None,
    scenario_stride: int | None,
    preprocessed_dir: str | None = None,
    num_workers: int = 1,
    recompute_hard_events: bool | None = None,
) -> dict[str, Any]:
    if preprocessed_dir:
        dataset = PreprocessedBDSEDataset(preprocessed_dir, split=split, max_scenarios=max_scenarios)
    else:
        dataset = NuPlanBDSEDataset(cfg, split=split, folders=folders, max_files=max_files, max_scenarios=max_scenarios, stride=scenario_stride, use_devkit=True)
    teacher_acc: dict[str, list[float]] = defaultdict(list)
    bdse_results = []
    budget_acc: dict[str, list[float]] = defaultdict(list)
    identity_items: list[dict[str, Any]] = []
    budgets = [float(x) for x in cfg.get("diagnostics", {}).get("budget_sweep", [4, 8, 16, 24, 32])]
    if recompute_hard_events is None:
        recompute_hard_events = bool(cfg.get("diagnostics", {}).get("recompute_hard_events", False))
    workers = max(1, int(num_workers or 1))
    # Threaded diagnostics is intended for preprocessed .npz caches.  Raw nuPlan
    # scenario objects may share DB/map handles, so keep those sequential.
    workers = workers if preprocessed_dir else 1
    skipped_missing = 0

    def consume(res) -> None:
        nonlocal skipped_missing
        if res is None:
            skipped_missing += 1
            return
        teacher_metrics, bdse_result, budget_metrics, identity = res
        _update_lists(teacher_acc, teacher_metrics)
        bdse_results.append(bdse_result)
        _update_lists(budget_acc, budget_metrics)
        identity_items.append(identity)

    n = len(dataset)
    if workers <= 1:
        for i in tqdm(range(n), total=n, desc=f"diagnostics:{split}"):
            consume(_diagnose_one_sample(dataset, i, cfg, budgets, bool(recompute_hard_events)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_diagnose_one_sample, dataset, i, cfg, budgets, bool(recompute_hard_events)): i for i in range(n)}
            for fut in tqdm(as_completed(futures), total=n, desc=f"diagnostics:{split}"):
                consume(fut.result())
    return {
        "num_loaded": int(len(dataset)),
        "num_samples": int(len(next(iter(teacher_acc.values()))) if teacher_acc else 0),
        "num_skipped_missing_labels": int(skipped_missing),
        "recomputed_hard_event_matrix": bool(recompute_hard_events),
        "split_identity": _identity_summary(identity_items),
        "E1_teacher_sanity_and_candidate_coverage": _summarize_lists(teacher_acc),
        "E2_evidence_sufficiency_oracle_teacher_interface": aggregate_metric_results(bdse_results),
        "E4_budget_sweep_oracle_teacher_interface": _summarize_lists(budget_acc),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, nargs="+", default=None, help="One or more splits. Kept for backward compatibility with a single value.")
    parser.add_argument("--splits", type=str, nargs="*", default=None, help="One or more splits. When multiple are given, metrics are returned per split.")
    parser.add_argument("--folders", type=str, nargs="*", default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--maps-root", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--scenario-stride", type=int, default=None)
    parser.add_argument("--preprocessed-dir", type=str, default=None, help="Load generated .npz cache instead of rebuilding samples through nuPlan devkit.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers for preprocessed-cache diagnostics.")
    parser.add_argument("--recompute-hard-events", action="store_true", help="Recompute exact per-hard-type event matrices. Slower; fast mode uses saved teacher hard-violation labels for aggregate metrics.")
    parser.add_argument("--output", type=str, default="outputs/diagnostics.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg.setdefault("paths", {})["data_cache_root"] = args.data_root
    if args.maps_root:
        cfg.setdefault("paths", {})["maps_root"] = args.maps_root
    splits = args.splits or (args.split if args.split else ["val"])
    if len(splits) == 1:
        split = splits[0]
        folders = args.folders or cfg.get("data", {}).get("split_folders", {}).get(split)
        metrics = run_diagnostics(cfg, split, folders, args.max_files, args.max_scenarios, args.scenario_stride, args.preprocessed_dir, num_workers=args.num_workers, recompute_hard_events=args.recompute_hard_events)
    else:
        metrics = {}
        for split in splits:
            folders = args.folders or cfg.get("data", {}).get("split_folders", {}).get(split)
            metrics[split] = run_diagnostics(cfg, split, folders, args.max_files, args.max_scenarios, args.scenario_stride, args.preprocessed_dir, num_workers=args.num_workers, recompute_hard_events=args.recompute_hard_events)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
