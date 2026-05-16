from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, PreprocessedBDSEDataset
from bdse.metrics.bdse_metrics import aggregate_metric_results, compute_bdse_diagnostics
from bdse.planner.fallback import runtime_safety_flags_from_runtime
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


def _teacher_sample_metrics(s) -> dict[str, float]:
    log_nearest, log_costs = _log_nearest(s)
    valid = s.candidates.valid_mask.astype(bool)
    a_star = int(s.teacher.a_star)
    route = np.asarray(s.runtime.map_features.get("route_centerline", []), dtype=np.float32)
    map_valid = float(bool(s.runtime.map_features.get("map_valid", False)))
    route_fallback = float(str(s.runtime.map_features.get("route_source", "")).startswith("fallback"))
    hard_valid = s.teacher.hard_violation_mask[valid]
    teacher_vs_log = float(log_nearest >= 0 and log_nearest != a_star)
    teacher_regret_to_log = float(s.teacher.J_T[log_nearest] - s.teacher.J_T[a_star]) if log_nearest >= 0 and np.isfinite(s.teacher.J_T[log_nearest]) else float("nan")
    family_counts = Counter(a.family for a in s.evidence_bank.atoms)
    hard_counts = Counter(a.type for a in s.evidence_bank.atoms if a.is_hard)
    return {
        "teacher_vs_log_disagreement": teacher_vs_log,
        "teacher_regret_to_log_nearest": teacher_regret_to_log,
        "safe_candidate_exists": float(((~s.teacher.hard_violation_mask) & valid).any()),
        "teacher_hard_violation": float(s.teacher.hard_violation_mask[a_star]),
        "candidate_hard_violation_rate": float(hard_valid.mean()) if hard_valid.size else float("nan"),
        "valid_candidate_count": float(valid.sum()),
        "candidate_log_ade_min": float(np.nanmin(log_costs)) if log_nearest >= 0 else float("nan"),
        "candidate_log_ade_teacher": float(log_costs[a_star]) if log_nearest >= 0 else float("nan"),
        "map_valid_rate": map_valid,
        "route_fallback_rate": route_fallback,
        "route_centerline_points": float(len(route)),
        "route_ids_nonempty": float(len(s.runtime.route_roadblock_ids) > 0),
        "mission_goal_nonempty": float(s.runtime.mission_goal is not None and np.asarray(s.runtime.mission_goal).size > 0),
        "traffic_light_nonempty": float(len(s.runtime.traffic_lights) > 0),
        "red_light_atom_count": float(sum(1 for a in s.evidence_bank.atoms if a.type == "red_light")),
        "drivable_polygon_count": float(len(s.runtime.map_features.get("drivable_polygons", []))),
        "stop_line_count": float(len(s.runtime.map_features.get("stop_lines", []))),
        "agent_future_valid_rate": float(s.label_future.agent_valid.mean()) if s.label_future is not None else float("nan"),
        "atom_count": float(len(s.evidence_bank.atoms)),
        "pair_count": float(0 if s.pairs is None else len(s.pairs.pairs)),
        "pair_nonempty": float(s.pairs is not None and len(s.pairs.pairs) > 0),
        "interaction_atom_count": float(family_counts.get("interaction", 0)),
        "rule_map_atom_count": float(family_counts.get("rule_map", 0)),
        "kinematic_atom_count": float(family_counts.get("kinematic", 0)),
        "hard_atom_count": float(sum(hard_counts.values())),
        "teacher_J_base_min": float(np.nanmin(s.teacher.J_base[valid])) if valid.any() else float("nan"),
        "teacher_J_evid_min": float(np.nanmin(s.teacher.J_evid[valid])) if valid.any() else float("nan"),
        "teacher_J_T_min": float(s.teacher.J_T[a_star]),
        "partition_max_abs_error": float(np.nanmax(np.abs(s.teacher.J_T[valid] - (s.teacher.J_base[valid] + s.teacher.J_evid[valid])))) if valid.any() else float("nan"),
        "evidence_sum_max_abs_error": float(np.nanmax(np.abs(s.teacher.J_evid[valid] - s.teacher.g_evid[:, valid].sum(axis=0)))) if valid.any() else float("nan"),
    }


def _update_lists(acc: dict[str, list[float]], vals: dict[str, float]) -> None:
    for k, v in vals.items():
        acc[k].append(float(v))


def _summarize_lists(acc: dict[str, list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, vals in sorted(acc.items()):
        out[k] = _finite_mean(vals)
        if k in {"candidate_log_ade_min", "candidate_log_ade_teacher", "teacher_J_T_min", "teacher_regret_to_log_nearest", "valid_candidate_count", "atom_count", "pair_count", "interaction_atom_count", "drivable_polygon_count"}:
            out[k + "_p50"] = _finite_pct(vals, 50)
            out[k + "_p90"] = _finite_pct(vals, 90)
    return out


def run_diagnostics(cfg: dict[str, Any], split: str, folders: list[str] | None, max_files: int | None, max_scenarios: int | None, scenario_stride: int | None, preprocessed_dir: str | None = None) -> dict[str, Any]:
    if preprocessed_dir:
        dataset = PreprocessedBDSEDataset(preprocessed_dir, split=split, max_scenarios=max_scenarios)
    else:
        dataset = NuPlanBDSEDataset(cfg, split=split, folders=folders, max_files=max_files, max_scenarios=max_scenarios, stride=scenario_stride, use_devkit=True)
    teacher_acc: dict[str, list[float]] = defaultdict(list)
    bdse_results = []
    budget_acc: dict[str, list[float]] = defaultdict(list)
    budgets = cfg.get("diagnostics", {}).get("budget_sweep", [4, 8, 16, 24, 32])
    iterator = tqdm(range(len(dataset)), total=len(dataset), desc=f"diagnostics:{split}")
    skipped_missing = 0
    for i in iterator:
        s = dataset[i]
        if s.teacher is None or s.label_future is None:
            skipped_missing += 1
            continue
        _update_lists(teacher_acc, _teacher_sample_metrics(s))
        J0 = s.teacher.J_base.copy()
        g = s.teacher.g_evid.copy()
        flags = runtime_safety_flags_from_runtime(s.runtime, s.candidates, cfg)
        runtime_sel = runtime_greedy_selector(J0, g, s.evidence_bank.budget_costs(), s.candidates.valid_mask, flags, float(cfg["evidence"]["budget"]), atom_active_mask=s.evidence_bank.active_mask)
        oracle_sel = oracle_greedy_selector(s.teacher.J_base, s.teacher.g_evid, s.pairs.pairs, s.pairs.margins, s.pairs.weights, s.evidence_bank.budget_costs(), float(cfg["evidence"]["budget"]), s.evidence_bank.active_mask)
        tour = run_tournament(J0, g, runtime_sel.selected, s.candidates.valid_mask, flags, cfg)
        bdse_results.append(compute_bdse_diagnostics(s.candidates, s.evidence_bank, s.teacher, s.pairs, J0, g, runtime_sel.selected, tour.action_index, runtime_sel.selected, oracle_sel.selected, cfg))
        for B in budgets:
            sel = oracle_greedy_selector(s.teacher.J_base, s.teacher.g_evid, s.pairs.pairs, s.pairs.margins, s.pairs.weights, s.evidence_bank.budget_costs(), float(B), s.evidence_bank.active_mask)
            t = run_tournament(J0, g, sel.selected, s.candidates.valid_mask, flags, cfg)
            budget_acc[f"B{int(B)}_decision_sufficiency"].append(float(t.action_index == s.teacher.a_star))
            hard = set(np.flatnonzero(s.evidence_bank.hard_mask() & s.evidence_bank.active_mask).astype(int).tolist())
            budget_acc[f"B{int(B)}_hard_recall"].append(float(len(hard & set(sel.selected)) / max(len(hard), 1)))
    return {
        "num_loaded": int(len(dataset)),
        "num_samples": int(len(next(iter(teacher_acc.values()))) if teacher_acc else 0),
        "num_skipped_missing_labels": int(skipped_missing),
        "E1_teacher_sanity_and_candidate_coverage": _summarize_lists(teacher_acc),
        "E2_evidence_sufficiency_oracle_teacher_interface": aggregate_metric_results(bdse_results),
        "E4_budget_sweep_oracle_teacher_interface": _summarize_lists(budget_acc),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--folders", type=str, nargs="*", default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--maps-root", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--scenario-stride", type=int, default=None)
    parser.add_argument("--preprocessed-dir", type=str, default=None, help="Load generated .npz cache instead of rebuilding samples through nuPlan devkit.")
    parser.add_argument("--output", type=str, default="outputs/diagnostics.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg.setdefault("paths", {})["data_cache_root"] = args.data_root
    if args.maps_root:
        cfg.setdefault("paths", {})["maps_root"] = args.maps_root
    folders = args.folders or cfg.get("data", {}).get("split_folders", {}).get(args.split)
    metrics = run_diagnostics(cfg, args.split, folders, args.max_files, args.max_scenarios, args.scenario_stride, args.preprocessed_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
