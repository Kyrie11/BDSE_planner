from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.nuplan_dataset import NuPlanBDSEDataset
from bdse.metrics.bdse_metrics import aggregate_metric_results, compute_bdse_diagnostics
from bdse.planner.fallback import runtime_safety_flags_from_runtime
from bdse.planner.selector import oracle_greedy_selector, runtime_greedy_selector
from bdse.planner.tournament import run_tournament


def teacher_sanity(samples):
    vals = []
    for s in samples:
        log_costs = np.linalg.norm(s.candidates.trajectories[:, :, :2] - s.label_future.logged_ego[None, :, :2], axis=-1).mean(axis=1)
        log_nearest = int(np.argmin(np.where(s.candidates.valid_mask, log_costs, np.inf)))
        vals.append(
            {
                "teacher_vs_log_disagreement": float(log_nearest != s.teacher.a_star),
                "safe_candidate_exists": float(((~s.teacher.hard_violation_mask) & s.candidates.valid_mask).any()),
                "teacher_hard_violation": float(s.teacher.hard_violation_mask[s.teacher.a_star]),
                "valid_candidate_count": float(s.candidates.valid_mask.sum()),
            }
        )
    return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=100)
    parser.add_argument("--output", type=str, default="outputs/diagnostics.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = NuPlanBDSEDataset(cfg, split=args.split, max_files=args.max_files, max_scenarios=args.max_scenarios, use_devkit=True)
    samples = list(tqdm(dataset.iter_samples(), total=len(dataset)))
    metrics = {"E1_teacher_sanity": teacher_sanity(samples)}
    bdse_results = []
    for s in samples:
        J0 = s.teacher.J_base.copy()
        g = s.teacher.g_evid.copy()
        flags = runtime_safety_flags_from_runtime(s.runtime, s.candidates, cfg)
        runtime_sel = runtime_greedy_selector(J0, g, s.evidence_bank.budget_costs(), s.candidates.valid_mask, flags, float(cfg["evidence"]["budget"]), atom_active_mask=s.evidence_bank.active_mask)
        oracle_sel = oracle_greedy_selector(s.teacher.J_base, s.teacher.g_evid, s.pairs.pairs, s.pairs.margins, s.pairs.weights, s.evidence_bank.budget_costs(), float(cfg["evidence"]["budget"]), s.evidence_bank.active_mask)
        tour = run_tournament(J0, g, runtime_sel.selected, s.candidates.valid_mask, flags, cfg)
        bdse_results.append(compute_bdse_diagnostics(s.candidates, s.evidence_bank, s.teacher, s.pairs, J0, g, runtime_sel.selected, tour.action_index, runtime_sel.selected, oracle_sel.selected, cfg))
    metrics["E2_evidence_sufficiency"] = aggregate_metric_results(bdse_results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
