from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.nuplan_dataset import NuPlanBDSEDataset
from bdse.metrics.bdse_metrics import aggregate_metric_results, compute_bdse_diagnostics
from bdse.model.bdse_model import BDSEModel
from bdse.planner.fallback import runtime_safety_flags_from_runtime
from bdse.planner.selector import runtime_greedy_selector
from bdse.planner.tournament import run_tournament


def load_model(checkpoint: str, cfg):
    model = BDSEModel(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/open_loop_bdse_metrics.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = load_model(args.checkpoint, cfg)
    dataset = NuPlanBDSEDataset(cfg, split=args.split, max_files=args.max_files, max_scenarios=args.max_scenarios, use_devkit=True)
    results = []
    for sample in tqdm(dataset.iter_samples(), total=len(dataset)):
        J0, g = model.predict_numpy(sample.runtime, sample.candidates, sample.evidence_bank)
        flags = runtime_safety_flags_from_runtime(sample.runtime, sample.candidates, cfg)
        sel = runtime_greedy_selector(J0, g, sample.evidence_bank.budget_costs(), sample.candidates.valid_mask, flags, float(cfg["evidence"]["budget"]), atom_active_mask=sample.evidence_bank.active_mask)
        tour = run_tournament(J0, g, sel.selected, sample.candidates.valid_mask, flags, cfg)
        results.append(compute_bdse_diagnostics(sample.candidates, sample.evidence_bank, sample.teacher, sample.pairs, J0, g, sel.selected, tour.action_index, cfg=cfg))
    summary = aggregate_metric_results(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
