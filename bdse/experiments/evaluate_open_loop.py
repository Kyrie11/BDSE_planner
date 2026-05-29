from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, PreprocessedBDSEDataset
from bdse.metrics.bdse_metrics import aggregate_metric_results, compute_bdse_diagnostics
from bdse.model.bdse_model import BDSEModel
from bdse.planner.nuplan_planner import BDSEPlannerCore


def load_model(checkpoint: str, cfg):
    model = BDSEModel(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, nargs="+", default=["val"])
    parser.add_argument("--preprocessed-dir", type=str, default=None, help="Evaluate cached .npz samples instead of rebuilding from nuPlan devkit.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--dense-full-interface", dest="dense_full_interface", action="store_true", default=True)
    parser.add_argument("--no-dense-full-interface", dest="dense_full_interface", action="store_false")
    parser.add_argument("--write-details", action="store_true")
    parser.add_argument("--output", type=str, default="outputs/open_loop_bdse_metrics.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = load_model(args.checkpoint, cfg)
    if args.preprocessed_dir:
        dataset = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split, max_scenarios=args.max_scenarios)
    else:
        if len(args.split) != 1:
            raise ValueError("On-the-fly open-loop evaluation supports one split at a time; use --preprocessed-dir for multiple split folders.")
        dataset = NuPlanBDSEDataset(cfg, split=args.split[0], max_files=args.max_files, max_scenarios=args.max_scenarios, use_devkit=True)
    core = BDSEPlannerCore(model=model, cfg=cfg)
    results = []
    detail_rows = []
    for sample in tqdm(dataset.iter_samples(), total=len(dataset)):
        if sample.teacher is None or sample.pairs is None:
            continue
        action, _, planner_diag = core.plan_from_components(sample.runtime, sample.candidates, sample.evidence_bank)
        if args.dense_full_interface and hasattr(model, "predict_dense_numpy"):
            J0, g_dense = model.predict_dense_numpy(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
            g_budget = g_dense
            g_full = g_dense
        else:
            pred = model.predict_certificate_numpy(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
            J0 = pred["J0"]
            g_budget = pred["g"]
            g_full = None
        res = compute_bdse_diagnostics(
            sample.candidates,
            sample.evidence_bank,
            sample.teacher,
            sample.pairs,
            J0,
            g_budget,
            planner_diag.get("selected_atoms", []),
            action,
            cfg=cfg,
            planner_diagnostics=planner_diag,
            full_predicted_atom_costs=g_full,
        )
        results.append(res)
        if args.write_details:
            detail_rows.append({
                "scenario_token": sample.scenario_token,
                "timestamp_us": int(sample.timestamp_us),
                **res.values,
                "action_index": int(action),
                "a_star": int(sample.teacher.a_star),
                "fallback_stage": planner_diag.get("fallback_stage", ""),
            })
    summary = aggregate_metric_results(results)
    summary["num_scenarios"] = float(len(results))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary}
    if args.write_details:
        payload["details"] = detail_rows
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
