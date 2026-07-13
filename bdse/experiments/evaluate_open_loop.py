from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, PreprocessedBDSEDataset
from bdse.metrics.bdse_metrics import aggregate_metric_results, compute_bdse_diagnostics
from bdse.planner.nuplan_planner import BDSEPlannerCore, runtime_query_diagnostics
from bdse.utils import configure_torch_for_device, resolve_torch_device
from bdse.external_baselines.model_factory import load_model_for_config


def load_model(checkpoint: str | None, cfg, device: torch.device):
    # Supports both native BDSE checkpoints and budget-compatible external
    # baseline adapters.  PDM-Closed is rule-based and may be evaluated without
    # a checkpoint; trainable external baselines and BDSE require one.
    return load_model_for_config(checkpoint, cfg, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, nargs="+", default=["val"])
    parser.add_argument("--preprocessed-dir", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/open_loop_bdse_metrics.json")
    parser.add_argument("--per-sample-output", type=str, default=None, help="Optional JSONL with one diagnostic row per sample for failure slicing.")
    parser.add_argument("--disable-dense-diagnostic", action="store_true", help="Skip diagnostic-only dense full-interface scoring.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Evaluation device. Defaults to auto, which uses CUDA when available and otherwise CPU.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_torch_device(args.device, context="Open-loop evaluation")
    configure_torch_for_device(device)
    model = load_model(args.checkpoint, cfg, device)
    print(f"Open-loop evaluation device: {device}")
    core = BDSEPlannerCore(model=model, cfg=cfg)
    if args.preprocessed_dir:
        dataset = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split, max_scenarios=args.max_scenarios)
    else:
        if len(args.split) != 1:
            raise ValueError("On-the-fly open-loop evaluation supports one split; use --preprocessed-dir for multiple split folders.")
        dataset = NuPlanBDSEDataset(cfg, split=args.split[0], max_files=args.max_files, max_scenarios=args.max_scenarios, use_devkit=True)
    results = []
    per_sample_rows = []
    for sample in tqdm(dataset.iter_samples(), total=len(dataset)):
        pred, sel, tour, _ = core._run_certificate_stage(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
        qdiag = runtime_query_diagnostics(pred, sel.selected)
        qdiag.update({k: v for k, v in getattr(tour, "diagnostics", {}).items() if k in {"normalized_margins", "margin_scale", "epsilon_cal", "pair_conditioned"}})
        qdiag["fallback_would_trigger"] = bool(core._needs_fallback(tour, sample.candidates, cfg))
        mode = str(getattr(sel, "diagnostics", {}).get("mode", ""))
        qdiag["selector_action_rank_active"] = float(mode == "runtime_pair_conditioned_action_rank")
        qdiag["selector_flip_rank_active"] = float(mode == "runtime_pair_conditioned_flip_rank")
        qdiag["selector_lcb_active"] = float(mode == "runtime_pair_conditioned_lcb_uncertainty")
        qdiag["top_m_atoms"] = list(map(int, np.asarray(pred.get("top_m_atoms", []), dtype=np.int64).reshape(-1).tolist()))
        dense = None
        if not args.disable_dense_diagnostic and hasattr(model, "predict_dense_numpy"):
            dense = model.predict_dense_numpy(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
        diag = compute_bdse_diagnostics(
            sample.candidates,
            sample.evidence_bank,
            sample.teacher,
            sample.pairs,
            pred["J0"],
            pred["g"],
            sel.selected,
            tour.action_index,
            cfg=cfg,
            inference_pairs=pred.get("rival_pair_indices", sel.pair_indices),
            query_diagnostics=qdiag,
            dense_predicted_base=None if dense is None else dense["J0"],
            dense_predicted_atom_costs=None if dense is None else dense["g"],
            certificate_margin_matrix=tour.margins,
        )
        results.append(diag)
        if args.per_sample_output:
            row = {
                "scenario_token": str(getattr(sample, "scenario_token", "")),
                "timestamp_us": int(getattr(sample, "timestamp_us", 0) or 0),
                **{k: float(v) for k, v in diag.values.items()},
                "teacher_action": int(getattr(sample.teacher, "a_star", -1)),
                "bdse_action": int(tour.action_index),
                "full_action": int(diag.details.get("full_action", -1)),
                "fallback_would_trigger": bool(qdiag.get("fallback_would_trigger", False)),
            }
            per_sample_rows.append(row)
    summary = aggregate_metric_results(results)
    summary["device"] = str(device)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.per_sample_output:
        ps = Path(args.per_sample_output)
        ps.parent.mkdir(parents=True, exist_ok=True)
        ps.write_text("\n".join(json.dumps(r, sort_keys=True) for r in per_sample_rows) + ("\n" if per_sample_rows else ""), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
