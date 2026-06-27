from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, PreprocessedBDSEDataset
from bdse.metrics.bdse_metrics import aggregate_metric_results, compute_bdse_diagnostics
from bdse.model.bdse_model import BDSEModel
from bdse.planner.nuplan_planner import BDSEPlannerCore, runtime_query_diagnostics


def _torch_load_any(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_device(device_arg: str | None) -> torch.device:
    """Resolve the evaluation device.

    Open-loop evaluation used to load the checkpoint on CPU and never moved the
    model afterwards.  Since BDSEModel creates runtime query tensors on
    ``next(self.parameters()).device``, moving the model is sufficient to make
    the neural parts of certificate scoring run on GPU while keeping the
    numpy/selector/metric parts on CPU.
    """
    requested = (device_arg or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA was requested via --device={device_arg!r}, but torch.cuda.is_available() is False; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_model(checkpoint: str, cfg, device: torch.device):
    # Keep deserialization on CPU so checkpoints saved from any device are safe
    # to load, then explicitly move the module to the requested evaluation device.
    model = BDSEModel(cfg)
    ckpt = _torch_load_any(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
    missing = sorted(set(current) - set(compatible))
    if missing:
        print(f"Loaded {len(compatible)}/{len(current)} compatible tensors; missing/new tensors include: {missing[:8]}")
    model.load_state_dict(compatible, strict=False)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, nargs="+", default=["val"])
    parser.add_argument("--preprocessed-dir", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/open_loop_bdse_metrics.json")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Evaluation device. Defaults to auto, which uses CUDA when available and otherwise CPU.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
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
    for sample in tqdm(dataset.iter_samples(), total=len(dataset)):
        pred, sel, tour, _ = core._run_certificate_stage(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
        results.append(
            compute_bdse_diagnostics(
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
                query_diagnostics=runtime_query_diagnostics(pred, sel.selected),
            )
        )
    summary = aggregate_metric_results(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
