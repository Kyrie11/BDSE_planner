from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, PreprocessedBDSEDataset
from bdse.model.bdse_model import BDSEModel
from bdse.model.checkpoint_contract import load_bdse_state_with_contract
from bdse.planner.nuplan_planner import BDSEPlannerCore
from bdse.utils import configure_torch_for_device, resolve_torch_device, torch_load_any


def load_model(checkpoint: str, cfg, device: torch.device):
    # Deserialize on CPU for portability, then explicitly move the model to the
    # requested inference device.  BDSEModel.predict_certificate_numpy places all
    # runtime tensors on next(self.parameters()).device, so this is the critical
    # step that enables CUDA for calibration.
    model = BDSEModel(cfg)
    ckpt = torch_load_any(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"BDSE checkpoint has no state dictionary: {checkpoint}")
    report = load_bdse_state_with_contract(
        model, state, cfg, context=f"BDSE calibration load: {checkpoint}"
    )
    if report["missing"] or report["unexpected"] or report["shape_mismatch"]:
        print(
            f"Loaded {report['loaded_tensor_count']}/{report['model_tensor_count']} tensors; "
            f"allowed missing/new={report['missing'][:8]} unexpected={report['unexpected'][:8]} "
            f"shape_mismatch={report['shape_mismatch'][:8]}",
            flush=True,
        )
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate post-hoc BDSE margin calibration epsilon on a validation cache.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--preprocessed-dir", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--delta", type=float, default=0.1, help="Allowed one-sided error rate; epsilon is the 1-delta residual quantile.")
    parser.add_argument("--output", type=str, default="outputs/calibration.json")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Calibration device. Defaults to auto, which uses CUDA when available and otherwise CPU.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_torch_device(args.device, context="Calibration")
    configure_torch_for_device(device)
    model = load_model(args.checkpoint, cfg, device)
    print(f"Calibration device: {device}")
    if args.preprocessed_dir:
        dataset = PreprocessedBDSEDataset(args.preprocessed_dir, split=[args.split], max_scenarios=args.max_scenarios)
    else:
        dataset = NuPlanBDSEDataset(cfg, split=args.split, max_files=args.max_files, max_scenarios=args.max_scenarios, use_devkit=True)

    residuals: list[float] = []
    safety_residuals: list[float] = []
    core = BDSEPlannerCore(model=model, cfg=cfg)
    for sample in tqdm(dataset.iter_samples(), total=len(dataset)):
        if sample.teacher is None or sample.pairs is None or len(sample.pairs.pairs) == 0:
            continue
        # Calibrate the same staged planner used at deployment.  This is important
        # when runtime.use_pair_conditioned_margins=True: a factorized budgeted_margin
        # calibration can report epsilon=0 even though the deployed pair scorer is the
        # object that actually drives the tournament.
        # Run one base stage so that we can access the selected margin matrix used
        # by the tournament without fallback side effects.
        candidates = sample.candidates
        evidence_bank = sample.evidence_bank
        pred, sel, tour, _ = core._run_certificate_stage(sample.runtime, candidates, evidence_bank, cfg)
        M_pred = np.asarray(tour.margins, dtype=np.float32)
        pair_source = np.asarray(sel.pair_indices, dtype=np.int64).reshape(-1, 2) if np.asarray(sel.pair_indices).size else np.asarray(sample.pairs.pairs, dtype=np.int64).reshape(-1, 2)
        if pair_source.size == 0:
            continue
        seen: set[tuple[int, int]] = set()
        for a_raw, b_raw in pair_source.tolist():
            a, b = int(a_raw), int(b_raw)
            if (a, b) in seen or not (0 <= a < M_pred.shape[0] and 0 <= b < M_pred.shape[1]):
                continue
            seen.add((a, b))
            true_margin = float(sample.teacher.J_T[b] - sample.teacher.J_T[a])
            if bool(cfg.get("model", {}).get("pair_margin_normalized", False)):
                mcfg = cfg.get("model", {})
                tcfg = cfg.get("training", {})
                scale_default = float(mcfg.get("margin_normalization_min_scale", tcfg.get("pair_margin_min_scale", 100.0)))
                scale = float(tour.diagnostics.get("margin_scale", pred.get("pair_margin_scale", scale_default)))
                true_margin = true_margin / max(scale, 1e-6)
            pred_margin = float(M_pred[a, b])
            err = max(0.0, pred_margin - true_margin)
            residuals.append(err)
            if bool(sample.teacher.hard_violation_mask[b]) and not bool(sample.teacher.hard_violation_mask[a]):
                safety_residuals.append(err)

    if not residuals:
        raise RuntimeError("No valid pair residuals found for calibration")
    q = 1.0 - float(args.delta)
    eps = float(np.quantile(np.asarray(residuals, dtype=np.float64), q))
    eps_safety = float(np.quantile(np.asarray(safety_residuals or residuals, dtype=np.float64), q))
    out = {
        "epsilon_cal": eps,
        "epsilon_cal_safety": eps_safety,
        "delta": float(args.delta),
        "pair_count": int(len(residuals)),
        "safety_pair_count": int(len(safety_residuals)),
        "device": str(device),
        "normalized_margins": bool(cfg.get("model", {}).get("pair_margin_normalized", False)),
        "recommendation": "Set tournament.epsilon_cal to epsilon_cal. If normalized_margins=true, this epsilon is dimensionless and should stay close to O(1), not raw teacher-cost units.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
