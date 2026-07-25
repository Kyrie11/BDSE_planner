from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.external_baselines.model_factory import load_model_for_config
from bdse.planner.nuplan_planner import BDSEPlannerCore
from bdse.utils import configure_torch_for_device, resolve_torch_device


def _conformal_quantile(values: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal upper quantile."""
    values = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    if values.size == 0:
        return float("nan")
    rank = int(math.ceil((values.size + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), values.size)
    return float(values[rank - 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate one-sided atom-pair adverse bounds on a held-out cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--split", nargs="+", default=["val"])
    parser.add_argument("--max-scenarios", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--prior-radius", type=float, default=0.10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be in (0,1)")
    cfg = load_config(args.config)
    device = resolve_torch_device(args.device, context="adverse-bound calibration")
    configure_torch_for_device(device)
    model = load_model_for_config(args.checkpoint, cfg, device)
    core = BDSEPlannerCore(model=model, cfg=cfg)
    dataset = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split, max_scenarios=args.max_scenarios)

    global_scores: list[float] = []
    family_scores: dict[int, list[float]] = defaultdict(list)
    raw_errors: list[float] = []
    variance_count = 0
    atom_pair_count = 0
    scene_count = 0

    for sample in tqdm(dataset.iter_samples(), total=len(dataset)):
        if sample.teacher is None:
            continue
        pred = core._predict_runtime_certificate(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
        pairs = np.asarray(pred.get("pair_indices", []), dtype=np.int64)
        pairs = pairs.reshape(-1, 2) if pairs.size else np.zeros((0, 2), dtype=np.int64)
        d_hat = np.asarray(pred.get("pair_atom_delta", []), dtype=np.float32)
        if not pairs.size or d_hat.ndim != 2 or d_hat.shape[1] != pairs.shape[0]:
            continue
        E = min(d_hat.shape[0], sample.teacher.g_evid.shape[0], sample.evidence_bank.E)
        a = pairs[:, 0]
        b = pairs[:, 1]
        ok_pair = (a >= 0) & (b >= 0) & (a < sample.teacher.g_evid.shape[1]) & (b < sample.teacher.g_evid.shape[1])
        if not bool(ok_pair.any()):
            continue
        pairs = pairs[ok_pair]
        d_hat = d_hat[:E, ok_pair]
        scale = float(pred.get("pair_margin_scale", 1.0)) if bool(cfg.get("model", {}).get("pair_margin_normalized", True)) else 1.0
        scale = max(scale, 1e-6)
        true = (sample.teacher.g_evid[:E, pairs[:, 1]] - sample.teacher.g_evid[:E, pairs[:, 0]]) / scale
        var = np.asarray(pred.get("pair_atom_var", np.zeros_like(d_hat)), dtype=np.float32)
        if var.shape != np.asarray(pred.get("pair_atom_delta")).shape:
            var = np.zeros_like(np.asarray(pred.get("pair_atom_delta")), dtype=np.float32)
        var = var[:E, ok_pair]
        variance_count += int(np.count_nonzero(var > 0.0))
        sigma = np.sqrt(np.maximum(var, 0.0) + max(float(args.prior_radius), 0.0) ** 2)
        # Need true d >= predicted d - beta*sigma - epsilon.
        scores = d_hat - true - max(float(args.beta), 0.0) * sigma
        active = np.asarray(sample.evidence_bank.active_mask[:E], dtype=bool)
        finite = active[:, None] & np.isfinite(scores) & np.isfinite(true) & np.isfinite(d_hat)
        fam = np.asarray(pred.get("family_ids", np.zeros((E,), dtype=np.int64)), dtype=np.int64).reshape(-1)[:E]
        global_scores.extend(scores[finite].astype(np.float64).tolist())
        raw_errors.extend((d_hat - true)[finite].astype(np.float64).tolist())
        for fid in np.unique(fam[active]):
            mask = finite & (fam[:, None] == int(fid))
            family_scores[int(fid)].extend(scores[mask].astype(np.float64).tolist())
        atom_pair_count += int(finite.sum())
        scene_count += 1

    global_arr = np.asarray(global_scores, dtype=np.float64)
    epsilon = max(0.0, _conformal_quantile(global_arr, args.alpha))
    family = {}
    for fid, values in sorted(family_scores.items()):
        arr = np.asarray(values, dtype=np.float64)
        family[str(fid)] = {
            "count": int(arr.size),
            "epsilon": max(0.0, _conformal_quantile(arr, args.alpha)),
            "empirical_violation_rate": float(np.mean(arr > max(0.0, _conformal_quantile(arr, args.alpha)))) if arr.size else float("nan"),
        }
    output = {
        "method": "split-conformal one-sided adverse residual calibration",
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "prior_radius": float(args.prior_radius),
        "recommended_adverse_certificate_epsilon": float(epsilon),
        "scene_count": int(scene_count),
        "atom_pair_count": int(atom_pair_count),
        "learned_variance_fraction": float(variance_count / max(atom_pair_count, 1)),
        "empirical_global_violation_rate": float(np.mean(global_arr > epsilon)) if global_arr.size else float("nan"),
        "raw_error_mae": float(np.mean(np.abs(np.asarray(raw_errors)))) if raw_errors else float("nan"),
        "family": family,
        "warning": "Use a disjoint evaluation split for claimed coverage; calibration-set coverage is not a test guarantee.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
