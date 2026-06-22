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
from bdse.planner.fallback import runtime_safety_flags_from_runtime
from bdse.planner.selector import runtime_greedy_selector, budgeted_margin


def load_model(checkpoint: str, cfg):
    model = BDSEModel(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
    model.load_state_dict(compatible, strict=False)
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = load_model(args.checkpoint, cfg)
    if args.preprocessed_dir:
        dataset = PreprocessedBDSEDataset(args.preprocessed_dir, split=[args.split], max_scenarios=args.max_scenarios)
    else:
        dataset = NuPlanBDSEDataset(cfg, split=args.split, max_files=args.max_files, max_scenarios=args.max_scenarios, use_devkit=True)

    residuals: list[float] = []
    safety_residuals: list[float] = []
    for sample in tqdm(dataset.iter_samples(), total=len(dataset)):
        if sample.teacher is None or sample.pairs is None or len(sample.pairs.pairs) == 0:
            continue
        pred = model.predict_certificate_numpy(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
        J0, g = pred["J0"], pred["g"]
        flags = runtime_safety_flags_from_runtime(sample.runtime, sample.candidates, cfg)
        topm = np.asarray(pred.get("top_m_atoms", np.flatnonzero(sample.evidence_bank.active_mask)), dtype=np.int64)
        atom_active = np.zeros((sample.evidence_bank.E,), dtype=bool)
        atom_active[topm[(topm >= 0) & (topm < sample.evidence_bank.E)]] = True
        atom_active &= sample.evidence_bank.active_mask
        sel = runtime_greedy_selector(
            J0, g, sample.evidence_bank.budget_costs(), sample.candidates.valid_mask, flags,
            float(cfg.get("evidence", {}).get("budget", 16)),
            L_infer=int(cfg.get("tournament", {}).get("L_infer", 16)),
            gamma_max=float(cfg.get("selector", {}).get("gamma_max_default", 100.0)),
            eta_pred=float(cfg.get("selector", {}).get("eta_pred", 1.0)),
            lambda_near=float(cfg.get("selector", {}).get("lambda_near", 1.0)),
            lambda_safety=float(cfg.get("selector", {}).get("lambda_safety", 2.0)),
            atom_active_mask=atom_active,
            predicted_atom_variance=pred.get("g_var", None),
            beta_uncertainty=float(cfg.get("tournament", {}).get("beta_uncertainty", 0.0)),
            epsilon_cal=float(cfg.get("tournament", {}).get("epsilon_cal", cfg.get("calibration", {}).get("epsilon_cal", 0.0))),
            lambda_info=float(cfg.get("selector", {}).get("lambda_info", 0.0)),
            prior_atom_variance=cfg.get("selector", {}).get("unqueried_atom_variance", None),
            family_ids=pred.get("family_ids", None),
            family_budget_caps=pred.get("family_budget_caps", None),
            bidirectional_pairs=bool(cfg.get("selector", {}).get("bidirectional_pairs", True)),
            reverse_pair_weight=float(cfg.get("selector", {}).get("reverse_pair_weight", 1.0)),
            pair_cap_multiplier=float(cfg.get("selector", {}).get("runtime_pair_cap_multiplier", 1.0)),
        )
        M_pred = budgeted_margin(J0, g, sel.selected)
        for (a, b), valid in zip(sample.pairs.pairs, sample.pairs.valid_mask):
            if not valid:
                continue
            true_margin = float(sample.teacher.J_T[b] - sample.teacher.J_T[a])
            pred_margin = float(M_pred[a, b])
            # One-sided certificate calibration: epsilon covers over-confident
            # margins.  Underestimation is conservative and should not inflate eps.
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
        "recommendation": "Set tournament.epsilon_cal to epsilon_cal. This is a one-sided over-confidence quantile, not an absolute residual.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
