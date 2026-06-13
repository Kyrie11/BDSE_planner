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
from bdse.planner.tournament import run_tournament, selected_pair_sigma_from_action_variance


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
        pred = model.predict_certificate_numpy(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
        J0, g = pred["J0"], pred["g"]
        g_var = pred.get("g_var", None)
        flags = runtime_safety_flags_from_runtime(sample.runtime, sample.candidates, cfg)
        topm = np.asarray(pred.get("top_m_atoms", np.flatnonzero(sample.evidence_bank.active_mask)), dtype=np.int64)
        atom_active = np.zeros((sample.evidence_bank.E,), dtype=bool)
        atom_active[topm[(topm >= 0) & (topm < sample.evidence_bank.E)]] = True
        atom_active &= sample.evidence_bank.active_mask
        sel = runtime_greedy_selector(
            J0, g, sample.evidence_bank.budget_costs(), sample.candidates.valid_mask, flags, float(cfg["evidence"]["budget"]),
            L_infer=int(cfg.get("tournament", {}).get("L_infer", 16)),
            gamma_max=float(cfg.get("selector", {}).get("gamma_max_default", 100.0)),
            eta_pred=float(cfg.get("selector", {}).get("eta_pred", 1.0)),
            lambda_near=float(cfg.get("selector", {}).get("lambda_near", 1.0)),
            lambda_safety=float(cfg.get("selector", {}).get("lambda_safety", 2.0)),
            atom_active_mask=atom_active,
            predicted_atom_variance=g_var,
            beta_uncertainty=float(cfg.get("tournament", {}).get("beta_uncertainty", 0.0)),
            epsilon_cal=float(cfg.get("tournament", {}).get("epsilon_cal", cfg.get("calibration", {}).get("epsilon_cal", 0.0))),
            lambda_info=float(cfg.get("selector", {}).get("lambda_info", 0.0)),
            prior_atom_variance=cfg.get("selector", {}).get("unqueried_atom_variance", None),
            family_ids=pred.get("family_ids", None),
            family_budget_caps=pred.get("family_budget_caps", None),
        )
        sigma = selected_pair_sigma_from_action_variance(g_var, sel.selected, sample.candidates.valid_mask)
        tour = run_tournament(J0, g, sel.selected, sample.candidates.valid_mask, flags, cfg, sigma=sigma)
        results.append(compute_bdse_diagnostics(sample.candidates, sample.evidence_bank, sample.teacher, sample.pairs, J0, g, sel.selected, tour.action_index, cfg=cfg))
    summary = aggregate_metric_results(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
