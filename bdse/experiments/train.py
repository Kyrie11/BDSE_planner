from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.cache_schema import Sample
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, PreprocessedBDSEDataset
from bdse.model.bdse_model import BDSEModel, EVIDENCE_TYPE_TO_ID, FAMILY_TO_ID
from bdse.model.losses import compute_bdse_losses
from bdse.planner.selector import oracle_greedy_selector, runtime_greedy_selector


class OnTheFlyDataset(Dataset):
    def __init__(self, source: NuPlanBDSEDataset):
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> Sample:
        return self.source[idx]


def sample_to_tensors(sample: Sample, cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    Emax = int(cfg.get("evidence", {}).get("max_atoms", 128))
    K = int(cfg.get("candidate", {}).get("K", 32))
    efd = int(cfg.get("model", {}).get("evidence_feature_dim", 24))
    qfd = int(cfg.get("model", {}).get("query_feature_dim", 12))
    Pmax = int(cfg.get("pairs", {}).get("target_max", 256))
    E = min(Emax, sample.evidence_bank.E)
    evidence_features = np.zeros((Emax, efd), dtype=np.float32)
    type_ids = np.zeros((Emax,), dtype=np.int64)
    family_ids = np.zeros((Emax,), dtype=np.int64)
    active = np.zeros((Emax,), dtype=bool)
    budget_costs = np.ones((Emax,), dtype=np.float32)
    for i, atom in enumerate(sample.evidence_bank.atoms[:E]):
        active[i] = bool(atom.active_mask)
        budget_costs[i] = float(atom.budget_cost)
        type_ids[i] = EVIDENCE_TYPE_TO_ID.get(atom.type, 0)
        family_ids[i] = FAMILY_TO_ID.get(atom.family, 0)
        evidence_features[i, 0] = float(atom.is_hard)
        evidence_features[i, 1] = float(atom.budget_cost)
        if "current_state" in atom.anchor:
            st = np.asarray(atom.anchor["current_state"], dtype=np.float32)
            evidence_features[i, 2 : 2 + min(10, len(st))] = st[: min(10, len(st))]
    query = np.zeros((Emax, K, qfd), dtype=np.float32)
    q = sample.evidence_bank.query_features[:E, :K, :qfd]
    query[: q.shape[0], : q.shape[1], : q.shape[2]] = q
    if sample.teacher is None or sample.pairs is None:
        raise ValueError("Training sample requires teacher and pair labels")
    J_base = sample.teacher.J_base.astype(np.float32)
    g = np.zeros((Emax, K), dtype=np.float32)
    g[: sample.teacher.g_evid.shape[0], : sample.teacher.g_evid.shape[1]] = sample.teacher.g_evid[:Emax, :K]
    JT = sample.teacher.J_T.astype(np.float32)
    pairs = np.zeros((Pmax, 2), dtype=np.int64)
    pair_valid = np.zeros((Pmax,), dtype=bool)
    margins = np.zeros((Pmax,), dtype=np.float32)
    weights = np.zeros((Pmax,), dtype=np.float32)
    residuals = np.zeros((Pmax,), dtype=np.float32)
    p = sample.pairs.pairs[:Pmax]
    n = len(p)
    pairs[:n] = p
    pair_valid[:n] = sample.pairs.valid_mask[:n]
    margins[:n] = sample.pairs.margins[:n]
    weights[:n] = sample.pairs.weights[:n]
    residuals[:n] = sample.pairs.residuals[:n]
    oracle = oracle_greedy_selector(J_base, g, pairs[:n], margins[:n], weights[:n], budget_costs, float(cfg.get("evidence", {}).get("budget", 16)), active)
    oracle_mask = np.zeros((Emax,), dtype=bool)
    oracle_mask[oracle.selected] = True
    runtime_flags = sample.teacher.hard_violation_mask.copy()
    runtime = runtime_greedy_selector(J_base, g, budget_costs, sample.candidates.valid_mask, runtime_flags, float(cfg.get("evidence", {}).get("budget", 16)), atom_active_mask=active)
    runtime_mask = np.zeros((Emax,), dtype=bool)
    runtime_mask[runtime.selected] = True
    return {
        "ego_history": torch.from_numpy(sample.runtime.ego_history).float(),
        "agent_history": torch.from_numpy(sample.runtime.agent_history).float(),
        "agent_valid": torch.from_numpy(sample.runtime.agent_valid),
        "candidate_trajectories": torch.from_numpy(sample.candidates.trajectories).float(),
        "candidate_valid": torch.from_numpy(sample.candidates.valid_mask),
        "candidate_maneuver_ids": torch.from_numpy(sample.candidates.maneuver_ids),
        "evidence_features": torch.from_numpy(evidence_features).float(),
        "evidence_query_features": torch.from_numpy(query).float(),
        "evidence_active": torch.from_numpy(active),
        "evidence_type_ids": torch.from_numpy(type_ids),
        "evidence_family_ids": torch.from_numpy(family_ids),
        "teacher_J_base": torch.from_numpy(J_base).float(),
        "teacher_g_evid": torch.from_numpy(g).float(),
        "teacher_J_T": torch.from_numpy(JT).float(),
        "teacher_a_star": torch.tensor(sample.teacher.a_star, dtype=torch.long),
        "pair_indices": torch.from_numpy(pairs),
        "pair_valid": torch.from_numpy(pair_valid),
        "pair_margins": torch.from_numpy(margins).float(),
        "pair_weights": torch.from_numpy(weights).float(),
        "pair_residuals": torch.from_numpy(residuals).float(),
        "oracle_selected_mask": torch.from_numpy(oracle_mask),
        "runtime_selected_mask": torch.from_numpy(runtime_mask),
    }


def collate(samples: list[Sample], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    items = [sample_to_tensors(s, cfg) for s in samples]
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--preprocessed-dir", type=str, default=None, help="Load generated .npz cache instead of building samples on the fly.")
    parser.add_argument("--output", type=str, default="outputs/bdse_model.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.preprocessed_dir:
        dataset = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split, max_scenarios=args.max_scenarios)
    else:
        dataset = NuPlanBDSEDataset(cfg, split=args.split, max_files=args.max_files, max_scenarios=args.max_scenarios, use_devkit=True)
    loader = DataLoader(OnTheFlyDataset(dataset), batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=0, collate_fn=lambda x: collate(x, cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BDSEModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        meters: dict[str, list[float]] = {}
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            losses = compute_bdse_losses(out, batch, cfg)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"]))
            opt.step()
            for k, v in losses.items():
                meters.setdefault(k, []).append(float(v.detach().cpu()))
        print({k: float(np.mean(v)) for k, v in meters.items()})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, out_path)


if __name__ == "__main__":
    main()
