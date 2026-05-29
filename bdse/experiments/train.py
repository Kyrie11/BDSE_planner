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
from bdse.data.tensorizer import sample_to_model_inputs
from bdse.model.bdse_model import BDSEModel
from bdse.model.losses import compute_bdse_losses


class OnTheFlyDataset(Dataset):
    def __init__(self, source: NuPlanBDSEDataset):
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> Sample:
        return self.source[idx]


def sample_to_tensors(sample: Sample, cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    # Unified tensorizer is shared by training and deployment to avoid train/deploy
    # feature skew.  It intentionally does not create a teacher-derived
    # runtime_selected_mask; L_act builds its certificate through predicted
    # proposal/greedy/tournament inside the loss.
    return sample_to_model_inputs(sample, cfg, include_teacher=True, include_dense_query=True)

def collate(samples: list[Sample], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    items = [sample_to_tensors(s, cfg) for s in samples]
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, nargs="+", default=["train"], help="One or more preprocessed splits/folders, e.g. train or train_1 train_2.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--preprocessed-dir", type=str, default=None, help="Load generated .npz cache instead of building samples on the fly.")
    parser.add_argument("--output", type=str, default="outputs/bdse_model.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    splits = args.split
    if args.preprocessed_dir:
        dataset = PreprocessedBDSEDataset(args.preprocessed_dir, split=splits, max_scenarios=args.max_scenarios)
    else:
        if len(splits) != 1:
            raise ValueError("On-the-fly training supports one split at a time; preprocess first to train from multiple split folders.")
        dataset = NuPlanBDSEDataset(cfg, split=splits[0], max_files=args.max_files, max_scenarios=args.max_scenarios, use_devkit=True)
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
            epoch_cfg = dict(cfg)
            epoch_cfg["training"] = {**cfg.get("training", {}), "current_epoch": epoch}
            losses = compute_bdse_losses(out, batch, epoch_cfg)
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
