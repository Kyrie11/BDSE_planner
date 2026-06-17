from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.cache_schema import Sample
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, PreprocessedBDSEDataset
from bdse.data.tensorizer import sample_to_model_inputs
from bdse.data.quality import quality_decision
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



def _json_loads_npz_scalar(z: Any, key: str, default: Any) -> Any:
    if key not in z.files:
        return default
    try:
        raw = z[key]
        text = str(raw.item()) if raw.shape == () else str(raw.tolist())
        return json.loads(text)
    except Exception:
        return default


def _quality_metrics_from_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        diag = _json_loads_npz_scalar(z, "teacher_diagnostics_json", {})
        if not isinstance(diag, dict):
            diag = {}
        out: dict[str, Any] = {}
        for k, v in diag.items():
            if str(k).startswith("quality_"):
                out[str(k)[len("quality_"):]] = v
        # Backward compatibility for old caches that do not contain quality_ keys.
        if "safe_candidate_exists" not in out and "teacher_hard_violation" in z.files and "candidate_valid" in z.files:
            valid = np.asarray(z["candidate_valid"], dtype=bool)
            hard = np.asarray(z["teacher_hard_violation"], dtype=bool)
            out["valid_candidate_count"] = int(valid.sum())
            out["safe_candidate_count"] = int((valid & ~hard).sum()) if hard.shape == valid.shape else 0
            out["safe_candidate_exists"] = bool(out["safe_candidate_count"] > 0)
        return out


def _apply_training_quality_filter(dataset: Any, cfg: dict[str, Any]) -> None:
    qcfg = cfg.get("training", {}).get("quality_filter", {})
    if not bool(qcfg.get("enabled", False)):
        return
    if not isinstance(dataset, PreprocessedBDSEDataset):
        print("[bdse] training quality_filter is enabled but only applies to preprocessed caches; continuing without filtering.", flush=True)
        return
    paths = list(dataset.build_index())
    kept: list[Path] = []
    dropped: dict[str, int] = {}
    for p in paths:
        try:
            metrics = _quality_metrics_from_npz(Path(p))
            dec = quality_decision(metrics, cfg)
        except Exception as exc:
            if bool(qcfg.get("drop_unreadable", True)):
                dropped[type(exc).__name__] = dropped.get(type(exc).__name__, 0) + 1
                continue
            kept.append(Path(p)); continue
        if dec.keep:
            kept.append(Path(p))
        else:
            for r in dec.reasons:
                dropped[r] = dropped.get(r, 0) + 1
    if not kept:
        raise RuntimeError(f"training quality_filter dropped all {len(paths)} samples; relax thresholds. dropped={dropped}")
    dataset._paths = kept
    print(f"[bdse] training quality_filter: kept={len(kept)} dropped={len(paths)-len(kept)} reasons={dropped}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, nargs="+", default=["train"], help="One or more preprocessed splits/folders, e.g. train or train_1 train_2.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--preprocessed-dir", type=str, default=None, help="Load generated .npz cache instead of building samples on the fly.")
    parser.add_argument("--output", type=str, default="outputs/bdse_model.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision for faster training when available.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("training", {})
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["training"]["lr"] = float(args.lr)
    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = float(args.weight_decay)
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = max(0, int(args.num_workers))
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    seed = int(cfg.get("seed", 17))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    splits = args.split
    if args.preprocessed_dir:
        dataset = PreprocessedBDSEDataset(args.preprocessed_dir, split=splits, max_scenarios=args.max_scenarios)
        _apply_training_quality_filter(dataset, cfg)
    else:
        if len(splits) != 1:
            raise ValueError("On-the-fly training supports one split at a time; preprocess first to train from multiple split folders.")
        dataset = NuPlanBDSEDataset(cfg, split=splits[0], max_files=args.max_files, max_scenarios=args.max_scenarios, use_devkit=True)
    loader = DataLoader(
        OnTheFlyDataset(dataset),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg["training"].get("num_workers", 0)) > 0,
        collate_fn=lambda x: collate(x, cfg),
    )
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BDSEModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(int(cfg["training"]["epochs"])):
        cfg["training"]["current_epoch"] = int(epoch)
        model.train()
        meters: dict[str, list[float]] = {}
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batch)
                losses = compute_bdse_losses(out, batch, cfg)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            for k, v in losses.items():
                meters.setdefault(k, []).append(float(v.detach().cpu()))
        print({k: float(np.mean(v)) for k, v in meters.items()})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, out_path)


if __name__ == "__main__":
    main()
