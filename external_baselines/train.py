from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.cache_schema import Sample, load_sample_npz
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.data.tensorizer import sample_to_model_inputs
from bdse.external_baselines.losses import compute_external_baseline_losses
from bdse.external_baselines.models import ExternalBaselineModel, external_variant
from bdse.metrics.bdse_metrics import aggregate_metric_results, compute_bdse_diagnostics
from bdse.planner.nuplan_planner import BDSEPlannerCore, runtime_query_diagnostics
from bdse.utils import configure_torch_for_device, resolve_torch_device, torch_load_any


class ExternalBaselineDataset(Dataset):
    def __init__(self, source: PreprocessedBDSEDataset):
        self.paths = list(source.build_index())

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Sample:
        return load_sample_npz(self.paths[idx], include_label_future=False, include_candidate_metadata=False)


def collate(samples: list[Sample], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    items = [sample_to_model_inputs(s, cfg, include_teacher=True, include_dense_query=False) for s in samples]
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}


def _save_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, cfg: dict[str, Any], epoch: int, metrics: dict[str, float]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "cfg": cfg, "epoch": int(epoch), "metrics": metrics}, tmp)
    tmp.replace(p)


def _load_resume(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    ckpt = torch_load_any(path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
    model.load_state_dict(compatible, strict=False)
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass
    return int(ckpt.get("epoch", -1)) + 1 if isinstance(ckpt, dict) else 0


@torch.no_grad()
def validation_loss(model: ExternalBaselineModel, loader: DataLoader, cfg: dict[str, Any], device: torch.device) -> dict[str, float]:
    was = model.training
    model.eval()
    meters: dict[str, list[float]] = {}
    for batch in tqdm(loader, desc="val-loss", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch)
        losses = compute_external_baseline_losses(out, batch, cfg)
        for k, v in losses.items():
            meters.setdefault(k, []).append(float(v.detach().cpu()))
    if was:
        model.train()
    return {f"val_{k}": float(np.mean(v)) for k, v in meters.items() if v}


@torch.no_grad()
def validation_open_loop(model: ExternalBaselineModel, dataset: PreprocessedBDSEDataset, cfg: dict[str, Any], max_scenarios: int | None = None) -> dict[str, float]:
    was = model.training
    model.eval()
    core = BDSEPlannerCore(model=model, cfg=cfg)
    results = []
    n = len(dataset) if max_scenarios is None else min(len(dataset), int(max_scenarios))
    for i in tqdm(range(n), desc="val-open-loop", leave=False):
        sample = dataset[i]
        pred, sel, tour, _ = core._run_certificate_stage(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
        qdiag = runtime_query_diagnostics(pred, sel.selected)
        qdiag["fallback_would_trigger"] = bool(core._needs_fallback(tour, sample.candidates, cfg))
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
            certificate_margin_matrix=tour.margins,
        )
        results.append(diag)
    if was:
        model.train()
    out = aggregate_metric_results(results) if results else {}
    return {f"val_{k}": float(v) for k, v in out.items() if isinstance(v, (int, float, np.floating))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train budget-compatible external baseline adapters on BDSE preprocessed caches.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, nargs="+", default=["train_boston", "train_pittsburgh", "train_singapore", "train_vegas_2"])
    parser.add_argument("--preprocessed-dir", type=str, required=True)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-scenarios-per-split", type=int, default=None)
    parser.add_argument("--val-preprocessed-dir", type=str, default=None)
    parser.add_argument("--val-split", type=str, nargs="+", default=["val"])
    parser.add_argument("--val-max-scenarios", type=int, default=1000)
    parser.add_argument("--val-every-n-epochs", type=int, default=1)
    parser.add_argument("--val-mode", type=str, choices=["loss", "open_loop", "none"], default="loss")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not bool(cfg.get("external_baseline", {}).get("enabled", False)):
        raise ValueError("External baseline training requires external_baseline.enabled: true in the config.")
    if external_variant(cfg) == "pdm_closed":
        raise ValueError("pdm_closed is a non-trainable rule baseline; run evaluation directly without training.")
    tcfg = cfg.get("training", {}) or {}
    ecfg = cfg.get("external_baseline", {}) or {}
    epochs = int(args.epochs if args.epochs is not None else ecfg.get("epochs", tcfg.get("epochs", 20)))
    batch_size = int(args.batch_size if args.batch_size is not None else ecfg.get("batch_size", tcfg.get("batch_size", 32)))
    num_workers = int(args.num_workers if args.num_workers is not None else ecfg.get("num_workers", tcfg.get("num_workers", 4)))
    lr = float(args.lr if args.lr is not None else ecfg.get("lr", tcfg.get("lr", 1e-4)))
    wd = float(args.weight_decay if args.weight_decay is not None else ecfg.get("weight_decay", tcfg.get("weight_decay", 1e-2)))
    grad_clip = float(args.grad_clip if args.grad_clip is not None else ecfg.get("grad_clip", tcfg.get("grad_clip", 5.0)))

    device = resolve_torch_device(args.device, context="external baseline training")
    configure_torch_for_device(device)
    train_source = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split, max_scenarios=args.max_scenarios, max_scenarios_per_split=args.max_scenarios_per_split)
    train_ds = ExternalBaselineDataset(train_source)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=lambda xs: collate(xs, cfg),
        prefetch_factor=1 if num_workers > 0 else None,
    )
    val_loader = None
    val_dataset = None
    if args.val_mode != "none" and args.val_preprocessed_dir:
        val_source = PreprocessedBDSEDataset(args.val_preprocessed_dir, split=args.val_split, max_scenarios=args.val_max_scenarios)
        if args.val_mode == "loss":
            val_loader = DataLoader(
                ExternalBaselineDataset(val_source),
                batch_size=batch_size,
                shuffle=False,
                num_workers=max(0, min(num_workers, 4)),
                pin_memory=(device.type == "cuda"),
                persistent_workers=(num_workers > 0),
                collate_fn=lambda xs: collate(xs, cfg),
                prefetch_factor=1 if num_workers > 0 else None,
            )
        else:
            val_dataset = val_source

    model = ExternalBaselineModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    start_epoch = 0
    if args.resume_from:
        start_epoch = _load_resume(args.resume_from, model, optimizer)
    log_path = Path(args.log_file) if args.log_file else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = str(Path(args.output).with_suffix(".best.pt"))

    for epoch in range(start_epoch, epochs):
        model.train()
        meters: dict[str, list[float]] = {}
        pbar = tqdm(loader, desc=f"external-{external_variant(cfg)} epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                out = model(batch)
                losses = compute_external_baseline_losses(out, batch, cfg)
                loss = losses["loss"]
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for k, v in losses.items():
                meters.setdefault(k, []).append(float(v.detach().cpu()))
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
        metrics = {k: float(np.mean(v)) for k, v in meters.items() if v}
        if (epoch + 1) % max(1, int(args.val_every_n_epochs)) == 0:
            if val_loader is not None:
                metrics.update(validation_loss(model, val_loader, cfg, device))
            elif val_dataset is not None:
                metrics.update(validation_open_loop(model, val_dataset, cfg, args.val_max_scenarios))
        val_key = "val_loss" if "val_loss" in metrics else "loss"
        if float(metrics.get(val_key, float("inf"))) < best_val:
            best_val = float(metrics.get(val_key, float("inf")))
            _save_checkpoint(best_path, model=model, optimizer=optimizer, cfg=cfg, epoch=epoch, metrics=metrics)
        _save_checkpoint(args.output, model=model, optimizer=optimizer, cfg=cfg, epoch=epoch, metrics=metrics)
        row = {"epoch": epoch + 1, "variant": external_variant(cfg), **metrics, "best_metric": best_val}
        if log_path:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
