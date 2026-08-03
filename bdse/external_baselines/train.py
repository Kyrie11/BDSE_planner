from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import time
from functools import partial
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
from bdse.external_baselines.models import ExternalBaselineModel, external_reference, external_variant
from bdse.metrics.bdse_metrics import aggregate_metric_results, compute_bdse_diagnostics
from bdse.planner.nuplan_planner import BDSEPlannerCore, runtime_query_diagnostics
from bdse.utils import configure_torch_for_device, resolve_torch_device, torch_load_any


class ExternalBaselineDataset(Dataset):
    def __init__(self, source: PreprocessedBDSEDataset):
        self.paths = [Path(p) for p in source.build_index()]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Sample:
        return load_sample_npz(self.paths[idx], include_label_future=False, include_candidate_metadata=False)


def collate(samples: list[Sample], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    items = [sample_to_model_inputs(s, cfg, include_teacher=True, include_dense_query=False) for s in samples]
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_manifest(paths: list[Path], root: str | Path, split: list[str]) -> dict[str, Any]:
    root_path = Path(root).resolve()
    rows: list[str] = []
    total_bytes = 0
    for p in paths:
        rp = p.resolve()
        try:
            rel = rp.relative_to(root_path).as_posix()
        except ValueError:
            rel = rp.as_posix()
        try:
            size = int(rp.stat().st_size)
        except OSError:
            size = -1
        total_bytes += max(size, 0)
        rows.append(f"{rel}\t{size}")
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    return {
        "root": str(root_path),
        "split": list(split),
        "count": len(paths),
        "total_bytes": total_bytes,
        "ordered_path_size_sha256": _sha256_bytes(payload),
        "first_paths": rows[:5],
        "last_paths": rows[-5:],
    }


def _config_sha256(path: str | Path) -> str:
    return _sha256_file(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _best_checkpoint_path(output: str | Path) -> Path:
    # Passing outputs/external/gameformer_budgeted.pt intentionally produces
    # outputs/external/gameformer_budgeted.best.pt, matching all sweep scripts.
    return Path(output).with_suffix(".best.pt")


def _save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    cfg: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
    best_metric: float,
    selection_metric: str,
    training_manifest: dict[str, Any],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
        "epoch": int(epoch),
        "metrics": metrics,
        "best_metric": float(best_metric),
        "selection_metric": str(selection_metric),
        "training_manifest": training_manifest,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, tmp)
    tmp.replace(p)


def _load_resume(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[int, float]:
    ckpt = torch_load_any(path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
    model.load_state_dict(compatible, strict=False)
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            print(f"[external] optimizer resume skipped: {exc}", flush=True)
    if scheduler is not None and isinstance(ckpt, dict) and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as exc:
            print(f"[external] scheduler resume skipped: {exc}", flush=True)
    start = int(ckpt.get("epoch", -1)) + 1 if isinstance(ckpt, dict) else 0
    best = float(ckpt.get("best_metric", float("inf"))) if isinstance(ckpt, dict) else float("inf")
    return start, best


def _make_optimizer(model: torch.nn.Module, *, lr: float, weight_decay: float, fused: bool, device: torch.device) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
    if fused and device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except (RuntimeError, TypeError) as exc:
        if kwargs.pop("fused", None):
            print(f"[external] fused AdamW unavailable; using standard AdamW: {exc}", flush=True)
            return torch.optim.AdamW(model.parameters(), **kwargs)
        raise


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if scheduler_name == "none":
        return None

    warmup = max(0, int(warmup_epochs))
    total = max(1, int(epochs))

    def scale(epoch_index: int) -> float:
        # LambdaLR calls epoch 0 before the first explicit scheduler.step().
        step = epoch_index + 1
        if warmup > 0 and step <= warmup:
            return max(step / float(warmup), 1e-3)
        progress = (step - warmup) / float(max(1, total - warmup))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def _amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(enabled and device.type == "cuda"))


def _make_scaler(device: torch.device, enabled: bool):
    use = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=use)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use)


def _accumulate(meters: dict[str, torch.Tensor], losses: dict[str, torch.Tensor]) -> None:
    for key, value in losses.items():
        v = value.detach().float()
        meters[key] = meters.get(key, torch.zeros((), device=v.device, dtype=torch.float32)) + v


def _finalize_meters(meters: dict[str, torch.Tensor], count: int, prefix: str = "") -> dict[str, float]:
    denom = max(int(count), 1)
    return {f"{prefix}{k}": float((v / denom).cpu()) for k, v in meters.items()}


@torch.no_grad()
def validation_loss(
    model: ExternalBaselineModel,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    amp: bool,
) -> dict[str, float]:
    was = model.training
    model.eval()
    meters: dict[str, torch.Tensor] = {}
    count = 0
    for batch in tqdm(loader, desc="val-loss", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with _amp_context(device, amp):
            out = model(batch)
            losses = compute_external_baseline_losses(out, batch, cfg)
        _accumulate(meters, losses)
        count += 1
    if was:
        model.train()
    return _finalize_meters(meters, count, prefix="val_")


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
    parser = argparse.ArgumentParser(description="Train budget-compatible external baseline adapters on matched BDSE caches.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, nargs="+", default=["train_boston", "train_pittsburgh", "train_singapore", "train_vegas_2"])
    parser.add_argument("--preprocessed-dir", type=str, required=True)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-scenarios-per-split", type=int, default=None)
    parser.add_argument("--val-preprocessed-dir", type=str, default=None)
    parser.add_argument("--val-split", type=str, nargs="+", default=["val_tune"])
    parser.add_argument("--val-max-scenarios", type=int, default=500)
    parser.add_argument("--val-every-n-epochs", type=int, default=3)
    parser.add_argument("--val-mode", type=str, choices=["loss", "open_loop", "none"], default="loss")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--selection-metric", type=str, default="val_action_ce")
    parser.add_argument("--early-stop-patience", type=int, default=3, help="Validation events without improvement; 0 disables.")
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--log-every-n-steps", type=int, default=25)
    parser.add_argument("--optimizer-fused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true", help="Optional torch.compile; disabled by default for reproducibility.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not bool(cfg.get("external_baseline", {}).get("enabled", False)):
        raise ValueError("External baseline training requires external_baseline.enabled: true in the config.")
    variant = external_variant(cfg)
    if variant == "pdm_closed":
        raise ValueError("pdm_closed is a non-trainable rule-style adapter; run evaluation directly without training.")
    tcfg = cfg.get("training", {}) or {}
    ecfg = cfg.get("external_baseline", {}) or {}
    epochs = int(args.epochs if args.epochs is not None else ecfg.get("epochs", tcfg.get("epochs", 20)))
    batch_size = int(args.batch_size if args.batch_size is not None else ecfg.get("batch_size", tcfg.get("batch_size", 32)))
    num_workers = int(args.num_workers if args.num_workers is not None else ecfg.get("num_workers", tcfg.get("num_workers", 4)))
    lr = float(args.lr if args.lr is not None else ecfg.get("lr", tcfg.get("lr", 1e-4)))
    wd = float(args.weight_decay if args.weight_decay is not None else ecfg.get("weight_decay", tcfg.get("weight_decay", 1e-2)))
    grad_clip = float(args.grad_clip if args.grad_clip is not None else ecfg.get("grad_clip", tcfg.get("grad_clip", 5.0)))

    _seed_everything(args.seed)
    device = resolve_torch_device(args.device, context="external baseline training")
    configure_torch_for_device(device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_source = PreprocessedBDSEDataset(
        args.preprocessed_dir,
        split=args.split,
        max_scenarios=args.max_scenarios,
        max_scenarios_per_split=args.max_scenarios_per_split,
    )
    train_ds = ExternalBaselineDataset(train_source)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    collate_fn = partial(collate, cfg=cfg)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_fn,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    loader = DataLoader(train_ds, **loader_kwargs)

    val_loader = None
    val_dataset = None
    val_ds_for_manifest: ExternalBaselineDataset | None = None
    if args.val_mode != "none" and args.val_preprocessed_dir:
        val_source = PreprocessedBDSEDataset(args.val_preprocessed_dir, split=args.val_split, max_scenarios=args.val_max_scenarios)
        if args.val_mode == "loss":
            val_ds_for_manifest = ExternalBaselineDataset(val_source)
            val_workers = max(0, min(num_workers, 6))
            val_kwargs: dict[str, Any] = {
                "batch_size": batch_size,
                "shuffle": False,
                "num_workers": val_workers,
                "pin_memory": device.type == "cuda",
                "persistent_workers": val_workers > 0,
                "collate_fn": collate_fn,
                "worker_init_fn": _seed_worker,
            }
            if val_workers > 0:
                val_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
            val_loader = DataLoader(val_ds_for_manifest, **val_kwargs)
        else:
            val_dataset = val_source
            val_ds_for_manifest = ExternalBaselineDataset(val_source)

    training_manifest: dict[str, Any] = {
        "schema_version": 1,
        "variant": variant,
        "implementation": external_reference(variant),
        "seed": int(args.seed),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": _config_sha256(args.config),
        "train": _dataset_manifest(train_ds.paths, args.preprocessed_dir, list(args.split)),
        "validation": None if val_ds_for_manifest is None else _dataset_manifest(val_ds_for_manifest.paths, args.val_preprocessed_dir or "", list(args.val_split)),
        "protocol": {
            "max_scenarios": args.max_scenarios,
            "max_scenarios_per_split": args.max_scenarios_per_split,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": wd,
            "warmup_epochs": args.warmup_epochs,
            "scheduler": args.scheduler,
            "selection_metric": args.selection_metric,
            "val_every_n_epochs": args.val_every_n_epochs,
            "val_max_scenarios": args.val_max_scenarios,
        },
    }
    manifest_path = Path(args.output).with_suffix(".data_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(training_manifest, indent=2, sort_keys=True), encoding="utf-8")

    model: torch.nn.Module = ExternalBaselineModel(cfg).to(device)
    optimizer = _make_optimizer(model, lr=lr, weight_decay=wd, fused=bool(args.optimizer_fused), device=device)
    scheduler = _make_scheduler(optimizer, scheduler_name=args.scheduler, epochs=epochs, warmup_epochs=args.warmup_epochs)
    scaler = _make_scaler(device, args.amp)
    start_epoch = 0
    best_val = float("inf")
    if args.resume_from:
        start_epoch, best_val = _load_resume(args.resume_from, model, optimizer, scheduler)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        model = torch.compile(model)  # type: ignore[assignment]

    log_path = Path(args.log_file) if args.log_file else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = _best_checkpoint_path(args.output)
    no_improve_events = 0
    start_wall = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        model.train()
        meters: dict[str, torch.Tensor] = {}
        batch_count = 0
        pbar = tqdm(loader, desc=f"external-{variant} epoch {epoch + 1}/{epochs}", mininterval=1.0)
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, args.amp):
                out = model(batch)
                losses = compute_external_baseline_losses(out, batch, cfg)
                loss = losses["loss"]
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            _accumulate(meters, losses)
            batch_count += 1
            if step % max(1, args.log_every_n_steps) == 0 or step == len(loader):
                pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        metrics = _finalize_meters(meters, batch_count)
        validation_ran = (epoch + 1) % max(1, int(args.val_every_n_epochs)) == 0
        if validation_ran:
            if val_loader is not None:
                metrics.update(validation_loss(model, val_loader, cfg, device, amp=args.amp))
            elif val_dataset is not None:
                metrics.update(validation_open_loop(model, val_dataset, cfg, args.val_max_scenarios))

        if scheduler is not None:
            scheduler.step()
        selection_key = args.selection_metric if args.selection_metric in metrics else ("val_loss" if "val_loss" in metrics else "loss")
        score = float(metrics.get(selection_key, float("inf")))
        improved = validation_ran and math.isfinite(score) and score < best_val - 1e-8
        if improved:
            best_val = score
            no_improve_events = 0
            _save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                epoch=epoch,
                metrics=metrics,
                best_metric=best_val,
                selection_metric=selection_key,
                training_manifest=training_manifest,
            )
        elif validation_ran:
            no_improve_events += 1

        _save_checkpoint(
            args.output,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            epoch=epoch,
            metrics=metrics,
            best_metric=best_val,
            selection_metric=selection_key,
            training_manifest=training_manifest,
        )
        row = {
            "epoch": epoch + 1,
            "variant": variant,
            "implementation_label": external_reference(variant).get("implementation_label", variant),
            **metrics,
            "selection_metric": selection_key,
            "selection_value": score,
            "best_metric": best_val,
            "best_checkpoint": str(best_path),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_wall_s": time.perf_counter() - start_wall,
        }
        if log_path:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

        if (
            args.early_stop_patience > 0
            and epoch + 1 >= max(1, args.min_epochs)
            and no_improve_events >= args.early_stop_patience
        ):
            print(
                f"[external] early stop at epoch {epoch + 1}: {no_improve_events} validation events without improvement in {selection_key}",
                flush=True,
            )
            break

    if not best_path.is_file():
        # Handles val_mode=none or an interrupted run before the first validation.
        latest = torch_load_any(args.output, map_location="cpu")
        torch.save(latest, best_path)
    summary = {
        "variant": variant,
        "implementation": external_reference(variant),
        "output": str(Path(args.output).resolve()),
        "best_checkpoint": str(best_path.resolve()),
        "best_metric": best_val,
        "selection_metric": args.selection_metric,
        "data_manifest": str(manifest_path.resolve()),
        "wall_time_s": time.perf_counter() - start_wall,
    }
    Path(args.output).with_suffix(".training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
