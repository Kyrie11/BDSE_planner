from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib
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
from bdse.data.cache_schema import Sample
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.external_baselines.data import external_sample_to_model_inputs, load_external_training_sample_npz
from bdse.external_baselines.compact_cache import CompactBatchLoader, CompactExternalCache, CompactSampleDataset
from bdse.external_baselines.losses import compute_external_baseline_losses
from bdse.external_baselines.models import ExternalBaselineModel, external_reference, external_variant
from bdse.metrics.bdse_metrics import aggregate_metric_results, compute_bdse_diagnostics
from bdse.planner.nuplan_planner import BDSEPlannerCore, runtime_query_diagnostics
from bdse.utils import configure_torch_for_device, resolve_torch_device, torch_load_any


class ExternalBaselineDataset(Dataset):
    def __init__(self, source: PreprocessedBDSEDataset, *, include_label_future: bool = False):
        self.paths = [Path(p) for p in source.build_index()]
        self.include_label_future = bool(include_label_future)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Sample:
        return load_external_training_sample_npz(
            self.paths[idx],
            include_label_future=self.include_label_future,
        )


def _planner_supervision(cfg: dict[str, Any]) -> str:
    return str((cfg.get("external_baseline", {}) or {}).get("planner_supervision", "teacher_cost")).strip().lower()


def collate(samples: list[Sample], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    items = [external_sample_to_model_inputs(sample, cfg) for sample in samples]
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
    scaler: Any | None = None,
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
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        },
    }
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    if device.type == "cuda":
        payload["rng_state"]["torch_cuda_device"] = torch.cuda.get_rng_state(device)
    if scaler is not None:
        try:
            payload["scaler"] = scaler.state_dict()
        except Exception:
            pass
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, tmp)
    tmp.replace(p)


def _load_resume(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: Any | None = None,
    restore_rng: bool = True,
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
    if scaler is not None and isinstance(ckpt, dict) and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as exc:
            print(f"[external] AMP scaler resume skipped: {exc}", flush=True)
    if restore_rng and isinstance(ckpt, dict) and isinstance(ckpt.get("rng_state"), dict):
        rng = ckpt["rng_state"]
        try:
            if "python" in rng:
                random.setstate(rng["python"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "torch_cpu" in rng:
                torch.random.set_rng_state(rng["torch_cpu"])
            if "torch_cuda_device" in rng and torch.cuda.is_available():
                try:
                    device = next(model.parameters()).device
                except StopIteration:
                    device = torch.device("cuda:0")
                if device.type == "cuda":
                    torch.cuda.set_rng_state(rng["torch_cuda_device"], device=device)
        except Exception as exc:
            print(f"[external] RNG resume skipped: {exc}", flush=True)
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


def _progress_iter(iterable, *, total: int | None, desc: str, style: str, leave: bool = False):
    if style == "tqdm":
        return tqdm(iterable, total=total, desc=desc, leave=leave, mininterval=1.0)
    return iterable


@torch.inference_mode()
def validation_loss(
    model: ExternalBaselineModel,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    amp: bool,
    progress_style: str = "tqdm",
    progress_prefix: str = "val",
    log_every_n_steps: int = 100,
) -> dict[str, float]:
    was = model.training
    model.eval()
    meters: dict[str, torch.Tensor] = {}
    count = 0
    total = len(loader)
    started = time.perf_counter()
    if progress_style == "lines":
        print(f"[{progress_prefix}-start] batches={total}", flush=True)
    iterator = _progress_iter(loader, total=total, desc="val-loss", style=progress_style, leave=False)
    for step, batch in enumerate(iterator, start=1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with _amp_context(device, amp):
            out = model(batch)
            losses = compute_external_baseline_losses(out, batch, cfg)
        _accumulate(meters, losses)
        count += 1
        if progress_style == "lines" and (step % max(1, int(log_every_n_steps)) == 0 or step == total):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"[{progress_prefix}-progress] step={step}/{total} ({100.0 * step / max(total, 1):.1f}%) "
                f"loss={float(losses['loss'].detach().cpu()):.4f} rate={step / elapsed:.2f} batch/s elapsed={elapsed:.1f}s",
                flush=True,
            )
    if was:
        model.train()
    out = _finalize_meters(meters, count, prefix="val_")
    if progress_style == "lines":
        print(
            f"[{progress_prefix}-done] batches={count} elapsed={time.perf_counter() - started:.1f}s "
            f"val_loss={out.get('val_loss', float('nan')):.4f}",
            flush=True,
        )
    return out


@torch.inference_mode()
def validation_open_loop(
    model: ExternalBaselineModel,
    dataset: PreprocessedBDSEDataset,
    cfg: dict[str, Any],
    max_scenarios: int | None = None,
    *,
    progress_style: str = "tqdm",
    progress_prefix: str = "val-open-loop",
    log_every_n_steps: int = 100,
) -> dict[str, float]:
    was = model.training
    model.eval()
    core = BDSEPlannerCore(model=model, cfg=cfg)
    results = []
    n = len(dataset) if max_scenarios is None else min(len(dataset), int(max_scenarios))
    started = time.perf_counter()
    if progress_style == "lines":
        print(f"[{progress_prefix}-start] scenarios={n}", flush=True)
    iterator = _progress_iter(range(n), total=n, desc="val-open-loop", style=progress_style, leave=False)
    for i in iterator:
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
        done = i + 1
        if progress_style == "lines" and (done % max(1, int(log_every_n_steps)) == 0 or done == n):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"[{progress_prefix}-progress] scenarios={done}/{n} ({100.0 * done / max(n, 1):.1f}%) "
                f"rate={done / elapsed:.2f} scenario/s elapsed={elapsed:.1f}s",
                flush=True,
            )
    if was:
        model.train()
    out = aggregate_metric_results(results) if results else {}
    if progress_style == "lines":
        print(f"[{progress_prefix}-done] scenarios={n} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return {f"val_{k}": float(v) for k, v in out.items() if isinstance(v, (int, float, np.floating))}



def _startup_training_preflight(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    cfg: dict[str, Any],
    device: torch.device,
    amp: bool,
    sample_count: int,
    variant: str,
    budget: int,
) -> None:
    """Fail fast on cache/shape/model-contract problems before a long two-GPU run.

    This is deliberately eager (uncompiled) and does not step an optimizer, so it
    cannot change training semantics.  It catches the common failure modes that
    previously surfaced only as the wrapper's generic ``FAILED: <model>`` line:
    missing expert futures/teacher proposal labels, heterogeneous tensor shapes,
    invalid candidate targets, and GameFormer/DTPP forward/backward shape errors.
    """
    n = min(max(int(sample_count), 0), len(dataset))
    if n <= 0:
        return
    print(
        f"[train-preflight-start] variant={variant} B={budget} samples={n} device={device}",
        flush=True,
    )
    was_training = model.training
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        samples = [dataset[i] for i in range(n)]
        if samples and isinstance(samples[0], dict):
            batch = {k: torch.stack([s[k] for s in samples], dim=0) for k in samples[0]}
        else:
            batch = collate(samples, cfg)  # type: ignore[arg-type]
        batch = {k: v.to(device, non_blocking=False) for k, v in batch.items()}
        model.train()
        model.zero_grad(set_to_none=True)
        with _amp_context(device, amp):
            out = model(batch)
            losses = compute_external_baseline_losses(out, batch, cfg)
            loss = losses["loss"]
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError(f"non-finite startup loss: {float(loss.detach().cpu())}")
        # Exercise the exact autograd path used by training without changing any
        # parameter: no optimizer exists yet and gradients are immediately cleared.
        loss.backward()
        bad_grad = None
        for name, param in model.named_parameters():
            if param.grad is not None and not bool(torch.isfinite(param.grad).all().item()):
                bad_grad = name
                break
        if bad_grad is not None:
            raise FloatingPointError(f"non-finite gradient during startup preflight: {bad_grad}")
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        shapes = {k: tuple(v.shape) for k, v in batch.items() if k in {
            "candidate_valid", "candidate_numeric_features", "evidence_features",
            "evidence_proposal_features", "expert_candidate_index", "expert_candidate_cost",
            "oracle_selected_mask",
        }}
        print(
            f"[train-preflight-ok] variant={variant} B={budget} loss={float(loss.detach().cpu()):.6f} shapes={shapes}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[train-preflight-failed] variant={variant} B={budget} error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    finally:
        model.zero_grad(set_to_none=True)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        model.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()


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
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--compact-cache-dir", type=str, default=None)
    parser.add_argument("--val-compact-cache-dir", type=str, default=None)
    parser.add_argument("--compact-shuffle-mode", choices=["global", "block", "none"], default="global")
    parser.add_argument("--compact-block-size", type=int, default=4096)
    parser.add_argument("--compact-prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compact-host-cache", choices=["off", "auto", "on"], default="auto")
    parser.add_argument("--compact-host-reserve-gib", type=float, default=16.0)
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
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps; useful to match published effective batch sizes on fewer GPUs.")
    parser.add_argument("--log-every-n-steps", type=int, default=25)
    parser.add_argument(
        "--progress-style",
        choices=["tqdm", "lines", "none"],
        default="tqdm",
        help="Progress rendering. Use 'lines' for two concurrent GPU jobs so output stays readable and tee-friendly.",
    )
    parser.add_argument(
        "--startup-preflight-samples",
        type=int,
        default=2,
        help="Synchronously validate this many training samples plus one eager forward/backward before spawning the long training loop; 0 disables.",
    )
    parser.add_argument("--optimizer-fused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true", help="Compile the training/eval forward graph while saving the original uncompiled state_dict.")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="reduce-overhead")
    parser.add_argument("--compile-fallback", action=argparse.BooleanOptionalAction, default=True, help="Let torch.compile fall back to eager for unsupported subgraphs instead of aborting a long run.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=False)
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
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    budget = int((cfg.get("evidence", {}) or {}).get("budget", (cfg.get("external_baseline", {}) or {}).get("budget", -1)))

    # Resolve/skip a completed run before opening or materializing the large
    # compact cache. This keeps legacy SHARED_DATALOADER=0 runs resumable too.
    auto_resume_selected = False
    if args.resume_from is None and args.auto_resume:
        latest_candidate = Path(args.output)
        best_candidate = _best_checkpoint_path(args.output)
        if latest_candidate.is_file():
            args.resume_from = str(latest_candidate)
            auto_resume_selected = True
        elif best_candidate.is_file():
            args.resume_from = str(best_candidate)
            auto_resume_selected = True
    if auto_resume_selected and args.resume_from and Path(args.resume_from).is_file():
        resume_meta = torch_load_any(args.resume_from, map_location="cpu")
        if isinstance(resume_meta, dict):
            manifest = resume_meta.get("training_manifest")
            if isinstance(manifest, dict):
                stored_hash = str(manifest.get("config_sha256", ""))
                current_hash = _config_sha256(args.config)
                if stored_hash and stored_hash != current_hash:
                    raise ValueError(
                        f"refusing auto-resume from config-mismatched checkpoint {args.resume_from}: "
                        f"stored={stored_hash} current={current_hash}"
                    )
            old_cfg = resume_meta.get("cfg")
            if isinstance(old_cfg, dict):
                old_variant = external_variant(old_cfg)
                old_budget = int((old_cfg.get("evidence", {}) or {}).get("budget", -1))
                old_refine = str((old_cfg.get("external_baseline", {}) or {}).get("refinement_mode", "legacy_repeated_encoder"))
                new_refine = str((cfg.get("external_baseline", {}) or {}).get("refinement_mode", "legacy_repeated_encoder"))
                if (old_variant, old_budget, old_refine) != (variant, budget, new_refine):
                    raise ValueError(
                        f"refusing incompatible resume {args.resume_from}: "
                        f"old={(old_variant, old_budget, old_refine)} new={(variant, budget, new_refine)}"
                    )
            completed_epochs = int(resume_meta.get("epoch", -1)) + 1
            if completed_epochs >= epochs:
                import shutil

                source = Path(args.resume_from)
                latest = Path(args.output)
                best = _best_checkpoint_path(args.output)
                latest.parent.mkdir(parents=True, exist_ok=True)
                if not latest.is_file():
                    shutil.copy2(source, latest)
                if not best.is_file():
                    shutil.copy2(source, best)
                print(
                    f"[train-resume-complete] variant={variant} B={budget} checkpoint={source} "
                    f"completed_epochs={completed_epochs} target_epochs={epochs}; reusing existing checkpoint",
                    flush=True,
                )
                return

    _seed_everything(args.seed)
    device = resolve_torch_device(args.device, context="external baseline training")
    configure_torch_for_device(device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # A30/Ampere can accelerate nn.MultiheadAttention/TransformerEncoder via
        # scaled-dot-product attention.  These calls are guarded for older torch.
        for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
            fn = getattr(torch.backends.cuda, name, None)
            if callable(fn):
                fn(True)
        props = torch.cuda.get_device_properties(device)
        print(
            f"[train-env] torch={torch.__version__} torch_cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} "
            f"gpu={props.name!r} capability={props.major}.{props.minor} total_mem_gib={props.total_memory / (1024**3):.1f} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
            f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}",
            flush=True,
        )
    else:
        print(f"[train-env] torch={torch.__version__} device={device}", flush=True)

    use_label_future = _planner_supervision(cfg) == "expert_imitation"
    collate_fn = partial(collate, cfg=cfg)
    compact_train = None
    if args.compact_cache_dir:
        if args.max_scenarios_per_split:
            raise ValueError("--compact-cache-dir does not support --max-scenarios-per-split")
        compact_train = CompactExternalCache.open(args.compact_cache_dir)
        compact_train.assert_compatible(cfg)
        compact_train = compact_train.materialize_host(
            mode=args.compact_host_cache,
            reserve_gib=args.compact_host_reserve_gib,
        )
        limit = int(args.max_scenarios) if args.max_scenarios is not None and int(args.max_scenarios) > 0 else None
        train_ds = CompactSampleDataset(compact_train, budget=budget, limit=limit)
        loader = CompactBatchLoader(
            compact_train, budget=budget, batch_size=batch_size, shuffle=True, seed=args.seed,
            pin_memory=device.type == "cuda", prefetch=args.compact_prefetch,
            shuffle_mode=args.compact_shuffle_mode, block_size=args.compact_block_size, limit=limit,
        )
        print(
            f"[train-data] compact_cache=1 storage={compact_train.storage_kind} path={compact_train.root} samples={len(train_ds)} "
            f"shuffle={args.compact_shuffle_mode} prefetch={args.compact_prefetch}", flush=True,
        )
    else:
        train_source = PreprocessedBDSEDataset(
            args.preprocessed_dir,
            split=args.split,
            max_scenarios=args.max_scenarios,
            max_scenarios_per_split=args.max_scenarios_per_split,
        )
        train_ds = ExternalBaselineDataset(train_source, include_label_future=use_label_future)
        generator = torch.Generator()
        generator.manual_seed(args.seed)
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
    val_ds_for_manifest = None
    compact_val = None
    if args.val_mode != "none" and args.val_preprocessed_dir:
        val_max_scenarios = None if args.val_max_scenarios is not None and int(args.val_max_scenarios) <= 0 else args.val_max_scenarios
        if args.val_mode == "loss":
            if args.val_compact_cache_dir:
                compact_val = CompactExternalCache.open(args.val_compact_cache_dir)
                compact_val.assert_compatible(cfg)
                compact_val = compact_val.materialize_host(
                    mode=args.compact_host_cache,
                    reserve_gib=args.compact_host_reserve_gib,
                )
                val_ds_for_manifest = CompactSampleDataset(compact_val, budget=budget, limit=val_max_scenarios)
                val_loader = CompactBatchLoader(
                    compact_val, budget=budget, batch_size=batch_size, shuffle=False, seed=args.seed,
                    pin_memory=device.type == "cuda", prefetch=args.compact_prefetch,
                    shuffle_mode="none", block_size=args.compact_block_size, limit=val_max_scenarios,
                )
                print(f"[val-data] compact_cache=1 storage={compact_val.storage_kind} path={compact_val.root} samples={len(val_ds_for_manifest)}", flush=True)
            else:
                val_source = PreprocessedBDSEDataset(args.val_preprocessed_dir, split=args.val_split, max_scenarios=val_max_scenarios)
                val_ds_for_manifest = ExternalBaselineDataset(val_source, include_label_future=use_label_future)
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
            val_source = PreprocessedBDSEDataset(args.val_preprocessed_dir, split=args.val_split, max_scenarios=val_max_scenarios)
            val_dataset = val_source
            val_ds_for_manifest = ExternalBaselineDataset(val_source, include_label_future=use_label_future)

    if compact_train is not None:
        train_manifest = compact_train.source_manifest()
    else:
        train_manifest = _dataset_manifest(train_ds.paths, args.preprocessed_dir, list(args.split))
    if compact_val is not None:
        validation_manifest = compact_val.source_manifest()
    elif val_ds_for_manifest is not None and hasattr(val_ds_for_manifest, "paths"):
        validation_manifest = _dataset_manifest(val_ds_for_manifest.paths, args.val_preprocessed_dir or "", list(args.val_split))
    else:
        validation_manifest = None

    training_manifest: dict[str, Any] = {
        "schema_version": 1,
        "variant": variant,
        "implementation": external_reference(variant),
        "seed": int(args.seed),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": _config_sha256(args.config),
        "train": train_manifest,
        "validation": validation_manifest,
        "protocol": {
            "max_scenarios": args.max_scenarios,
            "max_scenarios_per_split": args.max_scenarios_per_split,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size_per_process": batch_size * grad_accum_steps,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": wd,
            "warmup_epochs": args.warmup_epochs,
            "scheduler": args.scheduler,
            "selection_metric": args.selection_metric,
            "planner_supervision": _planner_supervision(cfg),
            "val_every_n_epochs": args.val_every_n_epochs,
            "torch_compile": bool(args.compile),
            "torch_compile_mode": args.compile_mode if args.compile else "disabled",
            "compact_mmap_cache": compact_train is not None,
            "compact_shuffle_mode": args.compact_shuffle_mode if compact_train is not None else None,
            "val_max_scenarios": None if args.val_max_scenarios is not None and int(args.val_max_scenarios) <= 0 else args.val_max_scenarios,
        },
    }
    manifest_path = Path(args.output).with_suffix(".data_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(training_manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Keep `model` as the canonical eager module for optimizer/checkpoint state.
    # `runtime_model` may be an OptimizedModule but shares the same parameters.
    model: torch.nn.Module = ExternalBaselineModel(cfg).to(device)
    param_count = int(sum(int(p.numel()) for p in model.parameters()))
    print(
        f"[train-init] variant={variant} B={budget} device={device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"params={param_count} train_samples={len(train_ds)} val_samples={0 if val_ds_for_manifest is None else len(val_ds_for_manifest)} "
        f"batch={batch_size} grad_accum={grad_accum_steps} effective_batch={batch_size * grad_accum_steps} workers={num_workers} "
        f"amp={bool(args.amp)} compile={bool(args.compile)}",
        flush=True,
    )
    _startup_training_preflight(
        model=model,
        dataset=train_ds,
        cfg=cfg,
        device=device,
        amp=bool(args.amp),
        sample_count=int(args.startup_preflight_samples),
        variant=variant,
        budget=budget,
    )
    optimizer = _make_optimizer(model, lr=lr, weight_decay=wd, fused=bool(args.optimizer_fused), device=device)
    scheduler = _make_scheduler(optimizer, scheduler_name=args.scheduler, epochs=epochs, warmup_epochs=args.warmup_epochs)
    scaler = _make_scaler(device, args.amp)
    start_epoch = 0
    best_val = float("inf")
    if args.resume_from:
        start_epoch, best_val = _load_resume(args.resume_from, model, optimizer, scheduler, scaler)
        print(
            f"[train-resume] variant={variant} B={budget} checkpoint={args.resume_from} "
            f"next_epoch={start_epoch + 1}/{epochs} best={best_val:.6f}",
            flush=True,
        )
    runtime_model: torch.nn.Module = model
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        if args.compile_fallback:
            try:
                dynamo = importlib.import_module("torch._dynamo")
                dynamo.config.suppress_errors = True
            except Exception:
                pass
        runtime_model = torch.compile(model, mode=args.compile_mode)  # type: ignore[assignment]
        print(
            f"[train-compile] variant={variant} B={budget} mode={args.compile_mode}; first batch may pause while PyTorch compiles the graph.",
            flush=True,
        )

    log_path = Path(args.log_file) if args.log_file else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = _best_checkpoint_path(args.output)
    no_improve_events = 0
    start_wall = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        model.train()
        runtime_model.train()
        meters: dict[str, torch.Tensor] = {}
        batch_count = 0
        epoch_started = time.perf_counter()
        if args.progress_style == "lines":
            print(
                f"[train-epoch-start] variant={variant} B={budget} epoch={epoch + 1}/{epochs} batches={len(loader)}",
                flush=True,
            )
        pbar = _progress_iter(
            loader,
            total=len(loader),
            desc=f"external-{variant} B{budget} epoch {epoch + 1}/{epochs}",
            style=args.progress_style,
            leave=True,
        )
        optimizer.zero_grad(set_to_none=True)
        total_steps = len(loader)
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            group_start = ((step - 1) // grad_accum_steps) * grad_accum_steps
            group_size = min(grad_accum_steps, total_steps - group_start)
            with _amp_context(device, args.amp):
                out = runtime_model(batch)
                losses = compute_external_baseline_losses(out, batch, cfg)
                loss = losses["loss"]
                backward_loss = loss / float(group_size)
            scaler.scale(backward_loss).backward()
            do_step = (step % grad_accum_steps == 0) or (step == total_steps)
            if do_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            _accumulate(meters, losses)
            batch_count += 1
            if step % max(1, args.log_every_n_steps) == 0 or step == total_steps:
                current_loss = float(loss.detach().cpu())
                if args.progress_style == "tqdm":
                    pbar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                elif args.progress_style == "lines":
                    elapsed = max(time.perf_counter() - epoch_started, 1e-9)
                    seen = min(step * batch_size, len(train_ds))
                    current_ce = float(losses["action_ce"].detach().cpu())
                    current_reg = float(losses["cost_reg"].detach().cpu())
                    current_prop = float(losses["proposal_bce"].detach().cpu())
                    current_deep = float(losses["deep_action_ce"].detach().cpu())
                    print(
                        f"[train-progress] variant={variant} B={budget} epoch={epoch + 1}/{epochs} "
                        f"step={step}/{total_steps} ({100.0 * step / max(total_steps, 1):.1f}%) "
                        f"samples~={seen}/{len(train_ds)} loss={current_loss:.4f} action_ce={current_ce:.4f} "
                        f"cost_reg={current_reg:.4f} proposal_bce={current_prop:.4f} deep_ce={current_deep:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} rate={step / elapsed:.2f} batch/s elapsed={elapsed:.1f}s",
                        flush=True,
                    )

        metrics = _finalize_meters(meters, batch_count)
        validation_ran = (epoch + 1) % max(1, int(args.val_every_n_epochs)) == 0
        if validation_ran:
            if val_loader is not None:
                metrics.update(
                    validation_loss(
                        runtime_model,
                        val_loader,
                        cfg,
                        device,
                        amp=args.amp,
                        progress_style=args.progress_style,
                        progress_prefix=f"val-{variant}-B{budget}-e{epoch + 1}",
                        log_every_n_steps=max(1, args.log_every_n_steps),
                    )
                )
            elif val_dataset is not None:
                metrics.update(
                    validation_open_loop(
                        runtime_model,
                        val_dataset,
                        cfg,
                        args.val_max_scenarios,
                        progress_style=args.progress_style,
                        progress_prefix=f"val-open-{variant}-B{budget}-e{epoch + 1}",
                        log_every_n_steps=max(1, args.log_every_n_steps),
                    )
                )

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
                scaler=scaler,
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
            scaler=scaler,
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
        if args.progress_style != "none":
            print(
                f"[train-epoch-done] variant={variant} B={budget} epoch={epoch + 1}/{epochs} "
                f"loss={metrics.get('loss', float('nan')):.4f} val_loss={metrics.get('val_loss', float('nan')):.4f} "
                f"selection={selection_key}:{score:.6f} best={best_val:.6f} improved={improved} "
                f"epoch_wall={time.perf_counter() - epoch_started:.1f}s total_wall={time.perf_counter() - start_wall:.1f}s",
                flush=True,
            )

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
    print(
        f"[train-complete] variant={variant} B={budget} best_metric={best_val:.6f} "
        f"checkpoint={best_path.resolve()} wall={summary['wall_time_s'] / 60.0:.1f}min",
        flush=True,
    )


if __name__ == "__main__":
    main()
