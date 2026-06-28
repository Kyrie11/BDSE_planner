from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import json
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.cache_schema import Sample, load_sample_npz
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, PreprocessedBDSEDataset
from bdse.data.tensorizer import sample_to_model_inputs
from bdse.data.quality import quality_decision
from bdse.metrics.bdse_metrics import compute_bdse_diagnostics
from bdse.model.bdse_model import BDSEModel
from bdse.model.losses import compute_bdse_losses
from bdse.planner.nuplan_planner import BDSEPlannerCore, runtime_query_diagnostics


class OnTheFlyDataset(Dataset):
    def __init__(self, source: NuPlanBDSEDataset | PreprocessedBDSEDataset):
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> Sample:
        if isinstance(self.source, PreprocessedBDSEDataset):
            # Training does not consume label-only futures or candidate JSON metadata.
            # Avoid moving those large unused arrays through DataLoader workers.
            return load_sample_npz(
                self.source.build_index()[idx],
                include_label_future=False,
                include_candidate_metadata=False,
            )
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




def _checkpoint_stem(output: str | Path) -> Path:
    out_path = Path(output)
    return out_path.with_suffix("") if out_path.suffix else out_path


def _checkpoint_paths(args: argparse.Namespace, epoch: int | None = None) -> dict[str, Path]:
    stem = _checkpoint_stem(args.output)
    ckpt_dir = Path(args.checkpoint_dir) if getattr(args, "checkpoint_dir", None) else stem.parent / "checkpoints"
    paths = {
        "final": Path(args.output),
        "latest": stem.parent / f"{stem.name}.latest.pt",
        "best": stem.parent / f"{stem.name}.best.pt",
    }
    if epoch is not None:
        # epoch is zero-based internally; filenames are one-based for readability.
        paths["epoch"] = ckpt_dir / f"{stem.name}.epoch_{epoch + 1:04d}.pt"
    return paths



def _torch_load_any(path: str | Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _torch_save_atomic(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Any) -> None:
    if not isinstance(state, dict):
        return
    try:
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:
        print(f"[bdse] warning: failed to restore RNG state: {type(exc).__name__}: {exc}", flush=True)


def _make_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
    best_metric: float | None,
    best_epoch: int | None,
    world_size: int,
) -> dict[str, Any]:
    raw_model = model.module if isinstance(model, DDP) else model
    return {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if hasattr(scaler, "state_dict") else {},
        "cfg": cfg,
        "args": vars(args),
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "metrics": {str(k): float(v) for k, v in metrics.items()},
        "best_metric": None if best_metric is None else float(best_metric),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "world_size": int(world_size),
        "rng_state": _rng_state(),
    }


def _load_checkpoint_if_requested(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    is_main: bool,
) -> tuple[int, float | None, int | None, Path | None]:
    if not args.resume and not args.resume_from:
        return 0, None, None, None
    paths = _checkpoint_paths(args)
    ckpt_path = Path(args.resume_from) if args.resume_from else paths["latest"]
    if args.resume and not ckpt_path.exists() and paths["final"].exists():
        # Backward compatibility: old runs only wrote the final output file.
        ckpt_path = paths["final"]
    if not ckpt_path.exists():
        if args.resume_from:
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        if is_main:
            print(f"[bdse] --resume requested but no checkpoint found at {paths['latest']}; starting from scratch.", flush=True)
        return 0, None, None, None

    ckpt = _torch_load_any(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    raw_model = model.module if isinstance(model, DDP) else model
    missing, unexpected = raw_model.load_state_dict(state, strict=False)
    if is_main and (missing or unexpected):
        print(
            f"[bdse] loaded checkpoint with non-strict state_dict: missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}",
            flush=True,
        )
    if isinstance(ckpt, dict) and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            if is_main:
                print(f"[bdse] warning: optimizer state was not restored: {type(exc).__name__}: {exc}", flush=True)
    if isinstance(ckpt, dict) and "scaler" in ckpt and hasattr(scaler, "load_state_dict"):
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as exc:
            if is_main:
                print(f"[bdse] warning: AMP scaler state was not restored: {type(exc).__name__}: {exc}", flush=True)
    _restore_rng_state(ckpt.get("rng_state") if isinstance(ckpt, dict) else None)
    start_epoch = int(ckpt.get("next_epoch", int(ckpt.get("epoch", -1)) + 1)) if isinstance(ckpt, dict) else 0
    best_metric = ckpt.get("best_metric") if isinstance(ckpt, dict) else None
    best_metric = None if best_metric is None else float(best_metric)
    best_epoch = ckpt.get("best_epoch") if isinstance(ckpt, dict) else None
    best_epoch = None if best_epoch is None else int(best_epoch)
    if is_main:
        print(f"[bdse] resumed from {ckpt_path} at epoch={start_epoch}", flush=True)
    return start_epoch, best_metric, best_epoch, ckpt_path


def _aggregate_meters(meters: dict[str, list[float]], device: torch.device, distributed: bool) -> dict[str, float]:
    keys = sorted(meters)
    if not keys:
        return {}
    local = torch.zeros((len(keys), 2), dtype=torch.float64, device=device)
    for i, k in enumerate(keys):
        vals = [float(v) for v in meters.get(k, []) if np.isfinite(float(v))]
        if vals:
            local[i, 0] = float(np.sum(vals, dtype=np.float64))
            local[i, 1] = float(len(vals))
    if distributed and dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    out: dict[str, float] = {}
    for i, k in enumerate(keys):
        count = float(local[i, 1].item())
        out[k] = float(local[i, 0].item() / max(count, 1.0))
    return out


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{k}": float(v) for k, v in metrics.items()}


def _validation_bdse_score(metrics: dict[str, float]) -> float:
    """Single scalar for validation-best checkpointing.

    Closed-loop nuPlan metrics are the final reporting target, but they are too
    expensive and simulator-dependent to run after every training epoch.  This
    score is therefore an open-loop proxy aligned with the paper's diagnostic
    claim: first preserve the teacher decision, then reduce teacher-cost regret,
    and finally reward evidence/margin quality.  The large coefficient on action
    match makes a 1 percentage point decision-sufficiency gain more important
    than small changes in auxiliary diagnostics, while regret/margin terms still
    break ties when match has plateaued.
    """
    def finite(name: str, default: float = 0.0) -> float:
        v = float(metrics.get(name, default))
        return v if np.isfinite(v) else default

    match = finite("val_teacher_action_match", finite("val_decision_sufficiency", 0.0))
    regret = max(0.0, finite("val_teacher_regret", 1e6))
    suff = finite("val_evidence_sufficiency", 0.0)
    hard = finite("val_hard_evidence_recall", 0.0)
    budget_full = finite("val_budget_vs_full_match", 0.0)
    full_match = finite("val_full_interface_action_match", 0.0)
    margin_err = max(0.0, finite("val_preserved_margin_error", 1e6))
    return float(
        200.0 * match
        + 80.0 * full_match
        + 10.0 * suff
        + 10.0 * hard
        + 5.0 * budget_full
        - 5.0 * np.log1p(regret / 1000.0)
        - 0.5 * np.log1p(margin_err / 1000.0)
    )


def _resolve_best_metric(metrics: dict[str, float], args: argparse.Namespace) -> tuple[str, float, str]:
    requested = str(getattr(args, "best_metric", "auto"))
    requested_mode = str(getattr(args, "best_mode", "auto"))
    if requested != "auto":
        metric_name = requested
        value = float(metrics.get(metric_name, float("nan")))
        mode = requested_mode if requested_mode in {"min", "max"} else ("min" if "loss" in metric_name or "regret" in metric_name or "error" in metric_name else "max")
        return metric_name, value, mode
    if "val_bdse_score" in metrics and np.isfinite(float(metrics["val_bdse_score"])):
        return "val_bdse_score", float(metrics["val_bdse_score"]), "max"
    if "val_teacher_regret" in metrics and np.isfinite(float(metrics["val_teacher_regret"])):
        return "val_teacher_regret", float(metrics["val_teacher_regret"]), "min"
    if "val_loss" in metrics and np.isfinite(float(metrics["val_loss"])):
        return "val_loss", float(metrics["val_loss"]), "min"
    return "loss", float(metrics.get("loss", float("nan"))), "min"


def _make_preprocessed_dataset(
    *,
    preprocessed_dir: str | None,
    splits: list[str],
    max_scenarios: int | None,
    max_scenarios_per_split: int | None,
    cfg: dict[str, Any],
    max_files: int | None = None,
    for_training: bool = False,
) -> NuPlanBDSEDataset | PreprocessedBDSEDataset:
    if preprocessed_dir:
        dataset = PreprocessedBDSEDataset(
            preprocessed_dir,
            split=splits,
            max_scenarios=max_scenarios,
            max_scenarios_per_split=max_scenarios_per_split,
        )
        if for_training:
            _apply_training_quality_filter(dataset, cfg)
        return dataset
    if len(splits) != 1:
        raise ValueError("On-the-fly mode supports one split at a time; preprocess first to use multiple split folders.")
    return NuPlanBDSEDataset(cfg, split=splits[0], max_files=max_files, max_scenarios=max_scenarios, use_devkit=True)


def _make_loader(
    *,
    dataset: NuPlanBDSEDataset | PreprocessedBDSEDataset,
    cfg: dict[str, Any],
    batch_size: int,
    num_workers: int,
    cuda_available: bool,
    distributed: bool,
    world_size: int,
    global_rank: int,
    shuffle: bool,
    seed: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    wrapped = OnTheFlyDataset(dataset)
    sampler = DistributedSampler(wrapped, num_replicas=world_size, rank=global_rank, shuffle=shuffle, seed=seed) if distributed else None
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle and sampler is None,
        "num_workers": num_workers,
        "pin_memory": bool(cfg["training"].get("pin_memory", cuda_available) and cuda_available),
        "persistent_workers": num_workers > 0,
        "collate_fn": lambda x: collate(x, cfg),
        "sampler": sampler,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg["training"].get("prefetch_factor", 1))
    return DataLoader(wrapped, **loader_kwargs), sampler


@torch.no_grad()
def _run_validation_loss(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    distributed: bool,
    is_main: bool,
    epoch: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    meters: dict[str, list[float]] = {}
    for batch in tqdm(loader, desc=f"val-loss {epoch}", disable=not is_main):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch)
        losses = compute_bdse_losses(out, batch, cfg)
        for k, v in losses.items():
            meters.setdefault(k, []).append(float(v.detach().cpu()))
    if was_training:
        model.train()
    return _prefix_metrics(_aggregate_meters(meters, device, distributed), "val_")


def _iter_distributed_indices(n: int, distributed: bool, world_size: int, global_rank: int) -> range:
    if distributed:
        return range(global_rank, n, world_size)
    return range(n)


@torch.no_grad()
def _run_validation_open_loop(
    *,
    model: torch.nn.Module,
    dataset: NuPlanBDSEDataset | PreprocessedBDSEDataset,
    cfg: dict[str, Any],
    device: torch.device,
    distributed: bool,
    world_size: int,
    global_rank: int,
    is_main: bool,
    epoch: int,
    strict: bool,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    core = BDSEPlannerCore(model=raw_model, cfg=cfg)
    meters: dict[str, list[float]] = {}
    failed = 0
    indices = list(_iter_distributed_indices(len(dataset), distributed, world_size, global_rank))
    for idx in tqdm(indices, desc=f"val-open-loop {epoch}", disable=not is_main):
        try:
            sample = dataset[int(idx)]
            pred, sel, tour, _ = core._run_certificate_stage(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
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
                query_diagnostics=runtime_query_diagnostics(pred, sel.selected),
            )
            for k, v in diag.values.items():
                meters.setdefault(k, []).append(float(v))
        except Exception:
            failed += 1
            if strict:
                raise
    metrics = _prefix_metrics(_aggregate_meters(meters, device, distributed), "val_")
    # Aggregate failure counts explicitly; _aggregate_meters would average them.
    fail_t = torch.tensor([float(failed), float(len(indices))], dtype=torch.float64, device=device)
    if distributed and dist.is_initialized():
        dist.all_reduce(fail_t, op=dist.ReduceOp.SUM)
    metrics["val_open_loop_failed"] = float(fail_t[0].item())
    metrics["val_open_loop_count"] = float(fail_t[1].item())
    metrics["val_bdse_score"] = _validation_bdse_score(metrics)
    if was_training:
        model.train()
    return metrics


def _is_better(metric: float, best: float | None, mode: str) -> bool:
    if best is None or not np.isfinite(best):
        return True
    if mode == "max":
        return metric > best
    return metric < best


def _save_training_checkpoints(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    cfg: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
    best_metric: float | None,
    best_epoch: int | None,
    world_size: int,
    is_main: bool,
) -> tuple[float | None, int | None]:
    metric_name, metric, metric_mode = _resolve_best_metric(metrics, args)
    improved = np.isfinite(metric) and _is_better(metric, best_metric, metric_mode)
    new_best_metric = metric if improved else best_metric
    new_best_epoch = int(epoch) if improved else best_epoch
    ckpt = _make_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        cfg=cfg,
        args=args,
        epoch=epoch,
        metrics=metrics,
        best_metric=new_best_metric,
        best_epoch=new_best_epoch,
        world_size=world_size,
    )
    if is_main:
        paths = _checkpoint_paths(args, epoch)
        _torch_save_atomic(ckpt, paths["latest"])
        if int(args.save_every_n_epochs) > 0 and ((epoch + 1) % int(args.save_every_n_epochs) == 0):
            _torch_save_atomic(ckpt, paths["epoch"])
        if bool(args.save_best) and improved:
            _torch_save_atomic(ckpt, paths["best"])
        print(
            f"[bdse] saved latest={paths['latest']} "
            f"epoch_ckpt={paths.get('epoch') if int(args.save_every_n_epochs) > 0 else '-'} "
            f"best={paths['best'] if improved and bool(args.save_best) else '(unchanged)'} "
            f"best_metric={metric_name} mode={metric_mode} best_value={new_best_metric}",
            flush=True,
        )
    return new_best_metric, new_best_epoch

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, nargs="+", default=["train"], help="One or more preprocessed splits/folders, e.g. train or train_1 train_2.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None, help="Total cap after split selection. With multiple concrete splits, the cap is balanced across splits.")
    parser.add_argument("--max-scenarios-per-split", type=int, default=None, help="Optional per-split cap for multi-city/cache training.")
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
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None, help="Local rank passed by torchrun; normally inferred from LOCAL_RANK.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch factor when num_workers > 0. Use 1 to reduce host/pinned-memory pressure.")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pinned host memory for DataLoader batches.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint derived from --output, or from --resume-from when provided.")
    parser.add_argument("--resume-from", type=str, default=None, help="Explicit checkpoint path to resume from.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for per-epoch checkpoints. Defaults to <output_dir>/checkpoints.")
    parser.add_argument("--save-every-n-epochs", type=int, default=1, help="Save an epoch checkpoint every N epochs. Set 0 to disable per-epoch files.")
    parser.add_argument("--save-best", dest="save_best", action="store_true", default=True, help="Save <output_stem>.best.pt using --best-metric.")
    parser.add_argument("--no-save-best", dest="save_best", action="store_false", help="Disable best checkpoint saving.")
    parser.add_argument("--best-metric", type=str, default="auto", help="Metric used for best checkpoint selection. 'auto' prefers val_bdse_score, then val_teacher_regret, then val_loss, then loss.")
    parser.add_argument("--best-mode", type=str, default="auto", choices=["auto", "min", "max"], help="Whether lower or higher --best-metric is better. Use auto for the default validation-aware behavior.")
    parser.add_argument("--val-split", type=str, nargs="+", default=None, help="Optional validation split/folder(s), e.g. val or val_vegas. Enables validation-best checkpoints.")
    parser.add_argument("--val-preprocessed-dir", type=str, default=None, help="Validation cache root. Defaults to --preprocessed-dir.")
    parser.add_argument("--val-max-scenarios", type=int, default=None, help="Cap validation samples, e.g. 1000 for fast per-epoch validation.")
    parser.add_argument("--val-max-scenarios-per-split", type=int, default=None, help="Optional validation per-split cap for multi-city validation.")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Validation loss batch size. Defaults to --batch-size / training batch_size.")
    parser.add_argument("--val-num-workers", type=int, default=None, help="Validation DataLoader workers for val loss. Defaults to training num_workers.")
    parser.add_argument("--val-every-n-epochs", type=int, default=1, help="Run validation every N epochs. Set 0 to disable validation even when --val-split is provided.")
    parser.add_argument("--val-mode", type=str, default="open_loop", choices=["loss", "open_loop", "both"], help="Validation signal. open_loop computes BDSE decision diagnostics; both also reports val loss.")
    parser.add_argument("--val-strict", action="store_true", help="Raise validation exceptions instead of counting failed validation samples.")
    parser.add_argument("--log-file", type=str, default=None, help="Optional JSONL file for per-epoch train/validation metrics. Defaults to <output_stem>.train_log.jsonl.")
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
    if args.prefetch_factor is not None:
        cfg["training"]["prefetch_factor"] = max(1, int(args.prefetch_factor))
    if args.no_pin_memory:
        cfg["training"]["pin_memory"] = False
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    seed = int(cfg.get("seed", 17))
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        cuda_available = False
        print(f"[bdse] CUDA availability check failed; falling back to CPU: {type(exc).__name__}: {exc}", flush=True)
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank is not None else 0))
    global_rank = int(os.environ.get("RANK", "0"))
    if distributed:
        if not cuda_available:
            raise RuntimeError("Distributed CUDA training was requested by torchrun, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    is_main = global_rank == 0
    splits = args.split
    dataset = _make_preprocessed_dataset(
        preprocessed_dir=args.preprocessed_dir,
        splits=splits,
        max_scenarios=args.max_scenarios,
        max_scenarios_per_split=args.max_scenarios_per_split,
        cfg=cfg,
        max_files=args.max_files,
        for_training=True,
    )
    dataset_len = len(dataset)
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 0))
    if is_main:
        print(
            f"[bdse] dataset_samples={dataset_len} splits={splits} batch_size_per_process={batch_size} "
            f"world_size={world_size} global_batch_size={batch_size * world_size} "
            f"num_workers={num_workers} max_scenarios={args.max_scenarios} "
            f"max_scenarios_per_split={args.max_scenarios_per_split}",
            flush=True,
        )
    loader, sampler = _make_loader(
        dataset=dataset,
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        cuda_available=cuda_available,
        distributed=distributed,
        world_size=world_size,
        global_rank=global_rank,
        shuffle=True,
        seed=seed,
    )

    val_dataset: NuPlanBDSEDataset | PreprocessedBDSEDataset | None = None
    val_loader: DataLoader | None = None
    val_sampler: DistributedSampler | None = None
    validation_enabled = bool(args.val_split) and int(args.val_every_n_epochs) > 0
    if validation_enabled:
        val_preprocessed_dir = args.val_preprocessed_dir or args.preprocessed_dir
        val_dataset = _make_preprocessed_dataset(
            preprocessed_dir=val_preprocessed_dir,
            splits=list(args.val_split),
            max_scenarios=args.val_max_scenarios,
            max_scenarios_per_split=args.val_max_scenarios_per_split,
            cfg=cfg,
            max_files=args.max_files,
            for_training=False,
        )
        if args.val_mode in {"loss", "both"}:
            val_loader, val_sampler = _make_loader(
                dataset=val_dataset,
                cfg=cfg,
                batch_size=int(args.val_batch_size or batch_size),
                num_workers=int(num_workers if args.val_num_workers is None else max(0, int(args.val_num_workers))),
                cuda_available=cuda_available,
                distributed=distributed,
                world_size=world_size,
                global_rank=global_rank,
                shuffle=False,
                seed=seed,
            )
        if is_main:
            print(
                f"[bdse] validation_samples={len(val_dataset)} val_split={args.val_split} "
                f"val_mode={args.val_mode} val_every_n_epochs={args.val_every_n_epochs} "
                f"val_best_metric={args.best_metric}",
                flush=True,
            )
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is false. "
                "Fix the NVIDIA driver / PyTorch CUDA build mismatch, or run with --device cpu."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if cuda_available else "cpu")
    if is_main:
        print(f"[bdse] device={device} distributed={distributed} cuda_available={cuda_available} amp={bool(args.amp and device.type == 'cuda')}", flush=True)
    model = BDSEModel(cfg).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    use_amp = bool(args.amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch, best_metric, best_epoch, _ = _load_checkpoint_if_requested(
        args=args, model=model, optimizer=opt, scaler=scaler, device=device, is_main=is_main
    )
    log_file = Path(args.log_file) if args.log_file else _checkpoint_stem(args.output).parent / f"{_checkpoint_stem(args.output).name}.train_log.jsonl"
    if is_main and start_epoch == 0:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("", encoding="utf-8")
    total_epochs = int(cfg["training"]["epochs"])
    for epoch in range(start_epoch, total_epochs):
        cfg["training"]["current_epoch"] = int(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        meters: dict[str, list[float]] = {}
        for batch in tqdm(loader, desc=f"epoch {epoch}", disable=not is_main):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)
            with autocast_ctx:
                out = model(batch)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                loss_autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
            else:
                loss_autocast_ctx = torch.cuda.amp.autocast(enabled=False)
            with loss_autocast_ctx:
                losses = compute_bdse_losses(out, batch, cfg)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            for k, v in losses.items():
                meters.setdefault(k, []).append(float(v.detach().cpu()))
        epoch_metrics = _aggregate_meters(meters, device, distributed)
        if validation_enabled and ((epoch + 1) % int(args.val_every_n_epochs) == 0):
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            val_metrics: dict[str, float] = {}
            if args.val_mode in {"loss", "both"}:
                assert val_loader is not None
                val_metrics.update(
                    _run_validation_loss(
                        model=model,
                        loader=val_loader,
                        cfg=cfg,
                        device=device,
                        distributed=distributed,
                        is_main=is_main,
                        epoch=epoch,
                    )
                )
            if args.val_mode in {"open_loop", "both"}:
                assert val_dataset is not None
                val_metrics.update(
                    _run_validation_open_loop(
                        model=model,
                        dataset=val_dataset,
                        cfg=cfg,
                        device=device,
                        distributed=distributed,
                        world_size=world_size,
                        global_rank=global_rank,
                        is_main=is_main,
                        epoch=epoch,
                        strict=bool(args.val_strict),
                    )
                )
            epoch_metrics.update(val_metrics)
        if is_main:
            print(epoch_metrics, flush=True)
            log_row = {"epoch": int(epoch), **{str(k): float(v) for k, v in epoch_metrics.items()}}
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_row, sort_keys=True) + "\n")
        best_metric, best_epoch = _save_training_checkpoints(
            args=args,
            model=model,
            optimizer=opt,
            scaler=scaler,
            cfg=cfg,
            epoch=epoch,
            metrics=epoch_metrics,
            best_metric=best_metric,
            best_epoch=best_epoch,
            world_size=world_size,
            is_main=is_main,
        )
        if distributed and dist.is_initialized():
            dist.barrier()
    if is_main:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_model = model.module if isinstance(model, DDP) else model
        final_metrics = {"best_metric": best_metric, "best_epoch": best_epoch}
        torch.save({"model": raw_model.state_dict(), "cfg": cfg, "metrics": final_metrics}, out_path)
        print(f"[bdse] saved final model={out_path} best_epoch={best_epoch} best_{args.best_metric}={best_metric}", flush=True)
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
