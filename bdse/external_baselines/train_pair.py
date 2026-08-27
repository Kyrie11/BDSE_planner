from __future__ import annotations

"""Train two fixed-budget external adapters from one shared DataLoader.

The original 2-GPU launcher starts two independent Python/DataLoader stacks. For
large NPZ caches this makes both jobs read, unzip, JSON-decode and tensorize the
same shuffled samples at the same time.  This module keeps one CPU input stream
and broadcasts each compact batch to two GPUs.  CUDA work is launched on the
independent per-device default streams, so the two models still execute in
parallel while host I/O/decoding is paid once.

This trainer intentionally supports the matched pairs used by
RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh: GameFormer+DTPP and
PlanTF+PLUTO. It preserves per-model optimizers, schedulers, losses, checkpoints,
and validation metrics.
"""

import argparse
import importlib
import json
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from bdse.config import load_config
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.external_baselines.losses import compute_external_baseline_losses
from bdse.external_baselines.models import ExternalBaselineModel, external_reference, external_variant
from bdse.external_baselines.compact_cache import CompactBatchLoader, CompactExternalCache
from bdse.external_baselines.train import (
    ExternalBaselineDataset,
    _accumulate,
    _amp_context,
    _best_checkpoint_path,
    _config_sha256,
    _dataset_manifest,
    _finalize_meters,
    _make_optimizer,
    _make_scaler,
    _make_scheduler,
    _planner_supervision,
    _save_checkpoint,
    _seed_everything,
    _seed_worker,
    collate,
)
from bdse.utils import configure_torch_for_device, torch_load_any


def _data_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fields that can change compact CPU tensorization/supervision."""
    ecfg = cfg.get("external_baseline", {}) or {}
    mcfg = cfg.get("model", {}) or {}
    return {
        "candidate": cfg.get("candidate", {}) or {},
        "evidence": cfg.get("evidence", {}) or {},
        "pairs": cfg.get("pairs", {}) or {},
        "runtime": cfg.get("runtime", {}) or {},
        "planner_supervision": _planner_supervision(cfg),
        "evidence_feature_dim": int(mcfg.get("evidence_feature_dim", 24)),
        "proposal_feature_dim": int(mcfg.get("proposal_feature_dim", 24)),
        "proposal_loss_enabled": float((ecfg.get("loss_weights", {}) or {}).get("proposal", 0.25)) != 0.0,
        "pair_loss_enabled": float((ecfg.get("loss_weights", {}) or {}).get("pair", 0.0)) != 0.0,
    }


def _assert_pair_compatible(cfg_a: dict[str, Any], cfg_b: dict[str, Any]) -> None:
    a, b = _data_contract(cfg_a), _data_contract(cfg_b)
    if a != b:
        # Give a useful compact diff instead of silently training model B with A's tensorizer.
        keys = sorted(set(a) | set(b))
        diff = {k: {"a": a.get(k), "b": b.get(k)} for k in keys if a.get(k) != b.get(k)}
        raise ValueError(f"paired trainer requires identical compact data contracts; differences={diff}")


def _setup_cuda(device: torch.device) -> None:
    configure_torch_for_device(device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
        fn = getattr(torch.backends.cuda, name, None)
        if callable(fn):
            fn(True)


def _maybe_compile(model: torch.nn.Module, *, enabled: bool, mode: str, fallback: bool) -> torch.nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch build")
    if fallback:
        try:
            dynamo = importlib.import_module("torch._dynamo")
            dynamo.config.suppress_errors = True
        except Exception:
            pass
    return torch.compile(model, mode=mode)  # type: ignore[return-value]


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.inference_mode()
def _validation_pair(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader: DataLoader,
    cfg_a: dict[str, Any],
    cfg_b: dict[str, Any],
    dev_a: torch.device,
    dev_b: torch.device,
    *,
    amp: bool,
    log_every: int,
    prefix: str,
) -> tuple[dict[str, float], dict[str, float]]:
    model_a.eval(); model_b.eval()
    ma: dict[str, torch.Tensor] = {}; mb: dict[str, torch.Tensor] = {}
    count = 0
    started = time.perf_counter()
    total = len(loader)
    print(f"[{prefix}-start] batches={total} shared_loader=1", flush=True)
    for step, host_batch in enumerate(loader, start=1):
        ba = _to_device(host_batch, dev_a)
        bb = _to_device(host_batch, dev_b)
        with _amp_context(dev_a, amp):
            la = compute_external_baseline_losses(model_a(ba), ba, cfg_a)
        with _amp_context(dev_b, amp):
            lb = compute_external_baseline_losses(model_b(bb), bb, cfg_b)
        _accumulate(ma, la); _accumulate(mb, lb); count += 1
        if step % max(1, log_every) == 0 or step == total:
            # CPU conversion synchronizes both streams, making the reported shared rate real.
            lva = float(la["loss"].detach().cpu()); lvb = float(lb["loss"].detach().cpu())
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"[{prefix}-progress] step={step}/{total} rate={step/elapsed:.2f} shared_batch/s "
                f"loss_a={lva:.4f} loss_b={lvb:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return _finalize_meters(ma, count, "val_"), _finalize_meters(mb, count, "val_")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a matched external-baseline pair with one shared NPZ DataLoader.")
    ap.add_argument("--config-a", required=True); ap.add_argument("--config-b", required=True)
    ap.add_argument("--output-a", required=True); ap.add_argument("--output-b", required=True)
    ap.add_argument("--log-file-a", default=None); ap.add_argument("--log-file-b", default=None)
    ap.add_argument("--split", nargs="+", default=["train_boston", "train_pittsburgh", "train_singapore", "train_vegas_2"])
    ap.add_argument("--preprocessed-dir", required=True)
    ap.add_argument("--val-preprocessed-dir", default=None); ap.add_argument("--val-split", nargs="+", default=["val"])
    ap.add_argument("--val-max-scenarios", type=int, default=0); ap.add_argument("--val-every-n-epochs", type=int, default=1)
    ap.add_argument("--max-scenarios", type=int, default=None); ap.add_argument("--max-scenarios-per-split", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=10); ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--compact-cache-dir", default=None)
    ap.add_argument("--val-compact-cache-dir", default=None)
    ap.add_argument("--compact-shuffle-mode", choices=["global", "block", "none"], default="global")
    ap.add_argument("--compact-block-size", type=int, default=4096)
    ap.add_argument("--compact-prefetch", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--warmup-epochs", type=int, default=3); ap.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    ap.add_argument("--selection-metric", default="val_action_ce")
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad-accum-steps", type=int, default=1); ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--log-every-n-steps", type=int, default=100)
    ap.add_argument("--optimizer-fused", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compile", action="store_true"); ap.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="reduce-overhead")
    ap.add_argument("--compile-fallback", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(f"train_pair requires two visible CUDA devices, got {torch.cuda.device_count()}")
    dev_a, dev_b = torch.device("cuda:0"), torch.device("cuda:1")
    _setup_cuda(dev_a); _setup_cuda(dev_b)
    _seed_everything(args.seed)

    cfg_a, cfg_b = load_config(args.config_a), load_config(args.config_b)
    _assert_pair_compatible(cfg_a, cfg_b)
    var_a, var_b = external_variant(cfg_a), external_variant(cfg_b)
    if var_a == "pdm_closed" or var_b == "pdm_closed":
        raise ValueError("pdm_closed is non-trainable")
    sup = _planner_supervision(cfg_a)
    use_label_future = sup == "expert_imitation"

    def hp(cfg: dict[str, Any], key: str, default: Any) -> Any:
        ecfg, tcfg = cfg.get("external_baseline", {}) or {}, cfg.get("training", {}) or {}
        return ecfg.get(key, tcfg.get(key, default))

    epochs_a = int(args.epochs if args.epochs is not None else hp(cfg_a, "epochs", 20))
    epochs_b = int(args.epochs if args.epochs is not None else hp(cfg_b, "epochs", 20))
    if epochs_a != epochs_b:
        raise ValueError(f"paired models must use the same epoch count; got {epochs_a} vs {epochs_b}")
    epochs = epochs_a
    batch_a = int(args.batch_size if args.batch_size is not None else hp(cfg_a, "batch_size", 32))
    batch_b = int(args.batch_size if args.batch_size is not None else hp(cfg_b, "batch_size", 32))
    if batch_a != batch_b:
        raise ValueError(f"paired models must use the same micro-batch size; got {batch_a} vs {batch_b}")
    batch_size = batch_a
    grad_accum = max(1, int(args.grad_accum_steps))
    lr_a, lr_b = float(hp(cfg_a, "lr", 1e-4)), float(hp(cfg_b, "lr", 1e-4))
    wd_a, wd_b = float(hp(cfg_a, "weight_decay", 1e-2)), float(hp(cfg_b, "weight_decay", 1e-2))
    gc_a = float(args.grad_clip if args.grad_clip is not None else hp(cfg_a, "grad_clip", 5.0))
    gc_b = float(args.grad_clip if args.grad_clip is not None else hp(cfg_b, "grad_clip", 5.0))

    collate_fn = partial(collate, cfg=cfg_a)
    ds = None
    compact_train = None
    if args.compact_cache_dir:
        if args.max_scenarios_per_split:
            raise ValueError("--compact-cache-dir does not support --max-scenarios-per-split; use the full cache or raw DataLoader for subset probes")
        compact_train = CompactExternalCache.open(args.compact_cache_dir)
        compact_train.assert_compatible(cfg_a)
        compact_train.assert_compatible(cfg_b)
        if budget := int((cfg_a.get("evidence", {}) or {}).get("budget", -1)):
            if budget not in compact_train.budgets:
                raise ValueError(f"training compact cache lacks B={budget}: {compact_train.root} budgets={compact_train.budgets}")
        limit = int(args.max_scenarios) if args.max_scenarios is not None and int(args.max_scenarios) > 0 else None
        loader = CompactBatchLoader(
            compact_train,
            budget=int((cfg_a.get("evidence", {}) or {}).get("budget", -1)),
            batch_size=batch_size,
            shuffle=True,
            seed=args.seed,
            pin_memory=True,
            prefetch=args.compact_prefetch,
            shuffle_mode=args.compact_shuffle_mode,
            block_size=args.compact_block_size,
            limit=limit,
        )
        train_count = loader.count
        print(
            f"[pair-data] compact_mmap=1 path={compact_train.root} samples={train_count} "
            f"shuffle={args.compact_shuffle_mode} prefetch={args.compact_prefetch}",
            flush=True,
        )
    else:
        src = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split, max_scenarios=args.max_scenarios, max_scenarios_per_split=args.max_scenarios_per_split)
        ds = ExternalBaselineDataset(src, include_label_future=use_label_future)
        gen = torch.Generator(); gen.manual_seed(args.seed)
        loader_kwargs: dict[str, Any] = dict(
            batch_size=batch_size, shuffle=True, num_workers=max(0, args.num_workers), pin_memory=True,
            persistent_workers=args.num_workers > 0, collate_fn=collate_fn, worker_init_fn=_seed_worker, generator=gen,
        )
        if args.num_workers > 0: loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
        loader = DataLoader(ds, **loader_kwargs)
        train_count = len(ds)

    val_ds = None; val_loader = None; compact_val = None
    if args.val_preprocessed_dir:
        val_max = None if args.val_max_scenarios <= 0 else args.val_max_scenarios
        if args.val_compact_cache_dir:
            compact_val = CompactExternalCache.open(args.val_compact_cache_dir)
            compact_val.assert_compatible(cfg_a)
            compact_val.assert_compatible(cfg_b)
            val_loader = CompactBatchLoader(
                compact_val,
                budget=int((cfg_a.get("evidence", {}) or {}).get("budget", -1)),
                batch_size=batch_size,
                shuffle=False,
                seed=args.seed,
                pin_memory=True,
                prefetch=args.compact_prefetch,
                shuffle_mode="none",
                block_size=args.compact_block_size,
                limit=val_max,
            )
            print(f"[pair-val-data] compact_mmap=1 path={compact_val.root} samples={val_loader.count}", flush=True)
        else:
            vsrc = PreprocessedBDSEDataset(args.val_preprocessed_dir, split=args.val_split, max_scenarios=val_max)
            val_ds = ExternalBaselineDataset(vsrc, include_label_future=use_label_future)
            vw = max(0, min(args.num_workers, 8))
            vkw: dict[str, Any] = dict(batch_size=batch_size, shuffle=False, num_workers=vw, pin_memory=True,
                                       persistent_workers=vw > 0, collate_fn=collate_fn, worker_init_fn=_seed_worker)
            if vw > 0: vkw["prefetch_factor"] = max(1, args.prefetch_factor)
            val_loader = DataLoader(val_ds, **vkw)

    # Match the legacy two-process protocol: each model process used the same
    # global seed before construction. Reset between constructions so pairing
    # does not silently alter initialization merely because both models now live
    # in one Python process. CUDA RNG streams remain per-device during training.
    _seed_everything(args.seed)
    model_a = ExternalBaselineModel(cfg_a).to(dev_a)
    _seed_everything(args.seed)
    model_b = ExternalBaselineModel(cfg_b).to(dev_b)
    runtime_a = _maybe_compile(model_a, enabled=args.compile, mode=args.compile_mode, fallback=args.compile_fallback)
    runtime_b = _maybe_compile(model_b, enabled=args.compile, mode=args.compile_mode, fallback=args.compile_fallback)
    opt_a = _make_optimizer(model_a, lr=lr_a, weight_decay=wd_a, fused=args.optimizer_fused, device=dev_a)
    opt_b = _make_optimizer(model_b, lr=lr_b, weight_decay=wd_b, fused=args.optimizer_fused, device=dev_b)
    sch_a = _make_scheduler(opt_a, scheduler_name=args.scheduler, epochs=epochs, warmup_epochs=args.warmup_epochs)
    sch_b = _make_scheduler(opt_b, scheduler_name=args.scheduler, epochs=epochs, warmup_epochs=args.warmup_epochs)
    sc_a, sc_b = _make_scaler(dev_a, args.amp), _make_scaler(dev_b, args.amp)

    budget = int((cfg_a.get("evidence", {}) or {}).get("budget", -1))
    print(
        f"[pair-train-init] A={var_a}@cuda:0 params={sum(p.numel() for p in model_a.parameters())} "
        f"B={var_b}@cuda:1 params={sum(p.numel() for p in model_b.parameters())} budget={budget} "
        f"samples={train_count} batch={batch_size} workers_shared={args.num_workers} amp={args.amp} compile={args.compile} "
        f"refineA={(cfg_a.get('external_baseline', {}) or {}).get('refinement_mode', 'legacy_repeated_encoder')} "
        f"refineB={(cfg_b.get('external_baseline', {}) or {}).get('refinement_mode', 'legacy_repeated_encoder')}", flush=True,
    )

    def manifest(cfg: dict[str, Any], variant: str, cfg_path: str) -> dict[str, Any]:
        if compact_train is not None:
            train_manifest = compact_train.source_manifest()
        else:
            assert ds is not None
            train_manifest = _dataset_manifest(ds.paths, args.preprocessed_dir, list(args.split))
        if compact_val is not None:
            validation_manifest = compact_val.source_manifest()
        elif val_ds is not None:
            validation_manifest = _dataset_manifest(val_ds.paths, args.val_preprocessed_dir or "", list(args.val_split))
        else:
            validation_manifest = None
        return {
            "schema_version": 2, "variant": variant, "implementation": external_reference(variant), "seed": args.seed,
            "config_path": str(Path(cfg_path).resolve()), "config_sha256": _config_sha256(cfg_path),
            "train": train_manifest,
            "validation": validation_manifest,
            "protocol": {"paired_shared_dataloader": True, "batch_size": batch_size, "grad_accum_steps": grad_accum,
                         "epochs": epochs, "warmup_epochs": args.warmup_epochs, "scheduler": args.scheduler,
                         "selection_metric": args.selection_metric, "planner_supervision": sup,
                         "val_every_n_epochs": args.val_every_n_epochs, "val_max_scenarios": args.val_max_scenarios,
                         "torch_compile": bool(args.compile),
                         "compact_mmap_cache": compact_train is not None,
                         "compact_shuffle_mode": args.compact_shuffle_mode if compact_train is not None else None},
        }
    man_a, man_b = manifest(cfg_a, var_a, args.config_a), manifest(cfg_b, var_b, args.config_b)
    for out, man in ((args.output_a, man_a), (args.output_b, man_b)):
        mp = Path(out).with_suffix(".data_manifest.json"); mp.parent.mkdir(parents=True, exist_ok=True); mp.write_text(json.dumps(man, indent=2, sort_keys=True))

    best_a = best_b = float("inf")
    best_path_a, best_path_b = _best_checkpoint_path(args.output_a), _best_checkpoint_path(args.output_b)
    log_a = Path(args.log_file_a) if args.log_file_a else None; log_b = Path(args.log_file_b) if args.log_file_b else None
    start_wall = time.perf_counter()

    for epoch in range(epochs):
        model_a.train(); model_b.train(); runtime_a.train(); runtime_b.train()
        ma: dict[str, torch.Tensor] = {}; mb: dict[str, torch.Tensor] = {}
        epoch_start = time.perf_counter(); total = len(loader)
        opt_a.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
        print(f"[pair-epoch-start] A={var_a} B={var_b} budget={budget} epoch={epoch+1}/{epochs} batches={total}", flush=True)
        for step, host_batch in enumerate(loader, start=1):
            ba = _to_device(host_batch, dev_a); bb = _to_device(host_batch, dev_b)
            group_start = ((step - 1) // grad_accum) * grad_accum
            group_size = min(grad_accum, total - group_start)
            with _amp_context(dev_a, args.amp):
                la = compute_external_baseline_losses(runtime_a(ba), ba, cfg_a); bla = la["loss"] / float(group_size)
            with _amp_context(dev_b, args.amp):
                lb = compute_external_baseline_losses(runtime_b(bb), bb, cfg_b); blb = lb["loss"] / float(group_size)
            sc_a.scale(bla).backward(); sc_b.scale(blb).backward()
            if step % grad_accum == 0 or step == total:
                if gc_a > 0: sc_a.unscale_(opt_a); torch.nn.utils.clip_grad_norm_(model_a.parameters(), gc_a)
                if gc_b > 0: sc_b.unscale_(opt_b); torch.nn.utils.clip_grad_norm_(model_b.parameters(), gc_b)
                sc_a.step(opt_a); sc_b.step(opt_b); sc_a.update(); sc_b.update()
                opt_a.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
            _accumulate(ma, la); _accumulate(mb, lb)
            if step % max(1, args.log_every_n_steps) == 0 or step == total:
                lva = float(la["loss"].detach().cpu()); lvb = float(lb["loss"].detach().cpu())
                cea = float(la["action_ce"].detach().cpu()); ceb = float(lb["action_ce"].detach().cpu())
                propa = float(la["proposal_bce"].detach().cpu()); propb = float(lb["proposal_bce"].detach().cpu())
                elapsed = max(time.perf_counter() - epoch_start, 1e-9); seen = min(step * batch_size, train_count)
                print(
                    f"[pair-train-progress] A={var_a} lossA={lva:.4f} ceA={cea:.4f} propA={propa:.4f} "
                    f"B={var_b} lossB={lvb:.4f} ceB={ceb:.4f} propB={propb:.4f} budget={budget} "
                    f"epoch={epoch+1}/{epochs} step={step}/{total} ({100*step/max(total,1):.1f}%) samples~={seen}/{train_count} "
                    f"lrA={opt_a.param_groups[0]['lr']:.2e} lrB={opt_b.param_groups[0]['lr']:.2e} "
                    f"rate={step/elapsed:.2f} shared_batch/s elapsed={elapsed:.1f}s", flush=True,
                )

        met_a, met_b = _finalize_meters(ma, total), _finalize_meters(mb, total)
        validation_ran = val_loader is not None and ((epoch + 1) % max(1, args.val_every_n_epochs) == 0)
        if validation_ran:
            va, vb = _validation_pair(runtime_a, runtime_b, val_loader, cfg_a, cfg_b, dev_a, dev_b, amp=args.amp,
                                      log_every=args.log_every_n_steps, prefix=f"pair-val-B{budget}-e{epoch+1}")
            met_a.update(va); met_b.update(vb)
        if sch_a is not None: sch_a.step()
        if sch_b is not None: sch_b.step()

        for side, (variant, model, opt, sch, cfg, metrics, out_path, best_path, man, log_path) in enumerate((
            (var_a, model_a, opt_a, sch_a, cfg_a, met_a, args.output_a, best_path_a, man_a, log_a),
            (var_b, model_b, opt_b, sch_b, cfg_b, met_b, args.output_b, best_path_b, man_b, log_b),
        )):
            key = args.selection_metric if args.selection_metric in metrics else ("val_loss" if "val_loss" in metrics else "loss")
            score = float(metrics.get(key, float("inf")))
            current_best = best_a if side == 0 else best_b
            improved = validation_ran and math.isfinite(score) and score < current_best - 1e-8
            if improved:
                current_best = score
                _save_checkpoint(best_path, model=model, optimizer=opt, scheduler=sch, cfg=cfg, epoch=epoch, metrics=metrics,
                                 best_metric=current_best, selection_metric=key, training_manifest=man)
            if side == 0: best_a = current_best
            else: best_b = current_best
            _save_checkpoint(out_path, model=model, optimizer=opt, scheduler=sch, cfg=cfg, epoch=epoch, metrics=metrics,
                             best_metric=current_best, selection_metric=key, training_manifest=man)
            if log_path:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": epoch+1, "variant": variant, **metrics, "selection_metric": key,
                                        "selection_value": score, "best_metric": current_best,
                                        "lr": opt.param_groups[0]["lr"], "paired_shared_dataloader": True,
                                        "elapsed_wall_s": time.perf_counter()-start_wall}, sort_keys=True) + "\n")
        print(
            f"[pair-epoch-done] A={var_a} lossA={met_a.get('loss', float('nan')):.4f} valA={met_a.get('val_loss', float('nan')):.4f} "
            f"B={var_b} lossB={met_b.get('loss', float('nan')):.4f} valB={met_b.get('val_loss', float('nan')):.4f} "
            f"epoch_wall={time.perf_counter()-epoch_start:.1f}s total_wall={time.perf_counter()-start_wall:.1f}s", flush=True,
        )

    final_wall = time.perf_counter() - start_wall
    for variant, out, best_path, best_metric, man in (
        (var_a, args.output_a, best_path_a, best_a, man_a),
        (var_b, args.output_b, best_path_b, best_b, man_b),
    ):
        if not best_path.is_file():
            torch.save(torch_load_any(out, map_location="cpu"), best_path)
        summary = {
            "variant": variant,
            "implementation": external_reference(variant),
            "output": str(Path(out).resolve()),
            "best_checkpoint": str(best_path.resolve()),
            "best_metric": best_metric,
            "selection_metric": args.selection_metric,
            "data_manifest": str(Path(out).with_suffix(".data_manifest.json").resolve()),
            "paired_shared_dataloader": True,
            "wall_time_s": final_wall,
        }
        Path(out).with_suffix(".training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[pair-train-complete] A={var_a} best={best_path_a} B={var_b} best={best_path_b} wall_min={final_wall/60:.1f}", flush=True)


if __name__ == "__main__":
    main()
