from __future__ import annotations

"""Small source-local profiler for external fixed-budget training.

It measures the two dominant components independently:
  1) DataLoader/NPZ decode + compact tensorization throughput.
  2) model forward/loss/backward/optimizer throughput while replaying one batch.

The separation makes it easy to distinguish storage/CPU starvation from an
expensive model graph before launching a multi-day sweep.
"""

import argparse
import time
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader

from bdse.config import load_config
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.external_baselines.losses import compute_external_baseline_losses
from bdse.external_baselines.models import ExternalBaselineModel, external_variant
from bdse.external_baselines.train import ExternalBaselineDataset, _amp_context, _make_scaler, _planner_supervision, _seed_worker, collate
from bdse.utils import configure_torch_for_device, resolve_torch_device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile external-baseline DataLoader and model compute separately.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--preprocessed-dir", required=True)
    ap.add_argument("--split", nargs="+", default=["train_boston", "train_pittsburgh", "train_singapore", "train_vegas_2"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_torch_device(args.device, context="external training profiler")
    configure_torch_for_device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    source = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split)
    ds = ExternalBaselineDataset(source, include_label_future=_planner_supervision(cfg) == "expert_imitation")
    kwargs: dict[str, Any] = dict(
        batch_size=args.batch_size, shuffle=True, num_workers=max(0, args.num_workers),
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        collate_fn=partial(collate, cfg=cfg), worker_init_fn=_seed_worker,
    )
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
    loader = DataLoader(ds, **kwargs)

    # Data-only: deliberately do not transfer to GPU.
    it = iter(loader)
    warm = min(max(args.warmup, 0), max(len(loader) - 1, 0))
    for _ in range(warm):
        try: next(it)
        except StopIteration: it = iter(loader); next(it)
    n = min(max(args.batches, 1), max(len(loader), 1))
    t0 = time.perf_counter(); first = None
    for i in range(n):
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        if first is None: first = batch
    data_s = time.perf_counter() - t0
    assert first is not None

    # Compute-only: replay a single already-decoded batch.
    model = ExternalBaselineModel(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = _make_scaler(device, args.amp)
    gpu_batch = {k: v.to(device, non_blocking=False) for k, v in first.items()}
    for _ in range(max(args.warmup, 1)):
        opt.zero_grad(set_to_none=True)
        with _amp_context(device, args.amp):
            losses = compute_external_baseline_losses(model(gpu_batch), gpu_batch, cfg)
        scaler.scale(losses["loss"]).backward(); scaler.step(opt); scaler.update()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        with _amp_context(device, args.amp):
            losses = compute_external_baseline_losses(model(gpu_batch), gpu_batch, cfg)
        scaler.scale(losses["loss"]).backward(); scaler.step(opt); scaler.update()
    _sync(device)
    compute_s = time.perf_counter() - t0

    variant = external_variant(cfg)
    data_rate = n / max(data_s, 1e-9); compute_rate = n / max(compute_s, 1e-9)
    predicted_serial = 1.0 / max((1.0 / max(data_rate, 1e-9)) + (1.0 / max(compute_rate, 1e-9)), 1e-9)
    bottleneck = "data/NPZ/CPU" if data_rate < compute_rate else "model/GPU"
    print(
        f"[profile-result] variant={variant} batches={n} batch_size={args.batch_size} workers={args.num_workers} "
        f"data_rate={data_rate:.3f} batch/s compute_rate={compute_rate:.3f} batch/s "
        f"serial_upper_estimate={predicted_serial:.3f} batch/s bottleneck={bottleneck}", flush=True,
    )
    print(
        "[profile-note] Compare this single-job result with the 2-GPU launcher. If both concurrent jobs collapse to the same "
        "rate while single-job data_rate is much higher, shared storage/CPU contention is the limiter; use train_pair/shared DataLoader.",
        flush=True,
    )


if __name__ == "__main__":
    main()
