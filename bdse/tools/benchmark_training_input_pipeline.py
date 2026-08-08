from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from bdse.config import load_config
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.experiments.train import _make_loader


def _bench(
    cfg: dict[str, Any],
    cache: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    max_scenarios: int,
    warmup_batches: int,
    measured_batches: int,
) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg["training"] = dict(cfg.get("training", {}) or {})
    cfg["training"]["prefetch_factor"] = int(prefetch_factor)
    dataset = PreprocessedBDSEDataset(
        cache,
        split=split,
        max_scenarios=max_scenarios,
    )
    loader, _ = _make_loader(
        dataset=dataset,
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        cuda_available=False,
        distributed=False,
        world_size=1,
        global_rank=0,
        shuffle=False,
        seed=17,
    )
    it = iter(loader)
    for _ in range(max(0, warmup_batches)):
        try:
            next(it)
        except StopIteration:
            break
    times: list[float] = []
    samples = 0
    for _ in range(max(1, measured_batches)):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        dt = time.perf_counter() - t0
        times.append(dt)
        # Any tensor batch dimension is sufficient.
        first = next(iter(batch.values()))
        samples += int(first.shape[0]) if hasattr(first, "shape") and len(first.shape) else batch_size
    arr = np.asarray(times, dtype=np.float64)
    total = float(arr.sum()) if arr.size else float("nan")
    return {
        "num_workers": int(num_workers),
        "prefetch_factor": int(prefetch_factor),
        "batch_size": int(batch_size),
        "measured_batches": int(arr.size),
        "samples": int(samples),
        "wall_s": total,
        "mean_batch_wait_ms": float(arr.mean() * 1000.0) if arr.size else float("nan"),
        "p50_batch_wait_ms": float(np.quantile(arr, 0.50) * 1000.0) if arr.size else float("nan"),
        "p95_batch_wait_ms": float(np.quantile(arr, 0.95) * 1000.0) if arr.size else float("nan"),
        "samples_per_s": float(samples / total) if arr.size and total > 0 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark BDSE NPZ decode + tensorization without model compute.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--preprocessed-dir", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--max-scenarios", type=int, default=4096)
    ap.add_argument("--warmup-batches", type=int, default=8)
    ap.add_argument("--measured-batches", type=int, default=64)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = [
        _bench(
            cfg,
            args.preprocessed_dir,
            args.split,
            args.batch_size,
            w,
            args.prefetch_factor,
            args.max_scenarios,
            args.warmup_batches,
            args.measured_batches,
        )
        for w in args.workers
    ]
    finite = [r for r in results if np.isfinite(float(r["samples_per_s"]))]
    best = max(finite, key=lambda r: float(r["samples_per_s"])) if finite else None
    report = {
        "benchmark": "v64_3_training_input_pipeline",
        "note": "Measures cache decode + sample tensorization only; no GPU/model work.",
        "cache": str(args.preprocessed_dir),
        "split": args.split,
        "results": results,
        "recommended_num_workers_per_process": None if best is None else int(best["num_workers"]),
        "recommended_prefetch_factor": int(args.prefetch_factor),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
