from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bdse.data.nuplan_dataset import PreprocessedBDSEDataset


@dataclass(frozen=True)
class SystemSpec:
    name: str
    config: Path
    checkpoint: Path


@dataclass(frozen=True)
class Task:
    system: SystemSpec
    shard_id: int
    shard_cache: Path
    output_json: Path
    output_jsonl: Path
    log_path: Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_system(text: str) -> SystemSpec:
    # NAME::CONFIG::CHECKPOINT avoids ambiguity with paths containing ':' on
    # mounted filesystems and remains readable in shell commands.
    parts = text.split("::", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("--system must be NAME::CONFIG::CHECKPOINT")
    spec = SystemSpec(parts[0], Path(parts[1]).expanduser().resolve(), Path(parts[2]).expanduser().resolve())
    if not spec.config.is_file():
        raise argparse.ArgumentTypeError(f"missing config: {spec.config}")
    if not spec.checkpoint.is_file():
        raise argparse.ArgumentTypeError(f"missing checkpoint: {spec.checkpoint}")
    return spec


def _write_manifests(cache_root: Path, split: str, limit: int, shard_root: Path, num_shards: int) -> int:
    paths = PreprocessedBDSEDataset(cache_root, split=[split], max_scenarios=limit).build_index()
    if not paths:
        raise RuntimeError(f"no samples found under {cache_root} split={split}")
    for shard_id in range(num_shards):
        out = shard_root / f"shard{shard_id}" / "val"
        out.mkdir(parents=True, exist_ok=True)
        records = [
            {"split": "val", "path": str(path.resolve()), "original_index": index}
            for index, path in enumerate(paths)
            if index % num_shards == shard_id
        ]
        (out / "manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
        )
    (shard_root / "shard_metadata.json").write_text(
        json.dumps({"split": split, "num_scenarios": len(paths), "num_shards": num_shards}, indent=2),
        encoding="utf-8",
    )
    return len(paths)


def _worker(
    slot_id: int,
    gpu: str,
    tasks: "queue.Queue[Task | None]",
    errors: "queue.Queue[str]",
    device: str,
    disable_dense: bool,
) -> None:
    while True:
        task = tasks.get()
        try:
            if task is None:
                return
            task.output_json.parent.mkdir(parents=True, exist_ok=True)
            task.log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, "-m", "bdse.experiments.evaluate_open_loop",
                "--config", str(task.system.config),
                "--checkpoint", str(task.system.checkpoint),
                "--split", "val",
                "--preprocessed-dir", str(task.shard_cache),
                "--device", device,
                "--output", str(task.output_json),
                "--per-sample-output", str(task.output_jsonl),
            ]
            if disable_dense:
                cmd.append("--disable-dense-diagnostic")
            env = os.environ.copy()
            if device == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            started = time.time()
            with task.log_path.open("w", encoding="utf-8") as log:
                log.write(f"slot={slot_id} gpu={gpu} system={task.system.name} shard={task.shard_id}\n")
                log.flush()
                proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"open-loop task failed system={task.system.name} shard={task.shard_id}; see {task.log_path}"
                )
            task.output_json.with_suffix(".done.json").write_text(
                json.dumps({"wall_time_s": time.time() - started, "slot": slot_id, "gpu": gpu}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            errors.put(f"slot={slot_id} gpu={gpu}: {type(exc).__name__}: {exc}")
        finally:
            tasks.task_done()


def _merge(system: SystemSpec, out_root: Path, num_shards: int, wall_time_s: float) -> dict[str, Any]:
    sys_root = out_root / system.name
    summaries: list[dict[str, Any]] = []
    shard_rows: list[list[dict[str, Any]]] = []
    task_wall = 0.0
    for shard_id in range(num_shards):
        base = sys_root / f"shard{shard_id}"
        summaries.append(json.loads((base / "metrics.json").read_text(encoding="utf-8")))
        rows = [json.loads(line) for line in (base / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        shard_rows.append(rows)
        done = json.loads((base / "metrics.done.json").read_text(encoding="utf-8"))
        task_wall += float(done.get("wall_time_s", 0.0))
    rows: list[dict[str, Any]] = []
    for group in itertools.zip_longest(*shard_rows):
        rows.extend(row for row in group if row is not None)
    keys_in_order = [
        (str(row.get("scenario_token", "")), int(row.get("timestamp_us", 0) or 0))
        for row in rows
    ]
    if any(not token for token, _ in keys_in_order):
        raise RuntimeError(f"{system.name} produced an empty scenario token")
    if len(set(keys_in_order)) != len(keys_in_order):
        raise RuntimeError(f"{system.name} produced duplicate scenario/timestamp keys")
    key_hash = hashlib.sha256(
        "".join(f"{token}::{timestamp}\n" for token, timestamp in keys_in_order).encode("utf-8")
    ).hexdigest()
    special = {"device", "cuda_peak_memory_mb"}
    out: dict[str, Any] = {}
    keys = set().union(*(row.keys() for row in rows))
    for key in sorted(keys - special):
        vals: list[float] = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
                vals.append(float(value))
        if vals:
            out[key] = float(np.mean(np.asarray(vals, dtype=np.float64)))
    latencies = np.asarray(
        [float(row["planner_latency_ms"]) for row in rows if math.isfinite(float(row.get("planner_latency_ms", float("nan"))))],
        dtype=np.float64,
    )
    if latencies.size:
        out.update({
            "planner_latency_ms_mean": float(latencies.mean()),
            "planner_latency_ms_p50": float(np.quantile(latencies, 0.50)),
            "planner_latency_ms_p90": float(np.quantile(latencies, 0.90)),
            "planner_latency_ms_p95": float(np.quantile(latencies, 0.95)),
            "planner_latency_ms_p99": float(np.quantile(latencies, 0.99)),
            "planner_latency_ms_max": float(latencies.max()),
        })
    peaks = [float(summary.get("cuda_peak_memory_mb", float("nan"))) for summary in summaries]
    finite_peaks = [x for x in peaks if math.isfinite(x)]
    out.update({
        "system": system.name,
        "device": f"parallel suite ({num_shards} shards)",
        "num_scenarios": len(rows),
        "suite_wall_time_s": float(wall_time_s),
        "sum_task_wall_time_s": float(task_wall),
        "scenario_timestamp_sha256": key_hash,
    })
    if finite_peaks:
        out["cuda_peak_memory_mb"] = max(finite_peaks)
    (sys_root / "metrics.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    with (sys_root / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run candidate/local/foundation open-loop systems in one bounded GPU/CPU worker pool.")
    parser.add_argument("--system", action="append", type=_parse_system, required=True)
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--split", default="val_eval")
    parser.add_argument("--max-scenarios", type=int, default=1000)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--disable-dense-diagnostic", action="store_true")
    args = parser.parse_args()

    systems: list[SystemSpec] = args.system
    gpus = [item.strip() for item in str(args.gpus).split(",") if item.strip()]
    if args.device == "cpu":
        gpus = ["cpu"]
    if not gpus:
        raise ValueError("at least one GPU id is required")
    slots = max(1, len(gpus) * max(1, int(args.workers_per_gpu)))
    out_root = args.output_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    shard_root = out_root / ".shared_shards"
    scenario_count = _write_manifests(
        args.preprocessed_dir.expanduser().resolve(), args.split, int(args.max_scenarios), shard_root, slots
    )

    task_queue: "queue.Queue[Task | None]" = queue.Queue()
    errors: "queue.Queue[str]" = queue.Queue()
    # Shard-major ordering starts every system before later shards, so all paired
    # controls overlap while the fixed-size pool prevents unbounded GPU contention.
    for shard_id in range(slots):
        for system in systems:
            base = out_root / system.name / f"shard{shard_id}"
            task_queue.put(Task(
                system=system,
                shard_id=shard_id,
                shard_cache=shard_root / f"shard{shard_id}" / "val",
                output_json=base / "metrics.json",
                output_jsonl=base / "metrics.jsonl",
                log_path=base / "run.log",
            ))

    threads: list[threading.Thread] = []
    started = time.time()
    for slot_id in range(slots):
        gpu = gpus[slot_id % len(gpus)]
        thread = threading.Thread(
            target=_worker,
            args=(slot_id, gpu, task_queue, errors, args.device, bool(args.disable_dense_diagnostic)),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    for _ in threads:
        task_queue.put(None)
    task_queue.join()
    for thread in threads:
        thread.join()
    if not errors.empty():
        failures = []
        while not errors.empty():
            failures.append(errors.get())
        raise RuntimeError("parallel open-loop suite failed:\n" + "\n".join(failures))
    wall_time = time.time() - started
    summaries = {system.name: _merge(system, out_root, slots, wall_time) for system in systems}
    paired_hashes = {name: summary.get("scenario_timestamp_sha256") for name, summary in summaries.items()}
    paired_counts = {name: int(summary.get("num_scenarios", -1)) for name, summary in summaries.items()}
    paired_protocol_pass = len(set(paired_hashes.values())) == 1 and len(set(paired_counts.values())) == 1
    if not paired_protocol_pass:
        raise RuntimeError(
            f"paired open-loop protocol mismatch: hashes={paired_hashes}, counts={paired_counts}"
        )
    report = {
        "systems": [system.name for system in systems],
        "system_sources": {
            system.name: {
                "config_path": str(system.config),
                "config_sha256": _sha256_file(system.config),
                "checkpoint_path": str(system.checkpoint),
                "checkpoint_sha256": _sha256_file(system.checkpoint),
            }
            for system in systems
        },
        "num_scenarios_per_system": scenario_count,
        "num_worker_slots": slots,
        "gpus": gpus,
        "workers_per_gpu": int(args.workers_per_gpu),
        "suite_wall_time_s": wall_time,
        "paired_protocol_pass": paired_protocol_pass,
        "scenario_timestamp_sha256": next(iter(paired_hashes.values())),
        "summaries": summaries,
    }
    (out_root / "parallel_open_loop_suite_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
