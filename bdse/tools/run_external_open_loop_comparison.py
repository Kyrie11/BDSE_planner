from __future__ import annotations

import argparse
import csv
import hashlib
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


@dataclass(frozen=True)
class Task:
    name: str
    label: str
    config: Path
    checkpoint: Path | None
    output_dir: Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_hash(path: Path) -> tuple[str, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [(str(r.get("scenario_token", "")), int(r.get("timestamp_us", 0) or 0)) for r in rows]
    if any(not token for token, _ in keys) or len(keys) != len(set(keys)):
        raise ValueError(f"invalid/duplicate scenario keys: {path}")
    payload = "".join(f"{token}::{ts}\n" for token, ts in keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(keys)


def worker(gpu: str, tasks: "queue.Queue[Task | None]", errors: "queue.Queue[str]", args: argparse.Namespace) -> None:
    while True:
        task = tasks.get()
        try:
            if task is None:
                return
            task.output_dir.mkdir(parents=True, exist_ok=True)
            metrics = task.output_dir / "metrics.json"
            rows = task.output_dir / "metrics.jsonl"
            run_manifest = task.output_dir / "run_manifest.json"
            expected_manifest = {
                "config_sha256": sha256(task.config),
                "checkpoint_sha256": "" if task.checkpoint is None else sha256(task.checkpoint),
                "split": args.split,
                "max_scenarios": int(args.max_scenarios),
                "preprocessed_dir": str(Path(args.preprocessed_dir).resolve()),
                "disable_dense": bool(args.disable_dense),
            }
            if args.resume and metrics.is_file() and rows.is_file() and run_manifest.is_file():
                try:
                    existing = json.loads(run_manifest.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
                if existing == expected_manifest:
                    continue
            cmd = [
                sys.executable, "-m", "bdse.experiments.evaluate_open_loop",
                "--config", str(task.config),
                "--split", args.split,
                "--preprocessed-dir", str(args.preprocessed_dir),
                "--max-scenarios", str(args.max_scenarios),
                "--device", args.device,
                "--output", str(metrics),
                "--per-sample-output", str(rows),
            ]
            if task.checkpoint is not None:
                cmd += ["--checkpoint", str(task.checkpoint)]
            if args.disable_dense:
                cmd.append("--disable-dense-diagnostic")
            env = os.environ.copy()
            if args.device == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = gpu
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env.setdefault(key, "1")
            started = time.time()
            with (task.output_dir / "run.log").open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            (task.output_dir / "wall_time_s.txt").write_text(f"{time.time() - started:.6f}\n", encoding="utf-8")
            if proc.returncode != 0:
                raise RuntimeError(f"{task.name} failed; see {task.output_dir / 'run.log'}")
            run_manifest.write_text(json.dumps(expected_manifest, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:
            errors.put(f"gpu={gpu}: {type(exc).__name__}: {exc}")
        finally:
            tasks.task_done()


def numeric_metrics(data: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[key] = float(value)
    return out


def summarize(tasks: list[Task], root: Path) -> None:
    rows: list[dict[str, Any]] = []
    scenario_hashes: set[str] = set()
    for task in tasks:
        metrics_path = task.output_dir / "metrics.json"
        sample_path = task.output_dir / "metrics.jsonl"
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        digest, count = row_hash(sample_path)
        scenario_hashes.add(digest)
        row: dict[str, Any] = {
            "system": task.name,
            "implementation_label": task.label,
            "num_scenarios": count,
            "scenario_timestamp_sha256": digest,
            "config": str(task.config.resolve()),
            "config_sha256": sha256(task.config),
            "checkpoint": "" if task.checkpoint is None else str(task.checkpoint.resolve()),
            "checkpoint_sha256": "" if task.checkpoint is None else sha256(task.checkpoint),
            "wall_time_s": float((task.output_dir / "wall_time_s.txt").read_text().strip()),
        }
        row.update(numeric_metrics(data))
        rows.append(row)
    if len(scenario_hashes) != 1:
        raise RuntimeError(f"open-loop systems are not paired: {sorted(scenario_hashes)}")
    all_fields = sorted({key for row in rows for key in row}, key=lambda x: (x not in {"system", "implementation_label", "num_scenarios", "scenario_timestamp_sha256"}, x))
    with (root / "open_loop_all_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader(); writer.writerows(rows)
    (root / "open_loop_all_metrics.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    key_metrics = [
        "teacher_action_match", "teacher_regret", "planner_latency_ms", "planner_latency_ms_p95",
        "collision_avoidance", "time_to_collision", "drivable_area_compliance", "progress",
        "comfort", "selected_decisive_atom_recall", "selected_interaction_decisive_recall",
        "effective_query_atom_count", "fallback_would_trigger_rate",
    ]
    lines = [
        "# Paired open-loop comparison",
        "",
        f"All systems use {rows[0]['num_scenarios'] if rows else 0} identical scenario/timestamp pairs.",
        "",
        "| System | Teacher match | Teacher regret | p95 latency ms | Effective atoms | Fallback |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def val(*names: str) -> float:
            for name in names:
                if isinstance(row.get(name), (int, float)):
                    return float(row[name])
            return float("nan")
        lines.append(
            f"| {row['system']} | {val('teacher_action_match'):.4f} | {val('teacher_regret'):.3f} | "
            f"{val('planner_latency_ms_p95', 'planner_latency_ms'):.1f} | "
            f"{val('effective_query_atom_count', 'decision_budget_atom_count'):.2f} | "
            f"{val('fallback_would_trigger_rate', 'fallback_would_trigger'):.4f} |"
        )
    lines += ["", "The CSV/JSON files retain every finite numeric metric emitted by evaluate_open_loop.", "", "## Suggested plotting columns", "", ", ".join(key_metrics)]
    (root / "open_loop_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run paired V60/external-adapter open-loop comparison on two GPUs.")
    p.add_argument("--v60-config", type=Path, required=True)
    p.add_argument("--v60-checkpoint", type=Path, required=True)
    p.add_argument("--external-checkpoint-root", type=Path, required=True)
    p.add_argument("--preprocessed-dir", type=Path, required=True)
    p.add_argument("--split", default="val_tune")
    p.add_argument("--max-scenarios", type=int, default=1000)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--disable-dense", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    specs = [
        ("v60", "V60 DWAPC-BFAR-DBAP", args.v60_config, args.v60_checkpoint),
        ("gameformer", "GameFormer-inspired budget adapter", Path("bdse/configs/external_gameformer_budgeted_fast_cl.yaml"), args.external_checkpoint_root / "gameformer_budgeted.best.pt"),
        ("dtpp", "DTPP-inspired budget adapter", Path("bdse/configs/external_dtpp_budgeted_fast_cl.yaml"), args.external_checkpoint_root / "dtpp_budgeted.best.pt"),
        ("plantf", "PlanTF-inspired budget adapter", Path("bdse/configs/external_plantf_budgeted_fast_cl.yaml"), args.external_checkpoint_root / "plantf_budgeted.best.pt"),
        ("pluto", "PLUTO-inspired budget adapter", Path("bdse/configs/external_pluto_budgeted_fast_cl.yaml"), args.external_checkpoint_root / "pluto_budgeted.best.pt"),
        ("pdm_closed_style", "PDM-Closed-style budget scorer", Path("bdse/configs/external_pdm_closed_budgeted_fast_cl.yaml"), None),
    ]
    tasks: list[Task] = []
    for name, label, config, checkpoint in specs:
        if not config.is_file():
            raise FileNotFoundError(config)
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        tasks.append(Task(name, label, config, checkpoint, args.output_root / name))

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()] if args.device == "cuda" else ["cpu"]
    slots = [gpu for gpu in gpus for _ in range(max(1, args.workers_per_gpu))]
    q: "queue.Queue[Task | None]" = queue.Queue()
    errors: "queue.Queue[str]" = queue.Queue()
    for task in tasks:
        q.put(task)
    threads = [threading.Thread(target=worker, args=(gpu, q, errors, args), daemon=True) for gpu in slots]
    for t in threads:
        t.start()
    for _ in threads:
        q.put(None)
    q.join()
    for t in threads:
        t.join()
    if not errors.empty():
        raise RuntimeError("\n".join(list(errors.queue)))
    summarize(tasks, args.output_root)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "systems": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()
