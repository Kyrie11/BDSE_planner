from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Task:
    name: str
    budget: int
    config: Path
    checkpoint: Path | None
    output_dir: Path


def _strict_budget_config(source: Path, output: Path, budget: int, *, bdse_local: bool) -> None:
    cfg = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"expected YAML mapping: {source}")
    cfg.setdefault("evidence", {})["budget"] = int(budget)
    selector = cfg.setdefault("selector", {})
    selector["min_selected_atoms"] = int(budget)
    selector["force_fill_budget"] = True
    max_atoms = int((cfg.get("evidence", {}) or {}).get("max_atoms", 128))
    selector["proposal_top_m"] = min(max_atoms, max(24, 3 * int(budget)))
    fallback = cfg.setdefault("fallback", {})
    fallback["enabled"] = False
    fallback["max_additional_stages"] = 0
    fallback["budget_stages"] = [int(budget)]
    external = cfg.get("external_baseline")
    if isinstance(external, dict):
        external["budget"] = int(budget)
    if bdse_local:
        runtime = cfg.setdefault("runtime", {})
        runtime["disable_pair_residual_intervention"] = True
        dual = runtime.setdefault("dual_certificate", {})
        dual["residual_epsilon_cal"] = 0.0
    cfg.setdefault("experiment", {})["strict_budget_sweep"] = {
        "budget": int(budget),
        "fallback_disabled": True,
        "proposal_top_m": int(selector["proposal_top_m"]),
        "bdse_mode": "selected_local_no_residual" if bdse_local else "external_adapter",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _row_hash(path: Path) -> tuple[str, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [(str(r.get("scenario_token", "")), int(r.get("timestamp_us", 0) or 0)) for r in rows]
    if any(not k[0] for k in keys) or len(set(keys)) != len(keys):
        raise ValueError(f"empty or duplicate scenario keys in {path}")
    payload = "".join(f"{a}::{b}\n" for a, b in keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(keys)


def _worker(gpu: str, tasks: "queue.Queue[Task | None]", errors: "queue.Queue[str]", args: argparse.Namespace) -> None:
    while True:
        task = tasks.get()
        try:
            if task is None:
                return
            task.output_dir.mkdir(parents=True, exist_ok=True)
            metrics = task.output_dir / "metrics.json"
            rows = task.output_dir / "metrics.jsonl"
            if args.resume and metrics.is_file() and rows.is_file():
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
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            with (task.output_dir / "run.log").open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"failed {task.name} B={task.budget}; see {task.output_dir / 'run.log'}")
        except Exception as exc:
            errors.put(f"gpu={gpu}: {type(exc).__name__}: {exc}")
        finally:
            tasks.task_done()


def _metric(data: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float("nan")


def _summarize(tasks: list[Task], output_root: Path) -> None:
    records: list[dict[str, Any]] = []
    hashes: dict[int, set[str]] = {}
    for task in tasks:
        data = json.loads((task.output_dir / "metrics.json").read_text(encoding="utf-8"))
        digest, count = _row_hash(task.output_dir / "metrics.jsonl")
        hashes.setdefault(task.budget, set()).add(digest)
        records.append({
            "system": task.name,
            "budget": task.budget,
            "num_scenarios": count,
            "teacher_action_match": _metric(data, "teacher_action_match"),
            "teacher_regret": _metric(data, "teacher_regret"),
            "planner_latency_ms_p95": _metric(data, "planner_latency_ms_p95", "planner_latency_ms"),
            "effective_query_atom_count": _metric(data, "effective_query_atom_count", "decision_budget_atom_count"),
            "selected_decisive_recall": _metric(data, "selected_decisive_atom_recall"),
            "interaction_decisive_recall": _metric(data, "selected_interaction_decisive_recall"),
            "fallback_rate": _metric(data, "fallback_would_trigger_rate", "fallback_would_trigger"),
            "scenario_timestamp_sha256": digest,
        })
    bad = {b: list(v) for b, v in hashes.items() if len(v) != 1}
    if bad:
        raise RuntimeError(f"systems are not paired on identical scenario order: {bad}")
    fields = list(records[0]) if records else []
    with (output_root / "budget_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
    (output_root / "budget_sweep.json").write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Strict fixed-budget open-loop comparison", "", "Fallback is disabled. BDSE uses the selected-local/no-residual path so cross-budget results do not reuse a B=16 residual certificate.", "", "| System | B | Teacher match | Teacher regret | p95 latency ms | Effective atoms |", "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(records, key=lambda x: (x["budget"], x["system"])):
        lines.append(f"| {r['system']} | {r['budget']} | {r['teacher_action_match']:.4f} | {r['teacher_regret']:.2f} | {r['planner_latency_ms_p95']:.1f} | {r['effective_query_atom_count']:.2f} |")
    (output_root / "budget_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run paired strict-budget BDSE/external-adapter open-loop sweeps.")
    p.add_argument("--bdse-config", type=Path, required=True)
    p.add_argument("--bdse-checkpoint", type=Path, required=True)
    p.add_argument("--external-checkpoint-root", type=Path, required=True)
    p.add_argument("--preprocessed-dir", type=Path, required=True)
    p.add_argument("--split", default="val_tune")
    p.add_argument("--max-scenarios", type=int, default=1000)
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 24, 32])
    p.add_argument("--external", nargs="+", default=["gameformer", "dtpp", "plantf", "pluto"])
    p.add_argument("--include-pdm-closed", action="store_true")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--disable-dense", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_root = args.output_root / "configs"
    tasks: list[Task] = []
    for budget in args.budgets:
        bdse_cfg = config_root / f"bdse_local_B{budget}.yaml"
        _strict_budget_config(args.bdse_config, bdse_cfg, budget, bdse_local=True)
        tasks.append(Task("bdse_local", budget, bdse_cfg, args.bdse_checkpoint, args.output_root / f"B{budget}" / "bdse_local"))
        for name in args.external:
            src = Path(f"bdse/configs/external_{name}_budgeted_fast_cl.yaml")
            checkpoint = args.external_checkpoint_root / f"{name}_budgeted.best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing external checkpoint: {checkpoint}")
            cfg = config_root / f"{name}_B{budget}.yaml"
            _strict_budget_config(src, cfg, budget, bdse_local=False)
            tasks.append(Task(name, budget, cfg, checkpoint, args.output_root / f"B{budget}" / name))
        if args.include_pdm_closed:
            src = Path("bdse/configs/external_pdm_closed_budgeted_fast_cl.yaml")
            cfg = config_root / f"pdm_closed_B{budget}.yaml"
            _strict_budget_config(src, cfg, budget, bdse_local=False)
            tasks.append(Task("pdm_closed", budget, cfg, None, args.output_root / f"B{budget}" / "pdm_closed"))
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()] if args.device == "cuda" else ["cpu"]
    slots = [gpu for gpu in gpus for _ in range(max(1, args.workers_per_gpu))]
    q: "queue.Queue[Task | None]" = queue.Queue(); errors: "queue.Queue[str]" = queue.Queue()
    for task in tasks: q.put(task)
    threads=[]
    for gpu in slots:
        t=threading.Thread(target=_worker,args=(gpu,q,errors,args),daemon=True); t.start(); threads.append(t)
    for _ in threads: q.put(None)
    q.join()
    for t in threads: t.join()
    if not errors.empty():
        raise RuntimeError("\n".join(list(errors.queue)))
    _summarize(tasks, args.output_root)
    print(json.dumps({"output_root": str(args.output_root), "tasks": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()
