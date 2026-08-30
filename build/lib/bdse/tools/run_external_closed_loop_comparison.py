from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bdse.data.nuplan_dataset import PreprocessedBDSEDataset


@dataclass(frozen=True)
class SystemTask:
    name: str
    label: str
    config: Path
    checkpoint: Path | None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tokens(cache_root: Path, split: str, limit: int, scan_max: int) -> list[str]:
    paths = PreprocessedBDSEDataset(cache_root, split=[split], max_scenarios=max(limit, scan_max)).build_index()
    tokens: list[str] = []
    seen: set[str] = set()
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as z:
                value = z["scenario_token"]
                token = str(value.item() if value.shape == () else value.reshape(-1)[0])
        except Exception:
            continue
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)
        if len(tokens) >= limit:
            break
    if len(tokens) < limit:
        raise RuntimeError(f"only found {len(tokens)} unique scenario tokens; need {limit}; increase --token-scan-max")
    return tokens


def parse_success(log_text: str) -> tuple[int, int]:
    success = re.findall(r"Number of successful simulations:\s*(\d+)", log_text)
    failed = re.findall(r"Number of failed simulations:\s*(\d+)", log_text)
    return (int(success[-1]) if success else -1, int(failed[-1]) if failed else -1)


def final_metric_row(root: Path) -> tuple[dict[str, float], Path]:
    files = sorted(root.glob("**/aggregator_metric/*.parquet"))
    if not files:
        raise RuntimeError(f"no aggregator_metric parquet under {root}")
    for path in files:
        df = pd.read_parquet(path)
        if "scenario" not in df.columns:
            continue
        final = df[df["scenario"] == "final_score"]
        if final.empty:
            continue
        row: dict[str, float] = {}
        for key, value in final.iloc[0].items():
            if isinstance(value, (bool, int, float, np.integer, np.floating)) and math.isfinite(float(value)):
                row[str(key)] = float(value)
        return row, path
    raise RuntimeError(f"no final_score row under {root}")


def run_shard(
    *,
    system: SystemTask,
    gpu: str,
    shard_id: int,
    shard_tokens: list[str],
    system_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    shard_root = system_root / f"shard{shard_id}"
    shard_root.mkdir(parents=True, exist_ok=True)
    token_file = shard_root / "scenario_tokens.json"
    token_file.write_text(json.dumps(shard_tokens, indent=2), encoding="utf-8")
    if not shard_tokens:
        return {"shard": shard_id, "scenario_count": 0, "metrics": {}, "wall_time_s": 0.0}
    token_override = "scenario_filter.scenario_tokens=" + json.dumps(shard_tokens, separators=(",", ":"))
    cmd = [
        sys.executable, "-m", "bdse.experiments.evaluate_closed_loop",
        "--config", str(system.config),
        "--device", args.device,
        "--challenge", args.challenge,
        "--metric-aggregator", args.metric_aggregator,
        "--output-dir", str(shard_root),
        "--experiment-uid", f"{system.name}_{args.challenge}_{args.limit}_shard{shard_id}",
        "--nuplan-module", "nuplan.planning.script.run_simulation",
        "--scenario-builder", "nuplan",
        "--worker", "single_machine_thread_pool",
        "--hydra-full-error",
        "--nuplan-data-root", str(args.nuplan_root),
        "--nuplan-map-root", str(args.nuplan_root / "maps"),
        "--nuplan-exp-root", str(args.nuplan_root / "exp"),
        "--nuplan-db-root", str(args.nuplan_db_root),
    ]
    if system.checkpoint is not None:
        cmd += ["--checkpoint", str(system.checkpoint)]
    cmd += [
        "--", token_override,
        f"scenario_filter.limit_total_scenarios={len(shard_tokens)}",
        "scenario_filter.shuffle=false",
        "worker.max_workers=1",
        "run_metric=true",
        "~callback.simulation_log_callback",
    ]
    env = os.environ.copy()
    if args.device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env.update({
        "BDSE_SHARE_MODEL_PER_PROCESS": "1",
        "BDSE_SERIALIZE_GPU_INFERENCE": "0",
        "BDSE_PROFILE_CLOSED_LOOP": "1",
        "BDSE_CLOSED_LOOP_PROFILE_JSON": str(shard_root / "bdse_closed_loop_profile.json"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    log_path = shard_root / "run.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    wall = time.time() - started
    text = log_path.read_text(encoding="utf-8", errors="replace")
    success, failed = parse_success(text)
    if proc.returncode != 0 or success != len(shard_tokens) or failed != 0:
        raise RuntimeError(
            f"{system.name} shard{shard_id} invalid: return={proc.returncode}, successful={success}, failed={failed}, expected={len(shard_tokens)}; see {log_path}"
        )
    metrics, metric_file = final_metric_row(shard_root)
    return {
        "shard": shard_id,
        "scenario_count": len(shard_tokens),
        "successful": success,
        "failed": failed,
        "wall_time_s": wall,
        "metric_file": str(metric_file),
        "metrics": metrics,
        "profile_file": str(shard_root / "bdse_closed_loop_profile.json") if (shard_root / "bdse_closed_loop_profile.json").is_file() else "",
    }


def combine_shards(system: SystemTask, system_root: Path, shard_rows: list[dict[str, Any]], tokens_sha: str, wall_time: float) -> dict[str, Any]:
    nonempty = [row for row in shard_rows if int(row["scenario_count"]) > 0]
    total = sum(int(row["scenario_count"]) for row in nonempty)
    if total <= 0:
        raise RuntimeError(f"no completed scenarios for {system.name}")
    common = set.intersection(*(set(row["metrics"]) for row in nonempty)) if nonempty else set()
    combined: dict[str, Any] = {
        "system": system.name,
        "implementation_label": system.label,
        "scenario_count": total,
        "successful": sum(int(row["successful"]) for row in nonempty),
        "failed": sum(int(row["failed"]) for row in nonempty),
        "parallel_wall_time_s": wall_time,
        "scenarios_per_wall_hour": total / max(wall_time, 1e-9) * 3600.0,
        "process_shards": len(nonempty),
        "scenario_token_sha256": tokens_sha,
        "config": str(system.config.resolve()),
        "config_sha256": sha256(system.config),
        "checkpoint": "" if system.checkpoint is None else str(system.checkpoint.resolve()),
        "checkpoint_sha256": "" if system.checkpoint is None else sha256(system.checkpoint),
        "profile_files": [row["profile_file"] for row in nonempty if row.get("profile_file")],
    }
    for key in sorted(common):
        values = [row["metrics"].get(key) for row in nonempty]
        if all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values):
            combined[key] = sum(float(row["metrics"][key]) * int(row["scenario_count"]) for row in nonempty) / total
    (system_root / "closed_loop_shards.json").write_text(json.dumps(shard_rows, indent=2, sort_keys=True), encoding="utf-8")
    (system_root / "closed_loop_combined_summary.json").write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    marker = {
        "complete": True,
        "successful": combined["successful"],
        "failed": combined["failed"],
        "scenario_count": total,
        "scenario_token_sha256": tokens_sha,
        "summary": str((system_root / "closed_loop_combined_summary.json").resolve()),
    }
    (system_root / ".closed_loop_complete.json").write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    return combined


def run_system(system: SystemTask, gpu: str, tokens: list[str], token_sha: str, args: argparse.Namespace) -> dict[str, Any]:
    system_root = args.output_root / system.name
    marker = system_root / ".closed_loop_complete.json"
    summary = system_root / "closed_loop_combined_summary.json"
    if args.resume and marker.is_file() and summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        expected_checkpoint_sha = "" if system.checkpoint is None else sha256(system.checkpoint)
        if (
            int(data.get("scenario_count", -1)) == len(tokens)
            and str(data.get("scenario_token_sha256")) == token_sha
            and str(data.get("config_sha256", "")) == sha256(system.config)
            and str(data.get("checkpoint_sha256", "")) == expected_checkpoint_sha
        ):
            return data
    if system_root.exists():
        import shutil
        shutil.rmtree(system_root)
    system_root.mkdir(parents=True, exist_ok=True)
    shards = [tokens[i :: args.processes_per_model] for i in range(args.processes_per_model)]
    started = time.time()
    results: list[dict[str, Any] | None] = [None] * len(shards)
    errors: list[str] = []

    def target(idx: int, shard_tokens: list[str]) -> None:
        try:
            results[idx] = run_shard(system=system, gpu=gpu, shard_id=idx, shard_tokens=shard_tokens, system_root=system_root, args=args)
        except Exception as exc:
            errors.append(f"shard{idx}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=target, args=(i, shard), daemon=True) for i, shard in enumerate(shards)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"{system.name} failed on gpu={gpu}: " + " | ".join(errors))
    return combine_shards(system, system_root, [x for x in results if x is not None], token_sha, time.time() - started)


def scheduler_worker(gpu: str, q: "queue.Queue[SystemTask | None]", errors: "queue.Queue[str]", summaries: list[dict[str, Any]], lock: threading.Lock, tokens: list[str], token_sha: str, args: argparse.Namespace) -> None:
    while True:
        task = q.get()
        try:
            if task is None:
                return
            result = run_system(task, gpu, tokens, token_sha, args)
            with lock:
                summaries.append(result)
        except Exception as exc:
            errors.put(f"system={getattr(task, 'name', '?')} gpu={gpu}: {type(exc).__name__}: {exc}")
        finally:
            q.task_done()


def write_comparison(summaries: list[dict[str, Any]], root: Path, challenge: str) -> None:
    summaries = sorted(summaries, key=lambda row: row["system"])
    all_fields = sorted({key for row in summaries for key in row if key != "profile_files"}, key=lambda x: (x not in {"system", "implementation_label", "scenario_count", "successful", "failed"}, x))
    with (root / "closed_loop_all_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summaries)
    (root / "closed_loop_all_metrics.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")

    preferred = {
        "closed_loop_nonreactive_agents": ["score", "final_score", "closed_loop_nonreactive_agents_weighted_average"],
        "closed_loop_reactive_agents": ["score", "final_score", "closed_loop_reactive_agents_weighted_average"],
    }
    def metric(row: dict[str, Any], *names: str) -> float:
        for name in names:
            if isinstance(row.get(name), (int, float)):
                return float(row[name])
        return float("nan")
    lines = [
        f"# Closed-loop comparison: {challenge}", "",
        "All systems use the identical deterministic scenario-token list. Completion requires successful==expected and failed==0 for every shard.", "",
        "| System | Aggregate score | Collision | TTC | Drivable | Progress | Comfort | Wall h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        score = metric(row, *preferred.get(challenge, []), "planner_expert_average_l2_error_within_bound", "score")
        lines.append(
            f"| {row['system']} | {score:.4f} | "
            f"{metric(row, 'no_ego_at_fault_collisions', 'collision_avoidance'):.4f} | "
            f"{metric(row, 'time_to_collision_within_bound', 'time_to_collision'):.4f} | "
            f"{metric(row, 'drivable_area_compliance'):.4f} | "
            f"{metric(row, 'ego_progress_along_expert_route', 'progress'):.4f} | "
            f"{metric(row, 'ego_is_comfortable', 'comfort'):.4f} | "
            f"{float(row['parallel_wall_time_s']) / 3600.0:.2f} |"
        )
    lines += ["", "All finite final_score metrics are retained in closed_loop_all_metrics.csv/json."]
    (root / "closed_loop_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run paired V60/external-adapter closed-loop comparison with two-model GPU scheduling.")
    p.add_argument("--v60-config", type=Path, required=True)
    p.add_argument("--v60-checkpoint", type=Path, required=True)
    p.add_argument("--external-checkpoint-root", type=Path, required=True)
    p.add_argument("--split-cache", type=Path, required=True)
    p.add_argument("--token-split", default="val_tune")
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--token-scan-max", type=int, default=2000)
    p.add_argument("--nuplan-root", type=Path, required=True)
    p.add_argument("--nuplan-db-root", type=Path, default=None)
    p.add_argument("--challenge", choices=["closed_loop_nonreactive_agents", "closed_loop_reactive_agents"], default="closed_loop_nonreactive_agents")
    p.add_argument("--metric-aggregator", default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--processes-per-model", type=int, default=1)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--systems", nargs="+", default=["v60", "gameformer", "dtpp", "plantf", "pluto", "pdm_closed_style"])
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.processes_per_model < 1:
        raise ValueError("--processes-per-model must be >=1")
    if args.nuplan_db_root is None:
        args.nuplan_db_root = args.nuplan_root / "data" / "cache" / "val"
    if args.metric_aggregator is None:
        args.metric_aggregator = (
            "closed_loop_nonreactive_agents_weighted_average"
            if args.challenge == "closed_loop_nonreactive_agents"
            else "closed_loop_reactive_agents_weighted_average"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    tokens = load_tokens(args.split_cache, args.token_split, args.limit, args.token_scan_max)
    token_path = args.output_root / "scenario_tokens_all.json"
    token_path.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    token_sha = sha256(token_path)

    specs = {
        "v60": SystemTask("v60", "V60 DWAPC-BFAR-DBAP", args.v60_config, args.v60_checkpoint),
        "gameformer": SystemTask("gameformer", "GameFormer-inspired budget adapter", Path("bdse/configs/external_gameformer_budgeted_fast_cl.yaml"), args.external_checkpoint_root / "gameformer_budgeted.best.pt"),
        "dtpp": SystemTask("dtpp", "DTPP-inspired budget adapter", Path("bdse/configs/external_dtpp_budgeted_fast_cl.yaml"), args.external_checkpoint_root / "dtpp_budgeted.best.pt"),
        "plantf": SystemTask("plantf", "PlanTF-inspired budget adapter", Path("bdse/configs/external_plantf_budgeted_fast_cl.yaml"), args.external_checkpoint_root / "plantf_budgeted.best.pt"),
        "pluto": SystemTask("pluto", "PLUTO-inspired budget adapter", Path("bdse/configs/external_pluto_budgeted_fast_cl.yaml"), args.external_checkpoint_root / "pluto_budgeted.best.pt"),
        "pdm_closed_style": SystemTask("pdm_closed_style", "PDM-Closed-style budget scorer", Path("bdse/configs/external_pdm_closed_budgeted_fast_cl.yaml"), None),
    }
    tasks = [specs[name] for name in args.systems]
    for task in tasks:
        if not task.config.is_file():
            raise FileNotFoundError(task.config)
        if task.checkpoint is not None and not task.checkpoint.is_file():
            raise FileNotFoundError(task.checkpoint)

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()] if args.device == "cuda" else ["cpu"]
    q: "queue.Queue[SystemTask | None]" = queue.Queue()
    errors: "queue.Queue[str]" = queue.Queue()
    summaries: list[dict[str, Any]] = []
    lock = threading.Lock()
    for task in tasks:
        q.put(task)
    threads = [threading.Thread(target=scheduler_worker, args=(gpu, q, errors, summaries, lock, tokens, token_sha, args), daemon=True) for gpu in gpus]
    for t in threads:
        t.start()
    for _ in threads:
        q.put(None)
    q.join()
    for t in threads:
        t.join()
    if not errors.empty():
        raise RuntimeError("\n".join(list(errors.queue)))
    if len(summaries) != len(tasks):
        raise RuntimeError(f"only collected {len(summaries)}/{len(tasks)} system summaries")
    if {row.get("scenario_token_sha256") for row in summaries} != {token_sha}:
        raise RuntimeError("closed-loop scenario token hash mismatch")
    write_comparison(summaries, args.output_root, args.challenge)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "systems": len(tasks), "scenario_count": len(tokens), "token_sha256": token_sha}, indent=2))


if __name__ == "__main__":
    main()
