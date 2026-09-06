from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bdse.data.nuplan_dataset import PreprocessedBDSEDataset


METRIC_SAFE_NUPLAN_MODULE = "bdse.tools.nuplan_metric_safe_run_simulation"
METRIC_SAFE_ENV_KEY = "BDSE_PIOR_METRIC_ENGINE_SERIALIZATION"


@dataclass(frozen=True)
class SystemTask:
    name: str
    label: str
    budget: int
    config: Path
    checkpoint: Path | None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _token_hash(tokens: list[str]) -> str:
    return hashlib.sha256(("\n".join(tokens) + "\n").encode("utf-8")).hexdigest()


def _resume_summary_compatible(
    data: dict[str, Any],
    *,
    scenario_count: int,
    token_sha: str,
    config_sha: str,
    checkpoint_sha: str,
) -> bool:
    """Only resume results produced by the metric-safe benchmark path.

    V50.4 established that concurrent access to nuPlan's shared stateful
    MetricsEngine can fail loudly or silently contaminate final metrics.  Older
    fixed-budget results did not carry a metric-safety provenance marker, so
    they must not be reused after this benchmark repair.
    """
    return bool(
        int(data.get("scenario_count", -1)) == int(scenario_count)
        and str(data.get("scenario_token_sha256", "")) == str(token_sha)
        and str(data.get("config_sha256", "")) == str(config_sha)
        and str(data.get("checkpoint_sha256", "")) == str(checkpoint_sha)
        and data.get("metric_engine_serialized") is True
        and str(data.get("nuplan_module", "")) == METRIC_SAFE_NUPLAN_MODULE
    )


def _read_token_and_hint(path: Path) -> tuple[str, str] | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            value = z["scenario_token"]
            token = str(value.item() if value.shape == () else value.reshape(-1)[0])
    except Exception:
        return None
    token = token.strip()
    if not token:
        return None
    return token, path.parent.name


def load_tokens(
    cache_root: Path,
    split: str,
    limit: int,
    scan_max: int,
    *,
    scan_workers: int = 8,
    progress_interval_s: float = 5.0,
) -> tuple[list[str], list[str]]:
    """Load paired scenario tokens plus conservative raw-log filename hints.

    The NPZ parent folder in the user's cache layout is the acquisition/log name.
    We retain one parent-name hint per unique scenario token so a flat raw DB
    directory can optionally be narrowed to only the required logs.
    """
    if limit > 0:
        max_scenarios: int | None = max(int(limit), int(scan_max))
    else:
        max_scenarios = None
    print(f"[manifest] indexing NPZ cache: root={cache_root} split={split} limit={limit or 'ALL'}", flush=True)
    index_started = time.time()
    paths = PreprocessedBDSEDataset(cache_root, split=[split], max_scenarios=max_scenarios).build_index()
    print(
        f"[manifest] index ready: files={len(paths)} elapsed={time.time() - index_started:.1f}s; "
        f"reading scenario_token with workers={max(1, int(scan_workers)) if limit <= 0 else 1}",
        flush=True,
    )
    tokens: list[str] = []
    log_hints: list[str] = []
    seen: set[str] = set()
    scan_started = time.time()
    last_progress = scan_started

    # A small --limit is primarily used for debugging; keep that path sequential
    # so we can stop immediately after enough unique tokens are found.  The full
    # paper test set benefits from parallel NPZ decompression / metadata reads.
    if limit > 0 or int(scan_workers) <= 1:
        iterator = ((_read_token_and_hint(path)) for path in paths)
        executor = None
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(scan_workers)))
        iterator = executor.map(_read_token_and_hint, paths)

    try:
        for scanned, result in enumerate(iterator, start=1):
            if result is not None:
                token, hint = result
                if token not in seen:
                    seen.add(token)
                    tokens.append(token)
                    log_hints.append(hint)
            now = time.time()
            if now - last_progress >= max(0.5, float(progress_interval_s)) or scanned == len(paths):
                elapsed = max(now - scan_started, 1e-9)
                print(
                    f"[manifest-progress] files={scanned}/{len(paths)} ({100.0 * scanned / max(len(paths), 1):.1f}%) "
                    f"unique_tokens={len(tokens)} rate={scanned / elapsed:.1f} files/s elapsed={elapsed:.1f}s",
                    flush=True,
                )
                last_progress = now
            if limit > 0 and len(tokens) >= limit:
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if not tokens:
        raise RuntimeError(f"no scenario_token values found under cache={cache_root} split={split}")
    if limit > 0 and len(tokens) < limit:
        raise RuntimeError(
            f"only found {len(tokens)} unique scenario tokens; need {limit}; increase --token-scan-max or use --limit 0"
        )
    print(
        f"[manifest] ready: unique_tokens={len(tokens)} sha256={_token_hash(tokens)[:16]}... "
        f"elapsed={time.time() - scan_started:.1f}s",
        flush=True,
    )
    return tokens, log_hints


def _restrict_flat_db_files(root: Path, log_hints: list[str]) -> list[Path]:
    """Resolve a safe DB subset from NPZ parent folders, or return [] to fall back.

    Restriction is all-or-nothing: every selected token's log hint must map to a
    direct ``<hint>.db`` file.  This prevents a naming mismatch from silently
    removing requested scenarios.
    """
    if not root.is_dir() or not log_hints:
        return []
    direct = sorted(root.glob("*.db"), key=lambda p: str(p))
    if not direct:
        return []
    by_stem = {p.stem: p for p in direct}
    resolved: list[Path] = []
    seen: set[Path] = set()
    for hint in log_hints:
        db = by_stem.get(str(hint))
        if db is None:
            return []
        if db not in seen:
            seen.add(db); resolved.append(db)
    return resolved


def _validate_db_root(root: Path) -> None:
    if root.is_file() and root.suffix == ".db":
        return
    if not root.is_dir():
        raise FileNotFoundError(f"nuPlan DB root does not exist: {root}")
    try:
        next(root.rglob("*.db"))
    except StopIteration as exc:
        raise RuntimeError(
            f"No .db files found under {root}. Closed-loop nuPlan simulation needs raw SQLite log DBs; "
            "the bdse_test_2/public_set_test directory contains NPZ tensor/cache artifacts and belongs in --split-cache instead."
        ) from exc


def strict_budget_config(source: Path, output: Path, *, budget: int, proposal_top_m: int, own_model: bool) -> None:
    cfg = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"expected YAML mapping: {source}")
    source_budget = int((cfg.get("evidence", {}) or {}).get("budget", 16))
    cfg.setdefault("evidence", {})["budget"] = int(budget)
    selector = cfg.setdefault("selector", {})
    selector["min_selected_atoms"] = int(budget)
    selector["force_fill_budget"] = True
    selector["proposal_top_m"] = int(proposal_top_m)
    fallback = cfg.setdefault("fallback", {})
    fallback["enabled"] = False
    fallback["max_additional_stages"] = 0
    fallback["budget_stages"] = [int(budget)]
    # Do not let an auxiliary fallback/rival stage silently become an extra
    # evidence budget.  Rival candidate count remains a separate quantity.
    external = cfg.get("external_baseline")
    if isinstance(external, dict):
        external["budget"] = int(budget)
    exp = cfg.setdefault("experiment", {})
    exp["fixed_budget_closed_loop"] = {
        "budget": int(budget),
        "proposal_top_m": int(proposal_top_m),
        "fallback_disabled": True,
        "source_budget": int(source_budget),
        "own_model": bool(own_model),
        "cross_budget_frozen_policy": bool(own_model and source_budget != int(budget)),
        "note": (
            "For the own model, B != source_budget changes only the deployed evidence interface while preserving the learned/fitted method. "
            "If calibration was fit only at source_budget, treat this as a frozen-policy cross-budget ablation unless per-budget calibration is refit."
            if own_model else
            "External adapters are expected to use the budget-specific checkpoint trained/selected for this B; the runner rejects shared checkpoints unless --allow-shared-external-checkpoint is explicitly requested."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


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


def _gpu_memory_used_mb(gpu: str) -> str:
    """Best-effort physical-GPU memory telemetry for progress logs."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
        if proc.returncode == 0:
            raw = proc.stdout.strip().splitlines()
            if raw:
                return f"{int(float(raw[0].strip()))}MB"
    except Exception:
        pass
    return "n/a"


def _tail_log(path: Path, max_bytes: int = 32768) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    # tqdm/progress bars often use carriage returns. Normalize them so the last
    # visible update can be surfaced in a one-line heartbeat.
    text = text.replace("\r", "\n")
    ansi = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
    lines = [ansi.sub("", line).strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    line = lines[-1]
    return line[-240:]


def _closed_loop_phase(log_path: Path) -> str:
    try:
        text = _tail_log(log_path, max_bytes=131072)
    except Exception:
        text = ""
    # _tail_log returns one line, so also inspect a bounded suffix for the
    # explicit planner-ready marker.
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            f.seek(max(0, size - 131072))
            suffix = f.read().decode("utf-8", errors="replace")
    except Exception:
        suffix = ""
    if "Number of successful simulations:" in suffix:
        return "metrics/finalizing"
    if "[planner-ready]" in suffix or "BDSEnuPlanPlanner device:" in suffix:
        return "planner-loaded/simulating"
    if text:
        return "nuplan-init/scenario-build"
    return "process-starting"


def _latest_marker_line(log_path: Path, marker: str, max_bytes: int = 262144) -> str:
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", errors="replace").replace("\r", "\n")
    except Exception:
        return ""
    matches = [line.strip() for line in text.splitlines() if marker in line]
    return matches[-1][-500:] if matches else ""


def run_task(task: SystemTask, gpu: str, tokens: list[str], token_sha: str, args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root / f"B{task.budget}" / task.name
    summary_path = root / "closed_loop_summary.json"
    complete_path = root / ".closed_loop_complete.json"
    expected_ckpt_sha = "" if task.checkpoint is None else sha256(task.checkpoint)
    if args.resume and summary_path.is_file() and complete_path.is_file():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if _resume_summary_compatible(
            data,
            scenario_count=len(tokens),
            token_sha=token_sha,
            config_sha=sha256(task.config),
            checkpoint_sha=expected_ckpt_sha,
        ):
            print(
                f"[task-resume] system={task.name} B={task.budget} already complete; "
                f"scenarios={len(tokens)} output={root}",
                flush=True,
            )
            return data
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    token_manifest = root / "scenario_tokens.json"
    token_manifest.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    # Never serialize the full test population into this process' argv.  The
    # full public_set_test contains O(10^5) tokens and a single Hydra override
    # such as scenario_filter.scenario_tokens=[...] can exceed Linux'
    # MAX_ARG_STRLEN before Python/nuPlan even starts.  Pass a short manifest
    # path to the evaluator; it validates the hash, constructs the Hydra
    # override in memory, and dispatches nuPlan in-process when needed.
    task_device = "cpu" if task.name == "pdm_closed_style" else args.device
    cmd = [
        sys.executable, "-m", "bdse.experiments.evaluate_closed_loop",
        "--config", str(task.config),
        "--device", task_device,
        "--challenge", args.challenge,
        "--metric-aggregator", args.metric_aggregator,
        "--output-dir", str(root),
        "--experiment-uid", f"fixedB{task.budget}_{task.name}_{args.challenge}_{len(tokens)}",
        "--nuplan-module", METRIC_SAFE_NUPLAN_MODULE,
        "--scenario-builder", "nuplan",
        "--worker", "single_machine_thread_pool",
        "--hydra-full-error",
        "--nuplan-data-root", str(args.nuplan_root),
        "--nuplan-map-root", str(args.nuplan_map_root),
        "--nuplan-exp-root", str(args.nuplan_exp_root),
        "--scenario-tokens-file", str(token_manifest),
        "--scenario-tokens-sha256", str(token_sha),
    ]
    if args.resolved_db_files:
        db_manifest = root / "nuplan_db_files.json"
        db_manifest.write_text(
            json.dumps([str(p.resolve()) for p in args.resolved_db_files], indent=2),
            encoding="utf-8",
        )
        cmd += ["--nuplan-db-files-file", str(db_manifest)]
    else:
        cmd += ["--nuplan-db-root", str(args.nuplan_db_root)]
    if task.checkpoint is not None:
        cmd += ["--checkpoint", str(task.checkpoint)]
    cmd += [
        "--",
        f"scenario_filter.limit_total_scenarios={len(tokens)}",
        "scenario_filter.shuffle=false",
        "scenario_filter.log_names=null",
        f"worker.max_workers={max(1, int(args.workers_per_job))}",
        "run_metric=true",
        "~callback.simulation_log_callback",
    ]
    env = os.environ.copy()
    if task_device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env.update({
        "PYTHONUNBUFFERED": "1",
        "BDSE_SHARE_MODEL_PER_PROCESS": "1",
        "BDSE_SERIALIZE_GPU_INFERENCE": "0",
        # Each suite subprocess is already pinned to one physical GPU via
        # CUDA_VISIBLE_DEVICES, so planner-level multi-GPU sharding is neither
        # needed nor desirable here.
        "BDSE_SHARD_PLANNERS_ACROSS_GPUS": "0",
        "BDSE_PROFILE_CLOSED_LOOP": "1",
        "BDSE_CLOSED_LOOP_PROFILE_JSON": str(root / "bdse_closed_loop_profile.json"),
        METRIC_SAFE_ENV_KEY: "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    started = time.time()
    log_path = root / "run.log"
    checkpoint_text = "none (CPU rule scorer)" if task.checkpoint is None else str(task.checkpoint)
    print(
        f"[task-start] system={task.name} B={task.budget} physical_gpu={gpu if task_device == 'cuda' else 'CPU'} "
        f"device={task_device} scenarios={len(tokens)} workers={args.workers_per_job} checkpoint={checkpoint_text}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(
            f"[task-process] system={task.name} B={task.budget} pid={proc.pid} log={log_path}",
            flush=True,
        )
        heartbeat = max(2.0, float(args.heartbeat_seconds))
        planner_ready_reported = False
        while True:
            try:
                returncode = proc.wait(timeout=heartbeat)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.time() - started
                phase = _closed_loop_phase(log_path)
                gpu_mem = _gpu_memory_used_mb(gpu) if task_device == "cuda" else "cpu"
                if not planner_ready_reported:
                    marker_line = _latest_marker_line(log_path, "[planner-ready]")
                    if marker_line:
                        planner_ready_reported = True
                        print(
                            f"[task-gpu-ready] system={task.name} B={task.budget} physical_gpu={gpu if task_device == 'cuda' else 'CPU'} "
                            f"{marker_line}",
                            flush=True,
                        )
                tail = _tail_log(log_path)
                print(
                    f"[task-heartbeat] system={task.name} B={task.budget} pid={proc.pid} "
                    f"phase={phase} elapsed={elapsed:.0f}s gpu_mem={gpu_mem}"
                    + (f" last='{tail}'" if tail else ""),
                    flush=True,
                )
        if not planner_ready_reported:
            marker_line = _latest_marker_line(log_path, "[planner-ready]")
            if marker_line:
                print(
                    f"[task-gpu-ready] system={task.name} B={task.budget} physical_gpu={gpu if task_device == 'cuda' else 'CPU'} "
                    f"{marker_line}",
                    flush=True,
                )
    wall = time.time() - started
    text = log_path.read_text(encoding="utf-8", errors="replace")
    success, failed = parse_success(text)
    if returncode != 0 or success != len(tokens) or failed != 0:
        tail_lines = "\n".join(text.replace("\r", "\n").splitlines()[-20:])
        print(
            f"[task-failed] system={task.name} B={task.budget} return={returncode} successful={success} failed={failed}\n"
            f"--- last 20 log lines: {log_path} ---\n{tail_lines}",
            flush=True,
        )
        raise RuntimeError(
            f"{task.name} B={task.budget} invalid: return={returncode}, successful={success}, failed={failed}, "
            f"expected={len(tokens)}; see {log_path}"
        )
    metrics, metric_file = final_metric_row(root)
    summary: dict[str, Any] = {
        "system": task.name,
        "implementation_label": task.label,
        "budget": int(task.budget),
        "proposal_top_m": int(args.proposal_top_m),
        "scenario_count": len(tokens),
        "successful": success,
        "failed": failed,
        "wall_time_s": wall,
        "scenarios_per_wall_hour": len(tokens) / max(wall, 1e-9) * 3600.0,
        "scenario_token_sha256": token_sha,
        "nuplan_db_mode": "restricted_files" if args.resolved_db_files else "root",
        "nuplan_db_file_count": len(args.resolved_db_files) if args.resolved_db_files else -1,
        "nuplan_workers_per_job": int(args.workers_per_job),
        "token_scan_workers": int(args.token_scan_workers),
        "heartbeat_seconds": float(args.heartbeat_seconds),
        "schedule_mode": str(args.schedule_mode),
        "device": task_device,
        "config": str(task.config.resolve()),
        "config_sha256": sha256(task.config),
        "checkpoint": "" if task.checkpoint is None else str(task.checkpoint.resolve()),
        "checkpoint_sha256": expected_ckpt_sha,
        "metric_file": str(metric_file),
        "metric_engine_serialized": True,
        "nuplan_module": METRIC_SAFE_NUPLAN_MODULE,
        "budget_semantics_warning": (
            "Current repository PDM-Closed-style scorer is not the official PDM-Closed planner and its static J0 does not consume selected evidence; B is interface accounting rather than the original planner's proposal count."
            if task.name == "pdm_closed_style" else
            (
                "BDSE was fit at B=16; B=8/B=24 are frozen-policy cross-budget robustness ablations, not budget-specific retraining. Use B=16 as the primary matched-interface comparison unless a preregistered per-budget BDSE fit is supplied."
                if task.name == "bdse" and int(task.budget) != 16 else ""
            )
        ),
    }
    summary.update(metrics)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    complete_path.write_text(json.dumps({
        "complete": True,
        "scenario_token_sha256": token_sha,
        "metric_engine_serialized": True,
        "nuplan_module": METRIC_SAFE_NUPLAN_MODULE,
    }, indent=2), encoding="utf-8")
    print(
        f"[task-done] system={task.name} B={task.budget} scenarios={success}/{len(tokens)} "
        f"wall={wall / 60.0:.1f}min throughput={summary['scenarios_per_wall_hour']:.1f} scenarios/h output={root}",
        flush=True,
    )
    return summary


def _worker(gpu: str, q: "queue.Queue[SystemTask | None]", errors: "queue.Queue[str]", summaries: list[dict[str, Any]], lock: threading.Lock, tokens: list[str], token_sha: str, args: argparse.Namespace) -> None:
    while True:
        task = q.get()
        try:
            if task is None:
                return
            result = run_task(task, gpu, tokens, token_sha, args)
            with lock:
                summaries.append(result)
        except Exception as exc:
            errors.put(f"system={getattr(task, 'name', '?')} B={getattr(task, 'budget', '?')} gpu={gpu}: {type(exc).__name__}: {exc}")
        finally:
            q.task_done()


def _sequence_worker(gpu: str, tasks: list[SystemTask], errors: "queue.Queue[str]", summaries: list[dict[str, Any]], lock: threading.Lock, tokens: list[str], token_sha: str, args: argparse.Namespace) -> None:
    """Run all budgets for one system on one fixed GPU before switching models."""
    for task in tasks:
        try:
            result = run_task(task, gpu, tokens, token_sha, args)
            with lock:
                summaries.append(result)
        except Exception as exc:
            errors.put(
                f"system={task.name} B={task.budget} gpu={gpu}: {type(exc).__name__}: {exc}"
            )
            return


def _metric(row: dict[str, Any], *names: str) -> float:
    for name in names:
        if isinstance(row.get(name), (int, float)):
            return float(row[name])
    return float("nan")


def write_comparison(rows: list[dict[str, Any]], root: Path, challenge: str) -> None:
    rows = sorted(rows, key=lambda x: (int(x["budget"]), str(x["system"])))
    fields = sorted({k for row in rows for k in row}, key=lambda x: (x not in {"budget", "system", "implementation_label", "scenario_count", "successful", "failed"}, x))
    with (root / "closed_loop_fixed_budget_all_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    (root / "closed_loop_fixed_budget_all_metrics.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    preferred = {
        "closed_loop_nonreactive_agents": ["score", "final_score", "closed_loop_nonreactive_agents_weighted_average"],
        "closed_loop_reactive_agents": ["score", "final_score", "closed_loop_reactive_agents_weighted_average"],
    }
    lines = [
        f"# Paired fixed-budget closed-loop comparison: {challenge}", "",
        "Every row uses the identical ordered test scenario-token manifest and one official nuPlan aggregation run (no shard-level weighted averaging).", "",
        "| B | System | Aggregate | Collision | TTC | Drivable | Progress | Comfort | Wall h |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        score = _metric(row, *preferred.get(challenge, []), "score")
        lines.append(
            f"| {row['budget']} | {row['system']} | {score:.4f} | "
            f"{_metric(row, 'no_ego_at_fault_collisions', 'collision_avoidance'):.4f} | "
            f"{_metric(row, 'time_to_collision_within_bound', 'time_to_collision'):.4f} | "
            f"{_metric(row, 'drivable_area_compliance'):.4f} | "
            f"{_metric(row, 'ego_progress_along_expert_route', 'progress'):.4f} | "
            f"{_metric(row, 'ego_is_comfortable', 'comfort'):.4f} | {float(row['wall_time_s']) / 3600.0:.2f} |"
        )
    lines += [
        "",
        "B=8 and B=24 are cross-budget ablations unless every B-dependent calibration/training artifact was independently refit on train/val.",
        "PDM-Closed-style is retained only as a repository adapter; do not label it official PDM-Closed.",
    ]
    (root / "closed_loop_fixed_budget_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Paired test-set nuPlan closed-loop suite for B in {8,16,24} with fixed upstream M.")
    p.add_argument("--own-config", type=Path, default=None, help="Resolved deployed config for the user's BDSE model; required only when --systems includes bdse.")
    p.add_argument("--own-checkpoint", type=Path, default=None, help="Required only when --systems includes bdse.")
    p.add_argument("--own-label", type=str, default="BDSE frozen deployable model", help="Display label for --systems bdse; does not change planner behavior.")
    p.add_argument("--external-checkpoint-root", type=Path, default=Path("outputs/external_fixed_budget"), help="Root containing B{budget}/{name}_budgeted.best.pt checkpoints from budget-specific training")
    p.add_argument("--allow-shared-external-checkpoint", action="store_true", help="Allow legacy one-checkpoint-for-all-B evaluation; use only for cross-budget ablations, not the strict primary comparison")
    p.add_argument("--split-cache", type=Path, required=True, help="BDSE NPZ test cache root, e.g. .../bdse_test_2")
    p.add_argument("--token-split", default="public_set_test")
    p.add_argument("--limit", type=int, default=0, help="0 = all unique test scenario tokens")
    p.add_argument("--token-scan-max", type=int, default=100000)
    p.add_argument("--token-scan-workers", type=int, default=8, help="Parallel workers used only while reading scenario_token from the full NPZ test cache")
    p.add_argument("--token-progress-seconds", type=float, default=5.0, help="How often to print NPZ manifest-scan progress")
    p.add_argument("--nuplan-root", type=Path, required=True, help="nuPlan data root used by the simulator for non-DB assets/config defaults")
    p.add_argument("--nuplan-map-root", type=Path, default=None, help="Optional map root; defaults to <nuplan-root>/maps")
    p.add_argument("--nuplan-exp-root", type=Path, default=None, help="Optional experiment root; defaults to <nuplan-root>/exp")
    p.add_argument("--nuplan-db-root", type=Path, required=True, help="Raw nuPlan test .db file/root, not the BDSE NPZ cache")
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 24])
    p.add_argument("--proposal-top-m", type=int, default=24, help="Fixed upstream proposal pool M from the manuscript")
    p.add_argument("--challenge", choices=["closed_loop_nonreactive_agents", "closed_loop_reactive_agents"], default="closed_loop_nonreactive_agents")
    p.add_argument("--metric-aggregator", default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--workers-per-job", type=int, default=4, help="nuPlan thread-pool workers inside each concurrently running model job")
    p.add_argument("--heartbeat-seconds", type=float, default=15.0, help="Closed-loop task heartbeat interval; reports phase, PID, elapsed time, and GPU memory")
    p.add_argument("--schedule-mode", choices=["model_pairs", "queue"], default="model_pairs", help="model_pairs pins one model (all budgets) to one GPU, two models at a time")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--systems", nargs="+", default=["bdse", "gameformer", "dtpp", "plantf", "pluto", "pdm_closed_style"])
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not args.budgets or any(int(b) <= 0 for b in args.budgets):
        raise ValueError("--budgets must contain positive integers")
    if int(args.proposal_top_m) <= 0:
        raise ValueError("--proposal-top-m must be >0")
    if int(args.workers_per_job) <= 0:
        raise ValueError("--workers-per-job must be >0")
    if int(args.token_scan_workers) <= 0:
        raise ValueError("--token-scan-workers must be >0")
    if float(args.heartbeat_seconds) <= 0:
        raise ValueError("--heartbeat-seconds must be >0")
    _validate_db_root(args.nuplan_db_root)
    args.nuplan_map_root = args.nuplan_map_root or (args.nuplan_root / "maps")
    args.nuplan_exp_root = args.nuplan_exp_root or (args.nuplan_root / "exp")
    if args.metric_aggregator is None:
        args.metric_aggregator = (
            "closed_loop_nonreactive_agents_weighted_average"
            if args.challenge == "closed_loop_nonreactive_agents"
            else "closed_loop_reactive_agents_weighted_average"
        )
    if "bdse" in args.systems:
        if args.own_config is None or args.own_checkpoint is None:
            raise ValueError("--own-config and --own-checkpoint are required when --systems includes bdse")
        for path in (args.own_config, args.own_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)

    # Fail fast before touching thousands of NPZ files. The closed-loop suite is
    # evaluation-only: it never trains missing external checkpoints. Previously
    # a missing checkpoint was discovered only after the full scenario-token
    # scan, which could look like a hung/no-GPU training run for many minutes.
    trainable_external = [x for x in args.systems if x in {"gameformer", "dtpp", "plantf", "pluto"}]
    missing_checkpoints: list[Path] = []
    found_checkpoints = 0
    for budget in args.budgets:
        for name in trainable_external:
            budget_ckpt = args.external_checkpoint_root / f"B{budget}" / f"{name}_budgeted.best.pt"
            legacy_ckpt = args.external_checkpoint_root / f"{name}_budgeted.best.pt"
            if budget_ckpt.is_file() or (args.allow_shared_external_checkpoint and legacy_ckpt.is_file()):
                found_checkpoints += 1
            else:
                missing_checkpoints.append(budget_ckpt)
    if missing_checkpoints:
        sample = "\n".join(f"  - {p}" for p in missing_checkpoints[:12])
        raise FileNotFoundError(
            "Closed-loop evaluation does not train models, and required budget-specific checkpoints are missing.\n"
            f"Missing {len(missing_checkpoints)} checkpoint(s):\n{sample}\n"
            "Run: bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh\n"
            "Then rerun the closed-loop script."
        )
    print(
        f"[preflight] evaluation-only suite: checkpoints_ok={found_checkpoints} "
        f"systems={args.systems} budgets={args.budgets} device={args.device}",
        flush=True,
    )

    tokens, log_hints = load_tokens(
        args.split_cache,
        args.token_split,
        args.limit,
        args.token_scan_max,
        scan_workers=int(args.token_scan_workers),
        progress_interval_s=float(args.token_progress_seconds),
    )
    token_sha = _token_hash(tokens)
    args.resolved_db_files = _restrict_flat_db_files(args.nuplan_db_root, log_hints)
    if args.resolved_db_files:
        print(
            f"[DB optimization] matched all {len(tokens)} scenario tokens to {len(args.resolved_db_files)} direct raw DB files; "
            "passing only those files to nuPlan.", flush=True,
        )
    else:
        print(
            "[DB optimization] cache-parent/raw-DB stems were not a complete match; using the full --nuplan-db-root safely.",
            flush=True,
        )
    (args.output_root / "scenario_tokens_all.json").write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    (args.output_root / "scenario_manifest.json").write_text(json.dumps({
        "split_cache": str(args.split_cache.resolve()),
        "token_split": args.token_split,
        "scenario_count": len(tokens),
        "ordered_token_sha256": token_sha,
        "nuplan_db_root": str(args.nuplan_db_root.resolve()),
        "nuplan_db_mode": "restricted_files" if args.resolved_db_files else "root",
        "nuplan_db_files": [str(p.resolve()) for p in args.resolved_db_files],
        "nuplan_map_root": str(args.nuplan_map_root.resolve()),
        "nuplan_exp_root": str(args.nuplan_exp_root.resolve()),
        "budgets": [int(x) for x in args.budgets],
        "proposal_top_m": int(args.proposal_top_m),
        "nuplan_db_file_count": len(args.resolved_db_files) if args.resolved_db_files else -1,
        "nuplan_workers_per_job": int(args.workers_per_job),
        "token_scan_workers": int(args.token_scan_workers),
        "heartbeat_seconds": float(args.heartbeat_seconds),
        "schedule_mode": str(args.schedule_mode),
    }, indent=2, sort_keys=True), encoding="utf-8")

    source_specs: dict[str, tuple[str, Path, str | None]] = {
        "gameformer": ("GameFormer-inspired budget adapter", Path("bdse/configs/external_gameformer_budgeted_fast_cl.yaml"), "gameformer"),
        "dtpp": ("DTPP-inspired budget adapter", Path("bdse/configs/external_dtpp_budgeted_fast_cl.yaml"), "dtpp"),
        "plantf": ("PlanTF-inspired budget adapter", Path("bdse/configs/external_plantf_budgeted_fast_cl.yaml"), "plantf"),
        "pluto": ("PLUTO-inspired budget adapter", Path("bdse/configs/external_pluto_budgeted_fast_cl.yaml"), "pluto"),
        "pdm_closed_style": ("PDM-Closed-style budget scorer (NOT official PDM-Closed)", Path("bdse/configs/external_pdm_closed_budgeted_fast_cl.yaml"), None),
    }
    if "bdse" in args.systems:
        assert args.own_config is not None
        source_specs["bdse"] = (str(args.own_label), args.own_config, None)
    unknown = [x for x in args.systems if x not in source_specs]
    if unknown:
        raise ValueError(f"unknown --systems: {unknown}")
    config_root = args.output_root / "resolved_configs"
    tasks: list[SystemTask] = []
    for budget in args.budgets:
        for name in args.systems:
            label, src, checkpoint_stem = source_specs[name]
            if not src.is_file():
                raise FileNotFoundError(src)
            if name == "bdse":
                assert args.own_checkpoint is not None
                ckpt: Path | None = args.own_checkpoint
            elif checkpoint_stem is None:
                ckpt = None
            else:
                budget_ckpt = args.external_checkpoint_root / f"B{budget}" / f"{checkpoint_stem}_budgeted.best.pt"
                legacy_ckpt = args.external_checkpoint_root / f"{checkpoint_stem}_budgeted.best.pt"
                if budget_ckpt.is_file():
                    ckpt = budget_ckpt
                elif args.allow_shared_external_checkpoint and legacy_ckpt.is_file():
                    ckpt = legacy_ckpt
                    label += " (shared-checkpoint cross-budget ablation)"
                else:
                    raise FileNotFoundError(
                        f"missing budget-specific checkpoint {budget_ckpt}; train each external adapter separately for B={budget}, "
                        "or pass --allow-shared-external-checkpoint only for a non-primary cross-budget ablation"
                    )
            cfg = config_root / f"B{budget}" / f"{name}.yaml"
            strict_budget_config(src, cfg, budget=int(budget), proposal_top_m=int(args.proposal_top_m), own_model=name == "bdse")
            tasks.append(SystemTask(name=name, label=label, budget=int(budget), config=cfg, checkpoint=ckpt))

    gpu_slots = [x.strip() for x in args.gpus.split(",") if x.strip()] if args.device == "cuda" else ["cpu"]
    if not gpu_slots:
        raise ValueError("no GPU slots specified")
    errors: "queue.Queue[str]" = queue.Queue()
    summaries: list[dict[str, Any]] = []
    lock = threading.Lock()
    if args.schedule_mode == "model_pairs":
        # Pin each model to one GPU for all requested budgets.  With two A30s this
        # runs (system0, system1), waits, then (system2, system3), etc.  Compared
        # with the old global task queue this improves locality and makes resource
        # usage/reproducibility much easier to audit.
        by_system = {
            name: [task for task in tasks if task.name == name]
            for name in args.systems
        }
        width = max(1, len(gpu_slots))
        for start_idx in range(0, len(args.systems), width):
            names = args.systems[start_idx : start_idx + width]
            threads: list[threading.Thread] = []
            for slot, name in enumerate(names):
                gpu = gpu_slots[slot]
                seq = by_system[name]
                print(
                    f"[closed-loop pair] gpu={gpu} system={name} budgets={[t.budget for t in seq]} workers={args.workers_per_job}",
                    flush=True,
                )
                t = threading.Thread(
                    target=_sequence_worker,
                    args=(gpu, seq, errors, summaries, lock, tokens, token_sha, args),
                    daemon=True,
                )
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            if not errors.empty():
                raise RuntimeError("\n".join(list(errors.queue)))
    else:
        q: "queue.Queue[SystemTask | None]" = queue.Queue()
        for task in tasks:
            q.put(task)
        threads = [threading.Thread(target=_worker, args=(gpu, q, errors, summaries, lock, tokens, token_sha, args), daemon=True) for gpu in gpu_slots]
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
        raise RuntimeError(f"collected {len(summaries)}/{len(tasks)} summaries")
    if {str(row.get("scenario_token_sha256")) for row in summaries} != {token_sha}:
        raise RuntimeError("scenario-token pairing invariant violated")
    write_comparison(summaries, args.output_root, args.challenge)
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "tasks": len(tasks),
        "scenario_count": len(tokens),
        "ordered_token_sha256": token_sha,
        "budgets": [int(x) for x in args.budgets],
        "proposal_top_m": int(args.proposal_top_m),
        "nuplan_db_mode": "restricted_files" if args.resolved_db_files else "root",
        "nuplan_db_file_count": len(args.resolved_db_files) if args.resolved_db_files else -1,
        "nuplan_workers_per_job": int(args.workers_per_job),
        "schedule_mode": str(args.schedule_mode),
    }, indent=2))


if __name__ == "__main__":
    main()
