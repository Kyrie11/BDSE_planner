from __future__ import annotations

import importlib
import json
import math
import os
import re
import time
import traceback
import sqlite3
import sys
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from bdse.config import load_config
from bdse.data.cache_schema import Sample, load_sample_npz, save_sample_npz
from bdse.data.label_builder import build_training_sample_from_scenario
from bdse.data.quality import quality_decision
from bdse.data.scenario_sampler import DBFileRecord, db_files_for_nuplan_builder, discover_db_files, normalize_split_name, select_records

@dataclass(frozen=True)
class ScenarioIndexRecord:
    db_path: Path
    split: str
    folder: str
    token: str
    timestamp_us: int
    iteration: int


@dataclass(frozen=True)
class DevkitScenarioIndexRecord:
    scenario: Any
    split: str
    folder: str
    log_name: str
    token: str
    iteration: int
    timestamp_us: int


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return set()
    return {str(r[1]) for r in rows}


def _cfg_digest_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    """Small JSON-serializable snapshot of knobs that affect cache labels/features."""
    out: dict[str, Any] = {}
    for section, keys in {
        "preprocess": [
            "scenario_stride", "scenario_iteration_policy", "max_samples_per_log",
            "max_samples_per_log_strategy", "max_scenarios_strategy",
            "label_agent_future_mode", "runtime_agent_history_mode",
            "candidate_aware_agent_selection", "materialize_quality_filter",
        ],
        "runtime": [
            "map_radius_m", "agent_radius_m", "max_agents", "include_drivable_polygons",
            "max_drivable_polygons", "max_polygon_points", "include_crosswalks",
        ],
        "candidate": ["K", "pool_K", "prune_pool_to_K", "min_valid_candidates"],
        "evidence": ["budget", "max_atoms", "max_interaction_atoms", "max_interaction_agents"],
        "teacher": ["cost_eval_stride", "demo_weight", "demo_scale", "route_weight", "progress_weight"],
        "selector": ["bidirectional_pairs", "reverse_pair_weight", "runtime_pair_cap_multiplier", "lambda_safety", "lambda_near"],
        "tournament": ["L_infer"],
    }.items():
        sec = cfg.get(section, {}) or {}
        out[section] = {k: sec.get(k) for k in keys if k in sec}
    return out


def _quality_metrics_from_teacher_diag(sample: Sample) -> dict[str, Any]:
    diag = {} if sample.teacher is None else dict(sample.teacher.diagnostics or {})
    out: dict[str, Any] = {}
    for k, v in diag.items():
        if str(k).startswith("quality_"):
            out[str(k)[len("quality_"):]] = v
    return out


def _preprocess_quality_decision_for_sample(sample: Sample, cfg: dict[str, Any]):
    metrics = _quality_metrics_from_teacher_diag(sample)
    # Force preprocess.quality_filter semantics even if training.quality_filter is
    # enabled in the same config file.  Training may be stricter; materialization
    # should be explicitly controlled by preprocess.materialize_quality_filter.
    qcfg = dict(cfg)
    qcfg["training"] = {"quality_filter": {"enabled": False}}
    return quality_decision(metrics, qcfg)


def scan_db_for_lidarpc_tokens(db_path: str | Path, split: str, folder: str, stride: int = 10, max_frames: int | None = None) -> list[ScenarioIndexRecord]:
    path = Path(db_path)
    records: list[ScenarioIndexRecord] = []
    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.Error:
        return records
    with conn:
        cols = _table_columns(conn, "lidar_pc")
        if not cols:
            return records
        token_col = "token" if "token" in cols else next(iter(cols))
        ts_col = "timestamp" if "timestamp" in cols else "time_stamp" if "time_stamp" in cols else None
        if ts_col is None:
            rows = conn.execute(f"SELECT {token_col} FROM lidar_pc ORDER BY rowid").fetchall()
            for i, row in enumerate(rows[::stride]):
                if max_frames is not None and len(records) >= max_frames:
                    break
                records.append(ScenarioIndexRecord(path, split, folder, str(row[0]), int(i * stride), i * stride))
        else:
            rows = conn.execute(f"SELECT {token_col}, {ts_col} FROM lidar_pc ORDER BY {ts_col}").fetchall()
            for i, row in enumerate(rows[::stride]):
                if max_frames is not None and len(records) >= max_frames:
                    break
                records.append(ScenarioIndexRecord(path, split, folder, str(row[0]), int(row[1]), i * stride))
    return records


class NuPlanScenarioSource:
    def __init__(self, cfg: dict[str, Any], records: list[DBFileRecord], split: str, num_workers=None, use_process_pool=None, max_scenarios: int | None = None):
        self.cfg = cfg
        self.records = records
        self.split = split
        self._scenarios: list[Any] | None = None
        self.num_workers = num_workers
        self.use_process_pool = use_process_pool
        self.max_scenarios = max_scenarios

    def _build_with_devkit(self) -> list[Any]:
        try:
            builder_mod = importlib.import_module("nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder")
            filter_mod = importlib.import_module("nuplan.planning.scenario_builder.scenario_filter")
            worker_mod = importlib.import_module("nuplan.planning.utils.multithreading.worker_parallel")
        except ImportError as exc:
            raise RuntimeError("nuPlan devkit is not installed; install nuplan-devkit or use preprocessed cache.") from exc
        NuPlanScenarioBuilder = getattr(builder_mod, "NuPlanScenarioBuilder")
        ScenarioFilter = getattr(filter_mod, "ScenarioFilter")
        SingleMachineParallelExecutor = getattr(worker_mod, "SingleMachineParallelExecutor")

        paths = self.cfg.get("paths", {})
        preprocess_cfg = self.cfg.get("preprocess", {})
        num_workers = self.num_workers
        if num_workers is None:
            num_workers = preprocess_cfg.get("scenario_builder_workers", 1)
        num_workers = max(1, int(num_workers or 1))

        # Important: do not reuse the sample-generation process-pool switch here.
        # nuPlan scenario discovery constructs DB/map-backed scenario objects; using a
        # process pool here can silently block before tqdm is created.  Keep discovery
        # threaded/single-process and parallelize sample materialization instead.
        builder_use_process_pool = bool(preprocess_cfg.get("scenario_builder_use_process_pool", False))
        # Do not implicitly pass --max-scenarios-per-split to nuPlan's ScenarioBuilder.
        # ScenarioBuilder's limit_total_scenarios is an order-dependent discovery cap,
        # while BDSE's max_scenarios should describe the final materialized subset.
        # Use preprocess.scenario_filter_limit_total_scenarios / --scenario-builder-limit
        # only for deliberate fast, potentially biased discovery probes.
        scenario_filter_limit = preprocess_cfg.get("scenario_filter_limit_total_scenarios", None)
        if scenario_filter_limit is None and bool(preprocess_cfg.get("use_max_scenarios_as_builder_limit", False)):
            scenario_filter_limit = self.max_scenarios

        db_files = db_files_for_nuplan_builder(self.records)
        print(
            f"[bdse] building nuPlan scenarios: split={self.split} db_files={len(db_files)} "
            f"builder_workers={num_workers} builder_process_pool={builder_use_process_pool} "
            f"scenario_filter_limit={scenario_filter_limit}",
            flush=True,
        )
        builder = NuPlanScenarioBuilder(
            data_root=str(paths.get("data_cache_root", "/data0/nuplan/data/cache")),
            map_root=str(paths.get("maps_root", "/data0/nuplan/dataset/maps")),
            sensor_root=str(paths.get("sensor_root", paths.get("data_cache_root", "/data0/nuplan/data/cache"))),
            db_files=db_files,
            map_version=str(paths.get("map_version", "nuplan-maps-v1.0")),
            include_cameras=False,
            max_workers=num_workers,
            verbose=False,
        )
        # Apply temporal thinning inside nuPlan ScenarioBuilder. Without this,
        # scenario_types=None can materialize one scenario per lidar frame; on full val
        # this can produce millions of Scenario objects before BDSE's own stride is used
        timestamp_threshold_s = preprocess_cfg.get("scenario_builder_timestamp_threshold_s", None)
        if timestamp_threshold_s is None:
            timestamp_threshold_s = max(0.1, float(preprocess_cfg.get("scenario_stride", 10)) * float(
                self.cfg.get("candidate", {}).get("step_s", 0.1)))

        scenario_filter = ScenarioFilter(
            scenario_types=None,
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=None,
            limit_total_scenarios=scenario_filter_limit,
            timestamp_threshold_s=float(timestamp_threshold_s),
            ego_displacement_minimum_m=None,
            expand_scenarios=False,
            remove_invalid_goals=True,
            shuffle=False,
        )

        worker = SingleMachineParallelExecutor(
            use_process_pool=builder_use_process_pool,
            max_workers=num_workers,
        )
        scenarios = list(builder.get_scenarios(scenario_filter, worker=worker))
        print(f"[bdse] built {len(scenarios)} nuPlan scenarios for split={self.split}", flush=True)
        return scenarios

    def scenarios(self) -> list[Any]:
        if self._scenarios is None:
            self._scenarios = self._build_with_devkit()
        return self._scenarios


def _safe_name(text: Any) -> str:
    s = str(text) if text is not None else "unknown"
    s = re.sub(r"[^A-Za-z0-9_.=-]+", "_", s)
    return s[:180] if len(s) > 180 else s


def _stable_log_name(text: Any) -> str:
    """Recover the DB/log identity from nuPlan scenario names when needed."""
    s = str(text) if text is not None else ""
    if s.endswith(".db"):
        s = Path(s).stem
    else:
        s = Path(s).name
    return _safe_name(re.sub(r"_\d{5,6}_\d{5,6}$", "", s))


def _scenario_token(scenario: Any) -> str:
    return _safe_name(getattr(scenario, "token", getattr(scenario, "scenario_name", getattr(scenario, "_scenario_name", "scenario"))))


def _time_us_value(value: Any) -> int | None:
    if value is None:
        return None
    val = getattr(value, "time_us", getattr(value, "timestamp_us", getattr(value, "timestamp", value)))
    if isinstance(val, (int, float, np.integer, np.floating)):
        return int(val)
    return None


def _scenario_iteration_timestamp_us(scenario: Any, iteration: int) -> int:
    """Best-effort absolute log timestamp for cache-local materialization order.

    The value is used only to order preprocessing jobs.  Sample contents and cache
    paths remain keyed by the original scenario/token/iteration, so failure to read
    a timestamp falls back to a stable per-scenario value instead of changing labels.
    """
    for name in ["get_time_point", "get_timestamp_at_iteration"]:
        fn = getattr(scenario, name, None)
        if callable(fn):
            try:
                ts = _time_us_value(fn(int(iteration)))
                if ts is not None:
                    return ts
            except Exception:
                pass
    for attr in ["start_time", "start_timestamp", "timestamp_us", "initial_lidar_timestamp"]:
        ts = _time_us_value(getattr(scenario, attr, None))
        if ts is not None:
            interval = getattr(scenario, "database_interval", getattr(scenario, "_database_interval", None))
            try:
                step_us = int(round(float(interval) * 1_000_000.0)) if interval is not None else 1
            except Exception:
                step_us = 1
            return int(ts) + int(iteration) * max(step_us, 1)
    return int(iteration)


def _scenario_log_name(scenario: Any) -> str:
    for name in ("log_name", "_log_name", "database_log_name", "_database_log_name", "db_name", "_db_name"):
        val = getattr(scenario, name, None)
        if callable(val):
            try:
                val = val()
            except Exception:
                val = None
        if val is not None and str(val) != "":
            return _stable_log_name(val)
    for name in ("database_path", "_database_path", "db_file", "_db_file"):
        val = getattr(scenario, name, None)
        if val is not None and str(val) != "":
            return _stable_log_name(Path(str(val)).stem)
    return _stable_log_name(getattr(scenario, "scenario_name", getattr(scenario, "_scenario_name", "log")))


def _scenario_folder_for_log(records: list[DBFileRecord], log_name: str, default_split: str) -> str:
    # Kept for compatibility, but build_index() uses a precomputed lookup.
    # The old implementation was called once per scenario and scanned all DB
    # records, which is O(num_scenarios * num_db_files) and can stall for tens
    # of minutes on full nuPlan val/train splits.
    lookup = _folder_lookup(records)
    return lookup.get(log_name, lookup.get(_safe_name(log_name), _default_folder(records, default_split)))


def _folder_lookup(records: Iterable[DBFileRecord]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in records:
        out[rec.log_name] = rec.folder
        out[_safe_name(rec.log_name)] = rec.folder
        out[rec.path.stem] = rec.folder
        out[_safe_name(rec.path.stem)] = rec.folder
    return out


def _default_folder(records: Iterable[DBFileRecord], default_split: str) -> str:
    for rec in records:
        if rec.split == default_split:
            return rec.folder
    return default_split


def _maybe_tqdm(iterable, total: int | None, desc: str, enable: bool):
    if not enable:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, file=sys.stdout, mininterval=2.0)
    except Exception:
        return iterable


def _num_iterations(scenario: Any) -> int:
    for name in ["get_number_of_iterations", "get_num_iterations", "num_iterations", "database_interval"]:
        obj = getattr(scenario, name, None)
        try:
            val = obj() if callable(obj) else obj
            if isinstance(val, (int, np.integer)) and int(val) > 0:
                return int(val)
        except Exception:
            pass
    return 1


def _sort_devkit_records_for_temporal_cache(records: list[DevkitScenarioIndexRecord]) -> list[DevkitScenarioIndexRecord]:
    """Order sample materialization by absolute log time within each log.

    BDSE label construction caches exact nuPlan frames by ``(log_name,
    timestamp_us)``.  Adjacent 1 Hz samples share most of their 8 s future
    windows, but that cache locality is lost if all scenario-local iteration-20
    records are processed before iteration-30 records from unrelated timestamps.
    Sorting only the work order preserves the selected sample set and teacher
    labels while turning repeated 80-frame future bulk reads into mostly cache
    hits plus a small tail fill.
    """
    return sorted(records, key=lambda r: (r.log_name, int(r.timestamp_us), r.token, int(r.iteration)))

def _uniform_indices(n: int, cap: int) -> np.ndarray:
    if n <= 0 or cap <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    idx = np.linspace(0, n - 1, cap).round().astype(np.int64)
    idx = np.unique(idx)
    if idx.size < cap:
        used = set(idx.tolist())
        missing = [i for i in range(n) if i not in used]
        idx = np.asarray(sorted(list(idx) + missing[: cap - idx.size]), dtype=np.int64)
    return idx[:cap]


def _uniform_block_indices(n: int, cap: int, block_size: int) -> np.ndarray:
    """Select a capped set as uniformly placed short temporal blocks.
    This keeps broad log coverage while making adjacent selected samples share
    most of their exact 8 s future/history windows. It changes only which frames
    are selected under a cap; teacher construction for each selected frame is
    unchanged.
    """
    if n <= 0 or cap <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    block = min(max(1, int(block_size)), cap)
    num_blocks = int(math.ceil(float(cap) / float(block)))
    max_anchor = max(0, n - block)
    anchors = np.linspace(0, max_anchor, num_blocks).round().astype(np.int64)
    picked: list[int] = []
    seen: set[int] = set()
    for anchor in anchors.tolist():
        a = int(max(0, min(max_anchor, anchor)))
        for j in range(block):
            idx = a + j
            if idx >= n:
                break
            if idx not in seen:
                picked.append(idx)
                seen.add(idx)
            if len(picked) >= cap:
                return np.asarray(picked, dtype=np.int64)
    for idx in _uniform_indices(n, cap).tolist():
        if idx not in seen:
            picked.append(int(idx))
            seen.add(int(idx))
        if len(picked) >= cap:
            break
    return np.asarray(picked[:cap], dtype=np.int64)

class NuPlanBDSEDataset:
    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        split: str = "train",
        folders: list[str] | None = None,
        max_files: int | None = None,
        max_scenarios: int | None = None,
        stride: int | None = None,
        use_devkit: bool = True,
        preprocessed_dir: str | Path | None = None,
        num_workers: int | None = None,
        use_process_pool: bool | None = None,
    ):
        self.cfg = cfg or load_config()
        records = discover_db_files(self.cfg.get("paths", {}).get("data_cache_root", "/data0/nuplan/data/cache"))
        self.records = select_records(records, split=split, folders=folders, max_files=max_files, seed=int(self.cfg.get("seed", 17)))
        self.split = split
        self.use_devkit = use_devkit
        self.preprocessed_dir = Path(preprocessed_dir or self.cfg.get("paths", {}).get("preprocessed_cache", "cache"))
        self.max_scenarios = max_scenarios
        self.stride = int(stride if stride is not None else self.cfg.get("preprocess", {}).get("scenario_stride", 10))
        max_per_log = self.cfg.get("preprocess", {}).get("max_samples_per_log", None)
        self.max_samples_per_log = None if max_per_log is None else max(1, int(max_per_log))
        iteration_policy = str(self.cfg.get("preprocess", {}).get("scenario_iteration_policy", "initial")).lower()
        if iteration_policy in {"start", "starts", "one", "once", "scenario"}:
            iteration_policy = "initial"
        if iteration_policy in {"all", "all_iterations", "expand", "expanded"}:
            iteration_policy = "expanded"
        if iteration_policy not in {"initial", "expanded"}:
            raise ValueError(
                f"Unsupported preprocess.scenario_iteration_policy={iteration_policy!r}; "
                "expected 'initial' or 'expanded'."
            )
        self.scenario_iteration_policy = iteration_policy
        self._index: list[Any] | None = None
        self.num_workers = int(num_workers if num_workers is not None else self.cfg.get("preprocess", {}).get("num_workers", 1) or 1)
        self.use_process_pool = bool(use_process_pool if use_process_pool is not None else self.cfg.get("preprocess", {}).get("use_process_pool", False))
        self._scenario_source = NuPlanScenarioSource(
            self.cfg,
            self.records,
            split,
            num_workers=self.cfg.get("preprocess", {}).get("scenario_builder_workers", min(self.num_workers, 4)),
            use_process_pool=self.cfg.get("preprocess", {}).get("scenario_builder_use_process_pool", False),
            max_scenarios=self.max_scenarios,
        ) if use_devkit else None

    def build_index(self) -> list[Any]:
        if self._index is not None:
            return self._index
        if self.use_devkit:
            scenarios = self._scenario_source.scenarios() if self._scenario_source is not None else []
            out: list[DevkitScenarioIndexRecord] = []
            folder_lookup = _folder_lookup(self.records)
            default_folder = _default_folder(self.records, self.split)
            total_scenarios = len(scenarios)
            print(
                f"[bdse] expanding scenario index: split={self.split} "
                f"scenario_objects={total_scenarios} stride={self.stride} "
                f"iteration_policy={self.scenario_iteration_policy} "
                f"max_scenarios={self.max_scenarios} max_samples_per_log={self.max_samples_per_log}",
                flush=True,
            )
            show_index_progress = total_scenarios >= int(self.cfg.get("preprocess", {}).get("index_progress_threshold", 10000))
            iterator = _maybe_tqdm(scenarios, total_scenarios, f"index:{self.split}", show_index_progress)
            cap_strategy = str(self.cfg.get("preprocess", {}).get("max_samples_per_log_strategy", "first")).lower()
            if cap_strategy in {"uniform-blocks", "blocked_uniform", "blocked-uniform"}:
                cap_strategy = "uniform_blocks"
            if cap_strategy not in {"first", "uniform", "uniform_blocks"}:
                raise ValueError(
                    f"Unsupported max_samples_per_log_strategy={cap_strategy!r}; "
                    "expected 'first', 'uniform', or 'uniform_blocks'."
                )
            per_log_counts: dict[str, int] = {}
            for scenario in iterator:
                token = _scenario_token(scenario)
                log_name = _scenario_log_name(scenario)
                folder = folder_lookup.get(log_name, folder_lookup.get(_safe_name(log_name), default_folder))
                if self.scenario_iteration_policy == "initial":
                    # nuPlan ScenarioBuilder already emits timestamp-filtered planning
                    # scenarios. Expanding every scenario-local iteration here creates
                    # many highly overlapping labels, duplicates actual log timesteps
                    # under different scenario tokens, and defeats the requested
                    # --scenario-stride. Keep one planning sample per builder scenario;
                    # use 'expanded' only for an explicit dense-window ablation.
                    iterations = (0,)
                else:
                    n_iter = _num_iterations(scenario)
                    iterations = range(0, n_iter, max(self.stride, 1))
                for iteration in iterations:
                    if self.max_samples_per_log is not None and cap_strategy == "first" and per_log_counts.get(log_name, 0) >= self.max_samples_per_log:
                        break
                    timestamp_us = _scenario_iteration_timestamp_us(scenario, iteration)
                    out.append(DevkitScenarioIndexRecord(scenario, self.split, folder, log_name, token, iteration,
                                                         timestamp_us))
                    per_log_counts[log_name] = per_log_counts.get(log_name, 0) + 1

            if self.max_samples_per_log is not None and cap_strategy in {"uniform", "uniform_blocks"}:
                grouped: dict[str, list[DevkitScenarioIndexRecord]] = {}
                for rec in out:
                    grouped.setdefault(rec.log_name, []).append(rec)
                capped: list[DevkitScenarioIndexRecord] = []
                cap = int(self.max_samples_per_log)
                block_size = max(1, int(self.cfg.get("preprocess", {}).get("max_samples_per_log_block_size", 8)))
                for log_name in sorted(grouped):
                    rows = _sort_devkit_records_for_temporal_cache(grouped[log_name])
                    if len(rows) <= cap:
                        capped.extend(rows)
                        continue
                    if cap_strategy == "uniform_blocks":
                        idx = _uniform_block_indices(len(rows), cap, block_size)
                    else:
                        idx = _uniform_indices(len(rows), cap)
                    capped.extend(rows[int(i)] for i in idx[:cap])
                out = _sort_devkit_records_for_temporal_cache(capped)
                print(
                    f"[bdse] applied {cap_strategy} per-log cap: split={self.split} "
                    f"logs={len(grouped)} cap={cap} "
                    f"block_size={block_size if cap_strategy == 'uniform_blocks' else '-'} records={len(out)}",
                    flush=True,
                )

            out = _sort_devkit_records_for_temporal_cache(out)

            if self.max_scenarios is not None and len(out) > int(self.max_scenarios):
                cap = max(1, int(self.max_scenarios))
                split_cap_strategy = str(self.cfg.get("preprocess", {}).get("max_scenarios_strategy", "uniform_blocks")).lower()
                if split_cap_strategy in {"uniform-blocks", "blocked_uniform", "blocked-uniform"}:
                    split_cap_strategy = "uniform_blocks"
                if split_cap_strategy not in {"first", "uniform", "uniform_blocks"}:
                    raise ValueError(
                        f"Unsupported max_scenarios_strategy={split_cap_strategy!r}; "
                        "expected 'first', 'uniform', or 'uniform_blocks'."
                    )
                if split_cap_strategy == "first":
                    out = out[:cap]
                elif split_cap_strategy == "uniform":
                    idx = _uniform_indices(len(out), cap)
                    out = [out[int(i)] for i in idx[:cap]]
                else:
                    block_size = max(1, int(self.cfg.get("preprocess", {}).get("max_samples_per_log_block_size", 8)))
                    idx = _uniform_block_indices(len(out), cap, block_size)
                    out = [out[int(i)] for i in idx[:cap]]
                out = _sort_devkit_records_for_temporal_cache(out)
                print(
                    f"[bdse] applied split cap: split={self.split} strategy={split_cap_strategy} "
                    f"cap={cap} records={len(out)}",
                    flush=True,
                )

            self._index = out
            print(f"[bdse] built scenario index: split={self.split} records={len(out)}", flush=True)
            return self._index
        idx: list[ScenarioIndexRecord] = []
        per_file = None if self.max_scenarios is None else max(1, self.max_scenarios // max(len(self.records), 1))
        for rec in self.records:
            idx.extend(scan_db_for_lidarpc_tokens(rec.path, rec.split, rec.folder, self.stride, per_file))
        if self.max_scenarios is not None:
            idx = idx[: self.max_scenarios]
        self._index = idx
        return self._index

    def __len__(self) -> int:
        return len(self.build_index())

    def __getitem__(self, idx: int) -> Sample:
        item = self.build_index()[idx]
        if self.use_devkit:
            assert isinstance(item, DevkitScenarioIndexRecord)
            sample = build_training_sample_from_scenario(item.scenario, item.iteration, self.cfg)
            if not sample.scenario_token:
                sample.scenario_token = item.token
            return sample
        raise RuntimeError("Raw SQLite indexing is available for discovery only; use nuPlan devkit for sample construction.")

    def iter_samples(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]

    def cache_path_for_index(self, idx: int, out_dir: str | Path | None = None) -> Path:
        out = Path(out_dir or self.preprocessed_dir)
        item = self.build_index()[idx]
        if isinstance(item, DevkitScenarioIndexRecord):
            parts = [out, Path(item.split)]
            # Avoid paths such as <out>/val/val/... when the folder is only the
            # split bucket.  City-specific train folders are preserved as
            # <out>/train/train_boston/... for traceability.
            if item.folder and item.folder != item.split:
                parts.append(Path(item.folder))
            return Path(*parts) / item.log_name / f"{item.token}_it{item.iteration:06d}.npz"
        if isinstance(item, ScenarioIndexRecord):
            parts = [out, Path(item.split)]
            if item.folder and item.folder != item.split:
                parts.append(Path(item.folder))
            return Path(*parts) / item.db_path.stem / f"{_safe_name(item.token)}_it{item.iteration:06d}.npz"
        return out / self.split / f"{idx:08d}.npz"

    def cache_path_aliases_for_index(self, idx: int, out_dir: str | Path | None = None) -> list[Path]:
        """Return canonical plus backward-compatible cache paths for resume.

        Older preprocessing runs may have written concrete train folders either as
        ``ROOT/train_boston/...`` or as ``ROOT/train/train_boston/...`` depending
        on whether the CLI used a canonical split plus folders or a concrete
        sub-split.  Resume should not rebuild a sample whose exact token/iteration
        file already exists in either layout.
        """
        out = Path(out_dir or self.preprocessed_dir)
        canonical = self.cache_path_for_index(idx, out)
        aliases: list[Path] = [canonical]
        item = self.build_index()[idx]
        if isinstance(item, DevkitScenarioIndexRecord):
            filename = f"{item.token}_it{item.iteration:06d}.npz"
            log_part = Path(item.log_name)
            norm_split = normalize_split_name(str(item.split))
            for base in [
                out / norm_split / item.folder,
                out / item.folder,
                out / norm_split,
            ]:
                aliases.append(Path(base) / log_part / filename)
        elif isinstance(item, ScenarioIndexRecord):
            filename = f"{_safe_name(item.token)}_it{item.iteration:06d}.npz"
            log_part = Path(item.db_path.stem)
            norm_split = normalize_split_name(str(item.split))
            for base in [
                out / norm_split / item.folder,
                out / item.folder,
                out / norm_split,
            ]:
                aliases.append(Path(base) / log_part / filename)
        return sorted(dict.fromkeys(aliases), key=lambda p: (p != canonical, str(p)))

    @staticmethod
    def _link_existing_cache_alias(existing: Path, canonical: Path) -> bool:
        if canonical.exists() or existing == canonical:
            return False
        canonical.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(existing, canonical)
            return True
        except OSError:
            pass
        try:
            os.symlink(existing, canonical)
            return True
        except OSError:
            return False

    def _cache_filename_for_index(self, idx: int) -> str:
        item = self.build_index()[idx]
        if isinstance(item, DevkitScenarioIndexRecord):
            return f"{item.token}_it{item.iteration:06d}.npz"
        if isinstance(item, ScenarioIndexRecord):
            return f"{_safe_name(item.token)}_it{item.iteration:06d}.npz"
        return f"{idx:08d}.npz"

    def _resume_scan_roots_for_index(self, idx: int, out_dir: str | Path | None = None) -> list[Path]:
        """Split-scoped roots used for robust filename-based resume lookup.

        Exact alias checks cover the normal layouts, but older runs sometimes used
        a slightly different nuPlan log-name/folder prefix while keeping the same
        globally unique lidar token and iteration filename.  Scanning only these
        split/folder roots lets resume find those files without accidentally
        treating another split as complete.
        """
        out = Path(out_dir or self.preprocessed_dir)
        item = self.build_index()[idx]
        roots: list[Path] = []
        if isinstance(item, (DevkitScenarioIndexRecord, ScenarioIndexRecord)):
            split = str(item.split)
            folder = str(item.folder) if getattr(item, "folder", None) else split
            norm = normalize_split_name(split)
            for root in (
                out / split,
                out / norm / folder,
                out / folder,
                out / norm,
            ):
                if root.exists():
                    roots.append(root)
        else:
            root = out / self.split
            if root.exists():
                roots.append(root)
        return sorted(dict.fromkeys(roots), key=lambda p: str(p))

    @staticmethod
    def _cache_file_looks_complete(path: str | Path, cfg: dict[str, Any] | None = None) -> bool:
        """Cheap resume guard for already materialized ``.npz`` samples.

        ``Path.exists()`` is too optimistic for long preprocessing jobs: an
        interrupted atomic write can leave dot/tmp files, a failed manual copy can
        leave a tiny placeholder, and a second concurrent run can materialize the
        same sample after the pre-submit cache check.  The default check stays
        O(1) per file (stat only) so resume over tens of thousands of samples is
        fast.  Set ``preprocess.resume_validate_existing=true`` for a one-time
        audit that opens matching ``.npz`` files and verifies the minimal schema.
        """
        p = Path(path)
        if p.name.startswith(".") or ".tmp." in p.name or p.suffix != ".npz":
            return False
        try:
            st = p.stat()
        except OSError:
            return False
        if not p.is_file():
            return False
        pcfg = (cfg or {}).get("preprocess", {}) if isinstance(cfg, dict) else {}
        min_bytes = max(1, int(pcfg.get("resume_min_file_bytes", 512)))
        if int(st.st_size) < min_bytes:
            return False
        if not bool(pcfg.get("resume_validate_existing", False)):
            return True
        required = {
            "scenario_token",
            "timestamp_us",
            "runtime_ego_history",
            "candidate_trajectories",
            "candidate_valid",
            "teacher_J_T",
            "teacher_a_star",
        }
        try:
            with np.load(p, allow_pickle=False) as z:
                if not required.issubset(set(z.files)):
                    return False
                if np.asarray(z["candidate_valid"]).size == 0:
                    return False
                if np.asarray(z["teacher_J_T"]).size == 0:
                    return False
                if int(np.asarray(z["teacher_a_star"]).reshape(-1)[0]) < 0:
                    return False
        except Exception:
            return False
        return True

    def _build_resume_filename_index(self, out_dir: str | Path | None = None) -> dict[str, Path | None]:
        """Return filename -> existing path for resume, scoped to this split.

        A value of None means the filename is duplicated in the scoped roots.  The
        caller then falls back to exact alias checks rather than risking a wrong
        hard-link.  In nuPlan lidar tokens are expected to be unique, so duplicates
        should be very rare and are surfaced as non-resumable rather than hidden.
        """
        out = Path(out_dir or self.preprocessed_dir)
        roots: list[Path] = []
        index = self.build_index()
        # Gather roots from a bounded prefix and suffix so this remains cheap even
        # for million-sample indexes while still covering every split/folder layout
        # present in the current run.
        probe_indices = list(range(min(len(index), 256)))
        if len(index) > 256:
            probe_indices.extend(range(max(0, len(index) - 256), len(index)))
        for i in probe_indices:
            roots.extend(self._resume_scan_roots_for_index(i, out))
        roots = sorted(dict.fromkeys(roots), key=lambda p: str(p))
        by_name: dict[str, Path | None] = {}
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("*.npz"):
                if not self._cache_file_looks_complete(p, self.cfg):
                    continue
                prev = by_name.get(p.name)
                if prev is None and p.name in by_name:
                    continue
                if prev is not None and prev != p:
                    by_name[p.name] = None
                else:
                    by_name[p.name] = p
        return by_name

    def _write_one_preprocessed_index(self, i: int, path: Path, manifest_path: Path | None, *, skip_existing: bool = False) -> tuple[int, Path, dict[str, Any] | None]:
        pcfg = self.cfg.get("preprocess", {})
        if skip_existing and self._cache_file_looks_complete(path, self.cfg):
            return i, path, None
        profile = bool(pcfg.get("profile", False))
        threshold_s = float(pcfg.get("profile_threshold_s", 2.0))
        t0 = time.perf_counter()
        sample = self[i]
        t_sample = time.perf_counter()
        if sample.runtime.metadata is None:
            sample.runtime.metadata = {}
        sample.runtime.metadata["bdse_cache_config_summary"] = _cfg_digest_summary(self.cfg)
        sample.runtime.metadata["bdse_cache_split"] = str(self.split)
        if bool(pcfg.get("materialize_quality_filter", False)):
            q_dec = _preprocess_quality_decision_for_sample(sample, self.cfg)
            if not bool(q_dec.keep):
                t_done = time.perf_counter()
                if profile and (t_done - t0) >= threshold_s:
                    print(
                        f"[bdse][profile-filtered] idx={i} build={t_sample - t0:.3f}s "
                        f"total={t_done - t0:.3f}s reasons={','.join(q_dec.reasons)} path={path}",
                        flush=True,
                    )
                rec = {
                    "path": str(path),
                    "split": self.split,
                    "scenario_token": sample.scenario_token,
                    "timestamp_us": int(sample.timestamp_us),
                    "iteration": int(getattr(self.build_index()[i], "iteration", -1)),
                    "filtered": True,
                    "quality_keep": False,
                    "quality_reasons": list(q_dec.reasons),
                    "quality_metrics": q_dec.metrics,
                }
                return i, Path(), rec
        # np.savez appends ".npz" when the target name does not already end with it.
        # Keep the temporary filename ending in .npz, otherwise os.replace(tmp, path)
        # will look for a different file than the one numpy created.
        tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{i}.npz")
        save_sample_npz(sample, tmp, compressed=bool(pcfg.get("compress_npz", False)))
        os.replace(tmp, path)
        t_done = time.perf_counter()
        if profile and (t_done - t0) >= threshold_s:
            print(
                f"[bdse][profile-write] idx={i} build={t_sample - t0:.3f}s "
                f"save={t_done - t_sample:.3f}s total={t_done - t0:.3f}s path={path}",
                flush=True,
            )
        rec = {
            "path": str(path),
            "split": self.split,
            "scenario_token": sample.scenario_token,
            "timestamp_us": int(sample.timestamp_us),
            "a_star": int(sample.teacher.a_star if sample.teacher is not None else -1),
            "valid_candidates": int(sample.candidates.valid_mask.sum()),
            "iteration": int(getattr(self.build_index()[i], "iteration", -1)),
            "filtered": False,
            "quality_keep": bool((sample.teacher.diagnostics or {}).get("quality_keep", True)) if sample.teacher is not None else True,
            "quality_reasons": list((sample.teacher.diagnostics or {}).get("quality_reasons", [])) if sample.teacher is not None else [],
            "cache_config_summary": _cfg_digest_summary(self.cfg),
        }
        return i, path, rec

    def write_preprocessed_cache(self, out_dir: str | Path | None = None, resume: bool = True, overwrite: bool = False, show_progress: bool = True, manifest_name: str | None = None, num_workers: int | None = None, use_process_pool: bool | None = None) -> list[Path]:
        out = Path(out_dir or self.preprocessed_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Keep each requested split/folder self-contained.  For example,
        # preprocessing ``--split train_1 train_2 --output-dir ROOT`` writes
        # samples and manifests under ``ROOT/train_1`` and ``ROOT/train_2``.
        manifest_path = out / self.split / (manifest_name or self.cfg.get("preprocess", {}).get("manifest_name", "manifest.jsonl"))

        # Build the index once. cache_path_for_index() is pure after this and can be
        # used to decide skip/write before expensive sample construction.
        index = self.build_index()
        total = len(index)
        if total == 0:
            print(
                f"[bdse] no scenarios found for split={self.split}; records={len(self.records)}. "
                "Check --data-root, --folders, map_root/map_version, and nuPlan devkit installation.",
                flush=True,
            )
            return []

        print(f"[bdse] checking existing cache files: split={self.split} total={total} resume={resume} overwrite={overwrite}", flush=True)
        all_paths: list[Path] = [Path()] * total
        pending: list[tuple[int, Path]] = []
        skipped = 0
        skipped_alias = 0
        skipped_filename = 0
        linked_alias = 0
        resume_by_filename: dict[str, Path | None] = {}
        if resume and not overwrite:
            resume_by_filename = self._build_resume_filename_index(out)
            if resume_by_filename:
                print(
                    f"[bdse] resume filename index: split={self.split} existing_filenames={len(resume_by_filename)}",
                    flush=True,
                )
        check_iter = _maybe_tqdm(range(total), total, f"cache-check:{self.split}", show_progress and total >= 10000)
        for i in check_iter:
            p = self.cache_path_for_index(i, out)
            all_paths[i] = p
            existing: Path | None = None
            if resume and not overwrite:
                for candidate_path in self.cache_path_aliases_for_index(i, out):
                    if self._cache_file_looks_complete(candidate_path, self.cfg):
                        existing = candidate_path
                        break
                if existing is None and resume_by_filename:
                    by_name = resume_by_filename.get(self._cache_filename_for_index(i))
                    if by_name is not None and self._cache_file_looks_complete(by_name, self.cfg):
                        existing = by_name
                        skipped_filename += 1
            if existing is not None:
                skipped += 1
                if existing != p:
                    skipped_alias += 1
                    if self._link_existing_cache_alias(existing, p):
                        linked_alias += 1
                    all_paths[i] = p if self._cache_file_looks_complete(p, self.cfg) else existing
            else:
                pending.append((i, p))

        print(
            f"[bdse] split={self.split}: records={len(self.records)} scenarios/iterations={total} "
            f"pending={len(pending)} skipped={skipped} skipped_alias={skipped_alias} "
            f"skipped_filename={skipped_filename} linked_alias={linked_alias} out={out}",
            flush=True,
        )
        if not pending:
            return [p for p in all_paths if p and self._cache_file_looks_complete(p, self.cfg)]

        workers = int(num_workers if num_workers is not None else self.num_workers or 1)
        workers = max(1, workers)
        requested_process_pool = bool(use_process_pool if use_process_pool is not None else self.use_process_pool)
        if requested_process_pool and self.use_devkit:
            print(
                "[bdse] --use-process-pool requested, but nuPlan scenario objects carry DB/map handles "
                "and are not safely pickleable. Using a ThreadPoolExecutor for sample materialization; "
                "scenario discovery remains single-process/threaded.",
                flush=True,
            )

        paths: list[Path] = list(all_paths)
        manifest_records: list[dict[str, Any]] = []

        def _append_manifest(records: list[dict[str, Any]]) -> None:
            if not records:
                return
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
            records.clear()

        failed_records: list[dict[str, Any]] = []
        failed_path = out / self.split / self.cfg.get("preprocess", {}).get("failed_manifest_name", "failed_preprocess.jsonl")
        skip_failed = bool(self.cfg.get("preprocess", {}).get("skip_failed_samples", False))

        def _index_debug_record(i0: int, path0: Path, exc: BaseException) -> dict[str, Any]:
            item = index[i0] if 0 <= int(i0) < len(index) else None
            return {
                "index": int(i0),
                "path": str(path0),
                "split": str(getattr(item, "split", self.split)),
                "folder": str(getattr(item, "folder", "")),
                "log_name": str(getattr(item, "log_name", getattr(getattr(item, "db_path", None), "stem", ""))),
                "token": str(getattr(item, "token", "")),
                "iteration": int(getattr(item, "iteration", -1)),
                "timestamp_us": int(getattr(item, "timestamp_us", 0) or 0),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            }

        def _append_failed(records: list[dict[str, Any]]) -> None:
            if not records:
                return
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            with failed_path.open("a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
            records.clear()

        def _handle_result(fut: Future, future_map: dict[Future, tuple[int, Path]]):
            i0, path0 = future_map.pop(fut)
            try:
                i, written, rec = fut.result()
            except Exception as exc:
                fail_rec = _index_debug_record(i0, path0, exc)
                failed_records.append(fail_rec)
                _append_failed(failed_records)
                print(
                    f"[bdse][preprocess-error] index={i0} path={path0} "
                    f"type={type(exc).__name__} err={exc}. "
                    f"Full traceback written to {failed_path}",
                    flush=True,
                )
                if skip_failed:
                    return None
                raise RuntimeError(
                    f"Failed preprocessing index={i0} path={path0}. "
                    f"Inner {type(exc).__name__}: {exc}. "
                    f"Full traceback written to {failed_path}"
                ) from exc
            paths[i] = written
            if rec is not None:
                manifest_records.append(rec)
            return i

        if workers <= 1:
            iterator = _maybe_tqdm(pending, len(pending), f"preprocess:{self.split}", show_progress)
            for i, path in iterator:
                _, written, rec = self._write_one_preprocessed_index(i, path, manifest_path, skip_existing=resume and not overwrite)
                paths[i] = written
                if rec is not None:
                    manifest_records.append(rec)
                    if len(manifest_records) >= 256:
                        _append_manifest(manifest_records)
        else:
            # Threading overlaps DB/map I/O without attempting to pickle nuPlan scenarios.
            # Keep only a bounded number of futures in flight; submitting millions of
            # futures before reading results can consume huge memory and makes it look
            # like preprocessing has hung.
            max_in_flight = max(workers, int(self.cfg.get("preprocess", {}).get("max_in_flight", workers * 4)))
            cache_local_scheduler = bool(
                self.cfg.get("preprocess", {}).get("cache_local_scheduler", True)) and self.use_devkit
            cache_local_log_parallelism = max(1,
                                              int(self.cfg.get("preprocess", {}).get("cache_local_log_parallelism", 1)))
            print(f"[bdse] materialization scheduler: split={self.split} workers={workers} "
                f"max_in_flight={max_in_flight} cache_local={cache_local_scheduler} "
                f"per_log_parallelism={cache_local_log_parallelism if cache_local_scheduler else '-'}",
                flush = True,)
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures: dict[Future, tuple[int, Path]] = {}
                if cache_local_scheduler:
                    grouped: dict[str, deque[tuple[int, Path]]] = {}
                    log_order: list[str] = []
                    for i, path in pending:
                        item = index[i]
                        log_name = str(getattr(item, "log_name", "default"))
                        if log_name not in grouped:
                            grouped[log_name] = deque()
                            log_order.append(log_name)
                        grouped[log_name].append((i, path))
                    ready_logs: deque[str] = deque(log_order)
                    ready_log_set: set[str] = set(log_order)
                    active_log_counts: dict[str, int] = {}
                    future_logs: dict[Future, str] = {}

                    def enqueue_log(log_name: str, *, left: bool = False) -> None:
                        q = grouped.get(log_name)
                        if not q:
                           return
                        if active_log_counts.get(log_name, 0) >= cache_local_log_parallelism:
                            return
                        if log_name in ready_log_set:
                            return
                        if left:
                            ready_logs.appendleft(log_name)
                        else:
                            ready_logs.append(log_name)
                        ready_log_set.add(log_name)

                    def submit_from_log(log_name: str) -> bool:
                        q = grouped.get(log_name)
                        if not q:
                            return False
                        i, path = q.popleft()
                        active_log_counts[log_name] = active_log_counts.get(log_name, 0) + 1
                        fut = ex.submit(self._write_one_preprocessed_index, i, path, manifest_path, skip_existing=resume and not overwrite)
                        futures[fut] = (i, path)
                        future_logs[fut] = log_name
                        # If the machine still has idle workers and this log has more
                        # adjacent work, it is safe to submit another job from the same
                        # log up to cache_local_log_parallelism.  Overlapping exact
                        # nuPlan frame fills are still serialized inside the frame-cache
                        # layer; later jobs re-check the cache after the first fill and
                        # usually become cheap tail fills.  This preserves sample bytes
                        # and only changes scheduling.
                        enqueue_log(log_name)
                        return True

                    def submit_next() -> bool:
                        while ready_logs and len(futures) < max_in_flight:
                            log_name = ready_logs.popleft()
                            ready_log_set.discard(log_name)
                            if active_log_counts.get(log_name, 0) >= cache_local_log_parallelism:
                                continue
                            if submit_from_log(log_name):
                                return True
                        return False
                else:
                    pending_iter = iter(pending)
                    future_logs = {}

                    def submit_next() -> bool:
                        try:
                            i, path = next(pending_iter)
                        except StopIteration:
                            return False
                        futures[ex.submit(self._write_one_preprocessed_index, i, path, manifest_path, skip_existing=resume and not overwrite)] = (i, path)
                        return True

                for _ in range(min(max_in_flight, len(pending))):
                    if not submit_next():
                        break
                pbar = _maybe_tqdm(range(len(pending)), len(pending), f"preprocess:{self.split}", show_progress)
                try:
                    while futures:
                        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                        for fut in done:
                            finished_log = future_logs.pop(fut, None)
                            _handle_result(fut, futures)
                            if cache_local_scheduler and finished_log is not None:
                                current = active_log_counts.get(finished_log, 0) - 1
                                if current > 0:
                                    active_log_counts[finished_log] = current
                                else:
                                    active_log_counts.pop(finished_log, None)
                                if grouped.get(finished_log):
                                    # Continue the same log in timestamp order.  This is the
                                    # critical bit for exact future/history frame cache reuse:
                                    # only a bounded number of windows from a log are allowed
                                    # to wait on/cache-fill at a time.
                                    enqueue_log(finished_log, left=True)
                            completed += 1
                            if hasattr(pbar, "update"):
                                pbar.update(1)
                            while len(futures) < max_in_flight and submit_next():
                                pass
                            if len(manifest_records) >= 256:
                                _append_manifest(manifest_records)
                finally:
                    if hasattr(pbar, "close"):
                        pbar.close()
        _append_manifest(manifest_records)
        _append_failed(failed_records)
        return [p for p in paths if p and self._cache_file_looks_complete(p, self.cfg)]


class PreprocessedBDSEDataset:
    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str | list[str] | tuple[str, ...] | None = None,
        manifest_name: str = "manifest.jsonl",
        max_scenarios: int | None = None,
        max_scenarios_per_split: int | None = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        if split is None:
            self.splits: list[str] | None = None
        elif isinstance(split, str):
            self.splits = [split]
        else:
            self.splits = [str(s) for s in split]
        # Backward-compatible public attribute used by a few scripts.
        self.split = None if self.splits is None else (self.splits[0] if len(self.splits) == 1 else list(self.splits))
        self.manifest_name = manifest_name
        self.max_scenarios = max_scenarios
        self.max_scenarios_per_split = max_scenarios_per_split
        self._paths: list[Path] | None = None

    @staticmethod
    def _is_canonical_split_name(split: str) -> bool:
        return split == normalize_split_name(split)

    def _record_matches_split(self, rec_split: Any) -> bool:
        if self.splits is None:
            return True
        rec = str(rec_split)
        rec_norm = normalize_split_name(rec)
        for split in self.splits:
            if rec == split:
                return True
            # Canonical selectors such as ``train`` should also match concrete
            # preprocessed folders/manifests like ``train_1`` or ``train_boston``.
            # Concrete selectors such as ``train_1`` remain exact-match only.
            if self._is_canonical_split_name(split) and rec_norm == split:
                return True
        return False

    def _root_matches_requested_split(self, split: str) -> bool:
        """Whether preprocessed_dir itself can be treated as the requested split.

        This supports calls such as PreprocessedBDSEDataset('/cache/train_boston',
        split='train_boston') without falling back to broad ROOT scans for missing
        concrete splits like train_pittsburgh.
        """
        root_name = self.preprocessed_dir.name
        if root_name == split:
            return True
        if self._is_canonical_split_name(split) and normalize_split_name(root_name) == split:
            return True
        return False

    def _path_matches_requested_split(self, path: Path, split: str) -> bool:
        """Best-effort split ownership for balanced multi-split caps.

        Cache layouts can be ROOT/train_boston/*.npz, ROOT/train/train_boston/*.npz,
        or manifest paths under log subfolders. Exact concrete split names are
        preferred; canonical names such as ``train`` intentionally match all
        concrete train_* folders.
        """
        parts = tuple(str(x) for x in Path(path).parts)
        if split in parts:
            return True
        norm = normalize_split_name(split)
        if self._is_canonical_split_name(split):
            return any(normalize_split_name(part) == norm for part in parts)
        return False

    def _concrete_split_key_for_path(self, path: Path, requested_split: str) -> str:
        """Infer the concrete cache bucket that owns ``path``.

        A canonical request such as ``split=train`` may scan several concrete
        folders (``train_boston``, ``train_singapore``, or
        ``train/train_boston``).  Per-split caps must apply to those concrete
        folders; otherwise ``--max-scenarios-per-split`` with the default
        ``--split train`` collapses the dataset to the first sorted folder.
        """
        norm = normalize_split_name(str(requested_split))
        parts = tuple(str(x) for x in Path(path).parts)
        # Prefer explicit concrete folder names such as train_boston.
        for part in parts:
            if part != norm and normalize_split_name(part) == norm:
                return part
        # Handle ROOT/train/train_boston/log/file.npz even if the loop above was
        # conservative because of unusual parent names.
        for idx, part in enumerate(parts[:-1]):
            nxt = parts[idx + 1]
            if part == norm and nxt != norm and normalize_split_name(nxt) == norm:
                return nxt
        return norm

    @staticmethod
    def _round_robin_cap(groups: dict[str, list[Path]], order: list[str], total_cap: int | None, other: list[Path] | None = None) -> list[Path]:
        if total_cap is None:
            out: list[Path] = []
            for key in order:
                out.extend(groups.get(key, []))
            if other:
                out.extend(other)
            return out
        out = []
        idx = 0
        while len(out) < total_cap:
            added = False
            for key in order:
                g = groups.get(key, [])
                if idx < len(g):
                    out.append(g[idx])
                    added = True
                    if len(out) >= total_cap:
                        break
            if not added:
                break
            idx += 1
        if other and len(out) < total_cap:
            used = set(out)
            for p in other:
                if p not in used:
                    out.append(p)
                    if len(out) >= total_cap:
                        break
        return out

    def _apply_training_caps(self, paths: list[Path]) -> list[Path]:
        """Apply caps without silently collapsing multi-city training to one folder.

        Earlier behavior sorted all paths globally and then sliced ``[:max_scenarios]``.
        With concrete split lists such as train_boston/train_singapore/... this could
        select only the alphabetically first city when that folder had enough files.
        The same issue also occurs for the canonical request ``split=train`` when
        the cache contains concrete train_* folders and ``max_scenarios_per_split``
        is set.
        """
        total_cap = None if self.max_scenarios is None else max(0, int(self.max_scenarios))
        per_cap = None if self.max_scenarios_per_split is None else max(0, int(self.max_scenarios_per_split))
        splits = list(self.splits or [])
        if not splits:
            if total_cap is not None:
                paths = paths[:total_cap]
            return paths
        if len(splits) <= 1:
            split = splits[0]
            if per_cap is not None and self._is_canonical_split_name(split):
                groups: dict[str, list[Path]] = {}
                for p in paths:
                    groups.setdefault(self._concrete_split_key_for_path(p, split), []).append(p)
                if len(groups) > 1:
                    groups = {k: v[:per_cap] for k, v in groups.items()}
                    order = sorted(groups)
                    return self._round_robin_cap(groups, order, total_cap)
            if per_cap is not None:
                paths = paths[:per_cap]
            if total_cap is not None:
                paths = paths[:total_cap]
            return paths

        groups: dict[str, list[Path]] = {s: [] for s in splits}
        other: list[Path] = []
        for p in paths:
            assigned = False
            for split in splits:
                if self._path_matches_requested_split(p, split):
                    groups[split].append(p)
                    assigned = True
                    break
            if not assigned:
                other.append(p)

        if per_cap is not None:
            groups = {k: v[:per_cap] for k, v in groups.items()}

        return self._round_robin_cap(groups, splits, total_cap, other)

    def _split_search_roots(self) -> list[Path]:
        root = self.preprocessed_dir
        if self.splits is None:
            return [root]
        roots: list[Path] = []
        for split in self.splits:
            if root.exists() and self._root_matches_requested_split(split):
                roots.append(root)
            direct = root / split
            if direct.exists():
                roots.append(direct)
            norm = normalize_split_name(split)
            # Concrete city/folder selectors such as train_boston can appear either
            # as ROOT/train_boston/... or as ROOT/train/train_boston/... depending on
            # whether preprocessing was invoked with --splits train_boston or with
            # --split train --folders train_boston.  Search both without broadening a
            # concrete selector to every training folder.
            nested = root / norm / split
            if split != norm and nested.exists():
                roots.append(nested)
            if self._is_canonical_split_name(split) and root.exists():
                for child in sorted(root.iterdir(), key=lambda p: p.name):
                    if child.is_dir() and normalize_split_name(child.name) == split:
                        roots.append(child)
        return sorted(dict.fromkeys(roots), key=lambda p: str(p))

    def _manifest_paths(self) -> list[Path]:
        root = self.preprocessed_dir
        paths = [root / self.manifest_name]
        for search_root in self._split_search_roots():
            paths.append(search_root / self.manifest_name)
        if self.splits is None and root.exists():
            for child in sorted(root.iterdir(), key=lambda p: p.name):
                if child.is_dir():
                    paths.append(child / self.manifest_name)
        return sorted(dict.fromkeys(paths), key=lambda p: str(p))

    def _paths_from_manifest(self, manifest_path: Path) -> list[Path]:
        paths: list[Path] = []
        if not manifest_path.exists():
            return paths
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not self._record_matches_split(rec.get("split", "")):
                    continue
                path = Path(str(rec.get("path", "")))
                if not path.is_absolute():
                    path = self.preprocessed_dir / path
                if path.exists():
                    paths.append(path)
        return paths

    def build_index(self) -> list[Path]:
        if self._paths is not None:
            return self._paths
        if not self.preprocessed_dir.exists():
            raise FileNotFoundError(f"preprocessed cache root does not exist: {self.preprocessed_dir}")
        paths: list[Path] = []
        for mp in self._manifest_paths():
            paths.extend(self._paths_from_manifest(mp))
        # Always union manifest entries with an on-disk scan.  Resumed preprocessing
        # can legitimately skip already-existing .npz files without appending fresh
        # manifest rows; relying only on a partial manifest would hide those samples
        # from training/evaluation.
        search_roots = self._split_search_roots()
        for search_root in search_roots:
            if search_root.exists():
                paths.extend(
                    p for p in search_root.rglob("*.npz")
                    if not p.name.startswith(".") and ".tmp." not in p.name
                )
        # Manifests may contain duplicates after resumed preprocessing. Keep one
        # copy of each path and sort deterministically.
        paths = sorted(dict.fromkeys(paths), key=lambda p: str(p))
        paths = self._apply_training_caps(paths)
        if not paths:
            hint = ""
            if self.splits is not None:
                hint = f" split={','.join(self.splits)}"
            raise FileNotFoundError(f"No .npz samples found under {self.preprocessed_dir}{hint}")
        self._paths = paths
        return self._paths

    def __len__(self) -> int:
        return len(self.build_index())

    def __getitem__(self, idx: int) -> Sample:
        return load_sample_npz(self.build_index()[idx])

    def iter_samples(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]

def discover_available_splits(data_cache_root: str | Path = "/data0/nuplan/data/cache") -> dict[str, list[str]]:
    records = discover_db_files(data_cache_root)
    out: dict[str, list[str]] = {}
    for rec in records:
        out.setdefault(rec.split, [])
        if rec.folder not in out[rec.split]:
            out[rec.split].append(rec.folder)
    return {k: sorted(v) for k, v in sorted(out.items())}
