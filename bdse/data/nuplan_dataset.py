from __future__ import annotations

import importlib
import json
import os
import re
import time
import sqlite3
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from bdse.config import load_config
from bdse.data.cache_schema import Sample, load_sample_npz, save_sample_npz
from bdse.data.label_builder import build_training_sample_from_scenario
from bdse.data.scenario_sampler import DBFileRecord, db_files_for_nuplan_builder, discover_db_files, normalize_split_name, select_records

@dataclass(frozen=True, slots=True)
class ScenarioIndexRecord:
    db_path: Path
    split: str
    folder: str
    token: str
    timestamp_us: int
    iteration: int


@dataclass(frozen=True, slots=True)
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
        scenario_filter_limit = self.max_scenarios
        if scenario_filter_limit is None:
            scenario_filter_limit = preprocess_cfg.get("scenario_filter_limit_total_scenarios", None)

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
            return int(ts) + int(iteration)
    return int(iteration)


def _scenario_log_name(scenario: Any) -> str:
    return _safe_name(getattr(scenario, "log_name", getattr(scenario, "_log_name", "log")))


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
                f"max_scenarios={self.max_scenarios} max_samples_per_log={self.max_samples_per_log}",
                flush=True,
            )
            show_index_progress = total_scenarios >= int(self.cfg.get("preprocess", {}).get("index_progress_threshold", 10000))
            iterator = _maybe_tqdm(scenarios, total_scenarios, f"index:{self.split}", show_index_progress)
            cap_strategy = str(self.cfg.get("preprocess", {}).get("max_samples_per_log_strategy", "first")).lower()
            if cap_strategy not in {"first", "uniform"}:
                raise ValueError(f"Unsupported max_samples_per_log_strategy={cap_strategy!r}; expected 'first' or 'uniform'.")
            per_log_counts: dict[str, int] = {}
            for scenario in iterator:
                token = _scenario_token(scenario)
                log_name = _scenario_log_name(scenario)
                folder = folder_lookup.get(log_name, folder_lookup.get(_safe_name(log_name), default_folder))
                n_iter = _num_iterations(scenario)
                for iteration in range(0, n_iter, max(self.stride, 1)):
                    if self.max_samples_per_log is not None and cap_strategy == "first" and per_log_counts.get(log_name, 0) >= self.max_samples_per_log:
                        break
                    timestamp_us = _scenario_iteration_timestamp_us(scenario, iteration)
                    out.append(DevkitScenarioIndexRecord(scenario, self.split, folder, log_name, token, iteration, timestamp_us))
                    per_log_counts[log_name] = per_log_counts.get(log_name, 0) + 1
                    if self.max_scenarios is not None and len(out) >= self.max_scenarios:
                        self._index = out
                        print(f"[bdse] built scenario index: split={self.split} records={len(out)}", flush=True)
                        return self._index

            if self.max_samples_per_log is not None and cap_strategy == "uniform":
                grouped: dict[str, list[DevkitScenarioIndexRecord]] = {}
                for rec in out:
                    grouped.setdefault(rec.log_name, []).append(rec)
                capped: list[DevkitScenarioIndexRecord] = []
                cap = int(self.max_samples_per_log)
                for log_name in sorted(grouped):
                    rows = grouped[log_name]
                    if len(rows) <= cap:
                        capped.extend(rows)
                        continue
                    # Evenly sample across the whole log. The previous implementation
                    # kept only the first cap frames, which can overrepresent the beginning
                    # of every DB file and bias route/traffic-light/interaction coverage.
                    idx = np.linspace(0, len(rows) - 1, cap).round().astype(np.int64)
                    idx = np.unique(idx)
                    if idx.size < cap:
                        used = set(idx.tolist())
                        missing = [i for i in range(len(rows)) if i not in used]
                        idx = np.asarray(sorted(list(idx) + missing[: cap - idx.size]), dtype=np.int64)
                    capped.extend(rows[int(i)] for i in idx[:cap])
                out = _sort_devkit_records_for_temporal_cache(capped)
                print(
                    f"[bdse] applied uniform per-log cap: split={self.split} "
                    f"logs={len(grouped)} cap={cap} records={len(out)}",
                    flush=True,
                )

            out = _sort_devkit_records_for_temporal_cache(out)
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

    def _write_one_preprocessed_index(self, i: int, path: Path, manifest_path: Path | None) -> tuple[int, Path, dict[str, Any] | None]:
        pcfg = self.cfg.get("preprocess", {})
        profile = bool(pcfg.get("profile", False))
        threshold_s = float(pcfg.get("profile_threshold_s", 2.0))
        t0 = time.perf_counter()
        sample = self[i]
        t_sample = time.perf_counter()
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
        check_iter = _maybe_tqdm(range(total), total, f"cache-check:{self.split}", show_progress and total >= 10000)
        for i in check_iter:
            p = self.cache_path_for_index(i, out)
            all_paths[i] = p
            if resume and not overwrite and p.exists():
                skipped += 1
            else:
                pending.append((i, p))

        print(
            f"[bdse] split={self.split}: records={len(self.records)} scenarios/iterations={total} "
            f"pending={len(pending)} skipped={skipped} out={out}",
            flush=True,
        )
        if not pending:
            return all_paths

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

        def _handle_result(fut: Future, future_map: dict[Future, tuple[int, Path]]):
            i0, path0 = future_map.pop(fut)
            try:
                i, written, rec = fut.result()
            except Exception as exc:
                raise RuntimeError(f"Failed preprocessing index={i0} path={path0}") from exc
            paths[i] = written
            if rec is not None:
                manifest_records.append(rec)

        if workers <= 1:
            iterator = _maybe_tqdm(pending, len(pending), f"preprocess:{self.split}", show_progress)
            for i, path in iterator:
                _, written, rec = self._write_one_preprocessed_index(i, path, manifest_path)
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
            pending_iter = iter(pending)
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures: dict[Future, tuple[int, Path]] = {}
                def submit_next() -> bool:
                    try:
                        i, path = next(pending_iter)
                    except StopIteration:
                        return False
                    futures[ex.submit(self._write_one_preprocessed_index, i, path, manifest_path)] = (i, path)
                    return True

                for _ in range(min(max_in_flight, len(pending))):
                    submit_next()
                pbar = _maybe_tqdm(range(len(pending)), len(pending), f"preprocess:{self.split}", show_progress)
                try:
                    while futures:
                        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                        for fut in done:
                            _handle_result(fut, futures)
                            completed += 1
                            if hasattr(pbar, "update"):
                                pbar.update(1)
                            submit_next()
                            if len(manifest_records) >= 256:
                                _append_manifest(manifest_records)
                finally:
                    if hasattr(pbar, "close"):
                        pbar.close()
        _append_manifest(manifest_records)
        return paths


class PreprocessedBDSEDataset:
    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str | list[str] | tuple[str, ...] | None = None,
        manifest_name: str = "manifest.jsonl",
        max_scenarios: int | None = None,
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

    def _split_search_roots(self) -> list[Path]:
        root = self.preprocessed_dir
        if self.splits is None:
            return [root]
        roots: list[Path] = []
        for split in self.splits:
            direct = root / split
            if direct.exists():
                roots.append(direct)
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
        if not paths:
            search_roots = self._split_search_roots()
            if not search_roots:
                search_roots = [self.preprocessed_dir]
            for search_root in search_roots:
                if search_root.exists():
                    paths.extend(
                        p for p in search_root.rglob("*.npz")
                        if not p.name.startswith(".") and ".tmp." not in p.name
                    )
        # Manifests may contain duplicates after resumed preprocessing. Keep one
        # copy of each path and sort deterministically.
        paths = sorted(dict.fromkeys(paths), key=lambda p: str(p))
        if self.max_scenarios is not None:
            paths = paths[: int(self.max_scenarios)]
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
