from __future__ import annotations

import importlib
import json
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from bdse.config import load_config
from bdse.data.cache_schema import Sample, save_sample_npz
from bdse.data.label_builder import build_training_sample_from_scenario
from bdse.data.scenario_sampler import DBFileRecord, db_files_for_nuplan_builder, discover_db_files, select_records

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
    def __init__(self, cfg: dict[str, Any], records: list[DBFileRecord], split: str, num_workers=None, use_process_pool=None):
        self.cfg = cfg
        self.records = records
        self.split = split
        self._scenarios: list[Any] | None = None
        self.num_workers = num_workers
        self.use_process_pool = use_process_pool

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

        db_files = db_files_for_nuplan_builder(self.records)
        print(
            f"[bdse] building nuPlan scenarios: split={self.split} db_files={len(db_files)} "
            f"builder_workers={num_workers} builder_process_pool={builder_use_process_pool}",
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
        scenario_filter = ScenarioFilter(
            scenario_types=None,
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=None,
            limit_total_scenarios=None,
            timestamp_threshold_s=None,
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


def _scenario_log_name(scenario: Any) -> str:
    return _safe_name(getattr(scenario, "log_name", getattr(scenario, "_log_name", "log")))


def _scenario_folder_for_log(records: list[DBFileRecord], log_name: str, default_split: str) -> str:
    for rec in records:
        if rec.log_name == log_name or rec.path.stem == log_name:
            return rec.folder
    for rec in records:
        if rec.split == default_split:
            return rec.folder
    return default_split


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
        self._index: list[Any] | None = None
        self.num_workers = int(num_workers if num_workers is not None else self.cfg.get("preprocess", {}).get("num_workers", 1) or 1)
        self.use_process_pool = bool(use_process_pool if use_process_pool is not None else self.cfg.get("preprocess", {}).get("use_process_pool", False))
        self._scenario_source = NuPlanScenarioSource(
            self.cfg,
            self.records,
            split,
            num_workers=self.cfg.get("preprocess", {}).get("scenario_builder_workers", min(self.num_workers, 4)),
            use_process_pool=self.cfg.get("preprocess", {}).get("scenario_builder_use_process_pool", False),
        ) if use_devkit else None

    def build_index(self) -> list[Any]:
        if self._index is not None:
            return self._index
        if self.use_devkit:
            scenarios = self._scenario_source.scenarios() if self._scenario_source is not None else []
            out: list[DevkitScenarioIndexRecord] = []
            for scenario in scenarios:
                token = _scenario_token(scenario)
                log_name = _scenario_log_name(scenario)
                folder = _scenario_folder_for_log(self.records, log_name, self.split)
                n_iter = _num_iterations(scenario)
                for iteration in range(0, n_iter, max(self.stride, 1)):
                    out.append(DevkitScenarioIndexRecord(scenario, self.split, folder, log_name, token, iteration))
                    if self.max_scenarios is not None and len(out) >= self.max_scenarios:
                        self._index = out
                        return self._index
            self._index = out
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
            return out / item.split / item.folder / item.log_name / f"{item.token}_it{item.iteration:06d}.npz"
        if isinstance(item, ScenarioIndexRecord):
            return out / item.split / item.folder / item.db_path.stem / f"{_safe_name(item.token)}_it{item.iteration:06d}.npz"
        return out / self.split / f"{idx:08d}.npz"

    def _write_one_preprocessed_index(self, i: int, path: Path, manifest_path: Path | None) -> tuple[int, Path, dict[str, Any] | None]:
        sample = self[i]
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{i}")
        save_sample_npz(sample, tmp)
        os.replace(tmp, path)
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
        manifest_path = out / (manifest_name or self.cfg.get("preprocess", {}).get("manifest_name", "manifest.jsonl"))

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

        all_paths = [self.cache_path_for_index(i, out) for i in range(total)]
        if resume and not overwrite:
            pending = [(i, p) for i, p in enumerate(all_paths) if not p.exists()]
            skipped = total - len(pending)
        else:
            pending = list(enumerate(all_paths))
            skipped = 0

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

        def _progress(iterable, total_count: int, desc: str):
            if not show_progress:
                return iterable
            from tqdm import tqdm
            return tqdm(iterable, total=total_count, desc=desc, file=sys.stdout, mininterval=1.0)

        if workers <= 1:
            for i, path in _progress(pending, len(pending), f"preprocess:{self.split}"):
                _, written, rec = self._write_one_preprocessed_index(i, path, manifest_path)
                paths[i] = written
                if rec is not None:
                    manifest_records.append(rec)
        else:
            # Threading overlaps DB/map I/O without attempting to pickle nuPlan scenarios.
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(self._write_one_preprocessed_index, i, path, manifest_path): (i, path) for i, path in pending}
                for fut in _progress(as_completed(futures), len(futures), f"preprocess:{self.split}"):
                    i0, path0 = futures[fut]
                    try:
                        i, written, rec = fut.result()
                    except Exception as exc:
                        raise RuntimeError(f"Failed preprocessing index={i0} path={path0}") from exc
                    paths[i] = written
                    if rec is not None:
                        manifest_records.append(rec)

        if manifest_records:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("a", encoding="utf-8") as f:
                for rec in sorted(manifest_records, key=lambda r: str(r["path"])):
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
        return paths


def discover_available_splits(data_cache_root: str | Path = "/data0/nuplan/data/cache") -> dict[str, list[str]]:
    records = discover_db_files(data_cache_root)
    out: dict[str, list[str]] = {}
    for rec in records:
        out.setdefault(rec.split, [])
        if rec.folder not in out[rec.split]:
            out[rec.split].append(rec.folder)
    return {k: sorted(v) for k, v in sorted(out.items())}
