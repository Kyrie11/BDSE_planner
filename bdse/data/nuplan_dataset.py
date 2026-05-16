from __future__ import annotations

import importlib
import sqlite3
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
    def __init__(self, cfg: dict[str, Any], records: list[DBFileRecord], split: str):
        self.cfg = cfg
        self.records = records
        self.split = split
        self._scenarios: list[Any] | None = None

    def _build_with_devkit(self) -> list[Any]:
        try:
            builder_mod = importlib.import_module("nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder")
            filter_mod = importlib.import_module("nuplan.planning.scenario_builder.scenario_filter")
        except ImportError as exc:
            raise RuntimeError("nuPlan devkit is not installed; install nuplan-devkit or use preprocessed cache.") from exc
        NuPlanScenarioBuilder = getattr(builder_mod, "NuPlanScenarioBuilder")
        ScenarioFilter = getattr(filter_mod, "ScenarioFilter")
        paths = self.cfg.get("paths", {})
        builder = NuPlanScenarioBuilder(
            data_root=str(paths.get("data_cache_root", "/data0/nuplan/data/cache")),
            map_root=str(paths.get("maps_root", "/data0/nuplan/dataset/maps")),
            sensor_root=str(paths.get("sensor_root", paths.get("data_cache_root", "/data0/nuplan/data/cache"))),
            db_files=db_files_for_nuplan_builder(self.records),
            map_version=str(paths.get("map_version", "nuplan-maps-v1.0")),
            include_cameras=False,
            max_workers=None,
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
        return list(builder.get_scenarios(scenario_filter, worker=None))

    def scenarios(self) -> list[Any]:
        if self._scenarios is None:
            self._scenarios = self._build_with_devkit()
        return self._scenarios


class NuPlanBDSEDataset:
    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        split: str = "train",
        folders: list[str] | None = None,
        max_files: int | None = None,
        max_scenarios: int | None = None,
        stride: int = 10,
        use_devkit: bool = True,
        preprocessed_dir: str | Path | None = None,
    ):
        self.cfg = cfg or load_config()
        records = discover_db_files(self.cfg.get("paths", {}).get("data_cache_root", "/data0/nuplan/data/cache"))
        self.records = select_records(records, split=split, folders=folders, max_files=max_files, seed=int(self.cfg.get("seed", 17)))
        self.split = split
        self.use_devkit = use_devkit
        self.preprocessed_dir = Path(preprocessed_dir or self.cfg.get("paths", {}).get("preprocessed_cache", "cache"))
        self.max_scenarios = max_scenarios
        self.stride = stride
        self._scenario_source = NuPlanScenarioSource(self.cfg, self.records, split) if use_devkit else None
        self._index: list[Any] | None = None

    def build_index(self) -> list[Any]:
        if self._index is not None:
            return self._index
        if self.use_devkit:
            scenarios = self._scenario_source.scenarios() if self._scenario_source is not None else []
            if self.max_scenarios is not None:
                scenarios = scenarios[: self.max_scenarios]
            self._index = scenarios
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
            scenario = item
            iteration = 0
            return build_training_sample_from_scenario(scenario, iteration, self.cfg)
        raise RuntimeError("Raw SQLite indexing is available for discovery only; use nuPlan devkit for sample construction.")

    def iter_samples(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]

    def write_preprocessed_cache(self, out_dir: str | Path | None = None) -> list[Path]:
        out = Path(out_dir or self.preprocessed_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, sample in enumerate(self.iter_samples()):
            path = out / self.split / f"{i:08d}_{sample.scenario_token}.npz"
            save_sample_npz(sample, path)
            paths.append(path)
        return paths


def discover_available_splits(data_cache_root: str | Path = "/data0/nuplan/data/cache") -> dict[str, list[str]]:
    records = discover_db_files(data_cache_root)
    out: dict[str, list[str]] = {}
    for rec in records:
        out.setdefault(rec.split, [])
        if rec.folder not in out[rec.split]:
            out[rec.split].append(rec.folder)
    return {k: sorted(v) for k, v in sorted(out.items())}
