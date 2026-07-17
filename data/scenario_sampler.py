from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class DBFileRecord:
    split: str
    folder: str
    path: Path

    @property
    def log_name(self) -> str:
        return self.path.stem


def normalize_split_name(folder_name: str) -> str:
    low = folder_name.lower()
    if low in {"val", "validation", "valid"} or low.startswith("val"):
        return "val"
    if low.startswith("train"):
        return "train"
    if low.startswith("test"):
        return "test"
    if low.startswith("mini"):
        return "mini"
    return low


def discover_db_files(cache_root: str | Path) -> list[DBFileRecord]:
    root = Path(cache_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"nuPlan DB cache root does not exist: {root}")
    records: list[DBFileRecord] = []
    if root.suffix == ".db":
        records.append(DBFileRecord(split=normalize_split_name(root.parent.name), folder=root.parent.name, path=root))
        return records
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            dbs = sorted(child.rglob("*.db"), key=lambda p: str(p))
            split = normalize_split_name(child.name)
            for db in dbs:
                records.append(DBFileRecord(split=split, folder=child.name, path=db))
        elif child.suffix == ".db":
            records.append(DBFileRecord(split=normalize_split_name(root.name), folder=root.name, path=child))
    if not records:
        raise FileNotFoundError(f"No .db files found under {root}")
    return records


def group_by_split(records: Iterable[DBFileRecord]) -> dict[str, list[DBFileRecord]]:
    grouped: dict[str, list[DBFileRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.split, []).append(rec)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda r: str(r.path))
    return grouped


def select_records(
    records: Iterable[DBFileRecord],
    split: str | None = None,
    folders: Iterable[str] | None = None,
    max_files: int | None = None,
    seed: int = 17,
) -> list[DBFileRecord]:
    recs = list(records)
    if split is not None:
        # ``train_1`` / ``train_2`` style folders should be addressable as
        # concrete DB buckets.  The previous predicate also matched their
        # normalized split (``train``), so requesting ``train_1`` accidentally
        # selected every training DB and then wrote them below a single
        # ``train_1`` output prefix.  Prefer an exact folder match whenever the
        # requested name is a sub-split folder; keep canonical names such as
        # ``train`` and ``val`` as aggregate split selectors.
        norm = normalize_split_name(split)
        exact_folders = {r.folder for r in recs}
        if split in exact_folders and split != norm:
            recs = [r for r in recs if r.folder == split]
        else:
            recs = [r for r in recs if r.split == norm]
    if folders is not None:
        folder_set = set(folders)
        recs = [r for r in recs if r.folder in folder_set]
    recs = sorted(recs, key=lambda r: str(r.path))
    if max_files is not None and len(recs) > max_files:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(recs), size=max_files, replace=False))
        recs = [recs[int(i)] for i in idx]
    return recs


def db_files_for_nuplan_builder(records: Iterable[DBFileRecord]) -> list[str]:
    return [str(r.path) for r in records]
