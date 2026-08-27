from __future__ import annotations

"""Persistent compact training cache for external fixed-budget baselines.

The nuPlan/BDSE source cache stores one NPZ per sample and contains many JSON and
teacher/runtime fields.  External planners only consume a small fixed-shape tensor
contract.  Re-decoding hundreds of thousands of NPZ/JSON samples every epoch is
therefore pure input-pipeline overhead.

This module materializes that compact tensor contract once into three row-major
NumPy mmap arrays (float32/int64/bool).  All requested evidence budgets share the
same scene/candidate/evidence features; only the proposal oracle mask is stored per
budget.  Training can then globally shuffle row indices while reading a few
contiguous mmap rows instead of opening/decompressing dozens of NPZ members.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bdse.config import load_config
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.external_baselines.data import (
    _oracle_selected_mask,
    external_sample_to_model_inputs,
    load_external_training_sample_npz,
)

CACHE_VERSION = 2
MANIFEST_NAME = "compact_manifest.json"
FLOAT_FILE = "float32.npy"
INT_FILE = "int64.npy"
BOOL_FILE = "bool.npy"
STATE_NAME = "build_state.json"


def _planner_supervision(cfg: dict[str, Any]) -> str:
    return str((cfg.get("external_baseline", {}) or {}).get("planner_supervision", "teacher_cost")).strip().lower()


def _cfg_for_budget(cfg: dict[str, Any], budget: int) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("evidence", {})["budget"] = int(budget)
    ext = out.setdefault("external_baseline", {})
    ext["budget"] = int(budget)
    selector = out.setdefault("selector", {})
    selector["min_selected_atoms"] = int(budget)
    selector["force_fill_budget"] = True
    fallback = out.setdefault("fallback", {})
    fallback["enabled"] = False
    fallback["max_additional_stages"] = 0
    fallback["budget_stages"] = [int(budget)]
    return out


def _path_order_sha256(paths: Sequence[Path], root: str | Path) -> str:
    root_p = Path(root).resolve()
    h = hashlib.sha256()
    for p in paths:
        rp = Path(p).resolve()
        try:
            rel = rp.relative_to(root_p).as_posix()
        except ValueError:
            rel = rp.as_posix()
        h.update(rel.encode("utf-8", errors="surrogatepass"))
        h.update(b"\n")
    return h.hexdigest()


def _data_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    ecfg = cfg.get("external_baseline", {}) or {}
    mcfg = cfg.get("model", {}) or {}
    return {
        "candidate": cfg.get("candidate", {}) or {},
        "evidence_max_atoms": int((cfg.get("evidence", {}) or {}).get("max_atoms", 128)),
        "pairs_target_max": int((cfg.get("pairs", {}) or {}).get("target_max", 256)),
        "runtime": cfg.get("runtime", {}) or {},
        "planner_supervision": _planner_supervision(cfg),
        "evidence_feature_dim": int(mcfg.get("evidence_feature_dim", 24)),
        "proposal_feature_dim": int(mcfg.get("proposal_feature_dim", 24)),
        "map_feature_dim": int(mcfg.get("map_feature_dim", 8)),
        "route_feature_dim": int(mcfg.get("route_feature_dim", 8)),
        "goal_feature_dim": int(mcfg.get("goal_feature_dim", 4)),
        "max_polyline_points": int(mcfg.get("max_polyline_points", 64)),
        "max_traffic_tokens": int(mcfg.get("max_traffic_tokens", 32)),
        "proposal_loss_enabled": float((ecfg.get("loss_weights", {}) or {}).get("proposal", 0.25)) != 0.0,
        "pair_loss_enabled": float((ecfg.get("loss_weights", {}) or {}).get("pair", 0.0)) != 0.0,
    }


def _contract_sha256(cfg: dict[str, Any]) -> str:
    payload = json.dumps(_data_contract(cfg), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _sample_tensor_dict(path: Path, cfg: dict[str, Any], budgets: Sequence[int], *, include_label_future: bool) -> dict[str, torch.Tensor]:
    sample = load_external_training_sample_npz(path, include_label_future=include_label_future)
    base_budget = int(budgets[0])
    base_cfg = _cfg_for_budget(cfg, base_budget)
    tensors = external_sample_to_model_inputs(sample, base_cfg)

    # The scene/candidate/evidence compact contract is budget-independent.  Only
    # the greedy proposal-supervision mask changes with B, so keep one mask per B.
    base_oracle = tensors.pop("oracle_selected_mask", None)
    proposal_enabled = float(((cfg.get("external_baseline", {}) or {}).get("loss_weights", {}) or {}).get("proposal", 0.25)) != 0.0
    if proposal_enabled:
        if base_oracle is None:
            raise RuntimeError("proposal supervision is enabled but external tensorizer returned no oracle_selected_mask")
        tensors[f"oracle_selected_mask_B{base_budget}"] = base_oracle
        for budget in budgets[1:]:
            bcfg = _cfg_for_budget(cfg, int(budget))
            tensors[f"oracle_selected_mask_B{int(budget)}"] = torch.from_numpy(_oracle_selected_mask(sample, bcfg))
    return tensors


class _CompactBuildDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[Path],
        cfg: dict[str, Any],
        budgets: Sequence[int],
        *,
        include_label_future: bool,
        start_index: int = 0,
    ) -> None:
        self.paths = list(paths)
        self.cfg = cfg
        self.budgets = tuple(int(x) for x in budgets)
        self.include_label_future = bool(include_label_future)
        self.start_index = int(start_index)

    def __len__(self) -> int:
        return max(0, len(self.paths) - self.start_index)

    def __getitem__(self, local_idx: int) -> dict[str, torch.Tensor]:
        idx = self.start_index + int(local_idx)
        return _sample_tensor_dict(
            self.paths[idx], self.cfg, self.budgets,
            include_label_future=self.include_label_future,
        )


def _schema_from_tensors(tensors: dict[str, torch.Tensor]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    fields: list[dict[str, Any]] = []
    offsets = {"float32": 0, "int64": 0, "bool": 0}
    for name, tensor in tensors.items():
        arr = tensor.detach().cpu().numpy()
        shape = list(arr.shape)
        size = int(arr.size) if arr.ndim else 1
        if arr.dtype == np.bool_:
            group = "bool"
        elif np.issubdtype(arr.dtype, np.integer):
            group = "int64"
        else:
            group = "float32"
        fields.append({"name": name, "group": group, "offset": offsets[group], "size": size, "shape": shape})
        offsets[group] += size
    return fields, offsets


def _open_group_memmaps(output_dir: Path, count: int, widths: dict[str, int], *, mode: str) -> dict[str, np.memmap]:
    specs = {
        "float32": (FLOAT_FILE, np.float32),
        "int64": (INT_FILE, np.int64),
        "bool": (BOOL_FILE, np.bool_),
    }
    out: dict[str, np.memmap] = {}
    for group, (filename, dtype) in specs.items():
        width = int(widths.get(group, 0))
        if width <= 0:
            continue
        path = output_dir / filename
        if mode == "w+":
            out[group] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(int(count), width))
        else:
            out[group] = np.load(path, mmap_mode=mode)
            if tuple(out[group].shape) != (int(count), width):
                raise RuntimeError(f"compact cache shape mismatch for {path}: {out[group].shape} != {(count, width)}")
    return out


def _write_batch(mem: dict[str, np.memmap], fields: list[dict[str, Any]], batch: dict[str, torch.Tensor], start: int, end: int) -> None:
    bsz = int(end - start)
    for f in fields:
        name, group = str(f["name"]), str(f["group"])
        off, size = int(f["offset"]), int(f["size"])
        t = batch[name].detach().cpu()
        if group == "float32":
            arr = t.numpy().astype(np.float32, copy=False).reshape(bsz, size)
        elif group == "int64":
            arr = t.numpy().astype(np.int64, copy=False).reshape(bsz, size)
        else:
            arr = t.numpy().astype(np.bool_, copy=False).reshape(bsz, size)
        mem[group][start:end, off:off + size] = arr


def build_compact_cache(
    *,
    cfg: dict[str, Any],
    preprocessed_dir: str | Path,
    split: Sequence[str],
    output_dir: str | Path,
    budgets: Sequence[int],
    num_workers: int = 10,
    prefetch_factor: int = 4,
    batch_size: int = 32,
    max_scenarios: int | None = None,
    max_scenarios_per_split: int | None = None,
    resume: bool = True,
    log_every: int = 100,
) -> Path:
    budgets = tuple(sorted({int(x) for x in budgets}))
    if not budgets or any(x <= 0 for x in budgets):
        raise ValueError(f"invalid budgets={budgets}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST_NAME
    state_path = output / STATE_NAME

    source = PreprocessedBDSEDataset(
        preprocessed_dir,
        split=list(split),
        max_scenarios=max_scenarios,
        max_scenarios_per_split=max_scenarios_per_split,
    )
    paths = [Path(p) for p in source.build_index()]
    if not paths:
        raise RuntimeError(f"no source NPZ files found under {preprocessed_dir} split={list(split)}")
    count = len(paths)
    path_hash = _path_order_sha256(paths, preprocessed_dir)
    contract_hash = _contract_sha256(cfg)
    include_label_future = _planner_supervision(cfg) == "expert_imitation"

    # Probe one real sample to make the cache schema fully data-contract driven.
    probe = _sample_tensor_dict(paths[0], cfg, budgets, include_label_future=include_label_future)
    fields, widths = _schema_from_tensors(probe)
    schema_payload = json.dumps({"fields": fields, "widths": widths}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    schema_hash = hashlib.sha256(schema_payload).hexdigest()

    expected = {
        "version": CACHE_VERSION,
        "count": count,
        "source_root": str(Path(preprocessed_dir).resolve()),
        "split": list(split),
        "path_order_sha256": path_hash,
        "contract_sha256": contract_hash,
        "schema_sha256": schema_hash,
        "budgets": list(budgets),
        "fields": fields,
        "widths": widths,
    }

    estimated_bytes = int(count) * (
        int(widths.get("float32", 0)) * np.dtype(np.float32).itemsize
        + int(widths.get("int64", 0)) * np.dtype(np.int64).itemsize
        + int(widths.get("bool", 0)) * np.dtype(np.bool_).itemsize
    )
    free_bytes = shutil.disk_usage(output).free
    if not manifest_path.is_file() and free_bytes < int(estimated_bytes * 1.10):
        raise RuntimeError(
            f"insufficient free space for compact cache {output}: estimated={estimated_bytes/(1024**3):.2f} GiB "
            f"free={free_bytes/(1024**3):.2f} GiB (need ~10% headroom)"
        )

    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        compatible = all(old.get(k) == expected.get(k) for k in (
            "version", "count", "source_root", "split", "path_order_sha256", "contract_sha256", "schema_sha256", "budgets", "fields", "widths"
        ))
        if compatible and bool(old.get("complete", False)):
            print(f"[compact-cache] READY path={output} samples={count} budgets={list(budgets)}", flush=True)
            return output
        if not compatible:
            raise RuntimeError(
                f"existing compact cache is incompatible: {output}. Remove it or choose a new output directory. "
                f"source/config/schema changed."
            )

    start_index = 0
    mode = "w+"
    if resume and state_path.is_file() and manifest_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_index = int(state.get("next_index", 0))
        if 0 < start_index < count:
            mode = "r+"
            print(f"[compact-cache] RESUME path={output} next_index={start_index}/{count}", flush=True)
    if mode == "w+":
        building = {**expected, "complete": False, "created_at_unix": time.time()}
        manifest_path.write_text(json.dumps(building, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        state_path.write_text(json.dumps({"next_index": 0}) + "\n", encoding="utf-8")

    mem = _open_group_memmaps(output, count, widths, mode=mode)
    if start_index >= count:
        start_index = count

    build_ds = _CompactBuildDataset(
        paths, cfg, budgets,
        include_label_future=include_label_future,
        start_index=start_index,
    )
    kwargs: dict[str, Any] = dict(
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=False,
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=_seed_worker,
    )
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = DataLoader(build_ds, **kwargs)

    t0 = time.perf_counter()
    written = start_index
    print(
        f"[compact-cache] BUILD path={output} samples={count} start={start_index} budgets={list(budgets)} "
        f"batch={batch_size} workers={num_workers} float_width={widths.get('float32',0)} "
        f"int_width={widths.get('int64',0)} bool_width={widths.get('bool',0)} estimated_gib={estimated_bytes/(1024**3):.2f}",
        flush=True,
    )
    for step, batch in enumerate(loader, start=1):
        bsz = int(next(iter(batch.values())).shape[0])
        end = written + bsz
        _write_batch(mem, fields, batch, written, end)
        written = end
        if step % max(1, int(log_every)) == 0 or written >= count:
            for mm in mem.values():
                mm.flush()
            state_path.write_text(json.dumps({"next_index": written}) + "\n", encoding="utf-8")
            elapsed = max(time.perf_counter() - t0, 1e-9)
            done_now = max(written - start_index, 0)
            rate = done_now / elapsed
            remain = max(count - written, 0) / max(rate, 1e-9)
            print(
                f"[compact-cache-progress] samples={written}/{count} ({100.0*written/max(count,1):.1f}%) "
                f"rate={rate:.1f} sample/s remaining_min={remain/60.0:.1f}",
                flush=True,
            )

    for mm in mem.values():
        mm.flush()
    final = {
        **expected,
        "complete": True,
        "completed_at_unix": time.time(),
        "source_manifest": {
            "root": str(Path(preprocessed_dir).resolve()),
            "split": list(split),
            "count": count,
            "ordered_path_sha256": path_hash,
            "compact_cache": True,
        },
        "storage_bytes": sum((output / f).stat().st_size for f in (FLOAT_FILE, INT_FILE, BOOL_FILE) if (output / f).is_file()),
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)
    state_path.unlink(missing_ok=True)
    elapsed = max(time.perf_counter() - t0, 1e-9)
    print(
        f"[compact-cache] DONE path={output} samples={count} new_samples={count-start_index} "
        f"rate={(count-start_index)/elapsed:.1f} sample/s size_gib={final['storage_bytes']/(1024**3):.2f}",
        flush=True,
    )
    return output


@dataclass
class CompactExternalCache:
    root: Path
    manifest: dict[str, Any]
    float_mm: np.ndarray | None
    int_mm: np.ndarray | None
    bool_mm: np.ndarray | None

    @classmethod
    def open(cls, root: str | Path) -> "CompactExternalCache":
        root_p = Path(root)
        manifest_path = root_p / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"compact cache manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("version", -1)) != CACHE_VERSION or not bool(manifest.get("complete", False)):
            raise RuntimeError(f"compact cache is incomplete or unsupported: {manifest_path}")
        count = int(manifest["count"])
        widths = manifest["widths"]
        def load(name: str, width: int):
            if int(width) <= 0:
                return None
            arr = np.load(root_p / name, mmap_mode="r")
            if tuple(arr.shape) != (count, int(width)):
                raise RuntimeError(f"compact cache shape mismatch: {root_p/name} {arr.shape}")
            return arr
        return cls(
            root=root_p,
            manifest=manifest,
            float_mm=load(FLOAT_FILE, int(widths.get("float32", 0))),
            int_mm=load(INT_FILE, int(widths.get("int64", 0))),
            bool_mm=load(BOOL_FILE, int(widths.get("bool", 0))),
        )

    def __len__(self) -> int:
        return int(self.manifest["count"])

    def assert_compatible(self, cfg: dict[str, Any]) -> None:
        current = _contract_sha256(cfg)
        stored = str(self.manifest.get("contract_sha256", ""))
        if current != stored:
            raise ValueError(
                f"compact cache data contract mismatch: cache={self.root} stored={stored} current={current}. "
                "Build a new compact cache for this external-baseline tensor contract."
            )

    @property
    def budgets(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.manifest.get("budgets", []))

    def source_manifest(self) -> dict[str, Any]:
        return dict(self.manifest.get("source_manifest", {}))

    def _group_batch(self, indices: np.ndarray, *, pin_memory: bool) -> dict[str, torch.Tensor]:
        groups: dict[str, torch.Tensor] = {}
        for group, mm in (("float32", self.float_mm), ("int64", self.int_mm), ("bool", self.bool_mm)):
            if mm is None:
                continue
            # Advanced indexing returns a compact writable ndarray, removing mmap
            # lifetime/writability hazards before torch takes ownership of the view.
            arr = np.asarray(mm[indices]).copy()
            t = torch.from_numpy(arr)
            if pin_memory and torch.cuda.is_available():
                t = t.pin_memory()
            groups[group] = t
        return groups

    def make_batch(self, indices: np.ndarray, *, budget: int, pin_memory: bool = True) -> dict[str, torch.Tensor]:
        budget = int(budget)
        if budget not in self.budgets:
            raise ValueError(f"budget B={budget} not present in compact cache {self.root}; available={self.budgets}")
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        groups = self._group_batch(indices, pin_memory=pin_memory)
        bsz = int(indices.size)
        out: dict[str, torch.Tensor] = {}
        oracle_name = f"oracle_selected_mask_B{budget}"
        for f in self.manifest["fields"]:
            stored_name = str(f["name"])
            if stored_name.startswith("oracle_selected_mask_B") and stored_name != oracle_name:
                continue
            public_name = "oracle_selected_mask" if stored_name == oracle_name else stored_name
            group = str(f["group"]); off = int(f["offset"]); size = int(f["size"]); shape = tuple(int(x) for x in f["shape"])
            base = groups[group][:, off:off + size]
            out[public_name] = base.reshape((bsz,) + shape) if shape else base.reshape(bsz)
        return out


class CompactBatchLoader:
    """Minimal mmap batch iterator with optional one-batch background prefetch."""

    def __init__(
        self,
        cache: CompactExternalCache,
        *,
        budget: int,
        batch_size: int,
        shuffle: bool,
        seed: int,
        pin_memory: bool = True,
        prefetch: bool = True,
        shuffle_mode: str = "global",
        block_size: int = 4096,
        limit: int | None = None,
    ) -> None:
        self.cache = cache
        self.budget = int(budget)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.pin_memory = bool(pin_memory)
        self.prefetch = bool(prefetch)
        self.shuffle_mode = str(shuffle_mode).strip().lower()
        if self.shuffle_mode not in {"global", "block", "none"}:
            raise ValueError("compact shuffle_mode must be one of: global, block, none")
        self.block_size = max(self.batch_size, int(block_size))
        self.count = min(len(cache), int(limit)) if limit is not None and int(limit) > 0 else len(cache)
        self._epoch = 0

    def __len__(self) -> int:
        return int(math.ceil(self.count / float(self.batch_size)))

    def _epoch_indices(self, epoch: int) -> np.ndarray:
        idx = np.arange(self.count, dtype=np.int64)
        if not self.shuffle or self.shuffle_mode == "none":
            return idx
        rng = np.random.default_rng(self.seed + int(epoch))
        if self.shuffle_mode == "global":
            rng.shuffle(idx)
            return idx
        # Locality-preserving shuffle: randomly permute blocks and shuffle rows
        # within each block.  Useful only when the compact mmap itself lives on
        # high-latency/rotational storage; global remains the default.
        blocks: list[np.ndarray] = []
        for start in range(0, self.count, self.block_size):
            block = idx[start:min(start + self.block_size, self.count)].copy()
            rng.shuffle(block)
            blocks.append(block)
        order = np.arange(len(blocks))
        rng.shuffle(order)
        return np.concatenate([blocks[int(i)] for i in order], axis=0)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        epoch = self._epoch
        self._epoch += 1
        idx = self._epoch_indices(epoch)
        chunks = [idx[s:min(s + self.batch_size, self.count)] for s in range(0, self.count, self.batch_size)]
        if not self.prefetch or len(chunks) <= 1:
            for chunk in chunks:
                yield self.cache.make_batch(chunk, budget=self.budget, pin_memory=self.pin_memory)
            return
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="compact-prefetch") as ex:
            fut = ex.submit(self.cache.make_batch, chunks[0], budget=self.budget, pin_memory=self.pin_memory)
            for i in range(len(chunks)):
                batch = fut.result()
                if i + 1 < len(chunks):
                    fut = ex.submit(self.cache.make_batch, chunks[i + 1], budget=self.budget, pin_memory=self.pin_memory)
                yield batch


class CompactSampleDataset(Dataset):
    """Small Dataset facade used by single-model startup preflight/debug paths."""

    def __init__(self, cache: CompactExternalCache, *, budget: int, limit: int | None = None) -> None:
        self.cache = cache
        self.budget = int(budget)
        self.count = min(len(cache), int(limit)) if limit is not None and int(limit) > 0 else len(cache)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = self.cache.make_batch(np.asarray([int(idx)], dtype=np.int64), budget=self.budget, pin_memory=False)
        return {k: v[0] for k, v in batch.items()}


def _cli_build(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    build_compact_cache(
        cfg=cfg,
        preprocessed_dir=args.preprocessed_dir,
        split=args.split,
        output_dir=args.output_dir,
        budgets=args.budgets,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        batch_size=args.batch_size,
        max_scenarios=args.max_scenarios,
        max_scenarios_per_split=args.max_scenarios_per_split,
        resume=not args.no_resume,
        log_every=args.log_every,
    )


def _cli_profile(args: argparse.Namespace) -> None:
    cache = CompactExternalCache.open(args.cache_dir)
    loader = CompactBatchLoader(
        cache,
        budget=args.budget,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        pin_memory=args.pin_memory,
        prefetch=not args.no_prefetch,
        shuffle_mode=args.shuffle_mode,
        block_size=args.block_size,
    )
    it = iter(loader)
    warm = min(max(0, args.warmup), max(len(loader) - 1, 0))
    for _ in range(warm):
        next(it)
    n = min(max(1, args.batches), len(loader))
    t0 = time.perf_counter(); samples = 0
    for _ in range(n):
        b = next(it); samples += int(next(iter(b.values())).shape[0])
    elapsed = max(time.perf_counter() - t0, 1e-9)
    print(
        f"[compact-profile] cache={cache.root} budget={args.budget} batches={n} samples={samples} "
        f"batch_rate={n/elapsed:.3f} batch/s sample_rate={samples/elapsed:.1f} sample/s "
        f"shuffle_mode={args.shuffle_mode} prefetch={not args.no_prefetch} pin_memory={args.pin_memory}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/profile persistent compact mmap cache for external baselines")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", required=True)
    b.add_argument("--preprocessed-dir", required=True)
    b.add_argument("--split", nargs="+", required=True)
    b.add_argument("--output-dir", required=True)
    b.add_argument("--budgets", nargs="+", type=int, default=[8, 16, 24])
    b.add_argument("--num-workers", type=int, default=10)
    b.add_argument("--prefetch-factor", type=int, default=4)
    b.add_argument("--batch-size", type=int, default=32)
    b.add_argument("--max-scenarios", type=int, default=None)
    b.add_argument("--max-scenarios-per-split", type=int, default=None)
    b.add_argument("--log-every", type=int, default=100)
    b.add_argument("--no-resume", action="store_true")
    b.set_defaults(func=_cli_build)

    p = sub.add_parser("profile")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--shuffle-mode", choices=["global", "block", "none"], default="global")
    p.add_argument("--block-size", type=int, default=4096)
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--no-prefetch", action="store_true")
    p.add_argument("--pin-memory", action="store_true")
    p.set_defaults(func=_cli_profile)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
