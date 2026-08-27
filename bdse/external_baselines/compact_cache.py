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
import statistics
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
    external_json_backend,
    external_sample_to_model_numpy,
    external_sample_to_model_inputs,
    load_external_training_sample_npz,
    oracle_selected_masks_for_budgets,
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


def _sample_array_dict(
    path: Path,
    cfg: dict[str, Any],
    budgets: Sequence[int],
    *,
    include_label_future: bool,
    npz_read_mode: str = "direct",
    compact_minimal: bool = True,
) -> dict[str, np.ndarray]:
    sample = load_external_training_sample_npz(
        path,
        include_label_future=include_label_future,
        read_mode=npz_read_mode,
        compact_minimal=compact_minimal,
    )
    base_budget = int(budgets[0])
    base_cfg = _cfg_for_budget(cfg, base_budget)
    proposal_enabled = float(((cfg.get("external_baseline", {}) or {}).get("loss_weights", {}) or {}).get("proposal", 0.25)) != 0.0
    tensor_cfg = base_cfg
    if proposal_enabled:
        # Avoid computing B0 once inside the generic NumPy tensorizer and then
        # recomputing all masks below.  Cache construction owns the multi-budget
        # oracle path explicitly.
        tensor_cfg = copy.deepcopy(base_cfg)
        tensor_cfg.setdefault("external_baseline", {}).setdefault("loss_weights", {})["proposal"] = 0.0
    arrays = external_sample_to_model_numpy(sample, tensor_cfg)

    # The scene/candidate/evidence compact contract is budget-independent.  Only
    # the greedy proposal-supervision mask changes with B, so keep one mask per B.
    if proposal_enabled:
        # Reuse one greedy max-budget sequence when the source really has unit
        # evidence costs (the fair protocol default); fall back exactly otherwise.
        masks = oracle_selected_masks_for_budgets(sample, cfg, tuple(int(x) for x in budgets))
        for budget in budgets:
            arrays[f"oracle_selected_mask_B{int(budget)}"] = np.asarray(masks[int(budget)], dtype=bool)
    return arrays


def _sample_tensor_dict(path: Path, cfg: dict[str, Any], budgets: Sequence[int], *, include_label_future: bool) -> dict[str, torch.Tensor]:
    arrays = _sample_array_dict(
        path, cfg, budgets,
        include_label_future=include_label_future,
        npz_read_mode="direct",
        compact_minimal=False,
    )
    out: dict[str, torch.Tensor] = {}
    for name, arr in arrays.items():
        a = np.asarray(arr)
        if a.dtype == np.bool_:
            out[name] = torch.from_numpy(a.astype(np.bool_, copy=False))
        elif np.issubdtype(a.dtype, np.integer):
            out[name] = torch.from_numpy(a.astype(np.int64, copy=False))
        else:
            out[name] = torch.from_numpy(a.astype(np.float32, copy=False))
    return out


class _CompactBuildDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[Path],
        cfg: dict[str, Any],
        budgets: Sequence[int],
        *,
        include_label_future: bool,
        start_index: int = 0,
        end_index: int | None = None,
        fields: Sequence[dict[str, Any]] | None = None,
        widths: dict[str, int] | None = None,
        npz_read_mode: str = "direct",
    ) -> None:
        self.paths = list(paths)
        self.cfg = cfg
        self.budgets = tuple(int(x) for x in budgets)
        self.include_label_future = bool(include_label_future)
        self.start_index = int(start_index)
        self.end_index = len(self.paths) if end_index is None else min(len(self.paths), int(end_index))
        self.fields = list(fields or [])
        self.widths = dict(widths or {})
        self.npz_read_mode = str(npz_read_mode)

    def __len__(self) -> int:
        return max(0, self.end_index - self.start_index)

    def __getitem__(self, local_idx: int):
        idx = self.start_index + int(local_idx)
        arrays = _sample_array_dict(
            self.paths[idx], self.cfg, self.budgets,
            include_label_future=self.include_label_future,
            npz_read_mode=self.npz_read_mode,
            compact_minimal=True,
        )
        if not self.fields:
            return arrays
        packed: dict[str, np.ndarray] = {
            "float32": np.empty((int(self.widths.get("float32", 0)),), dtype=np.float32),
            "int64": np.empty((int(self.widths.get("int64", 0)),), dtype=np.int64),
            "bool": np.empty((int(self.widths.get("bool", 0)),), dtype=np.bool_),
        }
        for f in self.fields:
            name, group = str(f["name"]), str(f["group"])
            off, size = int(f["offset"]), int(f["size"])
            arr = np.asarray(arrays[name]).reshape(-1)
            if group == "float32":
                packed[group][off:off + size] = arr.astype(np.float32, copy=False)
            elif group == "int64":
                packed[group][off:off + size] = arr.astype(np.int64, copy=False)
            else:
                packed[group][off:off + size] = arr.astype(np.bool_, copy=False)
        # Three flat tensors instead of ~15 independent tensors per sample greatly
        # reduces DataLoader collation/IPC metadata during one-time cache creation.
        return tuple(torch.from_numpy(packed[g]) for g in ("float32", "int64", "bool"))


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


def _write_packed_batch(mem: dict[str, np.memmap], batch: Sequence[torch.Tensor], start: int, end: int) -> None:
    """Write the three already-packed dtype groups emitted by build workers."""
    names = ("float32", "int64", "bool")
    bsz = int(end - start)
    for group, tensor in zip(names, batch):
        if group not in mem:
            continue
        t = tensor.detach().cpu()
        if group == "float32":
            arr = t.numpy().astype(np.float32, copy=False)
        elif group == "int64":
            arr = t.numpy().astype(np.int64, copy=False)
        else:
            arr = t.numpy().astype(np.bool_, copy=False)
        mem[group][start:end, :] = arr.reshape(bsz, mem[group].shape[1])


def _system_ram_gib() -> float:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return float(pages * page_size) / float(1024**3)
    except Exception:
        return float("nan")


def _loader_probe_rate(
    *,
    paths: Sequence[Path],
    cfg: dict[str, Any],
    budgets: Sequence[int],
    include_label_future: bool,
    fields: Sequence[dict[str, Any]],
    widths: dict[str, int],
    start_index: int,
    samples: int,
    workers: int,
    prefetch_factor: int,
    batch_size: int,
    npz_read_mode: str,
) -> float:
    end_index = min(len(paths), int(start_index) + max(1, int(samples)))
    if end_index <= start_index:
        return 0.0
    ds = _CompactBuildDataset(
        paths, cfg, budgets,
        include_label_future=include_label_future,
        start_index=start_index,
        end_index=end_index,
        fields=fields,
        widths=widths,
        npz_read_mode=npz_read_mode,
    )
    kwargs: dict[str, Any] = dict(
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        persistent_workers=False,
        worker_init_fn=_seed_worker,
    )
    if int(workers) > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = DataLoader(ds, **kwargs)
    it = iter(loader)
    total = 0
    # Exclude worker process startup / first-file cold metadata from steady-state
    # throughput; those costs are negligible over a 364k-sample build.
    try:
        first = next(it)
    except StopIteration:
        return 0.0
    total += int(first[0].shape[0])
    t0 = time.perf_counter()
    for batch in it:
        total += int(batch[0].shape[0])
    elapsed = max(time.perf_counter() - t0, 1e-9)
    timed = max(total - int(first[0].shape[0]), 0)
    return float(timed) / elapsed if timed > 0 else 0.0


def _autotune_build_pipeline(
    *,
    paths: Sequence[Path],
    cfg: dict[str, Any],
    budgets: Sequence[int],
    include_label_future: bool,
    fields: Sequence[dict[str, Any]],
    widths: dict[str, int],
    start_index: int,
    requested_workers: int,
    max_workers: int,
    prefetch_factor: int,
    batch_size: int,
    npz_read_mode: str,
    probe_samples: int,
) -> tuple[int, str]:
    """Benchmark target-host NPZ mode and worker count on short disjoint windows."""
    cpu = max(1, int(os.cpu_count() or 1))
    cap = min(cpu, max(1, int(max_workers)))
    ram_gib = _system_ram_gib()
    # Each torch DataLoader process imports NumPy/PyTorch and can consume a few
    # hundred MiB.  Avoid auto-selecting dozens of workers on a low-RAM host.
    if math.isfinite(ram_gib):
        cap = min(cap, max(2, int(ram_gib // 0.75)))
    requested = max(1, min(cap, int(requested_workers)))
    probe_n = max(batch_size * 3, int(probe_samples))
    cursor = int(start_index)

    modes = [str(npz_read_mode).strip().lower()]
    if modes[0] == "auto":
        modes = ["direct", "bytes"]
    mode_rates: dict[str, float] = {}
    mode_workers = min(requested, cap, 8)
    for mode in modes:
        if cursor + probe_n >= len(paths):
            cursor = max(0, start_index)
        rate = _loader_probe_rate(
            paths=paths, cfg=cfg, budgets=budgets, include_label_future=include_label_future,
            fields=fields, widths=widths, start_index=cursor, samples=probe_n,
            workers=mode_workers, prefetch_factor=prefetch_factor, batch_size=batch_size,
            npz_read_mode=mode,
        )
        cursor += probe_n
        mode_rates[mode] = rate
        print(f"[compact-autotune] read_mode={mode} workers={mode_workers} rate={rate:.1f} sample/s", flush=True)
    chosen_mode = max(mode_rates, key=mode_rates.get)

    candidates = sorted({
        max(1, min(cap, requested)),
        max(1, min(cap, 8)),
        max(1, min(cap, 12)),
        max(1, min(cap, 16)),
        max(1, min(cap, 24)),
        max(1, min(cap, 32)),
    })
    worker_rates: dict[int, float] = {}
    for workers in candidates:
        if cursor + probe_n >= len(paths):
            cursor = max(0, start_index)
        rate = _loader_probe_rate(
            paths=paths, cfg=cfg, budgets=budgets, include_label_future=include_label_future,
            fields=fields, widths=widths, start_index=cursor, samples=probe_n,
            workers=workers, prefetch_factor=prefetch_factor, batch_size=batch_size,
            npz_read_mode=chosen_mode,
        )
        cursor += probe_n
        worker_rates[workers] = rate
        print(f"[compact-autotune] read_mode={chosen_mode} workers={workers} rate={rate:.1f} sample/s", flush=True)

    best_workers = max(worker_rates, key=worker_rates.get)
    best_rate = worker_rates[best_workers]
    # Prefer fewer workers when rates are effectively tied; this lowers RAM and
    # metadata pressure without sacrificing meaningful throughput.
    near = [w for w, r in worker_rates.items() if r >= 0.97 * best_rate]
    if near:
        best_workers = min(near)
    print(
        f"[compact-autotune] SELECT read_mode={chosen_mode} workers={best_workers} "
        f"probe_best={best_rate:.1f} sample/s cpu={cpu} ram_gib={ram_gib:.1f}",
        flush=True,
    )
    return int(best_workers), str(chosen_mode)


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
    npz_read_mode: str = "auto",
    autotune: bool = False,
    autotune_max_workers: int = 32,
    autotune_probe_samples: int = 128,
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

    effective_workers = max(0, int(num_workers))
    effective_read_mode = str(npz_read_mode).strip().lower()
    if effective_read_mode not in {"auto", "direct", "bytes"}:
        raise ValueError("npz_read_mode must be auto|direct|bytes")
    if autotune and start_index < count:
        effective_workers, effective_read_mode = _autotune_build_pipeline(
            paths=paths,
            cfg=cfg,
            budgets=budgets,
            include_label_future=include_label_future,
            fields=fields,
            widths=widths,
            start_index=start_index,
            requested_workers=max(1, effective_workers),
            max_workers=autotune_max_workers,
            prefetch_factor=prefetch_factor,
            batch_size=batch_size,
            npz_read_mode=effective_read_mode,
            probe_samples=autotune_probe_samples,
        )
    elif effective_read_mode == "auto":
        effective_read_mode = "direct"

    build_ds = _CompactBuildDataset(
        paths, cfg, budgets,
        include_label_future=include_label_future,
        start_index=start_index,
        fields=fields,
        widths=widths,
        npz_read_mode=effective_read_mode,
    )
    kwargs: dict[str, Any] = dict(
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(effective_workers)),
        pin_memory=False,
        persistent_workers=int(effective_workers) > 0,
        worker_init_fn=_seed_worker,
    )
    if int(effective_workers) > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = DataLoader(build_ds, **kwargs)

    t0 = time.perf_counter()
    written = start_index
    print(
        f"[compact-cache] BUILD path={output} samples={count} start={start_index} budgets={list(budgets)} "
        f"batch={batch_size} workers={effective_workers} npz_read_mode={effective_read_mode} json={external_json_backend()} float_width={widths.get('float32',0)} "
        f"int_width={widths.get('int64',0)} bool_width={widths.get('bool',0)} estimated_gib={estimated_bytes/(1024**3):.2f}",
        flush=True,
    )
    for step, batch in enumerate(loader, start=1):
        bsz = int(batch[0].shape[0])
        end = written + bsz
        _write_packed_batch(mem, batch, written, end)
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


class CompactIndexLoader:
    """Yield only shuffled row indices for a device-resident compact cache."""

    def __init__(
        self,
        count: int,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        shuffle_mode: str = "global",
        block_size: int = 4096,
        limit: int | None = None,
    ) -> None:
        self.count = min(int(count), int(limit)) if limit is not None and int(limit) > 0 else int(count)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.shuffle_mode = str(shuffle_mode).strip().lower()
        if self.shuffle_mode not in {"global", "block", "none"}:
            raise ValueError("compact shuffle_mode must be one of: global, block, none")
        self.block_size = max(self.batch_size, int(block_size))
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
        blocks: list[np.ndarray] = []
        for start in range(0, self.count, self.block_size):
            block = idx[start:min(start + self.block_size, self.count)].copy()
            rng.shuffle(block)
            blocks.append(block)
        order = np.arange(len(blocks)); rng.shuffle(order)
        return np.concatenate([blocks[int(i)] for i in order], axis=0)

    def __iter__(self) -> Iterator[np.ndarray]:
        epoch = self._epoch; self._epoch += 1
        idx = self._epoch_indices(epoch)
        for start in range(0, self.count, self.batch_size):
            yield idx[start:min(start + self.batch_size, self.count)]


@dataclass
class DeviceCompactExternalCache:
    """Fully device-resident copy of the compact tensor contract.

    This is intentionally a cache of *already parsed numeric tensors*, never raw
    NPZ bytes.  It removes mmap/random-storage latency and per-step H2D copies.
    """

    manifest: dict[str, Any]
    device: torch.device
    groups: dict[str, torch.Tensor]
    source_root: Path

    @staticmethod
    def _torch_dtype(group: str, float_dtype: str) -> torch.dtype:
        if group == "float32":
            return torch.float16 if str(float_dtype).lower() == "float16" else torch.float32
        if group == "int64":
            return torch.int64
        if group == "bool":
            return torch.bool
        raise KeyError(group)

    @classmethod
    def estimated_bytes(cls, cache: CompactExternalCache, *, float_dtype: str = "float32") -> int:
        count = len(cache); widths = cache.manifest["widths"]
        fbytes = 2 if str(float_dtype).lower() == "float16" else 4
        return int(count) * (
            int(widths.get("float32", 0)) * fbytes
            + int(widths.get("int64", 0)) * 8
            + int(widths.get("bool", 0))
        )

    @classmethod
    def load(
        cls,
        cache: CompactExternalCache,
        device: torch.device,
        *,
        float_dtype: str = "float32",
        reserve_gib: float = 8.0,
        mode: str = "auto",
        chunk_rows: int = 4096,
    ) -> "DeviceCompactExternalCache | None":
        if device.type != "cuda":
            if str(mode).lower() == "on":
                raise ValueError("device compact cache requires CUDA")
            return None
        float_dtype = str(float_dtype).lower()
        if float_dtype not in {"float32", "float16"}:
            raise ValueError("compact device float dtype must be float32|float16")
        mode_l = str(mode).lower()
        if mode_l not in {"off", "auto", "on"}:
            raise ValueError("compact device cache mode must be off|auto|on")
        if mode_l == "off":
            return None
        need = cls.estimated_bytes(cache, float_dtype=float_dtype)
        with torch.cuda.device(device):
            free_b, total_b = torch.cuda.mem_get_info()
        reserve_b = int(float(reserve_gib) * (1024**3))
        fits = need + reserve_b <= int(free_b)
        print(
            f"[compact-device-cache-check] device={device} mode={mode_l} dtype={float_dtype} "
            f"need_gib={need/(1024**3):.2f} free_gib={free_b/(1024**3):.2f} "
            f"total_gib={total_b/(1024**3):.2f} reserve_gib={reserve_gib:.2f} fits={fits}",
            flush=True,
        )
        if not fits:
            if mode_l == "on":
                raise RuntimeError(
                    f"compact device cache does not fit on {device}: need={need/(1024**3):.2f} GiB "
                    f"free={free_b/(1024**3):.2f} GiB reserve={reserve_gib:.2f} GiB"
                )
            return None

        mm_by_group = {"float32": cache.float_mm, "int64": cache.int_mm, "bool": cache.bool_mm}
        groups: dict[str, torch.Tensor] = {}
        t0 = time.perf_counter(); copied = 0
        rows = max(1, int(chunk_rows))
        for group, mm in mm_by_group.items():
            if mm is None:
                continue
            dtype = cls._torch_dtype(group, float_dtype)
            dst = torch.empty(tuple(mm.shape), dtype=dtype, device=device)
            for start in range(0, len(cache), rows):
                end = min(start + rows, len(cache))
                # Writable host staging avoids PyTorch's read-only mmap warning and
                # bounds temporary RAM while the full destination stays on-device.
                host_np = np.array(mm[start:end], copy=True)
                host = torch.from_numpy(host_np)
                dst[start:end].copy_(host.to(device=device, dtype=dtype, non_blocking=False))
                copied += int(host_np.nbytes)
            groups[group] = dst
        elapsed = max(time.perf_counter() - t0, 1e-9)
        print(
            f"[compact-device-cache-ready] device={device} dtype={float_dtype} "
            f"resident_gib={sum(t.numel()*t.element_size() for t in groups.values())/(1024**3):.2f} "
            f"source_read_gib={copied/(1024**3):.2f} load_s={elapsed:.1f} read_gib_s={copied/(1024**3)/elapsed:.2f}",
            flush=True,
        )
        return cls(manifest=cache.manifest, device=device, groups=groups, source_root=cache.root)

    def make_batch(self, indices: np.ndarray, *, budget: int) -> dict[str, torch.Tensor]:
        budget = int(budget)
        budgets = tuple(int(x) for x in self.manifest.get("budgets", []))
        if budget not in budgets:
            raise ValueError(f"budget B={budget} not present in device compact cache; available={budgets}")
        idx = torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long, device=self.device)
        gathered = {group: tensor.index_select(0, idx) for group, tensor in self.groups.items()}
        bsz = int(idx.numel())
        out: dict[str, torch.Tensor] = {}
        oracle_name = f"oracle_selected_mask_B{budget}"
        for f in self.manifest["fields"]:
            stored_name = str(f["name"])
            if stored_name.startswith("oracle_selected_mask_B") and stored_name != oracle_name:
                continue
            public_name = "oracle_selected_mask" if stored_name == oracle_name else stored_name
            group = str(f["group"]); off = int(f["offset"]); size = int(f["size"])
            shape = tuple(int(x) for x in f["shape"])
            base = gathered[group][:, off:off + size]
            out[public_name] = base.reshape((bsz,) + shape) if shape else base.reshape(bsz)
        return out


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
        npz_read_mode=args.npz_read_mode,
        autotune=not args.no_autotune,
        autotune_max_workers=args.autotune_max_workers,
        autotune_probe_samples=args.autotune_probe_samples,
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


def _cli_diagnose(args: argparse.Namespace) -> None:
    """Break raw-cache construction into NPZ/JSON, feature, and oracle stages."""
    cfg = load_config(args.config)
    source = PreprocessedBDSEDataset(args.preprocessed_dir, split=list(args.split))
    paths = [Path(p) for p in source.build_index()]
    if not paths:
        raise RuntimeError("no raw samples for compact diagnose")
    n = min(max(1, int(args.samples)), len(paths))
    # Spread probes across the ordered cache so one unusually easy log/city does
    # not dominate the diagnosis while still opening only N samples.
    ids = np.linspace(0, len(paths) - 1, num=n, dtype=np.int64)
    probe_paths = [paths[int(i)] for i in ids]
    sizes = [p.stat().st_size for p in probe_paths]
    for mode in args.read_modes:
        load_s: list[float] = []; core_s: list[float] = []; oracle_s: list[float] = []
        for path in probe_paths:
            t0 = time.perf_counter()
            sample = load_external_training_sample_npz(
                path,
                include_label_future=_planner_supervision(cfg) == "expert_imitation",
                read_mode=mode,
                compact_minimal=True,
            )
            t1 = time.perf_counter()
            core_cfg = copy.deepcopy(cfg)
            core_cfg.setdefault("external_baseline", {}).setdefault("loss_weights", {})["proposal"] = 0.0
            external_sample_to_model_numpy(sample, core_cfg)
            t2 = time.perf_counter()
            for budget in args.budgets:
                _oracle_selected_mask(sample, _cfg_for_budget(cfg, int(budget)))
            t3 = time.perf_counter()
            load_s.append(t1 - t0); core_s.append(t2 - t1); oracle_s.append(t3 - t2)
        def ms(v: Sequence[float]) -> float:
            return 1000.0 * float(statistics.median(v))
        total_med = ms([a + b + c for a, b, c in zip(load_s, core_s, oracle_s)])
        print(
            f"[compact-diagnose] read_mode={mode} samples={n} raw_file_mib_median={statistics.median(sizes)/(1024**2):.2f} "
            f"npz_json_ms={ms(load_s):.1f} feature_target_ms={ms(core_s):.1f} "
            f"oracle_Bs_ms={ms(oracle_s):.1f} total_ms={total_med:.1f} budgets={list(args.budgets)} "
            f"json_backend={external_json_backend()}",
            flush=True,
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
    b.add_argument("--npz-read-mode", choices=["auto", "direct", "bytes"], default="auto")
    b.add_argument("--no-autotune", action="store_true")
    b.add_argument("--autotune-max-workers", type=int, default=32)
    b.add_argument("--autotune-probe-samples", type=int, default=128)
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

    d = sub.add_parser("diagnose")
    d.add_argument("--config", required=True)
    d.add_argument("--preprocessed-dir", required=True)
    d.add_argument("--split", nargs="+", required=True)
    d.add_argument("--budgets", nargs="+", type=int, default=[8, 16, 24])
    d.add_argument("--samples", type=int, default=16)
    d.add_argument("--read-modes", nargs="+", choices=["direct", "bytes"], default=["direct", "bytes"])
    d.set_defaults(func=_cli_diagnose)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
