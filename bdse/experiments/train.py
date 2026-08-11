from __future__ import annotations

import argparse
import hashlib
from contextlib import nullcontext
import os
import re
import time
from pathlib import Path
from typing import Any

import json
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.cache_schema import Sample, load_sample_npz
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, PreprocessedBDSEDataset
from bdse.data.tensorizer import sample_to_model_inputs
from bdse.data.quality import quality_decision
from bdse.experiments.evaluate_open_loop import (
    add_dense_bridge_diagnostics,
    _criticality_metrics,
    _frozen_family_slot_oracle_critical_recall,
)
from bdse.metrics.bdse_metrics import compute_bdse_diagnostics
from bdse.model.bdse_model import BDSEModel
from bdse.model.losses import compute_bdse_losses
from bdse.planner.nuplan_planner import BDSEPlannerCore, runtime_query_diagnostics
from bdse.planner.hab import select_topm_atoms_hab
from bdse.planner.fallback import runtime_safety_flags_from_runtime
from bdse.planner.tournament import run_pair_conditioned_tournament


class OnTheFlyDataset(Dataset):
    def __init__(self, source: NuPlanBDSEDataset | PreprocessedBDSEDataset):
        self.source = source
        self._preprocessed_paths: list[Path] | None = None
        if isinstance(source, PreprocessedBDSEDataset):
            # Resolve the index once in the parent process.  DataLoader workers
            # then receive a compact path list instead of repeatedly calling
            # build_index() for every sample.  build_index() is cached, but the
            # per-sample method call and split bookkeeping still show up in long
            # 50k-sample training runs.
            self._preprocessed_paths = list(source.build_index())

    def __len__(self) -> int:
        if self._preprocessed_paths is not None:
            return len(self._preprocessed_paths)
        return len(self.source)

    def __getitem__(self, idx: int) -> Sample:
        if self._preprocessed_paths is not None:
            # Training does not consume label-only futures or candidate JSON metadata.
            # Avoid moving those large unused arrays through DataLoader workers.
            return load_sample_npz(
                self._preprocessed_paths[idx],
                include_label_future=False,
                include_candidate_metadata=False,
                include_runtime_metadata=False,
                include_route_ids=False,
                include_evidence_aux_metadata=False,
                allow_pickle=False,
            )
        return self.source[idx]


class ResumableBatchSampler:
    """Skip completed DDP batches without loading or tensorizing them again.

    ``DistributedSampler.set_epoch`` deterministically reconstructs the same
    sample order after a restart.  Applying the offset at the batch-sampler
    layer therefore resumes at the exact next batch, whereas a ``continue`` in
    the training loop still makes DataLoader workers read, decode and collate
    every already-completed sample.
    """

    def __init__(self, batch_sampler: BatchSampler):
        self.batch_sampler = batch_sampler
        self.start_batch = 0

    @property
    def total_batches(self) -> int:
        return len(self.batch_sampler)

    def set_start_batch(self, start_batch: int) -> None:
        self.start_batch = min(max(0, int(start_batch)), self.total_batches)

    def __iter__(self):
        for batch_index, indices in enumerate(self.batch_sampler):
            if batch_index >= self.start_batch:
                yield indices

    def __len__(self) -> int:
        return max(0, self.total_batches - self.start_batch)


def sample_to_tensors(sample: Sample, cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    # Unified tensorizer is shared by training and deployment to avoid train/deploy
    # feature skew.  It intentionally does not create a teacher-derived
    # runtime_selected_mask; L_act builds its certificate through predicted
    # proposal/greedy/tournament inside the loss.
    return sample_to_model_inputs(sample, cfg, include_teacher=True, include_dense_query=True)

def collate(samples: list[Sample], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    items = [sample_to_tensors(s, cfg) for s in samples]
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}


_PAIR_ALIGNED_BATCH_KEYS = (
    "pair_indices",
    "pair_valid",
    "pair_margins",
    "pair_weights",
    "pair_residuals",
)


def _boundary_pair_sampler_uses_full_graph(train_cfg: dict[str, Any]) -> bool:
    """Return whether the current step must keep the complete cached pair graph.

    V52 uses a boundary-focused pair curriculum on most optimizer steps, but the
    exact AOCC supervision steps and the final alignment tail always see the full
    graph.  This keeps deployment supervision exact where it is applied while
    avoiding dense E x P pair-head work on every other step.
    """
    sampler_cfg = train_cfg.get("boundary_pair_sampler", {}) or {}
    if not bool(sampler_cfg.get("enabled", False)):
        return True
    step = int(train_cfg.get("global_step", 0))
    cadence = max(1, int(sampler_cfg.get("full_every_n_steps", 1)))
    if step % cadence == 0:
        return True
    full_last_steps = max(0, int(sampler_cfg.get("full_last_n_steps", 0)))
    if full_last_steps <= 0:
        return False
    epoch = int(train_cfg.get("current_epoch", 0))
    total_epochs = max(1, int(train_cfg.get("epochs", epoch + 1)))
    if epoch != total_epochs - 1:
        return False
    steps_per_epoch = max(1, int(train_cfg.get("steps_per_epoch", 1)))
    step_in_epoch = step % steps_per_epoch
    return step_in_epoch >= max(0, steps_per_epoch - full_last_steps)


def _boundary_focused_pair_subsample(
    batch: dict[str, torch.Tensor], cfg: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Keep the decision-critical pair subset used by BFAR-DBAP training.

    Priority is assigned to teacher-winner rivals, hard-feasibility crossings,
    near-tie margins, and high pair weights.  The sampler is deterministic and
    only changes training compute; cached labels and deployment-time pair graph
    construction remain untouched.  Full-graph steps are synchronized with the
    exact selector cadence through ``full_every_n_steps``.
    """
    train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    sampler_cfg = train_cfg.get("boundary_pair_sampler", {}) or {}
    if not bool(sampler_cfg.get("enabled", False)) or _boundary_pair_sampler_uses_full_graph(train_cfg):
        if "pair_valid" in batch:
            valid_count = batch["pair_valid"].bool().sum(dim=1).float()
            batch["training_pair_original_count"] = valid_count
            batch["training_pair_selected_count"] = valid_count
            batch["training_pair_fraction"] = torch.ones_like(valid_count)
            batch["training_pair_full_graph"] = torch.ones_like(valid_count)
        return batch
    if "pair_indices" not in batch or "pair_valid" not in batch:
        return batch

    pairs = batch["pair_indices"].long()
    pair_valid = batch["pair_valid"].bool()
    if pairs.ndim != 3 or pairs.shape[-1] != 2:
        return batch
    B, P, _ = pairs.shape
    max_pairs = max(1, int(sampler_cfg.get("max_pairs", P)))
    if max_pairs >= P:
        valid_count = pair_valid.sum(dim=1).float()
        batch["training_pair_original_count"] = valid_count
        batch["training_pair_selected_count"] = valid_count
        batch["training_pair_fraction"] = torch.ones_like(valid_count)
        batch["training_pair_full_graph"] = torch.ones_like(valid_count)
        return batch

    a = pairs[..., 0]
    b = pairs[..., 1]
    target = batch.get("teacher_a_star")
    if target is None:
        winner_pair = torch.zeros_like(pair_valid)
        target = torch.zeros((B, 1), dtype=torch.long, device=pairs.device)
    else:
        target = target.long().reshape(B, 1)
        winner_pair = (a == target) | (b == target)

    # V64.3.6 LBPR pair curriculum: reserve pair-batch capacity for the exact
    # teacher winner -> leave-one-atom-out flip boundaries.  This uses labels
    # only while constructing the offline training batch; runtime pair graphs
    # remain deployment-available and unchanged.
    literal_boundary_pair = torch.zeros_like(pair_valid)
    if int(sampler_cfg.get("literal_boundary_quota", 0)) > 0:
        teacher_cost = batch.get("teacher_J_T")
        teacher_g = batch.get("teacher_g_evid")
        evidence_active = batch.get("evidence_active")
        candidate_valid = batch.get("candidate_valid")
        if teacher_cost is not None and teacher_g is not None and evidence_active is not None and candidate_valid is not None:
            valid_action = candidate_valid.bool()
            invalid = teacher_cost.new_tensor(1.0e9)
            dense_teacher = teacher_cost.float().masked_fill(~valid_action, invalid)
            scalar_winner = dense_teacher.argmin(dim=1)
            aligned = scalar_winner[:, None].eq(target)
            active_e = evidence_active.bool()
            loo = dense_teacher[:, None, :] - teacher_g.float() * active_e[:, :, None].float()
            loo = loo.masked_fill(~valid_action[:, None, :], invalid)
            loo_winner = loo.argmin(dim=2)
            critical = active_e & aligned & loo_winner.ne(target)
            match_ab = (a == target) & (critical[:, :, None] & loo_winner[:, :, None].eq(b[:, None, :])).any(dim=1)
            match_ba = (b == target) & (critical[:, :, None] & loo_winner[:, :, None].eq(a[:, None, :])).any(dim=1)
            literal_boundary_pair = pair_valid & (match_ab | match_ba)

    hard = batch.get("teacher_hard_violation")
    if hard is None:
        hard_cross = torch.zeros_like(pair_valid)
    else:
        hard = hard.bool()
        K = hard.shape[1]
        hard_a = torch.gather(hard, 1, a.clamp(0, K - 1))
        hard_b = torch.gather(hard, 1, b.clamp(0, K - 1))
        hard_cross = hard_a ^ hard_b

    margins = batch.get("pair_margins")
    if margins is None:
        abs_margin = torch.zeros_like(pair_valid, dtype=torch.float32)
    else:
        abs_margin = margins.float().abs()
    inf = torch.full_like(abs_margin, float("inf"))
    valid_abs = torch.where(pair_valid, abs_margin, inf)
    sorted_abs = torch.sort(valid_abs, dim=1).values
    valid_count_int = pair_valid.sum(dim=1).clamp_min(1)
    median_index = ((valid_count_int - 1) // 2).reshape(B, 1)
    margin_scale = torch.gather(sorted_abs, 1, median_index).clamp_min(
        float(sampler_cfg.get("min_margin_scale", 1.0))
    )
    normalized_abs = abs_margin / margin_scale
    near_tau = max(float(sampler_cfg.get("near_tie_tau", 0.5)), 1e-6)
    near_score = 1.0 / (1.0 + normalized_abs / near_tau)

    weights = batch.get("pair_weights")
    if weights is None:
        weight_score = torch.zeros_like(abs_margin)
    else:
        w = weights.float().clamp_min(0.0)
        wmax = torch.where(pair_valid, w, torch.zeros_like(w)).max(dim=1, keepdim=True).values.clamp_min(1e-6)
        weight_score = w / wmax

    score = (
        float(sampler_cfg.get("literal_boundary_bonus", 32.0)) * literal_boundary_pair.float()
        + float(sampler_cfg.get("winner_bonus", 16.0)) * winner_pair.float()
        + float(sampler_cfg.get("hard_cross_bonus", 10.0)) * hard_cross.float()
        + float(sampler_cfg.get("near_tie_bonus", 6.0)) * near_score
        + float(sampler_cfg.get("pair_weight_bonus", 3.0)) * weight_score
    )
    # Weighted top-k alone can let one abundant category (typically hard-crossing
    # pairs) evict all near-boundary pairs.  BFAR reserves overlapping quotas for
    # winner, hard-crossing, and near-tie pairs, then fills by the joint score.
    #
    # V64.3.4 execution optimization: the previous implementation looped over
    # every local-batch row and repeatedly called .item()/nonzero/topk *after*
    # H2D.  On CUDA those scalar reads serialize the stream and pair sampling was
    # 15--23% of epoch time in V64.3.3.  The batched quota updates below are the
    # same deterministic selection rule, but require only four small batched
    # top-k operations and no host-device synchronization.  Training targets,
    # quotas, full-graph cadence, and deployment behavior are unchanged.
    index_preference = (P - torch.arange(P, device=score.device, dtype=score.dtype)) * 1e-7
    selected = torch.zeros((B, P), dtype=torch.bool, device=score.device)
    neg_rank = torch.finfo(score.dtype).min

    def _batched_take(candidate_mask: torch.Tensor, rank_values: torch.Tensor, raw_quota: int) -> None:
        nonlocal selected
        quota = min(max(int(raw_quota), 0), max_pairs, P)
        if quota <= 0:
            return
        available_capacity = (max_pairs - selected.sum(dim=1)).clamp_min(0)
        candidate_count = candidate_mask.sum(dim=1)
        take_count = torch.minimum(
            available_capacity,
            torch.minimum(candidate_count, torch.full_like(candidate_count, quota)),
        )
        ranked = (rank_values + index_preference[None, :]).masked_fill(~candidate_mask, neg_rank)
        ids = torch.topk(ranked, k=quota, dim=1, largest=True, sorted=True).indices
        slots = torch.arange(quota, device=score.device)[None, :] < take_count[:, None]
        selected.scatter_(1, ids, selected.gather(1, ids) | slots)

    _batched_take(
        literal_boundary_pair & pair_valid & ~selected,
        score,
        int(sampler_cfg.get("literal_boundary_quota", 0)),
    )
    _batched_take(
        winner_pair & pair_valid & ~selected,
        score,
        int(sampler_cfg.get("winner_quota", max_pairs // 4)),
    )
    _batched_take(
        hard_cross & pair_valid & ~selected,
        score,
        int(sampler_cfg.get("hard_cross_quota", max_pairs // 4)),
    )
    # Near-tie quota ranks all remaining valid pairs by boundary proximity, just
    # like the row-wise implementation; joint score is only a semantic tie-break.
    _batched_take(
        pair_valid & ~selected,
        near_score + 1e-3 * score,
        int(sampler_cfg.get("near_tie_quota", max_pairs // 3)),
    )
    # Fill the remaining valid capacity by the joint score.
    _batched_take(pair_valid & ~selected, score, max_pairs)
    # Rare caches can contain fewer than max_pairs valid pairs.  Preserve the
    # fixed gathered tensor shape by padding with the earliest unused source
    # slots; their gathered pair_valid remains false and contributes zero.
    pad_count = (max_pairs - selected.sum(dim=1)).clamp_min(0)
    if bool((pad_count > 0).any()):
        pad_rank = index_preference[None, :].expand(B, -1).masked_fill(selected, neg_rank)
        pad_ids = torch.topk(pad_rank, k=max_pairs, dim=1, largest=True, sorted=True).indices
        pad_slots = torch.arange(max_pairs, device=score.device)[None, :] < pad_count[:, None]
        selected.scatter_(1, pad_ids, selected.gather(1, pad_ids) | pad_slots)

    source_ids = torch.arange(P, device=score.device, dtype=torch.long)[None, :].expand(B, -1)
    chosen = torch.where(selected, source_ids, torch.full_like(source_ids, P))
    chosen = torch.sort(chosen, dim=1).values[:, :max_pairs]

    for key in _PAIR_ALIGNED_BATCH_KEYS:
        value = batch.get(key)
        if value is None or value.ndim < 2 or value.shape[0] != B or value.shape[1] != P:
            continue
        gather_shape = [B, max_pairs] + [1] * (value.ndim - 2)
        gather_index = chosen.reshape(gather_shape).expand(B, max_pairs, *value.shape[2:])
        batch[key] = torch.gather(value, 1, gather_index)

    original_count = pair_valid.sum(dim=1).float()
    selected_count = batch["pair_valid"].bool().sum(dim=1).float()
    batch["training_pair_original_count"] = original_count
    batch["training_pair_selected_count"] = selected_count
    batch["training_pair_fraction"] = selected_count / original_count.clamp_min(1.0)
    batch["training_pair_full_graph"] = torch.zeros_like(selected_count)
    return batch



def _json_loads_npz_scalar(z: Any, key: str, default: Any) -> Any:
    if key not in z.files:
        return default
    try:
        raw = z[key]
        text = str(raw.item()) if raw.shape == () else str(raw.tolist())
        return json.loads(text)
    except Exception:
        return default


def _quality_metrics_from_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        diag = _json_loads_npz_scalar(z, "teacher_diagnostics_json", {})
        if not isinstance(diag, dict):
            diag = {}
        out: dict[str, Any] = {}
        for k, v in diag.items():
            if str(k).startswith("quality_"):
                out[str(k)[len("quality_"):]] = v
        # Backward compatibility for old caches that do not contain quality_ keys.
        if "safe_candidate_exists" not in out and "teacher_hard_violation" in z.files and "candidate_valid" in z.files:
            valid = np.asarray(z["candidate_valid"], dtype=bool)
            hard = np.asarray(z["teacher_hard_violation"], dtype=bool)
            out["valid_candidate_count"] = int(valid.sum())
            out["safe_candidate_count"] = int((valid & ~hard).sum()) if hard.shape == valid.shape else 0
            out["safe_candidate_exists"] = bool(out["safe_candidate_count"] > 0)
        return out


def _apply_training_quality_filter(dataset: Any, cfg: dict[str, Any]) -> None:
    qcfg = cfg.get("training", {}).get("quality_filter", {})
    if not bool(qcfg.get("enabled", False)):
        return
    if not isinstance(dataset, PreprocessedBDSEDataset):
        print("[bdse] training quality_filter is enabled but only applies to preprocessed caches; continuing without filtering.", flush=True)
        return
    paths = list(dataset.build_index())
    kept: list[Path] = []
    dropped: dict[str, int] = {}
    for p in paths:
        try:
            metrics = _quality_metrics_from_npz(Path(p))
            dec = quality_decision(metrics, cfg)
        except Exception as exc:
            if bool(qcfg.get("drop_unreadable", True)):
                dropped[type(exc).__name__] = dropped.get(type(exc).__name__, 0) + 1
                continue
            kept.append(Path(p)); continue
        if dec.keep:
            kept.append(Path(p))
        else:
            for r in dec.reasons:
                dropped[r] = dropped.get(r, 0) + 1
    if not kept:
        raise RuntimeError(f"training quality_filter dropped all {len(paths)} samples; relax thresholds. dropped={dropped}")
    dataset._paths = kept
    print(f"[bdse] training quality_filter: kept={len(kept)} dropped={len(paths)-len(kept)} reasons={dropped}", flush=True)




def _checkpoint_stem(output: str | Path) -> Path:
    out_path = Path(output)
    return out_path.with_suffix("") if out_path.suffix else out_path


def _sanitize_best_label(label: str) -> str:
    text = str(label or "metric").strip()
    if text.startswith("val_"):
        text = text[4:]
    if text.startswith("best_"):
        text = text[5:]
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "metric"


def _checkpoint_paths(args: argparse.Namespace, epoch: int | None = None) -> dict[str, Path]:
    stem = _checkpoint_stem(args.output)
    ckpt_dir = Path(args.checkpoint_dir) if getattr(args, "checkpoint_dir", None) else stem.parent / "checkpoints"
    paths = {
        "final": Path(args.output),
        "latest": stem.parent / f"{stem.name}.latest.pt",
        "best": stem.parent / f"{stem.name}.best.pt",
    }
    if epoch is not None:
        # epoch is zero-based internally; filenames are one-based for readability.
        paths["epoch"] = ckpt_dir / f"{stem.name}.epoch_{epoch + 1:04d}.pt"
    return paths


def _best_checkpoint_path(args: argparse.Namespace, label: str) -> Path:
    stem = _checkpoint_stem(args.output)
    return stem.parent / f"{stem.name}.best_{_sanitize_best_label(label)}.pt"



def _torch_load_any(path: str | Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _torch_save_atomic(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Any) -> None:
    if not isinstance(state, dict):
        return
    try:
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:
        print(f"[bdse] warning: failed to restore RNG state: {type(exc).__name__}: {exc}", flush=True)


def _training_source_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    rels = (
        "bdse/model/bdse_model.py",
        "bdse/model/losses.py",
        "bdse/experiments/train.py",
    )
    out: dict[str, str] = {}
    for rel in rels:
        path = root / rel
        if path.is_file():
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _make_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
    best_metric: float | None,
    best_epoch: int | None,
    world_size: int,
    best_trackers: dict[str, dict[str, Any]] | None = None,
    next_batch_index: int = 0,
) -> dict[str, Any]:
    raw_model = model.module if isinstance(model, DDP) else model
    return {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if hasattr(scaler, "state_dict") else {},
        "cfg": cfg,
        "args": vars(args),
        "epoch": int(epoch),
        "next_epoch": int(epoch) if int(next_batch_index) > 0 else int(epoch) + 1,
        "next_batch_index": max(0, int(next_batch_index)),
        "metrics": {str(k): float(v) for k, v in metrics.items()},
        "best_metric": None if best_metric is None else float(best_metric),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_trackers": best_trackers or {},
        "world_size": int(world_size),
        "rng_state": _rng_state(),
        "source_sha256": _training_source_sha256(),
    }


def _load_checkpoint_if_requested(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    is_main: bool,
) -> tuple[int, int, float | None, int | None, dict[str, dict[str, Any]], Path | None]:
    warm_start_from = getattr(args, "warm_start_from", None)
    if warm_start_from:
        ckpt_path = Path(warm_start_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"warm-start checkpoint not found: {ckpt_path}")
        ckpt = _torch_load_any(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        raw_model = model.module if isinstance(model, DDP) else model
        missing, unexpected = raw_model.load_state_dict(state, strict=False)
        if is_main and (missing or unexpected):
            print(
                f"[bdse] warm-start loaded with non-strict state_dict: missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}",
                flush=True,
            )
        if is_main:
            print(f"[bdse] warm-started weights from {ckpt_path}; optimizer/scaler/rng/epoch reset to 0", flush=True)
        return 0, 0, None, None, {}, ckpt_path
    if not args.resume and not args.resume_from:
        return 0, 0, None, None, {}, None
    paths = _checkpoint_paths(args)
    ckpt_path = Path(args.resume_from) if args.resume_from else paths["latest"]
    if args.resume and not ckpt_path.exists() and paths["final"].exists():
        # Backward compatibility: old runs only wrote the final output file.
        ckpt_path = paths["final"]
    if not ckpt_path.exists():
        if args.resume_from:
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        if is_main:
            print(f"[bdse] --resume requested but no checkpoint found at {paths['latest']}; starting from scratch.", flush=True)
        return 0, 0, None, None, {}, None

    ckpt = _torch_load_any(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    raw_model = model.module if isinstance(model, DDP) else model
    missing, unexpected = raw_model.load_state_dict(state, strict=False)
    if is_main and (missing or unexpected):
        print(
            f"[bdse] loaded checkpoint with non-strict state_dict: missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}",
            flush=True,
        )
    if isinstance(ckpt, dict) and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            if is_main:
                print(f"[bdse] warning: optimizer state was not restored: {type(exc).__name__}: {exc}", flush=True)
    if isinstance(ckpt, dict) and "scaler" in ckpt and hasattr(scaler, "load_state_dict"):
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as exc:
            if is_main:
                print(f"[bdse] warning: AMP scaler state was not restored: {type(exc).__name__}: {exc}", flush=True)
    _restore_rng_state(ckpt.get("rng_state") if isinstance(ckpt, dict) else None)
    start_epoch = int(ckpt.get("next_epoch", int(ckpt.get("epoch", -1)) + 1)) if isinstance(ckpt, dict) else 0
    start_batch_index = max(0, int(ckpt.get("next_batch_index", 0))) if isinstance(ckpt, dict) else 0
    best_metric = ckpt.get("best_metric") if isinstance(ckpt, dict) else None
    best_metric = None if best_metric is None else float(best_metric)
    best_epoch = ckpt.get("best_epoch") if isinstance(ckpt, dict) else None
    best_epoch = None if best_epoch is None else int(best_epoch)
    raw_trackers = ckpt.get("best_trackers", {}) if isinstance(ckpt, dict) else {}
    best_trackers: dict[str, dict[str, Any]] = {}
    if isinstance(raw_trackers, dict):
        for label, rec in raw_trackers.items():
            if not isinstance(rec, dict):
                continue
            best_trackers[str(label)] = dict(rec)
    if is_main:
        print(
            f"[bdse] resumed from {ckpt_path} at epoch={start_epoch} batch_index={start_batch_index}",
            flush=True,
        )
    return start_epoch, start_batch_index, best_metric, best_epoch, best_trackers, ckpt_path




def _append_loss_meters(meters: dict[str, Any], losses: dict[str, torch.Tensor]) -> None:
    """Accumulate loss scalars on-device without synchronizing every step.

    The previous implementation copied every scalar loss to CPU after each
    optimizer update.  That introduces an implicit CUDA synchronization and is
    particularly costly when the CPU selector is already on the critical path.
    Host transfer now occurs only once per epoch in ``_aggregate_meters``.
    """
    for key, value in losses.items():
        if not torch.is_tensor(value):
            continue
        scalar = value.detach().float().reshape(())
        rec = meters.get(str(key))
        if isinstance(rec, list) and len(rec) == 2 and torch.is_tensor(rec[0]):
            rec[0].add_(scalar)
            rec[1] += 1
        else:
            meters[str(key)] = [scalar.clone(), 1]


def _scalar_loss_finite_flag(
    losses: dict[str, torch.Tensor],
) -> tuple[list[str], torch.Tensor]:
    """Return scalar-loss names and one on-device all-finite flag.

    A separate ``Tensor.item()`` for every reported loss serializes the CUDA
    stream once per component.  v46 exposes more than twenty scalar metrics, so
    the old safety check introduced dozens of avoidable synchronizations per
    optimizer step.  This helper keeps all predicates on device and reduces the
    common path to one aggregate synchronization.
    """
    names: list[str] = []
    flags: list[torch.Tensor] = []
    for name, value in losses.items():
        if torch.is_tensor(value) and value.numel() == 1:
            names.append(str(name))
            flags.append(torch.isfinite(value.detach()).reshape(()))
    if flags:
        return names, torch.stack(flags).all()
    device = next(
        (value.device for value in losses.values() if torch.is_tensor(value)),
        torch.device("cpu"),
    )
    return names, torch.ones((), dtype=torch.bool, device=device)


def _aggregate_meters(meters: dict[str, Any], device: torch.device, distributed: bool) -> dict[str, float]:
    keys = sorted(meters)
    if not keys:
        return {}
    local = torch.zeros((len(keys), 2), dtype=torch.float64, device=device)
    for i, key in enumerate(keys):
        rec = meters.get(key)
        if isinstance(rec, list) and len(rec) == 2 and torch.is_tensor(rec[0]):
            local[i, 0] = rec[0].to(device=device, dtype=torch.float64)
            local[i, 1] = float(rec[1])
            continue
        values = [float(v) for v in (rec or []) if np.isfinite(float(v))]
        if values:
            local[i, 0] = float(np.sum(values, dtype=np.float64))
            local[i, 1] = float(len(values))
    if distributed and dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    cpu = local.cpu().numpy()
    return {
        key: float(cpu[i, 0] / max(cpu[i, 1], 1.0))
        for i, key in enumerate(keys)
    }

def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{k}": float(v) for k, v in metrics.items()}


def _validation_bdse_score(metrics: dict[str, float]) -> float:
    """Single scalar for validation-best checkpointing.

    Closed-loop nuPlan metrics are the final reporting target, but they are too
    expensive and simulator-dependent to run after every training epoch.  This
    score is therefore an open-loop proxy aligned with the paper's diagnostic
    claim: first preserve the teacher decision, then reduce teacher-cost regret,
    and finally reward evidence/margin quality.  The large coefficient on action
    match makes a 1 percentage point decision-sufficiency gain more important
    than small changes in auxiliary diagnostics, while regret/margin terms still
    break ties when match has plateaued.
    """
    def finite(name: str, default: float = 0.0) -> float:
        v = float(metrics.get(name, default))
        return v if np.isfinite(v) else default

    match = finite("val_teacher_action_match", finite("val_decision_sufficiency", 0.0))
    regret = max(0.0, finite("val_teacher_regret", 1e6))
    suff = finite("val_evidence_sufficiency", 0.0)
    hard = finite("val_hard_evidence_recall", 0.0)
    budget_full = finite("val_budget_vs_full_match", 0.0)
    full_match = finite("val_full_interface_action_match", 0.0)
    margin_err = max(0.0, finite("val_preserved_margin_error", 1e6))
    return float(
        200.0 * match
        + 80.0 * full_match
        + 10.0 * suff
        + 10.0 * hard
        + 5.0 * budget_full
        - 5.0 * np.log1p(regret / 1000.0)
        - 0.5 * np.log1p(margin_err / 1000.0)
    )


def _validation_competitive_score(metrics: dict[str, float]) -> float:
    """Checkpoint score that cannot ignore whether the residual improves winners.

    The V57 primary score was dominated by evidence recall/certificate quality and
    selected epoch 3 even though every checkpoint had zero residual action gain.
    This score preserves fixed-budget quality but explicitly rewards candidate over
    selected-local teacher match and beneficial-minus-harmful interventions.
    """
    def finite(name: str, default: float = 0.0) -> float:
        value = float(metrics.get(name, default))
        return value if np.isfinite(value) else default

    candidate = finite("val_teacher_action_match", 0.0)
    local = finite(
        "val_selected_local_anchor_action_match",
        finite("val_local_pair_full_interface_action_match", 0.0),
    )
    pair_full = finite("val_pair_full_interface_action_match", candidate)
    beneficial = finite(
        "val_beneficial_pair_potential_intervention_rate",
        finite("val_beneficial_residual_intervention_rate", 0.0),
    )
    harmful = finite(
        "val_harmful_pair_potential_intervention_rate",
        finite("val_harmful_residual_intervention_rate", 0.0),
    )
    proposal_recall = finite("val_proposal_decisive_atom_recall", 0.0)
    selected_recall = finite("val_selected_decisive_atom_recall", 0.0)
    effective_recall = finite("val_effective_selected_decisive_atom_recall", 0.0)
    interaction_recall = finite(
        "val_selected_interaction_decisive_recall",
        finite("val_interaction_decisive_recall", 0.0),
    )
    evidence_certificate = finite(
        "val_pair_action_anchor_guard_evidence_certificate_fraction",
        finite("val_evidence_certificate_fraction", 0.0),
    )
    fallback = finite("val_fallback_would_trigger_rate", 1.0)
    robust_margin = finite("val_pair_action_anchor_robust_margin", -1.0)
    proposal_rate = finite("val_local_pair_full_to_residual_flip_rate", 0.0)
    dense_full = finite("val_full_interface_action_match", 0.0)
    sparse_full = finite("val_sparse_full_interface_action_match", 0.0)
    budget_vs_full = finite("val_budget_vs_full_match", 0.0)
    dense_proposal_drop = max(0.0, dense_full - sparse_full)
    residual_gain = candidate - local
    pair_gain = pair_full - local
    no_winner_progress_penalty = 100.0 if residual_gain <= 0.0 and pair_gain <= 0.0 else 0.0
    # Checkpoint selection is lexicographically constrained by the formal
    # minimum gate.  V60 selected epoch 3 over epoch 1 even though proposal
    # recall had already fallen below 0.72, because a tiny auxiliary-score gain
    # outweighed the missing gate constraint.  The shortfall penalty makes any
    # minimum-feasible checkpoint dominate an infeasible one while preserving
    # the competitive score as the tie-breaker inside the feasible set.
    minimum_shortfall = (
        max(0.0, 0.72 - proposal_recall)
        + max(0.0, 0.50 - selected_recall)
        + max(0.0, 0.62 - effective_recall)
        + max(0.0, 0.40 - interaction_recall)
        + max(0.0, 0.40 - evidence_certificate)
        + max(0.0, fallback - 0.60)
    )
    minimum_gate_penalty = 2000.0 * minimum_shortfall
    return float(
        250.0 * candidate
        + 500.0 * residual_gain
        + 350.0 * pair_gain
        + 500.0 * (beneficial - harmful)
        + 10.0 * proposal_recall
        + 20.0 * selected_recall
        + 10.0 * effective_recall
        + 15.0 * interaction_recall
        + 40.0 * robust_margin
        + 20.0 * proposal_rate
        + 120.0 * budget_vs_full
        + 80.0 * sparse_full
        - 180.0 * dense_proposal_drop
        - 15.0 * fallback
        - no_winner_progress_penalty
        - minimum_gate_penalty
    )


def _validation_fixed_budget_critical_score(metrics: dict[str, float]) -> float:
    """Checkpoint score for the exact deployed fixed-budget decision path.

    Missing pair-full/family diagnostics are treated as failures rather than
    silently replaced by sparse-full proxies.  This prevents a checkpoint from
    winning while the true dense->pair interface or cross-family budget
    competition is unmeasured.
    """
    def finite(name: str, default: float = 0.0) -> float:
        value = float(metrics.get(name, default))
        return value if np.isfinite(value) else default

    pair_metric_present = "val_pair_full_interface_action_match" in metrics
    local_pair_metric_present = "val_local_pair_full_interface_action_match" in metrics
    residual_metric_present = "val_harmful_residual_intervention_rate" in metrics
    interaction_metric_present = "val_selector_interaction_family_selected" in metrics
    teacher_match = finite("val_teacher_action_match", finite("val_decision_sufficiency", 0.0))
    full_match = finite("val_full_interface_action_match", 0.0)
    sparse_full = finite("val_sparse_full_interface_action_match", 0.0)
    dense_proposal_drop = max(0.0, full_match - sparse_full)
    pair_full = finite("val_pair_full_interface_action_match", 0.0)
    local_pair_full = finite(
        "val_selected_local_anchor_action_match",
        finite("val_local_pair_full_interface_action_match", 0.0),
    )
    harmful_residual = finite(
        "val_harmful_pair_potential_intervention_rate",
        finite("val_harmful_residual_intervention_rate", 1.0),
    )
    beneficial_residual = finite(
        "val_beneficial_pair_potential_intervention_rate",
        finite("val_beneficial_residual_intervention_rate", 0.0),
    )
    residual_interface_drop = max(0.0, local_pair_full - pair_full)
    budget_pair = finite("val_budget_vs_pair_full_match", 0.0)
    near_sign = finite("val_pair_sign_acc_near_tie", 0.0)
    winner_sign = finite("val_pair_sign_acc_winner_rival", 0.0)
    sufficiency = finite("val_evidence_sufficiency", 0.0)
    hard_recall = finite("val_selected_hard_decisive_recall", finite("val_hard_evidence_recall", 0.0))
    decisive_recall = finite("val_selected_decisive_atom_recall", 0.0)
    fallback = finite("val_fallback_would_trigger_rate", 0.0)
    regret = max(0.0, finite("val_teacher_regret", 1e6))
    latency_p95 = max(0.0, finite("val_planner_latency_ms_p95", 0.0))
    interaction_selected = max(0.0, finite("val_selector_interaction_family_selected", 0.0))
    decision_count = max(0.0, finite("val_decision_budget_atom_count", 0.0))
    configured_budget = max(1.0, finite("val_configured_decision_budget_atom_count", 1.0))
    interaction_fraction = interaction_selected / max(decision_count, 1.0)
    exact_budget_fill = min(decision_count / configured_budget, 1.0)

    hard_shortfall = max(0.0, 0.60 - hard_recall)
    pair_shortfall = max(0.0, 0.30 - pair_full)
    near_shortfall = max(0.0, 0.55 - near_sign)
    fallback_excess = max(0.0, fallback - 0.50)
    latency_excess = max(0.0, latency_p95 / 500.0 - 1.0) if latency_p95 > 0.0 else 0.0
    interaction_excess = max(0.0, interaction_fraction - 0.85)
    fill_shortfall = max(0.0, 0.95 - exact_budget_fill)
    missing_diag_penalty = (
        (0.0 if pair_metric_present else 180.0)
        + (0.0 if local_pair_metric_present else 120.0)
        + (0.0 if residual_metric_present else 120.0)
        + (0.0 if interaction_metric_present else 80.0)
    )

    return float(
        220.0 * teacher_match
        + 100.0 * full_match
        + 100.0 * sparse_full
        + 80.0 * local_pair_full
        + 140.0 * pair_full
        + 60.0 * budget_pair
        + 55.0 * near_sign
        + 15.0 * winner_sign
        + 20.0 * sufficiency
        + 5.0 * hard_recall
        + 2.0 * decisive_recall
        + 30.0 * exact_budget_fill
        - 9.0 * np.log1p(regret / 1000.0)
        - 100.0 * hard_shortfall
        - 180.0 * pair_shortfall
        - 180.0 * dense_proposal_drop
        - 220.0 * residual_interface_drop
        - 120.0 * max(0.0, harmful_residual - beneficial_residual)
        - 100.0 * near_shortfall
        - 50.0 * fallback_excess
        - 5.0 * latency_excess
        - 240.0 * interaction_excess
        - 120.0 * fill_shortfall
        - missing_diag_penalty
    )


def _infer_best_mode(metric_name: str, requested_mode: str = "auto") -> str:
    if requested_mode in {"min", "max"}:
        return requested_mode
    name = str(metric_name).lower()
    if any(tok in name for tok in ("loss", "regret", "error", "mae", "rmse", "latency", "violation")):
        return "min"
    return "max"


def _canonical_metric_name(requested: str, metrics: dict[str, float]) -> str:
    req = str(requested).strip()
    if req in metrics:
        return req
    # Most validation diagnostics are logged with a val_ prefix.  Accept the
    # paper-facing names so commands can stay readable.
    if not req.startswith("val_") and f"val_{req}" in metrics:
        return f"val_{req}"
    if req.startswith("best_"):
        bare = req[5:]
        if bare in metrics:
            return bare
        if f"val_{bare}" in metrics:
            return f"val_{bare}"
    return req


def _resolve_best_metric_for_request(metrics: dict[str, float], requested: str, requested_mode: str = "auto") -> tuple[str, float, str, str]:
    requested = str(requested or "auto")
    if requested == "auto":
        if "val_bdse_score" in metrics and np.isfinite(float(metrics["val_bdse_score"])):
            return "auto", "val_bdse_score", float(metrics["val_bdse_score"]), "max"
        if "val_teacher_regret" in metrics and np.isfinite(float(metrics["val_teacher_regret"])):
            return "auto", "val_teacher_regret", float(metrics["val_teacher_regret"]), "min"
        if "val_loss" in metrics and np.isfinite(float(metrics["val_loss"])):
            return "auto", "val_loss", float(metrics["val_loss"]), "min"
        return "auto", "loss", float(metrics.get("loss", float("nan"))), "min"
    metric_name = _canonical_metric_name(requested, metrics)
    value = float(metrics.get(metric_name, float("nan")))
    return _sanitize_best_label(requested), metric_name, value, _infer_best_mode(metric_name, requested_mode)


def _resolve_best_metric(metrics: dict[str, float], args: argparse.Namespace) -> tuple[str, float, str]:
    _, metric_name, value, mode = _resolve_best_metric_for_request(metrics, str(getattr(args, "best_metric", "auto")), str(getattr(args, "best_mode", "auto")))
    return metric_name, value, mode


def _requested_best_metrics(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "best_metrics", None)
    values = list(raw) if raw else [str(getattr(args, "best_metric", "auto"))]
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        label = str(v).strip()
        if not label:
            continue
        key = _sanitize_best_label(label) if label != "auto" else "auto"
        if key not in seen:
            out.append(label)
            seen.add(key)
    return out or ["auto"]


def _make_preprocessed_dataset(
    *,
    preprocessed_dir: str | None,
    splits: list[str],
    max_scenarios: int | None,
    max_scenarios_per_split: int | None,
    cfg: dict[str, Any],
    max_files: int | None = None,
    for_training: bool = False,
) -> NuPlanBDSEDataset | PreprocessedBDSEDataset:
    if preprocessed_dir:
        dataset = PreprocessedBDSEDataset(
            preprocessed_dir,
            split=splits,
            max_scenarios=max_scenarios,
            max_scenarios_per_split=max_scenarios_per_split,
        )
        if for_training:
            _apply_training_quality_filter(dataset, cfg)
        return dataset
    if len(splits) != 1:
        raise ValueError("On-the-fly mode supports one split at a time; preprocess first to use multiple split folders.")
    return NuPlanBDSEDataset(cfg, split=splits[0], max_files=max_files, max_scenarios=max_scenarios, use_devkit=True)


def _make_loader(
    *,
    dataset: NuPlanBDSEDataset | PreprocessedBDSEDataset,
    cfg: dict[str, Any],
    batch_size: int,
    num_workers: int,
    cuda_available: bool,
    distributed: bool,
    world_size: int,
    global_rank: int,
    shuffle: bool,
    seed: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    wrapped = OnTheFlyDataset(dataset)
    sampler = DistributedSampler(wrapped, num_replicas=world_size, rank=global_rank, shuffle=shuffle, seed=seed) if distributed else None
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg["training"].get("pin_memory", cuda_available) and cuda_available),
        "persistent_workers": num_workers > 0,
        "collate_fn": lambda x: collate(x, cfg),
    }
    if sampler is not None:
        loader_kwargs["batch_sampler"] = ResumableBatchSampler(
            BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        )
    else:
        loader_kwargs.update(
            {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "sampler": None,
            }
        )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg["training"].get("prefetch_factor", 1))
    return DataLoader(wrapped, **loader_kwargs), sampler


@torch.no_grad()
def _run_validation_loss(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    distributed: bool,
    is_main: bool,
    epoch: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    meters: dict[str, list[float]] = {}
    for batch in tqdm(loader, desc=f"val-loss {epoch}", disable=not is_main):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch)
        losses = compute_bdse_losses(out, batch, cfg)
        _append_loss_meters(meters, losses)
    if was_training:
        model.train()
    return _prefix_metrics(_aggregate_meters(meters, device, distributed), "val_")


def _iter_distributed_indices(n: int, distributed: bool, world_size: int, global_rank: int) -> range:
    if distributed:
        return range(global_rank, n, world_size)
    return range(n)


def _teacher_literal_criticality_full_support(
    sample, pred: dict[str, Any], selected_atoms, cfg: dict[str, Any] | None = None
) -> tuple[dict[str, float], dict[str, int]]:
    """Exact teacher winner-flip criticality on the full auditable evidence bank.

    This helper intentionally does *not* use the certificate-stage active mask:
    that mask is already HAB Top-M.  Using it as the label support makes Top-M
    recall tautologically 1.0 and cannot diagnose acquisition.
    """
    pred_g = np.asarray(pred["g"], dtype=np.float32)
    active_atoms = np.asarray(sample.evidence_bank.active_mask, dtype=bool).reshape(-1)
    if active_atoms.shape[0] < pred_g.shape[0]:
        active_atoms = np.pad(active_atoms, (0, pred_g.shape[0] - active_atoms.shape[0]), constant_values=False)
    active_atoms = active_atoms[: pred_g.shape[0]]
    valid_actions = np.asarray(sample.candidates.valid_mask, dtype=bool).reshape(-1)
    teacher_g = np.zeros_like(pred_g)
    source_teacher_g = np.asarray(sample.teacher.g_evid, dtype=np.float32)
    e_lim = min(teacher_g.shape[0], source_teacher_g.shape[0])
    a_lim = min(teacher_g.shape[1], source_teacher_g.shape[1])
    teacher_g[:e_lim, :a_lim] = source_teacher_g[:e_lim, :a_lim]
    teacher_cost = np.full((pred_g.shape[1],), np.inf, dtype=np.float32)
    source_teacher_cost = np.asarray(sample.teacher.J_T, dtype=np.float32).reshape(-1)
    k_lim = min(teacher_cost.shape[0], source_teacher_cost.shape[0])
    teacher_cost[:k_lim] = source_teacher_cost[:k_lim]
    teacher_base = teacher_cost - np.where(active_atoms[:, None], teacher_g, 0.0).sum(axis=0)
    values, details = _criticality_metrics(
        teacher_base,
        teacher_g,
        active_atoms,
        valid_actions,
        np.asarray(pred.get("top_m_atoms", []), dtype=np.int64).reshape(-1),
        np.asarray(selected_atoms, dtype=np.int64).reshape(-1),
        prefix="teacher_exact_winner_flip",
        forced_winner=int(sample.teacher.a_star),
        reference_action_cost=np.asarray(pred.get("J0", []), dtype=np.float32).reshape(-1),
    )

    cfg = cfg or {}

    # V64.3.6 instrumentation fix.  V64.3.5 implemented the frozen-family-slot
    # oracle only inside the optional dense-diagnostic evaluator; short training
    # screens therefore logged null even though this is precisely the diagnostic
    # needed to decide whether atom ranking or HAB family admission is limiting.
    # Compute it on the teacher-only validation path so every screen reports it.
    budget_costs_fn = getattr(sample.evidence_bank, "budget_costs", None)
    if callable(budget_costs_fn):
        values["teacher_exact_winner_flip_frozen_family_slot_oracle_topm_recall"] = (
            _frozen_family_slot_oracle_critical_recall(
                teacher_base, teacher_g, active_atoms, valid_actions,
                int(sample.teacher.a_star), pred, sample, cfg
            )
        )
    else:
        values["teacher_exact_winner_flip_frozen_family_slot_oracle_topm_recall"] = float("nan")

    dense_teacher = teacher_base + np.where(active_atoms[:, None], teacher_g, 0.0).sum(axis=0)
    dense_teacher = np.where(valid_actions, dense_teacher, np.inf)
    if np.isfinite(dense_teacher).any() and int(np.argmin(dense_teacher)) == int(sample.teacher.a_star):
        loo = dense_teacher[None, :] - np.where(active_atoms[:, None], teacher_g, 0.0)
        loo[:, ~valid_actions] = np.inf
        critical = active_atoms & (np.argmin(loo, axis=1) != int(sample.teacher.a_star))
        ncrit = int(critical.sum())
        if ncrit > 0 and callable(budget_costs_fn):
            selector = cfg.get("selector", {}) or {}
            budget = float((cfg.get("evidence", {}) or {}).get("budget", 16))
            M = int(selector.get("proposal_top_m", max(int(2 * budget), int(budget) + 1)))
            costs = np.asarray(sample.evidence_bank.budget_costs(), dtype=np.float32).reshape(-1)
            if costs.shape[0] < teacher_g.shape[0]:
                costs = np.pad(costs, (0, teacher_g.shape[0] - costs.shape[0]), constant_values=np.inf)
            oracle_logits = np.where(active_atoms, critical.astype(np.float32) * 1000.0, -1.0e9)
            fam = np.asarray(pred.get("family_ids", np.zeros((teacher_g.shape[0],), dtype=np.int64)), dtype=np.int64).reshape(-1)
            if fam.shape[0] < teacher_g.shape[0]:
                fam = np.pad(fam, (0, teacher_g.shape[0] - fam.shape[0]))
            top_global, _, _ = select_topm_atoms_hab(
                oracle_logits, fam[: teacher_g.shape[0]], active_atoms, costs[: teacher_g.shape[0]],
                budget, M, enabled=False
            )
            hit_mask = np.zeros_like(active_atoms, dtype=bool)
            hit_mask[np.asarray(top_global, dtype=np.int64)] = True
            global_recall = float((critical & hit_mask).sum() / ncrit)
            values["teacher_exact_winner_flip_global_oracle_topm_recall"] = global_recall
            frozen = values["teacher_exact_winner_flip_frozen_family_slot_oracle_topm_recall"]
            values["teacher_exact_winner_flip_family_slot_oracle_gap"] = (
                float(global_recall - frozen) if np.isfinite(frozen) else float("nan")
            )
        else:
            values["teacher_exact_winner_flip_global_oracle_topm_recall"] = float("nan")
            values["teacher_exact_winner_flip_family_slot_oracle_gap"] = float("nan")
    else:
        values["teacher_exact_winner_flip_global_oracle_topm_recall"] = float("nan")
        values["teacher_exact_winner_flip_family_slot_oracle_gap"] = float("nan")
    return values, details


@torch.no_grad()
def _run_validation_open_loop(
    *,
    model: torch.nn.Module,
    dataset: NuPlanBDSEDataset | PreprocessedBDSEDataset,
    cfg: dict[str, Any],
    device: torch.device,
    distributed: bool,
    world_size: int,
    global_rank: int,
    is_main: bool,
    epoch: int,
    strict: bool,
    dense_diagnostic: bool = False,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    core = BDSEPlannerCore(model=raw_model, cfg=cfg)
    meters: dict[str, list[float]] = {}
    failed = 0
    indices = list(_iter_distributed_indices(len(dataset), distributed, world_size, global_rank))
    for idx in tqdm(indices, desc=f"val-open-loop {epoch}", disable=not is_main):
        try:
            sample = dataset[int(idx)]
            prediction_scope_factory = getattr(raw_model, "runtime_prediction_cache_scope", None)
            prediction_scope = prediction_scope_factory() if callable(prediction_scope_factory) else nullcontext()
            with prediction_scope:
                planner_start = time.perf_counter()
                pred, sel, tour, stage_atom_active = core._run_certificate_stage(
                    sample.runtime, sample.candidates, sample.evidence_bank, cfg
                )
                planner_latency_ms = 1000.0 * (time.perf_counter() - planner_start)
                dense = None
                if dense_diagnostic and hasattr(raw_model, "predict_dense_numpy"):
                    dense = raw_model.predict_dense_numpy(
                        sample.runtime, sample.candidates, sample.evidence_bank, cfg
                    )
            qdiag = runtime_query_diagnostics(pred, sel.selected)
            qdiag["planner_latency_ms"] = float(planner_latency_ms)
            qdiag["configured_decision_budget_atom_count"] = float(
                max(1, int((cfg.get("evidence", {}) or {}).get("budget", 1)))
            )
            tour_diag = getattr(tour, "diagnostics", {}) or {}
            for key, value in tour_diag.items():
                if (
                    key.startswith("pair_potential_")
                    or key.startswith("pair_action_anchor_")
                    or key.startswith("decisive_anchor_margin_")
                    or key.startswith("evidence_certificate_")
                    or key.startswith("residual_flip_")
                    or key.startswith("dual_certificate_")
                    or key.startswith("set_conditioned_residual_")
                    or key.startswith("base_prior_")
                    or key.startswith("learned_base_")
                    or key.startswith("structural_residual_")
                ):
                    if isinstance(value, (bool, np.bool_)):
                        qdiag[key] = float(bool(value))
                    elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                        qdiag[key] = float(value)
            qdiag["fallback_would_trigger"] = bool(core._needs_fallback(tour, sample.candidates, cfg))
            sel_diag = getattr(sel, "diagnostics", {}) or {}
            mode = str(sel_diag.get("mode", ""))
            qdiag["selector_anytime_adverse_certificate_active"] = float(
                mode == "runtime_pair_conditioned_anytime_adverse_certificate"
            )
            for key, value in sel_diag.items():
                if isinstance(value, (bool, np.bool_)):
                    qdiag[f"selector_{key}"] = float(bool(value))
                elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                    qdiag[f"selector_{key}"] = float(value)
            qdiag["top_m_atoms"] = list(map(int, np.asarray(pred.get("top_m_atoms", []), dtype=np.int64).reshape(-1).tolist()))

            # Screen/formal-open-loop parity: training-time open-loop validation must
            # expose the same literal teacher winner-flip criticality metrics used by
            # the formal evaluator.  Earlier activation screens asked for these keys
            # even though this validation path never produced them, yielding NaN and
            # a false screen failure.  This computation is teacher-only and does not
            # require dense model inference.
            teacher_critical, teacher_critical_details = _teacher_literal_criticality_full_support(
                sample, pred, sel.selected, cfg
            )

            pair_full_action = -1
            local_pair_full_action = -1
            if "pair_atom_delta" in pred and "pair_indices" in pred:
                full_atoms = np.flatnonzero(np.asarray(stage_atom_active, dtype=bool)).astype(np.int64).tolist()
                sel_cfg = cfg.get("selector", {})
                if bool(sel_cfg.get("decision_budget_excludes_structural_safety", False)):
                    structural = np.asarray(
                        pred.get("mandatory_atom_mask", np.zeros_like(stage_atom_active)), dtype=bool
                    ).reshape(-1)
                    full_atoms = [i for i in full_atoms if i >= structural.shape[0] or not bool(structural[i])]
                runtime_flags = runtime_safety_flags_from_runtime(sample.runtime, sample.candidates, cfg)
                pair_full_tour = run_pair_conditioned_tournament(
                    pred["J0"],
                    pred.get("rival_pair_atom_delta", pred["pair_atom_delta"]),
                    pred.get("rival_pair_indices", pred["pair_indices"]),
                    full_atoms,
                    sample.candidates.valid_mask,
                    runtime_flags,
                    {**cfg, "runtime_pair_margin_scale": float(pred.get("rival_pair_margin_scale", pred.get("pair_margin_scale", 100.0)))},
                    pair_atom_variance=pred.get("rival_pair_atom_var", pred.get("pair_atom_var", None)),
                    candidate_trajectories=sample.candidates.trajectories,
                    maneuver_ids=sample.candidates.maneuver_ids,
                    predicted_atom_costs=pred["g"],
                    residual_action_potential=pred.get("residual_action_potential", None),
                    residual_action_variance=pred.get("residual_action_var", None),
                    residual_set_atom_factors=pred.get("residual_set_atom_factors", None),
                    residual_set_action_factors=pred.get("residual_set_action_factors", None),
                    evidence_certificate_fraction=1.0,
                )
                pair_full_tour = core._apply_all_flagged_structural_guard(
                    pair_full_tour, sample.runtime, sample.candidates, runtime_flags, cfg
                )
                pair_full_action = int(pair_full_tour.action_index)

                # Local-only pair-full ceiling.  This uses the exact same pair
                # graph and tournament as deployment, but removes the learned
                # residual intervention.  It separates an upstream local/pair
                # graph error from a harmful residual correction.
                rival_pairs_np = np.asarray(pred.get("rival_pair_indices", pred["pair_indices"]), dtype=np.int64).reshape(-1, 2)
                local_scale = max(float(pred.get("rival_pair_margin_scale", pred.get("pair_margin_scale", 100.0))), 1e-6)
                g_sparse_np = np.asarray(pred["g"], dtype=np.float32)
                if rival_pairs_np.size:
                    local_pair_delta = (g_sparse_np[:, rival_pairs_np[:, 1]] - g_sparse_np[:, rival_pairs_np[:, 0]])
                    if bool(pred.get("pair_margin_normalized", True)):
                        local_pair_delta = local_pair_delta / local_scale
                    local_pair_full_tour = run_pair_conditioned_tournament(
                        pred["J0"], local_pair_delta, rival_pairs_np, full_atoms,
                        sample.candidates.valid_mask, runtime_flags,
                        {**cfg, "runtime_pair_margin_scale": local_scale},
                        pair_atom_variance=None,
                        candidate_trajectories=sample.candidates.trajectories,
                        maneuver_ids=sample.candidates.maneuver_ids,
                        predicted_atom_costs=pred["g"],
                    )
                    local_pair_full_tour = core._apply_all_flagged_structural_guard(
                        local_pair_full_tour, sample.runtime, sample.candidates, runtime_flags, cfg
                    )
                    local_pair_full_action = int(local_pair_full_tour.action_index)

            diag = compute_bdse_diagnostics(
                sample.candidates,
                sample.evidence_bank,
                sample.teacher,
                sample.pairs,
                pred["J0"],
                pred["g"],
                sel.selected,
                tour.action_index,
                cfg=cfg,
                inference_pairs=pred.get("rival_pair_indices", sel.pair_indices),
                query_diagnostics=qdiag,
                dense_predicted_base=None if dense is None else dense.get("J0_model", dense["J0"]),
                dense_predicted_atom_costs=None if dense is None else dense["g"],
                certificate_margin_matrix=tour.margins,
            )
            diag.values.update(teacher_critical)
            diag.details.update(teacher_critical_details)
            if dense is not None:
                diag.details["deployed_action"] = int(tour.action_index)
                add_dense_bridge_diagnostics(
                    diag,
                    dense=dense,
                    pred=pred,
                    selected_atoms=sel.selected,
                    sample=sample,
                    cfg=cfg,
                )
            selected_local_anchor_action = int(tour_diag.get("pair_action_anchor_action", diag.details.get("sparse_full_action", -1)))
            teacher_action_for_anchor = int(sample.teacher.a_star)
            if selected_local_anchor_action >= 0:
                anchor_correct = selected_local_anchor_action == teacher_action_for_anchor
                deployed_correct = int(tour.action_index) == teacher_action_for_anchor
                diag.values["selected_local_anchor_action_match"] = float(anchor_correct)
                if 0 <= selected_local_anchor_action < len(sample.teacher.J_T):
                    diag.values["selected_local_anchor_teacher_regret"] = float(
                        sample.teacher.J_T[selected_local_anchor_action] - sample.teacher.J_T[teacher_action_for_anchor]
                    )
                diag.values["deployed_vs_selected_local_anchor_match"] = float(int(tour.action_index) == selected_local_anchor_action)
                diag.values["pair_potential_deployed_flip_rate"] = float(int(tour.action_index) != selected_local_anchor_action)
                diag.values["beneficial_pair_potential_intervention_rate"] = float((not anchor_correct) and deployed_correct)
                diag.values["harmful_pair_potential_intervention_rate"] = float(anchor_correct and not deployed_correct)

            if pair_full_action >= 0:
                teacher_action = int(sample.teacher.a_star)
                budget_action = int(tour.action_index)
                dense_action = int(diag.details.get("full_action", -1))
                pair_full_correct = pair_full_action == teacher_action
                budget_correct = budget_action == teacher_action
                diag.values["pair_full_interface_action_match"] = float(pair_full_correct)
                if 0 <= pair_full_action < len(sample.teacher.J_T):
                    diag.values["pair_full_teacher_regret"] = float(
                        sample.teacher.J_T[pair_full_action] - sample.teacher.J_T[teacher_action]
                    )
                diag.values["budget_vs_pair_full_match"] = float(budget_action == pair_full_action)
                diag.values["pair_full_to_budget_flip_rate"] = float(budget_action != pair_full_action)
                diag.values["harmful_pair_compression_rate"] = float(pair_full_correct and not budget_correct)
                diag.values["beneficial_pair_compression_rate"] = float((not pair_full_correct) and budget_correct)
                if dense_action >= 0:
                    dense_correct = dense_action == teacher_action
                    diag.values["dense_to_pair_full_flip_rate"] = float(dense_action != pair_full_action)
                    diag.values["harmful_pair_interface_rate"] = float(dense_correct and not pair_full_correct)
                    diag.values["beneficial_pair_interface_rate"] = float((not dense_correct) and pair_full_correct)
                if local_pair_full_action >= 0:
                    local_correct = local_pair_full_action == teacher_action
                    diag.values["local_pair_full_interface_action_match"] = float(local_correct)
                    if 0 <= local_pair_full_action < len(sample.teacher.J_T):
                        diag.values["local_pair_full_teacher_regret"] = float(
                            sample.teacher.J_T[local_pair_full_action] - sample.teacher.J_T[teacher_action]
                        )
                    diag.values["local_pair_full_to_residual_flip_rate"] = float(local_pair_full_action != pair_full_action)
                    diag.values["harmful_residual_intervention_rate"] = float(local_correct and not pair_full_correct)
                    diag.values["beneficial_residual_intervention_rate"] = float((not local_correct) and pair_full_correct)
                    if dense_action >= 0:
                        diag.values["dense_to_local_pair_full_flip_rate"] = float(dense_action != local_pair_full_action)
                cert_fraction = float(qdiag.get("selector_aocc_certified_pair_fraction", float("nan")))
                fully_certified = bool(np.isfinite(cert_fraction) and cert_fraction >= 1.0 - 1e-8)
                diag.values["aocc_fully_certified_scene_rate"] = float(fully_certified)
                diag.values["teacher_action_match_fully_certified"] = float(budget_correct) if fully_certified else float("nan")
                diag.values["teacher_action_match_not_fully_certified"] = float(budget_correct) if not fully_certified else float("nan")
            for k, v in diag.values.items():
                meters.setdefault(k, []).append(float(v))
        except Exception:
            failed += 1
            if strict:
                raise
    metrics = _prefix_metrics(_aggregate_meters(meters, device, distributed), "val_")
    # Micro-average literal critical recall is a more stable screening statistic
    # than the mean of per-scene recalls.  Because count/hit metrics are emitted
    # on every scalar-aligned scene (including zero-critical rows), ratio-of-means
    # equals total hits / total literal critical atoms over the frozen subset.
    for prefix in ("teacher_exact_winner_flip", "exact_winner_flip"):
        count = float(metrics.get(f"val_{prefix}_critical_count", float("nan")))
        topm_hits = float(metrics.get(f"val_{prefix}_critical_topm_hit_count", float("nan")))
        selected_hits = float(metrics.get(f"val_{prefix}_critical_selected_hit_count", float("nan")))
        metrics[f"val_{prefix}_critical_recall_topm_micro"] = (
            topm_hits / count if np.isfinite(count) and count > 0.0 and np.isfinite(topm_hits) else float("nan")
        )
        metrics[f"val_{prefix}_critical_recall_selected_micro"] = (
            selected_hits / count if np.isfinite(count) and count > 0.0 and np.isfinite(selected_hits) else float("nan")
        )
    # Aggregate failure counts explicitly; _aggregate_meters would average them.
    fail_t = torch.tensor([float(failed), float(len(indices))], dtype=torch.float64, device=device)
    if distributed and dist.is_initialized():
        dist.all_reduce(fail_t, op=dist.ReduceOp.SUM)
    metrics["val_open_loop_failed"] = float(fail_t[0].item())
    metrics["val_open_loop_count"] = float(fail_t[1].item())
    metrics["val_bdse_score"] = _validation_bdse_score(metrics)
    metrics["val_fixed_budget_critical_score"] = _validation_fixed_budget_critical_score(metrics)
    metrics["val_minimum_gate_feasible"] = float(
        float(metrics.get("val_proposal_decisive_atom_recall", 0.0)) >= 0.72
        and float(metrics.get("val_selected_decisive_atom_recall", 0.0)) >= 0.50
        and float(metrics.get("val_effective_selected_decisive_atom_recall", 0.0)) >= 0.62
        and float(metrics.get("val_selected_interaction_decisive_recall", 0.0)) >= 0.40
        and float(metrics.get("val_pair_action_anchor_guard_evidence_certificate_fraction", 0.0)) >= 0.40
        and float(metrics.get("val_fallback_would_trigger_rate", 1.0)) <= 0.60
    )
    metrics["val_competitive_score"] = _validation_competitive_score(metrics)
    if was_training:
        model.train()
    return metrics


def _is_better(metric: float, best: float | None, mode: str) -> bool:
    if best is None or not np.isfinite(best):
        return True
    if mode == "max":
        return metric > best
    return metric < best


def _save_training_checkpoints(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    cfg: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
    best_metric: float | None,
    best_epoch: int | None,
    best_trackers: dict[str, dict[str, Any]],
    world_size: int,
    is_main: bool,
) -> tuple[float | None, int | None, dict[str, dict[str, Any]]]:
    requests = _requested_best_metrics(args)
    requested_mode = str(getattr(args, "best_mode", "auto"))
    current_specs: list[tuple[str, str, float, str]] = [
        _resolve_best_metric_for_request(metrics, req, requested_mode) for req in requests
    ]

    # Backward-compatible primary best: the first requested metric also controls
    # <stem>.best.pt and the historical best_metric/best_epoch fields.
    primary_label, primary_metric_name, primary_value, primary_mode = current_specs[0]
    best_min_epoch = max(int(getattr(args, "best_min_epoch", 0)), 0)
    checkpoint_eligible = int(epoch) >= best_min_epoch
    primary_improved = checkpoint_eligible and np.isfinite(primary_value) and _is_better(primary_value, best_metric, primary_mode)
    new_best_metric = primary_value if primary_improved else best_metric
    new_best_epoch = int(epoch) if primary_improved else best_epoch

    # Update metric-specific trackers independently, e.g. best_auto,
    # best_teacher_action_match, best_full_interface_action_match, best_teacher_regret.
    new_trackers: dict[str, dict[str, Any]] = {str(k): dict(v) for k, v in (best_trackers or {}).items()}
    improved_labels: list[str] = []
    for label, metric_name, value, mode in current_specs:
        rec = new_trackers.get(label, {})
        prev = rec.get("best_value", None)
        prev_float = None if prev is None else float(prev)
        improved = checkpoint_eligible and np.isfinite(value) and _is_better(float(value), prev_float, mode)
        if improved:
            new_trackers[label] = {
                "request": label,
                "metric_name": metric_name,
                "best_value": float(value),
                "best_epoch": int(epoch),
                "mode": mode,
            }
            improved_labels.append(label)
        elif label not in new_trackers:
            new_trackers[label] = {
                "request": label,
                "metric_name": metric_name,
                "best_value": None,
                "best_epoch": None,
                "mode": mode,
            }

    ckpt = _make_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        cfg=cfg,
        args=args,
        epoch=epoch,
        metrics=metrics,
        best_metric=new_best_metric,
        best_epoch=new_best_epoch,
        world_size=world_size,
        best_trackers=new_trackers,
    )
    if is_main:
        paths = _checkpoint_paths(args, epoch)
        _torch_save_atomic(ckpt, paths["latest"])
        if int(args.save_every_n_epochs) > 0 and ((epoch + 1) % int(args.save_every_n_epochs) == 0):
            _torch_save_atomic(ckpt, paths["epoch"])
        saved_specific: list[str] = []
        if bool(args.save_best):
            if primary_improved:
                _torch_save_atomic(ckpt, paths["best"])
            for label in improved_labels:
                path = _best_checkpoint_path(args, label)
                _torch_save_atomic(ckpt, path)
                saved_specific.append(f"{label}:{path}")
        print(
            f"[bdse] saved latest={paths['latest']} "
            f"epoch_ckpt={paths.get('epoch') if int(args.save_every_n_epochs) > 0 else '-'} "
            f"primary_best={paths['best'] if primary_improved and bool(args.save_best) else '(unchanged)'} "
            f"metric_bests={saved_specific if saved_specific else '(unchanged)'} "
            f"primary_metric={primary_metric_name} mode={primary_mode} best_value={new_best_metric}",
            flush=True,
        )
    return new_best_metric, new_best_epoch, new_trackers

def _reinitialize_modules_after_warm_start(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    is_main: bool,
) -> list[str]:
    """Reset algorithm-specific heads after loading a foundation checkpoint.

    A direct pair head from a foundation checkpoint is not a valid initialization
    for a residual-over-local head even when the tensor shapes match.  Loading it
    silently gives the residual branch a large, semantically wrong intervention at
    step zero.  V51 makes this conversion explicit and reproducible.
    """
    raw = cfg.get("training", {}).get("reinitialize_modules_after_warm_start", []) if isinstance(cfg, dict) else []
    prefixes = [str(x).strip() for x in raw if str(x).strip()]
    if not prefixes:
        return []
    raw_model = model.module if isinstance(model, DDP) else model
    modules = dict(raw_model.named_modules())
    matched: list[str] = []
    reset_ids: set[int] = set()
    for prefix in prefixes:
        module = modules.get(prefix)
        if module is None:
            raise ValueError(f"training.reinitialize_modules_after_warm_start matched no module: {prefix}")
        matched.append(prefix)
        for child in module.modules():
            if id(child) in reset_ids:
                continue
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()
                reset_ids.add(id(child))
    # Residual-over-local heads must be a no-op at step zero.  A generic random
    # reset can immediately flip a good foundation winner before any residual
    # evidence has been learned.  Zeroing the final residual layer preserves the
    # anchor exactly; the variance head starts from a conservative constant.
    safe_cfg = cfg.get("training", {}).get("warm_start_safe_initialization", {}) or {}
    if bool(safe_cfg.get("enabled", False)):
        def _last_linear(module: torch.nn.Module) -> torch.nn.Linear | None:
            found = None
            for child in module.modules():
                if isinstance(child, torch.nn.Linear):
                    found = child
            return found

        if "pair_head" in matched:
            layer = _last_linear(modules["pair_head"])
            if layer is not None:
                torch.nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        if "pair_var_head" in matched:
            layer = _last_linear(modules["pair_var_head"])
            if layer is not None:
                torch.nn.init.zeros_(layer.weight)
                initial_raw = float(safe_cfg.get("pair_variance_raw_bias", 0.0))
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, initial_raw)
        if "residual_action_head" in matched:
            layer = _last_linear(modules["residual_action_head"])
            if layer is not None:
                torch.nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        if "residual_action_var_head" in matched:
            layer = _last_linear(modules["residual_action_var_head"])
            if layer is not None:
                torch.nn.init.zeros_(layer.weight)
                initial_raw = float(safe_cfg.get("residual_action_variance_raw_bias", -2.0))
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, initial_raw)
        if "residual_set_atom_head" in matched:
            layer = _last_linear(modules["residual_set_atom_head"])
            if layer is not None:
                atom_std = max(float(safe_cfg.get("set_atom_factor_init_std", 0.005)), 0.0)
                if atom_std > 0.0:
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=atom_std)
                else:
                    torch.nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        if "residual_set_action_head" in matched:
            layer = _last_linear(modules["residual_set_action_head"])
            if layer is not None:
                torch.nn.init.normal_(layer.weight, mean=0.0, std=float(safe_cfg.get("set_action_factor_init_std", 0.01)))
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
    if is_main:
        print(
            f"[bdse] reinitialized warm-start modules={matched} "
            f"safe_noop={bool(safe_cfg.get('enabled', False))}",
            flush=True,
        )
    return matched


def _configure_trainable_modules(model: torch.nn.Module, cfg: dict[str, Any], is_main: bool) -> list[str]:
    """Optionally freeze the pretrained planner and fine-tune only critical heads.

    V31 full-model fine-tuning changed almost no closed-loop actions and reduced
    safety on several scenes.  V32 supports low-rate critical-head adaptation so
    the stable scene/action/base representation is preserved while the evidence,
    pair-margin and HAB heads learn the fixed-budget decision boundary.
    """
    raw = cfg.get("training", {}).get("trainable_modules", []) if isinstance(cfg, dict) else []
    prefixes = [str(x).strip() for x in raw if str(x).strip()]
    if not prefixes:
        return [name for name, p in model.named_parameters() if p.requires_grad]
    for param in model.parameters():
        param.requires_grad_(False)
    selected: list[str] = []
    for name, param in model.named_parameters():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            param.requires_grad_(True)
            selected.append(name)
    if not selected:
        raise ValueError(f"training.trainable_modules matched no parameters: {prefixes}")
    if is_main:
        trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
        total = sum(int(p.numel()) for p in model.parameters())
        print(
            f"[bdse] critical-head finetune prefixes={prefixes} trainable_params={trainable}/{total} "
            f"({100.0 * trainable / max(total, 1):.2f}%)",
            flush=True,
        )
    return selected


def _set_frozen_top_level_modules_eval(model: torch.nn.Module, cfg: dict[str, Any]) -> list[str]:
    """Keep immutable foundation modules in eval mode during head-only finetuning.

    ``model.train()`` recursively enables dropout in frozen scene/action/evidence
    encoders.  With only AP-WCCA/residual heads trainable this makes the supposedly
    immutable acquisition anchor stochastic and changes proposal metrics even when
    adapter parameters do not move.  V64.3.2 restores a literal frozen anchor by
    leaving only top-level modules that contain trainable parameters in train mode.
    """
    if not bool((cfg.get("training", {}) or {}).get("eval_frozen_modules", False)):
        return []
    raw_model = model.module if isinstance(model, DDP) else model
    frozen: list[str] = []
    for name, child in raw_model.named_children():
        params = list(child.parameters())
        if params and not any(p.requires_grad for p in params):
            child.eval()
            frozen.append(name)
    return frozen


def _adapter_parameter_snapshot(model: torch.nn.Module, prefix: str = "critical_proposal_adapter") -> dict[str, torch.Tensor]:
    raw_model = model.module if isinstance(model, DDP) else model
    return {
        name: param.detach().clone()
        for name, param in raw_model.named_parameters()
        if name == prefix or name.startswith(prefix + ".")
    }


def _adapter_parameter_delta_metrics(
    model: torch.nn.Module,
    reference: dict[str, torch.Tensor],
    prefix: str = "critical_proposal_adapter",
    metric_prefix: str = "critical_adapter",
) -> dict[str, float]:
    raw_model = model.module if isinstance(model, DDP) else model
    sq = None
    count = 0
    max_abs = None
    param_sq = None
    for name, param in raw_model.named_parameters():
        if name not in reference:
            continue
        delta = param.detach().float() - reference[name].to(device=param.device, dtype=torch.float32)
        cur = param.detach().float()
        d2 = delta.square().sum()
        p2 = cur.square().sum()
        sq = d2 if sq is None else sq + d2
        param_sq = p2 if param_sq is None else param_sq + p2
        local_max = delta.abs().max() if delta.numel() else delta.new_tensor(0.0)
        max_abs = local_max if max_abs is None else torch.maximum(max_abs, local_max)
        count += int(delta.numel())
    if sq is None or count == 0:
        return {
            f"{metric_prefix}_parameter_delta_rms": float("nan"),
            f"{metric_prefix}_parameter_delta_max_abs": float("nan"),
            f"{metric_prefix}_parameter_rms": float("nan"),
        }
    return {
        f"{metric_prefix}_parameter_delta_rms": float(torch.sqrt(sq / max(count, 1)).item()),
        f"{metric_prefix}_parameter_delta_max_abs": float(max_abs.item()),
        f"{metric_prefix}_parameter_rms": float(torch.sqrt(param_sq / max(count, 1)).item()),
    }


def _validate_deployment_training_schedule(cfg: dict[str, Any]) -> None:
    """Reject schedules that never train the selector used at deployment.

    Earlier BDSE configurations could finish all epochs before
    ``predicted_selector_start_epoch``.  In that case the action-level loss was
    optimized only with oracle evidence masks, while inference used predicted
    masks.  The run was syntactically valid but did not train the deployed
    decision path.
    """
    train_cfg = cfg.get("training", {})
    epochs = int(train_cfg.get("epochs", 0))
    predicted_start = int(train_cfg.get("predicted_selector_start_epoch", 8))
    action_start = int(train_cfg.get("action_loss_start_epoch", 4))
    pair_action_weight = float(train_cfg.get("pair_action_loss_weight", 1.0))
    action_weight = float(train_cfg.get("loss_weights", {}).get("action", 1.0))
    allow_oracle_only = bool(train_cfg.get("allow_oracle_only_selector_training", False))
    if epochs <= 0:
        raise ValueError("training.epochs must be positive")
    if action_weight > 0.0 and pair_action_weight > 0.0 and not allow_oracle_only:
        if predicted_start >= epochs:
            raise ValueError(
                "Deployment selector is never action-supervised: "
                f"predicted_selector_start_epoch={predicted_start} but epochs={epochs}. "
                "Set predicted_selector_start_epoch < epochs, increase epochs, or explicitly set "
                "training.allow_oracle_only_selector_training=true for an oracle-only ablation."
            )
        if action_start >= epochs:
            raise ValueError(
                "Action loss is never enabled: "
                f"action_loss_start_epoch={action_start} but epochs={epochs}."
            )

    budgets = train_cfg.get("deployment_budgets", None)
    weights = train_cfg.get("deployment_budget_weights", None)
    parsed: list[float] | None = None
    if budgets is not None:
        if not isinstance(budgets, (list, tuple)) or not budgets:
            raise ValueError("training.deployment_budgets must be a non-empty list")
        parsed = [float(x) for x in budgets]
        if any((not np.isfinite(x)) or x <= 0.0 for x in parsed):
            raise ValueError("training.deployment_budgets must contain finite positive values")
        if weights is not None:
            if not isinstance(weights, (list, tuple)) or len(weights) != len(parsed):
                raise ValueError("training.deployment_budget_weights must match training.deployment_budgets")
            parsed_weights = [float(x) for x in weights]
            if any((not np.isfinite(x)) or x < 0.0 for x in parsed_weights) or sum(parsed_weights) <= 0.0:
                raise ValueError("training.deployment_budget_weights must be non-negative with positive total mass")

    strategy = str(train_cfg.get("deployment_budget_strategy", "all")).strip().lower()
    allowed_strategies = {"all", "full", "exact", "weighted_round_robin", "sampled", "stratified", "primary_plus_aux", "primary+aux", "primary_aux"}
    if strategy not in allowed_strategies:
        raise ValueError(f"unsupported training.deployment_budget_strategy={strategy!r}")
    if strategy in {"primary_plus_aux", "primary+aux", "primary_aux"}:
        if parsed is None:
            raise ValueError("primary_plus_aux requires training.deployment_budgets")
        primary = float(train_cfg.get("deployment_primary_budget", float("nan")))
        if not np.isfinite(primary):
            raise ValueError("primary_plus_aux requires finite training.deployment_primary_budget")
        if min(abs(value - primary) for value in parsed) > 1e-6:
            raise ValueError("training.deployment_primary_budget must be present in training.deployment_budgets")

    cpu_backend = str(train_cfg.get("deployment_selector_cpu_backend", "sequential")).strip().lower()
    if cpu_backend not in {"sequential", "serial", "none", "thread", "threads", "process", "processes", "spawn"}:
        raise ValueError(
            "training.deployment_selector_cpu_backend must be sequential, thread, or process"
        )
    cpu_workers = int(
        train_cfg.get(
            "deployment_selector_cpu_workers",
            train_cfg.get("deployment_selector_cpu_threads", 1),
        )
    )
    if cpu_workers < 1:
        raise ValueError("training.deployment_selector_cpu_workers must be >= 1")

    min_exact = train_cfg.get("min_deployment_exact_fraction", None)
    if min_exact is not None and not allow_oracle_only:
        min_exact = float(min_exact)
        if not (0.0 <= min_exact <= 1.0):
            raise ValueError("training.min_deployment_exact_fraction must be in [0, 1]")
        batch_size = max(1, int(train_cfg.get("batch_size", 1)))
        selector_backend = str(train_cfg.get("deployment_selector_backend", "exact_cpu")).strip().lower()
        fast_backends = {"hybrid_fast", "gpu_surrogate", "fast_gpu", "surrogate_plus_exact"}
        if selector_backend in fast_backends:
            scene_count = int(train_cfg.get("deployment_exact_distill_scenes_per_rank", 1))
            cadence = max(1, int(train_cfg.get("deployment_exact_distill_every_n_steps", 4)))
        else:
            scene_count = int(train_cfg.get("deployment_selector_scenes_per_rank", 0))
            cadence = max(1, int(train_cfg.get("deployment_selector_every_n_steps", 1)))
        scene_fraction = 1.0 if scene_count <= 0 else min(scene_count, batch_size) / float(batch_size)
        expected_fraction = scene_fraction / float(cadence)
        if expected_fraction + 1e-12 < min_exact:
            raise ValueError(
                "Deployment selector exact-supervision fraction is below the configured floor: "
                f"expected~{expected_fraction:.6f}, required>={min_exact:.6f}. "
                "For exact_cpu use deployment_selector_scenes_per_rank=0 and "
                "deployment_selector_every_n_steps=1. For hybrid_fast, lower the floor or "
                "increase deployment_exact_distill_scenes_per_rank / reduce "
                "deployment_exact_distill_every_n_steps."
            )


def _optimizer_parameter_groups(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    *,
    base_lr: float,
    is_main: bool,
) -> tuple[list[torch.nn.Parameter], list[dict[str, Any]]]:
    """Build named LR groups while preserving a flat list for gradient clipping.

    V57 fine-tuned the zero-initialized residual mean head with the same very low
    learning rate used to protect the selector.  Its winner losses decreased, but
    the action potential remained below every deployment certificate.  V58 allows
    a higher residual-head LR without destabilizing the already effective proposal
    and family heads.  Prefix matching is performed after stripping DDP's ``module.``.
    """
    train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    multipliers_cfg = train_cfg.get("lr_multipliers", {}) or {}
    prefix_mult = [(str(k), float(v)) for k, v in multipliers_cfg.items()]
    buckets: dict[float, list[torch.nn.Parameter]] = {}
    flat: list[torch.nn.Parameter] = []
    counts: dict[float, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean = name[7:] if name.startswith("module.") else name
        multiplier = 1.0
        best_len = -1
        for prefix, value in prefix_mult:
            if clean == prefix or clean.startswith(prefix + "."):
                if len(prefix) > best_len:
                    multiplier = value
                    best_len = len(prefix)
        multiplier = max(float(multiplier), 1.0e-4)
        buckets.setdefault(multiplier, []).append(param)
        counts[multiplier] = counts.get(multiplier, 0) + int(param.numel())
        flat.append(param)
    groups = [
        {"params": params, "lr": float(base_lr) * mult, "lr_multiplier": mult}
        for mult, params in sorted(buckets.items())
    ]
    if is_main:
        text = ", ".join(
            f"x{mult:g}:{counts[mult]} params@{base_lr * mult:.3g}"
            for mult in sorted(counts)
        )
        print(f"[bdse] optimizer LR groups: {text}", flush=True)
    return flat, groups


def _build_adamw_optimizer(
    parameters: list[torch.nn.Parameter] | list[dict[str, Any]],
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
    cfg: dict[str, Any],
    is_main: bool,
) -> torch.optim.AdamW:
    """Construct AdamW using faster kernels without changing its objective.

    ``fused=True`` is used only on CUDA when the installed PyTorch build accepts
    it.  Unsupported builds fall back to the standard implementation.  This
    changes kernel implementation, not the optimizer equations or schedule.
    """
    train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    request_fused = bool(train_cfg.get("optimizer_fused", False)) and device.type == "cuda"
    request_foreach = bool(train_cfg.get("optimizer_foreach", not request_fused))
    kwargs: dict[str, Any] = {"lr": float(lr), "weight_decay": float(weight_decay)}
    if request_fused:
        kwargs["fused"] = True
    elif request_foreach:
        kwargs["foreach"] = True
    try:
        optimizer = torch.optim.AdamW(parameters, **kwargs)
        if is_main:
            mode = "fused" if request_fused else ("foreach" if request_foreach else "standard")
            print(f"[bdse] AdamW implementation={mode}", flush=True)
        return optimizer
    except (TypeError, RuntimeError, ValueError) as exc:
        if is_main:
            print(
                f"[bdse] warning: requested accelerated AdamW is unavailable; "
                f"falling back to standard AdamW: {type(exc).__name__}: {exc}",
                flush=True,
            )
        return torch.optim.AdamW(parameters, lr=float(lr), weight_decay=float(weight_decay))


def _clip_grad_norm_fast(
    parameters: list[torch.nn.Parameter],
    max_norm: float,
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Use the foreach clipping kernel when available; preserve exact clipping semantics."""
    use_foreach = bool((cfg.get("training", {}) or {}).get("grad_clip_foreach", True))
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, float(max_norm), foreach=use_foreach)
    except TypeError:
        return torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, nargs="+", default=["train"], help="One or more preprocessed splits/folders, e.g. train or train_1 train_2.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None, help="Total cap after split selection. With multiple concrete splits, the cap is balanced across splits.")
    parser.add_argument("--max-scenarios-per-split", type=int, default=None, help="Optional per-split cap for multi-city/cache training.")
    parser.add_argument("--preprocessed-dir", type=str, default=None, help="Load generated .npz cache instead of building samples on the fly.")
    parser.add_argument("--output", type=str, default="outputs/bdse_model.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision for faster training when available.")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None, help="Local rank passed by torchrun; normally inferred from LOCAL_RANK.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch factor when num_workers > 0. Use 1 to reduce host/pinned-memory pressure.")
    parser.add_argument("--selector-scenes-per-rank", type=int, default=None, help="Exact CPU deployment-selector scenes per DDP rank on sampled steps. 0 means the full local batch.")
    parser.add_argument("--selector-every-n-steps", type=int, default=None, help="Run exact predicted-selector supervision every N optimizer steps before the final exact tail.")
    parser.add_argument("--selector-full-last-n-steps", type=int, default=None, help="Use exact selector supervision for all local scenes during the final N training steps.")
    parser.add_argument("--exact-distill-scenes-per-rank", type=int, default=None, help="Fast backend: exact CPU selector scenes per rank on distillation steps.")
    parser.add_argument("--exact-distill-every-n-steps", type=int, default=None, help="Fast backend: run exact CPU-mask distillation every N optimizer steps.")
    parser.add_argument("--selector-cpu-threads", type=int, default=None, help="Compatibility alias for exact-selector CPU workers per DDP rank.")
    parser.add_argument("--selector-cpu-workers", type=int, default=None, help="Exact CPU selector workers per DDP rank.")
    parser.add_argument("--selector-cpu-backend", choices=["sequential", "thread", "process"], default=None, help="Scene-parallel exact selector backend. process uses spawn workers and preserves exact masks.")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pinned host memory for DataLoader batches.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint derived from --output, or from --resume-from when provided.")
    parser.add_argument("--resume-from", type=str, default=None, help="Explicit checkpoint path to resume from and continue its epoch counter.")
    parser.add_argument("--warm-start-from", type=str, default=None, help="Load only model weights from a checkpoint and start a new run at epoch 0. Use this for finetuning from a completed best checkpoint.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for per-epoch checkpoints. Defaults to <output_dir>/checkpoints.")
    parser.add_argument("--save-every-n-epochs", type=int, default=1, help="Save an epoch checkpoint every N epochs. Set 0 to disable per-epoch files.")
    parser.add_argument("--save-every-n-steps", type=int, default=0, help="Also update the latest checkpoint every N optimizer steps for mid-epoch recovery. Set 0 to disable.")
    parser.add_argument("--save-best", dest="save_best", action="store_true", default=True, help="Save <output_stem>.best.pt using --best-metric.")
    parser.add_argument("--no-save-best", dest="save_best", action="store_false", help="Disable best checkpoint saving.")
    parser.add_argument("--best-metric", type=str, default="auto", help="Primary metric used for backward-compatible <output_stem>.best.pt. 'auto' prefers val_bdse_score, then val_teacher_regret, then val_loss, then loss.")
    parser.add_argument("--best-metrics", type=str, nargs="*", default=None, help="Save additional metric-specific best checkpoints in one run, e.g. auto teacher_action_match full_interface_action_match teacher_regret.")
    parser.add_argument("--best-mode", type=str, default="auto", choices=["auto", "min", "max"], help="Whether lower or higher --best-metric/--best-metrics is better. Use auto for validation-aware defaults.")
    parser.add_argument("--best-min-epoch", type=int, default=0, help="Do not promote any metric-specific or primary best checkpoint before this zero-based epoch. Useful when early curriculum epochs intentionally undertrain residual modules.")
    parser.add_argument("--val-split", type=str, nargs="+", default=None, help="Optional validation split/folder(s), e.g. val or val_vegas. Enables validation-best checkpoints.")
    parser.add_argument("--val-preprocessed-dir", type=str, default=None, help="Validation cache root. Defaults to --preprocessed-dir.")
    parser.add_argument("--val-max-scenarios", type=int, default=None, help="Cap validation samples, e.g. 1000 for fast per-epoch validation.")
    parser.add_argument("--val-max-scenarios-per-split", type=int, default=None, help="Optional validation per-split cap for multi-city validation.")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Validation loss batch size. Defaults to --batch-size / training batch_size.")
    parser.add_argument("--val-num-workers", type=int, default=None, help="Validation DataLoader workers for val loss. Defaults to training num_workers.")
    parser.add_argument("--val-every-n-epochs", type=int, default=1, help="Run validation every N epochs. Set 0 to disable validation even when --val-split is provided.")
    parser.add_argument("--val-mode", type=str, default="open_loop", choices=["loss", "open_loop", "both"], help="Validation signal. open_loop computes BDSE decision diagnostics; both also reports val loss.")
    parser.add_argument("--val-dense-diagnostic", action="store_true", help="During open-loop validation, also run dense full-interface scoring so val_full_interface_action_match is a true dense diagnostic. Slower but useful for metric-specific best checkpoints.")
    parser.add_argument("--val-before-training", action="store_true", help="Run one deterministic validation pass immediately after warm start and before optimizer updates. Used by activation screens to establish an exact step-zero anchor on the same validation rows.")
    parser.add_argument("--val-strict", action="store_true", help="Raise validation exceptions instead of counting failed validation samples.")
    parser.add_argument("--log-file", type=str, default=None, help="Optional JSONL file for per-epoch train/validation metrics. Defaults to <output_stem>.train_log.jsonl.")
    args = parser.parse_args()
    if args.warm_start_from and (args.resume or args.resume_from):
        raise ValueError("Use either --warm-start-from for finetuning or --resume/--resume-from for continuing an interrupted run, not both.")
    cfg = load_config(args.config)
    cfg.setdefault("training", {})
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["training"]["lr"] = float(args.lr)
    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = float(args.weight_decay)
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = max(0, int(args.num_workers))
    if args.prefetch_factor is not None:
        cfg["training"]["prefetch_factor"] = max(1, int(args.prefetch_factor))
    if args.selector_scenes_per_rank is not None:
        cfg["training"]["deployment_selector_scenes_per_rank"] = max(0, int(args.selector_scenes_per_rank))
    if args.selector_every_n_steps is not None:
        cfg["training"]["deployment_selector_every_n_steps"] = max(1, int(args.selector_every_n_steps))
    if args.selector_full_last_n_steps is not None:
        cfg["training"]["deployment_selector_full_last_n_steps"] = max(0, int(args.selector_full_last_n_steps))
    if args.exact_distill_scenes_per_rank is not None:
        cfg["training"]["deployment_exact_distill_scenes_per_rank"] = max(0, int(args.exact_distill_scenes_per_rank))
    if args.exact_distill_every_n_steps is not None:
        cfg["training"]["deployment_exact_distill_every_n_steps"] = max(1, int(args.exact_distill_every_n_steps))
    selector_workers = args.selector_cpu_workers if args.selector_cpu_workers is not None else args.selector_cpu_threads
    if selector_workers is not None:
        cfg["training"]["deployment_selector_cpu_workers"] = max(1, int(selector_workers))
        cfg["training"]["deployment_selector_cpu_threads"] = max(1, int(selector_workers))
    if args.selector_cpu_backend is not None:
        cfg["training"]["deployment_selector_cpu_backend"] = str(args.selector_cpu_backend)
    if args.no_pin_memory:
        cfg["training"]["pin_memory"] = False
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    _validate_deployment_training_schedule(cfg)
    seed = int(cfg.get("seed", 17))
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        cuda_available = False
        print(f"[bdse] CUDA availability check failed; falling back to CPU: {type(exc).__name__}: {exc}", flush=True)
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
        if bool(cfg.get("training", {}).get("allow_tf32", True)):
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank is not None else 0))
    global_rank = int(os.environ.get("RANK", "0"))
    if distributed:
        if not cuda_available:
            raise RuntimeError("Distributed CUDA training was requested by torchrun, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    is_main = global_rank == 0
    cfg["training"]["global_rank"] = int(global_rank)
    cfg["training"]["world_size"] = int(world_size)
    splits = args.split
    dataset = _make_preprocessed_dataset(
        preprocessed_dir=args.preprocessed_dir,
        splits=splits,
        max_scenarios=args.max_scenarios,
        max_scenarios_per_split=args.max_scenarios_per_split,
        cfg=cfg,
        max_files=args.max_files,
        for_training=True,
    )
    dataset_len = len(dataset)
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 0))
    if is_main:
        print(
            f"[bdse] dataset_samples={dataset_len} splits={splits} batch_size_per_process={batch_size} "
            f"world_size={world_size} global_batch_size={batch_size * world_size} "
            f"num_workers={num_workers} max_scenarios={args.max_scenarios} "
            f"max_scenarios_per_split={args.max_scenarios_per_split}",
            flush=True,
        )
    loader, sampler = _make_loader(
        dataset=dataset,
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        cuda_available=cuda_available,
        distributed=distributed,
        world_size=world_size,
        global_rank=global_rank,
        shuffle=True,
        seed=seed,
    )

    val_dataset: NuPlanBDSEDataset | PreprocessedBDSEDataset | None = None
    val_loader: DataLoader | None = None
    val_sampler: DistributedSampler | None = None
    validation_enabled = bool(args.val_split) and int(args.val_every_n_epochs) > 0
    if validation_enabled:
        val_preprocessed_dir = args.val_preprocessed_dir or args.preprocessed_dir
        val_dataset = _make_preprocessed_dataset(
            preprocessed_dir=val_preprocessed_dir,
            splits=list(args.val_split),
            max_scenarios=args.val_max_scenarios,
            max_scenarios_per_split=args.val_max_scenarios_per_split,
            cfg=cfg,
            max_files=args.max_files,
            for_training=False,
        )
        if args.val_mode in {"loss", "both"}:
            val_loader, val_sampler = _make_loader(
                dataset=val_dataset,
                cfg=cfg,
                batch_size=int(args.val_batch_size or batch_size),
                num_workers=int(num_workers if args.val_num_workers is None else max(0, int(args.val_num_workers))),
                cuda_available=cuda_available,
                distributed=distributed,
                world_size=world_size,
                global_rank=global_rank,
                shuffle=False,
                seed=seed,
            )
        if is_main:
            print(
                f"[bdse] validation_samples={len(val_dataset)} val_split={args.val_split} "
                f"val_mode={args.val_mode} val_every_n_epochs={args.val_every_n_epochs} "
                f"val_best_metrics={_requested_best_metrics(args)}",
                flush=True,
            )
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is false. "
                "Fix the NVIDIA driver / PyTorch CUDA build mismatch, or run with --device cpu."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if cuda_available else "cpu")
    if is_main:
        print(f"[bdse] device={device} distributed={distributed} cuda_available={cuda_available} amp={bool(args.amp and device.type == 'cuda')}", flush=True)
    model = BDSEModel(cfg).to(device)
    _configure_trainable_modules(model, cfg, is_main)
    if distributed:
        find_unused = bool(cfg.get("training", {}).get("ddp_find_unused_parameters", True))
        broadcast_buffers = bool(cfg.get("training", {}).get("ddp_broadcast_buffers", False))
        gradient_as_bucket_view = bool(cfg.get("training", {}).get("ddp_gradient_as_bucket_view", True))
        if is_main:
            print(
                f"[bdse] DDP find_unused_parameters={find_unused} "
                f"broadcast_buffers={broadcast_buffers} gradient_as_bucket_view={gradient_as_bucket_view}",
                flush=True,
            )
        ddp_kwargs: dict[str, Any] = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            "find_unused_parameters": find_unused,
            "broadcast_buffers": broadcast_buffers,
            "gradient_as_bucket_view": gradient_as_bucket_view,
        }
        if bool(cfg.get("training", {}).get("ddp_static_graph", False)):
            ddp_kwargs["static_graph"] = True
        model = DDP(model, **ddp_kwargs)
    trainable_parameters, optimizer_parameter_groups = _optimizer_parameter_groups(
        model, cfg, base_lr=float(cfg["training"]["lr"]), is_main=is_main
    )
    if not trainable_parameters:
        raise RuntimeError("no trainable parameters after applying training.trainable_modules")
    opt = _build_adamw_optimizer(
        optimizer_parameter_groups,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        device=device,
        cfg=cfg,
        is_main=is_main,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch, start_batch_index, best_metric, best_epoch, best_trackers, loaded_checkpoint = _load_checkpoint_if_requested(
        args=args, model=model, optimizer=opt, scaler=scaler, device=device, is_main=is_main
    )
    if loaded_checkpoint is not None and bool(getattr(args, "warm_start_from", None)):
        _reinitialize_modules_after_warm_start(model, cfg, is_main)
        # DDP parameters are initialized deterministically on every rank, but an
        # explicit broadcast makes the invariant robust to future rank-specific
        # seeding changes.
        if distributed and dist.is_initialized():
            raw_model = model.module if isinstance(model, DDP) else model
            for param in raw_model.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in raw_model.buffers():
                dist.broadcast(buffer.data, src=0)
    frozen_eval_modules = _set_frozen_top_level_modules_eval(model, cfg)
    if is_main and frozen_eval_modules:
        print(f"[bdse] frozen top-level modules kept in eval mode={frozen_eval_modules}", flush=True)
    critical_adapter_reference = _adapter_parameter_snapshot(model)
    literal_pair_adapter_reference = _adapter_parameter_snapshot(
        model, prefix="literal_boundary_pair_adapter"
    )
    decisive_pair_adapter_reference = _adapter_parameter_snapshot(
        model, prefix="decisive_boundary_pair_adapter"
    )

    log_file = Path(args.log_file) if args.log_file else _checkpoint_stem(args.output).parent / f"{_checkpoint_stem(args.output).name}.train_log.jsonl"
    if is_main and start_epoch == 0:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("", encoding="utf-8")
    if validation_enabled and bool(args.val_before_training) and start_epoch == 0:
        initial_metrics: dict[str, float] = {}
        if args.val_mode in {"loss", "both"}:
            assert val_loader is not None
            initial_metrics.update(_run_validation_loss(model=model, loader=val_loader, cfg=cfg, device=device, distributed=distributed, is_main=is_main, epoch=-1))
        if args.val_mode in {"open_loop", "both"}:
            assert val_dataset is not None
            initial_metrics.update(_run_validation_open_loop(model=model, dataset=val_dataset, cfg=cfg, device=device, distributed=distributed, world_size=world_size, global_rank=global_rank, is_main=is_main, epoch=-1, strict=bool(args.val_strict), dense_diagnostic=bool(args.val_dense_diagnostic)))
        initial_metrics.update(_adapter_parameter_delta_metrics(model, critical_adapter_reference))
        initial_metrics.update(
            _adapter_parameter_delta_metrics(
                model, literal_pair_adapter_reference,
                prefix="literal_boundary_pair_adapter", metric_prefix="literal_pair_adapter"
            )
        )
        initial_metrics.update(
            _adapter_parameter_delta_metrics(
                model, decisive_pair_adapter_reference,
                prefix="decisive_boundary_pair_adapter", metric_prefix="decisive_pair_adapter"
            )
        )
        if is_main:
            print({"epoch": -1, **initial_metrics}, flush=True)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": -1, **{str(k): float(v) for k, v in initial_metrics.items()}}, sort_keys=True) + "\n")
    total_epochs = int(cfg["training"]["epochs"])
    for epoch in range(start_epoch, total_epochs):
        cfg["training"]["current_epoch"] = int(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        _set_frozen_top_level_modules_eval(model, cfg)
        epoch_wall_start = time.perf_counter()
        meters: dict[str, Any] = {}
        resumable_batch_sampler = (
            loader.batch_sampler
            if isinstance(getattr(loader, "batch_sampler", None), ResumableBatchSampler)
            else None
        )
        steps_per_epoch = max(
            1,
            resumable_batch_sampler.total_batches
            if resumable_batch_sampler is not None
            else len(loader),
        )
        cfg["training"]["steps_per_epoch"] = int(steps_per_epoch)
        resume_at = int(start_batch_index) if epoch == start_epoch else 0
        if resumable_batch_sampler is not None:
            resumable_batch_sampler.set_start_batch(resume_at)
            if is_main and resume_at > 0:
                print(
                    f"[bdse] DataLoader resumes directly at batch_index={resume_at}; "
                    f"skipped_decode_batches={resume_at}",
                    flush=True,
                )
        processed_steps = 0
        stage_wall = {"data_wait": 0.0, "h2d": 0.0, "pair_sample": 0.0, "forward": 0.0, "loss": 0.0, "backward_step": 0.0}
        previous_step_end = time.perf_counter()
        for loader_batch_index, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}", disable=not is_main)):
            batch_index = loader_batch_index + (resume_at if resumable_batch_sampler is not None else 0)
            batch_received = time.perf_counter()
            stage_wall["data_wait"] += max(batch_received - previous_step_end, 0.0)
            if resumable_batch_sampler is None and batch_index < resume_at:
                continue
            cfg["training"]["global_step"] = int(epoch * steps_per_epoch + batch_index)
            _stage_started = time.perf_counter()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            stage_wall["h2d"] += time.perf_counter() - _stage_started
            _stage_started = time.perf_counter()
            batch = _boundary_focused_pair_subsample(batch, cfg)
            stage_wall["pair_sample"] += time.perf_counter() - _stage_started
            opt.zero_grad(set_to_none=True)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)
            _stage_started = time.perf_counter()
            with autocast_ctx:
                out = model(batch)
            stage_wall["forward"] += time.perf_counter() - _stage_started
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                loss_autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
            else:
                loss_autocast_ctx = torch.cuda.amp.autocast(enabled=False)
            _stage_started = time.perf_counter()
            with loss_autocast_ctx:
                losses = compute_bdse_losses(out, batch, cfg)
            stage_wall["loss"] += time.perf_counter() - _stage_started

            # Never let an invalid objective silently turn a finetuning run into
            # repeated GradScaler step skips.  Check every scalar component and
            # synchronize the decision across DDP ranks so all workers abort
            # together instead of hanging in the next collective.
            scalar_names, local_finite_bool = _scalar_loss_finite_flag(losses)
            local_finite = local_finite_bool.to(dtype=torch.int32)
            if distributed and dist.is_initialized():
                dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
            if int(local_finite.item()) == 0:
                nonfinite_names: list[str] = []
                if not bool(local_finite_bool.item()):
                    local_flags = torch.stack(
                        [
                            torch.isfinite(losses[name].detach()).reshape(())
                            for name in scalar_names
                        ]
                    ).to(device="cpu")
                    nonfinite_names = [
                        name for name, ok in zip(scalar_names, local_flags.tolist()) if not bool(ok)
                    ]
                detail = ", ".join(nonfinite_names) if nonfinite_names else "non-finite loss on another DDP rank"
                raise FloatingPointError(
                    f"Non-finite training objective at epoch={epoch} batch_index={batch_index} "
                    f"global_step={cfg['training']['global_step']}: {detail}"
                )

            _stage_started = time.perf_counter()
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            _clip_grad_norm_fast(trainable_parameters, float(cfg["training"]["grad_clip"]), cfg)
            scaler.step(opt)
            scaler.update()
            stage_wall["backward_step"] += time.perf_counter() - _stage_started
            _append_loss_meters(meters, losses)
            processed_steps += 1
            previous_step_end = time.perf_counter()
            save_steps = max(0, int(getattr(args, "save_every_n_steps", 0)))
            if (
                save_steps > 0
                and (batch_index + 1) % save_steps == 0
                and (batch_index + 1) < steps_per_epoch
            ):
                if is_main:
                    step_ckpt = _make_checkpoint(
                        model=model,
                        optimizer=opt,
                        scaler=scaler,
                        cfg=cfg,
                        args=args,
                        epoch=epoch,
                        metrics={},
                        best_metric=best_metric,
                        best_epoch=best_epoch,
                        world_size=world_size,
                        best_trackers=best_trackers,
                        next_batch_index=batch_index + 1,
                    )
                    _torch_save_atomic(step_ckpt, _checkpoint_paths(args)["latest"])
                    print(
                        f"[bdse] saved mid-epoch latest checkpoint: epoch={epoch} "
                        f"next_batch_index={batch_index + 1}",
                        flush=True,
                    )
                if distributed and dist.is_initialized():
                    dist.barrier()
        if resumable_batch_sampler is not None:
            resumable_batch_sampler.set_start_batch(0)
        start_batch_index = 0
        epoch_metrics = _aggregate_meters(meters, device, distributed)
        epoch_wall_s = max(time.perf_counter() - epoch_wall_start, 1e-9)
        epoch_metrics["train_epoch_wall_time_s"] = float(epoch_wall_s)
        epoch_metrics["train_samples_per_second"] = float(processed_steps * batch_size * world_size / epoch_wall_s)
        denom_steps = max(processed_steps, 1)
        for stage_name, stage_seconds in stage_wall.items():
            epoch_metrics[f"train_{stage_name}_wall_time_s"] = float(stage_seconds)
            epoch_metrics[f"train_{stage_name}_ms_per_step"] = float(1000.0 * stage_seconds / denom_steps)
        epoch_metrics.update(_adapter_parameter_delta_metrics(model, critical_adapter_reference))
        epoch_metrics.update(
            _adapter_parameter_delta_metrics(
                model, literal_pair_adapter_reference,
                prefix="literal_boundary_pair_adapter", metric_prefix="literal_pair_adapter"
            )
        )
        epoch_metrics.update(
            _adapter_parameter_delta_metrics(
                model, decisive_pair_adapter_reference,
                prefix="decisive_boundary_pair_adapter", metric_prefix="decisive_pair_adapter"
            )
        )
        if validation_enabled and ((epoch + 1) % int(args.val_every_n_epochs) == 0):
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            val_metrics: dict[str, float] = {}
            if args.val_mode in {"loss", "both"}:
                assert val_loader is not None
                val_metrics.update(
                    _run_validation_loss(
                        model=model,
                        loader=val_loader,
                        cfg=cfg,
                        device=device,
                        distributed=distributed,
                        is_main=is_main,
                        epoch=epoch,
                    )
                )
            if args.val_mode in {"open_loop", "both"}:
                assert val_dataset is not None
                val_metrics.update(
                    _run_validation_open_loop(
                        model=model,
                        dataset=val_dataset,
                        cfg=cfg,
                        device=device,
                        distributed=distributed,
                        world_size=world_size,
                        global_rank=global_rank,
                        is_main=is_main,
                        epoch=epoch,
                        strict=bool(args.val_strict),
                        dense_diagnostic=bool(args.val_dense_diagnostic),
                    )
                )
            epoch_metrics.update(val_metrics)
        if is_main:
            print(epoch_metrics, flush=True)
            log_row = {"epoch": int(epoch), **{str(k): float(v) for k, v in epoch_metrics.items()}}
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_row, sort_keys=True) + "\n")
        best_metric, best_epoch, best_trackers = _save_training_checkpoints(
            args=args,
            model=model,
            optimizer=opt,
            scaler=scaler,
            cfg=cfg,
            epoch=epoch,
            metrics=epoch_metrics,
            best_metric=best_metric,
            best_epoch=best_epoch,
            best_trackers=best_trackers,
            world_size=world_size,
            is_main=is_main,
        )
        if distributed and dist.is_initialized():
            dist.barrier()
        # Optional validation-aware early stopping.  It prevents the residual
        # head from continuing to overwrite a stronger early local interface,
        # while the saved best checkpoint remains the source for evaluation.
        patience_validations = max(0, int(cfg.get("training", {}).get("early_stopping_patience_validations", 0)))
        min_stop_epoch = max(0, int(cfg.get("training", {}).get("early_stopping_min_epoch", 0)))
        validated_this_epoch = validation_enabled and ((epoch + 1) % int(args.val_every_n_epochs) == 0)
        if patience_validations > 0 and validated_this_epoch and best_epoch is not None and epoch >= min_stop_epoch:
            stale_epochs = int(epoch) - int(best_epoch)
            patience_epochs = patience_validations * max(1, int(args.val_every_n_epochs))
            if stale_epochs >= patience_epochs:
                if is_main:
                    print(
                        f"[bdse] early stopping at epoch={epoch}: best_epoch={best_epoch}, "
                        f"stale_epochs={stale_epochs}, patience_epochs={patience_epochs}",
                        flush=True,
                    )
                break
    if is_main:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_model = model.module if isinstance(model, DDP) else model
        final_metrics = {"best_metric": best_metric, "best_epoch": best_epoch, "best_trackers": best_trackers}
        torch.save({"model": raw_model.state_dict(), "cfg": cfg, "metrics": final_metrics, "best_trackers": best_trackers}, out_path)
        print(f"[bdse] saved final model={out_path} best_epoch={best_epoch} best_{args.best_metric}={best_metric}", flush=True)
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
