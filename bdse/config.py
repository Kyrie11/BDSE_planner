from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    default_path = Path(__file__).resolve().parent / "configs" / "default.yaml"
    with default_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_update(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    teacher = cfg.get("teacher", {})
    if teacher.get("separate_hard_gate", False):
        debug = cfg.get("invalid_ablation_for_debug_only", False)
        if not debug:
            raise ValueError(
                "BDSE forbids teacher.separate_hard_gate in main experiments; hard events must be evidence atoms."
            )
    cand = cfg.get("candidate", {})
    K = int(cand.get("K", 32))
    pool_K = int(cand.get("pool_K", cand.get("K_pool", K)))
    if pool_K < K:
        raise ValueError(f"candidate.pool_K must be >= candidate.K; got pool_K={pool_K}, K={K}")
    counts = cand.get("counts", {})
    if counts:
        count_sum = sum(int(v) for v in counts.values())
        if count_sum != K:
            raise ValueError(
                f"candidate.counts must sum to candidate.K; got sum={count_sum}, K={K}. "
                "Use candidate.pool_K for a larger pre-pruning proposal pool while keeping candidate.K fixed."
            )
    pool_counts = cand.get("pool_counts", {})
    if pool_counts:
        pool_count_sum = sum(int(v) for v in pool_counts.values())
        if pool_count_sum != pool_K:
            raise ValueError(
                f"candidate.pool_counts must sum to candidate.pool_K; got sum={pool_count_sum}, pool_K={pool_K}."
            )
    if not cfg.get("teacher", {}).get("atom_normalization_first", True):
        raise ValueError("BDSE requires atom-level normalization before evidence summation.")
