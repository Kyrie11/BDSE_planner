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
    counts = cand.get("counts", {})
    if counts and sum(int(v) for v in counts.values()) != int(cand.get("K", 32)):
        raise ValueError("candidate.counts must sum to candidate.K")
    if not cfg.get("teacher", {}).get("atom_normalization_first", True):
        raise ValueError("BDSE requires atom-level normalization before evidence summation.")
