from __future__ import annotations

import os
from typing import Any

import torch

from bdse.utils import torch_load_any
from bdse.model.bdse_model import BDSEModel
from bdse.model.checkpoint_contract import load_bdse_state_with_contract
from bdse.external_baselines.models import ExternalBaselineModel, is_external_enabled, external_variant


def build_model_for_config(cfg: dict[str, Any]) -> torch.nn.Module:
    if is_external_enabled(cfg):
        return ExternalBaselineModel(cfg)
    return BDSEModel(cfg)


def _load_external_strict(model: torch.nn.Module, checkpoint: str, cfg: dict[str, Any], variant: str) -> None:
    ckpt = torch_load_any(checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"external checkpoint must be a dictionary: {checkpoint}")
    ckpt_cfg = ckpt.get("cfg", {}) or {}
    ckpt_variant = external_variant(ckpt_cfg) if isinstance(ckpt_cfg, dict) else ""
    if ckpt_variant != variant:
        raise ValueError(f"external checkpoint variant mismatch: config={variant!r}, checkpoint={ckpt_variant!r}, path={checkpoint}")
    state = ckpt.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"external checkpoint has no model state: {checkpoint}")
    current = model.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    shape_mismatch = sorted(k for k in set(current) & set(state) if tuple(current[k].shape) != tuple(state[k].shape))
    allow_partial = os.environ.get("BDSE_ALLOW_PARTIAL_EXTERNAL_CHECKPOINT", "0") in {"1", "true", "TRUE"}
    if (missing or unexpected or shape_mismatch) and not allow_partial:
        raise ValueError(
            "strict external checkpoint load failed: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, shape_mismatch={shape_mismatch[:8]}, path={checkpoint}. "
            "Retrain with the current adapter code; do not compare partially initialized models."
        )
    compatible = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
    model.load_state_dict(compatible, strict=not allow_partial)
    if allow_partial and (missing or unexpected or shape_mismatch):
        print(
            f"WARNING: partial external checkpoint load enabled: loaded={len(compatible)}/{len(current)} "
            f"missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape_mismatch)}",
            flush=True,
        )


def load_model_for_config(checkpoint: str | None, cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_model_for_config(cfg)
    external = is_external_enabled(cfg)
    variant = external_variant(cfg) if external else "bdse"
    if checkpoint:
        if external:
            _load_external_strict(model, checkpoint, cfg, variant)
        else:
            ckpt = torch_load_any(checkpoint, map_location="cpu")
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            if not isinstance(state, dict):
                raise ValueError(f"BDSE checkpoint has no state dictionary: {checkpoint}")
            report = load_bdse_state_with_contract(
                model, state, cfg, context=f"BDSE inference load: {checkpoint}"
            )
            if report["missing"] or report["unexpected"] or report["shape_mismatch"]:
                print(
                    f"Loaded {report['loaded_tensor_count']}/{report['model_tensor_count']} tensors for {variant}; "
                    f"allowed missing/new={report['missing'][:8]} unexpected={report['unexpected'][:8]} "
                    f"shape_mismatch={report['shape_mismatch'][:8]}",
                    flush=True,
                )
    elif not (external and variant == "pdm_closed"):
        raise ValueError("--checkpoint is required for BDSE and trainable external baselines; PDM-Closed-style can run without one.")
    model.to(device)
    model.eval()
    return model
