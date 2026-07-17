from __future__ import annotations

from typing import Any

import torch

from bdse.utils import torch_load_any
from bdse.model.bdse_model import BDSEModel
from bdse.external_baselines.models import ExternalBaselineModel, is_external_enabled, external_variant


def build_model_for_config(cfg: dict[str, Any]) -> torch.nn.Module:
    if is_external_enabled(cfg):
        return ExternalBaselineModel(cfg)
    return BDSEModel(cfg)


def load_model_for_config(checkpoint: str | None, cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_model_for_config(cfg)
    variant = external_variant(cfg) if is_external_enabled(cfg) else "bdse"
    if checkpoint:
        ckpt = torch_load_any(checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        current = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)} if isinstance(state, dict) else {}
        missing = sorted(set(current) - set(compatible))
        if missing:
            print(f"Loaded {len(compatible)}/{len(current)} compatible tensors for {variant}; missing/new tensors include: {missing[:8]}")
        model.load_state_dict(compatible, strict=False)
    elif not (is_external_enabled(cfg) and variant == "pdm_closed"):
        raise ValueError("--checkpoint is required for BDSE and trainable external baselines; PDM-Closed can run without one.")
    model.to(device)
    model.eval()
    return model
