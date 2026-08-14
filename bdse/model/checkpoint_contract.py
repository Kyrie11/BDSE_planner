from __future__ import annotations

from typing import Any, Mapping

import torch


_DEFAULT_ALLOWED_MISSING_PREFIXES = (
    "query_extension_proj.",
    "critical_proposal_adapter.",
    "residual_action_head.",
    "residual_action_var_head.",
    "residual_set_atom_head.",
    "residual_set_action_head.",
    "decisive_anchor_frontier_value_adapter.",
)


def load_bdse_state_with_contract(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    cfg: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Load a BDSE checkpoint without silently randomizing core tensors.

    Earlier inference/calibration loaders discarded every shape-mismatched tensor
    and continued.  That is useful for optional newly-added residual heads, but it
    can silently randomize the planner foundation/query interface and turn an
    engineering incompatibility into an apparent algorithm failure.  V64 permits
    only explicitly allow-listed new modules to be absent; any missing or
    shape-mismatched core tensor is fatal.
    """

    current = model.state_dict()
    load_cfg = cfg.get("checkpoint_loading", {}) or {}
    allowed_prefixes = tuple(
        str(x) for x in load_cfg.get(
            "allowed_missing_prefixes", _DEFAULT_ALLOWED_MISSING_PREFIXES
        )
    )
    # Missing newly introduced heads are acceptable when loading an older
    # foundation.  A *present but shape-incompatible* head is different: during
    # evaluation it means the checkpoint and runtime algorithm architectures do
    # not match and silently dropping it would randomize/zero a claimed module.
    # Shape mismatch is therefore fatal by default, with a separate explicit
    # escape hatch for deliberate migrations.
    allowed_shape_prefixes = tuple(str(x) for x in load_cfg.get("allowed_shape_mismatch_prefixes", ()))
    strict_core = bool(load_cfg.get("strict_core", True))

    shared = set(current).intersection(state)
    shape_mismatch = sorted(
        key for key in shared if tuple(current[key].shape) != tuple(state[key].shape)
    )
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    missing = sorted(set(current) - set(compatible))
    unexpected = sorted(set(state) - set(current))

    def allowed_missing(key: str) -> bool:
        return any(key.startswith(prefix) for prefix in allowed_prefixes)

    def allowed_shape(key: str) -> bool:
        return any(key.startswith(prefix) for prefix in allowed_shape_prefixes)

    fatal_missing = [key for key in missing if not allowed_missing(key)]
    fatal_shape = [key for key in shape_mismatch if not allowed_shape(key)]
    if strict_core and (fatal_missing or fatal_shape):
        raise ValueError(
            f"{context}: checkpoint violates the BDSE core-state contract; "
            f"fatal_missing={fatal_missing[:12]}, fatal_shape_mismatch={fatal_shape[:12]}. "
            "Do not evaluate a partially initialized foundation/query interface."
        )

    model.load_state_dict(compatible, strict=False)
    return {
        "context": context,
        "loaded_tensor_count": len(compatible),
        "model_tensor_count": len(current),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "allowed_missing_prefixes": list(allowed_prefixes),
        "allowed_shape_mismatch_prefixes": list(allowed_shape_prefixes),
        "strict_core": strict_core,
        "core_contract_pass": not fatal_missing and not fatal_shape,
    }
