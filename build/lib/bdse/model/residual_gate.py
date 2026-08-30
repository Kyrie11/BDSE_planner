from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _gate_parameters(cfg: dict[str, Any] | None) -> dict[str, float | bool]:
    rcfg = cfg or {}
    max_w = float(np.clip(float(rcfg.get("max_residual_weight", 0.25)), 0.0, 1.0))
    min_w = float(np.clip(float(rcfg.get("min_residual_weight", 0.0)), 0.0, max_w))
    return {
        "max_w": max_w,
        "min_w": min_w,
        "variance_tau": max(float(rcfg.get("variance_tau", 0.15)), 1e-6),
        "boundary_tau": max(float(rcfg.get("boundary_tau", 0.30)), 1e-6),
        "min_boundary_trust": float(np.clip(float(rcfg.get("min_boundary_trust", 0.05)), 0.0, 1.0)),
        "disagreement_penalty": float(np.clip(float(rcfg.get("disagreement_penalty", 1.0)), 0.0, 1.0)),
        "magnitude_ratio_tau": max(float(rcfg.get("magnitude_ratio_tau", 1.25)), 1e-3),
        "aggregate_max_correction_ratio": max(float(rcfg.get("aggregate_max_correction_ratio", 0.50)), 0.0),
        "aggregate_abs_cap": max(float(rcfg.get("aggregate_abs_cap", 0.05)), 0.0),
        "aggregate_preserve_sign_ratio": float(np.clip(float(rcfg.get("aggregate_preserve_sign_ratio", 0.80)), 0.0, 0.999)),
        "flip_confidence_beta": max(float(rcfg.get("flip_confidence_beta", 2.0)), 0.0),
        "flip_margin": max(float(rcfg.get("flip_margin", 0.05)), 0.0),
        "allow_confident_flips": bool(rcfg.get("allow_confident_flips", True)),
        "confident_flip_cap_ratio": max(float(rcfg.get("confident_flip_cap_ratio", 1.50)), 1.0),
    }


def confidence_shrunk_residual_pair_delta_numpy(
    local: np.ndarray,
    residual: np.ndarray,
    variance: np.ndarray,
    cfg: dict[str, Any] | None,
    base_margin: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Combine an integrable local interface with a bounded sparse residual.

    Trust is first computed per atom, then a pair-level intervention gate limits
    the *aggregate* correction.  The aggregate gate is essential: many tiny
    atom-wise residuals can otherwise sum to an unconfident action flip even
    when every individual trust coefficient is small.
    """
    local = np.asarray(local, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    if local.shape != residual.shape or local.size == 0:
        return residual.astype(np.float32), {
            "residual_trust_mean": 0.0,
            "residual_trust_p90": 0.0,
            "residual_sign_disagreement_rate": 0.0,
            "residual_pair_flip_proposal_rate": 0.0,
            "residual_pair_flip_allowed_rate": 0.0,
            "residual_pair_confident_flip_rate": 0.0,
            "residual_aggregate_scale_mean": 0.0,
        }
    p = _gate_parameters(cfg)
    var = np.asarray(variance, dtype=np.float32)
    if var.shape != residual.shape:
        var = np.zeros_like(residual, dtype=np.float32)

    std = np.sqrt(np.maximum(var, 0.0))
    variance_trust = 1.0 / (1.0 + std / float(p["variance_tau"]))
    local_sum = local.sum(axis=0)
    base = np.zeros_like(local_sum, dtype=np.float32) if base_margin is None else np.asarray(base_margin, dtype=np.float32)
    if base.shape != local_sum.shape:
        base = np.zeros_like(local_sum, dtype=np.float32)
    anchor_sum = base + local_sum
    # FAR-DBAP authorizes intervention at the *full foundation margin*, not at
    # the evidence-only margin.  Near-boundary anchors receive more trust;
    # well-separated anchors receive only the configured minimum trust.
    boundary_need = np.exp(-np.abs(anchor_sum) / float(p["boundary_tau"])).astype(np.float32)
    boundary_trust_pair = float(p["min_boundary_trust"]) + (1.0 - float(p["min_boundary_trust"])) * boundary_need
    disagreement = (local * residual < 0.0).astype(np.float32)
    sign_trust = 1.0 - float(p["disagreement_penalty"]) * disagreement
    ratio = np.abs(residual) / (np.abs(local) + float(p["boundary_tau"]))
    magnitude_trust = 1.0 / (1.0 + np.maximum(ratio / float(p["magnitude_ratio_tau"]) - 1.0, 0.0))

    trust = float(p["max_w"]) * variance_trust * boundary_trust_pair[None, ...] * sign_trust * magnitude_trust
    trust = np.clip(trust, float(p["min_w"]), float(p["max_w"])).astype(np.float32)
    correction = trust * residual

    # Evidence is the first axis and pairs are the remaining axes in runtime.
    correction_sum = correction.sum(axis=0)
    correction_var = ((trust * trust) * np.maximum(var, 0.0)).sum(axis=0)
    correction_std = np.sqrt(np.maximum(correction_var, 0.0))
    proposed_anchor = anchor_sum + correction_sum
    nonzero_anchor = np.abs(anchor_sum) > 1e-8
    flip_proposed = nonzero_anchor & (anchor_sum * proposed_anchor < 0.0)
    lower_confidence = np.abs(correction_sum) - float(p["flip_confidence_beta"]) * correction_std
    confident_flip = (
        flip_proposed
        & bool(p["allow_confident_flips"])
        & (lower_confidence >= np.abs(anchor_sum) + float(p["flip_margin"]))
    )

    normal_cap = float(p["aggregate_abs_cap"]) + float(p["aggregate_max_correction_ratio"]) * np.abs(anchor_sum)
    preserve_cap = float(p["aggregate_preserve_sign_ratio"]) * np.abs(anchor_sum)
    flip_cap = float(p["aggregate_abs_cap"]) + float(p["confident_flip_cap_ratio"]) * np.abs(anchor_sum)
    cap = np.where(flip_proposed & ~confident_flip, preserve_cap, normal_cap)
    cap = np.where(confident_flip, np.maximum(cap, flip_cap), cap)
    aggregate_scale = np.minimum(1.0, cap / (np.abs(correction_sum) + 1e-8)).astype(np.float32)
    correction = correction * np.expand_dims(aggregate_scale, axis=0)
    combined = local + correction
    final_sum = combined.sum(axis=0)
    final_anchor = base + final_sum
    flip_allowed = nonzero_anchor & (anchor_sum * final_anchor < 0.0)

    return combined.astype(np.float32), {
        "residual_trust_mean": float(np.mean(trust)),
        "residual_trust_p50": float(np.quantile(trust, 0.50)),
        "residual_trust_p90": float(np.quantile(trust, 0.90)),
        "residual_sign_disagreement_rate": float(np.mean(disagreement)),
        "residual_abs_mean": float(np.mean(np.abs(residual))),
        "local_abs_mean": float(np.mean(np.abs(local))),
        "combined_abs_mean": float(np.mean(np.abs(combined))),
        "residual_pair_flip_proposal_rate": float(np.mean(flip_proposed)),
        "residual_pair_flip_allowed_rate": float(np.mean(flip_allowed)),
        "residual_pair_confident_flip_rate": float(np.mean(confident_flip)),
        "residual_pair_sign_preserved_rate": float(np.mean(~flip_allowed | confident_flip)),
        "residual_aggregate_scale_mean": float(np.mean(aggregate_scale)),
        "residual_aggregate_scale_p10": float(np.quantile(aggregate_scale, 0.10)),
        "residual_aggregate_correction_abs_mean": float(np.mean(np.abs(correction.sum(axis=0)))),
        "residual_aggregate_local_abs_mean": float(np.mean(np.abs(local_sum))),
        "residual_aggregate_anchor_abs_mean": float(np.mean(np.abs(anchor_sum))),
        "residual_anchor_boundary_rate": float(np.mean(np.abs(anchor_sum) <= float(p["boundary_tau"]))),
    }


def confidence_shrunk_residual_pair_delta_torch(
    local: torch.Tensor,
    residual: torch.Tensor,
    variance: torch.Tensor | None,
    cfg: dict[str, Any] | None,
    base_margin: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable training counterpart of the NumPy deployment gate.

    Inputs have shape ``[B, E, P]``.  Discrete intervention masks are derived
    from detached tensors so gradients optimize the deployed continuous path
    without trying to differentiate through sign decisions.
    """
    if local.shape != residual.shape or local.numel() == 0:
        return residual
    p = _gate_parameters(cfg)
    var = torch.zeros_like(residual) if variance is None or variance.shape != residual.shape else variance.clamp_min(0.0)
    std = torch.sqrt(var + 1e-12)
    variance_trust = 1.0 / (1.0 + std / float(p["variance_tau"]))
    local_sum = local.sum(dim=1)
    base = torch.zeros_like(local_sum) if base_margin is None or base_margin.shape != local_sum.shape else base_margin.to(local)
    anchor_sum = base + local_sum
    boundary_need = torch.exp(-anchor_sum.abs() / float(p["boundary_tau"]))
    boundary_trust_pair = float(p["min_boundary_trust"]) + (1.0 - float(p["min_boundary_trust"])) * boundary_need
    disagreement = (local.detach() * residual.detach() < 0.0).to(local.dtype)
    sign_trust = 1.0 - float(p["disagreement_penalty"]) * disagreement
    ratio = residual.abs() / (local.abs() + float(p["boundary_tau"]))
    magnitude_trust = 1.0 / (1.0 + torch.clamp(ratio / float(p["magnitude_ratio_tau"]) - 1.0, min=0.0))
    trust = float(p["max_w"]) * variance_trust * boundary_trust_pair[:, None, :] * sign_trust * magnitude_trust
    trust = trust.clamp(min=float(p["min_w"]), max=float(p["max_w"]))
    correction = trust * residual

    correction_sum = correction.sum(dim=1)
    correction_var = ((trust * trust) * var).sum(dim=1)
    correction_std = torch.sqrt(correction_var + 1e-12)
    proposed_anchor = anchor_sum + correction_sum
    with torch.no_grad():
        nonzero_anchor = anchor_sum.abs() > 1e-8
        flip_proposed = nonzero_anchor & (anchor_sum * proposed_anchor < 0.0)
        lower_confidence = correction_sum.abs() - float(p["flip_confidence_beta"]) * correction_std
        confident_flip = (
            flip_proposed
            & bool(p["allow_confident_flips"])
            & (lower_confidence >= anchor_sum.abs() + float(p["flip_margin"]))
        )
        normal_cap = float(p["aggregate_abs_cap"]) + float(p["aggregate_max_correction_ratio"]) * anchor_sum.abs()
        preserve_cap = float(p["aggregate_preserve_sign_ratio"]) * anchor_sum.abs()
        flip_cap = float(p["aggregate_abs_cap"]) + float(p["confident_flip_cap_ratio"]) * anchor_sum.abs()
        cap = torch.where(flip_proposed & ~confident_flip, preserve_cap, normal_cap)
        cap = torch.where(confident_flip, torch.maximum(cap, flip_cap), cap)
    aggregate_scale = torch.minimum(torch.ones_like(correction_sum), cap / (correction_sum.abs() + 1e-8))
    return local + correction * aggregate_scale[:, None, :]
