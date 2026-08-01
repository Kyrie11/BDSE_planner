from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _numpy_valid_edges(
    pair_indices: np.ndarray,
    pair_residual: np.ndarray,
    valid_mask: np.ndarray,
    pair_weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pairs = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
    residual = np.asarray(pair_residual, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    count = min(int(pairs.shape[0]), int(residual.shape[0]))
    pairs = pairs[:count]
    residual = residual[:count]
    if pair_weights is None:
        weights = np.ones((count,), dtype=np.float64)
    else:
        weights = np.asarray(pair_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] < count:
            weights = np.pad(weights, (0, count - weights.shape[0]), constant_values=1.0)
        weights = weights[:count]
    if count == 0 or valid.size == 0:
        return pairs[:0], residual[:0], weights[:0], valid, np.zeros((count,), dtype=bool)
    a, b = pairs[:, 0], pairs[:, 1]
    keep = (
        (a >= 0)
        & (a < valid.size)
        & (b >= 0)
        & (b < valid.size)
        & (a != b)
        & valid[np.clip(a, 0, max(valid.size - 1, 0))]
        & valid[np.clip(b, 0, max(valid.size - 1, 0))]
        & np.isfinite(residual)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    return pairs[keep], residual[keep], weights[keep], valid, keep


def project_pair_residual_to_action_potential_numpy(
    pair_indices: np.ndarray,
    pair_residual: np.ndarray,
    valid_mask: np.ndarray,
    pair_weights: np.ndarray | None = None,
    *,
    anchor_margin: np.ndarray | None = None,
    ridge: float = 2.0e-2,
    boundary_tau: float = 0.35,
    boundary_gain: float = 2.0,
    weight_floor: float = 0.05,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project a directed residual edge field onto an integrable action potential.

    For an edge ``(a,b)``, the model residual is interpreted as a correction to
    the margin ``cost[b] - cost[a]``.  We solve the weighted least-squares Hodge
    projection

        min_phi sum_e w_e ((phi[b]-phi[a]) - r_e)^2 + ridge ||phi||^2.

    Only the conservative (gradient) component can alter action costs.  Cyclic
    inconsistency is retained as a diagnostic rather than being allowed to change
    a winner through an arbitrary tournament traversal.
    """
    pairs, residual, weights, valid, edge_keep = _numpy_valid_edges(
        pair_indices, pair_residual, valid_mask, pair_weights
    )
    k = int(valid.shape[0])
    potential = np.zeros((k,), dtype=np.float32)
    if pairs.shape[0] == 0 or int(valid.sum()) <= 1:
        return potential, {
            "pair_potential_edge_count": float(pairs.shape[0]),
            "pair_potential_reconstruction_rmse": 0.0,
            "pair_potential_cycle_fraction": 0.0,
            "pair_potential_l2": 0.0,
        }

    if anchor_margin is not None:
        margin = np.asarray(anchor_margin, dtype=np.float64).reshape(-1)
        if margin.shape[0] >= edge_keep.shape[0]:
            margin = margin[: edge_keep.shape[0]][edge_keep]
            tau = max(float(boundary_tau), 1.0e-6)
            weights = weights * (1.0 + max(float(boundary_gain), 0.0) * np.exp(-np.abs(margin) / tau))
    weights = np.maximum(weights, max(float(weight_floor), 1.0e-8))

    valid_ids = np.flatnonzero(valid)
    remap = np.full((k,), -1, dtype=np.int64)
    remap[valid_ids] = np.arange(valid_ids.size, dtype=np.int64)
    p = int(pairs.shape[0])
    d = np.zeros((p, valid_ids.size), dtype=np.float64)
    rows = np.arange(p, dtype=np.int64)
    d[rows, remap[pairs[:, 0]]] = -1.0
    d[rows, remap[pairs[:, 1]]] = 1.0
    wd = weights[:, None] * d
    lap = d.T @ wd
    rhs = d.T @ (weights * residual)
    lam = max(float(ridge), 1.0e-8)
    lap = lap + lam * np.eye(valid_ids.size, dtype=np.float64)
    try:
        phi = np.linalg.solve(lap, rhs)
    except np.linalg.LinAlgError:
        phi = np.linalg.lstsq(lap, rhs, rcond=None)[0]
    phi = phi - float(phi.mean())
    reconstructed = d @ phi
    err = reconstructed - residual
    denom = max(float(weights.sum()), 1.0e-12)
    mse = float(np.sum(weights * err * err) / denom)
    signal = float(np.sum(weights * residual * residual) / denom)
    potential[valid_ids] = phi.astype(np.float32)
    return potential, {
        "pair_potential_edge_count": float(p),
        "pair_potential_reconstruction_rmse": float(np.sqrt(max(mse, 0.0))),
        "pair_potential_cycle_fraction": float(mse / max(signal, 1.0e-12)),
        "pair_potential_l2": float(np.sqrt(np.mean(phi * phi))) if phi.size else 0.0,
    }


def project_pair_residual_to_action_potential_torch(
    pair_indices: torch.Tensor,
    pair_residual: torch.Tensor,
    pair_valid: torch.Tensor,
    valid_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    *,
    anchor_margin: torch.Tensor | None = None,
    ridge: float = 2.0e-2,
    boundary_tau: float = 0.35,
    boundary_gain: float = 2.0,
    weight_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched differentiable counterpart of the NumPy Hodge projection.

    Returns ``(potential, reconstruction_loss, cycle_fraction)``.  The two loss
    tensors are scalar means over scenes and retain gradients to the residual
    edge field.
    """
    if pair_residual.dim() != 2:
        raise ValueError("pair_residual must be [B,P]")
    bsz, pair_count = pair_residual.shape
    if pair_indices.shape[:2] != (bsz, pair_count):
        raise ValueError("pair_indices must be [B,P,2] aligned with pair_residual")
    k = int(valid_mask.shape[1])
    device, dtype = pair_residual.device, pair_residual.dtype
    a = pair_indices[..., 0].long().clamp(0, max(k - 1, 0))
    b = pair_indices[..., 1].long().clamp(0, max(k - 1, 0))
    edge_valid = pair_valid.bool() & valid_mask.gather(1, a) & valid_mask.gather(1, b) & a.ne(b)
    edge_valid = edge_valid & torch.isfinite(pair_residual)
    if pair_weights is None:
        weights = torch.ones_like(pair_residual)
    else:
        weights = pair_weights.to(device=device, dtype=dtype)
        if weights.shape != pair_residual.shape:
            weights = torch.ones_like(pair_residual)
    weights = weights.clamp_min(float(weight_floor)) * edge_valid.to(dtype)
    if anchor_margin is not None:
        tau = max(float(boundary_tau), 1.0e-6)
        boundary = torch.exp(-anchor_margin.abs() / tau)
        weights = weights * (1.0 + max(float(boundary_gain), 0.0) * boundary)

    incidence = torch.zeros((bsz, pair_count, k), dtype=dtype, device=device)
    incidence.scatter_add_(2, a.unsqueeze(-1), -edge_valid.to(dtype).unsqueeze(-1))
    incidence.scatter_add_(2, b.unsqueeze(-1), edge_valid.to(dtype).unsqueeze(-1))
    weighted_incidence = incidence * weights.unsqueeze(-1)
    lap = incidence.transpose(1, 2) @ weighted_incidence
    rhs = incidence.transpose(1, 2) @ (weights * pair_residual).unsqueeze(-1)
    eye = torch.eye(k, dtype=dtype, device=device).unsqueeze(0)
    lap = lap + max(float(ridge), 1.0e-8) * eye
    # Invalid actions are disconnected and therefore solve to zero under ridge.
    potential = torch.linalg.solve(lap, rhs).squeeze(-1)
    valid_f = valid_mask.to(dtype)
    mean = (potential * valid_f).sum(dim=1, keepdim=True) / valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    potential = (potential - mean) * valid_f
    reconstructed = (incidence @ potential.unsqueeze(-1)).squeeze(-1)
    err = (reconstructed - pair_residual) * edge_valid.to(dtype)
    denom = weights.sum(dim=1).clamp_min(1.0e-6)
    mse_scene = (weights * err.square()).sum(dim=1) / denom
    signal_scene = (weights * pair_residual.square()).sum(dim=1) / denom
    active_scene = edge_valid.any(dim=1)
    if bool(active_scene.any()):
        reconstruction_loss = mse_scene[active_scene].mean()
        cycle_fraction = (mse_scene[active_scene] / signal_scene[active_scene].clamp_min(1.0e-8)).mean()
    else:
        reconstruction_loss = pair_residual.new_zeros(())
        cycle_fraction = pair_residual.new_zeros(())
    return potential, reconstruction_loss, cycle_fraction
