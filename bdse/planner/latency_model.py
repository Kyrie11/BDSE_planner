"""Pure NumPy helpers for delayed-plan closed-loop evaluation."""
from __future__ import annotations

import math

import numpy as np


def _interp(traj: np.ndarray, times: np.ndarray) -> np.ndarray:
    src = np.asarray(traj, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] < 5 or len(src) == 0:
        raise ValueError("trajectory must have shape [T,5+]")
    out = np.zeros((len(times), src.shape[1]), dtype=np.float64)
    st = src[:, 4]
    for col in range(src.shape[1]):
        out[:, col] = np.interp(times, st, src[:, col], left=src[0, col], right=src[-1, col])
    out[:, 4] = times
    return out


def advance_local_trajectory(trajectory: np.ndarray, elapsed_s: float) -> np.ndarray:
    """Trim a prior local plan by elapsed time and rebase it at the new ego pose."""
    src = np.asarray(trajectory, dtype=np.float64)
    times = src[:, 4]
    horizon = float(times[-1])
    elapsed = float(np.clip(elapsed_s, 0.0, max(horizon, 0.0)))
    base = _interp(src, np.asarray([elapsed]))[0]
    query = np.minimum(times + elapsed, horizon)
    future = _interp(src, query)
    dx = future[:, 0] - base[0]; dy = future[:, 1] - base[1]
    c = math.cos(-base[2]); s = math.sin(-base[2])
    future[:, 0] = c * dx - s * dy
    future[:, 1] = s * dx + c * dy
    future[:, 2] = np.arctan2(np.sin(future[:, 2] - base[2]), np.cos(future[:, 2] - base[2]))
    future[:, 4] = times
    return future.astype(np.float32)


def stitch_delayed_plan(incumbent: np.ndarray, proposal: np.ndarray, delay_s: float) -> np.ndarray:
    """Follow incumbent until release, then execute an SE(2)-aligned proposal suffix."""
    new = np.asarray(proposal, dtype=np.float64)
    times = new[:, 4]
    old = _interp(np.asarray(incumbent, dtype=np.float64), times)
    delay = float(np.clip(delay_s, 0.0, float(times[-1])))
    old_at = _interp(old, np.asarray([delay]))[0]
    new_at = _interp(new, np.asarray([delay]))[0]
    aligned = new.copy()
    dx = new[:, 0] - new_at[0]; dy = new[:, 1] - new_at[1]
    rot = old_at[2] - new_at[2]
    c = math.cos(rot); s = math.sin(rot)
    aligned[:, 0] = old_at[0] + c * dx - s * dy
    aligned[:, 1] = old_at[1] + s * dx + c * dy
    aligned[:, 2] = np.arctan2(np.sin(new[:, 2] + rot), np.cos(new[:, 2] + rot))
    mask = times <= delay + 1e-9
    aligned[mask] = old[mask]
    aligned[:, 4] = times
    return aligned.astype(np.float32)
