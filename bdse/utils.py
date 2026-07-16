from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

Array = np.ndarray


def _finite_xy(arr: Array) -> Array:
    pts = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return pts.reshape(0, 2)
    return pts[np.isfinite(pts).all(axis=1)]


def as_array(x, dtype=np.float32) -> Array:
    if x is None:
        return np.zeros((0,), dtype=dtype)
    return np.asarray(x, dtype=dtype)


def stable_hash(text: str) -> int:
    h = 2166136261
    for ch in text.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def angle_wrap(angle: Array | float) -> Array | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rotation_matrix(theta: float) -> Array:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def transform_points_to_local(points_xy: Array, origin_xy: Sequence[float], origin_yaw: float) -> Array:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    rot = rotation_matrix(-origin_yaw)
    return (pts - np.asarray(origin_xy, dtype=np.float32)[None, :]) @ rot.T


def transform_yaw_to_local(yaw: Array | float, origin_yaw: float) -> Array | float:
    return angle_wrap(np.asarray(yaw) - origin_yaw)


def transform_states_to_local(states: Array, origin_xy: Sequence[float], origin_yaw: float) -> Array:
    arr = np.asarray(states, dtype=np.float32).copy()
    if arr.size == 0:
        return arr
    flat = arr.reshape(-1, arr.shape[-1])
    flat[:, :2] = transform_points_to_local(flat[:, :2], origin_xy, origin_yaw)
    if flat.shape[-1] > 2:
        flat[:, 2] = transform_yaw_to_local(flat[:, 2], origin_yaw)
    return flat.reshape(arr.shape)


def polyline_lengths(polyline: Array) -> Array:
    pts = _finite_xy(polyline)
    if len(pts) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(pts) == 1:
        return np.zeros((1,), dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)


def sample_polyline(polyline: Array, s_query: Array) -> tuple[Array, Array]:
    pts = _finite_xy(polyline)
    s_query = np.nan_to_num(np.asarray(s_query, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if pts.shape[0] < 2:
        xy = np.zeros((len(s_query), 2), dtype=np.float32)
        yaw = np.zeros((len(s_query),), dtype=np.float32)
        return xy, yaw
    s = polyline_lengths(pts)
    s_clipped = np.clip(s_query, 0.0, float(s[-1]))
    idx = np.searchsorted(s, s_clipped, side="right") - 1
    idx = np.clip(idx, 0, len(pts) - 2)
    denom = np.maximum(s[idx + 1] - s[idx], 1e-6)
    alpha = ((s_clipped - s[idx]) / denom).reshape(-1, 1)
    xy = pts[idx, :2] * (1.0 - alpha) + pts[idx + 1, :2] * alpha
    d = pts[idx + 1, :2] - pts[idx, :2]
    yaw = np.arctan2(d[:, 1], d[:, 0]).astype(np.float32)
    return xy.astype(np.float32), yaw


def nearest_polyline_distance(points: Array, polyline: Array) -> Array:
    pts = np.nan_to_num(np.asarray(points, dtype=np.float32).reshape(-1, 2), nan=0.0, posinf=1e6, neginf=-1e6)
    pl = _finite_xy(polyline)
    if len(pl) == 0:
        return np.full((len(pts),), 1e6, dtype=np.float32)
    if len(pl) == 1:
        return np.linalg.norm(pts - pl[0], axis=1).astype(np.float32)
    best = np.full((len(pts),), 1e6, dtype=np.float32)
    for a, b in zip(pl[:-1], pl[1:]):
        ab = b - a
        denom = max(float(ab @ ab), 1e-6)
        t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
        proj = a[None, :] + t[:, None] * ab[None, :]
        best = np.minimum(best, np.linalg.norm(pts - proj, axis=1))
    return best.astype(np.float32)


def route_progress_along_polyline(points: Array, polyline: Array) -> Array:
    pts = np.nan_to_num(np.asarray(points, dtype=np.float32).reshape(-1, 2), nan=0.0, posinf=1e6, neginf=-1e6)
    pl = _finite_xy(polyline)
    if len(pl) < 2:
        return pts[:, 0].astype(np.float32)
    s = polyline_lengths(pl)
    best_dist = np.full((len(pts),), 1e9, dtype=np.float32)
    best_s = np.zeros((len(pts),), dtype=np.float32)
    for i, (a, b) in enumerate(zip(pl[:-1], pl[1:])):
        ab = b - a
        denom = max(float(ab @ ab), 1e-6)
        t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
        proj = a[None, :] + t[:, None] * ab[None, :]
        dist = np.linalg.norm(pts - proj, axis=1)
        better = dist < best_dist
        best_dist[better] = dist[better]
        best_s[better] = s[i] + t[better] * (s[i + 1] - s[i])
    return best_s.astype(np.float32)


def finite_difference(x: Array, dt: float) -> Array:
    arr = np.asarray(x, dtype=np.float32)
    if len(arr) < 2:
        return np.zeros_like(arr)
    return np.gradient(arr, dt, axis=0).astype(np.float32)


def compute_curvature(xy: Array) -> Array:
    pts = np.nan_to_num(np.asarray(xy, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    if len(pts) < 3:
        return np.zeros((len(pts),), dtype=np.float32)
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-6)
    return ((dx * ddy - dy * ddx) / denom).astype(np.float32)


def oriented_box_corners(x: float, y: float, yaw: float, length: float, width: float) -> Array:
    x = 0.0 if not np.isfinite(float(x)) else float(x)
    y = 0.0 if not np.isfinite(float(y)) else float(y)
    yaw = 0.0 if not np.isfinite(float(yaw)) else float(yaw)
    length = 4.8 if not np.isfinite(float(length)) or float(length) <= 0 else float(length)
    width = 2.0 if not np.isfinite(float(width)) or float(width) <= 0 else float(width)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float32)
    return local @ rotation_matrix(yaw).T + np.array([x, y], dtype=np.float32)[None, :]


def polygons_overlap_sat(poly_a: Array, poly_b: Array) -> bool:
    a = np.asarray(poly_a, dtype=np.float32)
    b = np.asarray(poly_b, dtype=np.float32)
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        axes = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-6)
        for axis in axes:
            pa = a @ axis
            pb = b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def mask_invalid(values: Array, valid_mask: Array, invalid_value: float) -> Array:
    out = np.asarray(values).copy()
    mask = np.asarray(valid_mask, dtype=bool)
    out[~mask] = invalid_value
    return out


def softmin_np(values: Array, tau: float = 1.0) -> float:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return -1e9
    if tau <= 0:
        return float(vals.min())
    z = -vals / tau
    m = z.max()
    return float(-tau * (m + np.log(np.exp(z - m).sum())))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def deterministic_order(keys: Iterable[object]) -> list[int]:
    return sorted(range(len(list(keys))), key=lambda i: str(list(keys)[i]))


def resolve_torch_device(device: str | None = "auto", *, context: str = "BDSE"):
    """Resolve a torch device string for inference/evaluation code.

    ``auto`` chooses CUDA when available and CPU otherwise. Explicit CUDA
    requests such as ``cuda`` or ``cuda:0`` are respected when possible and
    safely fall back to CPU when the runtime PyTorch build cannot see CUDA.
    Torch is imported lazily so non-ML utility users do not pay the import cost.
    """
    import torch

    requested = str(device or "auto").strip().lower()
    if requested in {"", "auto", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"{context}: CUDA was requested via device={device!r}, but torch.cuda.is_available() is False; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def configure_torch_for_device(device) -> None:
    """Enable safe inference-time CUDA performance knobs for the resolved device."""
    import torch

    dev = torch.device(device)
    if dev.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def torch_load_any(path: str | Path, map_location="cpu"):
    """Compatibility wrapper for PyTorch versions with/without weights_only."""
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

