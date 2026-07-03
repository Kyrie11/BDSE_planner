from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bdse.data.cache_schema import LabelOnlyFuture, RuntimeFeatures


@dataclass()
class ResponseMode:
    name: str
    probability: float
    agent_futures: np.ndarray
    traffic_lights: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _mode_probs(cfg: dict[str, Any]) -> dict[str, float]:
    modes = cfg.get("teacher", {}).get("robust_modes", {})
    defaults = {
        "logged": {"enabled": True, "prob": 0.35},
        "cv": {"enabled": True, "prob": 0.20},
        "ca": {"enabled": True, "prob": 0.10},
        "brake": {"enabled": True, "prob": 0.15},
        "yield": {"enabled": True, "prob": 0.10},
        "nonyield": {"enabled": True, "prob": 0.10},
    }
    out: dict[str, float] = {}
    for name, d in defaults.items():
        row = modes.get(name, d) if isinstance(modes, dict) else d
        if bool(row.get("enabled", True)):
            out[name] = float(row.get("prob", d["prob"]))
    total = sum(max(p, 0.0) for p in out.values())
    if total <= 0.0:
        return {"cv": 1.0}
    return {k: max(v, 0.0) / total for k, v in out.items()}


def _roll_current_agents(runtime: RuntimeFeatures, T: int, dt: float, mode: str) -> np.ndarray:
    cur = np.asarray(runtime.current_agents, dtype=np.float32)
    N = cur.shape[0]
    out = np.zeros((N, T, 5), dtype=np.float32)
    times = np.arange(1, T + 1, dtype=np.float32) * dt
    for j in range(N):
        st = cur[j]
        x0, y0, yaw = float(st[0]), float(st[1]), float(st[2])
        v0 = float(st[3]) if st.shape[0] > 3 else 0.0
        vx = float(st[5]) if st.shape[0] > 5 else v0 * np.cos(yaw)
        vy = float(st[6]) if st.shape[0] > 6 else v0 * np.sin(yaw)
        ax = ay = 0.0
        if mode == "ca":
            # Conservative tiny acceleration along current velocity.
            norm = max(float(np.hypot(vx, vy)), 1e-3)
            ax, ay = 0.5 * vx / norm, 0.5 * vy / norm
        elif mode == "brake":
            norm = max(float(np.hypot(vx, vy)), 1e-3)
            ax, ay = -2.0 * vx / norm, -2.0 * vy / norm
        elif mode == "yield":
            vx, vy = 0.4 * vx, 0.4 * vy
        elif mode == "nonyield":
            vx, vy = 1.25 * vx, 1.25 * vy
        x = x0 + vx * times + 0.5 * ax * times * times
        y = y0 + vy * times + 0.5 * ay * times * times
        v = np.maximum(0.0, np.hypot(vx + ax * times, vy + ay * times))
        out[j] = np.stack([x, y, np.full_like(times, yaw), v, times], axis=1)
    return out


def build_response_modes(runtime: RuntimeFeatures, label_future: LabelOnlyFuture | None, cfg: dict[str, Any]) -> list[ResponseMode]:
    T = int(round(float(cfg.get("candidate", {}).get("horizon_s", 8.0)) / float(cfg.get("candidate", {}).get("step_s", 0.1))))
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    probs = _mode_probs(cfg)
    modes: list[ResponseMode] = []
    for name, prob in probs.items():
        if name == "logged" and label_future is not None and np.asarray(label_future.logged_agents).size:
            futures = np.asarray(label_future.logged_agents, dtype=np.float32)
            if futures.shape[1] < T:
                pad = np.repeat(futures[:, -1:, :], T - futures.shape[1], axis=1)
                futures = np.concatenate([futures, pad], axis=1)
            futures = futures[:, :T, :5]
            tls = list(label_future.future_traffic_lights or runtime.traffic_lights)
            modes.append(ResponseMode(name="logged", probability=prob, agent_futures=futures, traffic_lights=tls, metadata={"uses_label_future": True}))
        elif name != "logged":
            modes.append(ResponseMode(name=name, probability=prob, agent_futures=_roll_current_agents(runtime, T, dt, name), traffic_lights=list(runtime.traffic_lights), metadata={"uses_label_future": False}))
    if not modes:
        modes.append(ResponseMode(name="cv", probability=1.0, agent_futures=_roll_current_agents(runtime, T, dt, "cv"), traffic_lights=list(runtime.traffic_lights), metadata={"uses_label_future": False}))
    total = sum(m.probability for m in modes)
    return [ResponseMode(m.name, float(m.probability / max(total, 1e-6)), m.agent_futures, m.traffic_lights, m.metadata) for m in modes]


def mode_to_label_future(mode: ResponseMode, label_future: LabelOnlyFuture | None, runtime: RuntimeFeatures) -> LabelOnlyFuture:
    T = mode.agent_futures.shape[1]
    logged_ego = np.zeros((T, 5), dtype=np.float32) if label_future is None else np.asarray(label_future.logged_ego, dtype=np.float32)[:T]
    if logged_ego.shape[0] < T:
        pad = np.repeat(logged_ego[-1:, :], T - logged_ego.shape[0], axis=0) if logged_ego.size else np.zeros((T, 5), dtype=np.float32)
        logged_ego = np.concatenate([logged_ego, pad], axis=0)[:T]
    valid = np.asarray(runtime.agent_valid, dtype=bool)
    if valid.shape[0] < mode.agent_futures.shape[0]:
        valid = np.pad(valid, (0, mode.agent_futures.shape[0] - valid.shape[0]), constant_values=False)
    return LabelOnlyFuture(logged_ego=logged_ego, logged_agents=mode.agent_futures.astype(np.float32), agent_valid=valid[: mode.agent_futures.shape[0]], future_traffic_lights=mode.traffic_lights, metadata={"response_mode": mode.name, **mode.metadata})
