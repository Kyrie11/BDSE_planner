"""Opt-in planning profiling; no changes to the frozen algorithm or weights.

native: synchronize CUDA only at the complete-call boundaries.
attribution: additionally synchronize measured method boundaries. This mode
perturbs scheduling and must not be used as deployment deadline evidence.
The deadline is descriptive; this module never rejects or delays a plan.
"""
from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_ACTIVE = contextvars.ContextVar("bdse_budget_active", default=None)
_WRITE_LOCK = threading.Lock()


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if hasattr(value, "item"):
        return _finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def synchronize(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def memory_snapshot(device):
    if not str(device).startswith("cuda"):
        return {}
    import torch
    return {"cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))}


def trace_path(prefix, pid=None):
    return Path(str(prefix) + "." + str(os.getpid() if pid is None else pid) + ".jsonl")


def write_row(prefix, row):
    path = trace_path(prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    # One file per process; lock serializes simulation threads in that process.
    payload = json.dumps(_finite(row), ensure_ascii=False, allow_nan=False) + "\n"
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(payload)


@contextmanager
def span(name):
    state = _ACTIVE.get()
    if state is None or state["mode"] != "attribution":
        yield
        return
    synchronize(state["device"])
    started = time.perf_counter_ns()
    frame = {"children_ms": 0.0}
    state["stack"].append(frame)
    try:
        yield
    finally:
        synchronize(state["device"])
        elapsed = (time.perf_counter_ns() - started) / 1e6
        state["stack"].pop()
        if state["stack"]:
            state["stack"][-1]["children_ms"] += elapsed
        bucket = state["spans"].setdefault(name, {"inclusive_ms": 0.0, "exclusive_ms": 0.0, "calls": 0})
        bucket["inclusive_ms"] += elapsed
        bucket["exclusive_ms"] += max(0.0, elapsed - frame["children_ms"])
        bucket["calls"] += 1


def _wrap_method(obj, method, label):
    original = getattr(obj, method, None)
    if not callable(original) or getattr(original, "_bdse_budget_wrapped", False):
        return
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with span(label):
            return original(*args, **kwargs)
    wrapped._bdse_budget_wrapped = True
    setattr(obj, method, wrapped)


def instrument_core(core, attribution=False):
    """Wrap an individual core, preserving its result and frozen source files."""
    if not getattr(core, "_bdse_budget_capture_installed", False):
        original = core.plan_from_runtime
        @functools.wraps(original)
        def capture(runtime, *args, **kwargs):
            state = _ACTIVE.get()
            if state is not None:
                state["core_calls"] += 1
                state["runtime_shape"] = runtime_shape(runtime)
            result = original(runtime, *args, **kwargs)
            if state is not None:
                state["diagnostics"] = result[2]
                state["action_index"] = int(result[0])
                state["trajectory"] = result[1]
            return result
        core.plan_from_runtime = capture
        core._bdse_budget_capture_installed = True
    if attribution:
        for method, label in [("_run_certificate_stage", "certificate_stage"),
                              ("_predict_runtime_certificate", "certificate_prediction")]:
            _wrap_method(core, method, label)
        model = getattr(core, "model", None)
        if model is not None:
            for method, label in [("_make_batch", "tensorize_and_transfer"),
                                  ("encode_context", "dense_context_encoder"),
                                  ("score_sparse_queries", "action_atom_scorer"),
                                  ("score_sparse_pairs", "pair_atom_scorer")]:
                _wrap_method(model, method, label)


def runtime_shape(runtime):
    import numpy as np
    current = np.asarray(runtime.current_agents)
    valid = np.asarray(runtime.agent_valid, dtype=bool)
    map_features = runtime.map_features or {}
    route = np.asarray(map_features.get("route_centerline", []))
    return {"agent_slots": int(current.shape[0]) if current.ndim else 0,
            "active_agent_count": int(valid.sum()) if valid.ndim <= 1 else int(valid.any(axis=tuple(range(1, valid.ndim))).sum()),
            "history_valid_entries": int(valid.sum()),
            "agent_history_shape": list(np.asarray(runtime.agent_history).shape),
            "route_points": int(route.shape[0]) if route.ndim else 0,
            "traffic_light_count": len(runtime.traffic_lights)}


def measure_call(call, *, device="cpu", mode="native", context=None,
                 deadline_ms=None, output_prefix=None):
    if mode not in {"native", "attribution"}:
        raise ValueError("mode must be native or attribution")
    if deadline_ms is not None and (not math.isfinite(deadline_ms) or deadline_ms <= 0):
        raise ValueError("deadline_ms must be finite and positive")
    if _ACTIVE.get() is not None:
        raise RuntimeError("nested complete-call measurement is not supported")
    state = {"mode": mode, "device": str(device), "stack": [], "spans": {},
             "core_calls": 0, "diagnostics": {}, "runtime_shape": {}}
    token = _ACTIVE.set(state)
    # Pre-boundary draining is recorded separately, outside the service time.
    # It cannot be interpreted as real deployment queue wait.
    pre_started = time.perf_counter_ns()
    try:
        synchronize(device)
    except BaseException:
        _ACTIVE.reset(token)
        raise
    pre_sync_ms = (time.perf_counter_ns() - pre_started) / 1e6
    started = time.perf_counter_ns()
    result = None
    error = None
    try:
        result = call()
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            synchronize(device)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            diagnostics = state["diagnostics"]
            counts = {k: v for k, v in diagnostics.items()
                      if k.endswith("_count") or k in {
                          "effective_query_count", "configured_decision_budget_atom_count",
                          "upstream_configured_decision_budget_atom_count", "total_sparse_query_count",
                          "proposal_to_certificate_atom_expansion", "fallback_triggered", "fallback_stage"}}
            row = {"schema": "bdse_planning_budget_v1", "mode": mode,
                   "scope": "core_cached_runtime", "device": str(device),
                   "pid": os.getpid(), "thread_id": threading.get_ident(),
                   "start_perf_ns": started, "wall_ms": elapsed,
                   "pre_boundary_sync_ms": pre_sync_ms,
                   "deadline_ms": deadline_ms,
                   "service_exceeded_deadline": None if deadline_ms is None else elapsed > deadline_ms,
                   "deadline_is_enforced": False,
                   "core_calls": state["core_calls"], "cached_plan": state["core_calls"] == 0,
                   "action_index": state.get("action_index"), "counts": counts,
                   "runtime_shape": state["runtime_shape"], "spans": state["spans"],
                   "legacy_host_timings_not_cuda_attributed": {
                       k: diagnostics.get(k, {}) for k in ("timing_core", "model_timing")},
                   "status": "ok" if error is None else "error",
                   "error_type": None if error is None else type(error).__name__,
                   "memory": memory_snapshot(device), **(context or {})}
            if "trajectory" in state:
                import numpy as np
                trajectory = np.ascontiguousarray(state["trajectory"])
                row["trajectory_shape"] = list(trajectory.shape)
                row["trajectory_dtype"] = str(trajectory.dtype)
                row["trajectory_sha256"] = hashlib.sha256(trajectory.tobytes()).hexdigest()
            if output_prefix:
                write_row(output_prefix, row)
        finally:
            _ACTIVE.reset(token)
    return result, row


def latency_stats(values):
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not values:
        return {"n": 0}
    def quantile(p):
        pos = (len(values) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"n": len(values), "mean": sum(values) / len(values),
            "p50": quantile(.5), "p90": quantile(.9), "p95": quantile(.95),
            "p99": quantile(.99), "max": values[-1]}


def queue_diagnostic(service_ms, period_ms, deadline_ms):
    """Idealized single FIFO server; not a simulator or measured live queue."""
    if not math.isfinite(period_ms) or not math.isfinite(deadline_ms) or period_ms <= 0 or deadline_ms <= 0:
        raise ValueError("period and deadline must be positive")
    finish = 0.0
    waits, response = [], []
    for index, duration in enumerate(service_ms):
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid service duration")
        arrival = index * period_ms
        start = max(arrival, finish)
        finish = start + duration
        waits.append(start - arrival)
        response.append(finish - arrival)
    return {"assumption": "idealized_periodic_single_FIFO_server_no_drop",
            "period_ms": period_ms, "deadline_ms": deadline_ms,
            "utilization": sum(service_ms) / (len(service_ms) * period_ms) if service_ms else None,
            "wait_ms": latency_stats(waits), "response_ms": latency_stats(response),
            "miss_rate": sum(x > deadline_ms for x in response) / len(response) if response else None}


def read_runtime_npz(path):
    """Read a strict allowlist of current/past runtime fields, no teacher arrays.

    Cached metadata is deliberately discarded: old metadata may contain teacher
    diagnostics. Candidates/evidence must be regenerated by the runtime core.
    """
    import numpy as np
    from bdse.data.cache_schema import RuntimeFeatures, _json_loads_npz, _string_list
    with np.load(path, allow_pickle=False) as z:
        goal = np.asarray(z["mission_goal"], dtype=np.float32)
        map_features = _json_loads_npz(z, "runtime_map_features_json", {})
        if not map_features:
            raise ValueError(f"missing runtime map features: {path}")
        runtime = RuntimeFeatures(
            ego_history=np.asarray(z["runtime_ego_history"], dtype=np.float32),
            agent_history=np.asarray(z["runtime_agent_history"], dtype=np.float32),
            agent_valid=np.asarray(z["runtime_agent_valid"], dtype=bool),
            current_agents=np.asarray(z["runtime_current_agents"], dtype=np.float32),
            traffic_lights=_json_loads_npz(z, "runtime_traffic_lights_json", []),
            map_features=map_features,
            route_roadblock_ids=_string_list(np.asarray(z["route_roadblock_ids"])),
            mission_goal=None if goal.size == 0 else goal,
            metadata={},
        )
        return runtime, {"scenario_token": str(np.asarray(z["scenario_token"]).item()),
                         "timestamp_us": int(np.asarray(z["timestamp_us"]).item())}
