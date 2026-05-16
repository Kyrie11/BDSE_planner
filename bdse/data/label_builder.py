from __future__ import annotations

from typing import Any

import numpy as np

from bdse.data.cache_schema import LabelOnlyFuture, Sample, pad_array
from bdse.data.feature_builder import _call, _iter_tracked_objects, _state_to_array, _box_to_array, build_runtime_features_from_scenario
from bdse.planner.candidate_generator import generate_candidate_bank
from bdse.planner.evidence_atoms import enumerate_evidence_atoms
from bdse.planner.pair_builder import build_pair_labels
from bdse.planner.teacher_cost import evaluate_teacher_costs
from bdse.utils import transform_states_to_local


def _future_traffic_lights(scenario: Any, iteration: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not cfg.get("runtime", {}).get("use_future_traffic_lights_for_teacher", False):
        return []
    statuses = _call(scenario, ["get_future_traffic_light_status_history", "get_traffic_light_status_future"], iteration, default=[])
    out = []
    for st in statuses or []:
        out.append(
            {
                "lane_connector_id": str(getattr(st, "lane_connector_id", getattr(st, "connector_id", ""))),
                "status": str(getattr(st, "status", getattr(st, "traffic_light_status_type", "unknown"))).lower(),
                "timestamp_us": int(getattr(st, "timestamp", getattr(st, "timestamp_us", 0)) or 0),
                "label_only": True,
            }
        )
    return out


def build_label_future_from_scenario(scenario: Any, iteration: int, cfg: dict[str, Any]) -> LabelOnlyFuture:
    cand_cfg = cfg.get("candidate", {})
    runtime_cfg = cfg.get("runtime", {})
    horizon = float(cand_cfg.get("horizon_s", 8.0))
    step = float(cand_cfg.get("step_s", 0.1))
    T = int(round(horizon / step))
    max_agents = int(runtime_cfg.get("max_agents", 32))

    current_ego = _call(scenario, ["get_ego_state_at_iteration"], iteration, default=None)
    cur = _state_to_array(current_ego)
    origin_xy = cur[:2].copy()
    origin_yaw = float(cur[2])

    future_ego = _call(
        scenario,
        ["get_ego_future_trajectory", "get_future_ego_trajectory"],
        iteration,
        time_horizon=horizon,
        num_samples=T,
        default=[],
    )
    ego_arr = np.asarray([_state_to_array(s) for s in (list(future_ego) if future_ego is not None else [])], dtype=np.float32)
    logged_ego = transform_states_to_local(pad_array(ego_arr, (T, 5)), origin_xy, origin_yaw)

    current_objects = _call(scenario, ["get_tracked_objects_at_iteration"], iteration, default=[])
    cur_objs = _iter_tracked_objects(current_objects)
    raw_current = np.asarray([_box_to_array(o) for o in cur_objs], dtype=np.float32) if cur_objs else np.zeros((0, 10), dtype=np.float32)
    n = min(max_agents, len(raw_current))
    logged_agents = np.zeros((max_agents, T, 5), dtype=np.float32)
    valid = np.zeros((max_agents,), dtype=bool)
    future_objects = _call(
        scenario,
        ["get_future_tracked_objects", "get_tracked_objects_future_trajectory"],
        iteration,
        time_horizon=horizon,
        num_samples=T,
        default=None,
    )
    if future_objects is not None and n > 0:
        token_to_local_idx = {str(getattr(o, "track_token", getattr(o, "token", i))): i for i, o in enumerate(cur_objs[:n])}
        frames = list(future_objects) if not isinstance(future_objects, dict) else []
        for k, frame in enumerate(frames[:T]):
            for obj in _iter_tracked_objects(frame):
                token = str(getattr(obj, "track_token", getattr(obj, "token", "")))
                if token in token_to_local_idx:
                    i = token_to_local_idx[token]
                    arr = _box_to_array(obj)
                    st = np.asarray([arr[0], arr[1], arr[2], arr[3], (k + 1) * step], dtype=np.float32)
                    logged_agents[i, k] = transform_states_to_local(st[None], origin_xy, origin_yaw)[0]
                    valid[i] = True
    for i in range(n):
        if not valid[i]:
            arr = raw_current[i]
            st = np.asarray([arr[0], arr[1], arr[2], arr[3], 0.0], dtype=np.float32)
            st_local = transform_states_to_local(st[None], origin_xy, origin_yaw)[0]
            times = np.arange(1, T + 1, dtype=np.float32) * step
            logged_agents[i, :, 0] = st_local[0] + st_local[3] * np.cos(st_local[2]) * times
            logged_agents[i, :, 1] = st_local[1] + st_local[3] * np.sin(st_local[2]) * times
            logged_agents[i, :, 2] = st_local[2]
            logged_agents[i, :, 3] = st_local[3]
            logged_agents[i, :, 4] = times
            valid[i] = True
    return LabelOnlyFuture(
        logged_ego=logged_ego.astype(np.float32),
        logged_agents=logged_agents.astype(np.float32),
        agent_valid=valid,
        future_traffic_lights=_future_traffic_lights(scenario, iteration, cfg),
        metadata={"iteration": int(iteration), "label_only": True},
    )


def build_training_sample_from_scenario(scenario: Any, iteration: int, cfg: dict[str, Any]) -> Sample:
    runtime = build_runtime_features_from_scenario(scenario, iteration, cfg)
    candidates = generate_candidate_bank(runtime, cfg)
    label_future = build_label_future_from_scenario(scenario, iteration, cfg)
    evidence = enumerate_evidence_atoms(runtime, candidates, cfg)
    teacher = evaluate_teacher_costs(runtime, label_future, candidates, evidence, cfg)
    pairs = build_pair_labels(candidates, teacher, cfg)
    token = str(getattr(scenario, "token", getattr(scenario, "scenario_name", "")))
    timestamp_us = int(getattr(getattr(scenario, "start_time", None), "time_us", 0))
    return Sample(token, timestamp_us, runtime, label_future, candidates, evidence, teacher, pairs)


def build_training_sample_from_runtime_and_future(runtime, label_future, cfg: dict[str, Any], scenario_token: str = "synthetic", timestamp_us: int = 0) -> Sample:
    candidates = generate_candidate_bank(runtime, cfg)
    evidence = enumerate_evidence_atoms(runtime, candidates, cfg)
    teacher = evaluate_teacher_costs(runtime, label_future, candidates, evidence, cfg)
    pairs = build_pair_labels(candidates, teacher, cfg)
    return Sample(scenario_token, int(timestamp_us), runtime, label_future, candidates, evidence, teacher, pairs)
