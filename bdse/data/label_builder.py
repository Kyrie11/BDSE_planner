from __future__ import annotations

from typing import Any
import os
import time
import numpy as np

from bdse.data.cache_schema import LabelOnlyFuture, Sample, pad_array
from bdse.data.feature_builder import _call, _state_to_array, _object_token, build_runtime_features_from_scenario, resort_runtime_agents_for_candidates, cached_current_tracked_frame, cached_current_ego_state, cached_ego_window, cached_tracked_window, boxes_global_to_local
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


def build_label_future_from_scenario(scenario: Any, iteration: int, cfg: dict[str, Any], runtime=None) -> LabelOnlyFuture:
    cand_cfg = cfg.get("candidate", {})
    runtime_cfg = cfg.get("runtime", {})
    pcfg = cfg.get("preprocess", {})
    profile = bool(pcfg.get("profile", False)) or bool(os.environ.get("BDSE_PROFILE_PREPROCESS"))
    t0 = time.perf_counter()
    marks: dict[str, float] = {}

    def mark(name: str) -> None:
        if profile:
            marks[name] = time.perf_counter()

    horizon = float(cand_cfg.get("horizon_s", 8.0))
    step = float(cand_cfg.get("step_s", 0.1))
    T = int(round(horizon / step))
    max_agents = int(runtime_cfg.get("max_agents", 32))

    cur = cached_current_ego_state(scenario, iteration, cfg)
    origin_xy = cur[:2].copy()
    origin_yaw = float(cur[2])
    mark("current_ego")

    future_ego_states, ego_stats = cached_ego_window(
        scenario,
        iteration,
        cfg,
        direction="future",
        time_horizon=horizon,
        num_samples=T,
        step_s=step,
    )
    ego_arr = np.asarray(future_ego_states, dtype=np.float32)
    logged_ego = transform_states_to_local(pad_array(ego_arr, (T, 5)), origin_xy, origin_yaw)
    mark("future_ego")

    if runtime is not None:
        selected_tokens = list(getattr(runtime, "metadata", {}).get("selected_agent_tokens", []))
        valid_rows = np.flatnonzero(np.asarray(runtime.agent_valid, dtype=bool))[:max_agents]
        raw_current = np.asarray(runtime.current_agents[valid_rows], dtype=np.float32) if valid_rows.size else np.zeros((0, 10), dtype=np.float32)
        if not selected_tokens:
            selected_tokens = [str(i) for i in valid_rows.tolist()]
    else:
        current_frame = cached_current_tracked_frame(scenario, iteration, cfg)
        token_to_row = {str(t): i for i, t in enumerate(current_frame.tokens)}
        selected_tokens = list(current_frame.tokens)[:max_agents]
        selected_rows = [token_to_row[str(t)] for t in selected_tokens if str(t) in token_to_row][:max_agents]
        raw_current_global = current_frame.boxes[selected_rows] if selected_rows else np.zeros((0, 10), dtype=np.float32)
        raw_current = boxes_global_to_local(raw_current_global, origin_xy, origin_yaw)
    n = min(max_agents, len(raw_current))
    mark("current_agents")

    logged_agents = np.zeros((max_agents, T, 5), dtype=np.float32)
    valid = np.zeros((max_agents,), dtype=bool)
    logged_mask = np.zeros((max_agents,), dtype=bool)
    cv_fallback_mask = np.zeros((max_agents,), dtype=bool)
    future_stats: dict[str, float] = {"cache_hit_frames": 0.0, "cache_miss_frames": 0.0, "bulk_call": 0.0, "coalesced_recheck": 0.0, "individual_frame_calls": 0.0}
    agent_future_mode = str(pcfg.get("label_agent_future_mode", "logged")).lower()

    if agent_future_mode not in {"cv", "constant_velocity", "current_cv", "proxy", "skip_logged"}:
        future_frames, future_stats = cached_tracked_window(
            scenario,
            iteration,
            cfg,
            direction="future",
            time_horizon=horizon,
            num_samples=T,
            step_s=step,
        )
        mark("future_agents_fetch")
        if n > 0:
            # Use the same token canonicalization as runtime agent selection.  Only the
            # selected agent slots are label-relevant, so transform those rows instead of
            # converting every tracked object in every future frame.  This preserves the
            # exact selected-agent labels and avoids O(all_objects*T) local transforms on
            # dense nuPlan frames.
            token_to_local_idx = {str(tok): i for i, tok in enumerate(selected_tokens[:n])}
            for k, frame in enumerate(future_frames[:T]):
                if not frame.tokens or frame.boxes.size == 0:
                    continue
                src_rows: list[int] = []
                dst_rows: list[int] = []
                max_box = int(frame.boxes.shape[0])
                for j, token in enumerate(frame.tokens):
                    if j >= max_box:
                        break
                    i = token_to_local_idx.get(str(token))
                    if i is not None:
                        src_rows.append(j)
                        dst_rows.append(i)
                if not src_rows:
                    continue
                boxes_local = boxes_global_to_local(frame.boxes[np.asarray(src_rows, dtype=np.int64)], origin_xy, origin_yaw)
                for row, i in enumerate(dst_rows):
                    arr = boxes_local[row]
                    logged_agents[i, k] = np.asarray([arr[0], arr[1], arr[2], arr[3], (k + 1) * step], dtype=np.float32)
                    valid[i] = True
                    logged_mask[i] = True
    else:
        # Fast verification mode: keep logged ego imitation exact, but avoid the
        # expensive nuPlan future tracked-object window.  Interaction atoms then use
        # the same constant-velocity selected-agent proxy already used as the runtime
        # fallback/response mode, so runtime/label separation remains intact.
        future_stats["cv_proxy"] = 1.0
        mark("future_agents_fetch")
    mark("future_agents_project")

    times = np.arange(1, T + 1, dtype=np.float32) * step
    for i in range(n):
        if not valid[i]:
            st_local = raw_current[i, [0, 1, 2, 3, 4]]
            logged_agents[i, :, 0] = st_local[0] + st_local[3] * np.cos(st_local[2]) * times
            logged_agents[i, :, 1] = st_local[1] + st_local[3] * np.sin(st_local[2]) * times
            logged_agents[i, :, 2] = st_local[2]
            logged_agents[i, :, 3] = st_local[3]
            logged_agents[i, :, 4] = times
            valid[i] = True
            cv_fallback_mask[i] = True
    mark("cv_fallback")
    future_traffic_lights = _future_traffic_lights(scenario, iteration, cfg)
    mark("future_traffic_lights")

    metadata = {
        "iteration": int(iteration),
        "label_only": True,
        "selected_agent_count": int(n),
        "agent_future_logged_mask": logged_mask.tolist(),
        "agent_future_cv_fallback_mask": cv_fallback_mask.tolist(),
        "agent_future_logged_count": int(logged_mask.sum()),
        "agent_future_cv_fallback_count": int(cv_fallback_mask.sum()),
        "agent_future_mode": agent_future_mode,
    }
    if profile:
        prev = t0
        breakdown: dict[str, float] = {}
        for name in ["current_ego", "future_ego", "current_agents", "future_agents_fetch", "future_agents_project", "cv_fallback", "future_traffic_lights"]:
            if name in marks:
                breakdown[name] = float(marks[name] - prev)
                prev = marks[name]
        breakdown["future_agent_cache_hit_frames"] = float(future_stats.get("cache_hit_frames", 0))
        breakdown["future_agent_cache_miss_frames"] = float(future_stats.get("cache_miss_frames", 0))
        breakdown["future_agent_bulk_call"] = float(future_stats.get("bulk_call", 0))
        breakdown["future_agent_coalesced_recheck"] = float(future_stats.get("coalesced_recheck", 0))
        breakdown["future_agent_individual_calls"] = float(future_stats.get("individual_frame_calls", 0))
        breakdown["future_agent_cv_proxy"] = float(future_stats.get("cv_proxy", 0))
        breakdown["future_ego_cache_hit_frames"] = float(ego_stats.get("cache_hit_frames", 0))
        breakdown["future_ego_cache_miss_frames"] = float(ego_stats.get("cache_miss_frames", 0))
        breakdown["future_ego_bulk_call"] = float(ego_stats.get("bulk_call", 0))
        breakdown["future_ego_coalesced_recheck"] = float(ego_stats.get("coalesced_recheck", 0))
        breakdown["future_ego_individual_calls"] = float(ego_stats.get("individual_frame_calls", 0))
        metadata["profile_label_future"] = breakdown

    return LabelOnlyFuture(
        logged_ego=logged_ego.astype(np.float32),
        logged_agents=logged_agents.astype(np.float32),
        agent_valid=valid,
        future_traffic_lights=future_traffic_lights,
        metadata=metadata,
    )


def build_training_sample_from_scenario(scenario: Any, iteration: int, cfg: dict[str, Any]) -> Sample:
    pcfg = cfg.get("preprocess", {})
    profile = bool(pcfg.get("profile", False)) or bool(os.environ.get("BDSE_PROFILE_PREPROCESS"))
    threshold_s = float(pcfg.get("profile_threshold_s", 2.0))
    t0 = time.perf_counter()
    marks: dict[str, float] = {}

    def mark(name: str) -> None:
        if profile:
            marks[name] = time.perf_counter()

    runtime = build_runtime_features_from_scenario(scenario, iteration, cfg)
    mark("runtime")
    candidates = generate_candidate_bank(runtime, cfg)
    mark("candidates")
    if bool(pcfg.get("candidate_aware_agent_selection", False)):
        # Pass 1 builds a candidate bank so agent selection can be ordered by
        # candidate proximity.  Re-select agents from the all-agent arrays cached
        # in the first runtime pass, but do not re-extract ego/map/drivable polygon
        # features.  This preserves candidate-aware quality while removing the
        # dominant duplicate map-API cost.
        runtime = resort_runtime_agents_for_candidates(runtime, candidates, cfg)
        mark("runtime_resort")
        candidates = generate_candidate_bank(runtime, cfg)
        mark("candidates_resort")
    label_future = build_label_future_from_scenario(scenario, iteration, cfg, runtime=runtime)
    mark("label_future")
    evidence = enumerate_evidence_atoms(runtime, candidates, cfg)
    mark("evidence")
    teacher = evaluate_teacher_costs(runtime, label_future, candidates, evidence, cfg)
    mark("teacher")
    pairs = build_pair_labels(candidates, teacher, cfg)
    mark("pairs")
    token = str(getattr(scenario, "token", getattr(scenario, "scenario_name", "")))
    ts_obj = _call(scenario, ["get_time_point", "get_timestamp_at_iteration"], iteration, default=getattr(scenario, "start_time", None))
    timestamp_us = int(getattr(ts_obj, "time_us", getattr(ts_obj, "timestamp_us", ts_obj if isinstance(ts_obj, (int, float)) else 0)) or 0)
    total = time.perf_counter() - t0
    if profile and total >= threshold_s:
        prev = t0
        parts = []
        for name in ["runtime", "candidates", "runtime_resort", "candidates_resort", "label_future", "evidence", "teacher", "pairs"]:
            if name in marks:
                parts.append(f"{name}={marks[name] - prev:.3f}s")
                prev = marks[name]
        rt_profile = getattr(runtime, "metadata", {}).get("profile_runtime", {}) if runtime is not None else {}
        if rt_profile:
            rt_parts = ",".join(f"{k}={float(v):.3f}s" for k, v in sorted(rt_profile.items()))
            parts.append(f"runtime_breakdown=[{rt_parts}]")
        lf_profile = getattr(label_future, "metadata", {}).get("profile_label_future", {}) if label_future is not None else {}
        if lf_profile:
            lf_parts = ",".join(f"{k}={float(v):.3f}s" for k, v in sorted(lf_profile.items()))
            parts.append(f"label_future_breakdown=[{lf_parts}]")
        parts.append(f"total={total:.3f}s")
        print(f"[bdse][profile] token={token} it={iteration} " + " ".join(parts), flush=True)
    return Sample(token, timestamp_us, runtime, label_future, candidates, evidence, teacher, pairs)


def build_training_sample_from_runtime_and_future(runtime, label_future, cfg: dict[str, Any], scenario_token: str = "synthetic", timestamp_us: int = 0) -> Sample:
    candidates = generate_candidate_bank(runtime, cfg)
    evidence = enumerate_evidence_atoms(runtime, candidates, cfg)
    teacher = evaluate_teacher_costs(runtime, label_future, candidates, evidence, cfg)
    pairs = build_pair_labels(candidates, teacher, cfg)
    return Sample(scenario_token, int(timestamp_us), runtime, label_future, candidates, evidence, teacher, pairs)
