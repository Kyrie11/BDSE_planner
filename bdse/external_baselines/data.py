from __future__ import annotations

"""Lean data path for the fixed-budget external planning adapters.

The generic BDSE tensorizer intentionally materializes many teacher-only tensors
and dense query placeholders required by the native BDSE training objective.  The
external adapters use a much smaller contract.  This module keeps the exact same
runtime/candidate/evidence semantics while avoiding arrays that are never read by
ExternalBaselineModel.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from bdse.data.cache_schema import (
    CandidateBank,
    EvidenceAtom,
    EvidenceBank,
    LabelOnlyFuture,
    PairLabels,
    RuntimeFeatures,
    Sample,
    TeacherLabels,
    _json_loads_npz,
    _string_list,
)
from bdse.data.tensorizer import (
    _polyline_to_features,
    evidence_arrays,
)
from bdse.planner.selector import _greedy_cover_from_pair_delta


def load_external_training_sample_npz(path: str | Path, *, include_label_future: bool) -> Sample:
    """Load only fields consumed by external adapter training.

    In particular this skips logged-agent futures, candidate metadata, runtime
    metadata/route ids, evidence validity-domain JSON, dense cached query tensors,
    teacher diagnostics/J_evid and pair residuals.  Those fields can dominate CPU
    deserialization even though the external losses never read them.
    """
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        scenario_token = str(z["scenario_token"].item() if z["scenario_token"].shape == () else z["scenario_token"].reshape(-1)[0])
        timestamp_us = int(z["timestamp_us"].item())
        mission_goal = np.asarray(z["mission_goal"], dtype=np.float32)
        mission_goal_val = mission_goal.astype(np.float32) if mission_goal.size else None
        map_features = _json_loads_npz(z, "runtime_map_features_json", {})
        if not map_features:
            map_features = {
                "route_centerline": np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32),
                "route_corridor_width": 4.0,
                "map_valid": False,
            }
        runtime = RuntimeFeatures(
            ego_history=np.asarray(z["runtime_ego_history"], dtype=np.float32),
            agent_history=np.asarray(z["runtime_agent_history"], dtype=np.float32),
            agent_valid=np.asarray(z["runtime_agent_valid"], dtype=bool),
            current_agents=np.asarray(z["runtime_current_agents"], dtype=np.float32),
            traffic_lights=_json_loads_npz(z, "runtime_traffic_lights_json", []),
            map_features=map_features,
            route_roadblock_ids=[],
            mission_goal=mission_goal_val,
            metadata={},
        )

        if include_label_future and "label_logged_ego" in z.files and np.asarray(z["label_logged_ego"]).size:
            # Expert projection only consumes logged_ego.  Avoid loading the much
            # larger logged-agent future and future-traffic-light metadata.
            label_future = LabelOnlyFuture(
                logged_ego=np.asarray(z["label_logged_ego"], dtype=np.float32),
                logged_agents=np.zeros((0,), dtype=np.float32),
                agent_valid=np.zeros((0,), dtype=bool),
                future_traffic_lights=[],
                metadata={},
            )
        else:
            label_future = None

        trajectories = np.asarray(z["candidate_trajectories"], dtype=np.float32)
        K = int(trajectories.shape[0])
        candidates = CandidateBank(
            trajectories=trajectories,
            valid_mask=np.asarray(z["candidate_valid"], dtype=bool),
            maneuver_ids=np.asarray(z["candidate_maneuver_ids"], dtype=np.int64),
            theta=[{} for _ in range(K)],
            dynamic_flags=[{} for _ in range(K)],
            metadata=[{} for _ in range(K)],
        )

        types = _string_list(np.asarray(z["evidence_types"]))
        families = _string_list(np.asarray(z["evidence_families"]))
        hard = np.asarray(z["evidence_is_hard"], dtype=bool)
        costs = np.asarray(z["evidence_budget_costs"], dtype=np.float32)
        active = np.asarray(z["evidence_active"], dtype=bool)
        anchors = _json_loads_npz(z, "evidence_anchors_json", [{} for _ in types])
        cheap = _json_loads_npz(z, "evidence_cheap_features_json", [{} for _ in types])
        lambdas = np.asarray(z["evidence_lambda_weights"], dtype=np.float32) if "evidence_lambda_weights" in z.files else np.ones((len(types),), dtype=np.float32)
        if not isinstance(anchors, list) or len(anchors) != len(types):
            anchors = [{} for _ in types]
        if not isinstance(cheap, list) or len(cheap) != len(types):
            cheap = [{} for _ in types]
        atoms = [
            EvidenceAtom(
                atom_id=int(i),
                type=str(types[i]),
                anchor=dict(anchors[i] or {}),
                budget_cost=float(costs[i]),
                is_hard=bool(hard[i]),
                family=str(families[i]),
                active_mask=bool(active[i]),
                lambda_weight=float(lambdas[i]) if i < len(lambdas) else 1.0,
                cheap_features=dict(cheap[i] or {}),
            )
            for i in range(len(types))
        ]
        proposal = np.asarray(z["evidence_proposal_features"], dtype=np.float32) if "evidence_proposal_features" in z.files else None
        # The external model never reads dense cached query_features; do not even
        # touch the potentially large NPZ member.
        evidence_bank = EvidenceBank(
            atoms=atoms,
            query_features=np.zeros((0,), dtype=np.float32),
            active_mask=active,
            proposal_features=proposal,
        )

        teacher = None
        if "teacher_J_base" in z.files and np.asarray(z["teacher_J_base"]).size:
            j_base = np.asarray(z["teacher_J_base"], dtype=np.float64)
            g_evid = np.asarray(z["teacher_g_evid"], dtype=np.float32)
            j_t = np.asarray(z["teacher_J_T"], dtype=np.float64) if "teacher_J_T" in z.files else np.zeros((K,), dtype=np.float64)
            a_star = int(np.asarray(z["teacher_a_star"]).item()) if "teacher_a_star" in z.files else int(np.argmin(j_t))
            teacher = TeacherLabels(
                J_base=j_base,
                g_evid=g_evid,
                J_evid=np.zeros((K,), dtype=np.float64),
                J_T=j_t,
                a_star=a_star,
                hard_violation_mask=np.zeros((K,), dtype=bool),
                diagnostics={},
            )

        pair_indices = np.asarray(z["pair_indices"], dtype=np.int64)
        if pair_indices.size:
            pair_indices = pair_indices.reshape(-1, 2)
            pair_margins = np.asarray(z["pair_margins"], dtype=np.float32)
            pair_weights = np.asarray(z["pair_weights"], dtype=np.float32)
            pair_valid = np.asarray(z["pair_valid"], dtype=bool)
        else:
            pair_indices = np.zeros((0, 2), dtype=np.int64)
            pair_margins = np.zeros((0,), dtype=np.float32)
            pair_weights = np.zeros((0,), dtype=np.float32)
            pair_valid = np.zeros((0,), dtype=bool)
        pairs = PairLabels(
            pairs=pair_indices,
            margins=pair_margins,
            weights=pair_weights,
            residuals=np.zeros((len(pair_indices),), dtype=np.float32),
            valid_mask=pair_valid,
        )
    return Sample(scenario_token, timestamp_us, runtime, label_future, candidates, evidence_bank, teacher, pairs)


def _fit_last_dim_np(x: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[-1] == dim:
        return arr
    if arr.shape[-1] > dim:
        return arr[..., :dim]
    pad = [(0, 0)] * arr.ndim
    pad[-1] = (0, dim - arr.shape[-1])
    return np.pad(arr, pad, mode="constant")


def _masked_mean_np(x: np.ndarray, mask: np.ndarray | None, axis: int = 0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if mask is None:
        return arr.mean(axis=axis)
    m = np.asarray(mask, dtype=bool)
    mf = m.astype(np.float32)
    while mf.ndim < arr.ndim:
        mf = mf[..., None]
    denom = np.maximum(mf.sum(axis=axis), 1.0)
    return (arr * mf).sum(axis=axis) / denom




def _token_pool_summary_np(tokens: list[np.ndarray], *, feature_dim: int, capacity: int) -> np.ndarray:
    """Match mean/max pooling over padded token tensors without materializing them."""
    if not tokens:
        return np.zeros((2 * feature_dim,), dtype=np.float32)
    stacked = np.stack([_fit_last_dim_np(t, feature_dim) for t in tokens], axis=0).astype(np.float32)
    mean = stacked.mean(axis=0)
    mx = stacked.max(axis=0)
    # Generic tensorizer pads unused token slots with zero before amax.  Preserve
    # that exact behavior for negative-valued channels when capacity is not full.
    if len(tokens) < capacity:
        mx = np.maximum(mx, 0.0)
    return np.concatenate([mean, mx], axis=-1).astype(np.float32)


def _external_map_route_summaries_np(map_features: dict[str, Any], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Exact pooled equivalent of vectorize_map_features for external adapters.

    The generic path allocates max_map*max_poly_pts and
    max_route_tokens*max_poly_pts tensors although the external model immediately
    averages every polyline.  Compute those pooled token means directly.
    """
    mcfg = cfg.get("model", {}) or {}
    rcfg = cfg.get("runtime", {}) or {}
    max_map = int(rcfg.get("max_map_polylines", mcfg.get("max_map_polylines", 128)))
    max_route_total = int(rcfg.get("max_route_points", 256))
    max_poly_pts = int(mcfg.get("max_polyline_points", 64))
    max_route_tokens = int(rcfg.get("max_route_tokens", mcfg.get("max_route_tokens", max(1, int(np.ceil(max_route_total / max(max_poly_pts, 1)))))))
    max_route_tokens = max(1, max_route_tokens)
    map_dim = int(mcfg.get("map_feature_dim", 8))
    route_dim = int(mcfg.get("route_feature_dim", 8))

    map_token_means: list[np.ndarray] = []
    for p in map_features.get("drivable_polygons", []) or []:
        if len(map_token_means) >= max_map:
            break
        xy = np.asarray(p.get("xy", []), dtype=np.float32).reshape(-1, 2)
        if len(xy) >= 2:
            feat = _polyline_to_features(xy[:max_poly_pts], map_dim)
            feat[:, 5 if map_dim > 5 else -1] = 1.0
            pooled = _fit_last_dim_np(feat, 8).sum(axis=0) / float(max(max_poly_pts, 1))
            map_token_means.append(pooled.astype(np.float32))
    for sl in map_features.get("stop_lines", []) or []:
        if len(map_token_means) >= max_map:
            break
        xy = np.asarray(sl.get("xy", []), dtype=np.float32).reshape(-1, 2)
        if len(xy) >= 2:
            feat = _polyline_to_features(xy[:max_poly_pts], map_dim)
            if map_dim > 5:
                feat[:, 5] = 2.0
            if map_dim > 7:
                feat[:, 7] = float(bool(sl.get("red", False)))
            pooled = _fit_last_dim_np(feat, 8).sum(axis=0) / float(max(max_poly_pts, 1))
            map_token_means.append(pooled.astype(np.float32))
    map_summary = _token_pool_summary_np(map_token_means, feature_dim=8, capacity=max_map)

    route = np.asarray(map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    route = route[:max_route_total]
    route_token_means: list[np.ndarray] = []
    if len(route) >= 2:
        step = max(max_poly_pts - 1, 1)
        total_len = max(float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1))), 1e-3)
        cum = np.zeros((len(route),), dtype=np.float32)
        if len(route) > 1:
            cum[1:] = np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1)).astype(np.float32)
        for ridx, start in enumerate(range(0, len(route), step)):
            if ridx >= max_route_tokens:
                break
            seg = route[start : start + max_poly_pts]
            if len(seg) < 2:
                break
            feat = _polyline_to_features(seg, route_dim)
            if route_dim > 4:
                feat[:, 4] = cum[start : start + len(seg)] / total_len
            if route_dim > 5:
                feat[:, 5] = float(map_features.get("speed_limit_mps", 13.4)) / 30.0
            if route_dim > 6:
                feat[:, 6] = float(ridx) / max(float(max_route_tokens - 1), 1.0)
            if route_dim > 7:
                feat[:, 7] = 1.0 - feat[:, 4]
            pooled = _fit_last_dim_np(feat, 8).sum(axis=0) / float(max(max_poly_pts, 1))
            route_token_means.append(pooled.astype(np.float32))
    route_summary = _token_pool_summary_np(route_token_means, feature_dim=8, capacity=max_route_tokens)
    return map_summary, route_summary


def _external_traffic_summary_np(runtime: RuntimeFeatures, cfg: dict[str, Any]) -> np.ndarray:
    max_tl = int((cfg.get("model", {}) or {}).get("max_traffic_tokens", 32))
    rows: list[np.ndarray] = []
    for item in (runtime.traffic_lights or [])[:max_tl]:
        row = np.zeros((12,), dtype=np.float32)
        status = str(item.get("status", "")).lower()
        row[0] = float("red" in status); row[1] = float("yellow" in status); row[2] = float("green" in status)
        xy = np.asarray(item.get("xy", item.get("stop_line_center", [0.0, 0.0])), dtype=np.float32).reshape(-1)
        if xy.size >= 2:
            row[3:5] = xy[:2] / 100.0
        rows.append(row)
    return np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32) if rows else np.zeros((12,), dtype=np.float32)


def _external_goal_np(goal: np.ndarray | None, cfg: dict[str, Any]) -> np.ndarray:
    dim = int((cfg.get("model", {}) or {}).get("goal_feature_dim", 4))
    raw = np.zeros((dim,), dtype=np.float32)
    if goal is not None and np.asarray(goal).size:
        arr = np.asarray(goal, dtype=np.float32).reshape(-1)
        norm = np.asarray([100.0, 100.0, 1.0, 30.0], dtype=np.float32)
        n = min(dim, arr.size, norm.size)
        raw[:n] = arr[:n] / norm[:n]
    return _fit_last_dim_np(raw, 4).astype(np.float32)


def candidate_numeric_features_np(traj: np.ndarray, valid: np.ndarray, dt: float) -> np.ndarray:
    """NumPy equivalent of candidate_numeric_features_torch for compact transfer."""
    traj = np.asarray(traj, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    xy = traj[..., :2]
    T = max(int(traj.shape[1]), 1)
    if T > 1:
        dxy = xy[:, 1:] - xy[:, :-1]
        step = np.linalg.norm(dxy, axis=-1)
    else:
        dxy = np.zeros_like(xy[:, :1])
        step = np.zeros((xy.shape[0], 1), dtype=np.float32)
    path_len = step.sum(axis=-1)
    speed = step / max(float(dt), 1e-3)
    speed_mean = speed.mean(axis=-1) if speed.size else np.zeros_like(path_len)
    speed_max = speed.max(axis=-1) if speed.size else np.zeros_like(path_len)
    speed_final = speed[:, -1] if speed.shape[-1] else np.zeros_like(path_len)
    accel = speed[:, 1:] - speed[:, :-1] if speed.shape[-1] > 1 else np.zeros((xy.shape[0], 1), dtype=np.float32)
    jerk = accel[:, 1:] - accel[:, :-1] if accel.shape[-1] > 1 else np.zeros((xy.shape[0], 1), dtype=np.float32)
    acc_rms = np.sqrt(np.maximum((accel * accel).mean(axis=-1), 0.0)) if accel.size else np.zeros_like(path_len)
    jerk_rms = np.sqrt(np.maximum((jerk * jerk).mean(axis=-1), 0.0)) if jerk.size else np.zeros_like(path_len)
    if traj.shape[-1] > 2:
        yaw = traj[..., 2]
    else:
        mean_dxy = dxy.mean(axis=1) if dxy.size else np.zeros((xy.shape[0], 2), dtype=np.float32)
        yaw0 = np.arctan2(mean_dxy[:, 1], mean_dxy[:, 0]).astype(np.float32)
        yaw = np.repeat(yaw0[:, None], T, axis=1)
    if T > 1:
        raw = yaw[:, 1:] - yaw[:, :-1]
        yaw_delta = np.arctan2(np.sin(raw), np.cos(raw))
    else:
        yaw_delta = np.zeros((xy.shape[0], 1), dtype=np.float32)
    curvature_mean = np.abs(yaw_delta).mean(axis=-1) if yaw_delta.size else np.zeros_like(path_len)
    curvature_max = np.abs(yaw_delta).max(axis=-1) if yaw_delta.size else np.zeros_like(path_len)
    lat = xy[..., 1]
    progress = xy[..., 0]
    feats = np.stack(
        [
            progress[:, -1] / 120.0,
            lat[:, -1] / 20.0,
            np.abs(lat).mean(axis=-1) / 20.0,
            np.abs(lat).max(axis=-1) / 20.0,
            path_len / 120.0,
            speed_mean / 30.0,
            speed_max / 40.0,
            speed_final / 30.0,
            acc_rms / 5.0,
            jerk_rms / 10.0,
            curvature_mean,
            curvature_max,
            np.sin(yaw[:, -1]),
            np.cos(yaw[:, -1]),
            valid.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def external_runtime_to_model_numpy(
    runtime: RuntimeFeatures,
    candidates: CandidateBank,
    evidence_bank: EvidenceBank,
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Compact external-model input contract.

    The neural adapters only consume pooled scene descriptors, candidate numeric
    descriptors and selected-evidence descriptors.  Producing those summaries in
    the DataLoader worker avoids transferring full map/route/history/candidate
    tensors to the GPU on every step.
    """
    ego_hist = np.asarray(runtime.ego_history, dtype=np.float32)
    ego_cur = _fit_last_dim_np(ego_hist[-1], 8).astype(np.float32)

    agent = np.asarray(runtime.agent_history, dtype=np.float32)
    agent_valid = np.asarray(runtime.agent_valid, dtype=bool)
    if agent.size:
        agent_cur = _fit_last_dim_np(agent[:, -1], 8)
        av = agent_valid[:, -1] if agent_valid.ndim == 2 else agent_valid.reshape(-1)
        agent_mean = _masked_mean_np(agent_cur, av, axis=0)
        agent_max = np.where(av[:, None], agent_cur, np.zeros_like(agent_cur)).max(axis=0) if len(agent_cur) else np.zeros((8,), dtype=np.float32)
        agent_summary = np.concatenate([agent_mean, agent_max], axis=-1).astype(np.float32)
    else:
        agent_summary = np.zeros((16,), dtype=np.float32)

    map_summary, route_summary = _external_map_route_summaries_np(runtime.map_features or {}, cfg)
    traffic_summary = _external_traffic_summary_np(runtime, cfg)
    goal = _external_goal_np(runtime.mission_goal, cfg)

    ev = evidence_arrays(
        evidence_bank, candidates, runtime, cfg,
        include_dense_query=False, include_query_tensor=False,
    )
    # External adapters never consume dense evidence-query tensors.  The generic
    # BDSE path keeps them by default, while this lean path skips allocation.
    keep_ev = {
        k: ev[k]
        for k in (
            "evidence_features",
            "evidence_proposal_features",
            "evidence_active",
            "evidence_family_ids",
            "evidence_budget_costs",
        )
    }
    out: dict[str, np.ndarray] = {
        "external_ego_current": ego_cur,
        "external_agent_summary": agent_summary,
        "external_map_summary": map_summary,
        "external_route_summary": route_summary,
        "external_traffic_summary": traffic_summary,
        "mission_goal": goal,
        "candidate_numeric_features": candidate_numeric_features_np(
            candidates.trajectories, candidates.valid_mask, float(cfg.get("candidate", {}).get("step_s", 0.1))
        ),
        "candidate_valid": np.asarray(candidates.valid_mask, dtype=bool),
        "candidate_maneuver_ids": np.asarray(candidates.maneuver_ids, dtype=np.int64),
    }
    out.update(keep_ev)
    return out


def expert_candidate_targets(sample: Sample) -> tuple[int, np.ndarray]:
    if sample.label_future is None or np.asarray(sample.label_future.logged_ego).size == 0:
        raise ValueError(
            "planner_supervision=expert_imitation requires label_logged_ego in the training/validation cache; "
            f"missing for scenario={sample.scenario_token} timestamp_us={sample.timestamp_us}"
        )
    gt = np.asarray(sample.label_future.logged_ego, dtype=np.float32)
    trajs = np.asarray(sample.candidates.trajectories, dtype=np.float32)
    valid = np.asarray(sample.candidates.valid_mask, dtype=bool).reshape(-1)
    n = min(int(gt.shape[0]), int(trajs.shape[1]))
    if n <= 0 or not valid.any():
        raise ValueError(f"invalid expert/candidate tensors for scenario={sample.scenario_token}")
    ade = np.linalg.norm(trajs[:, :n, :2] - gt[None, :n, :2], axis=-1).mean(axis=1).astype(np.float32)
    ade[~valid] = np.inf
    if not np.isfinite(ade).any():
        raise ValueError(f"no finite candidate can represent expert future for scenario={sample.scenario_token}")
    return int(np.nanargmin(ade)), ade


def _oracle_selected_mask(sample: Sample, cfg: dict[str, Any]) -> np.ndarray:
    """Exact selected-mask subset of the generic BDSE teacher tensorizer.

    The proposal BCE only needs the greedy selected atom mask; computing the
    additional decisive/family/criticality labels in sample_to_model_inputs is
    wasted work for these adapters.
    """
    if sample.teacher is None or sample.pairs is None:
        raise ValueError("external proposal supervision requires teacher/pair labels")
    Emax = int(cfg.get("evidence", {}).get("max_atoms", 128))
    Pmax = int(cfg.get("pairs", {}).get("target_max", 256))
    oracle = np.zeros((Emax,), dtype=bool)
    n = min(Pmax, len(sample.pairs.pairs))
    if n <= 0:
        return oracle
    pairs = np.asarray(sample.pairs.pairs[:n], dtype=np.int64).reshape(-1, 2)
    margins = np.asarray(sample.pairs.margins[:n], dtype=np.float32)
    weights = np.asarray(sample.pairs.weights[:n], dtype=np.float32)
    a, b = pairs[:, 0], pairs[:, 1]
    base = np.asarray(sample.teacher.J_base, dtype=np.float64)
    g_src = np.asarray(sample.teacher.g_evid, dtype=np.float32)
    Etrue = min(Emax, g_src.shape[0])
    base_delta = (base[b] - base[a]).astype(np.float32)
    atom_delta = np.zeros((Emax, n), dtype=np.float32)
    atom_delta[:Etrue] = g_src[:Etrue, b] - g_src[:Etrue, a]
    caps = np.maximum(margins, 0.0).astype(np.float32)
    e_cost = np.ones((Emax,), dtype=np.float32)
    src_cost = sample.evidence_bank.budget_costs()
    e_cost[: min(Emax, len(src_cost))] = src_cost[:Emax]
    e_active = np.zeros((Emax,), dtype=bool)
    src_active = np.asarray(sample.evidence_bank.active_mask, dtype=bool)
    e_active[: min(Emax, len(src_active))] = src_active[:Emax]
    selected, _, _ = _greedy_cover_from_pair_delta(
        atom_delta,
        base_delta,
        caps,
        weights,
        e_cost,
        float(cfg.get("evidence", {}).get("budget", 16)),
        e_active,
    )
    oracle[np.asarray(selected, dtype=np.int64)] = True
    return oracle


def external_sample_to_model_inputs(sample: Sample, cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    arrays = external_runtime_to_model_numpy(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
    ecfg = cfg.get("external_baseline", {}) or {}
    supervision = str(ecfg.get("planner_supervision", "teacher_cost")).strip().lower()
    weights = ecfg.get("loss_weights", {}) or {}

    if supervision == "expert_imitation":
        target_idx, target_cost = expert_candidate_targets(sample)
        arrays["expert_candidate_index"] = np.asarray(target_idx, dtype=np.int64)
        arrays["expert_candidate_cost"] = np.asarray(target_cost, dtype=np.float32)
    else:
        if sample.teacher is None:
            raise ValueError("teacher-cost supervision requires teacher labels")
        arrays["teacher_a_star"] = np.asarray(sample.teacher.a_star, dtype=np.int64)
        arrays["teacher_J_T"] = np.asarray(sample.teacher.J_T, dtype=np.float32)

    if float(weights.get("proposal", 0.25)) != 0.0:
        arrays["oracle_selected_mask"] = _oracle_selected_mask(sample, cfg)

    # Pair tensors are omitted entirely when their loss weight is zero (the
    # published/fair configs all use pair=0).  This removes Pmax-sized transfers.
    if float(weights.get("pair", 0.0)) != 0.0 and sample.pairs is not None:
        Pmax = int(cfg.get("pairs", {}).get("target_max", 256))
        pairs = np.zeros((Pmax, 2), dtype=np.int64)
        valid = np.zeros((Pmax,), dtype=bool)
        n = min(Pmax, len(sample.pairs.pairs))
        if n:
            pairs[:n] = sample.pairs.pairs[:n]
            valid[:n] = sample.pairs.valid_mask[:n]
        arrays["pair_indices"] = pairs
        arrays["pair_valid"] = valid

    tensors: dict[str, torch.Tensor] = {}
    for k, v in arrays.items():
        arr = np.asarray(v)
        if arr.dtype == np.bool_:
            tensors[k] = torch.from_numpy(np.asarray(arr, dtype=bool))
        elif np.issubdtype(arr.dtype, np.integer):
            tensors[k] = torch.from_numpy(np.asarray(arr, dtype=np.int64))
        else:
            tensors[k] = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    return tensors
