from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bdse.data.cache_schema import CandidateBank, EvidenceBank, RuntimeFeatures, Sample
from bdse.planner.evidence_queries import FAMILY_NAMES, TYPE_NAMES, PROPOSAL_FEATURE_DIM, compute_proposal_features
from bdse.planner.evidence_atoms import ATOM_QUERY_DIM, compute_query_features
from bdse.planner.selector import _greedy_cover_from_pair_support


def _pad_2d(arr: np.ndarray, shape: tuple[int, int], value: float = 0.0) -> np.ndarray:
    out = np.full(shape, value, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, shape[1]) if arr.size else np.zeros((0, shape[1]), dtype=np.float32)
    rows = min(shape[0], arr.shape[0])
    cols = min(shape[1], arr.shape[1]) if arr.ndim >= 2 else 0
    if rows and cols:
        out[:rows, :cols] = arr[:rows, :cols]
    return out


def _polyline_to_features(xy: np.ndarray, dim: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    out = np.zeros((xy.shape[0], dim), dtype=np.float32)
    if xy.size == 0:
        return out
    delta = np.zeros_like(xy)
    if len(xy) > 1:
        delta[:-1] = xy[1:] - xy[:-1]
        delta[-1] = delta[-2]
    out[:, 0:2] = xy / 100.0
    out[:, 2:4] = delta / 10.0
    if dim > 4:
        out[:, 4] = np.linspace(0.0, 1.0, len(xy), dtype=np.float32)
    return out


def vectorize_map_features(map_features: dict[str, Any], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mcfg = cfg.get("model", {})
    max_map = int(cfg.get("runtime", {}).get("max_map_polylines", mcfg.get("max_map_polylines", 128)))
    max_route_pts = min(int(cfg.get("runtime", {}).get("max_route_points", 256)), int(mcfg.get("max_polyline_points", 64)))
    max_poly_pts = int(mcfg.get("max_polyline_points", 64))
    map_dim = int(mcfg.get("map_feature_dim", 8))
    route_dim = int(mcfg.get("route_feature_dim", 8))
    tl_dim = int(mcfg.get("traffic_feature_dim", 12))
    max_tl = int(mcfg.get("max_traffic_tokens", 32))

    map_tokens = np.zeros((max_map, max_poly_pts, map_dim), dtype=np.float32)
    map_valid = np.zeros((max_map,), dtype=bool)
    idx = 0
    for p in map_features.get("drivable_polygons", []) or []:
        if idx >= max_map:
            break
        xy = np.asarray(p.get("xy", []), dtype=np.float32).reshape(-1, 2)
        if len(xy) >= 2:
            feat = _polyline_to_features(xy[:max_poly_pts], map_dim)
            feat[:, 5 if map_dim > 5 else -1] = 1.0
            map_tokens[idx, : len(feat)] = feat
            map_valid[idx] = True
            idx += 1
    for sl in map_features.get("stop_lines", []) or []:
        if idx >= max_map:
            break
        xy = np.asarray(sl.get("xy", []), dtype=np.float32).reshape(-1, 2)
        if len(xy) >= 2:
            feat = _polyline_to_features(xy[:max_poly_pts], map_dim)
            if map_dim > 5:
                feat[:, 5] = 2.0
            if map_dim > 7:
                feat[:, 7] = float(bool(sl.get("red", False)))
            map_tokens[idx, : len(feat)] = feat
            map_valid[idx] = True
            idx += 1

    route = np.asarray(map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    route = route[:max_route_pts]
    route_tokens = np.zeros((1, max_route_pts, route_dim), dtype=np.float32)
    route_valid = np.zeros((1,), dtype=bool)
    if len(route) >= 2:
        route_feat = _polyline_to_features(route, route_dim)
        if route_dim > 5:
            route_feat[:, 5] = float(map_features.get("speed_limit_mps", 13.4)) / 30.0
        route_tokens[0, : len(route_feat)] = route_feat
        route_valid[0] = True

    tl = np.zeros((max_tl, tl_dim), dtype=np.float32)
    tl_valid = np.zeros((max_tl,), dtype=bool)
    for i, item in enumerate((map_features.get("traffic_lights", []) or [])[:max_tl]):
        tl_valid[i] = True
        status = str(item.get("status", "")).lower()
        tl[i, 0] = float("red" in status)
        tl[i, 1] = float("yellow" in status)
        tl[i, 2] = float("green" in status)
    return map_tokens, map_valid, route_tokens, route_valid, tl, tl_valid


def vectorize_traffic_lights(runtime: RuntimeFeatures, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    tl_dim = int(cfg.get("model", {}).get("traffic_feature_dim", 12))
    max_tl = int(cfg.get("model", {}).get("max_traffic_tokens", 32))
    out = np.zeros((max_tl, tl_dim), dtype=np.float32)
    valid = np.zeros((max_tl,), dtype=bool)
    for i, item in enumerate((runtime.traffic_lights or [])[:max_tl]):
        valid[i] = True
        status = str(item.get("status", "")).lower()
        out[i, 0] = float("red" in status)
        out[i, 1] = float("yellow" in status)
        out[i, 2] = float("green" in status)
        xy = np.asarray(item.get("xy", item.get("stop_line_center", [0.0, 0.0])), dtype=np.float32).reshape(-1)
        if xy.size >= 2:
            out[i, 3:5] = xy[:2] / 100.0
    return out, valid


def vectorize_mission_goal(goal: np.ndarray | None, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    dim = int(cfg.get("model", {}).get("goal_feature_dim", 4))
    out = np.zeros((dim,), dtype=np.float32)
    valid = np.asarray(False)
    if goal is not None and np.asarray(goal).size:
        arr = np.asarray(goal, dtype=np.float32).reshape(-1)
        out[: min(dim, arr.size)] = arr[: min(dim, arr.size)] / np.asarray([100.0, 100.0, 1.0, 30.0][:dim], dtype=np.float32)
        valid = np.asarray(True)
    return out, valid


def evidence_arrays(evidence_bank: EvidenceBank, candidates: CandidateBank, runtime: RuntimeFeatures, cfg: dict[str, Any], include_dense_query: bool) -> dict[str, np.ndarray]:
    Emax = int(cfg.get("evidence", {}).get("max_atoms", 128))
    efd = int(cfg.get("model", {}).get("evidence_feature_dim", 24))
    qfd = int(cfg.get("model", {}).get("query_feature_dim", ATOM_QUERY_DIM))
    prop_dim = int(cfg.get("model", {}).get("proposal_feature_dim", PROPOSAL_FEATURE_DIM))
    E = min(Emax, evidence_bank.E)
    feat = np.zeros((Emax, efd), dtype=np.float32)
    prop = np.zeros((Emax, prop_dim), dtype=np.float32)
    type_ids = np.zeros((Emax,), dtype=np.int64)
    fam_ids = np.zeros((Emax,), dtype=np.int64)
    active = np.zeros((Emax,), dtype=bool)
    budget = np.ones((Emax,), dtype=np.float32)
    if evidence_bank.proposal_features is None or np.asarray(evidence_bank.proposal_features).shape[0] < E:
        proposal_source = compute_proposal_features(evidence_bank.atoms, candidates, runtime, cfg)
    else:
        proposal_source = np.asarray(evidence_bank.proposal_features, dtype=np.float32)
    for i, atom in enumerate(evidence_bank.atoms[:E]):
        active[i] = bool(atom.active_mask)
        budget[i] = float(atom.budget_cost)
        type_ids[i] = TYPE_NAMES.get(atom.type, 0)
        fam_ids[i] = FAMILY_NAMES.get(atom.family, 0)
        feat[i, 0] = float(atom.is_hard)
        feat[i, 1] = float(atom.budget_cost)
        feat[i, 2] = float(atom.lambda_weight)
        feat[i, 3] = float(atom.cheap_features.get("ego_distance", 0.0)) / 100.0
        feat[i, 4] = float(atom.cheap_features.get("route_distance", 0.0)) / 50.0
        feat[i, 5] = float(atom.cheap_features.get("route_progress", 0.0)) / 100.0
        if "current_state" in atom.anchor:
            st = np.asarray(atom.anchor["current_state"], dtype=np.float32).reshape(-1)
            # normalized anchor state: x/y/yaw/speed/vx/vy/length/width where available
            if st.size >= 2:
                feat[i, 6:8] = st[:2] / 100.0
            if st.size >= 4:
                feat[i, 8] = st[2]
                feat[i, 9] = st[3] / 30.0
            if st.size >= 7:
                feat[i, 10:12] = st[5:7] / 30.0
            if st.size >= 9:
                feat[i, 12:14] = st[7:9] / 10.0
        if "stop_line_xy" in atom.anchor:
            xy = np.asarray(atom.anchor.get("stop_line_xy", []), dtype=np.float32).reshape(-1, 2)
            if len(xy):
                feat[i, 14:16] = xy.mean(axis=0) / 100.0
        if "speed_limit_mps" in atom.anchor:
            feat[i, 16] = float(atom.anchor.get("speed_limit_mps", 0.0)) / 30.0
        if i < proposal_source.shape[0]:
            cols = min(prop_dim, proposal_source.shape[1])
            prop[i, :cols] = proposal_source[i, :cols]
    query = np.zeros((Emax, candidates.K, qfd), dtype=np.float32)
    if include_dense_query:
        q_src = np.asarray(evidence_bank.query_features, dtype=np.float32)
        if q_src.ndim != 3 or q_src.shape[0] < E or q_src.shape[1] < candidates.K:
            q_src = compute_query_features(evidence_bank.atoms[:E], candidates, runtime, cfg)
        q = q_src[:E, : candidates.K, :qfd]
        query[: q.shape[0], : q.shape[1], : q.shape[2]] = q
    return {
        "evidence_features": feat,
        "evidence_proposal_features": prop,
        "evidence_query_features": query,
        "evidence_active": active,
        "evidence_type_ids": type_ids,
        "evidence_family_ids": fam_ids,
        "evidence_budget_costs": budget,
    }


def runtime_to_model_numpy(runtime: RuntimeFeatures, candidates: CandidateBank, evidence_bank: EvidenceBank, cfg: dict[str, Any], include_dense_query: bool = False) -> dict[str, np.ndarray]:
    map_polys, map_valid, route, route_valid, _, _ = vectorize_map_features(runtime.map_features or {}, cfg)
    traffic, traffic_valid = vectorize_traffic_lights(runtime, cfg)
    goal, goal_valid = vectorize_mission_goal(runtime.mission_goal, cfg)
    out: dict[str, np.ndarray] = {
        "ego_history": np.asarray(runtime.ego_history, dtype=np.float32),
        "agent_history": np.asarray(runtime.agent_history, dtype=np.float32),
        "agent_valid": np.asarray(runtime.agent_valid, dtype=bool),
        "candidate_trajectories": np.asarray(candidates.trajectories, dtype=np.float32),
        "candidate_valid": np.asarray(candidates.valid_mask, dtype=bool),
        "candidate_maneuver_ids": np.asarray(candidates.maneuver_ids, dtype=np.int64),
        "map_polylines": map_polys,
        "map_polyline_valid": map_valid,
        "route_polylines": route,
        "route_token_valid": route_valid,
        "traffic_control_tokens": traffic,
        "traffic_token_valid": traffic_valid,
        "mission_goal": goal,
        "mission_goal_valid": goal_valid,
    }
    out.update(evidence_arrays(evidence_bank, candidates, runtime, cfg, include_dense_query=include_dense_query))
    return out


def sample_to_model_inputs(sample: Sample, cfg: dict[str, Any], include_teacher: bool = True, include_dense_query: bool = True) -> dict[str, torch.Tensor]:
    arrays = runtime_to_model_numpy(sample.runtime, sample.candidates, sample.evidence_bank, cfg, include_dense_query=include_dense_query)
    if include_teacher:
        if sample.teacher is None or sample.pairs is None:
            raise ValueError("Training sample requires teacher and pair labels")
        Emax = int(cfg.get("evidence", {}).get("max_atoms", 128))
        K = int(cfg.get("candidate", {}).get("K", sample.candidates.K))
        Pmax = int(cfg.get("pairs", {}).get("target_max", 256))
        if sample.candidates.K != K:
            raise ValueError(f"Candidate count mismatch: cache has K={sample.candidates.K}, config expects K={K}.")
        g = np.zeros((Emax, K), dtype=np.float32)
        g[: min(Emax, sample.teacher.g_evid.shape[0]), : min(K, sample.teacher.g_evid.shape[1])] = sample.teacher.g_evid[:Emax, :K]
        pairs = np.zeros((Pmax, 2), dtype=np.int64)
        valid = np.zeros((Pmax,), dtype=bool)
        margins = np.zeros((Pmax,), dtype=np.float32)
        weights = np.zeros((Pmax,), dtype=np.float32)
        residuals = np.zeros((Pmax,), dtype=np.float32)
        n = min(Pmax, len(sample.pairs.pairs))
        if n:
            pairs[:n] = sample.pairs.pairs[:n]
            valid[:n] = sample.pairs.valid_mask[:n]
            margins[:n] = sample.pairs.margins[:n]
            weights[:n] = sample.pairs.weights[:n]
            residuals[:n] = sample.pairs.residuals[:n]
        arrays.update({
            "teacher_J_base": sample.teacher.J_base.astype(np.float32),
            "teacher_g_evid": g,
            "teacher_J_T": sample.teacher.J_T.astype(np.float32),
            "teacher_a_star": np.asarray(sample.teacher.a_star, dtype=np.int64),
            "teacher_hard_violation": sample.teacher.hard_violation_mask.astype(bool),
            "pair_indices": pairs,
            "pair_valid": valid,
            "pair_margins": margins,
            "pair_weights": weights,
            "pair_residuals": residuals,
        })
        # Oracle marginal-gain labels for the proposal head.  This mirrors the
        # certificate objective instead of using mean positive support: at every
        # greedy step, store the actual marginal gain of the selected atom.
        crit = np.zeros((Emax,), dtype=np.float32)
        oracle = np.zeros((Emax,), dtype=bool)
        if n:
            Etrue = min(Emax, sample.teacher.g_evid.shape[0])
            a = pairs[:n, 0]
            b = pairs[:n, 1]
            base_support = np.maximum(sample.teacher.J_base[b] - sample.teacher.J_base[a], 0.0).astype(np.float32)
            atom_support = np.zeros((Emax, n), dtype=np.float32)
            atom_support[:Etrue] = np.maximum(sample.teacher.g_evid[:Etrue, b] - sample.teacher.g_evid[:Etrue, a], 0.0)
            caps = np.maximum(margins[:n], 0.0).astype(np.float32)
            e_cost = np.ones((Emax,), dtype=np.float32)
            e_cost[: min(Emax, sample.evidence_bank.budget_costs().shape[0])] = sample.evidence_bank.budget_costs()[:Emax]
            e_active = np.zeros((Emax,), dtype=bool)
            e_active[: min(Emax, sample.evidence_bank.active_mask.shape[0])] = sample.evidence_bank.active_mask[:Emax]
            selected, _, _ = _greedy_cover_from_pair_support(
                atom_support, base_support, caps, weights[:n], e_cost, float(cfg.get("evidence", {}).get("budget", 16)), e_active
            )
            # Recompute the greedy trace to expose selected-step marginal gains.
            support = base_support.copy()
            current = np.minimum(caps, support)
            remaining = set(np.flatnonzero(e_active).tolist())
            spent = 0.0
            for idx in selected:
                if idx not in remaining:
                    continue
                trial = np.minimum(caps, support + atom_support[idx])
                gain = float(np.sum(weights[:n] * (trial - current), dtype=np.float64))
                crit[idx] = max(crit[idx], gain)
                oracle[idx] = True
                support += atom_support[idx]
                current = np.minimum(caps, support)
                spent += float(e_cost[idx])
                remaining.discard(idx)
        # Family-level oracle target for HAB.  It aggregates the same greedy
        # marginal gains used for atom proposals into certificate families, so
        # the family gate learns which evidence family deserves budget in this
        # scene/candidate set.
        num_families = max(int(cfg.get("model", {}).get("num_families", 0) or 0), max(FAMILY_NAMES.values()) + 1)
        family_gain = np.zeros((num_families,), dtype=np.float32)
        family_active = np.zeros((num_families,), dtype=bool)
        fam_ids = np.asarray(arrays["evidence_family_ids"], dtype=np.int64)[:Emax]
        e_active_all = np.asarray(arrays["evidence_active"], dtype=bool)[:Emax]
        for i in np.flatnonzero(e_active_all):
            f = int(np.clip(fam_ids[i], 0, num_families - 1))
            family_active[f] = True
            family_gain[f] += float(crit[i])
        if family_active[1:].any():
            family_active[0] = False
        if not family_active.any():
            family_active[0] = True
        arrays["oracle_selected_mask"] = oracle
        arrays["proposal_target_gain"] = crit
        arrays["family_target_gain"] = family_gain
        arrays["family_target_active"] = family_active
    tensors: dict[str, torch.Tensor] = {}
    for k, v in arrays.items():
        if np.asarray(v).dtype == np.bool_:
            tensors[k] = torch.from_numpy(np.asarray(v, dtype=bool))
        elif np.issubdtype(np.asarray(v).dtype, np.integer):
            tensors[k] = torch.from_numpy(np.asarray(v, dtype=np.int64))
        else:
            tensors[k] = torch.from_numpy(np.asarray(v, dtype=np.float32))
    return tensors
