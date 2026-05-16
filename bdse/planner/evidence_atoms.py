from __future__ import annotations

from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, EvidenceAtom, EvidenceBank, LabelOnlyFuture, RuntimeFeatures
from bdse.utils import angle_wrap, compute_curvature, finite_difference, nearest_polyline_distance, oriented_box_corners, polygons_overlap_sat

ATOM_QUERY_DIM = 12


def _unit_cost(cfg: dict[str, Any], atom_type: str, family: str) -> float:
    if cfg.get("evidence", {}).get("unit_cost", True):
        return 1.0
    if atom_type in {"occupancy", "collision"}:
        return 3.0
    if family == "interaction":
        return 2.0
    return 1.0


def _new_atom(atom_id: int, atom_type: str, anchor: dict[str, Any], family: str, is_hard: bool, cfg: dict[str, Any]) -> EvidenceAtom:
    return EvidenceAtom(atom_id=atom_id, type=atom_type, anchor=anchor, budget_cost=_unit_cost(cfg, atom_type, family), is_hard=is_hard, family=family, active_mask=True)


def _has_red_light(runtime: RuntimeFeatures) -> bool:
    return any("red" in str(tl.get("status", "")).lower() for tl in runtime.traffic_lights)


def _min_candidate_distance(candidates: CandidateBank, xy: np.ndarray) -> float:
    valid = candidates.trajectories[candidates.valid_mask]
    if len(valid) == 0:
        return 1e6
    pts = valid[..., :2].reshape(-1, 2)
    return float(np.linalg.norm(pts - xy[None, :], axis=1).min())


def enumerate_evidence_atoms(runtime: RuntimeFeatures, candidates: CandidateBank, cfg: dict[str, Any]) -> EvidenceBank:
    ecfg = cfg.get("evidence", {})
    atoms: list[EvidenceAtom] = []
    next_id = 0
    max_inter = int(ecfg.get("max_interaction_atoms", 64))
    max_map = int(ecfg.get("max_map_atoms", 32))
    max_kin = int(ecfg.get("max_kinematic_atoms", 16))

    if ecfg.get("include_interaction", True):
        for j, valid in enumerate(runtime.agent_valid.astype(bool)):
            if not valid:
                continue
            cur = runtime.current_agents[j]
            d = float(np.linalg.norm(cur[:2]))
            anchor = {"agent_index": int(j), "current_state": cur.copy(), "length": float(cur[7]) if cur.shape[0] > 7 else 4.8, "width": float(cur[8]) if cur.shape[0] > 8 else 2.0, "priority_distance": d}
            atoms.append(_new_atom(next_id, "occupancy", anchor, "interaction", True, cfg)); next_id += 1
            atoms.append(_new_atom(next_id, "ttc", anchor, "interaction", False, cfg)); next_id += 1
            atoms.append(_new_atom(next_id, "gap", anchor, "interaction", False, cfg)); next_id += 1
            if len([a for a in atoms if a.family == "interaction"]) >= max_inter:
                break

    if ecfg.get("include_rule_map", True):
        route = np.asarray(runtime.map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
        route_width = float(runtime.map_features.get("route_corridor_width", cfg.get("candidate", {}).get("route_width_m", 4.0)))
        drivable_polys = [np.asarray(p.get("xy"), dtype=np.float32).reshape(-1, 2) for p in runtime.map_features.get("drivable_polygons", []) if np.asarray(p.get("xy", [])).size >= 6]
        atoms.append(_new_atom(next_id, "drivable_area", {"route_centerline": route, "width": route_width, "drivable_polygons": drivable_polys, "map_valid": bool(runtime.map_features.get("map_valid", False))}, "rule_map", True, cfg)); next_id += 1
        atoms.append(_new_atom(next_id, "wrong_way", {"route_centerline": route}, "rule_map", True, cfg)); next_id += 1
        speed_limit = float(runtime.map_features.get("speed_limit_mps", 13.4))
        atoms.append(_new_atom(next_id, "speed_limit", {"speed_limit_mps": speed_limit}, "rule_map", False, cfg)); next_id += 1
        red_now = _has_red_light(runtime)
        if red_now:
            for sl in runtime.map_features.get("stop_lines", [])[: max(1, max_map - 4)]:
                xy = np.asarray(sl.get("xy", []), dtype=np.float32).reshape(-1, 2)
                if len(xy) >= 2 and _min_candidate_distance(candidates, xy.mean(axis=0)) < 80.0:
                    atoms.append(_new_atom(next_id, "red_light", {"stop_line_xy": xy, "red": True, "id": sl.get("id", "")}, "rule_map", True, cfg)); next_id += 1
        atoms.append(_new_atom(next_id, "route_connector", {"route_centerline": route}, "rule_map", False, cfg)); next_id += 1
        map_atoms = [a for a in atoms if a.family == "rule_map"]
        if len(map_atoms) > max_map:
            keep_ids = {a.atom_id for a in map_atoms[:max_map]}
            atoms = [a for a in atoms if a.family != "rule_map" or a.atom_id in keep_ids]

    if ecfg.get("include_kinematic", True):
        for atom_type in ["local_comfort_accel", "local_comfort_jerk", "local_comfort_curvature", "local_comfort_brake"][:max_kin]:
            atoms.append(_new_atom(next_id, atom_type, {}, "kinematic", False, cfg)); next_id += 1

    atoms = cap_evidence_atoms(atoms, candidates, runtime, cfg)
    q = compute_query_features(atoms, candidates, runtime, cfg)
    active = np.asarray([a.active_mask for a in atoms], dtype=bool)
    return EvidenceBank(atoms=atoms, query_features=q, active_mask=active)


def cap_evidence_atoms(atoms: list[EvidenceAtom], candidates: CandidateBank, runtime: RuntimeFeatures, cfg: dict[str, Any]) -> list[EvidenceAtom]:
    max_atoms = int(cfg.get("evidence", {}).get("max_atoms", 128))
    if len(atoms) <= max_atoms:
        return atoms
    valid_traj = candidates.trajectories[candidates.valid_mask]
    cand_xy = valid_traj[:, :, :2].reshape(-1, 2) if len(valid_traj) else np.zeros((0, 2), dtype=np.float32)

    def priority(atom: EvidenceAtom) -> tuple[int, float, str, int]:
        hard_rank = 0 if atom.is_hard else 1
        dist = 1e6
        if "current_state" in atom.anchor and len(cand_xy):
            dist = float(np.linalg.norm(cand_xy - atom.anchor["current_state"][:2][None, :], axis=1).min())
        elif "route_centerline" in atom.anchor and len(cand_xy):
            dist = float(nearest_polyline_distance(cand_xy, atom.anchor["route_centerline"]).min())
        elif "stop_line_xy" in atom.anchor and len(cand_xy):
            dist = float(np.linalg.norm(cand_xy - np.asarray(atom.anchor["stop_line_xy"]).mean(axis=0)[None, :], axis=1).min())
        return (hard_rank, dist, atom.type, atom.atom_id)

    return sorted(atoms, key=priority)[:max_atoms]


def _agent_future_for_atom(atom: EvidenceAtom, runtime: RuntimeFeatures, label_future: LabelOnlyFuture | None, T: int, dt: float) -> np.ndarray:
    j = int(atom.anchor.get("agent_index", -1))
    if label_future is not None and j >= 0 and j < label_future.logged_agents.shape[0] and label_future.agent_valid[j]:
        arr = np.asarray(label_future.logged_agents[j], dtype=np.float32)
        if arr.shape[0] >= T:
            return arr[:T]
        out = np.zeros((T, arr.shape[-1]), dtype=np.float32)
        out[: arr.shape[0]] = arr
        out[arr.shape[0] :] = arr[-1]
        return out
    cur = np.asarray(atom.anchor.get("current_state", np.zeros(10)), dtype=np.float32)
    times = np.arange(1, T + 1, dtype=np.float32) * dt
    vx = float(cur[5]) if cur.shape[0] > 5 else float(cur[3]) * np.cos(float(cur[2]))
    vy = float(cur[6]) if cur.shape[0] > 6 else float(cur[3]) * np.sin(float(cur[2]))
    x = cur[0] + vx * times
    y = cur[1] + vy * times
    yaw = np.full(T, cur[2], dtype=np.float32)
    v = np.full(T, cur[3], dtype=np.float32)
    return np.stack([x, y, yaw, v, times], axis=1).astype(np.float32)


def _collision_overlap_series(ego: np.ndarray, agent: np.ndarray, atom: EvidenceAtom) -> np.ndarray:
    length_ego, width_ego = 4.8, 2.0
    length_agent = float(atom.anchor.get("length", 4.8))
    width_agent = float(atom.anchor.get("width", 2.0))
    out = np.zeros((ego.shape[0],), dtype=np.float32)
    for k in range(ego.shape[0]):
        d = np.linalg.norm(ego[k, :2] - agent[k, :2])
        if d > (length_ego + length_agent):
            continue
        ego_box = oriented_box_corners(float(ego[k, 0]), float(ego[k, 1]), float(ego[k, 2]), length_ego, width_ego)
        agent_box = oriented_box_corners(float(agent[k, 0]), float(agent[k, 1]), float(agent[k, 2]), length_agent, width_agent)
        out[k] = 1.0 if polygons_overlap_sat(ego_box, agent_box) else 0.0
    return out


def _point_in_poly(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    px = poly[:, 0]
    py = poly[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        cond = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-9) + px[i])
        inside ^= cond
        j = i
    return inside


def _poly_boundary_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    d = np.full(points.shape[0], 1e6, dtype=np.float32)
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        ab = b - a
        t = np.clip(((points - a[None, :]) @ ab) / (float(ab @ ab) + 1e-9), 0.0, 1.0)
        proj = a[None, :] + t[:, None] * ab[None, :]
        d = np.minimum(d, np.linalg.norm(points - proj, axis=1))
    return d


def _outside_drivable(points: np.ndarray, polygons: list[np.ndarray], route: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    if polygons:
        inside_any = np.zeros(points.shape[0], dtype=bool)
        boundary = np.full(points.shape[0], 1e6, dtype=np.float32)
        for poly in polygons:
            if len(poly) < 3:
                continue
            inside_any |= _point_in_poly(points, poly)
            boundary = np.minimum(boundary, _poly_boundary_distance(points, poly))
        return ~inside_any, boundary
    dist = nearest_polyline_distance(points, route)
    return dist > width, np.maximum(0.0, dist - width)


def _route_heading_at(points: np.ndarray, route: np.ndarray) -> np.ndarray:
    if len(route) < 2:
        return np.zeros(points.shape[0], dtype=np.float32)
    seg_a = route[:-1]
    seg_b = route[1:]
    seg = seg_b - seg_a
    mids = 0.5 * (seg_a + seg_b)
    idx = np.argmin(np.linalg.norm(points[:, None, :] - mids[None, :, :], axis=2), axis=1)
    return np.arctan2(seg[idx, 1], seg[idx, 0]).astype(np.float32)


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
    return bool(ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d))


def _crosses_polyline(traj_xy: np.ndarray, line_xy: np.ndarray) -> bool:
    if len(line_xy) < 2 or len(traj_xy) < 2:
        return False
    for i in range(len(traj_xy) - 1):
        for j in range(len(line_xy) - 1):
            if _segments_intersect(traj_xy[i], traj_xy[i + 1], line_xy[j], line_xy[j + 1]):
                return True
    return False


def raw_local_costs(atoms: list[EvidenceAtom], candidates: CandidateBank, runtime: RuntimeFeatures, label_future: LabelOnlyFuture | None, cfg: dict[str, Any]) -> np.ndarray:
    E, K = len(atoms), candidates.K
    raw = np.zeros((E, K), dtype=np.float32)
    safety = cfg.get("evidence", {}).get("safety", {})
    d_safe = float(safety.get("d_safe_m", 2.0))
    tau_safe = float(safety.get("tau_safe_s", 2.0))
    front_gap = float(safety.get("front_gap_m", 8.0))
    rear_gap = float(safety.get("rear_gap_m", 12.0))
    margin = float(safety.get("boundary_margin_m", 0.5))
    c_col = float(safety.get("collision_raw", 100.0))
    c_red = float(safety.get("red_raw", 50.0))
    c_off = float(safety.get("off_raw", 50.0))
    c_wrong = float(safety.get("wrong_raw", 50.0))
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    for ei, atom in enumerate(atoms):
        for a in range(K):
            traj = candidates.trajectories[a]
            if atom.type == "occupancy":
                agent = _agent_future_for_atom(atom, runtime, label_future, candidates.T, dt)
                dist = np.linalg.norm(traj[:, :2] - agent[:, :2], axis=1)
                near = np.maximum(0.0, d_safe - dist) ** 2
                overlap = _collision_overlap_series(traj, agent, atom)
                raw[ei, a] = float(near.sum() + c_col * overlap.max())
            elif atom.type == "ttc":
                agent = _agent_future_for_atom(atom, runtime, label_future, candidates.T, dt)
                dist = np.linalg.norm(traj[:, :2] - agent[:, :2], axis=1)
                rel_speed = np.maximum(np.abs(traj[:, 3] - agent[:, 3]), 0.1)
                ttc = dist / rel_speed
                raw[ei, a] = float(np.maximum(0.0, tau_safe - ttc).max() ** 2)
            elif atom.type == "gap":
                cur = np.asarray(atom.anchor.get("current_state", np.zeros(10)), dtype=np.float32)
                dx = cur[0] - traj[:, 0]
                front = np.maximum(dx, 0.0)
                rear = np.maximum(-dx, 0.0)
                raw[ei, a] = float(np.maximum(0.0, front_gap - front.max()) ** 2 + np.maximum(0.0, rear_gap - rear.max()) ** 2)
            elif atom.type == "red_light":
                stop = np.asarray(atom.anchor.get("stop_line_xy", []), dtype=np.float32).reshape(-1, 2)
                red = bool(atom.anchor.get("red", True))
                crosses = _crosses_polyline(traj[:, :2], stop)
                d_line = float(np.linalg.norm(traj[:, None, :2] - stop[None, :, :], axis=2).min()) if len(stop) else 1e6
                raw[ei, a] = float(c_red * float(red and crosses) + np.maximum(0.0, 1.0 - d_line) ** 2)
            elif atom.type == "drivable_area":
                route = np.asarray(atom.anchor.get("route_centerline"), dtype=np.float32)
                width = float(atom.anchor.get("width", 4.0))
                polygons = [np.asarray(p, dtype=np.float32).reshape(-1, 2) for p in atom.anchor.get("drivable_polygons", [])]
                outside, dist = _outside_drivable(traj[:, :2], polygons, route, width)
                raw[ei, a] = float(c_off * outside.max() + np.maximum(0.0, margin + dist).sum())
            elif atom.type == "wrong_way":
                route = np.asarray(atom.anchor.get("route_centerline"), dtype=np.float32)
                route_dist = nearest_polyline_distance(traj[:, :2], route)
                route_yaw = _route_heading_at(traj[:, :2], route)
                heading_bad = np.abs(angle_wrap(traj[:, 2] - route_yaw)) > (0.5 * np.pi)
                raw[ei, a] = float(c_wrong * np.logical_and(heading_bad, route_dist < 5.0).max())
            elif atom.type == "route_connector":
                route = np.asarray(atom.anchor.get("route_centerline"), dtype=np.float32)
                raw[ei, a] = float(np.square(nearest_polyline_distance(traj[:, :2], route)).mean())
            elif atom.type == "speed_limit":
                limit = float(atom.anchor.get("speed_limit_mps", 13.4))
                raw[ei, a] = float(np.maximum(0.0, traj[:, 3] - limit).max() ** 2)
            elif atom.type.startswith("local_comfort"):
                v = traj[:, 3]
                acc = finite_difference(v, dt)
                jerk = finite_difference(acc, dt)
                curv = compute_curvature(traj[:, :2])
                if atom.type.endswith("accel"):
                    raw[ei, a] = float(np.maximum(0.0, np.abs(acc) - 3.0).sum())
                elif atom.type.endswith("jerk"):
                    raw[ei, a] = float(np.maximum(0.0, np.abs(jerk) - 5.0).sum())
                elif atom.type.endswith("curvature"):
                    raw[ei, a] = float(np.maximum(0.0, np.abs(curv) - 0.25).sum())
                elif atom.type.endswith("brake"):
                    raw[ei, a] = float(np.maximum(0.0, -acc - 5.0).sum())
    raw[:, ~candidates.valid_mask] = 0.0
    return raw


def atom_weight_scale_cap(atom: EvidenceAtom, cfg: dict[str, Any]) -> tuple[float, float, float]:
    ecfg = cfg.get("evidence", {})
    weights = ecfg.get("weights", {})
    scales = ecfg.get("scales", {})
    caps = ecfg.get("caps", {})
    if atom.is_hard:
        cap = float(caps.get("hard", 100.0))
    elif atom.family == "interaction":
        cap = float(caps.get("inter", 20.0))
    elif atom.family == "rule_map":
        cap = float(caps.get("rule", caps.get("map", 20.0)))
    else:
        cap = float(caps.get("kin", 10.0))
    base_type = atom.type.replace("local_comfort_accel", "local_comfort").replace("local_comfort_jerk", "local_comfort").replace("local_comfort_curvature", "local_comfort").replace("local_comfort_brake", "local_comfort")
    weight = float(weights.get(base_type, weights.get(atom.type, 1.0)))
    scale = float(scales.get(base_type, scales.get(atom.type, 1.0)))
    return weight, scale, cap


def normalize_atom_costs(raw_costs: np.ndarray, atoms: list[EvidenceAtom], cfg: dict[str, Any]) -> np.ndarray:
    eps = float(cfg.get("evidence", {}).get("eps", 1e-6))
    g = np.zeros_like(raw_costs, dtype=np.float32)
    for i, atom in enumerate(atoms):
        w, s, cap = atom_weight_scale_cap(atom, cfg)
        g[i] = w * np.clip(raw_costs[i] / (s + eps), 0.0, cap)
    return g.astype(np.float32)


def compute_query_features(atoms: list[EvidenceAtom], candidates: CandidateBank, runtime: RuntimeFeatures, cfg: dict[str, Any]) -> np.ndarray:
    E, K = len(atoms), candidates.K
    q = np.zeros((E, K, ATOM_QUERY_DIM), dtype=np.float32)
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    for ei, atom in enumerate(atoms):
        for a in range(K):
            traj = candidates.trajectories[a]
            feat = np.zeros((ATOM_QUERY_DIM,), dtype=np.float32)
            if "current_state" in atom.anchor:
                cur = np.asarray(atom.anchor["current_state"], dtype=np.float32)
                dist = np.linalg.norm(traj[:, :2] - cur[:2][None, :], axis=1)
                feat[0] = dist.min(); feat[1] = dist.mean(); feat[2] = traj[np.argmin(dist), 4]; feat[3] = cur[3] if len(cur) > 3 else 0.0
            if "route_centerline" in atom.anchor:
                dist = nearest_polyline_distance(traj[:, :2], atom.anchor["route_centerline"])
                feat[4] = dist.min(); feat[5] = dist.mean(); feat[6] = dist.max()
            if atom.type == "red_light":
                stop = np.asarray(atom.anchor.get("stop_line_xy", []), dtype=np.float32).reshape(-1, 2)
                feat[7] = float(_crosses_polyline(traj[:, :2], stop))
                feat[8] = float(np.linalg.norm(traj[-1, None, :2] - stop, axis=1).min()) if len(stop) else 1e6
            v = traj[:, 3]
            acc = finite_difference(v, dt)
            jerk = finite_difference(acc, dt)
            curv = compute_curvature(traj[:, :2])
            feat[9] = float(np.max(np.abs(acc))); feat[10] = float(np.max(np.abs(jerk))); feat[11] = float(np.max(np.abs(curv)))
            q[ei, a] = feat
    q[:, ~candidates.valid_mask, :] = 0.0
    return q


def hard_event_ownership(atoms: list[EvidenceAtom]) -> dict[str, int]:
    owners: dict[str, int] = {}
    for atom in atoms:
        if atom.is_hard:
            if atom.type in {"occupancy", "collision"}:
                key = "agent_collision"
            elif atom.type == "red_light":
                key = "red_light"
            elif atom.type == "drivable_area":
                key = "off_drivable"
            elif atom.type == "wrong_way":
                key = "wrong_way"
            else:
                key = atom.type
            owners[key] = owners.get(key, 0) + 1
    return owners
