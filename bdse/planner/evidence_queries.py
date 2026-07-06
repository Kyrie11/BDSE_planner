from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from bdse.data.cache_schema import CandidateBank, EvidenceAtom, RuntimeFeatures
from bdse.utils import nearest_polyline_distance, route_progress_along_polyline

ATOM_QUERY_DIM = 18
PROPOSAL_FEATURE_DIM = 24

FAMILY_NAMES = {
    "feasibility": 1,
    "reachability_interaction": 2,
    "precedence": 3,
    "decision_boundary": 4,
    "dynamic_regularity": 5,
    # Backward compatible names.
    "interaction": 2,
    "rule_map": 1,
    "kinematic": 5,
}

TYPE_NAMES = {
    "occupancy": 1,
    "collision": 1,
    "ttc": 2,
    "gap": 3,
    "yield": 4,
    "red_light": 5,
    "drivable_area": 6,
    "wrong_way": 7,
    "route_connector": 8,
    "speed_limit": 9,
    "local_comfort_accel": 10,
    "local_comfort_jerk": 11,
    "local_comfort_curvature": 12,
    "local_comfort_brake": 13,
}


def certificate_family(atom_type: str, family: str | None = None) -> str:
    if atom_type in {"drivable_area", "red_light", "wrong_way", "route_connector", "speed_limit"}:
        return "feasibility"
    if atom_type in {"occupancy", "collision", "ttc"}:
        return "reachability_interaction"
    if atom_type in {"gap", "yield"}:
        return "precedence"
    if atom_type.startswith("local_comfort"):
        return "dynamic_regularity"
    return family or "decision_boundary"


def _anchor_xy(atom: EvidenceAtom) -> np.ndarray:
    if "current_state" in atom.anchor:
        st = np.asarray(atom.anchor["current_state"], dtype=np.float32).reshape(-1)
        if st.size >= 2:
            return st[:2]
    if "stop_line_xy" in atom.anchor:
        xy = np.asarray(atom.anchor["stop_line_xy"], dtype=np.float32).reshape(-1, 2)
        if len(xy):
            return xy.mean(axis=0)
    if "route_centerline" in atom.anchor:
        xy = np.asarray(atom.anchor["route_centerline"], dtype=np.float32).reshape(-1, 2)
        if len(xy):
            return xy[min(len(xy) - 1, max(0, len(xy) // 4))]
    return np.zeros((2,), dtype=np.float32)


def compute_proposal_features(atoms: list[EvidenceAtom], candidates: CandidateBank, runtime: RuntimeFeatures, cfg: dict[str, Any]) -> np.ndarray:
    """Cheap atom-level proposal features used before action-conditioned queries.

    These features intentionally avoid local action--atom scoring.  They use only
    atom metadata, anchor geometry, current runtime state, route progress, and a
    coarse candidate envelope distance/overlap proxy.
    """
    E = len(atoms)
    z = np.zeros((E, PROPOSAL_FEATURE_DIM), dtype=np.float32)
    route = np.asarray(runtime.map_features.get("route_centerline", np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    if len(route) < 2:
        route = np.array([[0.0, 0.0], [160.0, 0.0]], dtype=np.float32)
    valid_traj = candidates.trajectories[np.asarray(candidates.valid_mask, dtype=bool)]
    cand_xy = valid_traj[..., :2].reshape(-1, 2) if len(valid_traj) else np.zeros((0, 2), dtype=np.float32)
    speed_limit = float(runtime.map_features.get("speed_limit_mps", 13.4))
    for i, atom in enumerate(atoms):
        xy = _anchor_xy(atom).astype(np.float32)
        fam = certificate_family(atom.type, atom.family)
        route_dist = float(nearest_polyline_distance(xy[None], route)[0]) if len(route) >= 2 else 0.0
        route_prog = float(route_progress_along_polyline(xy[None], route)[0]) if len(route) >= 2 else 0.0
        if len(cand_xy):
            d_cand = float(np.linalg.norm(cand_xy - xy[None], axis=1).min())
            overlap = float(d_cand < float(cfg.get("evidence", {}).get("proposal_overlap_radius_m", 8.0)))
        else:
            d_cand, overlap = 1e6, 0.0
        # Cheap TTC proxy from current relative position/speed when available.
        ttc_proxy = 1e6
        if "current_state" in atom.anchor:
            st = np.asarray(atom.anchor["current_state"], dtype=np.float32).reshape(-1)
            rel_x = float(st[0]) if st.size > 0 else 0.0
            rel_v = float(st[5]) if st.size > 5 else 0.0
            closing = max(-rel_v, 0.1)
            ttc_proxy = abs(rel_x) / closing
        z[i, 0] = float(atom.is_hard)
        z[i, 1] = float(atom.budget_cost)
        z[i, 2] = float(TYPE_NAMES.get(atom.type, 0)) / 16.0
        z[i, 3] = float(FAMILY_NAMES.get(fam, 0)) / 8.0
        z[i, 4:6] = xy / 100.0
        z[i, 6] = np.clip(route_dist / 50.0, 0.0, 20.0)
        z[i, 7] = np.clip(route_prog / 200.0, -5.0, 5.0)
        z[i, 8] = np.clip(d_cand / 100.0, 0.0, 20.0)
        z[i, 9] = np.clip(ttc_proxy / 10.0, 0.0, 100.0)
        z[i, 10] = overlap
        z[i, 11] = float(bool(atom.anchor.get("red", False)))
        z[i, 12] = float(bool(runtime.map_features.get("map_valid", False)))
        z[i, 13] = np.clip(speed_limit / 30.0, 0.0, 2.0)
        z[i, 14] = float(atom.lambda_weight)
        # Include user/debug cheap_features without changing the public dimension.
        for off, key in enumerate(sorted(atom.cheap_features)[: PROPOSAL_FEATURE_DIM - 15]):
            try:
                z[i, 15 + off] = float(atom.cheap_features[key])
            except Exception:
                pass
    return np.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def compute_query_features_for_pairs(
    atoms: list[EvidenceAtom],
    candidates: CandidateBank,
    runtime: RuntimeFeatures,
    atom_indices: Iterable[int],
    action_indices: Iterable[int],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute q_i(a) only for requested atom/action entries.

    Returns (atom_ids, action_ids, q) with matching first dimension.  The helper
    may internally evaluate each requested atom over K candidates for vectorized
    geometry reuse, but it never evaluates the full E x K evidence bank unless the
    caller explicitly passes all atoms.
    """
    atom_ids = np.asarray(list(dict.fromkeys(int(i) for i in atom_indices)), dtype=np.int64)
    action_ids = np.asarray(list(dict.fromkeys(int(i) for i in action_indices)), dtype=np.int64)
    if atom_ids.size == 0 or action_ids.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0, ATOM_QUERY_DIM), dtype=np.float32)
    atom_ids = atom_ids[(atom_ids >= 0) & (atom_ids < len(atoms))]
    action_ids = action_ids[(action_ids >= 0) & (action_ids < candidates.K)]
    if atom_ids.size == 0 or action_ids.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0, ATOM_QUERY_DIM), dtype=np.float32)
    from bdse.planner.evidence_atoms import compute_query_features

    sub_atoms = [atoms[int(i)] for i in atom_ids]
    q_sub = compute_query_features(sub_atoms, candidates, runtime, cfg)
    # Preserve the previous atom-major ordering while avoiding Python list
    # construction for the Top-M x action grid used at every closed-loop tick.
    A = int(atom_ids.size)
    U = int(action_ids.size)
    out_atoms = np.repeat(atom_ids.astype(np.int64), U)
    out_actions = np.tile(action_ids.astype(np.int64), A)
    out_q = q_sub[np.arange(A, dtype=np.int64)[:, None], action_ids[None, :]].reshape(A * U, ATOM_QUERY_DIM)
    return out_atoms, out_actions, np.asarray(out_q, dtype=np.float32)
