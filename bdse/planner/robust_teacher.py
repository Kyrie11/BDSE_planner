from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, EvidenceBank, LabelOnlyFuture, RuntimeFeatures, TeacherLabels
from bdse.planner.evidence_atoms import normalize_atom_costs, raw_local_costs_with_hard_events
from bdse.planner.response_modes import ResponseMode, build_response_modes, mode_to_label_future


@dataclass()
class RobustTeacherComponents:
    feasibility_level: np.ndarray
    J_quality: np.ndarray
    J_demo: np.ndarray
    J_risk_mean: np.ndarray
    J_risk_cvar: np.ndarray
    J_base: np.ndarray
    g_evid: np.ndarray
    J_T: np.ndarray
    mode_costs: dict[str, np.ndarray]


def _mode_dependent_atom(atom: Any) -> bool:
    """Whether an atom's raw cost changes with the response-mode future.

    In the current executable teacher, only occupancy/collision and TTC atoms read
    label_future.agent trajectories.  Route, drivable-area, red-light, speed-limit,
    gap, and kinematic atoms depend only on runtime map/current state and candidate
    trajectories.  Reusing their raw costs across robust modes preserves the exact
    teacher partition while avoiding 5--6 repeated static passes per sample.
    """
    return str(getattr(atom, "type", "")) in {"occupancy", "collision", "ttc"}


def _scatter_subset(values: np.ndarray, indices: list[int], shape: tuple[int, int], dtype: Any) -> np.ndarray:
    out = np.zeros(shape, dtype=dtype)
    if indices:
        out[np.asarray(indices, dtype=np.int64)] = values.astype(dtype, copy=False)
    return out


def weighted_cvar(values: np.ndarray, probs: np.ndarray, alpha: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    probs = probs / max(float(probs.sum()), 1e-6)
    M = vals.shape[0]
    flat = vals.reshape(M, -1)
    out = np.zeros((flat.shape[1],), dtype=np.float32)
    q = float(np.clip(alpha, 0.0, 0.999))
    for col in range(flat.shape[1]):
        order = np.argsort(flat[:, col])
        v = flat[order, col]
        p = probs[order]
        c = np.cumsum(p)
        start = np.searchsorted(c, q, side="left")
        tail_v = v[start:]
        tail_p = p[start:].copy()
        if tail_p.size == 0:
            out[col] = v[-1]
        else:
            # Fractional mass at quantile boundary.
            prev = c[start - 1] if start > 0 else 0.0
            tail_p[0] = max(c[start] - q, 0.0)
            denom = max(float(tail_p.sum()), 1e-6)
            out[col] = float((tail_v * tail_p).sum() / denom)
    return out.reshape(vals.shape[1:]).astype(np.float32)


def _hard_category_masks(atoms, hard_any: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-candidate hard event categories from owned hard atoms."""
    K = hard_any.shape[1] if hard_any.ndim == 2 else 0
    cats = {
        "collision": np.zeros((K,), dtype=bool),
        "off_drivable": np.zeros((K,), dtype=bool),
        "wrong_way": np.zeros((K,), dtype=bool),
        "red_light": np.zeros((K,), dtype=bool),
    }
    for ei, atom in enumerate(atoms):
        if ei >= hard_any.shape[0] or not bool(getattr(atom, "is_hard", False)):
            continue
        typ = str(getattr(atom, "type", ""))
        if typ in {"occupancy", "collision"}:
            cats["collision"] |= hard_any[ei]
        elif typ == "drivable_area":
            cats["off_drivable"] |= hard_any[ei]
        elif typ == "wrong_way":
            cats["wrong_way"] |= hard_any[ei]
        elif typ == "red_light":
            cats["red_light"] |= hard_any[ei]
    return cats


def _inject_lexicographic_hard_costs(g: np.ndarray, atoms, hard_any: np.ndarray, scale: float) -> np.ndarray:
    """Assign hard feasibility priority to the unique owning hard atoms.

    This keeps the paper partition J_T = J_base + sum_i g_i while making the
    teacher ordering lexicographic in collision/off-road/wrong-way/red-light
    categories.  The hierarchy is encoded by separated powers of ``scale`` so
    no soft progress/imitation term can compensate for a higher-priority hard
    event under normal cost ranges.
    """
    if scale <= 0 or hard_any.size == 0:
        return g
    priority = {"occupancy": 4.0, "collision": 4.0, "drivable_area": 3.0, "wrong_way": 2.0, "red_light": 1.0}
    out = np.asarray(g, dtype=np.float32).copy()
    for ei, atom in enumerate(atoms):
        if ei >= hard_any.shape[0] or not bool(getattr(atom, "is_hard", False)):
            continue
        mult = priority.get(str(getattr(atom, "type", "")), 1.0)
        out[ei] += float(scale) * mult * hard_any[ei].astype(np.float32)
    return out


def _lexsort_teacher_argmin(J_T: np.ndarray, valid: np.ndarray, cats: dict[str, np.ndarray]) -> int:
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        raise ValueError("Teacher cost requires at least one valid candidate")
    idx = np.flatnonzero(valid)
    # np.lexsort uses last key first; list keys from final tie-breaker to primary.
    keys = (
        idx,
        np.asarray(J_T, dtype=np.float64)[idx],
        cats["red_light"][idx].astype(np.int64),
        cats["wrong_way"][idx].astype(np.int64),
        cats["off_drivable"][idx].astype(np.int64),
        cats["collision"][idx].astype(np.int64),
    )
    return int(idx[np.lexsort(keys)[0]])


def evaluate_robust_teacher_costs(
    runtime: RuntimeFeatures,
    label_future: LabelOnlyFuture | None,
    candidates: CandidateBank,
    evidence_bank: EvidenceBank,
    cfg: dict[str, Any],
    base_cost_fn,
) -> TeacherLabels:
    J_base = base_cost_fn(runtime, label_future, candidates, cfg)
    modes = build_response_modes(runtime, label_future, cfg)
    raw_by_mode: list[np.ndarray] = []
    hard_by_mode: list[np.ndarray] = []
    mode_costs: dict[str, np.ndarray] = {}

    atoms = list(evidence_bank.atoms)
    E, K = evidence_bank.E, candidates.K
    dynamic_idx = [i for i, atom in enumerate(atoms) if _mode_dependent_atom(atom)]
    static_idx = [i for i in range(E) if i not in set(dynamic_idx)]
    static_raw_full = np.zeros((E, K), dtype=np.float32)
    static_hard_full = np.zeros((E, K), dtype=bool)
    if static_idx:
        static_atoms = [atoms[i] for i in static_idx]
        static_raw, static_hard = raw_local_costs_with_hard_events(static_atoms, candidates, runtime, label_future, cfg)
        static_raw_full = _scatter_subset(static_raw, static_idx, (E, K), np.float32)
        static_hard_full = _scatter_subset(static_hard, static_idx, (E, K), bool)

    for mode in modes:
        lf = mode_to_label_future(mode, label_future, runtime)
        raw_m = static_raw_full.copy()
        hard_m = static_hard_full.copy()
        if dynamic_idx:
            dyn_atoms = [atoms[i] for i in dynamic_idx]
            dyn_raw, dyn_hard = raw_local_costs_with_hard_events(dyn_atoms, candidates, runtime, lf, cfg)
            raw_m[np.asarray(dynamic_idx, dtype=np.int64)] = dyn_raw
            hard_m[np.asarray(dynamic_idx, dtype=np.int64)] = dyn_hard
        raw_m = np.nan_to_num(raw_m, nan=1e6, posinf=1e6, neginf=1e6)
        raw_by_mode.append(raw_m)
        hard_by_mode.append(hard_m)
        mode_costs[mode.name] = raw_m.sum(axis=0).astype(np.float32)
    raw_stack = np.stack(raw_by_mode, axis=0) if raw_by_mode else np.zeros((1, evidence_bank.E, candidates.K), dtype=np.float32)
    hard_stack = np.stack(hard_by_mode, axis=0) if hard_by_mode else np.zeros((1, evidence_bank.E, candidates.K), dtype=bool)
    probs = np.asarray([m.probability for m in modes], dtype=np.float32)
    probs = probs / max(float(probs.sum()), 1e-6)
    mean = np.tensordot(probs, raw_stack, axes=(0, 0)).astype(np.float32)
    rcfg = cfg.get("teacher", {}).get("risk_aggregation", {})
    alpha = float(rcfg.get("cvar_alpha", cfg.get("teacher", {}).get("cvar_alpha", 0.9)))
    cvar_weight = float(rcfg.get("cvar_weight", cfg.get("teacher", {}).get("cvar_weight", 0.4)))
    cvar = weighted_cvar(raw_stack, probs, alpha)
    raw_robust = (1.0 - cvar_weight) * mean + cvar_weight * cvar
    g = normalize_atom_costs(raw_robust, evidence_bank.atoms, cfg)

    hard_any = hard_stack.any(axis=0)
    hard_mask = evidence_bank.hard_mask()
    hard_violation = hard_any[hard_mask].any(axis=0) & candidates.valid_mask if hard_mask.size else np.zeros((candidates.K,), dtype=bool)
    hard_categories = _hard_category_masks(evidence_bank.atoms, hard_any)
    for key in hard_categories:
        hard_categories[key] &= candidates.valid_mask
    fcfg = cfg.get("teacher", {}).get("feasibility", {})
    H = float(fcfg.get("hard_priority_scale", cfg.get("teacher", {}).get("hard_priority_scale", 10000.0)))
    # raw_local_costs_with_hard_events already injects lexicographic hard
    # priority into the owning hard atoms before normalization, so do not add it
    # again here.
    J_evid = g.sum(axis=0, dtype=np.float64)
    J_T = J_base.astype(np.float64) + J_evid
    J_T[~candidates.valid_mask] = np.inf
    a_star = _lexsort_teacher_argmin(J_T, candidates.valid_mask, hard_categories)
    labels = TeacherLabels(
        J_base=J_base.astype(np.float64),
        g_evid=g.astype(np.float32),
        J_evid=J_evid.astype(np.float64),
        J_T=J_T.astype(np.float64),
        a_star=a_star,
        hard_violation_mask=hard_violation,
        diagnostics={
            "valid_candidate_count": int(candidates.valid_mask.sum()),
            "teacher_cost_min": float(J_T[a_star]),
            "atom_count": len(evidence_bank.atoms),
            "response_mode_count": int(len(modes)),
            "response_modes": [m.name for m in modes],
            "response_mode_probs": {m.name: float(m.probability) for m in modes},
            "risk_cvar_alpha": alpha,
            "risk_cvar_weight": cvar_weight,
            "hard_priority_scale": H,
            "hard_violation_rate": float(hard_violation[candidates.valid_mask].mean()) if np.any(candidates.valid_mask) else 0.0,
            "safe_candidate_exists": bool(np.any(candidates.valid_mask & ~hard_violation)),
            "hard_event_atom_count": int(hard_any[hard_mask].any(axis=1).sum()) if hard_mask.size else 0,
            "hard_collision_count": int(hard_categories["collision"].sum()),
            "hard_off_drivable_count": int(hard_categories["off_drivable"].sum()),
            "hard_wrong_way_count": int(hard_categories["wrong_way"].sum()),
            "hard_red_light_count": int(hard_categories["red_light"].sum()),
            "lexicographic_feasibility": True,
            "inject_hard_priority_costs": bool(fcfg.get("inject_hard_priority_costs", True)),
        },
    )
    labels.validate_partition(candidates.valid_mask)
    return labels
