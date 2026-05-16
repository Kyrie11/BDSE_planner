from __future__ import annotations

from typing import Any

import numpy as np

from bdse.data.cache_schema import CandidateBank, PairLabels, TeacherLabels
from bdse.planner.teacher_cost import residual_margin, validate_residual_closure


def margin_matrix(J: np.ndarray) -> np.ndarray:
    J = np.asarray(J, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        M = J[None, :] - J[:, None]
    return M


def add_positive_pair(pair_set: set[tuple[int, int]], J_T: np.ndarray, a: int, b: int, valid_mask: np.ndarray) -> None:
    if a == b or not (valid_mask[a] and valid_mask[b]):
        return
    m_ab = float(J_T[b] - J_T[a])
    if m_ab > 0:
        pair_set.add((int(a), int(b)))
        return
    m_ba = float(J_T[a] - J_T[b])
    if m_ba > 0:
        pair_set.add((int(b), int(a)))


def build_pair_labels(candidates: CandidateBank, teacher: TeacherLabels, cfg: dict[str, Any]) -> PairLabels:
    pcfg = cfg.get("pairs", {})
    valid = candidates.valid_mask.astype(bool)
    valid_idx = np.flatnonzero(valid)
    J = teacher.J_T
    pair_set: set[tuple[int, int]] = set()

    a_star = int(teacher.a_star)
    non_teacher = [i for i in valid_idx.tolist() if i != a_star]
    top_l = int(pcfg.get("winner_top_l", 16))
    top_by_cost = sorted(non_teacher, key=lambda i: (float(J[i]), int(i)))[:top_l]
    for b in top_by_cost:
        add_positive_pair(pair_set, J, a_star, b, valid)

    M = margin_matrix(J)
    pos = M[np.isfinite(M) & (M > 0)]
    eta = float(np.percentile(pos, int(pcfg.get("eta_percentile", 20)))) if pos.size else 1.0
    for a in valid_idx:
        for b in valid_idx:
            if a != b and 0.0 < M[a, b] < eta:
                pair_set.add((int(a), int(b)))

    safe = valid & (~teacher.hard_violation_mask.astype(bool))
    unsafe = valid & teacher.hard_violation_mask.astype(bool)
    for a in np.flatnonzero(safe):
        for b in np.flatnonzero(unsafe):
            add_positive_pair(pair_set, J, int(a), int(b), valid)

    all_pos = [(int(a), int(b)) for a in valid_idx for b in valid_idx if a != b and M[a, b] > 0]
    rng = np.random.default_rng(int(pcfg.get("random_seed", 17)))
    rng.shuffle(all_pos)
    target_min = int(pcfg.get("target_min", 64))
    target_max = int(pcfg.get("target_max", 256))
    for p in all_pos:
        if len(pair_set) >= target_min:
            break
        pair_set.add(p)
    if len(pair_set) > target_max:
        ordered = sorted(pair_set, key=lambda p: (0 if p[0] == a_star else 1, float(J[p[1]]), p[0], p[1]))
        pair_list = ordered[:target_max]
    else:
        pair_list = sorted(pair_set, key=lambda p: (p[0], p[1]))

    if not pair_list:
        pairs = np.zeros((0, 2), dtype=np.int64)
        margins = np.zeros((0,), dtype=np.float32)
        weights = np.zeros((0,), dtype=np.float32)
        residuals = np.zeros((0,), dtype=np.float32)
        pair_valid = np.zeros((0,), dtype=bool)
    else:
        pairs = np.asarray(pair_list, dtype=np.int64)
        margins = np.asarray([J[b] - J[a] for a, b in pairs], dtype=np.float32)
        residuals = np.asarray([residual_margin(teacher, int(a), int(b)) for a, b in pairs], dtype=np.float32)
        weights = np.ones((len(pairs),), dtype=np.float32)
        for i, (a, b) in enumerate(pairs):
            if a == a_star:
                weights[i] += 1.0
            if teacher.hard_violation_mask[b] and not teacher.hard_violation_mask[a]:
                weights[i] += 2.0
            if margins[i] < eta:
                weights[i] += 1.0
        pair_valid = np.ones((len(pairs),), dtype=bool)
    labels = PairLabels(pairs=pairs, margins=margins, weights=weights, residuals=residuals, valid_mask=pair_valid)
    labels.validate_positive_direction()
    validate_residual_closure(teacher, pairs)
    return labels


def decisive_rivals(teacher: TeacherLabels, candidates: CandidateBank, eta_D: float | None = None, top_L_D: int = 16) -> set[int]:
    valid = candidates.valid_mask.astype(bool)
    a = int(teacher.a_star)
    rivals = set()
    margins = teacher.J_T - teacher.J_T[a]
    pos = margins[(margins > 0) & valid]
    if eta_D is None:
        eta_D = float(np.percentile(pos, 20)) if pos.size else 1.0
    for b in np.flatnonzero(valid):
        if b != a and margins[b] < eta_D:
            rivals.add(int(b))
    for b in np.flatnonzero(teacher.hard_violation_mask & valid):
        if b != a:
            rivals.add(int(b))
    for b in sorted([i for i in np.flatnonzero(valid) if i != a], key=lambda i: (float(teacher.J_T[i]), int(i)))[:top_L_D]:
        rivals.add(int(b))
    return rivals
