from __future__ import annotations

import numpy as np

SAFE_LIKE_MANEUVER_IDS = {1, 2, 6}  # decelerate_stop, yield_creep, safe_fallback
PROGRESSIVE_MANEUVER_IDS = {0, 3, 4, 5}  # keep/follow, lane changes, connector


def _valid_finite_indices(J0: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.flatnonzero(valid & np.isfinite(J0))


def _trajectory_progress(trajectories: np.ndarray | None, K: int) -> np.ndarray:
    if trajectories is None:
        return np.zeros((K,), dtype=np.float32)
    tr = np.asarray(trajectories, dtype=np.float32)
    if tr.ndim < 3 or tr.shape[0] < K or tr.shape[2] < 1:
        return np.zeros((K,), dtype=np.float32)
    return np.nan_to_num(tr[:K, -1, 0], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _trajectory_terminal_speed(trajectories: np.ndarray | None, K: int) -> np.ndarray:
    if trajectories is None:
        return np.zeros((K,), dtype=np.float32)
    tr = np.asarray(trajectories, dtype=np.float32)
    if tr.ndim < 3 or tr.shape[0] < K or tr.shape[2] < 4:
        return np.zeros((K,), dtype=np.float32)
    return np.nan_to_num(tr[:K, -1, 3], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _maneuver_array(maneuver_ids: np.ndarray | None, K: int) -> np.ndarray:
    if maneuver_ids is None:
        return np.full((K,), -999, dtype=np.int64)
    arr = np.asarray(maneuver_ids, dtype=np.int64).reshape(-1)
    if arr.shape[0] < K:
        arr = np.pad(arr, (0, K - arr.shape[0]), constant_values=-999)
    return arr[:K]


def _add_enhanced_decisive_pairs(
    pair_set: dict[tuple[int, int], float],
    J0: np.ndarray,
    valid_idx: np.ndarray,
    valid: np.ndarray,
    safety: np.ndarray,
    trajectories: np.ndarray | None,
    maneuver_ids: np.ndarray | None,
    progress_pair_count: int,
    maneuver_pair_count: int,
    lambda_progress: float = 0.75,
    lambda_maneuver: float = 0.75,
) -> None:
    """Add cheap, deployment-available rival pairs beyond base-cost neighbors.

    The base Top-L graph is brittle in intersections: a conservative stop/yield
    candidate may have a worse base cost but becomes decisive after red-light,
    stop-line, or interaction evidence.  These pairs force the selector to query
    such cross-family comparisons without using teacher labels or future agents.
    """
    if valid_idx.size <= 1:
        return
    K = int(J0.shape[0])
    progress = _trajectory_progress(trajectories, K)
    speed = _trajectory_terminal_speed(trajectories, K)
    man = _maneuver_array(maneuver_ids, K)

    def add(a: int, b: int, w: float) -> None:
        if a == b or not valid[a] or not valid[b]:
            return
        key = (int(a), int(b))
        pair_set[key] = max(float(w), pair_set.get(key, 0.0))

    safe_like = np.asarray([i for i in valid_idx.tolist() if int(man[i]) in SAFE_LIKE_MANEUVER_IDS], dtype=np.int64)
    progressive = np.asarray([i for i in valid_idx.tolist() if int(man[i]) in PROGRESSIVE_MANEUVER_IDS], dtype=np.int64)
    if safe_like.size and progressive.size:
        safe_order = sorted(safe_like.tolist(), key=lambda i: (bool(safety[i]), float(J0[i]), float(speed[i]), i))
        prog_order = sorted(progressive.tolist(), key=lambda i: (-float(progress[i]), float(J0[i]), i))
        for a in safe_order[: max(0, int(maneuver_pair_count))]:
            for b in prog_order[: max(1, min(4, len(prog_order)))]:
                if a != b:
                    # Orient safe/yield/stop -> progressive so positive evidence
                    # can certify the conservative action when rules/interactions require it.
                    add(int(a), int(b), 1.0 + float(lambda_maneuver))
                    add(int(b), int(a), 0.5 * (1.0 + float(lambda_maneuver)))

    # Longitudinal progress extremes: stop-vs-go and slow-vs-fast candidates can
    # be decisive even if their base costs are far apart.
    n = max(0, int(progress_pair_count))
    if n > 0:
        low = sorted(valid_idx.tolist(), key=lambda i: (float(progress[i]), float(speed[i]), float(J0[i]), i))[:n]
        high = sorted(valid_idx.tolist(), key=lambda i: (-float(progress[i]), float(J0[i]), i))[:n]
        for a in low:
            for b in high[: max(1, min(3, len(high)))]:
                if a != b:
                    # Add both directions with lower weight: evidence may prefer
                    # either progress or stopping depending on the scene.
                    add(int(a), int(b), 1.0 + float(lambda_progress))
                    add(int(b), int(a), 1.0 + 0.5 * float(lambda_progress))


def build_runtime_pairs_from_base(
    predicted_base_cost: np.ndarray,
    valid_mask: np.ndarray,
    cheap_safety_flags: np.ndarray,
    L0: int = 16,
    eta0: float = 1.0,
    lambda_near: float = 1.0,
    lambda_safety: float = 2.0,
    preserve_safety_pairs: bool = True,
    bidirectional_pairs: bool = True,
    reverse_pair_weight: float = 1.0,
    pair_cap_multiplier: float = 1.0,
    candidate_trajectories: np.ndarray | None = None,
    maneuver_ids: np.ndarray | None = None,
    progress_pair_count: int = 0,
    maneuver_pair_count: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Runtime pair screen using base scores plus cheap decisive-rival priors.

    The graph still uses only deployment-time information.  Besides Top-L and
    near-base pairs, it now preserves safety pairs and cheap stop/yield-vs-go
    comparisons, which are important for traffic-light-heavy cities such as Las
    Vegas where the base score often ranks progressive candidates above the
    rule-correct stop candidate.
    """
    J0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    safety = np.asarray(cheap_safety_flags, dtype=bool).reshape(-1)
    valid_idx = _valid_finite_indices(J0, valid)
    if valid_idx.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    top = valid_idx[np.argsort(J0[valid_idx])[: min(max(int(L0), 1), valid_idx.size)]]
    pair_set: dict[tuple[int, int], float] = {}
    safety_pair_set: dict[tuple[int, int], float] = {}

    def _base_order(a: int, b: int) -> tuple[int, int]:
        if (float(J0[b]), int(b)) < (float(J0[a]), int(a)):
            return int(b), int(a)
        return int(a), int(b)

    def _add_direct(a: int, b: int, w: float) -> None:
        if a == b or not valid[a] or not valid[b]:
            return
        key = (int(a), int(b))
        pair_set[key] = max(float(w), pair_set.get(key, 0.0))

    def add(a: int, b: int, w: float) -> None:
        a0, b0 = _base_order(int(a), int(b))
        _add_direct(a0, b0, w)
        if bool(bidirectional_pairs):
            rw = max(0.0, float(reverse_pair_weight)) * float(w)
            if rw > 0.0:
                _add_direct(b0, a0, rw)

    for i, a in enumerate(top.tolist()):
        for b in top.tolist()[i + 1 :]:
            add(a, b, 1.0)
    for a in valid_idx.tolist():
        near = valid_idx[np.abs(J0[valid_idx] - J0[a]) < float(eta0)]
        for b in near.tolist():
            if a != b:
                add(a, b, 1.0 + float(lambda_near))

    safe_idx = valid_idx[~safety[valid_idx]]
    unsafe_idx = valid_idx[safety[valid_idx]]
    if safe_idx.size and unsafe_idx.size:
        safe_top = safe_idx[np.argsort(J0[safe_idx])[: min(max(int(L0), 1), safe_idx.size)]]
        for a in safe_top.tolist():
            for b in unsafe_idx.tolist():
                if a == b:
                    continue
                key = (int(a), int(b))
                w = 1.0 + float(lambda_safety)
                safety_pair_set[key] = max(w, safety_pair_set.get(key, 0.0))
                pair_set[key] = max(w, pair_set.get(key, 0.0))

    _add_enhanced_decisive_pairs(
        pair_set,
        J0,
        valid_idx,
        valid,
        safety,
        candidate_trajectories,
        maneuver_ids,
        progress_pair_count=progress_pair_count,
        maneuver_pair_count=maneuver_pair_count,
    )

    if not pair_set:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    items = sorted(pair_set.items(), key=lambda kv: (abs(float(J0[kv[0][1]] - J0[kv[0][0]])), kv[0][0], kv[0][1]))
    max_pairs = max(int(L0) * max(int(L0), 1), int(L0))
    max_pairs = int(np.ceil(max_pairs * max(1.0, float(pair_cap_multiplier))))
    if preserve_safety_pairs and safety_pair_set:
        safety_items = sorted(safety_pair_set.items(), key=lambda kv: (float(J0[kv[0][0]]), float(J0[kv[0][1]]), kv[0][0], kv[0][1]))
        safety_keys = {k for k, _ in safety_items}
        regular_items = [kv for kv in items if kv[0] not in safety_keys]
        items = safety_items + regular_items[:max_pairs]
    else:
        items = items[:max_pairs]
    pairs = np.asarray([k for k, _ in items], dtype=np.int64)
    weights = np.asarray([w for _, w in items], dtype=np.float32)
    return pairs, weights


def build_rival_sets_from_base(
    predicted_base_cost: np.ndarray,
    valid_mask: np.ndarray,
    cheap_safety_flags: np.ndarray,
    L_infer: int = 16,
    eta0: float = 1.0,
    preserve_safety_pairs: bool = True,
    candidate_trajectories: np.ndarray | None = None,
    maneuver_ids: np.ndarray | None = None,
    progress_rivals: int = 0,
    maneuver_rivals: int = 0,
) -> list[list[int]]:
    J0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    safety = np.asarray(cheap_safety_flags, dtype=bool).reshape(-1)
    K = len(valid)
    valid_idx = _valid_finite_indices(J0, valid)
    top = valid_idx[np.argsort(J0[valid_idx])[: min(max(int(L_infer), 1), valid_idx.size)]] if valid_idx.size else np.zeros((0,), dtype=np.int64)
    progress = _trajectory_progress(candidate_trajectories, K)
    speed = _trajectory_terminal_speed(candidate_trajectories, K)
    man = _maneuver_array(maneuver_ids, K)
    safe_like = [int(i) for i in valid_idx.tolist() if int(man[i]) in SAFE_LIKE_MANEUVER_IDS]
    progressive = [int(i) for i in valid_idx.tolist() if int(man[i]) in PROGRESSIVE_MANEUVER_IDS]
    safe_like = sorted(safe_like, key=lambda i: (bool(safety[i]), float(speed[i]), float(J0[i]), i))[: max(0, int(maneuver_rivals))]
    high_prog = sorted(valid_idx.tolist(), key=lambda i: (-float(progress[i]), float(J0[i]), i))[: max(0, int(progress_rivals))]
    low_prog = sorted(valid_idx.tolist(), key=lambda i: (float(progress[i]), float(speed[i]), float(J0[i]), i))[: max(0, int(progress_rivals))]

    rivals: list[list[int]] = []
    for a in range(K):
        if not valid[a] or not np.isfinite(J0[a]):
            rivals.append([])
            continue
        cand = [int(b) for b in top.tolist() if int(b) != a]
        cand += [int(b) for b in valid_idx.tolist() if int(b) != a and abs(float(J0[b] - J0[a])) < float(eta0)]
        cand += [int(b) for b in valid_idx.tolist() if int(b) != a and bool(safety[b])]
        # Decisive non-base rivals: every action compares against conservative
        # stop/yield options and against progress extremes, so tournament can
        # choose between stop/proceed rather than only among base-near actions.
        cand += [b for b in safe_like if b != a]
        if int(man[a]) in SAFE_LIKE_MANEUVER_IDS:
            cand += [b for b in high_prog if b != a]
        else:
            cand += [b for b in low_prog if b != a]
        limit = max(int(L_infer), int(safety[valid_idx].sum()) if valid_idx.size else 0, len(safe_like))
        seen = []
        for b in sorted(set(cand), key=lambda x: (abs(float(J0[x] - J0[a])), x)):
            seen.append(b)
            if len(seen) >= max(1, limit):
                break
        rivals.append(seen)
    return rivals
