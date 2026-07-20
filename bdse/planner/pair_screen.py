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
    critical_pair_set: set[tuple[int, int]] | None,
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
        if critical_pair_set is not None:
            critical_pair_set.add(key)

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
    critical_pair_set: set[tuple[int, int]] = set()

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
                critical_pair_set.add(key)

    _add_enhanced_decisive_pairs(
        pair_set,
        critical_pair_set,
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
    if preserve_safety_pairs and (safety_pair_set or critical_pair_set):
        # Preserve safety and enhanced stop/go-progress pairs before spending the
        # remaining pair budget on small-base-gap pairs.  Without this, far but
        # decisive stop-vs-go comparisons can be evicted from the runtime screen.
        critical_items = [
            (k, pair_set[k])
            for k in sorted(critical_pair_set, key=lambda k: (float(J0[k[0]]), float(J0[k[1]]), k[0], k[1]))
            if k in pair_set
        ]
        critical_keys = {k for k, _ in critical_items}
        regular_items = [kv for kv in items if kv[0] not in critical_keys]
        remaining = max(0, max_pairs - len(critical_items))
        items = critical_items + regular_items[:remaining]
    else:
        items = items[:max_pairs]
    pairs = np.asarray([k for k, _ in items], dtype=np.int64)
    weights = np.asarray([w for _, w in items], dtype=np.float32)
    return pairs, weights



def compact_runtime_pair_graph(
    pairs: np.ndarray,
    pair_weights: np.ndarray | None,
    predicted_base_cost: np.ndarray,
    valid_mask: np.ndarray,
    cheap_safety_flags: np.ndarray,
    *,
    maneuver_ids: np.ndarray | None = None,
    candidate_trajectories: np.ndarray | None = None,
    max_pairs: int = 0,
    canonicalize_reciprocals: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Compact a directed runtime pair graph before neural pair scoring.

    The final tournament consumes an antisymmetric margin matrix, so querying both
    directions of every unordered pair is redundant.  This helper keeps one
    deployment-meaningful orientation per unordered pair and then applies a strict
    pair budget while preserving safety, stop/yield-vs-go, and near-boundary pairs.
    It uses only runtime quantities and therefore does not leak teacher labels.
    """
    arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2) if np.asarray(pairs).size else np.zeros((0, 2), dtype=np.int64)
    if arr.size == 0:
        return arr, np.zeros((0,), dtype=np.float32), {
            "pair_count_before": 0.0,
            "pair_count_after": 0.0,
            "reciprocal_pairs_removed": 0.0,
            "pair_budget_pruned": 0.0,
        }
    J0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    flags = np.asarray(cheap_safety_flags, dtype=bool).reshape(-1)
    K = min(len(J0), len(valid))
    if flags.shape[0] < K:
        flags = np.pad(flags, (0, K - flags.shape[0]), constant_values=False)
    flags = flags[:K]
    man = _maneuver_array(maneuver_ids, K)
    progress = _trajectory_progress(candidate_trajectories, K)
    raw_w = np.ones((arr.shape[0],), dtype=np.float32)
    if pair_weights is not None:
        rw = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
        raw_w[: min(len(rw), len(raw_w))] = rw[: min(len(rw), len(raw_w))]

    def orient(a: int, b: int) -> tuple[int, int]:
        # Unflagged -> flagged is the safety-certificate direction.
        if bool(flags[a]) != bool(flags[b]):
            return (b, a) if bool(flags[a]) else (a, b)
        a_safe_like = int(man[a]) in SAFE_LIKE_MANEUVER_IDS
        b_safe_like = int(man[b]) in SAFE_LIKE_MANEUVER_IDS
        a_prog = int(man[a]) in PROGRESSIVE_MANEUVER_IDS
        b_prog = int(man[b]) in PROGRESSIVE_MANEUVER_IDS
        # Preserve the interpretable stop/yield -> go orientation.
        if a_safe_like and b_prog and not b_safe_like:
            return a, b
        if b_safe_like and a_prog and not a_safe_like:
            return b, a
        # Otherwise orient from the lower base-cost action to its rival.
        if (float(J0[b]), int(b)) < (float(J0[a]), int(a)):
            return b, a
        return a, b

    merged: dict[tuple[int, int], float] = {}
    original_directed: set[tuple[int, int]] = set()
    for idx, (a0, b0) in enumerate(arr.tolist()):
        a, b = int(a0), int(b0)
        if a == b or a < 0 or b < 0 or a >= K or b >= K or not valid[a] or not valid[b]:
            continue
        original_directed.add((a, b))
        key = orient(a, b) if bool(canonicalize_reciprocals) else (a, b)
        merged[key] = max(float(raw_w[idx]), merged.get(key, 0.0))

    items = list(merged.items())
    before_unique = len(original_directed)
    reciprocal_removed = max(0, before_unique - len(items))

    def priority(item: tuple[tuple[int, int], float]) -> tuple[float, float, float, int, int]:
        (a, b), w = item
        cross_safety = float(bool(flags[a]) != bool(flags[b]))
        a_safe_like = int(man[a]) in SAFE_LIKE_MANEUVER_IDS
        b_safe_like = int(man[b]) in SAFE_LIKE_MANEUVER_IDS
        a_prog = int(man[a]) in PROGRESSIVE_MANEUVER_IDS
        b_prog = int(man[b]) in PROGRESSIVE_MANEUVER_IDS
        maneuver_cross = float((a_safe_like and b_prog) or (b_safe_like and a_prog))
        gap = abs(float(J0[b] - J0[a]))
        near = 1.0 / (1.0 + gap)
        progress_span = min(abs(float(progress[b] - progress[a])) / 20.0, 2.0)
        score = 8.0 * cross_safety + 5.0 * maneuver_cross + 2.0 * float(w) + 1.5 * near + 0.5 * progress_span
        return (-score, gap, -float(w), int(a), int(b))

    items.sort(key=priority)
    cap = max(0, int(max_pairs))
    pruned = 0
    if cap > 0 and len(items) > cap:
        pruned = len(items) - cap
        items = items[:cap]
    out_pairs = np.asarray([k for k, _ in items], dtype=np.int64).reshape(-1, 2) if items else np.zeros((0, 2), dtype=np.int64)
    out_weights = np.asarray([w for _, w in items], dtype=np.float32) if items else np.zeros((0,), dtype=np.float32)
    return out_pairs, out_weights, {
        "pair_count_before": float(arr.shape[0]),
        "pair_count_after": float(out_pairs.shape[0]),
        "reciprocal_pairs_removed": float(reciprocal_removed),
        "pair_budget_pruned": float(pruned),
    }

def restrict_pairs_to_viability_frontier(
    pairs: np.ndarray,
    pair_weights: np.ndarray | None,
    valid_mask: np.ndarray,
    safety_flags: np.ndarray,
    predicted_base_cost: np.ndarray,
    *,
    hard_risk: np.ndarray | None = None,
    frontier_size: int = 8,
    single_safe_rivals: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Restrict the *selector* graph to decisions not already settled by safety.

    Hard feasibility is handled lexicographically by the runtime safety channel.
    The budgeted evidence selector should therefore spend its pair queries on
    comparisons inside the viable set.  When there is only one safe action, keep
    a small anchor graph around it for diagnostics/calibration.  If all candidates
    are flagged, fall back to a minimum-risk frontier instead of silently dropping
    every pair.

    The function never changes the final tournament rival graph; it only removes
    comparisons that cannot influence the decision after the hard safety guard.
    """
    arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2) if np.asarray(pairs).size else np.zeros((0, 2), dtype=np.int64)
    raw_w = np.ones((arr.shape[0],), dtype=np.float32)
    if pair_weights is not None:
        w = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
        raw_w[: min(len(w), len(raw_w))] = w[: min(len(w), len(raw_w))]
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    flags = np.asarray(safety_flags, dtype=bool).reshape(-1)
    J0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    K = min(len(valid), len(J0))
    valid = valid[:K]
    if flags.shape[0] < K:
        flags = np.pad(flags, (0, K - flags.shape[0]), constant_values=False)
    flags = flags[:K]
    if arr.size == 0 or K == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), {
            "pair_count_before_viability": float(arr.shape[0]),
            "pair_count_after_viability": 0.0,
            "viability_safe_action_count": float((valid & ~flags).sum()),
            "viability_scope_code": 0.0,
        }

    safe = valid & ~flags
    safe_idx = np.flatnonzero(safe)
    keep = np.zeros((arr.shape[0],), dtype=bool)
    scope_code = 0
    if safe_idx.size >= 2:
        a = arr[:, 0]; b = arr[:, 1]
        ok = (a >= 0) & (a < K) & (b >= 0) & (b < K)
        keep = ok & safe[np.clip(a, 0, K - 1)] & safe[np.clip(b, 0, K - 1)]
        scope_code = 3  # safe-safe
    elif safe_idx.size == 1:
        anchor = int(safe_idx[0])
        a = arr[:, 0]; b = arr[:, 1]
        ok = (a >= 0) & (a < K) & (b >= 0) & (b < K)
        incident = ok & ((a == anchor) | (b == anchor))
        incident_ids = np.flatnonzero(incident).tolist()
        incident_ids.sort(key=lambda i: (
            abs(float(J0[int(arr[i, 1])] - J0[int(arr[i, 0])])),
            -float(raw_w[i]),
            int(arr[i, 0]),
            int(arr[i, 1]),
        ))
        keep[np.asarray(incident_ids[: max(1, int(single_safe_rivals))], dtype=np.int64)] = True
        scope_code = 2  # one-safe anchor
    else:
        risk = np.full((K,), np.inf, dtype=np.float32)
        if hard_risk is not None:
            r = np.asarray(hard_risk, dtype=np.float32).reshape(-1)
            risk[: min(K, len(r))] = r[: min(K, len(r))]
        finite_valid = np.flatnonzero(valid & np.isfinite(J0))
        order = sorted(
            finite_valid.tolist(),
            key=lambda i: (float(risk[i]) if np.isfinite(risk[i]) else float('inf'), float(J0[i]), int(i)),
        )
        frontier = set(order[: max(2, int(frontier_size))])
        a = arr[:, 0]; b = arr[:, 1]
        keep = np.asarray([(int(x) in frontier and int(y) in frontier) for x, y in arr.tolist()], dtype=bool)
        scope_code = 1  # all-flagged minimum-risk frontier

    out_pairs = arr[keep]
    out_weights = raw_w[keep]
    # Do not return an empty selector graph when the input contained usable pairs.
    # A tiny near-base fallback is preferable to arbitrary proposal-only filling.
    if out_pairs.shape[0] == 0 and arr.shape[0] > 0:
        order = sorted(
            range(arr.shape[0]),
            key=lambda i: (
                abs(float(J0[int(arr[i, 1])] - J0[int(arr[i, 0])]))
                if 0 <= int(arr[i, 0]) < K and 0 <= int(arr[i, 1]) < K else float('inf'),
                -float(raw_w[i]),
                int(arr[i, 0]),
                int(arr[i, 1]),
            ),
        )
        take = np.asarray(order[: min(max(1, int(single_safe_rivals)), len(order))], dtype=np.int64)
        out_pairs = arr[take]
        out_weights = raw_w[take]
        scope_code = -1  # defensive near-base fallback

    return out_pairs.astype(np.int64), out_weights.astype(np.float32), {
        "pair_count_before_viability": float(arr.shape[0]),
        "pair_count_after_viability": float(out_pairs.shape[0]),
        "viability_safe_action_count": float(safe_idx.size),
        "viability_scope_code": float(scope_code),
    }


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
        base_cand = [int(b) for b in top.tolist() if int(b) != a]
        base_cand += [int(b) for b in valid_idx.tolist() if int(b) != a and abs(float(J0[b] - J0[a])) < float(eta0)]
        safety_cand = [int(b) for b in valid_idx.tolist() if int(b) != a and bool(safety[b])]
        # Decisive non-base rivals are preserved first, then base-near/top-L
        # rivals fill the remainder.  This keeps safety/progress alternatives in
        # every local tournament even if their base scores are far apart.
        critical = safety_cand + [b for b in safe_like if b != a]
        if int(man[a]) in SAFE_LIKE_MANEUVER_IDS:
            critical += [b for b in high_prog if b != a]
        else:
            critical += [b for b in low_prog if b != a]
        limit = max(int(L_infer), len(set(safety_cand)), len(set(safe_like)))
        seen: list[int] = []
        seen_set: set[int] = set()

        def add_ordered(seq, *, sort_key, max_total: int | None = None):
            for b in sorted(set(seq), key=sort_key):
                if max_total is not None and len(seen) >= max_total:
                    break
                if b == a or b in seen_set:
                    continue
                seen.append(int(b))
                seen_set.add(int(b))

        add_ordered(critical, sort_key=lambda x: (bool(safety[x]), abs(float(J0[x] - J0[a])), x), max_total=None)
        target = max(1, max(limit, len(seen)))
        add_ordered(base_cand, sort_key=lambda x: (abs(float(J0[x] - J0[a])), x), max_total=target)
        rivals.append(seen)
    return rivals


def reweight_pairs_by_viability_scope(
    pairs: np.ndarray,
    pair_weights: np.ndarray | None,
    valid_mask: np.ndarray,
    safety_flags: np.ndarray,
    predicted_base_cost: np.ndarray,
    *,
    hard_risk: np.ndarray | None = None,
    safe_safe_weight: float = 1.0,
    cross_safety_weight: float = 0.35,
    unsafe_unsafe_weight: float = 0.10,
    all_flagged_frontier_size: int = 8,
    outside_frontier_weight: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Keep the runtime pair graph intact while emphasizing viable decisions.

    V36 removed every pair outside the current hard-feasible frontier.  That is
    appropriate for a perfect safety oracle, but runtime flags are conservative
    proxies and the unchanged v30 checkpoint was trained on a broader pair graph.
    Hard removal therefore reduced teacher-margin fidelity.  This function keeps
    all deployment pairs and only changes their acquisition weights: safe-safe
    comparisons dominate, cross-safety pairs remain available for calibration,
    and unsafe-unsafe comparisons receive a small weight.  In all-flagged scenes
    a minimum-risk frontier is emphasized without erasing the remaining graph.
    """
    arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2) if np.asarray(pairs).size else np.zeros((0, 2), dtype=np.int64)
    weights = np.ones((arr.shape[0],), dtype=np.float32)
    if pair_weights is not None:
        raw = np.asarray(pair_weights, dtype=np.float32).reshape(-1)
        weights[: min(len(raw), len(weights))] = raw[: min(len(raw), len(weights))]
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    flags = np.asarray(safety_flags, dtype=bool).reshape(-1)
    J0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    K = min(len(valid), len(J0))
    valid = valid[:K]
    if flags.shape[0] < K:
        flags = np.pad(flags, (0, K - flags.shape[0]), constant_values=False)
    flags = flags[:K]
    safe = valid & ~flags
    safe_count = int(safe.sum())
    scope_code = 4.0
    frontier: set[int] = set()
    if safe_count == 0:
        risk = np.full((K,), np.inf, dtype=np.float32)
        if hard_risk is not None:
            raw_risk = np.asarray(hard_risk, dtype=np.float32).reshape(-1)
            risk[: min(K, len(raw_risk))] = raw_risk[: min(K, len(raw_risk))]
        finite_valid = np.flatnonzero(valid & np.isfinite(J0))
        order = sorted(
            finite_valid.tolist(),
            key=lambda i: (float(risk[i]) if np.isfinite(risk[i]) else float("inf"), float(J0[i]), int(i)),
        )
        frontier = set(order[: max(2, int(all_flagged_frontier_size))])
        scope_code = 1.0

    multipliers = np.ones((arr.shape[0],), dtype=np.float32)
    for idx, (a_raw, b_raw) in enumerate(arr.tolist()):
        a, b = int(a_raw), int(b_raw)
        if a < 0 or b < 0 or a >= K or b >= K or not valid[a] or not valid[b]:
            multipliers[idx] = 0.0
            continue
        if safe_count > 0:
            a_safe, b_safe = bool(safe[a]), bool(safe[b])
            if a_safe and b_safe:
                multipliers[idx] = max(float(safe_safe_weight), 0.0)
            elif a_safe != b_safe:
                multipliers[idx] = max(float(cross_safety_weight), 0.0)
            else:
                multipliers[idx] = max(float(unsafe_unsafe_weight), 0.0)
        else:
            inside = a in frontier and b in frontier
            multipliers[idx] = 1.0 if inside else max(float(outside_frontier_weight), 0.0)
    out = np.maximum(weights * multipliers, 1e-4).astype(np.float32)
    return arr.astype(np.int64), out, {
        "pair_count_before_viability": float(arr.shape[0]),
        "pair_count_after_viability": float(arr.shape[0]),
        "viability_safe_action_count": float(safe_count),
        "viability_scope_code": float(scope_code),
        "viability_pair_weight_mean": float(out.mean()) if out.size else 0.0,
        "viability_pair_weight_min": float(out.min()) if out.size else 0.0,
        "viability_pair_weight_max": float(out.max()) if out.size else 0.0,
        "viability_frontier_size": float(len(frontier)),
    }
