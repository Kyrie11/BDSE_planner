from __future__ import annotations

import numpy as np


def build_runtime_pairs_from_base(
    predicted_base_cost: np.ndarray,
    valid_mask: np.ndarray,
    cheap_safety_flags: np.ndarray,
    L0: int = 16,
    eta0: float = 1.0,
    lambda_near: float = 1.0,
    lambda_safety: float = 2.0,
    preserve_safety_pairs: bool = True,
    bidirectional_pairs: bool = False,
    reverse_pair_weight: float = 0.5,
    pair_cap_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Runtime pair screen using only base scores and cheap safety flags.

    By default the lower predicted base-cost action is stored first, matching the
    original runtime graph.  When ``bidirectional_pairs`` is enabled, ordinary
    top/near rival pairs are also stored in the reverse direction with a smaller
    weight.  This is important for deployment-style evidence selection: evidence
    often needs to overturn the base winner in favor of a safer/route-consistent
    action, and a one-way lower-base graph cannot score negative evidence deltas
    as useful margin support.  Cheap safe-vs-unsafe pairs remain oriented
    safe->unsafe and outside the ordinary pair cap.
    """
    J0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    safety = np.asarray(cheap_safety_flags, dtype=bool).reshape(-1)
    valid_idx = np.flatnonzero(valid & np.isfinite(J0))
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
        # Safety pairs are not evidence budget; they only decide which rival
        # comparisons are considered.  Keep them outside the ordinary pair cap so
        # a near/base pair screen cannot drop the only safe-vs-unsafe certificate.
        for a in safe_top.tolist():
            for b in unsafe_idx.tolist():
                if a == b:
                    continue
                key = (int(a), int(b))
                w = 1.0 + float(lambda_safety)
                safety_pair_set[key] = max(w, safety_pair_set.get(key, 0.0))
                pair_set[key] = max(w, pair_set.get(key, 0.0))
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
) -> list[list[int]]:
    J0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    safety = np.asarray(cheap_safety_flags, dtype=bool).reshape(-1)
    K = len(valid)
    valid_idx = np.flatnonzero(valid & np.isfinite(J0))
    top = valid_idx[np.argsort(J0[valid_idx])[: min(max(int(L_infer), 1), valid_idx.size)]] if valid_idx.size else np.zeros((0,), dtype=np.int64)
    rivals: list[list[int]] = []
    for a in range(K):
        if not valid[a] or not np.isfinite(J0[a]):
            rivals.append([])
            continue
        cand = [int(b) for b in top.tolist() if int(b) != a]
        cand += [int(b) for b in valid_idx.tolist() if int(b) != a and abs(float(J0[b] - J0[a])) < float(eta0)]
        cand += [int(b) for b in valid_idx.tolist() if int(b) != a and bool(safety[b])]
        seen = []
        for b in sorted(set(cand), key=lambda x: (abs(float(J0[x] - J0[a])), x)):
            seen.append(b)
            if len(seen) >= max(int(L_infer), int(safety[valid_idx].sum())):
                break
        rivals.append(seen)
    return rivals
