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
) -> tuple[np.ndarray, np.ndarray]:
    """Runtime pair screen using only base scores and cheap safety flags.

    The better/base-lower action is stored first.  No full-interface evidence
    margin is read here; this is the inference-time rival graph required by the
    two-stage BDSE certificate pipeline.
    """
    J0 = np.asarray(predicted_base_cost, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    safety = np.asarray(cheap_safety_flags, dtype=bool).reshape(-1)
    K = len(valid)
    valid_idx = np.flatnonzero(valid & np.isfinite(J0))
    if valid_idx.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    top = valid_idx[np.argsort(J0[valid_idx])[: min(max(int(L0), 1), valid_idx.size)]]
    pair_set: dict[tuple[int, int], float] = {}

    def add(a: int, b: int, w: float) -> None:
        if a == b or not valid[a] or not valid[b]:
            return
        # Store lower predicted base cost first.  Ties break by index.
        if (float(J0[b]), int(b)) < (float(J0[a]), int(a)):
            a, b = b, a
        key = (int(a), int(b))
        pair_set[key] = max(float(w), pair_set.get(key, 0.0))

    for i, a in enumerate(top.tolist()):
        for b in top.tolist()[i + 1 :]:
            add(a, b, 1.0)
    for i, a in enumerate(valid_idx.tolist()):
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
                # Preserve safe-vs-unsafe orientation when possible.
                key = (int(a), int(b)) if a != b else None
                if key is not None:
                    pair_set[key] = max(1.0 + float(lambda_safety), pair_set.get(key, 0.0))
    if not pair_set:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    items = sorted(pair_set.items(), key=lambda kv: (abs(float(J0[kv[0][1]] - J0[kv[0][0]])), kv[0][0], kv[0][1]))
    max_pairs = max(int(L0) * max(int(L0), 1), int(L0))
    items = items[: max_pairs]
    pairs = np.asarray([k for k, _ in items], dtype=np.int64)
    weights = np.asarray([w for _, w in items], dtype=np.float32)
    return pairs, weights


def build_rival_sets_from_base(
    predicted_base_cost: np.ndarray,
    valid_mask: np.ndarray,
    cheap_safety_flags: np.ndarray,
    L_infer: int = 16,
    eta0: float = 1.0,
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
