from __future__ import annotations

import numpy as np

from bdse.config import load_config
from bdse.model.bdse_model import BDSEModel
from bdse.planner.evidence_queries import FAMILY_NAMES
from bdse.planner.pair_screen import build_runtime_pairs_from_base


def test_model_family_embedding_covers_dynamic_regularity_even_if_config_is_stale():
    cfg = load_config(None)
    cfg["model"] = dict(cfg.get("model", {}))
    cfg["model"]["num_families"] = 5
    model = BDSEModel(cfg)
    assert model.num_families > FAMILY_NAMES["dynamic_regularity"]


def test_safety_pairs_are_not_truncated_by_regular_pair_cap():
    K = 32
    j0 = np.linspace(0.0, 100.0, K, dtype=np.float32)
    valid = np.ones((K,), dtype=bool)
    unsafe = np.zeros((K,), dtype=bool)
    unsafe[20:] = True
    pairs, _ = build_runtime_pairs_from_base(j0, valid, unsafe, L0=4, eta0=0.01, lambda_safety=2.0)
    pair_set = {tuple(map(int, p)) for p in pairs.tolist()}
    safe_top = [0, 1, 2, 3]
    for a in safe_top:
        for b in range(20, K):
            assert (a, b) in pair_set
