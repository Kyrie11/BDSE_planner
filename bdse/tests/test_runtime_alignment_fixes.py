from __future__ import annotations

import numpy as np
import torch

from bdse.model.scene_encoder import SceneEncoder
from bdse.planner.nuplan_planner import BDSEPlannerCore


def test_scene_encoder_ignores_invalid_padding_token_values():
    torch.manual_seed(0)
    enc = SceneEncoder(hidden_dim=16, layers=1, heads=4, dropout=0.0, map_feature_dim=8, route_feature_dim=8, traffic_feature_dim=12, goal_feature_dim=4)
    enc.eval()
    base = {
        "ego_history": torch.zeros(1, 3, 5),
        "agent_history": torch.zeros(1, 2, 3, 10),
        "agent_valid": torch.tensor([[True, False]]),
        "map_polylines": torch.zeros(1, 2, 4, 8),
        "map_polyline_valid": torch.tensor([[True, False]]),
        "route_polylines": torch.zeros(1, 1, 4, 8),
        "route_token_valid": torch.tensor([[True]]),
        "traffic_control_tokens": torch.zeros(1, 2, 12),
        "traffic_token_valid": torch.tensor([[False, False]]),
        "mission_goal": torch.zeros(1, 4),
        "mission_goal_valid": torch.tensor(False),
    }
    noisy = {k: v.clone() if torch.is_tensor(v) else v for k, v in base.items()}
    noisy["agent_history"][0, 1] = 999.0
    noisy["map_polylines"][0, 1] = -999.0
    noisy["traffic_control_tokens"][0] = 123.0
    with torch.no_grad():
        a = enc(base)
        b = enc(noisy)
    assert torch.allclose(a, b, atol=1e-5)


def test_teacher_hard_priority_prefers_safe_candidate_when_available(synthetic_sample):
    teacher = synthetic_sample.teacher
    valid = synthetic_sample.candidates.valid_mask.astype(bool)
    hard = teacher.hard_violation_mask.astype(bool) & valid
    safe = (~teacher.hard_violation_mask.astype(bool)) & valid
    if hard.any() and safe.any():
        assert not hard[int(teacher.a_star)]
        assert float(teacher.J_T[np.flatnonzero(safe)].min()) < float(teacher.J_T[np.flatnonzero(hard)].min())


def test_fallback_expands_budget_and_requeries_when_confidence_low(synthetic_sample, cfg):
    local_cfg = dict(cfg)
    local_cfg["selector"] = {**cfg.get("selector", {}), "proposal_top_m": 1}
    local_cfg["evidence"] = {**cfg.get("evidence", {}), "budget": 1}
    local_cfg["fallback"] = {
        **cfg.get("fallback", {}),
        "enabled": True,
        "tau_delta": 1e9,
        "rival_stages": [2, 4],
        "budget_stages": [1, 4],
        "proposal_multiplier": 2.0,
        "rule_rerank_top_k": 0,
    }
    core = BDSEPlannerCore(model=None, cfg=local_cfg)
    _, _, diag = core.plan_from_runtime(synthetic_sample.runtime)
    records = diag["fallback_stage_records"]
    assert diag["fallback_triggered"]
    assert len(records) >= 2
    assert max(len(r["top_m_atoms"]) for r in records) >= len(records[0]["top_m_atoms"])
    assert max(r["sparse_query_count"] for r in records) >= records[0]["sparse_query_count"]
