from __future__ import annotations

import numpy as np

from bdse.experiments.train import sample_to_tensors
from bdse.planner.nuplan_planner import BDSEPlannerCore


def test_train_tensorizer_has_v2_inputs_without_teacher_runtime_mask(synthetic_sample, cfg):
    batch = sample_to_tensors(synthetic_sample, cfg)
    assert "runtime_selected_mask" not in batch
    for key in [
        "map_polylines",
        "route_polylines",
        "traffic_control_tokens",
        "mission_goal",
        "evidence_proposal_features",
    ]:
        assert key in batch
    assert batch["evidence_query_features"].shape[0] == int(cfg["evidence"]["max_atoms"])


def test_runtime_plan_does_not_need_full_interface(monkeypatch, synthetic_sample, cfg):
    def boom(*args, **kwargs):
        raise AssertionError("full_interface_margin must not be used on runtime path")

    import bdse.planner.selector as selector
    import bdse.planner.tournament as tournament

    monkeypatch.setattr(selector, "full_interface_margin", boom)
    monkeypatch.setattr(tournament, "full_interface_margin", boom)
    local_cfg = dict(cfg)
    local_cfg["fallback"] = {**cfg.get("fallback", {}), "enabled": False}
    core = BDSEPlannerCore(model=None, cfg=local_cfg)
    action, trajectory, diag = core.plan_from_runtime(synthetic_sample.runtime)
    assert isinstance(action, int)
    assert trajectory.shape[-1] >= 5
    assert diag["sparse_query_count"] <= max(1, len(diag["proposal_top_m_atoms"])) * max(1, len(diag["queried_actions"]))
    assert diag["tournament"]["rival_source"] == "base_score_cheap_flags"


def test_proposal_top_m_limits_selected_atoms(synthetic_sample, cfg):
    local_cfg = dict(cfg)
    local_cfg["selector"] = {**cfg.get("selector", {}), "proposal_top_m": 3}
    local_cfg["fallback"] = {**cfg.get("fallback", {}), "enabled": False}
    core = BDSEPlannerCore(model=None, cfg=local_cfg)
    _, _, diag = core.plan_from_runtime(synthetic_sample.runtime)
    topm = set(diag["proposal_top_m_atoms"])
    assert len(topm) <= 3
    assert set(diag["selected_atoms"]).issubset(topm)
