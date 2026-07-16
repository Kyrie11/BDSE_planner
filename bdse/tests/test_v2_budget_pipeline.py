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


def test_runtime_query_diagnostics_separates_action_pair_and_certificate_counts():
    from bdse.planner.nuplan_planner import runtime_query_diagnostics

    pred = {
        "top_m_atoms": np.asarray([2, 4, 6]),
        "queried_actions": np.asarray([0, 1, 3, 5]),
        "runtime_pairs": np.asarray([[0, 1], [1, 0], [0, 3]]),
        "rival_pair_indices": np.asarray([[0, 1], [0, 3], [1, 3], [3, 5], [5, 0]]),
        "action_atom_query_count": 12,
        "selector_pair_atom_query_count": 9,
        "tournament_pair_atom_query_count": 15,
    }
    out = runtime_query_diagnostics(pred, selected_atoms=[2, 6])
    assert out["action_atom_query_count"] == 12
    assert out["selector_pair_atom_query_count"] == 9
    assert out["tournament_pair_atom_query_count"] == 15
    assert out["sparse_query_count"] == 36
    assert out["selected_certificate_query_count"] == 10
    assert out["effective_query_count"] == 10
