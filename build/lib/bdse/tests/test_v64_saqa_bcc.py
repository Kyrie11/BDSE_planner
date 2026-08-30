from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from bdse.config import load_config
from bdse.data.cache_schema import CandidateBank, EvidenceAtom, EvidenceBank, RuntimeFeatures
from bdse.data.tensorizer import evidence_arrays
from bdse.model.bdse_model import BDSEModel
from bdse.model.checkpoint_contract import load_bdse_state_with_contract
from bdse.model.losses import _exact_winner_flip_critical_proposal_loss


def _cfg() -> dict:
    return load_config("bdse/configs/v64_saqa_bcc_train_2gpu.yaml")


def test_zero_init_query_extension_adapter_preserves_legacy_projection() -> None:
    cfg_adapter = _cfg()
    cfg_legacy = copy.deepcopy(cfg_adapter)
    cfg_legacy["model"]["query_extension_adapter"]["enabled"] = False
    torch.manual_seed(7)
    adapted = BDSEModel(cfg_adapter).eval()
    legacy = BDSEModel(cfg_legacy).eval()
    legacy.load_state_dict(
        {k: v for k, v in adapted.state_dict().items() if k in legacy.state_dict()},
        strict=True,
    )
    q = torch.randn(2, 5, 18)
    with torch.no_grad():
        a = adapted._project_query(q)
        b = legacy._project_query(q)
        # Unsupported extension channels must be inert at warm-start step zero.
        c = adapted._project_query(torch.cat([q[..., :12], torch.zeros_like(q[..., 12:])], dim=-1))
    assert torch.equal(a, b)
    assert torch.equal(a, c)


def test_prefix_cache_runtime_extension_assembles_18d_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    cfg["runtime"]["dense_query_feature_source"] = "cache_prefix_runtime_extension"
    candidates = CandidateBank(
        trajectories=np.zeros((2, 2, 5), dtype=np.float32),
        valid_mask=np.array([True, True]),
        maneuver_ids=np.zeros((2,), dtype=np.int64),
        theta=[{}, {}],
        dynamic_flags=[{}, {}],
        metadata=[{}, {}],
    )
    atoms = [
        EvidenceAtom(
            atom_id=i,
            type="yield",
            anchor={},
            budget_cost=1.0,
            is_hard=False,
            family="interaction",
            active_mask=True,
        )
        for i in range(2)
    ]
    cached = np.arange(2 * 2 * 12, dtype=np.float32).reshape(2, 2, 12)
    evidence = EvidenceBank(
        atoms=atoms,
        query_features=cached,
        active_mask=np.array([True, True]),
    )
    runtime = RuntimeFeatures(
        ego_history=np.zeros((2, 8), dtype=np.float32),
        agent_history=np.zeros((1, 2, 8), dtype=np.float32),
        agent_valid=np.ones((1,), dtype=bool),
        current_agents=np.zeros((1, 8), dtype=np.float32),
        map_features={},
        traffic_lights=[],
        route_roadblock_ids=[],
        mission_goal=None,
    )
    extension = np.full((2, 2, 6), 3.25, dtype=np.float32)
    monkeypatch.setattr(
        "bdse.data.tensorizer.compute_query_feature_extension",
        lambda *args, **kwargs: extension,
    )
    out = evidence_arrays(evidence, candidates, runtime, cfg, include_dense_query=True)
    q = out["evidence_query_features"][:2, :2]
    assert q.shape[-1] == 18
    assert np.array_equal(q[..., :12], cached)
    assert np.array_equal(q[..., 12:], extension)


def test_checkpoint_contract_rejects_core_shape_mismatch() -> None:
    cfg = _cfg()
    model = BDSEModel(cfg)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["query_proj.0.weight"] = state["query_proj.0.weight"][:, :-1]
    with pytest.raises(ValueError, match="core-state contract"):
        load_bdse_state_with_contract(model, state, cfg, context="unit-test")


def test_checkpoint_contract_allows_missing_new_adapter_only() -> None:
    cfg = _cfg()
    model = BDSEModel(cfg)
    state = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if not k.startswith("query_extension_proj.")
    }
    report = load_bdse_state_with_contract(model, state, cfg, context="unit-test")
    assert report["core_contract_pass"] is True
    assert any(k.startswith("query_extension_proj.") for k in report["missing"])


def test_counterfactual_coverage_loss_backpropagates_to_soft_hab_mask() -> None:
    J0 = torch.tensor([[0.0, 0.2]])
    g = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, True]])
    active = torch.tensor([[True, True]])
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    deployed = torch.tensor([[False, True]])
    soft = torch.tensor([[0.2, 0.8]], requires_grad=True)
    cfg = {
        "training": {
            "exact_winner_flip_criticality": {
                "enabled": True,
                "target_source": "model_dense",
                "positive_weight": 8.0,
                "negative_weight": 0.25,
                "rank_weight": 0.0,
                "pairwise_rank_weight": 0.0,
                "coverage_weight": 2.0,
                "min_action_scale": 1.0,
            }
        }
    }
    loss, *_ = _exact_winner_flip_critical_proposal_loss(
        J0,
        g,
        valid,
        active,
        logits,
        deployed,
        torch.tensor([1]),
        torch.ones((1, 2)),
        cfg,
        deployment_soft_mask=soft,
    )
    loss.backward()
    assert soft.grad is not None
    assert float(soft.grad.abs().sum()) > 0.0
