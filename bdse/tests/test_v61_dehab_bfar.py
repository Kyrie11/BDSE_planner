from __future__ import annotations

import numpy as np
import torch

from bdse.experiments.train import _validation_competitive_score
from bdse.model.losses import (
    _fast_topm_mask_torch,
    _runtime_hab_topm_hard_mask,
    _straight_through_topm_mask,
)
from bdse.planner.hab import select_topm_atoms_hab
from bdse.planner.selector import (
    reserve_topm_candidates,
    restrict_topm_to_decision_evidence,
    structural_safety_mask,
)
from bdse.tools.check_v61_dehab_bfar_dbap_gate import _set_residual_observability


def _proposal_cfg() -> dict:
    return {
        "evidence": {"budget": 3},
        "selector": {
            "proposal_top_m": 4,
            "hab_enabled": True,
            "hab_reserve_fraction": 0.2,
            "decision_budget_excludes_structural_safety": True,
            "structural_safety_include_feasibility": True,
            "interaction_family_ids": [2, 3],
            "min_soft_interaction_topm_slots": 1,
            "min_family_topm_slots": {2: 1, 3: 1},
        },
    }


def test_st_topm_is_translation_invariant_in_forward_and_backward() -> None:
    weights = torch.tensor([[1.0, -2.0, 3.0, 0.5]])
    active = torch.tensor([[True, True, True, False]])
    x1 = torch.tensor([[0.1, 2.0, 1.0, -3.0]], requires_grad=True)
    x2 = (x1.detach() + 37.0).requires_grad_(True)
    st1, hard1 = _straight_through_topm_mask(x1, active, 2, 0.2)
    st2, hard2 = _straight_through_topm_mask(x2, active, 2, 0.2)
    (st1 * weights).sum().backward()
    (st2 * weights).sum().backward()
    assert torch.equal(hard1, hard2)
    assert torch.equal(st1.detach(), hard1.float())
    assert torch.equal(st2.detach(), hard2.float())
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6, rtol=1e-6)


def test_fast_hab_uses_family_scores_and_excludes_structural_atoms() -> None:
    cfg = _proposal_cfg()
    cfg["selector"]["proposal_top_m"] = 3
    logits = torch.tensor([[9.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]])
    active = torch.ones_like(logits, dtype=torch.bool)
    costs = torch.ones_like(logits)
    # Atom 0 is feasibility/structural and must not consume decision Top-M.
    family_ids = torch.tensor([[1, 2, 2, 2, 3, 3, 4, 5]])
    features = torch.zeros((1, 8, 2))
    favor_four = torch.tensor([[0.0, -100.0, 5.0, 4.0, 3.0, -20.0]])
    favor_five = torch.tensor([[0.0, -100.0, 5.0, 4.0, -20.0, 3.0]])
    mask_four, _ = _fast_topm_mask_torch(
        logits, active, costs, family_ids, features, cfg, family_scores=favor_four
    )
    mask_five, _ = _fast_topm_mask_torch(
        logits, active, costs, family_ids, features, cfg, family_scores=favor_five
    )
    selected_four = torch.nonzero(mask_four[0], as_tuple=False).flatten().tolist()
    selected_five = torch.nonzero(mask_five[0], as_tuple=False).flatten().tolist()
    assert 0 not in selected_four and 0 not in selected_five
    assert len(selected_four) == 3 and len(selected_five) == 3
    assert 6 in selected_four and 7 not in selected_four
    assert 7 in selected_five and 6 not in selected_five
    assert any(int(family_ids[0, i]) == 2 for i in selected_four)
    assert any(int(family_ids[0, i]) == 3 for i in selected_four)


def test_exact_runtime_hab_mask_matches_deployment_helpers() -> None:
    cfg = _proposal_cfg()
    logits = torch.tensor([[5.0, 1.0, 0.9, 0.8, 0.7, 0.6]])
    family_logits = torch.tensor([[0.0, -4.0, 2.0, 1.0, 0.0, -1.0]])
    family_ids = torch.tensor([[1, 2, 2, 3, 3, 4]])
    active = torch.ones_like(logits, dtype=torch.bool)
    costs = torch.ones_like(logits)
    features = torch.zeros((1, 6, 2))
    groups = torch.tensor([[-1, 10, 11, 20, 21, -1]])
    outputs = {"proposal_logits": logits, "family_logits": family_logits}
    batch = {
        "evidence_active": active,
        "evidence_budget_costs": costs,
        "evidence_family_ids": family_ids,
        "evidence_agent_group_ids": groups,
        "evidence_features": features,
        "decisive_hard_mask": torch.zeros_like(active),
    }
    got = _runtime_hab_topm_hard_mask(outputs, batch, cfg)[0].cpu().numpy()

    l = logits[0].numpy()
    f = family_ids[0].numpy()
    a = active[0].numpy()
    c = costs[0].numpy()
    topm, _, _ = select_topm_atoms_hab(
        l, f, a, c, 3, 4, family_scores=family_logits[0].numpy(),
        reserve_fraction=0.2, enabled=True, min_family_slots={2: 1, 3: 1},
    )
    hard = features[0, :, 0].numpy() > 0.5
    soft = np.isin(f, [2, 3]) & a & ~hard
    protected = structural_safety_mask(hard, f, a, include_feasibility=True)
    topm, _ = reserve_topm_candidates(
        topm, soft, l, 4, 1, protected_mask=None, group_ids=groups[0].numpy()
    )
    topm, _ = restrict_topm_to_decision_evidence(topm, a & ~protected, l, 4, family_ids=f)
    expected = np.zeros((6,), dtype=bool)
    expected[topm] = True
    assert np.array_equal(got, expected)


def test_checkpoint_score_prefers_minimum_gate_feasible_epoch() -> None:
    common = {
        "val_teacher_action_match": 0.141,
        "val_selected_local_anchor_action_match": 0.141,
        "val_pair_full_interface_action_match": 0.141,
        "val_beneficial_residual_intervention_rate": 0.0,
        "val_harmful_residual_intervention_rate": 0.0,
        "val_effective_selected_decisive_atom_recall": 0.73,
        "val_selected_interaction_decisive_recall": 0.54,
        "val_pair_action_anchor_guard_evidence_certificate_fraction": 0.88,
        "val_fallback_would_trigger_rate": 0.0,
        "val_full_interface_action_match": 0.359,
        "val_sparse_full_interface_action_match": 0.141,
        "val_budget_vs_full_match": 0.172,
    }
    feasible = dict(common, val_proposal_decisive_atom_recall=0.726, val_selected_decisive_atom_recall=0.567)
    infeasible = dict(common, val_proposal_decisive_atom_recall=0.700, val_selected_decisive_atom_recall=0.537)
    assert _validation_competitive_score(feasible) > _validation_competitive_score(infeasible)


def test_set_residual_observability_rejects_stale_rows(tmp_path) -> None:
    cfg = tmp_path / "candidate.yaml"
    cfg.write_text(
        "model:\n  set_residual_rank: 8\nruntime:\n  pair_tournament_aggregation_mode: evidence_action_potential\n",
        encoding="utf-8",
    )
    failures, stats = _set_residual_observability([{"teacher_action": 0}], cfg)
    assert failures
    assert stats["expected"] is True
