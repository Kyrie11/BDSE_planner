from __future__ import annotations

import numpy as np

from bdse.planner.hab import select_topm_atoms_hab
from bdse.planner.selector import runtime_greedy_selector
from bdse.planner.tournament import tournament_scores


def test_hab_topm_allocates_slots_across_active_families():
    logits = np.asarray([10.0, 9.0, 8.0, 7.0], dtype=np.float32)
    family_ids = np.asarray([1, 1, 2, 2], dtype=np.int64)
    active = np.ones((4,), dtype=bool)
    costs = np.ones((4,), dtype=np.float32)
    # Equal family logits and M=2 should give one proposal slot to each family,
    # then choose the highest-scoring atom within that family.
    top, fam_budget, diag = select_topm_atoms_hab(
        logits, family_ids, active, costs, budget=2, proposal_top_m=2, family_scores=np.asarray([0.0, 1.0, 1.0], dtype=np.float32)
    )
    assert set(top.tolist()) == {0, 2}
    assert fam_budget.family_caps[1] > 0 and fam_budget.family_caps[2] > 0
    assert diag["hab_enabled"] is True


def test_runtime_selector_respects_family_budget_caps():
    J0 = np.asarray([0.0, 0.05, 0.10], dtype=np.float32)
    valid = np.ones((3,), dtype=bool)
    flags = np.zeros((3,), dtype=bool)
    # Both atoms help the same screened pair, but the family cap allows spending
    # on only one atom from family 1.
    g = np.asarray([[0.0, 2.0, 2.0], [0.0, 1.5, 1.5]], dtype=np.float32)
    result = runtime_greedy_selector(
        J0,
        g,
        np.ones((2,), dtype=np.float32),
        valid,
        flags,
        budget=2,
        L_infer=3,
        eta_pred=1.0,
        atom_active_mask=np.ones((2,), dtype=bool),
        family_ids=np.asarray([1, 1], dtype=np.int64),
        family_budget_caps=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    assert len(result.selected) <= 1
    assert result.diagnostics["mode"] == "runtime_hab_lcb_uncertainty"


def test_uncertainty_penalty_changes_tournament_scores():
    margins = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [-1.0, 0.0, 1.0],
            [-2.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones((3,), dtype=bool)
    rivals = [[1, 2], [0, 2], [0, 1]]
    no_unc = tournament_scores(margins, valid, rivals, softmin_tau=0.0)
    sigma = np.zeros_like(margins)
    sigma[0, 1] = 5.0
    sigma[0, 2] = 5.0
    with_unc = tournament_scores(margins, valid, rivals, softmin_tau=0.0, beta_uncertainty=1.0, sigma=sigma)
    assert int(np.argmax(no_unc)) == 0
    assert with_unc[0] < no_unc[0]
