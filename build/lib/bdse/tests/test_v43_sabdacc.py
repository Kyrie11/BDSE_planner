from __future__ import annotations

from pathlib import Path

import numpy as np

from bdse.config import load_config
from bdse.planner.selector import runtime_greedy_selector_pair_conditioned


_TARGET_ATOMS = {5, 6, 7}


def _utility_stage_evaluator(selected: list[int], *, with_diagnostics: bool):
    state = set(map(int, selected))
    target_count = len(state & _TARGET_ATOMS)
    final_action = 0 if state == set(range(8)) or (len(state) == 3 and target_count == 3) else 1
    # Target is already the post-safety score winner in every state.  Only the
    # utility stage changes the final action, reproducing the zero-gradient v42
    # residual slice (raw target rank=1 and raw score deficit=0).
    scores = np.asarray([1.0, 0.8], dtype=np.float32)
    margins = np.zeros((2, 2), dtype=np.float32)
    margins[1, 0] = 0.06 - 0.04 * target_count
    diag = {
        "utility_refinement_action_before": 0,
        "utility_refinement_action_after": final_action,
        "utility_score_slack": 0.4,
        "utility_pair_certificate_enabled": True,
        "utility_pair_margin_tolerance": 0.04,
    }
    if with_diagnostics:
        return final_action, scores, margins, diag
    return final_action, scores, margins


def _run(*, with_diagnostics: bool):
    proposal = np.full((8,), 10.0, dtype=np.float32)
    proposal[list(_TARGET_ATOMS)] = -10.0
    return runtime_greedy_selector_pair_conditioned(
        predicted_base_cost=np.zeros((2,), dtype=np.float32),
        pair_atom_delta=np.zeros((8, 1), dtype=np.float32),
        pair_indices=np.asarray([[0, 1]], dtype=np.int64),
        pair_weights=np.ones((1,), dtype=np.float32),
        atom_budget_costs=np.ones((8,), dtype=np.float32),
        valid_mask=np.ones((2,), dtype=bool),
        runtime_safety_flags=np.zeros((2,), dtype=bool),
        budget=3.0,
        atom_active_mask=np.ones((8,), dtype=bool),
        proposal_scores=proposal,
        selector_cap_mode="stage_aware_budget_layer_coreset",
        deployment_evaluator=lambda selected: _utility_stage_evaluator(
            selected, with_diagnostics=with_diagnostics
        ),
        deployment_coreset_exact_candidates=1,
        deployment_coreset_swap_passes=0,
        deployment_coreset_lexicographic_action_preservation=True,
        deployment_coreset_preservation_scan_candidates=0,
        deployment_coreset_repair_one_swap=False,
        deployment_coreset_repair_two_swap_candidates=0,
        deployment_coreset_beam_width=0,
        deployment_coreset_budget_layer_width=4,
        deployment_coreset_budget_layer_branch=8,
        deployment_coreset_budget_layer_iterations=4,
        deployment_coreset_budget_layer_max_evaluations=200,
        deployment_coreset_budget_layer_exhaustive_first=True,
        deployment_coreset_budget_layer_seed_count=0,
        deployment_coreset_budget_layer_diversity_distance=2,
        force_fill_budget=True,
        min_selected_atoms=3,
    )


def test_stage_diagnostics_recover_utility_override_zero_gradient() -> None:
    old_information = _run(with_diagnostics=False)
    assert old_information.diagnostics["deployment_coreset_target_action_preserved"] == 0.0

    repaired = _run(with_diagnostics=True)
    assert set(repaired.selected) == _TARGET_ATOMS
    assert repaired.diagnostics["deployment_coreset_target_action_preserved"] == 1.0
    assert repaired.diagnostics["deployment_coreset_budget_layer_success"]
    assert repaired.diagnostics["deployment_coreset_budget_layer_best_stage"] == 0
    assert repaired.diagnostics["deployment_coreset_budget_layer_iteration_limit"] == 4


def test_v43_configs_are_stage_aware_and_distinct() -> None:
    root = Path(__file__).resolve().parents[2]
    main = root / "bdse/configs/v43_bdse_sabdacc_fast_cl.yaml"
    fallback = root / "bdse/configs/v43_bdse_sabdacc_fallback_fast_cl.yaml"
    control = root / "bdse/configs/v43_bdse_mars_control_fast_cl.yaml"
    main_cfg = load_config(str(main))
    fallback_cfg = load_config(str(fallback))
    control_cfg = load_config(str(control))
    assert main_cfg["selector"]["selector_cap_mode"] == "stage_aware_budget_layer_coreset"
    assert bool(main_cfg["selector"]["deployment_coreset_use_deployment_pair_graph"])
    assert int(main_cfg["selector"]["deployment_coreset_beam_width"]) == 0
    assert control_cfg["selector"]["selector_cap_mode"] == "margin_coreset"
    assert not bool(main_cfg["fallback"]["enabled"])
    assert bool(fallback_cfg["fallback"]["enabled"])
    assert len({main.read_bytes(), fallback.read_bytes(), control.read_bytes()}) == 3
