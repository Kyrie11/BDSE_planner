from __future__ import annotations

from pathlib import Path

import numpy as np

from bdse.config import load_config
from bdse.planner.selector import runtime_greedy_selector_pair_conditioned


_TARGET_ATOMS = {0, 1, 2}


def _three_swap_evaluator(selected: list[int]) -> tuple[int, np.ndarray, np.ndarray]:
    """Only a three-exchange B=3 subset restores the Top-M action.

    Greedy deletion is deliberately attracted to {5, 6, 7}.  No one-swap or
    two-swap repair can reach {0, 1, 2}, while the exact target-score deficit
    improves after each of three fixed-budget exchanges.
    """
    state = set(map(int, selected))
    target_count = len(state & _TARGET_ATOMS)
    if state == set(range(8)) or (len(state) == 3 and target_count == 3):
        target_score = 1.0
        action = 0
    elif len(state) > 3:
        # Removing a target atom moves the score closer to the full-set score,
        # so the lexicographic deletion path discards all three target atoms.
        target_score = 0.995 - 0.01 * target_count
        action = 0
    else:
        target_score = 0.10 + 0.19 * target_count
        action = 0 if target_count == 3 else 1
    scores = np.asarray([target_score, 1.0 - target_score], dtype=np.float32)
    margin = float(scores[0] - scores[1])
    margins = np.asarray([[0.0, margin], [-margin, 0.0]], dtype=np.float32)
    return action, scores, margins


def _run_three_swap(*, enable_budget_layer: bool):
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
        selector_cap_mode="counterfactual_budget_layer_coreset",
        deployment_evaluator=_three_swap_evaluator,
        deployment_coreset_exact_candidates=1,
        deployment_coreset_swap_passes=0,
        deployment_coreset_lexicographic_action_preservation=True,
        deployment_coreset_preservation_scan_candidates=0,
        deployment_coreset_repair_one_swap=False,
        deployment_coreset_repair_two_swap_candidates=0,
        deployment_coreset_beam_width=0,
        deployment_coreset_beam_branch=0,
        deployment_coreset_budget_layer_width=12 if enable_budget_layer else 0,
        deployment_coreset_budget_layer_branch=18 if enable_budget_layer else 0,
        deployment_coreset_budget_layer_iterations=6 if enable_budget_layer else 0,
        deployment_coreset_budget_layer_max_evaluations=300 if enable_budget_layer else 0,
        deployment_coreset_budget_layer_exhaustive_first=True,
        deployment_coreset_budget_layer_seed_count=0,
        deployment_coreset_budget_layer_diversity_distance=2,
        force_fill_budget=True,
        min_selected_atoms=3,
    )


def test_cbldacc_recovers_beyond_two_swap_neighborhood() -> None:
    trapped = _run_three_swap(enable_budget_layer=False)
    assert set(trapped.selected) == {5, 6, 7}
    assert trapped.diagnostics["deployment_coreset_target_action_preserved"] == 0.0

    repaired = _run_three_swap(enable_budget_layer=True)
    assert set(repaired.selected) == _TARGET_ATOMS
    assert repaired.diagnostics["deployment_coreset_target_action_preserved"] == 1.0
    assert repaired.diagnostics["deployment_coreset_budget_layer_attempted"]
    assert repaired.diagnostics["deployment_coreset_budget_layer_success"]
    assert repaired.diagnostics["deployment_coreset_budget_layer_iterations"] >= 3
    assert repaired.diagnostics["deployment_coreset_budget_layer_evaluations"] > 0
    assert repaired.diagnostics["deployment_coreset_budget_layer_best_target_rank"] == 1


def test_v42_configs_use_fixed_budget_search_and_disable_v41_beam() -> None:
    root = Path(__file__).resolve().parents[2]
    main = root / "bdse/configs/v42_bdse_cbldacc_fast_cl.yaml"
    fallback = root / "bdse/configs/v42_bdse_cbldacc_fallback_fast_cl.yaml"
    control = root / "bdse/configs/v42_bdse_mars_control_fast_cl.yaml"
    main_cfg = load_config(str(main))
    fallback_cfg = load_config(str(fallback))
    control_cfg = load_config(str(control))

    selector = main_cfg["selector"]
    assert selector["selector_cap_mode"] == "counterfactual_budget_layer_coreset"
    assert bool(selector["deployment_coreset_use_deployment_pair_graph"])
    assert int(selector["deployment_coreset_beam_width"]) == 0
    assert int(selector["deployment_coreset_beam_branch"]) == 0
    assert int(selector["deployment_coreset_budget_layer_width"]) > 1
    assert int(selector["deployment_coreset_budget_layer_branch"]) > 1
    assert int(selector["deployment_coreset_budget_layer_iterations"]) >= 3
    assert int(selector["deployment_coreset_budget_layer_max_evaluations"]) > 0
    assert bool(selector["deployment_coreset_budget_layer_exhaustive_first"])
    assert control_cfg["selector"]["selector_cap_mode"] == "margin_coreset"
    assert not bool(main_cfg["fallback"]["enabled"])
    assert bool(fallback_cfg["fallback"]["enabled"])
    assert main.read_bytes() != fallback.read_bytes()
