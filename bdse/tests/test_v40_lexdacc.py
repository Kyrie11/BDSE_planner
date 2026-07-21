from __future__ import annotations

from pathlib import Path

import numpy as np

from bdse.config import load_config
import bdse.planner.selector as selector_module
from bdse.planner.selector import runtime_greedy_selector_pair_conditioned


def _binary_deployment_evaluator(selected: list[int]) -> tuple[int, np.ndarray, np.ndarray]:
    """Action 0 is preserved exactly when the critical atom 0 is retained."""
    keep = 0 in set(map(int, selected))
    scores = np.asarray([1.0, 0.0] if keep else [0.0, 1.0], dtype=np.float32)
    margins = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    if not keep:
        margins = -margins
    return int(0 if keep else 1), scores, margins


def _run_lexdacc(**kwargs):
    return runtime_greedy_selector_pair_conditioned(
        predicted_base_cost=np.zeros((2,), dtype=np.float32),
        pair_atom_delta=np.zeros((3, 1), dtype=np.float32),
        pair_indices=np.asarray([[0, 1]], dtype=np.int64),
        pair_weights=np.ones((1,), dtype=np.float32),
        atom_budget_costs=np.ones((3,), dtype=np.float32),
        valid_mask=np.ones((2,), dtype=bool),
        runtime_safety_flags=np.zeros((2,), dtype=bool),
        budget=2.0,
        atom_active_mask=np.ones((3,), dtype=bool),
        selector_cap_mode="lexicographic_deployment_coreset",
        deployment_evaluator=_binary_deployment_evaluator,
        deployment_coreset_exact_candidates=1,
        deployment_coreset_lexicographic_action_preservation=True,
        deployment_coreset_preservation_scan_candidates=0,
        deployment_coreset_repair_one_swap=True,
        force_fill_budget=True,
        min_selected_atoms=2,
        **kwargs,
    )


def test_lexdacc_expands_exact_scan_to_find_preserving_deletion() -> None:
    # With all cheap scores tied, top-1 screens atom 0 first. Removing it flips
    # the exact deployed action. Lex-DACC must expand the exact scan and delete
    # atom 1 or 2 instead of accepting that flip.
    result = _run_lexdacc()
    assert 0 in result.selected
    assert len(result.selected) == 2
    assert result.diagnostics["deployment_coreset_lexicographic_active"]
    assert result.diagnostics["deployment_coreset_target_action_preserved"] == 1.0
    assert result.diagnostics["deployment_coreset_preservation_scan_evaluations"] >= 1
    assert result.diagnostics["deployment_coreset_forced_action_flip_steps"] == 0


def test_lexdacc_reaudits_and_reverts_postfill_action_flip(monkeypatch) -> None:
    # Simulate a legacy quota/post-fill replacement that drops the critical
    # atom after DACC has produced an action-preserving set.
    def destructive_postfill(*args, **kwargs):
        return [1, 2], 2.0, {"postfill_selected_atoms": 2, "postfill_spent_budget": 2.0}

    monkeypatch.setattr(selector_module, "_complete_safety_aware_selection", destructive_postfill)
    result = _run_lexdacc()
    assert 0 in result.selected
    assert result.diagnostics["deployment_coreset_postfill_changed"]
    assert result.diagnostics["deployment_coreset_postfill_reverted"]
    assert result.diagnostics["deployment_coreset_target_action_preserved"] == 1.0
    assert result.diagnostics["deployment_coreset_selected_action"] == 0


def test_v40_configs_are_distinct_loadable_and_strict() -> None:
    root = Path(__file__).resolve().parents[2]
    main = root / "bdse/configs/v40_bdse_lexdacc_fast_cl.yaml"
    fallback = root / "bdse/configs/v40_bdse_lexdacc_fallback_fast_cl.yaml"
    control = root / "bdse/configs/v40_bdse_mars_control_fast_cl.yaml"
    main_cfg = load_config(str(main))
    fallback_cfg = load_config(str(fallback))
    control_cfg = load_config(str(control))

    selector = main_cfg["selector"]
    assert selector["selector_cap_mode"] == "lexicographic_deployment_coreset"
    assert bool(selector["deployment_coreset_lexicographic_action_preservation"])
    assert int(selector["deployment_coreset_preservation_scan_candidates"]) == 0
    assert bool(selector["deployment_coreset_repair_one_swap"])
    assert int(selector["deployment_coreset_repair_two_swap_candidates"]) > 0
    assert control_cfg["selector"]["selector_cap_mode"] == "margin_coreset"
    assert not bool(main_cfg["fallback"]["enabled"])
    assert bool(fallback_cfg["fallback"]["enabled"])
    assert main.read_bytes() != fallback.read_bytes()
