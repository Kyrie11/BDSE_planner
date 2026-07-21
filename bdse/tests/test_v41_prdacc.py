from __future__ import annotations

from pathlib import Path

import numpy as np

from bdse.config import load_config
from bdse.planner.selector import runtime_greedy_selector_pair_conditioned


def _path_trap_evaluator(selected: list[int]) -> tuple[int, np.ndarray, np.ndarray]:
    """A target action that disappears at size three and recovers at size two.

    The single-path lexicographic greedy prefers {1,2,3} at the first deletion
    because it is the least distorted wrong-action state, but none of its size-2
    children recover action 0.  A second beam branch through {0,1,3} reaches the
    preserving subset {0,1}.  This reproduces the path dependence seen in the
    v40 runtime failures without depending on nuPlan assets.
    """
    key = frozenset(map(int, selected))
    if key == frozenset({0, 1, 2, 3}) or key == frozenset({0, 1}):
        scores = np.asarray([1.0, 0.0], dtype=np.float32)
        action = 0
    elif key == frozenset({1, 2, 3}):
        scores = np.asarray([0.49, 0.51], dtype=np.float32)
        action = 1
    elif key == frozenset({0, 1, 3}):
        scores = np.asarray([0.45, 0.55], dtype=np.float32)
        action = 1
    else:
        scores = np.asarray([0.0, 1.0], dtype=np.float32)
        action = 1
    margin = float(scores[0] - scores[1])
    margins = np.asarray([[0.0, margin], [-margin, 0.0]], dtype=np.float32)
    return action, scores, margins


def _run_path_trap(*, beam_width: int, beam_branch: int):
    return runtime_greedy_selector_pair_conditioned(
        predicted_base_cost=np.zeros((2,), dtype=np.float32),
        pair_atom_delta=np.zeros((4, 1), dtype=np.float32),
        pair_indices=np.asarray([[0, 1]], dtype=np.int64),
        pair_weights=np.ones((1,), dtype=np.float32),
        atom_budget_costs=np.ones((4,), dtype=np.float32),
        valid_mask=np.ones((2,), dtype=bool),
        runtime_safety_flags=np.zeros((2,), dtype=bool),
        budget=2.0,
        atom_active_mask=np.ones((4,), dtype=bool),
        selector_cap_mode="path_relaxed_deployment_coreset",
        deployment_evaluator=_path_trap_evaluator,
        deployment_coreset_exact_candidates=1,
        deployment_coreset_lexicographic_action_preservation=True,
        deployment_coreset_preservation_scan_candidates=0,
        deployment_coreset_repair_one_swap=False,
        deployment_coreset_repair_two_swap_candidates=0,
        deployment_coreset_beam_width=beam_width,
        deployment_coreset_beam_branch=beam_branch,
        deployment_coreset_beam_max_evaluations=100,
        deployment_coreset_beam_mismatch_fraction=0.5,
        force_fill_budget=True,
        min_selected_atoms=2,
    )


def test_prdacc_beam_recovers_after_temporary_action_flip() -> None:
    trapped = _run_path_trap(beam_width=0, beam_branch=0)
    assert trapped.diagnostics["deployment_coreset_target_action_preserved"] == 0.0

    repaired = _run_path_trap(beam_width=2, beam_branch=8)
    assert set(repaired.selected) == {0, 1}
    assert repaired.diagnostics["deployment_coreset_target_action_preserved"] == 1.0
    assert repaired.diagnostics["deployment_coreset_beam_attempted"]
    assert repaired.diagnostics["deployment_coreset_beam_success"]
    assert repaired.diagnostics["deployment_coreset_beam_evaluations"] > 0
    assert repaired.diagnostics["deployment_coreset_beam_terminal_count"] > 0


def test_v41_configs_enable_rival_graph_and_bounded_beam() -> None:
    root = Path(__file__).resolve().parents[2]
    main = root / "bdse/configs/v41_bdse_prdacc_fast_cl.yaml"
    fallback = root / "bdse/configs/v41_bdse_prdacc_fallback_fast_cl.yaml"
    control = root / "bdse/configs/v41_bdse_mars_control_fast_cl.yaml"
    main_cfg = load_config(str(main))
    fallback_cfg = load_config(str(fallback))
    control_cfg = load_config(str(control))

    selector = main_cfg["selector"]
    assert selector["selector_cap_mode"] == "path_relaxed_deployment_coreset"
    assert bool(selector["deployment_coreset_use_deployment_pair_graph"])
    assert int(selector["deployment_coreset_beam_width"]) > 1
    assert int(selector["deployment_coreset_beam_branch"]) > 1
    assert int(selector["deployment_coreset_beam_max_evaluations"]) > 0
    assert 0.0 < float(selector["deployment_coreset_beam_mismatch_fraction"]) < 1.0
    assert control_cfg["selector"]["selector_cap_mode"] == "margin_coreset"
    assert not bool(main_cfg["fallback"]["enabled"])
    assert bool(fallback_cfg["fallback"]["enabled"])
    assert main.read_bytes() != fallback.read_bytes()
