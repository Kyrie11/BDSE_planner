from __future__ import annotations

from pathlib import Path

import numpy as np

from bdse.config import load_config
from bdse.planner.selector import runtime_greedy_selector_pair_conditioned


def test_dacc_uses_exact_deployment_evaluator_and_preserves_target() -> None:
    base = np.asarray([0.0, 0.2, 0.4], dtype=np.float32)
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    delta = np.asarray(
        [[0.5, 0.4, 0.0], [0.3, -0.2, 0.2], [-0.1, 0.5, 0.3], [0.2, 0.1, 0.1]],
        dtype=np.float32,
    )

    def evaluate(selected: list[int]) -> tuple[int, np.ndarray, np.ndarray]:
        support = delta[np.asarray(selected, dtype=np.int64)].sum(axis=0) if selected else np.zeros((3,), dtype=np.float32)
        directed = (base[pairs[:, 1]] - base[pairs[:, 0]]) + support
        margins = base[None, :] - base[:, None]
        for value, (a, b) in zip(directed.tolist(), pairs.tolist()):
            margins[a, b] = value
            margins[b, a] = -value
        scores = np.asarray([min(margins[a, b] for b in range(3) if b != a) for a in range(3)], dtype=np.float32)
        return int(np.argmax(scores)), scores, margins

    result = runtime_greedy_selector_pair_conditioned(
        base,
        delta,
        pairs,
        np.ones((3,), dtype=np.float32),
        np.ones((4,), dtype=np.float32),
        np.ones((3,), dtype=bool),
        np.zeros((3,), dtype=bool),
        budget=2.0,
        atom_active_mask=np.ones((4,), dtype=bool),
        selector_cap_mode="deployment_coreset",
        deployment_evaluator=evaluate,
        deployment_coreset_exact_candidates=4,
        force_fill_budget=True,
        min_selected_atoms=2,
    )
    assert result.diagnostics["mode"] == "runtime_pair_conditioned_deployment_coreset"
    assert result.diagnostics["deployment_coreset_active"]
    assert result.diagnostics["deployment_coreset_target_action_preserved"] == 1.0
    assert len(result.selected) == 2


def test_v39_configs_are_distinct_and_loadable() -> None:
    root = Path(__file__).resolve().parents[2]
    main = root / "bdse/configs/v39_bdse_dacc_fast_cl.yaml"
    fallback = root / "bdse/configs/v39_bdse_dacc_fallback_fast_cl.yaml"
    control = root / "bdse/configs/v39_bdse_mars_control_fast_cl.yaml"
    main_cfg = load_config(str(main))
    fallback_cfg = load_config(str(fallback))
    control_cfg = load_config(str(control))
    assert main_cfg["selector"]["selector_cap_mode"] == "deployment_coreset"
    assert control_cfg["selector"]["selector_cap_mode"] == "margin_coreset"
    assert not bool(main_cfg["fallback"]["enabled"])
    assert bool(fallback_cfg["fallback"]["enabled"])
    assert main.read_bytes() != fallback.read_bytes()
