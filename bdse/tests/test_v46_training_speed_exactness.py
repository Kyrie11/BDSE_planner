from __future__ import annotations

import copy

import numpy as np
import torch

from bdse.experiments.train import _scalar_loss_finite_flag
from bdse.model.losses import (
    _config_with_evidence_budget,
    _pair_cycle_consistency_loss,
    _predicted_pair_certificate_masks,
    _predicted_pair_certificate_masks_multi_budget,
)
from bdse.planner.selector import runtime_greedy_selector_pair_conditioned


def _selector_case() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
    batch_size, atoms, actions, pair_count = 2, 8, 4, 4
    outputs = {
        "J0": torch.tensor(
            [[0.0, 0.2, 0.3, 0.4], [0.1, 0.0, 0.25, 0.35]], dtype=torch.float32
        ),
        "proposal_logits": torch.tensor(
            [
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ],
            dtype=torch.float32,
        ),
        "pair_atom_delta": torch.tensor(
            np.linspace(-0.25, 0.45, batch_size * atoms * pair_count, dtype=np.float32).reshape(
                batch_size, atoms, pair_count
            )
        ),
        "pair_atom_var": torch.full((batch_size, atoms, pair_count), 0.04, dtype=torch.float32),
    }
    batch = {
        "pair_indices": torch.tensor(
            [
                [[0, 1], [0, 2], [0, 3], [1, 2]],
                [[0, 1], [0, 2], [0, 3], [1, 3]],
            ],
            dtype=torch.long,
        ),
        "pair_valid": torch.ones((batch_size, pair_count), dtype=torch.bool),
        "pair_weights": torch.ones((batch_size, pair_count), dtype=torch.float32),
        "candidate_valid": torch.ones((batch_size, actions), dtype=torch.bool),
        "evidence_active": torch.ones((batch_size, atoms), dtype=torch.bool),
        "evidence_budget_costs": torch.ones((batch_size, atoms), dtype=torch.float32),
        "evidence_family_ids": torch.tensor(
            [[2, 2, 3, 3, 4, 4, 5, 5], [2, 2, 3, 3, 4, 4, 5, 5]], dtype=torch.long
        ),
        "evidence_agent_group_ids": torch.full((batch_size, atoms), -1, dtype=torch.long),
        "runtime_safety_flags": torch.zeros((batch_size, actions), dtype=torch.bool),
        "evidence_features": torch.zeros((batch_size, atoms, 2), dtype=torch.float32),
    }
    cfg = {
        "evidence": {"budget": 2},
        "model": {"pair_margin_normalized": True},
        "runtime": {"pair_delta_hybrid_local_weight": 0.0},
        "calibration": {"epsilon_cal": 0.05},
        "tournament": {"beta_uncertainty": 0.5, "epsilon_cal": 0.05},
        "selector": {
            "proposal_top_m": atoms,
            "hab_enabled": True,
            "selector_cap_mode": "anytime_adverse_certificate",
            "decision_budget_excludes_structural_safety": True,
            "collapse_reciprocal_pairs": True,
            "force_fill_budget": False,
            "min_selected_atoms": 0,
            "soft_interaction_quota": 0,
            "adverse_certificate_beta": 1.0,
            "adverse_certificate_epsilon": 0.05,
            "adverse_certificate_prior_radius": 0.10,
            "adverse_certificate_stop_when_certified": False,
        },
    }
    return outputs, batch, cfg


def test_multi_budget_exact_masks_equal_repeated_calls() -> None:
    outputs, batch, cfg = _selector_case()
    cfgs = [_config_with_evidence_budget(cfg, budget) for budget in (1.0, 2.0, 4.0)]
    repeated = [_predicted_pair_certificate_masks(outputs, batch, local_cfg) for local_cfg in cfgs]
    combined = _predicted_pair_certificate_masks_multi_budget(outputs, batch, cfgs)
    assert len(combined) == len(repeated)
    for expected, actual in zip(repeated, combined):
        assert torch.equal(expected, actual)


def test_aocc_state_cache_reuses_order_without_changing_result() -> None:
    rng = np.random.default_rng(17)
    actions, atoms, pair_count = 6, 32, 20
    pairs = np.stack(
        [rng.integers(0, actions, pair_count), rng.integers(0, actions, pair_count)], axis=1
    ).astype(np.int64)
    pairs[:, 1] = (pairs[:, 1] + (pairs[:, 1] == pairs[:, 0])) % actions
    kwargs = dict(
        predicted_base_cost=rng.normal(size=actions).astype(np.float32),
        pair_atom_delta=rng.normal(scale=0.2, size=(atoms, pair_count)).astype(np.float32),
        pair_indices=pairs,
        pair_weights=rng.uniform(0.2, 2.0, pair_count).astype(np.float32),
        atom_budget_costs=np.ones((atoms,), dtype=np.float32),
        valid_mask=np.ones((actions,), dtype=bool),
        runtime_safety_flags=np.zeros((actions,), dtype=bool),
        atom_active_mask=np.ones((atoms,), dtype=bool),
        pair_atom_variance=np.full((atoms, pair_count), 0.03, dtype=np.float32),
        selector_cap_mode="anytime_adverse_certificate",
        adverse_certificate_stop_when_certified=False,
        force_fill_budget=False,
        min_selected_atoms=0,
        soft_interaction_quota=0,
    )
    cache: dict = {}
    _ = runtime_greedy_selector_pair_conditioned(**kwargs, budget=8.0, aocc_state_cache=cache)
    state_id = id(cache["state"])
    cached = runtime_greedy_selector_pair_conditioned(**kwargs, budget=16.0, aocc_state_cache=cache)
    uncached = runtime_greedy_selector_pair_conditioned(**kwargs, budget=16.0)
    assert id(cache["state"]) == state_id
    assert cached.selected == uncached.selected
    assert cached.objective_value == uncached.objective_value
    assert cached.diagnostics["aocc_final_deficit"] == uncached.diagnostics["aocc_final_deficit"]


def _legacy_cycle_loss(
    predicted_margin: torch.Tensor,
    pairs: torch.Tensor,
    pair_mask: torch.Tensor,
    target_action: torch.Tensor,
    valid_actions: torch.Tensor,
    max_triangles: int,
    delta: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    batch_size, pair_count = predicted_margin.shape
    action_count = int(valid_actions.shape[1])
    for bi in range(batch_size):
        edge: dict[tuple[int, int], torch.Tensor] = {}
        for pi in range(pair_count):
            if not bool(pair_mask[bi, pi]):
                continue
            a = int(pairs[bi, pi, 0].item())
            b = int(pairs[bi, pi, 1].item())
            if a == b or not (0 <= a < action_count and 0 <= b < action_count):
                continue
            value = predicted_margin[bi, pi]
            edge[(a, b)] = value
            edge[(b, a)] = -value
        actions = [int(x) for x in torch.nonzero(valid_actions[bi], as_tuple=False).flatten().tolist()]
        target = int(target_action[bi].item())
        candidates = []
        for ia, a in enumerate(actions):
            for ib in range(ia + 1, len(actions)):
                b = actions[ib]
                for ic in range(ib + 1, len(actions)):
                    c = actions[ic]
                    if (a, b) not in edge or (b, c) not in edge or (c, a) not in edge:
                        continue
                    cyc = edge[(a, b)] + edge[(b, c)] + edge[(c, a)]
                    contains_target = 0 if target in (a, b, c) else 1
                    boundary = min(
                        abs(float(edge[(a, b)].detach().item())),
                        abs(float(edge[(b, c)].detach().item())),
                        abs(float(edge[(c, a)].detach().item())),
                    )
                    candidates.append((contains_target, boundary, a, b, c, cyc))
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
        for item in candidates[:max_triangles]:
            cyc = item[-1]
            abs_cyc = cyc.abs()
            losses.append(torch.where(abs_cyc <= delta, 0.5 * cyc.square() / delta, abs_cyc - 0.5 * delta))
    return torch.stack(losses).mean() if losses else predicted_margin.new_tensor(0.0)


def test_cycle_consistency_batched_snapshot_is_exact() -> None:
    pairs = torch.tensor(
        [[[0, 1], [1, 2], [2, 0], [0, 2], [2, 3], [3, 0]]], dtype=torch.long
    )
    mask = torch.ones((1, 6), dtype=torch.bool)
    target = torch.tensor([0], dtype=torch.long)
    valid = torch.ones((1, 4), dtype=torch.bool)
    cfg = {"training": {"cycle_consistency": {"max_triangles_per_scene": 64, "huber_delta": 0.15}}}

    pred_new = torch.tensor([[0.2, -0.1, 0.05, 0.08, -0.03, 0.12]], requires_grad=True)
    pred_old = pred_new.detach().clone().requires_grad_(True)
    actual = _pair_cycle_consistency_loss(pred_new, pairs, mask, target, valid, cfg)
    expected = _legacy_cycle_loss(pred_old, pairs, mask, target, valid, 64, 0.15)
    assert torch.equal(actual.detach(), expected.detach())
    actual.backward()
    expected.backward()
    assert torch.equal(pred_new.grad, pred_old.grad)


def test_scalar_finite_check_aggregates_on_device() -> None:
    names, flag = _scalar_loss_finite_flag(
        {"loss": torch.tensor(1.0), "aux": torch.tensor(float("inf")), "vector": torch.ones(2)}
    )
    assert names == ["loss", "aux"]
    assert flag.ndim == 0
    assert not bool(flag)


def test_aocc_skips_unused_postfill_utility(monkeypatch) -> None:
    import bdse.planner.selector as selector_mod

    rng = np.random.default_rng(23)
    actions, atoms, pair_count = 5, 24, 12
    pairs = np.stack(
        [rng.integers(0, actions, pair_count), rng.integers(0, actions, pair_count)], axis=1
    ).astype(np.int64)
    pairs[:, 1] = (pairs[:, 1] + (pairs[:, 1] == pairs[:, 0])) % actions
    kwargs = dict(
        predicted_base_cost=rng.normal(size=actions).astype(np.float32),
        pair_atom_delta=rng.normal(scale=0.2, size=(atoms, pair_count)).astype(np.float32),
        pair_indices=pairs,
        pair_weights=np.ones((pair_count,), dtype=np.float32),
        atom_budget_costs=np.ones((atoms,), dtype=np.float32),
        budget=8.0,
        valid_mask=np.ones((actions,), dtype=bool),
        runtime_safety_flags=np.zeros((actions,), dtype=bool),
        atom_active_mask=np.ones((atoms,), dtype=bool),
        pair_atom_variance=np.zeros((atoms, pair_count), dtype=np.float32),
        selector_cap_mode="anytime_adverse_certificate",
        adverse_certificate_stop_when_certified=False,
        force_fill_budget=False,
        min_selected_atoms=0,
        soft_interaction_quota=0,
        interaction_family_quota=0,
        decision_family_quota=0,
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("post-fill utility must not run when every post-fill mechanism is disabled")

    monkeypatch.setattr(selector_mod, "_action_rank_atom_utility", _unexpected)
    result = selector_mod.runtime_greedy_selector_pair_conditioned(**kwargs)
    assert result.selected
    assert result.diagnostics["postfill_selected_atoms"] == len(result.selected)


def test_threaded_multi_budget_exact_masks_equal_sequential() -> None:
    outputs, batch, cfg = _selector_case()
    seq_cfgs = [_config_with_evidence_budget(cfg, budget) for budget in (1.0, 2.0, 4.0)]
    for local_cfg in seq_cfgs:
        local_cfg.setdefault("training", {})["deployment_selector_cpu_threads"] = 1
        local_cfg["training"]["deployment_selector_cpu_backend"] = "sequential"
    expected = _predicted_pair_certificate_masks_multi_budget(outputs, batch, seq_cfgs)

    par_cfgs = [copy.deepcopy(local_cfg) for local_cfg in seq_cfgs]
    for local_cfg in par_cfgs:
        local_cfg.setdefault("training", {})["deployment_selector_cpu_threads"] = 2
        local_cfg["training"]["deployment_selector_cpu_backend"] = "thread"
    actual = _predicted_pair_certificate_masks_multi_budget(outputs, batch, par_cfgs)
    for sequential, threaded in zip(expected, actual):
        assert torch.equal(sequential, threaded)
