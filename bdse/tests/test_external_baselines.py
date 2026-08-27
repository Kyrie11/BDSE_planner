from __future__ import annotations

import torch

from bdse.config import load_config
from bdse.data.tensorizer import sample_to_model_inputs
from bdse.external_baselines.losses import compute_external_baseline_losses
from bdse.external_baselines.models import ExternalBaselineModel
from bdse.planner.nuplan_planner import BDSEPlannerCore


def _cfg_for_variant(variant: str):
    return load_config(overrides={
        "runtime": {"max_agents": 4, "use_pair_conditioned_margins": False},
        "evidence": {"max_atoms": 32, "max_interaction_atoms": 8, "budget": 4},
        "selector": {"proposal_top_m": 8},
        "pairs": {"target_min": 4, "target_max": 32},
        "candidate": {"K": 32},
        "planner": {"baseline_mode": "external_policy"},
        "external_baseline": {"enabled": True, "variant": variant, "budget": 4, "hidden_dim": 64, "transformer_layers": 1, "attention_heads": 4},
        "model": {"hidden_dim": 64, "transformer_layers": 1, "attention_heads": 4},
    })


def test_external_trainable_forward_and_loss(synthetic_sample):
    cfg = _cfg_for_variant("plantf")
    model = ExternalBaselineModel(cfg)
    batch = sample_to_model_inputs(synthetic_sample, cfg, include_teacher=True, include_dense_query=False)
    batch = {k: v.unsqueeze(0) for k, v in batch.items()}
    out = model(batch)
    assert out["J0"].shape == (1, 32)
    losses = compute_external_baseline_losses(out, batch, cfg)
    assert torch.isfinite(losses["loss"])


def test_external_policy_runtime_core(synthetic_sample):
    cfg = _cfg_for_variant("pdm_closed")
    model = ExternalBaselineModel(cfg)
    core = BDSEPlannerCore(model=model, cfg=cfg)
    pred, sel, tour, _ = core._run_certificate_stage(synthetic_sample.runtime, synthetic_sample.candidates, synthetic_sample.evidence_bank, cfg)
    assert pred["baseline_mode"] == "external_policy"
    assert pred["external_variant"] == "pdm_closed"
    assert len(sel.selected) <= int(cfg["evidence"]["budget"])
    assert 0 <= int(tour.action_index) < synthetic_sample.candidates.K


def test_external_unit_cost_budget_selection_is_exact_and_vectorized():
    cfg = _cfg_for_variant("gameformer")
    cfg["evidence"]["unit_cost"] = True
    model = ExternalBaselineModel(cfg)
    logits = torch.tensor([[1.0, 5.0, 3.0, -1.0, 4.0], [2.0, 1.0, 8.0, 7.0, 6.0]])
    active = torch.tensor([[True, True, False, True, True], [True, False, True, True, False]])
    costs = torch.ones_like(logits)
    model.budget = 2
    mask, indices, slot_valid = model._top_budget_selection(logits, active, costs)
    assert mask.sum(dim=1).tolist() == [2, 2]
    assert set(torch.nonzero(mask[0], as_tuple=False).reshape(-1).tolist()) == {1, 4}
    assert set(torch.nonzero(mask[1], as_tuple=False).reshape(-1).tolist()) == {2, 3}
    assert indices.shape == slot_valid.shape == (2, 2)


def test_all_primary_external_adapters_forward(synthetic_sample):
    for variant in ("gameformer", "dtpp", "plantf", "pluto"):
        cfg = _cfg_for_variant(variant)
        model = ExternalBaselineModel(cfg)
        batch = sample_to_model_inputs(synthetic_sample, cfg, include_teacher=True, include_dense_query=False)
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        out = model(batch)
        assert out["J0"].shape == (1, 32)
        assert out["external_selected_mask"].sum().item() <= 4
        assert "inspired" in str(out["external_implementation_label"]).lower()


def test_external_checkpoint_strict_variant_and_shapes(tmp_path):
    from bdse.external_baselines.model_factory import load_model_for_config

    cfg = _cfg_for_variant("plantf")
    model = ExternalBaselineModel(cfg)
    good = tmp_path / "plantf_budgeted.best.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, good)
    loaded = load_model_for_config(str(good), cfg, torch.device("cpu"))
    assert isinstance(loaded, ExternalBaselineModel)

    bad_cfg = _cfg_for_variant("gameformer")
    bad = tmp_path / "wrong.best.pt"
    torch.save({"model": model.state_dict(), "cfg": bad_cfg}, bad)
    try:
        load_model_for_config(str(bad), cfg, torch.device("cpu"))
    except ValueError as exc:
        assert "variant mismatch" in str(exc)
    else:
        raise AssertionError("strict external checkpoint load should reject wrong variants")


def test_external_runtime_budget_override_changes_selected_count(synthetic_sample):
    cfg = _cfg_for_variant("plantf")
    model = ExternalBaselineModel(cfg)
    batch = sample_to_model_inputs(synthetic_sample, cfg, include_teacher=True, include_dense_query=False)
    batch = {k: v.unsqueeze(0) for k, v in batch.items()}
    out2 = model(batch, budget_override=2)
    out6 = model(batch, budget_override=6)
    assert int(out2["external_selected_mask"].sum()) <= 2
    assert int(out6["external_selected_mask"].sum()) <= 6
    assert int(out6["external_selected_mask"].sum()) >= int(out2["external_selected_mask"].sum())


def test_external_compact_tensorizer_matches_generic_forward(synthetic_sample):
    from bdse.external_baselines.data import external_sample_to_model_inputs

    for variant in ("gameformer", "dtpp", "plantf", "pluto"):
        torch.manual_seed(7)
        cfg = _cfg_for_variant(variant)
        model = ExternalBaselineModel(cfg).eval()
        full = sample_to_model_inputs(synthetic_sample, cfg, include_teacher=True, include_dense_query=False)
        compact = external_sample_to_model_inputs(synthetic_sample, cfg)
        assert torch.equal(full["oracle_selected_mask"], compact["oracle_selected_mask"]), variant
        full_b = {k: v.unsqueeze(0) for k, v in full.items()}
        compact_b = {k: v.unsqueeze(0) for k, v in compact.items()}
        with torch.inference_mode():
            out_full = model(full_b)["J0"]
            out_compact = model(compact_b)["J0"]
        assert torch.allclose(out_full, out_compact, atol=2e-5, rtol=2e-5), variant


def test_external_minimal_npz_loader_preserves_external_contract(tmp_path, synthetic_sample):
    from bdse.data.cache_schema import save_sample_npz
    from bdse.external_baselines.data import (
        external_sample_to_model_inputs,
        load_external_training_sample_npz,
    )

    cfg = _cfg_for_variant("plantf")
    cfg["external_baseline"]["planner_supervision"] = "expert_imitation"
    path = tmp_path / "sample.npz"
    save_sample_npz(synthetic_sample, path)
    lean = load_external_training_sample_npz(path, include_label_future=True)
    tensors = external_sample_to_model_inputs(lean, cfg)
    assert lean.scenario_token == synthetic_sample.scenario_token
    assert torch.equal(tensors["candidate_valid"], torch.from_numpy(synthetic_sample.candidates.valid_mask))
    assert int(tensors["expert_candidate_index"]) >= 0
    assert tensors["expert_candidate_cost"].shape[0] == synthetic_sample.candidates.K


def test_external_train_main_does_not_shadow_module_torch():
    import ast
    import inspect
    from bdse.external_baselines import train as train_module

    tree = ast.parse(inspect.getsource(train_module.main))
    shadowing_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    shadowing_imports.append(alias.name)
    assert shadowing_imports == []


def test_external_vectorized_oracle_matches_reference_randomized():
    import numpy as np
    from bdse.external_baselines.data import _greedy_cover_from_pair_delta_vectorized
    from bdse.planner.selector import _greedy_cover_from_pair_delta

    rng = np.random.default_rng(2026)
    for _ in range(50):
        E, P = 32, 64
        atom_delta = rng.normal(0.0, 0.25, size=(E, P)).astype(np.float32)
        base = rng.normal(0.0, 0.5, size=(P,)).astype(np.float32)
        caps = rng.uniform(0.0, 1.5, size=(P,)).astype(np.float32)
        weights = rng.uniform(0.1, 2.0, size=(P,)).astype(np.float32)
        costs = rng.choice(np.array([1.0, 2.0, 3.0], dtype=np.float32), size=(E,)).astype(np.float32)
        active = rng.random(E) > 0.15
        budget = float(rng.choice([4, 8, 16]))
        ref = _greedy_cover_from_pair_delta(atom_delta, base, caps, weights, costs, budget, active)
        fast = _greedy_cover_from_pair_delta_vectorized(atom_delta, base, caps, weights, costs, budget, active)
        assert ref[0] == fast[0]
        assert abs(ref[1] - fast[1]) < 1e-5
        assert abs(ref[2] - fast[2]) < 1e-6


def test_external_fast_refinement_forward(synthetic_sample):
    for variant in ("gameformer", "dtpp"):
        cfg = _cfg_for_variant(variant)
        cfg["external_baseline"]["refinement_mode"] = "shared_encoder_light_refine"
        cfg["external_baseline"]["refinement_layers_per_stage"] = 1
        cfg["external_baseline"]["refinement_ff_dim"] = 128
        model = ExternalBaselineModel(cfg)
        assert len(model.stage_refiners) == (model.reasoning_levels if variant == "gameformer" else model.tree_depth)
        batch = sample_to_model_inputs(synthetic_sample, cfg, include_teacher=True, include_dense_query=False)
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        out = model(batch)
        losses = compute_external_baseline_losses(out, batch, cfg)
        assert out["J0"].shape == (1, 32)
        assert torch.isfinite(losses["loss"])
