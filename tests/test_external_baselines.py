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
