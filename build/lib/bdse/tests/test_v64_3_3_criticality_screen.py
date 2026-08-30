from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from bdse.config import load_config
from bdse.experiments.train import _teacher_literal_criticality_full_support
from bdse.model.bdse_model import BDSEModel
from bdse.tools.check_v64_3_3_acquisition_screen import build_report


def test_training_forward_exports_critical_adapter_residual() -> None:
    cfg = load_config("bdse/configs/v64_3_3_cc_aocc_apwcca_daepc_screen_2gpu.yaml")
    model = BDSEModel(cfg)
    h = int(model.hidden_dim)
    residual = torch.tensor([[0.0, 0.25, -0.10]])
    ctx = {
        "J0": torch.tensor([[0.0, 1.0]]),
        "action_h": torch.zeros((1, 2, h)),
        "evidence_h": torch.zeros((1, 3, h)),
        "scene": torch.zeros((1, h)),
        "proposal_logits": torch.zeros((1, 3)),
        "critical_proposal_residual_logits": residual,
        "family_logits": torch.zeros((1, int(model.num_families))),
        "family_pi": torch.full((1, int(model.num_families)), 1.0 / float(model.num_families)),
        "family_active": torch.ones((1, int(model.num_families)), dtype=torch.bool),
    }
    model.encode_context = lambda batch: ctx  # type: ignore[method-assign]
    model.set_residual_factors = lambda context: (None, None)  # type: ignore[method-assign]
    out = model({})
    assert "critical_proposal_residual_logits" in out
    assert torch.equal(out["critical_proposal_residual_logits"], residual)


def test_screen_teacher_criticality_uses_full_support_not_topm_support() -> None:
    # Atom 1 is literally critical but is not in Top-M.  A Top-M-as-active bug
    # would erase atom 1 from the label universe and make recall tautologically 1.
    sample = SimpleNamespace(
        evidence_bank=SimpleNamespace(active_mask=np.array([True, True])),
        candidates=SimpleNamespace(valid_mask=np.array([True, True])),
        teacher=SimpleNamespace(
            g_evid=np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            J_T=np.array([0.5, 1.0], dtype=np.float32),
            a_star=0,
        ),
    )
    pred = {
        "g": np.zeros((2, 2), dtype=np.float32),
        "top_m_atoms": np.array([0], dtype=np.int64),
    }
    values, details = _teacher_literal_criticality_full_support(sample, pred, [0])
    assert details["teacher_exact_winner_flip_critical_count"] == 1
    assert values["teacher_exact_winner_flip_critical_recall_topm"] == 0.0
    assert values["teacher_exact_winner_flip_critical_recall_selected"] == 0.0


def test_screen_requires_wired_acra_and_full_support_improvement() -> None:
    rows = [
        {
            "epoch": -1,
            "val_teacher_exact_winner_flip_critical_recall_topm": 0.35,
            "val_teacher_exact_winner_flip_critical_recall_topm_micro": 0.34,
            "val_teacher_exact_winner_flip_critical_recall_selected_micro": 0.32,
            "val_teacher_exact_winner_flip_critical_scene_rate": 0.45,
            "val_teacher_exact_winner_flip_critical_count": 1.2,
            "val_proposal_decisive_atom_recall": 0.80,
            "critical_adapter_parameter_delta_rms": 0.0,
        },
        {
            "epoch": 0,
            "val_teacher_exact_winner_flip_critical_recall_topm": 0.37,
            "val_teacher_exact_winner_flip_critical_recall_topm_micro": 0.36,
            "val_teacher_exact_winner_flip_critical_recall_selected_micro": 0.33,
            "val_teacher_exact_winner_flip_critical_scene_rate": 0.45,
            "val_teacher_exact_winner_flip_critical_count": 1.2,
            "val_proposal_decisive_atom_recall": 0.79,
            "val_teacher_action_match": 0.26,
            "critical_adapter_parameter_delta_rms": 1e-3,
            "critical_adapter_parameter_delta_max_abs": 2e-3,
            "critical_proposal_residual_rms": 0.05,
            "critical_proposal_residual_abs_mean": 0.04,
            "L_critical_adapter_residual_alignment": 0.1,
        },
    ]
    report = build_report(rows, "unit")
    assert report["screen_instrumentation_valid"] is True
    assert report["adapter_parameter_activated"] is True
    assert report["adapter_forward_activated"] is True
    assert report["acra_wired"] is True
    assert report["delta_val_critical_topm_recall_micro"] > 0
    assert report["continue_to_full_run"] is True
