from __future__ import annotations

import torch

from bdse.model.bdse_model import BDSEModel


def test_candidate_set_summary_is_amp_safe():
    cfg = {"model": {"hidden_dim": 16, "action_set_summary_dim": 16}, "candidate": {"num_maneuvers": 7}, "selector": {"eta_pred": 1.0}}
    model = BDSEModel(cfg)
    J0 = torch.tensor([[0.0, 1.0, float("inf"), 2.0]], dtype=torch.float16)
    traj = torch.zeros((1, 4, 3, 5), dtype=torch.float16)
    traj[0, :, -1, 0] = torch.tensor([5.0, 4.0, 3.0, 2.0], dtype=torch.float16)
    maneuver = torch.tensor([[0, 1, 2, 6]], dtype=torch.long)
    valid = torch.tensor([[True, True, False, True]])
    summary = model._candidate_set_summary(J0, traj, maneuver, valid)
    assert summary.dtype == torch.float32
    assert torch.isfinite(summary).all()
    assert summary.shape == (1, 16)
