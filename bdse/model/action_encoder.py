from __future__ import annotations

import torch
from torch import nn


class ActionEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, max_maneuver_id: int = 16):
        super().__init__()
        self.traj_proj = nn.Sequential(nn.Linear(5, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        # terminal pose, speed mean/max, displacement, terminal time, valid,
        # route-relative proxy progress, terminal lateral offset, max lateral, heading change.
        self.summary_proj = nn.Sequential(nn.Linear(12, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.maneuver_emb = nn.Embedding(max_maneuver_id + 2, hidden_dim)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, trajectories: torch.Tensor, maneuver_ids: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        traj = trajectories.float()
        B, K, T, D = traj.shape
        valid_f = valid_mask.float().unsqueeze(-1)
        traj_tokens = self.traj_proj(traj).mean(dim=2)
        xy0 = traj[:, :, 0, :2]
        xyT = traj[:, :, -1, :2]
        v = traj[:, :, :, 3]
        speed_mean = v.mean(dim=2, keepdim=True)
        speed_max = v.max(dim=2, keepdim=True).values
        disp = torch.linalg.norm(xyT - xy0, dim=-1, keepdim=True)
        terminal = traj[:, :, -1, :3]
        route_progress = xyT[..., 0:1]
        terminal_lateral = xyT[..., 1:2]
        max_abs_lateral = traj[:, :, :, 1].abs().max(dim=2, keepdim=True).values
        heading_change = torch.atan2(torch.sin(traj[:, :, -1, 2] - traj[:, :, 0, 2]), torch.cos(traj[:, :, -1, 2] - traj[:, :, 0, 2])).unsqueeze(-1)
        summary = torch.cat([
            terminal, speed_mean, speed_max, disp, traj[:, :, -1, 4:5], valid_f,
            route_progress, terminal_lateral, max_abs_lateral, heading_change,
        ], dim=-1)
        summary_tokens = self.summary_proj(summary)
        mids = torch.clamp(maneuver_ids.long() + 1, 0, self.maneuver_emb.num_embeddings - 1)
        emb = self.maneuver_emb(mids)
        out = self.out(torch.cat([traj_tokens, summary_tokens, emb], dim=-1))
        return out * valid_f
