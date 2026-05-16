from __future__ import annotations

import torch
from torch import nn


class SceneEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, layers: int = 4, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ego_proj = nn.Linear(5, hidden_dim)
        self.agent_proj = nn.Linear(10, hidden_dim)
        self.map_proj = nn.Linear(4, hidden_dim)
        self.route_proj = nn.Linear(4, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        B = batch["ego_history"].shape[0]
        tokens = [self.cls.expand(B, -1, -1)]
        ego = batch["ego_history"].float()
        tokens.append(self.ego_proj(ego))
        if "agent_history" in batch:
            agent = batch["agent_history"].float()
            B, N, H, D = agent.shape
            agent_tokens = self.agent_proj(agent.reshape(B, N * H, D))
            if "agent_valid" in batch:
                valid = batch["agent_valid"].bool().unsqueeze(-1).expand(B, N, H).reshape(B, N * H).unsqueeze(-1)
                agent_tokens = agent_tokens * valid.float()
            tokens.append(agent_tokens)
        if "map_polylines" in batch:
            mp = batch["map_polylines"].float()
            B, M, P, D = mp.shape
            d = min(D, 4)
            map_tokens = self.map_proj(torch.nn.functional.pad(mp[..., :d], (0, 4 - d)).mean(dim=2))
            tokens.append(map_tokens)
        x = torch.cat(tokens, dim=1)
        h = self.encoder(x)
        return self.norm(h[:, 0])
