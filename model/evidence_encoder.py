from __future__ import annotations

import torch
from torch import nn


class EvidenceEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, evidence_feature_dim: int = 24, num_types: int = 32, num_families: int = 8):
        super().__init__()
        self.type_emb = nn.Embedding(num_types, hidden_dim)
        self.family_emb = nn.Embedding(num_families, hidden_dim)
        self.feature_proj = nn.Sequential(nn.Linear(evidence_feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, features: torch.Tensor, type_ids: torch.Tensor, family_ids: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        f = self.feature_proj(features.float())
        t = self.type_emb(torch.clamp(type_ids.long(), 0, self.type_emb.num_embeddings - 1))
        fam = self.family_emb(torch.clamp(family_ids.long(), 0, self.family_emb.num_embeddings - 1))
        out = self.out(torch.cat([f, t, fam], dim=-1))
        return out * active_mask.float().unsqueeze(-1)
