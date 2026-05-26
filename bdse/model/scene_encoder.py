from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SceneEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        layers: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        map_feature_dim: int = 8,
        route_feature_dim: int = 8,
        traffic_feature_dim: int = 12,
        goal_feature_dim: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ego_proj = nn.Linear(5, hidden_dim)
        self.agent_proj = nn.Linear(10, hidden_dim)
        self.map_proj = nn.Linear(map_feature_dim, hidden_dim)
        self.route_proj = nn.Linear(route_feature_dim, hidden_dim)
        self.traffic_proj = nn.Linear(traffic_feature_dim, hidden_dim)
        self.goal_proj = nn.Linear(goal_feature_dim, hidden_dim)
        self.type_emb = nn.Embedding(6, hidden_dim)
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

    @staticmethod
    def _fit_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[-1] == dim:
            return x
        if x.shape[-1] > dim:
            return x[..., :dim]
        return F.pad(x, (0, dim - x.shape[-1]))

    def _add_type(self, x: torch.Tensor, type_id: int) -> torch.Tensor:
        emb = self.type_emb(torch.full((x.shape[0], x.shape[1]), type_id, dtype=torch.long, device=x.device))
        return x + emb

    @staticmethod
    def _polyline_nonempty(poly: torch.Tensor) -> torch.Tensor:
        # poly: [B,N,P,D].  Treat a row with at least one non-zero coordinate/feature
        # as a real token when no explicit mask is provided.
        return poly.abs().sum(dim=(-1, -2)) > 0

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        B = batch["ego_history"].shape[0]
        device = batch["ego_history"].device
        tokens: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []

        cls = self.cls.expand(B, -1, -1) + self.type_emb(torch.zeros((B, 1), dtype=torch.long, device=device))
        tokens.append(cls)
        pad_masks.append(torch.zeros((B, 1), dtype=torch.bool, device=device))

        ego = self._fit_last_dim(batch["ego_history"].float(), 5)
        ego_tokens = self._add_type(self.ego_proj(ego), 1)
        tokens.append(ego_tokens)
        pad_masks.append(torch.zeros((B, ego_tokens.shape[1]), dtype=torch.bool, device=device))

        if "agent_history" in batch:
            agent = self._fit_last_dim(batch["agent_history"].float(), 10)
            B0, N, H, D = agent.shape
            agent_tokens = self._add_type(self.agent_proj(agent.reshape(B0, N * H, D)), 2)
            if "agent_valid" in batch:
                valid = batch["agent_valid"].bool().unsqueeze(-1).expand(B0, N, H).reshape(B0, N * H)
            else:
                valid = agent.abs().sum(dim=-1).reshape(B0, N * H) > 0
            tokens.append(agent_tokens)
            pad_masks.append(~valid)

        if "map_polylines" in batch:
            mp = self._fit_last_dim(batch["map_polylines"].float(), self.map_proj.in_features)
            map_tokens = self._add_type(self.map_proj(mp.mean(dim=2)), 3)
            valid = batch.get("map_polyline_valid", self._polyline_nonempty(mp)).bool()
            tokens.append(map_tokens)
            pad_masks.append(~valid)

        if "route_polylines" in batch:
            rt = self._fit_last_dim(batch["route_polylines"].float(), self.route_proj.in_features)
            route_tokens = self._add_type(self.route_proj(rt.mean(dim=2)), 4)
            valid = batch.get("route_token_valid", self._polyline_nonempty(rt)).bool()
            tokens.append(route_tokens)
            pad_masks.append(~valid)

        if "traffic_control_tokens" in batch:
            tl = self._fit_last_dim(batch["traffic_control_tokens"].float(), self.traffic_proj.in_features)
            tl_tokens = self._add_type(self.traffic_proj(tl), 5)
            valid = batch.get("traffic_token_valid", tl.abs().sum(dim=-1) > 0).bool()
            tokens.append(tl_tokens)
            pad_masks.append(~valid)

        if "mission_goal" in batch:
            goal = self._fit_last_dim(batch["mission_goal"].float(), self.goal_proj.in_features)
            if goal.dim() == 2:
                goal = goal[:, None, :]
            goal_tokens = self._add_type(self.goal_proj(goal), 4)
            if "mission_goal_valid" in batch:
                valid = batch["mission_goal_valid"].bool().reshape(B, 1)
            else:
                valid = goal.abs().sum(dim=-1) > 0
            tokens.append(goal_tokens)
            pad_masks.append(~valid)

        x = torch.cat(tokens, dim=1)
        src_key_padding_mask = torch.cat(pad_masks, dim=1)
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(h[:, 0])
