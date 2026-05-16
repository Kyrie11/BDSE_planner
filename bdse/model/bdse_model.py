from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from bdse.model.action_encoder import ActionEncoder
from bdse.model.evidence_encoder import EvidenceEncoder
from bdse.model.scene_encoder import SceneEncoder


EVIDENCE_TYPE_TO_ID = {
    "occupancy": 1,
    "ttc": 2,
    "gap": 3,
    "yield": 4,
    "red_light": 5,
    "drivable_area": 6,
    "wrong_way": 7,
    "route_connector": 8,
    "speed_limit": 9,
    "local_comfort_accel": 10,
    "local_comfort_jerk": 11,
    "local_comfort_curvature": 12,
    "local_comfort_brake": 13,
}
FAMILY_TO_ID = {"interaction": 1, "rule_map": 2, "kinematic": 3}


class BDSEModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        mcfg = cfg.get("model", {})
        h = int(mcfg.get("hidden_dim", 256))
        self.cfg = cfg
        self.scene = SceneEncoder(h, int(mcfg.get("transformer_layers", 4)), int(mcfg.get("attention_heads", 8)), float(mcfg.get("dropout", 0.1)))
        self.action = ActionEncoder(h)
        self.evidence = EvidenceEncoder(h, int(mcfg.get("evidence_feature_dim", 24)))
        self.query_proj = nn.Sequential(nn.Linear(int(mcfg.get("query_feature_dim", 12)), h), nn.ReLU(), nn.Linear(h, h))
        self.base_head = nn.Sequential(nn.LayerNorm(h * 2), nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, 1))
        self.local_head = nn.Sequential(nn.LayerNorm(h * 4), nn.Linear(h * 4, h), nn.ReLU(), nn.Linear(h, 1), nn.Softplus())
        self.selector_head = nn.Sequential(nn.LayerNorm(h * 2), nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scene = self.scene(batch)
        traj = batch["candidate_trajectories"].float()
        valid = batch["candidate_valid"].bool()
        mid = batch.get("candidate_maneuver_ids", torch.zeros(traj.shape[:2], dtype=torch.long, device=traj.device))
        action_h = self.action(traj, mid, valid)
        B, K, _ = action_h.shape
        scene_a = scene[:, None, :].expand(B, K, -1)
        J0 = self.base_head(torch.cat([action_h, scene_a], dim=-1)).squeeze(-1)
        J0 = J0.masked_fill(~valid, float("inf"))

        E = batch["evidence_features"].shape[1]
        e_valid = batch.get("evidence_active", torch.ones((B, E), dtype=torch.bool, device=traj.device)).bool()
        type_ids = batch.get("evidence_type_ids", torch.zeros((B, E), dtype=torch.long, device=traj.device))
        fam_ids = batch.get("evidence_family_ids", torch.zeros((B, E), dtype=torch.long, device=traj.device))
        evid_h = self.evidence(batch["evidence_features"].float(), type_ids, fam_ids, e_valid)
        q_h = self.query_proj(batch["evidence_query_features"].float())
        a_exp = action_h[:, None, :, :].expand(B, E, K, -1)
        e_exp = evid_h[:, :, None, :].expand(B, E, K, -1)
        s_exp = scene[:, None, None, :].expand(B, E, K, -1)
        local = self.local_head(torch.cat([a_exp, e_exp, q_h, s_exp], dim=-1)).squeeze(-1)
        local = local.masked_fill(~valid[:, None, :], 0.0).masked_fill(~e_valid[:, :, None], 0.0)
        selector_logits = self.selector_head(torch.cat([evid_h, scene[:, None, :].expand(B, E, -1)], dim=-1)).squeeze(-1)
        selector_logits = selector_logits.masked_fill(~e_valid, -1e9)
        return {"J0": J0, "g": local, "selector_logits": selector_logits, "scene": scene, "action_h": action_h, "evidence_h": evid_h}

    def predict_numpy(self, runtime, candidates, evidence_bank):
        K = candidates.K
        E = evidence_bank.E
        mcfg = self.cfg.get("model", {})
        efd = int(mcfg.get("evidence_feature_dim", 24))
        features = np.zeros((E, efd), dtype=np.float32)
        type_ids = np.zeros((E,), dtype=np.int64)
        family_ids = np.zeros((E,), dtype=np.int64)
        for i, atom in enumerate(evidence_bank.atoms):
            type_ids[i] = EVIDENCE_TYPE_TO_ID.get(atom.type, 0)
            family_ids[i] = FAMILY_TO_ID.get(atom.family, 0)
            features[i, 0] = float(atom.is_hard)
            features[i, 1] = float(atom.budget_cost)
            if "current_state" in atom.anchor:
                st = atom.anchor["current_state"]
                features[i, 2 : 2 + min(10, len(st))] = st[: min(10, len(st))]
        batch = {
            "ego_history": torch.from_numpy(runtime.ego_history[None]).float(),
            "agent_history": torch.from_numpy(runtime.agent_history[None]).float(),
            "agent_valid": torch.from_numpy(runtime.agent_valid[None]),
            "candidate_trajectories": torch.from_numpy(candidates.trajectories[None]).float(),
            "candidate_valid": torch.from_numpy(candidates.valid_mask[None]),
            "candidate_maneuver_ids": torch.from_numpy(candidates.maneuver_ids[None]),
            "evidence_features": torch.from_numpy(features[None]).float(),
            "evidence_query_features": torch.from_numpy(evidence_bank.query_features[None]).float(),
            "evidence_active": torch.from_numpy(evidence_bank.active_mask[None]),
            "evidence_type_ids": torch.from_numpy(type_ids[None]),
            "evidence_family_ids": torch.from_numpy(family_ids[None]),
        }
        device = next(self.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        self.eval()
        with torch.no_grad():
            out = self(batch)
        return out["J0"][0].detach().cpu().numpy().astype(np.float32), out["g"][0].detach().cpu().numpy().astype(np.float32)
