from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from bdse.data.tensorizer import runtime_to_model_numpy
from bdse.model.action_encoder import ActionEncoder
from bdse.model.evidence_encoder import EvidenceEncoder
from bdse.model.scene_encoder import SceneEncoder
from bdse.planner.evidence_queries import FAMILY_NAMES, TYPE_NAMES, compute_query_features_for_pairs
from bdse.planner.fallback import runtime_safety_flags_from_runtime
from bdse.planner.pair_screen import build_runtime_pairs_from_base

EVIDENCE_TYPE_TO_ID = TYPE_NAMES
FAMILY_TO_ID = FAMILY_NAMES


class BDSEModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        mcfg = cfg.get("model", {})
        h = int(mcfg.get("hidden_dim", 256))
        self.cfg = cfg
        self.scene = SceneEncoder(
            h,
            int(mcfg.get("transformer_layers", 4)),
            int(mcfg.get("attention_heads", 8)),
            float(mcfg.get("dropout", 0.1)),
            int(mcfg.get("map_feature_dim", 8)),
            int(mcfg.get("route_feature_dim", 8)),
            int(mcfg.get("traffic_feature_dim", 12)),
            int(mcfg.get("goal_feature_dim", 4)),
        )
        self.action = ActionEncoder(h)
        self.evidence = EvidenceEncoder(h, int(mcfg.get("evidence_feature_dim", 24)))
        self.proposal_feature_proj = nn.Sequential(
            nn.Linear(int(mcfg.get("proposal_feature_dim", 24)), h), nn.ReLU(), nn.Linear(h, h)
        )
        self.action_set_proj = nn.Sequential(nn.Linear(16, h), nn.ReLU(), nn.Linear(h, h))
        self.query_proj = nn.Sequential(nn.Linear(int(mcfg.get("query_feature_dim", 12)), h), nn.ReLU(), nn.Linear(h, h))
        self.base_head = nn.Sequential(nn.LayerNorm(h * 2), nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, 1))
        self.local_head = nn.Sequential(nn.LayerNorm(h * 4), nn.Linear(h * 4, h), nn.ReLU(), nn.Linear(h, 1), nn.Softplus())
        # p_i = f_prop(h_x, z_i, u(A_t)); u(A_t) is projected by action_set_proj.
        self.proposal_head = nn.Sequential(nn.LayerNorm(h * 4), nn.Linear(h * 4, h), nn.ReLU(), nn.Linear(h, 1))
        # Backward-compatible name used by old checkpoints/tests.
        self.selector_head = self.proposal_head

    @staticmethod
    def _fit_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[-1] == dim:
            return x
        if x.shape[-1] > dim:
            return x[..., :dim]
        return torch.nn.functional.pad(x, (0, dim - x.shape[-1]))

    def _candidate_set_summary(self, J0: torch.Tensor, trajectories: torch.Tensor, maneuver_ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, K = J0.shape
        finite = torch.where(torch.isfinite(J0) & valid, J0, torch.full_like(J0, 1e6))
        masked = finite.masked_fill(~valid, 1e6)
        valid_count = valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        sorted_cost, _ = torch.sort(masked, dim=1)
        top1 = sorted_cost[:, :1]
        top2 = sorted_cost[:, 1:2] if K > 1 else top1
        gap12 = (top2 - top1).clamp(min=0.0, max=1e4)
        centered = torch.where(valid, -(finite - top1), torch.full_like(finite, -1e6))
        prob = torch.softmax(centered, dim=1)
        entropy = -(prob * torch.log(prob.clamp_min(1e-9))).sum(dim=1, keepdim=True)
        near_count = ((masked - top1).abs() < float(self.cfg.get("selector", {}).get("eta_pred", 1.0))).float().sum(dim=1, keepdim=True) / valid_count
        progress = trajectories[:, :, -1, 0].float().masked_fill(~valid, 0.0)
        lateral = trajectories[:, :, -1, 1].float().abs().masked_fill(~valid, 0.0)
        speed = trajectories[:, :, -1, 3].float().masked_fill(~valid, 0.0)
        prog_mean = progress.sum(dim=1, keepdim=True) / valid_count
        prog_max = progress.masked_fill(~valid, -1e6).max(dim=1, keepdim=True).values.clamp_min(0.0)
        lat_mean = lateral.sum(dim=1, keepdim=True) / valid_count
        speed_mean = speed.sum(dim=1, keepdim=True) / valid_count
        # Compact maneuver coverage histogram for ids [0..5].
        hists = []
        for m in range(6):
            hists.append(((maneuver_ids == m) & valid).float().sum(dim=1, keepdim=True) / valid_count)
        raw = torch.cat([valid_count / max(float(K), 1.0), entropy, gap12 / 100.0, near_count, prog_mean / 100.0, prog_max / 100.0, lat_mean / 10.0, speed_mean / 30.0, *hists], dim=1)
        if raw.shape[1] < 16:
            raw = torch.nn.functional.pad(raw, (0, 16 - raw.shape[1]))
        return raw[:, :16]

    def encode_context(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scene = self.scene(batch)
        traj = batch["candidate_trajectories"].float()
        valid = batch["candidate_valid"].bool()
        mid = batch.get("candidate_maneuver_ids", torch.zeros(traj.shape[:2], dtype=torch.long, device=traj.device))
        action_h = self.action(traj, mid, valid)
        B, K, _ = action_h.shape
        scene_a = scene[:, None, :].expand(B, K, -1)
        J0_raw = self.base_head(torch.cat([action_h, scene_a], dim=-1)).squeeze(-1)
        J0 = J0_raw.masked_fill(~valid, float("inf"))

        E = batch["evidence_features"].shape[1]
        e_valid = batch.get("evidence_active", torch.ones((B, E), dtype=torch.bool, device=traj.device)).bool()
        type_ids = batch.get("evidence_type_ids", torch.zeros((B, E), dtype=torch.long, device=traj.device))
        fam_ids = batch.get("evidence_family_ids", torch.zeros((B, E), dtype=torch.long, device=traj.device))
        evid_h = self.evidence(batch["evidence_features"].float(), type_ids, fam_ids, e_valid)
        prop_feat = self._fit_last_dim(batch.get("evidence_proposal_features", batch["evidence_features"]).float(), self.proposal_feature_proj[0].in_features)
        prop_h = self.proposal_feature_proj(prop_feat)
        scene_e = scene[:, None, :].expand(B, E, -1)
        u_A = self.action_set_proj(self._candidate_set_summary(J0, traj, mid, valid))
        u_e = u_A[:, None, :].expand(B, E, -1)
        proposal_logits = self.proposal_head(torch.cat([evid_h, prop_h, scene_e, u_e], dim=-1)).squeeze(-1)
        proposal_logits = proposal_logits.masked_fill(~e_valid, -1e9)
        return {"scene": scene, "action_h": action_h, "evidence_h": evid_h, "J0": J0, "proposal_logits": proposal_logits, "evidence_valid": e_valid, "action_set_summary": u_A}

    def score_sparse_queries(
        self,
        context: dict[str, torch.Tensor],
        atom_indices: torch.Tensor,
        action_indices: torch.Tensor,
        query_features: torch.Tensor,
    ) -> torch.Tensor:
        # atom/action indices: [B,Q], query_features: [B,Q,Dq]
        action_h = context["action_h"]
        evid_h = context["evidence_h"]
        scene = context["scene"]
        B, Q = atom_indices.shape
        H = action_h.shape[-1]
        a_idx = action_indices.long().clamp_min(0).clamp_max(action_h.shape[1] - 1)
        e_idx = atom_indices.long().clamp_min(0).clamp_max(evid_h.shape[1] - 1)
        a_h = torch.gather(action_h, 1, a_idx[..., None].expand(B, Q, H))
        e_h = torch.gather(evid_h, 1, e_idx[..., None].expand(B, Q, H))
        q_h = self.query_proj(self._fit_last_dim(query_features.float(), self.query_proj[0].in_features))
        s_h = scene[:, None, :].expand(B, Q, H)
        return self.local_head(torch.cat([a_h, e_h, q_h, s_h], dim=-1)).squeeze(-1)

    def propose_atoms(self, context: dict[str, torch.Tensor], M: int) -> torch.Tensor:
        logits = context["proposal_logits"]
        k = min(max(int(M), 1), logits.shape[1])
        return torch.topk(logits, k=k, dim=1).indices

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ctx = self.encode_context(batch)
        J0 = ctx["J0"]
        action_h = ctx["action_h"]
        evid_h = ctx["evidence_h"]
        scene = ctx["scene"]
        valid = batch["candidate_valid"].bool()
        e_valid = ctx["evidence_valid"]
        B, K, H = action_h.shape
        E = evid_h.shape[1]
        if "evidence_query_features" in batch:
            q_h = self.query_proj(self._fit_last_dim(batch["evidence_query_features"].float(), self.query_proj[0].in_features))
            a_exp = action_h[:, None, :, :].expand(B, E, K, -1)
            e_exp = evid_h[:, :, None, :].expand(B, E, K, -1)
            s_exp = scene[:, None, None, :].expand(B, E, K, -1)
            local = self.local_head(torch.cat([a_exp, e_exp, q_h, s_exp], dim=-1)).squeeze(-1)
            local = local.masked_fill(~valid[:, None, :], 0.0).masked_fill(~e_valid[:, :, None], 0.0)
        else:
            local = torch.zeros((B, E, K), dtype=J0.dtype, device=J0.device)
        out = {
            "J0": J0,
            "g": local,
            "proposal_logits": ctx["proposal_logits"],
            "selector_logits": ctx["proposal_logits"],
            "scene": scene,
            "action_h": action_h,
            "evidence_h": evid_h,
        }
        return out

    def _make_batch(self, runtime, candidates, evidence_bank, include_dense_query: bool = False) -> dict[str, torch.Tensor]:
        arrays = runtime_to_model_numpy(runtime, candidates, evidence_bank, self.cfg, include_dense_query=include_dense_query)
        batch = {}
        for k, v in arrays.items():
            arr = np.asarray(v)
            if arr.dtype == np.bool_:
                t = torch.from_numpy(arr[None].astype(bool))
            elif np.issubdtype(arr.dtype, np.integer):
                t = torch.from_numpy(arr[None].astype(np.int64))
            else:
                t = torch.from_numpy(arr[None].astype(np.float32))
            batch[k] = t
        device = next(self.parameters()).device
        return {k: v.to(device) for k, v in batch.items()}

    def predict_certificate_numpy(self, runtime, candidates, evidence_bank, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = cfg or self.cfg
        batch = self._make_batch(runtime, candidates, evidence_bank, include_dense_query=False)
        self.eval()
        with torch.no_grad():
            ctx = self.encode_context(batch)
        J0 = ctx["J0"][0].detach().cpu().numpy().astype(np.float32)
        proposal_logits = ctx["proposal_logits"][0].detach().cpu().numpy().astype(np.float32)
        flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
        pairs, pair_weights = build_runtime_pairs_from_base(
            J0,
            candidates.valid_mask,
            flags,
            L0=int(cfg.get("tournament", {}).get("L_infer", 16)),
            eta0=float(cfg.get("selector", {}).get("eta_pred", 1.0)),
            lambda_near=float(cfg.get("selector", {}).get("lambda_near", 1.0)),
            lambda_safety=float(cfg.get("selector", {}).get("lambda_safety", 2.0)),
        )
        budget = int(cfg.get("evidence", {}).get("budget", 16))
        M = int(cfg.get("selector", {}).get("proposal_top_m", max(2 * budget, budget + 1)))
        active = np.asarray(evidence_bank.active_mask, dtype=bool)
        masked_logits = np.where(active, proposal_logits[: len(active)], -1e9)
        topm = np.argsort(-masked_logits)[: min(M, int(active.sum()) if active.any() else len(masked_logits))].astype(np.int64)
        if len(pairs):
            action_ids = np.unique(pairs.reshape(-1))
        else:
            action_ids = np.flatnonzero(candidates.valid_mask)[: max(1, int(cfg.get("tournament", {}).get("L_infer", 16)))]
        atom_ids, action_ids_rep, q = compute_query_features_for_pairs(evidence_bank.atoms, candidates, runtime, topm, action_ids, cfg)
        g_sparse = np.zeros((evidence_bank.E, candidates.K), dtype=np.float32)
        if len(atom_ids):
            atom_t = torch.from_numpy(atom_ids[None].astype(np.int64)).to(next(self.parameters()).device)
            act_t = torch.from_numpy(action_ids_rep[None].astype(np.int64)).to(next(self.parameters()).device)
            q_t = torch.from_numpy(q[None].astype(np.float32)).to(next(self.parameters()).device)
            with torch.no_grad():
                vals = self.score_sparse_queries(ctx, atom_t, act_t, q_t)[0].detach().cpu().numpy().astype(np.float32)
            g_sparse[atom_ids, action_ids_rep] = vals
        g_sparse[:, ~np.asarray(candidates.valid_mask, dtype=bool)] = 0.0
        return {
            "J0": J0,
            "g": g_sparse,
            "proposal_logits": proposal_logits,
            "top_m_atoms": topm,
            "queried_actions": np.asarray(action_ids, dtype=np.int64),
            "queried_pair_count": int(len(atom_ids)),
            "runtime_pairs": pairs,
            "runtime_pair_weights": pair_weights,
        }

    def predict_numpy(self, runtime, candidates, evidence_bank):
        # Legacy API: return only base and sparse-scored local costs.  The runtime
        # planner uses predict_certificate_numpy to also obtain proposal diagnostics.
        out = self.predict_certificate_numpy(runtime, candidates, evidence_bank, self.cfg)
        return out["J0"], out["g"]
