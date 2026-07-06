from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bdse.data.tensorizer import runtime_to_model_numpy
from bdse.model.action_encoder import ActionEncoder
from bdse.model.evidence_encoder import EvidenceEncoder
from bdse.model.scene_encoder import SceneEncoder
from bdse.planner.evidence_queries import FAMILY_NAMES, TYPE_NAMES, compute_query_features_for_pairs
from bdse.planner.fallback import runtime_safety_flags_from_runtime
from bdse.planner.hab import max_family_id, select_topm_atoms_hab
from bdse.planner.pair_screen import build_runtime_pairs_from_base, build_rival_sets_from_base
from bdse.planner.selector import margin_normalization_scale, structural_safety_mask

EVIDENCE_TYPE_TO_ID = TYPE_NAMES
FAMILY_TO_ID = FAMILY_NAMES


class BDSEModel(nn.Module):
    """Neural BDSE interface model.

    The model follows the paper decomposition

        J_T(a) = J_0(a) + sum_i g_i(a)

    and exposes all quantities needed by the deployment-time HAB selector:
    base costs, proposal logits, family gates, local atom means, and local atom
    heteroscedastic variances.  The dense forward path is used for training;
    ``predict_certificate_numpy`` mirrors runtime by only scoring Top-M HAB atoms
    on actions that appear in the screened rival graph.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        mcfg = cfg.get("model", {})
        h = int(mcfg.get("hidden_dim", 256))
        self.cfg = cfg
        self.hidden_dim = h
        configured_families = int(mcfg.get("num_families", int(mcfg.get("max_family_id", max_family_id())) + 1))
        # Do not trust the yaml value alone: FAMILY_NAMES currently uses ids
        # 0..5.  A too-small embedding silently collapses family 5 into 4 after
        # clamp(), which mixes dynamic_regularity with decision_boundary.
        self.num_families = max(configured_families, int(max_family_id()) + 1)
        self.pair_conditioned = bool(mcfg.get("pair_conditioned", True))
        self.pair_delta_scale = float(mcfg.get("pair_delta_scale", 1.0))
        # Pair-conditioned deltas are trained/deployed in a normalized margin
        # space by default.  This keeps hard feasibility penalties (often 1e4)
        # out of the learnable interaction-margin scale and makes calibration
        # epsilon dimensionless.
        self.pair_margin_normalized = bool(mcfg.get("pair_margin_normalized", True))
        self.var_floor = float(mcfg.get("variance_floor", 1e-4))
        self.scene = SceneEncoder(
            h,
            int(mcfg.get("transformer_layers", 4)),
            int(mcfg.get("attention_heads", 8)),
            float(mcfg.get("dropout", 0.1)),
            int(mcfg.get("map_feature_dim", 8)),
            int(mcfg.get("route_feature_dim", 8)),
            int(mcfg.get("traffic_feature_dim", 12)),
            int(mcfg.get("goal_feature_dim", 4)),
            float(mcfg.get("map_token_dropout", 0.0)),
            float(mcfg.get("route_token_dropout", 0.0)),
            float(mcfg.get("traffic_token_dropout", 0.0)),
        )
        self.action = ActionEncoder(h)
        self.evidence = EvidenceEncoder(h, int(mcfg.get("evidence_feature_dim", 24)), num_families=self.num_families)
        self.proposal_feature_proj = nn.Sequential(
            nn.Linear(int(mcfg.get("proposal_feature_dim", 24)), h), nn.ReLU(), nn.Linear(h, h)
        )
        # Candidate-set context includes fixed scalar summaries plus a maneuver histogram.
        # Keep the projection input configurable/backward compatible; the default
        # remains 16, which is enough for 7 maneuvers (including safe_fallback).
        self.action_set_summary_dim = int(mcfg.get("action_set_summary_dim", 16))
        self.action_set_proj = nn.Sequential(nn.Linear(self.action_set_summary_dim, h), nn.ReLU(), nn.Linear(h, h))
        self.query_proj = nn.Sequential(nn.Linear(int(mcfg.get("query_feature_dim", 12)), h), nn.ReLU(), nn.Linear(h, h))
        self.base_head = nn.Sequential(nn.LayerNorm(h * 2), nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, 1))

        # g_i(a): non-negative local cost contribution; cost *differences* are
        # antisymmetric by construction: d_i(a,b)=g_i(b)-g_i(a).
        self.local_head = nn.Sequential(nn.LayerNorm(h * 4), nn.Linear(h * 4, h), nn.ReLU(), nn.Linear(h, 1), nn.Softplus())
        # Heteroscedastic epistemic/aleatoric uncertainty head for the same query.
        self.local_var_head = nn.Sequential(nn.LayerNorm(h * 4), nn.Linear(h * 4, h), nn.ReLU(), nn.Linear(h, 1))

        # Pair-conditioned margin scorer f_d(h_x, h_a, h_b, h_i, q_i(a), q_i(b)).
        # It predicts signed atom-level deltas d_i(a,b) directly and enforces
        # antisymmetry by scoring both orders and subtracting.  The non-negative
        # local g_i(a) head remains for backward-compatible cost diagnostics.
        # Relation-aware pair head.  Besides the ordered action/evidence/query
        # embeddings used by earlier versions, include explicit pair-difference and
        # pair-product terms.  The head still enforces antisymmetry by subtracting
        # the score of the reversed order.
        self.pair_feature_blocks = 10
        self.pair_head = nn.Sequential(nn.LayerNorm(h * self.pair_feature_blocks), nn.Linear(h * self.pair_feature_blocks, h), nn.ReLU(), nn.Linear(h, 1))
        self.pair_var_head = nn.Sequential(nn.LayerNorm(h * self.pair_feature_blocks), nn.Linear(h * self.pair_feature_blocks, h), nn.ReLU(), nn.Linear(h, 1))

        # Hierarchical Atom Builder (HAB): family gate pi_tau followed by an
        # atom proposal conditioned on family embedding and candidate-set summary.
        self.family_embed = nn.Embedding(self.num_families, h)
        self.family_activity_proj = nn.Sequential(nn.Linear(8, h), nn.ReLU(), nn.Linear(h, h))
        self.family_head = nn.Sequential(nn.LayerNorm(h * 4), nn.Linear(h * 4, h), nn.ReLU(), nn.Linear(h, 1))
        self.proposal_head = nn.Sequential(nn.LayerNorm(h * 5), nn.Linear(h * 5, h), nn.ReLU(), nn.Linear(h, 1))
        # Backward-compatible name used by old checkpoints/tests.
        self.selector_head = self.proposal_head

    @staticmethod
    def _fit_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[-1] == dim:
            return x
        if x.shape[-1] > dim:
            return x[..., :dim]
        return F.pad(x, (0, dim - x.shape[-1]))

    @staticmethod
    def _positive_variance(raw: torch.Tensor, floor: float) -> torch.Tensor:
        return F.softplus(raw).clamp_min(float(floor))

    def _candidate_set_summary(
        self,
        J0: torch.Tensor,
        trajectories: torch.Tensor,
        maneuver_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        B, K = J0.shape
        # This summary is used inside autocast.  Keep sentinel arithmetic in
        # float32: fp16 cannot represent 1e6/-1e6 and will overflow when AMP is
        # enabled.  Returning fp32 is safe because the following Linear layer is
        # autocast-aware.
        valid = valid.bool()
        J0f = J0.float()
        large = J0f.new_tensor(1e6)
        neg_large = J0f.new_tensor(-1e6)
        finite = torch.where(torch.isfinite(J0f) & valid, J0f, large)
        masked = finite.masked_fill(~valid, large)
        valid_count = valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        sorted_cost, _ = torch.sort(masked, dim=1)
        top1 = sorted_cost[:, :1]
        top2 = sorted_cost[:, 1:2] if K > 1 else top1
        gap12 = (top2 - top1).clamp(min=0.0, max=1e4)
        centered = torch.where(valid, -(finite - top1), neg_large.expand_as(finite))
        prob = torch.softmax(centered, dim=1)
        entropy = -(prob * torch.log(prob.clamp_min(1e-9))).sum(dim=1, keepdim=True)
        near_count = ((masked - top1).abs() < float(self.cfg.get("selector", {}).get("eta_pred", 1.0))).float().sum(dim=1, keepdim=True) / valid_count
        progress = trajectories[:, :, -1, 0].float().masked_fill(~valid, 0.0)
        lateral = trajectories[:, :, -1, 1].float().abs().masked_fill(~valid, 0.0)
        speed = trajectories[:, :, -1, 3].float().masked_fill(~valid, 0.0)
        prog_mean = progress.sum(dim=1, keepdim=True) / valid_count
        prog_max = progress.masked_fill(~valid, neg_large).max(dim=1, keepdim=True).values.clamp_min(0.0)
        lat_mean = lateral.sum(dim=1, keepdim=True) / valid_count
        speed_mean = speed.sum(dim=1, keepdim=True) / valid_count
        # Compact maneuver coverage histogram.  Earlier versions hard-coded
        # ids [0..5] and silently dropped id=6 (safe_fallback), which made the
        # proposal head blind to whether the candidate bank contained conservative
        # stop/fallback options.
        max_mid = int(self.cfg.get("candidate", {}).get("num_maneuvers", 7))
        max_hist = max(0, int(self.action_set_summary_dim) - 8)
        num_bins = min(max(max_mid, 7), max_hist if max_hist > 0 else max(max_mid, 7))
        hists = []
        for m in range(num_bins):
            hists.append(((maneuver_ids == m) & valid).float().sum(dim=1, keepdim=True) / valid_count)
        raw = torch.cat(
            [
                valid_count / max(float(K), 1.0),
                entropy,
                gap12 / 100.0,
                near_count,
                prog_mean / 100.0,
                prog_max / 100.0,
                lat_mean / 10.0,
                speed_mean / 30.0,
                *hists,
            ],
            dim=1,
        )
        target_dim = int(self.action_set_summary_dim)
        if raw.shape[1] < target_dim:
            raw = F.pad(raw, (0, target_dim - raw.shape[1]))
        return raw[:, :target_dim]

    def _family_activity_features(self, prop_feat: torch.Tensor, fam_ids: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Summarise atom availability/proposal metadata per certificate family.

        Features are intentionally low-dimensional and robust to older cached
        tensors with fewer proposal columns.  Family 0 is padding/unknown; it is
        allowed only when no semantic family is present.
        """
        B, E, D = prop_feat.shape
        device = prop_feat.device
        dtype = prop_feat.dtype
        n_fam = int(self.num_families)
        fam = fam_ids.long().clamp(min=0, max=n_fam - 1)
        out = torch.zeros((B, n_fam, 8), dtype=dtype, device=device)
        present_list = []

        def col(idx: int) -> torch.Tensor:
            if D <= 0:
                return torch.zeros((B, E), dtype=dtype, device=device)
            j = min(max(int(idx), 0), D - 1)
            return prop_feat[..., j]

        for f_id in range(n_fam):
            mask = (fam == f_id) & active
            present_list.append(mask.any(dim=1))
            mf = mask.float()
            count = mf.sum(dim=1)
            denom = count.clamp_min(1.0)
            out[:, f_id, 0] = count / max(float(E), 1.0)
            out[:, f_id, 1] = mask.any(dim=1).float()
            # Proposal feature schema from evidence_queries: hard flag, budget,
            # active, radius, lambda, route distance, nearest candidate distance,
            # overlap/urgency/rule activity.  The index guards keep this usable
            # for legacy tensors.
            for dst, src in [(2, 0), (3, 1), (4, 6), (6, 10), (7, 11)]:
                out[:, f_id, dst] = (col(src) * mf).sum(dim=1) / denom
            cand_dist = col(8)
            large = torch.full_like(cand_dist, 1e3)
            min_dist = torch.where(mask, cand_dist, large).min(dim=1).values
            out[:, f_id, 5] = torch.where(mask.any(dim=1), min_dist.clamp(0.0, 1e3) / 100.0, torch.zeros_like(min_dist))

        family_active = torch.stack(present_list, dim=1) if present_list else torch.zeros((B, n_fam), dtype=torch.bool, device=device)
        semantic_present = family_active[:, 1:].any(dim=1) if n_fam > 1 else torch.zeros((B,), dtype=torch.bool, device=device)
        if n_fam > 1:
            family_active = family_active.clone()
            family_active[:, 0] = family_active[:, 0] & ~semantic_present
        no_active = ~family_active.any(dim=1)
        if no_active.any():
            family_active = family_active.clone()
            family_active[no_active, 0] = True
            out[no_active, 0, 1] = 1.0
        return out, family_active

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
        fam_ids_raw = batch.get("evidence_family_ids", torch.zeros((B, E), dtype=torch.long, device=traj.device))
        fam_ids = fam_ids_raw.long().clamp(min=0, max=self.num_families - 1)
        evid_h = self.evidence(batch["evidence_features"].float(), type_ids, fam_ids, e_valid)
        prop_feat = self._fit_last_dim(batch.get("evidence_proposal_features", batch["evidence_features"]).float(), self.proposal_feature_proj[0].in_features)
        prop_h = self.proposal_feature_proj(prop_feat)
        scene_e = scene[:, None, :].expand(B, E, -1)
        u_A = self.action_set_proj(self._candidate_set_summary(J0, traj, mid, valid))
        u_e = u_A[:, None, :].expand(B, E, -1)

        family_feat, family_active = self._family_activity_features(prop_feat, fam_ids, e_valid)
        family_act_h = self.family_activity_proj(family_feat)
        fam_token = torch.arange(self.num_families, dtype=torch.long, device=traj.device)
        family_emb = self.family_embed(fam_token)[None, :, :].expand(B, -1, -1)
        scene_f = scene[:, None, :].expand(B, self.num_families, -1)
        u_f = u_A[:, None, :].expand(B, self.num_families, -1)
        family_logits_raw = self.family_head(torch.cat([scene_f, u_f, family_emb, family_act_h], dim=-1)).squeeze(-1)
        neg_mask = torch.finfo(family_logits_raw.dtype).min / 2.0
        family_logits = family_logits_raw.masked_fill(~family_active, neg_mask)
        family_pi = torch.softmax(family_logits.float(), dim=1).to(family_logits.dtype) * family_active.float()
        family_pi = family_pi / family_pi.sum(dim=1, keepdim=True).clamp_min(1e-6)

        atom_family_h = self.family_embed(fam_ids)
        atom_pi = torch.gather(family_pi, 1, fam_ids).clamp_min(1e-6)
        proposal_logits = self.proposal_head(torch.cat([evid_h, prop_h, scene_e, u_e, atom_family_h], dim=-1)).squeeze(-1)
        # Condition the atom proposal by the learned family gate.  Invalid atoms
        # are masked after adding the log gate so gradients still reach the gate.
        proposal_logits = proposal_logits + torch.log(atom_pi)
        proposal_neg_mask = torch.finfo(proposal_logits.dtype).min / 2.0
        proposal_logits = proposal_logits.masked_fill(~e_valid, proposal_neg_mask)
        return {
            "scene": scene,
            "action_h": action_h,
            "evidence_h": evid_h,
            "J0": J0,
            "proposal_logits": proposal_logits,
            "selector_logits": proposal_logits,
            "family_logits": family_logits,
            "family_pi": family_pi,
            "family_active": family_active,
            "evidence_valid": e_valid,
            "action_set_summary": u_A,
        }

    def _sparse_pair_features(
        self,
        context: dict[str, torch.Tensor],
        atom_indices: torch.Tensor,
        action_a_indices: torch.Tensor,
        action_b_indices: torch.Tensor,
        query_a_features: torch.Tensor,
        query_b_features: torch.Tensor,
    ) -> torch.Tensor:
        action_h = context["action_h"]
        evid_h = context["evidence_h"]
        scene = context["scene"]
        B, Q = atom_indices.shape
        H = action_h.shape[-1]
        a_idx = action_a_indices.long().clamp_min(0).clamp_max(action_h.shape[1] - 1)
        b_idx = action_b_indices.long().clamp_min(0).clamp_max(action_h.shape[1] - 1)
        e_idx = atom_indices.long().clamp_min(0).clamp_max(evid_h.shape[1] - 1)
        a_h = torch.gather(action_h, 1, a_idx[..., None].expand(B, Q, H))
        b_h = torch.gather(action_h, 1, b_idx[..., None].expand(B, Q, H))
        e_h = torch.gather(evid_h, 1, e_idx[..., None].expand(B, Q, H))
        q_a = self.query_proj(self._fit_last_dim(query_a_features.float(), self.query_proj[0].in_features))
        q_b = self.query_proj(self._fit_last_dim(query_b_features.float(), self.query_proj[0].in_features))
        s_h = scene[:, None, :].expand(B, Q, H)
        return torch.cat([a_h, b_h, e_h, q_a, q_b, s_h, b_h - a_h, a_h * b_h, q_b - q_a, q_a * q_b], dim=-1)

    def score_sparse_pairs(
        self,
        context: dict[str, torch.Tensor],
        atom_indices: torch.Tensor,
        action_a_indices: torch.Tensor,
        action_b_indices: torch.Tensor,
        query_a_features: torch.Tensor,
        query_b_features: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z_ab = self._sparse_pair_features(context, atom_indices, action_a_indices, action_b_indices, query_a_features, query_b_features)
        z_ba = self._sparse_pair_features(context, atom_indices, action_b_indices, action_a_indices, query_b_features, query_a_features)
        delta = (self.pair_head(z_ab) - self.pair_head(z_ba)).squeeze(-1) * float(self.pair_delta_scale)
        if not return_uncertainty:
            return delta
        # Variance is symmetric for the ordered comparison.
        var = self._positive_variance(self.pair_var_head(z_ab).squeeze(-1), self.var_floor)
        var = var + self._positive_variance(self.pair_var_head(z_ba).squeeze(-1), self.var_floor)
        return delta, var

    def _sparse_query_features(
        self,
        context: dict[str, torch.Tensor],
        atom_indices: torch.Tensor,
        action_indices: torch.Tensor,
        query_features: torch.Tensor,
    ) -> torch.Tensor:
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
        return torch.cat([a_h, e_h, q_h, s_h], dim=-1)

    def score_sparse_queries(
        self,
        context: dict[str, torch.Tensor],
        atom_indices: torch.Tensor,
        action_indices: torch.Tensor,
        query_features: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # atom/action indices: [B,Q], query_features: [B,Q,Dq]
        z = self._sparse_query_features(context, atom_indices, action_indices, query_features)
        mean = self.local_head(z).squeeze(-1)
        if not return_uncertainty:
            return mean
        var = self._positive_variance(self.local_var_head(z).squeeze(-1), self.var_floor)
        return mean, var

    def propose_atoms(self, context: dict[str, torch.Tensor], M: int) -> torch.Tensor:
        logits = context["proposal_logits"]
        k = min(max(int(M), 1), logits.shape[1])
        return torch.topk(logits, k=k, dim=1).indices

    def _training_chunk_size(self, key: str, default: int) -> int:
        train_cfg = self.cfg.get("training", {}) if isinstance(self.cfg, dict) else {}
        model_cfg = self.cfg.get("model", {}) if isinstance(self.cfg, dict) else {}
        value = train_cfg.get(key, model_cfg.get(key, default))
        try:
            return max(1, int(value))
        except Exception:
            return max(1, int(default))

    def _dense_local_from_batch(self, context: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        action_h = context["action_h"]
        evid_h = context["evidence_h"]
        scene = context["scene"]
        J0 = context["J0"]
        valid = batch["candidate_valid"].bool()
        e_valid = context["evidence_valid"]
        q_raw_all = self._fit_last_dim(batch["evidence_query_features"].float(), self.query_proj[0].in_features)
        B, K, H = action_h.shape
        E = evid_h.shape[1]
        chunk_e = min(E, self._training_chunk_size("local_forward_atom_chunk", 32))
        local_parts: list[torch.Tensor] = []
        local_var_parts: list[torch.Tensor] = []
        for e0 in range(0, E, chunk_e):
            e1 = min(E, e0 + chunk_e)
            Ce = e1 - e0
            q_h = self.query_proj(q_raw_all[:, e0:e1])
            a_exp = action_h[:, None, :, :].expand(B, Ce, K, H)
            e_exp = evid_h[:, e0:e1, None, :].expand(B, Ce, K, H)
            s_exp = scene[:, None, None, :].expand(B, Ce, K, H)
            z = torch.cat([a_exp, e_exp, q_h, s_exp], dim=-1)
            local_chunk = self.local_head(z).squeeze(-1)
            local_var_chunk = self._positive_variance(self.local_var_head(z).squeeze(-1), self.var_floor)
            local_chunk = local_chunk.masked_fill(~valid[:, None, :], 0.0).masked_fill(~e_valid[:, e0:e1, None], 0.0)
            local_var_chunk = local_var_chunk.masked_fill(~valid[:, None, :], 0.0).masked_fill(~e_valid[:, e0:e1, None], 0.0)
            local_parts.append(local_chunk)
            local_var_parts.append(local_var_chunk)
        if not local_parts:
            return (
                torch.zeros((B, E, K), dtype=J0.dtype, device=J0.device),
                torch.zeros((B, E, K), dtype=J0.dtype, device=J0.device),
            )
        return torch.cat(local_parts, dim=1), torch.cat(local_var_parts, dim=1)

    def _dense_pair_delta_from_batch(self, context: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.pair_conditioned or "pair_indices" not in batch or "evidence_query_features" not in batch:
            return None
        pairs = batch["pair_indices"].long()
        action_h = context["action_h"]
        evid_h = context["evidence_h"]
        scene = context["scene"]
        B, P, _ = pairs.shape
        E = evid_h.shape[1]
        H = action_h.shape[-1]
        K = action_h.shape[1]
        a_idx_all = pairs[..., 0].clamp_min(0).clamp_max(K - 1)
        b_idx_all = pairs[..., 1].clamp_min(0).clamp_max(K - 1)
        q_raw_all = self._fit_last_dim(batch["evidence_query_features"].float(), self.query_proj[0].in_features)
        chunk_e = min(E, self._training_chunk_size("pair_forward_atom_chunk", 16))
        chunk_p = min(P, self._training_chunk_size("pair_forward_pair_chunk", 32))
        delta_chunks: list[torch.Tensor] = []
        var_chunks: list[torch.Tensor] = []
        for e0 in range(0, E, chunk_e):
            e1 = min(E, e0 + chunk_e)
            Ce = e1 - e0
            delta_p_chunks: list[torch.Tensor] = []
            var_p_chunks: list[torch.Tensor] = []
            q_raw_e = q_raw_all[:, e0:e1]
            e_h = evid_h[:, e0:e1, None, :].expand(B, Ce, 1, H)
            for p0 in range(0, P, chunk_p):
                p1 = min(P, p0 + chunk_p)
                Cp = p1 - p0
                a_idx = a_idx_all[:, p0:p1]
                b_idx = b_idx_all[:, p0:p1]
                a_h = torch.gather(action_h, 1, a_idx[..., None].expand(B, Cp, H))[:, None, :, :].expand(B, Ce, Cp, H)
                b_h = torch.gather(action_h, 1, b_idx[..., None].expand(B, Cp, H))[:, None, :, :].expand(B, Ce, Cp, H)
                idx_a_q = a_idx[:, None, :, None].expand(B, Ce, Cp, q_raw_e.shape[-1])
                idx_b_q = b_idx[:, None, :, None].expand(B, Ce, Cp, q_raw_e.shape[-1])
                q_a = self.query_proj(torch.gather(q_raw_e, 2, idx_a_q))
                q_b = self.query_proj(torch.gather(q_raw_e, 2, idx_b_q))
                e_hp = e_h.expand(B, Ce, Cp, H)
                s_h = scene[:, None, None, :].expand(B, Ce, Cp, H)
                z_ab = torch.cat([a_h, b_h, e_hp, q_a, q_b, s_h, b_h - a_h, a_h * b_h, q_b - q_a, q_a * q_b], dim=-1)
                z_ba = torch.cat([b_h, a_h, e_hp, q_b, q_a, s_h, a_h - b_h, a_h * b_h, q_a - q_b, q_a * q_b], dim=-1)
                delta = (self.pair_head(z_ab) - self.pair_head(z_ba)).squeeze(-1) * float(self.pair_delta_scale)
                var = self._positive_variance(self.pair_var_head(z_ab).squeeze(-1), self.var_floor)
                var = var + self._positive_variance(self.pair_var_head(z_ba).squeeze(-1), self.var_floor)
                pair_valid = batch.get("pair_valid")
                if pair_valid is not None:
                    p_mask = pair_valid[:, p0:p1].bool()
                    delta = delta.masked_fill(~p_mask[:, None, :], 0.0)
                    var = var.masked_fill(~p_mask[:, None, :], 0.0)
                e_valid = context.get("evidence_valid")
                if e_valid is not None:
                    e_mask = e_valid[:, e0:e1].bool()
                    delta = delta.masked_fill(~e_mask[:, :, None], 0.0)
                    var = var.masked_fill(~e_mask[:, :, None], 0.0)
                delta_p_chunks.append(delta)
                var_p_chunks.append(var)
            delta_chunks.append(torch.cat(delta_p_chunks, dim=2))
            var_chunks.append(torch.cat(var_p_chunks, dim=2))
        return torch.cat(delta_chunks, dim=1), torch.cat(var_chunks, dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ctx = self.encode_context(batch)
        J0 = ctx["J0"]
        action_h = ctx["action_h"]
        evid_h = ctx["evidence_h"]
        scene = ctx["scene"]
        B, K, H = action_h.shape
        E = evid_h.shape[1]
        if "evidence_query_features" in batch:
            local, local_var = self._dense_local_from_batch(ctx, batch)
            pair_out = self._dense_pair_delta_from_batch(ctx, batch)
        else:
            local = torch.zeros((B, E, K), dtype=J0.dtype, device=J0.device)
            local_var = torch.zeros((B, E, K), dtype=J0.dtype, device=J0.device)
            pair_out = None
        out = {
            "J0": J0,
            "g": local,
            "g_var": local_var,
            "proposal_logits": ctx["proposal_logits"],
            "selector_logits": ctx["proposal_logits"],
            "family_logits": ctx["family_logits"],
            "family_pi": ctx["family_pi"],
            "family_active": ctx["family_active"],
            "scene": scene,
            "action_h": action_h,
            "evidence_h": evid_h,
        }
        if pair_out is not None:
            out["pair_atom_delta"], out["pair_atom_var"] = pair_out
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

    def _score_pair_indices_numpy(
        self,
        context: dict[str, torch.Tensor],
        runtime,
        candidates,
        evidence_bank,
        atom_indices: np.ndarray,
        pair_indices: np.ndarray,
        cfg: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        atom_indices = np.asarray(atom_indices, dtype=np.int64).reshape(-1)
        pair_indices = np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2)
        E = evidence_bank.E
        P = int(pair_indices.shape[0])
        out = np.zeros((E, P), dtype=np.float32)
        var_out = np.zeros((E, P), dtype=np.float32)
        atom_indices = atom_indices[(atom_indices >= 0) & (atom_indices < E)]
        if atom_indices.size == 0 or P == 0:
            return out, var_out
        action_ids = np.unique(pair_indices.reshape(-1))
        action_ids = action_ids[(action_ids >= 0) & (action_ids < candidates.K)]
        atom_ids_flat, action_ids_rep, q = compute_query_features_for_pairs(evidence_bank.atoms, candidates, runtime, atom_indices, action_ids, cfg)
        qfd = int(self.cfg.get("model", {}).get("query_feature_dim", 12))
        q_dense = np.zeros((E, candidates.K, qfd), dtype=np.float32)
        if len(atom_ids_flat):
            d = min(qfd, q.shape[1])
            q_dense[np.asarray(atom_ids_flat, dtype=np.int64), np.asarray(action_ids_rep, dtype=np.int64), :d] = q[:, :d]

        a_all = pair_indices[:, 0]
        b_all = pair_indices[:, 1]
        valid_pairs = (a_all >= 0) & (a_all < candidates.K) & (b_all >= 0) & (b_all < candidates.K)
        pidx_valid = np.flatnonzero(valid_pairs).astype(np.int64)
        if pidx_valid.size == 0:
            return out, var_out

        # Flatten the Top-M atom x pair grid without Python list construction.
        # Closed-loop profiling showed this function can be called repeatedly per
        # planner step and can easily evaluate tens of thousands of atom-pair
        # scores; vectorizing the index grid avoids a large CPU bottleneck before
        # the batched MLP call.
        A = int(atom_indices.size)
        P_valid = int(pidx_valid.size)
        atoms_grid = np.repeat(atom_indices.astype(np.int64), P_valid)
        pidx_grid = np.tile(pidx_valid, A)
        a_grid = np.tile(a_all[pidx_valid].astype(np.int64), A)
        b_grid = np.tile(b_all[pidx_valid].astype(np.int64), A)
        qa = q_dense[atoms_grid, a_grid]
        qb = q_dense[atoms_grid, b_grid]

        device = next(self.parameters()).device
        runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
        model_cfg = self.cfg.get("model", {}) if isinstance(self.cfg, dict) else {}
        chunk = int(runtime_cfg.get("pair_score_chunk", model_cfg.get("runtime_pair_score_chunk", 65536)))
        chunk = max(1, chunk)
        with torch.inference_mode():
            for q0 in range(0, int(atoms_grid.size), chunk):
                q1 = min(int(atoms_grid.size), q0 + chunk)
                atom_t = torch.as_tensor(atoms_grid[q0:q1][None], dtype=torch.long, device=device)
                a_t = torch.as_tensor(a_grid[q0:q1][None], dtype=torch.long, device=device)
                b_t = torch.as_tensor(b_grid[q0:q1][None], dtype=torch.long, device=device)
                qa_t = torch.as_tensor(qa[q0:q1][None], dtype=torch.float32, device=device)
                qb_t = torch.as_tensor(qb[q0:q1][None], dtype=torch.float32, device=device)
                vals, variances = self.score_sparse_pairs(context, atom_t, a_t, b_t, qa_t, qb_t, return_uncertainty=True)
                vals_np = vals[0].detach().cpu().numpy().astype(np.float32)
                var_np = variances[0].detach().cpu().numpy().astype(np.float32)
                out[atoms_grid[q0:q1], pidx_grid[q0:q1]] = vals_np
                var_out[atoms_grid[q0:q1], pidx_grid[q0:q1]] = var_np
        return out, var_out

    def predict_certificate_numpy(self, runtime, candidates, evidence_bank, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = cfg or self.cfg
        batch = self._make_batch(runtime, candidates, evidence_bank, include_dense_query=False)
        self.eval()
        with torch.inference_mode():
            ctx = self.encode_context(batch)
        J0 = ctx["J0"][0].detach().cpu().numpy().astype(np.float32)
        proposal_logits = ctx["proposal_logits"][0].detach().cpu().numpy().astype(np.float32)
        family_logits = ctx["family_logits"][0].detach().cpu().numpy().astype(np.float32)
        family_pi = ctx["family_pi"][0].detach().cpu().numpy().astype(np.float32)
        flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
        pairs, pair_weights = build_runtime_pairs_from_base(
            J0,
            candidates.valid_mask,
            flags,
            L0=int(cfg.get("tournament", {}).get("L_infer", 16)),
            eta0=float(cfg.get("selector", {}).get("eta_pred", 1.0)),
            lambda_near=float(cfg.get("selector", {}).get("lambda_near", 1.0)),
            lambda_safety=float(cfg.get("selector", {}).get("lambda_safety", 2.0)),
            bidirectional_pairs=bool(cfg.get("selector", {}).get("bidirectional_pairs", True)),
            reverse_pair_weight=float(cfg.get("selector", {}).get("reverse_pair_weight", 1.0)),
            pair_cap_multiplier=float(cfg.get("selector", {}).get("runtime_pair_cap_multiplier", 1.0)),
            candidate_trajectories=candidates.trajectories,
            maneuver_ids=candidates.maneuver_ids,
            progress_pair_count=int(cfg.get("selector", {}).get("progress_pair_count", 8)),
            maneuver_pair_count=int(cfg.get("selector", {}).get("maneuver_pair_count", 8)),
        )
        budget = float(cfg.get("evidence", {}).get("budget", 16))
        M = int(cfg.get("selector", {}).get("proposal_top_m", max(2 * int(budget), int(budget) + 1)))
        active = np.asarray(evidence_bank.active_mask, dtype=bool)
        costs = np.asarray(evidence_bank.budget_costs(), dtype=np.float32)
        if "evidence_family_ids" in batch:
            family_ids = batch["evidence_family_ids"][0].detach().cpu().numpy().astype(np.int64)[: evidence_bank.E]
        else:
            family_ids = np.asarray([getattr(a, "family_id", 0) for a in evidence_bank.atoms], dtype=np.int64)
        topm, family_budget, hab_diag = select_topm_atoms_hab(
            proposal_logits[: evidence_bank.E],
            family_ids,
            active,
            costs,
            budget,
            M,
            family_scores=family_logits,
            free_budget=cfg.get("selector", {}).get("hab_free_budget", None),
            reserve_fraction=float(cfg.get("selector", {}).get("hab_reserve_fraction", 0.2)),
            enabled=bool(cfg.get("selector", {}).get("hab_enabled", True)),
            min_family_slots=cfg.get("selector", {}).get("min_family_topm_slots", None),
        )
        try:
            raw_hard_mask = np.asarray(evidence_bank.hard_mask(), dtype=bool)[: evidence_bank.E]
        except Exception:
            raw_hard_mask = np.zeros((evidence_bank.E,), dtype=bool)
        mandatory_hard_mask = structural_safety_mask(
            raw_hard_mask,
            family_ids,
            active,
            include_feasibility=bool(cfg.get("selector", {}).get("structural_safety_include_feasibility", True)),
        )
        if bool(cfg.get("selector", {}).get("force_hard_topm", True)):
            forced = np.flatnonzero(mandatory_hard_mask)
            if forced.size:
                forced_cap = int(cfg.get("selector", {}).get("max_forced_hard_topm", max(1, M // 2)))
                forced = np.asarray(sorted(forced.tolist(), key=lambda i: (-float(proposal_logits[int(i)]), int(i)))[:forced_cap], dtype=np.int64)
                forced_set = set(forced.tolist())
                non_forced = [int(i) for i in np.asarray(topm, dtype=np.int64).reshape(-1).tolist() if int(i) not in forced_set]
                topm = np.asarray((forced.tolist() + non_forced)[:M], dtype=np.int64)
                hab_diag = dict(hab_diag)
                hab_diag["forced_hard_topm"] = int(forced.size)
        rival_sets = build_rival_sets_from_base(
            J0,
            candidates.valid_mask,
            flags,
            L_infer=int(cfg.get("tournament", {}).get("L_infer", 16)),
            eta0=float(cfg.get("selector", {}).get("eta_pred", 1.0)),
            candidate_trajectories=candidates.trajectories,
            maneuver_ids=candidates.maneuver_ids,
            progress_rivals=int(cfg.get("selector", {}).get("progress_rivals", 4)),
            maneuver_rivals=int(cfg.get("selector", {}).get("maneuver_rivals", 4)),
        )
        action_set: set[int] = set()
        rival_pair_list: list[tuple[int, int]] = []
        for a_idx, rivals in enumerate(rival_sets):
            if not bool(candidates.valid_mask[a_idx]) or not rivals:
                continue
            action_set.add(int(a_idx))
            action_set.update(int(r) for r in rivals)
            for r in rivals:
                rival_pair_list.append((int(a_idx), int(r)))
        rival_pairs = np.asarray(rival_pair_list, dtype=np.int64).reshape(-1, 2) if rival_pair_list else np.zeros((0, 2), dtype=np.int64)
        if action_set:
            action_ids = np.asarray(sorted(action_set), dtype=np.int64)
        elif len(pairs):
            action_ids = np.unique(pairs.reshape(-1))
        else:
            action_ids = np.flatnonzero(candidates.valid_mask)[: max(1, int(cfg.get("tournament", {}).get("L_infer", 16)))]
        runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
        mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        use_pair_runtime = bool(runtime_cfg.get("use_pair_conditioned_margins", self.pair_conditioned))
        hybrid_local_weight = float(runtime_cfg.get("pair_delta_hybrid_local_weight", 0.0))
        need_action_sparse = (
            not use_pair_runtime
            or bool(mcfg.get("pair_head_residual_over_local", False))
            or hybrid_local_weight > 0.0
        )
        g_sparse = np.zeros((evidence_bank.E, candidates.K), dtype=np.float32)
        g_var_sparse = np.zeros((evidence_bank.E, candidates.K), dtype=np.float32)
        if need_action_sparse:
            atom_ids, action_ids_rep, q = compute_query_features_for_pairs(evidence_bank.atoms, candidates, runtime, topm, action_ids, cfg)
            if len(atom_ids):
                atom_t = torch.from_numpy(atom_ids[None].astype(np.int64)).to(next(self.parameters()).device)
                act_t = torch.from_numpy(action_ids_rep[None].astype(np.int64)).to(next(self.parameters()).device)
                q_t = torch.from_numpy(q[None].astype(np.float32)).to(next(self.parameters()).device)
                with torch.inference_mode():
                    vals, var = self.score_sparse_queries(ctx, atom_t, act_t, q_t, return_uncertainty=True)
                    vals_np = vals[0].detach().cpu().numpy().astype(np.float32)
                    var_np = var[0].detach().cpu().numpy().astype(np.float32)
                g_sparse[atom_ids, action_ids_rep] = vals_np
                g_var_sparse[atom_ids, action_ids_rep] = var_np
        else:
            atom_ids = np.zeros((0,), dtype=np.int64)
            action_ids_rep = np.zeros((0,), dtype=np.int64)

        # Score selector pairs and tournament-rival pairs in one GPU call over the
        # unique pair union. The previous runtime called _score_pair_indices_numpy
        # twice, recomputing query features and launching the pair MLP twice.
        if len(pairs) or len(rival_pairs):
            all_pairs = np.concatenate([pairs.reshape(-1, 2), rival_pairs.reshape(-1, 2)], axis=0)
            unique_pairs, inverse = np.unique(all_pairs, axis=0, return_inverse=True)
            all_pair_delta, all_pair_var = self._score_pair_indices_numpy(ctx, runtime, candidates, evidence_bank, topm, unique_pairs, cfg)
            n_selector_pairs = int(len(pairs))
            selector_inv = inverse[:n_selector_pairs]
            rival_inv = inverse[n_selector_pairs:]
            selector_pair_delta = all_pair_delta[:, selector_inv] if n_selector_pairs else np.zeros((evidence_bank.E, 0), dtype=np.float32)
            selector_pair_var = all_pair_var[:, selector_inv] if n_selector_pairs else np.zeros((evidence_bank.E, 0), dtype=np.float32)
            rival_pair_delta = all_pair_delta[:, rival_inv] if len(rival_pairs) else np.zeros((evidence_bank.E, 0), dtype=np.float32)
            rival_pair_var = all_pair_var[:, rival_inv] if len(rival_pairs) else np.zeros((evidence_bank.E, 0), dtype=np.float32)
        else:
            selector_pair_delta = np.zeros((evidence_bank.E, 0), dtype=np.float32)
            selector_pair_var = np.zeros((evidence_bank.E, 0), dtype=np.float32)
            rival_pair_delta = np.zeros((evidence_bank.E, 0), dtype=np.float32)
            rival_pair_var = np.zeros((evidence_bank.E, 0), dtype=np.float32)
        normalize_pairs = bool(cfg.get("model", {}).get("pair_margin_normalized", self.pair_margin_normalized))
        tcfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
        min_scale = float(mcfg.get("margin_normalization_min_scale", tcfg.get("pair_margin_min_scale", 100.0)))
        q_scale = float(mcfg.get("margin_normalization_quantile", 0.75))
        if normalize_pairs and len(pairs):
            pair_margin_scale = margin_normalization_scale(J0[pairs[:, 1]] - J0[pairs[:, 0]], min_scale=min_scale, quantile=q_scale)
        else:
            pair_margin_scale = 1.0
        if normalize_pairs and len(rival_pairs):
            rival_pair_margin_scale = margin_normalization_scale(J0[rival_pairs[:, 1]] - J0[rival_pairs[:, 0]], min_scale=min_scale, quantile=q_scale)
        else:
            rival_pair_margin_scale = pair_margin_scale

        def _local_pair_delta(pair_arr: np.ndarray, scale: float) -> np.ndarray:
            pair_arr = np.asarray(pair_arr, dtype=np.int64).reshape(-1, 2) if np.asarray(pair_arr).size else np.zeros((0, 2), dtype=np.int64)
            if pair_arr.size == 0:
                return np.zeros((evidence_bank.E, 0), dtype=np.float32)
            a = np.clip(pair_arr[:, 0], 0, candidates.K - 1)
            b = np.clip(pair_arr[:, 1], 0, candidates.K - 1)
            local = g_sparse[:, b] - g_sparse[:, a]
            return (local / max(float(scale), 1e-6)).astype(np.float32) if normalize_pairs else local.astype(np.float32)

        if bool(mcfg.get("pair_head_residual_over_local", False)):
            selector_pair_delta = _local_pair_delta(pairs, pair_margin_scale) + selector_pair_delta
            rival_pair_delta = _local_pair_delta(rival_pairs, rival_pair_margin_scale) + rival_pair_delta
        else:
            w_local = float(cfg.get("runtime", {}).get("pair_delta_hybrid_local_weight", 0.0)) if isinstance(cfg, dict) else 0.0
            if w_local > 0.0:
                w_local = min(max(w_local, 0.0), 1.0)
                selector_pair_delta = (1.0 - w_local) * selector_pair_delta + w_local * _local_pair_delta(pairs, pair_margin_scale)
                rival_pair_delta = (1.0 - w_local) * rival_pair_delta + w_local * _local_pair_delta(rival_pairs, rival_pair_margin_scale)
        valid_mask = np.asarray(candidates.valid_mask, dtype=bool)
        g_sparse[:, ~valid_mask] = 0.0
        g_var_sparse[:, ~valid_mask] = 0.0
        return {
            "J0": J0,
            "g": g_sparse,
            "g_var": g_var_sparse,
            "proposal_logits": proposal_logits,
            "family_logits": family_logits,
            "family_pi": family_pi,
            "family_ids": family_ids,
            "family_budget_caps": family_budget.family_caps,
            "family_budgets": family_budget.family_budgets,
            "mandatory_atom_mask": mandatory_hard_mask.astype(bool),
            "mandatory_hard_atoms": np.flatnonzero(mandatory_hard_mask).astype(np.int64),
            "hab_diagnostics": hab_diag,
            "top_m_atoms": topm,
            "queried_actions": np.asarray(action_ids, dtype=np.int64),
            # Query accounting uses explicit categories.  Keep queried_pair_count
            # as a backward-compatible alias for the total number of sparse model
            # scores actually evaluated in this runtime certificate stage.
            "action_atom_query_count": int(len(atom_ids)),
            "selector_pair_atom_query_count": int(len(topm) * len(pairs)),
            "tournament_pair_atom_query_count": int(len(topm) * len(rival_pairs)),
            "runtime_pair_count": int(len(pairs)),
            "tournament_pair_count": int(len(rival_pairs)),
            "queried_pair_count": int(len(atom_ids) + len(topm) * len(pairs) + len(topm) * len(rival_pairs)),
            "pair_atom_delta": selector_pair_delta,
            "pair_atom_var": selector_pair_var,
            "pair_indices": pairs,
            "pair_margin_scale": float(pair_margin_scale),
            "rival_pair_margin_scale": float(rival_pair_margin_scale),
            "pair_margin_normalized": bool(normalize_pairs),
            "rival_pair_atom_delta": rival_pair_delta,
            "rival_pair_atom_var": rival_pair_var,
            "rival_pair_indices": rival_pairs,
            "runtime_pairs": pairs,
            "runtime_pair_weights": pair_weights,
        }

    def predict_dense_numpy(self, runtime, candidates, evidence_bank, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        """Diagnostic-only dense BDSE interface.

        Runtime planning must not call this path.  It scores every active atom/action
        pair so open-loop diagnostics can separately measure dense full-interface
        reconstruction versus the sparse selected certificate.
        """
        cfg = cfg or self.cfg
        batch = self._make_batch(runtime, candidates, evidence_bank, include_dense_query=True)
        self.eval()
        with torch.no_grad():
            out = self.forward(batch)
        J0 = out["J0"][0].detach().cpu().numpy().astype(np.float32)
        g = out["g"][0].detach().cpu().numpy().astype(np.float32)
        g_var = out.get("g_var")
        g_var_np = g_var[0].detach().cpu().numpy().astype(np.float32) if g_var is not None else np.zeros_like(g, dtype=np.float32)
        valid = np.asarray(candidates.valid_mask, dtype=bool)
        active = np.asarray(evidence_bank.active_mask, dtype=bool)
        if active.shape[0] < g.shape[0]:
            active = np.pad(active, (0, g.shape[0] - active.shape[0]), constant_values=False)
        active = active[: g.shape[0]]
        g[~active, :] = 0.0
        g_var_np[~active, :] = 0.0
        g[:, ~valid] = 0.0
        g_var_np[:, ~valid] = 0.0
        return {"J0": J0, "g": g, "g_var": g_var_np, "dense_atom_count": int(active.sum()), "dense_action_count": int(valid.sum())}

    def predict_numpy(self, runtime, candidates, evidence_bank):
        # Legacy API: return only base and sparse-scored local costs.  The runtime
        # planner uses predict_certificate_numpy to also obtain proposal/HAB diagnostics.
        out = self.predict_certificate_numpy(runtime, candidates, evidence_bank, self.cfg)
        return out["J0"], out["g"]
