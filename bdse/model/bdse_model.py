from __future__ import annotations

from typing import Any

from contextlib import contextmanager
import os
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bdse.data.tensorizer import runtime_to_model_numpy
from bdse.model.action_encoder import ActionEncoder
from bdse.model.evidence_encoder import EvidenceEncoder
from bdse.model.residual_gate import confidence_shrunk_residual_pair_delta_numpy
from bdse.model.scene_encoder import SceneEncoder
from bdse.planner.evidence_queries import FAMILY_NAMES, TYPE_NAMES, compute_query_features_for_pairs
from bdse.planner.fallback import runtime_risk_scores, runtime_safety_flags_from_runtime
from bdse.planner.hab import max_family_id, select_topm_atoms_hab
from bdse.planner.pair_screen import build_runtime_pairs_from_base, build_rival_sets_from_base, compact_runtime_pair_graph, restrict_pairs_to_viability_frontier, reweight_pairs_by_viability_scope
from bdse.planner.selector import margin_normalization_scale, reserve_topm_candidates, restrict_topm_to_decision_evidence, structural_safety_mask



_RUNTIME_PREDICTION_CACHE = threading.local()


class _RuntimePredictionMemo:
    def __init__(self) -> None:
        self.cache: dict[tuple[int, int, int, int], tuple[Any, Any, Any, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.depth = 0

EVIDENCE_TYPE_TO_ID = TYPE_NAMES
FAMILY_TO_ID = FAMILY_NAMES


def _confidence_shrunk_residual_pair_delta_np(
    local: np.ndarray,
    residual: np.ndarray,
    variance: np.ndarray,
    cfg: dict[str, Any],
    base_margin: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Foundation-margin-aware wrapper around the shared FAR residual gate."""
    return confidence_shrunk_residual_pair_delta_numpy(local, residual, variance, cfg, base_margin=base_margin)

def _robust_rank_cost_np(cost: np.ndarray, valid_mask: np.ndarray, *, clip: float = 4.0) -> np.ndarray:
    """Return a robust dimensionless cost where lower remains better.

    The runtime base prior is intentionally rank/scale normalized: the BDSE pair
    head is trained in normalized margin units, so injecting raw handcrafted
    costs would either be ignored or dominate depending on arbitrary units.
    Robust normalization makes the base prior a decision prior, not a hidden
    full-cost oracle.
    """
    c = np.asarray(cost, dtype=np.float32).reshape(-1).copy()
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.shape[0] < c.shape[0]:
        valid = np.pad(valid, (0, c.shape[0] - valid.shape[0]), constant_values=False)
    valid = valid[: c.shape[0]] & np.isfinite(c)
    out = np.zeros_like(c, dtype=np.float32)
    if int(valid.sum()) <= 1:
        out[~valid] = np.inf
        return out
    vals = c[valid].astype(np.float32)
    med = float(np.median(vals))
    q25, q75 = np.percentile(vals, [25.0, 75.0])
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(vals - med)))
    scale = max(iqr / 1.349, 1.4826 * mad, 1e-3)
    out[valid] = ((c[valid] - med) / scale).astype(np.float32)
    if clip and clip > 0:
        out[valid] = np.clip(out[valid], -float(clip), float(clip)).astype(np.float32)
    out[~valid] = np.inf
    return out


def _trajectory_decision_prior_cost_np(candidates: Any, runtime_flags: np.ndarray | None, cfg: dict[str, Any]) -> np.ndarray:
    """Cheap route-progress/comfort prior over the existing candidate bank.

    This is a base-action prior J_prior(a), not an evidence query.  It uses only
    candidate geometry and runtime hard flags, so it preserves the fixed evidence
    budget while giving BDSE a stronger prior for progress/route tracking.  The
    learned evidence selector still decides which budgeted atoms can overturn or
    certify this prior.
    """
    traj = np.asarray(getattr(candidates, "trajectories", np.zeros((0, 1, 3), dtype=np.float32)), dtype=np.float32)
    valid = np.asarray(getattr(candidates, "valid_mask", np.ones((traj.shape[0],), dtype=bool)), dtype=bool).reshape(-1)
    K = int(traj.shape[0]) if traj.ndim >= 3 else int(valid.shape[0])
    if K <= 0:
        return np.zeros((0,), dtype=np.float32)
    if traj.ndim < 3 or traj.shape[0] != K or traj.shape[1] < 1:
        out = np.zeros((K,), dtype=np.float32); out[~valid[:K]] = np.inf; return out
    xy = traj[..., :2]
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    if xy.shape[1] > 1:
        dxy = np.diff(xy, axis=1)
        step = np.linalg.norm(dxy, axis=-1)
    else:
        step = np.zeros((K, 1), dtype=np.float32)
    progress = xy[:, -1, 0]
    lateral = xy[..., 1]
    lateral_mean = np.mean(np.abs(lateral), axis=1)
    lateral_final = np.abs(lateral[:, -1])
    path_len = step.sum(axis=1)
    speed = step / max(dt, 1e-3)
    speed_mean = speed.mean(axis=1) if speed.size else np.zeros((K,), dtype=np.float32)
    speed_final = speed[:, -1] if speed.ndim == 2 and speed.shape[1] else np.zeros((K,), dtype=np.float32)
    acc = np.diff(speed, axis=1) if speed.ndim == 2 and speed.shape[1] > 1 else np.zeros((K, 1), dtype=np.float32)
    jerk = np.diff(acc, axis=1) if acc.ndim == 2 and acc.shape[1] > 1 else np.zeros((K, 1), dtype=np.float32)
    acc_rms = np.sqrt(np.mean(acc * acc, axis=1)) if acc.size else np.zeros((K,), dtype=np.float32)
    jerk_rms = np.sqrt(np.mean(jerk * jerk, axis=1)) if jerk.size else np.zeros((K,), dtype=np.float32)
    comfort = 0.25 * acc_rms + 0.10 * jerk_rms
    yaw = traj[..., 2] if traj.shape[-1] > 2 else np.zeros((K, traj.shape[1]), dtype=np.float32)
    yaw_delta = np.arctan2(np.sin(np.diff(yaw, axis=1)), np.cos(np.diff(yaw, axis=1))) if yaw.shape[1] > 1 else np.zeros((K, 1), dtype=np.float32)
    curvature = np.mean(np.abs(yaw_delta), axis=1) if yaw_delta.size else np.zeros((K,), dtype=np.float32)
    pcfg = ((cfg.get("runtime", {}) or {}).get("base_prior", {}) or {}) if isinstance(cfg, dict) else {}
    cost = (
        float(pcfg.get("lateral_mean_weight", 2.0)) * lateral_mean
        + float(pcfg.get("lateral_final_weight", 0.75)) * lateral_final
        + float(pcfg.get("comfort_weight", 0.5)) * comfort
        + float(pcfg.get("curvature_weight", 0.25)) * curvature
        - float(pcfg.get("progress_weight", 0.05)) * progress
        - float(pcfg.get("path_length_weight", 0.01)) * path_len
    )
    cost += np.where(speed_mean < float(pcfg.get("low_speed_threshold", 0.3)), float(pcfg.get("low_speed_penalty", 0.15)), 0.0)
    cost += np.where(speed_final < float(pcfg.get("low_final_speed_threshold", -1.0)), float(pcfg.get("low_final_speed_penalty", 0.0)), 0.0)
    if runtime_flags is not None:
        flags = np.asarray(runtime_flags, dtype=bool).reshape(-1)
        if flags.shape[0] < K:
            flags = np.pad(flags, (0, K - flags.shape[0]), constant_values=False)
        cost += flags[:K].astype(np.float32) * float(pcfg.get("unsafe_penalty", 1000.0))
    if valid.shape[0] < K:
        valid = np.pad(valid, (0, K - valid.shape[0]), constant_values=False)
    cost = np.asarray(cost, dtype=np.float32)
    cost[~valid[:K]] = np.inf
    return cost


def _apply_runtime_base_prior_np(J0: np.ndarray, candidates: Any, runtime_flags: np.ndarray | None, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
    pcfg = runtime_cfg.get("base_prior", {}) or {}
    if not bool(pcfg.get("enabled", False)):
        return np.asarray(J0, dtype=np.float32), {"base_prior_enabled": False}
    valid = np.asarray(getattr(candidates, "valid_mask", np.isfinite(J0)), dtype=bool).reshape(-1)
    J0_in = np.asarray(J0, dtype=np.float32).reshape(-1)
    if valid.shape[0] < J0_in.shape[0]:
        valid = np.pad(valid, (0, J0_in.shape[0] - valid.shape[0]), constant_values=False)
    valid = valid[: J0_in.shape[0]]
    prior_raw = _trajectory_decision_prior_cost_np(candidates, runtime_flags, cfg)
    if prior_raw.shape[0] < J0_in.shape[0]:
        prior_raw = np.pad(prior_raw, (0, J0_in.shape[0] - prior_raw.shape[0]), constant_values=np.inf)
    prior_raw = prior_raw[: J0_in.shape[0]]
    clip = float(pcfg.get("z_clip", 4.0))
    learned_z = _robust_rank_cost_np(J0_in, valid, clip=clip)
    prior_z = _robust_rank_cost_np(prior_raw, valid, clip=clip)
    w = min(max(float(pcfg.get("weight", 0.5)), 0.0), 1.0)
    mode = str(pcfg.get("mode", "blend")).lower()
    if mode == "prior_only":
        blended_z = prior_z
    elif mode == "learned_only":
        blended_z = learned_z
    else:
        blended_z = (1.0 - w) * learned_z + w * prior_z
    mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    tcfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    min_scale = float(mcfg.get("margin_normalization_min_scale", tcfg.get("pair_margin_min_scale", 20000.0)))
    scale = max(float(pcfg.get("scale", min_scale)), 1e-6) * float(pcfg.get("scale_multiplier", 1.0))
    J0_out = (blended_z * scale).astype(np.float32)
    J0_out[~valid] = np.inf
    diag = {
        "base_prior_enabled": True,
        "base_prior_mode": mode,
        "base_prior_weight": float(w),
        "base_prior_scale": float(scale),
        "base_prior_best_action": int(np.nanargmin(np.where(valid & np.isfinite(prior_raw), prior_raw, np.inf))) if bool((valid & np.isfinite(prior_raw)).any()) else -1,
        "learned_base_best_action": int(np.nanargmin(np.where(valid & np.isfinite(J0_in), J0_in, np.inf))) if bool((valid & np.isfinite(J0_in)).any()) else -1,
        "base_prior_replaced_best": bool(np.nanargmin(np.where(valid & np.isfinite(prior_raw), prior_raw, np.inf)) != np.nanargmin(np.where(valid & np.isfinite(J0_in), J0_in, np.inf))) if bool((valid & np.isfinite(prior_raw) & np.isfinite(J0_in)).any()) else False,
    }
    return J0_out, diag


def _apply_structural_safety_residual_prior_np(
    J0: np.ndarray,
    runtime: Any,
    candidates: Any,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend a graded runtime-safety residual into the base decision prior.

    Hard constraints remain lexicographic and budget-exempt.  Their *continuous*
    near-boundary information (agent clearance, TTC and route-boundary distance)
    still affects ranking among feasible actions, however.  V36 discarded that
    graded component together with the hard atoms and therefore changed the
    teacher-margin decomposition.  This helper compresses the complete structural
    channel into one dimensionless action prior without spending evidence budget
    or querying teacher futures.
    """
    rcfg = (((cfg.get("runtime", {}) or {}).get("structural_safety_residual", {}) or {})
            if isinstance(cfg, dict) else {})
    if not bool(rcfg.get("enabled", False)):
        return np.asarray(J0, dtype=np.float32), {"structural_residual_enabled": False}

    base = np.asarray(J0, dtype=np.float32).reshape(-1)
    valid = np.asarray(getattr(candidates, "valid_mask", np.isfinite(base)), dtype=bool).reshape(-1)
    if valid.shape[0] < base.shape[0]:
        valid = np.pad(valid, (0, base.shape[0] - valid.shape[0]), constant_values=False)
    valid = valid[: base.shape[0]] & np.isfinite(base)
    risks = runtime_risk_scores(runtime, candidates, cfg)

    components = {
        "hard_agent": float(rcfg.get("hard_agent_weight", 0.28)),
        "agent_ttc": float(rcfg.get("ttc_weight", 0.30)),
        "hard_off_route": float(rcfg.get("hard_offroute_weight", 0.18)),
        "soft_agent": float(rcfg.get("soft_agent_weight", 0.12)),
        "soft_off_route": float(rcfg.get("soft_offroute_weight", 0.08)),
        "red_light": float(rcfg.get("red_light_weight", 0.40)),
    }
    combined = np.zeros_like(base, dtype=np.float32)
    used = 0
    for key, weight in components.items():
        if abs(weight) <= 0.0:
            continue
        raw = np.asarray(risks.get(key, np.zeros_like(base)), dtype=np.float32).reshape(-1)
        if raw.shape[0] < base.shape[0]:
            raw = np.pad(raw, (0, base.shape[0] - raw.shape[0]), constant_values=np.inf)
        z = _robust_rank_cost_np(raw[: base.shape[0]], valid, clip=float(rcfg.get("component_z_clip", 3.0)))
        finite = valid & np.isfinite(z)
        combined[finite] += float(weight) * z[finite]
        used += 1
    if used == 0 or not bool(valid.any()):
        return base, {"structural_residual_enabled": True, "structural_residual_component_count": 0}

    combined = _robust_rank_cost_np(combined, valid, clip=float(rcfg.get("combined_z_clip", 3.0)))
    mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    tcfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    scale = float(mcfg.get("margin_normalization_min_scale", tcfg.get("pair_margin_min_scale", 20000.0)))
    scale *= float(rcfg.get("scale_multiplier", 1.0))
    weight = max(0.0, float(rcfg.get("weight", 0.25)))
    out = base.copy()
    out[valid] = out[valid] + weight * scale * combined[valid]
    out[~valid] = np.inf
    best = int(np.argmin(np.where(valid, combined, np.inf))) if bool(valid.any()) else -1
    return out.astype(np.float32), {
        "structural_residual_enabled": True,
        "structural_residual_component_count": int(used),
        "structural_residual_weight": float(weight),
        "structural_residual_scale": float(scale),
        "structural_residual_best_action": int(best),
    }


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

        # V56 DCIP: evidence-attributable, globally integrable residual action
        # potential.  Each queried atom contributes a signed correction h_i(a)
        # to an action cost; pair corrections are derived as h_i(b)-h_i(a), so
        # cycle consistency is exact by construction rather than repaired after
        # an arbitrary pair field has already been learned.
        self.residual_action_head = nn.Sequential(
            nn.LayerNorm(h * 4), nn.Linear(h * 4, h), nn.ReLU(), nn.Linear(h, 1)
        )
        self.residual_action_var_head = nn.Sequential(
            nn.LayerNorm(h * 4), nn.Linear(h * 4, h), nn.ReLU(), nn.Linear(h, 1)
        )
        if bool(mcfg.get("zero_init_residual_action_head", True)):
            nn.init.zeros_(self.residual_action_head[-1].weight)
            nn.init.zeros_(self.residual_action_head[-1].bias)
            # Start from a moderate, finite uncertainty.  softplus(-2) ~= 0.127.
            nn.init.zeros_(self.residual_action_var_head[-1].weight)
            nn.init.constant_(self.residual_action_var_head[-1].bias, -2.0)

        # V59 FSCIP: a low-rank selected-set interaction potential.  The model
        # predicts one factor per evidence atom and one factor per action before
        # selection.  After the fixed-budget set is known, their DeepSets-style
        # pooled interaction yields a global action potential without a second
        # neural forward.  It is integrable by construction and captures evidence
        # interactions that an independent atom sum cannot represent.
        self.set_residual_rank = max(0, int(mcfg.get("set_residual_rank", 0)))
        if self.set_residual_rank > 0:
            self.residual_set_atom_head = nn.Sequential(
                nn.LayerNorm(h * 2), nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, self.set_residual_rank)
            )
            self.residual_set_action_head = nn.Sequential(
                nn.LayerNorm(h * 2), nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, self.set_residual_rank)
            )
            if bool(mcfg.get("zero_init_set_residual_head", True)):
                nn.init.zeros_(self.residual_set_atom_head[-1].weight)
                nn.init.zeros_(self.residual_set_atom_head[-1].bias)
                nn.init.normal_(self.residual_set_action_head[-1].weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.residual_set_action_head[-1].bias)
        else:
            self.residual_set_atom_head = None
            self.residual_set_action_head = None

        if bool(cfg.get("training", {}).get("freeze_unused_pair_variance_head", False)):
            for param in self.pair_var_head.parameters():
                param.requires_grad_(False)

        # Hierarchical Atom Builder (HAB): family gate pi_tau followed by an
        # atom proposal conditioned on family embedding and candidate-set summary.
        self.family_embed = nn.Embedding(self.num_families, h)
        self.family_activity_proj = nn.Sequential(nn.Linear(8, h), nn.ReLU(), nn.Linear(h, h))
        self.family_head = nn.Sequential(nn.LayerNorm(h * 4), nn.Linear(h * 4, h), nn.ReLU(), nn.Linear(h, 1))
        self.proposal_head = nn.Sequential(nn.LayerNorm(h * 5), nn.Linear(h * 5, h), nn.ReLU(), nn.Linear(h, 1))
        # Backward-compatible name used by old checkpoints/tests.
        self.selector_head = self.proposal_head

        # v24 training uses the pair-conditioned uncertainty head for L_unc and
        # sets action_conditioned_action_loss_weight=0, so the action-conditioned
        # local variance head can be safely frozen/skipped to avoid DDP unused-
        # parameter graph traversal and a dense E x K variance forward.
        if bool(cfg.get("training", {}).get("freeze_unused_local_variance_head", False)):
            for param in self.local_var_head.parameters():
                param.requires_grad_(False)

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
            "action_valid": valid,
            "action_set_summary": u_A,
        }

    def set_residual_factors(self, context: dict[str, torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.set_residual_rank <= 0 or self.residual_set_atom_head is None or self.residual_set_action_head is None:
            return None, None
        scene = context["scene"]
        evidence_h = context["evidence_h"]
        action_h = context["action_h"]
        scene_e = scene[:, None, :].expand(-1, evidence_h.shape[1], -1)
        scene_a = scene[:, None, :].expand(-1, action_h.shape[1], -1)
        atom = self.residual_set_atom_head(torch.cat([evidence_h, scene_e], dim=-1))
        action = self.residual_set_action_head(torch.cat([action_h, scene_a], dim=-1))
        atom = atom.masked_fill(~context["evidence_valid"][:, :, None], 0.0)
        action = action.masked_fill(~context["action_valid"][:, :, None], 0.0)
        return atom, action

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
        # Runtime closed-loop inference calls this for small but frequent
        # atom-pair batches.  Build the AB and BA tensors from shared gathers and
        # shared query projections instead of calling _sparse_pair_features twice;
        # this is algebraically identical but removes two query_proj forwards and
        # redundant gather work per planner tick.
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

        z_ab = torch.cat([a_h, b_h, e_h, q_a, q_b, s_h, b_h - a_h, a_h * b_h, q_b - q_a, q_a * q_b], dim=-1)
        z_ba = torch.cat([b_h, a_h, e_h, q_b, q_a, s_h, a_h - b_h, a_h * b_h, q_a - q_b, q_a * q_b], dim=-1)
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
        return_residual_action: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # atom/action indices: [B,Q], query_features: [B,Q,Dq]
        z = self._sparse_query_features(context, atom_indices, action_indices, query_features)
        mean = self.local_head(z).squeeze(-1)
        if not return_uncertainty and not return_residual_action:
            return mean
        var = self._positive_variance(self.local_var_head(z).squeeze(-1), self.var_floor)
        if not return_residual_action:
            return mean, var
        residual = self.residual_action_head(z).squeeze(-1)
        residual_var = self._positive_variance(
            self.residual_action_var_head(z).squeeze(-1), self.var_floor
        )
        return mean, var, residual, residual_var

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

    def _need_dense_local_variance(self) -> bool:
        train_cfg = self.cfg.get("training", {}) if isinstance(self.cfg, dict) else {}
        value = train_cfg.get("compute_local_variance", True)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"0", "false", "no", "off", "skip"}:
                return False
            if text in {"1", "true", "yes", "on", "always"}:
                return True
            if text == "auto":
                lw = train_cfg.get("loss_weights", {}) if isinstance(train_cfg.get("loss_weights", {}), dict) else {}
                action_cond = float(train_cfg.get("action_conditioned_action_loss_weight", 0.0)) > 0.0
                beta_unc = float(self.cfg.get("tournament", {}).get("beta_uncertainty", 0.0))
                unc_weight = float(lw.get("uncertainty", 0.0))
                # g_var is only consumed by the action-conditioned path or by L_unc
                # when no pair-conditioned variance is produced.
                return (action_cond and beta_unc > 0.0) or ((not self.pair_conditioned) and unc_weight > 0.0)
        return bool(value)

    def _need_pair_uncertainty(self) -> bool:
        train_cfg = self.cfg.get("training", {}) if isinstance(self.cfg, dict) else {}
        value = train_cfg.get("compute_pair_uncertainty", "auto")
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"0", "false", "no", "off", "skip"}:
                return False
            if text in {"1", "true", "yes", "on", "always"}:
                return True
        lw = train_cfg.get("loss_weights", {}) if isinstance(train_cfg.get("loss_weights", {}), dict) else {}
        return float(lw.get("uncertainty", 0.0)) > 0.0 or float(self.cfg.get("tournament", {}).get("beta_uncertainty", 0.0)) > 0.0

    def _dense_query_projection(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q_raw_all = self._fit_last_dim(batch["evidence_query_features"].float(), self.query_proj[0].in_features)
        B, E, K, _ = q_raw_all.shape
        chunk_e = min(E, self._training_chunk_size("query_projection_atom_chunk", self._training_chunk_size("local_forward_atom_chunk", 32)))
        parts: list[torch.Tensor] = []
        for e0 in range(0, E, chunk_e):
            e1 = min(E, e0 + chunk_e)
            parts.append(self.query_proj(q_raw_all[:, e0:e1]))
        if not parts:
            return torch.zeros((B, E, K, self.hidden_dim), dtype=batch["evidence_query_features"].dtype, device=batch["evidence_query_features"].device)
        return torch.cat(parts, dim=1)

    def _dense_local_from_batch(
        self,
        context: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        q_h_all: torch.Tensor | None = None,
        compute_variance: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_h = context["action_h"]
        evid_h = context["evidence_h"]
        scene = context["scene"]
        J0 = context["J0"]
        valid = batch["candidate_valid"].bool()
        e_valid = context["evidence_valid"]
        q_raw_all = None if q_h_all is not None else self._fit_last_dim(batch["evidence_query_features"].float(), self.query_proj[0].in_features)
        B, K, H = action_h.shape
        E = evid_h.shape[1]
        chunk_e = min(E, self._training_chunk_size("local_forward_atom_chunk", 32))
        local_parts: list[torch.Tensor] = []
        local_var_parts: list[torch.Tensor] = []
        residual_parts: list[torch.Tensor] = []
        residual_var_parts: list[torch.Tensor] = []
        use_residual_action = bool((self.cfg.get("model", {}) or {}).get("evidence_action_residual", False))
        for e0 in range(0, E, chunk_e):
            e1 = min(E, e0 + chunk_e)
            Ce = e1 - e0
            q_h = q_h_all[:, e0:e1] if q_h_all is not None else self.query_proj(q_raw_all[:, e0:e1])
            a_exp = action_h[:, None, :, :].expand(B, Ce, K, H)
            e_exp = evid_h[:, e0:e1, None, :].expand(B, Ce, K, H)
            s_exp = scene[:, None, None, :].expand(B, Ce, K, H)
            z = torch.cat([a_exp, e_exp, q_h, s_exp], dim=-1)
            local_chunk = self.local_head(z).squeeze(-1)
            local_chunk = local_chunk.masked_fill(~valid[:, None, :], 0.0).masked_fill(~e_valid[:, e0:e1, None], 0.0)
            local_parts.append(local_chunk)
            if compute_variance:
                local_var_chunk = self._positive_variance(self.local_var_head(z).squeeze(-1), self.var_floor)
                local_var_chunk = local_var_chunk.masked_fill(~valid[:, None, :], 0.0).masked_fill(~e_valid[:, e0:e1, None], 0.0)
                local_var_parts.append(local_var_chunk)
            if use_residual_action:
                residual_chunk = self.residual_action_head(z).squeeze(-1)
                residual_chunk = residual_chunk.masked_fill(~valid[:, None, :], 0.0).masked_fill(~e_valid[:, e0:e1, None], 0.0)
                residual_var_chunk = self._positive_variance(
                    self.residual_action_var_head(z).squeeze(-1), self.var_floor
                )
                residual_var_chunk = residual_var_chunk.masked_fill(~valid[:, None, :], 0.0).masked_fill(~e_valid[:, e0:e1, None], 0.0)
                residual_parts.append(residual_chunk)
                residual_var_parts.append(residual_var_chunk)
        if not local_parts:
            zeros = torch.zeros((B, E, K), dtype=J0.dtype, device=J0.device)
            return zeros, zeros, zeros, zeros
        local = torch.cat(local_parts, dim=1)
        if compute_variance and local_var_parts:
            local_var = torch.cat(local_var_parts, dim=1)
        else:
            local_var = torch.zeros((B, E, K), dtype=local.dtype, device=local.device)
        residual = torch.cat(residual_parts, dim=1) if residual_parts else torch.zeros_like(local)
        residual_var = torch.cat(residual_var_parts, dim=1) if residual_var_parts else torch.zeros_like(local)
        return local, local_var, residual, residual_var

    def _dense_pair_delta_from_batch(
        self,
        context: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        q_h_all: torch.Tensor | None = None,
        return_uncertainty: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
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
        if q_h_all is None:
            q_h_all = self._dense_query_projection(batch)
        # Pair action embeddings are independent of atoms; gather once and reuse
        # across all atom chunks instead of repeating this work E/chunk_e times.
        a_pair_h_all = torch.gather(action_h, 1, a_idx_all[..., None].expand(B, P, H))
        b_pair_h_all = torch.gather(action_h, 1, b_idx_all[..., None].expand(B, P, H))
        chunk_e = min(E, self._training_chunk_size("pair_forward_atom_chunk", 16))
        chunk_p = min(P, self._training_chunk_size("pair_forward_pair_chunk", 32))
        delta_chunks: list[torch.Tensor] = []
        var_chunks: list[torch.Tensor] = []
        pair_valid_all = batch.get("pair_valid")
        e_valid_all = context.get("evidence_valid")
        for e0 in range(0, E, chunk_e):
            e1 = min(E, e0 + chunk_e)
            Ce = e1 - e0
            delta_p_chunks: list[torch.Tensor] = []
            var_p_chunks: list[torch.Tensor] = []
            q_h_e = q_h_all[:, e0:e1]
            e_h = evid_h[:, e0:e1, None, :].expand(B, Ce, 1, H)
            for p0 in range(0, P, chunk_p):
                p1 = min(P, p0 + chunk_p)
                Cp = p1 - p0
                a_idx = a_idx_all[:, p0:p1]
                b_idx = b_idx_all[:, p0:p1]
                a_h = a_pair_h_all[:, p0:p1, :][:, None, :, :].expand(B, Ce, Cp, H)
                b_h = b_pair_h_all[:, p0:p1, :][:, None, :, :].expand(B, Ce, Cp, H)
                idx_a_q = a_idx[:, None, :, None].expand(B, Ce, Cp, H)
                idx_b_q = b_idx[:, None, :, None].expand(B, Ce, Cp, H)
                q_a = torch.gather(q_h_e, 2, idx_a_q)
                q_b = torch.gather(q_h_e, 2, idx_b_q)
                e_hp = e_h.expand(B, Ce, Cp, H)
                s_h = scene[:, None, None, :].expand(B, Ce, Cp, H)
                z_ab = torch.cat([a_h, b_h, e_hp, q_a, q_b, s_h, b_h - a_h, a_h * b_h, q_b - q_a, q_a * q_b], dim=-1)
                z_ba = torch.cat([b_h, a_h, e_hp, q_b, q_a, s_h, a_h - b_h, a_h * b_h, q_a - q_b, q_a * q_b], dim=-1)
                delta = (self.pair_head(z_ab) - self.pair_head(z_ba)).squeeze(-1) * float(self.pair_delta_scale)
                var = None
                if return_uncertainty:
                    var = self._positive_variance(self.pair_var_head(z_ab).squeeze(-1), self.var_floor)
                    var = var + self._positive_variance(self.pair_var_head(z_ba).squeeze(-1), self.var_floor)
                if pair_valid_all is not None:
                    p_mask = pair_valid_all[:, p0:p1].bool()
                    delta = delta.masked_fill(~p_mask[:, None, :], 0.0)
                    if var is not None:
                        var = var.masked_fill(~p_mask[:, None, :], 0.0)
                if e_valid_all is not None:
                    e_mask = e_valid_all[:, e0:e1].bool()
                    delta = delta.masked_fill(~e_mask[:, :, None], 0.0)
                    if var is not None:
                        var = var.masked_fill(~e_mask[:, :, None], 0.0)
                delta_p_chunks.append(delta)
                if var is not None:
                    var_p_chunks.append(var)
            delta_chunks.append(torch.cat(delta_p_chunks, dim=2))
            if return_uncertainty and var_p_chunks:
                var_chunks.append(torch.cat(var_p_chunks, dim=2))
        delta_all = torch.cat(delta_chunks, dim=1)
        if return_uncertainty and var_chunks:
            return delta_all, torch.cat(var_chunks, dim=1)
        return delta_all, None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ctx = self.encode_context(batch)
        J0 = ctx["J0"]
        action_h = ctx["action_h"]
        evid_h = ctx["evidence_h"]
        scene = ctx["scene"]
        B, K, H = action_h.shape
        E = evid_h.shape[1]
        if "evidence_query_features" in batch:
            share_q = bool(self.cfg.get("training", {}).get("shared_dense_query_projection", True))
            q_h_all = self._dense_query_projection(batch) if (share_q and self.pair_conditioned) else None
            local, local_var, residual_action, residual_action_var = self._dense_local_from_batch(
                ctx,
                batch,
                q_h_all=q_h_all,
                compute_variance=self._need_dense_local_variance(),
            )
            if bool((self.cfg.get("training", {}) or {}).get("skip_pair_head_forward", False)):
                # DCIP trains a per-evidence action potential and derives every
                # pair correction from potential differences.  The legacy pair
                # MLP is therefore outside the deployed computation graph and
                # can be skipped entirely during training.
                pair_out = None
            else:
                pair_out = self._dense_pair_delta_from_batch(
                    ctx,
                    batch,
                    q_h_all=q_h_all,
                    return_uncertainty=self._need_pair_uncertainty(),
                )
        else:
            local = torch.zeros((B, E, K), dtype=J0.dtype, device=J0.device)
            local_var = torch.zeros((B, E, K), dtype=J0.dtype, device=J0.device)
            residual_action = torch.zeros((B, E, K), dtype=J0.dtype, device=J0.device)
            residual_action_var = torch.zeros((B, E, K), dtype=J0.dtype, device=J0.device)
            pair_out = None
        set_atom_factors, set_action_factors = self.set_residual_factors(ctx)
        out = {
            "J0": J0,
            "g": local,
            "g_var": local_var,
            "residual_action_potential": residual_action,
            "residual_action_var": residual_action_var,
            "residual_set_atom_factors": set_atom_factors,
            "residual_set_action_factors": set_action_factors,
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
            out["pair_atom_delta"] = pair_out[0]
            if pair_out[1] is not None:
                out["pair_atom_var"] = pair_out[1]
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

    def _selector_pair_thresholds(self, cfg: dict[str, Any]) -> tuple[float, float]:
        """Return selector thresholds in the same margin units used by the pair head.

        A normalized pair head predicts d_i(a,b) / scale.  Training already uses
        selector.normalized_eta_pred / selector.normalized_gamma_max, but the old
        deployment path used raw eta_pred when constructing the runtime pair graph
        and tournament rival sets.  That train/deploy mismatch makes the learned
        selector optimize a different set of comparisons than it saw in training.
        """
        sel = cfg.get("selector", {}) if isinstance(cfg, dict) else {}
        mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        normalize = bool(mcfg.get("pair_margin_normalized", self.pair_margin_normalized))
        if normalize:
            eta = float(sel.get("normalized_eta_pred", sel.get("eta_pred", 0.1)))
            gamma = float(sel.get("normalized_gamma_max", sel.get("gamma_max_default", 5.0)))
        else:
            eta = float(sel.get("eta_pred", 1.0))
            gamma = float(sel.get("gamma_max_default", sel.get("gamma_max", 100.0)))
        return eta, gamma

    @contextmanager
    def runtime_prediction_cache_scope(self):
        """Reuse the expensive scene encoder across fallback stages of one plan.

        The fixed runtime/candidate/evidence objects are immutable inside a
        planner call, while B/M/L only affect proposal truncation and the sparse
        query graph.  A thread-local scope is safe with a shared read-only model
        and never leaks tensors across scenes.
        """
        memo = getattr(_RUNTIME_PREDICTION_CACHE, "memo", None)
        owner = not isinstance(memo, _RuntimePredictionMemo) or memo.depth <= 0
        if owner:
            memo = _RuntimePredictionMemo()
            _RUNTIME_PREDICTION_CACHE.memo = memo
        memo.depth += 1
        try:
            yield memo
        finally:
            memo.depth -= 1
            if owner:
                try:
                    delattr(_RUNTIME_PREDICTION_CACHE, "memo")
                except AttributeError:
                    pass

    def _runtime_encoded_context(self, runtime, candidates, evidence_bank):
        memo = getattr(_RUNTIME_PREDICTION_CACHE, "memo", None)
        key = (id(self), id(runtime), id(candidates), id(evidence_bank))
        if isinstance(memo, _RuntimePredictionMemo) and memo.depth > 0 and key in memo.cache:
            memo.hits += 1
            return (*memo.cache[key], True, 0.0, 0.0)
        started = time.perf_counter()
        batch = self._make_batch(runtime, candidates, evidence_bank, include_dense_query=False)
        make_batch_s = float(time.perf_counter() - started)
        started = time.perf_counter()
        self.eval()
        with torch.inference_mode():
            ctx = self.encode_context(batch)
            set_atom_factors_t, set_action_factors_t = self.set_residual_factors(ctx)
        encode_context_s = float(time.perf_counter() - started)
        if isinstance(memo, _RuntimePredictionMemo) and memo.depth > 0:
            memo.cache[key] = (batch, ctx, set_atom_factors_t, set_action_factors_t)
            memo.misses += 1
        return batch, ctx, set_atom_factors_t, set_action_factors_t, False, make_batch_s, encode_context_s

    def predict_certificate_numpy(self, runtime, candidates, evidence_bank, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = cfg or self.cfg
        profile_enabled = os.environ.get("BDSE_PROFILE_CLOSED_LOOP", "0").lower() in {"1", "true", "yes", "on"}
        model_timing: dict[str, float] = {}
        t_model = time.perf_counter()
        (batch, ctx, set_atom_factors_t, set_action_factors_t, context_cache_hit,
         make_batch_s, encode_context_s) = self._runtime_encoded_context(runtime, candidates, evidence_bank)
        if profile_enabled:
            model_timing["model_context_cache_hit"] = float(context_cache_hit)
            model_timing["model_make_batch_s"] = float(make_batch_s)
            model_timing["model_encode_context_s"] = float(encode_context_s)
            t_model = time.perf_counter()
        J0 = ctx["J0"][0].detach().cpu().numpy().astype(np.float32)
        proposal_logits = ctx["proposal_logits"][0].detach().cpu().numpy().astype(np.float32)
        family_logits = ctx["family_logits"][0].detach().cpu().numpy().astype(np.float32)
        family_pi = ctx["family_pi"][0].detach().cpu().numpy().astype(np.float32)
        set_atom_factors_np = (
            set_atom_factors_t[0].detach().cpu().numpy().astype(np.float32)
            if set_atom_factors_t is not None else None
        )
        set_action_factors_np = (
            set_action_factors_t[0].detach().cpu().numpy().astype(np.float32)
            if set_action_factors_t is not None else None
        )
        flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
        # v12: an evidence-free, robustly normalized decision prior becomes part
        # of J0.  This keeps the fixed evidence budget intact while giving the
        # budgeted residual evidence a stronger progress/route anchor.
        J0, base_prior_diag = _apply_runtime_base_prior_np(J0, candidates, flags, cfg)
        J0, structural_residual_diag = _apply_structural_safety_residual_prior_np(J0, runtime, candidates, cfg)
        selector_eta, selector_gamma = self._selector_pair_thresholds(cfg)
        pairs, pair_weights = build_runtime_pairs_from_base(
            J0,
            candidates.valid_mask,
            flags,
            L0=int(cfg.get("tournament", {}).get("L_infer", 16)),
            eta0=selector_eta,
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
        # v34 ABIQ: compact the pair graph before neural scoring.  The final
        # tournament is antisymmetric, so reciprocal directed queries are redundant.
        # A strict logical-pair cap makes the fixed query budget an executed
        # constraint rather than a post-hoc reporting threshold.
        query_pair_cap = int(cfg.get("selector", {}).get("max_runtime_pair_queries", 512))
        pairs, pair_weights, runtime_pair_compact_diag = compact_runtime_pair_graph(
            pairs,
            pair_weights,
            J0,
            candidates.valid_mask,
            flags,
            maneuver_ids=candidates.maneuver_ids,
            candidate_trajectories=candidates.trajectories,
            max_pairs=query_pair_cap,
            canonicalize_reciprocals=bool(cfg.get("selector", {}).get("canonicalize_reciprocal_queries", True)),
        )
        selector_cfg_early = cfg.get("selector", {}) if isinstance(cfg, dict) else {}
        viability_pair_diag: dict[str, float] = {}
        viability_mode = str(selector_cfg_early.get("decision_pair_viability_mode", "hard_frontier" if selector_cfg_early.get("decision_pairs_within_viability_frontier", False) else "full")).lower()
        if viability_mode in {"hard_frontier", "restrict", "strict"}:
            risk_dict = runtime_risk_scores(runtime, candidates, cfg)
            pairs, pair_weights, viability_pair_diag = restrict_pairs_to_viability_frontier(
                pairs,
                pair_weights,
                candidates.valid_mask,
                flags,
                J0,
                hard_risk=risk_dict.get("hard", None),
                frontier_size=int(selector_cfg_early.get("all_flagged_frontier_size", 8)),
                single_safe_rivals=int(selector_cfg_early.get("single_safe_anchor_rivals", 8)),
            )
        elif viability_mode in {"soft_weight", "weighted", "reweight"}:
            risk_dict = runtime_risk_scores(runtime, candidates, cfg)
            pairs, pair_weights, viability_pair_diag = reweight_pairs_by_viability_scope(
                pairs,
                pair_weights,
                candidates.valid_mask,
                flags,
                J0,
                hard_risk=risk_dict.get("hard", None),
                safe_safe_weight=float(selector_cfg_early.get("viability_safe_safe_weight", 1.0)),
                cross_safety_weight=float(selector_cfg_early.get("viability_cross_safety_weight", 0.35)),
                unsafe_unsafe_weight=float(selector_cfg_early.get("viability_unsafe_unsafe_weight", 0.10)),
                all_flagged_frontier_size=int(selector_cfg_early.get("all_flagged_frontier_size", 8)),
                outside_frontier_weight=float(selector_cfg_early.get("viability_outside_frontier_weight", 0.20)),
            )
        else:
            viability_pair_diag = {
                "pair_count_before_viability": float(len(pairs)),
                "pair_count_after_viability": float(len(pairs)),
                "viability_safe_action_count": float((np.asarray(candidates.valid_mask, dtype=bool) & ~np.asarray(flags, dtype=bool)).sum()),
                "viability_scope_code": 4.0,
            }
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
        interaction_family_set = set(int(x) for x in selector_cfg_early.get("interaction_family_ids", [2, 3]))
        soft_interaction_mask = np.asarray(
            [int(f) in interaction_family_set for f in family_ids.tolist()], dtype=bool
        ) & active & ~raw_hard_mask
        interaction_group_ids = np.full((evidence_bank.E,), -1, dtype=np.int64)
        for i, atom in enumerate(evidence_bank.atoms[: evidence_bank.E]):
            try:
                interaction_group_ids[i] = int(getattr(atom, "anchor", {}).get("agent_index", -1))
            except Exception:
                interaction_group_ids[i] = -1
        mandatory_hard_mask = structural_safety_mask(
            raw_hard_mask,
            family_ids,
            active,
            include_feasibility=bool(cfg.get("selector", {}).get("structural_safety_include_feasibility", True)),
        )
        structural_safety_bypass = bool(selector_cfg_early.get("decision_budget_excludes_structural_safety", False))
        if structural_safety_bypass:
            decision_mask = active & ~mandatory_hard_mask
            topm, decision_topm_diag = restrict_topm_to_decision_evidence(
                topm,
                decision_mask,
                proposal_logits[: evidence_bank.E],
                M,
                family_ids=family_ids,
            )
            hab_diag = dict(hab_diag)
            hab_diag.update({f"scide_{k}": int(v) for k, v in decision_topm_diag.items()})
            hab_diag["structural_safety_bypass"] = 1
        elif bool(cfg.get("selector", {}).get("force_hard_topm", True)):
            forced = np.flatnonzero(mandatory_hard_mask)
            if forced.size:
                forced_cap = int(cfg.get("selector", {}).get("max_forced_hard_topm", max(1, M // 2)))
                forced = np.asarray(sorted(forced.tolist(), key=lambda i: (-float(proposal_logits[int(i)]), int(i)))[:forced_cap], dtype=np.int64)
                forced_set = set(forced.tolist())
                non_forced = [int(i) for i in np.asarray(topm, dtype=np.int64).reshape(-1).tolist() if int(i) not in forced_set]
                topm = np.asarray((forced.tolist() + non_forced)[:M], dtype=np.int64)
                hab_diag = dict(hab_diag)
                hab_diag["forced_hard_topm"] = int(forced.size)
        min_soft_topm = int(selector_cfg_early.get("min_soft_interaction_topm_slots", 0))
        if min_soft_topm > 0 and bool(soft_interaction_mask.any()):
            topm, soft_topm_diag = reserve_topm_candidates(
                topm,
                soft_interaction_mask,
                proposal_logits[: evidence_bank.E],
                M,
                min_soft_topm,
                protected_mask=mandatory_hard_mask,
                group_ids=interaction_group_ids,
            )
            hab_diag = dict(hab_diag)
            hab_diag.update({f"soft_interaction_topm_{k}": int(v) for k, v in soft_topm_diag.items()})
        rival_sets = build_rival_sets_from_base(
            J0,
            candidates.valid_mask,
            flags,
            L_infer=int(cfg.get("tournament", {}).get("L_infer", 16)),
            eta0=selector_eta,
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
        rival_pairs, _, rival_pair_compact_diag = compact_runtime_pair_graph(
            rival_pairs,
            None,
            J0,
            candidates.valid_mask,
            flags,
            maneuver_ids=candidates.maneuver_ids,
            candidate_trajectories=candidates.trajectories,
            max_pairs=query_pair_cap,
            canonicalize_reciprocals=bool(cfg.get("selector", {}).get("canonicalize_reciprocal_queries", True)),
        )
        if action_set:
            action_ids = np.asarray(sorted(action_set), dtype=np.int64)
        elif len(pairs):
            action_ids = np.unique(pairs.reshape(-1))
        else:
            action_ids = np.flatnonzero(candidates.valid_mask)[: max(1, int(cfg.get("tournament", {}).get("L_infer", 16)))]
        runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
        mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        scfg = cfg.get("selector", {}) if isinstance(cfg, dict) else {}
        use_pair_runtime = bool(runtime_cfg.get("use_pair_conditioned_margins", self.pair_conditioned))
        hybrid_local_weight = float(runtime_cfg.get("pair_delta_hybrid_local_weight", 0.0))
        normalize_pairs = bool(mcfg.get("pair_margin_normalized", self.pair_margin_normalized))
        tcfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
        min_scale = float(mcfg.get("margin_normalization_min_scale", tcfg.get("pair_margin_min_scale", 100.0)))
        q_scale = float(mcfg.get("margin_normalization_quantile", 0.75))

        # Runtime v11 option: optimize the evidence budget on the same pair graph
        # that will be consumed by the final tournament.  The old selector covered
        # only the cheap base-screened pairs and then used a different rival graph
        # for the action tournament, which created a train/deploy decision gap.
        selection_pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2) if len(pairs) else np.zeros((0, 2), dtype=np.int64)
        selection_pair_weights = np.asarray(pair_weights, dtype=np.float32).reshape(-1) if len(pair_weights) else np.zeros((0,), dtype=np.float32)
        selector_pair_union_enabled = False
        if bool(scfg.get("use_tournament_pair_union", False)) and len(rival_pairs):
            selector_pair_union_enabled = True
            raw_selection = np.concatenate([selection_pairs, rival_pairs.reshape(-1, 2)], axis=0) if len(selection_pairs) else rival_pairs.reshape(-1, 2)
            selection_pairs = np.unique(raw_selection.astype(np.int64), axis=0)
            if normalize_pairs and len(selection_pairs):
                selection_scale_for_weights = margin_normalization_scale(
                    J0[selection_pairs[:, 1]] - J0[selection_pairs[:, 0]],
                    min_scale=min_scale,
                    quantile=q_scale,
                )
            else:
                selection_scale_for_weights = 1.0
            weight_map: dict[tuple[int, int], float] = {}
            for idx, (a_raw, b_raw) in enumerate(np.asarray(pairs, dtype=np.int64).reshape(-1, 2).tolist() if len(pairs) else []):
                key = (int(a_raw), int(b_raw))
                w = float(pair_weights[idx]) if idx < len(pair_weights) else 1.0
                weight_map[key] = max(weight_map.get(key, 0.0), w)
            tournament_pair_weight = float(scfg.get("tournament_pair_weight", 1.0))
            for a_raw, b_raw in rival_pairs.reshape(-1, 2).tolist():
                key = (int(a_raw), int(b_raw))
                weight_map[key] = weight_map.get(key, 0.0) + tournament_pair_weight
            near_bonus = float(scfg.get("tournament_near_pair_weight", 0.0))
            safety_bonus = float(scfg.get("tournament_safety_pair_weight", scfg.get("lambda_safety", 2.0)))
            weights = []
            for a_raw, b_raw in selection_pairs.tolist():
                a_i, b_i = int(a_raw), int(b_raw)
                w = float(weight_map.get((a_i, b_i), 1.0))
                if 0 <= b_i < len(flags) and bool(flags[b_i]):
                    w += safety_bonus
                if 0 <= a_i < len(J0) and 0 <= b_i < len(J0):
                    raw_margin = float(J0[b_i] - J0[a_i])
                    norm_margin = raw_margin / max(float(selection_scale_for_weights), 1e-6) if normalize_pairs else raw_margin
                    if abs(norm_margin) <= float(selector_eta):
                        w += near_bonus
                weights.append(max(w, 1e-3))
            selection_pair_weights = np.asarray(weights, dtype=np.float32)

        # v16: keep the selector pair graph focused on plausible winner/rival
        # decisions.  v15 increased selected decisive recall but diffused the
        # budget over many non-decision pairs; in closed-loop this also raised GPU
        # memory use.  This filter is deployment-only and uses only base cost,
        # cheap safety flags, and candidate geometry.  It preserves safety pairs
        # while prioritizing pairs involving top base/progress anchors and pairs
        # near the action-order boundary.
        max_selector_pairs = int(scfg.get("max_selector_pairs", 0))
        if max_selector_pairs > 0 and len(selection_pairs) > max_selector_pairs:
            valid_for_filter = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)
            valid_idx_filter = np.flatnonzero(valid_for_filter & np.isfinite(J0)).astype(np.int64)
            anchor_topk = max(1, int(scfg.get("selector_anchor_topk", 6)))
            anchors: set[int] = set()
            if valid_idx_filter.size:
                base_order = valid_idx_filter[np.argsort(J0[valid_idx_filter])[: min(anchor_topk, valid_idx_filter.size)]]
                anchors.update(map(int, base_order.tolist()))
            progress_anchor_topk = max(0, int(scfg.get("selector_progress_anchor_topk", 4)))
            if progress_anchor_topk > 0 and getattr(candidates, "trajectories", None) is not None and valid_idx_filter.size:
                traj = np.asarray(candidates.trajectories, dtype=np.float32)
                if traj.ndim >= 3 and traj.shape[0] >= candidates.K and traj.shape[2] >= 1:
                    prog = np.nan_to_num(traj[: candidates.K, -1, 0], nan=0.0, posinf=0.0, neginf=0.0)
                    safe_valid = valid_idx_filter[~np.asarray(flags, dtype=bool).reshape(-1)[: candidates.K][valid_idx_filter]]
                    pool = safe_valid if safe_valid.size else valid_idx_filter
                    prog_order = pool[np.argsort(-prog[pool])[: min(progress_anchor_topk, pool.size)]]
                    anchors.update(map(int, prog_order.tolist()))
            if anchors and len(selection_pairs):
                pair_arr_filter = np.asarray(selection_pairs, dtype=np.int64).reshape(-1, 2)
                a_f = pair_arr_filter[:, 0]
                b_f = pair_arr_filter[:, 1]
                base_gap = J0[b_f] - J0[a_f]
                if normalize_pairs:
                    scale_for_filter = margin_normalization_scale(base_gap, min_scale=min_scale, quantile=q_scale)
                    norm_gap = base_gap / max(float(scale_for_filter), 1e-6)
                else:
                    norm_gap = base_gap
                flags_arr = np.asarray(flags, dtype=bool).reshape(-1)
                if flags_arr.shape[0] < candidates.K:
                    flags_arr = np.pad(flags_arr, (0, candidates.K - flags_arr.shape[0]), constant_values=False)
                anchors_mask = np.asarray([(int(x) in anchors) or (int(y) in anchors) for x, y in pair_arr_filter.tolist()], dtype=bool)
                safety_mask = flags_arr[np.clip(a_f, 0, candidates.K - 1)] | flags_arr[np.clip(b_f, 0, candidates.K - 1)]
                near_mask = np.abs(norm_gap) <= float(scfg.get("selector_filter_near_eta_mult", 2.0)) * max(float(selector_eta), 1e-6)
                base_w = np.asarray(selection_pair_weights, dtype=np.float32).reshape(-1)
                if base_w.shape[0] != pair_arr_filter.shape[0]:
                    base_w = np.ones((pair_arr_filter.shape[0],), dtype=np.float32)
                score_filter = (
                    base_w
                    + float(scfg.get("selector_filter_anchor_bonus", 2.0)) * anchors_mask.astype(np.float32)
                    + float(scfg.get("selector_filter_safety_bonus", 4.0)) * safety_mask.astype(np.float32)
                    + float(scfg.get("selector_filter_near_bonus", 1.0)) * near_mask.astype(np.float32)
                )
                order = sorted(range(pair_arr_filter.shape[0]), key=lambda i: (-float(score_filter[i]), abs(float(norm_gap[i])), int(a_f[i]), int(b_f[i])))
                keep = np.asarray(order[: max_selector_pairs], dtype=np.int64)
                selection_pairs = pair_arr_filter[keep]
                selection_pair_weights = base_w[keep].astype(np.float32)

        if normalize_pairs and len(selection_pairs):
            pair_margin_scale = margin_normalization_scale(J0[selection_pairs[:, 1]] - J0[selection_pairs[:, 0]], min_scale=min_scale, quantile=q_scale)
        else:
            pair_margin_scale = 1.0
        if bool(runtime_cfg.get("shared_pair_margin_scale", False)):
            rival_pair_margin_scale = pair_margin_scale
        elif normalize_pairs and len(rival_pairs):
            rival_pair_margin_scale = margin_normalization_scale(J0[rival_pairs[:, 1]] - J0[rival_pairs[:, 0]], min_scale=min_scale, quantile=q_scale)
        else:
            rival_pair_margin_scale = pair_margin_scale

        pair_cal_cfg = (runtime_cfg.get("pair_delta_calibration", {}) or {}) if isinstance(runtime_cfg, dict) else {}
        pair_cal_enabled = bool(pair_cal_cfg.get("enabled", False))
        need_action_sparse = (
            not use_pair_runtime
            or bool(mcfg.get("pair_head_residual_over_local", False))
            or hybrid_local_weight > 0.0
            or pair_cal_enabled
        )
        g_sparse = np.zeros((evidence_bank.E, candidates.K), dtype=np.float32)
        g_var_sparse = np.zeros((evidence_bank.E, candidates.K), dtype=np.float32)
        residual_action_sparse = np.zeros((evidence_bank.E, candidates.K), dtype=np.float32)
        residual_action_var_sparse = np.zeros((evidence_bank.E, candidates.K), dtype=np.float32)
        if need_action_sparse:
            t_sparse = time.perf_counter()
            atom_ids, action_ids_rep, q = compute_query_features_for_pairs(evidence_bank.atoms, candidates, runtime, topm, action_ids, cfg)
            if len(atom_ids):
                atom_t = torch.from_numpy(atom_ids[None].astype(np.int64)).to(next(self.parameters()).device)
                act_t = torch.from_numpy(action_ids_rep[None].astype(np.int64)).to(next(self.parameters()).device)
                q_t = torch.from_numpy(q[None].astype(np.float32)).to(next(self.parameters()).device)
                with torch.inference_mode():
                    vals, var, residual_vals, residual_var = self.score_sparse_queries(
                        ctx, atom_t, act_t, q_t,
                        return_uncertainty=True,
                        return_residual_action=True,
                    )
                    vals_np = vals[0].detach().cpu().numpy().astype(np.float32)
                    var_np = var[0].detach().cpu().numpy().astype(np.float32)
                    residual_np = residual_vals[0].detach().cpu().numpy().astype(np.float32)
                    residual_var_np = residual_var[0].detach().cpu().numpy().astype(np.float32)
                g_sparse[atom_ids, action_ids_rep] = vals_np
                g_var_sparse[atom_ids, action_ids_rep] = var_np
                residual_action_sparse[atom_ids, action_ids_rep] = residual_np
                residual_action_var_sparse[atom_ids, action_ids_rep] = residual_var_np
            if profile_enabled:
                model_timing["model_action_sparse_s"] = float(time.perf_counter() - t_sparse)
        else:
            atom_ids = np.zeros((0,), dtype=np.int64)
            action_ids_rep = np.zeros((0,), dtype=np.int64)

        # Score selector pairs and tournament-rival pairs in one GPU call over the
        # unique pair union.  With selector.use_tournament_pair_union=true, the
        # selector graph itself is the decision/tournament graph, while query
        # accounting still reports the original cheap runtime-pair count.
        actual_unique_pair_count = 0
        scored_unique_pair_count = 0
        pair_residual_refinement_diag: dict[str, Any] = {
            "pair_residual_refinement_enabled": False,
            "pair_residual_refined_pair_count": 0,
            "pair_residual_total_pair_count": 0,
            "pair_residual_refined_fraction": 0.0,
        }
        if len(selection_pairs) or len(rival_pairs):
            t_pair = time.perf_counter()
            all_pairs = np.concatenate([selection_pairs.reshape(-1, 2), rival_pairs.reshape(-1, 2)], axis=0)
            unique_pairs, inverse = np.unique(all_pairs, axis=0, return_inverse=True)
            actual_unique_pair_count = int(len(unique_pairs))

            # D3CE selective residual refinement: the local action-conditioned
            # interface provides an integrable margin for every pair.  The more
            # expensive pair head is treated as a residual and is evaluated only
            # on near-boundary, safety-crossing, or current-winner pairs.  Unscored
            # residuals are exactly zero, so all pairs still have a valid local
            # margin and the final tournament graph is unchanged.
            refine_cap = int(runtime_cfg.get("pair_residual_refine_max_pairs", 0))
            residual_mode = bool(mcfg.get("pair_head_residual_over_local", False))
            refine_ids = np.arange(len(unique_pairs), dtype=np.int64)
            if residual_mode and refine_cap > 0 and len(unique_pairs) > refine_cap:
                a_u = np.clip(unique_pairs[:, 0], 0, candidates.K - 1)
                b_u = np.clip(unique_pairs[:, 1], 0, candidates.K - 1)
                local_full_cost = J0 + g_sparse[np.asarray(topm, dtype=np.int64)].sum(axis=0)
                local_margin_raw = local_full_cost[b_u] - local_full_cost[a_u]
                local_scale = margin_normalization_scale(
                    local_margin_raw, min_scale=min_scale, quantile=q_scale
                ) if normalize_pairs else 1.0
                local_margin = local_margin_raw / max(float(local_scale), 1e-6)
                flags_u = np.asarray(flags, dtype=bool).reshape(-1)
                if flags_u.shape[0] < candidates.K:
                    flags_u = np.pad(flags_u, (0, candidates.K - flags_u.shape[0]), constant_values=False)
                safety_cross = flags_u[a_u] ^ flags_u[b_u]
                valid_cost = np.asarray(local_full_cost, dtype=np.float64).copy()
                valid_cost[~np.asarray(candidates.valid_mask, dtype=bool)] = np.inf
                local_winner = int(np.argmin(valid_cost)) if np.isfinite(valid_cost).any() else -1
                winner_pair = (a_u == local_winner) | (b_u == local_winner)
                tau_refine = max(float(runtime_cfg.get("pair_residual_refine_boundary_tau", 0.35)), 1e-6)
                priority = (
                    1.0 / (np.abs(local_margin) + tau_refine)
                    + float(runtime_cfg.get("pair_residual_refine_safety_bonus", 4.0)) * safety_cross.astype(np.float32)
                    + float(runtime_cfg.get("pair_residual_refine_winner_bonus", 2.0)) * winner_pair.astype(np.float32)
                )
                refine_ids = np.argsort(-priority, kind="stable")[:refine_cap].astype(np.int64)
                pair_residual_refinement_diag.update({
                    "pair_residual_refinement_enabled": True,
                    "pair_residual_refined_pair_count": int(len(refine_ids)),
                    "pair_residual_total_pair_count": int(len(unique_pairs)),
                    "pair_residual_refined_fraction": float(len(refine_ids) / max(len(unique_pairs), 1)),
                    "pair_residual_local_winner": int(local_winner),
                })

            skip_pair_head_scoring = bool(runtime_cfg.get("skip_pair_head_scoring", False))
            if skip_pair_head_scoring:
                # V56 deployment derives all action corrections from the
                # evidence-attributable action-potential head.  The legacy pair
                # MLP is not consumed by either the evidence certificate or the
                # final action rule, so scoring it wastes a large E x P forward
                # and can only contaminate diagnostics.
                scored_pairs = np.zeros((0, 2), dtype=np.int64)
                scored_delta = np.zeros((evidence_bank.E, 0), dtype=np.float32)
                scored_var = np.zeros((evidence_bank.E, 0), dtype=np.float32)
                refine_ids = np.zeros((0,), dtype=np.int64)
                pair_residual_refinement_diag.update({
                    "pair_head_scoring_skipped": True,
                    "pair_residual_refined_pair_count": 0,
                    "pair_residual_total_pair_count": int(len(unique_pairs)),
                    "pair_residual_refined_fraction": 0.0,
                })
            else:
                scored_pairs = unique_pairs[refine_ids]
                scored_delta, scored_var = self._score_pair_indices_numpy(
                    ctx, runtime, candidates, evidence_bank, topm, scored_pairs, cfg
                )
            scored_unique_pair_count = int(len(scored_pairs))
            all_pair_delta = np.zeros((evidence_bank.E, len(unique_pairs)), dtype=np.float32)
            all_pair_var = np.zeros((evidence_bank.E, len(unique_pairs)), dtype=np.float32)
            if len(refine_ids):
                all_pair_delta[:, refine_ids] = scored_delta
                all_pair_var[:, refine_ids] = scored_var
            if (not skip_pair_head_scoring) and (not pair_residual_refinement_diag["pair_residual_refinement_enabled"]):
                pair_residual_refinement_diag.update({
                    "pair_residual_refined_pair_count": int(len(unique_pairs)),
                    "pair_residual_total_pair_count": int(len(unique_pairs)),
                    "pair_residual_refined_fraction": 1.0 if len(unique_pairs) else 0.0,
                })
            if profile_enabled:
                model_timing["model_pair_scoring_s"] = float(time.perf_counter() - t_pair)
                model_timing["model_unique_pair_count"] = float(len(unique_pairs))
                model_timing["model_scored_unique_pair_count"] = float(len(scored_pairs))
                model_timing["model_pair_atom_score_count"] = float(len(topm) * len(scored_pairs))
            n_selector_pairs = int(len(selection_pairs))
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

        def _local_pair_delta(pair_arr: np.ndarray, scale: float) -> np.ndarray:
            pair_arr = np.asarray(pair_arr, dtype=np.int64).reshape(-1, 2) if np.asarray(pair_arr).size else np.zeros((0, 2), dtype=np.int64)
            if pair_arr.size == 0:
                return np.zeros((evidence_bank.E, 0), dtype=np.float32)
            a = np.clip(pair_arr[:, 0], 0, candidates.K - 1)
            b = np.clip(pair_arr[:, 1], 0, candidates.K - 1)
            local = g_sparse[:, b] - g_sparse[:, a]
            return (local / max(float(scale), 1e-6)).astype(np.float32) if normalize_pairs else local.astype(np.float32)

        def _local_pair_variance(pair_arr: np.ndarray, scale: float) -> np.ndarray:
            pair_arr = np.asarray(pair_arr, dtype=np.int64).reshape(-1, 2) if np.asarray(pair_arr).size else np.zeros((0, 2), dtype=np.int64)
            if pair_arr.size == 0:
                return np.zeros((evidence_bank.E, 0), dtype=np.float32)
            a = np.clip(pair_arr[:, 0], 0, candidates.K - 1)
            b = np.clip(pair_arr[:, 1], 0, candidates.K - 1)
            var = np.maximum(g_var_sparse[:, a], 0.0) + np.maximum(g_var_sparse[:, b], 0.0)
            if normalize_pairs:
                var = var / max(float(scale) ** 2, 1e-12)
            return var.astype(np.float32)

        pair_delta_calibration_diag: dict[str, Any] = {"pair_delta_calibration_enabled": bool(pair_cal_enabled)}

        def _calibrate_pair_delta(
            pair_pred: np.ndarray,
            pair_var: np.ndarray,
            pair_arr: np.ndarray,
            scale: float,
            prefix: str,
        ) -> np.ndarray:
            local = _local_pair_delta(pair_arr, scale)
            if pair_pred.shape != local.shape or pair_pred.size == 0:
                return pair_pred
            base_w = float(pair_cal_cfg.get("base_local_weight", 0.35))
            min_w = float(pair_cal_cfg.get("min_local_weight", 0.20))
            max_w = float(pair_cal_cfg.get("max_local_weight", 0.70))
            variance_gain = float(pair_cal_cfg.get("variance_gain", 0.25))
            disagreement_gain = float(pair_cal_cfg.get("disagreement_gain", 0.20))
            variance_tau = max(float(pair_cal_cfg.get("variance_tau", 0.35)), 1e-6)
            magnitude_tau = max(float(pair_cal_cfg.get("magnitude_tau", 0.25)), 1e-6)
            var = np.asarray(pair_var, dtype=np.float32)
            if var.shape != pair_pred.shape:
                var = np.zeros_like(pair_pred, dtype=np.float32)
            std = np.sqrt(np.maximum(var, 0.0))
            var_term = std / (std + variance_tau)
            disagree = (pair_pred * local < 0.0).astype(np.float32)
            local_strength = np.tanh(np.abs(local) / magnitude_tau).astype(np.float32)
            w = base_w + variance_gain * var_term + disagreement_gain * disagree * local_strength
            w = np.clip(w, min(min_w, max_w), max(min_w, max_w)).astype(np.float32)
            blended = (1.0 - w) * pair_pred + w * local
            pair_delta_calibration_diag[f"pair_delta_{prefix}_local_weight_mean"] = float(np.mean(w))
            pair_delta_calibration_diag[f"pair_delta_{prefix}_local_weight_p90"] = float(np.quantile(w, 0.90))
            pair_delta_calibration_diag[f"pair_delta_{prefix}_sign_disagreement_rate"] = float(np.mean(disagree))
            pair_delta_calibration_diag[f"pair_delta_{prefix}_local_abs_mean"] = float(np.mean(np.abs(local)))
            pair_delta_calibration_diag[f"pair_delta_{prefix}_pair_abs_mean"] = float(np.mean(np.abs(pair_pred)))
            return blended.astype(np.float32)

        if bool(mcfg.get("pair_head_residual_over_local", False)):
            # FAR-DBAP treats the pretrained local interface as the immutable
            # anchor.  A runtime switch exposes an exact same-checkpoint local
            # control for causal attribution; the candidate path authorizes
            # residual flips against the full foundation margin (base + local).
            residual_cfg = (runtime_cfg.get("pair_residual_trust", {}) or {})
            selector_local = _local_pair_delta(selection_pairs, pair_margin_scale)
            rival_local = _local_pair_delta(rival_pairs, rival_pair_margin_scale)
            if bool(runtime_cfg.get("disable_pair_residual_intervention", False)) or bool(runtime_cfg.get("skip_pair_head_scoring", False)):
                selector_pair_delta = selector_local
                rival_pair_delta = rival_local
                # A same-checkpoint local control must remove the complete
                # residual intervention, not only its mean.  Leaving pair-head
                # variance active changes AOCC selection and uncertainty guards,
                # which contaminated the V54 causal comparison.
                selector_pair_var = np.zeros_like(selector_pair_delta, dtype=np.float32)
                rival_pair_var = np.zeros_like(rival_pair_delta, dtype=np.float32)
                selector_residual_diag = {
                    "residual_disabled_control": float(bool(runtime_cfg.get("disable_pair_residual_intervention", False))),
                    "legacy_pair_head_skipped": float(bool(runtime_cfg.get("skip_pair_head_scoring", False))),
                    "residual_uncertainty_disabled_control": 1.0,
                }
                rival_residual_diag = dict(selector_residual_diag)
            else:
                selector_base = (J0[selection_pairs[:, 1]] - J0[selection_pairs[:, 0]]) / max(float(pair_margin_scale), 1e-6) if selection_pairs.size else np.zeros((0,), dtype=np.float32)
                rival_base = (J0[rival_pairs[:, 1]] - J0[rival_pairs[:, 0]]) / max(float(rival_pair_margin_scale), 1e-6) if rival_pairs.size else np.zeros((0,), dtype=np.float32)
                selector_pair_delta, selector_residual_diag = _confidence_shrunk_residual_pair_delta_np(
                    selector_local, selector_pair_delta, selector_pair_var, residual_cfg, base_margin=selector_base
                )
                rival_pair_delta, rival_residual_diag = _confidence_shrunk_residual_pair_delta_np(
                    rival_local, rival_pair_delta, rival_pair_var, residual_cfg, base_margin=rival_base
                )
            pair_delta_calibration_diag["pair_delta_calibration_enabled"] = True
            for key, value in selector_residual_diag.items():
                pair_delta_calibration_diag[f"pair_delta_selector_{key}"] = float(value)
            for key, value in rival_residual_diag.items():
                pair_delta_calibration_diag[f"pair_delta_tournament_{key}"] = float(value)
        elif pair_cal_enabled:
            selector_pair_delta = _calibrate_pair_delta(selector_pair_delta, selector_pair_var, selection_pairs, pair_margin_scale, "selector")
            rival_pair_delta = _calibrate_pair_delta(rival_pair_delta, rival_pair_var, rival_pairs, rival_pair_margin_scale, "tournament")
        else:
            w_local = float(cfg.get("runtime", {}).get("pair_delta_hybrid_local_weight", 0.0)) if isinstance(cfg, dict) else 0.0
            if w_local > 0.0:
                w_local = min(max(w_local, 0.0), 1.0)
                selector_pair_delta = (1.0 - w_local) * selector_pair_delta + w_local * _local_pair_delta(selection_pairs, pair_margin_scale)
                rival_pair_delta = (1.0 - w_local) * rival_pair_delta + w_local * _local_pair_delta(rival_pairs, rival_pair_margin_scale)
        valid_mask = np.asarray(candidates.valid_mask, dtype=bool)
        if bool(runtime_cfg.get("disable_pair_residual_intervention", False)):
            residual_action_sparse.fill(0.0)
            residual_action_var_sparse.fill(0.0)
            if set_atom_factors_np is not None:
                set_atom_factors_np.fill(0.0)
            if set_action_factors_np is not None:
                set_action_factors_np.fill(0.0)
        g_sparse[:, ~valid_mask] = 0.0
        g_var_sparse[:, ~valid_mask] = 0.0
        residual_action_sparse[:, ~valid_mask] = 0.0
        residual_action_var_sparse[:, ~valid_mask] = 0.0

        # Dual certificate: evidence selection is certified against the
        # selected-local anchor only.  Residual uncertainty is deliberately not
        # injected into AOCC; a separate global residual-flip guard certifies any
        # action change after the evidence certificate has been established.
        certificate_selector_delta = _local_pair_delta(selection_pairs, pair_margin_scale)
        certificate_rival_delta = _local_pair_delta(rival_pairs, rival_pair_margin_scale)
        dual_cfg = (runtime_cfg.get("dual_certificate", {}) or {})
        if str(dual_cfg.get("evidence_uncertainty_source", "none")).lower() == "local":
            certificate_selector_var = _local_pair_variance(selection_pairs, pair_margin_scale)
            certificate_rival_var = _local_pair_variance(rival_pairs, rival_pair_margin_scale)
        else:
            certificate_selector_var = np.zeros_like(certificate_selector_delta, dtype=np.float32)
            certificate_rival_var = np.zeros_like(certificate_rival_delta, dtype=np.float32)

        result = {
            "J0": J0,
            "g": g_sparse,
            "g_var": g_var_sparse,
            "residual_action_potential": residual_action_sparse,
            "residual_action_var": residual_action_var_sparse,
            "residual_set_atom_factors": set_atom_factors_np,
            "residual_set_action_factors": set_action_factors_np,
            "certificate_pair_atom_delta": certificate_selector_delta,
            "certificate_pair_atom_var": certificate_selector_var,
            "certificate_rival_pair_atom_delta": certificate_rival_delta,
            "certificate_rival_pair_atom_var": certificate_rival_var,
            "dual_certificate_active": bool(dual_cfg.get("enabled", False)),
            "proposal_logits": proposal_logits,
            "family_logits": family_logits,
            "family_pi": family_pi,
            "family_ids": family_ids,
            "family_budget_caps": family_budget.family_caps,
            "family_budgets": family_budget.family_budgets,
            "mandatory_atom_mask": mandatory_hard_mask.astype(bool),
            "structural_safety_bypass": bool(structural_safety_bypass),
            "structural_safety_include_feasibility": bool(cfg.get("selector", {}).get("structural_safety_include_feasibility", True)),
            "structural_safety_atom_count": int(mandatory_hard_mask.sum()),
            "soft_interaction_mask": soft_interaction_mask.astype(bool),
            "interaction_group_ids": interaction_group_ids.astype(np.int64),
            "mandatory_hard_atoms": np.flatnonzero(mandatory_hard_mask).astype(np.int64),
            "hab_diagnostics": hab_diag,
            "top_m_atoms": topm,
            "queried_actions": np.asarray(action_ids, dtype=np.int64),
            # Query accounting uses explicit categories.  Keep queried_pair_count
            # as a backward-compatible alias for the total number of sparse model
            # scores actually evaluated in this runtime certificate stage.
            "action_atom_query_count": int(len(atom_ids)),
            # Legacy decompositions remain available for diagnostics, but the
            # actual model call scores the unique selector/tournament union once.
            "selector_pair_atom_query_count": int(len(topm) * len(selection_pairs)),
            "tournament_pair_atom_query_count": int(len(topm) * len(rival_pairs)),
            "unique_pair_atom_query_count": int(len(topm) * scored_unique_pair_count),
            "runtime_pair_count": int(len(pairs)),
            "selector_pair_count": int(len(selection_pairs)),
            "tournament_pair_count": int(len(rival_pairs)),
            "actual_unique_pair_count": int(actual_unique_pair_count),
            "queried_pair_count": int(len(atom_ids) + len(topm) * scored_unique_pair_count),
            "pair_atom_delta": selector_pair_delta,
            "pair_atom_var": selector_pair_var,
            "pair_indices": selection_pairs,
            "pair_margin_scale": float(pair_margin_scale),
            "rival_pair_margin_scale": float(rival_pair_margin_scale),
            "pair_margin_normalized": bool(normalize_pairs),
            "selector_eta_used": float(selector_eta),
            "selector_gamma_used": float(selector_gamma),
            "rival_pair_atom_delta": rival_pair_delta,
            "rival_pair_atom_var": rival_pair_var,
            "rival_pair_indices": rival_pairs,
            "runtime_pairs": pairs,
            "runtime_pair_weights": selection_pair_weights,
            "runtime_base_pair_weights": pair_weights,
            "selector_pair_union_enabled": bool(selector_pair_union_enabled),
            **{f"runtime_{k}": v for k, v in runtime_pair_compact_diag.items()},
            **{f"viability_{k}": v for k, v in viability_pair_diag.items()},
            **{f"rival_{k}": v for k, v in rival_pair_compact_diag.items()},
            **base_prior_diag,
            **structural_residual_diag,
            **pair_delta_calibration_diag,
            **pair_residual_refinement_diag,
        }
        if profile_enabled:
            result["model_timing"] = model_timing
        return result

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
