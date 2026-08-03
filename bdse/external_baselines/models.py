from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bdse.data.cache_schema import RuntimeFeatures
from bdse.data.tensorizer import runtime_to_model_numpy
from bdse.planner.fallback import runtime_safety_flags_from_runtime
from bdse.planner.hab import family_ids_from_atoms
from bdse.planner.pair_screen import build_runtime_pairs_from_base, build_rival_sets_from_base
from bdse.planner.selector import structural_safety_mask

SUPPORTED_EXTERNAL_BASELINES = {
    "pdm_closed",
    "gameformer",
    "dtpp",
    "plantf",
    "pluto",
    "ppad",
}

# These entries deliberately distinguish the original method from this repository's
# budget-compatible adapter.  The adapters share BDSE candidate/evidence inputs and
# are therefore suitable for controlled fixed-budget comparisons, but they are not
# drop-in reproductions of the authors' complete systems.
EXTERNAL_BASELINE_REFERENCES: dict[str, dict[str, str]] = {
    "gameformer": {
        "paper": "GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving (ICCV 2023)",
        "source": "https://github.com/MCZhi/GameFormer and https://github.com/MCZhi/GameFormer-Planner",
        "implementation_label": "GameFormer-inspired budget adapter",
        "fidelity": "partial: hierarchical level-k refinement retained; joint multi-agent prediction/decoder omitted",
    },
    "dtpp": {
        "paper": "DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving (2023)",
        "source": "https://research.nvidia.com/labs/avg/publication/huang.karkus.etal.arxiv2023/",
        "implementation_label": "DTPP-inspired budget adapter",
        "fidelity": "partial: maneuver/tree branch scoring retained; ego-conditioned prediction and scenario tree omitted",
    },
    "plantf": {
        "paper": "Rethinking Imitation-based Planner for Autonomous Driving / PlanTF (ICRA 2024)",
        "source": "https://github.com/jchengai/planTF",
        "implementation_label": "PlanTF-inspired budget adapter",
        "fidelity": "partial: Transformer imitation planner and state dropout retained; official object-token pipeline/augmentations omitted",
    },
    "pluto": {
        "paper": "PLUTO: Pushing the Limit of Imitation Learning-based Planning for Autonomous Driving (2024)",
        "source": "https://github.com/jchengai/pluto",
        "implementation_label": "PLUTO-inspired budget adapter",
        "fidelity": "partial: longitudinal/lateral cost decomposition retained; auxiliary/CIL/augmentation framework omitted",
    },
    "pdm_closed": {
        "paper": "Parting with Misconceptions about Learning-based Vehicle Motion Planning (CoRL 2023)",
        "source": "https://github.com/autonomousvision/tuplan_garage",
        "implementation_label": "PDM-Closed-style budget scorer",
        "fidelity": "low: centerline/progress/comfort/safety prior retained; official proposal generation, IDM rollout and scoring stack omitted",
    },
    "ppad": {
        "paper": "PPAD-style iterative policy refinement adapter",
        "source": "repository-local adapter",
        "implementation_label": "PPAD-inspired budget adapter",
        "fidelity": "partial",
    },
}


def external_reference(variant: str) -> dict[str, str]:
    return dict(EXTERNAL_BASELINE_REFERENCES.get(str(variant), {}))


def external_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    ecfg = dict(cfg.get("external_baseline", {}) or {})
    if "budget" not in ecfg:
        ecfg["budget"] = int(cfg.get("evidence", {}).get("budget", 16))
    return ecfg


def external_variant(cfg: dict[str, Any]) -> str:
    return str(external_cfg(cfg).get("variant", "plantf")).lower().replace("-", "_")


def is_external_enabled(cfg: dict[str, Any]) -> bool:
    ecfg = cfg.get("external_baseline", {}) or {}
    return bool(ecfg.get("enabled", False))


def _torch_bool(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.bool else x.bool()


def _fit_last_dim_torch(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    return F.pad(x, (0, dim - x.shape[-1]))


def _fit_last_dim_np(x: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[-1] == dim:
        return arr
    if arr.shape[-1] > dim:
        return arr[..., :dim]
    pad = [(0, 0)] * arr.ndim
    pad[-1] = (0, dim - arr.shape[-1])
    return np.pad(arr, pad, mode="constant")


def candidate_numeric_features_torch(traj: torch.Tensor, valid: torch.Tensor, dt: float) -> torch.Tensor:
    """Small candidate descriptor shared by the external baseline adapters.

    The descriptor is intentionally built only from the candidate bank and
    runtime-derived candidate validity, so it can be used both in cached
    open-loop evaluation and in nuPlan closed-loop simulation.
    """
    traj = traj.float()
    valid = _torch_bool(valid)
    xy = traj[..., :2]
    T = int(max(traj.shape[2], 1))
    dxy = xy[:, :, 1:] - xy[:, :, :-1] if T > 1 else torch.zeros_like(xy[:, :, :1])
    step = torch.linalg.norm(dxy, dim=-1) if T > 1 else torch.zeros((*xy.shape[:2], 1), device=xy.device, dtype=xy.dtype)
    path_len = step.sum(dim=-1)
    speed = step / max(float(dt), 1e-3)
    speed_mean = speed.mean(dim=-1) if speed.numel() else torch.zeros_like(path_len)
    speed_max = speed.max(dim=-1).values if speed.numel() else torch.zeros_like(path_len)
    speed_final = speed[:, :, -1] if speed.shape[-1] else torch.zeros_like(path_len)
    accel = speed[:, :, 1:] - speed[:, :, :-1] if speed.shape[-1] > 1 else torch.zeros((*xy.shape[:2], 1), device=xy.device, dtype=xy.dtype)
    jerk = accel[:, :, 1:] - accel[:, :, :-1] if accel.shape[-1] > 1 else torch.zeros((*xy.shape[:2], 1), device=xy.device, dtype=xy.dtype)
    acc_rms = torch.sqrt((accel * accel).mean(dim=-1).clamp_min(0.0)) if accel.numel() else torch.zeros_like(path_len)
    jerk_rms = torch.sqrt((jerk * jerk).mean(dim=-1).clamp_min(0.0)) if jerk.numel() else torch.zeros_like(path_len)
    yaw = traj[..., 2] if traj.shape[-1] > 2 else torch.atan2(dxy[..., 1].mean(dim=-1), dxy[..., 0].mean(dim=-1)).unsqueeze(-1).expand(*xy.shape[:2], T)
    yaw_delta = torch.atan2(torch.sin(yaw[:, :, 1:] - yaw[:, :, :-1]), torch.cos(yaw[:, :, 1:] - yaw[:, :, :-1])) if T > 1 else torch.zeros((*xy.shape[:2], 1), device=xy.device, dtype=xy.dtype)
    curvature_mean = yaw_delta.abs().mean(dim=-1) if yaw_delta.numel() else torch.zeros_like(path_len)
    curvature_max = yaw_delta.abs().max(dim=-1).values if yaw_delta.numel() else torch.zeros_like(path_len)
    lat = xy[..., 1]
    progress = xy[..., 0]
    feats = torch.stack(
        [
            progress[:, :, -1] / 120.0,
            lat[:, :, -1] / 20.0,
            lat.abs().mean(dim=-1) / 20.0,
            lat.abs().max(dim=-1).values / 20.0,
            path_len / 120.0,
            speed_mean / 30.0,
            speed_max / 40.0,
            speed_final / 30.0,
            acc_rms / 5.0,
            jerk_rms / 10.0,
            curvature_mean,
            curvature_max,
            yaw[:, :, -1].sin(),
            yaw[:, :, -1].cos(),
            valid.float(),
        ],
        dim=-1,
    )
    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def candidate_rule_cost_np(candidates: Any, runtime_flags: np.ndarray | None, cfg: dict[str, Any]) -> np.ndarray:
    traj = np.asarray(candidates.trajectories, dtype=np.float32)
    valid = np.asarray(candidates.valid_mask, dtype=bool).reshape(-1)
    xy = traj[..., :2]
    dt = float(cfg.get("candidate", {}).get("step_s", 0.1))
    if traj.shape[1] > 1:
        dxy = np.diff(xy, axis=1)
        step = np.linalg.norm(dxy, axis=-1)
    else:
        step = np.zeros((traj.shape[0], 1), dtype=np.float32)
    progress = xy[:, -1, 0]
    lateral_mean = np.mean(np.abs(xy[..., 1]), axis=1)
    lateral_final = np.abs(xy[:, -1, 1])
    path_len = step.sum(axis=1)
    speed = step / max(dt, 1e-3)
    speed_mean = speed.mean(axis=1) if speed.size else np.zeros((traj.shape[0],), dtype=np.float32)
    acc = np.diff(speed, axis=1) if speed.shape[1] > 1 else np.zeros((traj.shape[0], 1), dtype=np.float32)
    jerk = np.diff(acc, axis=1) if acc.shape[1] > 1 else np.zeros((traj.shape[0], 1), dtype=np.float32)
    comfort = 0.25 * np.sqrt(np.mean(acc * acc, axis=1)) + 0.1 * np.sqrt(np.mean(jerk * jerk, axis=1))
    flags = np.zeros((traj.shape[0],), dtype=bool) if runtime_flags is None else np.asarray(runtime_flags, dtype=bool).reshape(-1)
    if flags.shape[0] < traj.shape[0]:
        flags = np.pad(flags, (0, traj.shape[0] - flags.shape[0]), constant_values=False)
    # PDM-style prior: stay near route/centerline, keep smoothness, prefer safe
    # progress, and strongly penalize cheap safety rule flags.
    cost = 2.0 * lateral_mean + 0.75 * lateral_final + 0.5 * comfort - 0.05 * progress - 0.01 * path_len
    cost += np.where(speed_mean < 0.3, 0.15, 0.0)
    cost += flags[: traj.shape[0]].astype(np.float32) * float(cfg.get("teacher", {}).get("feasibility", {}).get("hard_priority_scale", 10000.0))
    cost = cost.astype(np.float32)
    cost[~valid] = np.inf
    return cost


def cheap_proposal_logits_np(evidence_bank: Any, cfg: dict[str, Any]) -> np.ndarray:
    E = int(getattr(evidence_bank, "E", len(getattr(evidence_bank, "atoms", []))))
    if E <= 0:
        return np.zeros((0,), dtype=np.float32)
    prop = np.asarray(getattr(evidence_bank, "proposal_features", np.zeros((E, 0), dtype=np.float32)), dtype=np.float32)
    if prop.ndim != 2 or prop.shape[0] < E:
        prop2 = np.zeros((E, max(1, prop.shape[1] if prop.ndim == 2 else 1)), dtype=np.float32)
        if prop.ndim == 2 and prop.size:
            prop2[: min(E, prop.shape[0]), : prop.shape[1]] = prop[:E]
        prop = prop2
    logits = np.zeros((E,), dtype=np.float32)
    if prop.shape[1] > 0:
        # Features differ slightly across cache versions; this weighted sum uses
        # only cheap runtime proposal metadata and stays robust to missing columns.
        weights = np.asarray([2.0, -0.25, 1.0, 0.5, 0.5, 1.0, 0.25, 0.25, -0.5, 0.5, 1.0, 0.25], dtype=np.float32)
        n = min(prop.shape[1], weights.shape[0])
        logits += np.nan_to_num(prop[:, :n], nan=0.0, posinf=0.0, neginf=0.0) @ weights[:n]
    try:
        hard = np.asarray(evidence_bank.hard_mask(), dtype=bool)
        logits[: hard.shape[0]] += hard[:E].astype(np.float32) * 5.0
    except Exception:
        pass
    try:
        active = np.asarray(evidence_bank.active_mask, dtype=bool).reshape(-1)
        logits[: active.shape[0]] = np.where(active[:E], logits[: active.shape[0]], -np.inf)
    except Exception:
        pass
    return logits.astype(np.float32)


def select_budget_atoms_np(evidence_bank: Any, cfg: dict[str, Any], scores: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    E = int(getattr(evidence_bank, "E", len(getattr(evidence_bank, "atoms", []))))
    if E <= 0:
        return np.zeros((0,), dtype=np.int64), 0.0
    costs = np.asarray(evidence_bank.budget_costs(), dtype=np.float32).reshape(-1)
    if costs.shape[0] < E:
        costs = np.pad(costs, (0, E - costs.shape[0]), constant_values=1.0)
    active = np.asarray(getattr(evidence_bank, "active_mask", np.ones((E,), dtype=bool)), dtype=bool).reshape(-1)
    if active.shape[0] < E:
        active = np.pad(active, (0, E - active.shape[0]), constant_values=False)
    logits = cheap_proposal_logits_np(evidence_bank, cfg) if scores is None else np.asarray(scores, dtype=np.float32).reshape(-1)
    if logits.shape[0] < E:
        logits = np.pad(logits, (0, E - logits.shape[0]), constant_values=-np.inf)
    budget = float(external_cfg(cfg).get("budget", cfg.get("evidence", {}).get("budget", 16)))
    order = sorted([int(i) for i in np.flatnonzero(active[:E])], key=lambda i: (-float(logits[i]), float(costs[i]), i))
    selected: list[int] = []
    spent = 0.0
    for i in order:
        c = float(costs[i]) if np.isfinite(costs[i]) else 1.0
        if spent + c <= budget + 1e-6:
            selected.append(int(i))
            spent += c
    return np.asarray(selected, dtype=np.int64), float(spent)


class ExternalBaselineModel(nn.Module):
    """Budget-compatible external planning baselines implemented as candidate scorers.

    These adapters preserve the algorithmic structure of common nuPlan planning
    baselines while using this repository's runtime inputs, candidate bank, and
    evidence-budget accounting.  All variants output a cost for each candidate;
    deployment uses the existing BDSE planner/tournament shell with
    ``planner.baseline_mode=external_policy``.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.ecfg = external_cfg(cfg)
        self.variant = external_variant(cfg)
        if self.variant not in SUPPORTED_EXTERNAL_BASELINES:
            raise ValueError(f"Unsupported external baseline variant={self.variant!r}; supported={sorted(SUPPORTED_EXTERNAL_BASELINES)}")
        h = int(self.ecfg.get("hidden_dim", cfg.get("model", {}).get("hidden_dim", 256)))
        self.hidden_dim = h
        self.num_families = int(max(cfg.get("model", {}).get("num_families", 6), 8))
        self.dropout_p = float(self.ecfg.get("dropout", cfg.get("model", {}).get("dropout", 0.1)))
        self.state_dropout = float(self.ecfg.get("state_dropout", 0.0))
        self.reasoning_levels = int(self.ecfg.get("reasoning_levels", 3))
        self.tree_depth = int(self.ecfg.get("tree_depth", 2))
        self.step_s = float(cfg.get("candidate", {}).get("step_s", 0.1))
        self.budget = int(self.ecfg.get("budget", cfg.get("evidence", {}).get("budget", 16)))
        self.unit_cost = bool(cfg.get("evidence", {}).get("unit_cost", True))
        self.reference = external_reference(self.variant)

        self.ego_mlp = nn.Sequential(nn.Linear(8, h), nn.ReLU(), nn.LayerNorm(h), nn.Linear(h, h))
        self.agent_mlp = nn.Sequential(nn.Linear(16, h), nn.ReLU(), nn.LayerNorm(h), nn.Linear(h, h))
        self.map_mlp = nn.Sequential(nn.Linear(16, h), nn.ReLU(), nn.LayerNorm(h), nn.Linear(h, h))
        self.route_mlp = nn.Sequential(nn.Linear(16, h), nn.ReLU(), nn.LayerNorm(h), nn.Linear(h, h))
        self.traffic_mlp = nn.Sequential(nn.Linear(12, h), nn.ReLU(), nn.LayerNorm(h), nn.Linear(h, h))
        self.goal_mlp = nn.Sequential(nn.Linear(4, h), nn.ReLU(), nn.LayerNorm(h), nn.Linear(h, h))
        self.cand_mlp = nn.Sequential(nn.Linear(15, h), nn.ReLU(), nn.LayerNorm(h), nn.Linear(h, h))
        self.maneuver_embed = nn.Embedding(int(self.ecfg.get("num_maneuvers", 32)), h)
        self.family_embed = nn.Embedding(self.num_families, h)
        ev_dim = int(cfg.get("model", {}).get("evidence_feature_dim", 24)) + int(cfg.get("model", {}).get("proposal_feature_dim", 24))
        self.evidence_mlp = nn.Sequential(nn.Linear(ev_dim, h), nn.ReLU(), nn.LayerNorm(h), nn.Linear(h, h))
        self.proposal_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))

        layers = int(self.ecfg.get("transformer_layers", cfg.get("model", {}).get("transformer_layers", 4)))
        heads = int(self.ecfg.get("attention_heads", cfg.get("model", {}).get("attention_heads", 8)))
        ff = int(self.ecfg.get("ff_dim", 4 * h))
        enc_layer = nn.TransformerEncoderLayer(d_model=h, nhead=heads, dim_feedforward=ff, dropout=self.dropout_p, batch_first=True, norm_first=True)
        self.scene_transformer = nn.TransformerEncoder(enc_layer, num_layers=max(1, layers))
        self.cross_attn = nn.MultiheadAttention(h, heads, dropout=self.dropout_p, batch_first=True)
        self.update_gru = nn.GRUCell(h, h)
        self.scene_token = nn.Parameter(torch.zeros(1, 1, h))
        self.cost_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.maneuver_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, int(self.ecfg.get("num_maneuvers", 32))))
        self.dtpp_branch_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, int(self.ecfg.get("num_maneuvers", 32))))
            for _ in range(max(1, self.tree_depth))
        ])
        if self.variant == "pluto":
            self.long_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))
            self.lat_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))
        else:
            self.long_head = None
            self.lat_head = None

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor | None, dims: tuple[int, ...]) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=dims)
        m = mask.bool()
        while m.ndim < x.ndim:
            m = m.unsqueeze(-1)
        mf = m.float()
        return (x * mf).sum(dim=dims) / mf.sum(dim=dims).clamp_min(1.0)

    def _scene_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ego = batch["ego_history"].float()
        ego_cur = _fit_last_dim_torch(ego[:, -1], 8)
        if self.training and self.state_dropout > 0.0:
            drop = (torch.rand_like(ego_cur) > self.state_dropout).float()
            ego_cur = ego_cur * drop
        agent = batch.get("agent_history")
        agent_valid = batch.get("agent_valid")
        if agent is not None:
            agent_cur = _fit_last_dim_torch(agent.float()[:, :, -1], 8)
            if agent_valid is not None:
                av = agent_valid.bool()
                if av.ndim == 3:
                    av = av[:, :, -1]
            else:
                av = torch.ones(agent_cur.shape[:2], device=agent_cur.device, dtype=torch.bool)
            agent_mean = self._masked_mean(agent_cur, av, dims=(1,))
            agent_max = torch.where(av.unsqueeze(-1), agent_cur, torch.zeros_like(agent_cur)).amax(dim=1)
            agent_feat = torch.cat([agent_mean, agent_max], dim=-1)
        else:
            B = ego.shape[0]
            agent_feat = torch.zeros(B, 16, device=ego.device, dtype=ego.dtype)
        map_poly = batch.get("map_polylines")
        map_valid = batch.get("map_polyline_valid")
        if map_poly is not None:
            mp = _fit_last_dim_torch(map_poly.float(), 8)
            poly_mean = mp.mean(dim=2)
            map_mean = self._masked_mean(poly_mean, map_valid, dims=(1,)) if map_valid is not None else poly_mean.mean(dim=1)
            map_max = torch.where(map_valid.bool().unsqueeze(-1), poly_mean, torch.zeros_like(poly_mean)).amax(dim=1) if map_valid is not None else poly_mean.amax(dim=1)
            map_feat = torch.cat([map_mean, map_max], dim=-1)
        else:
            map_feat = torch.zeros(ego.shape[0], 16, device=ego.device, dtype=ego.dtype)
        route = batch.get("route_polylines")
        route_valid = batch.get("route_token_valid")
        if route is not None:
            rt = _fit_last_dim_torch(route.float(), 8)
            rt_mean_poly = rt.mean(dim=2)
            route_mean = self._masked_mean(rt_mean_poly, route_valid, dims=(1,)) if route_valid is not None else rt_mean_poly.mean(dim=1)
            route_max = torch.where(route_valid.bool().unsqueeze(-1), rt_mean_poly, torch.zeros_like(rt_mean_poly)).amax(dim=1) if route_valid is not None else rt_mean_poly.amax(dim=1)
            route_feat = torch.cat([route_mean, route_max], dim=-1)
        else:
            route_feat = torch.zeros(ego.shape[0], 16, device=ego.device, dtype=ego.dtype)
        tl = batch.get("traffic_control_tokens")
        tlv = batch.get("traffic_token_valid")
        if tl is not None:
            traffic_feat = self._masked_mean(_fit_last_dim_torch(tl.float(), 12), tlv, dims=(1,)) if tlv is not None else _fit_last_dim_torch(tl.float(), 12).mean(dim=1)
        else:
            traffic_feat = torch.zeros(ego.shape[0], 12, device=ego.device, dtype=ego.dtype)
        goal = _fit_last_dim_torch(batch.get("mission_goal", torch.zeros(ego.shape[0], 4, device=ego.device, dtype=ego.dtype)).float(), 4)
        return (
            self.ego_mlp(ego_cur)
            + self.agent_mlp(agent_feat)
            + self.map_mlp(map_feat)
            + self.route_mlp(route_feat)
            + self.traffic_mlp(traffic_feat)
            + self.goal_mlp(goal)
        )

    def _evidence_tokens(self, batch: dict[str, torch.Tensor], scene: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ev = batch["evidence_features"].float()
        prop = batch.get("evidence_proposal_features", torch.zeros_like(ev)).float()
        active = batch.get("evidence_active", torch.ones(ev.shape[:2], dtype=torch.bool, device=ev.device)).bool()
        fam = batch.get("evidence_family_ids", torch.zeros(ev.shape[:2], dtype=torch.long, device=ev.device)).long().clamp_min(0).clamp_max(self.num_families - 1)
        token = self.evidence_mlp(torch.cat([ev, prop], dim=-1)) + self.family_embed(fam)
        logits = self.proposal_head(token + scene[:, None, :]).squeeze(-1)
        # Keep the neural proposal aligned with cheap hard/active priors.
        hard_bonus = 5.0 * ev[..., 0].clamp(0.0, 1.0)
        # Under AMP the proposal head can produce fp16 logits.  fp16 cannot
        # represent -1e9, so do mask/sentinel arithmetic in fp32 while leaving
        # the network forward itself unchanged.
        logits = (logits + hard_bonus).float()
        logits = logits.masked_fill(~active, logits.new_tensor(-1e9))
        return token, logits, active

    def _top_budget_selection(
        self, logits: torch.Tensor, active: torch.Tensor, budget_costs: torch.Tensor, budget_override: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mask, ranked selected indices and valid-slot mask without GPU/CPU sync.

        BDSE caches use unit evidence cost by default.  That path is an exact, fully
        vectorized top-k.  The variable-cost fallback keeps the greedy semantics but
        loops only over evidence rank while updating the whole batch on-device.
        """
        B, E = logits.shape
        active = active.bool()
        effective_budget = float(self.budget if budget_override is None else budget_override)
        masked = logits.float().masked_fill(~active, -torch.inf)
        if E == 0:
            empty_i = torch.zeros((B, 1), dtype=torch.long, device=logits.device)
            empty_v = torch.zeros((B, 1), dtype=torch.bool, device=logits.device)
            return active.clone(), empty_i, empty_v

        if self.unit_cost:
            slots = max(1, min(E, int(effective_budget)))
            values, indices = torch.topk(masked, k=slots, dim=1, largest=True, sorted=True)
            slot_valid = torch.isfinite(values)
            selected = torch.zeros((B, E), dtype=torch.bool, device=logits.device)
            selected.scatter_(1, indices, slot_valid)
            return selected, indices, slot_valid

        order = torch.argsort(masked, dim=1, descending=True)
        ordered_cost = torch.gather(torch.nan_to_num(budget_costs.float(), nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-6), 1, order)
        ordered_active = torch.gather(active, 1, order)
        accepted = torch.zeros((B, E), dtype=torch.bool, device=logits.device)
        spent = torch.zeros((B,), dtype=ordered_cost.dtype, device=logits.device)
        budget = ordered_cost.new_full((B,), effective_budget)
        for rank in range(E):
            take = ordered_active[:, rank] & torch.isfinite(masked.gather(1, order[:, rank : rank + 1]).squeeze(1))
            take = take & (spent + ordered_cost[:, rank] <= budget + 1e-6)
            accepted[:, rank] = take
            spent = spent + torch.where(take, ordered_cost[:, rank], torch.zeros_like(spent))
        selected = torch.zeros((B, E), dtype=torch.bool, device=logits.device)
        selected.scatter_(1, order, accepted)
        # Keeping ranked E slots avoids a data-dependent CPU synchronization.  Padding
        # masks ensure unselected slots do not affect attention.
        return selected, order, accepted

    def _candidate_tokens(self, batch: dict[str, torch.Tensor], scene: torch.Tensor) -> torch.Tensor:
        traj = batch["candidate_trajectories"].float()
        valid = batch["candidate_valid"].bool()
        cfeat = candidate_numeric_features_torch(traj, valid, self.step_s)
        mid = batch.get("candidate_maneuver_ids", torch.zeros(valid.shape, dtype=torch.long, device=valid.device)).long().clamp_min(0).clamp_max(self.maneuver_embed.num_embeddings - 1)
        tok = self.cand_mlp(cfeat) + self.maneuver_embed(mid) + scene[:, None, :]
        return tok

    def forward(self, batch: dict[str, torch.Tensor], budget_override: float | None = None) -> dict[str, Any]:
        valid = batch["candidate_valid"].bool()
        scene = self._scene_embedding(batch)
        cand = self._candidate_tokens(batch, scene)
        ev_tokens, prop_logits, ev_active = self._evidence_tokens(batch, scene)
        costs = batch.get("evidence_budget_costs", torch.ones_like(prop_logits)).float().clamp_min(1e-6)
        selected_mask, selected_indices, selected_valid = self._top_budget_selection(prop_logits, ev_active, costs, budget_override=budget_override)
        gather_index = selected_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        ev_sel = torch.gather(ev_tokens, 1, gather_index)
        ev_sel = torch.where(selected_valid.unsqueeze(-1), ev_sel, torch.zeros_like(ev_sel))
        # MultiheadAttention cannot consume a row with every key masked.  Reserve one
        # zero context token only for genuinely empty scenes.
        empty = ~selected_valid.any(dim=1)
        first_slot = torch.zeros_like(selected_valid)
        first_slot[:, 0] = True
        selected_valid = selected_valid | (empty[:, None] & first_slot)
        selected_padding = ~selected_valid
        candidate_padding = ~valid
        scene_tok = self.scene_token.expand(cand.shape[0], -1, -1) + scene[:, None, :]
        auxiliary_costs: list[torch.Tensor] = []

        if self.variant == "pdm_closed":
            # Neural forward is not used for PDM.  Keep a simple differentiable
            # equivalent for py_compile/tests if somebody calls it.
            j = -batch["candidate_trajectories"][:, :, -1, 0].float() * 0.05 + batch["candidate_trajectories"][:, :, :, 1].float().abs().mean(dim=-1)
        elif self.variant in {"plantf", "pluto"}:
            tokens = torch.cat([scene_tok, cand, ev_sel], dim=1)
            padding = torch.cat([torch.zeros((valid.shape[0], 1), dtype=torch.bool, device=valid.device), candidate_padding, selected_padding], dim=1)
            enc = self.scene_transformer(tokens, src_key_padding_mask=padding)
            cand_enc = enc[:, 1 : 1 + cand.shape[1]]
            j = self.cost_head(cand_enc).squeeze(-1)
            if self.variant == "pluto" and self.long_head is not None and self.lat_head is not None:
                # Longitudinal-lateral aware decomposition: a trajectory can be
                # good longitudinally but fail laterally near route/corridor.
                j = j + self.long_head(cand_enc).squeeze(-1) + self.lat_head(cand_enc).squeeze(-1)
        elif self.variant == "gameformer":
            cand_ctx = cand
            cand_padding = torch.cat([torch.zeros((valid.shape[0], 1), dtype=torch.bool, device=valid.device), candidate_padding], dim=1)
            for _ in range(max(1, self.reasoning_levels)):
                attn, _ = self.cross_attn(cand_ctx, ev_sel, ev_sel, key_padding_mask=selected_padding, need_weights=False)
                cand_ctx = cand_ctx + attn
                cand_ctx = self.scene_transformer(torch.cat([scene_tok, cand_ctx], dim=1), src_key_padding_mask=cand_padding)[:, 1:]
                auxiliary_costs.append(self.cost_head(cand_ctx).squeeze(-1))
            j = auxiliary_costs[-1]
        elif self.variant == "dtpp":
            mid = batch.get("candidate_maneuver_ids", torch.zeros(valid.shape, dtype=torch.long, device=valid.device)).long().clamp_min(0).clamp_max(self.maneuver_head[-1].out_features - 1)
            cand_ctx = cand
            scene_ctx = scene_tok[:, 0]
            cand_padding = torch.cat([torch.zeros((valid.shape[0], 1), dtype=torch.bool, device=valid.device), candidate_padding], dim=1)
            for depth, branch_head in enumerate(self.dtpp_branch_heads):
                # A budget-compatible tree-policy analogue: each stage updates the
                # candidate branch state from selected interaction evidence, then
                # adds a maneuver-level branch cost and trajectory refinement cost.
                attn, _ = self.cross_attn(cand_ctx, ev_sel, ev_sel, key_padding_mask=selected_padding, need_weights=False)
                enc = self.scene_transformer(torch.cat([scene_ctx[:, None, :], cand_ctx + attn], dim=1), src_key_padding_mask=cand_padding)
                scene_ctx, cand_ctx = enc[:, 0], enc[:, 1:]
                maneuver_cost = torch.gather(branch_head(scene_ctx), 1, mid)
                auxiliary_costs.append(maneuver_cost + self.cost_head(cand_ctx).squeeze(-1))
            j = auxiliary_costs[-1]
        elif self.variant == "ppad":
            iters = int(self.ecfg.get("iterations", 4))
            cand_ctx = cand
            for _ in range(max(1, iters)):
                attn, _ = self.cross_attn(cand_ctx, ev_sel, ev_sel, key_padding_mask=selected_padding, need_weights=False)
                flat = cand_ctx.reshape(-1, self.hidden_dim)
                upd = self.update_gru(attn.reshape(-1, self.hidden_dim), flat).reshape_as(cand_ctx)
                cand_ctx = 0.5 * cand_ctx + 0.5 * upd
            j = self.cost_head(cand_ctx).squeeze(-1)
        else:  # pragma: no cover
            raise AssertionError(self.variant)
        # Cost masking also runs inside autocast.  Keep the sentinel value at
        # 1e6 for the training objective, but cast to fp32 first because fp16
        # max is only 65504.
        j = j.float()
        j = torch.nan_to_num(j, nan=0.0, posinf=1e6, neginf=-1e6)
        j = j.masked_fill(~valid, j.new_tensor(1e6))
        result: dict[str, torch.Tensor | str] = {
            "J0": j,
            "proposal_logits": prop_logits,
            "external_selected_mask": selected_mask,
            "external_selected_indices": selected_indices,
            "external_selected_valid": selected_valid,
            "external_variant": self.variant,
            "external_implementation_label": self.reference.get("implementation_label", self.variant),
            "external_fidelity": self.reference.get("fidelity", "unspecified"),
        }
        if auxiliary_costs:
            result["external_aux_costs"] = torch.stack([x.float().masked_fill(~valid, 1e6) for x in auxiliary_costs], dim=1)
        return result

    def _numpy_pred_common(self, runtime: RuntimeFeatures, candidates: Any, evidence_bank: Any, cfg: dict[str, Any], J0: np.ndarray, proposal_logits: np.ndarray) -> dict[str, Any]:
        E = int(getattr(evidence_bank, "E", len(getattr(evidence_bank, "atoms", []))))
        K = int(getattr(candidates, "K", len(np.asarray(getattr(candidates, "valid_mask", [])))))
        valid = np.asarray(candidates.valid_mask, dtype=bool)
        J0 = np.asarray(J0, dtype=np.float32).reshape(-1)
        if J0.shape[0] < K:
            J0 = np.pad(J0, (0, K - J0.shape[0]), constant_values=np.inf)
        J0 = J0[:K]
        J0[~valid] = np.inf
        runtime_flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
        pairs, pair_weights = build_runtime_pairs_from_base(
            J0,
            valid,
            runtime_flags,
            L0=int(cfg.get("tournament", {}).get("L_infer", 16)),
            eta0=float(cfg.get("selector", {}).get("eta_pred", 1.0)),
            lambda_near=float(cfg.get("selector", {}).get("lambda_near", 1.0)),
            lambda_safety=float(cfg.get("selector", {}).get("lambda_safety", 2.0)),
            preserve_safety_pairs=bool(cfg.get("selector", {}).get("preserve_safety_pairs", True)),
            bidirectional_pairs=bool(cfg.get("selector", {}).get("bidirectional_pairs", True)),
            reverse_pair_weight=float(cfg.get("selector", {}).get("reverse_pair_weight", 1.0)),
            pair_cap_multiplier=float(cfg.get("selector", {}).get("runtime_pair_cap_multiplier", 1.0)),
            candidate_trajectories=candidates.trajectories,
            maneuver_ids=candidates.maneuver_ids,
            progress_pair_count=int(cfg.get("selector", {}).get("progress_pair_count", 0)),
            maneuver_pair_count=int(cfg.get("selector", {}).get("maneuver_pair_count", 0)),
        )
        rival_sets = build_rival_sets_from_base(
            J0,
            valid,
            runtime_flags,
            L_infer=int(cfg.get("tournament", {}).get("L_infer", 16)),
            eta0=float(cfg.get("selector", {}).get("eta_pred", 1.0)),
            candidate_trajectories=candidates.trajectories,
            maneuver_ids=candidates.maneuver_ids,
            progress_rivals=int(cfg.get("selector", {}).get("progress_rivals", 0)),
            maneuver_rivals=int(cfg.get("selector", {}).get("maneuver_rivals", 0)),
        )
        rival_pair_list: list[tuple[int, int]] = []
        for a_idx, rivals in enumerate(rival_sets):
            if not bool(valid[a_idx]) or not rivals:
                continue
            for r in rivals:
                rival_pair_list.append((int(a_idx), int(r)))
        rival_pairs = np.asarray(rival_pair_list, dtype=np.int64).reshape(-1, 2) if rival_pair_list else np.zeros((0, 2), dtype=np.int64)
        selected, spent = select_budget_atoms_np(evidence_bank, cfg, proposal_logits)
        M = int(cfg.get("selector", {}).get("proposal_top_m", max(2 * int(external_cfg(cfg).get("budget", 16)), 1)))
        finite_order = np.argsort(np.where(np.isfinite(proposal_logits), -proposal_logits, np.inf))[: max(0, min(M, E))]
        active = np.asarray(getattr(evidence_bank, "active_mask", np.ones((E,), dtype=bool)), dtype=bool).reshape(-1)
        topm = np.asarray([int(i) for i in finite_order if i < active.shape[0] and active[i]], dtype=np.int64)
        queried_actions = np.flatnonzero(valid).astype(np.int64)
        family_ids = family_ids_from_atoms(evidence_bank.atoms, max_atoms=E)
        try:
            hard = np.asarray(evidence_bank.hard_mask(), dtype=bool)
        except Exception:
            hard = np.zeros((E,), dtype=bool)
        mandatory = structural_safety_mask(
            hard,
            family_ids,
            active,
            include_feasibility=bool(cfg.get("selector", {}).get("structural_safety_include_feasibility", True)),
        )
        zero_g = np.zeros((E, K), dtype=np.float32)
        return {
            "J0": J0,
            "g": zero_g,
            "g_var": np.zeros((E, K), dtype=np.float32),
            "proposal_logits": proposal_logits.astype(np.float32),
            "family_ids": family_ids.astype(np.int64),
            "family_budget_caps": None,
            "mandatory_atom_mask": np.asarray(mandatory, dtype=bool)[:E],
            "top_m_atoms": topm.astype(np.int64),
            "queried_actions": queried_actions,
            "runtime_pairs": pairs.astype(np.int64),
            "runtime_pair_weights": pair_weights.astype(np.float32),
            "rival_pair_indices": rival_pairs.astype(np.int64),
            "action_atom_query_count": int(len(topm) * len(queried_actions)),
            "selector_pair_atom_query_count": int(len(topm) * len(pairs)),
            "tournament_pair_atom_query_count": int(len(topm) * len(rival_pairs)),
            "external_selected_atoms": selected.astype(np.int64),
            "external_spent_budget": float(spent),
            "external_variant": self.variant,
            "external_implementation_label": self.reference.get("implementation_label", self.variant),
            "external_fidelity": self.reference.get("fidelity", "unspecified"),
        }

    @torch.no_grad()
    def predict_certificate_numpy(self, runtime: RuntimeFeatures, candidates: Any, evidence_bank: Any, cfg: dict[str, Any]) -> dict[str, Any]:
        if self.variant == "pdm_closed":
            flags = runtime_safety_flags_from_runtime(runtime, candidates, cfg)
            J0 = candidate_rule_cost_np(candidates, flags, cfg)
            proposal_logits = cheap_proposal_logits_np(evidence_bank, cfg)
            return self._numpy_pred_common(runtime, candidates, evidence_bank, cfg, J0, proposal_logits)
        arrays = runtime_to_model_numpy(runtime, candidates, evidence_bank, cfg, include_dense_query=False)
        device = next(self.parameters()).device
        batch: dict[str, torch.Tensor] = {}
        for k, v in arrays.items():
            arr = np.asarray(v)
            if arr.dtype == np.bool_:
                t = torch.from_numpy(arr.astype(bool)).to(device)
            elif np.issubdtype(arr.dtype, np.integer):
                t = torch.from_numpy(arr.astype(np.int64)).to(device)
            else:
                t = torch.from_numpy(arr.astype(np.float32)).to(device)
            batch[k] = t.unsqueeze(0)
        runtime_budget = float(external_cfg(cfg).get("budget", cfg.get("evidence", {}).get("budget", self.budget)))
        out = self.forward(batch, budget_override=runtime_budget)
        J0 = out["J0"][0].detach().cpu().numpy().astype(np.float32)
        proposal = out["proposal_logits"][0].detach().cpu().numpy().astype(np.float32)
        return self._numpy_pred_common(runtime, candidates, evidence_bank, cfg, J0, proposal)

    @torch.no_grad()
    def predict_dense_numpy(self, runtime: RuntimeFeatures, candidates: Any, evidence_bank: Any, cfg: dict[str, Any]) -> dict[str, Any]:
        pred = self.predict_certificate_numpy(runtime, candidates, evidence_bank, cfg)
        return {"J0": pred["J0"], "g": pred["g"]}
