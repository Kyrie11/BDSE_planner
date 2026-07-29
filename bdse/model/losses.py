from __future__ import annotations

from typing import Any
import time

import numpy as np
import torch
import torch.nn.functional as F

from bdse.model.residual_gate import confidence_shrunk_residual_pair_delta_numpy, confidence_shrunk_residual_pair_delta_torch

from bdse.planner.hab import select_topm_atoms_hab
from bdse.planner.pair_screen import build_runtime_pairs_from_base, build_rival_sets_from_base, restrict_pairs_to_viability_frontier
from bdse.planner.selector import (
    margin_normalization_scale,
    reserve_topm_candidates,
    restrict_topm_to_decision_evidence,
    runtime_greedy_selector,
    runtime_greedy_selector_pair_conditioned,
    structural_safety_mask,
)


def _to_numpy(t: torch.Tensor | None, dtype: Any | None = None) -> np.ndarray | None:
    """Detach once and convert to NumPy without an extra dtype copy when possible."""
    if t is None:
        return None
    arr = t.detach().cpu().numpy()
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def _packed_numpy_snapshot(
    tensors: dict[str, tuple[torch.Tensor | None, torch.dtype]],
) -> dict[str, np.ndarray | None]:
    """Copy heterogeneous selector inputs to CPU with one sync per dtype.

    The exact deployment selector intentionally runs on NumPy/CPU.  Copying
    each CUDA tensor independently serializes the stream once per ``.cpu()``
    call, which made the selector path pay more than a dozen synchronization
    points per optimizer step.  Packing tensors by destination dtype preserves
    every value and shape while reducing that to at most float/bool/int syncs.
    Returned arrays are views of the packed CPU buffers and remain valid for
    the lifetime of the result dictionary.
    """
    result: dict[str, np.ndarray | None] = {
        name: None for name, (tensor, _) in tensors.items() if tensor is None
    }
    groups: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for name, (tensor, dtype) in tensors.items():
        if tensor is not None:
            groups.setdefault(dtype, []).append((name, tensor.detach()))

    for dtype, entries in groups.items():
        shapes = [tuple(tensor.shape) for _, tensor in entries]
        sizes = [int(tensor.numel()) for _, tensor in entries]
        flat = [
            tensor.to(dtype=dtype, copy=False).reshape(-1)
            for _, tensor in entries
        ]
        packed = torch.cat(flat, dim=0) if len(flat) > 1 else flat[0]
        packed_np = packed.cpu().numpy()
        offset = 0
        for (name, _), shape, size in zip(entries, shapes, sizes):
            result[name] = packed_np[offset : offset + size].reshape(shape)
            offset += size
    return result


def robust_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    mask = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
    safe_pred = torch.where(mask, pred, torch.zeros_like(pred))
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    terms = F.huber_loss(safe_pred, safe_target, delta=delta, reduction="none")
    weights = mask.to(dtype=terms.dtype)
    return (terms * weights).sum() / weights.sum().clamp_min(1.0)


def pair_gather(values: torch.Tensor, pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, P, _ = pairs.shape
    a = pairs[..., 0].long()
    b = pairs[..., 1].long()
    if values.dim() == 2:
        va = torch.gather(values, 1, a)
        vb = torch.gather(values, 1, b)
        return va, vb
    if values.dim() == 3:
        E = values.shape[1]
        idx_a = a[:, None, :].expand(B, E, P)
        idx_b = b[:, None, :].expand(B, E, P)
        va = torch.gather(values, 2, idx_a)
        vb = torch.gather(values, 2, idx_b)
        return va, vb
    raise ValueError("values must be [B,K] or [B,E,K]")


def full_interface_predicted_margin(J0: torch.Tensor, g: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    J_a, J_b = pair_gather(J0, pairs)
    g_a, g_b = pair_gather(g, pairs)
    return (J_b - J_a) + (g_b - g_a).sum(dim=1)


def selected_budget_margin(J0: torch.Tensor, g: torch.Tensor, pairs: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
    J_a, J_b = pair_gather(J0, pairs)
    g_a, g_b = pair_gather(g, pairs)
    support = ((g_b - g_a) * selected_mask[:, :, None].float()).sum(dim=1)
    return (J_b - J_a) + support


def _neg_mask_value(x: torch.Tensor) -> float:
    if torch.is_floating_point(x):
        return float(torch.finfo(x.dtype).min / 2.0)
    return -1e9


def _predicted_certificate_masks(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop-gradient deployment certificate masks for L_act.

    Dense ``g`` is available during training, but the deployment path only sees
    scores for HAB Top-M atoms and actions contained in the base rival graph.
    This helper mirrors that path and returns the selected atom mask plus the
    sparse atom-action query mask, preventing dense-label leakage into action
    supervision.
    """
    J0 = _to_numpy(outputs["J0"], np.float32)
    g = _to_numpy(outputs["g"], np.float32)
    g_var = outputs.get("g_var")
    g_var_np = _to_numpy(g_var, np.float32) if (g_var is not None and (float(cfg.get("tournament", {}).get("beta_uncertainty", 0.0)) > 0.0 or float(cfg.get("selector", {}).get("lambda_info", 0.0)) > 0.0)) else None
    logits = _to_numpy(outputs["proposal_logits"], np.float32)
    family_logits = outputs.get("family_logits")
    family_logits_np = _to_numpy(family_logits, np.float32) if family_logits is not None else None
    valid = _to_numpy(batch["candidate_valid"], bool)
    active = _to_numpy(batch.get("evidence_active", torch.ones_like(outputs["proposal_logits"]).bool()), bool)
    costs = _to_numpy(batch.get("evidence_budget_costs", torch.ones_like(outputs["proposal_logits"])), np.float32)
    fam_ids_t = batch.get("evidence_family_ids")
    fam_ids_np = _to_numpy(fam_ids_t, np.int64) if fam_ids_t is not None else np.zeros_like(logits, dtype=np.int64)
    group_ids_t = batch.get("evidence_agent_group_ids")
    group_ids_np = _to_numpy(group_ids_t, np.int64) if group_ids_t is not None else np.full_like(fam_ids_np, -1, dtype=np.int64)
    B, E = logits.shape
    K = valid.shape[1]
    traj_np = batch.get("candidate_trajectories")
    traj_np = _to_numpy(traj_np, np.float32) if traj_np is not None else None
    man_np = batch.get("candidate_maneuver_ids")
    man_np = _to_numpy(man_np, np.int64) if man_np is not None else None
    flags_t = batch.get("runtime_safety_flags")
    flags_np_all = _to_numpy(flags_t, bool) if flags_t is not None else None
    evidence_features_np = _to_numpy(batch.get("evidence_features"), np.float32) if "evidence_features" in batch else None
    decisive_hard_np = _to_numpy(batch.get("decisive_hard_mask"), bool) if "decisive_hard_mask" in batch else None
    selected_mask = np.zeros((B, E), dtype=bool)
    query_mask = np.zeros((B, E, K), dtype=bool)

    e_cfg = cfg.get("evidence", {})
    s_cfg = cfg.get("selector", {})
    train_cfg = cfg.get("training", {})
    t_cfg = cfg.get("tournament", {})
    c_cfg = cfg.get("calibration", {})
    budget = float(e_cfg.get("budget", 16))
    M = int(s_cfg.get("proposal_top_m", max(int(2 * budget), int(budget) + 1)))
    L0 = int(t_cfg.get("L_infer", 16))
    eta = float(s_cfg.get("eta_pred", 1.0))
    gamma = float(s_cfg.get("gamma_max_default", 100.0))
    lambda_near = float(s_cfg.get("lambda_near", 1.0))
    lambda_safety = float(s_cfg.get("lambda_safety", 2.0))
    beta_unc = float(t_cfg.get("beta_uncertainty", 0.0))
    eps_cal = float(t_cfg.get("epsilon_cal", c_cfg.get("epsilon_cal", 0.0)))
    lambda_info = float(s_cfg.get("lambda_info", 0.0))
    prior_var = s_cfg.get("unqueried_atom_variance", None)

    for bidx in range(B):
        structural_bypass = bool(s_cfg.get("decision_budget_excludes_structural_safety", False))
        topm, fam_budget, _ = select_topm_atoms_hab(
            logits[bidx],
            fam_ids_np[bidx],
            active[bidx],
            costs[bidx],
            budget,
            M,
            family_scores=family_logits_np[bidx] if family_logits_np is not None else None,
            free_budget=s_cfg.get("hab_free_budget", None),
            reserve_fraction=float(s_cfg.get("hab_reserve_fraction", 0.2)),
            enabled=bool(s_cfg.get("hab_enabled", True)),
            min_family_slots=s_cfg.get("min_family_topm_slots", None),
        )
        # Mirror the hybrid selector used at deployment: hard/rule atoms that
        # support supervised margins must remain queryable even when proposal
        # logits are still immature.  This is not dense leakage into the final
        # certificate; it only expands the Top-M candidate pool before the same
        # budgeted greedy selector runs.
        if (not structural_bypass) and bool(s_cfg.get("force_decisive_hard_topm", True)) and "decisive_hard_mask" in batch:
            hard_train = (decisive_hard_np[bidx] if decisive_hard_np is not None else np.zeros((E,), dtype=bool)) & active[bidx]
            forced = np.flatnonzero(hard_train)
            if forced.size:
                forced_cap = int(s_cfg.get("max_forced_hard_topm", max(1, M // 2)))
                forced = np.asarray(sorted(forced.tolist(), key=lambda i: (-float(logits[bidx, i]), int(i)))[:forced_cap], dtype=np.int64)
                non_forced = [int(i) for i in np.asarray(topm, dtype=np.int64).reshape(-1).tolist() if int(i) not in set(forced.tolist())]
                topm = np.asarray((forced.tolist() + non_forced)[:M], dtype=np.int64)
        hard_feature_for_pool = evidence_features_np[bidx, :, 0] > 0.5 if evidence_features_np is not None else np.zeros((E,), dtype=bool)
        interaction_family_set = set(int(x) for x in s_cfg.get("interaction_family_ids", [2, 3]))
        soft_interaction_pool = np.asarray(
            [int(f) in interaction_family_set for f in fam_ids_np[bidx].tolist()], dtype=bool
        ) & active[bidx] & ~hard_feature_for_pool
        min_soft_topm = int(s_cfg.get("min_soft_interaction_topm_slots", 0))
        if min_soft_topm > 0 and bool(soft_interaction_pool.any()):
            protected_for_pool = structural_safety_mask(
                hard_feature_for_pool,
                fam_ids_np[bidx],
                active[bidx],
                include_feasibility=bool(s_cfg.get("structural_safety_include_feasibility", True)),
            )
            topm, _ = reserve_topm_candidates(
                topm,
                soft_interaction_pool,
                logits[bidx],
                M,
                min_soft_topm,
                protected_mask=protected_for_pool,
                group_ids=group_ids_np[bidx],
            )
        if structural_bypass:
            structural_for_pool = structural_safety_mask(
                hard_feature_for_pool,
                fam_ids_np[bidx],
                active[bidx],
                include_feasibility=bool(s_cfg.get("structural_safety_include_feasibility", True)),
            )
            topm, _ = restrict_topm_to_decision_evidence(
                topm,
                active[bidx] & ~structural_for_pool,
                logits[bidx],
                M,
                family_ids=fam_ids_np[bidx],
            )
        atom_active = np.zeros((E,), dtype=bool)
        atom_active[topm] = True
        atom_active &= active[bidx]
        flag_np = flags_np_all[bidx] if flags_np_all is not None else np.zeros((K,), dtype=bool)
        pairs, _ = build_runtime_pairs_from_base(
            J0[bidx], valid[bidx], flag_np, L0=L0, eta0=eta, lambda_near=lambda_near, lambda_safety=lambda_safety,
            candidate_trajectories=traj_np[bidx] if traj_np is not None else None,
            maneuver_ids=man_np[bidx] if man_np is not None else None,
            progress_pair_count=int(s_cfg.get("progress_pair_count", 8)),
            maneuver_pair_count=int(s_cfg.get("maneuver_pair_count", 8)),
        )
        rival_sets = build_rival_sets_from_base(
            J0[bidx], valid[bidx], flag_np, L_infer=L0, eta0=eta,
            candidate_trajectories=traj_np[bidx] if traj_np is not None else None,
            maneuver_ids=man_np[bidx] if man_np is not None else None,
            progress_rivals=int(s_cfg.get("progress_rivals", 4)),
            maneuver_rivals=int(s_cfg.get("maneuver_rivals", 4)),
        )
        action_set: set[int] = set()
        for a_idx, rivals in enumerate(rival_sets):
            if not bool(valid[bidx, a_idx]) or not rivals:
                continue
            action_set.add(int(a_idx))
            action_set.update(int(r) for r in rivals)
        if action_set:
            action_ids = np.asarray(sorted(action_set), dtype=np.int64)
            for ei in np.flatnonzero(atom_active):
                query_mask[bidx, ei, action_ids] = True
        elif len(pairs):
            action_ids = np.unique(pairs.reshape(-1))
            for ei in np.flatnonzero(atom_active):
                query_mask[bidx, ei, action_ids] = True
        g_sparse = np.where(query_mask[bidx], g[bidx], 0.0)
        var_sparse = np.where(query_mask[bidx], g_var_np[bidx], 0.0) if g_var_np is not None else None
        hard_feature = evidence_features_np[bidx, :, 0] > 0.5 if evidence_features_np is not None else np.zeros((E,), dtype=bool)
        mandatory = structural_safety_mask(
            hard_feature,
            fam_ids_np[bidx],
            active[bidx],
            include_feasibility=bool(s_cfg.get("structural_safety_include_feasibility", True)),
        )
        if (not structural_bypass) and "decisive_hard_mask" in batch and bool(s_cfg.get("force_decisive_hard_topm", True)):
            mandatory = mandatory | ((decisive_hard_np[bidx] if decisive_hard_np is not None else np.zeros((E,), dtype=bool)) & active[bidx])
        if structural_bypass:
            atom_active &= ~mandatory
        result = runtime_greedy_selector(
            J0[bidx],
            g_sparse,
            costs[bidx],
            valid[bidx],
            flag_np,
            budget,
            L_infer=L0,
            gamma_max=gamma,
            eta_pred=eta,
            lambda_near=lambda_near,
            lambda_safety=lambda_safety,
            atom_active_mask=atom_active,
            bidirectional_pairs=bool(s_cfg.get("bidirectional_pairs", True)),
            reverse_pair_weight=float(s_cfg.get("reverse_pair_weight", 1.0)),
            pair_cap_multiplier=float(s_cfg.get("runtime_pair_cap_multiplier", 1.0)),
            predicted_atom_variance=var_sparse,
            beta_uncertainty=beta_unc,
            epsilon_cal=eps_cal,
            lambda_info=lambda_info,
            prior_atom_variance=prior_var,
            family_ids=fam_ids_np[bidx],
            family_budget_caps=fam_budget.family_caps,
            mandatory_atom_mask=None if structural_bypass else mandatory,
            mandatory_quota=0 if structural_bypass else int(s_cfg.get("mandatory_hard_quota", 0)),
            min_selected_atoms=int(s_cfg.get("min_selected_atoms", 0)),
            force_fill_budget=bool(s_cfg.get("force_fill_budget", False)),
            prioritize_mandatory_fill=bool(s_cfg.get("prioritize_mandatory_fill", True)),
        )
        selected_mask[bidx, result.selected] = True
    device = outputs["J0"].device
    return torch.from_numpy(selected_mask).to(device), torch.from_numpy(query_mask).to(device)




def _slice_scene_batch(
    values: dict[str, torch.Tensor],
    scene_indices: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Slice only tensors whose leading dimension is the scene batch.

    The deployment selector is a stop-gradient CPU routine.  Copying a full
    training batch to CPU when only a small rotating subset needs exact selector
    supervision wastes PCIe bandwidth and blocks both DDP ranks.  This helper
    performs the slice on-device first so only selected scenes cross to CPU.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in values.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == int(batch_size):
            out[key] = value.index_select(0, scene_indices)
        else:
            out[key] = value
    return out


def _deployment_selector_scene_indices(
    batch_size: int,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Return rotating scenes that receive exact predicted-selector supervision.

    ``deployment_selector_scenes_per_rank=0`` means all scenes.  Before the
    optional final full-alignment epoch, a small deterministic subset is selected
    every N steps.  The remaining scenes keep the existing oracle curriculum,
    yielding a low-variance mixed objective without running the CPU selector for
    every scene on every optimizer step.
    """
    if batch_size <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    epoch = int(train_cfg.get("current_epoch", 0))
    step = int(train_cfg.get("global_step", 0))
    full_start = int(train_cfg.get("deployment_selector_full_start_epoch", 10**9))
    if epoch >= full_start:
        return torch.arange(batch_size, dtype=torch.long, device=device)
    # Optional short exact-alignment tail.  This is substantially cheaper than
    # making the entire final epoch exact while still ending optimization on the
    # deployment selector rather than the oracle curriculum.
    full_last_steps = max(0, int(train_cfg.get("deployment_selector_full_last_n_steps", 0)))
    steps_per_epoch = max(1, int(train_cfg.get("steps_per_epoch", 1)))
    total_epochs = max(1, int(train_cfg.get("epochs", epoch + 1)))
    step_in_epoch = step % steps_per_epoch
    if epoch == total_epochs - 1 and full_last_steps > 0 and step_in_epoch >= max(0, steps_per_epoch - full_last_steps):
        return torch.arange(batch_size, dtype=torch.long, device=device)
    cadence = max(1, int(train_cfg.get("deployment_selector_every_n_steps", 1)))
    if step % cadence != 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    count = int(train_cfg.get("deployment_selector_scenes_per_rank", 0))
    if count <= 0 or count >= batch_size:
        return torch.arange(batch_size, dtype=torch.long, device=device)
    # Rotate the exact scenes so every position receives predicted-selector
    # supervision over a short window.  Rank offset is harmless because each DDP
    # rank already owns a different sample shard.
    rank = int(train_cfg.get("global_rank", 0))
    start = (step * count + rank * count) % batch_size
    return (torch.arange(count, device=device, dtype=torch.long) + start) % batch_size


def _build_predicted_pair_numpy_cache(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Copy all exact-selector inputs to CPU once for one local batch.

    Multi-budget supervision changes only the evidence budget.  The model
    outputs, pair graph, masks, costs and metadata are identical across those
    budgets, so copying them from CUDA for every budget is pure duplication.
    """
    s_cfg = cfg.get("selector", {})
    t_cfg = cfg.get("tournament", {})
    pair_head_needs_local = bool(cfg.get("model", {}).get("pair_head_residual_over_local", False)) or float(cfg.get("runtime", {}).get("pair_delta_hybrid_local_weight", 0.0)) > 0.0
    selector_needs_pair_var = (
        float(t_cfg.get("beta_uncertainty", 0.0)) > 0.0
        or float(s_cfg.get("lambda_info", 0.0)) > 0.0
        or bool(s_cfg.get("force_uncertainty_objective", False))
    )
    family_logits = outputs.get("family_logits")
    fam_ids_t = batch.get("evidence_family_ids")
    group_ids_t = batch.get("evidence_agent_group_ids")
    flags = batch.get("runtime_safety_flags")
    pair_var_t = outputs.get("pair_atom_var")
    active_t = batch.get("evidence_active")
    costs_t = batch.get("evidence_budget_costs")
    pair_weights_t = batch.get("pair_weights")
    evidence_features_t = batch.get("evidence_features")
    decisive_hard_t = batch.get("decisive_hard_mask")
    teacher_a_star_t = batch.get("teacher_a_star")

    snapshot = _packed_numpy_snapshot(
        {
            "J0": (outputs["J0"], torch.float32),
            "g_np": (
                outputs.get("g") if pair_head_needs_local and "g" in outputs else None,
                torch.float32,
            ),
            "delta": (outputs["pair_atom_delta"], torch.float32),
            "pair_var": (
                pair_var_t if pair_var_t is not None and selector_needs_pair_var else None,
                torch.float32,
            ),
            "pair_weights": (
                pair_weights_t
                if pair_weights_t is not None
                else torch.ones_like(batch["pair_valid"], dtype=torch.float32),
                torch.float32,
            ),
            "logits": (outputs["proposal_logits"], torch.float32),
            "family_logits_np": (family_logits, torch.float32),
            "costs": (
                costs_t
                if costs_t is not None
                else torch.ones_like(outputs["proposal_logits"]),
                torch.float32,
            ),
            "evidence_features_np": (evidence_features_t, torch.float32),
            "pairs": (batch["pair_indices"], torch.int64),
            "fam_ids_np": (fam_ids_t, torch.int64),
            "group_ids_np": (group_ids_t, torch.int64),
            "teacher_a_star_np": (teacher_a_star_t, torch.int64),
            "valid": (batch["candidate_valid"], torch.bool),
            "pair_valid": (batch["pair_valid"], torch.bool),
            "active": (
                active_t
                if active_t is not None
                else torch.ones_like(outputs["proposal_logits"], dtype=torch.bool),
                torch.bool,
            ),
            "flags_np": (flags, torch.bool),
            "decisive_hard_np": (decisive_hard_t, torch.bool),
        }
    )
    logits = snapshot["logits"]
    valid = snapshot["valid"]
    assert logits is not None and valid is not None
    fam_ids_np = (
        snapshot["fam_ids_np"]
        if snapshot["fam_ids_np"] is not None
        else np.zeros_like(logits, dtype=np.int64)
    )
    return {
        "J0": snapshot["J0"],
        "g_np": snapshot["g_np"],
        "delta": snapshot["delta"],
        "pair_var": snapshot["pair_var"],
        "pairs": snapshot["pairs"],
        "pair_valid": snapshot["pair_valid"],
        "pair_weights": snapshot["pair_weights"],
        "logits": logits,
        "family_logits_np": snapshot["family_logits_np"],
        "valid": valid,
        "active": snapshot["active"],
        "costs": snapshot["costs"],
        "fam_ids_np": fam_ids_np,
        "group_ids_np": (
            snapshot["group_ids_np"]
            if snapshot["group_ids_np"] is not None
            else np.full_like(fam_ids_np, -1, dtype=np.int64)
        ),
        "flags_np": (
            snapshot["flags_np"]
            if snapshot["flags_np"] is not None
            else np.zeros_like(valid, dtype=bool)
        ),
        "evidence_features_np": snapshot["evidence_features_np"],
        "decisive_hard_np": snapshot["decisive_hard_np"],
        "teacher_a_star_np": snapshot["teacher_a_star_np"],
    }


def _predicted_pair_certificate_masks(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    scene_indices: torch.Tensor | None = None,
    _numpy_cache: dict[str, Any] | None = None,
    _aocc_scene_caches: list[dict[str, Any]] | None = None,
) -> torch.Tensor:
    """Stop-gradient HAB masks for pair-conditioned action supervision.

    Runtime pair-conditioned BDSE selects atoms using signed atom-pair deltas
    d_i(a,b), not dense action costs g_i(a).  Training therefore needs a
    parallel stop-gradient selection path so L_act can send gradients through
    the same pair-margin head used at deployment.
    """
    full_batch_size = int(outputs["J0"].shape[0])
    if scene_indices is not None:
        scene_indices = scene_indices.to(device=outputs["J0"].device, dtype=torch.long)
        outputs = _slice_scene_batch(outputs, scene_indices, full_batch_size)
        batch = _slice_scene_batch(batch, scene_indices, full_batch_size)
    if "pair_atom_delta" not in outputs or "pair_indices" not in batch:
        return outputs["J0"].new_zeros(outputs["proposal_logits"].shape, dtype=torch.bool)
    e_cfg = cfg.get("evidence", {})
    s_cfg = cfg.get("selector", {})
    train_cfg = cfg.get("training", {})
    t_cfg = cfg.get("tournament", {})
    c_cfg = cfg.get("calibration", {})
    normalize_margins = bool(cfg.get("model", {}).get("pair_margin_normalized", True))
    pair_head_needs_local = bool(cfg.get("model", {}).get("pair_head_residual_over_local", False)) or float(cfg.get("runtime", {}).get("pair_delta_hybrid_local_weight", 0.0)) > 0.0

    cache = _numpy_cache if _numpy_cache is not None else _build_predicted_pair_numpy_cache(outputs, batch, cfg)
    J0 = cache["J0"]
    g_np = cache["g_np"]
    delta = cache["delta"]
    pair_var = cache["pair_var"]
    pairs = cache["pairs"]
    pair_valid = cache["pair_valid"]
    pair_weights = cache["pair_weights"]
    logits = cache["logits"]
    family_logits_np = cache["family_logits_np"]
    valid = cache["valid"]
    active = cache["active"]
    costs = cache["costs"]
    fam_ids_np = cache["fam_ids_np"]
    group_ids_np = cache["group_ids_np"]
    flags_np = cache["flags_np"]
    evidence_features_np = cache["evidence_features_np"]
    decisive_hard_np = cache["decisive_hard_np"]
    teacher_a_star_np = cache.get("teacher_a_star_np")

    Bsz, E = logits.shape
    selected_mask = np.zeros((Bsz, E), dtype=bool)
    budget = float(e_cfg.get("budget", 16))
    M = int(s_cfg.get("proposal_top_m", max(int(2 * budget), int(budget) + 1)))
    gamma = float(s_cfg.get("normalized_gamma_max", 5.0) if normalize_margins else s_cfg.get("gamma_max_default", 100.0))
    eta = float(s_cfg.get("normalized_eta_pred", 0.1) if normalize_margins else s_cfg.get("eta_pred", 1.0))
    beta_unc = float(t_cfg.get("beta_uncertainty", 0.0))
    eps_cal = float(t_cfg.get("epsilon_cal", c_cfg.get("epsilon_cal", 0.0)))
    lambda_info = float(s_cfg.get("lambda_info", 0.0))
    prior_var = s_cfg.get("unqueried_atom_variance", None)

    for bidx in range(Bsz):
        structural_bypass = bool(s_cfg.get("decision_budget_excludes_structural_safety", False))
        topm, fam_budget, _ = select_topm_atoms_hab(
            logits[bidx],
            fam_ids_np[bidx],
            active[bidx],
            costs[bidx],
            budget,
            M,
            family_scores=family_logits_np[bidx] if family_logits_np is not None else None,
            free_budget=s_cfg.get("hab_free_budget", None),
            reserve_fraction=float(s_cfg.get("hab_reserve_fraction", 0.2)),
            enabled=bool(s_cfg.get("hab_enabled", True)),
            min_family_slots=s_cfg.get("min_family_topm_slots", None),
        )
        # Mirror the hybrid selector used at deployment: hard/rule atoms that
        # support supervised margins must remain queryable even when proposal
        # logits are still immature.  This is not dense leakage into the final
        # certificate; it only expands the Top-M candidate pool before the same
        # budgeted greedy selector runs.
        if (not structural_bypass) and bool(s_cfg.get("force_decisive_hard_topm", True)) and decisive_hard_np is not None:
            hard_train = (decisive_hard_np[bidx] if decisive_hard_np is not None else np.zeros((E,), dtype=bool)) & active[bidx]
            forced = np.flatnonzero(hard_train)
            if forced.size:
                forced_cap = int(s_cfg.get("max_forced_hard_topm", max(1, M // 2)))
                forced = np.asarray(sorted(forced.tolist(), key=lambda i: (-float(logits[bidx, i]), int(i)))[:forced_cap], dtype=np.int64)
                non_forced = [int(i) for i in np.asarray(topm, dtype=np.int64).reshape(-1).tolist() if int(i) not in set(forced.tolist())]
                topm = np.asarray((forced.tolist() + non_forced)[:M], dtype=np.int64)
        hard_feature_for_pool = evidence_features_np[bidx, :, 0] > 0.5 if evidence_features_np is not None else np.zeros((E,), dtype=bool)
        interaction_family_set = set(int(x) for x in s_cfg.get("interaction_family_ids", [2, 3]))
        soft_interaction_pool = np.asarray([int(f) in interaction_family_set for f in fam_ids_np[bidx].tolist()], dtype=bool) & active[bidx] & ~hard_feature_for_pool
        min_soft_topm = int(s_cfg.get("min_soft_interaction_topm_slots", 0))
        protected_for_pool = structural_safety_mask(
            hard_feature_for_pool,
            fam_ids_np[bidx],
            active[bidx],
            include_feasibility=bool(s_cfg.get("structural_safety_include_feasibility", True)),
        )
        if min_soft_topm > 0 and bool(soft_interaction_pool.any()):
            topm, _ = reserve_topm_candidates(
                topm, soft_interaction_pool, logits[bidx], M, min_soft_topm,
                protected_mask=None if structural_bypass else protected_for_pool,
                group_ids=group_ids_np[bidx],
            )
        if structural_bypass:
            topm, _ = restrict_topm_to_decision_evidence(
                topm, active[bidx] & ~protected_for_pool, logits[bidx], M,
                family_ids=fam_ids_np[bidx],
            )
        atom_active = np.zeros((E,), dtype=bool)
        atom_active[topm] = True
        atom_active &= active[bidx]
        pv = pair_valid[bidx]
        pair_arr = pairs[bidx][pv]
        weight_arr = pair_weights[bidx][pv]
        delta_arr = delta[bidx][:, pv]
        var_arr = pair_var[bidx][:, pv] if pair_var is not None else None
        # Remove out-of-range pairs defensively; padded pairs are already masked.
        if pair_arr.size:
            ok = (pair_arr[:, 0] >= 0) & (pair_arr[:, 0] < valid.shape[1]) & (pair_arr[:, 1] >= 0) & (pair_arr[:, 1] < valid.shape[1])
            ok &= valid[bidx, pair_arr[:, 0]] & valid[bidx, pair_arr[:, 1]]
            pair_arr = pair_arr[ok]
            weight_arr = weight_arr[ok]
            delta_arr = delta_arr[:, ok]
            if var_arr is not None:
                var_arr = var_arr[:, ok]
        if bool(s_cfg.get("decision_pairs_within_viability_frontier", False)) and pair_arr.size:
            pair_arr_before = pair_arr.copy()
            pair_arr, weight_arr, _ = restrict_pairs_to_viability_frontier(
                pair_arr,
                weight_arr,
                valid[bidx],
                flags_np[bidx],
                J0[bidx],
                hard_risk=None,
                frontier_size=int(s_cfg.get("all_flagged_frontier_size", 8)),
                single_safe_rivals=int(s_cfg.get("single_safe_anchor_rivals", 8)),
            )
            if pair_arr.size:
                lookup = {tuple(map(int, p)): i for i, p in enumerate(pair_arr_before.tolist())}
                idx = np.asarray([lookup[tuple(map(int, p))] for p in pair_arr.tolist()], dtype=np.int64)
                delta_arr = delta_arr[:, idx]
                if var_arr is not None:
                    var_arr = var_arr[:, idx]
            else:
                delta_arr = delta_arr[:, :0]
                if var_arr is not None:
                    var_arr = var_arr[:, :0]
        hard_feature = evidence_features_np[bidx, :, 0] > 0.5 if evidence_features_np is not None else np.zeros((E,), dtype=bool)
        mandatory = structural_safety_mask(
            hard_feature,
            fam_ids_np[bidx],
            active[bidx],
            include_feasibility=bool(s_cfg.get("structural_safety_include_feasibility", True)),
        )
        if (not structural_bypass) and decisive_hard_np is not None and bool(s_cfg.get("force_decisive_hard_topm", True)):
            mandatory = mandatory | ((decisive_hard_np[bidx] if decisive_hard_np is not None else np.zeros((E,), dtype=bool)) & active[bidx])
        if structural_bypass:
            atom_active &= ~mandatory
        base_deltas = J0[bidx, pair_arr[:, 1]] - J0[bidx, pair_arr[:, 0]] if pair_arr.size else np.zeros((0,), dtype=np.float32)
        mscale = margin_normalization_scale(base_deltas, min_scale=_margin_norm_min_scale(cfg), quantile=_margin_norm_quantile(cfg)) if normalize_margins else 1.0
        if g_np is not None and pair_arr.size:
            a_np = pair_arr[:, 0].clip(0, g_np.shape[2] - 1)
            b_np = pair_arr[:, 1].clip(0, g_np.shape[2] - 1)
            local_delta_arr = (g_np[bidx][:, b_np] - g_np[bidx][:, a_np]) / max(float(mscale), 1e-6) if normalize_margins else (g_np[bidx][:, b_np] - g_np[bidx][:, a_np])
            if bool(cfg.get("model", {}).get("pair_head_residual_over_local", False)):
                delta_arr, _ = confidence_shrunk_residual_pair_delta_numpy(
                    local_delta_arr,
                    delta_arr,
                    np.zeros_like(delta_arr, dtype=np.float32) if var_arr is None else var_arr,
                    (cfg.get("runtime", {}).get("pair_residual_trust", {}) or {}),
                )
            else:
                w_local = float(cfg.get("runtime", {}).get("pair_delta_hybrid_local_weight", 0.0))
                if w_local > 0.0:
                    w_local = min(max(w_local, 0.0), 1.0)
                    delta_arr = (1.0 - w_local) * delta_arr + w_local * local_delta_arr
        result = runtime_greedy_selector_pair_conditioned(
            J0[bidx],
            delta_arr,
            pair_arr,
            weight_arr,
            costs[bidx],
            valid[bidx],
            flags_np[bidx],
            budget=budget,
            gamma_max=gamma,
            eta_pred=eta,
            atom_active_mask=atom_active,
            pair_atom_variance=var_arr,
            beta_uncertainty=beta_unc,
            epsilon_cal=eps_cal,
            lambda_info=lambda_info,
            prior_atom_variance=prior_var,
            family_ids=fam_ids_np[bidx],
            family_budget_caps=fam_budget.family_caps,
            mandatory_atom_mask=None if structural_bypass else mandatory,
            mandatory_quota=0 if structural_bypass else int(s_cfg.get("mandatory_hard_quota", 0)),
            min_selected_atoms=int(s_cfg.get("min_selected_atoms", 0)),
            force_fill_budget=bool(s_cfg.get("force_fill_budget", False)),
            normalize_margins=normalize_margins,
            margin_scale=mscale,
            proposal_scores=logits[bidx],
            proposal_fill_weight=float(s_cfg.get("proposal_fill_weight", 0.25)),
            prioritize_mandatory_fill=bool(s_cfg.get("prioritize_mandatory_fill", True)),
            selector_cap_mode=str(s_cfg.get("selector_cap_mode", "legacy_abs")),
            boundary_certificate_cap=s_cfg.get("boundary_certificate_cap", None),
            base_margin_cap_multiplier=float(s_cfg.get("base_margin_cap_multiplier", 1.0)),
            flip_bonus=float(s_cfg.get("flip_bonus", 0.0)),
            flip_window=float(s_cfg.get("flip_window", 0.5)),
            certify_margin=float(s_cfg.get("certify_margin", 0.0)),
            flip_mode=str(s_cfg.get("flip_mode", "hard")),
            flip_temperature=float(s_cfg.get("flip_temperature", 0.08)),
            action_rank_certificate_weight=float(s_cfg.get("action_rank_certificate_weight", 1.0)),
            action_rank_score_weight=float(s_cfg.get("action_rank_score_weight", 0.0)),
            action_rank_gap_weight=float(s_cfg.get("action_rank_gap_weight", 0.0)),
            action_rank_flip_weight=float(s_cfg.get("action_rank_flip_weight", 0.0)),
            action_rank_softmin_tau=float(s_cfg.get("action_rank_softmin_tau", 0.2)),
            action_utility_weight=float(s_cfg.get("action_utility_weight", 0.0)),
            action_pair_utility_weight=float(s_cfg.get("action_pair_utility_weight", 0.0)),
            action_rank_fast_greedy=bool(s_cfg.get("action_rank_fast_greedy", False)),
            hybrid_lcb_budget_frac=float(s_cfg.get("hybrid_lcb_budget_frac", 0.55)),
            hybrid_lcb_cap_mode=str(s_cfg.get("hybrid_lcb_cap_mode", "legacy_abs")),
            hybrid_protect_lcb_seed=bool(s_cfg.get("hybrid_protect_lcb_seed", True)),
            hybrid_min_action_budget_frac=float(s_cfg.get("hybrid_min_action_budget_frac", 0.0)),
            hybrid_max_lcb_seed_atoms=int(s_cfg.get("hybrid_max_lcb_seed_atoms", 0)),
            adaptive_hybrid_lcb_budget=bool(s_cfg.get("adaptive_hybrid_lcb_budget", False)),
            adaptive_lcb_min_frac=float(s_cfg.get("adaptive_lcb_min_frac", 0.45)),
            adaptive_lcb_max_frac=float(s_cfg.get("adaptive_lcb_max_frac", 0.80)),
            adaptive_lcb_safety_weight=float(s_cfg.get("adaptive_lcb_safety_weight", 0.25)),
            adaptive_lcb_fallback_weight=float(s_cfg.get("adaptive_lcb_fallback_weight", 0.20)),
            adaptive_lcb_uncertainty_weight=float(s_cfg.get("adaptive_lcb_uncertainty_weight", 0.10)),
            adaptive_lcb_boundary_action_weight=float(s_cfg.get("adaptive_lcb_boundary_action_weight", 0.25)),
            adaptive_lcb_boundary_tau=float(s_cfg.get("adaptive_lcb_boundary_tau", 0.35)),
            decision_family_boost=float(s_cfg.get("decision_family_boost", 0.0)),
            decision_family_ids=s_cfg.get("decision_family_ids", [2, 3]),
            decision_family_quota=int(s_cfg.get("decision_family_quota", 0)),
            interaction_family_ids=s_cfg.get("interaction_family_ids", [2, 3]),
            interaction_family_quota=int(s_cfg.get("interaction_family_quota", 0)),
            soft_interaction_mask=soft_interaction_pool,
            soft_interaction_quota=int(s_cfg.get("soft_interaction_quota", 0)),
            interaction_group_ids=group_ids_np[bidx],
            direction_invariant_interaction_weight=float(s_cfg.get("direction_invariant_interaction_weight", 0.0)),
            direction_invariant_boundary_tau=float(s_cfg.get("direction_invariant_boundary_tau", 0.35)),
            direction_invariant_flip_bonus=float(s_cfg.get("direction_invariant_flip_bonus", 0.5)),
            collapse_reciprocal_pairs=bool(s_cfg.get("collapse_reciprocal_pairs", True)),
            force_uncertainty_objective=bool(s_cfg.get("force_uncertainty_objective", False)),
            adverse_certificate_beta=float(s_cfg.get("adverse_certificate_beta", 1.0)),
            adverse_certificate_epsilon=float(s_cfg.get("adverse_certificate_epsilon", 0.05)),
            adverse_certificate_prior_radius=float(s_cfg.get("adverse_certificate_prior_radius", 0.10)),
            adverse_certificate_margin=float(s_cfg.get("adverse_certificate_margin", 0.0)),
            adverse_certificate_stop_when_certified=bool(s_cfg.get("adverse_certificate_stop_when_certified", True)),
            adverse_certificate_max_target_rivals=int(s_cfg.get("adverse_certificate_max_target_rivals", 0)),
            adverse_certificate_target_action=(
                int(teacher_a_star_np[bidx])
                if bool(train_cfg.get("aocc_teacher_target_training", True))
                and teacher_a_star_np is not None
                else None
            ),
            adverse_certificate_calibrated=False,
            adverse_certificate_fill_to_budget_after_certified=bool(
                s_cfg.get("adverse_certificate_fill_to_budget_after_certified", False)
            ),
            adverse_certificate_max_interaction_prefix_fraction=float(
                s_cfg.get("adverse_certificate_max_interaction_prefix_fraction", 1.0)
            ),
            aocc_state_cache=(
                _aocc_scene_caches[bidx]
                if _aocc_scene_caches is not None and bidx < len(_aocc_scene_caches)
                else None
            ),
        )
        selected_mask[bidx, result.selected] = True
    return torch.from_numpy(selected_mask).to(outputs["J0"].device)



def _predicted_pair_certificate_masks_multi_budget(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    budget_cfgs: list[dict[str, Any]],
    scene_indices: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Exact masks for several budgets with one CUDA-to-CPU snapshot.

    The selector itself remains unchanged and is still evaluated independently
    for each configured budget.  Only duplicated tensor slicing and host copies
    are removed, so this is numerically identical to repeated calls to
    :func:`_predicted_pair_certificate_masks`.
    """
    if not budget_cfgs:
        return []
    full_batch_size = int(outputs["J0"].shape[0])
    if scene_indices is not None:
        scene_indices = scene_indices.to(device=outputs["J0"].device, dtype=torch.long)
        outputs = _slice_scene_batch(outputs, scene_indices, full_batch_size)
        batch = _slice_scene_batch(batch, scene_indices, full_batch_size)
    cache = _build_predicted_pair_numpy_cache(outputs, batch, budget_cfgs[0])
    scene_caches = [dict() for _ in range(int(outputs["J0"].shape[0]))]
    return [
        _predicted_pair_certificate_masks(
            outputs,
            batch,
            local_cfg,
            scene_indices=None,
            _numpy_cache=cache,
            _aocc_scene_caches=scene_caches,
        )
        for local_cfg in budget_cfgs
    ]



def _deployment_exact_distill_scene_indices(
    batch_size: int,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Sample scenes for expensive exact CPU-mask distillation.

    The action loss may use the fast GPU margin surrogate on every scene, while
    exact runtime masks are sampled only to supervise the proposal head.  This
    keeps the fixed-budget pathway trained without forcing every DDP rank to
    block on serial NumPy selector search at every optimizer step.
    """
    local_cfg = dict(train_cfg)
    local_cfg["deployment_selector_scenes_per_rank"] = int(
        train_cfg.get(
            "deployment_exact_distill_scenes_per_rank",
            train_cfg.get("deployment_selector_scenes_per_rank", 1),
        )
    )
    local_cfg["deployment_selector_every_n_steps"] = int(
        train_cfg.get(
            "deployment_exact_distill_every_n_steps",
            train_cfg.get("deployment_selector_every_n_steps", 4),
        )
    )
    local_cfg["deployment_selector_full_last_n_steps"] = int(
        train_cfg.get("deployment_exact_distill_full_last_n_steps", 0)
    )
    return _deployment_selector_scene_indices(batch_size, local_cfg, device)


def _fast_topm_mask_torch(
    proposal_logits: torch.Tensor,
    active_mask: torch.Tensor,
    costs: torch.Tensor,
    family_ids: torch.Tensor,
    evidence_features: torch.Tensor | None,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU Top-M approximation used only by the training-time fast surrogate.

    It preserves the deployment-critical constraints used by v45: inactive and
    structural safety atoms are excluded from the decision budget, and a fixed
    number of soft interaction atoms is reserved in Top-M.  Runtime evaluation
    continues to use the exact HAB implementation.
    """
    s_cfg = cfg.get("selector", {})
    B, E = proposal_logits.shape
    M = min(max(int(s_cfg.get("proposal_top_m", 64)), 1), E)
    active = active_mask.bool() & torch.isfinite(costs) & (costs > 0)
    hard_feature = (
        evidence_features[..., 0] > 0.5
        if evidence_features is not None and evidence_features.ndim >= 3 and evidence_features.shape[-1] > 0
        else torch.zeros_like(active)
    )
    structural_bypass = bool(s_cfg.get("decision_budget_excludes_structural_safety", False))
    include_feasibility = bool(s_cfg.get("structural_safety_include_feasibility", True))
    structural = hard_feature | ((family_ids == 1) if include_feasibility else torch.zeros_like(active))
    if structural_bypass:
        active = active & ~structural

    interaction_ids = [int(x) for x in s_cfg.get("interaction_family_ids", [2, 3])]
    interaction = torch.zeros_like(active)
    for fam in interaction_ids:
        interaction |= family_ids == fam
    soft_interaction = active & interaction & ~hard_feature
    reserve = max(0, int(s_cfg.get("min_soft_interaction_topm_slots", 0)))

    neg_inf = torch.finfo(proposal_logits.dtype).min
    topm_mask = torch.zeros_like(active)
    row = torch.arange(B, device=proposal_logits.device)[:, None]
    reserve_k = min(reserve, M, E)
    if reserve_k > 0:
        soft_scores = proposal_logits.masked_fill(~soft_interaction, neg_inf)
        soft_idx = torch.topk(soft_scores, k=reserve_k, dim=1, largest=True, sorted=True).indices
        soft_ok = soft_interaction.gather(1, soft_idx)
        topm_mask[row.expand_as(soft_idx)[soft_ok], soft_idx[soft_ok]] = True
    active_count = active.sum(dim=1).clamp_max(M)
    remaining = (active_count - topm_mask.sum(dim=1)).clamp_min(0)
    candidate = active & ~topm_mask
    global_order = torch.argsort(proposal_logits.masked_fill(~candidate, neg_inf), dim=1, descending=True)
    global_ok = candidate.gather(1, global_order)
    positions = torch.arange(E, device=proposal_logits.device)[None, :]
    take = global_ok & (positions < remaining[:, None])
    topm_mask[row.expand_as(global_order)[take], global_order[take]] = True
    return topm_mask, soft_interaction


def _pairwise_action_from_margin_torch(
    margins: torch.Tensor,
    pairs: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    safety_flags: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Return the predicted pairwise-preference winner for each scene."""
    B, P = margins.shape
    K = valid.shape[1]
    a = pairs[..., 0].long().clamp(0, K - 1)
    b = pairs[..., 1].long().clamp(0, K - 1)
    p = torch.sigmoid(margins / max(float(tau), 1e-3))
    score = margins.new_zeros((B, K))
    denom = margins.new_zeros((B, K))
    score.scatter_add_(1, a, weights * p)
    score.scatter_add_(1, b, weights * (1.0 - p))
    denom.scatter_add_(1, a, weights)
    denom.scatter_add_(1, b, weights)
    avg = score / denom.clamp_min(1e-9)
    eligible = valid.bool()
    safe = eligible & ~safety_flags.bool()
    has_safe = safe.any(dim=1, keepdim=True)
    eligible = torch.where(has_safe, safe, eligible)
    avg = avg.masked_fill(~eligible, torch.finfo(avg.dtype).min)
    return avg.argmax(dim=1)


def _fast_pair_margin_surrogate_masks(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    budgets: list[float],
) -> dict[float, torch.Tensor]:
    """Build nested fixed-budget masks entirely on the accelerator.

    This is a one-shot approximation to the exact backward-elimination MARS
    selector.  It scores each Top-M atom by the damage caused when that atom is
    removed from the full predicted signed margin field, including residual,
    sign, winner-certificate, and action-preservation terms.  The expensive exact
    NumPy selector remains the source of distillation targets on sampled scenes.
    """
    J0 = outputs["J0"].detach().float()
    delta = outputs["pair_atom_delta"].detach().float()
    logits = outputs["proposal_logits"].detach().float()
    pair_indices = batch["pair_indices"].long()
    pair_valid = batch["pair_valid"].bool()
    valid = batch["candidate_valid"].bool()
    weights = batch.get("pair_weights", torch.ones_like(pair_valid, dtype=torch.float32)).float()
    weights = torch.where(pair_valid, weights.clamp_min(0.0), torch.zeros_like(weights))
    active = batch.get("evidence_active", torch.ones_like(logits, dtype=torch.bool)).bool()
    costs = batch.get("evidence_budget_costs", torch.ones_like(logits)).float()
    fam = batch.get("evidence_family_ids", torch.zeros_like(logits, dtype=torch.long)).long()
    features = batch.get("evidence_features")
    flags = batch.get("runtime_safety_flags", torch.zeros_like(valid)).bool()
    topm_mask, soft_interaction = _fast_topm_mask_torch(logits, active, costs, fam, features, cfg)

    B, E, P = delta.shape
    K = J0.shape[1]
    a = pair_indices[..., 0].clamp(0, K - 1)
    b = pair_indices[..., 1].clamp(0, K - 1)
    pvalid = pair_valid & valid.gather(1, a) & valid.gather(1, b)
    weights = weights * pvalid.float()
    base = J0.gather(1, b) - J0.gather(1, a)
    normalize = bool(cfg.get("model", {}).get("pair_margin_normalized", True))
    if normalize:
        abs_base = base.abs().masked_fill(~pvalid, 0.0)
        q = float(cfg.get("model", {}).get("margin_normalization_quantile", 0.9))
        q = min(max(q, 0.5), 0.99)
        # torch.quantile over padded zero values is stable and cheap; the configured
        # minimum dominates sparse scenes and matches the deployment normalization.
        scale = torch.quantile(abs_base, q, dim=1, keepdim=True).clamp_min(
            float(cfg.get("model", {}).get("margin_normalization_min_scale", 20000.0))
        )
        base = base / scale
    delta = delta * pvalid[:, None, :].float()
    target = base + (delta * topm_mask[:, :, None].float()).sum(dim=1)

    s_cfg = cfg.get("selector", {})
    tau = max(float(s_cfg.get("margin_coreset_boundary_tau", 0.3)), 1e-3)
    clip = max(float(s_cfg.get("margin_coreset_target_clip", 3.0)), tau)
    huber_delta = max(float(s_cfg.get("margin_coreset_huber_delta", 0.25)), 1e-3)
    residual_w = max(float(s_cfg.get("margin_coreset_residual_weight", 1.0)), 0.0)
    sign_w = max(float(s_cfg.get("margin_coreset_sign_weight", 1.0)), 0.0)
    winner_w = max(float(s_cfg.get("margin_coreset_winner_weight", 2.2)), 0.0)
    action_w = max(float(s_cfg.get("margin_coreset_action_weight", 0.8)), 0.0)

    t = target.clamp(-clip, clip)
    boundary = torch.exp(-t.abs() / tau)
    decisive = torch.tanh(t.abs() / tau)
    pw = weights * (1.0 + 1.5 * boundary + 0.5 * decisive)
    pw_norm = pw.sum(dim=1, keepdim=True).clamp_min(1e-9)
    trial = (target[:, None, :] - delta).clamp(-clip, clip)
    err = (trial - t[:, None, :]) / huber_delta
    abs_err = err.abs()
    huber = torch.where(abs_err <= 1.0, 0.5 * err.square(), abs_err - 0.5)
    score = residual_w * (huber * pw[:, None, :]).sum(dim=2) / pw_norm
    if sign_w > 0.0:
        mismatch = (trial * t[:, None, :] < 0.0) & (t.abs()[:, None, :] >= 0.05)
        sign_penalty = (
            pw[:, None, :] * mismatch.float() * t.abs().clamp_max(1.0)[:, None, :]
        ).sum(dim=2) / pw_norm
        score = score + sign_w * sign_penalty

    target_action = _pairwise_action_from_margin_torch(target, pair_indices, weights, valid, flags, tau)
    orient = torch.zeros_like(target)
    orient = torch.where(a == target_action[:, None], torch.ones_like(orient), orient)
    orient = torch.where(b == target_action[:, None], -torch.ones_like(orient), orient)
    wanted = (orient != 0.0) & (t * orient > 0.0) & pvalid
    if winner_w > 0.0:
        oriented_trial = trial * orient[:, None, :]
        winner_penalty = F.softplus(-oriented_trial / tau)
        winner_norm = (pw * wanted.float()).sum(dim=1, keepdim=True).clamp_min(1e-9)
        winner_value = (winner_penalty * pw[:, None, :] * wanted[:, None, :].float()).sum(dim=2) / winner_norm
        score = score + winner_w * winner_value

    if action_w > 0.0:
        # Batched action-preservation penalty for every leave-one-out candidate.
        for bi in range(B):
            prob = torch.sigmoid(trial[bi] / tau)  # [E,P]
            inc_a = trial.new_zeros((P, K))
            inc_b = trial.new_zeros((P, K))
            inc_a.scatter_add_(1, a[bi, :, None], weights[bi, :, None])
            inc_b.scatter_add_(1, b[bi, :, None], weights[bi, :, None])
            denom = (inc_a + inc_b).sum(dim=0).clamp_min(1e-9)
            action_score = prob @ inc_a + (1.0 - prob) @ inc_b
            avg = action_score / denom[None, :]
            safe = valid[bi] & ~flags[bi]
            eligible = torch.where(safe.any(), safe, valid[bi])
            avg[:, ~eligible] = torch.finfo(avg.dtype).min
            inferred = avg.argmax(dim=1)
            score[bi] = score[bi] + action_w * (inferred != target_action[bi]).float()

    score = score.masked_fill(~topm_mask, torch.finfo(score.dtype).min)
    ratio = score / costs.clamp_min(1e-6)
    soft_quota = max(0, int(s_cfg.get("soft_interaction_quota", 0)))
    out_masks: dict[float, torch.Tensor] = {}
    row = torch.arange(B, device=score.device)[:, None]
    positions = torch.arange(E, device=score.device)[None, :]
    for budget in sorted(set(float(x) for x in budgets)):
        selected = torch.zeros_like(topm_mask)
        soft_pool = topm_mask & soft_interaction & (costs <= float(budget) + 1e-6)
        if soft_quota > 0:
            soft_order = torch.argsort(ratio.masked_fill(~soft_pool, torch.finfo(ratio.dtype).min), dim=1, descending=True)
            soft_ok = soft_pool.gather(1, soft_order)
            soft_cost = costs.gather(1, soft_order)
            soft_cum = soft_cost.cumsum(dim=1)
            take_soft = soft_ok & (positions < soft_quota) & (soft_cum <= float(budget) + 1e-6)
            selected[row.expand_as(soft_order)[take_soft], soft_order[take_soft]] = True
        spent = (costs * selected.float()).sum(dim=1)
        remaining = (float(budget) - spent).clamp_min(0.0)
        pool = topm_mask & ~selected & (costs <= remaining[:, None] + 1e-6)
        order = torch.argsort(ratio.masked_fill(~pool, torch.finfo(ratio.dtype).min), dim=1, descending=True)
        ok = pool.gather(1, order)
        sorted_cost = costs.gather(1, order)
        cumulative = sorted_cost.cumsum(dim=1)
        take = ok & (cumulative <= remaining[:, None] + 1e-6)
        selected[row.expand_as(order)[take], order[take]] = True
        out_masks[float(budget)] = selected
    return out_masks


def _pair_conditioned_tournament_scores(
    J0: torch.Tensor,
    pair_delta: torch.Tensor,
    pairs: torch.Tensor,
    pair_valid: torch.Tensor,
    selected_mask: torch.Tensor,
    valid: torch.Tensor,
    tau: float,
    epsilon_cal: float = 0.0,
    pair_var: torch.Tensor | None = None,
    beta_uncertainty: float = 0.0,
    pair_scale: torch.Tensor | None = None,
    normalize_margins: bool = True,
) -> torch.Tensor:
    """Differentiable soft tournament from selected pair-conditioned deltas.

    Missing queried pairs fall back to the base margin, while present pairs update
    both directions to preserve antisymmetry: M(a,b)+=d and M(b,a)-=d.
    """
    B, K = J0.shape
    M = J0[:, None, :] - J0[:, :, None]
    if normalize_margins:
        scale = J0.new_full((B, 1, 1), 100.0) if pair_scale is None else pair_scale.view(B, 1, 1).clamp_min(1e-6)
        M = M / scale
    support = (pair_delta * selected_mask[:, :, None].float()).sum(dim=1)
    pvalid = pair_valid.bool()
    a = pairs[..., 0].long().clamp(0, K - 1)
    b = pairs[..., 1].long().clamp(0, K - 1)
    pvalid = pvalid & valid.gather(1, a) & valid.gather(1, b)
    flat = torch.zeros((B, K * K), dtype=J0.dtype, device=J0.device)
    lin_ab = a * K + b
    lin_ba = b * K + a
    val = support.masked_fill(~pvalid, 0.0)
    flat.scatter_add_(1, lin_ab, val)
    flat.scatter_add_(1, lin_ba, -val)
    M = M + flat.view(B, K, K)
    sigma = None
    if pair_var is not None and beta_uncertainty > 0:
        var_support = (pair_var.clamp_min(0.0) * selected_mask[:, :, None].float()).sum(dim=1).masked_fill(~pvalid, 0.0)
        vflat = torch.zeros((B, K * K), dtype=J0.dtype, device=J0.device)
        vflat.scatter_add_(1, lin_ab, var_support)
        vflat.scatter_add_(1, lin_ba, var_support)
        sigma = torch.sqrt(vflat.view(B, K, K).clamp_min(0.0) + 1e-12)
    if sigma is not None:
        M = M - float(beta_uncertainty) * sigma
    M = M - float(epsilon_cal)
    eye = torch.eye(K, dtype=torch.bool, device=J0.device)[None]
    rival_valid = valid[:, None, :].expand(B, K, K) & (~eye)
    vals = M.masked_fill(~rival_valid, float("inf"))
    scores = _softmin(vals, tau, dim=2)
    no_rivals = ~rival_valid.any(dim=2)
    scores = scores.masked_fill(no_rivals, 0.0).masked_fill(~valid, _neg_mask_value(scores))
    return scores

def _softmin(vals: torch.Tensor, tau: float, dim: int) -> torch.Tensor:
    if tau <= 0:
        return vals.min(dim=dim).values
    return -float(tau) * torch.logsumexp(-vals / float(tau), dim=dim)


def _budgeted_tournament_scores(
    cost: torch.Tensor,
    valid: torch.Tensor,
    tau: float,
    epsilon_cal: float = 0.0,
    sigma: torch.Tensor | None = None,
    beta_uncertainty: float = 0.0,
) -> torch.Tensor:
    # M(a,b)=cost[b]-cost[a].  Higher score means larger lower margin against rivals.
    B, K = cost.shape
    margins = cost[:, None, :] - cost[:, :, None]
    if sigma is not None:
        margins = margins - float(beta_uncertainty) * sigma
    margins = margins - float(epsilon_cal)
    eye = torch.eye(K, dtype=torch.bool, device=cost.device)[None]
    rival_valid = valid[:, None, :].expand(B, K, K) & (~eye)
    vals = margins.masked_fill(~rival_valid, float("inf"))
    scores = _softmin(vals, tau, dim=2)
    no_rivals = ~rival_valid.any(dim=2)
    scores = scores.masked_fill(no_rivals, 0.0).masked_fill(~valid, _neg_mask_value(scores))
    return scores


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool() & torch.isfinite(values)
    safe = torch.where(mask, values, torch.zeros_like(values))
    weights = mask.to(dtype=values.dtype)
    return safe.sum() / weights.sum().clamp_min(1.0)


def _weighted_mean(loss: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool() & torch.isfinite(loss) & torch.isfinite(weights)
    safe_loss = torch.where(mask, loss, torch.zeros_like(loss))
    safe_weights = torch.where(mask, weights, torch.zeros_like(weights))
    return (safe_loss * safe_weights).sum() / safe_weights.sum().clamp_min(1e-6)


def _valid_row_scale(values: torch.Tensor, valid: torch.Tensor, min_scale: float = 1.0) -> torch.Tensor:
    """Per-sample robust scale for costs/margins with invalid entries masked out."""
    mask = valid.bool() & torch.isfinite(values)
    count = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    safe = torch.where(mask, values, torch.zeros_like(values))
    mean = safe.sum(dim=1, keepdim=True) / count
    var = (torch.where(mask, values - mean, torch.zeros_like(values)).pow(2).sum(dim=1, keepdim=True) / count).clamp_min(0.0)
    scale = torch.sqrt(var).clamp_min(float(min_scale))
    return scale.detach()


def _negative_cost_logits(cost: torch.Tensor, valid: torch.Tensor, min_scale: float = 1.0) -> torch.Tensor:
    """Convert arbitrary-scale costs to stable CE logits without changing argmin order."""
    finite_cost = torch.where(torch.isfinite(cost), cost, torch.zeros_like(cost))
    scale = _valid_row_scale(finite_cost, valid, min_scale=min_scale)
    center = torch.where(valid.bool(), finite_cost, torch.zeros_like(finite_cost)).sum(dim=1, keepdim=True) / valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    logits = -(finite_cost - center.detach()) / scale
    return logits.masked_fill(~valid.bool(), _neg_mask_value(logits))


def _config_with_evidence_budget(cfg: dict[str, Any], budget: float) -> dict[str, Any]:
    """Return a shallow config view with a different evidence budget.

    The deployment selector reads only nested configuration dictionaries and
    does not mutate them.  A shallow top-level copy plus a copied ``evidence``
    block is therefore enough and avoids a costly deepcopy in every batch.
    """
    local_cfg = dict(cfg)
    local_evidence = dict(cfg.get("evidence", {}))
    local_evidence["budget"] = float(budget)
    local_cfg["evidence"] = local_evidence
    return local_cfg


def _deployment_budget_entries_for_step(
    budgets: list[float],
    weights: list[float],
    train_cfg: dict[str, Any],
) -> list[tuple[float, float]]:
    """Choose deployment budgets for one optimizer step.

    ``all`` reproduces the original objective exactly.  ``weighted_round_robin``
    evaluates one budget per DDP rank/step using a deterministic schedule whose
    frequency is proportional to the configured loss weights.  Because the
    original multi-budget loss is a weighted mean, this is an unbiased
    stratified estimator of that objective while avoiding two of the three
    expensive CPU deployment-selector calls in the common [8, 16, 24] setup.
    """
    entries = [(float(b), max(float(w), 0.0)) for b, w in zip(budgets, weights) if float(w) > 0.0]
    if not entries:
        return []
    strategy = str(train_cfg.get("deployment_budget_strategy", "all")).strip().lower()
    if strategy in {"all", "full", "exact"} or len(entries) == 1:
        return entries
    if strategy in {"primary_plus_aux", "primary+aux", "primary_aux"}:
        primary_budget = float(train_cfg.get("deployment_primary_budget", entries[0][0]))
        primary_idx = min(range(len(entries)), key=lambda i: abs(entries[i][0] - primary_budget))
        primary = entries[primary_idx]
        aux = [entry for i, entry in enumerate(entries) if i != primary_idx]
        if not aux:
            return [primary]
        # Always optimize the paper's primary fixed budget.  The auxiliary slot
        # is deterministically sampled in proportion to the remaining weights.
        # Giving it the total auxiliary mass makes the per-step weighted mean an
        # unbiased estimator of the full any-budget objective.
        aux_total = sum(weight for _, weight in aux)
        slots = max(len(aux), int(train_cfg.get("deployment_budget_schedule_slots", 16)))
        raw = [weight / aux_total * slots for _, weight in aux]
        counts = [max(1, int(round(value))) for value in raw]
        while sum(counts) > slots:
            candidates = [i for i, count in enumerate(counts) if count > 1]
            if not candidates:
                break
            idx = max(candidates, key=lambda i: counts[i] - raw[i])
            counts[idx] -= 1
        while sum(counts) < slots:
            idx = max(range(len(counts)), key=lambda i: raw[i] - counts[i])
            counts[idx] += 1
        schedule: list[float] = []
        remaining = counts[:]
        while any(count > 0 for count in remaining):
            for i, (budget, _) in enumerate(aux):
                if remaining[i] > 0:
                    schedule.append(float(budget))
                    remaining[i] -= 1
        step = int(train_cfg.get("global_step", 0))
        rank = int(train_cfg.get("global_rank", 0))
        world = max(1, int(train_cfg.get("world_size", 1)))
        slot = (step * world + rank) % max(len(schedule), 1)
        return [(float(primary[0]), float(primary[1])), (float(schedule[slot]), float(aux_total))]
    if strategy not in {"weighted_round_robin", "sampled", "stratified"}:
        raise ValueError(
            "training.deployment_budget_strategy must be one of "
            "all|weighted_round_robin|primary_plus_aux; got %r" % strategy
        )

    # Convert arbitrary positive weights to a compact deterministic slot table.
    # The largest table is intentionally capped; this is a training schedule,
    # not a floating-point exact rational expansion.
    total = sum(w for _, w in entries)
    max_slots = max(len(entries), int(train_cfg.get("deployment_budget_schedule_slots", 16)))
    raw = [w / total * max_slots for _, w in entries]
    counts = [max(1, int(round(x))) for x in raw]
    while sum(counts) > max_slots:
        candidates = [i for i, c in enumerate(counts) if c > 1]
        if not candidates:
            break
        idx = max(candidates, key=lambda i: counts[i] - raw[i])
        counts[idx] -= 1
    while sum(counts) < max_slots:
        idx = max(range(len(counts)), key=lambda i: raw[i] - counts[i])
        counts[idx] += 1
    schedule: list[float] = []
    # Interleave rather than grouping identical budgets, improving short-window
    # coverage when validation/checkpointing interrupts a run.
    remaining = counts[:]
    while any(c > 0 for c in remaining):
        for i, (budget, _) in enumerate(entries):
            if remaining[i] > 0:
                schedule.append(float(budget))
                remaining[i] -= 1

    step = int(train_cfg.get("global_step", 0))
    rank = int(train_cfg.get("global_rank", 0))
    world = max(1, int(train_cfg.get("world_size", 1)))
    slot = (step * world + rank) % max(len(schedule), 1)
    # A sampled/stratified entry already has the correct expectation; giving it
    # unit averaging weight avoids multiplying the loss twice by its probability.
    return [(float(schedule[slot]), 1.0)]


def _teacher_regret_weights(
    logits: torch.Tensor,
    target_action: torch.Tensor,
    valid: torch.Tensor,
    teacher_cost: torch.Tensor | None,
    *,
    strength: float,
    clip: float,
    min_scale: float,
) -> torch.Tensor:
    """Robust, stop-gradient weights for deployment mistakes with high regret.

    The raw teacher costs in nuPlan-style supervision are heavy tailed.  We
    therefore normalize regret by a per-scene robust scale and use ``log1p``
    before clipping.  This emphasizes consequential selector failures without
    letting a few extreme scenes dominate the batch.
    """
    if teacher_cost is None or float(strength) <= 0.0:
        return logits.new_ones((logits.shape[0],))
    with torch.no_grad():
        pred_action = logits.argmax(dim=1)
        safe_teacher = teacher_cost.float().masked_fill(~valid.bool(), float("inf"))
        pred_cost = torch.gather(safe_teacher, 1, pred_action[:, None]).squeeze(1)
        target_cost = torch.gather(safe_teacher, 1, target_action[:, None]).squeeze(1)
        regret = (pred_cost - target_cost).clamp_min(0.0)
        finite_teacher = torch.where(torch.isfinite(safe_teacher), safe_teacher, torch.zeros_like(safe_teacher))
        scale = _valid_row_scale(finite_teacher, valid, min_scale=float(min_scale)).squeeze(1)
        normalized = torch.log1p(regret / scale.clamp_min(1e-6))
        if float(clip) > 0.0:
            normalized = normalized.clamp_max(float(clip))
        return 1.0 + float(strength) * normalized


def _weighted_action_target_loss(
    logits: torch.Tensor,
    target_action: torch.Tensor,
    *,
    soft_target: torch.Tensor | None,
    scene_weights: torch.Tensor,
) -> torch.Tensor:
    """Teacher action loss with optional soft targets and per-scene weights."""
    if soft_target is not None:
        per_scene = -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1)
    else:
        per_scene = F.cross_entropy(logits, target_action, reduction="none")
    weights = scene_weights.detach().to(dtype=per_scene.dtype)
    return (per_scene * weights).sum() / weights.sum().clamp_min(1e-6)


def _pair_margin_scale(pair_margins: torch.Tensor, pair_mask: torch.Tensor, default: float = 100.0, quantile: float = 0.75) -> torch.Tensor:
    vals = pair_margins[pair_mask].detach().abs()
    if vals.numel() == 0:
        return pair_margins.new_tensor(float(default))
    q = max(0.5, min(float(quantile), 0.99))
    return torch.quantile(vals.float(), q).clamp_min(float(default))


def _masked_value_scale(values: torch.Tensor, mask: torch.Tensor, default: float = 100.0, quantile: float = 0.75) -> torch.Tensor:
    vals = values[mask.bool() & torch.isfinite(values)].detach().abs()
    if vals.numel() == 0:
        return values.new_tensor(float(default))
    q = max(0.5, min(float(quantile), 0.99))
    return torch.quantile(vals.float(), q).clamp_min(float(default))


def _pair_margin_scale_per_scene(pair_margins: torch.Tensor, pair_mask: torch.Tensor, default: float = 100.0, quantile: float = 0.75) -> torch.Tensor:
    """Robust per-sample normalization scale for pair margins. Output: [B,1].

    This is algebraically the same per-row quantile used before, but computes all
    rows in one tensor op instead of launching ``torch.quantile`` once per scene.
    The fallback branch preserves compatibility with older PyTorch builds that do
    not expose ``nanquantile``.
    """
    mask = pair_mask.bool() & torch.isfinite(pair_margins)
    q = max(0.5, min(float(quantile), 0.99))
    if hasattr(torch, "nanquantile"):
        vals = pair_margins.detach().abs().float().masked_fill(~mask, float("nan"))
        scales = torch.nanquantile(vals, q, dim=1)
        scales = torch.where(torch.isfinite(scales), scales, scales.new_full(scales.shape, float(default)))
        return scales.clamp_min(float(default)).view(pair_margins.shape[0], 1).to(pair_margins.device).detach()
    vals = pair_margins.detach().abs().masked_fill(~mask, 0.0)
    scales = []
    for b in range(vals.shape[0]):
        row = vals[b][mask[b]]
        scales.append(pair_margins.new_tensor(float(default)) if row.numel() == 0 else torch.quantile(row.float(), q).clamp_min(float(default)))
    return torch.stack(scales, dim=0).view(vals.shape[0], 1).detach()


def _margin_norm_min_scale(cfg: dict[str, Any]) -> float:
    mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    tcfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    return float(mcfg.get("margin_normalization_min_scale", tcfg.get("pair_margin_min_scale", 100.0)))


def _margin_norm_quantile(cfg: dict[str, Any]) -> float:
    mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    return float(mcfg.get("margin_normalization_quantile", 0.75))


def _safety_atom_mask_from_batch(batch: dict[str, torch.Tensor], e_mask: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    hard = torch.zeros_like(e_mask, dtype=torch.bool)
    feats = batch.get("evidence_features")
    if feats is not None and feats.shape[-1] > 0:
        hard = feats[..., 0].float() > 0.5
    fam = batch.get("evidence_family_ids")
    if bool(cfg.get("selector", {}).get("structural_safety_include_feasibility", True)) and fam is not None:
        hard = hard | (fam.long() == 1)
    return hard & e_mask.bool()


def _interaction_atom_mask_from_batch(batch: dict[str, torch.Tensor], e_mask: torch.Tensor) -> torch.Tensor:
    fam = batch.get("evidence_family_ids")
    if fam is None:
        return e_mask.bool()
    return ((fam.long() == 2) | (fam.long() == 3)) & e_mask.bool()


def _decision_pair_weights(
    base_weights: torch.Tensor,
    pairs: torch.Tensor,
    target_action: torch.Tensor,
    pair_margins: torch.Tensor,
    pair_scale: torch.Tensor,
    pair_mask: torch.Tensor,
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
) -> torch.Tensor:
    train_cfg = cfg.get("training", {})
    w = base_weights.float().clone()
    tgt = target_action[:, None]
    involves_teacher = (pairs[..., 0].long() == tgt) | (pairs[..., 1].long() == tgt)
    near = pair_margins.float().abs() <= pair_scale * float(train_cfg.get("decision_weight_near_margin", 0.5))
    w = w * (1.0 + float(train_cfg.get("decision_weight_teacher_pair", 2.0)) * involves_teacher.float())
    w = w * (1.0 + float(train_cfg.get("decision_weight_near_pair", 1.0)) * near.float())
    hard = batch.get("teacher_hard_violation")
    if hard is not None and bool(train_cfg.get("decision_weight_safety_pairs", True)):
        hard = hard.bool()
        a = pairs[..., 0].long().clamp(0, hard.shape[1] - 1)
        b = pairs[..., 1].long().clamp(0, hard.shape[1] - 1)
        crossing = torch.gather(hard, 1, a) ^ torch.gather(hard, 1, b)
        w = w * (1.0 + float(train_cfg.get("decision_weight_hard_crossing_pair", 1.0)) * crossing.float())
    return w.masked_fill(~pair_mask.bool(), 0.0)




def _online_teacher_hard_rival_loss(
    predicted_margin: torch.Tensor,
    pairs: torch.Tensor,
    pair_mask: torch.Tensor,
    target_action: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Mine the currently most dangerous teacher-vs-rival pairs online.

    Cached pair labels already contain teacher-vs-top rivals and near ties.  This
    term changes *which of those pairs receives the strongest gradient at the
    current checkpoint*: it selects the smallest predicted oriented margins,
    including sign-flipped failures, instead of relying only on static weights.
    """
    train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    hcfg = train_cfg.get("online_hard_rival", {}) or {}
    top_k = max(1, int(hcfg.get("top_k", 4)))
    tau = max(float(hcfg.get("tau", 0.12)), 1e-6)
    margin = float(hcfg.get("margin", 0.08))
    B, P = predicted_margin.shape
    tgt = target_action[:, None]
    a = pairs[..., 0].long()
    b = pairs[..., 1].long()
    orient = torch.where(a == tgt, torch.ones_like(predicted_margin), torch.zeros_like(predicted_margin))
    orient = torch.where(b == tgt, -torch.ones_like(predicted_margin), orient)
    valid = pair_mask.bool() & (orient != 0.0)
    oriented = predicted_margin * orient
    # Lowest oriented margins are the online hard rivals.  Detached selection
    # keeps the mining discrete while the selected margins retain gradients.
    rank_value = oriented.detach().masked_fill(~valid, torch.finfo(oriented.dtype).max)
    k = min(top_k, P)
    idx = torch.topk(rank_value, k=k, dim=1, largest=False, sorted=False).indices
    chosen_valid = valid.gather(1, idx)
    chosen_margin = oriented.gather(1, idx)
    loss = F.softplus((margin - chosen_margin) / tau)
    return (loss * chosen_valid.float()).sum() / chosen_valid.float().sum().clamp_min(1.0)


def _counterfactual_critical_evidence_loss(
    true_atom_delta: torch.Tensor,
    pred_atom_delta: torch.Tensor,
    predicted_margin: torch.Tensor,
    true_margin: torch.Tensor,
    pairs: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_weights: torch.Tensor,
    target_action: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_costs: torch.Tensor,
    proposal_logits: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decision-boundary critical evidence supervision.

    V47 rewarded every positive teacher/rival atom contribution.  That target
    increased interaction recall, but it did not identify whether removing an
    atom would actually erase the teacher action's margin.  V48 instead uses a
    leave-one-atom-out boundary deficit.  Rivals are mined from the union of the
    teacher-nearest and model-most-confused pairs, and each atom is weighted by
    the increase in decision deficit caused by removing it, divided by query
    cost.  The pair head and proposal head are trained with the same listwise
    target, so the learned ranking matches the fixed-budget selector objective.
    """
    train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    ccfg = train_cfg.get("counterfactual_critical_evidence", {}) or {}
    top_k = max(1, int(ccfg.get("top_k_rivals", 4)))
    rival_tau = max(float(ccfg.get("rival_temperature", 0.12)), 1e-6)
    atom_tau = max(float(ccfg.get("atom_temperature", 0.20)), 1e-6)
    proposal_tau = max(float(ccfg.get("proposal_temperature", 0.35)), 1e-6)
    cost_power = max(float(ccfg.get("cost_power", 1.0)), 0.0)
    min_gain = max(float(ccfg.get("min_gain", 1e-5)), 0.0)
    boundary_margin = float(ccfg.get("boundary_margin", 0.08))
    support_floor = max(float(ccfg.get("positive_support_floor", 0.0)), 0.0)
    target_top_k_atoms = max(0, int(ccfg.get("target_top_k_atoms", 0)))
    min_relative_gain = min(max(float(ccfg.get("min_relative_gain", 0.0)), 0.0), 1.0)
    teacher_rival_mix = float(ccfg.get("teacher_rival_mix", 0.6))
    teacher_rival_mix = min(max(teacher_rival_mix, 0.0), 1.0)

    B, E, P = pred_atom_delta.shape
    tgt = target_action[:, None]
    a = pairs[..., 0].long()
    b = pairs[..., 1].long()
    orient = torch.where(a == tgt, torch.ones_like(predicted_margin), torch.zeros_like(predicted_margin))
    orient = torch.where(b == tgt, -torch.ones_like(predicted_margin), orient)
    valid_pair = pair_mask.bool() & (orient != 0.0)

    oriented_pred_margin = predicted_margin * orient
    oriented_true_margin = true_margin * orient
    rank_teacher = oriented_true_margin.detach().masked_fill(~valid_pair, torch.finfo(oriented_true_margin.dtype).max)
    rank_model = oriented_pred_margin.detach().masked_fill(~valid_pair, torch.finfo(oriented_pred_margin.dtype).max)
    k = min(top_k, P)
    teacher_idx = torch.topk(rank_teacher, k=k, dim=1, largest=False, sorted=False).indices
    model_idx = torch.topk(rank_model, k=k, dim=1, largest=False, sorted=False).indices
    hard_mask = torch.zeros_like(valid_pair)
    hard_mask.scatter_(1, teacher_idx, valid_pair.gather(1, teacher_idx))
    hard_mask.scatter_(1, model_idx, valid_pair.gather(1, model_idx))

    base_pair_weight = pair_weights.float().clamp_min(0.0)
    if base_pair_weight.shape != predicted_margin.shape:
        base_pair_weight = torch.ones_like(predicted_margin)
    teacher_logits = -oriented_true_margin.detach() / rival_tau
    model_logits = -oriented_pred_margin.detach() / rival_tau
    hard_logits = teacher_rival_mix * teacher_logits + (1.0 - teacher_rival_mix) * model_logits
    hard_logits = hard_logits + torch.log1p(base_pair_weight.detach())
    hard_logits = hard_logits.masked_fill(~hard_mask, -1e4)
    hard_weight = torch.softmax(hard_logits, dim=1) * hard_mask.float()
    hard_weight = hard_weight / hard_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)

    oriented_true_atom = true_atom_delta * orient[:, None, :]
    oriented_pred_atom = pred_atom_delta * orient[:, None, :]
    # Removing atom i changes m to m-d_i.  The positive difference between the
    # resulting boundary deficit and the full-interface deficit is the causal
    # contribution of that atom to preserving the teacher/rival decision.
    full_true_deficit = torch.relu(boundary_margin - oriented_true_margin)[:, None, :]
    loo_true_deficit = torch.relu(boundary_margin - (oriented_true_margin[:, None, :] - oriented_true_atom))
    true_utility_pair = torch.relu(loo_true_deficit - full_true_deficit)
    if support_floor > 0.0:
        true_utility_pair = true_utility_pair + support_floor * torch.relu(oriented_true_atom)

    full_pred_deficit = torch.relu(boundary_margin - oriented_pred_margin)[:, None, :]
    loo_pred_deficit = torch.relu(boundary_margin - (oriented_pred_margin[:, None, :] - oriented_pred_atom))
    pred_utility_pair = torch.relu(loo_pred_deficit - full_pred_deficit)
    if support_floor > 0.0:
        pred_utility_pair = pred_utility_pair + support_floor * torch.relu(oriented_pred_atom)

    true_gain = (true_utility_pair * hard_weight[:, None, :]).sum(dim=2)
    pred_gain = (pred_utility_pair * hard_weight[:, None, :]).sum(dim=2)

    costs = atom_costs.float().clamp_min(1e-3)
    if costs.shape != true_gain.shape:
        costs = torch.ones_like(true_gain)
    cost_adjust = costs.pow(cost_power)
    target_gain = (true_gain / cost_adjust).masked_fill(~atom_mask.bool(), 0.0)
    # DBAP uses a sparse boundary-critical target.  Retaining every atom with a
    # tiny positive gain turns the objective back into positive-support recall
    # and makes the pair loss approach the entropy of the entire evidence bank.
    # The optional relative threshold and top-k are computed from detached
    # teacher utilities, so they only define the target support.
    if min_relative_gain > 0.0:
        scene_max = target_gain.detach().amax(dim=1, keepdim=True)
        target_gain = target_gain.masked_fill(target_gain < min_relative_gain * scene_max, 0.0)
    if target_top_k_atoms > 0 and target_top_k_atoms < E:
        k_atom = min(target_top_k_atoms, E)
        keep_idx = torch.topk(target_gain.detach(), k=k_atom, dim=1, largest=True, sorted=False).indices
        keep_mask = torch.zeros_like(atom_mask, dtype=torch.bool)
        keep_mask.scatter_(1, keep_idx, True)
        target_gain = target_gain.masked_fill(~keep_mask, 0.0)
    scene_has_target = target_gain.sum(dim=1) > min_gain
    target_dist = target_gain / target_gain.sum(dim=1, keepdim=True).clamp_min(1e-6)

    mask_value = _neg_mask_value(pred_gain)
    pair_logits = (pred_gain / cost_adjust) / atom_tau
    pair_logits = pair_logits.masked_fill(~atom_mask.bool(), mask_value)
    pair_ce = -(target_dist * F.log_softmax(pair_logits, dim=1)).sum(dim=1)
    proposal_score = (proposal_logits / cost_adjust) / proposal_tau
    proposal_score = proposal_score.masked_fill(~atom_mask.bool(), mask_value)
    proposal_ce = -(target_dist * F.log_softmax(proposal_score, dim=1)).sum(dim=1)
    denom = scene_has_target.float().sum().clamp_min(1.0)
    return (pair_ce * scene_has_target.float()).sum() / denom, (proposal_ce * scene_has_target.float()).sum() / denom

def _pair_cycle_consistency_loss(
    predicted_margin: torch.Tensor,
    pairs: torch.Tensor,
    pair_mask: torch.Tensor,
    target_action: torch.Tensor,
    valid_actions: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Triangle consistency for pair-conditioned scalar-cost differences.

    The selected triangle set is discrete and therefore stop-gradient.  Build
    it from a single batched CPU snapshot instead of repeatedly calling
    ``item()`` / ``bool()`` on CUDA tensors inside Python loops.  The selected
    margins are then gathered from the original GPU tensor, preserving the exact
    objective and gradients while eliminating hundreds of per-step CUDA
    synchronizations.
    """
    train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    ccfg = train_cfg.get("cycle_consistency", {}) or {}
    max_triangles = max(1, int(ccfg.get("max_triangles_per_scene", 64)))
    delta = max(float(ccfg.get("huber_delta", 0.15)), 1e-6)
    B, P = predicted_margin.shape
    K = int(valid_actions.shape[1])

    # Collect all topology/ranking inputs in one batched snapshot phase rather
    # than synchronizing from inside the pair/triangle loops. ``predicted_margin``
    # is detached only for triangle ranking; gradients flow through the gather below.
    pair_np = pairs.detach().to(device="cpu").numpy()
    mask_np = pair_mask.detach().to(device="cpu").numpy().astype(bool, copy=False)
    target_np = target_action.detach().to(device="cpu").numpy()
    valid_np = valid_actions.detach().to(device="cpu").numpy().astype(bool, copy=False)
    margin_np = predicted_margin.detach().float().to(device="cpu").numpy()

    scene_ids: list[int] = []
    pair_ids: list[tuple[int, int, int]] = []
    pair_signs: list[tuple[float, float, float]] = []
    for bi in range(B):
        # Match the legacy dict semantics exactly: later duplicate pair entries
        # overwrite earlier ones, and every directed edge also installs its
        # antisymmetric reverse edge.
        edge: dict[tuple[int, int], tuple[int, float]] = {}
        for pi in range(P):
            if not bool(mask_np[bi, pi]):
                continue
            a = int(pair_np[bi, pi, 0])
            b = int(pair_np[bi, pi, 1])
            if a == b or not (0 <= a < K and 0 <= b < K):
                continue
            edge[(a, b)] = (pi, 1.0)
            edge[(b, a)] = (pi, -1.0)

        actions = np.flatnonzero(valid_np[bi]).astype(np.int64, copy=False).tolist()
        tgt = int(target_np[bi])
        candidates: list[
            tuple[
                int,
                float,
                int,
                int,
                int,
                tuple[int, int, int],
                tuple[float, float, float],
            ]
        ] = []
        for ia, a in enumerate(actions):
            for ib in range(ia + 1, len(actions)):
                b = int(actions[ib])
                for ic in range(ib + 1, len(actions)):
                    c = int(actions[ic])
                    refs = (edge.get((a, b)), edge.get((b, c)), edge.get((c, a)))
                    if any(ref is None for ref in refs):
                        continue
                    r0, r1, r2 = refs  # type: ignore[misc]
                    ids = (int(r0[0]), int(r1[0]), int(r2[0]))
                    signs = (float(r0[1]), float(r1[1]), float(r2[1]))
                    boundary = min(
                        abs(signs[0] * float(margin_np[bi, ids[0]])),
                        abs(signs[1] * float(margin_np[bi, ids[1]])),
                        abs(signs[2] * float(margin_np[bi, ids[2]])),
                    )
                    contains_tgt = 0 if tgt in (a, b, c) else 1
                    candidates.append((contains_tgt, boundary, a, b, c, ids, signs))
        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        for item in candidates[:max_triangles]:
            scene_ids.append(bi)
            pair_ids.append(item[5])
            pair_signs.append(item[6])

    if not scene_ids:
        return predicted_margin.new_tensor(0.0)

    scene_t = torch.as_tensor(scene_ids, dtype=torch.long, device=predicted_margin.device)
    pair_t = torch.as_tensor(pair_ids, dtype=torch.long, device=predicted_margin.device)
    sign_t = torch.as_tensor(pair_signs, dtype=predicted_margin.dtype, device=predicted_margin.device)
    gathered = predicted_margin[scene_t[:, None], pair_t]
    cyc = (gathered * sign_t).sum(dim=1)
    abs_cyc = cyc.abs()
    return torch.where(abs_cyc <= delta, 0.5 * cyc.square() / delta, abs_cyc - 0.5 * delta).mean()



def _critical_pair_mask(
    pairs: torch.Tensor,
    target_action: torch.Tensor,
    pair_margins: torch.Tensor,
    pair_scale: torch.Tensor,
    pair_mask: torch.Tensor,
    valid: torch.Tensor,
    hard_action: torch.Tensor | None,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return closed-loop-critical pair mask and weights for CACE.

    This supervision focuses on the exact pairs whose evidence must survive a
    small query budget: teacher-vs-rival decision pairs, near-boundary pairs, and
    safe-vs-unsafe crossings.  Unlike proposal BCE over all oracle atoms, it
    supervises the pair-conditioned margin head and proposal head on the same
    frontier that the closed-loop selector/tournament uses at deployment.
    """
    train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    cm = train_cfg.get("critical_evidence", {}) or {}
    B, P = pair_mask.shape
    K = valid.shape[1]
    a = pairs[..., 0].long().clamp(0, K - 1)
    b = pairs[..., 1].long().clamp(0, K - 1)
    tgt = target_action.view(B, 1).expand(B, P)
    teacher_pair = (a == tgt) | (b == tgt)
    norm_margin = pair_margins.float().abs() / pair_scale.view(B, 1).clamp_min(1e-6)
    near_tau = float(cm.get("near_margin_tau", train_cfg.get("decision_weight_near_margin", 0.5)))
    near_pair = norm_margin <= max(near_tau, 1e-6)
    crossing = torch.zeros_like(pair_mask, dtype=torch.bool)
    if hard_action is not None:
        h = hard_action.bool() & valid.bool()
        ha = torch.gather(h, 1, a)
        hb = torch.gather(h, 1, b)
        crossing = ha ^ hb
    crit = pair_mask.bool() & (teacher_pair | near_pair | crossing)
    base_w = torch.ones_like(pair_margins.float())
    base_w = base_w + float(cm.get("teacher_pair_weight", 3.0)) * teacher_pair.float()
    base_w = base_w + float(cm.get("near_pair_weight", 2.0)) * near_pair.float()
    base_w = base_w + float(cm.get("hard_crossing_weight", 4.0)) * crossing.float()
    return crit, base_w.masked_fill(~crit, 0.0)


def _critical_atom_targets(
    true_atom_delta: torch.Tensor,
    pred_atom_delta: torch.Tensor,
    critical_pair_mask: torch.Tensor,
    critical_pair_weights: torch.Tensor,
    atom_mask: torch.Tensor,
    interaction_atom_mask: torch.Tensor,
    safety_atom_mask: torch.Tensor | None,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Atom-level criticality targets derived from positive support on critical pairs."""
    train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    cm = train_cfg.get("critical_evidence", {}) or {}
    pos_thr = float(cm.get("positive_support_threshold", 1e-4))
    # Positive true support means the atom increases the margin of the logged
    # winner over the rival on at least one closed-loop-critical pair.
    pair_w = critical_pair_weights[:, None, :]
    crit_pair = critical_pair_mask[:, None, :]
    active = atom_mask.bool()
    true_pos = (true_atom_delta > pos_thr) & crit_pair & active[:, :, None]
    gain = torch.relu(true_atom_delta) * pair_w * active[:, :, None].float()
    gain = gain.masked_fill(~crit_pair, 0.0).sum(dim=2)
    # Normalize per scene for a stable listwise proposal target.
    hard_focus = torch.zeros_like(interaction_atom_mask, dtype=torch.bool) if safety_atom_mask is None else safety_atom_mask.bool()
    focus_mask = interaction_atom_mask.bool()
    if bool(cm.get("include_safety_atoms", False)):
        focus_mask = focus_mask | hard_focus
    gain = gain * (
        1.0
        + float(cm.get("interaction_gain_boost", 2.0)) * interaction_atom_mask.float()
        + float(cm.get("safety_gain_boost", 1.5)) * hard_focus.float()
    )
    target = true_pos.any(dim=2) & active
    if bool(cm.get("interaction_only", True)):
        # v24 used interaction_only=True and therefore dropped hard/rule atoms
        # even on safe-vs-unsafe critical crossings.  v25 keeps the interaction
        # focus but can include structural safety atoms, which are exactly the
        # atoms needed to reduce closed-loop collision/TTC failures.
        target = target & focus_mask
        gain = gain * focus_mask.float()
    # Predicted signed support on the same pair set is used for a ranking loss.
    pred_gain = (pred_atom_delta * pair_w).masked_fill(~crit_pair, 0.0).sum(dim=2)
    return target, gain, pred_gain



def _certificate_action_gap_loss(
    logits: torch.Tensor | None,
    target_action: torch.Tensor,
    valid: torch.Tensor,
    hard_action: torch.Tensor | None,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable proxy for the deployment fallback gate.

    CACE makes the right atoms visible, but v24 closed-loop still triggered the
    fallback stage on essentially every replan.  This loss teaches the
    selected-budget tournament to produce an accepted action certificate: the
    teacher action should beat its best valid rival by a small normalized gap,
    and hard-unsafe actions should not outrank the teacher/safe frontier.
    """
    if logits is None:
        z = target_action.new_zeros((), dtype=torch.float32)
        return z, z, z
    train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    gc = train_cfg.get("certificate_gap", {}) or {}
    B, K = logits.shape
    tgt = target_action.long().clamp(0, K - 1)
    tscore = torch.gather(logits, 1, tgt[:, None]).squeeze(1)
    rival_mask = valid.bool().clone()
    rival_mask.scatter_(1, tgt[:, None], False)
    rival_scores = logits.masked_fill(~rival_mask, _neg_mask_value(logits))
    best_rival = rival_scores.max(dim=1).values
    margin = float(gc.get("margin", 0.08))
    tau = max(float(gc.get("tau", 0.08)), 1e-6)
    L_gap = F.softplus((best_rival - tscore + margin) / tau).mean()
    if hard_action is None:
        z = logits.new_tensor(0.0)
        return L_gap, z, z
    h = hard_action.bool() & valid.bool()
    safe = (~hard_action.bool()) & valid.bool()
    hard_scores = logits.masked_fill(~h, _neg_mask_value(logits))
    safe_scores = logits.masked_fill(~safe, _neg_mask_value(logits))
    best_hard = hard_scores.max(dim=1).values
    best_safe = safe_scores.max(dim=1).values
    has_hard = h.any(dim=1)
    has_safe = safe.any(dim=1)
    safe_margin = float(gc.get("safety_margin", margin))
    # Mask *before* softplus.  Computing softplus on sentinel differences and
    # multiplying by a zero mask afterwards is numerically unsafe: an all-hard
    # scene has best_safe=-mask_sentinel, which can yield inf, and inf*0 is NaN.
    safe_arg = torch.where(
        has_hard,
        (best_hard - tscore + safe_margin) / tau,
        torch.zeros_like(best_hard),
    )
    L_safe_terms = F.softplus(safe_arg)
    L_safe = (L_safe_terms * has_hard.float()).sum() / has_hard.float().sum().clamp_min(1.0)
    frontier_margin = float(gc.get("safe_frontier_margin", safe_margin))
    frontier_mask = has_hard & has_safe
    frontier_arg = torch.where(
        frontier_mask,
        (best_hard - best_safe + frontier_margin) / tau,
        torch.zeros_like(best_hard),
    )
    frontier_terms = F.softplus(frontier_arg)
    L_frontier = (frontier_terms * frontier_mask.float()).sum() / frontier_mask.float().sum().clamp_min(1.0)
    return L_gap, L_safe, L_frontier

def compute_bdse_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    # Losses include large teacher costs and masking sentinels.  Compute them in
    # float32 even when model forward uses CUDA AMP; otherwise fp16 masks such as
    # 1e9/-1e9 overflow.
    out = {
        k: (v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v)
        for k, v in outputs.items()
    }
    J0 = out["J0"]
    g = out["g"]
    g_var = out.get("g_var")
    valid = batch["candidate_valid"].bool()
    J_base_T = batch["teacher_J_base"].float()
    pairs = batch["pair_indices"].long()
    pair_valid = batch["pair_valid"].bool()
    pair_weights = batch["pair_weights"].float()
    residual_T = batch["pair_residuals"].float()
    target_action = batch["teacher_a_star"].long()
    train_cfg = cfg.get("training", {})
    lw = train_cfg.get("loss_weights", {})

    finite_J0 = torch.where(torch.isfinite(J0), J0, torch.zeros_like(J0))
    if bool(train_cfg.get("normalize_base_loss", False)):
        base_scale = _valid_row_scale(
            J_base_T.masked_fill(~valid, 0.0),
            valid,
            min_scale=float(train_cfg.get("base_loss_min_scale", 100.0)),
        )
        base_center = (
            torch.where(valid, J_base_T, torch.zeros_like(J_base_T)).sum(dim=1, keepdim=True)
            / valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        ).detach()
        pred_base_norm = (finite_J0 - base_center) / base_scale
        target_base_norm = (J_base_T - base_center) / base_scale
        base_clip = float(train_cfg.get("base_target_clip_normalized", 0.0))
        if base_clip > 0.0:
            pred_base_norm = pred_base_norm.clamp(-base_clip, base_clip)
            target_base_norm = target_base_norm.clamp(-base_clip, base_clip)
        L_base = robust_loss(pred_base_norm, target_base_norm, valid)
    else:
        L_base = robust_loss(finite_J0, J_base_T, valid)

    g_a, g_b = pair_gather(g, pairs)
    pair_mask = pair_valid & torch.isfinite(residual_T)
    pair_scale = _pair_margin_scale_per_scene(
        batch["pair_margins"].float(),
        pair_mask,
        default=_margin_norm_min_scale(cfg),
        quantile=_margin_norm_quantile(cfg),
    )
    normalize_pair_losses = bool(train_cfg.get("normalize_pair_losses", True))
    pair_target_clip = float(train_cfg.get(
        "pair_target_clip_normalized" if normalize_pair_losses else "pair_target_clip",
        0.0,
    ))
    residual_target_clip = float(train_cfg.get(
        "residual_target_clip_normalized" if normalize_pair_losses else "residual_target_clip",
        pair_target_clip,
    ))
    margin_target_clip = float(train_cfg.get(
        "full_pair_margin_clip_normalized" if normalize_pair_losses else "full_pair_margin_clip",
        residual_target_clip,
    ))

    e_mask = batch.get("evidence_active", torch.ones_like(out["proposal_logits"]).bool()).bool()
    safety_atom_mask = _safety_atom_mask_from_batch(batch, e_mask, cfg)
    interaction_atom_mask = _interaction_atom_mask_from_batch(batch, e_mask)
    pair_atom_train_mask = e_mask
    if bool(train_cfg.get("exclude_safety_atoms_from_pair_regression", True)):
        pair_atom_train_mask = pair_atom_train_mask & (~safety_atom_mask)

    # Prefer the paper-faithful pair-conditioned scorer d_i(a,b) when present.
    # In the normalized-margin version, pair_atom_delta is treated as a
    # dimensionless interaction evidence contribution.  The raw factorized
    # g_i(b)-g_i(a) path is normalized on the fly for backward-compatible
    # ablations and dense diagnostics.
    raw_atom_delta = g_b - g_a
    local_atom_delta = raw_atom_delta / pair_scale[:, None, :].clamp_min(1e-6) if normalize_pair_losses else raw_atom_delta
    pair_head_residual = bool(cfg.get("model", {}).get("pair_head_residual_over_local", False))
    pair_var_pred = out.get("pair_atom_var")
    if "pair_atom_delta" in out:
        pair_head_delta = out["pair_atom_delta"]
        if pair_head_residual:
            pred_atom_delta = confidence_shrunk_residual_pair_delta_torch(
                local_atom_delta,
                pair_head_delta,
                pair_var_pred,
                (cfg.get("runtime", {}).get("pair_residual_trust", {}) or {}),
            )
        else:
            pred_atom_delta = pair_head_delta
    else:
        pair_head_delta = local_atom_delta
        pred_atom_delta = local_atom_delta
    res_pred = pred_atom_delta.sum(dim=1)

    hard_pair_mask = torch.zeros_like(pair_mask, dtype=torch.bool)
    hard_action = batch.get("teacher_hard_violation")
    if hard_action is not None:
        hard_action = hard_action.bool() & valid
        a_idx = pairs[..., 0].long().clamp(0, hard_action.shape[1] - 1)
        b_idx = pairs[..., 1].long().clamp(0, hard_action.shape[1] - 1)
        hard_pair_mask = torch.gather(hard_action, 1, a_idx) | torch.gather(hard_action, 1, b_idx)
    pair_train_mask = pair_mask
    if bool(train_cfg.get("exclude_hard_action_pairs_from_pair_regression", True)):
        pair_train_mask = pair_train_mask & (~hard_pair_mask)
    decision_w = _decision_pair_weights(
        pair_weights,
        pairs,
        target_action,
        batch["pair_margins"].float(),
        pair_scale,
        pair_mask,
        batch,
        cfg,
    )

    # V50 DBAP-RI: the residual head is a boundary intervention, not a second
    # full reconstruction interface.  Concentrate residual regression on pairs
    # for which the integrable local interface is wrong or close to the teacher
    # boundary.  Decision/action losses below still supervise the exact gated
    # deployment path on every decision-weighted pair.
    correction_focus = torch.ones_like(decision_w)
    focus_cfg = train_cfg.get("residual_correction_focus", {}) or {}
    if pair_head_residual and bool(focus_cfg.get("enabled", True)):
        early_J_a, early_J_b = pair_gather(finite_J0, pairs)
        early_base = (early_J_b - early_J_a) / pair_scale.clamp_min(1e-6) if normalize_pair_losses else (early_J_b - early_J_a)
        local_margin_for_focus = early_base + local_atom_delta.detach().sum(dim=1)
        teacher_margin_for_focus = batch["pair_margins"].float() / pair_scale.clamp_min(1e-6) if normalize_pair_losses else batch["pair_margins"].float()
        tau_focus = float(focus_cfg.get("boundary_tau", 0.35))
        wrong_sign = (local_margin_for_focus.detach() * teacher_margin_for_focus.detach()) <= 0.0
        near_boundary = (local_margin_for_focus.detach().abs() <= tau_focus) | (teacher_margin_for_focus.detach().abs() <= tau_focus)
        correction_focus = torch.full_like(decision_w, float(focus_cfg.get("default_weight", 0.10)))
        correction_focus = correction_focus + float(focus_cfg.get("wrong_sign_weight", 4.0)) * wrong_sign.float()
        correction_focus = correction_focus + float(focus_cfg.get("near_boundary_weight", 2.0)) * near_boundary.float()
        correction_focus = correction_focus.clamp(max=float(focus_cfg.get("max_weight", 6.0)))
        correction_focus = torch.where(pair_mask, correction_focus, torch.zeros_like(correction_focus))

    residual_target_norm = residual_T / pair_scale.clamp_min(1e-6)
    if residual_target_clip > 0:
        residual_target_norm = residual_target_norm.clamp(-residual_target_clip, residual_target_clip)
    residual_weight = decision_w * correction_focus
    residual_target_for_loss = residual_target_norm if normalize_pair_losses else residual_T
    residual_delta = float(train_cfg.get("normalized_huber_delta", 1.0)) if normalize_pair_losses else 1.0
    safe_res_pred = torch.where(pair_train_mask, res_pred, torch.zeros_like(res_pred))
    safe_res_target = torch.where(pair_train_mask, residual_target_for_loss, torch.zeros_like(residual_target_for_loss))
    L_res_terms = F.huber_loss(
        safe_res_pred,
        safe_res_target,
        delta=residual_delta,
        reduction="none",
    )
    L_res = _weighted_mean(L_res_terms, residual_weight, pair_train_mask)

    # Explicit action-conditioned local-cost supervision keeps the factorized
    # g_i(a) head usable for diagnostics/fallback.  For hard/safety atoms this
    # remains separate from pair-margin regression: hard feasibility is handled
    # by L_hard_feas and structural selection, not by learning huge pair deltas.
    teacher_g = batch.get("teacher_g_evid")
    if teacher_g is not None:
        atom_cost_mask = e_mask[:, :, None] & valid[:, None, :] & torch.isfinite(teacher_g.float())
        if bool(train_cfg.get("normalize_atom_cost_loss", True)):
            atom_scale = _masked_value_scale(teacher_g.float(), atom_cost_mask, default=float(train_cfg.get("atom_cost_min_scale", _margin_norm_min_scale(cfg))), quantile=_margin_norm_quantile(cfg))
            atom_cost_err = F.huber_loss(g / atom_scale, teacher_g.float() / atom_scale, delta=float(train_cfg.get("normalized_huber_delta", 1.0)), reduction="none")
        else:
            atom_cost_err = F.huber_loss(g, teacher_g.float(), delta=1.0, reduction="none")
        nonzero_g = teacher_g.float().abs() > 1e-6
        atom_zero_w = float(train_cfg.get("atom_zero_weight", train_cfg.get("pair_zero_weight", 0.1)))
        atom_cost_w = atom_zero_w + (1.0 - atom_zero_w) * nonzero_g.float()
        safety_cost_w = float(train_cfg.get("safety_atom_cost_weight", 1.0))
        if safety_cost_w != 1.0:
            atom_cost_w = torch.where(safety_atom_mask[:, :, None], atom_cost_w * safety_cost_w, atom_cost_w)
        L_atom = _weighted_mean(atom_cost_err, atom_cost_w, atom_cost_mask)
    else:
        L_atom = J0.new_tensor(0.0)

    if teacher_g is not None:
        tg_a, tg_b = pair_gather(teacher_g.float(), pairs)
        true_atom_delta_raw = tg_b - tg_a
        true_atom_delta = true_atom_delta_raw / pair_scale[:, None, :].clamp_min(1e-6) if normalize_pair_losses else true_atom_delta_raw
        if pair_target_clip > 0:
            true_atom_delta = true_atom_delta.clamp(-pair_target_clip, pair_target_clip)
        atom_pair_mask = pair_atom_train_mask[:, :, None] & pair_train_mask[:, None, :]
        nonzero = true_atom_delta.abs() > 1e-6
        zero_w = float(train_cfg.get("pair_zero_weight", 0.1))
        atom_weights = (decision_w * correction_focus)[:, None, :] * (zero_w + (1.0 - zero_w) * nonzero.float())
        if bool(train_cfg.get("upweight_interaction_atoms", True)):
            atom_weights = torch.where(
                interaction_atom_mask[:, :, None],
                atom_weights * float(train_cfg.get("interaction_atom_pair_weight", 2.0)),
                atom_weights,
            )
        if pair_head_residual and "pair_atom_delta" in out:
            pair_pred_for_loss = pair_head_delta
            pair_target_for_loss = true_atom_delta - local_atom_delta.detach()
        else:
            pair_pred_for_loss = pred_atom_delta
            pair_target_for_loss = true_atom_delta
        if normalize_pair_losses:
            L_pair_terms = F.huber_loss(pair_pred_for_loss, pair_target_for_loss, delta=float(train_cfg.get("normalized_huber_delta", 1.0)), reduction="none")
        else:
            L_pair_terms = F.huber_loss(pair_pred_for_loss, pair_target_for_loss, delta=1.0, reduction="none")
        L_pair = _weighted_mean(L_pair_terms, atom_weights, atom_pair_mask)
    else:
        true_atom_delta = None
        atom_pair_mask = None
        atom_weights = None
        L_pair = J0.new_tensor(0.0)

    # Heteroscedastic uncertainty in normalized pair-margin space.
    pair_var_pred = out.get("pair_atom_var")
    uncertainty_loss_enabled = float(lw.get("uncertainty", 0.0)) > 0.0
    if uncertainty_loss_enabled and true_atom_delta is not None and atom_pair_mask is not None and atom_weights is not None and (pair_var_pred is not None or g_var is not None):
        if pair_var_pred is not None:
            pair_var = pair_var_pred.clamp_min(1e-6)
        else:
            v_a, v_b = pair_gather(g_var, pairs)
            pair_var = ((v_a + v_b) / pair_scale[:, None, :].pow(2).clamp_min(1e-6)).clamp_min(1e-6)
        err2 = (pred_atom_delta - true_atom_delta).pow(2)
        pair_var_norm = pair_var.clamp_min(float(train_cfg.get("uncertainty_normalized_var_floor", 5e-2)))
        var_ceiling = train_cfg.get("uncertainty_normalized_var_ceiling", 20.0)
        if var_ceiling is not None and float(var_ceiling) > 0:
            pair_var_norm = pair_var_norm.clamp_max(float(var_ceiling))
        nll = 0.5 * (err2 / pair_var_norm + torch.log(pair_var_norm))
        nll_max = train_cfg.get("uncertainty_nll_clamp", 20.0)
        if nll_max is not None and float(nll_max) > 0:
            nll = nll.clamp_max(float(nll_max))
        nll_min = train_cfg.get("uncertainty_nll_min", 0.0)
        if nll_min is not None:
            nll = nll.clamp_min(float(nll_min))
        L_unc = _weighted_mean(nll, atom_weights, atom_pair_mask)
    else:
        L_unc = J0.new_tensor(0.0)

    J_a_pair, J_b_pair = pair_gather(finite_J0, pairs)
    base_margin_norm = (J_b_pair - J_a_pair) / pair_scale.clamp_min(1e-6) if normalize_pair_losses else (J_b_pair - J_a_pair)
    M_hat_E = base_margin_norm + pred_atom_delta.sum(dim=1)
    tau_rank = float(train_cfg.get("rank_tau", train_cfg.get("rank_margin", 1.0)))
    rank_terms = F.softplus(-M_hat_E / max(tau_rank, 1e-6))
    L_rank_cls = _weighted_mean(rank_terms, decision_w, pair_train_mask)
    target_margin_norm = batch["pair_margins"].float() / pair_scale.clamp_min(1e-6) if normalize_pair_losses else batch["pair_margins"].float()
    if margin_target_clip > 0:
        target_margin_norm = target_margin_norm.clamp(-margin_target_clip, margin_target_clip)
    safe_rank_pred = torch.where(pair_train_mask, M_hat_E, torch.zeros_like(M_hat_E))
    safe_rank_target = torch.where(pair_train_mask, target_margin_norm, torch.zeros_like(target_margin_norm))
    reg_terms = F.huber_loss(
        safe_rank_pred,
        safe_rank_target,
        delta=float(train_cfg.get("pair_margin_reg_delta", 1.0)),
        reduction="none",
    )
    L_rank_reg = _weighted_mean(reg_terms, decision_w, pair_train_mask)
    L_rank = L_rank_cls + float(train_cfg.get("pair_margin_reg_weight", 1.0)) * L_rank_reg
    L_online_hard_rival = _online_teacher_hard_rival_loss(
        M_hat_E, pairs, pair_train_mask, target_action, cfg
    )
    L_cycle = _pair_cycle_consistency_loss(
        M_hat_E, pairs, pair_train_mask, target_action, valid, cfg
    )
    if true_atom_delta is not None:
        atom_costs_for_critical = batch.get(
            "evidence_budget_costs", torch.ones_like(out["proposal_logits"])
        )
        L_cf_critical_pair, L_cf_critical_proposal = _counterfactual_critical_evidence_loss(
            true_atom_delta, pred_atom_delta, M_hat_E, target_margin_norm,
            pairs, pair_train_mask, pair_weights, target_action, e_mask,
            atom_costs_for_critical, out["proposal_logits"], cfg
        )
    else:
        L_cf_critical_pair = J0.new_tensor(0.0)
        L_cf_critical_proposal = J0.new_tensor(0.0)

    target_sel = batch.get("oracle_selected_mask")
    decisive = batch.get("decisive_atom_mask")
    decisive_hard = batch.get("decisive_hard_mask")
    gain = batch.get("proposal_target_gain")
    if target_sel is not None:
        target_bool = target_sel.bool()
        if bool(train_cfg.get("proposal_focus_interaction_only", True)):
            target_bool = target_bool & interaction_atom_mask
        if decisive is not None and bool(train_cfg.get("proposal_include_decisive_atoms", True)):
            decisive_target = decisive.bool() & (interaction_atom_mask if bool(train_cfg.get("proposal_focus_interaction_only", True)) else e_mask)
            target_bool = target_bool | decisive_target
        if decisive_hard is not None and bool(train_cfg.get("proposal_include_decisive_hard", False)):
            target_bool = target_bool | decisive_hard.bool()
        bce = F.binary_cross_entropy_with_logits(out["proposal_logits"], target_bool.float(), reduction="none")
        pos_w = float(train_cfg.get("proposal_positive_weight", 4.0))
        hard_w = float(train_cfg.get("proposal_decisive_hard_weight", 1.0))
        weights_prop = torch.ones_like(bce)
        weights_prop = torch.where(target_bool, weights_prop * pos_w, weights_prop)
        if decisive_hard is not None and bool(train_cfg.get("proposal_include_decisive_hard", False)):
            weights_prop = torch.where(decisive_hard.bool(), weights_prop * hard_w, weights_prop)
        mask_prop = e_mask & torch.isfinite(bce)
        L_prop_bce = _weighted_mean(bce, weights_prop, mask_prop)
    else:
        L_prop_bce = J0.new_tensor(0.0)
    if gain is not None:
        target_gain = gain.float().masked_fill(~e_mask, 0.0)
        if bool(train_cfg.get("proposal_focus_interaction_only", True)):
            target_gain = target_gain * interaction_atom_mask.float()
        if decisive is not None:
            target_gain = target_gain + float(train_cfg.get("proposal_interaction_gain_boost", 1.0)) * decisive.float() * interaction_atom_mask.float()
        if decisive_hard is not None and bool(train_cfg.get("proposal_include_decisive_hard", False)):
            boost = float(train_cfg.get("proposal_decisive_hard_gain_boost", 0.0))
            target_gain = target_gain + boost * decisive_hard.float()
        target_dist = target_gain / target_gain.sum(dim=1, keepdim=True).clamp_min(1e-6)
        logp = F.log_softmax(out["proposal_logits"].masked_fill(~e_mask, _neg_mask_value(out["proposal_logits"])), dim=1)
        L_prop_rank = -(target_dist * logp).sum(dim=1).mean()
    else:
        L_prop_rank = J0.new_tensor(0.0)
    L_prop = L_prop_bce + float(train_cfg.get("proposal_rank_weight", 0.5)) * L_prop_rank

    # v24 CACE: Closed-loop Action-Critical Evidence supervision.  v23 showed
    # that better budget allocation alone cannot recover interaction recall or
    # closed-loop score.  This term directly trains the pair-conditioned head and
    # proposal logits on teacher-vs-rival / near-boundary / hard-crossing pairs,
    # i.e. the evidence that must be preserved under a fixed budget.
    cm_cfg = train_cfg.get("critical_evidence", {}) or {}
    enable_cace = bool(cm_cfg.get("enabled", False)) and true_atom_delta is not None
    if enable_cace and atom_pair_mask is not None:
        crit_pair_mask, crit_pair_w = _critical_pair_mask(
            pairs,
            target_action,
            batch["pair_margins"].float(),
            pair_scale,
            pair_mask,
            valid,
            hard_action,
            cfg,
        )
        critical_atom_mask = pair_atom_train_mask
        crit_target, crit_gain, crit_pred_gain = _critical_atom_targets(
            true_atom_delta,
            pred_atom_delta,
            crit_pair_mask,
            crit_pair_w,
            critical_atom_mask,
            interaction_atom_mask,
            safety_atom_mask,
            cfg,
        )
        # Pair-head signed support: critical atoms should assign positive mass to
        # the critical frontier; non-critical active atoms are downweighted but not
        # ignored, which improves ranking calibration without forcing sparsity.
        cace_margin = float(cm_cfg.get("atom_margin", 0.05))
        pos_loss = F.softplus(-(crit_pred_gain - cace_margin))
        neg_loss = F.softplus(crit_pred_gain - cace_margin)
        neg_w = float(cm_cfg.get("negative_weight", 0.15))
        cace_w = torch.where(crit_target, torch.ones_like(crit_pred_gain), torch.full_like(crit_pred_gain, neg_w))
        cace_mask = critical_atom_mask.bool() & torch.isfinite(crit_pred_gain)
        L_cace_pair = _weighted_mean(torch.where(crit_target, pos_loss, neg_loss), cace_w, cace_mask)

        # Proposal/listwise target: make the Top-M proposal bank retain these
        # atoms before the selector budget split happens.
        prop_logits = out["proposal_logits"]
        bce_c = F.binary_cross_entropy_with_logits(prop_logits, crit_target.float(), reduction="none")
        pos_w_c = float(cm_cfg.get("proposal_positive_weight", 10.0))
        prop_w_c = torch.where(crit_target, torch.full_like(bce_c, pos_w_c), torch.ones_like(bce_c) * neg_w)
        L_cace_prop_bce = _weighted_mean(bce_c, prop_w_c, e_mask & torch.isfinite(bce_c))
        dist = crit_gain / crit_gain.sum(dim=1, keepdim=True).clamp_min(1e-6)
        has_gain = (crit_gain.sum(dim=1) > 1e-6).float()
        logp_c = F.log_softmax(prop_logits.masked_fill(~e_mask, _neg_mask_value(prop_logits)), dim=1)
        L_cace_prop_rank = -((dist * logp_c).sum(dim=1) * has_gain).sum() / has_gain.sum().clamp_min(1.0)
        L_cace_prop = L_cace_prop_bce + float(cm_cfg.get("proposal_rank_weight", 1.0)) * L_cace_prop_rank
    else:
        L_cace_pair = J0.new_tensor(0.0)
        L_cace_prop = J0.new_tensor(0.0)

    # Family-listwise loss for the HAB family gate.
    fam_gain = batch.get("family_target_gain")
    family_logits = out.get("family_logits")
    if fam_gain is not None and family_logits is not None:
        fam_active = batch.get("family_target_active", out.get("family_active", torch.ones_like(family_logits).bool())).bool()
        target = fam_gain.float().masked_fill(~fam_active, 0.0)
        mass = target.sum(dim=1, keepdim=True)
        uniform = fam_active.float() / fam_active.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        target_dist = torch.where(mass > 1e-6, target / mass.clamp_min(1e-6), uniform)
        logp = F.log_softmax(family_logits.masked_fill(~fam_active, _neg_mask_value(family_logits)), dim=1)
        L_fam = -(target_dist * logp).sum(dim=1).mean()
    else:
        L_fam = J0.new_tensor(0.0)

    tau_q = float(cfg.get("tournament", {}).get("softmin_tau", 1.0))
    eps_cal = float(cfg.get("tournament", {}).get("epsilon_cal", cfg.get("calibration", {}).get("epsilon_cal", 0.0)))
    beta_unc = float(cfg.get("tournament", {}).get("beta_uncertainty", 0.0))
    cur_epoch = int(train_cfg.get("current_epoch", 0))
    action_loss_start = int(train_cfg.get("action_loss_start_epoch", 4))
    predicted_selector_start = int(train_cfg.get("predicted_selector_start_epoch", 8))
    pair_act_weight_cfg = float(train_cfg.get("pair_action_loss_weight", 1.0))
    action_act_weight_cfg = float(train_cfg.get("action_conditioned_action_loss_weight", 0.0))
    enable_action_loss = (cur_epoch >= action_loss_start) and (float(lw.get("action", 1.0)) > 0.0)
    logits_action = None
    logits_pair = None
    pair_logits_entries: list[tuple[float, float, torch.Tensor]] = []
    deployment_mask_entries: list[tuple[float, torch.Tensor, torch.Tensor]] = []
    selector_exact_fraction = J0.new_tensor(0.0)
    selector_surrogate_exact_agreement = J0.new_tensor(0.0)
    selector_fast_wall_time_s = J0.new_tensor(0.0)
    selector_exact_wall_time_s = J0.new_tensor(0.0)

    if enable_action_loss and action_act_weight_cfg > 0.0:
        selected_mask, query_mask = _predicted_certificate_masks(out, batch, cfg)
        g_runtime = g * query_mask.float()
        budgeted_cost = finite_J0 + (g_runtime * selected_mask[:, :, None].float()).sum(dim=1)
        if g_var is not None:
            selected_var = (g_var * query_mask.float() * selected_mask[:, :, None].float()).sum(dim=1).clamp_min(0.0)
            sigma = torch.sqrt(selected_var[:, :, None] + selected_var[:, None, :] + 1e-12)
        else:
            sigma = None
        logits_action = _budgeted_tournament_scores(budgeted_cost, valid, tau_q, eps_cal, sigma=sigma, beta_uncertainty=beta_unc)

    use_pair_act = bool(cfg.get("runtime", {}).get("use_pair_conditioned_margins", cfg.get("model", {}).get("pair_conditioned", True)))
    if enable_action_loss and pair_act_weight_cfg > 0.0 and use_pair_act and "pair_atom_delta" in out:
        exclude_safety_from_pair_action = bool(train_cfg.get("exclude_safety_atoms_from_pair_action_loss", True))
        pair_action_atom_mask = e_mask & ((~safety_atom_mask) if exclude_safety_from_pair_action else torch.ones_like(safety_atom_mask, dtype=torch.bool))
        if cur_epoch < predicted_selector_start and target_sel is not None:
            # Oracle-to-predicted curriculum: early action supervision can either
            # exclude structural safety atoms (legacy behavior) or include clipped
            # hard/safety margins for metric-aligned deployment training.
            pair_selected_mask = target_sel.bool() & pair_action_atom_mask
            logits_pair = _pair_conditioned_tournament_scores(
                finite_J0,
                pred_atom_delta,
                pairs,
                pair_valid,
                pair_selected_mask,
                valid,
                tau_q,
                eps_cal,
                pair_var=out.get("pair_atom_var"),
                beta_uncertainty=beta_unc,
                pair_scale=pair_scale,
                normalize_margins=bool(cfg.get("model", {}).get("pair_margin_normalized", True)),
            )
            pair_logits_entries = [(float(cfg.get("evidence", {}).get("budget", 16)), 1.0, logits_pair)]
        else:
            configured_budgets = train_cfg.get("deployment_budgets", None)
            if configured_budgets is None:
                configured_budgets = [float(cfg.get("evidence", {}).get("budget", 16))]
            budgets = [float(x) for x in configured_budgets]
            configured_weights = train_cfg.get("deployment_budget_weights", None)
            if configured_weights is None:
                budget_weights = [1.0] * len(budgets)
            else:
                budget_weights = [float(x) for x in configured_weights]
                if len(budget_weights) != len(budgets):
                    raise ValueError("training.deployment_budget_weights must match training.deployment_budgets")
            budget_entries = _deployment_budget_entries_for_step(budgets, budget_weights, train_cfg)
            primary_budget = float(cfg.get("evidence", {}).get("budget", 16))
            primary_distance = float("inf")
            selector_backend = str(train_cfg.get("deployment_selector_backend", "exact_cpu")).strip().lower()
            fast_backends = {"hybrid_fast", "gpu_surrogate", "fast_gpu", "surrogate_plus_exact"}
            budget_values = [float(value) for value, _ in budget_entries]
            fast_masks: dict[float, torch.Tensor] = {}
            if selector_backend in fast_backends:
                _fast_started = time.perf_counter()
                fast_masks = _fast_pair_margin_surrogate_masks(out, batch, cfg, budget_values)
                selector_fast_wall_time_s = J0.new_tensor(time.perf_counter() - _fast_started)

                # Exact masks are used only as sparse proposal-distillation targets.
                # This is the only CPU/NumPy selector call in the fast pathway.
                exact_scene_indices = _deployment_exact_distill_scene_indices(
                    int(J0.shape[0]), train_cfg, J0.device
                )
                exact_scene_mask = torch.zeros((int(J0.shape[0]),), dtype=torch.bool, device=J0.device)
                if exact_scene_indices.numel() > 0:
                    exact_scene_mask.index_fill_(0, exact_scene_indices, True)
                    exact_cfg = _config_with_evidence_budget(cfg, primary_budget)
                    train_swap_passes = train_cfg.get("deployment_selector_swap_passes", None)
                    if train_swap_passes is not None:
                        exact_cfg = dict(exact_cfg)
                        local_selector = dict(exact_cfg.get("selector", {}))
                        local_selector["margin_coreset_swap_passes"] = max(0, int(train_swap_passes))
                        exact_cfg["selector"] = local_selector
                    _exact_started = time.perf_counter()
                    exact_small = _predicted_pair_certificate_masks(
                        out, batch, exact_cfg, scene_indices=exact_scene_indices
                    )
                    selector_exact_wall_time_s = J0.new_tensor(time.perf_counter() - _exact_started)
                    exact_small = exact_small & pair_action_atom_mask.index_select(0, exact_scene_indices)
                    exact_full = torch.zeros_like(pair_action_atom_mask)
                    exact_full.index_copy_(0, exact_scene_indices, exact_small)
                    deployment_mask_entries.append((1.0, exact_full.detach(), exact_scene_mask))
                    surrogate_small = fast_masks[min(fast_masks, key=lambda value: abs(value - primary_budget))].index_select(0, exact_scene_indices)
                    union = (surrogate_small | exact_small).float().sum(dim=1)
                    inter = (surrogate_small & exact_small).float().sum(dim=1)
                    selector_surrogate_exact_agreement = torch.where(
                        union > 0.0, inter / union.clamp_min(1.0), torch.ones_like(union)
                    ).mean()
                selector_exact_fraction = J0.new_tensor(
                    float(exact_scene_indices.numel()) / max(float(J0.shape[0]), 1.0)
                )

            prepared_budget_entries: list[tuple[float, float, dict[str, Any]]] = []
            for budget_value, budget_weight in budget_entries:
                budget_cfg = _config_with_evidence_budget(cfg, budget_value)
                train_swap_passes = train_cfg.get("deployment_selector_swap_passes", None)
                if train_swap_passes is not None:
                    budget_cfg = dict(budget_cfg)
                    local_selector = dict(budget_cfg.get("selector", {}))
                    local_selector["margin_coreset_swap_passes"] = max(0, int(train_swap_passes))
                    budget_cfg["selector"] = local_selector
                prepared_budget_entries.append((float(budget_value), float(budget_weight), budget_cfg))

            exact_scene_indices = torch.empty((0,), dtype=torch.long, device=J0.device)
            exact_scene_mask = torch.zeros((int(J0.shape[0]),), dtype=torch.bool, device=J0.device)
            exact_budget_masks: list[torch.Tensor] = []
            if selector_backend not in fast_backends and prepared_budget_entries:
                exact_scene_indices = _deployment_selector_scene_indices(
                    int(J0.shape[0]), train_cfg, J0.device
                )
                if exact_scene_indices.numel() > 0:
                    exact_scene_mask.index_fill_(0, exact_scene_indices, True)
                    _exact_started = time.perf_counter()
                    exact_budget_masks = _predicted_pair_certificate_masks_multi_budget(
                        out,
                        batch,
                        [entry[2] for entry in prepared_budget_entries],
                        scene_indices=(
                            None
                            if exact_scene_indices.numel() == int(J0.shape[0])
                            else exact_scene_indices
                        ),
                    )
                    selector_exact_wall_time_s = J0.new_tensor(time.perf_counter() - _exact_started)
                selector_exact_fraction = J0.new_tensor(
                    float(exact_scene_indices.numel()) / max(float(J0.shape[0]), 1.0)
                )

            for budget_index, (budget_value, budget_weight, budget_cfg) in enumerate(prepared_budget_entries):
                if selector_backend in fast_backends:
                    mask_key = min(fast_masks, key=lambda value: abs(value - float(budget_value)))
                    pair_selected_mask = fast_masks[mask_key] & pair_action_atom_mask
                else:
                    if exact_scene_indices.numel() == int(J0.shape[0]):
                        pair_selected_mask = exact_budget_masks[budget_index] & pair_action_atom_mask
                    else:
                        if target_sel is not None:
                            pair_selected_mask = target_sel.bool() & pair_action_atom_mask
                        else:
                            pair_selected_mask = torch.zeros_like(pair_action_atom_mask)
                        if exact_scene_indices.numel() > 0:
                            exact_mask = exact_budget_masks[budget_index]
                            exact_mask = exact_mask & pair_action_atom_mask.index_select(0, exact_scene_indices)
                            pair_selected_mask = pair_selected_mask.clone()
                            pair_selected_mask.index_copy_(0, exact_scene_indices, exact_mask)
                    deployment_mask_entries.append(
                        (max(float(budget_weight), 0.0), pair_selected_mask.detach(), exact_scene_mask)
                    )

                budget_logits = _pair_conditioned_tournament_scores(
                    finite_J0,
                    pred_atom_delta,
                    pairs,
                    pair_valid,
                    pair_selected_mask,
                    valid,
                    tau_q,
                    eps_cal,
                    pair_var=out.get("pair_atom_var"),
                    beta_uncertainty=beta_unc,
                    pair_scale=pair_scale,
                    normalize_margins=bool(cfg.get("model", {}).get("pair_margin_normalized", True)),
                )
                pair_logits_entries.append((budget_value, max(float(budget_weight), 0.0), budget_logits))
                distance = abs(float(budget_value) - primary_budget)
                if distance < primary_distance:
                    logits_pair = budget_logits
                    primary_distance = distance

    # Stop-gradient deployment-selection distillation.  The CPU margin-coreset
    # selector is discrete, so action loss alone cannot teach proposal logits
    # which atoms survive the fixed budget.  Distilling the exact runtime masks
    # into the proposal head closes that training/deployment gap without changing
    # the deployed selector or the paper's fixed-budget semantics.
    if deployment_mask_entries:
        target_mass = torch.zeros_like(out["proposal_logits"])
        weight_mass = torch.zeros((int(J0.shape[0]), 1), dtype=target_mass.dtype, device=target_mass.device)
        for mask_weight, selected_mask, exact_scene_mask in deployment_mask_entries:
            if mask_weight <= 0.0:
                continue
            scene_weight = exact_scene_mask[:, None].float() * float(mask_weight)
            target_mass = target_mass + selected_mask.float() * scene_weight
            weight_mass = weight_mass + scene_weight
        deploy_target = target_mass / weight_mass.clamp_min(1e-6)
        deploy_mask = e_mask & (weight_mass > 0.0)
        deploy_bce = F.binary_cross_entropy_with_logits(
            out["proposal_logits"], deploy_target, reduction="none"
        )
        deploy_pos_weight = float(train_cfg.get("deployment_selection_positive_weight", 4.0))
        deploy_weights = 1.0 + (deploy_pos_weight - 1.0) * deploy_target
        L_deploy_select = _weighted_mean(deploy_bce, deploy_weights, deploy_mask)
    else:
        L_deploy_select = J0.new_tensor(0.0)

    L_cert_gap, L_cert_safety, L_cert_frontier = _certificate_action_gap_loss(logits_pair, target_action, valid, hard_action, cfg)

    full_pred_cost = finite_J0 + (g * e_mask[:, :, None].float()).sum(dim=1)
    full_pred_cost = full_pred_cost.masked_fill(~valid, J0.new_tensor(1e6))
    full_logits = _negative_cost_logits(full_pred_cost, valid, min_scale=float(train_cfg.get("full_action_min_scale", 1.0)))
    hard_action_targets = bool(train_cfg.get("hard_action_targets", True))
    soft_teacher_target = None
    if "teacher_J_T" in batch and not hard_action_targets:
        tau_T = float(train_cfg.get("teacher_soft_target_tau", 1.0))
        teacher_cost = batch["teacher_J_T"].float().masked_fill(~valid, J0.new_tensor(1e9))
        teacher_logits = _negative_cost_logits(teacher_cost, valid, min_scale=float(train_cfg.get("teacher_action_min_scale", 1.0))) / max(tau_T, 1e-6)
        soft_teacher_target = torch.softmax(teacher_logits, dim=1)
        L_full_action = -(soft_teacher_target * F.log_softmax(full_logits, dim=1)).sum(dim=1).mean()
    else:
        L_full_action = F.cross_entropy(full_logits, target_action)

    teacher_cost_for_weight = batch.get("teacher_J_T")
    regret_strength = float(train_cfg.get("deployment_regret_weight", 0.0)) if cur_epoch >= predicted_selector_start else 0.0
    regret_clip = float(train_cfg.get("deployment_regret_clip", 4.0))
    regret_min_scale = float(train_cfg.get("deployment_regret_min_scale", train_cfg.get("teacher_action_min_scale", 1.0)))
    if logits_action is not None:
        action_scene_weights = _teacher_regret_weights(
            logits_action,
            target_action,
            valid,
            teacher_cost_for_weight,
            strength=regret_strength,
            clip=regret_clip,
            min_scale=regret_min_scale,
        )
        L_act_action = _weighted_action_target_loss(
            logits_action,
            target_action,
            soft_target=soft_teacher_target,
            scene_weights=action_scene_weights,
        )
    else:
        L_act_action = J0.new_tensor(0.0)

    if pair_logits_entries:
        pair_losses = []
        pair_weights_for_average = []
        for _, budget_weight, budget_logits in pair_logits_entries:
            if budget_weight <= 0.0:
                continue
            scene_weights = _teacher_regret_weights(
                budget_logits,
                target_action,
                valid,
                teacher_cost_for_weight,
                strength=regret_strength,
                clip=regret_clip,
                min_scale=regret_min_scale,
            )
            pair_losses.append(
                _weighted_action_target_loss(
                    budget_logits,
                    target_action,
                    soft_target=soft_teacher_target,
                    scene_weights=scene_weights,
                )
            )
            pair_weights_for_average.append(float(budget_weight))
        if pair_losses:
            denom = max(sum(pair_weights_for_average), 1e-6)
            L_act_pair = sum(w * loss for w, loss in zip(pair_weights_for_average, pair_losses)) / denom
        else:
            L_act_pair = J0.new_tensor(0.0)
    else:
        L_act_pair = J0.new_tensor(0.0)

    # Dense full-interface margin distillation.  Use a per-scene scale so this
    # term optimizes ordering rather than absolute cost units.
    Bsz, Ksz = full_pred_cost.shape
    tgt = target_action.clamp_min(0).clamp_max(Ksz - 1)
    c_star = torch.gather(full_pred_cost, 1, tgt[:, None]).expand(Bsz, Ksz)
    margin_mask = valid.clone()
    margin_mask.scatter_(1, tgt[:, None], False)
    if "teacher_J_T" in batch:
        full_scale = _valid_row_scale(batch["teacher_J_T"].float().masked_fill(~valid, 0.0), valid, min_scale=float(train_cfg.get("full_margin_min_scale", 100.0)))
    else:
        full_scale = _valid_row_scale(full_pred_cost, valid, min_scale=float(train_cfg.get("full_margin_min_scale", 100.0)))
    L_full_margin_terms = F.softplus((c_star - full_pred_cost) / (full_scale * max(float(train_cfg.get("full_margin_tau", 1.0)), 1e-6)))
    L_full_margin = _masked_mean(L_full_margin_terms, margin_mask)

    hard_mask = batch.get("teacher_hard_violation")
    if hard_mask is not None:
        hard_mask = hard_mask.bool() & valid
        safe_mask = (~hard_mask) & valid
        safe_cost = full_pred_cost.masked_fill(~safe_mask, J0.new_tensor(1e6)).min(dim=1, keepdim=True).values
        hard_cost = full_pred_cost.masked_fill(~hard_mask, J0.new_tensor(1e6))
        feasible_pair = safe_mask.any(dim=1, keepdim=True) & hard_mask
        hard_scale = _valid_row_scale(batch["teacher_J_T"].float().masked_fill(~valid, 0.0), valid, min_scale=float(train_cfg.get("hard_feasibility_min_scale", 100.0))) if "teacher_J_T" in batch else _valid_row_scale(full_pred_cost, valid, min_scale=float(train_cfg.get("hard_feasibility_min_scale", 100.0)))
        L_hard_feas = F.softplus((safe_cost - hard_cost + float(train_cfg.get("hard_feasibility_margin", 10.0))) / (hard_scale * max(float(train_cfg.get("hard_feasibility_tau", 1.0)), 1e-6)))
        L_hard_feas = _masked_mean(L_hard_feas, feasible_pair)
    else:
        L_hard_feas = J0.new_tensor(0.0)
    pair_act_weight = pair_act_weight_cfg if logits_pair is not None else 0.0
    action_act_weight = action_act_weight_cfg if logits_action is not None else 0.0
    norm_act = max(pair_act_weight + action_act_weight, 1e-6)
    L_act = (pair_act_weight * L_act_pair + action_act_weight * L_act_action) / norm_act if enable_action_loss else J0.new_tensor(0.0)

    # Optional post-hoc-style calibration surrogate: penalize pair margin residuals
    # above the configured epsilon_cal so the validation quantile has a training signal.
    if eps_cal > 0:
        target_margin_for_cal = batch["pair_margins"].float() / pair_scale.clamp_min(1e-6) if normalize_pair_losses else batch["pair_margins"].float()
        if margin_target_clip > 0:
            target_margin_for_cal = target_margin_for_cal.clamp(-margin_target_clip, margin_target_clip)
        cal_err = (M_hat_E - target_margin_for_cal).abs()
        if bool(train_cfg.get("normalize_calibration_loss", True)):
            eps_train = float(train_cfg.get("epsilon_cal_normalized", 1.0))
        else:
            eps_train = eps_cal
        L_cal = _masked_mean(F.relu(cal_err - eps_train), pair_mask)
    else:
        L_cal = J0.new_tensor(0.0)

    total = (
        float(lw.get("base", 1.0)) * L_base
        + float(lw.get("pair", 1.0)) * L_pair
        + float(lw.get("residual", 1.0)) * L_res
        + float(lw.get("atom_cost", 0.25)) * L_atom
        + float(lw.get("uncertainty", 0.1)) * L_unc
        + float(lw.get("full_interface_rank_aux", lw.get("rank", 0.1))) * L_rank
        + float(lw.get("family", 0.5)) * L_fam
        + float(lw.get("proposal", lw.get("selection", 1.0))) * L_prop
        + float(lw.get("action", 1.0)) * L_act
        + float(lw.get("full_action", 1.0)) * L_full_action
        + float(lw.get("full_margin", 0.5)) * L_full_margin
        + float(lw.get("hard_feasibility", 0.5)) * L_hard_feas
        + float(lw.get("calibration", 0.0)) * L_cal
        + float(lw.get("critical_pair", 0.0)) * L_cace_pair
        + float(lw.get("critical_proposal", 0.0)) * L_cace_prop
        + float(lw.get("deployment_selection", 0.0)) * L_deploy_select
        + float(lw.get("certificate_gap", 0.0)) * L_cert_gap
        + float(lw.get("certificate_safety", 0.0)) * L_cert_safety
        + float(lw.get("certificate_safe_frontier", 0.0)) * L_cert_frontier
        + float(lw.get("online_hard_rival", 0.0)) * L_online_hard_rival
        + float(lw.get("cycle_consistency", 0.0)) * L_cycle
        + float(lw.get("counterfactual_critical_pair", 0.0)) * L_cf_critical_pair
        + float(lw.get("counterfactual_critical_proposal", 0.0)) * L_cf_critical_proposal
    )
    return {
        "loss": total,
        "L_base": L_base,
        "L_pair": L_pair,
        "L_res": L_res,
        "L_atom": L_atom,
        "L_unc": L_unc,
        "L_rank": L_rank,
        "L_fam": L_fam,
        "L_prop": L_prop,
        "L_sel": L_prop,
        "L_act": L_act,
        "L_act_pair": L_act_pair if 'L_act_pair' in locals() else J0.new_tensor(0.0),
        "L_act_action": L_act_action if 'L_act_action' in locals() else J0.new_tensor(0.0),
        "L_full_action": L_full_action if 'L_full_action' in locals() else J0.new_tensor(0.0),
        "L_full_margin": L_full_margin if 'L_full_margin' in locals() else J0.new_tensor(0.0),
        "L_hard_feas": L_hard_feas if 'L_hard_feas' in locals() else J0.new_tensor(0.0),
        "L_cal": L_cal,
        "L_cace_pair": L_cace_pair if 'L_cace_pair' in locals() else J0.new_tensor(0.0),
        "L_cace_prop": L_cace_prop if 'L_cace_prop' in locals() else J0.new_tensor(0.0),
        "L_deploy_select": L_deploy_select if 'L_deploy_select' in locals() else J0.new_tensor(0.0),
        "L_cert_gap": L_cert_gap if 'L_cert_gap' in locals() else J0.new_tensor(0.0),
        "L_cert_safety": L_cert_safety if 'L_cert_safety' in locals() else J0.new_tensor(0.0),
        "L_cert_frontier": L_cert_frontier if 'L_cert_frontier' in locals() else J0.new_tensor(0.0),
        "L_online_hard_rival": L_online_hard_rival,
        "L_cycle": L_cycle,
        "L_cf_critical_pair": L_cf_critical_pair,
        "L_cf_critical_proposal": L_cf_critical_proposal,
        "selector_exact_fraction": selector_exact_fraction,
        "selector_surrogate_exact_agreement": selector_surrogate_exact_agreement,
        "selector_fast_wall_time_s": selector_fast_wall_time_s,
        "selector_exact_wall_time_s": selector_exact_wall_time_s,
    }
