from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from bdse.config import load_config
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, PreprocessedBDSEDataset
from bdse.metrics.bdse_metrics import OnlineMetricMean, compute_bdse_diagnostics
from bdse.planner.nuplan_planner import BDSEPlannerCore, runtime_query_diagnostics
from bdse.planner.fallback import runtime_safety_flags_from_runtime
from bdse.planner.tournament import full_interface_action, run_pair_conditioned_tournament
from bdse.utils import configure_torch_for_device, resolve_torch_device
from bdse.external_baselines.model_factory import load_model_for_config


def load_model(checkpoint: str | None, cfg, device: torch.device):
    # Supports both native BDSE checkpoints and budget-compatible external
    # baseline adapters.  PDM-Closed is rule-based and may be evaluated without
    # a checkpoint; trainable external baselines and BDSE require one.
    return load_model_for_config(checkpoint, cfg, device)


def _align_bool_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Return a boolean mask with exactly ``size`` entries.

    Cached samples store only the atoms/actions that actually exist in the scene,
    while dense model outputs use the configured padded dimensions.  Slicing a
    shorter mask does not extend it and therefore cannot be broadcast against a
    padded tensor.  Explicit zero-padding preserves the intended semantics:
    nonexistent padded entries are inactive/invalid.
    """

    if size < 0:
        raise ValueError(f"mask size must be non-negative, got {size}")
    source = np.asarray(mask, dtype=bool).reshape(-1)
    aligned = np.zeros((size,), dtype=bool)
    count = min(size, source.shape[0])
    if count:
        aligned[:count] = source[:count]
    return aligned


def _validate_dense_prediction(
    dense_j0: np.ndarray,
    dense_g: np.ndarray,
    active_atoms: np.ndarray,
    valid_actions: np.ndarray,
    *,
    scenario_token: str,
) -> None:
    """Fail early with a useful message when a dense diagnostic is malformed."""

    if dense_j0.ndim != 1:
        raise ValueError(
            f"dense J0 must have shape [K], got {dense_j0.shape} for scenario={scenario_token!r}"
        )
    if dense_g.ndim != 2 or dense_g.shape[1] != dense_j0.shape[0]:
        raise ValueError(
            "dense g must have shape [E,K] with the same K as J0; "
            f"got J0={dense_j0.shape}, g={dense_g.shape} for scenario={scenario_token!r}"
        )
    if active_atoms.shape != (dense_g.shape[0],) or valid_actions.shape != (dense_g.shape[1],):
        raise ValueError(
            "aligned diagnostic masks have inconsistent shapes: "
            f"active={active_atoms.shape}, valid={valid_actions.shape}, g={dense_g.shape}"
        )
    if valid_actions.any() and not np.isfinite(dense_j0[valid_actions]).all():
        raise FloatingPointError(f"non-finite dense J0 on a valid action for scenario={scenario_token!r}")
    if active_atoms.any() and valid_actions.any():
        active_valid_g = dense_g[np.ix_(active_atoms, valid_actions)]
        if not np.isfinite(active_valid_g).all():
            raise FloatingPointError(
                f"non-finite dense atom/action contribution for scenario={scenario_token!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, nargs="+", default=["val"])
    parser.add_argument("--preprocessed-dir", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/open_loop_bdse_metrics.json")
    parser.add_argument("--per-sample-output", type=str, default=None, help="Optional JSONL with one diagnostic row per sample for failure slicing.")
    parser.add_argument("--disable-dense-diagnostic", action="store_true", help="Skip diagnostic-only dense full-interface scoring.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Evaluation device. Defaults to auto, which uses CUDA when available and otherwise CPU.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_torch_device(args.device, context="Open-loop evaluation")
    configure_torch_for_device(device)
    model = load_model(args.checkpoint, cfg, device)
    print(f"Open-loop evaluation device: {device}")
    core = BDSEPlannerCore(model=model, cfg=cfg)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if args.preprocessed_dir:
        dataset = PreprocessedBDSEDataset(args.preprocessed_dir, split=args.split, max_scenarios=args.max_scenarios)
    else:
        if len(args.split) != 1:
            raise ValueError("On-the-fly open-loop evaluation supports one split; use --preprocessed-dir for multiple split folders.")
        dataset = NuPlanBDSEDataset(cfg, split=args.split[0], max_files=args.max_files, max_scenarios=args.max_scenarios, use_devkit=True)
    metric_means = OnlineMetricMean()
    per_sample_file = None
    if args.per_sample_output:
        per_sample_path = Path(args.per_sample_output)
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        per_sample_file = per_sample_path.open("w", encoding="utf-8", buffering=1024 * 1024)
    planner_latencies_ms: list[float] = []
    for sample in tqdm(dataset.iter_samples(), total=len(dataset)):
        # Keep the sparse certificate stage and the optional dense diagnostic in
        # one model cache scope.  The dense path can then reuse the identical scene,
        # action and evidence encodings without changing any planner output.
        prediction_scope_factory = getattr(model, "runtime_prediction_cache_scope", None)
        prediction_scope = prediction_scope_factory() if callable(prediction_scope_factory) else nullcontext()
        with prediction_scope:
            planner_start = time.perf_counter()
            pred, sel, tour, stage_atom_active = core._run_certificate_stage(
                sample.runtime, sample.candidates, sample.evidence_bank, cfg
            )
            # Native/external model adapters return NumPy arrays, so their CUDA-to-
            # CPU transfers already synchronize the kernels that determine the
            # planner result.  Extra full-device synchronizations only serialize
            # concurrent paper-grade workers and do not improve this wall-clock
            # measurement.
            planner_latency_ms = 1000.0 * (time.perf_counter() - planner_start)
            dense = None
            if not args.disable_dense_diagnostic and hasattr(model, "predict_dense_numpy"):
                dense = model.predict_dense_numpy(
                    sample.runtime, sample.candidates, sample.evidence_bank, cfg
                )
        planner_latencies_ms.append(float(planner_latency_ms))
        qdiag = runtime_query_diagnostics(pred, sel.selected)
        qdiag["planner_latency_ms"] = float(planner_latency_ms)
        qdiag["configured_decision_budget_atom_count"] = float(
            max(1, int((cfg.get("evidence", {}) or {}).get("budget", 1)))
        )
        tour_diag = getattr(tour, "diagnostics", {}) or {}
        qdiag.update({k: v for k, v in tour_diag.items() if k in {"normalized_margins", "margin_scale", "epsilon_cal", "pair_conditioned", "selected_action_safety_flag", "avoidable_selected_action_safety_flag", "all_actions_safety_flagged", "all_flagged_risk_guard_applied", "all_flagged_hard_risk_regret", "hard_filter_applied", "safe_action_available"}})
        for key, value in tour_diag.items():
            if (
                key.startswith("pair_potential_")
                or key.startswith("pair_action_anchor_")
                or key.startswith("evidence_certificate_")
                or key.startswith("residual_flip_")
                or key.startswith("dual_certificate_")
                or key.startswith("set_conditioned_residual_")
            ):
                if isinstance(value, (bool, np.bool_)):
                    qdiag[key] = float(bool(value))
                elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                    qdiag[key] = float(value)
        qdiag["fallback_would_trigger"] = bool(core._needs_fallback(tour, sample.candidates, cfg))
        sel_diag = getattr(sel, "diagnostics", {}) or {}
        mode = str(sel_diag.get("mode", ""))
        qdiag["selector_action_rank_active"] = float(mode.startswith("runtime_pair_conditioned_action_rank") or mode == "runtime_pair_conditioned_hybrid_lcb_action_rank")
        qdiag["selector_margin_coreset_active"] = float(mode == "runtime_pair_conditioned_margin_coreset")
        qdiag["selector_anytime_adverse_certificate_active"] = float(mode == "runtime_pair_conditioned_anytime_adverse_certificate")
        qdiag["selector_deployment_coreset_active"] = float(mode == "runtime_pair_conditioned_deployment_coreset")
        qdiag["selector_hybrid_lcb_action_rank_active"] = float(mode == "runtime_pair_conditioned_hybrid_lcb_action_rank")
        qdiag["selector_flip_rank_active"] = float(mode == "runtime_pair_conditioned_flip_rank")
        qdiag["selector_lcb_active"] = float(mode == "runtime_pair_conditioned_lcb_uncertainty" or mode == "runtime_pair_conditioned_hybrid_lcb_action_rank")
        for k, v in sel_diag.items():
            if isinstance(v, (bool, np.bool_)):
                qdiag[f"selector_{k}"] = float(bool(v))
            elif isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)):
                qdiag[f"selector_{k}"] = float(v)
        qdiag["top_m_atoms"] = list(map(int, np.asarray(pred.get("top_m_atoms", []), dtype=np.int64).reshape(-1).tolist()))
        # Pair-conditioned full-support diagnostic using the exact same final
        # tournament as deployment.  The legacy full_interface_action_match is
        # computed from the action-conditioned g head and is not the ceiling of
        # the pair-conditioned selector/tournament path.
        pair_full_action = -1
        local_pair_full_action = -1
        if "pair_atom_delta" in pred and "pair_indices" in pred:
            full_atoms = np.flatnonzero(np.asarray(stage_atom_active, dtype=bool)).astype(np.int64).tolist()
            sel_cfg = cfg.get("selector", {})
            if bool(sel_cfg.get("decision_budget_excludes_structural_safety", False)):
                structural = np.asarray(pred.get("mandatory_atom_mask", np.zeros_like(stage_atom_active)), dtype=bool).reshape(-1)
                full_atoms = [i for i in full_atoms if i >= structural.shape[0] or not bool(structural[i])]
            runtime_flags = runtime_safety_flags_from_runtime(sample.runtime, sample.candidates, cfg)
            pair_full_tour = run_pair_conditioned_tournament(
                pred["J0"],
                pred.get("rival_pair_atom_delta", pred["pair_atom_delta"]),
                pred.get("rival_pair_indices", pred["pair_indices"]),
                full_atoms,
                sample.candidates.valid_mask,
                runtime_flags,
                {**cfg, "runtime_pair_margin_scale": float(pred.get("rival_pair_margin_scale", pred.get("pair_margin_scale", 100.0)))},
                pair_atom_variance=pred.get("rival_pair_atom_var", pred.get("pair_atom_var", None)),
                candidate_trajectories=sample.candidates.trajectories,
                maneuver_ids=sample.candidates.maneuver_ids,
                predicted_atom_costs=pred["g"],
                residual_action_potential=pred.get("residual_action_potential", None),
                residual_action_variance=pred.get("residual_action_var", None),
                residual_set_atom_factors=pred.get("residual_set_atom_factors", None),
                residual_set_action_factors=pred.get("residual_set_action_factors", None),
                evidence_certificate_fraction=1.0,
            )
            pair_full_tour = core._apply_all_flagged_structural_guard(
                pair_full_tour, sample.runtime, sample.candidates, runtime_flags, cfg
            )
            pair_full_tour = core._finalize_pair_anchor_after_structural_guard(
                pair_full_tour, sample.runtime, sample.candidates, runtime_flags, cfg
            )
            pair_full_action = int(pair_full_tour.action_index)

            # Local-only pair-full ceiling, using the exact deployed pair graph
            # and tournament but no learned residual intervention.
            rival_pairs_np = np.asarray(pred.get("rival_pair_indices", pred["pair_indices"]), dtype=np.int64).reshape(-1, 2)
            local_scale = max(float(pred.get("rival_pair_margin_scale", pred.get("pair_margin_scale", 100.0))), 1e-6)
            g_sparse_np = np.asarray(pred["g"], dtype=np.float32)
            if rival_pairs_np.size:
                local_pair_delta = (g_sparse_np[:, rival_pairs_np[:, 1]] - g_sparse_np[:, rival_pairs_np[:, 0]])
                if bool(pred.get("pair_margin_normalized", True)):
                    local_pair_delta = local_pair_delta / local_scale
                local_pair_full_tour = run_pair_conditioned_tournament(
                    pred["J0"], local_pair_delta, rival_pairs_np, full_atoms,
                    sample.candidates.valid_mask, runtime_flags,
                    {**cfg, "runtime_pair_margin_scale": local_scale},
                    pair_atom_variance=None,
                    candidate_trajectories=sample.candidates.trajectories,
                    maneuver_ids=sample.candidates.maneuver_ids,
                    predicted_atom_costs=pred["g"],
                    residual_action_potential=None,
                    residual_action_variance=None,
                )
                local_pair_full_tour = core._apply_all_flagged_structural_guard(
                    local_pair_full_tour, sample.runtime, sample.candidates, runtime_flags, cfg
                )
                local_pair_full_action = int(local_pair_full_tour.action_index)

        diag = compute_bdse_diagnostics(
            sample.candidates,
            sample.evidence_bank,
            sample.teacher,
            sample.pairs,
            pred["J0"],
            pred["g"],
            sel.selected,
            tour.action_index,
            cfg=cfg,
            inference_pairs=pred.get("rival_pair_indices", sel.pair_indices),
            query_diagnostics=qdiag,
            dense_predicted_base=None if dense is None else dense["J0"],
            dense_predicted_atom_costs=None if dense is None else dense["g"],
            certificate_margin_matrix=tour.margins,
        )
        if dense is not None:
            dense_g = np.asarray(dense["g"], dtype=np.float32)
            dense_j0 = np.asarray(dense["J0"], dtype=np.float32).reshape(-1)
            valid_actions = _align_bool_mask(sample.candidates.valid_mask, dense_j0.shape[0])
            active_atoms = _align_bool_mask(sample.evidence_bank.active_mask, dense_g.shape[0])
            _validate_dense_prediction(
                dense_j0,
                dense_g,
                active_atoms,
                valid_actions,
                scenario_token=str(getattr(sample, "scenario_token", "")),
            )
            topm_atoms = np.asarray(pred.get("top_m_atoms", []), dtype=np.int64).reshape(-1)
            topm_atoms = topm_atoms[(topm_atoms >= 0) & (topm_atoms < dense_g.shape[0])]
            selected_atoms = np.asarray(sel.selected, dtype=np.int64).reshape(-1)
            selected_atoms = selected_atoms[(selected_atoms >= 0) & (selected_atoms < dense_g.shape[0])]
            dense_topm_g = np.zeros_like(dense_g)
            dense_selected_g = np.zeros_like(dense_g)
            if topm_atoms.size:
                dense_topm_g[topm_atoms] = dense_g[topm_atoms]
            if selected_atoms.size:
                dense_selected_g[selected_atoms] = dense_g[selected_atoms]
            dense_full_action = int(diag.details.get("full_action", -1))
            dense_topm_action = full_interface_action(dense_j0, dense_topm_g, valid_actions, cfg)
            dense_selected_action = full_interface_action(dense_j0, dense_selected_g, valid_actions, cfg)
            runtime_sparse_full_action = int(diag.details.get("sparse_full_action", -1))
            teacher_action = int(sample.teacher.a_star)
            diag.values.update(
                {
                    "hab_topm_dense_value_action_match": float(dense_topm_action == teacher_action),
                    "dense_vs_hab_topm_dense_value_match": float(dense_topm_action == dense_full_action),
                    "hab_topm_dense_value_vs_runtime_sparse_full_match": float(dense_topm_action == runtime_sparse_full_action),
                    "runtime_sparse_value_bridge_flip_rate": float(dense_topm_action != runtime_sparse_full_action),
                    "selected_budget_dense_value_action_match": float(dense_selected_action == teacher_action),
                    "dense_vs_selected_budget_dense_value_match": float(dense_selected_action == dense_full_action),
                    "selected_budget_dense_value_vs_deployed_match": float(dense_selected_action == int(tour.action_index)),
                }
            )

            # Literal novelty diagnostic: atom i is critical iff removing it
            # changes the dense winner action.  This is evaluated independently
            # of the historical margin-deficit proxy.
            # np.where (rather than multiplication) also sanitizes any malformed
            # non-finite values in padded/inactive rows: 0 * NaN is still NaN.
            active_dense_g = np.where(active_atoms[:, None], dense_g, 0.0).astype(
                dense_g.dtype, copy=False
            )
            dense_cost = dense_j0 + active_dense_g.sum(axis=0)
            dense_cost = np.where(valid_actions, dense_cost, np.inf)
            if np.isfinite(dense_cost).any() and active_atoms.any():
                winner = int(np.nanargmin(dense_cost))
                loo_cost = dense_cost[None, :] - active_dense_g
                loo_cost[:, ~valid_actions] = np.inf
                loo_winner = np.nanargmin(loo_cost, axis=1)
                critical_mask = active_atoms & (loo_winner != winner)
                critical_count = int(critical_mask.sum())
                if critical_count > 0:
                    topm_mask = np.zeros_like(critical_mask)
                    selected_mask = np.zeros_like(critical_mask)
                    topm_mask[topm_atoms] = True
                    selected_mask[selected_atoms] = True
                    diag.values["exact_winner_flip_critical_recall_topm"] = float((critical_mask & topm_mask).sum() / critical_count)
                    diag.values["exact_winner_flip_critical_recall_selected"] = float((critical_mask & selected_mask).sum() / critical_count)
                else:
                    diag.values["exact_winner_flip_critical_recall_topm"] = float("nan")
                    diag.values["exact_winner_flip_critical_recall_selected"] = float("nan")
                diag.values["exact_winner_flip_critical_atom_fraction"] = float(critical_count / max(int(active_atoms.sum()), 1))
                diag.values["exact_winner_flip_critical_scene_rate"] = float(critical_count > 0)
                diag.details["dense_hab_topm_action"] = int(dense_topm_action)
                diag.details["dense_selected_budget_action"] = int(dense_selected_action)
        selected_local_anchor_action = int(tour_diag.get("pair_action_anchor_action", diag.details.get("sparse_full_action", -1)))
        teacher_action_for_anchor = int(sample.teacher.a_star)
        if selected_local_anchor_action >= 0:
            anchor_correct = selected_local_anchor_action == teacher_action_for_anchor
            deployed_correct = int(tour.action_index) == teacher_action_for_anchor
            diag.values["selected_local_anchor_action_match"] = float(anchor_correct)
            if 0 <= selected_local_anchor_action < len(sample.teacher.J_T):
                diag.values["selected_local_anchor_teacher_regret"] = float(
                    sample.teacher.J_T[selected_local_anchor_action] - sample.teacher.J_T[teacher_action_for_anchor]
                )
            diag.values["deployed_vs_selected_local_anchor_match"] = float(int(tour.action_index) == selected_local_anchor_action)
            diag.values["pair_potential_deployed_flip_rate"] = float(int(tour.action_index) != selected_local_anchor_action)
            diag.values["beneficial_pair_potential_intervention_rate"] = float((not anchor_correct) and deployed_correct)
            diag.values["harmful_pair_potential_intervention_rate"] = float(anchor_correct and not deployed_correct)
            diag.details["selected_local_anchor_action"] = int(selected_local_anchor_action)

        if pair_full_action >= 0:
            teacher_action = int(sample.teacher.a_star)
            budget_action = int(tour.action_index)
            dense_action = int(diag.details.get("full_action", -1))
            pair_full_correct = pair_full_action == teacher_action
            budget_correct = budget_action == teacher_action
            diag.values["pair_full_interface_action_match"] = float(pair_full_correct)
            if 0 <= pair_full_action < len(sample.teacher.J_T):
                diag.values["pair_full_teacher_regret"] = float(sample.teacher.J_T[pair_full_action] - sample.teacher.J_T[teacher_action])
            diag.values["budget_vs_pair_full_match"] = float(budget_action == pair_full_action)
            diag.values["pair_full_to_budget_flip_rate"] = float(budget_action != pair_full_action)
            diag.values["harmful_pair_compression_rate"] = float(pair_full_correct and not budget_correct)
            diag.values["beneficial_pair_compression_rate"] = float((not pair_full_correct) and budget_correct)
            if dense_action >= 0:
                dense_correct = dense_action == teacher_action
                diag.values["dense_to_pair_full_flip_rate"] = float(dense_action != pair_full_action)
                diag.values["harmful_pair_interface_rate"] = float(dense_correct and not pair_full_correct)
                diag.values["beneficial_pair_interface_rate"] = float((not dense_correct) and pair_full_correct)
            if local_pair_full_action >= 0:
                local_correct = local_pair_full_action == teacher_action
                diag.values["local_pair_full_interface_action_match"] = float(local_correct)
                if 0 <= local_pair_full_action < len(sample.teacher.J_T):
                    diag.values["local_pair_full_teacher_regret"] = float(sample.teacher.J_T[local_pair_full_action] - sample.teacher.J_T[teacher_action])
                diag.values["local_pair_full_to_residual_flip_rate"] = float(local_pair_full_action != pair_full_action)
                diag.values["harmful_residual_intervention_rate"] = float(local_correct and not pair_full_correct)
                diag.values["beneficial_residual_intervention_rate"] = float((not local_correct) and pair_full_correct)
                if dense_action >= 0:
                    diag.values["dense_to_local_pair_full_flip_rate"] = float(dense_action != local_pair_full_action)
            cert_fraction = float(qdiag.get("selector_aocc_certified_pair_fraction", float("nan")))
            fully_certified = bool(np.isfinite(cert_fraction) and cert_fraction >= 1.0 - 1e-8)
            diag.values["aocc_fully_certified_scene_rate"] = float(fully_certified)
            diag.values["teacher_action_match_fully_certified"] = float(budget_correct) if fully_certified else float("nan")
            diag.values["teacher_action_match_not_fully_certified"] = float(budget_correct) if not fully_certified else float("nan")
            diag.details["pair_full_action"] = int(pair_full_action)
            diag.details["local_pair_full_action"] = int(local_pair_full_action)
        metric_means.update(diag)
        if args.per_sample_output:
            row = {
                "scenario_token": str(getattr(sample, "scenario_token", "")),
                "timestamp_us": int(getattr(sample, "timestamp_us", 0) or 0),
                **{k: float(v) for k, v in diag.values.items()},
                "teacher_action": int(getattr(sample.teacher, "a_star", -1)),
                "bdse_action": int(tour.action_index),
                "full_action": int(diag.details.get("full_action", -1)),
                "sparse_full_action": int(diag.details.get("sparse_full_action", -1)),
                "selected_local_anchor_action": int(diag.details.get("selected_local_anchor_action", -1)),
                "pair_full_action": int(diag.details.get("pair_full_action", -1)),
                "local_pair_full_action": int(diag.details.get("local_pair_full_action", -1)),
                "fallback_would_trigger": bool(qdiag.get("fallback_would_trigger", False)),
                "planner_latency_ms": float(planner_latency_ms),
            }
            assert per_sample_file is not None
            per_sample_file.write(json.dumps(row, sort_keys=True) + "\n")
    if per_sample_file is not None:
        per_sample_file.close()
    summary = metric_means.result()
    summary["device"] = str(device)
    if planner_latencies_ms:
        latency = np.asarray(planner_latencies_ms, dtype=np.float64)
        summary.update(
            {
                "planner_latency_ms_mean": float(latency.mean()),
                "planner_latency_ms_p50": float(np.quantile(latency, 0.50)),
                "planner_latency_ms_p90": float(np.quantile(latency, 0.90)),
                "planner_latency_ms_p95": float(np.quantile(latency, 0.95)),
                "planner_latency_ms_p99": float(np.quantile(latency, 0.99)),
                "planner_latency_ms_max": float(latency.max()),
            }
        )
    if device.type == "cuda":
        summary["cuda_peak_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
