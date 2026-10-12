"""Offline protocols for decision-sufficient evidence and plan intervention.

This module deliberately separates *scientific replay* from latency evidence.
Dense predictions and teacher outcomes may be materialized once to compare
acquisition policies on identical scenes.  Such replay is valid for quality,
calibration, criticality, stopping and intervention diagnostics, but it is not
evidence that an online implementation saved computation.  Executed-work claims
must come from ``profile_planning_budget`` or the profiled nuPlan planner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _aligned(values: Any, shape: tuple[int, ...], fill: float = 0.0, dtype=np.float64) -> np.ndarray:
    out = np.full(shape, fill, dtype=dtype)
    src = np.asarray(values, dtype=dtype)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(src.shape, shape))
    if len(slices) == len(shape):
        out[slices] = src[slices]
    return out


def _winner(cost: np.ndarray, valid: np.ndarray) -> int:
    masked = np.where(valid, np.asarray(cost, dtype=np.float64), np.inf)
    return int(np.argmin(masked)) if np.isfinite(masked).any() else -1


def _stable_desc(score: np.ndarray, active: np.ndarray) -> np.ndarray:
    ids = np.flatnonzero(active)
    return ids[np.argsort(-np.nan_to_num(score[ids], nan=-np.inf), kind="stable")]


def acquisition_orders(
    *, proposal: np.ndarray, pred_g: np.ndarray, pred_var: np.ndarray,
    active: np.ndarray, valid: np.ndarray, reference: int,
    hard: np.ndarray, random_seed: int,
) -> dict[str, np.ndarray]:
    """Return comparable descriptor orders; only ``adaptive`` is deployment-cheap.

    ``magnitude`` and ``uncertainty`` intentionally use dense counterfactual
    predictions and are therefore diagnostic upper-bound baselines unless their
    ranking overhead is separately profiled.  ``adaptive`` uses the learned,
    scene-conditioned proposal score and later adapts query count through the
    stopping rule; the frozen model does not yet outcome-condition the next rank.
    """
    E, _ = pred_g.shape
    proposal = _aligned(proposal, (E,), fill=-np.inf)
    hard = _aligned(hard, (E,), fill=0, dtype=bool)
    if reference < 0:
        reference = int(np.flatnonzero(valid)[0])
    rel = pred_g - pred_g[:, [reference]]
    magnitude = np.max(np.abs(rel[:, valid]), axis=1) if valid.any() else np.zeros(E)
    rel_var = pred_var + pred_var[:, [reference]]
    uncertainty = np.max(np.sqrt(np.maximum(rel_var[:, valid], 0.0)), axis=1) if valid.any() else np.zeros(E)
    static_score = hard.astype(np.float64) * 1e6 - np.arange(E, dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    random_score = rng.random(E)
    return {
        "adaptive": _stable_desc(proposal, active),
        "static": _stable_desc(static_score, active),
        "random": _stable_desc(random_score, active),
        "magnitude": _stable_desc(magnitude, active),
        "uncertainty": _stable_desc(uncertainty, active),
    }


def select_prefix(order: np.ndarray, costs: np.ndarray, max_atoms: int, max_cost: float | None = None) -> np.ndarray:
    chosen: list[int] = []
    spent = 0.0
    for atom in np.asarray(order, dtype=np.int64).tolist():
        if len(chosen) >= max_atoms:
            break
        cost = max(float(costs[atom]), 0.0)
        if max_cost is not None and chosen and spent + cost > max_cost + 1e-9:
            continue
        if max_cost is not None and not chosen and cost > max_cost + 1e-9:
            continue
        chosen.append(int(atom))
        spent += cost
    return np.asarray(chosen, dtype=np.int64)


def decision_from_atoms(j0: np.ndarray, pred_g: np.ndarray, selected: np.ndarray, valid: np.ndarray) -> tuple[int, np.ndarray]:
    cost = np.asarray(j0, dtype=np.float64).copy()
    if len(selected):
        cost += np.asarray(pred_g, dtype=np.float64)[selected].sum(axis=0)
    return _winner(cost, valid), cost


def literal_decision_critical(teacher_j0: np.ndarray, teacher_g: np.ndarray, active: np.ndarray, valid: np.ndarray) -> np.ndarray:
    full = np.asarray(teacher_j0, dtype=np.float64) + np.where(active[:, None], teacher_g, 0.0).sum(axis=0)
    win = _winner(full, valid)
    out = np.zeros(len(active), dtype=bool)
    if win < 0:
        return out
    for atom in np.flatnonzero(active):
        out[atom] = _winner(full - teacher_g[atom], valid) != win
    return out


def physical_critical_proxy(
    teacher_g: np.ndarray, active: np.ndarray, reference: int, comparator: int,
    hard: np.ndarray, abs_threshold: float,
) -> np.ndarray:
    """Cost-domain proxy; final paper claims require raw physical consequence fields."""
    if reference < 0 or comparator < 0:
        return np.zeros(len(active), dtype=bool)
    contrast = np.abs(teacher_g[:, comparator] - teacher_g[:, reference])
    return active & ((contrast >= abs_threshold) | (hard & (contrast > 0)))


@dataclass
class StopResult:
    action: int
    selected: np.ndarray
    reason: str
    decided_update: bool
    predicted_improvement: float
    radius: float


def sequential_stop(
    order: np.ndarray, j0: np.ndarray, pred_g: np.ndarray, pred_var: np.ndarray,
    valid: np.ndarray, reference: int, max_atoms: int, *, z: float,
    update_margin: float, unqueried_radius: float,
) -> StopResult:
    selected: list[int] = []
    last_action = reference
    last_imp = 0.0
    last_radius = math.inf
    n = min(max_atoms, len(order))
    for step, atom in enumerate(np.asarray(order, dtype=np.int64)[:n], start=1):
        selected.append(int(atom))
        ids = np.asarray(selected, dtype=np.int64)
        action, cost = decision_from_atoms(j0, pred_g, ids, valid)
        if action < 0:
            continue
        improvement = float(cost[reference] - cost[action])
        variance = float(np.maximum(pred_var[ids, reference] + pred_var[ids, action], 0.0).sum())
        remaining = max(n - step, 0)
        radius = float(z) * math.sqrt(max(variance + remaining * unqueried_radius ** 2, 0.0))
        last_action, last_imp, last_radius = action, improvement, radius
        if action != reference and improvement - radius > update_margin:
            return StopResult(action, ids, "support", True, improvement, radius)
        if improvement + radius < update_margin:
            return StopResult(reference, ids, "refute", False, improvement, radius)
    ids = np.asarray(selected, dtype=np.int64)
    action, cost = decision_from_atoms(j0, pred_g, ids, valid)
    if action >= 0:
        last_imp = float(cost[reference] - cost[action])
    return StopResult(action, ids, "budget", action != reference, last_imp, last_radius)


def intervention_label(
    reference: int, action: int, oracle: int, teacher_cost: np.ndarray,
    hard_violation: np.ndarray, delta: float,
) -> str:
    if min(reference, action, oracle) < 0:
        return "invalid"
    oracle_gain = float(teacher_cost[reference] - teacher_cost[oracle])
    if action == reference:
        return "missed" if oracle_gain > delta else "retained"
    gain = float(teacher_cost[reference] - teacher_cost[action])
    hard_worse = bool(hard_violation[action] and not hard_violation[reference])
    if hard_worse or gain < -delta:
        return "harmful"
    if gain > delta:
        return "beneficial"
    return "unnecessary"


class ScalarLog:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)

    def add(self, key: str, value: float) -> None:
        if np.isfinite(value):
            self.values[key].append(float(value))

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key, values in sorted(self.values.items()):
            arr = np.asarray(values, dtype=np.float64)
            out[key] = {
                "n": int(len(arr)), "mean": float(arr.mean()),
                "p50": float(np.quantile(arr, .5)), "p95": float(np.quantile(arr, .95)),
            }
        return out


class CalibrationLog:
    def __init__(self, bins: int = 10) -> None:
        self.abs_error: list[np.ndarray] = []
        self.std: list[np.ndarray] = []
        self.bins = bins

    def add(self, error: np.ndarray, variance: np.ndarray) -> None:
        error = np.asarray(error, dtype=np.float64).reshape(-1)
        std = np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64).reshape(-1), 0.0))
        ok = np.isfinite(error) & np.isfinite(std)
        if ok.any():
            self.abs_error.append(np.abs(error[ok])); self.std.append(std[ok])

    def summary(self) -> dict[str, Any]:
        if not self.abs_error:
            return {"n": 0}
        err = np.concatenate(self.abs_error); std = np.concatenate(self.std)
        result: dict[str, Any] = {"n": int(len(err))}
        for z in (1.0, 1.645, 1.96):
            result[f"coverage_z{z:g}"] = float(np.mean(err <= z * std))
        edges = np.quantile(std, np.linspace(0, 1, self.bins + 1))
        rows, weighted = [], 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (std >= lo) & (std <= hi if hi == edges[-1] else std < hi)
            if not mask.any():
                continue
            rmse = float(np.sqrt(np.mean(err[mask] ** 2))); pred = float(np.mean(std[mask]))
            weighted += float(mask.mean()) * abs(rmse - pred)
            rows.append({"n": int(mask.sum()), "mean_std": pred, "rmse": rmse})
        result["ence"] = float(weighted); result["bins"] = rows
        return result


def _seed_for(token: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{token}:{seed}".encode()).digest()[:8], "little")


def _reference_action(sample: Any, j0: np.ndarray, valid: np.ndarray) -> tuple[int, str]:
    for container in (getattr(sample.runtime, "metadata", {}) or {},):
        for key in ("reference_action_index", "incumbent_action_index", "published_action_index"):
            try:
                value = int(container[key])
            except Exception:
                continue
            if 0 <= value < len(valid) and valid[value]:
                return value, f"runtime_metadata.{key}"
    return _winner(j0, valid), "argmin_predicted_base_proxy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--preprocessed-dir", required=True)
    p.add_argument("--split", nargs="+", default=["val"])
    p.add_argument("--max-scenarios", type=int, default=1000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 12, 16, 24])
    p.add_argument("--budget-unit", choices=["atoms", "measured_cost"], default="atoms")
    p.add_argument("--random-seeds", type=int, nargs="+", default=[17, 29, 43])
    p.add_argument("--physical-delta", type=float, default=0.05)
    p.add_argument("--intervention-delta", type=float, default=0.05)
    p.add_argument("--stop-z", type=float, nargs="+", default=[0.0, 1.0, 1.645, 1.96])
    p.add_argument("--stop-unqueried-radius", type=float, default=0.05)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-scene-output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.budgets) <= 0 or args.max_scenarios <= 0:
        raise SystemExit("budgets and max-scenarios must be positive")
    import torch
    from bdse.config import load_config
    from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
    from bdse.external_baselines.model_factory import load_model_for_config
    from bdse.utils import configure_torch_for_device, resolve_torch_device

    cfg = load_config(args.config)
    device = resolve_torch_device(args.device, context="evidence protocol")
    configure_torch_for_device(device)
    model = load_model_for_config(args.checkpoint, cfg, device)
    if not hasattr(model, "predict_dense_numpy") or not hasattr(model, "predict_certificate_numpy"):
        raise RuntimeError("evidence protocol requires native dense and certificate prediction APIs")
    dataset = PreprocessedBDSEDataset(
        args.preprocessed_dir, split=args.split, max_scenarios=args.max_scenarios,
        max_scenarios_strategy="uniform_blocks",
    )
    log = ScalarLog(); cal_abs = CalibrationLog(); cal_rel = CalibrationLog()
    intervention_counts: dict[str, int] = defaultdict(int)
    reference_sources: dict[str, int] = defaultdict(int)
    scene_rows: list[dict[str, Any]] = []
    processed = 0
    for sample in dataset.iter_samples():
        if sample.teacher is None:
            continue
        scope_factory = getattr(model, "runtime_prediction_cache_scope", None)
        with (scope_factory() if callable(scope_factory) else nullcontext()):
            pred = model.predict_certificate_numpy(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
            dense = model.predict_dense_numpy(sample.runtime, sample.candidates, sample.evidence_bank, cfg)
        pred_g = np.asarray(dense["g"], dtype=np.float64)
        pred_var = np.asarray(dense.get("g_var", np.zeros_like(pred_g)), dtype=np.float64)
        E, K = pred_g.shape
        valid = _aligned(sample.candidates.valid_mask, (K,), fill=0, dtype=bool)
        active = _aligned(sample.evidence_bank.active_mask, (E,), fill=0, dtype=bool)
        hard = _aligned(sample.evidence_bank.hard_mask(), (E,), fill=0, dtype=bool)
        costs = _aligned(sample.evidence_bank.budget_costs(), (E,), fill=np.inf)
        j0 = _aligned(dense.get("J0_deployment", dense["J0"]), (K,), fill=np.inf)
        teacher_g = _aligned(sample.teacher.g_evid, (E, K))
        teacher_j0 = _aligned(sample.teacher.J_base, (K,), fill=np.inf)
        teacher_cost = _aligned(sample.teacher.J_T, (K,), fill=np.inf)
        hard_violation = _aligned(sample.teacher.hard_violation_mask, (K,), fill=1, dtype=bool)
        oracle = _winner(teacher_cost, valid)
        reference, reference_source = _reference_action(sample, j0, valid)
        reference_sources[reference_source] += 1
        if oracle < 0 or reference < 0:
            continue
        mask = active[:, None] & valid[None, :]
        error = pred_g - teacher_g
        cal_abs.add(error[mask], pred_var[mask])
        rel_error = (pred_g - pred_g[:, [reference]]) - (teacher_g - teacher_g[:, [reference]])
        rel_var = pred_var + pred_var[:, [reference]]
        cal_rel.add(rel_error[mask], rel_var[mask])
        log.add("evidence/absolute_mae", float(np.mean(np.abs(error[mask]))))
        log.add("evidence/relative_mae", float(np.mean(np.abs(rel_error[mask]))))
        nonzero = mask & (np.abs(teacher_g - teacher_g[:, [reference]]) > args.physical_delta)
        if nonzero.any():
            log.add("evidence/relative_sign_accuracy", float(np.mean(
                np.sign(pred_g - pred_g[:, [reference]])[nonzero] == np.sign(teacher_g - teacher_g[:, [reference]])[nonzero]
            )))
        decision_critical = literal_decision_critical(teacher_j0, teacher_g, active, valid)
        physical_proxy = physical_critical_proxy(
            teacher_g, active, reference, oracle, hard, args.physical_delta
        )
        dc = int(decision_critical.sum()); pc = int(physical_proxy.sum()); both = int((decision_critical & physical_proxy).sum())
        log.add("criticality/decision_scene_rate", float(dc > 0))
        log.add("criticality/physical_proxy_scene_rate", float(pc > 0))
        if pc: log.add("criticality/decision_precision_given_physical", both / pc)
        if dc: log.add("criticality/physical_recall_of_decision", both / dc)
        proposal = _aligned(pred.get("proposal_logits", np.zeros(E)), (E,), fill=-np.inf)
        policy_orders = acquisition_orders(
            proposal=proposal, pred_g=pred_g, pred_var=pred_var, active=active, valid=valid,
            reference=reference, hard=hard, random_seed=_seed_for(str(sample.scenario_token), args.random_seeds[0]),
        )
        # Average random policy over registered seeds without changing other arms.
        random_orders = []
        for seed in args.random_seeds:
            random_orders.append(acquisition_orders(
                proposal=proposal, pred_g=pred_g, pred_var=pred_var, active=active, valid=valid,
                reference=reference, hard=hard, random_seed=_seed_for(str(sample.scenario_token), seed),
            )["random"])
        for budget in sorted(set(args.budgets)):
            for policy, order in policy_orders.items():
                orders = random_orders if policy == "random" else [order]
                for rep_order in orders:
                    selected = select_prefix(
                        rep_order, costs,
                        budget if args.budget_unit == "atoms" else E,
                        None if args.budget_unit == "atoms" else float(budget),
                    )
                    action, _ = decision_from_atoms(j0, pred_g, selected, valid)
                    prefix = f"policy/{policy}/B{budget}"
                    log.add(prefix + "/query_atoms", len(selected))
                    log.add(prefix + "/query_cost", float(costs[selected].sum()) if len(selected) else 0.0)
                    log.add(prefix + "/oracle_match", float(action == oracle))
                    log.add(prefix + "/teacher_regret", float(teacher_cost[action] - teacher_cost[oracle]))
                    log.add(prefix + "/decision_critical_recall", float((decision_critical[selected]).sum() / dc) if dc else float("nan"))
                    label = intervention_label(reference, action, oracle, teacher_cost, hard_violation, args.intervention_delta)
                    intervention_counts[f"{policy}/B{budget}/{label}"] += 1
            learned_order = policy_orders["adaptive"]
            full_ids = select_prefix(
                learned_order, costs,
                budget if args.budget_unit == "atoms" else E,
                None if args.budget_unit == "atoms" else float(budget),
            )
            for z in args.stop_z:
                stopped = sequential_stop(
                    full_ids, j0, pred_g, pred_var, valid, reference, len(full_ids),
                    z=z, update_margin=args.intervention_delta,
                    unqueried_radius=args.stop_unqueried_radius,
                )
                full_action, _ = decision_from_atoms(j0, pred_g, full_ids, valid)
                prefix = f"stopping/z{z:g}/B{budget}"
                early = stopped.reason != "budget"
                log.add(prefix + "/coverage_early", float(early))
                log.add(prefix + "/query_atoms", len(stopped.selected))
                log.add(prefix + "/teacher_risk", float(stopped.action != oracle))
                if early:
                    log.add(prefix + "/selective_teacher_risk", float(stopped.action != oracle))
                log.add(prefix + "/premature_stop", float(early and stopped.action != full_action))
                log.add(prefix + "/post_stop_predicted_flip", float(stopped.action != full_action))
                log.add(prefix + "/post_stop_oracle_flip", float(stopped.action != oracle))
                log.add(prefix + "/teacher_regret", float(teacher_cost[stopped.action] - teacher_cost[oracle]))
                stop_label = intervention_label(
                    reference, stopped.action, oracle, teacher_cost, hard_violation, args.intervention_delta
                )
                intervention_counts[f"stopping_z{z:g}/B{budget}/{stop_label}"] += 1
        scene_rows.append({
            "scenario_token": str(sample.scenario_token), "timestamp_us": int(sample.timestamp_us),
            "reference_action": reference, "reference_source": reference_source, "teacher_oracle_action": oracle,
            "active_atoms": int(active.sum()), "valid_actions": int(valid.sum()),
            "physical_critical_proxy_count": pc, "decision_critical_count": dc,
        })
        processed += 1
        if processed >= args.max_scenarios:
            break
    payload = {
        "schema": "bdse_evidence_protocol_v1",
        "scenes": processed,
        "configuration": vars(args),
        "capability_and_claim_boundary": {
            "adaptive_scene_conditioned_ranking": True,
            "adaptive_query_count_via_stopping": True,
            "outcome_conditioned_next_query_reranking": False,
            "dense_replay_precomputes_all_predictions": True,
            "dense_replay_is_latency_evidence": False,
            "magnitude_and_uncertainty_ranking_are_diagnostic_upper_bounds": True,
            "physical_criticality_is_cost_domain_proxy": True,
            "late_invalid_requires_latency_aware_closed_loop": True,
            "reference_fallback_is_predicted_base_proxy": True,
        },
        "reference_sources": dict(sorted(reference_sources.items())),
        "metrics": log.summary(),
        "absolute_consequence_calibration": cal_abs.summary(),
        "relative_consequence_calibration": cal_rel.summary(),
        "intervention_counts": dict(sorted(intervention_counts.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if args.per_scene_output is not None:
        args.per_scene_output.parent.mkdir(parents=True, exist_ok=True)
        args.per_scene_output.write_text("".join(json.dumps(row) + "\n" for row in scene_rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scenes": processed}))


if __name__ == "__main__":
    main()
