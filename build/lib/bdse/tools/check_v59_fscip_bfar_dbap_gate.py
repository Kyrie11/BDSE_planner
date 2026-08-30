from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _finite(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value if math.isfinite(value) else default


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("scenario_token", "")), int(row.get("timestamp_us", 0) or 0)


def _index(rows: list[dict[str, Any]], label: str) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if not key[0] or key in out:
            raise ValueError(f"{label} has empty/duplicate key {key}")
        out[key] = row
    return out


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"median": float("nan"), "p90": float("nan"), "cvar90": float("nan")}
    q90 = float(np.quantile(values, 0.90))
    return {
        "median": float(np.quantile(values, 0.50)),
        "p90": q90,
        "cvar90": float(values[values >= q90].mean()),
    }


def _paired_action_mismatch(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], field: str, a_label: str, b_label: str) -> tuple[float, int]:
    a = _index(a_rows, a_label)
    b = _index(b_rows, b_label)
    if set(a) != set(b):
        raise ValueError(f"{a_label}/{b_label} scenario keys differ: {len(a)} vs {len(b)}")
    keys = sorted(a)
    mismatches = 0
    valid = 0
    for key in keys:
        av, bv = a[key].get(field), b[key].get(field)
        if av is None or bv is None:
            continue
        valid += 1
        mismatches += int(int(av) != int(bv))
    return (float(mismatches / valid) if valid else float("nan"), valid)


def _paired_deployed_flip_stats(
    candidate_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    candidate_label: str = "candidate",
    control_label: str = "local",
) -> dict[str, float]:
    candidate = _index(candidate_rows, candidate_label)
    control = _index(control_rows, control_label)
    if set(candidate) != set(control):
        raise ValueError(f"{candidate_label}/{control_label} scenario keys differ: {len(candidate)} vs {len(control)}")
    flips = beneficial = harmful = unchanged = valid = 0
    for key in sorted(candidate):
        c_row, l_row = candidate[key], control[key]
        if any(field not in c_row for field in ("bdse_action", "teacher_action")) or "bdse_action" not in l_row:
            continue
        c_action = int(c_row["bdse_action"]); l_action = int(l_row["bdse_action"]); teacher = int(c_row["teacher_action"])
        valid += 1
        if c_action == l_action:
            unchanged += 1
            continue
        flips += 1
        beneficial += int(l_action != teacher and c_action == teacher)
        harmful += int(l_action == teacher and c_action != teacher)
    denom = max(valid, 1)
    return {
        "n": float(valid),
        "flip_rate": float(flips / denom),
        "beneficial_rate": float(beneficial / denom),
        "harmful_rate": float(harmful / denom),
        "beneficial_given_flip": float(beneficial / max(flips, 1)),
        "harmful_given_flip": float(harmful / max(flips, 1)),
        "unchanged_rate": float(unchanged / denom),
    }


def _paired_regret(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], a_label: str, b_label: str):
    a = _index(a_rows, a_label)
    b = _index(b_rows, b_label)
    if set(a) != set(b):
        raise ValueError(f"{a_label}/{b_label} scenario keys differ: {len(a)} vs {len(b)}")
    keys = sorted(a)
    av = np.asarray([_finite(a[k], "teacher_regret") for k in keys], dtype=np.float64)
    bv = np.asarray([_finite(b[k], "teacher_regret") for k in keys], dtype=np.float64)
    ok = np.isfinite(av) & np.isfinite(bv)
    av, bv = av[ok], bv[ok]
    return _quantiles(av), _quantiles(bv), _quantiles(av - bv), int(ok.sum())


def _training_health(
    path: Path, min_last_exact: float, train_config: dict[str, Any] | None = None
) -> tuple[list[str], dict[str, float]]:
    failures: list[str] = []
    rows = _load_rows(path)
    epochs = [int(r.get("epoch", -1)) for r in rows]
    duplicates = sorted({e for e in epochs if epochs.count(e) > 1})
    if duplicates:
        failures.append(f"duplicate training epochs: {duplicates[:12]}")
    exact, pair_fraction, action_family_active = [], [], []
    winner_loss_keys = (
        "L_pair_full_action",
        "L_pair_full_winner_margin",
        "L_budget_preserve_pair_full",
        "L_pair_full_anchor_preserve",
        "L_action_potential_teacher",
        "L_residual_winner_correction",
        "L_certified_residual_winner",
        "L_residual_boundary_margin_distill",
    )
    winner_loss_values: list[float] = []
    winner_loss_by_key: dict[str, list[float]] = {key: [] for key in winner_loss_keys}
    deploy_loss_values: list[float] = []
    uncertainty_loss_values: list[float] = []
    for row in rows:
        for key, value in row.items():
            if key == "epoch" or not isinstance(value, (int, float)):
                continue
            if key == "loss" or key.startswith("L_") or key in {"selector_exact_fraction", "training_pair_fraction"}:
                if not math.isfinite(float(value)):
                    failures.append(f"non-finite training metric epoch={row.get('epoch')}: {key}={value}")
                    break
        x = _finite(row, "selector_exact_fraction")
        if math.isfinite(x): exact.append(x)
        x = _finite(row, "training_pair_fraction")
        if math.isfinite(x): pair_fraction.append(x)
        x = _finite(row, "action_family_enabled")
        if math.isfinite(x): action_family_active.append(x)
        for key in winner_loss_keys:
            value = _finite(row, key)
            if math.isfinite(value):
                winner_loss_values.append(abs(value))
                winner_loss_by_key[key].append(abs(value))
        uncertainty_value = _finite(row, "L_residual_action_uncertainty")
        if math.isfinite(uncertainty_value):
            uncertainty_loss_values.append(abs(uncertainty_value))
        value = _finite(row, "L_deploy_select")
        if math.isfinite(value): deploy_loss_values.append(abs(value))
    last_exact = exact[-1] if exact else float("nan")
    if not math.isfinite(last_exact) or last_exact < min_last_exact:
        failures.append(f"last selector_exact_fraction={last_exact} < {min_last_exact}")
    if not exact or max(exact) <= 0.0:
        failures.append("no exact selector supervision observed")

    training_cfg = (train_config or {}).get("training", {}) or {}
    loss_weights = training_cfg.get("loss_weights", {}) or {}
    configured_winner_supervision = any(
        float(loss_weights.get(name, 0.0)) > 0.0
        for name in (
            "pair_full_action", "pair_full_winner_margin",
            "budget_preserve_pair_full", "pair_full_anchor_preserve",
            "action_potential_teacher", "residual_winner_correction",
            "certified_residual_winner",
            "residual_boundary_margin_distill",
        )
    )
    if configured_winner_supervision:
        if action_family_active and max(action_family_active) <= 0.0:
            failures.append("winner/deployment action-family branch never activated")
        if not winner_loss_values or max(winner_loss_values) <= 1.0e-12:
            failures.append("no non-zero winner-level supervision observed")
        metric_for_weight = {
            "pair_full_action": "L_pair_full_action",
            "pair_full_winner_margin": "L_pair_full_winner_margin",
            "budget_preserve_pair_full": "L_budget_preserve_pair_full",
            "pair_full_anchor_preserve": "L_pair_full_anchor_preserve",
            "action_potential_teacher": "L_action_potential_teacher",
            "residual_winner_correction": "L_residual_winner_correction",
            "certified_residual_winner": "L_certified_residual_winner",
            "residual_boundary_margin_distill": "L_residual_boundary_margin_distill",
        }
        for weight_name, metric_name in metric_for_weight.items():
            if float(loss_weights.get(weight_name, 0.0)) <= 0.0:
                continue
            values = winner_loss_by_key.get(metric_name, [])
            if not values or max(values) <= 1.0e-12:
                failures.append(
                    f"configured winner loss {weight_name}/{metric_name} never became non-zero"
                )
    if float(loss_weights.get("residual_action_uncertainty", 0.0)) > 0.0:
        if not uncertainty_loss_values or max(uncertainty_loss_values) <= 1.0e-12:
            failures.append("configured residual uncertainty loss never became non-zero")
    if float(loss_weights.get("deployment_selection", 0.0)) > 0.0:
        if not deploy_loss_values or max(deploy_loss_values) <= 1.0e-12:
            failures.append("no non-zero exact deployment-selection distillation observed")
    return failures, {
        "rows": float(len(rows)),
        "unique_epochs": float(len(set(epochs))),
        "last_exact_fraction": float(last_exact),
        "max_exact_fraction": float(max(exact)) if exact else float("nan"),
        "mean_training_pair_fraction": float(np.mean(pair_fraction)) if pair_fraction else float("nan"),
        "max_action_family_enabled": float(max(action_family_active)) if action_family_active else float("nan"),
        "max_winner_level_loss": float(max(winner_loss_values)) if winner_loss_values else float("nan"),
        "max_deployment_selection_loss": float(max(deploy_loss_values)) if deploy_loss_values else float("nan"),
        "max_residual_uncertainty_loss": float(max(uncertainty_loss_values)) if uncertainty_loss_values else float("nan"),
        "max_winner_losses_by_key": {
            key: (float(max(values)) if values else float("nan"))
            for key, values in winner_loss_by_key.items()
        },
    }


def _evaluation_config_health(
    candidate_path: Path | None,
    local_path: Path | None,
    foundation_path: Path | None,
    train_config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Verify that calibrated candidate and causal controls differ only as intended."""
    if candidate_path is None and local_path is None and foundation_path is None:
        return [], {"checked": False}
    failures: list[str] = []
    if candidate_path is None or local_path is None or foundation_path is None:
        return ["candidate/local/foundation calibrated configs must be supplied together"], {"checked": False}

    def load(path: Path) -> dict[str, Any]:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"expected YAML mapping: {path}")
        return data

    try:
        cand, local, foundation = map(load, (candidate_path, local_path, foundation_path))
    except Exception as exc:
        return [f"failed to load calibrated evaluation configs: {type(exc).__name__}: {exc}"], {"checked": False}

    systems = {"candidate": cand, "local": local, "foundation": foundation}
    evidence_eps: dict[str, float] = {}
    tournament_eps: dict[str, float] = {}
    residual_eps: dict[str, float] = {}
    beta: dict[str, float] = {}
    residual_disabled: dict[str, bool] = {}
    independently_calibrated: dict[str, bool] = {}
    selector_calibrated: dict[str, bool] = {}
    for label, cfg in systems.items():
        selector = cfg.get("selector", {}) or {}
        calibration = cfg.get("calibration", {}) or {}
        runtime = cfg.get("runtime", {}) or {}
        dual = runtime.get("dual_certificate", {}) or {}
        tournament = cfg.get("tournament", {}) or {}
        evidence_eps[label] = float(selector.get("adverse_certificate_epsilon", float("nan")))
        tournament_eps[label] = float(tournament.get("epsilon_cal", float("nan")))
        residual_eps[label] = float(dual.get("residual_epsilon_cal", float("nan")))
        beta[label] = float(tournament.get("beta_uncertainty", float("nan")))
        residual_disabled[label] = bool(runtime.get("disable_pair_residual_intervention", False))
        independently_calibrated[label] = bool(calibration.get("independent", False))
        selector_calibrated[label] = bool(selector.get("adverse_certificate_calibrated", False))
        if not independently_calibrated[label] or not selector_calibrated[label]:
            failures.append(f"{label} evidence certificate is not marked independently calibrated")
        if not math.isfinite(evidence_eps[label]) or evidence_eps[label] < 0.0:
            failures.append(f"{label} evidence epsilon is invalid: {evidence_eps[label]}")
        if not math.isfinite(residual_eps[label]) or residual_eps[label] < 0.0:
            failures.append(f"{label} residual epsilon is invalid: {residual_eps[label]}")
        if not math.isfinite(beta[label]) or beta[label] < 0.0:
            failures.append(f"{label} residual uncertainty beta is invalid: {beta[label]}")

    if max(evidence_eps.values()) - min(evidence_eps.values()) > 1.0e-9:
        failures.append(f"shared evidence calibration differs across systems: {evidence_eps}")
    if max(tournament_eps.values()) - min(tournament_eps.values()) > 1.0e-9:
        failures.append(f"tournament action-rule epsilon differs across systems: {tournament_eps}")
    if residual_disabled["candidate"]:
        failures.append("candidate residual intervention is disabled")
    for label in ("local", "foundation"):
        if not residual_disabled[label]:
            failures.append(f"{label} control did not disable residual intervention")
        if abs(residual_eps[label]) > 1.0e-12:
            failures.append(f"{label} control has non-zero residual calibration epsilon={residual_eps[label]}")

    train_beta = float(
        (((train_config.get("training", {}) or {}).get("certified_residual_winner", {}) or {}).get(
            "beta_uncertainty",
            (train_config.get("tournament", {}) or {}).get("beta_uncertainty", float("nan")),
        ))
    )
    if not math.isfinite(train_beta):
        failures.append("training certified-winner uncertainty beta is missing/non-finite")
    elif abs(beta["candidate"] - train_beta) > 1.0e-9:
        failures.append(
            f"training/deployment residual uncertainty beta mismatch: train={train_beta}, candidate={beta['candidate']}"
        )

    return failures, {
        "checked": True,
        "evidence_epsilon": evidence_eps,
        "tournament_epsilon": tournament_eps,
        "residual_epsilon": residual_eps,
        "beta_uncertainty": beta,
        "residual_disabled": residual_disabled,
        "independent_calibration": independently_calibrated,
        "selector_calibrated": selector_calibrated,
        "training_beta_uncertainty": train_beta,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Three-tier V59 FSCIP-BFAR open-loop gate")
    p.add_argument("candidate", type=Path)
    p.add_argument("local_control", type=Path)
    p.add_argument("foundation_control", type=Path)
    p.add_argument("--candidate-jsonl", type=Path, required=True)
    p.add_argument("--local-control-jsonl", type=Path, required=True)
    p.add_argument("--foundation-control-jsonl", type=Path, required=True)
    p.add_argument("--train-log", type=Path, required=True)
    p.add_argument("--report-json", type=Path, required=True)
    p.add_argument("--train-config", type=Path, required=True)
    p.add_argument("--candidate-config", type=Path)
    p.add_argument("--local-control-config", type=Path)
    p.add_argument("--foundation-control-config", type=Path)
    p.add_argument("--dual-calibration-json", type=Path, required=True)
    p.add_argument("--min-uniform-residual-calibration-scenes", type=int, default=1000)
    p.add_argument("--min-last-exact-fraction", type=float, default=None)
    p.add_argument("--latency-target-ms", type=float, default=500.0)
    p.add_argument("--enforce-latency", action="store_true")
    args = p.parse_args()

    cand, local, foundation = map(_load_json, (args.candidate, args.local_control, args.foundation_control))
    cand_rows, local_rows, foundation_rows = map(
        _load_rows, (args.candidate_jsonl, args.local_control_jsonl, args.foundation_control_jsonl)
    )
    train_config = yaml.safe_load(args.train_config.read_text(encoding="utf-8")) or {}
    configured_exact_floor = float((train_config.get("training", {}) or {}).get("min_deployment_exact_fraction", 0.0))
    exact_floor = configured_exact_floor if args.min_last_exact_fraction is None else float(args.min_last_exact_fraction)
    protocol_failures, train_stats = _training_health(args.train_log, exact_floor, train_config)
    config_failures, config_stats = _evaluation_config_health(
        args.candidate_config,
        args.local_control_config,
        args.foundation_control_config,
        train_config,
    )
    protocol_failures.extend(config_failures)
    calibration_stats: dict[str, Any] = {"checked": False}
    try:
        calibration_stats = _load_json(args.dual_calibration_json)
        calibration_stats["checked"] = True
        method = str(calibration_stats.get("method", ""))
        uniform_count = int(calibration_stats.get("residual_uniform_scene_count", 0) or 0)
        scene_count = int(calibration_stats.get("scene_count", 0) or 0)
        residual_eps_value = float(calibration_stats.get("recommended_residual_flip_epsilon", float("nan")))
        independent = bool(calibration_stats.get("independent_calibration", False))
        if "scene-uniform" not in method.lower():
            protocol_failures.append(f"residual calibration is not scene-uniform: method={method!r}")
        if not independent:
            protocol_failures.append("dual calibration is not marked independent")
        if uniform_count < int(args.min_uniform_residual_calibration_scenes):
            protocol_failures.append(
                f"uniform residual calibration scenes={uniform_count} < {args.min_uniform_residual_calibration_scenes}"
            )
        if scene_count > 0 and uniform_count / scene_count < 0.80:
            protocol_failures.append(
                f"uniform residual calibration coverage={uniform_count}/{scene_count} < 0.80"
            )
        if not math.isfinite(residual_eps_value) or residual_eps_value < 0.0:
            protocol_failures.append(f"invalid residual calibration epsilon={residual_eps_value}")
    except Exception as exc:
        protocol_failures.append(f"failed to audit dual calibration: {type(exc).__name__}: {exc}")
    train_stats["configured_min_exact_fraction"] = float(configured_exact_floor)
    train_stats["gate_min_exact_fraction"] = float(exact_floor)
    min_failures: list[str] = []
    comp_failures: list[str] = []
    warnings: list[str] = []

    def val(d: dict[str, Any], k: str) -> float: return _finite(d, k)
    cm, lm, fm = val(cand, "teacher_action_match"), val(local, "teacher_action_match"), val(foundation, "teacher_action_match")
    cpair, clocal = val(cand, "pair_full_interface_action_match"), val(cand, "local_pair_full_interface_action_match")
    local_pair_anchor = val(local, "local_pair_full_interface_action_match")
    internal_harmful = val(cand, "harmful_pair_potential_intervention_rate")
    internal_beneficial = val(cand, "beneficial_pair_potential_intervention_rate")
    if not math.isfinite(internal_harmful):
        internal_harmful = val(cand, "harmful_residual_intervention_rate")
    if not math.isfinite(internal_beneficial):
        internal_beneficial = val(cand, "beneficial_residual_intervention_rate")
    # Under the dual-certificate protocol, minimum/competitive evidence quality
    # is measured by the selected-local evidence certificate. Residual flip
    # uncertainty is evaluated separately and must not lower this value when the
    # residual proposes no action change.
    cert = val(cand, "evidence_certificate_fraction")
    if not math.isfinite(cert):
        cert = val(cand, "selector_aocc_certified_pair_fraction")
    residual_flip_cert = val(cand, "residual_flip_certificate_pass")
    dual_deployment_cert = val(cand, "dual_certificate_deployment_certified")
    residual_proposal_rate = val(cand, "residual_flip_proposed")
    residual_margin_pass_conditional = val(cand, "residual_flip_certificate_pass_conditional")
    dual_pass_conditional = val(cand, "dual_certificate_pass_conditional")
    frontier = val(cand, "selector_aocc_frontier_retained_weight_fraction")
    proposal, selected = val(cand, "proposal_decisive_atom_recall"), val(cand, "selected_decisive_atom_recall")
    effective, interaction = val(cand, "effective_selected_decisive_atom_recall"), val(cand, "selected_interaction_decisive_recall")
    fallback = val(cand, "fallback_would_trigger_rate")
    decision_atoms, configured = val(cand, "decision_budget_atom_count"), val(cand, "configured_decision_budget_atom_count")
    if not math.isfinite(decision_atoms): decision_atoms = val(cand, "selector_decision_budget_atom_count")
    if not math.isfinite(configured): configured = val(cand, "selector_budget")
    fill = decision_atoms / max(configured, 1e-9)
    calibrated, exact_target = val(cand, "selector_aocc_bound_calibrated"), val(cand, "selector_aocc_exact_tournament_target_active")
    latency = val(cand, "planner_latency_ms_p95")

    # V59 protocol integrity compares the same immutable interface row-by-row.
    # V53 incorrectly compared candidate.local_pair_full against local-control
    # pair_full, which are different interfaces and produced a false drift.
    try:
        local_anchor_drift, local_anchor_n = _paired_action_mismatch(
            cand_rows, local_rows, "local_pair_full_action", "candidate", "local"
        )
        dense_anchor_drift, dense_anchor_n = _paired_action_mismatch(
            cand_rows, local_rows, "full_action", "candidate", "local"
        )
        deployed_residual_flip_rate, deployed_flip_n = _paired_action_mismatch(
            cand_rows, local_rows, "bdse_action", "candidate", "local"
        )
        deployed_flip_stats = _paired_deployed_flip_stats(cand_rows, local_rows)
    except ValueError as exc:
        protocol_failures.append(str(exc))
        local_anchor_drift = dense_anchor_drift = deployed_residual_flip_rate = float("nan")
        local_anchor_n = dense_anchor_n = deployed_flip_n = 0
        deployed_flip_stats = {"n": 0.0, "flip_rate": float("nan"), "beneficial_rate": float("nan"), "harmful_rate": float("nan"), "beneficial_given_flip": float("nan"), "harmful_given_flip": float("nan"), "unchanged_rate": float("nan")}

    harmful = float(deployed_flip_stats.get("harmful_rate", float("nan")))
    beneficial = float(deployed_flip_stats.get("beneficial_rate", float("nan")))

    protocol_checks = [
        (math.isfinite(local_anchor_drift) and local_anchor_drift <= 0.005, f"local frozen-anchor row drift={local_anchor_drift} > 0.005"),
        (math.isfinite(dense_anchor_drift) and dense_anchor_drift <= 0.005, f"dense frozen-anchor row drift={dense_anchor_drift} > 0.005"),
        (math.isfinite(calibrated) and calibrated >= 0.5, f"AOCC evidence certificate is not independently calibrated: {calibrated}"),
        (math.isfinite(exact_target) and exact_target >= 0.5, f"exact downstream tournament target inactive: {exact_target}"),
    ]
    for ok, msg in protocol_checks:
        if not ok: protocol_failures.append(msg)

    # Minimum-completeness gate: protects against catastrophic algorithm or
    # protocol failures while allowing paired CL20 to generate the evidence
    # needed to decide what to optimize next.
    checks = [
        (math.isfinite(cm) and cm >= max(lm, fm) - 0.005, f"teacher match {cm} < best control {max(lm, fm)} - 0.005"),
        (math.isfinite(cpair) and cpair >= clocal - 0.005, f"pair-full match {cpair} < local anchor {clocal} - 0.005"),
        (math.isfinite(harmful) and harmful <= 0.05, f"harmful residual rate={harmful} > 0.05"),
        (math.isfinite(beneficial) and math.isfinite(harmful) and beneficial + 0.01 >= harmful, f"residual strongly net harmful: {beneficial}/{harmful}"),
        (math.isfinite(cert) and cert >= 0.40, f"certified fraction={cert} < 0.40"),
        (math.isfinite(frontier) and frontier >= 0.45, f"frontier retained={frontier} < 0.45"),
        (math.isfinite(proposal) and proposal >= 0.72, f"proposal decisive recall={proposal} < 0.72"),
        (math.isfinite(selected) and selected >= 0.50, f"selected decisive recall={selected} < 0.50"),
        (math.isfinite(effective) and effective >= 0.62, f"effective decisive recall={effective} < 0.62"),
        (math.isfinite(interaction) and interaction >= 0.40, f"interaction decisive recall={interaction} < 0.40"),
        (math.isfinite(fallback) and fallback <= 0.60, f"fallback rate={fallback} > 0.60"),
    ]
    for ok, msg in checks:
        if not ok: min_failures.append(msg)

    paired: dict[str, Any] = {}
    for label, rows in (("local", local_rows), ("foundation", foundation_rows)):
        try:
            cq, bq, dq, n = _paired_regret(cand_rows, rows, "candidate", label)
            paired[label] = {"candidate": cq, "control": bq, "delta": dq, "n": n}
            med_tol = max(250.0, 0.05 * abs(bq["median"]))
            p90_tol = max(500.0, 0.05 * abs(bq["p90"]))
            if cq["median"] > bq["median"] + med_tol or cq["p90"] > bq["p90"] + p90_tol:
                min_failures.append(f"paired regret catastrophically regressed vs {label}: {cq} / {bq}")
        except ValueError as exc:
            min_failures.append(str(exc))

    residual_epsilon = float(calibration_stats.get("recommended_residual_flip_epsilon", float("nan")))
    reserve = float((((train_config.get("training", {}) or {}).get("certified_residual_winner", {}) or {}).get("residual_epsilon_reserve", 0.0)))
    if math.isfinite(residual_epsilon) and residual_epsilon > reserve + 1.0e-9:
        warnings.append(
            f"calibrated residual epsilon={residual_epsilon:.6f} exceeds training reserve={reserve:.6f}; certified flips may be suppressed"
        )

    if math.isfinite(latency) and latency > args.latency_target_ms:
        msg = f"latency p95={latency} ms > {args.latency_target_ms} ms"
        if args.enforce_latency: min_failures.append(msg)
        else: warnings.append(msg + "; CL20 remains allowed, no real-time claim")

    # Competitive gate: unchanged paper-grade intent.  It does not suppress
    # CL20; it controls CL100/official-result escalation.
    competitive_checks = [
        (math.isfinite(cm) and cm >= fm + 0.015, f"total teacher-match gain={cm-fm:+.6f} < +0.015"),
        (math.isfinite(cm) and cm >= lm + 0.005, f"residual teacher-match gain={cm-lm:+.6f} < +0.005"),
        (math.isfinite(cpair) and cpair >= clocal + 0.005, f"pair-full residual gain={cpair-clocal:+.6f} < +0.005"),
        (math.isfinite(beneficial) and math.isfinite(harmful) and beneficial > harmful, f"residual not net beneficial={beneficial}/{harmful}"),
        (math.isfinite(harmful) and harmful <= 0.03, f"harmful residual rate={harmful} > 0.03"),
        (math.isfinite(cert) and cert >= 0.55, f"certified fraction={cert} < 0.55"),
        (math.isfinite(frontier) and frontier >= 0.55, f"frontier retained={frontier} < 0.55"),
        (math.isfinite(proposal) and proposal >= 0.80, f"proposal decisive recall={proposal} < 0.80"),
        (math.isfinite(selected) and selected >= 0.55, f"selected decisive recall={selected} < 0.55"),
        (math.isfinite(effective) and effective >= 0.70, f"effective decisive recall={effective} < 0.70"),
        (math.isfinite(interaction) and interaction >= 0.50, f"interaction decisive recall={interaction} < 0.50"),
        (math.isfinite(fallback) and fallback <= 0.40, f"fallback rate={fallback} > 0.40"),
        (math.isfinite(residual_proposal_rate) and residual_proposal_rate >= 0.001, f"raw residual proposal rate={residual_proposal_rate} < 0.001"),
    ]
    for ok, msg in competitive_checks:
        if not ok: comp_failures.append(msg)
    for label, stats in paired.items():
        if stats["candidate"]["median"] > stats["control"]["median"] + 1e-9 or stats["candidate"]["p90"] > stats["control"]["p90"] + 1e-9:
            comp_failures.append(f"paired regret regressed vs {label}: {stats['candidate']} / {stats['control']}")

    protocol_pass = not protocol_failures
    minimum_metrics_pass = not min_failures
    minimum_pass = protocol_pass and minimum_metrics_pass
    competitive_metrics_pass = not comp_failures
    competitive_pass = minimum_pass and competitive_metrics_pass
    report = {
        "gate": "v59_fscip_bfar",
        "protocol_pass": protocol_pass,
        "minimum_metrics_pass": minimum_metrics_pass,
        "minimum_pass": minimum_pass,
        "competitive_metrics_pass": competitive_metrics_pass,
        "competitive_pass": competitive_pass,
        "protocol_failures": protocol_failures,
        "minimum_failures": min_failures,
        "competitive_failures": comp_failures,
        "warnings": warnings,
        "metrics": {
            "teacher_match_candidate": cm, "teacher_match_local": lm, "teacher_match_foundation": fm,
            "pair_full_match": cpair, "local_pair_full_match": clocal, "local_control_local_pair_full_match": local_pair_anchor,
            "local_anchor_row_drift": local_anchor_drift, "dense_anchor_row_drift": dense_anchor_drift,
            "deployed_residual_flip_rate": deployed_residual_flip_rate,
            "selector_gain_vs_foundation": lm - fm, "residual_gain_vs_local": cm - lm,
            "pair_residual_gain_vs_local_pair": cpair - clocal,
            "harmful_residual_rate": harmful, "beneficial_residual_rate": beneficial,
            "internal_harmful_pair_potential_rate": internal_harmful,
            "internal_beneficial_pair_potential_rate": internal_beneficial,
            "paired_deployed_flip_stats": deployed_flip_stats,
            "evidence_certified_fraction": cert,
            "residual_flip_certificate_pass": residual_flip_cert,
            "dual_certificate_deployment_certified": dual_deployment_cert,
            "residual_proposal_rate": residual_proposal_rate,
            "residual_margin_pass_conditional": residual_margin_pass_conditional,
            "dual_pass_conditional": dual_pass_conditional,
            "residual_calibration_epsilon": residual_epsilon,
            "training_residual_epsilon_reserve": reserve,
            "frontier_retained": frontier,
            "proposal_decisive_recall": proposal, "selected_decisive_recall": selected,
            "effective_decisive_recall": effective, "interaction_decisive_recall": interaction,
            "fallback_rate": fallback, "budget_fill": fill, "latency_p95_ms": latency,
        },
        "training": train_stats,
        "evaluation_config_protocol": config_stats,
        "dual_calibration_protocol": calibration_stats,
        "paired_regret": paired,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nV59 protocol-integrity gate [{'PASS' if protocol_pass else 'FAIL'}]")
    print(f"V59 minimum-completeness gate [{'PASS' if minimum_pass else 'FAIL'}]")
    print(f"V59 competitive gate [{'PASS' if competitive_pass else 'FAIL'}]")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    for warning in warnings: print(f"  ! WARNING: {warning}")
    for failure in protocol_failures: print(f"  - PROTOCOL: {failure}")
    for failure in min_failures: print(f"  - MINIMUM: {failure}")
    for failure in comp_failures: print(f"  - COMPETITIVE: {failure}")
    return 0 if protocol_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
