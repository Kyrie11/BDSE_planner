from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


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


def _training_health(path: Path, min_last_exact: float) -> tuple[list[str], dict[str, float]]:
    failures: list[str] = []
    rows = _load_rows(path)
    epochs = [int(r.get("epoch", -1)) for r in rows]
    duplicates = sorted({e for e in epochs if epochs.count(e) > 1})
    if duplicates:
        failures.append(f"duplicate training epochs: {duplicates[:12]}")
    exact, pair_fraction = [], []
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
    last_exact = exact[-1] if exact else float("nan")
    if not math.isfinite(last_exact) or last_exact < min_last_exact:
        failures.append(f"last selector_exact_fraction={last_exact} < {min_last_exact}")
    if not exact or max(exact) <= 0.0:
        failures.append("no exact selector supervision observed")
    return failures, {
        "rows": float(len(rows)),
        "unique_epochs": float(len(set(epochs))),
        "last_exact_fraction": float(last_exact),
        "max_exact_fraction": float(max(exact)) if exact else float("nan"),
        "mean_training_pair_fraction": float(np.mean(pair_fraction)) if pair_fraction else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Two-tier V53 WC-BFAR open-loop gate")
    p.add_argument("candidate", type=Path)
    p.add_argument("local_control", type=Path)
    p.add_argument("foundation_control", type=Path)
    p.add_argument("--candidate-jsonl", type=Path, required=True)
    p.add_argument("--local-control-jsonl", type=Path, required=True)
    p.add_argument("--foundation-control-jsonl", type=Path, required=True)
    p.add_argument("--train-log", type=Path, required=True)
    p.add_argument("--report-json", type=Path, required=True)
    p.add_argument("--min-last-exact-fraction", type=float, default=0.03)
    p.add_argument("--latency-target-ms", type=float, default=500.0)
    p.add_argument("--enforce-latency", action="store_true")
    args = p.parse_args()

    cand, local, foundation = map(_load_json, (args.candidate, args.local_control, args.foundation_control))
    cand_rows, local_rows, foundation_rows = map(
        _load_rows, (args.candidate_jsonl, args.local_control_jsonl, args.foundation_control_jsonl)
    )
    min_failures, train_stats = _training_health(args.train_log, args.min_last_exact_fraction)
    comp_failures: list[str] = []
    warnings: list[str] = []

    def val(d: dict[str, Any], k: str) -> float: return _finite(d, k)
    cm, lm, fm = val(cand, "teacher_action_match"), val(local, "teacher_action_match"), val(foundation, "teacher_action_match")
    cpair, clocal = val(cand, "pair_full_interface_action_match"), val(cand, "local_pair_full_interface_action_match")
    local_pair = val(local, "pair_full_interface_action_match")
    drift = abs(clocal - local_pair) if math.isfinite(clocal) and math.isfinite(local_pair) else float("nan")
    harmful, beneficial = val(cand, "harmful_residual_intervention_rate"), val(cand, "beneficial_residual_intervention_rate")
    cert, frontier = val(cand, "selector_aocc_certified_pair_fraction"), val(cand, "selector_aocc_frontier_retained_weight_fraction")
    proposal, selected = val(cand, "proposal_decisive_atom_recall"), val(cand, "selected_decisive_atom_recall")
    effective, interaction = val(cand, "effective_selected_decisive_atom_recall"), val(cand, "selected_interaction_decisive_recall")
    fallback = val(cand, "fallback_would_trigger_rate")
    decision_atoms, configured = val(cand, "decision_budget_atom_count"), val(cand, "configured_decision_budget_atom_count")
    if not math.isfinite(decision_atoms): decision_atoms = val(cand, "selector_decision_budget_atom_count")
    if not math.isfinite(configured): configured = val(cand, "selector_budget")
    fill = decision_atoms / max(configured, 1e-9)
    calibrated, exact_target = val(cand, "selector_aocc_bound_calibrated"), val(cand, "selector_aocc_exact_tournament_target_active")
    latency = val(cand, "planner_latency_ms_p95")

    # Minimum-completeness gate: protects against catastrophic algorithm or
    # protocol failures while allowing paired CL20 to generate the evidence
    # needed to decide what to optimize next.
    checks = [
        (math.isfinite(cm) and cm >= max(lm, fm) - 0.01, f"teacher match {cm} < best control {max(lm, fm)} - 0.01"),
        (math.isfinite(cpair) and cpair >= clocal - 0.01, f"pair-full match {cpair} < local anchor {clocal} - 0.01"),
        (math.isfinite(drift) and drift <= 0.005, f"frozen anchor drift={drift} > 0.005"),
        (math.isfinite(harmful) and harmful <= 0.05, f"harmful residual rate={harmful} > 0.05"),
        (math.isfinite(beneficial) and math.isfinite(harmful) and beneficial + 0.01 >= harmful, f"residual strongly net harmful: {beneficial}/{harmful}"),
        (math.isfinite(cert) and cert >= 0.40, f"certified fraction={cert} < 0.40"),
        (math.isfinite(frontier) and frontier >= 0.45, f"frontier retained={frontier} < 0.45"),
        (math.isfinite(proposal) and proposal >= 0.72, f"proposal decisive recall={proposal} < 0.72"),
        (math.isfinite(selected) and selected >= 0.50, f"selected decisive recall={selected} < 0.50"),
        (math.isfinite(effective) and effective >= 0.62, f"effective decisive recall={effective} < 0.62"),
        (math.isfinite(interaction) and interaction >= 0.40, f"interaction decisive recall={interaction} < 0.40"),
        (math.isfinite(fallback) and fallback <= 0.60, f"fallback rate={fallback} > 0.60"),
        (math.isfinite(fill) and fill >= 0.95, f"budget fill={fill} < 0.95"),
        (math.isfinite(calibrated) and calibrated >= 0.99, f"independent calibration active={calibrated}"),
        (math.isfinite(exact_target) and exact_target >= 0.99, f"exact target active={exact_target}"),
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
    ]
    for ok, msg in competitive_checks:
        if not ok: comp_failures.append(msg)
    for label, stats in paired.items():
        if stats["candidate"]["median"] > stats["control"]["median"] + 1e-9 or stats["candidate"]["p90"] > stats["control"]["p90"] + 1e-9:
            comp_failures.append(f"paired regret regressed vs {label}: {stats['candidate']} / {stats['control']}")

    minimum_pass = not min_failures
    competitive_pass = minimum_pass and not comp_failures
    report = {
        "gate": "v53_wc_bfar",
        "minimum_pass": minimum_pass,
        "competitive_pass": competitive_pass,
        "minimum_failures": min_failures,
        "competitive_failures": comp_failures,
        "warnings": warnings,
        "metrics": {
            "teacher_match_candidate": cm, "teacher_match_local": lm, "teacher_match_foundation": fm,
            "pair_full_match": cpair, "local_anchor_match": clocal, "anchor_drift": drift,
            "harmful_residual_rate": harmful, "beneficial_residual_rate": beneficial,
            "certified_fraction": cert, "frontier_retained": frontier,
            "proposal_decisive_recall": proposal, "selected_decisive_recall": selected,
            "effective_decisive_recall": effective, "interaction_decisive_recall": interaction,
            "fallback_rate": fallback, "budget_fill": fill, "latency_p95_ms": latency,
        },
        "training": train_stats,
        "paired_regret": paired,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nV53 minimum-completeness gate [{'PASS' if minimum_pass else 'FAIL'}]")
    print(f"V53 competitive gate [{'PASS' if competitive_pass else 'FAIL'}]")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    for warning in warnings: print(f"  ! WARNING: {warning}")
    for failure in min_failures: print(f"  - MINIMUM: {failure}")
    for failure in comp_failures: print(f"  - COMPETITIVE: {failure}")
    return 0 if minimum_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
