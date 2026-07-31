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


def _paired_regret(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], a_label: str, b_label: str) -> tuple[dict[str, float], dict[str, float], dict[str, float], int]:
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

    exact: list[float] = []
    pair_fraction: list[float] = []
    critical_prefixes = ("L_", "train_")
    critical_names = {"loss", "selector_exact_fraction", "training_pair_fraction"}
    for row in rows:
        for key, value in row.items():
            if key == "epoch" or not isinstance(value, (int, float)):
                continue
            if key in critical_names or key.startswith(critical_prefixes):
                if not math.isfinite(float(value)):
                    failures.append(f"non-finite critical training metric epoch={row.get('epoch')}: {key}={value}")
                    break
        value = _finite(row, "selector_exact_fraction")
        if math.isfinite(value):
            exact.append(value)
        value = _finite(row, "training_pair_fraction")
        if math.isfinite(value):
            pair_fraction.append(value)

    last_exact = exact[-1] if exact else float("nan")
    max_exact = max(exact) if exact else float("nan")
    if not math.isfinite(last_exact) or last_exact < min_last_exact:
        failures.append(f"last selector_exact_fraction={last_exact} < {min_last_exact}")
    if not exact or max_exact <= 0.0:
        failures.append("no exact full-graph selector supervision was observed")
    return failures, {
        "rows": float(len(rows)),
        "unique_epochs": float(len(set(epochs))),
        "last_exact_fraction": float(last_exact),
        "max_exact_fraction": float(max_exact),
        "mean_training_pair_fraction": float(np.mean(pair_fraction)) if pair_fraction else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Three-way V52 BFAR-DBAP causal/open-loop gate")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("local_control", type=Path, help="same candidate checkpoint with residual intervention disabled")
    parser.add_argument("foundation_control", type=Path, help="immutable foundation checkpoint with matched runtime")
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--local-control-jsonl", type=Path, required=True)
    parser.add_argument("--foundation-control-jsonl", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--min-total-match-gain", type=float, default=0.015)
    parser.add_argument("--min-residual-match-gain", type=float, default=0.005)
    parser.add_argument("--min-sufficiency-gain", type=float, default=0.01)
    parser.add_argument("--min-sign-gain", type=float, default=0.0)
    parser.add_argument("--min-near-tie-gain", type=float, default=0.0)
    parser.add_argument("--max-local-anchor-drift", type=float, default=0.005)
    parser.add_argument("--max-harmful-residual-rate", type=float, default=0.03)
    parser.add_argument("--min-certified-pair-fraction", type=float, default=0.55)
    parser.add_argument("--min-frontier-retained-weight", type=float, default=0.55)
    parser.add_argument("--max-interaction-budget-fraction", type=float, default=0.85)
    parser.add_argument("--min-budget-fill-fraction", type=float, default=0.95)
    parser.add_argument("--max-fallback-rate", type=float, default=0.40)
    parser.add_argument("--min-proposal-decisive-recall", type=float, default=0.80)
    parser.add_argument("--min-selected-decisive-recall", type=float, default=0.55)
    parser.add_argument("--min-effective-decisive-recall", type=float, default=0.70)
    parser.add_argument("--min-selected-interaction-decisive-recall", type=float, default=0.50)
    parser.add_argument("--latency-target-ms", type=float, default=500.0)
    parser.add_argument(
        "--enforce-latency",
        action="store_true",
        help="Make latency a hard prerequisite for closed-loop. By default latency is reported separately so algorithm quality can still be evaluated in simulation.",
    )
    parser.add_argument("--min-last-exact-fraction", type=float, default=0.03)
    args = parser.parse_args()

    cand = _load_json(args.candidate)
    local = _load_json(args.local_control)
    foundation = _load_json(args.foundation_control)
    cand_rows = _load_rows(args.candidate_jsonl)
    local_rows = _load_rows(args.local_control_jsonl)
    foundation_rows = _load_rows(args.foundation_control_jsonl)
    failures, train_stats = _training_health(args.train_log, args.min_last_exact_fraction)
    warnings: list[str] = []

    def gain(a: dict[str, Any], b: dict[str, Any], key: str, threshold: float, label: str) -> tuple[float, float]:
        av, bv = _finite(a, key), _finite(b, key)
        if not (math.isfinite(av) and math.isfinite(bv)):
            failures.append(f"{label}: {key} missing")
        elif av < bv + threshold:
            failures.append(f"{label} gain={av-bv:+.6f} < {threshold:+.6f}")
        return av, bv

    cand_match, foundation_match = gain(cand, foundation, "teacher_action_match", args.min_total_match_gain, "total teacher match")
    _, local_match = gain(cand, local, "teacher_action_match", args.min_residual_match_gain, "residual-only teacher match")
    cand_suff, foundation_suff = gain(cand, foundation, "evidence_sufficiency", args.min_sufficiency_gain, "total sufficiency")
    cand_win, foundation_win = gain(cand, foundation, "pair_sign_acc_winner_rival", args.min_sign_gain, "winner/rival sign")
    cand_near, foundation_near = gain(cand, foundation, "pair_sign_acc_near_tie", args.min_near_tie_gain, "near-tie sign")

    # Because the base/local modules are frozen and the local control uses the
    # same checkpoint, the two local-interface diagnostics should be invariant.
    cand_local = _finite(cand, "local_pair_full_interface_action_match")
    local_pair = _finite(local, "pair_full_interface_action_match")
    anchor_drift = abs(cand_local - local_pair) if math.isfinite(cand_local) and math.isfinite(local_pair) else float("nan")
    if not math.isfinite(anchor_drift) or anchor_drift > args.max_local_anchor_drift:
        failures.append(f"frozen local-anchor drift={anchor_drift} > {args.max_local_anchor_drift:.6f}")

    pair_full = _finite(cand, "pair_full_interface_action_match")
    harmful = _finite(cand, "harmful_residual_intervention_rate")
    beneficial = _finite(cand, "beneficial_residual_intervention_rate", 0.0)
    if not math.isfinite(pair_full) or pair_full < cand_local + args.min_residual_match_gain:
        failures.append(f"pair-full residual gain={pair_full-cand_local:+.6f} < {args.min_residual_match_gain:+.6f}")
    if not math.isfinite(harmful) or harmful > args.max_harmful_residual_rate:
        failures.append(f"harmful_residual_intervention_rate={harmful} > {args.max_harmful_residual_rate}")
    if math.isfinite(harmful) and math.isfinite(beneficial) and beneficial <= harmful:
        failures.append(f"residual is not net beneficial: beneficial={beneficial:.6f}, harmful={harmful:.6f}")

    certified = _finite(cand, "selector_aocc_certified_pair_fraction")
    frontier = _finite(cand, "selector_aocc_frontier_retained_weight_fraction")
    decision_atoms = _finite(cand, "decision_budget_atom_count")
    configured = _finite(cand, "configured_decision_budget_atom_count")
    interaction = _finite(cand, "selector_interaction_family_selected")
    fill = decision_atoms / max(configured, 1e-9)
    interaction_frac = interaction / max(decision_atoms, 1e-9)
    fallback = _finite(cand, "fallback_would_trigger_rate")
    proposal_decisive = _finite(cand, "proposal_decisive_atom_recall")
    selected_decisive = _finite(cand, "selected_decisive_atom_recall")
    effective_decisive = _finite(cand, "effective_selected_decisive_atom_recall")
    selected_interaction = _finite(cand, "selected_interaction_decisive_recall")
    calibrated = _finite(cand, "selector_aocc_bound_calibrated")
    exact_target = _finite(cand, "selector_aocc_exact_tournament_target_active")
    latency = _finite(cand, "planner_latency_ms_p95")
    for ok, msg in [
        (math.isfinite(certified) and certified >= args.min_certified_pair_fraction, f"certified pair fraction={certified}"),
        (math.isfinite(frontier) and frontier >= args.min_frontier_retained_weight, f"frontier retained weight={frontier}"),
        (math.isfinite(fill) and fill >= args.min_budget_fill_fraction, f"fixed-budget fill={fill}"),
        (math.isfinite(interaction_frac) and interaction_frac <= args.max_interaction_budget_fraction, f"interaction fraction={interaction_frac}"),
        (math.isfinite(fallback) and fallback <= args.max_fallback_rate, f"fallback rate={fallback}"),
        (math.isfinite(proposal_decisive) and proposal_decisive >= args.min_proposal_decisive_recall, f"proposal decisive recall={proposal_decisive}"),
        (math.isfinite(selected_decisive) and selected_decisive >= args.min_selected_decisive_recall, f"selected decisive recall={selected_decisive}"),
        (math.isfinite(effective_decisive) and effective_decisive >= args.min_effective_decisive_recall, f"effective decisive recall={effective_decisive}"),
        (math.isfinite(selected_interaction) and selected_interaction >= args.min_selected_interaction_decisive_recall, f"selected interaction decisive recall={selected_interaction}"),
        (math.isfinite(calibrated) and calibrated >= 0.99, f"independent calibration active={calibrated}"),
        (math.isfinite(exact_target) and exact_target >= 0.99, f"exact AOCC target active={exact_target}"),
    ]:
        if not ok:
            failures.append(msg)

    latency_ok = math.isfinite(latency) and latency <= args.latency_target_ms
    if not latency_ok:
        message = f"latency p95={latency} ms exceeds deployment target={args.latency_target_ms} ms"
        if args.enforce_latency:
            failures.append(message)
        else:
            warnings.append(message + "; closed-loop algorithm evaluation remains allowed, but no real-time claim is permitted")

    paired = {}
    for label, rows in [("local", local_rows), ("foundation", foundation_rows)]:
        try:
            cq, bq, dq, n = _paired_regret(cand_rows, rows, "candidate", label)
            paired[label] = {"candidate": cq, "control": bq, "delta": dq, "n": n}
            if cq["median"] > bq["median"] + 1e-9 or cq["p90"] > bq["p90"] + 1e-9:
                failures.append(f"paired regret regressed vs {label}: candidate={cq}, control={bq}")
        except ValueError as exc:
            failures.append(str(exc))

    print(f"\nV52 BFAR-DBAP gate [{'PASS' if not failures else 'FAIL'}]")
    print(f"  teacher match candidate/local/foundation: {cand_match} / {local_match} / {foundation_match}")
    print(f"  winner sign candidate/foundation: {cand_win} / {foundation_win}")
    print(f"  near-tie sign candidate/foundation: {cand_near} / {foundation_near}")
    print(f"  sufficiency candidate/foundation: {cand_suff} / {foundation_suff}")
    print(f"  local anchor candidate/local-control: {cand_local} / {local_pair}; drift={anchor_drift}")
    print(f"  residual pair-full/local: {pair_full} / {cand_local}; harmful/beneficial={harmful}/{beneficial}")
    print(f"  AOCC certified={certified}, frontier={frontier}, fill={fill}, interaction={interaction_frac}")
    print(f"  decisive evidence proposal/selected/effective/interaction={proposal_decisive}/{selected_decisive}/{effective_decisive}/{selected_interaction}")
    print(f"  latency p95={latency} ms; train={train_stats}")
    for label, value in paired.items():
        print(f"  paired regret vs {label}: {value}")
    for warning in warnings:
        print(f"  ! WARNING: {warning}")
    for failure in failures:
        print(f"  - {failure}")
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
