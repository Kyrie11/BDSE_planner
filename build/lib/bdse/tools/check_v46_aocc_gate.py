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


def _index_rows(rows: list[dict[str, Any]], label: str) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if not key[0]:
            raise ValueError(f"{label} JSONL has an empty scenario_token")
        if key in out:
            raise ValueError(f"{label} JSONL has duplicate key {key}")
        out[key] = row
    return out


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"median": float("nan"), "p90": float("nan"), "p95": float("nan"), "cvar90": float("nan")}
    q90 = float(np.quantile(values, 0.90))
    return {
        "median": float(np.quantile(values, 0.50)),
        "p90": q90,
        "p95": float(np.quantile(values, 0.95)),
        "cvar90": float(values[values >= q90].mean()),
    }


def _paired_regret(candidate_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float], dict[str, float], int]:
    cand = _index_rows(candidate_rows, "candidate")
    ctrl = _index_rows(control_rows, "control")
    if set(cand) != set(ctrl):
        missing_ctrl = sorted(set(cand) - set(ctrl))[:5]
        missing_cand = sorted(set(ctrl) - set(cand))[:5]
        raise ValueError(
            "candidate/control JSONL scenario keys differ; "
            f"candidate_only={missing_ctrl}, control_only={missing_cand}, "
            f"counts=({len(cand)},{len(ctrl)})"
        )
    keys = sorted(cand)
    c = np.asarray([_finite(cand[k], "teacher_regret") for k in keys], dtype=np.float64)
    r = np.asarray([_finite(ctrl[k], "teacher_regret") for k in keys], dtype=np.float64)
    valid = np.isfinite(c) & np.isfinite(r)
    c, r = c[valid], r[valid]
    return _quantiles(c), _quantiles(r), _quantiles(c - r), int(valid.sum())


def _row_summary_check(summary: dict[str, Any], rows: list[dict[str, Any]], label: str, failures: list[str]) -> None:
    for key in ("teacher_action_match", "evidence_sufficiency"):
        vals = np.asarray([_finite(row, key) for row in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        reported = _finite(summary, key)
        if vals.size and math.isfinite(reported):
            observed = float(vals.mean())
            if abs(observed - reported) > 5e-4:
                failures.append(
                    f"{label} summary/JSONL mismatch for {key}: summary={reported:.6f}, rows={observed:.6f}"
                )


def _training_health(path: Path | None, min_exact_fraction: float) -> tuple[list[str], dict[str, float]]:
    failures: list[str] = []
    stats = {"epochs": 0.0, "min_exact_fraction": float("nan")}
    if path is None:
        return ["training log is required"], stats
    rows = _load_rows(path)
    stats["epochs"] = float(len(rows))
    exact: list[float] = []
    for row in rows:
        epoch = int(row.get("epoch", -1))
        for key, value in row.items():
            if key != "epoch" and isinstance(value, (int, float)) and not math.isfinite(float(value)):
                failures.append(f"non-finite train metric at epoch={epoch}: {key}={value}")
                break
        if epoch >= 1 and math.isfinite(_finite(row, "selector_exact_fraction")):
            exact.append(_finite(row, "selector_exact_fraction"))
    if not exact:
        failures.append("selector_exact_fraction missing")
    else:
        stats["min_exact_fraction"] = float(min(exact))
        if stats["min_exact_fraction"] + 1e-12 < min_exact_fraction:
            failures.append(
                f"selector_exact_fraction min={stats['min_exact_fraction']:.6f} < {min_exact_fraction:.6f}"
            )
    return failures, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict v46 AOCC open-loop gate with token-paired regret")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--control-jsonl", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--latency-target-ms", type=float, default=500.0)
    parser.add_argument("--min-match-gain", type=float, default=0.02)
    parser.add_argument("--min-sufficiency-gain", type=float, default=0.01)
    parser.add_argument("--min-sign-gain", type=float, default=0.0)
    parser.add_argument("--min-exact-fraction", type=float, default=0.99)
    parser.add_argument("--min-pair-full-match", type=float, default=0.30)
    parser.add_argument("--min-certified-pair-fraction", type=float, default=0.50)
    parser.add_argument("--require-independent-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-exact-aocc-target", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    candidate = _load_json(args.candidate)
    control = _load_json(args.control)
    cand_rows = _load_rows(args.candidate_jsonl)
    ctrl_rows = _load_rows(args.control_jsonl)
    failures, train_stats = _training_health(args.train_log, args.min_exact_fraction)
    _row_summary_check(candidate, cand_rows, "candidate", failures)
    _row_summary_check(control, ctrl_rows, "control", failures)

    def gain_gate(key: str, minimum: float, label: str) -> tuple[float, float]:
        c, r = _finite(candidate, key), _finite(control, key)
        if not math.isfinite(c) or not math.isfinite(r):
            failures.append(f"{key} missing")
        elif c < r + minimum:
            failures.append(f"{label} gain={c-r:+.6f} < +{minimum:.6f}")
        return c, r

    cand_match, ctrl_match = gain_gate("teacher_action_match", args.min_match_gain, "teacher_action_match")
    cand_sign, ctrl_sign = gain_gate("pair_sign_acc_winner_rival", args.min_sign_gain, "winner/rival sign")
    cand_suff, ctrl_suff = gain_gate("evidence_sufficiency", args.min_sufficiency_gain, "evidence_sufficiency")
    pair_full = _finite(candidate, "pair_full_interface_action_match")
    if not math.isfinite(pair_full):
        failures.append("pair_full_interface_action_match missing; rerun with v46 evaluator")
    elif pair_full < args.min_pair_full_match:
        failures.append(f"pair_full_interface_action_match={pair_full:.6f} < {args.min_pair_full_match:.6f}")
    certified_fraction = _finite(candidate, "selector_aocc_certified_pair_fraction")
    if not math.isfinite(certified_fraction) or certified_fraction < args.min_certified_pair_fraction:
        failures.append(
            f"selector_aocc_certified_pair_fraction={certified_fraction} < {args.min_certified_pair_fraction:.6f}"
        )
    calibrated_rate = _finite(candidate, "selector_aocc_bound_calibrated")
    if args.require_independent_calibration and (not math.isfinite(calibrated_rate) or calibrated_rate < 0.99):
        failures.append("independent AOCC calibration provenance is missing or not active")
    exact_target_rate = _finite(candidate, "selector_aocc_exact_tournament_target_active")
    if args.require_exact_aocc_target and (not math.isfinite(exact_target_rate) or exact_target_rate < 0.99):
        failures.append("AOCC target is not consistently sourced from the exact deployment tournament")

    latency = _finite(candidate, "planner_latency_ms_p95")
    if not math.isfinite(latency) or latency > args.latency_target_ms:
        failures.append(f"planner_latency_ms_p95={latency} > target={args.latency_target_ms:.3f}")

    try:
        cand_q, ctrl_q, delta_q, matched = _paired_regret(cand_rows, ctrl_rows)
        for key in ("median", "p90"):
            if cand_q[key] > ctrl_q[key] + 1e-9:
                failures.append(f"teacher regret {key} regressed: {cand_q[key]:.6f} > {ctrl_q[key]:.6f}")
    except ValueError as exc:
        failures.append(str(exc))
        cand_q = ctrl_q = delta_q = None
        matched = 0

    print(f"\nV47 D3CE/AOCC open-loop gate [{'PASS' if not failures else 'FAIL'}]")
    print(f"  teacher match: {cand_match} vs {ctrl_match} ({cand_match-ctrl_match:+.6f})")
    print(f"  winner/rival sign: {cand_sign} vs {ctrl_sign} ({cand_sign-ctrl_sign:+.6f})")
    print(f"  sufficiency: {cand_suff} vs {ctrl_suff} ({cand_suff-ctrl_suff:+.6f})")
    print(f"  pair-full interface match: {pair_full}")
    print(f"  certified pair fraction: {certified_fraction}; calibrated rate: {calibrated_rate}; exact target rate: {exact_target_rate}")
    print(f"  latency p95: {latency} ms; matched regret rows: {matched}")
    print(f"  training epochs: {train_stats['epochs']}; min exact fraction: {train_stats['min_exact_fraction']}")
    if cand_q is not None:
        print(f"  candidate regret: {cand_q}")
        print(f"  control regret:   {ctrl_q}")
        print(f"  paired delta:     {delta_q}")
    for failure in failures:
        print(f"  - {failure}")
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
