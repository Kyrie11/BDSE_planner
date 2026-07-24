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
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finite(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value if math.isfinite(value) else default


def _regret_quantiles(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = np.asarray(
        [_finite(row, "teacher_regret") for row in rows if math.isfinite(_finite(row, "teacher_regret"))],
        dtype=np.float64,
    )
    if values.size == 0:
        return {"median": float("nan"), "p90": float("nan"), "p95": float("nan"), "cvar90": float("nan")}
    q90 = float(np.quantile(values, 0.90))
    return {
        "median": float(np.quantile(values, 0.50)),
        "p90": q90,
        "p95": float(np.quantile(values, 0.95)),
        "cvar90": float(values[values >= q90].mean()),
    }


def _training_health(path: Path | None, min_exact_fraction: float) -> tuple[list[str], dict[str, float]]:
    failures: list[str] = []
    stats = {"epochs": 0.0, "min_exact_fraction": float("nan")}
    if path is None:
        failures.append("training log is required to verify finite optimization")
        return failures, stats
    rows = _load_rows(path)
    stats["epochs"] = float(len(rows))
    exact: list[float] = []
    for row in rows:
        epoch = int(row.get("epoch", -1))
        for key, value in row.items():
            if key == "epoch":
                continue
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                failures.append(f"non-finite train metric at epoch={epoch}: {key}={value}")
                break
        if epoch >= 1:
            value = _finite(row, "selector_exact_fraction")
            if math.isfinite(value):
                exact.append(value)
    if exact:
        stats["min_exact_fraction"] = float(min(exact))
        if stats["min_exact_fraction"] + 1e-12 < min_exact_fraction:
            failures.append(
                f"selector_exact_fraction min={stats['min_exact_fraction']:.6f} < {min_exact_fraction:.6f}"
            )
    else:
        failures.append("selector_exact_fraction missing after deployment supervision starts")
    return failures, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict v45 PB-RADS open-loop gate")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--candidate-jsonl", type=Path, default=None)
    parser.add_argument("--control-jsonl", type=Path, default=None)
    parser.add_argument("--train-log", type=Path, default=None)
    parser.add_argument("--latency-target-ms", type=float, required=True)
    parser.add_argument("--min-match-gain", type=float, default=0.02)
    parser.add_argument("--min-sufficiency-gain", type=float, default=0.01)
    parser.add_argument("--min-sign-gain", type=float, default=0.0)
    parser.add_argument("--min-exact-fraction", type=float, default=0.99)
    args = parser.parse_args()

    candidate = _load_json(args.candidate)
    control = _load_json(args.control)
    candidate_jsonl = args.candidate_jsonl or args.candidate.with_suffix(".jsonl")
    control_jsonl = args.control_jsonl or args.control.with_suffix(".jsonl")
    failures, train_stats = _training_health(args.train_log, args.min_exact_fraction)

    cand_match = _finite(candidate, "teacher_action_match")
    ctrl_match = _finite(control, "teacher_action_match")
    if not math.isfinite(cand_match) or not math.isfinite(ctrl_match):
        failures.append("teacher_action_match missing")
    elif cand_match < ctrl_match + args.min_match_gain:
        failures.append(
            f"teacher_action_match gain={cand_match - ctrl_match:+.6f} < +{args.min_match_gain:.6f}"
        )

    cand_sign = _finite(candidate, "pair_sign_acc_winner_rival")
    ctrl_sign = _finite(control, "pair_sign_acc_winner_rival")
    if not math.isfinite(cand_sign) or not math.isfinite(ctrl_sign):
        failures.append("pair_sign_acc_winner_rival missing")
    elif cand_sign < ctrl_sign + args.min_sign_gain:
        failures.append(
            f"winner/rival sign gain={cand_sign - ctrl_sign:+.6f} < +{args.min_sign_gain:.6f}"
        )

    cand_suff = _finite(candidate, "evidence_sufficiency")
    ctrl_suff = _finite(control, "evidence_sufficiency")
    if not math.isfinite(cand_suff) or not math.isfinite(ctrl_suff):
        failures.append("evidence_sufficiency missing")
    elif cand_suff < ctrl_suff + args.min_sufficiency_gain:
        failures.append(
            f"evidence_sufficiency gain={cand_suff - ctrl_suff:+.6f} < +{args.min_sufficiency_gain:.6f}"
        )

    latency = _finite(candidate, "planner_latency_ms_p95")
    if not math.isfinite(latency):
        failures.append("planner_latency_ms_p95 missing")
    elif latency > args.latency_target_ms:
        failures.append(f"planner_latency_ms_p95={latency:.3f} > target={args.latency_target_ms:.3f}")

    cand_q = ctrl_q = None
    if candidate_jsonl.exists() and control_jsonl.exists():
        cand_q = _regret_quantiles(_load_rows(candidate_jsonl))
        ctrl_q = _regret_quantiles(_load_rows(control_jsonl))
        for key in ("median", "p90"):
            if not math.isfinite(cand_q[key]) or not math.isfinite(ctrl_q[key]):
                failures.append(f"teacher regret {key} unavailable")
            elif cand_q[key] > ctrl_q[key] + 1e-9:
                failures.append(
                    f"teacher regret {key} regressed: {cand_q[key]:.6f} > {ctrl_q[key]:.6f}"
                )
    else:
        failures.append("candidate/control JSONL are required for median and p90 regret gate")

    print("\nV45 PB-RADS open-loop gate")
    print(f"[{'PASS' if not failures else 'FAIL'}] {args.candidate.name}")
    print(f"  teacher_action_match: candidate={cand_match} control={ctrl_match} gain={cand_match-ctrl_match:+.6f}")
    print(f"  winner/rival sign: candidate={cand_sign} control={ctrl_sign} gain={cand_sign-ctrl_sign:+.6f}")
    print(f"  evidence_sufficiency: candidate={cand_suff} control={ctrl_suff} gain={cand_suff-ctrl_suff:+.6f}")
    print(f"  planner_latency_ms_p95: {latency} (target {args.latency_target_ms})")
    print(f"  training epochs: {train_stats['epochs']}; min exact fraction: {train_stats['min_exact_fraction']}")
    if cand_q is not None and ctrl_q is not None:
        print(f"  regret quantiles candidate: {cand_q}")
        print(f"  regret quantiles control:   {ctrl_q}")
    if failures:
        for failure in failures:
            print(f"  - {failure}")
        return 3
    print("  open-loop gate passed; closed-loop CL20 is authorized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
