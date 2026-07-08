from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


def _values(rows: list[dict], path: tuple[str, ...]) -> list[float]:
    out: list[float] = []
    for row in rows:
        cur = row
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if not ok:
            continue
        try:
            v = float(cur)
        except Exception:
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def _stats(xs: Iterable[float]) -> str:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return "n=0"
    n = len(vals)
    def q(p: float) -> float:
        return vals[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return (
        f"n={n} mean={statistics.mean(vals):.4f} p50={q(0.50):.4f} "
        f"p90={q(0.90):.4f} p95={q(0.95):.4f} p99={q(0.99):.4f} "
        f"max={vals[-1]:.4f} sum={sum(vals):.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize BDSE_CLOSED_LOOP_DIAG jsonl using the nested diagnostics.timing fields.")
    parser.add_argument("jsonl", type=str)
    parser.add_argument("--slowest", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.jsonl).expanduser()
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"ticks={len(rows)} file={path}")
    if not rows:
        return

    diag_rows = [r.get("diagnostics", {}) for r in rows]
    cached = sum(1 for d in diag_rows if bool(d.get("cached_plan", False)))
    fallback = collections.Counter(str(d.get("fallback_stage", "")) for d in diag_rows if not bool(d.get("cached_plan", False)))
    triggered = sum(1 for d in diag_rows if bool(d.get("fallback_triggered", False)))
    print(f"cached_plan={cached}/{len(rows)} ({cached / max(len(rows), 1):.1%})")
    print(f"fallback_triggered={triggered}/{len(rows)} ({triggered / max(len(rows), 1):.1%}); non_cached_denominator={len(rows) - cached}")
    print("fallback_stage_counts=", dict(fallback.most_common()))

    print("\n[timing: diagnostics.timing]")
    for key in ["compute_planner_trajectory_total_s", "runtime_from_planner_input_s", "core_plan_s", "to_nuplan_trajectory_s"]:
        print(f"{key}: {_stats(_values(diag_rows, ('timing', key)))}")

    print("\n[timing_core: diagnostics.timing_core]")
    core_keys = sorted({k for d in diag_rows for k in d.get("timing_core", {}).keys()})
    for key in core_keys:
        print(f"{key}: {_stats(_values(diag_rows, ('timing_core', key)))}")

    print("\n[model_timing: diagnostics.model_timing]")
    model_keys = sorted({k for d in diag_rows for k in d.get("model_timing", {}).keys()})
    for key in model_keys:
        print(f"{key}: {_stats(_values(diag_rows, ('model_timing', key)))}")

    print("\n[counts: non-cached rows only]")
    non_cached_diag_rows = [d for d in diag_rows if not bool(d.get("cached_plan", False))]
    for key in ["queried_action_count", "proposal_atom_count", "runtime_pair_count", "tournament_pair_count", "effective_query_count", "total_sparse_query_count"]:
        print(f"{key}: {_stats(_values(non_cached_diag_rows, (key,)))}")

    slow = []
    for i, row in enumerate(rows):
        d = row.get("diagnostics", {})
        t = d.get("timing", {}).get("compute_planner_trajectory_total_s")
        try:
            tval = float(t)
        except Exception:
            continue
        slow.append((tval, i, int(row.get("iteration_index", -1)), d.get("fallback_stage", ""), d.get("timing", {}), d.get("timing_core", {}), d.get("model_timing", {})))
    print(f"\n[slowest {args.slowest} ticks]")
    for tval, i, iteration, stage, timing, timing_core, model_timing in sorted(slow, reverse=True)[: max(0, args.slowest)]:
        print(f"idx={i} iter={iteration} total={tval:.4f} stage={stage} timing={timing} timing_core={timing_core} model_timing={model_timing}")


if __name__ == "__main__":
    main()
