from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _finite(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _stats(values: list[float]) -> dict[str, float | int]:
    arr = _finite(values)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


def _match(row: dict[str, Any], key: str, action_key: str | None = None) -> bool | None:
    value = row.get(key)
    if isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
        return bool(round(float(value)))
    if action_key is not None and action_key in row and "teacher_action" in row:
        return int(row[action_key]) == int(row["teacher_action"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dense/pair-full/budget failure modes from v46 JSONL")
    parser.add_argument("jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    cross_dense_budget: Counter[str] = Counter()
    cross_pair_budget: Counter[str] = Counter()
    triple: Counter[str] = Counter()

    latency_keys = [
        "planner_latency_ms",
        "stage_predict_ms",
        "stage_selector_ms",
        "stage_tournament_ms",
        "stage_total_internal_ms",
    ]
    latency: dict[str, list[float]] = {key: [] for key in latency_keys}

    for row in rows:
        budget = _match(row, "teacher_action_match", "bdse_action")
        dense = _match(row, "full_interface_action_match", "full_action")
        pair_full = _match(row, "pair_full_interface_action_match", "pair_full_action")
        if dense is not None and budget is not None:
            cross_dense_budget[f"dense_{'correct' if dense else 'wrong'}__budget_{'correct' if budget else 'wrong'}"] += 1
        if pair_full is not None and budget is not None:
            cross_pair_budget[f"pair_full_{'correct' if pair_full else 'wrong'}__budget_{'correct' if budget else 'wrong'}"] += 1
        if dense is not None and pair_full is not None and budget is not None:
            triple[f"dense_{int(dense)}__pair_full_{int(pair_full)}__budget_{int(budget)}"] += 1
        for key in latency_keys:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                latency[key].append(float(value))

    output = {
        "jsonl": str(Path(args.jsonl)),
        "scene_count": len(rows),
        "dense_vs_budget": dict(sorted(cross_dense_budget.items())),
        "pair_full_vs_budget": dict(sorted(cross_pair_budget.items())),
        "dense_pair_full_budget": dict(sorted(triple.items())),
        "latency": {key: _stats(values) for key, values in latency.items()},
        "diagnostic_note": (
            "pair_full uses the deployment-identical tournament with full Top-M support; "
            "dense full may use a different action-conditioned aggregation and should not be treated as the final selector upper bound."
        ),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
