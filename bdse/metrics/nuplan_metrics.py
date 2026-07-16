from __future__ import annotations

from pathlib import Path
from typing import Any

import json


def load_nuplan_metric_summary(metric_file: str | Path) -> dict[str, Any]:
    path = Path(metric_file)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("Use nuPlan's native metric runner for parquet/msgpack outputs, or export summary JSON for this helper.")


def required_closed_loop_metrics() -> list[str]:
    return [
        "overall_planning_score",
        "no_at_fault_collision",
        "drivable_area_compliance",
        "route_progress",
        "speed_limit_compliance",
        "time_to_collision",
        "comfort",
        "latency_ms",
    ]


def validate_metric_summary(summary: dict[str, Any]) -> None:
    missing = [k for k in required_closed_loop_metrics() if k not in summary]
    if missing:
        raise AssertionError(f"Missing nuPlan metrics: {missing}")
