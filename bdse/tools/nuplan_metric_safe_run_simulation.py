"""Thread-safe nuPlan simulation entrypoint for V64.3.50.5 PIOR.

This is an engineering-only wrapper around nuPlan's official
``nuplan.planning.script.run_simulation`` module.  nuPlan builds one MetricsEngine
per scenario type and reuses that stateful engine across simulations.  PIOR runs
scenarios with a thread-pool worker, so concurrent ``MetricCallback`` calls can
otherwise interleave stateful metrics (notably EgoLaneChangeStatistics and
metrics that depend on it).  We serialize the *entire* metric-engine callback
within this Python process while leaving simulation/planner execution parallel.

No planner, proposal, outcome definition, metric definition, or PIOR fitting
logic is changed.
"""

from __future__ import annotations

import functools
import os
import runpy
import threading
from types import ModuleType
from typing import Any, Callable

_METRIC_ENGINE_LOCK = threading.RLock()
_PATCH_MARKER = "_bdse_v64_3_50_5_metric_engine_serialized"


def install_metric_engine_serialization(metric_callback_module: ModuleType | Any | None = None) -> Callable[..., Any]:
    """Serialize nuPlan ``run_metric_engine`` calls in this process.

    The helper accepts a module-like object to make the concurrency contract
    unit-testable without a nuPlan installation.
    """
    if metric_callback_module is None:
        from nuplan.planning.simulation.callback import metric_callback as metric_callback_module  # type: ignore

    current = getattr(metric_callback_module, "run_metric_engine")
    if bool(getattr(current, _PATCH_MARKER, False)):
        return current

    original = current

    @functools.wraps(original)
    def serialized_run_metric_engine(*args: Any, **kwargs: Any) -> Any:
        with _METRIC_ENGINE_LOCK:
            return original(*args, **kwargs)

    setattr(serialized_run_metric_engine, _PATCH_MARKER, True)
    setattr(serialized_run_metric_engine, "_bdse_original_run_metric_engine", original)
    setattr(metric_callback_module, "run_metric_engine", serialized_run_metric_engine)
    return serialized_run_metric_engine


def main() -> None:
    if os.environ.get("BDSE_PIOR_METRIC_ENGINE_SERIALIZATION") != "1":
        raise RuntimeError(
            "V64.3.50.5 metric-safe nuPlan wrapper requires "
            "BDSE_PIOR_METRIC_ENGINE_SERIALIZATION=1"
        )
    install_metric_engine_serialization()
    print(
        "[BDSE-PIOR-METRIC-SAFE] V64.3.50.5 process-wide metric-engine "
        "serialization enabled; simulation/planner worker concurrency unchanged",
        flush=True,
    )
    runpy.run_module("nuplan.planning.script.run_simulation", run_name="__main__")


if __name__ == "__main__":
    main()
