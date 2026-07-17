from __future__ import annotations

import inspect

from bdse.planner.nuplan_planner import BDSEnuPlanPlanner


def test_runtime_label_separation_signature():
    sig = inspect.signature(BDSEnuPlanPlanner.compute_trajectory)
    assert "label_future" not in sig.parameters
    assert "future_agents" not in sig.parameters
    assert "logged_future" not in sig.parameters
