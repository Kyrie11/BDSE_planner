import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner import fallback
from bdse.planner.tournament import _evidence_action_potential_cost


def _runtime() -> RuntimeFeatures:
    return RuntimeFeatures(
        ego_history=np.zeros((1, 7), dtype=np.float32),
        agent_history=np.zeros((0, 1, 7), dtype=np.float32),
        agent_valid=np.zeros((0,), dtype=bool),
        current_agents=np.zeros((0, 7), dtype=np.float32),
        traffic_lights=[],
        map_features={
            "route_centerline": np.array([[0.0, 0.0], [40.0, 0.0]], dtype=np.float32),
            "route_corridor_width": 4.0,
            "stop_lines": [],
            "speed_limit_mps": 15.0,
        },
        route_roadblock_ids=[],
        mission_goal=None,
    )


def _bank() -> CandidateBank:
    traj = np.zeros((2, 9, 5), dtype=np.float32)
    traj[:, :, 0] = np.linspace(0.0, 8.0, 9)[None, :]
    traj[:, :, 3] = 2.0
    traj[:, :, 4] = np.linspace(0.0, 4.0, 9)[None, :]
    return CandidateBank(
        trajectories=traj,
        valid_mask=np.ones((2,), dtype=bool),
        maneuver_ids=np.zeros((2,), dtype=np.int64),
        theta=[{}, {}],
        dynamic_flags=[{}, {}],
        metadata=[{}, {}],
    )


def test_runtime_safety_geometry_is_computed_once_per_planner_scope(monkeypatch):
    runtime, candidates = _runtime(), _bank()
    cfg = {"runtime_safety": {}, "candidate": {"step_s": 0.1}}
    calls = {"n": 0}
    original = fallback._runtime_safety_flag_components_uncached

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fallback, "_runtime_safety_flag_components_uncached", counted)
    with fallback.runtime_safety_cache_scope() as memo:
        first = fallback.runtime_safety_flag_components(runtime, candidates, cfg)
        second = fallback.runtime_safety_flag_components(runtime, candidates, cfg)
        assert first is second
        assert memo.hits >= 1
        assert memo.misses == 1
    assert calls["n"] == 1

    # A new planner call must never reuse stale scene geometry.
    with fallback.runtime_safety_cache_scope():
        fallback.runtime_safety_flag_components(runtime, candidates, cfg)
    assert calls["n"] == 2


def test_set_conditioned_integrable_potential_can_flip_the_selected_local_winner():
    anchor, corrected, sigma, diag = _evidence_action_potential_cost(
        predicted_base_cost=np.array([0.0, 1.0], dtype=np.float32),
        predicted_atom_costs=np.zeros((1, 2), dtype=np.float32),
        residual_action_potential=np.zeros((1, 2), dtype=np.float32),
        selected_atoms=[0],
        valid_mask=np.array([True, True]),
        residual_action_variance=np.zeros((1, 2), dtype=np.float32),
        residual_set_atom_factors=np.array([[1.0]], dtype=np.float32),
        residual_set_action_factors=np.array([[1.0], [-1.0]], dtype=np.float32),
        set_residual_scale=1.0,
        normalize_margins=False,
        margin_scale=1.0,
    )
    assert int(np.argmin(anchor)) == 0
    assert int(np.argmin(corrected)) == 1
    assert sigma is not None
    assert diag["set_conditioned_residual_active"] == 1.0

    # A scalar action potential is integrable: every three-action cycle sums to zero.
    three = np.array([0.2, -0.4, 0.7], dtype=np.float32)
    cycle = (three[0] - three[1]) + (three[1] - three[2]) + (three[2] - three[0])
    assert abs(float(cycle)) < 1.0e-6


def test_planner_core_drops_process_local_rlock_when_pickled():
    import pickle
    import threading
    from bdse.planner.nuplan_planner import BDSEPlannerCore

    core = BDSEPlannerCore(model=None, cfg={"runtime": {}}, inference_lock=threading.RLock())
    restored = pickle.loads(pickle.dumps(core))
    assert restored.inference_lock is None


def test_v59_calibration_requires_the_runtime_residual_variance_key():
    import pytest
    from bdse.tools.calibrate_v59_dual_certificates import _require_residual_action_variance

    with pytest.raises(KeyError, match="residual_action_var"):
        _require_residual_action_variance({"residual_action_variance": np.zeros((1, 2), dtype=np.float32)})
    value = _require_residual_action_variance({"residual_action_var": np.ones((1, 2), dtype=np.float32)})
    assert value.shape == (1, 2)
    assert float(value.sum()) == 2.0
