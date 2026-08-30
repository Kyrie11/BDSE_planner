import numpy as np
import pytest

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.planner.value_observables import QUALITY_DIM, VALUE_OBSERVABLE_NAMES, runtime_value_observable_costs
from bdse.tools.fit_v64_3_42_eaf_icer_ovdr import _fit_observable_residual
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES


def _bank():
    T = 41
    t = np.linspace(0, 4, T, dtype=np.float32)
    tr = np.zeros((2, T, 5), dtype=np.float32)
    tr[0, :, 0] = 4.0 * t
    tr[0, :, 3] = 4.0
    tr[1, :, 0] = 3.5 * t
    tr[1, :, 1] = 0.8
    tr[1, :, 3] = 3.5
    tr[:, :, 4] = t
    return CandidateBank(
        trajectories=tr,
        valid_mask=np.ones(2, dtype=bool),
        maneuver_ids=np.zeros(2, dtype=np.int64),
        theta=[{}, {}], dynamic_flags=[{}, {}], metadata=[{}, {}],
    )


def _runtime():
    cur = np.array([[12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.8, 2.0, 1.0]], dtype=np.float32)
    return RuntimeFeatures(
        ego_history=np.zeros((1, 5), dtype=np.float32),
        agent_history=cur[:, None, :].copy(), agent_valid=np.ones(1, dtype=bool), current_agents=cur,
        traffic_lights=[],
        map_features={"route_centerline": np.array([[0.0, 0.0], [80.0, 0.0]], dtype=np.float32), "route_corridor_width": 4.0, "stop_lines": [], "speed_limit_mps": 20.0},
        route_roadblock_ids=[], mission_goal=None,
    )


def _cfg():
    return {"candidate": {"step_s": 0.1}, "teacher": {}, "runtime_safety": {"flag_mode": "hard"}}


def test_value_observables_are_finite_deployment_only_costs():
    x, names = runtime_value_observable_costs(_runtime(), _bank(), _cfg())
    assert names == VALUE_OBSERVABLE_NAMES
    assert x.shape == (2, len(names))
    assert np.all(np.isfinite(x))
    # Candidate 1 is laterally offset, so teacher-aligned route cost is worse.
    assert x[1, 0] > x[0, 0]


def _runtime_value_cfg(mode):
    base = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    fn = [f"delta::{n}" for n in base] + ["delta::support_logit"]
    return {
        "feature_names": fn,
        "feature_mean": [0.0] * 19,
        "feature_std": [1.0] * 19,
        "weights": [0.0] * 19,
        "bias": 0.0,
        "post_selection_value_enabled": True,
        "post_selection_value_mode": mode,
        "post_selection_endpoint_feature_names": EPV_NAMES,
        "post_selection_endpoint_feature_scale": [1.0] * 38,
        "post_selection_endpoint_weights": [0.0] * 38,
        "post_selection_endpoint_bias": 0.0,
        "post_selection_observable_names": VALUE_OBSERVABLE_NAMES,
        "post_selection_observable_quality_dim": QUALITY_DIM,
        "post_selection_observable_scale": [1.0] * (QUALITY_DIM if mode.endswith("quality_observable") else len(VALUE_OBSERVABLE_NAMES) - QUALITY_DIM if mode.endswith("risk_observable") else len(VALUE_OBSERVABLE_NAMES)),
        "post_selection_observable_weights": [1.0] * (QUALITY_DIM if mode.endswith("quality_observable") else len(VALUE_OBSERVABLE_NAMES) - QUALITY_DIM if mode.endswith("risk_observable") else len(VALUE_OBSERVABLE_NAMES)),
        "post_selection_selected_bias": 0.0,
    }


def test_v42_runtime_observable_value_is_incumbent_minus_candidate_cost_and_never_reranks():
    raw_names = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    raw_feat = np.zeros((2, len(raw_names)), dtype=float)
    support = np.zeros(2, dtype=float)
    X = np.zeros((2, 19), dtype=float)
    mu = np.zeros(2, dtype=float)
    obs = np.zeros((2, len(VALUE_OBSERVABLE_NAMES)), dtype=float)
    obs[0, :QUALITY_DIM] = 2.0
    obs[1, :QUALITY_DIM] = 1.0
    c = _runtime_value_cfg("endpoint_potential_quality_observable")
    v, feat, names = _icer_post_selection_value(
        1, mu, X, c["feature_names"], c,
        raw_feat=raw_feat, raw_feature_names=raw_names, support_logits=support, legacy_action=0,
        value_observable_matrix=obs, value_observable_names=VALUE_OBSERVABLE_NAMES,
    )
    assert v == pytest.approx(float(QUALITY_DIM))
    assert feat.shape[0] == 38 + QUALITY_DIM
    assert any("observable_improvement" in n for n in names)


def _a(obs, y):
    q = np.zeros(19)
    return {"q_inc": q, "q_cand": q, "delta_endpoint": q, "x": q, "y": y, "action": 1, "support": 1.0, "margin": 1.0, "utility_prior": 0, "observable_inc": np.zeros(9), "observable_cand": -np.asarray(obs, dtype=float), "observable_improvement": np.asarray(obs, dtype=float)}


def test_scene_equal_observable_residual_is_invariant_to_uniform_candidate_duplication():
    epv = {"mode": "epv", "names": EPV_NAMES, "scale": np.ones(38), "weights": np.zeros(38), "bias": 0.0}
    a1 = _a([1, 0, 0, 0, 0, 0, 0, 0, 0], 1.0)
    a2 = _a([-1, 0, 0, 0, 0, 0, 0, 0, 0], -1.0)
    b = _a([0, 1, 0, 0, 0, 0, 0, 0, 0], 0.5)
    m1 = _fit_observable_residual({"A": [a1, a2], "B": [b]}, ["A", "B"], epv, "quality")
    m2 = _fit_observable_residual({"A": [dict(a1), dict(a1), dict(a2), dict(a2)], "B": [b]}, ["A", "B"], epv, "quality")
    assert np.max(np.abs(m1["scale"] - m2["scale"])) < 1e-12
    assert np.max(np.abs(m1["weights"] - m2["weights"])) < 1e-10
