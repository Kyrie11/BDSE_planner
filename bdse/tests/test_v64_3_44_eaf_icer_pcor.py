import numpy as np
import pytest

from bdse.data.cache_schema import CandidateBank, EvidenceBank, LabelOnlyFuture, RuntimeFeatures, Sample
from bdse.planner.response_modes import build_response_modes
from bdse.planner.response_value_observables import (
    FUTURE_RESPONSE_OBSERVABLE_NAMES,
    PLAN_CONDITIONED_RESPONSE_ALL_NAMES,
    PLAN_CONDITIONED_RESPONSE_OBSERVABLE_NAMES,
    PLAN_RESPONSE_CONDITIONING_NAMES,
    RUNTIME_RESPONSE_MODE_NAMES,
    _plan_conditioned_mode_probabilities,
    runtime_plan_conditioned_response_observable_costs,
    runtime_plan_response_mode_features,
)
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.planner.value_observables import QUALITY_NAMES, VALUE_OBSERVABLE_NAMES, runtime_value_observable_costs
from bdse.tools.build_v64_3_44_behavior_supervision import behavior_supervision_example
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES
from bdse.tools.fit_v64_3_44_eaf_icer_pcor import _behavior_diag, _fit_behavior_posterior


def _cfg(*, pc=False):
    ic = {
        "instrument_value_observables": True,
        "instrument_future_response_observables": True,
        "instrument_plan_conditioned_response_observables": pc,
    }
    return {
        "candidate": {"step_s": 0.2, "horizon_s": 8.0},
        "teacher": {
            "robust_modes": {
                "logged": {"enabled": True, "prob": 0.35},
                "cv": {"enabled": True, "prob": 0.20},
                "ca": {"enabled": True, "prob": 0.10},
                "brake": {"enabled": True, "prob": 0.15},
                "yield": {"enabled": True, "prob": 0.10},
                "nonyield": {"enabled": True, "prob": 0.10},
            },
            "risk_aggregation": {"cvar_alpha": 0.9, "cvar_weight": 0.4},
        },
        "runtime_safety": {
            "flag_mode": "hard",
            "hard_check_horizon_s": 1.0,
            "soft_check_horizon_s": 2.0,
            "soft_agent_radius_m": 1.5,
            "risk_hard_agent_weight": 6.0,
            "risk_soft_agent_weight": 1.2,
            "risk_hard_ttc_weight": 2.5,
            "risk_soft_ttc_weight": 1.0,
        },
        "runtime": {"decisive_frontier_value": {"incumbent_contrastive_extremal_recovery": ic}},
    }


def _runtime():
    # A stopped agent lies beyond the V43 2 s risk horizon.  Candidate 0 reaches
    # it later in the full planning horizon; candidate 1 remains laterally apart.
    cur = np.array([[24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.8, 2.0, 1.0]], dtype=np.float32)
    return RuntimeFeatures(
        ego_history=np.zeros((1, 5), dtype=np.float32),
        agent_history=cur[:, None, :].copy(),
        agent_valid=np.ones(1, dtype=bool),
        current_agents=cur,
        traffic_lights=[],
        map_features={
            "route_centerline": np.array([[0.0, 0.0], [80.0, 0.0]], dtype=np.float32),
            "route_corridor_width": 4.0,
            "stop_lines": [],
            "speed_limit_mps": 20.0,
        },
        route_roadblock_ids=[],
        mission_goal=None,
    )


def _bank():
    T = 41
    t = np.arange(1, T + 1, dtype=np.float32) * 0.2
    tr = np.zeros((2, T, 5), dtype=np.float32)
    tr[0, :, 0] = 3.0 * t
    tr[0, :, 3] = 3.0
    tr[1, :, 0] = 3.0 * t
    tr[1, :, 1] = 6.0
    tr[1, :, 3] = 3.0
    tr[:, :, 4] = t[None, :]
    return CandidateBank(
        trajectories=tr,
        valid_mask=np.ones(2, dtype=bool),
        maneuver_ids=np.zeros(2, dtype=np.int64),
        theta=[{}, {}],
        dynamic_flags=[{}, {}],
        metadata=[{}, {}],
    )


def test_v44_ungated_occupancy_preserves_future_support_beyond_v43_gate():
    raw, names = runtime_plan_response_mode_features(_runtime(), _bank(), _cfg())
    assert names == PLAN_RESPONSE_CONDITIONING_NAMES
    assert raw.shape == (2, 10) and np.all(np.isfinite(raw))
    # With the short V43 check horizon, all gated mode costs can still be zero.
    assert np.max(np.abs(raw[0, :5])) < 1e-12
    # The full-horizon occupancy support remains non-zero and distinguishes plans.
    assert np.max(raw[0, 5:]) > 0.0
    assert float(np.mean(raw[0, 5:])) > float(np.mean(raw[1, 5:]))


def test_v44_plan_conditioned_posterior_is_candidate_specific_and_normalized():
    X = np.zeros((2, 10), dtype=np.float64)
    X[0, 0] = 2.0
    X[1, 5] = 2.0
    W = np.zeros((10, 5), dtype=np.float64)
    W[0, 0] = 3.0
    W[5, 1] = 3.0
    cfg = _cfg(pc=True)
    sc = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"].setdefault(
        "selection_conditioned_intervention_recovery", {}
    )
    sc["plan_conditioned_response_posterior"] = {
        "enabled": True,
        "feature_names": list(PLAN_RESPONSE_CONDITIONING_NAMES),
        "feature_scale": [1.0] * 10,
        "weights": W.tolist(),
        "bias": [0.0] * 5,
    }
    p = _plan_conditioned_mode_probabilities(X, cfg)
    assert p.shape == (2, 5)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert int(np.argmax(p[0])) == 0
    assert int(np.argmax(p[1])) == 1
    assert not np.allclose(p[0], p[1])


def test_v44_runtime_instrumentation_appends_without_changing_v43_prefix():
    x43, n43 = runtime_value_observable_costs(_runtime(), _bank(), _cfg(pc=False))
    x44, n44 = runtime_value_observable_costs(_runtime(), _bank(), _cfg(pc=True))
    assert n43 == VALUE_OBSERVABLE_NAMES + FUTURE_RESPONSE_OBSERVABLE_NAMES
    assert n44 == n43 + PLAN_CONDITIONED_RESPONSE_ALL_NAMES
    assert x44.shape[1] == len(n44)
    assert np.max(np.abs(x43 - x44[:, : len(n43)])) < 1e-12
    pc, pn = runtime_plan_conditioned_response_observable_costs(_runtime(), _bank(), _cfg(pc=True))
    assert pn == PLAN_CONDITIONED_RESPONSE_ALL_NAMES
    assert pc.shape == (2, len(pn)) and np.all(np.isfinite(pc))


def test_v44_behavior_supervision_uses_logged_future_only_for_runtime_mode_target():
    cfg = _cfg(pc=False)
    rt = _runtime()
    modes = {m.name: m for m in build_response_modes(rt, None, cfg)}
    assert set(RUNTIME_RESPONSE_MODE_NAMES).issubset(modes)
    cv = np.asarray(modes["cv"].agent_futures, dtype=np.float32)
    # Logged ego trajectory is only a conditioning plan; logged agent future makes
    # the nearest runtime-only behavior label exactly CV.
    bank = _bank()
    lab = LabelOnlyFuture(
        logged_ego=bank.trajectories[0].copy(),
        logged_agents=cv.copy(),
        agent_valid=np.ones(cv.shape[0], dtype=bool),
    )
    sample = Sample(
        scenario_token="synthetic-v44",
        timestamp_us=0,
        runtime=rt,
        label_future=lab,
        candidates=bank,
        evidence_bank=EvidenceBank(atoms=[], query_features=np.zeros((0, 1), dtype=np.float32), active_mask=np.zeros((0,), dtype=bool)),
        teacher=None,
        pairs=None,
    )
    ex = behavior_supervision_example(sample, cfg)
    assert ex is not None
    assert ex["target_mode"] == "cv"
    assert ex["supervision_source"] == "TRAIN_logged_agent_future_nearest_runtime_mode_only"
    assert "teacher" not in ex and "teacher_improvement" not in ex
    assert len(ex["conditioning_features"]) == 10


def test_v44_behavior_ridge_can_learn_nontrivial_mode_signal_without_value_labels():
    behavior = {}
    for i in range(300):
        x = np.zeros(10, dtype=np.float64)
        target = i % 2
        x[target * 5] = 2.0
        behavior[f"tok{i}"] = {"x": x, "target": target}
    fit = _fit_behavior_posterior(behavior, list(behavior)[:270])
    diag = _behavior_diag(behavior, list(behavior)[270:], fit)
    assert fit["lambda"] == pytest.approx(1.0)
    assert diag["accuracy"] > diag["majority_baseline_accuracy"] + 0.2


def test_v44_tournament_accepts_plan_conditioned_observable_without_reranking():
    raw_names = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    base_names = raw_names
    feature_names = [f"delta::{n}" for n in base_names] + ["delta::support_logit"]
    obs_names = VALUE_OBSERVABLE_NAMES + FUTURE_RESPONSE_OBSERVABLE_NAMES + PLAN_CONDITIONED_RESPONSE_ALL_NAMES
    sc = {
        "feature_names": feature_names,
        "feature_mean": [0.0] * 19,
        "feature_std": [1.0] * 19,
        "weights": [0.0] * 19,
        "bias": 0.0,
        "base_feature_names": base_names,
        "post_selection_value_enabled": True,
        "post_selection_value_mode": "endpoint_potential_quality_plan_conditioned_response",
        "post_selection_endpoint_feature_names": EPV_NAMES,
        "post_selection_endpoint_feature_scale": [1.0] * 38,
        "post_selection_endpoint_weights": [0.0] * 38,
        "post_selection_endpoint_bias": 0.0,
        "post_selection_observable_names": obs_names,
        "post_selection_quality_observable_names": QUALITY_NAMES,
        "post_selection_quality_observable_scale": [1.0] * 3,
        "post_selection_quality_observable_weights": [0.0] * 3,
        "post_selection_future_response_observable_name": "plan_conditioned_occupancy_mean_cost",
        "post_selection_future_response_scale": 1.0,
        "post_selection_future_response_weight": 1.0,
        "post_selection_selected_bias": 0.0,
    }
    raw = np.zeros((2, len(raw_names)), dtype=np.float64)
    sup = np.zeros(2, dtype=np.float64)
    X = np.zeros((2, 19), dtype=np.float64)
    mu = np.zeros(2, dtype=np.float64)
    obs = np.zeros((2, len(obs_names)), dtype=np.float64)
    ridx = obs_names.index("plan_conditioned_occupancy_mean_cost")
    obs[0, ridx] = 2.0
    obs[1, ridx] = 0.5
    v, feat, names = _icer_post_selection_value(
        1, mu, X, feature_names, sc,
        raw_feat=raw, raw_feature_names=raw_names, support_logits=sup, legacy_action=0,
        value_observable_matrix=obs, value_observable_names=obs_names,
    )
    assert v == pytest.approx(1.5)
    assert feat.shape[0] == 38 + 3 + 1
    assert names[-1].endswith("plan_conditioned_occupancy_mean_cost")
