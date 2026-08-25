import numpy as np
import pytest

from bdse.data.cache_schema import CandidateBank, EvidenceAtom, EvidenceBank, RuntimeFeatures
from bdse.planner.response_modes import ResponseMode
from bdse.planner.response_value_observables import (
    RESPONSE_VALUE_OBSERVABLE_NAMES,
    runtime_selected_response_costs,
)
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.planner.value_observables import VALUE_OBSERVABLE_NAMES
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES
from bdse.tools.fit_v64_3_43_eaf_icer_crvr import ALL_OBSERVABLE_NAMES, _anchor_improvement


def _bank():
    T = 81
    t = np.arange(T, dtype=np.float32) * 0.1
    tr = np.zeros((2, T, 5), dtype=np.float32)
    tr[0, :, 0] = 4.0 * t
    tr[0, :, 3] = 4.0
    tr[1, :, 0] = 1.0 * t
    tr[1, :, 1] = 8.0
    tr[1, :, 3] = 1.0
    tr[:, :, 4] = t
    return CandidateBank(
        trajectories=tr,
        valid_mask=np.ones(2, dtype=bool),
        maneuver_ids=np.zeros(2, dtype=np.int64),
        theta=[{}, {}],
        dynamic_flags=[{}, {}],
        metadata=[{}, {}],
    )


def _runtime():
    cur = np.array([[12.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 4.8, 2.0, 1.0]], dtype=np.float32)
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


def _cfg():
    return {
        "candidate": {"step_s": 0.1, "horizon_s": 8.0},
        "teacher": {
            # Logged is deliberately enabled in config.  With label_future=None,
            # runtime response construction must omit it and renormalize only
            # current-state-generated modes.
            "robust_modes": {
                "logged": {"enabled": True, "prob": 0.50},
                "cv": {"enabled": True, "prob": 0.20},
                "ca": {"enabled": True, "prob": 0.05},
                "brake": {"enabled": True, "prob": 0.10},
                "yield": {"enabled": True, "prob": 0.10},
                "nonyield": {"enabled": True, "prob": 0.05},
            },
            "risk_aggregation": {"cvar_alpha": 0.9, "cvar_weight": 0.25},
            "feasibility": {"inject_hard_priority_costs": False},
        },
        "evidence": {
            "weights": {"ttc": 1.0},
            "scales": {"ttc": 1.0},
            "safety": {"tau_safe_s": 2.0},
        },
    }


def _ttc_bank():
    cur = _runtime().current_agents[0]
    atom = EvidenceAtom(
        atom_id=0,
        type="ttc",
        anchor={"agent_index": 0, "current_state": cur.copy(), "length": 4.8, "width": 2.0},
        budget_cost=1.0,
        is_hard=False,
        family="interaction",
        active_mask=True,
    )
    return EvidenceBank(
        atoms=[atom],
        query_features=np.zeros((1, 18), dtype=np.float32),
        active_mask=np.ones(1, dtype=bool),
    )


def test_v43_response_observables_are_runtime_only_selected_evidence_costs():
    x, names = runtime_selected_response_costs(_runtime(), _bank(), _ttc_bank(), [0], _cfg())
    assert names == RESPONSE_VALUE_OBSERVABLE_NAMES
    assert x.shape == (2, 3)
    assert np.all(np.isfinite(x))
    # CV/mean/robust are genuinely different counterfactual functionals in this
    # synthetic interaction, rather than three aliases of a current-state score.
    assert not np.allclose(x[:, 0], x[:, 1])
    assert not np.allclose(x[:, 1], x[:, 2])
    # No selected evidence means exactly zero added consequence cost.
    z, zn = runtime_selected_response_costs(_runtime(), _bank(), _ttc_bank(), [], _cfg())
    assert zn == names
    assert np.array_equal(z, np.zeros_like(z))


def test_v43_response_observable_fails_closed_on_invalid_selection_or_logged_future(monkeypatch):
    with pytest.raises(ValueError, match="out-of-range"):
        runtime_selected_response_costs(_runtime(), _bank(), _ttc_bank(), [1], _cfg())
    with pytest.raises(ValueError, match="unique"):
        runtime_selected_response_costs(_runtime(), _bank(), _ttc_bank(), [0, 0], _cfg())

    import bdse.planner.response_value_observables as mod

    T = _bank().T
    leaked = ResponseMode(
        name="logged",
        probability=1.0,
        agent_futures=np.zeros((1, T, 5), dtype=np.float32),
        metadata={"uses_label_future": True},
    )
    monkeypatch.setattr(mod, "build_response_modes", lambda runtime, label_future, cfg: [leaked])
    with pytest.raises(ValueError, match="never consume"):
        mod.runtime_selected_response_costs(_runtime(), _bank(), _ttc_bank(), [0], _cfg())


def _value_cfg(mode):
    base = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    feature_names = [f"delta::{n}" for n in base] + ["delta::support_logit"]
    return {
        "feature_names": feature_names,
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
        "post_selection_observable_names": ALL_OBSERVABLE_NAMES,
        "post_selection_observable_quality_dim": 3,
    }


def test_v43_runtime_uses_fixed_physical_anchor_and_same_frozen_winner():
    raw_names = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    raw_feat = np.zeros((2, len(raw_names)), dtype=float)
    support = np.zeros(2, dtype=float)
    X = np.zeros((2, 19), dtype=float)
    mu = np.zeros(2, dtype=float)
    obs = np.zeros((2, len(ALL_OBSERVABLE_NAMES)), dtype=float)
    # lower-is-better: incumbent(0)-candidate(1) = +1 for each quality term.
    obs[0, :3] = [2.0, 2.0, 2.0]
    obs[1, :3] = [1.0, 1.0, 1.0]
    # CV response contribution improves by +4.  Other response columns must not
    # enter the CV arm.
    off = len(VALUE_OBSERVABLE_NAMES)
    obs[0, off:] = [5.0, 40.0, 70.0]
    obs[1, off:] = [1.0, 2.0, 3.0]
    c = _value_cfg("endpoint_residual_quality_cv_evidence_anchor")
    v, feat, names = _icer_post_selection_value(
        1,
        mu,
        X,
        c["feature_names"],
        c,
        raw_feat=raw_feat,
        raw_feature_names=raw_names,
        support_logits=support,
        legacy_action=0,
        value_observable_matrix=obs,
        value_observable_names=ALL_OBSERVABLE_NAMES,
    )
    assert v == pytest.approx(7.0)  # three exact quality deltas + exact CV delta
    assert feat.shape[0] == 38 + 4
    assert names[-1] == "analytic_anchor::selected_evidence_cv_cost"

    v2, _, _ = _icer_post_selection_value(
        1,
        mu,
        X,
        c["feature_names"],
        c,
        raw_feat=raw_feat,
        raw_feature_names=raw_names,
        support_logits=support,
        legacy_action=0,
        value_observable_matrix=obs,
        value_observable_names=ALL_OBSERVABLE_NAMES,
        value_target_scale=2.0,
    )
    assert v2 == pytest.approx(3.5)


def test_v43_training_anchor_coefficients_are_analytic_not_learned():
    a = {
        "observable_improvement": np.array([1.0, -2.0, 3.0] + [100.0] * 6),
        "response_observable_improvement": np.array([4.0, 5.0, 6.0]),
    }
    assert _anchor_improvement(a, "q_anchor") == pytest.approx(2.0)
    assert _anchor_improvement(a, "cv_anchor") == pytest.approx(6.0)
    assert _anchor_improvement(a, "mean_anchor") == pytest.approx(7.0)
    assert _anchor_improvement(a, "robust_anchor") == pytest.approx(8.0)
