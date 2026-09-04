from __future__ import annotations

import numpy as np
import pytest

from bdse.planner.paired_outcome_dominance_retention import (
    HODR_STATE_NAMES,
    operator_state,
    runtime_certificate,
)
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.planner.future_state_factorization import FSFR_OBSERVABLE_NAMES
from bdse.planner.value_observables import VALUE_OBSERVABLE_NAMES
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES
from bdse.tools.fit_v64_3_52_eaf_icer_hodr import (
    _effect_support,
    _fit_models,
    _pareto_pairs,
    _risk,
    _safety_names,
    _state,
)


def _row(i: int, *, effect: bool, beneficial: bool, score: float, safety: float, d: float) -> dict:
    return {
        "scenario_token": f"t{i}",
        "quality_value": 0.1 * i,
        "plan_control_value": 0.1 * i + 0.2,
        "ego_ref_value": 0.1 * i + 0.1,
        "operator_execution_contrast_linf": d,
        "closed_loop_beneficial": beneficial,
        "closed_loop_hard_harm": safety < 0.0,
        "closed_loop_score_delta": score if effect else 0.0,
        "safety_delta": {"collision": safety if effect else 0.0, "ttc": 0.0, "drivable": 0.0},
        "rsm_selected_teacher_improvement": 1.0 if beneficial else -1.0,
    }


def _component(names, w):
    return {
        "model": "unit",
        "feature_names": list(names),
        "feature_mean": [0.0] * len(names),
        "feature_std": [1.0] * len(names),
        "weights": list(w),
        "bias": 0.0,
        "lambda": 1.0,
        "fit_beneficial_score_mean": 0.0,
        "fit_beneficial_score_std": 1.0,
    }


def test_v52_state_is_exact_v51_additive_operator_state():
    p = np.zeros((2, 5)); i = np.zeros((2, 5)); p[1, 1] = 3.0
    z = operator_state(1.0, 2.5, 2.0, p, i)
    assert HODR_STATE_NAMES == ["quality_value", "prospective_response_increment", "ego_reference_increment", "operator_execution_contrast_linf"]
    assert z.tolist() == pytest.approx([1.0, 1.5, -0.5, 3.0])


def test_v52_effect_support_is_structural_not_beneficial_sign():
    null = _row(0, effect=False, beneficial=False, score=0.0, safety=0.0, d=0.0)
    harmful = _row(1, effect=True, beneficial=False, score=-0.5, safety=-1.0, d=2.0)
    assert _effect_support(null) is False
    assert _effect_support(harmful) is True


def test_v52_pareto_pairs_use_unweighted_score_and_safety_dominance():
    rows = [
        _row(0, effect=True, beneficial=False, score=-1.0, safety=-1.0, d=3.0),
        _row(1, effect=True, beneficial=True, score=+1.0, safety=0.0, d=2.0),
        _row(2, effect=True, beneficial=False, score=+2.0, safety=-1.0, d=4.0),  # trade-off: higher score, worse safety
    ]
    names = _safety_names(rows)
    pairs = _pareto_pairs(rows, names)
    assert (0, 1) in pairs
    assert (2, 1) not in pairs and (1, 2) not in pairs  # ambiguous trade-off is omitted, not scalarized


def test_v52_hurdle_models_can_separate_null_and_conditional_badness():
    rows = []
    for i in range(40):
        rows.append(_row(i, effect=False, beneficial=False, score=0.0, safety=0.0, d=0.05 + 0.001 * i))
    for i in range(40, 80):
        rows.append(_row(i, effect=True, beneficial=True, score=1.0, safety=0.0, d=1.0 + 0.01 * i))
    for i in range(80, 120):
        rows.append(_row(i, effect=True, beneficial=False, score=-1.0, safety=-1.0, d=3.0 + 0.01 * i))
    models = _fit_models(rows, "hurdle_sign", _safety_names(rows))
    assert models["effect_support_risk"]["lambda"] == 1.0
    assert models["conditional_outcome_risk"]["lambda"] == 1.0
    null_risk = _risk(rows[0], models)[0]
    good_risk = _risk(rows[50], models)[0]
    bad_risk = _risk(rows[-1], models)[0]
    assert null_risk > good_risk
    assert bad_risk > good_risk


def test_v52_runtime_uses_max_of_two_components_and_one_threshold():
    cfg = {
        "functional": "hurdle_pareto",
        "aggregation": "max_support_outcome",
        "components": {
            "effect_support_risk": _component(HODR_STATE_NAMES, [0, 0, 0, 1]),
            "conditional_outcome_risk": _component(HODR_STATE_NAMES, [1, 0, 0, 0]),
        },
        "retention_threshold": 2.5,
    }
    z = np.asarray([3.0, 0.0, 0.0, 2.0])
    cert, risk, parts = runtime_certificate(z, cfg)
    assert parts["effect_support_risk"] == pytest.approx(2.0)
    assert parts["conditional_outcome_risk"] == pytest.approx(3.0)
    assert risk == pytest.approx(3.0)
    assert cert == pytest.approx(-0.5)


def test_v52_tournament_keeps_frozen_winner_and_computes_only_retention_certificate():
    raw_names = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    feature_names = [f"delta::{n}" for n in raw_names] + ["delta::support_logit"]
    obs_names = VALUE_OBSERVABLE_NAMES + FSFR_OBSERVABLE_NAMES
    pnames = ["fsfr_plan_1d_occupancy_cost"]
    enames = ["fsfr_plan_1d_occupancy_cost", "fsfr_predicted_demo_cost"]
    support = _component(HODR_STATE_NAMES, [0.0, 0.0, 0.0, 0.0])
    outcome = _component(HODR_STATE_NAMES, [0.0, 0.0, 0.0, 1.0])
    sc = {
        "feature_names": feature_names, "feature_mean": [0.] * 19, "feature_std": [1.] * 19, "weights": [0.] * 19, "bias": 0.,
        "base_feature_names": raw_names, "post_selection_value_enabled": True,
        "post_selection_value_mode": "endpoint_potential_quality_paired_outcome_dominance_retention",
        "post_selection_endpoint_feature_names": EPV_NAMES, "post_selection_endpoint_feature_scale": [1.] * 38,
        "post_selection_endpoint_weights": [0.] * 38, "post_selection_endpoint_bias": 0.,
        "post_selection_observable_names": obs_names,
        "post_selection_quality_observable_names": ["route_deviation_cost", "progress_deficit_cost", "global_comfort_cost"],
        "post_selection_quality_observable_scale": [1.] * 3, "post_selection_quality_observable_weights": [0.] * 3,
        "selected_policy_risk_plan_response_names": pnames, "selected_policy_risk_plan_response_scales": [1.], "selected_policy_risk_plan_response_weights": [0.],
        "selected_policy_risk_ego_reference_names": enames, "selected_policy_risk_ego_reference_scales": [1., 1.], "selected_policy_risk_ego_reference_weights": [0., 0.],
        "paired_outcome_dominance_retention": {
            "feature_names": HODR_STATE_NAMES, "functional": "hurdle_pareto", "aggregation": "max_support_outcome",
            "components": {"effect_support_risk": support, "conditional_outcome_risk": outcome}, "retention_threshold": 2.0,
        },
    }
    raw = np.zeros((2, len(raw_names))); sup = np.zeros(2); X = np.zeros((2, 19)); mu = np.zeros(2); obs = np.zeros((2, len(obs_names)))
    traj = np.zeros((2, 2, 5)); traj[1, 1, 0] = 3.0
    value, _, names = _icer_post_selection_value(
        1, mu, X, feature_names, sc, raw_feat=raw, raw_feature_names=raw_names, support_logits=sup, legacy_action=0,
        value_observable_matrix=obs, value_observable_names=obs_names, candidate_trajectories=traj,
    )
    assert value == pytest.approx(-1.0)  # risk=3, threshold=2
    assert "hodr_risk" in names
