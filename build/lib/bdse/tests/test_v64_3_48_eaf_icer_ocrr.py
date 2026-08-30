import math

import numpy as np
import pytest

from bdse.planner.future_state_factorization import FSFR_OBSERVABLE_NAMES
from bdse.planner.operator_conditioned_risk_retention import (
    OCRR_STATE_NAMES,
    operator_state,
    runtime_certificate,
)
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.planner.value_observables import VALUE_OBSERVABLE_NAMES
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES
from bdse.tools.fit_v64_3_48_eaf_icer_ocrr import (
    _conformal_threshold,
    _fit_sign_ranker,
    _retention_alpha,
    _state,
)


def _risk_model(mult_weight=1.0):
    return {
        "model": "zero_bias_pairwise_selected_sign_risk",
        "feature_names": list(OCRR_STATE_NAMES),
        "feature_mean": [0.0] * 4,
        "feature_std": [1.0] * 4,
        "weights": [0.0, 0.0, 0.0, float(mult_weight)],
        "bias": 0.0,
        "lambda": 1.0,
        "use_extremal_multiplicity": True,
        "fit_positive_score_mean": 0.0,
        "fit_positive_score_std": 1.0,
    }


def test_v48_operator_state_factorizes_existing_values_and_multiplicity():
    z = operator_state(1.0, 2.5, 2.0, 4)
    assert z.tolist()[:3] == pytest.approx([1.0, 1.5, -0.5])
    assert z[3] == pytest.approx(math.log(4.0))


def test_v48_no_multiplicity_ablation_changes_only_operator_coordinate():
    row = {
        "quality_value": 0.2,
        "plan_control_value": 0.5,
        "ego_ref_value": 0.4,
        "candidate_count": 9,
        "scenario_token": "x",
    }
    a = _state(row, False)
    b = _state(row, True)
    assert np.allclose(a[:3], b[:3], atol=0.0, rtol=0.0)
    assert a[3] == 0.0
    assert b[3] == pytest.approx(math.log(9.0))


def test_v48_runtime_risk_is_veto_only_more_competition_can_reduce_certificate():
    cfg = {
        "aggregation": "sign_only",
        "components": {"sign_risk": _risk_model(1.0)},
        "retention_threshold": 2.0,
    }
    c2, r2, _ = runtime_certificate(operator_state(0, 0, 0, 2), cfg)
    c20, r20, _ = runtime_certificate(operator_state(0, 0, 0, 20), cfg)
    assert r20 > r2
    assert c20 < c2


def test_v48_pairwise_selected_sign_ranker_identifies_multiplicity_signal():
    rows = []
    for i in range(40):
        rows.append({
            "rsm_selected_action": 1,
            "rsm_selected_teacher_improvement": 1.0,
            "quality_value": 0.0,
            "plan_control_value": 0.0,
            "ego_ref_value": 0.0,
            "candidate_count": 2 + (i % 2),
            "scenario_token": f"g{i}",
        })
    for i in range(40):
        rows.append({
            "rsm_selected_action": 1,
            "rsm_selected_teacher_improvement": -1.0,
            "quality_value": 0.0,
            "plan_control_value": 0.0,
            "ego_ref_value": 0.0,
            "candidate_count": 16 + (i % 2),
            "scenario_token": f"b{i}",
        })
    m = _fit_sign_ranker(rows, True)
    assert m["bias"] == 0.0
    assert m["weights"][3] > 0.0
    assert m["objective_final"] < m["objective_at_zero"]


def test_v48_capture_budget_maps_existing_absolute_tolerance_to_conditional_retention():
    alpha = _retention_alpha({"positive_capture_rate": 0.38501742160278746})
    assert alpha == pytest.approx(0.03 / 0.38501742160278746)
    assert 0.07 < alpha < 0.08


def test_v48_split_calibration_threshold_uses_only_frozen_policy_positives():
    model = _risk_model(1.0)
    cal = []
    for k in range(1, 21):
        cal.append({
            "rsm_selected_action": 1,
            "rsm_selected_teacher_improvement": 1.0,
            "quality_value": 0.0,
            "plan_control_value": 0.0,
            "ego_ref_value": 0.0,
            "candidate_count": k,
            "scenario_token": f"p{k}",
        })
    # A negative selected sample must not alter the positive-retention quantile.
    cal.append({**cal[-1], "scenario_token": "neg", "rsm_selected_teacher_improvement": -1.0, "candidate_count": 1000})
    tau, info = _conformal_threshold(cal, model, alpha=0.1)
    rank = math.ceil((20 + 1) * 0.9)
    assert info["positive_calibration_count"] == 20
    assert info["conformal_rank"] == rank
    assert tau == pytest.approx(math.log(rank))


def test_v48_tournament_consumes_operator_risk_without_reranking():
    raw_names = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    feature_names = [f"delta::{n}" for n in raw_names] + ["delta::support_logit"]
    obs_names = VALUE_OBSERVABLE_NAMES + FSFR_OBSERVABLE_NAMES
    pnames = ["fsfr_plan_1d_occupancy_cost"]
    enames = ["fsfr_plan_1d_occupancy_cost", "fsfr_predicted_demo_cost"]
    risk = _risk_model(1.0)
    sc = {
        "feature_names": feature_names,
        "feature_mean": [0.0] * 19,
        "feature_std": [1.0] * 19,
        "weights": [0.0] * 19,
        "bias": 0.0,
        "base_feature_names": raw_names,
        "post_selection_value_enabled": True,
        "post_selection_value_mode": "endpoint_potential_quality_operator_conditioned_risk_retention",
        "post_selection_endpoint_feature_names": EPV_NAMES,
        "post_selection_endpoint_feature_scale": [1.0] * 38,
        "post_selection_endpoint_weights": [0.0] * 38,
        "post_selection_endpoint_bias": 0.0,
        "post_selection_observable_names": obs_names,
        "post_selection_quality_observable_names": ["route_deviation_cost", "progress_deficit_cost", "global_comfort_cost"],
        "post_selection_quality_observable_scale": [1.0] * 3,
        "post_selection_quality_observable_weights": [0.0] * 3,
        "selected_policy_risk_plan_response_names": pnames,
        "selected_policy_risk_plan_response_scales": [1.0],
        "selected_policy_risk_plan_response_weights": [0.0],
        "selected_policy_risk_ego_reference_names": enames,
        "selected_policy_risk_ego_reference_scales": [1.0, 1.0],
        "selected_policy_risk_ego_reference_weights": [0.0, 0.0],
        "operator_conditioned_risk_retention": {
            "feature_names": list(OCRR_STATE_NAMES),
            "aggregation": "sign_only",
            "use_extremal_multiplicity": True,
            "components": {"sign_risk": risk},
            "retention_threshold": 2.0,
        },
    }
    raw = np.zeros((2, len(raw_names)))
    sup = np.zeros(2)
    X = np.zeros((2, 19))
    mu = np.zeros(2)
    obs = np.zeros((2, len(obs_names)))
    v2, _, _ = _icer_post_selection_value(
        1, mu, X, feature_names, sc,
        raw_feat=raw, raw_feature_names=raw_names, support_logits=sup, legacy_action=0,
        value_observable_matrix=obs, value_observable_names=obs_names, selection_multiplicity=2,
    )
    v20, _, _ = _icer_post_selection_value(
        1, mu, X, feature_names, sc,
        raw_feat=raw, raw_feature_names=raw_names, support_logits=sup, legacy_action=0,
        value_observable_matrix=obs, value_observable_names=obs_names, selection_multiplicity=20,
    )
    assert v2 == pytest.approx(2.0 - math.log(2.0))
    assert v20 == pytest.approx(2.0 - math.log(20.0))
    assert v2 > 0.0 > v20
