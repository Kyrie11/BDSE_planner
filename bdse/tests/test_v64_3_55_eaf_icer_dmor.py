from __future__ import annotations

import numpy as np
import pytest

from bdse.planner.paired_dynamic_mediator_outcome_retention import (
    DMOR_PREDICTED_STATE_NAMES,
    DMOR_REALIZED_STATE_NAMES,
    MEDIATOR_MODEL_SCHEMA,
    fit_zero_preserving_mediator_ridge,
    outcome_state,
    planned_mediator_input,
    predict_realized_endpoint,
)
from bdse.planner.paired_operator_trajectory_retention import PROFILE_SCHEMA


def _profile(endpoint=None, temporal=None):
    return {
        "schema": PROFILE_SCHEMA,
        "execution_contrast_linf": 1.0,
        "endpoint_signed": list(endpoint if endpoint is not None else [1.0, -2.0, 0.1, 3.0]),
        "cosine_modes_1_2": list(temporal if temporal is not None else np.arange(8, dtype=float)),
        "trajectory_steps": 16,
    }


def test_planned_mediator_input_schema_and_shape():
    x = planned_mediator_input(_profile())
    assert x.shape == (12,)
    assert np.allclose(x[:4], [1.0, -2.0, 0.1, 3.0])
    with pytest.raises(ValueError):
        planned_mediator_input({**_profile(), "schema": "wrong"})


def test_zero_preserving_mediator_predictor_exact_zero():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(128, 12))
    Y = rng.normal(scale=0.1, size=(128, 4))
    model = fit_zero_preserving_mediator_ridge(X, Y)
    assert model["schema"] == MEDIATOR_MODEL_SCHEMA
    pred = predict_realized_endpoint(np.zeros(12), model)
    assert np.array_equal(pred, np.zeros(4))


def test_zero_preserving_mediator_predictor_recovers_linear_mapping():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(256, 12))
    W = rng.normal(scale=0.08, size=(12, 4))
    Y = X @ W + rng.normal(scale=0.002, size=(256, 4))
    model = fit_zero_preserving_mediator_ridge(X, Y)
    P = np.stack([predict_realized_endpoint(x, model) for x in X])
    mse = np.mean((P - Y) ** 2)
    zero = np.mean(Y ** 2)
    assert mse < 0.05 * zero
    assert model["fit_normalized_mse"] < model["zero_baseline_normalized_mse"]


def test_predictor_rejects_bias_drift():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 12)); Y = rng.normal(size=(100, 4))
    model = fit_zero_preserving_mediator_ridge(X, Y)
    model["bias"][0] = 1e-3
    with pytest.raises(ValueError):
        predict_realized_endpoint(np.zeros(12), model)


def test_outcome_state_realized_and_predicted_have_fixed_shape():
    med = np.asarray([0.1, -0.2, 0.03, 0.4])
    zr = outcome_state(1.0, 1.2, 0.8, 2.5, med, predicted=False)
    zp = outcome_state(1.0, 1.2, 0.8, 2.5, med, predicted=True)
    assert zr.shape == (len(DMOR_REALIZED_STATE_NAMES),)
    assert zp.shape == (len(DMOR_PREDICTED_STATE_NAMES),)
    assert np.allclose(zr[:4], [1.0, 0.2, -0.4, 2.5])
    assert np.allclose(zr[4:], med)


def test_outcome_state_rejects_negative_planned_d():
    with pytest.raises(ValueError):
        outcome_state(0.0, 0.0, 0.0, -1.0, np.zeros(4), predicted=False)


def test_nested_does_not_evaluate_predicted_branch_after_oracle_stop(monkeypatch):
    from bdse.tools import fit_v64_3_55_eaf_icer_dmor as fit

    calls = []
    def fake_safety(_rows):
        return ["h"]
    def fake_eval(_rows, *, arm, alpha, safety, control, cagg):
        calls.append(arm)
        assert arm == "realized_dominance"
        return {
            "identification": {"functional_identified": False},
            "deployment_gate": {"pass": False},
            "pass": False,
        }
    monkeypatch.setattr(fit.v52, "_safety_names", fake_safety)
    monkeypatch.setattr(fit, "_evaluate_arm", fake_eval)
    report = {
        "nested_crossfit": {
            "arms": {
                "hurdle_pareto": {
                    "identification": {"pareto_concordance": fit.EXPECTED_V52_PARETO_CONCORDANCE},
                    "folds": [{"fold": k, "pareto_concordance": 0.5} for k in range(fit.FOLDS)],
                },
                "hurdle_sign": {"identification": {"support_auc": 0.65}},
            }
        }
    }
    out = fit._nested([{"outer_test_fold": 0}], 0.07, report, {})
    assert calls == ["realized_dominance"]
    assert out["arms"]["predicted_dominance"]["status"] == "NOT_EVALUATED_BY_PREREGISTERED_BRANCH_ORDER"
    assert out["deployable_train_gate_pass"] is False


def test_nested_evaluates_predicted_only_after_oracle_full_pass(monkeypatch):
    from bdse.tools import fit_v64_3_55_eaf_icer_dmor as fit

    calls = []
    def fake_safety(_rows):
        return ["h"]
    def fake_eval(_rows, *, arm, alpha, safety, control, cagg):
        calls.append(arm)
        if arm == "realized_dominance":
            return {
                "identification": {"functional_identified": True},
                "deployment_gate": {"pass": True},
                "pass": False,
            }
        return {
            "identification": {
                "functional_identified": True,
                "mediator_prediction": {"identified": True},
            },
            "deployment_gate": {"pass": True},
            "pass": False,
        }
    monkeypatch.setattr(fit.v52, "_safety_names", fake_safety)
    monkeypatch.setattr(fit, "_evaluate_arm", fake_eval)
    report = {
        "nested_crossfit": {
            "arms": {
                "hurdle_pareto": {
                    "identification": {"pareto_concordance": fit.EXPECTED_V52_PARETO_CONCORDANCE},
                    "folds": [{"fold": k, "pareto_concordance": 0.5} for k in range(fit.FOLDS)],
                },
                "hurdle_sign": {"identification": {"support_auc": 0.65}},
            }
        }
    }
    out = fit._nested([{"outer_test_fold": 0}], 0.07, report, {})
    assert calls == ["realized_dominance", "predicted_dominance"]
    assert out["deployable_train_gate_pass"] is True
    assert out["arms"]["predicted_dominance"]["eligible_by_branch_order"] is True
