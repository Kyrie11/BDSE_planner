from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from bdse.planner.future_state_factorization import FSFR_OBSERVABLE_NAMES
from bdse.planner.selection_interventional_risk_retention import (
    SIIR_STATE_NAMES,
    consequence_state,
    intervention_prefix,
    runtime_certificate,
    select_interventional_winner,
)
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.planner.value_observables import VALUE_OBSERVABLE_NAMES
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES
from bdse.tools.fit_v64_3_49_eaf_icer_siir import _compress_nomult_model

ROOT = Path(__file__).resolve().parents[2]


def _m4() -> dict:
    return {
        "model": "zero_bias_pairwise_selected_sign_risk",
        "feature_names": [
            "quality_value", "prospective_response_increment", "ego_reference_increment", "log_extremal_multiplicity"
        ],
        "feature_mean": [1.0, 2.0, 3.0, 0.0],
        "feature_std": [2.0, 4.0, 5.0, 1.0e-6],
        "weights": [0.5, -0.25, 0.1, 0.0],
        "bias": 0.0,
        "lambda": 1.0,
        "use_extremal_multiplicity": False,
        "fit_positive_score_mean": 0.2,
        "fit_positive_score_std": 0.7,
    }


def test_v49_runtime_state_has_no_multiplicity_coordinate() -> None:
    z = consequence_state(1.0, 2.5, 2.0)
    assert SIIR_STATE_NAMES == ["quality_value", "prospective_response_increment", "ego_reference_increment"]
    assert z.tolist() == pytest.approx([1.0, 1.5, -0.5])
    assert z.shape == (3,)


def test_v49_label_free_prefix_intervention_is_order_invariant_and_deterministic() -> None:
    acts = [9, 3, 5, 1, 7]
    p1, m1 = intervention_prefix("scene", acts, seed="fixed")
    p2, m2 = intervention_prefix("scene", list(reversed(acts)), seed="fixed")
    assert p1 == p2
    assert m1 == m2
    assert 1 <= m1 <= len(acts)
    assert sorted(p1) == sorted(acts)


def test_v49_interventional_winner_uses_only_frozen_rsmr_runtime_quantities() -> None:
    acts = [10, 20, 30, 40]
    scores = [0.3, -0.1, 0.8, 0.5]
    support = [1.0, 1.0, 2.0, 3.0]
    margin = [0.2, 0.2, 0.1, 0.4]
    prior = [0, 0, 1, 0]
    j1, m1 = select_interventional_winner("s", acts, scores, support, margin, prior, seed="fixed")
    # Reorder inputs.  Because selection is keyed by action id and uses the same
    # frozen tie-break, the selected action must remain identical.
    perm = [3, 1, 0, 2]
    j2, m2 = select_interventional_winner(
        "s", [acts[i] for i in perm], [scores[i] for i in perm], [support[i] for i in perm],
        [margin[i] for i in perm], [prior[i] for i in perm], seed="fixed",
    )
    a1 = None if j1 is None else acts[j1]
    a2 = None if j2 is None else [acts[i] for i in perm][j2]
    assert a1 == a2
    assert m1 == m2


def test_v49_compressed_model_is_exactly_v48_nomult_in_three_dimensions() -> None:
    m4 = _m4()
    m3 = _compress_nomult_model(m4)
    z3 = consequence_state(0.4, 0.9, 0.7)
    raw4 = float(((np.r_[z3, 0.0] - np.asarray(m4["feature_mean"])) / np.asarray(m4["feature_std"])) @ np.asarray(m4["weights"]))
    risk4 = (raw4 - m4["fit_positive_score_mean"]) / m4["fit_positive_score_std"]
    cert3, risk3 = runtime_certificate(z3, {"aggregation": "sign_only", "components": {"sign_risk": m3}, "retention_threshold": 1.0})
    assert risk3 == pytest.approx(risk4)
    assert cert3 == pytest.approx(1.0 - risk4)
    assert len(m3["weights"]) == 3


def test_v49_reuses_locked_v48_nomult_runtime_and_is_invariant_to_multiplicity() -> None:
    raw_names = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    feature_names = [f"delta::{n}" for n in raw_names] + ["delta::support_logit"]
    obs_names = VALUE_OBSERVABLE_NAMES + FSFR_OBSERVABLE_NAMES
    pnames = ["fsfr_plan_1d_occupancy_cost"]
    enames = ["fsfr_plan_1d_occupancy_cost", "fsfr_predicted_demo_cost"]
    m4 = _m4()
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
            "feature_names": m4["feature_names"],
            "aggregation": "sign_only",
            "use_extremal_multiplicity": False,
            "components": {"sign_risk": m4},
            "retention_threshold": 1.0,
        },
    }
    raw = np.zeros((2, len(raw_names)))
    X = np.zeros((2, 19)); mu = np.zeros(2); sup = np.zeros(2); obs = np.zeros((2, len(obs_names)))
    v2, _, _ = _icer_post_selection_value(
        1, mu, X, feature_names, sc, raw_feat=raw, raw_feature_names=raw_names,
        support_logits=sup, legacy_action=0, value_observable_matrix=obs, value_observable_names=obs_names, selection_multiplicity=2,
    )
    v999, _, _ = _icer_post_selection_value(
        1, mu, X, feature_names, sc, raw_feat=raw, raw_feature_names=raw_names,
        support_logits=sup, legacy_action=0, value_observable_matrix=obs, value_observable_names=obs_names, selection_multiplicity=999,
    )
    assert v2 == pytest.approx(v999)


def test_v49_consumed_v48_2_fresh_ledger_is_frozen() -> None:
    p = ROOT / "bdse/configs/v64_3_48_2_consumed_fresh1000_tokens.txt"
    xs = [x.strip() for x in p.read_text().splitlines() if x.strip()]
    assert len(xs) == 1000
    assert len(set(xs)) == 1000


def test_v49_launcher_locks_v48_science_and_both_consumed_fresh_ledgers() -> None:
    text = (ROOT / "RUN_V64_3_49_EAF_ICER_SIIR_SCREEN_2GPU.sh").read_text()
    assert "sha256sum -c V64_3_49_SOURCE_MANIFEST.sha256" in text
    assert "sha256sum -c V64_3_48_OCRR_SCIENCE_LOCK.sha256" in text
    assert "v64_3_48_consumed_fresh1000_tokens.txt" in text
    assert "v64_3_48_2_consumed_fresh1000_tokens.txt" in text
    assert "v64.3.49-eaf-icer-siir-double-fresh-v1" in text
    assert "[[ $FIT_STATUS -eq 0 ]] || exit \"$FIT_STATUS\"" in text
