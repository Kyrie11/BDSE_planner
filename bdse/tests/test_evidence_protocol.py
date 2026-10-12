from types import SimpleNamespace

import numpy as np

from bdse.experiments.evidence_protocol import (
    acquisition_orders,
    intervention_label,
    literal_decision_critical,
    sequential_stop,
)
from bdse.planner.latency_model import advance_local_trajectory, stitch_delayed_plan
from bdse.tools.profile_planning_budget import controlled_overrides


def _trajectory(offset=0.0):
    t = np.arange(0.1, 1.1, 0.1, dtype=np.float32)
    return np.stack([offset + 2 * t, np.zeros_like(t), np.zeros_like(t), np.full_like(t, 2), t], axis=1)


def test_strict_profile_treatment_disables_budget_expansion():
    base = {
        "selector": {"min_selected_atoms": 16},
        "fallback": {"enabled": True, "budget_stages": [16]},
        "candidate": {"K": 32, "pool_K": 96, "min_valid_candidates": 16,
                      "counts": {"a": 20, "b": 12}},
        "teacher": {"robust_modes": {"logged": {"enabled": True, "prob": .5},
                                      "cv": {"enabled": True, "prob": .5}}},
    }
    args = SimpleNamespace(budget=8, proposal_top_m=12, sweep_semantics="strict",
                           candidate_k=16, candidate_pool_k=32, response_modes="logged")
    overrides, audit = controlled_overrides(base, args)
    assert overrides["selector"]["min_selected_atoms"] == 8
    assert overrides["fallback"]["enabled"] is False
    assert sum(overrides["candidate"]["counts"].values()) == 16
    assert overrides["candidate"]["pool_K"] == 32
    assert audit["requested_budget"] == 8


def test_acquisition_and_stopping_are_deterministic():
    g = np.asarray([[0, 1, 2], [0, .2, .1], [0, -1, 0], [0, .3, -.2]], dtype=float)
    var = np.full_like(g, .01)
    orders = acquisition_orders(proposal=np.asarray([1, 4, 2, 3]), pred_g=g, pred_var=var,
                                active=np.ones(4, bool), valid=np.ones(3, bool), reference=0,
                                hard=np.zeros(4, bool), random_seed=1)
    assert orders["adaptive"].tolist() == [1, 3, 2, 0]
    stopped = sequential_stop(orders["adaptive"], np.asarray([0, .1, .2]), g, var,
                              np.ones(3, bool), 0, 4, z=1.0,
                              update_margin=.01, unqueried_radius=.01)
    assert stopped.reason in {"support", "refute", "budget"}
    assert 1 <= len(stopped.selected) <= 4


def test_criticality_and_intervention_taxonomy():
    j0 = np.asarray([0.0, .1])
    g = np.asarray([[0.0, -1.0], [0.0, .2]])
    critical = literal_decision_critical(j0, g, np.ones(2, bool), np.ones(2, bool))
    assert critical.tolist() == [True, False]
    teacher = np.asarray([1.0, 0.0])
    assert intervention_label(0, 1, 1, teacher, np.zeros(2, bool), .05) == "beneficial"
    assert intervention_label(1, 0, 1, teacher, np.zeros(2, bool), .05) == "harmful"


def test_latency_stitch_is_continuous_at_release():
    incumbent = _trajectory(0.0)
    proposal = _trajectory(1.0)
    advanced = advance_local_trajectory(incumbent, .2)
    assert np.allclose(advanced[0, 0], .2, atol=1e-5)
    stitched = stitch_delayed_plan(incumbent, proposal, .5)
    release = int(np.argmin(np.abs(stitched[:, 4] - .5)))
    assert np.allclose(stitched[release, :3], incumbent[release, :3], atol=1e-5)
    assert np.all(np.diff(stitched[:, 4]) > 0)
