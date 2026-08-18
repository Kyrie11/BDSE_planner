from __future__ import annotations

import numpy as np

from bdse.planner.tournament import (
    _DACER_FEATURE_NAMES,
    _DALER_FEATURE_NAMES,
    _apply_decisive_frontier_dacer,
    _decisive_frontier_dacer_features,
    _decisive_frontier_guard_admissible_mask,
    _decisive_frontier_value_star_residual_numpy,
)
from bdse.tools.fit_v64_3_18_eaf_dacer import _fit, _logits


def _base_cfg(*, enabled: bool, weights: list[float] | None = None, bias: float = 0.0) -> dict:
    w = [0.0] * len(_DACER_FEATURE_NAMES) if weights is None else list(weights)
    return {
        "runtime": {
            "pair_action_anchor_guard": {"enabled": True, "flip_margin": 0.015, "score_margin": 0.0},
            "dual_certificate": {
                "enabled": True,
                "require_evidence_certificate_before_residual_flip": True,
                "min_evidence_certificate_fraction_for_residual_flip": 1.0,
            },
            "decisive_frontier_value": {
                "deployment_admissible_counterfactual_extremal_recovery": {
                    "enabled": enabled,
                    "instrument_features": True,
                    "feature_mode": "profile",
                    "feature_names": list(_DACER_FEATURE_NAMES),
                    "feature_mean": [0.0] * len(_DACER_FEATURE_NAMES),
                    "feature_std": [1.0] * len(_DACER_FEATURE_NAMES),
                    "weights": w,
                    "bias": bias,
                    "anchor_logit": 0.0,
                    "require_guard_admissible": True,
                    "require_safe_available_for_learned_intervention": True,
                    "utility_equivalence_role": "diagnostic_tiebreak_only_not_hard_mask",
                }
            },
        },
        "tournament": {"utility_refinement": {"enabled": False}},
    }


def _matrix() -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float32)
    for b, v in [(1, 0.20), (2, 0.40), (3, 0.30)]:
        m[b, 0] = v
        m[0, b] = -v
    return m


def _diag() -> dict:
    return {
        "decisive_frontier_value_active": 1.0,
        "decisive_frontier_value_residual_rms": 0.2,
        "decisive_frontier_value_residual_abs_mean": 0.15,
        "decisive_frontier_value_attribution_scale_rms": 0.12,
        "decisive_frontier_value_attribution_scale_mean": 0.10,
    }


def test_dacer_guard_admissibility_does_not_hard_gate_on_legacy_utility_pool() -> None:
    cfg = _base_cfg(enabled=False)
    margins = _matrix()[:, 0]
    scores = np.asarray([0.0, 0.2, 0.4, 0.3], dtype=np.float32)
    mask = _decisive_frontier_guard_admissible_mask(
        margins, scores, np.ones(4, bool), np.zeros(4, bool), 0, 1.0, cfg
    )
    # All three challengers satisfy the actual frozen final guard prerequisites.
    assert mask.tolist() == [False, True, True, True]


def test_dacer_can_recover_alternative_outside_legacy_utility_prior() -> None:
    weights = [0.0] * len(_DACER_FEATURE_NAMES)
    weights[_DACER_FEATURE_NAMES.index("attribution_scale")] = 10.0
    cfg = _base_cfg(enabled=True, weights=weights, bias=-0.5)
    utility_diag = {
        # The legacy pool contains only action 2; action 1 must remain learnably recoverable.
        "_utility_refinement_eligible_mask": np.asarray([False, False, True, False]),
        "_utility_refinement_cost": np.asarray([0.0, 0.5, 0.1, 0.4], dtype=np.float32),
    }
    selected, d = _apply_decisive_frontier_dacer(
        2, 0, _matrix(), np.asarray([0.0, 1.0, 0.01, 0.02]), np.zeros((2, 4), dtype=np.float32),
        np.asarray([0.0, 0.2, 0.4, 0.3]), np.ones(4, bool), np.zeros(4, bool),
        _diag(), 1.0, utility_diag, cfg,
    )
    assert not bool(d["_decisive_frontier_dacer_utility_prior_mask"][1])
    assert bool(d["_decisive_frontier_dacer_admissible_mask"][1])
    assert selected == 1


def test_dacer_frozen_guard_and_evidence_certificate_cannot_be_bypassed() -> None:
    weights = [0.0] * len(_DACER_FEATURE_NAMES)
    weights[_DACER_FEATURE_NAMES.index("raw_margin")] = 100.0
    cfg = _base_cfg(enabled=True, weights=weights, bias=10.0)
    selected, d = _apply_decisive_frontier_dacer(
        2, 0, _matrix(), np.asarray([0.0, 0.1, 0.1, 0.1]), np.zeros((2, 4), dtype=np.float32),
        np.asarray([0.0, 0.2, 0.4, 0.3]), np.ones(4, bool), np.zeros(4, bool),
        _diag(), 0.5, None, cfg,
    )
    assert int(np.asarray(d["_decisive_frontier_dacer_admissible_mask"]).sum()) == 0
    assert selected == 0


def test_dacer_all_flagged_bank_abstains_to_frozen_structural_guard() -> None:
    cfg = _base_cfg(enabled=True, bias=10.0)
    selected, d = _apply_decisive_frontier_dacer(
        2, 0, _matrix(), np.asarray([0.0, 0.1, 0.1, 0.1]), np.zeros((2, 4), dtype=np.float32),
        np.asarray([0.0, 0.2, 0.4, 0.3]), np.ones(4, bool), np.ones(4, bool),
        _diag(), 1.0, None, cfg,
    )
    assert int(np.asarray(d["_decisive_frontier_dacer_admissible_mask"]).sum()) == 0
    assert selected == 0


def test_exact_signed_atom_attribution_columns_sum_to_frozen_eaf_residual() -> None:
    rng = np.random.default_rng(18)
    atom = rng.normal(size=(8, 5)).astype(np.float32)
    signed = rng.normal(size=(4, 5)).astype(np.float32)
    context = rng.normal(size=(4, 5)).astype(np.float32)
    residual, d = _decisive_frontier_value_star_residual_numpy(
        [0, 2, 4, 6], np.ones(4, bool), 0, atom, signed, context, scale=0.7
    )
    contrib = np.asarray(d["_decisive_frontier_value_atom_contrib_star"], dtype=np.float64)
    np.testing.assert_allclose(contrib.sum(axis=0), residual, rtol=2e-5, atol=2e-6)


def test_dacer_profile_contains_incumbent_relative_signed_attribution_structure() -> None:
    cfg = _base_cfg(enabled=False)
    m = _matrix(); margins = m[:, 0]; scores = np.asarray([0.0, 0.2, 0.4, 0.3])
    admissible = np.asarray([False, True, True, True])
    contrib = np.asarray([
        [0.0, 0.4, 0.1, -0.2],
        [0.0, 0.1, 0.2, 0.1],
        [0.0, -0.1, 0.1, 0.2],
    ], dtype=np.float32)
    feat, names, prior = _decisive_frontier_dacer_features(
        margins, np.sqrt((contrib * contrib).sum(axis=0)), contrib, scores, np.ones(4, bool),
        0, 2, _diag(), 1.0,
        {"_utility_refinement_eligible_mask": np.asarray([False, False, True, False])},
        admissible, cfg,
    )
    assert names == _DACER_FEATURE_NAMES
    assert feat.shape == (4, len(_DACER_FEATURE_NAMES))
    assert np.all(np.isfinite(feat))
    assert prior.tolist() == [False, False, True, False]
    # Action 1 differs from legacy action 2 at atom level, so at least one delta-profile feature is non-zero.
    delta_start = len(_DALER_FEATURE_NAMES) + 6
    assert np.any(np.abs(feat[1, delta_start:]) > 1e-8)


def test_dacer_counterfactual_fitter_learns_incumbent_relative_ordering() -> None:
    rng = np.random.default_rng(1818)
    X=[]; y=[]; tm=[]; tokens=[]; challenger=[]; legacy=[]; mask=[]
    for s in range(300):
        lg = 1
        base = rng.normal(scale=0.2)
        for act in [1, 2, 3]:
            margin = base + (0.9 if act == 2 and s % 2 == 0 else 0.35 if act == 1 else -0.25)
            row = np.zeros(len(_DACER_FEATURE_NAMES), dtype=float)
            row[0] = margin + rng.normal(scale=0.1)
            row[_DACER_FEATURE_NAMES.index("delta_atom_contrib_l1")] = (margin - (base + 0.35)) + rng.normal(scale=0.1)
            X.append(row); tm.append(margin); y.append(float(margin > 0)); tokens.append(f"s{s}")
            challenger.append(act); legacy.append(lg); mask.append(True)
    X=np.asarray(X); y=np.asarray(y); tm=np.asarray(tm); mask=np.asarray(mask,bool)
    w,b,mean,std=_fit(
        X,y,tm,tokens,np.asarray(challenger),np.asarray(legacy),mask,
        feature_mode="profile",objective_mode="counterfactual",steps=250,lr=0.04,l2=1e-3,seed=18,
    )
    logits=_logits(X,w,b,mean,std)
    correct=[]
    for s in range(300):
        idx=np.arange(3*s,3*s+3); best=int(idx[np.argmax(logits[idx])]); target=int(idx[np.argmax(tm[idx])])
        correct.append(best==target)
    assert float(np.mean(correct)) > 0.80
