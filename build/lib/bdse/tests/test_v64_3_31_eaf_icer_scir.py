from __future__ import annotations

import copy

import numpy as np
import pytest

from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES
from bdse.tests.test_v64_3_19_eaf_icer import _cfg, _diag, _matrix


def _scir_cfg(*, mode: str = "rank_only", q: float = 0.0) -> dict:
    c = _cfg(support_bias=1.0, dominance_bias=-10.0, policy="scalar_only")
    ic = c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic["all_flagged_policy"] = "preserve_legacy_for_structural_guard"
    ic["incumbent_retention_policy"] = "preserve_admissible_incumbent"
    names = [f"delta::{n}" for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES] + ["delta::support_logit"]
    w = [0.0] * len(names)
    w[names.index("delta::raw_margin")] = -10.0
    ic["selection_conditioned_intervention_recovery"] = {
        "enabled": True,
        "mode": mode,
        "base_feature_names": list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES),
        "feature_names": names,
        "feature_mean": [0.0] * len(names),
        "feature_std": [1.0] * len(names),
        "weights": w,
        "bias": 3.0,
        "conformal_alpha": 0.05,
        "conformal_overprediction_quantile": q,
    }
    return c


def _run(cfg: dict, safety: np.ndarray | None = None):
    return __import__("bdse.planner.tournament", fromlist=["_apply_decisive_frontier_icer"])._apply_decisive_frontier_icer(
        2, 0, _matrix(), np.asarray([0.0, 0.10, 0.05, 0.08]), np.zeros((3, 4), dtype=np.float32),
        np.asarray([0.0, 0.20, 0.40, 0.30]), np.ones(4, bool), np.zeros(4, bool) if safety is None else safety,
        _diag(), 1.0, None, cfg,
    )


def test_scir_replaces_old_binary_dominance_with_same_scene_continuous_improvement():
    selected, d = _run(_scir_cfg())
    assert selected == 1
    assert float(d["decisive_frontier_icer_scir_proposal_exists"]) == 1.0
    assert float(d["decisive_frontier_icer_scir_proposal_predicted_improvement"]) > 0.0
    # Frozen scalar dominance is deliberately negative; it remains diagnostic only.
    assert float(np.asarray(d["_decisive_frontier_icer_scalar_dominance_logit_star"])[selected]) < 0.0


def test_scir_conformal_veto_is_no_fallback_subset_of_rank_proposal():
    rank_action, rank = _run(_scir_cfg(mode="rank_only"))
    main_action, main = _run(_scir_cfg(mode="conformal_veto", q=6.0))
    assert rank_action == 1
    assert int(main["decisive_frontier_icer_scir_proposal_action"]) == rank_action
    assert main_action == 2  # incumbent, not a second-best alternative
    assert float(main["decisive_frontier_icer_scir_certificate_accepted"]) == 0.0
    assert float(main["decisive_frontier_icer_scir_proposal_lower_bound"]) < 0.0


def test_scir_preserves_all_flagged_structural_domain_exactly():
    selected, d = _run(_scir_cfg(), safety=np.ones(4, dtype=bool))
    assert selected == 2
    assert float(d["decisive_frontier_icer_structural_domain_delegated"]) == 1.0
    assert float(d["decisive_frontier_icer_scir_proposal_exists"]) == 0.0


def test_scir_schema_mismatch_fails_closed():
    c = _scir_cfg()
    scir = c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    scir["feature_names"] = scir["feature_names"][:-1]
    with pytest.raises(ValueError, match="SCIR feature schema"):
        _run(c)


def test_scir_disabled_is_exact_legacy_behavior():
    base = _cfg(support_bias=1.0, dominance_bias=-1.5, raw_margin_weight=10.0, policy="scalar_only")
    test = copy.deepcopy(base)
    test["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"] = {"enabled": False}
    a0, d0 = _run(base)
    a1, d1 = _run(test)
    assert a0 == a1
    assert np.allclose(d0["_decisive_frontier_icer_dominance_logit_star"], d1["_decisive_frontier_icer_dominance_logit_star"])
