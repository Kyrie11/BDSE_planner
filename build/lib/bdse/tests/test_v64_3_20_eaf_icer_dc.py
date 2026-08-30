from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from bdse.planner.tournament import _apply_decisive_frontier_icer
from bdse.tests.test_v64_3_19_eaf_icer import _cfg, _diag, _matrix


def _run(cfg: dict, *, safety: np.ndarray | None = None) -> tuple[int, dict]:
    return _apply_decisive_frontier_icer(
        2,
        0,
        _matrix(),
        np.asarray([0.0, 0.10, 0.05, 0.08]),
        np.zeros((3, 4), dtype=np.float32),
        np.asarray([0.0, 0.20, 0.40, 0.30]),
        np.ones(4, bool),
        np.zeros(4, bool) if safety is None else safety,
        _diag(),
        1.0,
        None,
        cfg,
    )


def _dc_cfg(**kwargs) -> dict:
    c = _cfg(**kwargs)
    ic = c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic["all_flagged_policy"] = "preserve_legacy_for_structural_guard"
    return c


def test_v20_all_flagged_delegates_by_preserving_raw_legacy_proposal() -> None:
    selected, d = _run(_dc_cfg(dominance_bias=10.0), safety=np.ones(4, dtype=bool))
    assert selected == 2
    assert int(np.asarray(d["_decisive_frontier_icer_admissible_mask"]).sum()) == 0
    assert float(d["decisive_frontier_icer_all_flagged_domain"]) == 1.0
    assert float(d["decisive_frontier_icer_structural_domain_delegated"]) == 1.0
    assert float(d["decisive_frontier_icer_all_flagged_preserved_legacy"]) == 1.0


def test_v20_safe_domain_is_action_identical_to_v19_operator() -> None:
    old = _cfg(dominance_bias=-1.5, raw_margin_weight=10.0)
    new = copy.deepcopy(old)
    new["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["all_flagged_policy"] = "preserve_legacy_for_structural_guard"
    s0, d0 = _run(old)
    s1, d1 = _run(new)
    assert s0 == s1
    assert np.array_equal(d0["_decisive_frontier_icer_admissible_mask"], d1["_decisive_frontier_icer_admissible_mask"])
    assert np.allclose(d0["_decisive_frontier_icer_support_logit_star"], d1["_decisive_frontier_icer_support_logit_star"])
    assert np.allclose(d0["_decisive_frontier_icer_dominance_logit_star"], d1["_decisive_frontier_icer_dominance_logit_star"])
    assert float(d1["decisive_frontier_icer_structural_domain_delegated"]) == 0.0


def test_v20_fitted_configs_copy_v19_heads_exactly_and_only_change_deployment_policy() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    pairs = [
        (root / "v64_3_19_icer_scalar_frozen_uploaded.yaml", root / "v64_3_20_icer_dc_scalar.yaml"),
        (root / "v64_3_19_icer_dual_frozen_uploaded.yaml", root / "v64_3_20_icer_dc_dual.yaml"),
    ]
    for oldp, newp in pairs:
        old = yaml.safe_load(oldp.read_text(encoding="utf-8"))
        new = yaml.safe_load(newp.read_text(encoding="utf-8"))
        oi = old["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
        ni = new["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
        for prefix in ["support_", "scalar_dominance_", "profile_dominance_"]:
            for k, v in oi.items():
                if k.startswith(prefix):
                    assert ni[k] == v
        assert ni["dominance_policy"] == oi["dominance_policy"]
        assert ni["all_flagged_policy"] == "preserve_legacy_for_structural_guard"
        assert new["metadata"]["algorithm_version"] == "V64.3.20-EAF-ICER-DC-DARM-DBR"


def test_v20_design_exclusion_is_exact_prior_union_plus_v19_fresh() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    old = {x.strip() for x in (root / "v64_3_19_design_exclude_v64_3_18_screen_tokens.txt").read_text().splitlines() if x.strip()}
    new = {x.strip() for x in (root / "v64_3_20_design_exclude_v64_3_19_screen_tokens.txt").read_text().splitlines() if x.strip()}
    assert len(old) == 2700
    assert len(new) == 3200
    assert old <= new
