from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES
from bdse.tests.test_v64_3_22_eaf_icer_tcr import _tcr_cfg, _run
from bdse.tools.fit_v64_3_23_eaf_icer_rcr import _feature_metric_weight


def _memory(tmp_path: Path, delta: float) -> tuple[str, str]:
    n = len(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    path = tmp_path / ("pos.npz" if delta > 0 else "neg.npz")
    names = np.asarray([f"evidence::{x}" for x in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES], dtype="U128")
    np.savez_compressed(
        path,
        memory_metric_z=np.zeros((96, n), dtype=np.float32),
        teacher_improvement=np.full((96,), delta, dtype=np.float32),
        feature_mean=np.zeros((n,), dtype=np.float32),
        feature_std=np.ones((n,), dtype=np.float32),
        feature_names=names,
        feature_metric_weight=np.full((n,), 1.0 / n, dtype=np.float32),
        neighbor_k_values=np.asarray([32, 64], dtype=np.int32),
        se_multiplier=np.asarray([1.0], dtype=np.float32),
    )
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _rcr_cfg(tmp_path: Path, delta: float, policy: str = "scalar_positive_dual_mean_positive") -> dict:
    c = _tcr_cfg(rep_bias=1.0, ret_bias=1.0, mode="evidence_only", dominance=policy)
    ic = c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    p, h = _memory(tmp_path, delta)
    ic.update({
        "incumbent_retention_policy": "preserve_admissible_incumbent",
        "regret_risk_enabled": True,
        "retention_regret_risk_enabled": False,
        "replacement_regret_risk_enabled": True,
        "regret_risk_model_type": "local_multiscale_regret_lower_bound",
        "regret_risk_feature_mode": "evidence_only",
        "replacement_local_regret_memory_path": p,
        "replacement_local_regret_memory_sha256": h,
        "regret_risk_threshold_policy": "fixed_zero_multiscale_local_lower_bound_no_validation_sweep",
    })
    return c


def test_group_balanced_metric_does_not_let_transition_dimensionality_dominate() -> None:
    e = _feature_metric_weight("evidence_only")
    t = _feature_metric_weight("transition_conditioned")
    assert len(e) == 18 and np.isclose(e.sum(), 1.0)
    assert len(t) == 59
    assert np.isclose(t[:18].sum(), 1.0)
    assert np.isclose(t[18:39].sum(), 1.0)
    assert np.isclose(t[39:].sum(), 1.0)


def test_local_regret_lower_bound_vetoes_negative_train_neighborhood(tmp_path: Path) -> None:
    c = _rcr_cfg(tmp_path, -1.0)
    selected, diag = _run(c)
    assert selected == 2  # final-guard-admissible incumbent preserved by default
    assert float(diag["decisive_frontier_icer_replacement_regret_risk_enabled"]) == 1.0
    assert float(diag["decisive_frontier_icer_regret_risk_local_memory"]) == 1.0


def test_local_regret_lower_bound_allows_positive_replacement_neighborhood(tmp_path: Path) -> None:
    c = _rcr_cfg(tmp_path, 1.0, policy="scalar_only")
    selected, _ = _run(c)
    assert selected != 2


def test_self_consistent_dual_requires_actual_equal_mean_rank_score_positive(tmp_path: Path) -> None:
    c = _rcr_cfg(tmp_path, 1.0, policy="scalar_positive_dual_mean_positive")
    ic = c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic["scalar_dominance_bias"] = 2.0
    ic["profile_dominance_bias"] = -6.0
    selected, diag = _run(c)
    assert selected == 2
    assert float(diag["decisive_frontier_icer_dominance_policy_scalar_positive_dual_mean_positive"]) == 1.0


def test_rcr_never_learned_vetoes_an_admissible_incumbent(tmp_path: Path) -> None:
    c = _rcr_cfg(tmp_path, -1.0)
    ic = c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic["support_bias"] = -20.0  # generic all-edge support may dislike the incumbent
    selected, _ = _run(c)
    assert selected == 2


def test_v23_design_exclusion_is_unchanged_because_v22_never_selected_fresh_validation() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    old = {x.strip() for x in (root / "v64_3_22_design_exclude_v64_3_21_screen_tokens.txt").read_text().splitlines() if x.strip()}
    new = {x.strip() for x in (root / "v64_3_23_design_exclude_v64_3_21_screen_tokens.txt").read_text().splitlines() if x.strip()}
    assert len(old) == len(new) == 4700 and old == new
