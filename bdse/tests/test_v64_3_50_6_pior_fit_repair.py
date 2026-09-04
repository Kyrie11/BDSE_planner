from __future__ import annotations

import math
from pathlib import Path

import pytest

from bdse.tools import fit_v64_3_50_6_eaf_icer_pior as repair


def _model() -> dict:
    return {
        "use_extremal_multiplicity": False,
        "feature_mean": [0.0, 0.0, 0.0, 0.0],
        "feature_std": [1.0, 1.0, 1.0, 1.0],
        "weights": [1.0, 0.0, 0.0, 0.0],
        "fit_positive_score_mean": 0.0,
        "fit_positive_score_std": 1.0,
    }


def _row(i: int, positive: bool = True) -> dict:
    return {
        "scenario_token": f"tok{i:03d}",
        "rsm_selected_action": 0,
        "rsm_selected_teacher_improvement": 1.0 if positive else -1.0,
        "quality_value": float(i),
        "plan_control_value": float(i),
        "ego_ref_value": float(i),
        "candidate_count": 1,
    }


def test_v50_6_frozen_alpha_accepts_15_positive_calibration_examples() -> None:
    alpha = 0.0779185520
    cal = [_row(i) for i in range(15)]
    tau, info = repair._pior_conformal_threshold(cal, _model(), alpha)
    assert info["positive_calibration_count"] == 15
    assert info["minimum_positive_count_for_finite_rank"] == 12
    assert info["conformal_rank"] == math.ceil(16 * (1.0 - alpha)) == 15
    assert math.isfinite(tau)


def test_v50_6_fails_closed_if_frozen_alpha_has_no_finite_empirical_rank() -> None:
    alpha = 0.0779185520
    cal = [_row(i) for i in range(11)]
    with pytest.raises(ValueError, match="insufficient paired beneficial outcomes"):
        repair._pior_conformal_threshold(cal, _model(), alpha)


def test_v50_6_oof_keep_is_aligned_by_token_not_fold_concatenation_order() -> None:
    rows = [
        {"scenario_token": "c"},
        {"scenario_token": "a"},
        {"scenario_token": "b"},
    ]
    keep_by_token = {"a": True, "b": False, "c": True}
    assert repair._align_oof_keep_by_token(rows, keep_by_token) == [True, True, False]


def test_v50_6_oof_keep_refuses_missing_or_extra_tokens() -> None:
    rows = [{"scenario_token": "a"}, {"scenario_token": "b"}]
    with pytest.raises(RuntimeError, match="identity mismatch"):
        repair._align_oof_keep_by_token(rows, {"a": True, "c": False})


def test_v50_6_keeps_v50_5_science_critical_files_unchanged() -> None:
    import hashlib
    root = Path(__file__).resolve().parents[2]
    expected = {
        "bdse/tools/run_v64_3_50_pior_paired_closed_loop.py": "7c5472442e5a76ee6cbb6ef3189e086e4e800e8337deb14cacb26488b9050d53",
        "bdse/planner/nuplan_planner.py": "c3a6e37901349408b7c8e6ab7b3811f905f3a81b0c441e6aa7ddf4dde92131ef",
        "bdse/planner/tournament.py": "291b3b77202974b74fe42431ee7954de8c401d927591c19a12a5837f18374044",
        "bdse/tools/fit_v64_3_50_eaf_icer_pior.py": "c1c1d297766d8a2e43430739d639ff1ed73f866ec193b47a5dc9e7cb997727aa",
    }
    # V51 intentionally extends tournament.py with a new post-selection mode.
    # Historical V48/V50 locks still apply to every unchanged parent file;
    # tournament behavior is protected by exact V50 control replay plus V51
    # focused tests and the V51 full-source manifest.
    if (root / "bdse/planner/paired_operator_contrast_retention.py").is_file():
        expected.pop("bdse/planner/tournament.py", None)
    for rel, want in expected.items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == want
