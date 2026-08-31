from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from bdse.planner.nuplan_planner import BDSEPlannerCore
from bdse.tools.build_v64_3_50_pior_train_manifest import CITY_TO_RAW
from bdse.tools.run_v64_3_50_pior_paired_closed_loop import _pair, SAFETY_METRICS

ROOT = Path(__file__).resolve().parents[2]


def _tournament(proposal: int = 3, baseline: int = 1, exists: bool = True, action: int = 1):
    return SimpleNamespace(
        action_index=action,
        diagnostics={
            "decisive_frontier_icer_scir_proposal_exists": float(exists),
            "decisive_frontier_icer_scir_proposal_action": float(proposal),
            "decisive_frontier_icer_baseline_action": float(baseline),
        },
    )


def _candidates():
    return SimpleNamespace(K=5, valid_mask=np.asarray([True, True, True, True, True], dtype=bool))


def _cfg(arm: str | None, *, fallback: bool = False):
    c = {"fallback": {"enabled": fallback, "rule_rerank_top_k": 0}}
    if arm is not None:
        c["selected_outcome_probe"] = {"enabled": True, "arm": arm, "one_shot": True}
    return c


def test_v50_probe_is_semantic_noop_when_disabled() -> None:
    core = BDSEPlannerCore(cfg=_cfg(None))
    action, diag = core._apply_selected_outcome_probe(_tournament(action=4), 4, _candidates())
    assert action == 4
    assert diag["pior_probe_enabled"] is False
    assert core._pior_probe_used is False


def test_v50_treatment_executes_exact_frozen_proposal_once_then_incumbent() -> None:
    core = BDSEPlannerCore(cfg=_cfg("treatment"))
    action, d1 = core._apply_selected_outcome_probe(_tournament(proposal=3, baseline=1, action=1), 1, _candidates())
    assert action == 3
    assert d1["pior_probe_fired"] is True
    assert d1["pior_probe_contract_same_frozen_proposal_or_incumbent"] is True
    action2, d2 = core._apply_selected_outcome_probe(_tournament(proposal=4, baseline=1, action=4), 4, _candidates())
    assert action2 == 1
    assert d2["pior_probe_fired"] is False
    assert d2["pior_probe_phase"] == "post_intervention_incumbent"
    assert core._pior_probe_event_count == 1


def test_v50_control_marks_same_event_but_never_leaves_incumbent() -> None:
    core = BDSEPlannerCore(cfg=_cfg("control"))
    action, d = core._apply_selected_outcome_probe(_tournament(proposal=3, baseline=1, action=3), 3, _candidates())
    assert action == 1
    assert d["pior_probe_fired"] is True
    assert d["pior_probe_proposal_action"] == 3
    assert d["pior_probe_baseline_action"] == 1


def test_v50_probe_refuses_fallback_or_second_path() -> None:
    core = BDSEPlannerCore(cfg=_cfg("treatment", fallback=True))
    with pytest.raises(RuntimeError, match="fallback.enabled=false"):
        core._apply_selected_outcome_probe(_tournament(), 1, _candidates())


def test_v50_dataset_contract_matches_user_train_and_raw_db_layout() -> None:
    assert CITY_TO_RAW == {
        "train_boston": "train_boston",
        "train_pittsburgh": "train_pittsburgh",
        "train_singapore": "train_singapore",
        "train_vegas_2": "train_vegas",
    }


def test_v50_paired_outcome_is_positive_only_when_score_improves_without_hard_safety_degradation() -> None:
    metrics = {"score": 0.7, **{k: 1.0 for k in SAFETY_METRICS}}
    control = {"a": {"identity": {}, "metrics": metrics}}
    treat_good = {"a": {"identity": {}, "metrics": {**metrics, "score": 0.8}}}
    row = _pair(control, treat_good, ["a"])[0]
    assert row["closed_loop_beneficial"] is True
    assert row["pior_interventional_outcome"] == 1.0
    treat_harm = {"a": {"identity": {}, "metrics": {**metrics, "score": 0.9, "time_to_collision_within_bound": 0.0}}}
    row2 = _pair(control, treat_harm, ["a"])[0]
    assert row2["closed_loop_hard_harm"] is True
    assert row2["closed_loop_beneficial"] is False
    assert row2["pior_interventional_outcome"] == -1.0


def test_v50_launcher_changes_evidence_source_and_stops_before_untouched_validation() -> None:
    text = (ROOT / "RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh").read_text()
    assert "v64_3_49_siir_fit.json" in text
    assert "805c4f8088051413edeb568623bc6d225d1b3c301c52612f89109216b38be296" in text
    assert "bdse_train_v2" in text
    assert "train_vegas_2" in text and "train_vegas" in text
    assert "run_v64_3_50_pior_paired_closed_loop" in text
    assert "closed_loop_nonreactive_agents" in text
    assert "STOP before untouched closed-loop validation" in text or "do not consume untouched closed-loop validation" in text
    assert "A/B pooling" not in text  # no offline fresh-rescue stage exists in V50 launcher
