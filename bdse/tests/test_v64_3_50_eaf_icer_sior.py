from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from bdse.planner.selected_outcome_probe import SelectedOutcomeProbeState, apply_selected_outcome_probe
from bdse.tools.fit_v64_3_50_eaf_icer_sior import ALPHA_RET, V49_FAILURE, _check_v49
from bdse.tools.prepare_v64_3_50_eaf_icer_sior_probe_configs import _make
from bdse.tools.run_v64_3_50_paired_selected_outcome_collection import HARD_METRICS, _hard_noninferiority

ROOT = Path(__file__).resolve().parents[2]


def _diag(proposal: int = 7, baseline: int = 2, selected: int = 7) -> dict:
    return {
        "decisive_frontier_icer_scir_proposal_exists": 1.0,
        "decisive_frontier_icer_scir_proposal_action": float(proposal),
        "decisive_frontier_icer_baseline_action": float(baseline),
        "decisive_frontier_icer_selected_action": float(selected),
    }


def _cfg(role: str) -> dict:
    return {"fallback": {"enabled": False}, "selected_outcome_probe": {"enabled": True, "role": role}}


def test_v50_probe_disabled_is_exact_noop() -> None:
    s = SelectedOutcomeProbeState()
    out, d = apply_selected_outcome_probe(7, _diag(), {}, s)
    assert out == 7
    assert d["enabled"] is False
    assert s.executed_intervention_count == 0


def test_v50_control_always_preserves_incumbent_at_rsmr_proposal() -> None:
    s = SelectedOutcomeProbeState()
    out, d = apply_selected_outcome_probe(7, _diag(), _cfg("control"), s)
    assert out == 2
    assert d["first_proposal_now"] is True
    assert d["intervention_executed"] is False
    assert s.executed_intervention_count == 0


def test_v50_treatment_executes_exact_frozen_winner_once_then_incumbent() -> None:
    s = SelectedOutcomeProbeState()
    out1, d1 = apply_selected_outcome_probe(7, _diag(), _cfg("treatment"), s)
    out2, d2 = apply_selected_outcome_probe(7, _diag(), _cfg("treatment"), s)
    assert out1 == 7 and d1["intervention_executed"] is True
    assert out2 == 2 and d2["intervention_executed"] is False
    assert s.executed_intervention_count == 1
    s.reset()
    assert s.executed_intervention_count == 0 and s.first_proposal_seen is False


def test_v50_treatment_fail_closes_if_proposal_is_not_actual_frozen_selected_action() -> None:
    s = SelectedOutcomeProbeState()
    with pytest.raises(ValueError, match="not the frozen RSMR selected action"):
        apply_selected_outcome_probe(2, _diag(proposal=7, baseline=2, selected=7), _cfg("treatment"), s)


def test_v50_probe_requires_no_fallback() -> None:
    s = SelectedOutcomeProbeState()
    c = _cfg("control"); c["fallback"]["enabled"] = True
    with pytest.raises(ValueError, match="fallback.enabled=false"):
        apply_selected_outcome_probe(7, _diag(), c, s)


def test_v50_probe_config_is_evidence_collection_only() -> None:
    src = {
        "fallback": {"enabled": True},
        "runtime": {"decisive_frontier_value": {"incumbent_contrastive_extremal_recovery": {
            "enabled": True,
            "selection_conditioned_intervention_recovery": {"post_selection_value_enabled": False},
        }}},
        "metadata": {}, "provenance": {}, "experiment": {},
    }
    c = _make(src, "control"); t = _make(src, "treatment")
    assert c["fallback"]["enabled"] is False and t["fallback"]["enabled"] is False
    assert c["selected_outcome_probe"]["role"] == "control"
    assert t["selected_outcome_probe"]["role"] == "treatment"
    assert c["metadata"]["selected_outcome_probe_role"] == "control"
    assert t["metadata"]["selected_outcome_probe_role"] == "treatment"
    assert src["metadata"] == {}  # config preparation must not mutate the frozen source object
    assert t["selected_outcome_probe"]["proposal_source"] == "frozen_full_set_RSMR"
    assert t["selected_outcome_probe"]["teacher_or_logged_future_inputs"] is False



def test_v50_paired_hard_metrics_fail_closed_on_schema_drift() -> None:
    control = {m: 1.0 for m in HARD_METRICS}
    treatment = dict(control)
    ok, bad = _hard_noninferiority(control, treatment)
    assert ok is True and bad == []
    treatment[HARD_METRICS[0]] = 0.0
    ok, bad = _hard_noninferiority(control, treatment)
    assert ok is False and bad == [HARD_METRICS[0]]
    missing = dict(treatment); missing.pop(HARD_METRICS[-1])
    with pytest.raises(RuntimeError, match="missing preregistered hard metrics"):
        _hard_noninferiority(control, missing)

def test_v50_fit_hard_locks_v49_preregistered_failure(tmp_path: Path) -> None:
    p = tmp_path / "v49.json"
    p.write_text(json.dumps({"nested_crossfit": {
        "train_gate_pass": False,
        "failure_diagnosis": V49_FAILURE,
        "risk_identification": {"siir": {"aggregate_nonpositive_risk_auc": 0.6081222524597028}},
    }}))
    _check_v49(p)
    bad = json.loads(p.read_text()); bad["nested_crossfit"]["train_gate_pass"] = True
    p.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="V49 preregistered offline-family failure signature changed"):
        _check_v49(p)
    assert ALPHA_RET == pytest.approx(0.0779185520361991)


def test_v50_launcher_is_closed_loop_evidence_source_and_stops_before_fresh_on_fit_failure() -> None:
    p = ROOT / "RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh"
    if not p.exists():
        pytest.skip("launcher generated after focused unit layer")
    text = p.read_text()
    assert "run_v64_3_50_paired_selected_outcome_collection" in text
    assert "closed_loop_reactive_agents" in text
    assert "fit_v64_3_50_eaf_icer_sior" in text
    assert "[[ $FIT_STATUS -eq 0 ]] || exit \"$FIT_STATUS\"" in text
    assert "select_fresh_preprocessed_tokens" not in text
