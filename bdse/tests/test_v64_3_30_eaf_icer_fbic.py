from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from bdse.planner.full_bank_capacity_probe import full_bank_capacity_probe


def test_fbic_exposes_complete_already_queried_bank_when_budget_feasible() -> None:
    out = full_bank_capacity_probe(
        baseline_selected=list(range(16)),
        reference_atoms=list(range(24)),
        atom_budget_costs=np.ones(24, dtype=np.float32),
        budget=24.0,
        structural_domain=False,
        expected_top_m=24,
    )
    assert out.selected == list(range(24))
    assert out.diagnostics["full_bank_capacity_probe_applied"] == 1.0
    assert out.diagnostics["full_bank_capacity_probe_added_atom_count"] == 8.0
    assert out.diagnostics["full_bank_capacity_probe_no_new_query"] == 1.0


def test_fbic_is_strict_noop_in_structural_domain() -> None:
    baseline = list(range(16))
    out = full_bank_capacity_probe(
        baseline_selected=baseline,
        reference_atoms=list(range(24)),
        atom_budget_costs=np.ones(24, dtype=np.float32),
        budget=24.0,
        structural_domain=True,
        expected_top_m=24,
    )
    assert out.selected == baseline
    assert out.diagnostics["full_bank_capacity_probe_applied"] == 0.0
    assert out.diagnostics["full_bank_capacity_probe_reason_code"] == 1.0


def test_fbic_fails_closed_when_full_bank_exceeds_interface_budget() -> None:
    baseline = list(range(16))
    out = full_bank_capacity_probe(
        baseline_selected=baseline,
        reference_atoms=list(range(24)),
        atom_budget_costs=np.ones(24, dtype=np.float32),
        budget=23.0,
        structural_domain=False,
        expected_top_m=24,
    )
    assert out.selected == baseline
    assert out.diagnostics["full_bank_capacity_probe_applied"] == 0.0
    assert out.diagnostics["full_bank_capacity_probe_reason_code"] == 4.0


def test_fbic_fails_closed_on_duplicate_or_invalid_reference() -> None:
    baseline = list(range(16))
    out = full_bank_capacity_probe(
        baseline_selected=baseline,
        reference_atoms=list(range(23)) + [22],
        atom_budget_costs=np.ones(24, dtype=np.float32),
        budget=24.0,
        structural_domain=False,
        expected_top_m=24,
    )
    assert out.selected == baseline
    assert out.diagnostics["full_bank_capacity_probe_reason_code"] == 3.0



def test_fbic_fails_closed_if_capacity_ceiling_would_reallocate_baseline_atoms() -> None:
    baseline = list(range(15)) + [30]
    out = full_bank_capacity_probe(
        baseline_selected=baseline,
        reference_atoms=list(range(24)),
        atom_budget_costs=np.ones(32, dtype=np.float32),
        budget=24.0,
        structural_domain=False,
        expected_top_m=24,
    )
    assert out.selected == baseline
    assert out.diagnostics["full_bank_capacity_probe_applied"] == 0.0
    assert out.diagnostics["full_bank_capacity_probe_reason_code"] == 7.0


def test_fbic_fails_closed_on_invalid_baseline_contract() -> None:
    baseline = list(range(15)) + [14]
    out = full_bank_capacity_probe(
        baseline_selected=baseline,
        reference_atoms=list(range(24)),
        atom_budget_costs=np.ones(24, dtype=np.float32),
        budget=24.0,
        structural_domain=False,
        expected_top_m=24,
    )
    assert out.selected == baseline
    assert out.diagnostics["full_bank_capacity_probe_reason_code"] == 8.0

def test_v30_config_is_one_point_capacity_ceiling_not_fcr_or_budget_sweep() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/v64_3_30_eaf_icer_fbic_v20.yaml").read_text())
    assert cfg["evidence"]["budget"] == 16
    assert cfg["selector"]["proposal_top_m"] == 24
    assert cfg["selector"]["min_selected_atoms"] == 16
    fbic = cfg["selector"]["full_bank_capacity_probe"]
    assert fbic["enabled"] is True
    assert fbic["baseline_selector_budget"] == 16
    assert fbic["interface_budget"] == 24
    assert fbic["additional_evidence_queries"] == 0
    assert "frontier_contrast_rebinding" not in cfg["selector"]
    assert cfg["fallback"]["budget_stages"] == [16]
    assert cfg["metadata"]["fbic_full_bank_capacity_probe"] is True


def test_fbic_query_accounting_separates_upstream_b16_from_retained_b24() -> None:
    from bdse.planner.nuplan_planner import runtime_query_diagnostics

    pred = {
        "top_m_atoms": list(range(24)),
        "queried_actions": [0, 1],
        "action_atom_query_count": 48,
        "unique_pair_atom_query_count": 0,
        "configured_decision_budget_atom_count": 24,
        "upstream_configured_decision_budget_atom_count": 16,
    }
    diag = runtime_query_diagnostics(pred, list(range(24)))
    assert diag["configured_decision_budget_atom_count"] == 24
    assert diag["upstream_configured_decision_budget_atom_count"] == 16
    assert diag["decision_budget_atom_count"] == 24
    assert diag["retained_interface_atom_budget_pass"] == 1.0
