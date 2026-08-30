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



def test_fbic_b24_runtime_attribution_instrumentation_expands_without_touching_b16_schema() -> None:
    from bdse.planner import tournament as tour

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/v64_3_30_eaf_icer_fbic_v20.yaml").read_text())
    assert tour._icer_runtime_attribution_spectrum_budget(cfg) == 24

    contrib = np.arange(24 * 3, dtype=float).reshape(24, 3) + 1.0
    valid = np.ones(3, dtype=bool)
    b24, names24 = tour._icer_attribution_resolved_feature_matrix(
        contrib, valid, 0, 1, budget=tour._icer_runtime_attribution_spectrum_budget(cfg)
    )
    assert b24.shape == (3, 48)
    assert len(names24) == 48
    assert names24[0] == "candidate_atom_signed_spectrum_00"
    assert names24[23] == "candidate_atom_signed_spectrum_23"
    assert names24[24] == "delta_atom_signed_spectrum_00"
    assert names24[-1] == "delta_atom_signed_spectrum_23"
    assert np.count_nonzero(b24[2, :24]) == 24
    assert np.count_nonzero(b24[2, 24:]) == 24
    assert np.isclose(np.abs(b24[2, :24]).sum(), 1.0)
    assert np.isclose(np.abs(b24[2, 24:]).sum(), 1.0)

    # The historical helper default must remain exactly V24's B16 schema.
    b16, names16 = tour._icer_attribution_resolved_feature_matrix(contrib[:16], valid, 0, 1)
    assert b16.shape == (3, 32)
    assert names16 == list(tour._ICER_ATTRIBUTION_RESOLVED_FEATURE_NAMES)


def test_fbic_b24_full_icer_tournament_preflight_no_longer_hits_b16_spectrum_guard() -> None:
    from bdse.planner.tournament import run_pair_conditioned_tournament

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/v64_3_30_eaf_icer_fbic_v20.yaml").read_text())
    K, S, R = 3, 24, 4
    J0 = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    g = np.zeros((S, K), dtype=np.float32)
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    pair_delta = np.zeros((S, 1), dtype=np.float32)
    valid = np.ones(K, dtype=bool)
    safety = np.zeros(K, dtype=bool)
    atom = np.zeros((S, R), dtype=np.float32)
    signed = np.zeros((K, R), dtype=np.float32)
    context = np.zeros((K, R), dtype=np.float32)
    out = run_pair_conditioned_tournament(
        J0, pair_delta, pairs, list(range(S)), valid, safety, cfg,
        predicted_atom_costs=g,
        frontier_value_atom_factors=atom,
        frontier_value_action_signed_factors=signed,
        frontier_value_action_context_factors=context,
        selected_atom_family_ids=np.ones(S, dtype=np.int64),
        selected_atom_type_names=["occupancy"] * S,
    )
    names = out.diagnostics["_decisive_frontier_icer_attribution_resolved_feature_names"]
    mat = np.asarray(out.diagnostics["_decisive_frontier_icer_attribution_resolved_feature_matrix"])
    assert len(names) == 48
    assert mat.shape == (K, 48)


def test_fbic_generated_drc_keeps_evidence_only_risk_and_capacity_probe() -> None:
    import bdse.tools.fit_v64_3_30_eaf_icer_fbic as fit30

    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((root / "configs/v64_3_30_eaf_icer_fbic_v20.yaml").read_text())
    cfg = fit30._v30_cfg(base, {"path": "/tmp/fake.npz", "sha256": "0" * 64}, "downside_rms", "aggregate_downside")
    probe = cfg["selector"]["full_bank_capacity_probe"]
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    assert cfg["evidence"]["budget"] == 16
    assert probe["enabled"] is True and probe["interface_budget"] == 24
    assert ic["regret_risk_enabled"] is True
    assert ic["replacement_regret_risk_enabled"] is True
    assert ic["regret_risk_feature_mode"] == "evidence_only"
    assert ic["regret_risk_model_type"] == "local_multiscale_downside_regret_certificate"


def test_dynamic_b24_attribution_view_has_self_consistent_schema_if_a_diagnostic_uses_it() -> None:
    from bdse.planner import tournament as tour

    K = 3
    feat = np.zeros((K, len(tour._DACER_FEATURE_NAMES)), dtype=float)
    tr = np.zeros((K, 0), dtype=float)
    ar = np.zeros((K, 48), dtype=float)
    names48 = tour._icer_attribution_resolved_feature_names(24)
    x, names = tour._icer_regret_risk_feature_matrix(
        feat, list(tour._DACER_FEATURE_NAMES), tr, [], "attribution_resolved", ar, names48
    )
    assert x.shape == (K, 66)
    assert len(names) == 66
    assert names[-1] == "attribution::delta_atom_signed_spectrum_23"
