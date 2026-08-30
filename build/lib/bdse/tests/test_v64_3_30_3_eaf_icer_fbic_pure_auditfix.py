from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from bdse.tools.check_v64_3_30_3_eaf_icer_fbic_pure_split import _fbic_integrity_diag


def _row(*, code: int, applied: int, baseline: int, reference: int, final: int, added: int = 0) -> dict:
    return {
        'selector_full_bank_capacity_probe_reason_code': float(code),
        'selector_full_bank_capacity_probe_applied': float(applied),
        'selector_full_bank_capacity_probe_baseline_count': float(baseline),
        'selector_full_bank_capacity_probe_reference_count': float(reference),
        'selector_full_bank_capacity_probe_final_count': float(final),
        'selector_full_bank_capacity_probe_added_atom_count': float(added),
        'selector_full_bank_capacity_probe_removed_atom_count': 0.0,
        'selector_full_bank_capacity_probe_enabled': 1.0,
        'selector_full_bank_capacity_probe_no_new_query': 1.0,
        'upstream_configured_decision_budget_atom_count': 16.0,
        'configured_decision_budget_atom_count': 24.0,
        'retained_interface_atom_budget_pass': 1.0,
    }


def test_reason6_is_valid_noop_not_engineering_failure() -> None:
    rows = {
        'expand': _row(code=0, applied=1, baseline=16, reference=24, final=24, added=8),
        'noexpand': _row(code=6, applied=0, baseline=11, reference=11, final=11),
        'structural': _row(code=1, applied=0, baseline=16, reference=24, final=16),
    }
    d = _fbic_integrity_diag(rows, {'expand', 'noexpand'}, {'structural'})
    assert d['pointwise_integrity_valid'] is True
    assert d['safe_expandable_rate'] == 0.5
    assert d['safe_applied_given_expandable_rate'] == 1.0
    assert d['safe_reason6_noop_valid_rate'] == 1.0
    assert d['structural_reason1_noop_valid_rate'] == 1.0


def test_invalid_subset_fallback_remains_engineering_failure() -> None:
    rows = {'bad': _row(code=7, applied=0, baseline=16, reference=24, final=16)}
    d = _fbic_integrity_diag(rows, {'bad'}, set())
    assert d['pointwise_integrity_valid'] is False
    assert d['safe_contract_violation_count'] == 1


def test_double_screen_requires_new_integrity_and_exposure_fields(tmp_path: Path) -> None:
    train = {
        'engineering_contract_valid': True,
        'historical_B16_V25_reproduced': True,
        'B24_DRC_fail_is_selected_path_fold_safety_failure_not_runtime_error': True,
    }
    split = {
        'fbic_contract_valid': True,
        'fbic_integrity_valid': True,
        'capacity_exposure_adequate': True,
        'structural_preservation_valid': True,
        'pure_capacity_capture_signal': False,
        'capacity_action_switch_nonharmful': True,
        'capacity_common_B16_opportunity_nonharmful': True,
        'endpoint_noninferior': True,
    }
    pa, pb, pt, po = [tmp_path / x for x in ['A.json', 'B.json', 'T.json', 'O.json']]
    pa.write_text(json.dumps(split)); pb.write_text(json.dumps(split)); pt.write_text(json.dumps(train))
    subprocess.run([
        sys.executable, '-m', 'bdse.tools.check_v64_3_30_3_eaf_icer_fbic_pure_screen',
        '--split-a-report', str(pa), '--split-b-report', str(pb), '--train-audit', str(pt), '--output', str(po),
    ], check=True, capture_output=True, text=True)
    out = json.loads(po.read_text())
    assert out['engineering_valid'] is True
    assert out['fbic_integrity_valid_both'] is True
    assert out['scientific_conclusion'] == 'retained_capacity_is_not_the_reproducible_first_order_missing_mediator_under_current_frozen_consumer'


def test_v30_3_launcher_uses_new_seed_and_excludes_spent_v30_2_fresh() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / 'RUN_V64_3_30_3_EAF_ICER_FBIC_PURE_AUDITFIX_SCREEN_2GPU.sh').read_text()
    assert 'v64.3.30.3-eaf-icer-fbic-pure-auditfix-double-fresh-v1' in text
    assert 'v64_3_30_3_design_exclude_v64_3_30_2_screen_tokens.txt' in text
    assert '9700' in text
    assert 'check_v64_3_30_3_eaf_icer_fbic_pure_split' in text
    assert 'B24_DRC=' not in text
