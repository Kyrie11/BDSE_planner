from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from bdse.tools.check_v64_3_30_2_eaf_icer_fbic_pure_split import (
    _b16_common_opportunity_diag,
    _capacity_switch_diag,
)


def test_capacity_switch_diag_uses_teacher_regret_direction() -> None:
    base = {
        'a': {'bdse_action': 1, 'teacher_regret': 10.0},
        'b': {'bdse_action': 2, 'teacher_regret': 5.0},
        'c': {'bdse_action': 3, 'teacher_regret': 7.0},
    }
    cap = {
        'a': {'bdse_action': 4, 'teacher_regret': 4.0},   # beneficial +6
        'b': {'bdse_action': 5, 'teacher_regret': 8.0},   # harmful -3
        'c': {'bdse_action': 3, 'teacher_regret': 1.0},   # same action ignored
    }
    d = _capacity_switch_diag(base, cap, set(base))
    assert d['changed_action_count'] == 2
    assert d['beneficial_change_count'] == 1
    assert d['harmful_change_count'] == 1
    assert d['B16_minus_B24_teacher_regret_sum'] == 3.0
    assert d['net_nonharmful'] is True


def test_common_B16_opportunity_is_paired_on_fixed_B16_definition(tmp_path: Path) -> None:
    p = tmp_path / 'edges.jsonl'
    rows = [
        {'scenario_token': 's', 'anchor_action': 0, 'raw_top_action': 1, 'challenger_action': 0, 'icer_admissible': 1.0, 'teacher_margin': 0.0},
        {'scenario_token': 's', 'anchor_action': 0, 'raw_top_action': 1, 'challenger_action': 1, 'icer_admissible': 1.0, 'teacher_margin': 0.2},
        {'scenario_token': 's', 'anchor_action': 0, 'raw_top_action': 1, 'challenger_action': 2, 'icer_admissible': 1.0, 'teacher_margin': 0.5},
    ]
    p.write_text('\n'.join(json.dumps(x) for x in rows) + '\n')
    b16 = {'s': {'teacher_regret': 10.0}}
    b24 = {'s': {'teacher_regret': 8.0}}
    d = _b16_common_opportunity_diag(str(p), b16, b24, {'s'})
    assert d['B16_defined_positive_opportunity_scene_count'] == 1
    assert d['B24_better_final_action_count_on_B16_opportunities'] == 1
    assert d['paired_nonharmful_on_B16_opportunities'] is True


def test_double_screen_branching(tmp_path: Path) -> None:
    train = {
        'engineering_contract_valid': True,
        'historical_B16_V25_reproduced': True,
        'B24_DRC_fail_is_selected_path_fold_safety_failure_not_runtime_error': True,
    }
    split = {
        'fbic_contract_valid': True,
        'structural_preservation_valid': True,
        'pure_capacity_capture_signal': False,
        'capacity_action_switch_nonharmful': True,
        'capacity_common_B16_opportunity_nonharmful': True,
        'endpoint_noninferior': True,
    }
    pa, pb, pt, po = [tmp_path / x for x in ['A.json','B.json','T.json','O.json']]
    pa.write_text(json.dumps(split)); pb.write_text(json.dumps(split)); pt.write_text(json.dumps(train))
    subprocess.run([
        sys.executable, '-m', 'bdse.tools.check_v64_3_30_2_eaf_icer_fbic_pure_screen',
        '--split-a-report', str(pa), '--split-b-report', str(pb), '--train-audit', str(pt), '--output', str(po)
    ], check=True, capture_output=True, text=True)
    out = json.loads(po.read_text())
    assert out['engineering_valid'] is True
    assert out['pure_capacity_capture_signal_both'] is False
    assert out['scientific_conclusion'] == 'retained_capacity_is_not_the_reproducible_first_order_missing_mediator_under_current_frozen_consumer'


def test_v30_2_launcher_keeps_same_fresh_seed_and_has_no_B24_DRC_fresh_arm() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / 'RUN_V64_3_30_2_EAF_ICER_FBIC_PURE_SCREEN_2GPU.sh').read_text()
    assert 'v64.3.30-eaf-icer-fbic-double-fresh-v1' in text
    assert 'B24_DRC=' not in text
    assert 'b24_drc' not in text
    assert 'raw/B16-V20/B24-V20' in text
