from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Combine two untouched V64.3.30.2 pure-capacity blocks.')
    ap.add_argument('--split-a-report', required=True)
    ap.add_argument('--split-b-report', required=True)
    ap.add_argument('--train-audit', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    A = json.load(open(a.split_a_report, encoding='utf-8'))
    B = json.load(open(a.split_b_report, encoding='utf-8'))
    T = json.load(open(a.train_audit, encoding='utf-8'))

    engineering = bool(
        T.get('engineering_contract_valid')
        and T.get('historical_B16_V25_reproduced')
        and T.get('B24_DRC_fail_is_selected_path_fold_safety_failure_not_runtime_error')
        and A.get('fbic_contract_valid') and B.get('fbic_contract_valid')
        and A.get('structural_preservation_valid') and B.get('structural_preservation_valid')
    )
    capture_both = bool(A.get('pure_capacity_capture_signal') and B.get('pure_capacity_capture_signal'))
    switch_both = bool(A.get('capacity_action_switch_nonharmful') and B.get('capacity_action_switch_nonharmful'))
    common_both = bool(A.get('capacity_common_B16_opportunity_nonharmful') and B.get('capacity_common_B16_opportunity_nonharmful'))
    endpoint_both = bool(A.get('endpoint_noninferior') and B.get('endpoint_noninferior'))

    if not engineering:
        conclusion = 'engineering_or_preservation_failure'
        next_action = 'STOP_fix_contract_do_not_interpret_capacity'
    elif capture_both and switch_both and common_both and endpoint_both:
        conclusion = 'B16_retained_capacity_is_a_reproducible_useful_mediator_but_current_DRC_consumer_is_already_TRAIN_falsified_on_B24'
        next_action = 'freeze_capacity_result_do_not_make_B24_the_thesis_design_candidate_conditioned_selection_aware_recovery_on_the_capacity_signal_then_later_compress_adaptively'
    elif capture_both:
        conclusion = 'capacity_exposes_recovery_signal_but_frozen_downstream_operator_is_not_safe_or_endpoint_coherent'
        next_action = 'stop_budget_and_selector_tuning_redesign_candidate_conditioned_extremal_recovery_semantics_selection_calibration'
    else:
        conclusion = 'retained_capacity_is_not_the_reproducible_first_order_missing_mediator_under_current_frozen_consumer'
        next_action = 'stop_B_sweeps_and_same_bank_rebinding_focus_on_candidate_conditioned_counterfactual_recovery_semantics_and_extremal_selection_operator'

    out = {
        'audit': 'v64_3_30_2_eaf_icer_fbic_pure_double_fresh',
        'engineering_valid': engineering,
        'pure_capacity_capture_signal_both': capture_both,
        'capacity_action_switch_nonharmful_both': switch_both,
        'capacity_common_B16_opportunity_nonharmful_both': common_both,
        'endpoint_noninferior_both': endpoint_both,
        'scientific_conclusion': conclusion,
        'next_action': next_action,
        'split_A': A,
        'split_B': B,
        'train_development_audit': T,
        'interpretation': (
            'This is the missing pure capacity mediator test. B=16 remains only the controlled baseline operating point. '
            'No B24 DRC fresh result is produced because the unchanged DRC already failed the frozen TRAIN 5-fold path-safety gate; lowering that gate is forbidden.'
        ),
    }
    p = Path(a.output); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({k: out[k] for k in [
        'engineering_valid', 'pure_capacity_capture_signal_both', 'capacity_action_switch_nonharmful_both',
        'capacity_common_B16_opportunity_nonharmful_both', 'endpoint_noninferior_both', 'scientific_conclusion', 'next_action'
    ]}, indent=2))
    if not engineering:
        raise SystemExit('STOP V30.2: engineering/preservation contract failed')


if __name__ == '__main__':
    main()
