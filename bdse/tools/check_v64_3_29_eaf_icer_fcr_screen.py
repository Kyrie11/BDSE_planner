from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Combine two untouched V64.3.29 FCR mechanism blocks.')
    ap.add_argument('--split-a-report', required=True)
    ap.add_argument('--split-b-report', required=True)
    ap.add_argument('--train-fit-report', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    a = json.load(open(args.split_a_report, encoding='utf-8'))
    b = json.load(open(args.split_b_report, encoding='utf-8'))
    tr = json.load(open(args.train_fit_report, encoding='utf-8'))
    train_ok = bool(tr.get('train_gate_pass', False))
    both = bool(a.get('full_split_pass', False) and b.get('full_split_pass', False))
    endpoint_signal = bool(a.get('endpoint_strict_gain_over_V25_DRC', False) or b.get('endpoint_strict_gain_over_V25_DRC', False))
    promote = bool(train_ok and both and endpoint_signal)

    if promote:
        next_action = 'freeze_V29_FCR_and_run_one_independent_full_val_reproduction_before_any_test_or_closed_loop'
    elif not train_ok:
        next_action = 'TRAIN_gate_invalid_stop_before_interpreting_fresh'
    elif not both:
        next_action = 'FCR_not_independently_reproduced_stop_do_not_tune_B_objective_weights_or_acceptance_thresholds'
    else:
        next_action = 'safe_coverage_mechanism_reproduced_but_no_endpoint_signal_stop_before_full_val_and_audit_recovery_ranking'

    report = {
        'audit': 'v64_3_29_eaf_icer_fcr_double_fresh_screen',
        'train_gate_pass': train_ok,
        'split_A_pass': bool(a.get('full_split_pass', False)),
        'split_B_pass': bool(b.get('full_split_pass', False)),
        'both_independent_blocks_pass': both,
        'safe_coverage_gain_both': bool(
            a.get('safe_recovery_coverage_gain_over_V25_DRC', False)
            and b.get('safe_recovery_coverage_gain_over_V25_DRC', False)
        ),
        'tail_noninferior_both': bool(
            a.get('selected_tail_noninferior_to_V25_DRC', False)
            and b.get('selected_tail_noninferior_to_V25_DRC', False)
        ),
        'catastrophe_free_both': bool(
            a.get('selected_replacement_catastrophe_free', False)
            and b.get('selected_replacement_catastrophe_free', False)
        ),
        'fcr_monotone_contract_both': bool(
            a.get('fcr_monotone_interface_contract', False)
            and b.get('fcr_monotone_interface_contract', False)
        ),
        'endpoint_noninferior_both': bool(
            a.get('endpoint_noninferior_to_V25_DRC', False)
            and b.get('endpoint_noninferior_to_V25_DRC', False)
        ),
        'endpoint_strict_signal_at_least_one_block': endpoint_signal,
        'full_promotion_to_independent_full_val_reproduction': promote,
        'test_or_closed_loop_allowed': False,
        'next_action': next_action,
        'split_A_next_action': a.get('next_action'),
        'split_B_next_action': b.get('next_action'),
        'interpretation': (
            'V29 is a mechanism screen for fixed-budget decision-evidence sufficiency. Promotion requires the deterministic FCR invariants, materially higher safe direct-recovery coverage, '
            'and a non-inferior selected tail with zero selected catastrophic replacement (teacher improvement <= -0.5) independently on both untouched 500-scene blocks, plus at least one endpoint signal. Literal B remains frozen; this screen does not tune B.'
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
