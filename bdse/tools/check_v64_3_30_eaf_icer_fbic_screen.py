from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Combine the two independent V64.3.30 FBIC fresh blocks.')
    ap.add_argument('--split-a-report', required=True)
    ap.add_argument('--split-b-report', required=True)
    ap.add_argument('--b24-train-fit-report', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    A = json.load(open(a.split_a_report, encoding='utf-8'))
    B = json.load(open(a.split_b_report, encoding='utf-8'))
    T = json.load(open(a.b24_train_fit_report, encoding='utf-8'))

    contracts = bool(A.get('fbic_contract_valid') and B.get('fbic_contract_valid'))
    preservation = bool(A.get('structural_preservation_valid') and B.get('structural_preservation_valid') and A.get('incumbent_default_invariant') and B.get('incumbent_default_invariant'))
    pure_both = bool(A.get('pure_capacity_signal') and B.get('pure_capacity_signal'))
    safe_both = bool(A.get('safe_DRC_coverage_gain') and B.get('safe_DRC_coverage_gain'))
    tail_both = bool(A.get('B24_selected_catastrophe_free') and B.get('B24_selected_catastrophe_free') and A.get('B24_selected_tail_noninferior') and B.get('B24_selected_tail_noninferior'))
    endpoint_both = bool(A.get('endpoint_noninferior') and B.get('endpoint_noninferior'))

    if not contracts or not preservation:
        conclusion = 'engineering_or_preservation_failure'
        next_action = 'STOP_fix_capacity_contract_do_not_interpret_B24'
    elif pure_both and safe_both and tail_both and endpoint_both:
        conclusion = 'B16_retained_interface_capacity_bottleneck_supported_on_double_fresh'
        next_action = 'design_V31_adaptive_bounded_candidate_conditioned_completion_targeting_B24_signal_with_lower_average_interface_and_same_monotone_safety_contracts'
    elif pure_both and not safe_both:
        conclusion = 'capacity_signal_exists_but_DRC_is_the_dominant_unsafe_consumer'
        next_action = 'freeze_query_and_capacity_stop_selector_tuning_design_candidate_conditioned_recovery_reliability_using_capacity_complete_evidence'
    elif not pure_both:
        conclusion = 'retained_interface_capacity_is_not_the_reproducible_dominant_bottleneck'
        next_action = 'stop_budget_sweeps_and_same_bank_rebinding_focus_on_candidate_conditioned_counterfactual_recovery_semantics_and_operator_mismatch'
    else:
        conclusion = 'capacity_effect_not_reproduced_consistently_across_splits'
        next_action = 'stop_budget_tuning_use_split_gate_decomposition_to_localize_the_nonstationary_recovery_stage'

    out = {
        'audit': 'v64_3_30_eaf_icer_fbic_double_fresh',
        'train_gate_pass': bool(T.get('train_gate_pass', False)),
        'fbic_contract_both': contracts,
        'preservation_contract_both': preservation,
        'pure_capacity_signal_both': pure_both,
        'safe_DRC_capacity_gain_both': safe_both,
        'tail_safe_both': tail_both,
        'endpoint_noninferior_both': endpoint_both,
        'scientific_conclusion': conclusion,
        'next_action': next_action,
        'split_A': A,
        'split_B': B,
        'interpretation': 'This screen is diagnostic by design: even a double-fresh B24 win is evidence for a capacity mediator, not a paper claim that B=24 is the algorithm. The allowed next step after a clean win is an adaptive bounded interface; after a clean failure, further B sweeps or same-bank selector objectives are disallowed.',
    }
    p = Path(a.output); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({k: out[k] for k in ['train_gate_pass','fbic_contract_both','preservation_contract_both','pure_capacity_signal_both','safe_DRC_capacity_gain_both','tail_safe_both','endpoint_noninferior_both','scientific_conclusion','next_action']}, indent=2))
    if not bool(T.get('train_gate_pass', False)):
        raise SystemExit('STOP TRAIN: unchanged aggregate DRC recipe is not stable on the capacity-complete TRAIN representation')


if __name__ == '__main__':
    main()
