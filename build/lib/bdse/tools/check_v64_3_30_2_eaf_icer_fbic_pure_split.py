from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _f, _icer_edge_diag, _metric_pack
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _identity_rate, _load_rows, _path_diag
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _edge_groups
from bdse.tools.check_v64_3_30_eaf_icer_fbic_split import _fbic_diag, _gate_decomposition, _query_diag


def _capacity_switch_diag(base: dict[str, dict[str, Any]], cap: dict[str, dict[str, Any]], tokens: set[str]) -> dict[str, Any]:
    gains: list[float] = []
    beneficial = harmful = equal = 0
    for t in sorted(tokens):
        a = int(round(_f(base[t], 'bdse_action', -999.0)))
        b = int(round(_f(cap[t], 'bdse_action', -999.0)))
        if a == b:
            continue
        # Positive gain means capacity arm has lower teacher regret.
        gain = _f(base[t], 'teacher_regret') - _f(cap[t], 'teacher_regret')
        gains.append(gain)
        if gain > 1e-12:
            beneficial += 1
        elif gain < -1e-12:
            harmful += 1
        else:
            equal += 1
    return {
        'changed_action_count': len(gains),
        'beneficial_change_count': beneficial,
        'harmful_change_count': harmful,
        'equal_change_count': equal,
        'B16_minus_B24_teacher_regret_sum': float(sum(gains)),
        'B16_minus_B24_teacher_regret_mean': float(np.mean(gains)) if gains else float('nan'),
        'B24_worst_regret_increase': float(-min(gains)) if gains else float('nan'),
        'B24_best_regret_reduction': float(max(gains)) if gains else float('nan'),
        'net_nonharmful': bool(sum(gains) >= -1e-9),
    }


def _b16_common_opportunity_diag(
    b16_edge_path: str,
    b16_rows: dict[str, dict[str, Any]],
    b24_rows: dict[str, dict[str, Any]],
    safe: set[str],
) -> dict[str, Any]:
    groups = _edge_groups(b16_edge_path, safe)
    opp: list[str] = []
    for token, rs in groups.items():
        if not rs:
            continue
        anchor = int(rs[0].get('anchor_action', -1))
        incumbent = int(rs[0].get('raw_top_action', -1))
        by = {int(r.get('challenger_action', -999)): r for r in rs}
        inc = by.get(incumbent)
        if inc is None or _f(inc, 'icer_admissible', 0.0) < 0.5:
            continue
        threshold = max(0.0, _f(inc, 'teacher_margin', 0.0))
        positive = any(
            _f(r, 'icer_admissible', 0.0) >= 0.5
            and int(r.get('challenger_action', -999)) not in {anchor, incumbent}
            and _f(r, 'teacher_margin', -float('inf')) > threshold
            for r in rs
        )
        if positive:
            opp.append(token)
    gains = [_f(b16_rows[t], 'teacher_regret') - _f(b24_rows[t], 'teacher_regret') for t in opp]
    return {
        'B16_defined_positive_opportunity_scene_count': len(opp),
        'B24_better_final_action_count_on_B16_opportunities': int(sum(x > 1e-12 for x in gains)),
        'B24_worse_final_action_count_on_B16_opportunities': int(sum(x < -1e-12 for x in gains)),
        'B16_minus_B24_teacher_regret_sum_on_B16_opportunities': float(sum(gains)),
        'B16_minus_B24_teacher_regret_mean_on_B16_opportunities': float(np.mean(gains)) if gains else float('nan'),
        'paired_nonharmful_on_B16_opportunities': bool(sum(gains) >= -1e-9),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='One untouched V64.3.30.2 pure FBIC capacity block: raw/B16-V20/B24-V20 only.')
    ap.add_argument('--split-name', required=True)
    for name in ['raw', 'b16-v20', 'b24-v20']:
        ap.add_argument(f'--{name}-metrics', required=True)
        ap.add_argument(f'--{name}-rows', required=True)
        if name != 'raw':
            ap.add_argument(f'--{name}-edges', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()

    cli = ['raw', 'b16-v20', 'b24-v20']
    tags = [x.replace('-', '_') for x in cli]
    metrics = {t: json.load(open(getattr(a, t + '_metrics'), encoding='utf-8')) for t in tags}
    rows = {t: _load_rows(getattr(a, t + '_rows')) for t in tags}
    tokens = set(rows['raw'])
    if len(tokens) != 500 or any(set(rows[t]) != tokens for t in tags[1:]):
        raise SystemExit('STOP DATA: every V30.2 pure-capacity arm must contain the exact same 500 scenes')

    all_flagged = {t for t in tokens if _f(rows['raw'][t], 'all_actions_safety_flagged_rate', 0.0) >= 0.5}
    safe = tokens - all_flagged
    b16_edge = getattr(a, 'b16_v20_edges')
    b24_edge = getattr(a, 'b24_v20_edges')
    edge16 = _icer_edge_diag(Path(b16_edge), safe)
    edge24 = _icer_edge_diag(Path(b24_edge), safe)
    gate16 = _gate_decomposition(b16_edge, safe)
    gate24 = _gate_decomposition(b24_edge, safe)
    path16 = _path_diag(rows['raw'], rows['b16_v20'], b16_edge, safe, all_flagged)
    path24 = _path_diag(rows['raw'], rows['b24_v20'], b24_edge, safe, all_flagged)
    M = {t: _metric_pack(metrics[t]) for t in tags}
    fbic = _fbic_diag(rows['b24_v20'], safe, all_flagged)
    query = _query_diag(rows['b16_v20'], rows['b24_v20'], tokens)
    switch = _capacity_switch_diag(rows['b16_v20'], rows['b24_v20'], safe)
    common = _b16_common_opportunity_diag(b16_edge, rows['b16_v20'], rows['b24_v20'], safe)

    fbic_contract = bool(
        fbic['enabled_rate'] == 1.0
        and fbic['safe_applied_rate'] >= 0.90
        and (not all_flagged or fbic['structural_applied_rate'] == 0.0)
        and fbic['safe_final_count_mean'] >= fbic['safe_baseline_count_mean'] + 4.0
        and abs(fbic['safe_removed_atom_count_mean']) <= 1e-12
        and abs(fbic['upstream_configured_budget_mean'] - 16.0) <= 1e-12
        and abs(fbic['retained_interface_configured_budget_mean'] - 24.0) <= 1e-12
        and fbic['retained_interface_budget_pass_rate'] == 1.0
        and fbic['no_new_query_rate'] == 1.0
        and query['all_query_counts_exact_scene_parity']
    )
    structural = {
        'all_flagged_scene_count': len(all_flagged),
        'B24_all_flagged_final_identity_vs_raw': _identity_rate(rows['b24_v20'], rows['raw'], all_flagged),
        'B24_structural_probe_applied_rate': fbic['structural_applied_rate'],
    }
    structural_ok = bool(
        (not all_flagged or structural['B24_all_flagged_final_identity_vs_raw'] == 1.0)
        and (not all_flagged or structural['B24_structural_probe_applied_rate'] == 0.0)
    )

    capture16 = float(edge16['direct_incumbent_opportunity_capture_rate'])
    capture24 = float(edge24['direct_incumbent_opportunity_capture_rate'])
    pure_gain = capture24 - capture16
    pure_capture_signal = bool(math.isfinite(pure_gain) and pure_gain >= 0.03)
    endpoint_noninferior = bool(
        M['b24_v20']['match'] >= M['b16_v20']['match'] - 0.002
        and M['b24_v20']['regret'] <= M['b16_v20']['regret'] * 1.005
    )
    switch_nonharm = bool(switch['net_nonharmful'])
    common_nonharm = bool(common['paired_nonharmful_on_B16_opportunities'])

    if not fbic_contract:
        diagnosis = 'engineering_capacity_contract_failure'
        next_action = 'STOP_fix_FBIC_contract_before_scientific_interpretation'
    elif not structural_ok:
        diagnosis = 'structural_preservation_failure'
        next_action = 'STOP_capacity_probe_must_be_strict_noop_in_all_flagged_domain'
    elif pure_capture_signal and switch_nonharm and common_nonharm and endpoint_noninferior:
        diagnosis = 'B16_retained_capacity_is_a_useful_mediator_if_second_block_agrees'
        next_action = 'if_second_block_agrees_capacity_signal_exists_but_B24_DRC_ALREADY_failed_TRAIN_so_redesign_candidate_conditioned_consumer_before_any_adaptive_completion'
    elif pure_capture_signal:
        diagnosis = 'capacity_exposes_more_direct_recovery_but_current_downstream_operator_converts_it_unsafely'
        next_action = 'stop_budget_tuning_focus_on_candidate_conditioned_extremal_recovery_semantics_and_selection_calibration'
    else:
        diagnosis = 'capacity_only_transmission_does_not_raise_recovery_under_frozen_downstream_semantics'
        next_action = 'stop_B_sweeps_and_same_bank_selector_work_focus_on_candidate_conditioned_counterfactual_recovery_semantics_operator_mismatch'

    report = {
        'audit': 'v64_3_30_2_eaf_icer_fbic_pure_split',
        'split_name': a.split_name,
        'fbic_contract_valid': fbic_contract,
        'structural_preservation_valid': structural_ok,
        'pure_B24_capacity_capture_gain_vs_B16_v20': pure_gain,
        'pure_capacity_capture_signal': pure_capture_signal,
        'capacity_action_switch_nonharmful': switch_nonharm,
        'capacity_common_B16_opportunity_nonharmful': common_nonharm,
        'endpoint_noninferior': endpoint_noninferior,
        'diagnosis': diagnosis,
        'next_action': next_action,
        'fbic_diagnostics': fbic,
        'query_accounting': query,
        'structural': structural,
        'capacity_action_switch': switch,
        'common_B16_opportunity_paired_effect': common,
        'gate_decomposition': {'b16_v20': gate16, 'b24_v20': gate24},
        'edge_diagnostics': {'b16_v20': edge16, 'b24_v20': edge24},
        'path_diagnostics': {'b16_v20': path16, 'b24_v20': path24},
        'metrics': M,
        'preregistered_thresholds': {
            'capacity_probe_safe_applied_rate_min': 0.90,
            'mean_retained_atom_increase_min': 4.0,
            'removed_baseline_atoms_allowed': 0,
            'query_count_change_allowed': 0,
            'upstream_selector_budget_required': 16,
            'retained_interface_ceiling_required': 24,
            'pure_v20_capture_gain_min': 0.03,
            'endpoint_match_tolerance': 0.002,
            'endpoint_regret_tolerance': 0.005,
            'changed_action_capacity_effect_required_nonharmful': True,
            'B16_defined_opportunity_paired_effect_required_nonharmful': True,
            'capacity_test': 'single ceiling B16->B24 with queried M fixed at 24; no DRC arm because B24 DRC failed the frozen TRAIN fold-safety gate',
        },
        'interpretation': (
            'B=16 is an operating point used to identify the causal mediator, not the paper thesis. '
            'This pure screen asks only whether transmitting the rest of the already queried M=24 bank through the frozen V20 downstream semantics creates reproducible useful recovery signal. '
            'It does not claim B=24 is an algorithm or novelty.'
        ),
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
