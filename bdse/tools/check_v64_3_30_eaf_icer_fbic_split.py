from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _f, _icer_edge_diag, _metric_pack
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _guard_block_rate, _identity_rate, _load_rows, _path_diag
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _replacement_tail_diag, _edge_groups


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float('nan')


def _positive_count(tail: dict[str, Any]) -> int:
    count = int(tail.get('count', 0))
    precision = float(tail.get('teacher_positive_precision', float('nan')))
    return int(round(count * precision)) if math.isfinite(precision) else 0


def _fbic_diag(rows: dict[str, dict[str, Any]], safe: set[str], all_flagged: set[str]) -> dict[str, Any]:
    vals = list(rows.values())
    safe_rows = [rows[t] for t in safe]
    structural_rows = [rows[t] for t in all_flagged]
    reasons: dict[str, int] = {}
    for r in vals:
        code = str(int(round(_f(r, 'selector_full_bank_capacity_probe_reason_code', -1))))
        reasons[code] = reasons.get(code, 0) + 1
    return {
        'scene_count': len(vals),
        'enabled_rate': _mean([float(_f(r, 'selector_full_bank_capacity_probe_enabled', 0) >= 0.5) for r in vals]),
        'safe_applied_rate': _mean([float(_f(r, 'selector_full_bank_capacity_probe_applied', 0) >= 0.5) for r in safe_rows]),
        'structural_applied_rate': _mean([float(_f(r, 'selector_full_bank_capacity_probe_applied', 0) >= 0.5) for r in structural_rows]),
        'safe_baseline_count_mean': _mean([_f(r, 'selector_full_bank_capacity_probe_baseline_count') for r in safe_rows]),
        'safe_reference_count_mean': _mean([_f(r, 'selector_full_bank_capacity_probe_reference_count') for r in safe_rows]),
        'safe_final_count_mean': _mean([_f(r, 'selector_full_bank_capacity_probe_final_count') for r in safe_rows]),
        'safe_added_atom_count_mean': _mean([_f(r, 'selector_full_bank_capacity_probe_added_atom_count', 0.0) for r in safe_rows]),
        'safe_removed_atom_count_mean': _mean([_f(r, 'selector_full_bank_capacity_probe_removed_atom_count', 0.0) for r in safe_rows]),
        'upstream_configured_budget_mean': _mean([_f(r, 'upstream_configured_decision_budget_atom_count') for r in vals]),
        'retained_interface_configured_budget_mean': _mean([_f(r, 'configured_decision_budget_atom_count') for r in vals]),
        'retained_interface_budget_pass_rate': _mean([float(_f(r, 'retained_interface_atom_budget_pass', 0) >= 0.5) for r in vals]),
        'no_new_query_rate': _mean([float(_f(r, 'selector_full_bank_capacity_probe_no_new_query', 0) >= 0.5) for r in vals]),
        'reason_code_counts': reasons,
    }


def _query_diag(base_rows: dict[str, dict[str, Any]], cap_rows: dict[str, dict[str, Any]], tokens: set[str]) -> dict[str, Any]:
    keys = ['action_atom_query_count', 'proposal_candidate_atom_count', 'effective_query_action_count']
    out: dict[str, Any] = {}
    parity = True
    for k in keys:
        a = [_f(base_rows[t], k) for t in tokens]
        b = [_f(cap_rows[t], k) for t in tokens]
        finite = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
        equal = bool(finite) and all(abs(x-y) <= 1e-9 for x, y in finite)
        out[k] = {'b16_mean': _mean([x for x, _ in finite]), 'b24_mean': _mean([y for _, y in finite]), 'exact_scene_parity': equal}
        parity = parity and equal
    out['all_query_counts_exact_scene_parity'] = parity
    return out


def _gate_decomposition(edge_path: str, safe: set[str]) -> dict[str, int | float]:
    groups = _edge_groups(edge_path, safe)
    opp = support = scalar = both = drc = selected = 0
    for rows in groups.values():
        if not rows:
            continue
        anchor = int(rows[0].get('anchor_action', -1))
        incumbent = int(rows[0].get('raw_top_action', -1))
        sel = int(rows[0].get('icer_selected_action', incumbent))
        by = {int(r.get('challenger_action', -999)): r for r in rows}
        inc = by.get(incumbent)
        if inc is None or _f(inc, 'icer_admissible', 0.0) < 0.5:
            continue
        inc_tm = _f(inc, 'teacher_margin', 0.0)
        threshold = max(0.0, inc_tm)
        positives = [
            r for r in rows
            if _f(r, 'icer_admissible', 0.0) >= 0.5
            and int(r.get('challenger_action', -999)) not in {anchor, incumbent}
            and _f(r, 'teacher_margin', -float('inf')) > threshold
        ]
        if not positives:
            continue
        opp += 1
        s = any(_f(r, 'icer_support_logit', -float('inf')) > 0.0 for r in positives)
        d = any(_f(r, 'icer_scalar_dominance_logit', -float('inf')) > 0.0 for r in positives)
        sd = any(_f(r, 'icer_support_logit', -float('inf')) > 0.0 and _f(r, 'icer_scalar_dominance_logit', -float('inf')) > 0.0 for r in positives)
        rr = any(
            _f(r, 'icer_support_logit', -float('inf')) > 0.0
            and _f(r, 'icer_scalar_dominance_logit', -float('inf')) > 0.0
            and _f(r, 'icer_replacement_regret_risk_logit', -float('inf')) > 0.0
            for r in positives
        )
        support += int(s); scalar += int(d); both += int(sd); drc += int(rr)
        sr = by.get(sel)
        selected += int(sr is not None and sel not in {anchor, incumbent} and _f(sr, 'teacher_margin', -float('inf')) > threshold)
    return {
        'direct_positive_opportunity_scenes': opp,
        'support_positive_scenes': support,
        'scalar_positive_scenes': scalar,
        'support_and_scalar_positive_scenes': both,
        'support_scalar_drc_positive_scenes': drc,
        'selected_positive_recovery_scenes': selected,
        'selected_positive_recovery_capture': float(selected / opp) if opp else float('nan'),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='One untouched V64.3.30 B16-vs-B24 FBIC capacity-ceiling block.')
    ap.add_argument('--split-name', required=True)
    for name in ['raw', 'b16-v20', 'b16-drc', 'b24-v20', 'b24-drc']:
        ap.add_argument(f'--{name}-metrics', required=True)
        ap.add_argument(f'--{name}-rows', required=True)
        if name != 'raw':
            ap.add_argument(f'--{name}-edges', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()

    cli = ['raw', 'b16-v20', 'b16-drc', 'b24-v20', 'b24-drc']
    tags = [x.replace('-', '_') for x in cli]
    metrics = {t: json.load(open(getattr(a, t + '_metrics'), encoding='utf-8')) for t in tags}
    rows = {t: _load_rows(getattr(a, t + '_rows')) for t in tags}
    tokens = set(rows['raw'])
    if len(tokens) != 500 or any(set(rows[t]) != tokens for t in tags[1:]):
        raise SystemExit('STOP DATA: every FBIC arm must contain the exact same 500 scenes')

    all_flagged = {t for t in tokens if _f(rows['raw'][t], 'all_actions_safety_flagged_rate', 0.0) >= 0.5}
    safe = tokens - all_flagged
    edge_paths = {t: getattr(a, t + '_edges') for t in tags[1:]}
    edge = {t: _icer_edge_diag(Path(edge_paths[t]), safe) for t in tags[1:]}
    gates = {t: _gate_decomposition(edge_paths[t], safe) for t in tags[1:]}
    path = {t: _path_diag(rows['raw'], rows[t], edge_paths[t], safe, all_flagged) for t in tags[1:]}
    tails = {t: _replacement_tail_diag(rows['raw'], rows[t], edge_paths[t], safe) for t in ['b16_drc', 'b24_drc']}
    M = {t: _metric_pack(metrics[t]) for t in tags}
    fbic = {'b24_v20': _fbic_diag(rows['b24_v20'], safe, all_flagged), 'b24_drc': _fbic_diag(rows['b24_drc'], safe, all_flagged)}
    query = {
        'v20': _query_diag(rows['b16_v20'], rows['b24_v20'], tokens),
        'drc': _query_diag(rows['b16_drc'], rows['b24_drc'], tokens),
    }

    fbic_contract = bool(
        fbic['b24_v20']['enabled_rate'] == 1.0
        and fbic['b24_drc']['enabled_rate'] == 1.0
        and fbic['b24_v20']['safe_applied_rate'] >= 0.90
        and fbic['b24_drc']['safe_applied_rate'] >= 0.90
        and (not all_flagged or fbic['b24_v20']['structural_applied_rate'] == 0.0)
        and (not all_flagged or fbic['b24_drc']['structural_applied_rate'] == 0.0)
        and fbic['b24_v20']['safe_final_count_mean'] >= fbic['b24_v20']['safe_baseline_count_mean'] + 4.0
        and fbic['b24_drc']['safe_final_count_mean'] >= fbic['b24_drc']['safe_baseline_count_mean'] + 4.0
        and abs(fbic['b24_v20']['safe_removed_atom_count_mean']) <= 1e-12
        and abs(fbic['b24_drc']['safe_removed_atom_count_mean']) <= 1e-12
        and abs(fbic['b24_v20']['upstream_configured_budget_mean'] - 16.0) <= 1e-12
        and abs(fbic['b24_drc']['upstream_configured_budget_mean'] - 16.0) <= 1e-12
        and abs(fbic['b24_v20']['retained_interface_configured_budget_mean'] - 24.0) <= 1e-12
        and abs(fbic['b24_drc']['retained_interface_configured_budget_mean'] - 24.0) <= 1e-12
        and fbic['b24_v20']['retained_interface_budget_pass_rate'] == 1.0
        and fbic['b24_drc']['retained_interface_budget_pass_rate'] == 1.0
        and query['v20']['all_query_counts_exact_scene_parity']
        and query['drc']['all_query_counts_exact_scene_parity']
    )

    structural = {
        'all_flagged_scene_count': len(all_flagged),
        'b24_drc_all_flagged_final_identity_vs_raw': _identity_rate(rows['b24_drc'], rows['raw'], all_flagged),
        'b24_drc_all_flagged_delegation_rate': (
            sum(_f(rows['b24_drc'][t], 'decisive_frontier_icer_structural_domain_delegated', 0.0) >= 0.5 for t in all_flagged) / len(all_flagged)
            if all_flagged else float('nan')
        ),
        'b24_drc_safe_guard_block_rate': _guard_block_rate(rows['b24_drc'], safe),
    }
    structural_ok = bool(
        (not all_flagged or structural['b24_drc_all_flagged_final_identity_vs_raw'] == 1.0)
        and (not all_flagged or structural['b24_drc_all_flagged_delegation_rate'] == 1.0)
        and structural['b24_drc_safe_guard_block_rate'] <= 0.001
    )
    asymmetric = bool(
        path['b24_drc']['admissible_incumbent_to_anchor']['count'] == 0
        and abs(path['b24_drc']['admissible_incumbent_to_anchor']['regret_delta_sum']) <= 1e-9
    )

    pure_gain = float(edge['b24_v20']['direct_incumbent_opportunity_capture_rate'] - edge['b16_v20']['direct_incumbent_opportunity_capture_rate'])
    drc_gain = float(edge['b24_drc']['direct_incumbent_opportunity_capture_rate'] - edge['b16_drc']['direct_incumbent_opportunity_capture_rate'])
    b16_tail, b24_tail = tails['b16_drc'], tails['b24_drc']
    b16_pos, b24_pos = _positive_count(b16_tail), _positive_count(b24_tail)
    catastrophe_free = bool(math.isfinite(b24_tail['teacher_improvement_worst']) and b24_tail['teacher_improvement_worst'] > -0.5)
    tail_noninferior = bool(
        math.isfinite(b24_tail['teacher_negative_rms']) and math.isfinite(b16_tail['teacher_negative_rms'])
        and b24_tail['teacher_negative_rms'] <= b16_tail['teacher_negative_rms'] + 1e-9
        and b24_tail['teacher_improvement_worst'] >= b16_tail['teacher_improvement_worst'] - 1e-9
        and b24_tail['regret_positive_rms'] <= b16_tail['regret_positive_rms'] + 1e-9
        and b24_tail['worst_regret_increase'] <= b16_tail['worst_regret_increase'] + 1e-9
    )
    safe_coverage_gain = bool(drc_gain >= 0.03 and b24_pos >= b16_pos + 5 and catastrophe_free and tail_noninferior)
    pure_capacity_signal = bool(pure_gain >= 0.03)
    endpoint_noninferior = bool(
        M['b24_drc']['match'] >= M['b16_drc']['match'] - 0.002
        and M['b24_drc']['regret'] <= M['b16_drc']['regret'] * 1.005
        and M['b24_drc']['regret'] <= M['raw']['regret'] * 1.02
    )

    if not fbic_contract:
        diagnosis = 'engineering_capacity_contract_failure'
        next_action = 'STOP_fix_FBIC_interface_or_query_accounting_before_any_new_fresh_interpretation'
    elif not structural_ok or not asymmetric:
        diagnosis = 'preservation_contract_failure'
        next_action = 'STOP_capacity_probe_must_not_change_structural_or_incumbent_preservation_domains'
    elif pure_capacity_signal and safe_coverage_gain and endpoint_noninferior:
        diagnosis = 'retained_interface_capacity_is_a_reproducible_mediator_if_second_block_agrees'
        next_action = 'if_second_block_agrees_design_adaptive_bounded_interface_that_recovers_B24_signal_without_making_B24_the_paper_thesis'
    elif pure_capacity_signal and not safe_coverage_gain:
        diagnosis = 'capacity_exposes_more_recovery_signal_but_current_DRC_cannot_use_it_safely'
        next_action = 'stop_selector_work_redesign_candidate_conditioned_recovery_reliability_on_the_capacity-complete_view'
    elif not pure_capacity_signal:
        diagnosis = 'B16_retained_capacity_not_the_main_missing_mediator'
        next_action = 'stop_budget_and_same_bank_allocation_work_focus_on_candidate_conditioned_counterfactual_recovery_semantics'
    else:
        diagnosis = 'capacity_effect_is_mixed'
        next_action = 'do_not_tune_budget_use_second_independent_block_and_gate_decomposition_to_localize_support_scalar_vs_DRC'

    report = {
        'audit': 'v64_3_30_eaf_icer_fbic_split',
        'split_name': a.split_name,
        'fbic_contract_valid': fbic_contract,
        'structural_preservation_valid': structural_ok,
        'incumbent_default_invariant': asymmetric,
        'pure_B24_capacity_capture_gain_vs_B16_v20': pure_gain,
        'B24_DRC_capture_gain_vs_B16_DRC': drc_gain,
        'pure_capacity_signal': pure_capacity_signal,
        'safe_DRC_coverage_gain': safe_coverage_gain,
        'B24_selected_catastrophe_free': catastrophe_free,
        'B24_selected_tail_noninferior': tail_noninferior,
        'endpoint_noninferior': endpoint_noninferior,
        'diagnosis': diagnosis,
        'next_action': next_action,
        'fbic_diagnostics': fbic,
        'query_accounting': query,
        'structural': structural,
        'gate_decomposition': gates,
        'edge_diagnostics': edge,
        'path_diagnostics': path,
        'selected_replacement_tail_diagnostics': tails,
        'positive_direct_replacements': {'b16_drc': b16_pos, 'b24_drc': b24_pos},
        'metrics': M,
        'preregistered_thresholds': {
            'capacity_probe_safe_applied_rate_min': 0.90,
            'mean_retained_atom_increase_min': 4.0,
            'removed_baseline_atoms_allowed': 0,
            'query_count_change_allowed': 0,
            'upstream_selector_budget_required': 16,
            'retained_interface_ceiling_required': 24,
            'retained_interface_budget_pass_required': 1.0,
            'pure_v20_capture_gain_min': 0.03,
            'safe_drc_capture_gain_min': 0.03,
            'extra_positive_direct_replacements_min': 5,
            'catastrophic_teacher_improvement_threshold': -0.5,
            'tail_noninferiority': 'B24 no worse than B16 on teacher negative RMS, worst teacher improvement, positive-regret RMS and worst regret increase',
            'endpoint_match_tolerance': 0.002,
            'endpoint_regret_tolerance': 0.005,
            'capacity_test': 'single ceiling B16->B24 with M fixed at 24; not a sweep',
        },
        'interpretation': (
            'FBIC is a causal capacity ceiling. Historical AOCC is still constructed at B=16; only the retained downstream interface is expanded to all already-queried M=24 evidence in the safe domain. '
            'If this does not improve pure recovery coverage on untouched data, another B=16 allocation objective is not justified. If it does improve pure coverage but DRC remains unsafe, the next bottleneck is candidate-conditioned recovery semantics rather than evidence acquisition.'
        ),
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
