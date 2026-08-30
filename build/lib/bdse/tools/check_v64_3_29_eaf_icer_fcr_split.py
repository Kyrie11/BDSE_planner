from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _icer_edge_diag, _metric_pack, _f
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _load_rows, _path_diag, _guard_block_rate, _identity_rate
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _replacement_tail_diag


def _fcr_diag(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    vals = list(rows.values())
    n = len(vals)
    accepted = [r for r in vals if _f(r, 'selector_frontier_contrast_rebinding_accepted', 0.0) >= 0.5]
    enabled_rate = sum(_f(r, 'selector_frontier_contrast_rebinding_enabled', 0.0) >= 0.5 for r in vals) / max(n, 1)
    attempted_rate = sum(_f(r, 'selector_frontier_contrast_rebinding_attempted', 0.0) >= 0.5 for r in vals) / max(n, 1)

    error_nonincrease = 0
    finite_error = 0
    reductions: list[float] = []
    rms_reductions: list[float] = []
    for r in vals:
        bi = _f(r, 'selector_frontier_contrast_rebinding_baseline_linf_error')
        fi = _f(r, 'selector_frontier_contrast_rebinding_final_linf_error')
        br = _f(r, 'selector_frontier_contrast_rebinding_baseline_rms_error')
        fr = _f(r, 'selector_frontier_contrast_rebinding_final_rms_error')
        if all(math.isfinite(x) for x in (bi, fi, br, fr)):
            finite_error += 1
            ok = bool(fi < bi - 1e-8 or (abs(fi - bi) <= 1e-8 and fr <= br + 1e-8))
            error_nonincrease += int(ok)
            if _f(r, 'selector_frontier_contrast_rebinding_accepted', 0.0) >= 0.5:
                reductions.append(float(bi - fi))
                rms_reductions.append(float(br - fr))

    accepted_contract = []
    for r in accepted:
        bi = _f(r, 'selector_frontier_contrast_rebinding_baseline_linf_error')
        fi = _f(r, 'selector_frontier_contrast_rebinding_final_linf_error')
        br = _f(r, 'selector_frontier_contrast_rebinding_baseline_rms_error')
        fr = _f(r, 'selector_frontier_contrast_rebinding_final_rms_error')
        strict = bool(
            all(math.isfinite(x) for x in (bi, fi, br, fr))
            and (fi < bi - 1e-8 or (abs(fi - bi) <= 1e-8 and fr < br - 1e-8))
        )
        accepted_contract.append(bool(
            strict
            and _f(r, 'selector_frontier_contrast_rebinding_cardinality_preserved', 0.0) >= 0.5
            and _f(r, 'selector_frontier_contrast_rebinding_budget_preserved', 0.0) >= 0.5
            and _f(r, 'selector_frontier_contrast_rebinding_local_anchor_preserved', 0.0) >= 0.5
            and _f(r, 'selector_frontier_contrast_rebinding_candidate_exact_certificate', 0.0) >= 0.5
            and _f(r, 'selector_frontier_contrast_rebinding_certificate_non_decreasing', 0.0) >= 0.5
        ))

    reasons: dict[str, int] = {}
    for r in vals:
        code = str(int(round(_f(r, 'selector_frontier_contrast_rebinding_reason_code', -1.0))))
        reasons[code] = reasons.get(code, 0) + 1
    return {
        'scene_count': n,
        'enabled_rate': float(enabled_rate),
        'attempted_rate': float(attempted_rate),
        'accepted_count': int(len(accepted)),
        'accepted_rate': float(len(accepted) / max(n, 1)),
        'finite_frontier_error_rate': float(finite_error / max(n, 1)),
        'frontier_error_nonincrease_rate': float(error_nonincrease / max(finite_error, 1)),
        'accepted_contract_pass_rate': float(np.mean(accepted_contract)) if accepted_contract else float('nan'),
        'accepted_linf_reduction_mean': float(np.mean(reductions)) if reductions else float('nan'),
        'accepted_rms_reduction_mean': float(np.mean(rms_reductions)) if rms_reductions else float('nan'),
        'reason_code_counts': reasons,
    }


def _positive_count(tail: dict[str, Any]) -> int:
    count = int(tail.get('count', 0))
    precision = float(tail.get('teacher_positive_precision', float('nan')))
    if not math.isfinite(precision):
        return 0
    return int(round(count * precision))


def main() -> None:
    ap = argparse.ArgumentParser(description='One untouched 500-scene V64.3.29 FCR mechanism screen block.')
    ap.add_argument('--split-name', required=True)
    for name in ['raw', 'v20', 'aggregate-downside', 'fcr-v20', 'fcr-downside']:
        ap.add_argument(f'--{name}-metrics', required=True)
        ap.add_argument(f'--{name}-rows', required=True)
        if name != 'raw':
            ap.add_argument(f'--{name}-edges', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    cli_tags = ['raw', 'v20', 'aggregate-downside', 'fcr-v20', 'fcr-downside']
    tags = [x.replace('-', '_') for x in cli_tags]
    metrics = {t: json.load(open(getattr(args, t + '_metrics'), encoding='utf-8')) for t in tags}
    rows = {t: _load_rows(getattr(args, t + '_rows')) for t in tags}
    tokens = set(rows['raw'])
    if len(tokens) != 500 or any(set(rows[t]) != tokens for t in tags[1:]):
        raise SystemExit('STOP DATA: paired token identity mismatch; every arm must contain the exact same 500 scenes')

    all_flagged = {t for t in tokens if _f(rows['raw'][t], 'all_actions_safety_flagged_rate', 0.0) >= 0.5}
    safe = tokens - all_flagged
    edge_paths = {t: getattr(args, t + '_edges') for t in tags[1:]}
    edge = {t: _icer_edge_diag(Path(edge_paths[t]), safe) for t in tags[1:]}
    path = {t: _path_diag(rows['raw'], rows[t], edge_paths[t], safe, all_flagged) for t in tags[1:]}
    tails = {
        t: _replacement_tail_diag(rows['raw'], rows[t], edge_paths[t], safe)
        for t in ['aggregate_downside', 'fcr_downside']
    }
    M = {t: _metric_pack(metrics[t]) for t in tags}
    fcr = {
        'fcr_v20': _fcr_diag(rows['fcr_v20']),
        'fcr_downside': _fcr_diag(rows['fcr_downside']),
    }

    # The control and main FCR arms share the same selector; both must show the
    # exact same activation contract.  A mechanism that is effectively a no-op on
    # untouched data is not promotable even if endpoints fluctuate favorably.
    fcr_contract = bool(
        fcr['fcr_v20']['enabled_rate'] == 1.0
        and fcr['fcr_downside']['enabled_rate'] == 1.0
        and fcr['fcr_v20']['accepted_count'] >= 5
        and fcr['fcr_downside']['accepted_count'] >= 5
        and fcr['fcr_v20']['finite_frontier_error_rate'] >= 0.99
        and fcr['fcr_downside']['finite_frontier_error_rate'] >= 0.99
        and fcr['fcr_v20']['frontier_error_nonincrease_rate'] == 1.0
        and fcr['fcr_downside']['frontier_error_nonincrease_rate'] == 1.0
        and fcr['fcr_v20']['accepted_contract_pass_rate'] == 1.0
        and fcr['fcr_downside']['accepted_contract_pass_rate'] == 1.0
    )

    structural = {
        'all_flagged_scene_count': len(all_flagged),
        'main_all_flagged_final_identity_vs_raw': _identity_rate(rows['fcr_downside'], rows['raw'], all_flagged),
        'main_all_flagged_delegation_rate': (
            sum(_f(rows['fcr_downside'][t], 'decisive_frontier_icer_structural_domain_delegated', 0.0) >= 0.5 for t in all_flagged) / len(all_flagged)
            if all_flagged else float('nan')
        ),
        'main_safe_guard_block_rate': _guard_block_rate(rows['fcr_downside'], safe),
    }
    structural_ok = bool(
        len(all_flagged) >= 3
        and structural['main_all_flagged_final_identity_vs_raw'] == 1.0
        and structural['main_all_flagged_delegation_rate'] == 1.0
        and structural['main_safe_guard_block_rate'] <= 0.001
    )

    baseline_tail = tails['aggregate_downside']
    main_tail = tails['fcr_downside']
    baseline_path = path['aggregate_downside']['direct_incumbent_to_alternative']
    main_path = path['fcr_downside']['direct_incumbent_to_alternative']
    baseline_edge = edge['aggregate_downside']
    main_edge = edge['fcr_downside']

    # Preserve the already successful asymmetric incumbent contract.  V19-V21
    # demonstrated that reopening learned incumbent->anchor veto/recovery is
    # split-unstable, so V29 is not allowed to buy coverage through that path.
    asymmetric = bool(
        path['fcr_downside']['admissible_incumbent_to_anchor']['count'] == 0
        and abs(path['fcr_downside']['admissible_incumbent_to_anchor']['regret_delta_sum']) <= 1e-9
    )
    catastrophe_free = bool(
        math.isfinite(main_tail['teacher_improvement_worst'])
        and main_tail['teacher_improvement_worst'] > -0.5
    )
    selected_path_safe = bool(
        main_path['count'] >= 8
        and main_path['regret_delta_sum'] <= 0.0
        and main_tail['teacher_positive_precision'] >= 0.60
        and catastrophe_free
    )

    # The causal V29 claim is safe *coverage expansion*, not a prettier classifier
    # score.  Both independent blocks must gain at least 3 percentage points of
    # direct positive-opportunity capture and at least five extra teacher-positive
    # direct replacements, while the selected downside tail is non-inferior to
    # the frozen V25 aggregate DRC control.
    baseline_pos = _positive_count(baseline_tail)
    main_pos = _positive_count(main_tail)
    coverage_gain = bool(
        main_edge['direct_incumbent_opportunity_capture_rate'] >= baseline_edge['direct_incumbent_opportunity_capture_rate'] + 0.03
        and main_pos >= baseline_pos + 5
    )
    tail_noninferior = bool(
        math.isfinite(main_tail['regret_positive_rms']) and math.isfinite(baseline_tail['regret_positive_rms'])
        and main_tail['regret_positive_rms'] <= baseline_tail['regret_positive_rms'] + 1e-9
        and main_tail['worst_regret_increase'] <= baseline_tail['worst_regret_increase'] + 1e-9
        and main_tail['teacher_negative_rms'] <= baseline_tail['teacher_negative_rms'] + 1e-9
        and main_tail['teacher_improvement_worst'] >= baseline_tail['teacher_improvement_worst'] - 1e-9
    )

    preservation = bool(
        M['fcr_downside']['harmful'] <= M['raw']['harmful'] + 0.005
        and M['fcr_downside']['flip'] <= M['raw']['flip'] + 0.01
        and selected_path_safe
        and asymmetric
    )
    endpoint_noninferior = bool(
        M['fcr_downside']['match'] >= M['aggregate_downside']['match'] - 0.002
        and M['fcr_downside']['regret'] <= M['aggregate_downside']['regret'] * 1.005
        and M['fcr_downside']['regret'] <= M['raw']['regret'] * 1.02
    )
    endpoint_strict_gain = bool(
        M['fcr_downside']['match'] >= M['aggregate_downside']['match'] + 0.002
        or M['fcr_downside']['regret'] < M['aggregate_downside']['regret'] - 1e-6
    )

    instrumentation = bool(
        edge['fcr_downside']['scene_count'] >= 450
        and edge['fcr_downside']['admissible_edge_count'] >= 1800
        and _f(metrics['fcr_downside'], 'decisive_frontier_value_complete_star_coverage') >= 0.99
        and fcr_contract
    )

    full = bool(
        instrumentation and structural_ok and asymmetric and selected_path_safe
        and coverage_gain and tail_noninferior and preservation and endpoint_noninferior
    )
    if not instrumentation:
        next_action = 'engineering_or_FCR_inactive_stop_do_not_tune_rebinding_objective'
    elif not structural_ok or not asymmetric:
        next_action = 'preservation_contract_failure_stop_FCR'
    elif not selected_path_safe or not tail_noninferior:
        next_action = 'FCR_reintroduced_selected_tail_harm_stop_do_not_relax_acceptance_guards'
    elif not coverage_gain:
        next_action = 'FCR_preserves_frontier_but_does_not_expand_safe_recovery_coverage_stop_and_reassess_interface_target'
    elif not preservation:
        next_action = 'coverage_gain_but_raw_preservation_failed_stop'
    elif not endpoint_noninferior:
        next_action = 'safe_coverage_gain_without_endpoint_noninferiority_audit_recovery_ranking_not_budget_size'
    else:
        next_action = 'split_pass_freeze_FCR_and_wait_for_second_independent_block'

    report = {
        'audit': 'v64_3_29_eaf_icer_fcr_split',
        'split_name': args.split_name,
        'full_split_pass': full,
        'instrumentation_valid': instrumentation,
        'fcr_monotone_interface_contract': fcr_contract,
        'deployment_alignment': structural_ok,
        'incumbent_default_invariant': asymmetric,
        'selected_replacement_path_nonharmful': selected_path_safe,
        'selected_replacement_catastrophe_free': catastrophe_free,
        'safe_recovery_coverage_gain_over_V25_DRC': coverage_gain,
        'selected_tail_noninferior_to_V25_DRC': tail_noninferior,
        'asymmetric_preservation_non_degradation': preservation,
        'endpoint_noninferior_to_V25_DRC': endpoint_noninferior,
        'endpoint_strict_gain_over_V25_DRC': endpoint_strict_gain,
        'next_action': next_action,
        'fcr_diagnostics': fcr,
        'structural': structural,
        'edge_diagnostics': edge,
        'path_diagnostics': path,
        'selected_replacement_tail_diagnostics': tails,
        'positive_direct_replacements': {'aggregate_downside': baseline_pos, 'fcr_downside': main_pos},
        'metrics': M,
        'thresholds_preregistered_before_new_fresh': {
            'FCR_min_accepted_scenes_per_500': 5,
            'direct_opportunity_capture_absolute_gain_min': 0.03,
            'extra_teacher_positive_direct_replacements_min': 5,
            'direct_precision_min': 0.60,
            'selected_catastrophic_teacher_improvement_threshold': -0.5,
            'selected_catastrophic_replacement_count_max': 0,
            'selected_tail_noninferiority': 'no increase in positive-regret RMS, worst regret increase, teacher negative RMS, or worst teacher improvement versus V25 aggregate DRC',
            'raw_harmful_tolerance': 0.005,
            'raw_flip_tolerance': 0.01,
            'endpoint_match_tolerance': 0.002,
            'endpoint_regret_tolerance_vs_V25': 0.005,
            'B_and_M': 'fixed B<=16, M=24; no sweep',
        },
        'interpretation': (
            'This block tests whether a post-EAF fixed-cardinality evidence rebind can transmit more complete full-M action-decision contrast through the same retained interface. '
            'The mechanism is promoted only if its monotone frontier/certificate invariants hold, direct recovery coverage rises materially, and the already-clean V25 selected tail does not degrade. '
            'No PTMC confirmation, B sweep, learned acquisition loss, incumbent->anchor reopening, or beam/swap repair is permitted.'
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
