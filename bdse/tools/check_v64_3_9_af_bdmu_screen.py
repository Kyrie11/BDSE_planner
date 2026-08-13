from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float('nan')
    return v if math.isfinite(v) else float('nan')


def _pick(row: dict[str, Any], key: str) -> float:
    return _finite(row.get(key))


def _delta(anchor: float, cur: float) -> float:
    return cur - anchor if math.isfinite(anchor) and math.isfinite(cur) else float('nan')


def _summary(row: dict[str, Any]) -> dict[str, float]:
    return {
        'teacher': _pick(row, 'val_teacher_action_match'),
        'teacher_regret': _pick(row, 'val_teacher_regret'),
        'pairfull': _pick(row, 'val_pair_full_interface_action_match'),
        'pairfull_regret': _pick(row, 'val_pair_full_teacher_regret'),
        'budget_vs_pairfull': _pick(row, 'val_budget_vs_pair_full_match'),
        'critical_topm': _pick(row, 'val_teacher_exact_winner_flip_critical_recall_topm_micro'),
        'critical_selected': _pick(row, 'val_teacher_exact_winner_flip_critical_recall_selected_micro'),
        'proposal_decisive': _pick(row, 'val_proposal_decisive_atom_recall'),
        'exact_evidence_certificate': _pick(row, 'val_evidence_certificate_fraction'),
        'bdmu_topm_capture': _pick(row, 'val_teacher_bdmu_topm_utility_capture'),
        'bdmu_selected_capture': _pick(row, 'val_teacher_bdmu_selected_utility_capture'),
        'bdmu_missed_fraction': _pick(row, 'val_teacher_bdmu_missed_utility_fraction'),
        'bdmu_margin_deficit': _pick(row, 'val_teacher_bdmu_reference_margin_deficit'),
        'bdmu_mean_margin_deficit': _pick(row, 'val_teacher_bdmu_reference_mean_margin_deficit'),
        'bdmu_worst_margin_deficit': _pick(row, 'val_teacher_bdmu_reference_worst_margin_deficit'),
        'bdmu_frontier_rival_count': _pick(row, 'val_teacher_bdmu_frontier_rival_count'),
        'bdmu_scene_has_utility': _pick(row, 'val_teacher_bdmu_scene_has_utility'),
        'adapter_delta_rms': _pick(row, 'critical_adapter_parameter_delta_rms'),
        'adapter_residual_rms': _pick(row, 'critical_proposal_residual_rms'),
        'swap_rank_scene_fraction': _pick(row, 'bdmu_topm_swap_rank_scene_fraction'),
        'swap_rank_pairs': _pick(row, 'bdmu_topm_swap_rank_pairs'),
        'frontier_train_count': _pick(row, 'bdmu_frontier_rival_count'),
        'beneficial_compression': _pick(row, 'val_beneficial_pair_compression_rate'),
        'harmful_compression': _pick(row, 'val_harmful_pair_compression_rate'),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Audit V64.3.9 AF-BDMU frozen-value acquisition screen')
    ap.add_argument('--train-log', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--min-topm-utility-gain', type=float, default=0.015)
    ap.add_argument('--min-selected-utility-gain', type=float, default=0.005)
    ap.add_argument('--min-critical-topm-gain', type=float, default=0.005)
    ap.add_argument('--min-teacher-gain', type=float, default=0.005)
    ap.add_argument('--min-regret-relative-gain', type=float, default=0.02)
    ap.add_argument('--teacher-nonworse-tol', type=float, default=0.004)
    ap.add_argument('--regret-nonworse-relative-tol', type=float, default=0.02)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.train_log).read_text(encoding='utf-8').splitlines() if line.strip()]
    anchors = [r for r in rows if int(r.get('epoch', 999999)) < 0]
    trained = [r for r in rows if int(r.get('epoch', -999999)) >= 0]
    if not anchors or not trained:
        raise SystemExit('AF-BDMU screen requires epoch<0 validation anchor plus trained epochs')
    anchor_row = anchors[-1]
    anchor = _summary(anchor_row)
    audited: list[dict[str, Any]] = []
    for row in trained:
        cur = _summary(row)
        deltas = {k: _delta(anchor[k], cur[k]) for k in anchor}
        topm_gain = math.isfinite(deltas['bdmu_topm_capture']) and deltas['bdmu_topm_capture'] >= args.min_topm_utility_gain
        selected_gain = math.isfinite(deltas['bdmu_selected_capture']) and deltas['bdmu_selected_capture'] >= args.min_selected_utility_gain
        boundary_gain = math.isfinite(deltas['critical_topm']) and deltas['critical_topm'] >= args.min_critical_topm_gain
        # Primary evidence is continuous decisive-margin utility. Literal winner
        # flips are retained only as a limiting-case boundary stress test.
        mechanism_gain = bool(topm_gain and (selected_gain or boundary_gain))

        regret_rel = float('nan')
        if math.isfinite(anchor['teacher_regret']) and abs(anchor['teacher_regret']) > 1e-9 and math.isfinite(cur['teacher_regret']):
            regret_rel = (anchor['teacher_regret'] - cur['teacher_regret']) / abs(anchor['teacher_regret'])
        deployment_gain = bool(
            (math.isfinite(deltas['teacher']) and deltas['teacher'] >= args.min_teacher_gain)
            or (math.isfinite(regret_rel) and regret_rel >= args.min_regret_relative_gain)
        )
        teacher_nonworse = math.isfinite(deltas['teacher']) and deltas['teacher'] >= -args.teacher_nonworse_tol
        regret_nonworse = (
            math.isfinite(cur['teacher_regret']) and math.isfinite(anchor['teacher_regret'])
            and cur['teacher_regret'] <= anchor['teacher_regret'] * (1.0 + args.regret_nonworse_relative_tol)
        )
        frontier_active = (
            (math.isfinite(cur['bdmu_frontier_rival_count']) and cur['bdmu_frontier_rival_count'] >= 4.0)
            or (math.isfinite(cur['frontier_train_count']) and cur['frontier_train_count'] >= 4.0)
        )
        swap_rank_active = (
            (math.isfinite(cur['swap_rank_scene_fraction']) and cur['swap_rank_scene_fraction'] > 0.0)
            or (math.isfinite(cur['swap_rank_pairs']) and cur['swap_rank_pairs'] > 0.0)
        )
        instrumentation = bool(
            math.isfinite(cur['bdmu_scene_has_utility']) and cur['bdmu_scene_has_utility'] > 0.0
            and math.isfinite(cur['adapter_delta_rms']) and cur['adapter_delta_rms'] > 1e-7
            and frontier_active and swap_rank_active
        )
        promotion = bool(instrumentation and mechanism_gain and deployment_gain and teacher_nonworse and regret_nonworse)
        # Explicit scientific exit rule: if the acquisition mechanism moves on a
        # frozen value interface but the teacher endpoint does not, further
        # selector tuning is no longer the next justified experiment.
        pivot_to_value_frontier = bool(instrumentation and mechanism_gain and not deployment_gain)
        audited.append({
            'epoch': int(row['epoch']), 'selected': cur, 'deltas': deltas,
            'teacher_regret_relative_gain': regret_rel,
            'instrumentation_valid': instrumentation, 'frontier_active': frontier_active,
            'topm_swap_rank_active': swap_rank_active, 'mechanism_gain': mechanism_gain,
            'deployment_gain': deployment_gain, 'teacher_nonworse': teacher_nonworse,
            'teacher_regret_nonworse': regret_nonworse, 'full_promotion': promotion,
            'pivot_to_value_frontier': pivot_to_value_frontier,
        })

    # Mechanism first, then paper endpoint. This avoids choosing an epoch only
    # because a finite validation subset happened to improve action match.
    pool = [r for r in audited if r['instrumentation_valid'] and r['mechanism_gain']] or audited
    def score(r: dict[str, Any]) -> tuple[float, float, float, float, int]:
        s = r['selected']
        return (
            float(bool(r['full_promotion'])),
            s['teacher'] if math.isfinite(s['teacher']) else -1e9,
            -(s['teacher_regret'] if math.isfinite(s['teacher_regret']) else 1e18),
            s['bdmu_topm_capture'] if math.isfinite(s['bdmu_topm_capture']) else -1e9,
            -int(r['epoch']),
        )
    best = max(pool, key=score)
    report = {
        'audit': 'v64_3_9_af_bdmu_screen',
        'audit_version': 'v64.3.9.0',
        'objective': 'adaptive-frontier, cost-normalized, budget-feasible one-sided decisive-margin marginal utility with Top-M swap ranking',
        'anchor_epoch': int(anchor_row.get('epoch', -1)), 'anchor': anchor,
        'epochs': audited, 'selected_epoch': best['epoch'], 'selected': best['selected'],
        'deltas': best['deltas'], 'teacher_regret_relative_gain': best['teacher_regret_relative_gain'],
        'instrumentation_valid': best['instrumentation_valid'], 'mechanism_gain': best['mechanism_gain'],
        'deployment_gain': best['deployment_gain'], 'full_promotion': best['full_promotion'],
        'pivot_to_value_frontier': best['pivot_to_value_frontier'],
        'thresholds': {
            'min_topm_utility_gain': args.min_topm_utility_gain,
            'min_selected_utility_gain': args.min_selected_utility_gain,
            'min_critical_topm_gain': args.min_critical_topm_gain,
            'min_teacher_gain': args.min_teacher_gain,
            'min_regret_relative_gain': args.min_regret_relative_gain,
            'teacher_nonworse_tolerance': args.teacher_nonworse_tol,
            'regret_nonworse_relative_tolerance': args.regret_nonworse_relative_tol,
        },
        'scientific_note': (
            'The exact B->TopM evidence certificate is reported but is not a promotion target: the uploaded '
            'V64.3.7 result is already high on that metric. AF-BDMU targets the upstream Top-M utility ranking. '
            'If utility/critical recall improves without teacher decision/regret improvement, stop acquisition '
            'iterations and pivot to a new frozen-interface value/frontier model.'
        ),
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['selected_epoch','instrumentation_valid','mechanism_gain','deployment_gain','full_promotion','pivot_to_value_frontier']}, indent=2))
    return 0 if report['instrumentation_valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
