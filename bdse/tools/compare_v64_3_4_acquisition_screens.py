from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def main() -> int:
    ap = argparse.ArgumentParser(description='Rank V64.3.4 acquisition screens using causal same-subset deltas.')
    ap.add_argument('--screen', action='append', default=[], help='NAME=path/to/critical_acquisition_screen.json')
    ap.add_argument('--min-critical-topm-gain', type=float, default=0.01)
    ap.add_argument('--min-selected-gain', type=float, default=-0.005)
    ap.add_argument('--min-proposal-gain', type=float, default=-0.02)
    ap.add_argument('--min-teacher-match-gain', type=float, default=-0.005)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if not args.screen:
        raise SystemExit('at least one --screen NAME=PATH is required')

    rows = []
    for item in args.screen:
        if '=' not in item:
            raise SystemExit(f'invalid --screen {item!r}; expected NAME=PATH')
        name, raw_path = item.split('=', 1)
        path = Path(raw_path)
        d = json.loads(path.read_text())
        critical = _finite(d.get('delta_val_critical_topm_recall_micro'))
        selected = _finite(d.get('delta_val_critical_selected_recall_micro'))
        proposal = _finite(d.get('delta_val_proposal_decisive_recall'))
        teacher = _finite(d.get('delta_val_teacher_action_match'))
        valid = bool(
            d.get('screen_instrumentation_valid')
            and d.get('adapter_parameter_activated')
            and d.get('adapter_forward_activated')
            and d.get('acra_wired')
            and d.get('lba_wired', True)
            and d.get('literal_critical_support_nonempty')
        )
        safeguards = bool(
            selected is not None and selected >= args.min_selected_gain
            and proposal is not None and proposal >= args.min_proposal_gain
            and (teacher is None or teacher >= args.min_teacher_match_gain)
        )
        meaningful = bool(valid and critical is not None and critical >= args.min_critical_topm_gain and safeguards)
        # Ranking is used only among experiments that pass the same safety gates;
        # critical acquisition remains the primary score by construction.
        rank_score = None
        if meaningful:
            rank_score = critical + 0.35 * max(selected or 0.0, -0.02) + 0.15 * max(teacher or 0.0, -0.02)
        rows.append({
            'name': name,
            'path': str(path),
            'variant': d.get('variant'),
            'valid': valid,
            'meaningful_acquisition_gain': meaningful,
            'delta_critical_topm_micro': critical,
            'delta_critical_selected_micro': selected,
            'delta_proposal_decisive': proposal,
            'delta_teacher_action_match': teacher,
            'boundary_representable_fraction_max': _finite(d.get('boundary_representable_fraction_max')),
            'critical_boundary_in_base_top6_anchor': _finite(d.get('anchor_val_critical_boundary_in_base_top6_fraction')),
            'rank_score': rank_score,
        })

    winners = sorted(
        [r for r in rows if r['meaningful_acquisition_gain']],
        key=lambda r: (r['rank_score'], r['delta_critical_topm_micro']),
        reverse=True,
    )
    report = {
        'audit': 'v64_3_4_acquisition_screen_comparison',
        'thresholds': {
            'min_critical_topm_gain': args.min_critical_topm_gain,
            'min_selected_gain': args.min_selected_gain,
            'min_proposal_gain': args.min_proposal_gain,
            'min_teacher_match_gain': args.min_teacher_match_gain,
        },
        'screens': rows,
        'winner': winners[0]['name'] if winners else None,
        'run_full_pipeline': bool(winners),
        'note': 'A screen is promotion-eligible only after a >=1pp same-subset literal-critical Top-M gain and no material selected/proposal/teacher regression.',
    }
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
