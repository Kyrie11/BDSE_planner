from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--screen', action='append', default=[])
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    screens = []
    for spec in a.screen:
        name, path = spec.split('=', 1)
        d = json.loads(Path(path).read_text())
        screens.append({'name': name, 'path': path, **d})

    interpretable = [s for s in screens if s.get('instrumentation_valid', s.get('valid'))]
    promoted = [s for s in interpretable if s.get('full_promotion')]

    def rank(s):
        d = s.get('deltas', {})
        return (
            d.get('teacher') if d.get('teacher') is not None else -9,
            d.get('pairfull') if d.get('pairfull') is not None else -9,
            s.get('pair_full_advantage_over_local') if s.get('pair_full_advantage_over_local') is not None else -9,
            s.get('residual_intervention_net') if s.get('residual_intervention_net') is not None else -9,
            s.get('budget_compression_net') if s.get('budget_compression_net') is not None else -9,
            -(d.get('teacher_regret') if d.get('teacher_regret') is not None else 9e18),
        )

    winner = max(promoted, key=rank)['name'] if promoted else None
    mechanism_arms = [s for s in interpretable if s.get('meaningful_value_gain')]
    mechanism = max(mechanism_arms, key=rank)['name'] if mechanism_arms else None
    report = {
        'audit': 'v64_3_7_1_darm_dbr_screen_comparison',
        'winner': winner,
        'best_mechanism_arm': mechanism,
        'run_full_pipeline': bool(winner),
        'screens': screens,
        'note': (
            'V64.3.7.1 separates protocol validity from algorithm promotion. '
            'Pair-star coverage and budget-vs-learned-pair-full agreement are diagnostics; '
            'promotion follows fixed-B teacher decision/regret, pair-full-over-local value gain, '
            'and beneficial-vs-harmful intervention evidence.'
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
