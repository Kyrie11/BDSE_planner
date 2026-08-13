from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    ap = argparse.ArgumentParser(description='Fail fast when a completed BDSE training stage did not persist its artifacts.')
    ap.add_argument('--output-root', required=True, type=Path)
    ap.add_argument('--stem', default='bdse_v64_saqa_bcc')
    ap.add_argument('--require-epoch-checkpoint', action='store_true')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()

    root = args.output_root
    train = root / 'train'
    final_ckpt = train / f'{args.stem}.pt'
    train_log = train / f'{args.stem}.train_log.jsonl'
    rows: list[dict[str, Any]] = []
    parse_error = None
    if train_log.is_file():
        try:
            rows = [json.loads(x) for x in train_log.read_text(encoding='utf-8').splitlines() if x.strip()]
        except Exception as exc:  # report, do not hide a corrupted log
            parse_error = f'{type(exc).__name__}: {exc}'
    trained_epochs = sorted({int(r.get('epoch', -1)) for r in rows if int(r.get('epoch', -1)) >= 0})
    checkpoints = sorted((train / 'checkpoints').glob(f'{args.stem}.epoch_*.pt')) if (train / 'checkpoints').is_dir() else []
    checks = {
        'train_log_exists': train_log.is_file() and train_log.stat().st_size > 0,
        'train_log_parseable': parse_error is None,
        'at_least_one_trained_epoch_logged': bool(trained_epochs),
        'final_checkpoint_exists': final_ckpt.is_file() and final_ckpt.stat().st_size > 0,
        'epoch_checkpoint_exists_if_required': (not args.require_epoch_checkpoint) or bool(checkpoints),
    }
    report = {
        'audit': 'bdse_training_artifact_contract',
        'output_root': str(root),
        'stem': args.stem,
        'checks': checks,
        'pass': all(checks.values()),
        'logged_trained_epochs': trained_epochs,
        'final_checkpoint': str(final_ckpt),
        'epoch_checkpoints': [str(p) for p in checkpoints],
        'train_log': str(train_log),
        'parse_error': parse_error,
        'note': 'This contract distinguishes a completed training process from a launcher/promotion failure. It intentionally does not accept a validation log without the final checkpoint.',
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
