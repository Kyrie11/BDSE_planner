from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import bdse.tools.fit_v64_3_25_eaf_icer_drc as v25

EXPECTED_TRAIN_SCENES = 3000
MIN_ACTIVE_REBINDS = 1


def _f(row: dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        x = float(row.get(key, default))
    except Exception:
        return float(default)
    return x if np.isfinite(x) else float(default)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f'STOP TRAIN DATA: malformed per-sample JSONL line {line_no}: {exc}') from exc
    return rows


def _fcr_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_TRAIN_SCENES:
        raise SystemExit(f'STOP TRAIN DATA: expected {EXPECTED_TRAIN_SCENES} FCR per-sample rows, got {len(rows)}')
    tokens = [str(r.get('scenario_token', '')) for r in rows]
    if not all(tokens) or len(set(tokens)) != EXPECTED_TRAIN_SCENES:
        raise SystemExit('STOP TRAIN DATA: FCR per-sample tokens are not exactly 3000 unique scenes')

    enabled = np.asarray([_f(r, 'selector_frontier_contrast_rebinding_enabled', 0.0) for r in rows])
    attempted = np.asarray([_f(r, 'selector_frontier_contrast_rebinding_attempted', 0.0) for r in rows])
    accepted = np.asarray([_f(r, 'selector_frontier_contrast_rebinding_accepted', 0.0) for r in rows]) >= 0.5
    b_linf = np.asarray([_f(r, 'selector_frontier_contrast_rebinding_baseline_linf_error') for r in rows])
    f_linf = np.asarray([_f(r, 'selector_frontier_contrast_rebinding_final_linf_error') for r in rows])
    b_rms = np.asarray([_f(r, 'selector_frontier_contrast_rebinding_baseline_rms_error') for r in rows])
    f_rms = np.asarray([_f(r, 'selector_frontier_contrast_rebinding_final_rms_error') for r in rows])

    finite_error = np.isfinite(b_linf) & np.isfinite(f_linf) & np.isfinite(b_rms) & np.isfinite(f_rms)
    final_nonincrease = np.ones(len(rows), dtype=bool)
    final_nonincrease[finite_error] = (
        (f_linf[finite_error] < b_linf[finite_error] - 1e-8)
        | (
            np.abs(f_linf[finite_error] - b_linf[finite_error]) <= 1e-8
        )
    )
    # If L_inf ties, FCR may only improve RMS; fallback rows are exactly equal.
    tie = finite_error & (np.abs(f_linf - b_linf) <= 1e-8)
    final_nonincrease[tie] &= f_rms[tie] <= b_rms[tie] + 1e-8

    accepted_contract = []
    reductions: list[float] = []
    rms_reductions: list[float] = []
    for i, r in enumerate(rows):
        if not accepted[i]:
            continue
        ok = bool(
            _f(r, 'selector_frontier_contrast_rebinding_cardinality_preserved', 0.0) >= 0.5
            and _f(r, 'selector_frontier_contrast_rebinding_budget_preserved', 0.0) >= 0.5
            and _f(r, 'selector_frontier_contrast_rebinding_local_anchor_preserved', 0.0) >= 0.5
            and _f(r, 'selector_frontier_contrast_rebinding_candidate_exact_certificate', 0.0) >= 0.5
            and _f(r, 'selector_frontier_contrast_rebinding_certificate_non_decreasing', 0.0) >= 0.5
            and finite_error[i]
            and (
                f_linf[i] < b_linf[i] - 1e-8
                or (
                    abs(f_linf[i] - b_linf[i]) <= 1e-8
                    and f_rms[i] < b_rms[i] - 1e-8
                )
            )
        )
        accepted_contract.append(ok)
        if finite_error[i]:
            reductions.append(float(b_linf[i] - f_linf[i]))
            rms_reductions.append(float(b_rms[i] - f_rms[i]))

    reasons = Counter(int(round(_f(r, 'selector_frontier_contrast_rebinding_reason_code', -1.0))) for r in rows)
    accepted_count = int(np.sum(accepted))
    report = {
        'scene_count': len(rows),
        'enabled_rate': float(np.mean(enabled >= 0.5)),
        'attempted_rate': float(np.mean(attempted >= 0.5)),
        'accepted_count': accepted_count,
        'accepted_rate': float(accepted_count / len(rows)),
        'finite_frontier_error_rate': float(np.mean(finite_error)),
        'final_frontier_error_nonincrease_rate': float(np.mean(final_nonincrease)),
        'accepted_contract_pass_rate': float(np.mean(accepted_contract)) if accepted_contract else float('nan'),
        'accepted_linf_reduction_mean': float(np.mean(reductions)) if reductions else float('nan'),
        'accepted_rms_reduction_mean': float(np.mean(rms_reductions)) if rms_reductions else float('nan'),
        'reason_code_counts': {str(k): int(v) for k, v in sorted(reasons.items())},
        'all_rows_enabled': bool(np.all(enabled >= 0.5)),
        'all_final_errors_nonincreasing': bool(np.all(final_nonincrease)),
        'all_accepted_contracts_valid': bool(accepted_contract and all(accepted_contract)),
        'mechanism_active': bool(accepted_count >= MIN_ACTIVE_REBINDS),
    }
    report['gate_pass'] = bool(
        report['all_rows_enabled']
        and report['finite_frontier_error_rate'] >= 0.99
        and report['all_final_errors_nonincreasing']
        and report['all_accepted_contracts_valid']
        and report['mechanism_active']
    )
    return report


def _retag_cfg(base: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    cfg = v25._cfg(base, memory, 'downside_rms', 'fcr_aggregate_downside')
    version = 'V64.3.29-EAF-ICER-FCR-DRC'
    cfg.setdefault('metadata', {})['algorithm_version'] = version
    cfg['metadata']['fcr_post_eaf_rebinding'] = True
    cfg.setdefault('provenance', {})['algorithm_version'] = version
    cfg['provenance']['screening_only'] = True
    exp = cfg.setdefault('experiment', {})
    exp['name'] = 'v64_3_29_eaf_icer_fcr_aggregate_downside'
    exp['algorithm'] = (
        'V64.3.29 EAF-ICER-FCR-DRC: Fixed-Budget Frontier-Contrast Evidence Rebinding '
        'with Frozen Aggregate Downside-Regret Recovery'
    )
    exp['calibration_protocol'] = (
        'FCR itself is deterministic and teacher-free. The unchanged V25 aggregate downside-regret '
        'certificate recipe is re-fit only on the same frozen 3000 TRAIN scenes because the selected-B '
        'evidence distribution has changed; K={32,64}, downside multiplier=1, boundary=0 remain frozen.'
    )
    exp['mechanism_chain'] = (
        'fixed planner-interface budget -> frozen Top-M/EAF -> complete full-M DARM+EAF frontier target -> '
        'monotone same-cardinality FCR -> frozen support/scalar dominance -> unchanged aggregate downside-regret '
        'extremal recovery -> incumbent preservation -> unchanged structural guard -> endpoint'
    )
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description='Fit the unchanged DRC recipe after V64.3.29 FCR and audit the rebinding invariants on frozen TRAIN.')
    ap.add_argument('--train-frontier-edges', required=True)
    ap.add_argument('--train-rows', required=True)
    ap.add_argument('--base-fcr-v20-config', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--output-train-token-file', required=True)
    ap.add_argument('--output-report', required=True)
    args = ap.parse_args()

    edge_path = Path(args.train_frontier_edges)
    row_path = Path(args.train_rows)
    if not edge_path.is_file() or edge_path.stat().st_size <= 0:
        raise SystemExit(f'STOP TRAIN DATA: missing frontier provenance {edge_path}')
    if not row_path.is_file() or row_path.stat().st_size <= 0:
        raise SystemExit(f'STOP TRAIN DATA: missing FCR per-sample provenance {row_path}')

    by, frontier_row_count = v25._load_minimal_scenes(edge_path)
    if len(by) != EXPECTED_TRAIN_SCENES:
        raise SystemExit(f'STOP TRAIN DATA: expected exactly {EXPECTED_TRAIN_SCENES} frozen TRAIN scenes, got {len(by)}')
    data = v25._build(by)
    crossfit = v25._crossfit(data, 'downside_rms')
    fcr = _fcr_audit(_load_rows(row_path))
    drc_gate = bool(
        crossfit['all_folds_path_safe']
        and crossfit['selected_count'] >= v25.MAIN_MIN_SELECTED
        and crossfit['teacher_improvement_sum'] >= -1.0e-9
    )
    gate = bool(fcr['gate_pass'] and drc_gate)

    tokens = sorted(by)
    token_path = Path(args.output_train_token_file)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text('\n'.join(tokens) + '\n', encoding='utf-8')

    y = data['delta']
    report: dict[str, Any] = {
        'audit': 'v64_3_29_eaf_icer_fcr_train_fit',
        'algorithm': 'V64.3.29 EAF-ICER-FCR-DRC',
        'train_scene_count': int(len(by)),
        'frontier_row_count': int(frontier_row_count),
        'replacement_edges': int(len(y)),
        'replacement_scenes': int(data['replacement_scene_count']),
        'population_teacher_positive_fraction': float(np.mean(y > 0.0)),
        'population_teacher_improvement_sum': float(y.sum()),
        'population_teacher_improvement_worst': float(y.min()),
        'drc_crossfit': crossfit,
        'fcr_train_audit': fcr,
        'drc_gate_pass': drc_gate,
        'train_gate_pass': gate,
        'frozen_drc_contract': {
            'feature_mode': 'aggregate_evidence_only',
            'neighbor_k_values': list(v25.KS),
            'certificate': 'mean_minus_downside_rms',
            'downside_multiplier': 1.0,
            'decision_boundary': 0.0,
            'ranking': 'frozen_scalar_dominance',
            'no_validation_tuning': True,
        },
        'input_frontier': {
            'path': str(edge_path), 'bytes': int(edge_path.stat().st_size), 'sha256': v25._sha256_file(edge_path),
        },
        'input_rows': {
            'path': str(row_path), 'bytes': int(row_path.stat().st_size), 'sha256': v25._sha256_file(row_path),
        },
        'train_token_manifest': {
            'path': str(token_path), 'count': int(len(tokens)), 'sha256': v25._sha256_file(token_path),
        },
        'fresh_validation_used': False,
        'gate_contract': {
            'fcr_enabled_on_all_train_scenes': True,
            'fcr_nontrivial_acceptance_min': MIN_ACTIVE_REBINDS,
            'frontier_error_never_increases': True,
            'every_accepted_rebind_preserves_cardinality_budget_fullM_local_anchor_and_exact_target': True,
            'drc_all_5_scene_folds_path_safe': True,
            'drc_selected_count_min': v25.MAIN_MIN_SELECTED,
            'drc_teacher_improvement_sum_min': 0.0,
            'stop_before_fresh_on_fail': True,
        },
        'configs': {},
        'memories': {},
    }
    v25._write_report(Path(args.output_report), report)
    if not gate:
        raise SystemExit(
            'STOP TRAIN FCR: either the deterministic rebinding contract was inactive/violated or the unchanged DRC recipe '
            'lost its 5-fold TRAIN safety gate. Do not tune FCR weights/thresholds or DRC K/boundary; audit interface observability.'
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    memory_path = out_dir / 'v64_3_29_fcr_aggregate_downside_memory.npz'
    memory = v25._save_memory(memory_path, data, 'downside_rms')
    base = yaml.safe_load(Path(args.base_fcr_v20_config).read_text(encoding='utf-8'))
    cfg = _retag_cfg(base, memory)
    config_path = out_dir / 'v64_3_29_fcr_aggregate_downside.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')

    report['memories'] = {'fcr_aggregate_downside': memory}
    report['configs'] = {'fcr_aggregate_downside': str(config_path)}
    v25._write_report(Path(args.output_report), report)
    print(json.dumps({
        'pass': True,
        'train_gate_pass': gate,
        'fcr_accepted_count': fcr['accepted_count'],
        'drc_fold_pass_count': crossfit['fold_pass_count'],
        'drc_selected_count': crossfit['selected_count'],
        'drc_teacher_improvement_sum': crossfit['teacher_improvement_sum'],
    }, sort_keys=True))


if __name__ == '__main__':
    main()
