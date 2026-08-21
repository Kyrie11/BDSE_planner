from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _icer(cfg):
    return cfg['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']


def main() -> None:
    ap = argparse.ArgumentParser(description='Hard contract audit for V64.3.29 FCR-DRC.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--frozen-fcr-v20-config', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    base = yaml.safe_load(Path(args.frozen_fcr_v20_config).read_text(encoding='utf-8'))
    errors: list[str] = []

    if cfg.get('evidence', {}).get('budget') != 16:
        errors.append('literal retained evidence budget changed')
    if cfg.get('selector', {}).get('proposal_top_m') != 24:
        errors.append('proposal Top-M changed')
    fcr = cfg.get('selector', {}).get('frontier_contrast_rebinding', {}) or {}
    if not fcr.get('enabled', False):
        errors.append('FCR disabled')
    if fcr.get('teacher_labels', True):
        errors.append('FCR must be teacher-free')
    if int(fcr.get('additional_evidence_queries', -1)) != 0:
        errors.append('FCR added evidence queries')
    if fcr.get('beam_swap_repair', True):
        errors.append('historical beam/swap/repair search was reopened')
    if fcr.get('objective') != 'lexicographic_linf_then_rms_complete_darm_eaf_anchor_star':
        errors.append('FCR objective changed')
    if fcr.get('candidate_pool') != 'frozen_decision_top_m':
        errors.append('FCR candidate pool changed')
    if fcr.get('retained_cardinality') != 'exact_baseline_aocc_selected_count':
        errors.append('FCR retained-cardinality contract changed')
    if fcr.get('acceptance_contract') != 'strict_error_improvement_and_full_m_local_anchor_and_exact_downstream_target_preservation':
        errors.append('FCR monotone acceptance contract changed')
    sel = cfg.get('selector', {}) or {}
    if sel.get('selector_cap_mode') != 'anytime_adverse_certificate':
        errors.append('frozen AOCC selector cap mode changed')
    if sel.get('evidence_certificate_mode') != 'exact_downstream_winner_preservation':
        errors.append('exact downstream evidence certificate changed')
    if not bool(sel.get('force_fill_budget', False)) or int(sel.get('min_selected_atoms', -1)) != 16:
        errors.append('frozen selected-interface fill/cardinality policy changed')

    ic = _icer(cfg)
    bic = _icer(base)
    if not ic.get('enabled'):
        errors.append('ICER disabled')
    if ic.get('dominance_policy') != 'scalar_only':
        errors.append('dominance policy changed')
    if ic.get('incumbent_retention_policy') != 'preserve_admissible_incumbent':
        errors.append('incumbent preservation changed')
    if not ic.get('regret_risk_enabled') or ic.get('retention_regret_risk_enabled') or not ic.get('replacement_regret_risk_enabled'):
        errors.append('replacement-only DRC contract broken')
    if ic.get('regret_risk_feature_mode') != 'evidence_only':
        errors.append('aggregate DRC feature mode changed')
    if ic.get('regret_risk_model_type') != 'local_multiscale_downside_regret_certificate':
        errors.append('DRC estimator changed')
    if ic.get('replacement_local_regret_certificate') != 'mean_minus_downside_rms':
        errors.append('DRC certificate changed')
    if list(ic.get('replacement_local_regret_neighbor_k_values', [])) != [32, 64]:
        errors.append('DRC K changed')
    if ic.get('all_flagged_policy') != 'preserve_legacy_for_structural_guard':
        errors.append('structural guard delegation changed')
    # V28 PTMC must be completely absent: V29 tests interface sufficiency, not a
    # new confirmation head stacked on top of a failed fresh mechanism.
    for key in ic:
        if 'confirmation' in str(key).lower() or 'tail_mode' in str(key).lower():
            if key.startswith('replacement_confirmation') or 'tail_mode' in key:
                errors.append(f'failed PTMC confirmation leaked into V29: {key}')

    frozen_heads = [
        'support_feature_names', 'support_feature_mean', 'support_feature_std', 'support_weights', 'support_bias',
        'scalar_dominance_feature_names', 'scalar_dominance_base_feature_names', 'scalar_dominance_feature_mean',
        'scalar_dominance_feature_std', 'scalar_dominance_weights', 'scalar_dominance_bias',
        'profile_dominance_feature_names', 'profile_dominance_base_feature_names', 'profile_dominance_feature_mean',
        'profile_dominance_feature_std', 'profile_dominance_weights', 'profile_dominance_bias',
    ]
    for key in frozen_heads:
        if ic.get(key) != bic.get(key):
            errors.append('frozen ICER head changed: ' + key)

    mem_path = Path(str(ic.get('replacement_local_regret_memory_path', '')))
    mem_sha = str(ic.get('replacement_local_regret_memory_sha256', ''))
    memory_info = {}
    if not mem_path.is_file():
        errors.append(f'missing FCR DRC memory {mem_path}')
    else:
        got = _sha(mem_path)
        if got != mem_sha:
            errors.append('FCR DRC memory SHA mismatch')
        try:
            with np.load(mem_path, allow_pickle=False) as z:
                names = [str(x) for x in z['feature_names'].reshape(-1)]
                weights = np.asarray(z['feature_metric_weight'], dtype=float).reshape(-1)
                ks = [int(x) for x in z['neighbor_k_values'].reshape(-1)]
                kind = str(z['certificate_kind'].reshape(-1)[0])
                dm = float(z['downside_multiplier'].reshape(-1)[0])
                mem = np.asarray(z['memory_metric_z'])
                y = np.asarray(z['teacher_improvement']).reshape(-1)
            expected = [f'evidence::{n}' for n in v22._ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]
            d = len(expected)
            if names != expected:
                errors.append('aggregate DRC memory schema changed')
            if len(weights) != d or not np.allclose(weights, np.full(d, 1.0 / d), atol=1e-7):
                errors.append('aggregate DRC metric weights changed')
            if ks != [32, 64]:
                errors.append('aggregate DRC memory K changed')
            if kind != 'mean_minus_downside_rms':
                errors.append('aggregate DRC memory certificate changed')
            if abs(dm - 1.0) > 1e-8:
                errors.append('aggregate DRC downside multiplier changed')
            if mem.shape != (len(y), d):
                errors.append('aggregate DRC memory shape mismatch')
            memory_info = {'rows': int(len(y)), 'features': d, 'sha256': got, 'kind': kind}
        except Exception as exc:
            errors.append(f'FCR DRC memory read failed: {exc}')

    report = {
        'pass': not errors,
        'errors': errors,
        'algorithm_version': cfg.get('metadata', {}).get('algorithm_version'),
        'fcr_contract': fcr,
        'memory': memory_info,
        'fixed_constants': {'B_max': 16, 'M': 24, 'K': [32, 64], 'downside_multiplier': 1.0, 'decision_boundary': 0.0},
        'failed_ptmc_removed': not any('PTMC' in e or 'confirmation leaked' in e for e in errors),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    if errors:
        raise SystemExit('STOP CONTRACT: ' + '; '.join(errors))
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
