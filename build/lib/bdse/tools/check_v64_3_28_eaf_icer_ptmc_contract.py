from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22
import bdse.tools.fit_v64_3_28_eaf_icer_ptmc as v28
from bdse.planner.tournament import _ICER_SEMANTIC_TYPE_FEATURE_NAMES


def _ic(cfg):
    return cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_memory(path: Path, sha: str, expected_names: list[str], errors: list[str]) -> dict[str, object]:
    info: dict[str, object] = {}
    if not path.is_file():
        errors.append(f"missing aggregate memory {path}")
        return info
    got = _sha(path)
    if got != sha:
        errors.append("aggregate memory SHA mismatch")
    try:
        with np.load(path, allow_pickle=False) as z:
            names = [str(x) for x in z["feature_names"].reshape(-1)]
            weights = np.asarray(z["feature_metric_weight"], dtype=float).reshape(-1)
            ks = [int(x) for x in z["neighbor_k_values"].reshape(-1)]
            kind = str(z["certificate_kind"].reshape(-1)[0])
            dm = float(z["downside_multiplier"].reshape(-1)[0])
            mem = np.asarray(z["memory_metric_z"])
            y = np.asarray(z["teacher_improvement"]).reshape(-1)
        d = len(expected_names)
        if names != expected_names: errors.append("aggregate memory feature schema mismatch")
        if len(weights) != d or not np.allclose(weights, np.full(d, 1.0/d), atol=1e-7): errors.append("aggregate metric weights changed")
        if ks != [32,64]: errors.append("aggregate K changed")
        if kind != "mean_minus_downside_rms": errors.append("aggregate certificate kind changed")
        if abs(dm-1.0)>1e-8: errors.append("aggregate downside multiplier changed")
        if mem.shape != (len(y), d): errors.append("aggregate memory shape mismatch")
        info = {"rows": int(len(y)), "features": d, "sha256": got, "kind": kind}
    except Exception as exc:
        errors.append(f"aggregate memory read failed: {exc}")
    return info


def _check_tail_model(path: Path, sha: str, expected_names: list[str], errors: list[str]) -> dict[str, object]:
    info: dict[str, object] = {}
    if not path.is_file():
        errors.append(f"missing tail-mode model {path}")
        return info
    got = _sha(path)
    if got != sha:
        errors.append("tail-mode model SHA mismatch")
    try:
        with np.load(path, allow_pickle=False) as z:
            names = [str(x) for x in z["feature_names"].reshape(-1)]
            arrays = {k: np.asarray(z[k], dtype=float).reshape(-1) for k in [
                "feature_mean","feature_std","catastrophic_mean","catastrophic_var","benign_mean","benign_var"
            ]}
            thr = float(np.asarray(z["risk_threshold"]).reshape(-1)[0])
            label = float(np.asarray(z["catastrophic_delta_threshold"]).reshape(-1)[0])
            cov = float(np.asarray(z["positive_proposal_coverage"]).reshape(-1)[0])
            ncat = int(np.asarray(z["catastrophic_count"]).reshape(-1)[0])
            nben = int(np.asarray(z["benign_count"]).reshape(-1)[0])
            nprop = int(np.asarray(z["calibration_proposal_count"]).reshape(-1)[0])
            npos = int(np.asarray(z["calibration_positive_proposal_count"]).reshape(-1)[0])
        d=len(expected_names)
        if names != expected_names: errors.append("tail-mode feature schema mismatch")
        if any(len(a)!=d for a in arrays.values()): errors.append("tail-mode array shape mismatch")
        if any(not np.all(np.isfinite(a)) for a in arrays.values()) or not np.isfinite(thr): errors.append("tail-mode model non-finite")
        if np.any(arrays["feature_std"]<=0) or np.any(arrays["catastrophic_var"]<=0) or np.any(arrays["benign_var"]<=0): errors.append("tail-mode model invalid variance")
        if abs(label-v28.CATASTROPHIC_DELTA_THRESHOLD)>1e-8: errors.append("catastrophic label threshold changed")
        if abs(cov-v28.POSITIVE_PROPOSAL_COVERAGE)>1e-6: errors.append("positive-proposal coverage changed")
        if ncat < 32 or nben < 128 or nprop < 64 or npos < 32: errors.append("tail-mode TRAIN support too small")
        info={"features":d,"sha256":got,"risk_threshold":thr,"catastrophic_delta_threshold":label,"positive_proposal_coverage":cov,"catastrophic_count":ncat,"benign_count":nben,"calibration_proposal_count":nprop,"calibration_positive_proposal_count":npos}
    except Exception as exc:
        errors.append(f"tail-mode model read failed: {exc}")
    return info


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--expect', choices=['aggregate-downside','tail-mode-confirmed'], required=True)
    ap.add_argument('--frozen-v20-dual-config', required=True)
    ap.add_argument('--output', required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    base=yaml.safe_load(Path(args.frozen_v20_dual_config).read_text(encoding='utf-8'))
    ic,bic=_ic(cfg),_ic(base)
    main=args.expect=='tail-mode-confirmed'
    errors: list[str]=[]
    if not ic.get('enabled'): errors.append('ICER disabled')
    if ic.get('dominance_policy')!='scalar_only': errors.append('dominance must remain scalar_only')
    if ic.get('incumbent_retention_policy')!='preserve_admissible_incumbent': errors.append('incumbent preservation changed')
    if not ic.get('regret_risk_enabled') or ic.get('retention_regret_risk_enabled') or not ic.get('replacement_regret_risk_enabled'): errors.append('replacement-only risk contract broken')
    if ic.get('regret_risk_feature_mode')!='evidence_only': errors.append('aggregate proposal feature mode changed')
    expected_model='local_multiscale_downside_regret_with_global_type_tail_confirmation' if main else 'local_multiscale_downside_regret_certificate'
    if ic.get('regret_risk_model_type')!=expected_model: errors.append('risk model mismatch')
    if ic.get('replacement_local_regret_certificate')!='mean_minus_downside_rms': errors.append('aggregate certificate changed')
    if list(ic.get('replacement_local_regret_neighbor_k_values',[]))!=[32,64]: errors.append('aggregate K changed')
    if ic.get('all_flagged_policy')!='preserve_legacy_for_structural_guard': errors.append('structural delegation changed')
    agg_names=[f'evidence::{n}' for n in v22._ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]
    memories={'aggregate':_check_memory(Path(str(ic.get('replacement_local_regret_memory_path',''))),str(ic.get('replacement_local_regret_memory_sha256','')),agg_names,errors)}
    models={}
    if main:
        if ic.get('replacement_confirmation_regret_risk_feature_mode')!='semantic_type_only': errors.append('tail confirmation feature mode changed')
        if ic.get('replacement_selection_monotonicity')!='selected_replacements_are_subset_of_V25_aggregate_DRC_selected_replacements_no_fallback': errors.append('no-fallback subset invariant missing')
        if abs(float(ic.get('replacement_confirmation_tail_mode_label_threshold',999))-v28.CATASTROPHIC_DELTA_THRESHOLD)>1e-8: errors.append('config catastrophic label threshold changed')
        if abs(float(ic.get('replacement_confirmation_positive_proposal_coverage',-1))-v28.POSITIVE_PROPOSAL_COVERAGE)>1e-8: errors.append('config proposal coverage changed')
        names=[f'semantic_type::{n}' for n in _ICER_SEMANTIC_TYPE_FEATURE_NAMES]
        models['type_tail_mode']=_check_tail_model(Path(str(ic.get('replacement_confirmation_tail_mode_model_path',''))),str(ic.get('replacement_confirmation_tail_mode_model_sha256','')),names,errors)
    frozen=[
        'support_feature_names','support_feature_mean','support_feature_std','support_weights','support_bias',
        'scalar_dominance_feature_names','scalar_dominance_base_feature_names','scalar_dominance_feature_mean','scalar_dominance_feature_std','scalar_dominance_weights','scalar_dominance_bias',
        'profile_dominance_feature_names','profile_dominance_base_feature_names','profile_dominance_feature_mean','profile_dominance_feature_std','profile_dominance_weights','profile_dominance_bias',
    ]
    for key in frozen:
        if ic.get(key)!=bic.get(key): errors.append('frozen head changed: '+key)
    report={'pass':not errors,'errors':errors,'expect':args.expect,'memories':memories,'models':models,'frozen_head_identity':not any(x.startswith('frozen head') for x in errors),'selection_subset_contract':bool(main),'frozen_constants':{'catastrophic_delta_threshold':v28.CATASTROPHIC_DELTA_THRESHOLD,'positive_proposal_coverage':v28.POSITIVE_PROPOSAL_COVERAGE,'K':[32,64],'downside_multiplier':1.0}}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    if errors: raise SystemExit('STOP CONTRACT: '+'; '.join(errors))
    print(json.dumps(report,sort_keys=True))

if __name__=='__main__': main()
