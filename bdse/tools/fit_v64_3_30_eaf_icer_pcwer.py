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
CATASTROPHIC_DELTA = -0.5
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


def _pcwer_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_TRAIN_SCENES:
        raise SystemExit(f'STOP TRAIN DATA: expected {EXPECTED_TRAIN_SCENES} PCWER rows, got {len(rows)}')
    tokens = [str(r.get('scenario_token', '')) for r in rows]
    if not all(tokens) or len(set(tokens)) != EXPECTED_TRAIN_SCENES:
        raise SystemExit('STOP TRAIN DATA: PCWER per-sample tokens are not exactly 3000 unique scenes')

    prefix = 'selector_proposal_conditioned_witness_rebinding_'
    enabled = np.asarray([_f(r, prefix + 'enabled', 0.0) for r in rows]) >= 0.5
    attempted = np.asarray([_f(r, prefix + 'attempted', 0.0) for r in rows]) >= 0.5
    accepted = np.asarray([_f(r, prefix + 'accepted', 0.0) for r in rows]) >= 0.5
    reasons = Counter(int(round(_f(r, prefix + 'reason_code', -1.0))) for r in rows)
    accepted_contract: list[bool] = []
    margin_reduction: list[float] = []
    attr_reduction: list[float] = []
    for i, r in enumerate(rows):
        if not accepted[i]:
            continue
        bm = _f(r, prefix + 'baseline_margin_linf_error')
        fm = _f(r, prefix + 'final_margin_linf_error')
        ba = _f(r, prefix + 'baseline_attribution_linf_error')
        fa = _f(r, prefix + 'final_attribution_linf_error')
        br = _f(r, prefix + 'baseline_margin_rms_error')
        fr = _f(r, prefix + 'final_margin_rms_error')
        bar = _f(r, prefix + 'baseline_attribution_rms_error')
        far = _f(r, prefix + 'final_attribution_rms_error')
        before = (bm, ba, br, bar); after = (fm, fa, fr, far)
        finite = all(np.isfinite(x) for x in (*before, *after))
        strict = False
        if finite:
            for c, b in zip(after, before):
                if c < b - 1e-8:
                    strict = True; break
                if c > b + 1e-8:
                    break
        ok = bool(
            _f(r, prefix + 'cardinality_preserved', 0.0) >= 0.5
            and _f(r, prefix + 'budget_preserved', 0.0) >= 0.5
            and _f(r, prefix + 'proposal_lock', 0.0) >= 0.5
            and int(round(_f(r, prefix + 'candidate_proposal_action', -1.0)))
                == int(round(_f(r, prefix + 'baseline_proposal_action', -2.0)))
            and int(round(_f(r, prefix + 'candidate_incumbent_action', -1.0)))
                == int(round(_f(r, prefix + 'baseline_incumbent_action', -2.0)))
            and int(round(_f(r, prefix + 'candidate_anchor_action', -1.0)))
                == int(round(_f(r, prefix + 'baseline_anchor_action', -2.0)))
            and finite and strict
        )
        accepted_contract.append(ok)
        if np.isfinite(bm) and np.isfinite(fm):
            margin_reduction.append(float(bm - fm))
        if np.isfinite(ba) and np.isfinite(fa):
            attr_reduction.append(float(ba - fa))

    accepted_count = int(np.sum(accepted))
    report = {
        'scene_count': len(rows),
        'all_rows_enabled': bool(np.all(enabled)),
        'attempted_count': int(np.sum(attempted)),
        'accepted_count': accepted_count,
        'accepted_rate': float(accepted_count / len(rows)),
        'all_accepted_contracts_valid': bool(accepted_contract and all(accepted_contract)),
        'accepted_margin_linf_reduction_mean': float(np.mean(margin_reduction)) if margin_reduction else float('nan'),
        'accepted_attribution_linf_reduction_mean': float(np.mean(attr_reduction)) if attr_reduction else float('nan'),
        'reason_code_counts': {str(k): int(v) for k, v in sorted(reasons.items())},
        'mechanism_active': bool(accepted_count >= MIN_ACTIVE_REBINDS),
    }
    report['gate_pass'] = bool(
        report['all_rows_enabled']
        and report['all_accepted_contracts_valid']
        and report['mechanism_active']
    )
    return report


def _proposal_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return the exact runtime PCWER proposal-lock map from per-scene diagnostics.

    Do not reconstruct this from frontier edges: V25's memory-efficient frontier
    loader intentionally drops `icer_selected_action`, and, more importantly, the
    final V20 ICER action is not the V30 proposal contract on fail-closed rebinding
    paths.  The selector diagnostics are the authoritative source because they are
    emitted at the generation/rebinding boundary that supplies
    `recovery_proposal_action` to the downstream tournament.
    """
    out: dict[str, int] = {}
    prefix = 'selector_proposal_conditioned_witness_rebinding_'
    for r in rows:
        token = str(r.get('scenario_token', ''))
        if not token or _f(r, prefix + 'proposal_lock', 0.0) < 0.5:
            continue
        q = int(round(_f(r, prefix + 'baseline_proposal_action', -1.0)))
        incumbent = int(round(_f(r, prefix + 'baseline_incumbent_action', -1.0)))
        anchor = int(round(_f(r, prefix + 'baseline_anchor_action', -1.0)))
        if min(q, incumbent, anchor) < 0 or q in {incumbent, anchor}:
            raise SystemExit(f'STOP TRAIN INSTRUMENTATION: invalid locked PCWER proposal for {token}: q={q} incumbent={incumbent} anchor={anchor}')
        out[token] = q
    return out


def _selection_locked(data: dict[str, Any], score: np.ndarray, hold: set[str], proposal: dict[str, int]) -> dict[str, Any]:
    toks = data['tok']; delta = data['delta']; action = data['action']
    selected: list[float] = []; opportunities = captured = scene_count = 0
    for token in sorted(hold):
        idx = np.flatnonzero(toks == token)
        if not len(idx):
            continue
        scene_count += 1
        opportunities += int(np.any(delta[idx] > 0.0))
        q = int(proposal.get(token, -1))
        qidx = idx[action[idx] == q]
        if len(qidx) != 1:
            continue
        j = int(qidx[0])
        if np.isfinite(score[j]) and score[j] > 0.0:
            selected.append(float(delta[j]))
            captured += int(delta[j] > 0.0)
    arr = np.asarray(selected, dtype=np.float64)
    neg = np.minimum(arr, 0.0)
    return {
        'scene_count': int(scene_count), 'count': int(len(arr)),
        'precision': float(np.mean(arr > 0.0)) if len(arr) else float('nan'),
        'sum': float(arr.sum()) if len(arr) else 0.0,
        'mean': float(arr.mean()) if len(arr) else float('nan'),
        'worst': float(arr.min()) if len(arr) else float('nan'),
        'negative_rms': float(np.sqrt(np.mean(neg * neg))) if len(arr) else float('nan'),
        'opportunities': int(opportunities),
        'capture': float(captured / opportunities) if opportunities else float('nan'),
    }


def _crossfit_locked(data: dict[str, Any], proposal: dict[str, int]) -> dict[str, Any]:
    X=data['X']; y=data['delta']; toks=data['tok']
    unique=sorted(set(map(str,toks)))
    folds=[]
    for f in range(v25.FOLDS):
        hold={t for t in unique if v25._fold(t)==f}
        if len(hold) < v25.MIN_FOLD_SCENES:
            raise SystemExit(f'STOP TRAIN SPLIT: fold too small {f}: {len(hold)}')
        hm=np.asarray([str(t) in hold for t in toks],dtype=bool)
        score=np.full(len(X),np.nan,dtype=np.float64)
        score[hm]=v25._score(X[~hm],y[~hm],X[hm],'downside_rms')
        m=_selection_locked(data,score,hold,proposal)
        m['fold']=int(f); m['hold_scenes']=int(len(hold))
        m['path_safe']=bool(m['count'] >= v25.MIN_SELECTED and m['sum'] >= -1e-9)
        m['catastrophe_free']=bool(m['count'] == 0 or (np.isfinite(m['worst']) and m['worst'] > CATASTROPHIC_DELTA))
        folds.append(m)
    return {
        'mode':'proposal_locked_aggregate_evidence_only',
        'certificate':'mean_minus_downside_rms',
        'folds':folds,
        'all_folds_path_safe':bool(all(x['path_safe'] for x in folds)),
        'all_folds_catastrophe_free':bool(all(x['catastrophe_free'] for x in folds)),
        'fold_pass_count':int(sum(x['path_safe'] for x in folds)),
        'selected_count':int(sum(x['count'] for x in folds)),
        'teacher_improvement_sum':float(sum(x['sum'] for x in folds)),
        'mean_precision':float(np.nanmean([x['precision'] for x in folds])) if any(np.isfinite(x['precision']) for x in folds) else float('nan'),
        'mean_capture':float(np.nanmean([x['capture'] for x in folds])) if any(np.isfinite(x['capture']) for x in folds) else float('nan'),
    }


def _retag_cfg(base: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    cfg = v25._cfg(base, memory, 'downside_rms', 'pcwer_aggregate_downside')
    version='V64.3.30-EAF-ICER-PCWER-DRC'
    cfg.setdefault('metadata', {})['algorithm_version']=version
    cfg['metadata']['fcr_post_eaf_rebinding']=False
    cfg['metadata']['pcwer_proposal_conditioned_rebinding']=True
    cfg.setdefault('provenance', {})['algorithm_version']=version
    cfg['provenance']['screening_only']=True
    exp=cfg.setdefault('experiment', {})
    exp['name']='v64_3_30_eaf_icer_pcwer_aggregate_downside'
    exp['algorithm']='V64.3.30 EAF-ICER-PCWER: proposal-conditioned fixed-B witness rebinding with same-proposal DRC confirmation'
    exp['mechanism_chain']=(
        'fixed B=16/M=24 -> AOCC/EAF -> risk-free unique direct proposal -> proposal/anchor + '
        'proposal/incumbent witness rebinding -> same-proposal aggregate downside-regret confirmation -> '
        'incumbent default -> unchanged final/structural guards'
    )
    exp['calibration_protocol']=(
        'PCWER is runtime teacher-free. The unchanged V25 aggregate downside certificate is re-fit on the same '
        'frozen 3000 TRAIN scenes under the PCWER representation. K={32,64}, downside multiplier=1 and zero boundary '
        'remain frozen. Cross-fitting evaluates the same proposal-locked operator and now requires catastrophe-free folds.'
    )
    return cfg


def main() -> None:
    ap=argparse.ArgumentParser(description='Fit unchanged DRC after V64.3.30 PCWER and audit proposal-locked TRAIN safety.')
    ap.add_argument('--train-frontier-edges',required=True)
    ap.add_argument('--train-rows',required=True)
    ap.add_argument('--base-pcwer-v20-config',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--output-train-token-file',required=True)
    ap.add_argument('--output-report',required=True)
    args=ap.parse_args()
    edge_path=Path(args.train_frontier_edges); row_path=Path(args.train_rows)
    if not edge_path.is_file() or edge_path.stat().st_size<=0: raise SystemExit(f'STOP TRAIN DATA: missing frontier {edge_path}')
    if not row_path.is_file() or row_path.stat().st_size<=0: raise SystemExit(f'STOP TRAIN DATA: missing rows {row_path}')
    by,nrows=v25._load_minimal_scenes(edge_path)
    if len(by)!=EXPECTED_TRAIN_SCENES: raise SystemExit(f'STOP TRAIN DATA: expected 3000 scenes, got {len(by)}')
    sample_rows=_load_rows(row_path)
    data=v25._build(by); proposal=_proposal_map(sample_rows)
    crossfit=_crossfit_locked(data,proposal)
    pcwer=_pcwer_audit(sample_rows)
    pcwer['proposal_lock_count'] = int(len(proposal))
    pcwer['proposal_lock_map_complete'] = bool(len(proposal) >= pcwer['accepted_count'])
    pcwer['gate_pass'] = bool(pcwer['gate_pass'] and pcwer['proposal_lock_map_complete'])
    drc_gate=bool(
        crossfit['all_folds_path_safe']
        and crossfit['all_folds_catastrophe_free']
        and crossfit['selected_count'] >= v25.MAIN_MIN_SELECTED
        and crossfit['teacher_improvement_sum'] >= -1e-9
    )
    gate=bool(pcwer['gate_pass'] and drc_gate)
    tokens=sorted(by); token_path=Path(args.output_train_token_file); token_path.parent.mkdir(parents=True,exist_ok=True)
    token_path.write_text('\n'.join(tokens)+'\n',encoding='utf-8')
    report={
        'audit':'v64_3_30_eaf_icer_pcwer_train_fit',
        'algorithm':'V64.3.30 EAF-ICER-PCWER-DRC',
        'train_scene_count':len(by),'frontier_row_count':int(nrows),
        'replacement_edges':int(len(data['delta'])),'replacement_scenes':int(data['replacement_scene_count']),
        'population_teacher_positive_fraction':float(np.mean(data['delta']>0.0)),
        'population_teacher_improvement_sum':float(data['delta'].sum()),
        'population_teacher_improvement_worst':float(data['delta'].min()),
        'proposal_locked_drc_crossfit':crossfit,'pcwer_train_audit':pcwer,
        'drc_gate_pass':drc_gate,'train_gate_pass':gate,
        'fresh_validation_used':False,
        'frozen_contract':{
            'B':16,'M':24,'DRC_K':[32,64],'DRC_boundary':0.0,'downside_multiplier':1.0,
            'proposal_lock':'confirm_same_risk_free_proposal_or_preserve_incumbent',
            'catastrophic_delta_threshold':CATASTROPHIC_DELTA,
            'no_validation_tuning':True,
        },
        'input_frontier':{'path':str(edge_path),'bytes':edge_path.stat().st_size,'sha256':v25._sha256_file(edge_path)},
        'input_rows':{'path':str(row_path),'bytes':row_path.stat().st_size,'sha256':v25._sha256_file(row_path)},
        'configs':{},'memories':{},
    }
    v25._write_report(Path(args.output_report),report)
    if not gate:
        raise SystemExit(
            'STOP TRAIN PCWER: proposal-conditioned rebinding or proposal-locked DRC failed the pre-registered TRAIN gate. '
            'Do not tune B/M, witness weights, DRC K/boundary, or add a classifier; inspect the causal failure before fresh.'
        )
    out_dir=Path(args.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
    mem_path=out_dir/'v64_3_30_pcwer_aggregate_downside_memory.npz'
    memory=v25._save_memory(mem_path,data,'downside_rms')
    base=yaml.safe_load(Path(args.base_pcwer_v20_config).read_text(encoding='utf-8'))
    cfg=_retag_cfg(base,memory)
    cfg_path=out_dir/'v64_3_30_pcwer_aggregate_downside.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    report['memories']={'pcwer_aggregate_downside':memory}; report['configs']={'pcwer_aggregate_downside':str(cfg_path)}
    v25._write_report(Path(args.output_report),report)
    print(json.dumps({'pass':True,'train_gate_pass':gate,'pcwer_accepted_count':pcwer['accepted_count'],
                      'drc_fold_pass_count':crossfit['fold_pass_count'],'drc_catastrophe_free':crossfit['all_folds_catastrophe_free'],
                      'drc_selected_count':crossfit['selected_count'],'drc_teacher_improvement_sum':crossfit['teacher_improvement_sum']},sort_keys=True))


if __name__=='__main__':
    main()
