from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import yaml
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FEATURE_NAMES, _select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _structured_scores
from bdse.tools.fit_v64_3_38_eaf_icer_davr import MIN_VALUE_CAL_PROPOSALS
from bdse.tools.fit_v64_3_39_eaf_icer_cfsr import _fit_translation
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import _pred as _epv_pred
from bdse.tools.fit_v64_3_43_eaf_icer_cfrv import _scene, _quality_x, _future_x, _lin, FUTURE_ROBUST_NAME

EXPECTED_SCENES=500

def _tokens(p): return [str(json.loads(x).get('scenario_token','')) for x in Path(p).read_text().splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--calibration-rows',required=True); ap.add_argument('--calibration-edges',required=True); ap.add_argument('--future-robust-config',required=True); ap.add_argument('--output-main-config',required=True); ap.add_argument('--output-report',required=True); a=ap.parse_args()
    rt=_tokens(a.calibration_rows)
    if len(rt)!=EXPECTED_SCENES or len(set(rt))!=EXPECTED_SCENES: raise SystemExit('V43 CAL rows must be 500 unique scenes')
    groups={t:[] for t in rt}; allowed=set(rt)
    for line in Path(a.calibration_edges).read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line); t=str(r.get('scenario_token',''))
        if t not in allowed: raise SystemExit('V43 CAL edge outside row set')
        groups[t].append(r)
    scene=_scene(groups)
    cfg=yaml.safe_load(Path(a.future_robust_config).read_text()); sc=cfg['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['selection_conditioned_intervention_recovery']
    if str(sc.get('post_selection_value_mode',''))!='endpoint_potential_quality_future_response_robust': raise SystemExit('V43 CAL requires raw future-response robust config')
    rmean=np.asarray(sc['feature_mean'],float); rstd=np.asarray(sc['feature_std'],float); rw=np.asarray(sc['weights'],float)
    if list(sc.get('feature_names',[]))!=FEATURE_NAMES or rmean.size!=len(FEATURE_NAMES) or rstd.size!=len(FEATURE_NAMES) or rw.size!=len(FEATURE_NAMES) or np.max(np.abs(rmean))>1e-12 or abs(float(sc.get('bias',0.0)))>1e-12: raise SystemExit('V43 CAL frozen RSMR schema invalid')
    rsm=(rw,rstd,{'source':'frozen_full_TRAIN_RSMR'})
    epv={'mode':'epv','names':list(sc['post_selection_endpoint_feature_names']),'scale':np.asarray(sc['post_selection_endpoint_feature_scale'],float),'weights':np.asarray(sc['post_selection_endpoint_weights'],float),'bias':0.0}
    q={'names':list(sc['post_selection_quality_observable_names']),'scale':np.asarray(sc['post_selection_quality_observable_scale'],float),'weights':np.asarray(sc['post_selection_quality_observable_weights'],float),'bias':0.0}
    response={'names':[str(sc['post_selection_future_response_observable_name'])],'scale':np.asarray([float(sc['post_selection_future_response_scale'])]),'weights':np.asarray([float(sc['post_selection_future_response_weight'])]),'bias':0.0}
    if response['names'][0]!=FUTURE_ROBUST_NAME: raise SystemExit('V43 CAL robust response name mismatch')
    ys=[]; pv=[]; used=[]
    for t in rt:
        ss=scene.get(t,[])
        if not ss: continue
        idx=_select(ss,_structured_scores(ss,rsm))
        if idx is None: continue
        x=ss[idx]; pred=float(_epv_pred(x,epv)+_lin(x,q,_quality_x(x))+_lin(x,response,_future_x(x,FUTURE_ROBUST_NAME)))
        ys.append(float(x['y'])); pv.append(pred); used.append(t)
    if len(used)<MIN_VALUE_CAL_PROPOSALS: raise SystemExit(f'V43 CAL proposals {len(used)} < {MIN_VALUE_CAL_PROPOSALS}')
    fit=_fit_translation(np.asarray(pv),np.asarray(ys),'quality_plus_runtime_future_response_robust')
    sc['post_selection_value_mode']='endpoint_potential_quality_future_response_robust_shift'; sc['post_selection_selected_bias']=float(fit['selected_policy_bias']); sc['post_selection_value_training']='dense_all_TRAIN_quality_plus_runtime_future_response_robust_plus_independent_CAL500_unit_slope_translation'
    cfg.setdefault('metadata',{})['algorithm_version']='V64.3.43-EAF-ICER-CFRV'; cfg.setdefault('provenance',{})['algorithm_version']='V64.3.43-EAF-ICER-CFRV'; cfg.setdefault('experiment',{})['name']='v64_3_43_eaf_icer_cfrv'; cfg['experiment']['algorithm']='V64.3.43 counterfactual future-response valuation'
    Path(a.output_main_config).write_text(yaml.safe_dump(cfg,sort_keys=False))
    rep={'audit':'v64_3_43_cfrv_independent_CAL500_translation','selected_policy_proposal_count':len(used),'selected_policy_proposal_count_min':MIN_VALUE_CAL_PROPOSALS,'calibration_tokens_sha256':hashlib.sha256(('\n'.join(rt)+'\n').encode()).hexdigest(),'selected_policy_tokens_sha256':hashlib.sha256(('\n'.join(used)+'\n').encode()).hexdigest(),'translation_fit':fit,'causal_contract':'RSMR winner and full-TRAIN QUALITY+future-response valuation are frozen; CAL500 learns unit-slope translation only.'}
    Path(a.output_report).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n'); print(json.dumps({'pass':True,'selected_policy_proposals':len(used),'output':a.output_main_config},sort_keys=True))
if __name__=='__main__':main()
