from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,yaml
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FEATURE_NAMES,_select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _structured_scores
from bdse.tools.fit_v64_3_38_eaf_icer_davr import MIN_VALUE_CAL_PROPOSALS
from bdse.tools.fit_v64_3_39_eaf_icer_cfsr import _fit_translation
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import _endpoint_scene,_pred
EXPECTED_SCENES=500

def _tokens(p): return [str(json.loads(x).get('scenario_token','')) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--calibration-rows',required=True); ap.add_argument('--calibration-edges',required=True); ap.add_argument('--epv-config',required=True); ap.add_argument('--output-epvr-config',required=True); ap.add_argument('--output-report',required=True); a=ap.parse_args()
 rt=_tokens(a.calibration_rows)
 if len(rt)!=500 or len(set(rt))!=500: raise SystemExit('V41 CAL rows must be 500 unique scenes')
 groups={t:[] for t in rt}; allowed=set(rt)
 for line in Path(a.calibration_edges).read_text().splitlines():
  if not line.strip(): continue
  r=json.loads(line); t=str(r.get('scenario_token',''))
  if t not in allowed: raise SystemExit('V41 CAL edge outside row set')
  groups[t].append(r)
 scene=_endpoint_scene(groups); cfg=yaml.safe_load(Path(a.epv_config).read_text()); sc=cfg['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['selection_conditioned_intervention_recovery']
 if str(sc.get('post_selection_value_mode',''))!='endpoint_potential_value': raise SystemExit('V41 CAL requires raw endpoint_potential_value config')
 rmean=np.asarray(sc['feature_mean'],dtype=float); rstd=np.asarray(sc['feature_std'],dtype=float); rw=np.asarray(sc['weights'],dtype=float)
 if list(sc.get('feature_names',[])) != FEATURE_NAMES or rmean.size!=len(FEATURE_NAMES) or rstd.size!=len(FEATURE_NAMES) or rw.size!=len(FEATURE_NAMES) or np.max(np.abs(rmean))>1e-12 or abs(float(sc.get('bias',0.0)))>1e-12: raise SystemExit('V41 CAL frozen RSMR schema/zero-origin contract invalid')
 rsm=(rw,rstd,{'source':'frozen_full_TRAIN_RSMR'})
 model={'mode':'epv','names':list(sc['post_selection_endpoint_feature_names']),'scale':np.asarray(sc['post_selection_endpoint_feature_scale'],dtype=float),'weights':np.asarray(sc['post_selection_endpoint_weights'],dtype=float),'bias':0.0}
 ys=[]; pv=[]; used=[]
 for t in rt:
  ss=scene.get(t,[])
  if not ss: continue
  score=_structured_scores(ss,rsm); idx=_select(ss,score)
  if idx is None: continue
  ys.append(float(ss[idx]['y'])); pv.append(_pred(ss[idx],model)); used.append(t)
 if len(used)<MIN_VALUE_CAL_PROPOSALS: raise SystemExit(f'V41 CAL proposals {len(used)} < {MIN_VALUE_CAL_PROPOSALS}')
 fit=_fit_translation(np.asarray(pv),np.asarray(ys),'endpoint_potential_value'); bias=float(fit['selected_policy_bias'])
 sc['post_selection_value_mode']='endpoint_potential_shift'; sc['post_selection_selected_bias']=bias; sc['post_selection_value_training']='dense_all_TRAIN_endpoint_potential_plus_independent_CAL500_unit_slope_translation'; cfg.setdefault('metadata',{})['algorithm_version']='V64.3.41-EAF-ICER-EPVR'; cfg.setdefault('provenance',{})['algorithm_version']='V64.3.41-EAF-ICER-EPVR'; cfg.setdefault('experiment',{})['name']='v64_3_41_eaf_icer_epvr'; cfg['experiment']['algorithm']='V64.3.41 endpoint-potential value recovery'
 Path(a.output_epvr_config).write_text(yaml.safe_dump(cfg,sort_keys=False)); rep={'audit':'v64_3_41_epvr_independent_CAL500_translation','selected_policy_proposal_count':len(used),'selected_policy_proposal_count_min':MIN_VALUE_CAL_PROPOSALS,'calibration_tokens_sha256':hashlib.sha256(('\n'.join(rt)+'\n').encode()).hexdigest(),'selected_policy_tokens_sha256':hashlib.sha256(('\n'.join(used)+'\n').encode()).hexdigest(),'translation_fit':fit,'causal_contract':'RSMR winner is frozen before endpoint potential value; CAL500 learns translation only and cannot rerank or create proposal.'}; Path(a.output_report).write_text(json.dumps(rep,indent=2,sort_keys=True)); print(json.dumps({'pass':True,'selected_policy_proposals':len(used),'output':a.output_epvr_config},sort_keys=True))
if __name__=='__main__': main()
