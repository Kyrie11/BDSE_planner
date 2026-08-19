from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,yaml
import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22
from bdse.tools.fit_v64_3_24_eaf_icer_arc import ATTR_NAMES

def _ic(c): return c['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--expect',choices=['aggregate-meanse','aggregate-downside','attribution-meanse','attribution-downside'],required=True);ap.add_argument('--frozen-v20-dual-config',required=True);ap.add_argument('--output',required=True);a=ap.parse_args(); c=yaml.safe_load(Path(a.config).read_text());b=yaml.safe_load(Path(a.frozen_v20_dual_config).read_text());ic,bic=_ic(c),_ic(b);errs=[]
 mode='attribution_resolved' if a.expect.startswith('attribution') else 'evidence_only'; downside=a.expect.endswith('downside'); expected_model='local_multiscale_downside_regret_certificate' if downside else 'local_multiscale_regret_lower_bound'
 if not ic.get('enabled'): errs.append('ICER disabled')
 if ic.get('dominance_policy')!='scalar_only': errs.append('dominance must remain frozen scalar-only')
 if ic.get('incumbent_retention_policy')!='preserve_admissible_incumbent': errs.append('incumbent default-preservation missing')
 if not ic.get('regret_risk_enabled') or ic.get('retention_regret_risk_enabled') or not ic.get('replacement_regret_risk_enabled'): errs.append('replacement-only risk contract broken')
 if ic.get('regret_risk_feature_mode')!=mode: errs.append('risk feature mode mismatch')
 if ic.get('regret_risk_model_type')!=expected_model: errs.append('risk model mismatch')
 if ic.get('all_flagged_policy')!='preserve_legacy_for_structural_guard': errs.append('structural delegation changed')
 p=Path(str(ic.get('replacement_local_regret_memory_path',''))); sha=str(ic.get('replacement_local_regret_memory_sha256','')); info={}
 if not p.is_file(): errs.append(f'missing memory {p}')
 else:
  got=hashlib.sha256(p.read_bytes()).hexdigest();
  if got!=sha: errs.append('memory SHA mismatch')
  try:
   with np.load(p,allow_pickle=False) as z:
    names=[str(x) for x in z['feature_names'].reshape(-1)]; w=np.asarray(z['feature_metric_weight'],float).reshape(-1); ks=[int(x) for x in z['neighbor_k_values'].reshape(-1)]; kind=str(z['certificate_kind'].reshape(-1)[0]); dm=float(z['downside_multiplier'].reshape(-1)[0]); mem=np.asarray(z['memory_metric_z']); y=np.asarray(z['teacher_improvement']).reshape(-1)
   exp=[f'evidence::{n}' for n in v22._ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]
   if mode=='attribution_resolved': exp += [f'attribution::{n}' for n in ATTR_NAMES]
   if names!=exp: errs.append('memory feature schema mismatch')
   if ks!=[32,64]: errs.append('K must remain fixed 32/64')
   if kind != ('mean_minus_downside_rms' if downside else 'mean_minus_standard_error'): errs.append('certificate kind mismatch')
   if abs(dm-1.0)>1e-8: errs.append('downside multiplier must remain fixed 1.0')
   if mem.shape!=(len(y),len(names)) or len(w)!=len(names): errs.append('memory shape mismatch')
   if mode=='attribution_resolved' and (len(w)!=50 or any(abs(float(w[s:e].sum())-1)>1e-5 for s,e in [(0,18),(18,34),(34,50)])): errs.append('attribution metric groups not equally balanced')
   info=dict(rows=int(len(y)),features=len(names),kind=kind,sha256=got)
  except Exception as e: errs.append(f'memory read failed {e}')
 frozen=['support_feature_names','support_feature_mean','support_feature_std','support_weights','support_bias','scalar_dominance_feature_names','scalar_dominance_base_feature_names','scalar_dominance_feature_mean','scalar_dominance_feature_std','scalar_dominance_weights','scalar_dominance_bias','profile_dominance_feature_names','profile_dominance_base_feature_names','profile_dominance_feature_mean','profile_dominance_feature_std','profile_dominance_weights','profile_dominance_bias']
 for k in frozen:
  if ic.get(k)!=bic.get(k): errs.append('frozen head changed: '+k)
 rep={'pass':not errs,'errors':errs,'expect':a.expect,'memory':info,'frozen_head_identity':not any(x.startswith('frozen head') for x in errs)};Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True));
 if errs: raise SystemExit('STOP CONTRACT: '+'; '.join(errs))
 print(json.dumps(rep,sort_keys=True))
if __name__=='__main__': main()
