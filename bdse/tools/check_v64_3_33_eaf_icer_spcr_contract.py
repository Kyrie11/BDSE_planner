from __future__ import annotations
import argparse,json,math
from pathlib import Path
import yaml

def _ic(c): return c.get('runtime',{}).get('decisive_frontier_value',{}).get('incumbent_contrastive_extremal_recovery',{}) or {}
def main():
 ap=argparse.ArgumentParser()
 for x in ['v20-config','preserve-config','mean-config','rank-config','main-config','calibration-report','output']: ap.add_argument('--'+x,required=True)
 a=ap.parse_args(); V,P,M,R,S=[yaml.safe_load(Path(getattr(a,n.replace('-','_'))).read_text()) for n in ['v20-config','preserve-config','mean-config','rank-config','main-config']]; C=json.load(open(a.calibration_report)); vi,pi,mi,ri,si=map(_ic,[V,P,M,R,S]); ms=mi.get('selection_conditioned_intervention_recovery',{}) or {}; rs=ri.get('selection_conditioned_intervention_recovery',{}) or {}; ss=si.get('selection_conditioned_intervention_recovery',{}) or {}
 keys=['feature_names','base_feature_names','feature_mean','feature_std','weights','bias','ridge_lambda','training_target','training_weighting']
 checks={}
 checks['preserve_incumbent_default']=pi.get('incumbent_retention_policy')=='preserve_admissible_incumbent' and not bool((pi.get('selection_conditioned_intervention_recovery',{}) or {}).get('enabled',False))
 checks['all_learned_arms_preserve_incumbent_default']=all(x.get('incumbent_retention_policy')=='preserve_admissible_incumbent' for x in [mi,ri,si])
 checks['mean_is_corrected_control']=bool(ms.get('enabled')) and ms.get('mode')=='mean_rank'
 checks['rank_structured_mode']=bool(rs.get('enabled')) and rs.get('mode')=='rank_only' and rs.get('model_type')=='scene_equal_incumbent_augmented_teacher_best_pair_gap_ridge'
 checks['main_selected_policy_veto']=bool(ss.get('enabled')) and ss.get('mode')=='conformal_veto' and bool(ss.get('no_fallback'))
 checks['rank_main_same_selector']=all(rs.get(k)==ss.get(k) for k in keys)
 q=float(ss.get('conformal_overprediction_quantile',float('nan'))); cq=float(C.get('selected_policy_conformal_quantile',float('nan'))); checks['q_exact']=math.isfinite(q) and q>=0 and abs(q-cq)<=1e-12
 checks['calibration_valid']=C.get('calibration_total_scene_count')==500 and int(C.get('selected_policy_proposal_count',0))>=64 and abs(float(C.get('alpha',0))-.05)<=1e-12 and C.get('calibration_uses_promotion_labels') is False
 checks['no_leverage_or_old_risk_head']=rs.get('leverage_inverse',[])==[] and all(not bool(x.get('regret_risk_enabled',False)) and not bool(x.get('replacement_regret_risk_enabled',False)) and not bool(x.get('retention_regret_risk_enabled',False)) for x in [pi,mi,ri,si])
 checks['same_budget']=V.get('evidence',{}).get('budget')==P.get('evidence',{}).get('budget')==M.get('evidence',{}).get('budget')==R.get('evidence',{}).get('budget')==S.get('evidence',{}).get('budget')
 checks['structural_delegation']=all(x.get('all_flagged_policy')=='preserve_legacy_for_structural_guard' for x in [pi,mi,ri,si])
 ok=all(checks.values()); rep={'audit':'v64_3_33_eaf_icer_spcr_contract','pass':ok,'checks':checks,'interpretation':'SPCR changes the direct selector from edge-wise conditional-mean regression to an incumbent-augmented scene-structured teacher-best-vs-rivals gap objective. Independent calibration is attached to the one frozen policy proposal, and MAIN may only accept that proposal or return incumbent.'}; Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)); print(json.dumps(rep,indent=2,sort_keys=True));
 if not ok: raise SystemExit('V64.3.33 SPCR contract failed')
if __name__=='__main__': main()
