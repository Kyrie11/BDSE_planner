from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split-a',required=True); ap.add_argument('--split-b',required=True); ap.add_argument('--value-report',required=True); ap.add_argument('--fit-report',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    A=json.load(open(a.split_a)); B=json.load(open(a.split_b)); C=json.load(open(a.value_report)); F=json.load(open(a.fit_report))
    eng=bool(A.get('engineering_valid') and B.get('engineering_valid')); train=bool(F.get('train_gate_pass')); cal=bool(C.get('calibration_total_scene_count')==500 and int(C.get('selected_policy_proposal_count',0))>=64 and float(C.get('frozen_rsmr_score_replay_max_abs',1.0))<=1e-5)
    both=bool(eng and train and cal and A.get('full_split_pass') and B.get('full_split_pass'))
    rep={'audit':'v64_3_37_eaf_icer_pvr_double_fresh_screen','engineering_valid_both':eng,'train_nested_proposal_value_gate_pass':train,'independent_CAL500_value_fit_valid':cal,'double_fresh_promotion_pass':both,'scientific_conclusion':'OPVR_reproduces_on_double_fresh' if both else ('engineering_invalid_stop' if not eng else 'PVR_not_promoted_current_frozen_mechanism_fails_at_least_one_independent_gate_or_block'),'next_action':'freeze_OPVR_and_run_one_independent_full_validation_reproduction' if both else 'STOP_do_not_pool_AB_or_sweep_thresholds_use_affine_vs_orthogonal_post_selection_value_failure_slice_for_next_mechanism','train_fit':F,'independent_value_fit':C,'split_A':A,'split_B':B,'protocol':'Exact V36 TRAIN failure prerequisite -> nested TRAIN frozen V34 RSMR ordering -> independent-fold selected-proposal score-affine value diagnostic + score-orthogonal proposal-value main -> if TRAIN passes, independent CAL500 fits both heads on frozen RSMR selected-policy outputs -> untouched A500/B500. No value arm can re-rank challengers, create a proposal, or fall through to second best.'}
    Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n'); print(json.dumps({'pass':both,'engineering':eng,'train':train,'calibration':cal,'conclusion':rep['scientific_conclusion']},sort_keys=True))
if __name__=='__main__': main()
