from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description='V64.3.23 two-block RCR screen checker.')
    ap.add_argument('--split-a-report',required=True); ap.add_argument('--split-b-report',required=True); ap.add_argument('--train-fit-report',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    A=json.load(open(a.split_a_report)); B=json.load(open(a.split_b_report)); T=json.load(open(a.train_fit_report))
    if A.get('split_name')!='A' or B.get('split_name')!='B': raise SystemExit('STOP DATA: split report identity mismatch')
    train=bool(T.get('train_gate_pass',False))
    both=bool(train and A.get('full_split_pass',False) and B.get('full_split_pass',False))
    signed_both=bool(A.get('signed_profile_ranking_incremental',False) and B.get('signed_profile_ranking_incremental',False))
    transition_both=bool(A.get('transition_conditioning_incremental_diagnostic',False) and B.get('transition_conditioning_incremental_diagnostic',False))
    if both:
        next_action='double_fresh_pass_freeze_V23_evidence_local_RCR_then_run_one_independent_full_val_reproduction_only_test_closed_loop_still_forbidden'
    elif not train:
        next_action='TRAIN_local_regret_path_contract_failed_do_not_run_or_tune_fresh_validation'
    elif not A.get('full_split_pass',False):
        next_action='split_A_failed_follow_split_A_next_action_no_pooled_rescue'
    else:
        next_action='split_B_failed_follow_split_B_next_action_no_pooled_rescue'
    rep={'audit':'v64_3_23_eaf_icer_rcr_double_fresh_screen','train_gate_pass':train,'split_A_pass':bool(A.get('full_split_pass',False)),'split_B_pass':bool(B.get('full_split_pass',False)),'both_independent_500_scene_blocks_pass':both,'transition_conditioning_incremental_both_diagnostic':transition_both,'transition_conditioning_required_for_main_promotion':False,'signed_profile_ranking_incremental_both':signed_both,'full_promotion_to_independent_full_val_reproduction':both,'test_closed_loop_allowed':False,'next_action':next_action,'split_A_next_action':A.get('next_action'),'split_B_next_action':B.get('next_action')}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rep,indent=2,sort_keys=True),encoding='utf-8');print(json.dumps(rep,indent=2,sort_keys=True))

if __name__=='__main__':main()
