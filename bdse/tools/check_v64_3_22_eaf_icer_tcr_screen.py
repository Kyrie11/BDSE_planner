from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split-a-report',required=True); ap.add_argument('--split-b-report',required=True); ap.add_argument('--train-fit-report',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    A=json.load(open(a.split_a_report)); B=json.load(open(a.split_b_report)); T=json.load(open(a.train_fit_report))
    both=bool(A.get('main_split_pass',False) and B.get('main_split_pass',False) and T.get('main_train_holdout_safe',False))
    signed=bool(A.get('signed_profile_incremental_diagnostic',False) and B.get('signed_profile_incremental_diagnostic',False))
    scalar_both=bool(A.get('transition_scalar_split_pass',False) and B.get('transition_scalar_split_pass',False))
    if both:
        nxt='PASS_DOUBLE_FRESH_ONLY_freeze_V64_3_22_transition_risk_dual_and_run_one_independent_full_val_reproduction_no_test_closed_loop'
    elif scalar_both:
        nxt='SIGNED_PROFILE_NOT_NEEDED_transition_scalar_reproduces_on_both_blocks_freeze_scalar_candidate_before_full_val_no_weight_tuning'
    elif not T.get('main_train_holdout_safe',False):
        nxt='STOP_TRAIN_regret_risk_path_not_safe_do_not_use_fresh_validation_audit_transition_representation'
    else:
        nxt='STOP_DOUBLE_FRESH_follow_split_specific_next_actions_no_threshold_budget_or_guard_tuning'
    r={'audit':'v64_3_22_eaf_icer_tcr_double_fresh','both_independent_500_scene_blocks_pass':both,'transition_scalar_both_blocks_pass':scalar_both,'signed_profile_incremental_on_both_blocks':signed,'full_promotion_to_independent_full_val_reproduction':both or scalar_both,'next_action':nxt,'split_A':A,'split_B':B,'train_fit_summary':T}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(r,indent=2,sort_keys=True)); print(json.dumps({'both_pass':both,'scalar_both':scalar_both,'signed_both':signed,'next_action':nxt},sort_keys=True))
if __name__=='__main__': main()
