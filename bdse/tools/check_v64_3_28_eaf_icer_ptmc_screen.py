from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--split-a-report', required=True); ap.add_argument('--split-b-report', required=True)
    ap.add_argument('--train-fit-report', required=True); ap.add_argument('--output', required=True)
    args=ap.parse_args()
    A=json.load(open(args.split_a_report, encoding='utf-8')); B=json.load(open(args.split_b_report, encoding='utf-8'))
    T=json.load(open(args.train_fit_report, encoding='utf-8'))
    train=bool(T.get('train_gate_pass'))
    both=bool(train and A.get('full_split_pass') and B.get('full_split_pass'))
    semantic=bool(A.get('tail_mode_confirmation_tail_incremental') and B.get('tail_mode_confirmation_tail_incremental'))
    if both:
        next_action='freeze_V64_3_28_PTMC_and_run_one_independent_full_val_reproduction_only'
    elif not train:
        next_action='TRAIN_tail_mode_confirmation_tail_contract_failed_do_not_run_or_tune_fresh'
    elif not A.get('full_split_pass'):
        next_action='split_A_failed_follow_split_A_next_action_no_pooled_rescue'
    else:
        next_action='split_B_failed_follow_split_B_next_action_no_pooled_rescue'
    report={
        'audit':'v64_3_28_eaf_icer_ptmc_double_fresh_screen','train_gate_pass':train,
        'split_A_pass':bool(A.get('full_split_pass')),'split_B_pass':bool(B.get('full_split_pass')),
        'both_independent_blocks_pass':both,'tail_mode_confirmation_tail_incremental_both':semantic,
        'full_promotion_to_independent_full_val_reproduction':both,'test_closed_loop_allowed':False,
        'next_action':next_action,'split_A_next_action':A.get('next_action'),'split_B_next_action':B.get('next_action'),
    }
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))
if __name__=='__main__': main()
