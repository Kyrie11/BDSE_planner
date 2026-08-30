from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split-a-report',required=True);ap.add_argument('--split-b-report',required=True);ap.add_argument('--train-fit-report',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();A=json.load(open(a.split_a_report));B=json.load(open(a.split_b_report));T=json.load(open(a.train_fit_report)); train=bool(T.get('train_gate_pass')); both=bool(train and A.get('full_split_pass') and B.get('full_split_pass')); attr=bool(A.get('attribution_resolved_incremental') and B.get('attribution_resolved_incremental')); downside=bool(A.get('downside_certificate_incremental') and B.get('downside_certificate_incremental'))
 if both:nxt='freeze_V64_3_24_ARC_and_run_one_independent_full_val_reproduction_only'
 elif not train:nxt='TRAIN_attribution_downside_path_contract_failed_do_not_run_or_tune_fresh'
 elif not A.get('full_split_pass'):nxt='split_A_failed_follow_split_A_next_action_no_pooled_rescue'
 else:nxt='split_B_failed_follow_split_B_next_action_no_pooled_rescue'
 r={'audit':'v64_3_24_eaf_icer_arc_double_fresh_screen','train_gate_pass':train,'split_A_pass':bool(A.get('full_split_pass')),'split_B_pass':bool(B.get('full_split_pass')),'both_independent_blocks_pass':both,'downside_certificate_incremental_both':downside,'attribution_resolved_incremental_both':attr,'full_promotion_to_independent_full_val_reproduction':both,'test_closed_loop_allowed':False,'next_action':nxt,'split_A_next_action':A.get('next_action'),'split_B_next_action':B.get('next_action')};Path(a.output).write_text(json.dumps(r,indent=2,sort_keys=True));print(json.dumps(r,indent=2,sort_keys=True))
if __name__=='__main__':main()
