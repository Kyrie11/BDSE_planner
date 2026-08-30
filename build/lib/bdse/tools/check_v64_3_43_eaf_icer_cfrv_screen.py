from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split-a',required=True); ap.add_argument('--split-b',required=True); ap.add_argument('--value-report',required=True); ap.add_argument('--fit-report',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    A=json.load(open(a.split_a));B=json.load(open(a.split_b));F=json.load(open(a.fit_report));C=json.load(open(a.value_report)); eng=bool(A.get('engineering_valid') and B.get('engineering_valid')); train=bool(F.get('train_gate_pass')); cal=int(C.get('selected_policy_proposal_count',0))>=64; both=bool(A.get('full_split_pass') and B.get('full_split_pass'))
    rep={'audit':'v64_3_43_eaf_icer_cfrv_double_fresh_screen','engineering_valid_both':eng,'train_nested_counterfactual_future_response_gate_pass':train,'independent_CAL500_translation_fit_valid':cal,'double_fresh_promotion_pass':both,'scientific_conclusion':'CFRV_reproduces_on_double_fresh' if both else ('engineering_invalid_stop' if not eng else 'CFRV_not_promoted_at_least_one_mechanism_or_fresh_block_gate_fails'),'next_action':'freeze_CFRV_and_run_one_independent_full_validation_plus_official_nuPlan_closed_loop' if both else 'STOP_do_not_pool_AB_or_sweep_parameters_use_QUALITY_vs_FUTURE_MEAN_vs_FUTURE_ROBUST_failure_slice_for_next_behavior_observable','split_A':A,'split_B':B,'fit':F,'calibration':C}
    Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n'); print(json.dumps(rep,indent=2,sort_keys=True))
if __name__=='__main__':main()
