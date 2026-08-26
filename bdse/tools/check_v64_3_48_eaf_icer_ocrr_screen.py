from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split-a',required=True);ap.add_argument('--split-b',required=True);ap.add_argument('--fit-report',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    A=json.load(open(a.split_a));B=json.load(open(a.split_b));F=json.load(open(a.fit_report));pa=bool(A.get('full_split_pass'));pb=bool(B.get('full_split_pass'));same=A.get('preferred_arm')==B.get('preferred_arm')==F.get('nested_crossfit',{}).get('preferred_promotion_arm');ok=bool(pa and pb and same and F.get('train_gate_pass'))
    rep={'audit':'v64_3_48_eaf_icer_ocrr_double_fresh','pass':ok,'split_A_pass':pa,'split_B_pass':pb,'preferred_arm':F.get('nested_crossfit',{}).get('preferred_promotion_arm'),'same_preregistered_arm_on_both_splits':same,'next_action':'freeze_V48_and_run_full_validation_plus_official_closed_loop' if ok else 'STOP_no_promotion_do_not_pool_A_B_or_tune'}
    Path(a.output).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');print(json.dumps(rep,indent=2,sort_keys=True))
if __name__=='__main__': main()
