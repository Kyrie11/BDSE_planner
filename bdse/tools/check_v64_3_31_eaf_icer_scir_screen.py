from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split-a',required=True); ap.add_argument('--split-b',required=True); ap.add_argument('--calibration-report',required=True); ap.add_argument('--fit-report',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    A=json.load(open(a.split_a,encoding='utf-8')); B=json.load(open(a.split_b,encoding='utf-8')); C=json.load(open(a.calibration_report,encoding='utf-8')); F=json.load(open(a.fit_report,encoding='utf-8'))
    engineering=bool(A.get('engineering_valid') and B.get('engineering_valid'))
    both=bool(A.get('full_split_pass') and B.get('full_split_pass') and F.get('train_gate_pass') and C.get('selected_proposal_count',0)>=64)
    if not engineering: conclusion='engineering_invalid_stop'
    elif both: conclusion='SCIR_selection_conditioned_intervention_semantics_and_selected_path_conformal_certificate_reproduce_on_double_fresh'
    else: conclusion='SCIR_not_promoted_current_frozen_mechanism_fails_at_least_one_independent_block'
    nxt='freeze_and_run_one_independent_full_validation_reproduction_no_more_tuning' if both else 'STOP_do_not_pool_AB_or_sweep_alpha_thresholds_use_failed_gate_to_choose_next_mechanism'
    r={'audit':'v64_3_31_eaf_icer_scir_double_fresh_screen','engineering_valid_both':engineering,'double_fresh_promotion_pass':both,'scientific_conclusion':conclusion,'next_action':nxt,'train_fit':F,'independent_calibration':C,'split_A':A,'split_B':B,'protocol':'CAL500 is calibration/design data and is excluded from A/B; A and B are independent 500-scene promotion blocks judged separately; no pooled rescue.'}
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n'); print(json.dumps({'pass':both,'engineering':engineering,'conclusion':conclusion,'next_action':nxt},sort_keys=True))
if __name__=='__main__': main()
