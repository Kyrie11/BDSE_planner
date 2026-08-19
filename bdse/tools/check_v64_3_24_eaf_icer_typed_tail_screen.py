from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap=argparse.ArgumentParser(description="V64.3.24 two-block typed-tail causal screen checker.")
    ap.add_argument("--split-c-report",required=True); ap.add_argument("--split-d-report",required=True)
    ap.add_argument("--train-fit-report",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    C=json.load(open(a.split_c_report)); D=json.load(open(a.split_d_report)); T=json.load(open(a.train_fit_report))
    if C.get("split_name")!="C" or D.get("split_name")!="D": raise SystemExit("STOP DATA: split identity mismatch")
    train=bool(T.get("train_gate_pass",False)); both=bool(train and C.get("full_split_pass",False) and D.get("full_split_pass",False))
    rep_both=bool(C.get("typed_representation_incremental",False) and D.get("typed_representation_incremental",False))
    tail_both=bool(C.get("tail_objective_incremental",False) and D.get("tail_objective_incremental",False))
    rank_both=bool(C.get("risk_first_extremal_alignment_incremental",False) and D.get("risk_first_extremal_alignment_incremental",False))
    tail_safe_both=bool(C.get("selected_path_material_tail_safe",False) and D.get("selected_path_material_tail_safe",False))
    if both:
        next_action="double_fresh_pass_freeze_V24_TTCR_then_run_one_independent_full_val_reproduction_only_test_closed_loop_still_forbidden"
    elif not train:
        next_action="TRAIN_typed_tail_contract_failed_do_not_spend_fresh_validation_GPU"
    elif not C.get("full_split_pass",False):
        next_action="split_C_failed_follow_split_C_next_action_no_pooled_rescue"
    else:
        next_action="split_D_failed_follow_split_D_next_action_no_pooled_rescue"
    rep={
        "audit":"v64_3_24_eaf_icer_typed_tail_double_fresh_screen", "train_gate_pass":train,
        "split_C_pass":bool(C.get("full_split_pass",False)), "split_D_pass":bool(D.get("full_split_pass",False)),
        "both_independent_500_scene_blocks_pass":both, "selected_material_tail_safe_both":tail_safe_both,
        "typed_representation_incremental_both":rep_both, "tail_objective_incremental_both":tail_both,
        "risk_first_extremal_alignment_incremental_both":rank_both,
        "full_promotion_to_independent_full_val_reproduction":both, "test_closed_loop_allowed":False,
        "next_action":next_action, "split_C_next_action":C.get("next_action"), "split_D_next_action":D.get("next_action"),
    }
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rep,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(rep,indent=2,sort_keys=True))

if __name__=="__main__": main()
