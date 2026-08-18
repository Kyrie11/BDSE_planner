from __future__ import annotations
import argparse,json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description="Aggregate two independent V64.3.21 MCR fresh replication blocks.")
    ap.add_argument("--split-a-report",required=True); ap.add_argument("--split-b-report",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    A=json.load(open(a.split_a_report)); B=json.load(open(a.split_b_report))
    both=bool(A.get("full_split_pass",False) and B.get("full_split_pass",False))
    # Profile-retention attribution is a diagnostic rather than a promotion shortcut:
    # if it fails, the main mechanism can still be real, but paper claims about signed
    # attribution in the retention head must be demoted until an independent ablation supports it.
    profile_both=bool(A.get("profile_retention_incremental_diagnostic",False) and B.get("profile_retention_incremental_diagnostic",False))
    if both:
        next_action="independent_frozen_full_val_reproduction_of_V64_3_21_MCR_consensus_only_no_test_or_closed_loop_until_reproduced"
    else:
        fails=[r.get("next_action") for r in [A,B] if not r.get("full_split_pass",False)]
        next_action="STOP_DOUBLE_FRESH_REPLICATION:"+"|".join(str(x) for x in fails)
    r={"audit":"v64_3_21_eaf_icer_mcr_double_fresh_screen","full_promotion_to_independent_full_val_reproduction":both,"both_independent_500_scene_blocks_pass":both,"signed_profile_retention_incremental_on_both_blocks":profile_both,"split_a":A,"split_b":B,"next_action":next_action,"interpretation":"V64.3.21 is allowed to leave screen only if the complete mechanism and endpoint pass independently on two disjoint label-free 500-scene validation blocks. Pooled success cannot rescue a failed block. Passing this screen permits only one frozen independent full-val reproduction; test/closed-loop remain forbidden."}
    p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2,sort_keys=True),encoding="utf-8");print(json.dumps(r,indent=2,sort_keys=True))

if __name__=="__main__":main()
