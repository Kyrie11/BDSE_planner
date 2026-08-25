from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-a", required=True)
    ap.add_argument("--split-b", required=True)
    ap.add_argument("--fit-report", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    A = json.load(open(a.split_a))
    B = json.load(open(a.split_b))
    F = json.load(open(a.fit_report))
    eng = bool(A.get("engineering_valid") and B.get("engineering_valid"))
    train = bool(F.get("train_gate_pass"))
    both = bool(A.get("full_split_pass") and B.get("full_split_pass"))
    promoted = (F.get("nested_crossfit", {}) or {}).get("promoted_arm")
    rep = {
        "audit": "v64_3_43_eaf_icer_crvr_double_fresh_screen",
        "engineering_valid_both": eng,
        "train_nested_crvr_gate_pass": train,
        "promoted_train_arm": promoted,
        "double_fresh_promotion_pass": bool(train and both),
        "scientific_conclusion": (
            "CRVR_reproduces_on_double_fresh"
            if train and both
            else ("engineering_invalid_stop" if not eng else "CRVR_not_promoted_on_at_least_one_fresh_block")
        ),
        "next_action": (
            "freeze_CRVR_and_run_one_independent_full_validation_plus_official_nuPlan_closed_loop"
            if train and both
            else "STOP_do_not_pool_AB_or_sweep_parameters_use_preregistered_response_arm_and_teacher_oracle_diagnostics"
        ),
        "split_A": A,
        "split_B": B,
        "fit": F,
    }
    Path(a.output).write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
