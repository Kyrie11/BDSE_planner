from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine two untouched V64.3.31 OMCER blocks without pooled rescue.")
    ap.add_argument("--split-a-report", required=True)
    ap.add_argument("--split-b-report", required=True)
    ap.add_argument("--train-fit-report", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    A = json.load(open(args.split_a_report, encoding="utf-8"))
    B = json.load(open(args.split_b_report, encoding="utf-8"))
    tr = json.load(open(args.train_fit_report, encoding="utf-8"))
    train = bool(tr.get("train_gate_pass"))
    both = bool(A.get("full_split_pass") and B.get("full_split_pass"))
    endpoint_signal = bool(A.get("endpoint_strict_signal_over_V25_DRC") or B.get("endpoint_strict_signal_over_V25_DRC"))
    promote = bool(train and both and endpoint_signal)

    if promote:
        nxt = "freeze_V31_OMCER_and_run_one_independent_full_val_reproduction_only_no_test_or_closed_loop"
    elif not train:
        nxt = "TRAIN_gate_invalid_stop_before_spending_or_interpreting_fresh"
    elif not both:
        nxt = "OMCER_not_independently_reproduced_stop_no_K_cat_threshold_multiplier_operator_margin_or_B_tuning"
    else:
        nxt = "mechanism_reproduced_without_endpoint_signal_stop_before_full_val"

    r = {
        "audit": "v64_3_31_eaf_icer_omcer_double_fresh_screen",
        "train_gate_pass": train,
        "train_catastrophe_free": bool(tr.get("crossfit", {}).get("omcer_main", {}).get("all_folds_catastrophe_free", False)),
        "split_A_pass": bool(A.get("full_split_pass")),
        "split_B_pass": bool(B.get("full_split_pass")),
        "both_independent_blocks_pass": both,
        "safe_coverage_gain_vs_V25_both": bool(A.get("safe_recovery_coverage_gain_over_V25_DRC") and B.get("safe_recovery_coverage_gain_over_V25_DRC")),
        "safe_coverage_gain_vs_lock_both": bool(A.get("safe_recovery_coverage_gain_over_proposal_lock_control") and B.get("safe_recovery_coverage_gain_over_proposal_lock_control")),
        "catastrophe_free_both": bool(A.get("selected_replacement_catastrophe_free") and B.get("selected_replacement_catastrophe_free")),
        "tail_noninferior_to_V25_both": bool(A.get("selected_tail_noninferior_to_V25_DRC") and B.get("selected_tail_noninferior_to_V25_DRC")),
        "tail_noninferior_to_lock_both": bool(A.get("selected_tail_noninferior_to_proposal_lock_control") and B.get("selected_tail_noninferior_to_proposal_lock_control")),
        "endpoint_noninferior_both": bool(A.get("endpoint_noninferior_to_V25_DRC") and B.get("endpoint_noninferior_to_V25_DRC")),
        "endpoint_strict_signal_at_least_one_block": endpoint_signal,
        "full_promotion_to_independent_full_val_reproduction": promote,
        "test_or_closed_loop_allowed": False,
        "next_action": nxt,
        "split_A_next_action": A.get("next_action"),
        "split_B_next_action": B.get("next_action"),
        "interpretation": (
            "V31 tests operator-aligned catastrophic-tail admissibility at fixed B=16. The V25 arm controls the historical evidence-only downside certificate; "
            "the V30 lock-only arm controls post-extremal same-proposal confirmation. Promotion requires the OMCER pre-extremal risk formulation to improve safe "
            "coverage over both controls independently on A and B while remaining catastrophe-free, tail-noninferior, and endpoint-noninferior."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(r, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
