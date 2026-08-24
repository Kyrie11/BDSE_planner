from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="V64.3.32.1 SSIR double-fresh promotion screen.")
    ap.add_argument("--split-a", required=True)
    ap.add_argument("--split-b", required=True)
    ap.add_argument("--calibration-report", required=True)
    ap.add_argument("--fit-report", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    A = json.load(open(a.split_a, encoding="utf-8"))
    B = json.load(open(a.split_b, encoding="utf-8"))
    C = json.load(open(a.calibration_report, encoding="utf-8"))
    F = json.load(open(a.fit_report, encoding="utf-8"))
    engineering = bool(A.get("engineering_valid") and B.get("engineering_valid"))
    train = bool(F.get("train_gate_pass") and F.get("nested_crossfit", {}).get("fold_pass_count") == 5)
    cal = bool(C.get("calibration_total_scene_count") == 500 and int(C.get("direct_eligible_scene_count", 0)) >= 64 and abs(float(C.get("alpha", 0.0)) - 0.05) <= 1e-12)
    both = bool(engineering and train and cal and A.get("full_split_pass") and B.get("full_split_pass"))
    if not engineering:
        conclusion = "engineering_invalid_stop"
    elif both:
        conclusion = "SSIR_scene_simultaneous_candidate_specific_lower_bound_ordering_reproduces_selection_stable_direct_recovery_on_double_fresh"
    else:
        conclusion = "SSIR_not_promoted_current_frozen_selection_stable_mechanism_fails_at_least_one_independent_gate_or_block"
    nxt = "freeze_and_run_one_independent_full_validation_reproduction_no_more_tuning" if both else "STOP_do_not_pool_AB_or_sweep_alpha_scale_ridge_thresholds_use_failed_gate_to_choose_next_mechanism"
    r = {
        "audit": "v64_3_32_1_eaf_icer_ssir_double_fresh_screen",
        "engineering_valid_both": engineering,
        "train_nested_full_mechanism_gate_pass": train,
        "independent_calibration_valid": cal,
        "double_fresh_promotion_pass": both,
        "scientific_conclusion": conclusion,
        "next_action": nxt,
        "train_fit": F,
        "independent_calibration": C,
        "split_A": A,
        "split_B": B,
        "protocol": "Frozen TRAIN nested 3-fit/1-cal/1-test gate evaluates the complete SSIR bound-selection mechanism; only then is an independent CAL500 used once to calibrate the frozen direct-domain scene-simultaneous score; A500/B500 are untouched promotion blocks judged separately with no pooled rescue.",
    }
    p = Path(a.output); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": both, "engineering": engineering, "train": train, "calibration": cal, "conclusion": conclusion, "next_action": nxt}, sort_keys=True))


if __name__ == "__main__":
    main()
