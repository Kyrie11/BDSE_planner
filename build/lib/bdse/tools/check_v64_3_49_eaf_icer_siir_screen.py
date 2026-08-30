from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="V64.3.49 SIIR double-fresh screen")
    ap.add_argument("--split-a", required=True)
    ap.add_argument("--split-b", required=True)
    ap.add_argument("--fit-report", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    A = json.load(open(a.split_a)); B = json.load(open(a.split_b)); F = json.load(open(a.fit_report))
    if F.get("train_gate_pass") is not True or F.get("nested_crossfit", {}).get("preferred_promotion_arm") != "siir":
        raise SystemExit("STOP V49: fresh screen requires preregistered SIIR TRAIN pass")
    ok = bool(A.get("full_split_pass", False) and B.get("full_split_pass", False))
    rep = {
        "audit": "v64_3_49_eaf_icer_siir_double_fresh",
        "preferred_arm": "siir",
        "split_A_pass": bool(A.get("full_split_pass", False)),
        "split_B_pass": bool(B.get("full_split_pass", False)),
        "pass": ok,
        "next_action": (
            "FREEZE_SIIR_and_run_full_validation_then_official_closed_loop_interventional_evidence"
            if ok else
            "STOP_close_current_offline_selected_risk_family_do_not_pool_AB_or_tune_intervention_loss_threshold_features"
        ),
    }
    Path(a.output).write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit("V64.3.49 SIIR untouched A/B failed; scientific STOP")


if __name__ == "__main__":
    main()
