from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare V64.3.12 RET vs CET exact-transmission screens")
    ap.add_argument("--ret", required=True)
    ap.add_argument("--cet", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    ret, cet = _load(args.ret), _load(args.cet)

    if not ret.get("instrumentation_valid", False) or not cet.get("instrumentation_valid", False):
        decision = "repair_instrumentation"
        reason = "At least one arm lacks valid exact-training/exact-validation instrumentation. Do not interpret endpoint differences."
    elif cet.get("full_promotion", False):
        decision = "promote_cet_full"
        reason = "CET improves exact C2-B and endpoint under non-harm; controlled budget exchange is supported by the causal screen."
    elif ret.get("full_promotion", False):
        decision = "promote_ret_full"
        reason = "Exact-runtime training alignment is sufficient; the extra controlled-exchange mechanism is unnecessary or less stable on screen."
    elif cet.get("pivot_to_value_frontier", False):
        decision = "pivot_to_value_frontier"
        reason = "After exact-runtime training alignment plus controlled B-set exchange, acquisition still does not yield an acceptable C2-B/C3 path. Stop acquisition variants."
    else:
        decision = "inspect_exact_pair_density"
        reason = "The two-arm result is not promotable and does not meet the terminal CET stop rule; inspect exact pair density/instrumentation before any algorithm claim."

    def compact(r: dict) -> dict:
        return {
            "selected_epoch": r.get("selected_epoch"),
            "anchor_budget_oracle_gap": r.get("anchor_budget_oracle_gap"),
            "budget_oracle_gap_closure": r.get("budget_oracle_gap_closure"),
            "mechanism_nonharm": r.get("mechanism_nonharm"),
            "mechanism_gain": r.get("mechanism_gain"),
            "deployment_gain": r.get("deployment_gain"),
            "full_promotion": r.get("full_promotion"),
            "pivot_to_value_frontier": r.get("pivot_to_value_frontier"),
            "exact_acquisition_exhausted": r.get("exact_acquisition_exhausted"),
            "selected": r.get("selected"),
            "deltas": r.get("deltas"),
            "diagnosis": r.get("diagnosis"),
        }

    report = {
        "audit": "v64_3_12_ret_cet_screen_comparison",
        "decision": decision,
        "reason": reason,
        "ret": compact(ret),
        "cet": compact(cet),
        "causal_readout": {
            "ret_minus_v64_3_11": "isolates fast-surrogate -> sampled exact-runtime B training-target alignment",
            "cet_minus_ret": "isolates controlled displacement of exact-current-B evidence that exact-oracle-B removes",
            "terminal_rule": "If CET cannot improve exact C2-B under non-harm, or improves C2-B without C3, pivot to decisive value/frontier.",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "reason": reason}, indent=2))
    return 0 if decision != "repair_instrumentation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
