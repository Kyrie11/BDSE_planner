from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean(values: list[float]) -> float:
    arr = np.asarray([x for x in values if math.isfinite(x)], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit whether the AOCC certificate tracks the exact deployed decision.

    This report is deliberately diagnostic.  It never upgrades an uncertified
    scene to certified and it never changes the gate.  ``budget_vs_pair_full``
    is the exact equality between the fixed-B deployed action and the same
    downstream decision operator run with all proposed Top-M evidence atoms.
    If that decision-aligned quantity and the AOCC pairwise certificate strongly
    disagree, the next algorithmic change should target the certificate
    formulation rather than silently relaxing B or a threshold.
    """
    total = len(rows)
    cert = []
    preserved = []
    teacher = []
    critical = []
    initial_deficit: list[float] = []
    reduction: list[float] = []
    final_deficit: list[float] = []
    full_target_pair_cert: list[float] = []

    for row in rows:
        c = _finite_float(row.get("evidence_certificate_fraction", row.get("aocc_fully_certified_scene_rate")))
        # evidence_certificate_fraction is a scene-level 0/1 metric in the
        # per-sample JSONL.  Use the strict fully-certified interpretation.
        cert.append(bool(math.isfinite(c) and c >= 1.0 - 1.0e-8))
        p = _finite_float(row.get("budget_vs_pair_full_match"))
        preserved.append(bool(math.isfinite(p) and p >= 0.5))
        t = _finite_float(row.get("teacher_action_match"))
        teacher.append(t)
        cr = _finite_float(row.get("teacher_exact_winner_flip_critical_scene_rate"))
        critical.append(bool(math.isfinite(cr) and cr >= 0.5))
        initial_deficit.append(_finite_float(row.get("selector_aocc_initial_deficit")))
        reduction.append(_finite_float(row.get("selector_aocc_deficit_reduction")))
        final_deficit.append(_finite_float(row.get("selector_aocc_final_deficit")))
        full_target_pair_cert.append(_finite_float(row.get("selector_aocc_full_target_certified_pair_fraction")))

    n_cert = sum(cert)
    n_pres = sum(preserved)
    both = sum(c and p for c, p in zip(cert, preserved))
    cert_not_pres = sum(c and not p for c, p in zip(cert, preserved))
    uncert_pres = sum((not c) and p for c, p in zip(cert, preserved))
    neither = total - both - cert_not_pres - uncert_pres

    def conditional_mean(mask: list[bool], values: list[float]) -> float:
        return _mean([v for m, v in zip(mask, values) if m])

    critical_pres = sum(cr and p for cr, p in zip(critical, preserved))
    critical_n = sum(critical)
    critical_uncert_pres = sum(cr and (not c) and p for cr, c, p in zip(critical, cert, preserved))

    cert_rate = _rate(n_cert, total)
    preservation_rate = _rate(n_pres, total)
    report = {
        "audit": "decision_aligned_certificate_action_alignment",
        "diagnostic_only": True,
        "num_scenarios": total,
        "evidence_fully_certified_rate": cert_rate,
        "exact_budget_vs_pair_full_winner_preservation_rate": preservation_rate,
        "certificate_action_preservation_gap": (
            float(preservation_rate - cert_rate)
            if math.isfinite(cert_rate) and math.isfinite(preservation_rate)
            else float("nan")
        ),
        "quadrants": {
            "certified_and_preserved": both,
            "certified_but_not_preserved": cert_not_pres,
            "uncertified_but_preserved": uncert_pres,
            "uncertified_and_not_preserved": neither,
        },
        "preservation_given_certified": _rate(both, n_cert),
        "preservation_given_uncertified": _rate(uncert_pres, total - n_cert),
        "teacher_match_given_certified": conditional_mean(cert, teacher),
        "teacher_match_given_uncertified": conditional_mean([not x for x in cert], teacher),
        "teacher_match_given_preserved": conditional_mean(preserved, teacher),
        "teacher_match_given_not_preserved": conditional_mean([not x for x in preserved], teacher),
        "critical_scene_rate": _rate(critical_n, total),
        "budget_winner_preservation_on_critical_scenes": _rate(critical_pres, critical_n),
        "critical_uncertified_but_preserved_rate_over_critical": _rate(critical_uncert_pres, critical_n),
        "aocc_initial_deficit_mean": _mean(initial_deficit),
        "aocc_deficit_reduction_mean": _mean(reduction),
        "aocc_final_deficit_mean": _mean(final_deficit),
        "aocc_full_target_certified_pair_fraction_mean": _mean(full_target_pair_cert),
    }
    # A certified scene should normally preserve the exact reference winner if
    # the certificate is meant to certify that decision.  Do not mutate gate
    # semantics here; flag the mismatch for algorithm review.
    report["certificate_soundness_warning"] = bool(
        n_cert > 0 and _rate(both, n_cert) < 0.99
    )
    report["certificate_conservatism_warning"] = bool(
        math.isfinite(report["certificate_action_preservation_gap"])
        and report["certificate_action_preservation_gap"] > 0.25
    )
    report["recommended_interpretation"] = (
        "Audit certificate formulation against the exact downstream winner before changing B or gate thresholds."
        if report["certificate_soundness_warning"] or report["certificate_conservatism_warning"]
        else "Certificate and exact winner preservation are sufficiently aligned for this diagnostic."
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit AOCC certificate alignment with exact fixed-B winner preservation.")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    with args.jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    if not rows:
        raise RuntimeError(f"no rows found in {args.jsonl}")
    report = analyze_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
