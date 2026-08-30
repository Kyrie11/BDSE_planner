from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _f, _metric_pack
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _load_rows
from bdse.tools.check_v64_3_30_eaf_icer_fbic_split import _fbic_diag, _query_diag


def _write(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _switch_diag(b16_rows: dict[str, dict[str, Any]], b24_rows: dict[str, dict[str, Any]], safe: set[str]) -> dict[str, Any]:
    gains: list[float] = []
    beneficial = harmful = equal = 0
    for t in sorted(safe):
        a = int(round(_f(b16_rows[t], "bdse_action", -999.0)))
        b = int(round(_f(b24_rows[t], "bdse_action", -999.0)))
        if a == b:
            continue
        gain = _f(b16_rows[t], "teacher_regret") - _f(b24_rows[t], "teacher_regret")
        gains.append(gain)
        if gain > 1e-12:
            beneficial += 1
        elif gain < -1e-12:
            harmful += 1
        else:
            equal += 1
    return {
        "safe_action_change_count": len(gains),
        "B24_beneficial_change_count": beneficial,
        "B24_harmful_change_count": harmful,
        "B24_equal_change_count": equal,
        "B16_minus_B24_teacher_regret_sum_on_changed_actions": float(sum(gains)),
        "B16_minus_B24_teacher_regret_mean_on_changed_actions": float(np.mean(gains)) if gains else float("nan"),
        "B24_worst_regret_increase_on_changed_actions": float(-min(gains)) if gains else float("nan"),
        "B24_best_regret_reduction_on_changed_actions": float(max(gains)) if gains else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Lightweight audit of repaired V30.1 paired TRAIN result; consumes fitted reports instead of rescanning ~1.6 GB edge logs.")
    ap.add_argument("--b16-metrics", required=True)
    ap.add_argument("--b16-rows", required=True)
    ap.add_argument("--b24-metrics", required=True)
    ap.add_argument("--b24-rows", required=True)
    ap.add_argument("--b16-fit-report", required=True)
    ap.add_argument("--b24-fit-report", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    b16_rows = _load_rows(a.b16_rows)
    b24_rows = _load_rows(a.b24_rows)
    if len(b16_rows) != 3000 or len(b24_rows) != 3000 or set(b16_rows) != set(b24_rows):
        raise SystemExit("STOP DATA: repaired V30.1 TRAIN arms must contain the exact same 3000 scenes")
    all_flagged = {t for t in b16_rows if _f(b16_rows[t], "all_actions_safety_flagged_rate", 0.0) >= 0.5}
    safe = set(b16_rows) - all_flagged
    fbic = _fbic_diag(b24_rows, safe, all_flagged)
    query = _query_diag(b16_rows, b24_rows, set(b16_rows))
    M16_raw = json.load(open(a.b16_metrics, encoding="utf-8"))
    M24_raw = json.load(open(a.b24_metrics, encoding="utf-8"))
    M16 = _metric_pack(M16_raw)
    M24 = _metric_pack(M24_raw)
    F16 = json.load(open(a.b16_fit_report, encoding="utf-8"))
    F24 = json.load(open(a.b24_fit_report, encoding="utf-8"))
    switch = _switch_diag(b16_rows, b24_rows, safe)

    contract = bool(
        fbic["enabled_rate"] == 1.0
        and fbic["safe_applied_rate"] >= 0.90
        and (not all_flagged or fbic["structural_applied_rate"] == 0.0)
        and fbic["safe_final_count_mean"] >= fbic["safe_baseline_count_mean"] + 4.0
        and abs(fbic["safe_removed_atom_count_mean"]) <= 1e-12
        and abs(fbic["upstream_configured_budget_mean"] - 16.0) <= 1e-12
        and abs(fbic["retained_interface_configured_budget_mean"] - 24.0) <= 1e-12
        and fbic["retained_interface_budget_pass_rate"] == 1.0
        and fbic["no_new_query_rate"] == 1.0
        and query["all_query_counts_exact_scene_parity"]
    )
    cf16 = F16.get("crossfit", {}).get("aggregate_downside", {})
    cf24 = F24.get("crossfit", {}).get("aggregate_downside", {})
    baseline_reproduced = bool(
        F16.get("train_gate_pass")
        and int(F16.get("replacement_edges", -1)) == 1455
        and int(F16.get("replacement_scenes", -1)) == 310
        and int(cf16.get("fold_pass_count", -1)) == 5
        and int(cf16.get("selected_count", -1)) == 71
        and abs(float(cf16.get("teacher_improvement_sum", float("nan"))) - 5.527642325753739) <= 1e-9
    )
    b24_drc_fail_is_algorithmic = bool(
        (not F24.get("train_gate_pass", False))
        and int(cf24.get("fold_pass_count", -1)) < 5
        and int(cf24.get("selected_count", 0)) >= 64
        and float(cf24.get("teacher_improvement_sum", -1.0)) >= 0.0
    )

    out = {
        "audit": "v64_3_30_1_repaired_fbic_train_result",
        "engineering_contract_valid": contract,
        "historical_B16_V25_reproduced": baseline_reproduced,
        "B24_DRC_train_gate_pass": bool(F24.get("train_gate_pass", False)),
        "B24_DRC_fail_is_selected_path_fold_safety_failure_not_runtime_error": b24_drc_fail_is_algorithmic,
        "fresh_validation_used": False,
        "fresh_capacity_question_resolved": False,
        "scientific_scope": "development-only; untouched A/B pure-capacity evaluation remains required",
        "safe_scene_count": len(safe),
        "all_flagged_scene_count": len(all_flagged),
        "fbic_diagnostics": fbic,
        "query_accounting": query,
        "B16_metrics": M16,
        "B24_metrics": M24,
        "selected_decisive_atom_recall": {"B16": M16_raw.get("selected_decisive_atom_recall"), "B24": M24_raw.get("selected_decisive_atom_recall")},
        "evidence_certificate_fraction": {"B16": M16_raw.get("evidence_certificate_fraction"), "B24": M24_raw.get("evidence_certificate_fraction")},
        "capacity_action_switch": switch,
        "B16_DRC_fit_report": F16,
        "B24_DRC_fit_report": F24,
        "development_interpretation": (
            "FBIC materially increases evidence transmission/coverage, but on frozen TRAIN the pure downstream endpoint worsens and the unchanged DRC loses fold safety. "
            "This is directional evidence for downstream operator/reliability mismatch, not independent evidence that B16 capacity is irrelevant."
        ),
        "next_experiment": "untouched double-fresh raw/B16-V20/B24-V20 pure-capacity screen; B24 DRC remains fail-closed",
    }
    _write(Path(a.output), out)
    print(json.dumps({
        "engineering_contract_valid": contract,
        "historical_B16_V25_reproduced": baseline_reproduced,
        "B24_DRC_train_gate_pass": bool(F24.get("train_gate_pass", False)),
        "B16_match": M16.get("match"), "B24_match": M24.get("match"),
        "B16_regret": M16.get("regret"), "B24_regret": M24.get("regret"),
        "B24_changed_action_regret_gain_sum": switch["B16_minus_B24_teacher_regret_sum_on_changed_actions"],
    }, indent=2))
    if not contract or not baseline_reproduced or not b24_drc_fail_is_algorithmic:
        raise SystemExit("STOP V30.2 TRAIN AUDIT: repaired V30.1 branch identity/contract did not reproduce")


if __name__ == "__main__":
    main()
