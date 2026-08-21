from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bdse.tools.check_v64_3_19_eaf_icer_screen import _icer_edge_diag, _metric_pack, _f
from bdse.tools.check_v64_3_21_eaf_icer_mcr_split import _load_rows, _path_diag, _guard_block_rate, _identity_rate


def _edge_groups(path: str, allowed: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            token = str(row.get("scenario_token", ""))
            if allowed is not None and token not in allowed:
                continue
            groups.setdefault(token, []).append(row)
    return groups


def _replacement_tail_diag(
    raw_rows: dict[str, dict[str, Any]],
    alg_rows: dict[str, dict[str, Any]],
    edge_path: str,
    safe: set[str],
) -> dict[str, Any]:
    """Direct incumbent->alternative outcome tail, reported in both endpoint and frontier units."""
    groups = _edge_groups(edge_path, safe)
    regret_delta: list[float] = []
    teacher_improvement: list[float] = []
    tokens: list[str] = []
    for token in sorted(safe):
        rows = groups.get(token, [])
        if not rows:
            continue
        anchor = int(rows[0].get("anchor_action", -1))
        incumbent = int(rows[0].get("raw_top_action", -1))
        selected = int(rows[0].get("icer_selected_action", incumbent))
        inc = next((r for r in rows if int(r.get("challenger_action", -999)) == incumbent), None)
        if inc is None or _f(inc, "icer_admissible", _f(inc, "dacer_admissible", 0.0)) < 0.5:
            continue
        if selected in {anchor, incumbent}:
            continue
        sel = next((r for r in rows if int(r.get("challenger_action", -999)) == selected), None)
        if sel is None:
            continue
        inc_tm = _f(inc, "teacher_margin", float("nan"))
        sel_tm = _f(sel, "teacher_margin", float("nan"))
        if not (math.isfinite(inc_tm) and math.isfinite(sel_tm)):
            continue
        if token not in raw_rows or token not in alg_rows:
            continue
        rd = _f(alg_rows[token], "teacher_regret", float("nan")) - _f(raw_rows[token], "teacher_regret", float("nan"))
        if not math.isfinite(rd):
            continue
        regret_delta.append(float(rd))
        teacher_improvement.append(float(sel_tm - inc_tm))
        tokens.append(token)

    r = np.asarray(regret_delta, dtype=np.float64)
    y = np.asarray(teacher_improvement, dtype=np.float64)
    harmful = np.maximum(r, 0.0)
    downside = np.minimum(y, 0.0)
    return {
        "count": int(len(r)),
        "regret_delta_sum": float(r.sum()) if len(r) else 0.0,
        "regret_delta_mean": float(r.mean()) if len(r) else float("nan"),
        "regret_positive_rms": float(np.sqrt(np.mean(harmful * harmful))) if len(r) else float("nan"),
        "worst_regret_increase": float(r.max()) if len(r) else float("nan"),
        "teacher_improvement_sum": float(y.sum()) if len(y) else 0.0,
        "teacher_improvement_mean": float(y.mean()) if len(y) else float("nan"),
        "teacher_positive_precision": float(np.mean(y > 0.0)) if len(y) else float("nan"),
        "teacher_improvement_worst": float(y.min()) if len(y) else float("nan"),
        "teacher_negative_rms": float(np.sqrt(np.mean(downside * downside))) if len(y) else float("nan"),
        "token_count": int(len(set(tokens))),
    }


def _strictly_better_or_zero(main: float, ctrl: float, eps: float = 1e-9) -> bool:
    if not (math.isfinite(main) and math.isfinite(ctrl)):
        return False
    return bool(main < ctrl - eps)


def _selected_replacement_map(edge_path: str, safe: set[str]) -> dict[str, tuple[int, int, int]]:
    groups = _edge_groups(edge_path, safe)
    out: dict[str, tuple[int, int, int]] = {}
    for token, rows in groups.items():
        if not rows:
            continue
        anchor = int(rows[0].get("anchor_action", -1))
        incumbent = int(rows[0].get("raw_top_action", -1))
        selected = int(rows[0].get("icer_selected_action", incumbent))
        out[token] = (anchor, incumbent, selected)
    return out


def _no_fallback_subset_contract(main_edge: str, ctrl_edge: str, safe: set[str]) -> dict[str, Any]:
    m = _selected_replacement_map(main_edge, safe)
    c = _selected_replacement_map(ctrl_edge, safe)
    violations: list[dict[str, Any]] = []
    main_replacements = 0
    for token in sorted(safe):
        if token not in m or token not in c:
            continue
        ma, mi, ms = m[token]
        ca, ci, cs = c[token]
        if (ma, mi) != (ca, ci):
            violations.append({"token": token, "reason": "anchor_or_incumbent_identity_mismatch"})
            continue
        if ms not in {ma, mi}:
            main_replacements += 1
            if cs != ms:
                violations.append({"token": token, "main_selected": ms, "aggregate_selected": cs, "incumbent": mi, "anchor": ma})
    return {
        "pass": not violations,
        "main_replacement_count": int(main_replacements),
        "violation_count": int(len(violations)),
        "violations": violations[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="One independent 500-scene V64.3.27 TRCC screen block.")
    ap.add_argument("--split-name", required=True)
    for name in ["raw", "v20", "aggregate-downside", "type-confirmed"]:
        ap.add_argument(f"--{name}-metrics", required=True)
        ap.add_argument(f"--{name}-rows", required=True)
        if name != "raw":
            ap.add_argument(f"--{name}-edges", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cli_tags = ["raw", "v20", "aggregate-downside", "type-confirmed"]
    tags = [x.replace("-", "_") for x in cli_tags]
    metrics = {t: json.load(open(getattr(args, t + "_metrics"), encoding="utf-8")) for t in tags}
    rows = {t: _load_rows(getattr(args, t + "_rows")) for t in tags}
    tokens = set(rows["raw"])
    if len(tokens) < 480 or any(set(rows[t]) != tokens for t in tags[1:]):
        raise SystemExit("STOP DATA: paired token identity mismatch")

    all_flagged = {t for t in tokens if _f(rows["raw"][t], "all_actions_safety_flagged_rate", 0.0) >= 0.5}
    safe = tokens - all_flagged
    edge_paths = {t: getattr(args, t + "_edges") for t in tags[1:]}
    edge = {t: _icer_edge_diag(Path(edge_paths[t]), safe) for t in tags[1:]}
    path = {t: _path_diag(rows["raw"], rows[t], edge_paths[t], safe, all_flagged) for t in tags[1:]}
    tails = {
        t: _replacement_tail_diag(rows["raw"], rows[t], edge_paths[t], safe)
        for t in ["aggregate_downside", "type_confirmed"]
    }
    subset_contract = _no_fallback_subset_contract(edge_paths["type_confirmed"], edge_paths["aggregate_downside"], safe)
    M = {t: _metric_pack(metrics[t]) for t in tags}
    main = "type_confirmed"
    ctrl = "aggregate_downside"
    anchor = {
        "match": _f(metrics["raw"], "selected_local_anchor_action_match"),
        "regret": _f(metrics["raw"], "selected_local_anchor_teacher_regret"),
    }

    structural = {
        "all_flagged_scene_count": len(all_flagged),
        "main_all_flagged_final_identity_vs_raw": _identity_rate(rows[main], rows["raw"], all_flagged),
        "main_all_flagged_delegation_rate": (
            sum(_f(rows[main][t], "decisive_frontier_icer_structural_domain_delegated", 0.0) >= 0.5 for t in all_flagged) / len(all_flagged)
            if all_flagged
            else float("nan")
        ),
        "main_safe_guard_block_rate": _guard_block_rate(rows[main], safe),
    }
    frozen_keys = [
        "selected_local_anchor_action_match", "pair_full_interface_action_match", "local_pair_full_interface_action_match",
        "evidence_certificate_fraction", "decision_budget_atom_count", "proposal_candidate_atom_count",
        "proposal_decisive_atom_recall", "selected_decisive_atom_recall", "effective_selected_decisive_atom_recall",
    ]
    frozen = {
        k: bool(
            math.isfinite(_f(metrics["raw"], k))
            and math.isfinite(_f(metrics[main], k))
            and abs(_f(metrics["raw"], k) - _f(metrics[main], k)) <= 1.0e-6
        )
        for k in frozen_keys
    }

    c = edge[main]
    instrumentation = bool(
        c["scene_count"] >= 450
        and c["admissible_edge_count"] >= 1800
        and c["direct_counterfactual_dominance_edge_count"] >= 400
        and _f(metrics[main], "decisive_frontier_value_complete_star_coverage") >= 0.99
        and all(frozen.values())
    )
    structural_ok = bool(
        len(all_flagged) >= 3
        and structural["main_all_flagged_final_identity_vs_raw"] == 1.0
        and structural["main_all_flagged_delegation_rate"] == 1.0
        and structural["main_safe_guard_block_rate"] <= 0.001
    )
    candidate = bool(c["multi_admissible_proposal_rate"] >= 0.25 and c["admissible_candidates_per_proposal_mean"] >= 3.0)
    reliability = bool(c["support_auc"] >= 0.65 and c["direct_counterfactual_dominance_auc"] >= 0.70)

    incumbent_to_anchor = path[main]["admissible_incumbent_to_anchor"]
    replacement = path[main]["direct_incumbent_to_alternative"]
    asymmetric = bool(incumbent_to_anchor["count"] == 0 and abs(incumbent_to_anchor["regret_delta_sum"]) <= 1.0e-9)
    path_safe = bool(replacement["count"] >= 8 and replacement["regret_delta_sum"] <= 0.0)
    recovery = bool(
        c["alternative_recovery_rate"] >= 0.03
        and c["alternative_recovery_precision"] >= 0.80
        and c["direct_incumbent_replacement_rate"] >= 0.02
        and c["direct_incumbent_replacement_precision"] >= 0.60
        and c["direct_incumbent_opportunity_capture_rate"] >= 0.08
        and c["alternative_teacher_margin_mean"] > 0.0
    )

    # Causal claim for V27: the downside objective, K, zero boundary, support,
    # dominance and extremal ranking are frozen from V25.  The only learned-risk
    # change is identity-preserving type-resolved coordinates in the local
    # neighborhood.  The main must therefore improve the *selected negative tail*
    # over the V25 aggregate-DRC control, not merely change AUC or endpoint noise.
    main_tail = tails[main]
    ctrl_tail = tails[ctrl]
    tail_instrumentation = bool(
        main_tail["count"] == replacement["count"]
        and ctrl_tail["count"] == path[ctrl]["direct_incumbent_to_alternative"]["count"]
    )
    tail_nonworse = bool(
        tail_instrumentation
        and math.isfinite(main_tail["teacher_negative_rms"])
        and math.isfinite(ctrl_tail["teacher_negative_rms"])
        and math.isfinite(main_tail["teacher_improvement_worst"])
        and math.isfinite(ctrl_tail["teacher_improvement_worst"])
        and main_tail["teacher_negative_rms"] <= ctrl_tail["teacher_negative_rms"] + 1.0e-9
        and main_tail["teacher_improvement_worst"] >= ctrl_tail["teacher_improvement_worst"] - 1.0e-9
    )
    tail_strict = bool(
        tail_instrumentation
        and (
            main_tail["teacher_negative_rms"] < ctrl_tail["teacher_negative_rms"] - 1.0e-9
            or main_tail["teacher_improvement_worst"] > ctrl_tail["teacher_improvement_worst"] + 1.0e-9
        )
    )
    type_confirmation_incremental = bool(
        tail_nonworse
        and tail_strict
        and M[main]["regret"] <= M[ctrl]["regret"] * 1.02
    )


    preservation = bool(
        M[main]["harmful"] <= M["raw"]["harmful"] + 0.005
        and M[main]["flip"] <= M["raw"]["flip"] + 0.01
        and path_safe
        and asymmetric
    )
    endpoint = bool(
        M[main]["match"] >= anchor["match"] + 0.005
        and M[main]["regret"] <= M["raw"]["regret"] * 1.02
        and M[main]["regret"] <= M["v20"]["regret"] * 1.02
    )

    full = bool(
        instrumentation and tail_instrumentation and structural_ok and candidate and reliability and asymmetric
        and subset_contract["pass"] and path_safe and recovery and type_confirmation_incremental and preservation and endpoint
    )
    if full:
        next_action = "split_pass_freeze_TRCC_wait_for_second_block"
    elif not instrumentation or not tail_instrumentation:
        next_action = "engineering_or_frozen_interface_failure"
    elif not structural_ok:
        next_action = "deployment_semantics_failure_do_not_change_regret_certificate"
    elif not candidate or not reliability:
        next_action = "frozen_frontier_or_support_dominance_regression"
    elif not asymmetric:
        next_action = "incumbent_default_contract_broken_engineering_failure"
    elif not subset_contract["pass"]:
        next_action = "TRCC_no_fallback_subset_contract_broken_engineering_failure"
    elif not path_safe:
        next_action = "TRCC_failed_selected_tail_do_not_tune_type_weights_K_multiplier_or_zero_audit_residual_aliasing"
    elif not recovery:
        next_action = "path_safe_but_too_conservative_do_not_tune_zero_or_K_audit_support_capture"
    elif not type_confirmation_incremental:
        next_action = "type_confirmation_not_incremental_over_V25_DRC_do_not_tune_type_weights_K_or_thresholds"
    elif not preservation:
        next_action = "replacement_path_safe_but_raw_harm_non_degradation_failed_audit_action_changes"
    else:
        next_action = "mechanism_pass_but_endpoint_fail_audit_remaining_teacher_regret_tail"

    report = {
        "audit": "v64_3_27_eaf_icer_trcc_split",
        "split_name": args.split_name,
        "main_algorithm_arm": main,
        "full_split_pass": full,
        "instrumentation_valid": instrumentation,
        "selected_tail_instrumentation_valid": tail_instrumentation,
        "deployment_alignment": structural_ok,
        "candidate_support_valid": candidate,
        "frozen_support_dominance_signal": reliability,
        "incumbent_default_invariant": asymmetric,
        "no_fallback_selected_path_subset_invariant": bool(subset_contract["pass"]),
        "selected_replacement_path_nonharmful": path_safe,
        "counterfactual_recovery_mechanism": recovery,
        "type_confirmation_tail_incremental": type_confirmation_incremental,
        "type_confirmation_tail_nonworse": tail_nonworse,
        "type_confirmation_tail_strict": tail_strict,
        "asymmetric_preservation_non_degradation": preservation,
        "endpoint_gain": endpoint,
        "next_action": next_action,
        "structural": structural,
        "no_fallback_subset_contract": subset_contract,
        "edge_diagnostics": edge,
        "path_diagnostics": path,
        "selected_replacement_tail_diagnostics": tails,
        "metrics": {"anchor": anchor, **M},
        "frozen_interface": frozen,
        "thresholds": {
            "support_auc_min": 0.65,
            "dominance_auc_min": 0.70,
            "incumbent_to_anchor_count_max": 0,
            "replacement_path_regret_sum_max": 0.0,
            "direct_precision_min": 0.60,
            "capture_min": 0.08,
            "raw_harmful_non_degradation_tolerance": 0.005,
            "raw_flip_non_degradation_tolerance": 0.01,
            "match_over_anchor_min": 0.005,
            "regret_tolerance": 0.02,
            "K_downside_multiplier_zero_boundary": "fixed on TRAIN: K=32/64, multiplier=1, boundary=0; no validation sweep",
            "type_confirmation_metric": "aggregate DRC proposes one candidate; independent 24-D fixed atom-type signed view may only veto that same candidate; no fallback/reselection and no type/group weight sweep",
        },
        "interpretation": (
            "V27 tests one causal change after V26: a finer fixed atom-type evidence view confirms only the single candidate proposed by the frozen V25 aggregate DRC. "
            "The V25 aggregate-downside arm is the sole mechanism control; support, scalar dominance, K=32/64, downside multiplier=1, zero boundary and extremal proposal ranking are frozen.  Failed confirmation returns directly to the incumbent, so the main selected replacement set must be a strict subset of the control and cannot resurrect a new alternative. "
            "Promotion requires non-harmful selected replacements, recovery, strictly incremental selected negative-tail behavior over V25 DRC, asymmetric preservation and endpoint non-inferiority independently on this block."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
