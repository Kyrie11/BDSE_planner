from __future__ import annotations

"""V64.3.50.6 engineering repair for the preregistered V50 PIOR fit.

Scientific mechanism is unchanged from V64.3.50 PIOR.  This module fixes two
analysis-pipeline defects that prevented/corrupted evaluation of the already
collected paired closed-loop outcomes:

1. V50 imported V48's historical fixed ``n >= 16`` calibration guard.  The
   actual V50 paired-outcome calibration fold has 15 beneficial examples, even
   though 15 is sufficient for the frozen alpha under the exact finite-sample
   split-conformal rank.  We therefore use the mathematically required finite
   rank condition ``ceil((n+1)(1-alpha)) <= n`` rather than V48's unrelated
   fixed support guard.  Alpha, calibration fold, score, and quantile rule are
   unchanged.
2. V50 concatenated per-fold OOF keep decisions in fold order and then applied
   them positionally to token-sorted rows for the aggregate deployment gate.
   We align OOF decisions by scenario_token before aggregate metrics.

No feature, label, loss, lambda, selector, threshold budget, safety definition,
or runtime policy is changed.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bdse.tools import fit_v64_3_50_eaf_icer_pior as v50
from bdse.tools.fit_v64_3_48_eaf_icer_ocrr import _fit_sign_ranker, _risk
from bdse.tools.fit_v64_3_49_eaf_icer_siir import _auc

EPS = v50.EPS
FOLDS = v50.FOLDS


def _pior_conformal_threshold(
    cal: list[dict[str, Any]], model: dict[str, Any], alpha: float
) -> tuple[float, dict[str, Any]]:
    """Frozen-alpha split-conformal threshold with the exact finite-sample rank.

    A finite empirical threshold exists iff
        ceil((n + 1) * (1 - alpha)) <= n.
    This is the only sample-size condition needed by the rank rule used in V48
    and preregistered for V50.  We fail closed rather than clipping the rank when
    the condition is not met.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"V50.6 invalid retention alpha: {alpha}")
    vals = sorted(
        _risk(r, model)
        for r in cal
        if int(r["rsm_selected_action"]) >= 0
        and float(r["rsm_selected_teacher_improvement"]) > 0.0
    )
    n = len(vals)
    rank = int(math.ceil((n + 1) * (1.0 - float(alpha)))) if n else 1
    min_n_for_finite_rank = int(math.ceil((1.0 - float(alpha)) / float(alpha)))
    if n == 0 or rank > n:
        raise ValueError(
            "V50.6 calibration has insufficient paired beneficial outcomes for "
            f"a finite frozen-alpha split-conformal threshold: n={n}, "
            f"required_at_least={min_n_for_finite_rank}, alpha={alpha:.12g}, "
            f"requested_rank={rank}"
        )
    tau = float(vals[rank - 1])
    return tau, {
        "positive_calibration_count": n,
        "conformal_rank": rank,
        "alpha": float(alpha),
        "threshold": tau,
        "finite_sample_condition": "ceil((n+1)*(1-alpha))<=n",
        "minimum_positive_count_for_finite_rank": min_n_for_finite_rank,
        "engineering_repair": "V50.6_removed_unrelated_V48_fixed_n16_guard_without_changing_rank_rule",
    }


def _align_oof_keep_by_token(
    rows: list[dict[str, Any]], keep_by_token: dict[str, bool]
) -> list[bool]:
    tokens = [str(r["scenario_token"]) for r in rows]
    if len(tokens) != len(set(tokens)):
        raise RuntimeError("V50.6 ENGINEERING STOP: duplicate scenario_token in PIOR rows")
    if set(tokens) != set(keep_by_token):
        missing = sorted(set(tokens) - set(keep_by_token))
        extra = sorted(set(keep_by_token) - set(tokens))
        raise RuntimeError(
            "V50.6 ENGINEERING STOP: OOF keep/token identity mismatch "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    return [bool(keep_by_token[t]) for t in tokens]


def _nested(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    all_target: list[bool] = []
    all_ego: list[float] = []
    all_obs: list[float] = []
    all_pior: list[float] = []
    keep_by_token: dict[str, bool] = {}
    p_better_obs = p_better_ego = 0

    for k in range(FOLDS):
        cf = (k + 1) % FOLDS
        fit = [r for r in rows if int(r["outer_test_fold"]) not in {k, cf}]
        cal = [r for r in rows if int(r["outer_test_fold"]) == cf]
        test = [r for r in rows if int(r["outer_test_fold"]) == k]
        model = _fit_sign_ranker(fit, False)
        tau, cal_info = _pior_conformal_threshold(cal, model, alpha)

        target = np.asarray([not bool(r["closed_loop_beneficial"]) for r in test], dtype=bool)
        ego = np.asarray([-float(r["ego_ref_value"]) for r in test], dtype=np.float64)
        obs = np.asarray([float(r["v49_obs_sign_risk"]) for r in test], dtype=np.float64)
        pior = np.asarray([_risk(r, model) for r in test], dtype=np.float64)
        keep = [bool(x <= tau) for x in pior]
        for r, kkeep in zip(test, keep):
            tok = str(r["scenario_token"])
            if tok in keep_by_token:
                raise RuntimeError(f"V50.6 ENGINEERING STOP: duplicate OOF keep for {tok}")
            keep_by_token[tok] = bool(kkeep)

        ae = _auc(target, ego)
        ao = _auc(target, obs)
        ap = _auc(target, pior)
        p_better_ego += int(math.isfinite(ap) and math.isfinite(ae) and ap > ae + EPS)
        p_better_obs += int(math.isfinite(ap) and math.isfinite(ao) and ap > ao + EPS)
        base_m = v50._delta_metrics(test, [True] * len(test))
        pior_m = v50._delta_metrics(test, keep)
        folds.append(
            {
                "fold": k,
                "fit_events": len(fit),
                "calibration_events": len(cal),
                "test_events": len(test),
                "rsmr": base_m,
                "pior": pior_m,
                "risk_identification": {
                    "ego_ref_auc": ae,
                    "offline_obs_auc": ao,
                    "pior_auc": ap,
                    "pior_better_ego": ap > ae + EPS,
                    "pior_better_obs": ap > ao + EPS,
                },
                "calibration": cal_info,
            }
        )
        all_target.extend(target.tolist())
        all_ego.extend(ego.tolist())
        all_obs.extend(obs.tolist())
        all_pior.extend(pior.tolist())

    target = np.asarray(all_target, dtype=bool)
    ego = np.asarray(all_ego)
    obs = np.asarray(all_obs)
    pior = np.asarray(all_pior)
    ae = _auc(target, ego)
    ao = _auc(target, obs)
    ap = _auc(target, pior)
    identified = bool(ap > max(ae, ao) + EPS and p_better_ego >= 4 and p_better_obs >= 4)

    aligned_keep = _align_oof_keep_by_token(rows, keep_by_token)
    base = v50._delta_metrics(rows, [True] * len(rows))
    kept = v50._delta_metrics(rows, aligned_keep)
    gate = v50._gate(base, kept, alpha, folds)
    return {
        "folds": folds,
        "retention_alpha": alpha,
        "rsmr_interventional_outcome_aggregate": base,
        "pior_aggregate": kept,
        "risk_identification": {
            "aggregate_ego_ref_auc_on_closed_loop_outcome": ae,
            "aggregate_offline_obs_auc_on_closed_loop_outcome": ao,
            "aggregate_pior_auc_on_closed_loop_outcome": ap,
            "pior_better_ego_fold_count": p_better_ego,
            "pior_better_offline_obs_fold_count": p_better_obs,
            "identified": identified,
        },
        "deployment_gate": gate,
        "train_gate_pass": bool(identified and gate["pass"]),
        "failure_diagnosis": (
            "paired_interventional_outcome_identification_and_retention_pass"
            if identified and gate["pass"]
            else "closed_loop_outcome_identifiable_but_retention_functional_insufficient"
            if identified
            else "paired_closed_loop_outcome_source_does_not_identify_transportable_QPE_retention_risk"
        ),
        "engineering_repair": {
            "oof_keep_alignment": "scenario_token_keyed_before_aggregate_metrics",
            "calibration": "exact_finite_sample_rank_same_frozen_alpha_no_threshold_sweep",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="V64.3.50.6 fit-only engineering repair for V50 PIOR on already collected paired outcomes."
    )
    ap.add_argument("--v49-fit-report", type=Path, required=True)
    ap.add_argument("--v49-candidate-audit", type=Path, required=True)
    ap.add_argument("--v49-scene-audit", type=Path, required=True)
    ap.add_argument("--v49-siir-config", type=Path, required=True)
    ap.add_argument("--paired-outcomes", type=Path, required=True)
    ap.add_argument("--output-config", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    a = ap.parse_args()

    v49 = v50._check_v49(a.v49_fit_report)
    alpha = float(v49["nested_crossfit"]["retention_alpha"])
    states = v50._candidate_states(a.v49_candidate_audit)
    obs = v50._obs_risk(a.v49_scene_audit)
    rows = v50._join(states, a.paired_outcomes, obs)
    nested = _nested(rows, alpha)

    fit = [r for r in rows if int(r["outer_test_fold"]) != 0]
    cal = [r for r in rows if int(r["outer_test_fold"]) == 0]
    model = _fit_sign_ranker(fit, False)
    tau, cal_info = _pior_conformal_threshold(cal, model, alpha)

    report = {
        "audit": "v64_3_50_6_eaf_icer_pior_fit_repair",
        "algorithm_version": "V64.3.50-EAF-ICER-PIOR",
        "engineering_revision": "V64.3.50.6",
        "scientific_mechanism_unchanged": True,
        "repair_scope": [
            "replace_inherited_V48_fixed_n16_calibration_guard_with_exact_finite_sample_rank_condition",
            "align_nested_OOF_keep_decisions_to_rows_by_scenario_token_before_aggregate_gate",
        ],
        "frozen_contract": {
            "RSMR_selector_unchanged": True,
            "Q_P_E_runtime_state_unchanged": True,
            "zero_bias_pairwise_sign_risk_unchanged": True,
            "lambda": 1.0,
            "retention_alpha_unchanged": True,
            "split_conformal_rank_rule_unchanged": True,
            "paired_outcome_labels_unchanged": True,
            "hard_safety_definition_unchanged": True,
            "same_winner_or_incumbent_only": True,
            "no_rerank_second_best_fallback": True,
        },
        "nested_crossfit": nested,
        "final_runtime_fit": {"model": model, "calibration": cal_info},
        "train_gate_pass": bool(nested["train_gate_pass"]),
        "preregistered_next_branch": {
            "if_train_pass": "freeze_PIOR_and_run_untouched_paired_closed_loop_validation; no TRAIN tuning",
            "if_train_fail_identification": "QPE state is insufficient for paired selected-outcome discrimination; move to on-policy structured outcome state/evidence, not offline sweeps",
            "if_train_fail_retention": "low-capacity sign-risk functional is insufficient despite identifiable paired outcome; next branch is structured closed-loop outcome functional under on-policy evidence",
        },
    }
    a.output_report.parent.mkdir(parents=True, exist_ok=True)
    a.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not nested["train_gate_pass"]:
        try:
            if a.output_config.exists():
                a.output_config.unlink()
        except OSError:
            pass
        print(
            json.dumps(
                {
                    "pass": False,
                    "failure_diagnosis": nested["failure_diagnosis"],
                    "output_config_emitted": False,
                    "engineering_revision": "V64.3.50.6",
                },
                sort_keys=True,
            )
        )
        raise SystemExit(
            f"V64.3.50.6 PIOR nested TRAIN gate failed ({nested['failure_diagnosis']}); "
            "scientific STOP before untouched closed-loop validation"
        )

    # Runtime mechanism is still exactly V50 PIOR; decorate with the frozen V50
    # runtime schema after the repaired scientific gate passes.
    v50._decorate(a.v49_siir_config, model, tau, a.output_config)
    print(
        json.dumps(
            {
                "pass": True,
                "failure_diagnosis": nested["failure_diagnosis"],
                "output_config": str(a.output_config),
                "output_config_emitted": True,
                "engineering_revision": "V64.3.50.6",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
