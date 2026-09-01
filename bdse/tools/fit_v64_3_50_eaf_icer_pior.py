from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.tools.fit_v64_3_48_eaf_icer_ocrr import _fit_sign_ranker, _risk, _conformal_threshold
from bdse.tools.fit_v64_3_49_eaf_icer_siir import _auc

EPS = 1.0e-12
FOLDS = 5
EXPECTED_V49_FAILURE = "selection_interventional_risk_does_not_outperform_observational_selected_risk_close_current_offline_selected_risk_family"
EXPECTED_V49_OBS_AUC = 0.6139192605594113
EXPECTED_V49_SIIR_AUC = 0.6081222524597028


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _check_v49(report_path: Path) -> dict[str, Any]:
    r = json.loads(report_path.read_text(encoding="utf-8"))
    n = r.get("nested_crossfit", {})
    if r.get("train_gate_pass") is not False or n.get("train_gate_pass") is not False:
        raise RuntimeError("V64.3.50 ENGINEERING STOP: V49 must be the preregistered nested-TRAIN failure")
    if n.get("failure_diagnosis") != EXPECTED_V49_FAILURE:
        raise RuntimeError(f"V64.3.50 ENGINEERING STOP: V49 failure branch changed: {n.get('failure_diagnosis')}")
    ri = n.get("risk_identification", {})
    if abs(float(ri.get("aggregate_obs_sign_auc", float("nan"))) - EXPECTED_V49_OBS_AUC) > 1e-12:
        raise RuntimeError("V64.3.50 ENGINEERING STOP: V49 OBS AUC signature changed")
    if abs(float(ri.get("aggregate_siir_auc", float("nan"))) - EXPECTED_V49_SIIR_AUC) > 1e-12:
        raise RuntimeError("V64.3.50 ENGINEERING STOP: V49 SIIR AUC signature changed")
    return r


def _candidate_states(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in _read_jsonl(path):
        tok = str(r["scenario_token"])
        a = int(r.get("full_selected_action", -1))
        if a < 0:
            continue
        cand = [x for x in r.get("candidates", []) if int(x.get("action", -999)) == a]
        if len(cand) != 1:
            raise RuntimeError(f"V64.3.50 ENGINEERING STOP: full selected candidate not unique for {tok}: action={a} n={len(cand)}")
        c = cand[0]
        out[tok] = {
            "scenario_token": tok,
            "outer_test_fold": int(r["outer_test_fold"]),
            "candidate_count": int(r["candidate_count"]),
            "rsm_selected_action": a,
            "quality_value": float(c["quality_value"]),
            "plan_control_value": float(c["plan_control_value"]),
            "ego_ref_value": float(c["ego_ref_value"]),
            "offline_teacher_improvement": float(c["y"]),
        }
    if len(out) != 502:
        raise RuntimeError(f"V64.3.50 ENGINEERING STOP: V49 exact full-set RSMR event population must be 502, got {len(out)}")
    return out


def _obs_risk(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(float(r["rsm_selected_action"])) < 0:
                continue
            out[str(r["scenario_token"])] = float(r["v49_obs_sign_risk"])
    if len(out) != 502:
        raise RuntimeError(f"V64.3.50 ENGINEERING STOP: V49 OBS risk audit expected 502 selected events, got {len(out)}")
    return out


def _join(states: dict[str, dict[str, Any]], outcomes_path: Path, obs: dict[str, float]) -> list[dict[str, Any]]:
    outcomes = {str(r["scenario_token"]): r for r in _read_jsonl(outcomes_path)}
    if set(outcomes) != set(states):
        miss = sorted(set(states)-set(outcomes)); extra = sorted(set(outcomes)-set(states))
        raise RuntimeError(f"V64.3.50 PIOR paired outcome identity mismatch missing={miss[:10]} extra={extra[:10]}")
    rows: list[dict[str, Any]] = []
    for tok in sorted(states):
        s = dict(states[tok]); o = outcomes[tok]
        y = float(o["pior_interventional_outcome"])
        if y not in {-1.0, 1.0}:
            raise RuntimeError(f"invalid PIOR sign label for {tok}: {y}")
        s.update({
            "rsm_selected_teacher_improvement": y,  # adapter field consumed by unchanged V48 pairwise sign-ranker; source is closed-loop intervention, not teacher.
            "pior_interventional_outcome": y,
            "closed_loop_beneficial": bool(o["closed_loop_beneficial"]),
            "closed_loop_hard_harm": bool(o["closed_loop_hard_harm"]),
            "closed_loop_score_delta": float(o["closed_loop_score_delta"]),
            "official_score_metric": str(o["official_score_metric"]),
            "safety_delta": dict(o.get("safety_delta", {})),
            "v49_obs_sign_risk": float(obs[tok]),
        })
        rows.append(s)
    if len(rows) != 502:
        raise RuntimeError("PIOR join must remain 502 events")
    return rows


def _delta_metrics(rows: list[dict[str, Any]], keep: list[bool]) -> dict[str, Any]:
    chosen = [r for r, k in zip(rows, keep) if k]
    pos_total = sum(int(r["closed_loop_beneficial"]) for r in rows)
    pos = sum(int(r["closed_loop_beneficial"]) for r in chosen)
    bad = len(chosen) - pos
    harm = sum(int(r["closed_loop_hard_harm"]) for r in chosen)
    vals = np.asarray([float(r["closed_loop_score_delta"]) for r in chosen], dtype=np.float64)
    neg = np.minimum(vals, 0.0)
    return {
        "selected_count": len(chosen),
        "beneficial_selected_count": pos,
        "beneficial_total_count": pos_total,
        "beneficial_retention_recall": float(pos / pos_total) if pos_total else float("nan"),
        "nonbeneficial_selected_count": bad,
        "hard_harm_selected_count": harm,
        "closed_loop_score_delta_sum": float(vals.sum()) if vals.size else 0.0,
        "closed_loop_score_delta_worst": float(vals.min()) if vals.size else float("nan"),
        "closed_loop_negative_rms": float(np.sqrt(np.mean(neg * neg))) if vals.size else 0.0,
    }


def _gate(base: dict[str, Any], pior: dict[str, Any], alpha: float, folds: list[dict[str, Any]]) -> dict[str, Any]:
    retention = bool(pior["beneficial_retention_recall"] >= 1.0 - alpha - EPS)
    hard_tail = bool(
        pior["hard_harm_selected_count"] <= base["hard_harm_selected_count"]
        and (base["hard_harm_selected_count"] == 0 or pior["hard_harm_selected_count"] < base["hard_harm_selected_count"])
        and pior["closed_loop_negative_rms"] <= base["closed_loop_negative_rms"] + EPS
    )
    bad_reduction = bool(pior["nonbeneficial_selected_count"] < base["nonbeneficial_selected_count"])
    utility_nonharm = bool(pior["closed_loop_score_delta_sum"] >= base["closed_loop_score_delta_sum"] - EPS)
    fold_nonharm = all(
        f["pior"]["hard_harm_selected_count"] <= f["rsmr"]["hard_harm_selected_count"]
        and f["pior"]["closed_loop_score_delta_sum"] >= f["rsmr"]["closed_loop_score_delta_sum"] - EPS
        for f in folds
    )
    population = bool(pior["selected_count"] >= 64 and pior["beneficial_selected_count"] >= 32)
    return {
        "beneficial_retention": retention,
        "hard_tail": hard_tail,
        "nonbeneficial_reduction": bad_reduction,
        "utility_nonharm": utility_nonharm,
        "all_folds_nonharm": fold_nonharm,
        "population": population,
        "pass": bool(retention and hard_tail and bad_reduction and utility_nonharm and fold_nonharm and population),
    }


def _nested(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    all_target: list[bool] = []
    all_ego: list[float] = []
    all_obs: list[float] = []
    all_pior: list[float] = []
    all_keep: list[bool] = []
    p_better_obs = p_better_ego = 0
    for k in range(FOLDS):
        cf = (k + 1) % FOLDS
        fit = [r for r in rows if int(r["outer_test_fold"]) not in {k, cf}]
        cal = [r for r in rows if int(r["outer_test_fold"]) == cf]
        test = [r for r in rows if int(r["outer_test_fold"]) == k]
        model = _fit_sign_ranker(fit, False)
        tau, cal_info = _conformal_threshold(cal, model, alpha)
        target = np.asarray([not bool(r["closed_loop_beneficial"]) for r in test], dtype=bool)
        ego = np.asarray([-float(r["ego_ref_value"]) for r in test], dtype=np.float64)
        obs = np.asarray([float(r["v49_obs_sign_risk"]) for r in test], dtype=np.float64)
        pior = np.asarray([_risk(r, model) for r in test], dtype=np.float64)
        keep = [bool(x <= tau) for x in pior]
        ae = _auc(target, ego); ao = _auc(target, obs); ap = _auc(target, pior)
        p_better_ego += int(math.isfinite(ap) and math.isfinite(ae) and ap > ae + EPS)
        p_better_obs += int(math.isfinite(ap) and math.isfinite(ao) and ap > ao + EPS)
        base_m = _delta_metrics(test, [True] * len(test)); pior_m = _delta_metrics(test, keep)
        folds.append({
            "fold": k, "fit_events": len(fit), "calibration_events": len(cal), "test_events": len(test),
            "rsmr": base_m, "pior": pior_m,
            "risk_identification": {"ego_ref_auc": ae, "offline_obs_auc": ao, "pior_auc": ap, "pior_better_ego": ap > ae + EPS, "pior_better_obs": ap > ao + EPS},
            "calibration": cal_info,
        })
        all_target.extend(target.tolist()); all_ego.extend(ego.tolist()); all_obs.extend(obs.tolist()); all_pior.extend(pior.tolist()); all_keep.extend(keep)
    target = np.asarray(all_target, dtype=bool); ego = np.asarray(all_ego); obs = np.asarray(all_obs); pior = np.asarray(all_pior)
    ae = _auc(target, ego); ao = _auc(target, obs); ap = _auc(target, pior)
    identified = bool(ap > max(ae, ao) + EPS and p_better_ego >= 4 and p_better_obs >= 4)
    base = _delta_metrics(rows, [True] * len(rows)); kept = _delta_metrics(rows, all_keep)
    gate = _gate(base, kept, alpha, folds)
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
            if identified and gate["pass"] else
            "closed_loop_outcome_identifiable_but_retention_functional_insufficient"
            if identified else
            "paired_closed_loop_outcome_source_does_not_identify_transportable_QPE_retention_risk"
        ),
    }


def _decorate(v49_cfg_path: Path, model: dict[str, Any], tau: float, output: Path) -> None:
    cfg = yaml.safe_load(v49_cfg_path.read_text(encoding="utf-8"))
    cfg.pop("selected_outcome_probe", None)
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    sc = ic["selection_conditioned_intervention_recovery"]
    rr = sc["operator_conditioned_risk_retention"]
    rr.update({
        "aggregation": "sign_only",
        "use_extremal_multiplicity": False,
        "components": {"sign_risk": model},
        "retention_threshold": float(tau),
        "threshold_calibration": "TRAIN_paired_closed_loop_beneficial_RSMR_proposals_split_conformal_using_frozen_V48_false_veto_budget",
        "identification_distribution": "TRAIN_paired_one_shot_closed_loop_actual_full_set_RSMR_proposal_vs_same_incumbent",
        "runtime_candidate_set": "full_frozen_deployment_candidate_set",
        "evidence_source": "interventional_selected_outcome_supervision_not_offline_teacher_surrogate",
    })
    sc["post_selection_value_training"] = "paired_closed_loop_selected_outcome_pairwise_sign_risk_fixed_lambda_1_same_QPE_no_multiplicity"
    sc["post_selection_operator"] = "freeze_full_set_RSMR_winner_then_PIOR_veto_only_same_winner_or_incumbent_no_rerank_no_fallback"
    v = "V64.3.50-EAF-ICER-PIOR"
    cfg.setdefault("metadata", {})["algorithm_version"] = v
    cfg.setdefault("provenance", {})["algorithm_version"] = v
    cfg.setdefault("experiment", {})["name"] = "v64_3_50_eaf_icer_pior"
    cfg["experiment"]["algorithm"] = v
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.50 Paired Interventional Outcome Retention on actual one-shot closed-loop selected outcomes.")
    ap.add_argument("--v49-fit-report", type=Path, required=True)
    ap.add_argument("--v49-candidate-audit", type=Path, required=True)
    ap.add_argument("--v49-scene-audit", type=Path, required=True)
    ap.add_argument("--v49-siir-config", type=Path, required=True)
    ap.add_argument("--paired-outcomes", type=Path, required=True)
    ap.add_argument("--output-config", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    a = ap.parse_args()

    v49 = _check_v49(a.v49_fit_report)
    alpha = float(v49["nested_crossfit"]["retention_alpha"])
    states = _candidate_states(a.v49_candidate_audit)
    obs = _obs_risk(a.v49_scene_audit)
    rows = _join(states, a.paired_outcomes, obs)
    nested = _nested(rows, alpha)

    # Runtime artifact: same Q/P/E state, same zero-bias pairwise model and fixed
    # lambda=1. Only the outcome evidence source changed. Fit/cal split mirrors the
    # historical fold-0 holdout convention after the nested scientific gate.
    fit = [r for r in rows if int(r["outer_test_fold"]) != 0]
    cal = [r for r in rows if int(r["outer_test_fold"]) == 0]
    model = _fit_sign_ranker(fit, False)
    tau, cal_info = _conformal_threshold(cal, model, alpha)

    report = {
        "audit": "v64_3_50_eaf_icer_pior_fit",
        "algorithm_version": "V64.3.50-EAF-ICER-PIOR",
        "scientific_role": "evidence_source_intervention_after_preregistered_V49_offline_selected_risk_family_closure",
        "mechanism_hypothesis": "The remaining error is not another missing offline observable or loss. V49 shows that changing the offline selected-event measure with label-free random-prefix intervention does not improve either full-set or held-out-interventional risk AUC. PIOR therefore identifies the outcome law on the actual frozen full-set RSMR proposal using paired one-shot closed-loop proposal-vs-incumbent interventions, while keeping the deployment state Q/P/E, ranker family, lambda, selector, and veto-only operator fixed.",
        "frozen_contract": {
            "RSMR_selector_unchanged": True,
            "Q_P_E_runtime_state_unchanged": True,
            "zero_bias_pairwise_sign_risk_unchanged": True,
            "lambda": 1.0,
            "multiplicity_or_other_operator_observable": False,
            "same_winner_or_incumbent_only": True,
            "no_rerank_second_best_fallback": True,
            "new_information_is_TRAIN_only_paired_closed_loop_outcome_supervision": True,
            "deployment_logged_future_or_teacher": False,
        },
        "nested_crossfit": nested,
        "final_runtime_fit": {"model": model, "calibration": cal_info},
        "train_gate_pass": bool(nested["train_gate_pass"]),
        "preregistered_next_branch": {
            "if_train_pass": "freeze_PIOR_and_run_untouched_paired_closed_loop_validation; no offline A/B rescue or tuning",
            "if_train_fail": "QPE_retention_state_is_not_sufficient_even_with_aligned_selected-outcome evidence; next research must enrich on-policy outcome state/evidence acquisition or learn a structured closed-loop outcome functional, not return to offline feature/loss/threshold sweeps",
        },
        "prohibited_tuning": [
            "V49 intervention seed/prefix sweep", "K/logK/operator observable", "pairwise loss or lambda",
            "class/focal/catastrophe weighting", "retention threshold sweep", "bigger MLP", "new offline future observable",
            "CVaR/variance/handcrafted temporal profile", "AGENT-2D constant drift", "selected translation",
            "binary catastrophe veto", "RSMR/B/M/topK/candidate-count change", "second-best/fallback", "A/B pooling",
        ],
    }
    a.output_report.parent.mkdir(parents=True, exist_ok=True)
    a.output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not nested["train_gate_pass"]:
        # Do not leave a deployment-looking YAML after a scientific failure.
        try:
            if a.output_config.exists():
                a.output_config.unlink()
        except OSError:
            pass
        print(json.dumps({
            "pass": False,
            "failure_diagnosis": nested["failure_diagnosis"],
            "output_config_emitted": False,
        }, sort_keys=True))
        raise SystemExit(f"V64.3.50 PIOR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before untouched closed-loop validation")

    _decorate(a.v49_siir_config, model, tau, a.output_config)
    print(json.dumps({
        "pass": True,
        "failure_diagnosis": nested["failure_diagnosis"],
        "output_config": str(a.output_config),
        "output_config_emitted": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
