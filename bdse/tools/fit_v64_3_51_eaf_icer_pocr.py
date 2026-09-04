from __future__ import annotations

"""V64.3.51 EAF-ICER-POCR: Paired Operator-Contrast Retention.

Preregistered V50 branch: paired outcomes are complete and provenance-valid,
but Q/P/E-only selected-outcome risk fails the 4/5-fold identification gate.
V51 therefore changes only the *on-policy selected-outcome state*, not the
selector, outcome labels, loss family, lambda, retention alpha, calibration
rule, or monotone deployment operator.

Two causal arms are preregistered in fixed order:
  1. QPE+DOSE: [Q, P-Q, E-P, D]
  2. QPE+DOSE-X: [Q, P-Q, E-P, D, D*Q, D*(P-Q), D*(E-P)]
where D is the exact pre-execution physical trajectory contrast between the
frozen RSMR proposal and runtime incumbent.  The second arm is evaluated only
as the preregistered interaction alternative; promotion preference is always
the simpler additive arm when both pass.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.planner.paired_operator_contrast_retention import (
    POCR_ADDITIVE_STATE_NAMES,
    POCR_INTERACTION_STATE_NAMES,
)
from bdse.tools import fit_v64_3_50_eaf_icer_pior as v50
from bdse.tools import fit_v64_3_50_6_eaf_icer_pior as v506
from bdse.tools.fit_v64_3_49_eaf_icer_siir import _auc

EPS = 1.0e-12
FOLDS = 5
RIDGE_LAMBDA = 1.0
ARMS = ("qpe_dose", "qpe_dose_interaction")

EXPECTED_V50_FAILURE = "paired_closed_loop_outcome_source_does_not_identify_transportable_QPE_retention_risk"
EXPECTED_V50_EGO_AUC = 0.4746100952257001
EXPECTED_V50_OBS_AUC = 0.4994251751588903
EXPECTED_V50_PIOR_AUC = 0.5087091386303985
EXPECTED_V50_BETTER_EGO = 3
EXPECTED_V50_BETTER_OBS = 3
EXPECTED_V50_PAIRED_COUNT = 502


def _check_v50(report_path: Path) -> dict[str, Any]:
    d = json.loads(report_path.read_text(encoding="utf-8"))
    n = d.get("nested_crossfit", {})
    ri = n.get("risk_identification", {})
    if d.get("train_gate_pass") is not False or n.get("train_gate_pass") is not False:
        raise RuntimeError("V51 ENGINEERING STOP: V50.7 must replay the preregistered V50 TRAIN failure")
    if n.get("failure_diagnosis") != EXPECTED_V50_FAILURE:
        raise RuntimeError(f"V51 ENGINEERING STOP: V50 failure branch changed: {n.get('failure_diagnosis')}")
    checks = [
        ("ego", float(ri.get("aggregate_ego_ref_auc_on_closed_loop_outcome", float("nan"))), EXPECTED_V50_EGO_AUC),
        ("obs", float(ri.get("aggregate_offline_obs_auc_on_closed_loop_outcome", float("nan"))), EXPECTED_V50_OBS_AUC),
        ("pior", float(ri.get("aggregate_pior_auc_on_closed_loop_outcome", float("nan"))), EXPECTED_V50_PIOR_AUC),
    ]
    for name, actual, expected in checks:
        if not math.isfinite(actual) or abs(actual - expected) > 1e-12:
            raise RuntimeError(f"V51 ENGINEERING STOP: V50 {name} AUC signature changed: {actual} vs {expected}")
    if int(ri.get("pior_better_ego_fold_count", -1)) != EXPECTED_V50_BETTER_EGO or int(ri.get("pior_better_offline_obs_fold_count", -1)) != EXPECTED_V50_BETTER_OBS:
        raise RuntimeError("V51 ENGINEERING STOP: V50 fold-identification signature changed")
    base = n.get("rsmr_interventional_outcome_aggregate", {})
    if int(base.get("selected_count", -1)) != EXPECTED_V50_PAIRED_COUNT or int(base.get("beneficial_total_count", -1)) != 121:
        raise RuntimeError("V51 ENGINEERING STOP: V50 paired population signature changed")
    return d


def _load_treatment_contrast(root: Path, expected_tokens: set[str]) -> dict[str, float]:
    by: dict[str, float] = {}
    files = sorted((root / "closed_loop_train" / "treatment" / "batches").glob("batch_*/pior_probe_events.jsonl"))
    if not files:
        raise RuntimeError("V51 ENGINEERING STOP: no V50.5 treatment probe-event files")
    for path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                tok = str(r.get("scenario_token", ""))
                if tok not in expected_tokens:
                    continue
                if tok in by:
                    raise RuntimeError(f"V51 ENGINEERING STOP: duplicate treatment probe event for {tok}")
                if str(r.get("pior_probe_arm", "")) != "treatment" or not bool(r.get("pior_probe_fired", False)) or int(r.get("pior_probe_event_count", -1)) != 1:
                    raise RuntimeError(f"V51 ENGINEERING STOP: invalid one-shot treatment probe for {tok}")
                if str(r.get("pior_probe_physical_identity_contract", "")) != "cached_V49_proposal_trajectory_vs_runtime_incumbent_trajectory":
                    raise RuntimeError(f"V51 ENGINEERING STOP: missing V50 physical contrast contract for {tok}")
                if not bool(r.get("pior_probe_contract_same_frozen_proposal_or_incumbent", False)) or not bool(r.get("pior_probe_contract_no_rerank_second_best_fallback", False)):
                    raise RuntimeError(f"V51 ENGINEERING STOP: V50 containment contract failed for {tok}")
                d = float(r.get("pior_probe_frozen_vs_runtime_incumbent_geometry_max_abs_error", float("nan")))
                if not math.isfinite(d) or d < 0.0:
                    raise RuntimeError(f"V51 ENGINEERING STOP: invalid execution contrast for {tok}: {d}")
                if bool(r.get("pior_probe_frozen_equals_runtime_incumbent_physical", False)) and abs(d) > EPS:
                    raise RuntimeError(f"V51 ENGINEERING STOP: physical equality but nonzero contrast for {tok}: {d}")
                by[tok] = d
    if set(by) != expected_tokens:
        miss = sorted(expected_tokens - set(by)); extra = sorted(set(by) - expected_tokens)
        raise RuntimeError(f"V51 ENGINEERING STOP: treatment contrast identity mismatch {len(by)}/502 missing={miss[:10]} extra={extra[:10]}")
    return by


def _arm_state(r: dict[str, Any], arm: str) -> np.ndarray:
    q = float(r["quality_value"]); p = float(r["plan_control_value"]); e = float(r["ego_ref_value"])
    base = np.asarray([q, p - q, e - p], dtype=np.float64)
    d = float(r["operator_execution_contrast_linf"])
    if not math.isfinite(d) or d < 0.0 or np.any(~np.isfinite(base)):
        raise ValueError(f"V51 non-finite operator-contrast state {r['scenario_token']}")
    if arm == "qpe_dose":
        return np.concatenate([base, np.asarray([d], dtype=np.float64)])
    if arm == "qpe_dose_interaction":
        return np.concatenate([base, np.asarray([d], dtype=np.float64), d * base])
    raise ValueError(f"unknown V51 arm {arm}")


def _arm_names(arm: str) -> list[str]:
    if arm == "qpe_dose": return list(POCR_ADDITIVE_STATE_NAMES)
    if arm == "qpe_dose_interaction": return list(POCR_INTERACTION_STATE_NAMES)
    raise ValueError(arm)


def _fit_ranker(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    X = np.stack([_arm_state(r, arm) for r in rows])
    y = np.asarray([float(r["rsm_selected_teacher_improvement"]) for r in rows], dtype=np.float64)
    mean = X.mean(axis=0); std = np.maximum(X.std(axis=0), 1.0e-6)
    Z = (X - mean[None, :]) / std[None, :]
    good = Z[y > 0.0]; bad = Z[y <= 0.0]
    if good.shape[0] < 32 or bad.shape[0] < 32:
        raise ValueError("V51 selected-outcome ranker has insufficient sign populations")
    D = (bad[:, None, :] - good[None, :, :]).reshape(-1, Z.shape[1])
    w = np.zeros((Z.shape[1],), dtype=np.float64)
    before = float(D.shape[0] * math.log(2.0))
    for _ in range(80):
        s = D @ w
        q = 1.0 / (1.0 + np.exp(np.clip(s, -60.0, 60.0)))
        grad = -(D.T @ q) + RIDGE_LAMBDA * w
        hw = q * (1.0 - q)
        H = (D.T * hw) @ D + RIDGE_LAMBDA * np.eye(D.shape[1], dtype=np.float64)
        step = np.linalg.solve(H, grad)
        w = w - step
        if float(np.linalg.norm(step)) < 1.0e-9:
            break
    s = D @ w
    after = float(np.sum(np.logaddexp(0.0, -s)) + 0.5 * RIDGE_LAMBDA * np.dot(w, w))
    raw = Z @ w; pos = raw[y > 0.0]
    return {
        "model": "zero_bias_pairwise_selected_sign_risk",
        "feature_names": _arm_names(arm),
        "feature_mean": [float(x) for x in mean], "feature_std": [float(x) for x in std],
        "weights": [float(x) for x in w], "bias": 0.0, "lambda": RIDGE_LAMBDA,
        "arm": arm, "selected_count": len(rows), "positive_count": int(np.sum(y > 0.0)), "nonpositive_count": int(np.sum(y <= 0.0)),
        "pair_count": int(D.shape[0]), "objective_at_zero": before, "objective_final": after,
        "fit_positive_score_mean": float(pos.mean()), "fit_positive_score_std": max(float(pos.std()), 1.0e-6),
    }


def _risk(r: dict[str, Any], m: dict[str, Any]) -> float:
    z = _arm_state(r, str(m["arm"]))
    mean = np.asarray(m["feature_mean"], dtype=np.float64); std = np.asarray(m["feature_std"], dtype=np.float64); w = np.asarray(m["weights"], dtype=np.float64)
    raw = float(((z - mean) / np.maximum(std, 1.0e-6)) @ w)
    return float((raw - float(m["fit_positive_score_mean"])) / max(float(m["fit_positive_score_std"]), 1.0e-6))


def _threshold(cal: list[dict[str, Any]], model: dict[str, Any], alpha: float) -> tuple[float, dict[str, Any]]:
    vals = sorted(_risk(r, model) for r in cal if bool(r["closed_loop_beneficial"]))
    n = len(vals); rank = int(math.ceil((n + 1) * (1.0 - alpha))) if n else 1
    min_n = int(math.ceil((1.0 - alpha) / alpha))
    if n == 0 or rank > n:
        raise ValueError(f"V51 calibration insufficient for frozen alpha: n={n}, min={min_n}, rank={rank}")
    return float(vals[rank-1]), {
        "positive_calibration_count": n, "conformal_rank": rank, "alpha": float(alpha), "threshold": float(vals[rank-1]),
        "finite_sample_condition": "ceil((n+1)*(1-alpha))<=n", "minimum_positive_count_for_finite_rank": min_n,
    }


def _effect_support(r: dict[str, Any]) -> bool:
    if abs(float(r["closed_loop_score_delta"])) > EPS:
        return True
    return any(abs(float(v)) > EPS for v in dict(r.get("safety_delta", {})).values())


def _scalar_contrast_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target_name, target_fn in [
        ("any_closed_loop_effect", _effect_support),
        ("beneficial", lambda r: bool(r["closed_loop_beneficial"])),
        ("hard_harm", lambda r: bool(r["closed_loop_hard_harm"])),
    ]:
        ys = np.asarray([bool(target_fn(r)) for r in rows], dtype=bool)
        ds = np.asarray([float(r["operator_execution_contrast_linf"]) for r in rows], dtype=np.float64)
        folds=[]
        for k in range(FOLDS):
            rr=[r for r in rows if int(r["outer_test_fold"])==k]
            folds.append(_auc(np.asarray([bool(target_fn(r)) for r in rr]), np.asarray([float(r["operator_execution_contrast_linf"]) for r in rr])))
        agg=_auc(ys,ds)
        out[target_name]={"auc":agg,"fold_auc":folds,"folds_above_random":sum(int(math.isfinite(x) and x>0.5+EPS) for x in folds)}
    out["effect_support_identified"] = bool(out["any_closed_loop_effect"]["auc"] > 0.5 + EPS and out["any_closed_loop_effect"]["folds_above_random"] >= 4)
    out["physical_equal_count"] = int(sum(abs(float(r["operator_execution_contrast_linf"])) <= EPS for r in rows))
    out["physical_equal_all_null_effect"] = bool(all(not _effect_support(r) for r in rows if abs(float(r["operator_execution_contrast_linf"])) <= EPS))
    return out


def _nested(rows: list[dict[str, Any]], alpha: float, v50_report: dict[str, Any]) -> dict[str, Any]:
    # Exact V50.6 control replay first; it is the causal control for every V51 arm.
    control = v506._nested(rows, alpha)
    expected = v50_report["nested_crossfit"]
    c_ri=control["risk_identification"]; e_ri=expected["risk_identification"]
    for k in ["aggregate_ego_ref_auc_on_closed_loop_outcome","aggregate_offline_obs_auc_on_closed_loop_outcome","aggregate_pior_auc_on_closed_loop_outcome"]:
        if abs(float(c_ri[k])-float(e_ri[k]))>1e-12:
            raise RuntimeError(f"V51 ENGINEERING STOP: exact V50 QPE control replay drift for {k}")
    if int(c_ri["pior_better_ego_fold_count"])!=int(e_ri["pior_better_ego_fold_count"]) or int(c_ri["pior_better_offline_obs_fold_count"])!=int(e_ri["pior_better_offline_obs_fold_count"]):
        raise RuntimeError("V51 ENGINEERING STOP: V50 QPE control fold signature drift")

    arms: dict[str, Any] = {}
    control_fold_auc=[float(f["risk_identification"]["pior_auc"]) for f in expected["folds"]]
    control_auc=float(e_ri["aggregate_pior_auc_on_closed_loop_outcome"])
    ego_auc=float(e_ri["aggregate_ego_ref_auc_on_closed_loop_outcome"]); obs_auc=float(e_ri["aggregate_offline_obs_auc_on_closed_loop_outcome"])
    ego_fold=[float(f["risk_identification"]["ego_ref_auc"]) for f in expected["folds"]]
    obs_fold=[float(f["risk_identification"]["offline_obs_auc"]) for f in expected["folds"]]

    for arm in ARMS:
        folds=[]; all_target=[]; all_risk=[]; keep_by_token={}
        better_control=better_ego=better_obs=0
        for k in range(FOLDS):
            cf=(k+1)%FOLDS
            fit=[r for r in rows if int(r["outer_test_fold"]) not in {k,cf}]
            cal=[r for r in rows if int(r["outer_test_fold"])==cf]
            test=[r for r in rows if int(r["outer_test_fold"])==k]
            model=_fit_ranker(fit,arm); tau,ci=_threshold(cal,model,alpha)
            target=np.asarray([not bool(r["closed_loop_beneficial"]) for r in test],dtype=bool)
            risk=np.asarray([_risk(r,model) for r in test],dtype=np.float64)
            keep=[bool(x<=tau) for x in risk]
            for r,kk in zip(test,keep):
                tok=str(r["scenario_token"])
                if tok in keep_by_token: raise RuntimeError(f"V51 duplicate OOF keep {tok}")
                keep_by_token[tok]=kk
            a=_auc(target,risk)
            better_control += int(math.isfinite(a) and a > control_fold_auc[k]+EPS)
            better_ego += int(math.isfinite(a) and a > ego_fold[k]+EPS)
            better_obs += int(math.isfinite(a) and a > obs_fold[k]+EPS)
            bm=v50._delta_metrics(test,[True]*len(test)); pm=v50._delta_metrics(test,keep)
            folds.append({"fold":k,"fit_events":len(fit),"calibration_events":len(cal),"test_events":len(test),"rsmr":bm,"pocr":pm,"risk_auc":a,"qpe_control_auc":control_fold_auc[k],"ego_ref_auc":ego_fold[k],"offline_obs_auc":obs_fold[k],"better_qpe_control":a>control_fold_auc[k]+EPS,"better_ego":a>ego_fold[k]+EPS,"better_obs":a>obs_fold[k]+EPS,"calibration":ci})
            all_target.extend(target.tolist()); all_risk.extend(risk.tolist())
        auc=float(_auc(np.asarray(all_target,dtype=bool),np.asarray(all_risk,dtype=np.float64)))
        identified=bool(auc>max(control_auc,ego_auc,obs_auc)+EPS and better_control>=4 and better_ego>=4 and better_obs>=4)
        keep=v506._align_oof_keep_by_token(rows,keep_by_token)
        base=v50._delta_metrics(rows,[True]*len(rows)); chosen=v50._delta_metrics(rows,keep)
        dep=v50._gate(base,chosen,alpha,[{"rsmr":f["rsmr"],"pior":f["pocr"]} for f in folds])
        arms[arm]={"folds":folds,"risk_identification":{"aggregate_auc":auc,"qpe_control_auc":control_auc,"ego_ref_auc":ego_auc,"offline_obs_auc":obs_auc,"better_qpe_control_fold_count":better_control,"better_ego_fold_count":better_ego,"better_obs_fold_count":better_obs,"identified":identified},"aggregate":chosen,"deployment_gate":dep,"pass":bool(identified and dep["pass"])}

    preferred = "qpe_dose" if arms["qpe_dose"]["pass"] else "qpe_dose_interaction" if arms["qpe_dose_interaction"]["pass"] else None
    any_identified=any(bool(arms[a]["risk_identification"]["identified"]) for a in ARMS)
    diag=_scalar_contrast_diagnostics(rows)
    if preferred == "qpe_dose": diagnosis="additive_operator_execution_contrast_state_sufficient"
    elif preferred == "qpe_dose_interaction": diagnosis="operator_execution_contrast_requires_dose_conditioned_consequence_interaction"
    elif any_identified: diagnosis="operator_contrast_state_identified_but_low_capacity_sign_retention_functional_insufficient"
    elif diag["effect_support_identified"]: diagnosis="scalar_operator_contrast_identifies_effect_support_but_not_selected_outcome_sign"
    else: diagnosis="scalar_operator_contrast_state_not_identified"
    return {"retention_alpha":alpha,"qpe_control":control,"operator_contrast_diagnostic":diag,"arms":arms,"preferred_promotion_arm":preferred,"train_gate_pass":preferred is not None,"failure_diagnosis":diagnosis}


def _decorate(base_cfg: Path, model: dict[str, Any], tau: float, arm: str, out: Path) -> None:
    cfg=yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    ic=cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    sc=ic["selection_conditioned_intervention_recovery"]
    sc["post_selection_value_mode"]="endpoint_potential_quality_paired_operator_contrast_retention"
    sc["paired_operator_contrast_retention"]={
        "feature_names":_arm_names(arm),"aggregation":"sign_only","include_dose_interactions":arm=="qpe_dose_interaction",
        "components":{"sign_risk":model},"retention_threshold":float(tau),
        "operator_contrast":"max_abs_frozen_RSMR_proposal_vs_runtime_incumbent_bounded_trajectory_tensor",
        "outcome_supervision":"TRAIN_only_metric_safe_paired_one_shot_closed_loop_outcomes",
        "runtime_candidate_set":"full_frozen_deployment_candidate_set",
    }
    sc["post_selection_selected_bias"]=0.0
    sc["post_selection_value_training"]="paired_selected_outcome_pairwise_sign_risk_fixed_lambda_1_with_operator_execution_contrast"
    sc["post_selection_operator"]="freeze_full_set_RSMR_winner_then_POCR_veto_only_same_winner_or_incumbent_no_rerank_no_fallback"
    cfg.setdefault("metadata",{})["algorithm_version"]="V64.3.51-EAF-ICER-POCR"
    cfg.setdefault("provenance",{})["algorithm_version"]="V64.3.51-EAF-ICER-POCR"
    cfg.setdefault("experiment",{})["name"]="v64_3_51_eaf_icer_pocr"
    cfg["experiment"]["algorithm"]="V64.3.51-EAF-ICER-POCR"
    out.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser(description="Fit V64.3.51 POCR on provenance-locked V50 paired outcomes.")
    ap.add_argument("--v49-fit-report",type=Path,required=True); ap.add_argument("--v49-candidate-audit",type=Path,required=True); ap.add_argument("--v49-scene-audit",type=Path,required=True); ap.add_argument("--v49-siir-config",type=Path,required=True)
    ap.add_argument("--v50-fit-report",type=Path,required=True); ap.add_argument("--paired-outcomes",type=Path,required=True); ap.add_argument("--v50-5-root",type=Path,required=True)
    ap.add_argument("--output-config",type=Path,required=True); ap.add_argument("--output-report",type=Path,required=True)
    a=ap.parse_args()
    v50_report=_check_v50(a.v50_fit_report)
    v49=v50._check_v49(a.v49_fit_report); alpha=float(v49["nested_crossfit"]["retention_alpha"])
    states=v50._candidate_states(a.v49_candidate_audit); obs=v50._obs_risk(a.v49_scene_audit); rows=v50._join(states,a.paired_outcomes,obs)
    contrast=_load_treatment_contrast(a.v50_5_root,set(states))
    for r in rows: r["operator_execution_contrast_linf"]=float(contrast[str(r["scenario_token"])])
    nested=_nested(rows,alpha,v50_report)
    preferred=nested["preferred_promotion_arm"]
    final=None
    if preferred is not None:
        fit=[r for r in rows if int(r["outer_test_fold"])!=0]; cal=[r for r in rows if int(r["outer_test_fold"])==0]
        model=_fit_ranker(fit,preferred); tau,ci=_threshold(cal,model,alpha); final={"arm":preferred,"model":model,"calibration":ci}
    report={
        "audit":"v64_3_51_eaf_icer_pocr_fit","algorithm_version":"V64.3.51-EAF-ICER-POCR",
        "scientific_role":"preregistered_V50_identification_failure_branch_on_policy_structured_selected_outcome_state",
        "mechanism_hypothesis":"Paired outcome labels alone are insufficient because Q/P/E compress proposal consequence but omit the physical treatment-control execution contrast. Condition selected-outcome risk on the exact frozen-proposal-vs-incumbent trajectory contrast; test additive dose first, then a fixed dose-by-QPE interaction state.",
        "frozen_contract":{"RSMR_selector_unchanged":True,"Q_P_E_coordinates_preserved":True,"paired_outcome_labels_unchanged":True,"pairwise_sign_risk_family_unchanged":True,"lambda":1.0,"retention_alpha_and_conformal_rule_unchanged":True,"same_winner_or_incumbent_only":True,"no_rerank_second_best_fallback":True,"no_K_multiplicity":True,"no_new_offline_future_observable":True},
        "nested_crossfit":nested,"final_runtime_fit":final,"train_gate_pass":bool(nested["train_gate_pass"]),
        "preregistered_branch_order":["qpe_dose","qpe_dose_interaction"],
        "preregistered_next_branch":{
            "if_qpe_dose_pass":"promote minimal additive operator-contrast state; freeze and run untouched paired validation",
            "if_only_interaction_pass":"promote fixed dose-conditioned consequence interaction; freeze and run untouched paired validation",
            "if_state_identified_but_deployment_fail":"V52 structured paired closed-loop outcome functional; keep paired evidence and operator contrast, do not add offline features",
            "if_effect_support_identified_but_sign_identification_fail":"close scalar-dose-as-sufficient-sign-state; next test richer temporal treatment-control contrast profile under paired evidence",
            "if_effect_support_not_identified":"close scalar operator-contrast family and reassess selected-outcome measurement/state acquisition",
        },
        "prohibited_tuning":["QPE-only rescue","pairwise loss/lambda/class/focal/catastrophe weighting","retention threshold/calibration sweep","bigger MLP","K/logK/K interaction","new offline future observable","V46 variance/temporal profile resurrection","V47 AGENT-2D resurrection","CVaR","selected translation","binary catastrophe veto","RSMR/B/M/topK/candidate-count change","rerank/second-best/fallback","A/B pooling"],
    }
    a.output_report.parent.mkdir(parents=True,exist_ok=True); a.output_report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not nested["train_gate_pass"]:
        if a.output_config.exists(): a.output_config.unlink()
        print(json.dumps({"pass":False,"failure_diagnosis":nested["failure_diagnosis"],"output_config_emitted":False},sort_keys=True))
        raise SystemExit(f"V64.3.51 POCR TRAIN scientific STOP ({nested['failure_diagnosis']}); do not consume untouched validation")
    assert final is not None
    _decorate(a.v49_siir_config,final["model"],float(final["calibration"]["threshold"]),str(final["arm"]),a.output_config)
    print(json.dumps({"pass":True,"preferred_promotion_arm":preferred,"output_config":str(a.output_config)},sort_keys=True))


if __name__=="__main__": main()
