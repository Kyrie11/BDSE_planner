from __future__ import annotations

"""Fit V64.3.55 DMOR: Dynamic-Mediator Outcome Retention.

Preregistered order:
  A. REALIZED-DOMINANCE: keep V54's identified realized endpoint mediator and
     change only the conditional outcome functional from binary sign to
     unweighted Pareto dominance over the existing paired score/safety vector.
     This is a diagnostic oracle because the mediator is post-intervention.
  B. PREDICTED-DOMINANCE: only eligible if A passes. Distill the realized
     endpoint from the already-fixed pre-execution V53 planned operator profile
     with a zero-preserving ridge nuisance model, then apply the *same* Pareto
     functional. This is the first t0-available candidate mechanism.

No new outcome labels, no horizon/basis/threshold/loss tuning, no safety
scalarization, no larger MLP, and no change to RSMR/no-fallback containment.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bdse.planner.paired_dynamic_mediator_outcome_retention import (
    DMOR_PREDICTED_STATE_NAMES,
    DMOR_REALIZED_STATE_NAMES,
    fit_zero_preserving_mediator_ridge,
    outcome_state,
    planned_mediator_input,
    predict_realized_endpoint,
)
from bdse.tools import fit_v64_3_50_eaf_icer_pior as v50
from bdse.tools import fit_v64_3_50_6_eaf_icer_pior as v506
from bdse.tools import fit_v64_3_51_eaf_icer_pocr as v51
from bdse.tools import fit_v64_3_52_eaf_icer_hodr as v52
from bdse.tools import fit_v64_3_53_eaf_icer_potr as v53
from bdse.tools import fit_v64_3_54_eaf_icer_pdrm as v54

FOLDS = 5
EPS = 1.0e-12
RIDGE_LAMBDA = 1.0
ARMS = ("realized_dominance", "predicted_dominance")
EXPECTED_V54_FIT_SHA256 = "10f3e60c82bb8b82f1f688e866a27008e1498b67d2e194b0c7aadec5368536d8"
EXPECTED_V54_DYNAMIC_SHA256 = "dd2bdd809a757ce74973d7ce2c3189fad60dc0d3e0125d7fcec9ca7ad1bda373"
EXPECTED_V53_PROFILE_SHA256 = "9a69c196a1d76e9c5d424068df223ec26f0e481252f25f67d5bb17fd355aaef6"
EXPECTED_V54_ENDPOINT_AUC = 0.6117518844791572
EXPECTED_V52_PARETO_CONCORDANCE = 0.49772853185595567


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_v54(path: Path) -> dict[str, Any]:
    got = _sha256(path)
    if got != EXPECTED_V54_FIT_SHA256:
        raise RuntimeError(f"V55 ENGINEERING STOP: V54 fit hash drift {got}")
    d = json.loads(path.read_text(encoding="utf-8"))
    n = d.get("nested_crossfit", {})
    ep = n.get("arms", {}).get("realized_endpoint", {})
    tp = n.get("arms", {}).get("realized_temporal", {})
    if d.get("mediator_identification_pass") is not True or n.get("preferred_mediator_arm") != "realized_endpoint":
        raise RuntimeError("V55 ENGINEERING STOP: V54 realized-endpoint mediator signature drift")
    if ep.get("identification", {}).get("identified") is not True:
        raise RuntimeError("V55 ENGINEERING STOP: V54 endpoint must be identified")
    if abs(float(ep["identification"]["aggregate_auc"]) - EXPECTED_V54_ENDPOINT_AUC) > 1e-12:
        raise RuntimeError("V55 ENGINEERING STOP: V54 endpoint AUC drift")
    if ep.get("retrospective_oracle_gate", {}).get("pass") is not False:
        raise RuntimeError("V55 ENGINEERING STOP: V54 static sign oracle must be deployment STOP")
    if tp.get("identification", {}).get("temporal_necessity_identified") is not False:
        raise RuntimeError("V55 ENGINEERING STOP: V54 temporal necessity signature drift")
    return d


def _load_rows(
    *, v49_candidate_audit: Path, v49_scene_audit: Path, paired_outcomes: Path,
    v50_5_root: Path, v53_profiles: Path, v54_dynamic: Path,
) -> list[dict[str, Any]]:
    states = v50._candidate_states(v49_candidate_audit)
    obs = v50._obs_risk(v49_scene_audit)
    rows = v50._join(states, paired_outcomes, obs)
    expected = set(states)
    scalar = v51._load_treatment_contrast(v50_5_root, expected)
    planned = v53._load_profiles(v53_profiles, expected)
    dynamic = v54._load_dynamic(v54_dynamic, expected)
    for r in rows:
        tok = str(r["scenario_token"])
        d = float(scalar[tok])
        if abs(float(planned[tok]["execution_contrast_linf"]) - d) > 1e-9:
            raise RuntimeError(f"V55 ENGINEERING STOP: V53 planned D drift token={tok}")
        if abs(float(dynamic[tok]["planned_execution_contrast_linf"]) - d) > 1e-9:
            raise RuntimeError(f"V55 ENGINEERING STOP: V54 planned D drift token={tok}")
        r["operator_execution_contrast_linf"] = d
        r["operator_trajectory_profile"] = planned[tok]
        r["dynamic_response_profile"] = dynamic[tok]
    return rows


def _realized_mediator(r: dict[str, Any]) -> np.ndarray:
    x = np.asarray(r["dynamic_response_profile"]["endpoint_signed"], dtype=np.float64).reshape(-1)
    if x.size != 4 or np.any(~np.isfinite(x)):
        raise ValueError("V55 invalid realized mediator")
    return x


def _planned_input(r: dict[str, Any]) -> np.ndarray:
    return planned_mediator_input(dict(r["operator_trajectory_profile"]))


def _fit_mediator(rows: list[dict[str, Any]]) -> dict[str, Any]:
    X = np.stack([_planned_input(r) for r in rows])
    Y = np.stack([_realized_mediator(r) for r in rows])
    return fit_zero_preserving_mediator_ridge(X, Y, ridge_lambda=1.0)


def _pred_map(rows: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, np.ndarray]:
    return {str(r["scenario_token"]): predict_realized_endpoint(_planned_input(r), model) for r in rows}


def _inner_oof_pred_map(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    folds = sorted({int(r["outer_test_fold"]) for r in rows})
    out: dict[str, np.ndarray] = {}
    for f in folds:
        tr = [r for r in rows if int(r["outer_test_fold"]) != f]
        te = [r for r in rows if int(r["outer_test_fold"]) == f]
        m = _fit_mediator(tr)
        for r in te:
            tok = str(r["scenario_token"])
            if tok in out:
                raise RuntimeError(f"V55 duplicate inner OOF mediator token={tok}")
            out[tok] = predict_realized_endpoint(_planned_input(r), m)
    if len(out) != len(rows):
        raise RuntimeError("V55 inner OOF mediator coverage mismatch")
    return out


def _state(r: dict[str, Any], *, predicted: bool, pred: dict[str, np.ndarray] | None = None) -> np.ndarray:
    tok = str(r["scenario_token"])
    med = pred[tok] if predicted else _realized_mediator(r)
    return outcome_state(
        float(r["quality_value"]), float(r["plan_control_value"]), float(r["ego_ref_value"]),
        float(r["operator_execution_contrast_linf"]), med, predicted=predicted,
    )


def _fit_outcome(rows: list[dict[str, Any]], safety_names: list[str], *, predicted: bool, pred: dict[str, np.ndarray] | None) -> dict[str, Any]:
    pairs = v52._pareto_pairs(rows, safety_names)
    if len(rows) < 64 or len(pairs) < 64:
        raise ValueError(f"V55 insufficient Pareto support rows={len(rows)} pairs={len(pairs)}")
    X = np.stack([_state(r, predicted=predicted, pred=pred) for r in rows])
    mean = X.mean(axis=0); std = np.maximum(X.std(axis=0), 1e-6); Z = (X - mean) / std
    D = np.stack([Z[i] - Z[j] for i, j in pairs])
    w = np.zeros(Z.shape[1], dtype=np.float64)
    before = float(D.shape[0] * math.log(2.0))
    for _ in range(80):
        s = D @ w; q = 1.0 / (1.0 + np.exp(np.clip(s, -60.0, 60.0)))
        grad = -(D.T @ q) + RIDGE_LAMBDA * w
        hw = q * (1.0 - q); H = (D.T * hw) @ D + RIDGE_LAMBDA * np.eye(D.shape[1], dtype=np.float64)
        step = np.linalg.solve(H, grad); w -= step
        if float(np.linalg.norm(step)) < 1e-9: break
    raw = Z @ w
    ben = raw[np.asarray([bool(r["closed_loop_beneficial"]) for r in rows], dtype=bool)]
    if ben.size < 24: raise ValueError("V55 insufficient beneficial normalization")
    after = float(np.sum(np.logaddexp(0.0, -D @ w)) + 0.5 * np.dot(w, w))
    names = DMOR_PREDICTED_STATE_NAMES if predicted else DMOR_REALIZED_STATE_NAMES
    return {
        "model": "zero_bias_pairwise_dynamic_mediator_pareto_risk",
        "state_family": "predicted_mediator" if predicted else "realized_mediator",
        "feature_names": list(names), "feature_mean": [float(x) for x in mean], "feature_std": [float(x) for x in std],
        "weights": [float(x) for x in w], "bias": 0.0, "lambda": 1.0,
        "fit_row_count": len(rows), "pareto_pair_count": len(pairs), "objective_at_zero": before, "objective_final": after,
        "fit_beneficial_score_mean": float(ben.mean()), "fit_beneficial_score_std": max(float(ben.std()), 1e-6),
    }


def _outcome_risk(r: dict[str, Any], model: dict[str, Any], *, predicted: bool, pred: dict[str, np.ndarray] | None) -> float:
    z = _state(r, predicted=predicted, pred=pred)
    mean = np.asarray(model["feature_mean"], dtype=np.float64); std = np.asarray(model["feature_std"], dtype=np.float64); w = np.asarray(model["weights"], dtype=np.float64)
    raw = float(((z - mean) / np.maximum(std, 1e-6)) @ w)
    return float((raw - float(model["fit_beneficial_score_mean"])) / max(float(model["fit_beneficial_score_std"]), 1e-6))


def _threshold(cal: list[dict[str, Any]], support_model: dict[str, Any], outcome_model: dict[str, Any], alpha: float, *, predicted: bool, pred: dict[str, np.ndarray] | None) -> tuple[float, dict[str, Any]]:
    vals = sorted(max(v52._component_risk(r, support_model), _outcome_risk(r, outcome_model, predicted=predicted, pred=pred)) for r in cal if bool(r["closed_loop_beneficial"]))
    n = len(vals); rank = int(math.ceil((n + 1) * (1.0 - alpha))) if n else 1
    if n == 0 or rank > n: raise ValueError(f"V55 calibration insufficient n={n} rank={rank}")
    return float(vals[rank - 1]), {"positive_calibration_count": n, "conformal_rank": rank, "alpha": alpha, "threshold": float(vals[rank-1]), "single_joint_threshold": True}


def _mediator_test_metric(test: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    Y = np.stack([_realized_mediator(r) for r in test])
    P = np.stack([predict_realized_endpoint(_planned_input(r), model) for r in test])
    scale = np.asarray(model["output_rms"], dtype=np.float64)
    err = ((P - Y) / scale[None, :]) ** 2
    zero = (Y / scale[None, :]) ** 2
    return {
        "normalized_mse": float(np.mean(err)), "zero_baseline_normalized_mse": float(np.mean(zero)),
        "better_zero": bool(float(np.mean(err)) < float(np.mean(zero)) - EPS),
        "component_normalized_mse": [float(v) for v in np.mean(err, axis=0)],
        "component_zero_normalized_mse": [float(v) for v in np.mean(zero, axis=0)],
        "normalized_sse": float(np.sum(err)), "zero_normalized_sse": float(np.sum(zero)), "elements": int(err.size),
    }


def _evaluate_arm(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    alpha: float,
    safety: list[str],
    control: dict[str, Any],
    cagg: float,
) -> dict[str, Any]:
    predicted = arm == "predicted_dominance"
    folds: list[dict[str, Any]] = []
    keep_by_token: dict[str, bool] = {}
    all_dom: list[float] = []
    above = better_ctrl = 0
    pred_sse = pred_zero_sse = 0.0
    pred_elements = 0
    pred_better_folds = 0
    control_by_fold = {int(f["fold"]): f for f in control["folds"]}
    if set(control_by_fold) != set(range(FOLDS)):
        raise RuntimeError("V55 ENGINEERING STOP: V52 Pareto fold control coverage drift")

    for k in range(FOLDS):
        cf = (k + 1) % FOLDS
        fit = [r for r in rows if int(r["outer_test_fold"]) not in {k, cf}]
        cal = [r for r in rows if int(r["outer_test_fold"]) == cf]
        test = [r for r in rows if int(r["outer_test_fold"]) == k]
        support = v52._fit_models(fit, "hurdle_sign", safety)["effect_support_risk"]
        fit_pred = cal_pred = test_pred = None
        pred_diag = None
        if predicted:
            # The outcome model sees nuisance predictions that are out-of-fold even
            # inside the outer fitting population. Calibration/test predictions use
            # a nuisance fit that excludes both outer test and calibration folds.
            fit_pred = _inner_oof_pred_map(fit)
            med_model = _fit_mediator(fit)
            cal_pred = _pred_map(cal, med_model)
            test_pred = _pred_map(test, med_model)
            pred_diag = _mediator_test_metric(test, med_model)
            pred_sse += float(pred_diag["normalized_sse"])
            pred_zero_sse += float(pred_diag["zero_normalized_sse"])
            pred_elements += int(pred_diag["elements"])
            pred_better_folds += int(bool(pred_diag["better_zero"]))

        outcome = _fit_outcome(fit, safety, predicted=predicted, pred=fit_pred)
        tau, ci = _threshold(cal, support, outcome, alpha, predicted=predicted, pred=cal_pred)
        sr = np.asarray([v52._component_risk(r, support) for r in test], dtype=np.float64)
        rr = np.asarray([_outcome_risk(r, outcome, predicted=predicted, pred=test_pred) for r in test], dtype=np.float64)
        joint = np.maximum(sr, rr)
        keep = [bool(x <= tau) for x in joint]
        for r, kk in zip(test, keep):
            tok = str(r["scenario_token"])
            if tok in keep_by_token:
                raise RuntimeError(f"V55 duplicate OOF keep token={tok}")
            keep_by_token[tok] = kk

        dpairs = v52._pareto_pairs(test, safety)
        dom = v52._concordance(test, rr, dpairs)
        cdom = float(control_by_fold[k]["pareto_concordance"])
        ar = bool(math.isfinite(dom) and dom > 0.5 + EPS)
        bc = bool(math.isfinite(dom) and math.isfinite(cdom) and dom > cdom + EPS)
        above += int(ar)
        better_ctrl += int(bc)
        for bad, good in dpairs:
            d = float(rr[bad] - rr[good])
            all_dom.append(1.0 if d > EPS else 0.5 if abs(d) <= EPS else 0.0)

        bm = v50._delta_metrics(test, [True] * len(test))
        dm = v50._delta_metrics(test, keep)
        folds.append({
            "fold": k,
            "fit_events": len(fit),
            "calibration_events": len(cal),
            "test_events": len(test),
            "pareto_pair_count": len(dpairs),
            "pareto_concordance": dom,
            "v52_static_pareto_concordance": cdom,
            "above_random": ar,
            "better_v52_static_pareto": bc,
            "mediator_prediction": pred_diag,
            "rsmr": bm,
            "dmor": dm,
            "calibration": ci,
        })

    dom = float(np.mean(all_dom)) if all_dom else float("nan")
    functional_identified = bool(
        math.isfinite(dom)
        and dom > 0.5 + EPS
        and dom > cagg + EPS
        and above >= 4
        and better_ctrl >= 4
    )
    pred_agg = None
    if predicted:
        mse = pred_sse / max(pred_elements, 1)
        zmse = pred_zero_sse / max(pred_elements, 1)
        predictor_identified = bool(mse < zmse - EPS and pred_better_folds >= 4)
        pred_agg = {
            "normalized_mse": mse,
            "zero_baseline_normalized_mse": zmse,
            "better_zero_fold_count": pred_better_folds,
            "identified": predictor_identified,
        }
    keep = v506._align_oof_keep_by_token(rows, keep_by_token)
    base = v50._delta_metrics(rows, [True] * len(rows))
    chosen = v50._delta_metrics(rows, keep)
    dep = v50._gate(base, chosen, alpha, [{"rsmr": f["rsmr"], "pior": f["dmor"]} for f in folds])
    return {
        "folds": folds,
        "identification": {
            "pareto_concordance": dom,
            "v52_static_pareto_concordance": cagg,
            "folds_above_random": above,
            "better_v52_static_pareto_fold_count": better_ctrl,
            "functional_identified": functional_identified,
            "mediator_prediction": pred_agg,
        },
        "aggregate": chosen,
        "deployment_gate": dep,
        "pass": False,
    }


def _nested(rows: list[dict[str, Any]], alpha: float, v52_report: dict[str, Any], v54_report: dict[str, Any]) -> dict[str, Any]:
    safety = v52._safety_names(rows)
    control = v52_report["nested_crossfit"]["arms"]["hurdle_pareto"]
    cagg = float(control["identification"]["pareto_concordance"])
    if abs(cagg - EXPECTED_V52_PARETO_CONCORDANCE) > 1e-12:
        raise RuntimeError(f"V55 ENGINEERING STOP: V52 Pareto control drift {cagg}")

    # Preregistered sequential fail-closed evaluation.  Arm B is not even fit
    # unless Arm A both identifies the structured functional and closes the
    # unchanged deployment gate.  This prevents post-hoc TRAIN peeking into the
    # deployable bridge after a failed diagnostic oracle.
    oracle = _evaluate_arm(
        rows, arm="realized_dominance", alpha=alpha, safety=safety,
        control=control, cagg=cagg,
    )
    oracle_pass = bool(oracle["identification"]["functional_identified"] and oracle["deployment_gate"]["pass"])
    oracle["pass"] = oracle_pass
    arms: dict[str, Any] = {"realized_dominance": oracle}

    if not oracle_pass:
        arms["predicted_dominance"] = {
            "status": "NOT_EVALUATED_BY_PREREGISTERED_BRANCH_ORDER",
            "eligible_by_branch_order": False,
            "pass": False,
        }
        return {
            "retention_alpha": alpha,
            "safety_delta_names": safety,
            "v54_realized_endpoint_control": {"auc": EXPECTED_V54_ENDPOINT_AUC, "identified": True},
            "v52_effect_support_control": {
                "auc": float(v52_report["nested_crossfit"]["arms"]["hurdle_sign"]["identification"]["support_auc"]),
                "frozen": True,
            },
            "arms": arms,
            "diagnostic_oracle_pass": False,
            "deployable_train_gate_pass": False,
            "failure_diagnosis": "realized_mediator_plus_static_pareto_functional_still_deployment_insufficient",
        }

    pred = _evaluate_arm(
        rows, arm="predicted_dominance", alpha=alpha, safety=safety,
        control=control, cagg=cagg,
    )
    pred["eligible_by_branch_order"] = True
    pdiag = pred["identification"].get("mediator_prediction")
    pred_ident = bool(pdiag and pdiag.get("identified") and pred["identification"]["functional_identified"])
    pred_pass = bool(pred_ident and pred["deployment_gate"]["pass"])
    pred["pass"] = pred_pass
    arms["predicted_dominance"] = pred

    if pred_pass:
        diagnosis = "deployable_predicted_dynamic_mediator_pareto_retention_closes_train_gate"
    elif not bool(pdiag and pdiag.get("identified")):
        diagnosis = "realized_structured_oracle_sufficient_but_preexecution_plan_does_not_predict_mediator"
    elif not bool(pred["identification"]["functional_identified"]):
        diagnosis = "predicted_mediator_does_not_preserve_structured_outcome_order"
    else:
        diagnosis = "predicted_mediator_structured_retention_identified_but_deployment_insufficient"
    return {
        "retention_alpha": alpha,
        "safety_delta_names": safety,
        "v54_realized_endpoint_control": {"auc": EXPECTED_V54_ENDPOINT_AUC, "identified": True},
        "v52_effect_support_control": {
            "auc": float(v52_report["nested_crossfit"]["arms"]["hurdle_sign"]["identification"]["support_auc"]),
            "frozen": True,
        },
        "arms": arms,
        "diagnostic_oracle_pass": True,
        "deployable_train_gate_pass": pred_pass,
        "failure_diagnosis": diagnosis,
    }


def _final_runtime(rows: list[dict[str, Any]], alpha: float, safety: list[str]) -> dict[str, Any]:
    # Fold 0 remains the frozen final calibration block, matching historical convention.
    fit=[r for r in rows if int(r["outer_test_fold"])!=0]; cal=[r for r in rows if int(r["outer_test_fold"])==0]
    med=_fit_mediator(fit); fit_pred=_inner_oof_pred_map(fit); cal_pred=_pred_map(cal,med)
    support=v52._fit_models(fit,"hurdle_sign",safety)["effect_support_risk"]
    outcome=_fit_outcome(fit,safety,predicted=True,pred=fit_pred)
    tau,ci=_threshold(cal,support,outcome,alpha,predicted=True,pred=cal_pred)
    return {"state_family":"predicted_realized_endpoint","mediator_predictor":med,"effect_support_risk":support,"conditional_pareto_risk":outcome,"calibration":ci,
        "runtime_inputs":"QPE+D plus V53 fixed pre-execution planned endpoint+DCT operator profile; no post-intervention state at runtime",
        "operator":"same frozen full-set RSMR winner or incumbent; veto-only; no rerank/second-best/fallback"}


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--v49-candidate-audit",type=Path,required=True); p.add_argument("--v49-scene-audit",type=Path,required=True)
    p.add_argument("--paired-outcomes",type=Path,required=True); p.add_argument("--v50-5-root",type=Path,required=True)
    p.add_argument("--v52-fit-report",type=Path,required=True); p.add_argument("--v53-operator-profiles",type=Path,required=True)
    p.add_argument("--v54-fit-report",type=Path,required=True); p.add_argument("--v54-dynamic-profiles",type=Path,required=True)
    p.add_argument("--output-report",type=Path,required=True); p.add_argument("--output-runtime-artifact",type=Path,required=True)
    a=p.parse_args()
    v54_report=_check_v54(a.v54_fit_report); v52_report=v53._check_v52(a.v52_fit_report)
    if _sha256(a.v53_operator_profiles) != EXPECTED_V53_PROFILE_SHA256:
        raise RuntimeError(f"V55 ENGINEERING STOP: V53 operator-profile hash drift {_sha256(a.v53_operator_profiles)}")
    if _sha256(a.v54_dynamic_profiles)!=EXPECTED_V54_DYNAMIC_SHA256:
        raise RuntimeError(f"V55 ENGINEERING STOP: V54 dynamic profile hash drift {_sha256(a.v54_dynamic_profiles)}")
    rows=_load_rows(v49_candidate_audit=a.v49_candidate_audit,v49_scene_audit=a.v49_scene_audit,paired_outcomes=a.paired_outcomes,
        v50_5_root=a.v50_5_root,v53_profiles=a.v53_operator_profiles,v54_dynamic=a.v54_dynamic_profiles)
    alpha=float(v52_report["nested_crossfit"]["retention_alpha"]); nested=_nested(rows,alpha,v52_report,v54_report)
    final=None
    if nested["deployable_train_gate_pass"]:
        final=_final_runtime(rows,alpha,list(nested["safety_delta_names"]))
        a.output_runtime_artifact.parent.mkdir(parents=True,exist_ok=True); a.output_runtime_artifact.write_text(json.dumps(final,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    elif a.output_runtime_artifact.exists():
        a.output_runtime_artifact.unlink()
    report={"audit":"v64_3_55_eaf_icer_dmor_fit","algorithm_version":"V64.3.55-EAF-ICER-DMOR",
        "scientific_role":"V54_endpoint_mediator_identified_but_sign_oracle_fold_nonharm_failed_branch_structured_dynamic_mediator_then_deployable_distillation",
        "mechanism_hypothesis":"A realized one-replan endpoint is a genuine mediator, but binary sign ranking loses deployment severity/order. First test an unweighted paired outcome dominance functional on the true mediator; only if sufficient, distill that mediator from fixed pre-execution planned operator geometry and test the identical functional as a t0-available mechanism.",
        "frozen_contract":{"RSMR_selector_unchanged":True,"V52_effect_support_unchanged":True,"V50_5_paired_outcomes_unchanged":True,"V54_realized_endpoint_mediator_frozen":True,
            "V53_planned_profile_basis_unchanged":True,"lambda":1.0,"retention_alpha":alpha,"no_safety_scalarization":True,"no_rerank_second_best_fallback":True},
        "nested_crossfit":nested,"final_runtime_fit":final,"train_gate_pass":bool(nested["deployable_train_gate_pass"]),
        "internal_convergence_candidate":bool(nested["deployable_train_gate_pass"]),
        "preregistered_branch_order":["realized_dominance","predicted_dominance"],
        "next_branch":{"if_deployable_pass":"freeze immediately; engineering-only runtime integration then untouched paired validation; no more TRAIN algorithm tuning",
            "if_oracle_pass_predicted_fail":"mediator prediction is the sole remaining static bottleneck; allow one predeclared runtime dynamics nuisance model, no outcome-state/functional changes",
            "if_oracle_fail":"close static one-replan ego mediator plus Pareto functional as deployment-sufficient; next/last internal family is realized interaction/safety consequence process, not more ego geometry",
            "if_predicted_identified_deployment_fail":"static predicted-mediator state+Pareto functional exhausted; do not add state or tune thresholds; move to temporal/constraint process only if oracle evidence supports it"},
        "prohibited_tuning":["using true post-intervention mediator at t0","mediator horizon/basis/mode sweep","safety weights/scalarization","threshold/alpha/lambda/loss/class/focal/catastrophe weighting",
            "bigger MLP/attention","new offline future observables","RSMR/B/M/topK/candidate-count change","rerank/second-best/fallback","A/B pooling","untouched validation before deployable TRAIN pass"]}
    a.output_report.parent.mkdir(parents=True,exist_ok=True); a.output_report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if nested["deployable_train_gate_pass"]:
        print(json.dumps({"pass":True,"branch":"predicted_dominance","next":"freeze; runtime integration + untouched paired validation; no TRAIN tuning"},sort_keys=True)); return
    print(json.dumps({"pass":False,"diagnostic_oracle_pass":nested["diagnostic_oracle_pass"],"failure_diagnosis":nested["failure_diagnosis"],"runtime_artifact_emitted":False},sort_keys=True))
    raise SystemExit(f"V64.3.55 DMOR scientific STOP ({nested['failure_diagnosis']})")


if __name__=="__main__": main()
