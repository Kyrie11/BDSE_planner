from __future__ import annotations

"""V64.3.53 EAF-ICER-POTR nested TRAIN state experiment.

V52 cleanly identified the scalar QPE+D effect-support hurdle, but neither the
conditional sign nor Pareto order was stable once structural nulls were
removed.  V53 therefore freezes the V52 support model and changes only the
conditional-outcome state using pre-execution proposal-vs-incumbent trajectory
contrast collected at the exact paired intervention event.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from bdse.planner.paired_operator_trajectory_retention import (
    PROFILE_SCHEMA,
    POTR_ENDPOINT_STATE_NAMES,
    POTR_TEMPORAL_STATE_NAMES,
    outcome_state_from_profile,
)
from bdse.tools import fit_v64_3_50_eaf_icer_pior as v50
from bdse.tools import fit_v64_3_50_6_eaf_icer_pior as v506
from bdse.tools import fit_v64_3_51_eaf_icer_pocr as v51
from bdse.tools import fit_v64_3_52_eaf_icer_hodr as v52
from bdse.tools.fit_v64_3_49_eaf_icer_siir import _auc

EPS = 1.0e-12
FOLDS = 5
RIDGE_LAMBDA = 1.0
ARMS = ("endpoint", "temporal")
EXPECTED_V52_FIT_SHA256 = "7a21fead5383ebd6aafaeb5da586346a77e5707050cb359511a06971b742a16b"
EXPECTED_V52_FAILURE = "effect_support_identified_but_operator_state_does_not_identify_conditional_outcome_order"
EXPECTED_V52_SUPPORT_AUC = 0.6516244589283049
EXPECTED_V52_SUPPORT_FOLDS = 5


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_v52(path: Path) -> dict[str, Any]:
    if _sha256(path) != EXPECTED_V52_FIT_SHA256:
        raise RuntimeError(f"V53 ENGINEERING STOP: V52 fit JSON SHA drift {_sha256(path)}")
    d = json.loads(path.read_text(encoding="utf-8"))
    n = d.get("nested_crossfit", {})
    if d.get("train_gate_pass") is not False or n.get("train_gate_pass") is not False:
        raise RuntimeError("V53 ENGINEERING STOP: V52 must be the preregistered TRAIN STOP")
    if n.get("failure_diagnosis") != EXPECTED_V52_FAILURE or n.get("preferred_promotion_arm") is not None:
        raise RuntimeError(f"V53 ENGINEERING STOP: V52 branch drift {n.get('failure_diagnosis')}")
    for arm in ("hurdle_sign", "hurdle_pareto"):
        ri = n.get("arms", {}).get(arm, {}).get("identification", {})
        if ri.get("support_identified") is not True or ri.get("functional_identified") is not False:
            raise RuntimeError(f"V53 ENGINEERING STOP: V52 {arm} identification signature drift")
        if abs(float(ri.get("support_auc", float("nan"))) - EXPECTED_V52_SUPPORT_AUC) > 1e-12:
            raise RuntimeError(f"V53 ENGINEERING STOP: V52 {arm} support AUC drift")
        if int(ri.get("support_folds_above_random", -1)) != EXPECTED_V52_SUPPORT_FOLDS:
            raise RuntimeError(f"V53 ENGINEERING STOP: V52 {arm} support fold signature drift")
    return d


def _load_profiles(path: Path, expected_tokens: set[str]) -> dict[str, dict[str, Any]]:
    by: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tok = str(r.get("scenario_token", ""))
        if tok in by:
            raise RuntimeError(f"V53 ENGINEERING STOP: duplicate operator profile {tok}")
        if str(r.get("schema", "")) != PROFILE_SCHEMA:
            raise RuntimeError(f"V53 ENGINEERING STOP: profile schema mismatch {tok}")
        if len(r.get("endpoint_signed", [])) != 4 or len(r.get("cosine_modes_1_2", [])) != 8:
            raise RuntimeError(f"V53 ENGINEERING STOP: profile feature size mismatch {tok}")
        by[tok] = r
    if set(by) != expected_tokens:
        raise RuntimeError(f"V53 ENGINEERING STOP: profile token identity mismatch {len(by)}/{len(expected_tokens)}")
    return by


def _state(r: dict[str, Any], arm: str) -> np.ndarray:
    return outcome_state_from_profile(
        float(r["quality_value"]), float(r["plan_control_value"]), float(r["ego_ref_value"]),
        dict(r["operator_trajectory_profile"]), state_family=arm,
    )


def _names(arm: str) -> list[str]:
    if arm == "endpoint": return list(POTR_ENDPOINT_STATE_NAMES)
    if arm == "temporal": return list(POTR_TEMPORAL_STATE_NAMES)
    raise ValueError(arm)


def _sign_pairs(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    good = [i for i, r in enumerate(rows) if v52._effect_support(r) and bool(r["closed_loop_beneficial"])]
    bad = [i for i, r in enumerate(rows) if v52._effect_support(r) and not bool(r["closed_loop_beneficial"])]
    return [(i, j) for i in bad for j in good]


def _fit_outcome(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    pairs = _sign_pairs(rows)
    if len(rows) < 64 or len(pairs) < 64:
        raise ValueError(f"V53 {arm} insufficient fit support rows={len(rows)} pairs={len(pairs)}")
    X = np.stack([_state(r, arm) for r in rows])
    mean = X.mean(axis=0); std = np.maximum(X.std(axis=0), 1.0e-6)
    Z = (X - mean[None, :]) / std[None, :]
    D = np.stack([Z[i] - Z[j] for i, j in pairs])
    w = np.zeros((Z.shape[1],), dtype=np.float64)
    before = float(D.shape[0] * math.log(2.0))
    for _ in range(80):
        s = D @ w
        q = 1.0 / (1.0 + np.exp(np.clip(s, -60.0, 60.0)))
        grad = -(D.T @ q) + RIDGE_LAMBDA * w
        hw = q * (1.0 - q)
        H = (D.T * hw) @ D + RIDGE_LAMBDA * np.eye(D.shape[1], dtype=np.float64)
        step = np.linalg.solve(H, grad)
        w -= step
        if float(np.linalg.norm(step)) < 1.0e-9:
            break
    s = D @ w
    after = float(np.sum(np.logaddexp(0.0, -s)) + 0.5 * RIDGE_LAMBDA * np.dot(w, w))
    raw = Z @ w
    ben = raw[np.asarray([bool(r["closed_loop_beneficial"]) for r in rows], dtype=bool)]
    if ben.size < 24:
        raise ValueError(f"V53 {arm} insufficient beneficial normalization support={ben.size}")
    return {
        "model": "zero_bias_pairwise_effectful_conditional_sign_risk",
        "state_family": arm,
        "feature_names": _names(arm),
        "feature_mean": [float(x) for x in mean], "feature_std": [float(x) for x in std],
        "weights": [float(x) for x in w], "bias": 0.0, "lambda": RIDGE_LAMBDA,
        "fit_row_count": len(rows), "pair_count": len(pairs),
        "objective_at_zero": before, "objective_final": after,
        "fit_beneficial_score_mean": float(ben.mean()),
        "fit_beneficial_score_std": max(float(ben.std()), 1.0e-6),
    }


def _outcome_risk(r: dict[str, Any], model: dict[str, Any]) -> float:
    z = _state(r, str(model["state_family"]))
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    w = np.asarray(model["weights"], dtype=np.float64)
    raw = float(((z - mean) / np.maximum(std, 1.0e-6)) @ w)
    return float((raw - float(model["fit_beneficial_score_mean"])) / max(float(model["fit_beneficial_score_std"]), 1.0e-6))


def _support_risk(r: dict[str, Any], support_model: dict[str, Any]) -> float:
    return v52._component_risk(r, support_model)


def _threshold(cal: list[dict[str, Any]], support_model: dict[str, Any], outcome_model: dict[str, Any], alpha: float) -> tuple[float, dict[str, Any]]:
    vals = sorted(max(_support_risk(r, support_model), _outcome_risk(r, outcome_model)) for r in cal if bool(r["closed_loop_beneficial"]))
    n = len(vals); rank = int(math.ceil((n + 1) * (1.0 - alpha))) if n else 1
    min_n = int(math.ceil((1.0 - alpha) / alpha))
    if n == 0 or rank > n:
        raise ValueError(f"V53 calibration insufficient n={n} min={min_n} rank={rank}")
    return float(vals[rank - 1]), {
        "positive_calibration_count": n, "conformal_rank": rank, "alpha": float(alpha),
        "threshold": float(vals[rank - 1]), "single_joint_threshold": True, "no_alpha_split": True,
        "finite_sample_condition": "ceil((n+1)*(1-alpha))<=n",
        "minimum_positive_count_for_finite_rank": min_n,
    }


def _nested(rows: list[dict[str, Any]], alpha: float, v52_report: dict[str, Any]) -> dict[str, Any]:
    safety_names = v52._safety_names(rows)
    scalar_report = v52_report["nested_crossfit"]["arms"]["hurdle_sign"]
    arms: dict[str, Any] = {}
    oof_outcome_risk: dict[str, dict[str, float]] = {a: {} for a in ARMS}

    for arm in ARMS:
        folds=[]; keep_by_token={}
        all_y=[]; all_r=[]; all_v52=[]; all_v51=[]
        above_random = better_v52 = better_v51 = 0
        for k in range(FOLDS):
            cf=(k+1)%FOLDS
            fit=[r for r in rows if int(r["outer_test_fold"]) not in {k,cf}]
            cal=[r for r in rows if int(r["outer_test_fold"])==cf]
            test=[r for r in rows if int(r["outer_test_fold"])==k]

            # Freeze exact V52 support hurdle; same split/state/loss.
            scalar_models=v52._fit_models(fit,"hurdle_sign",safety_names)
            support_model=scalar_models["effect_support_risk"]
            scalar_outcome_model=scalar_models["conditional_outcome_risk"]
            om=_fit_outcome(fit,arm)
            tau,ci=_threshold(cal,support_model,om,alpha)
            srisk=np.asarray([_support_risk(r,support_model) for r in test],dtype=np.float64)
            orisk=np.asarray([_outcome_risk(r,om) for r in test],dtype=np.float64)
            scalar_risk=np.asarray([v52._component_risk(r,scalar_outcome_model) for r in test],dtype=np.float64)
            cm=v51._fit_ranker(fit,"qpe_dose")
            v51risk=np.asarray([v51._risk(r,cm) for r in test],dtype=np.float64)
            joint=np.maximum(srisk,orisk)
            keep=[bool(x<=tau) for x in joint]
            for r,kk,rr in zip(test,keep,orisk):
                tok=str(r["scenario_token"])
                if tok in keep_by_token: raise RuntimeError(f"V53 duplicate OOF token {tok}")
                keep_by_token[tok]=kk; oof_outcome_risk[arm][tok]=float(rr)

            # Exact support replay gate.
            null_y=np.asarray([not v52._effect_support(r) for r in test],dtype=bool)
            support_auc=_auc(null_y,srisk)
            expected_support=float(scalar_report["folds"][k]["effect_support_auc"])
            if abs(support_auc-expected_support)>1e-12:
                raise RuntimeError(f"V53 ENGINEERING STOP: V52 support fold {k} replay drift {support_auc} vs {expected_support}")

            eff=[i for i,r in enumerate(test) if v52._effect_support(r)]
            y=np.asarray([not bool(test[i]["closed_loop_beneficial"]) for i in eff],dtype=bool)
            idx=np.asarray(eff,dtype=np.int64)
            auc=_auc(y,orisk[idx]); s_auc=_auc(y,scalar_risk[idx]); v51_auc=_auc(y,v51risk[idx])
            exp_scalar=float(scalar_report["folds"][k]["conditional_sign_auc"])
            exp_v51=float(scalar_report["folds"][k]["v51_control_conditional_sign_auc"])
            if abs(s_auc-exp_scalar)>1e-12 or abs(v51_auc-exp_v51)>1e-12:
                raise RuntimeError(f"V53 ENGINEERING STOP: scalar control replay drift fold={k}")
            ar=bool(math.isfinite(auc) and auc>0.5+EPS)
            bv52=bool(math.isfinite(auc) and math.isfinite(s_auc) and auc>s_auc+EPS)
            bv51=bool(math.isfinite(auc) and math.isfinite(v51_auc) and auc>v51_auc+EPS)
            above_random+=int(ar); better_v52+=int(bv52); better_v51+=int(bv51)
            all_y.extend(y.tolist()); all_r.extend(orisk[idx].tolist()); all_v52.extend(scalar_risk[idx].tolist()); all_v51.extend(v51risk[idx].tolist())
            bm=v50._delta_metrics(test,[True]*len(test)); pm=v50._delta_metrics(test,keep)
            folds.append({
                "fold":k,"fit_events":len(fit),"calibration_events":len(cal),"test_events":len(test),
                "rsmr":bm,"potr":pm,"calibration":ci,"effect_support_auc":support_auc,
                "conditional_sign_auc":auc,"v52_scalar_conditional_sign_auc":s_auc,"v51_conditional_sign_auc":v51_auc,
                "above_random":ar,"better_v52_scalar":bv52,"better_v51_scalar":bv51,
            })

        auc=_auc(np.asarray(all_y,dtype=bool),np.asarray(all_r,dtype=np.float64))
        v52_auc=_auc(np.asarray(all_y,dtype=bool),np.asarray(all_v52,dtype=np.float64))
        v51_auc=_auc(np.asarray(all_y,dtype=bool),np.asarray(all_v51,dtype=np.float64))
        expected_scalar=float(scalar_report["identification"]["conditional_sign_auc"])
        expected_v51=float(scalar_report["identification"]["v51_control_conditional_sign_auc"])
        if abs(v52_auc-expected_scalar)>1e-12 or abs(v51_auc-expected_v51)>1e-12:
            raise RuntimeError("V53 ENGINEERING STOP: scalar aggregate controls do not replay V52")
        identified=bool(
            math.isfinite(auc) and auc>0.5+EPS and auc>v52_auc+EPS and auc>v51_auc+EPS
            and above_random>=4 and better_v52>=4 and better_v51>=4
        )
        keep=v506._align_oof_keep_by_token(rows,keep_by_token)
        base=v50._delta_metrics(rows,[True]*len(rows)); chosen=v50._delta_metrics(rows,keep)
        dep=v50._gate(base,chosen,alpha,[{"rsmr":f["rsmr"],"pior":f["potr"]} for f in folds])
        arms[arm]={
            "folds":folds,
            "identification":{
                "aggregate_auc":auc,"v52_scalar_conditional_sign_auc":v52_auc,"v51_conditional_sign_auc":v51_auc,
                "folds_above_random":above_random,"better_v52_scalar_fold_count":better_v52,
                "better_v51_scalar_fold_count":better_v51,"identified":identified,
            },
            "aggregate":chosen,"deployment_gate":dep,"pass":bool(identified and dep["pass"]),
        }

    # Temporal necessity is a stronger claim: it must beat the endpoint state itself.
    ep=arms["endpoint"]; tp=arms["temporal"]
    both_tokens=sorted(oof_outcome_risk["endpoint"])
    effrows={str(r["scenario_token"]):r for r in rows if v52._effect_support(r)}
    y=np.asarray([not bool(effrows[t]["closed_loop_beneficial"]) for t in both_tokens if t in effrows],dtype=bool)
    ep_r=np.asarray([oof_outcome_risk["endpoint"][t] for t in both_tokens if t in effrows],dtype=np.float64)
    tp_r=np.asarray([oof_outcome_risk["temporal"][t] for t in both_tokens if t in effrows],dtype=np.float64)
    ep_auc=_auc(y,ep_r); tp_auc=_auc(y,tp_r)
    temporal_better_folds=sum(int(float(tp["folds"][k]["conditional_sign_auc"]) > float(ep["folds"][k]["conditional_sign_auc"]) + EPS) for k in range(FOLDS))
    tp["identification"]["endpoint_control_auc"] = ep_auc
    tp["identification"]["better_endpoint_fold_count"] = temporal_better_folds
    tp["identification"]["temporal_necessity_identified"] = bool(tp["identification"]["identified"] and tp_auc > ep_auc + EPS and temporal_better_folds >= 4)
    tp["pass"] = bool(tp["identification"]["temporal_necessity_identified"] and tp["deployment_gate"]["pass"])

    preferred="endpoint" if ep["pass"] else "temporal" if tp["pass"] else None
    if preferred=="endpoint": diagnosis="signed_terminal_operator_contrast_is_sufficient_for_effectful_selected_outcome"
    elif preferred=="temporal": diagnosis="fixed_temporal_operator_contrast_shape_is_required_beyond_terminal_direction"
    elif ep["identification"]["identified"] or tp["identification"]["temporal_necessity_identified"]:
        diagnosis="operator_trajectory_state_identified_but_static_sign_retention_functional_still_deployment_insufficient"
    else:
        diagnosis="preexecution_operator_trajectory_contrast_does_not_identify_effectful_outcome_order"
    return {
        "retention_alpha":alpha,
        "v52_effect_support_control":{"auc":EXPECTED_V52_SUPPORT_AUC,"folds_above_random":EXPECTED_V52_SUPPORT_FOLDS,"frozen":True},
        "arms":arms,"preferred_promotion_arm":preferred,"train_gate_pass":preferred is not None,"failure_diagnosis":diagnosis,
    }


def _decorate(base_cfg: Path, support_model: dict[str,Any], outcome_model: dict[str,Any], tau: float, arm: str, out: Path) -> None:
    cfg=yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    ic=cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    sc=ic["selection_conditioned_intervention_recovery"]
    sc["post_selection_value_mode"]="endpoint_potential_quality_paired_operator_trajectory_retention"
    sc["paired_operator_trajectory_retention"]={
        "state_family":arm,
        "aggregation":"max_support_outcome",
        "components":{"effect_support_risk":support_model,"conditional_outcome_risk":outcome_model},
        "retention_threshold":float(tau),
        "support_state":"exact_frozen_V52_QPE_plus_scalar_D_effect_support",
        "conditional_outcome_state":"preexecution_signed_frozen_proposal_vs_runtime_incumbent_trajectory_contrast",
        "outcome_supervision":"frozen_V50_metric_safe_paired_one_shot_closed_loop_outcomes",
        "runtime_candidate_set":"full_frozen_deployment_candidate_set",
    }
    sc["post_selection_selected_bias"]=0.0
    sc["post_selection_value_training"]="frozen_effect_support_plus_pairwise_effectful_conditional_sign_fixed_lambda_1"
    sc["post_selection_operator"]="freeze_full_set_RSMR_winner_then_POTR_veto_only_same_winner_or_incumbent_no_rerank_no_fallback"
    cfg.setdefault("version",{})["algorithm"]="V64.3.53-EAF-ICER-POTR"
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--v49-fit-report",type=Path,required=True); p.add_argument("--v49-candidate-audit",type=Path,required=True)
    p.add_argument("--v49-scene-audit",type=Path,required=True); p.add_argument("--v49-siir-config",type=Path,required=True)
    p.add_argument("--paired-outcomes",type=Path,required=True); p.add_argument("--v50-5-root",type=Path,required=True); p.add_argument("--v51-fit-report",type=Path,required=True)
    p.add_argument("--v52-fit-report",type=Path,required=True); p.add_argument("--operator-profiles",type=Path,required=True)
    p.add_argument("--output-config",type=Path,required=True); p.add_argument("--output-report",type=Path,required=True)
    a=p.parse_args()

    v52_report=_check_v52(a.v52_fit_report)
    v51_report=v52_report["nested_crossfit"]  # only alpha is needed from V52 below
    states=v50._candidate_states(a.v49_candidate_audit); obs=v50._obs_risk(a.v49_scene_audit)
    rows=v50._join(states,a.paired_outcomes,obs)
    # Reuse exact V51 D and require the newly acquired profile to reproduce it token-by-token.
    scalar_d=v51._load_treatment_contrast(a.v50_5_root,set(states))
    profiles=_load_profiles(a.operator_profiles,set(states))
    for r in rows:
        tok=str(r["scenario_token"]); prof=profiles[tok]; d=float(prof["execution_contrast_linf"])
        if abs(d-float(scalar_d[tok]))>1.0e-9:
            raise RuntimeError(f"V53 ENGINEERING STOP: profile D does not replay V51 scalar D token={tok}: {d} vs {scalar_d[tok]}")
        r["operator_execution_contrast_linf"]=d; r["operator_trajectory_profile"]=prof
    # Cross-check the parent scientific signature independently of the state-only replay.
    _ = v52._check_v51(a.v51_fit_report)
    alpha=float(v52_report["nested_crossfit"]["retention_alpha"])
    nested=_nested(rows,alpha,v52_report)
    preferred=nested["preferred_promotion_arm"]
    final=None
    if preferred is not None:
        safety=v52._safety_names(rows); fit=[r for r in rows if int(r["outer_test_fold"])!=0]; cal=[r for r in rows if int(r["outer_test_fold"])==0]
        scalar=v52._fit_models(fit,"hurdle_sign",safety); sm=scalar["effect_support_risk"]; om=_fit_outcome(fit,preferred)
        tau,ci=_threshold(cal,sm,om,alpha); final={"state_family":preferred,"support_model":sm,"outcome_model":om,"calibration":ci}

    report={
        "audit":"v64_3_53_eaf_icer_potr_fit","algorithm_version":"V64.3.53-EAF-ICER-POTR",
        "scientific_role":"preregistered_V52_support_identified_conditional_order_failed_branch_operator_trajectory_state",
        "mechanism_hypothesis":"Freeze the identified QPE+D effect-support hurdle and test whether effectful outcome direction requires signed proposal-vs-incumbent treatment geometry: terminal contrast first, then a fixed two-mode temporal contrast basis.",
        "frozen_contract":{
            "RSMR_selector_unchanged":True,"V52_effect_support_state_and_functional_unchanged":True,
            "paired_outcome_labels_unchanged":True,"lambda":1.0,"retention_alpha_and_single_conformal_threshold_unchanged":True,
            "same_winner_or_incumbent_only":True,"no_rerank_second_best_fallback":True,"no_new_offline_future_observable":True,
            "no_safety_weight_or_catastrophe_veto":True,
        },
        "operator_profile_source":"TRAIN_only_state_replay_at_exact_metric_safe_V50_treatment_anchor; preexecution; labels reused, not recollected",
        "nested_crossfit":nested,"final_runtime_fit":final,"train_gate_pass":bool(nested["train_gate_pass"]),
        "preregistered_branch_order":["endpoint","temporal"],
        "preregistered_next_branch":{
            "if_endpoint_pass":"promote minimal signed terminal operator contrast; freeze and use untouched paired validation",
            "if_only_temporal_pass":"temporal treatment shape is necessary beyond terminal direction; freeze and use untouched paired validation",
            "if_state_identified_but_deployment_fail":"operator state solved; return bottleneck to deployment functional using paired evidence, not more state",
            "if_both_state_arms_fail_identification":"close preexecution operator-trajectory geometry as sufficient; next evidence/state must be post-intervention paired dynamic response process",
        },
        "prohibited_tuning":["D transform/interaction sweep","trajectory horizon or basis sweep","peak/early handcrafted features","attention/MLP",
            "V46 agent-occupancy temporal-profile resurrection","threshold/calibration/lambda/loss/class/focal/catastrophe weighting","safety scalarization",
            "new offline future observable","RSMR/B/M/topK/candidate-count change","rerank/second-best/fallback","validation pooling"],
    }
    a.output_report.parent.mkdir(parents=True,exist_ok=True); a.output_report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not nested["train_gate_pass"]:
        if a.output_config.exists(): a.output_config.unlink()
        print(json.dumps({"pass":False,"failure_diagnosis":nested["failure_diagnosis"],"output_config_emitted":False},sort_keys=True))
        raise SystemExit(f"V64.3.53 POTR TRAIN scientific STOP ({nested['failure_diagnosis']}); do not consume untouched validation")
    assert final is not None
    _decorate(a.v49_siir_config,final["support_model"],final["outcome_model"],float(final["calibration"]["threshold"]),preferred,a.output_config)
    print(json.dumps({"pass":True,"preferred_promotion_arm":preferred,"output_config":str(a.output_config)},sort_keys=True))

if __name__=="__main__": main()
