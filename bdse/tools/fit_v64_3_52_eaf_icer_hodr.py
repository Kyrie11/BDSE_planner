from __future__ import annotations

"""V64.3.52 EAF-ICER-HODR fit-only TRAIN mechanism test.

V51 identifies the operator-relative QPE+D state but fails the unchanged paired
causal deployment gate with a single sign-only ranker.  V52 freezes the state,
paired evidence, RSMR selector, alpha, lambda and veto-only operator, and tests
only a structured paired-outcome functional.

Preregistered causal arms:
  A. HURDLE-SIGN   : separate structural-null/effect support, then binary sign
                     ranking conditional on an effect.
  B. HURDLE-PARETO : same support hurdle, but replace sign compression by a
                     Pareto pairwise order over official-score and hard-safety
                     deltas.  No safety weights or standalone safety veto.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from bdse.planner.paired_outcome_dominance_retention import HODR_STATE_NAMES
from bdse.tools import fit_v64_3_50_eaf_icer_pior as v50
from bdse.tools import fit_v64_3_50_6_eaf_icer_pior as v506
from bdse.tools import fit_v64_3_51_eaf_icer_pocr as v51
from bdse.tools.fit_v64_3_49_eaf_icer_siir import _auc

EPS = 1.0e-12
FOLDS = 5
RIDGE_LAMBDA = 1.0
ARMS = ("hurdle_sign", "hurdle_pareto")
EXPECTED_V51_FAILURE = "operator_contrast_state_identified_but_low_capacity_sign_retention_functional_insufficient"
EXPECTED_V51_ADDITIVE_AUC = 0.5767120019088523
EXPECTED_V51_INTERACTION_AUC = 0.5484913559358799
EXPECTED_V51_FIT_SHA256 = "54d3664e378c85ba482485c49ddcb3e10e83a59d5d04cd049ccaf05fdbb23049"


def _check_v51(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    n = d.get("nested_crossfit", {})
    if d.get("train_gate_pass") is not False or n.get("train_gate_pass") is not False:
        raise RuntimeError("V52 ENGINEERING STOP: V51 must be the preregistered TRAIN failure")
    if n.get("failure_diagnosis") != EXPECTED_V51_FAILURE or n.get("preferred_promotion_arm") is not None:
        raise RuntimeError(f"V52 ENGINEERING STOP: V51 branch drift {n.get('failure_diagnosis')}")
    a = n.get("arms", {}).get("qpe_dose", {})
    x = n.get("arms", {}).get("qpe_dose_interaction", {})
    if not bool(a.get("risk_identification", {}).get("identified", False)) or not bool(x.get("risk_identification", {}).get("identified", False)):
        raise RuntimeError("V52 ENGINEERING STOP: V51 operator state must be identified in both preregistered arms")
    if abs(float(a["risk_identification"]["aggregate_auc"]) - EXPECTED_V51_ADDITIVE_AUC) > 1e-12:
        raise RuntimeError("V52 ENGINEERING STOP: V51 additive AUC signature drift")
    if abs(float(x["risk_identification"]["aggregate_auc"]) - EXPECTED_V51_INTERACTION_AUC) > 1e-12:
        raise RuntimeError("V52 ENGINEERING STOP: V51 interaction AUC signature drift")
    if bool(a.get("deployment_gate", {}).get("pass", False)) or bool(x.get("deployment_gate", {}).get("pass", False)):
        raise RuntimeError("V52 ENGINEERING STOP: V51 deployment failure signature drift")
    diag = n.get("operator_contrast_diagnostic", {})
    if diag.get("effect_support_identified") is not True or diag.get("physical_equal_all_null_effect") is not True:
        raise RuntimeError("V52 ENGINEERING STOP: V51 effect-support diagnostic drift")
    return d


def _state(r: dict[str, Any]) -> np.ndarray:
    return v51._arm_state(r, "qpe_dose")


def _effect_support(r: dict[str, Any]) -> bool:
    return v51._effect_support(r)


def _outcome_vector(r: dict[str, Any], safety_names: list[str]) -> np.ndarray:
    vals = [float(r["closed_loop_score_delta"])]
    sd = dict(r.get("safety_delta", {}))
    vals.extend(float(sd.get(k, 0.0)) for k in safety_names)
    x = np.asarray(vals, dtype=np.float64)
    if np.any(~np.isfinite(x)):
        raise ValueError(f"V52 non-finite paired outcome vector {r['scenario_token']}")
    return x


def _safety_names(rows: list[dict[str, Any]]) -> list[str]:
    keys = sorted({str(k) for r in rows for k in dict(r.get("safety_delta", {})).keys()})
    if not keys:
        raise ValueError("V52 paired outcomes contain no hard-safety delta coordinates")
    return keys


def _pareto_pairs(rows: list[dict[str, Any]], safety_names: list[str]) -> list[tuple[int, int]]:
    """Return (bad, good) pairs using only unambiguous deployment dominance.

    Higher official-score delta and higher safety deltas are all better.  A
    pair is used only when one effectful intervention is no better on every
    coordinate and strictly worse on at least one.  Ambiguous trade-offs are
    omitted; no scalar safety weight is introduced.
    """
    effect_idx = [i for i, r in enumerate(rows) if _effect_support(r)]
    vec = {i: _outcome_vector(rows[i], safety_names) for i in effect_idx}
    pairs: list[tuple[int, int]] = []
    for ai in range(len(effect_idx)):
        i = effect_idx[ai]
        for aj in range(ai + 1, len(effect_idx)):
            j = effect_idx[aj]
            vi, vj = vec[i], vec[j]
            i_worse = bool(np.all(vi <= vj + EPS) and np.any(vi < vj - EPS))
            j_worse = bool(np.all(vj <= vi + EPS) and np.any(vj < vi - EPS))
            if i_worse and not j_worse:
                pairs.append((i, j))
            elif j_worse and not i_worse:
                pairs.append((j, i))
    return pairs


def _sign_pairs(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    good = [i for i, r in enumerate(rows) if _effect_support(r) and bool(r["closed_loop_beneficial"])]
    bad = [i for i, r in enumerate(rows) if _effect_support(r) and not bool(r["closed_loop_beneficial"])]
    return [(i, j) for i in bad for j in good]


def _support_pairs(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    effect = [i for i, r in enumerate(rows) if _effect_support(r)]
    null = [i for i, r in enumerate(rows) if not _effect_support(r)]
    return [(i, j) for i in null for j in effect]


def _fit_pairwise(rows: list[dict[str, Any]], pairs: list[tuple[int, int]], model_name: str) -> dict[str, Any]:
    if len(rows) < 64 or len(pairs) < 64:
        raise ValueError(f"V52 {model_name} has insufficient fit support rows={len(rows)} pairs={len(pairs)}")
    X = np.stack([_state(r) for r in rows])
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
        w = w - step
        if float(np.linalg.norm(step)) < 1.0e-9:
            break
    s = D @ w
    after = float(np.sum(np.logaddexp(0.0, -s)) + 0.5 * RIDGE_LAMBDA * np.dot(w, w))
    raw = Z @ w
    ben = raw[np.asarray([bool(r["closed_loop_beneficial"]) for r in rows], dtype=bool)]
    if ben.size < 24:
        raise ValueError(f"V52 {model_name} has insufficient beneficial normalization support {ben.size}")
    return {
        "model": model_name,
        "feature_names": list(HODR_STATE_NAMES),
        "feature_mean": [float(x) for x in mean],
        "feature_std": [float(x) for x in std],
        "weights": [float(x) for x in w],
        "bias": 0.0,
        "lambda": RIDGE_LAMBDA,
        "fit_row_count": len(rows),
        "pair_count": len(pairs),
        "objective_at_zero": before,
        "objective_final": after,
        "fit_beneficial_score_mean": float(ben.mean()),
        "fit_beneficial_score_std": max(float(ben.std()), 1.0e-6),
    }


def _component_risk(r: dict[str, Any], model: dict[str, Any]) -> float:
    z = _state(r)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    w = np.asarray(model["weights"], dtype=np.float64)
    raw = float(((z - mean) / np.maximum(std, 1.0e-6)) @ w)
    return float((raw - float(model["fit_beneficial_score_mean"])) / max(float(model["fit_beneficial_score_std"]), 1.0e-6))


def _fit_models(rows: list[dict[str, Any]], arm: str, safety_names: list[str]) -> dict[str, Any]:
    support = _fit_pairwise(rows, _support_pairs(rows), "zero_bias_pairwise_effect_support_risk")
    if arm == "hurdle_sign":
        outcome = _fit_pairwise(rows, _sign_pairs(rows), "zero_bias_pairwise_conditional_sign_risk")
    elif arm == "hurdle_pareto":
        outcome = _fit_pairwise(rows, _pareto_pairs(rows, safety_names), "zero_bias_pairwise_pareto_outcome_risk")
    else:
        raise ValueError(arm)
    return {"arm": arm, "effect_support_risk": support, "conditional_outcome_risk": outcome}


def _risk(r: dict[str, Any], models: dict[str, Any]) -> tuple[float, float, float]:
    sr = _component_risk(r, models["effect_support_risk"])
    orisk = _component_risk(r, models["conditional_outcome_risk"])
    return float(max(sr, orisk)), float(sr), float(orisk)


def _threshold(cal: list[dict[str, Any]], models: dict[str, Any], alpha: float) -> tuple[float, dict[str, Any]]:
    vals = sorted(_risk(r, models)[0] for r in cal if bool(r["closed_loop_beneficial"]))
    n = len(vals); rank = int(math.ceil((n + 1) * (1.0 - alpha))) if n else 1
    min_n = int(math.ceil((1.0 - alpha) / alpha))
    if n == 0 or rank > n:
        raise ValueError(f"V52 calibration insufficient for frozen alpha n={n} min={min_n} rank={rank}")
    return float(vals[rank - 1]), {
        "positive_calibration_count": n,
        "conformal_rank": rank,
        "alpha": float(alpha),
        "threshold": float(vals[rank - 1]),
        "finite_sample_condition": "ceil((n+1)*(1-alpha))<=n",
        "minimum_positive_count_for_finite_rank": min_n,
        "single_joint_threshold": True,
        "no_alpha_split": True,
    }


def _concordance(rows: list[dict[str, Any]], risk: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    vals = []
    for bad, good in pairs:
        d = float(risk[bad] - risk[good])
        vals.append(1.0 if d > EPS else 0.5 if abs(d) <= EPS else 0.0)
    return float(np.mean(vals))


def _nested(rows: list[dict[str, Any]], alpha: float, v51_report: dict[str, Any]) -> dict[str, Any]:
    safety_names = _safety_names(rows)
    control = v51_report["nested_crossfit"]["arms"]["qpe_dose"]
    arms: dict[str, Any] = {}
    for arm in ARMS:
        folds = []
        keep_by_token: dict[str, bool] = {}
        all_support_y: list[bool] = []; all_support_risk: list[float] = []
        all_sign_y: list[bool] = []; all_outcome_risk: list[float] = []; all_control_risk: list[float] = []
        all_full_y: list[bool] = []; all_full_control_risk: list[float] = []
        all_dom_correct: list[float] = []; all_dom_control_correct: list[float] = []
        support_good_folds = sign_better_folds = dom_better_folds = 0
        for k in range(FOLDS):
            cf = (k + 1) % FOLDS
            fit = [r for r in rows if int(r["outer_test_fold"]) not in {k, cf}]
            cal = [r for r in rows if int(r["outer_test_fold"]) == cf]
            test = [r for r in rows if int(r["outer_test_fold"]) == k]
            models = _fit_models(fit, arm, safety_names)
            tau, ci = _threshold(cal, models, alpha)
            full = [_risk(r, models) for r in test]
            risk = np.asarray([x[0] for x in full], dtype=np.float64)
            srisk = np.asarray([x[1] for x in full], dtype=np.float64)
            orisk = np.asarray([x[2] for x in full], dtype=np.float64)
            keep = [bool(x <= tau) for x in risk]
            for r, kk in zip(test, keep):
                tok = str(r["scenario_token"])
                if tok in keep_by_token:
                    raise RuntimeError(f"V52 duplicate OOF keep {tok}")
                keep_by_token[tok] = kk

            # Exact low-capacity V51 additive sign-risk control on same split.
            cm = v51._fit_ranker(fit, "qpe_dose")
            crisk = np.asarray([v51._risk(r, cm) for r in test], dtype=np.float64)
            full_target = np.asarray([not bool(r["closed_loop_beneficial"]) for r in test], dtype=bool)
            control_full_auc = _auc(full_target, crisk)
            expected_control_fold_auc = float(control["folds"][k]["risk_auc"])
            if not math.isfinite(control_full_auc) or abs(control_full_auc - expected_control_fold_auc) > 1e-12:
                raise RuntimeError(f"V52 ENGINEERING STOP: V51 additive control fold {k} replay drift {control_full_auc} vs {expected_control_fold_auc}")
            all_full_y.extend(full_target.tolist()); all_full_control_risk.extend(crisk.tolist())

            null_target = np.asarray([not _effect_support(r) for r in test], dtype=bool)
            support_auc = _auc(null_target, srisk)
            support_good = bool(math.isfinite(support_auc) and support_auc > 0.5 + EPS)
            support_good_folds += int(support_good)

            eff_idx = [i for i, r in enumerate(test) if _effect_support(r)]
            sign_auc = control_sign_auc = float("nan")
            sign_better = False
            if eff_idx:
                sign_target = np.asarray([not bool(test[i]["closed_loop_beneficial"]) for i in eff_idx], dtype=bool)
                sign_auc = _auc(sign_target, orisk[np.asarray(eff_idx, dtype=np.int64)])
                control_sign_auc = _auc(sign_target, crisk[np.asarray(eff_idx, dtype=np.int64)])
                sign_better = bool(math.isfinite(sign_auc) and math.isfinite(control_sign_auc) and sign_auc > control_sign_auc + EPS)
                all_sign_y.extend(sign_target.tolist())
                all_outcome_risk.extend(orisk[np.asarray(eff_idx, dtype=np.int64)].tolist())
                all_control_risk.extend(crisk[np.asarray(eff_idx, dtype=np.int64)].tolist())
            sign_better_folds += int(sign_better)

            dpairs = _pareto_pairs(test, safety_names)
            dom = _concordance(test, orisk, dpairs)
            cdom = _concordance(test, crisk, dpairs)
            dom_better = bool(math.isfinite(dom) and math.isfinite(cdom) and dom > cdom + EPS)
            dom_better_folds += int(dom_better)
            for bad, good in dpairs:
                d = float(orisk[bad] - orisk[good]); cd = float(crisk[bad] - crisk[good])
                all_dom_correct.append(1.0 if d > EPS else 0.5 if abs(d) <= EPS else 0.0)
                all_dom_control_correct.append(1.0 if cd > EPS else 0.5 if abs(cd) <= EPS else 0.0)

            bm = v50._delta_metrics(test, [True] * len(test)); pm = v50._delta_metrics(test, keep)
            folds.append({
                "fold": k, "fit_events": len(fit), "calibration_events": len(cal), "test_events": len(test),
                "rsmr": bm, "hodr": pm, "calibration": ci,
                "effect_support_auc": support_auc, "effect_support_identified": support_good,
                "conditional_sign_auc": sign_auc, "v51_control_conditional_sign_auc": control_sign_auc,
                "v51_control_full_nonbeneficial_auc": control_full_auc,
                "conditional_sign_better_v51": sign_better,
                "pareto_concordance": dom, "v51_control_pareto_concordance": cdom,
                "pareto_better_v51": dom_better, "pareto_pair_count": len(dpairs),
            })
            all_support_y.extend(null_target.tolist()); all_support_risk.extend(srisk.tolist())

        full_control_auc = _auc(np.asarray(all_full_y, dtype=bool), np.asarray(all_full_control_risk, dtype=np.float64))
        expected_full_control_auc = float(control["risk_identification"]["aggregate_auc"])
        if not math.isfinite(full_control_auc) or abs(full_control_auc - expected_full_control_auc) > 1e-12:
            raise RuntimeError(f"V52 ENGINEERING STOP: V51 additive aggregate control replay drift {full_control_auc} vs {expected_full_control_auc}")
        support_auc = _auc(np.asarray(all_support_y, dtype=bool), np.asarray(all_support_risk, dtype=np.float64))
        sign_auc = _auc(np.asarray(all_sign_y, dtype=bool), np.asarray(all_outcome_risk, dtype=np.float64))
        control_sign_auc = _auc(np.asarray(all_sign_y, dtype=bool), np.asarray(all_control_risk, dtype=np.float64))
        dom = float(np.mean(all_dom_correct)) if all_dom_correct else float("nan")
        cdom = float(np.mean(all_dom_control_correct)) if all_dom_control_correct else float("nan")
        support_identified = bool(math.isfinite(support_auc) and support_auc > 0.5 + EPS and support_good_folds >= 4)
        if arm == "hurdle_sign":
            outcome_identified = bool(math.isfinite(sign_auc) and math.isfinite(control_sign_auc) and sign_auc > control_sign_auc + EPS and sign_better_folds >= 4)
        else:
            outcome_identified = bool(math.isfinite(dom) and math.isfinite(cdom) and dom > cdom + EPS and dom_better_folds >= 4)
        functional_identified = bool(support_identified and outcome_identified)
        keep = v506._align_oof_keep_by_token(rows, keep_by_token)
        base = v50._delta_metrics(rows, [True] * len(rows)); chosen = v50._delta_metrics(rows, keep)
        dep = v50._gate(base, chosen, alpha, [{"rsmr": f["rsmr"], "pior": f["hodr"]} for f in folds])
        arms[arm] = {
            "folds": folds,
            "identification": {
                "v51_control_full_nonbeneficial_auc": full_control_auc,
                "support_auc": support_auc, "support_folds_above_random": support_good_folds,
                "support_identified": support_identified,
                "conditional_sign_auc": sign_auc, "v51_control_conditional_sign_auc": control_sign_auc,
                "v51_control_full_nonbeneficial_auc": control_full_auc,
                "conditional_sign_better_v51_fold_count": sign_better_folds,
                "pareto_concordance": dom, "v51_control_pareto_concordance": cdom,
                "pareto_better_v51_fold_count": dom_better_folds,
                "functional_identified": functional_identified,
            },
            "aggregate": chosen,
            "deployment_gate": dep,
            "pass": bool(functional_identified and dep["pass"]),
        }

    preferred = "hurdle_sign" if arms["hurdle_sign"]["pass"] else "hurdle_pareto" if arms["hurdle_pareto"]["pass"] else None
    any_identified = any(bool(arms[a]["identification"]["functional_identified"]) for a in ARMS)
    if preferred == "hurdle_sign":
        diagnosis = "structural_null_factorization_is_sufficient_with_conditional_sign_functional"
    elif preferred == "hurdle_pareto":
        diagnosis = "deployment_pareto_outcome_order_is_required_beyond_binary_sign"
    elif any_identified:
        diagnosis = "structured_static_paired_outcome_functional_identified_but_deployment_still_insufficient"
    elif any(bool(arms[a]["identification"]["support_identified"]) for a in ARMS):
        diagnosis = "effect_support_identified_but_operator_state_does_not_identify_conditional_outcome_order"
    else:
        diagnosis = "structured_effect_support_functional_not_identified"
    return {
        "retention_alpha": alpha,
        "state_control": {
            "name": "V51-QPE+DOSE",
            "aggregate_auc": float(control["risk_identification"]["aggregate_auc"]),
            "identified": bool(control["risk_identification"]["identified"]),
            "deployment_gate_pass": bool(control["deployment_gate"]["pass"]),
        },
        "safety_delta_names": safety_names,
        "arms": arms,
        "preferred_promotion_arm": preferred,
        "train_gate_pass": preferred is not None,
        "failure_diagnosis": diagnosis,
    }


def _decorate(base_cfg: Path, models: dict[str, Any], tau: float, arm: str, out: Path) -> None:
    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    sc = ic["selection_conditioned_intervention_recovery"]
    sc["post_selection_value_mode"] = "endpoint_potential_quality_paired_outcome_dominance_retention"
    sc["paired_outcome_dominance_retention"] = {
        "feature_names": list(HODR_STATE_NAMES),
        "functional": arm,
        "aggregation": "max_support_outcome",
        "components": {
            "effect_support_risk": models["effect_support_risk"],
            "conditional_outcome_risk": models["conditional_outcome_risk"],
        },
        "retention_threshold": float(tau),
        "operator_contrast": "exact_V51_max_abs_frozen_RSMR_proposal_vs_runtime_incumbent_bounded_trajectory_tensor",
        "outcome_supervision": "TRAIN_only_metric_safe_paired_one_shot_closed_loop_outcomes",
        "outcome_functional": "effect_hurdle_then_conditional_sign_or_pareto_order_no_safety_scalarization",
        "runtime_candidate_set": "full_frozen_deployment_candidate_set",
    }
    sc["post_selection_selected_bias"] = 0.0
    sc["post_selection_value_training"] = "paired_effect_support_plus_conditional_outcome_pairwise_fixed_lambda_1"
    sc["post_selection_operator"] = "freeze_full_set_RSMR_winner_then_HODR_veto_only_same_winner_or_incumbent_no_rerank_no_fallback"
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.52-EAF-ICER-HODR"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.52-EAF-ICER-HODR"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v49-fit-report", type=Path, required=True)
    p.add_argument("--v49-candidate-audit", type=Path, required=True)
    p.add_argument("--v49-scene-audit", type=Path, required=True)
    p.add_argument("--v49-siir-config", type=Path, required=True)
    p.add_argument("--paired-outcomes", type=Path, required=True)
    p.add_argument("--v50-5-root", type=Path, required=True)
    p.add_argument("--v51-fit-report", type=Path, required=True)
    p.add_argument("--output-config", type=Path, required=True)
    p.add_argument("--output-report", type=Path, required=True)
    a = p.parse_args()

    v51_report = _check_v51(a.v51_fit_report)
    v50_report = v51_report["nested_crossfit"]["qpe_control"]
    alpha = float(v50_report["retention_alpha"])
    states = v50._candidate_states(a.v49_candidate_audit)
    obs = v50._obs_risk(a.v49_scene_audit)
    rows = v50._join(states, a.paired_outcomes, obs)
    contrast = v51._load_treatment_contrast(a.v50_5_root, set(states))
    for r in rows:
        r["operator_execution_contrast_linf"] = float(contrast[str(r["scenario_token"])])

    nested = _nested(rows, alpha, v51_report)
    preferred = nested["preferred_promotion_arm"]
    final = None
    if preferred is not None:
        safety_names = nested["safety_delta_names"]
        fit = [r for r in rows if int(r["outer_test_fold"]) != 0]
        cal = [r for r in rows if int(r["outer_test_fold"]) == 0]
        models = _fit_models(fit, preferred, safety_names)
        tau, ci = _threshold(cal, models, alpha)
        final = {"arm": preferred, "models": models, "calibration": ci}

    report = {
        "audit": "v64_3_52_eaf_icer_hodr_fit",
        "algorithm_version": "V64.3.52-EAF-ICER-HODR",
        "scientific_role": "preregistered_V51_state_identified_deployment_failed_branch_structured_paired_outcome_functional",
        "mechanism_hypothesis": "V51 identifies the minimal operator-relative QPE+D state, but a single binary sign ranker discards two deployment-relevant outcome structures: a large structural-null/effect-support layer and the partial order induced by paired official-score and hard-safety deltas. Factor effect support first, then identify the conditional outcome order without safety scalarization.",
        "frozen_contract": {
            "RSMR_selector_unchanged": True,
            "QPE_plus_scalar_operator_contrast_state_unchanged": True,
            "paired_outcome_evidence_unchanged": True,
            "lambda": 1.0,
            "retention_alpha_and_single_conformal_threshold_unchanged": True,
            "same_winner_or_incumbent_only": True,
            "no_rerank_second_best_fallback": True,
            "no_new_runtime_or_offline_observable": True,
            "no_safety_weight_scalarization": True,
            "no_standalone_catastrophe_veto": True,
        },
        "nested_crossfit": nested,
        "final_runtime_fit": final,
        "train_gate_pass": bool(nested["train_gate_pass"]),
        "preregistered_branch_order": ["hurdle_sign", "hurdle_pareto"],
        "preregistered_next_branch": {
            "if_hurdle_sign_pass": "promote minimal structural-null factorization; freeze and run untouched paired validation",
            "if_only_hurdle_pareto_pass": "promote Pareto-ordered paired outcome functional; freeze and run untouched paired validation",
            "if_functional_identified_but_deployment_fail": "static operator-relative state+functional exhausted; next collect/model paired temporal closed-loop outcome process, not offline features",
            "if_support_identified_but_conditional_order_fail": "operator-relative scalar state insufficient for structured outcome order; next state acquisition must come from paired/on-policy treatment-control process",
            "if_support_fail": "ENGINEERING/scientific audit of effect-support definition because V51 scalar-D effect-support signature should replay",
        },
        "prohibited_tuning": [
            "V51 QPE+D state expansion before functional test completes",
            "QPE-only rescue", "loss/lambda/class/focal/catastrophe weighting", "threshold/calibration sweep", "bigger MLP",
            "K/logK/K interaction", "new offline future observable", "V46 variance/handcrafted temporal profile resurrection",
            "V47 AGENT-2D resurrection", "V49 SIIR/random-prefix resurrection", "CVaR", "selected translation",
            "standalone binary catastrophe veto", "RSMR/B/M/topK/candidate-count change", "rerank/second-best/fallback", "validation pooling",
        ],
    }
    a.output_report.parent.mkdir(parents=True, exist_ok=True)
    a.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not nested["train_gate_pass"]:
        if a.output_config.exists():
            a.output_config.unlink()
        print(json.dumps({"pass": False, "failure_diagnosis": nested["failure_diagnosis"], "output_config_emitted": False}, sort_keys=True))
        raise SystemExit(f"V64.3.52 HODR TRAIN scientific STOP ({nested['failure_diagnosis']}); do not consume untouched validation")
    assert final is not None
    _decorate(a.v49_siir_config, final["models"], float(final["calibration"]["threshold"]), str(final["arm"]), a.output_config)
    print(json.dumps({"pass": True, "preferred_promotion_arm": preferred, "output_config": str(a.output_config)}, sort_keys=True))


if __name__ == "__main__":
    main()
