from __future__ import annotations

"""V64.3.48 OCRR nested TRAIN mechanism test.

V47 preregistered a representation stop when both future-state nuisances were
identifiable but the deployment gate remained open.  V48 therefore freezes all
V47 runtime observables and changes only the selected-policy functional.

The key hypothesis is post-selection multiplicity: the outcome law of an
extremal RSMR winner depends on the size of the admissible challenger set from
which it was selected.  A zero-bias pairwise sign-risk ranker is fit on frozen
RSMR proposals, and a split-conformal threshold protects the already registered
capture tolerance.  No action is re-ranked: the risk layer can only retain the
same RSMR proposal or return to the incumbent.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.planner.operator_conditioned_risk_retention import OCRR_STATE_NAMES
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS
from bdse.tools.fit_v64_3_38_eaf_icer_davr import CAPTURE_TOL, CATASTROPHE_REDUCTION_MIN, NOOP_REDUCTION_MIN, _auc
from bdse.tools.fit_v64_3_43_eaf_icer_cfrv import _gate, _metrics

EPS = 1.0e-12
RIDGE_LAMBDA = 1.0
ARMS = ["sign_nomult", "sign_mult"]

V47_EXPECTED = {
    "rsmr_rank_aggregate": (502, 221, 107, 28, 43.29405361274824),
    "quality_control_aggregate": (205, 129, 30, 13, 43.905547394411805),
    "v45_plan_control_aggregate": (217, 121, 38, 9, 56.55117310290402),
    "agent_2d_aggregate": (213, 118, 36, 10, 52.305649566059444),
    "ego_reference_aggregate": (251, 136, 45, 9, 59.53269591505746),
    "fsfr_joint_aggregate": (249, 135, 42, 9, 57.004928000622115),
}


def _sig(d: dict[str, Any]) -> tuple[Any, Any, Any, Any, float]:
    return (
        d.get("selected_count"), d.get("selected_positive_count"),
        d.get("no_positive_opportunity_false_intervention_count"), d.get("catastrophic_count"),
        float(d.get("teacher_improvement_sum", float("nan"))),
    )


def _check_v47(path: Path) -> dict[str, Any]:
    r = json.loads(path.read_text())
    n = r.get("nested_crossfit", {})
    for k, e in V47_EXPECTED.items():
        g = _sig(n.get(k, {}))
        if any(g[i] != e[i] for i in range(4)) or abs(g[4] - e[4]) > 1.0e-9:
            raise RuntimeError(f"V48 ENGINEERING STOP: V47 signature mismatch {k}: {g}")
    if r.get("train_gate_pass") is not False or n.get("train_gate_pass") is not False:
        raise RuntimeError("V48 ENGINEERING STOP: V47 unexpectedly passed TRAIN gate")
    ident = n.get("future_state_identification", {})
    if not bool(ident.get("agent_2d_response_identified", False)) or not bool(ident.get("ego_reference_identified", False)):
        raise RuntimeError("V48 ENGINEERING STOP: V47 nuisance-identification signature changed")
    if n.get("failure_diagnosis") != "future_state_nuisances_are_identifiable_but_absolute_zero_requires_selected_deployment_decision_functional_or_more_general_future_state":
        raise RuntimeError("V48 ENGINEERING STOP: V47 scientific-stop signature changed")
    # V46/V47 preregistration resolves the legacy diagnosis string: once both
    # nuisance families are identifiable and deployment still fails, no third
    # future-state block may be appended before testing the selected operator.
    return r


def _read_audit(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            z: dict[str, Any] = dict(r)
            for k in [
                "outer_test_fold", "calibration_fold", "candidate_count", "positive_opportunity",
                "rsm_selected_action", "quality_selected_action", "plan_control_selected_action",
                "agent_2d_selected_action", "ego_ref_selected_action", "fsfr_joint_selected_action",
            ]:
                z[k] = int(float(r[k]))
            for k in [
                "rsm_selected_teacher_improvement", "quality_value", "plan_control_value", "agent_2d_value",
                "ego_ref_value", "fsfr_joint_value", "fsfr_plan_1d_occupancy_cost_improvement",
                "fsfr_plan_2d_occupancy_cost_improvement", "fsfr_predicted_demo_cost_improvement",
            ]:
                z[k] = float(r[k]) if str(r[k]).strip() not in {"", "nan", "NaN"} else float("nan")
            rows.append(z)
    toks = [str(r["scenario_token"]) for r in rows]
    if len(rows) != 782 or len(set(toks)) != 782:
        raise RuntimeError(f"V48 ENGINEERING STOP: V47 scene audit must be 782/782 unique, got {len(rows)}/{len(set(toks))}")
    if sorted(set(int(r["outer_test_fold"]) for r in rows)) != list(range(FOLDS)):
        raise RuntimeError("V48 ENGINEERING STOP: V47 audit outer folds are incomplete")
    return rows


def _state(r: dict[str, Any], use_multiplicity: bool) -> np.ndarray:
    q = float(r["quality_value"])
    p = float(r["plan_control_value"])
    e = float(r["ego_ref_value"])
    k = max(int(r["candidate_count"]), 1)
    z = np.asarray([q, p - q, e - p, math.log(float(k)) if use_multiplicity else 0.0], dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise ValueError(f"non-finite V48 operator state {r['scenario_token']}")
    return z


def _fit_sign_ranker(rows: list[dict[str, Any]], use_multiplicity: bool) -> dict[str, Any]:
    rr = [r for r in rows if int(r["rsm_selected_action"]) >= 0]
    X = np.stack([_state(r, use_multiplicity) for r in rr])
    y = np.asarray([float(r["rsm_selected_teacher_improvement"]) for r in rr], dtype=np.float64)
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1.0e-6)
    Z = (X - mean[None, :]) / std[None, :]
    good = Z[y > 0.0]; bad = Z[y <= 0.0]
    if good.shape[0] < 32 or bad.shape[0] < 32:
        raise ValueError("V48 selected sign ranker has insufficient sign populations")
    D = (bad[:, None, :] - good[None, :, :]).reshape(-1, Z.shape[1])
    w = np.zeros((Z.shape[1],), dtype=np.float64)
    before = float(D.shape[0] * math.log(2.0))
    for _ in range(80):
        s = D @ w
        q = 1.0 / (1.0 + np.exp(np.clip(s, -60.0, 60.0)))  # sigmoid(-margin)
        grad = -(D.T @ q) + RIDGE_LAMBDA * w
        hw = q * (1.0 - q)
        H = (D.T * hw) @ D + RIDGE_LAMBDA * np.eye(D.shape[1], dtype=np.float64)
        step = np.linalg.solve(H, grad)
        w2 = w - step
        w = w2
        if float(np.linalg.norm(step)) < 1.0e-9:
            break
    s = D @ w
    after = float(np.sum(np.logaddexp(0.0, -s)) + 0.5 * RIDGE_LAMBDA * np.dot(w, w))
    raw = Z @ w
    pos = raw[y > 0.0]
    pmean = float(pos.mean()); pstd = max(float(pos.std()), 1.0e-6)
    return {
        "model": "zero_bias_pairwise_selected_sign_risk",
        "feature_names": list(OCRR_STATE_NAMES),
        "feature_mean": [float(x) for x in mean], "feature_std": [float(x) for x in std],
        "weights": [float(x) for x in w], "bias": 0.0, "lambda": RIDGE_LAMBDA,
        "use_extremal_multiplicity": bool(use_multiplicity),
        "selected_count": int(len(y)), "positive_count": int(np.sum(y > 0.0)), "nonpositive_count": int(np.sum(y <= 0.0)),
        "pair_count": int(D.shape[0]), "objective_at_zero": before, "objective_final": after,
        "fit_positive_score_mean": pmean, "fit_positive_score_std": pstd,
    }


def _risk(r: dict[str, Any], m: dict[str, Any]) -> float:
    z = _state(r, bool(m.get("use_extremal_multiplicity", False)))
    mean = np.asarray(m["feature_mean"], dtype=np.float64)
    std = np.asarray(m["feature_std"], dtype=np.float64)
    w = np.asarray(m["weights"], dtype=np.float64)
    raw = float(((z - mean) / np.maximum(std, 1.0e-6)) @ w)
    return float((raw - float(m["fit_positive_score_mean"])) / max(float(m["fit_positive_score_std"]), 1.0e-6))


def _retention_alpha(rsmr: dict[str, Any]) -> float:
    # Existing gate permits at most CAPTURE_TOL absolute loss over opportunity
    # scenes.  Conditional on an RSMR-selected true positive, this is exactly
    # the allowed false-veto probability budget below.
    base_capture = float(rsmr["positive_capture_rate"])
    if not (base_capture > CAPTURE_TOL > 0.0):
        raise ValueError("V48 invalid frozen capture/tolerance relation")
    return float(CAPTURE_TOL / base_capture)


def _conformal_threshold(cal: list[dict[str, Any]], m: dict[str, Any], alpha: float) -> tuple[float, dict[str, Any]]:
    vals = sorted(_risk(r, m) for r in cal if int(r["rsm_selected_action"]) >= 0 and float(r["rsm_selected_teacher_improvement"]) > 0.0)
    n = len(vals)
    if n < 16:
        raise ValueError("V48 calibration has too few frozen-policy positives")
    rank = min(max(int(math.ceil((n + 1) * (1.0 - alpha))), 1), n)
    tau = float(vals[rank - 1])
    return tau, {"positive_calibration_count": n, "conformal_rank": rank, "alpha": float(alpha), "threshold": tau}


def _nested(rows: list[dict[str, Any]], v47: dict[str, Any], audit_csv: Path) -> dict[str, Any]:
    v47n = v47["nested_crossfit"]
    rsmr = v47n["rsmr_rank_aggregate"]
    alpha = _retention_alpha(rsmr)
    arms = ["rsmr", "plan_control", "ego_ref"] + ARMS
    vals = {a: [] for a in arms}; caps = {a: 0 for a in arms}; noops = {a: 0 for a in arms}; oppsels = {a: 0 for a in arms}
    total_opp = total_noop = 0
    folds: list[dict[str, Any]] = []
    aud: list[dict[str, Any]] = []
    risk_sign_all = {a: [] for a in ARMS}; risk_y_all: list[float] = []; base_risk_all: list[float] = []
    better = {a: 0 for a in ARMS}

    for k in range(FOLDS):
        cf = (k + 1) % FOLDS
        fit = [r for r in rows if int(r["outer_test_fold"]) not in {k, cf}]
        cal = [r for r in rows if int(r["outer_test_fold"]) == cf]
        test = [r for r in rows if int(r["outer_test_fold"]) == k]
        models = {
            "sign_nomult": _fit_sign_ranker(fit, False),
            "sign_mult": _fit_sign_ranker(fit, True),
        }
        taus = {a: _conformal_threshold(cal, models[a], alpha) for a in ARMS}
        fv = {a: [] for a in arms}; fc = {a: 0 for a in arms}; fn = {a: 0 for a in arms}; fo = {a: 0 for a in arms}
        opp = sum(int(r["positive_opportunity"]) for r in test); noopsc = len(test) - opp
        fold_risk = {a: [] for a in ARMS}; fold_y: list[float] = []; fold_base: list[float] = []
        for r in test:
            has = bool(int(r["positive_opportunity"]))
            y = float(r["rsm_selected_teacher_improvement"])
            rsm_sel = int(r["rsm_selected_action"]) >= 0
            chosen = {
                "rsmr": rsm_sel,
                "plan_control": int(r["plan_control_selected_action"]) >= 0,
                "ego_ref": int(r["ego_ref_selected_action"]) >= 0,
                "sign_nomult": False,
                "sign_mult": False,
            }
            rs: dict[str, float] = {a: float("nan") for a in ARMS}
            if rsm_sel:
                for a in ARMS:
                    rs[a] = _risk(r, models[a]); chosen[a] = bool(rs[a] <= taus[a][0])
                fold_y.append(y); fold_base.append(-float(r["ego_ref_value"]))
                for a in ARMS: fold_risk[a].append(rs[a])
            for a in arms:
                if not chosen[a]: continue
                fv[a].append(y); fc[a] += int(has and y > 0.0); fn[a] += int(not has); fo[a] += int(has)
            aud.append({
                **r,
                "v48_sign_nomult_risk": rs["sign_nomult"], "v48_sign_mult_risk": rs["sign_mult"],
                "v48_sign_nomult_threshold": taus["sign_nomult"][0], "v48_sign_mult_threshold": taus["sign_mult"][0],
                "v48_sign_nomult_selected_action": int(r["rsm_selected_action"]) if chosen["sign_nomult"] else -1,
                "v48_sign_mult_selected_action": int(r["rsm_selected_action"]) if chosen["sign_mult"] else -1,
            })
        total_opp += opp; total_noop += noopsc
        fd: dict[str, Any] = {}
        for a in arms:
            fd[a] = _metrics(fv[a], fc[a], opp, fn[a], fo[a], noopsc)
            vals[a] += fv[a]; caps[a] += fc[a]; noops[a] += fn[a]; oppsels[a] += fo[a]
        fy = np.asarray(fold_y, dtype=np.float64); fb = np.asarray(fold_base, dtype=np.float64)
        diag: dict[str, Any] = {}
        for a in ARMS:
            rr = np.asarray(fold_risk[a], dtype=np.float64)
            auc = _auc(fy <= 0.0, rr); bauc = _auc(fy <= 0.0, fb)
            better[a] += int(math.isfinite(auc) and math.isfinite(bauc) and auc > bauc + 1.0e-12)
            diag[a] = {"selected_nonpositive_risk_auc": auc, "baseline_neg_ego_ref_value_auc": bauc, "auc_better": bool(auc > bauc + 1.0e-12)}
            risk_sign_all[a] += rr.tolist()
        risk_y_all += fy.tolist(); base_risk_all += fb.tolist()
        folds.append({
            "fold": k, "fit_scenes": len(fit), "value_calibration_scenes": len(cal), "test_scenes": len(test),
            **{a: fd[a] for a in arms},
            "risk_identification": diag,
            "calibration": {a: taus[a][1] for a in ARMS},
            "monotone_frozen_winner_contract_valid": True,
        })

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(aud[0].keys())); w.writeheader(); w.writerows(aud)
    A = {a: _metrics(vals[a], caps[a], total_opp, noops[a], oppsels[a], total_noop) for a in arms}
    gates = {a: _gate(A[a], A["rsmr"], folds, a) for a in ["plan_control", "ego_ref"] + ARMS}
    yy = np.asarray(risk_y_all, dtype=np.float64); bb = np.asarray(base_risk_all, dtype=np.float64)
    risk_id: dict[str, Any] = {}
    for a in ARMS:
        ss = np.asarray(risk_sign_all[a], dtype=np.float64)
        auc = _auc(yy <= 0.0, ss); bauc = _auc(yy <= 0.0, bb)
        risk_id[a] = {
            "aggregate_nonpositive_risk_auc": auc, "aggregate_baseline_neg_ego_ref_value_auc": bauc,
            "better_fold_count": int(better[a]),
            "identified": bool(math.isfinite(auc) and auc > bauc + 1.0e-12 and better[a] >= 4),
        }
    promotion = {
        "sign_nomult": bool(gates["sign_nomult"]["pass"] and risk_id["sign_nomult"]["identified"]),
        "sign_mult": bool(gates["sign_mult"]["pass"] and risk_id["sign_mult"]["identified"]),
    }
    preferred = "sign_nomult" if promotion["sign_nomult"] else ("sign_mult" if promotion["sign_mult"] else None)
    if preferred == "sign_nomult":
        diag = "selected_policy_sign_risk_is_sufficient_without_extremal_multiplicity"
    elif preferred == "sign_mult":
        diag = "extremal_selection_multiplicity_is_required_for_selected_policy_risk_retention"
    else:
        diag = "operator_conditioned_sign_risk_improves_identification_but_selected_zero_tail_functional_remains_insufficient"
    return {
        "folds": folds, "scene_audit_csv": str(audit_csv),
        "rsmr_rank_aggregate": A["rsmr"], "v45_plan_control_aggregate": A["plan_control"], "v47_ego_reference_aggregate": A["ego_ref"],
        "sign_nomult_aggregate": A["sign_nomult"], "sign_mult_aggregate": A["sign_mult"],
        "gates": gates, "risk_identification": risk_id,
        "retention_alpha": alpha, "retention_alpha_derivation": "CAPTURE_TOL / frozen_RSMR_capture_rate",
        "promotion_eligible": promotion, "preferred_promotion_arm": preferred,
        "train_gate_pass": bool(preferred is not None), "failure_diagnosis": diag,
        "monotone_frozen_winner_contract_valid": True,
    }


def _extract_plan_and_ego_params(plan_cfg_path: Path, ego_cfg_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_cfg = yaml.safe_load(plan_cfg_path.read_text()); ego_cfg = yaml.safe_load(ego_cfg_path.read_text())
    def sc(cfg): return cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    p = sc(plan_cfg); e = sc(ego_cfg)
    plan = {
        "names": list(p["post_selection_future_response_observable_names"]),
        "scales": [float(x) for x in p["post_selection_future_response_scales"]],
        "weights": [float(x) for x in p["post_selection_future_response_weights"]],
    }
    ego = {
        "names": list(e["post_selection_future_response_observable_names"]),
        "scales": [float(x) for x in e["post_selection_future_response_scales"]],
        "weights": [float(x) for x in e["post_selection_future_response_weights"]],
    }
    if plan["names"] != ["fsfr_plan_1d_occupancy_cost"] or ego["names"] != ["fsfr_plan_1d_occupancy_cost", "fsfr_predicted_demo_cost"]:
        raise RuntimeError("V48 ENGINEERING STOP: V47 PLAN/EGOREF config schemas changed")
    return plan_cfg, plan, ego


def _decorate(base_cfg: dict[str, Any], plan: dict[str, Any], ego: dict[str, Any], risk: dict[str, Any], tau: float, out: Path, arm: str) -> None:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False))
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    sc = ic["selection_conditioned_intervention_recovery"]
    sc["post_selection_value_mode"] = "endpoint_potential_quality_operator_conditioned_risk_retention"
    sc["selected_policy_risk_plan_response_names"] = list(plan["names"])
    sc["selected_policy_risk_plan_response_scales"] = list(plan["scales"])
    sc["selected_policy_risk_plan_response_weights"] = list(plan["weights"])
    sc["selected_policy_risk_ego_reference_names"] = list(ego["names"])
    sc["selected_policy_risk_ego_reference_scales"] = list(ego["scales"])
    sc["selected_policy_risk_ego_reference_weights"] = list(ego["weights"])
    sc["operator_conditioned_risk_retention"] = {
        "feature_names": list(OCRR_STATE_NAMES), "aggregation": "sign_only",
        "use_extremal_multiplicity": bool(risk["use_extremal_multiplicity"]),
        "components": {"sign_risk": risk}, "retention_threshold": float(tau),
        "threshold_calibration": "TRAIN_selected_positive_split_conformal_from_frozen_capture_tolerance_no_sweep",
    }
    sc["post_selection_selected_bias"] = 0.0
    sc["post_selection_value_training"] = "selected_RSMR_pairwise_sign_risk_fixed_lambda_1_plus_operator_multiplicity_ablation"
    sc["post_selection_operator"] = "freeze_RSMR_winner_then_OCRR_veto_only_same_winner_or_incumbent_no_rerank_no_fallback"
    cfg.setdefault("metadata", {})["algorithm_version"] = f"V64.3.48-EAF-ICER-OCRR-{arm.upper()}"
    cfg.setdefault("provenance", {})["algorithm_version"] = f"V64.3.48-EAF-ICER-OCRR-{arm.upper()}"
    cfg.setdefault("experiment", {})["name"] = f"v64_3_48_eaf_icer_ocrr_{arm}"
    cfg["experiment"]["algorithm"] = f"V64.3.48-EAF-ICER-OCRR-{arm.upper()}"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _final_model(rows: list[dict[str, Any]], use_mult: bool, alpha: float) -> tuple[dict[str, Any], float, dict[str, Any]]:
    # Keep one deterministic calibration fold out of the selected-risk fit so
    # final fresh deployment retains calibration independence.
    fit = [r for r in rows if int(r["outer_test_fold"]) != 0]
    cal = [r for r in rows if int(r["outer_test_fold"]) == 0]
    model = _fit_sign_ranker(fit, use_mult)
    tau, info = _conformal_threshold(cal, model, alpha)
    info["final_fit_folds"] = [1, 2, 3, 4]; info["final_calibration_fold"] = 0
    return model, tau, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v47-fit-report", required=True); ap.add_argument("--v47-scene-audit", required=True)
    ap.add_argument("--v47-plan-config", required=True); ap.add_argument("--v47-ego-ref-config", required=True)
    ap.add_argument("--output-sign-nomult-config", required=True); ap.add_argument("--output-sign-mult-config", required=True)
    ap.add_argument("--output-report", required=True); ap.add_argument("--output-scene-audit", required=True)
    a = ap.parse_args()

    v47 = _check_v47(Path(a.v47_fit_report)); rows = _read_audit(Path(a.v47_scene_audit))
    nested = _nested(rows, v47, Path(a.output_scene_audit))
    # Hard replay: this layer must not alter historical policies before OCRR.
    for key, expkey in [("rsmr_rank_aggregate", "rsmr_rank_aggregate"), ("v45_plan_control_aggregate", "v45_plan_control_aggregate"), ("v47_ego_reference_aggregate", "ego_reference_aggregate")]:
        got = _sig(nested[key]); exp = V47_EXPECTED[expkey]
        if any(got[i] != exp[i] for i in range(4)) or abs(got[4] - exp[4]) > 1.0e-9:
            raise RuntimeError(f"V48 ENGINEERING STOP: historical replay changed {key}: {got}")

    plan_cfg, plan, ego = _extract_plan_and_ego_params(Path(a.v47_plan_config), Path(a.v47_ego_ref_config))
    alpha = float(nested["retention_alpha"])
    m0, t0, i0 = _final_model(rows, False, alpha); m1, t1, i1 = _final_model(rows, True, alpha)
    _decorate(plan_cfg, plan, ego, m0, t0, Path(a.output_sign_nomult_config), "sign-nomult")
    _decorate(plan_cfg, plan, ego, m1, t1, Path(a.output_sign_mult_config), "sign-mult")

    report = {
        "audit": "v64_3_48_eaf_icer_ocrr_fit",
        "scientific_role": "TRAIN_only_operator_functional_after_V47_preregistered_representation_stop",
        "mechanism_hypothesis": "The absolute outcome law of a frozen extremal RSMR proposal is post-selection shifted by admissible challenger multiplicity. Use already validated QUALITY/PLAN/EGOREF consequence coordinates only as fixed context, learn selected-policy sign risk rather than an all-edge signed mean, and calibrate a veto-only retention threshold from the existing capture tolerance.",
        "nested_crossfit": nested, "train_gate_pass": nested["train_gate_pass"],
        "final_runtime_fit": {"sign_nomult": i0, "sign_mult": i1},
        "train_gate_contract": {
            "V47_failure_signature_is_exact_hard_gate": True,
            "representation_expansion_after_V47_is_forbidden": True,
            "RSMR_is_sole_challenger_selector": True,
            "same_winner_veto_only_no_rerank_no_fallback": True,
            "candidate_bank_and_candidate_count_are_not_changed_or_swept": True,
            "multiplicity_is_only_the_observed_size_of_the_existing_admissible_challenger_set": True,
            "no_new_future_state_observable": True,
            "no_catastrophe_classifier": True,
            "no_selected_translation_or_free_threshold_sweep": True,
            "retention_threshold_is_split_conformal_and_derived_from_existing_capture_tolerance": True,
            "lambda": RIDGE_LAMBDA, "capture_tolerance": CAPTURE_TOL,
            "noop_reduction_min": NOOP_REDUCTION_MIN, "catastrophe_reduction_min": CATASTROPHE_REDUCTION_MIN,
        },
    }
    Path(a.output_report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": nested["train_gate_pass"], "preferred_promotion_arm": nested["preferred_promotion_arm"], "failure_diagnosis": nested["failure_diagnosis"]}, sort_keys=True))
    if not nested["train_gate_pass"]:
        raise SystemExit(f"V64.3.48 OCRR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before fresh selection")


if __name__ == "__main__":
    main()
