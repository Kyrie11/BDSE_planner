from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.planner.distributional_interaction_response import DIRP_OBSERVABLE_NAMES
from bdse.planner.interaction_response_field import RESPONSE_FIELD_OBSERVABLE_NAMES
from bdse.planner.response_value_observables import FUTURE_RESPONSE_OBSERVABLE_NAMES
from bdse.planner.value_observables import QUALITY_NAMES, VALUE_OBSERVABLE_NAMES
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS, RIDGE_LAMBDA, _fold, _read_edges, _select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _fit_regret_structured_margin, _structured_scores
from bdse.tools.fit_v64_3_38_eaf_icer_davr import CAPTURE_TOL, CATASTROPHE_REDUCTION_MIN, NOOP_REDUCTION_MIN, _value_diag, _write_rsmr
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import _base_cfg, _fit_zero_ridge, _pred as _epv_pred
from bdse.tools.fit_v64_3_43_eaf_icer_cfrv import _fit_weighted_zero_ridge, _gate, _metrics, _quality_value, _quality_x, _scene as _v43_scene, _write_quality_control

ALL_OBSERVABLE_NAMES = list(VALUE_OBSERVABLE_NAMES) + list(FUTURE_RESPONSE_OBSERVABLE_NAMES) + list(RESPONSE_FIELD_OBSERVABLE_NAMES) + list(DIRP_OBSERVABLE_NAMES)
ARM_FEATURES = {
    "plan_control": ["dirp_plan_mean_occupancy_cost"],
    "dist_mean": ["dirp_distribution_mean_occupancy_cost"],
    "temporal_profile": [
        "dirp_plan_mean_occupancy_cost",
        "dirp_plan_peak_occupancy_cost",
        "dirp_plan_early_occupancy_cost",
        "dirp_plan_second_moment_occupancy_cost",
    ],
    "dirp_joint": [
        "dirp_distribution_mean_occupancy_cost",
        "dirp_distribution_peak_occupancy_cost",
        "dirp_distribution_early_occupancy_cost",
        "dirp_distribution_second_moment_occupancy_cost",
    ],
}


def _read_sidecar(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = str(r["scenario_token"])
        names = [str(x) for x in r["observable_names"]]
        if names != DIRP_OBSERVABLE_NAMES:
            raise ValueError(f"V46 sidecar schema mismatch {t}")
        c = np.asarray(r["costs"], dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != len(DIRP_OBSERVABLE_NAMES) or not np.all(np.isfinite(c)):
            raise ValueError(f"V46 invalid sidecar costs {t}")
        if t in out:
            raise ValueError(f"duplicate V46 sidecar {t}")
        out[t] = c
    return out


def _scene(groups: dict[str, list[dict[str, Any]]], side: dict[str, np.ndarray]) -> dict[str, list[dict[str, Any]]]:
    base = _v43_scene(groups)
    out: dict[str, list[dict[str, Any]]] = {}
    for t, ss in base.items():
        if t not in side:
            raise ValueError(f"V46 sidecar missing {t}")
        c = side[t]
        inc = int(groups[t][0].get("raw_top_action", -1))
        if not (0 <= inc < len(c)):
            raise ValueError(f"V46 incumbent out of sidecar range {t}")
        rr: list[dict[str, Any]] = []
        for a in ss:
            act = int(a["action"])
            if not (0 <= act < len(c)):
                raise ValueError(f"V46 action out of sidecar range {t}/{act}")
            z = dict(a)
            for j, n in enumerate(DIRP_OBSERVABLE_NAMES):
                z[n + "_improvement"] = float(c[inc, j] - c[act, j])
            rr.append(z)
        out[t] = rr
    return out


def _rx(a: dict[str, Any], names: list[str]) -> np.ndarray:
    return np.asarray([float(a[n + "_improvement"]) for n in names], dtype=np.float64)


def _rval(a: dict[str, Any], epv: dict[str, Any], q: dict[str, Any], r: dict[str, Any], names: list[str]) -> float:
    base = _quality_value(a, epv, q)
    x = _rx(a, names)
    return float(np.clip(base + (x / np.maximum(np.asarray(r["scale"]), 1.0e-6)) @ np.asarray(r["weights"]), -40.0, 40.0))


def _check_v45(path: Path, response_report: Path) -> None:
    r = json.loads(path.read_text())
    n = r.get("nested_crossfit", {})
    exp = {
        "rsmr_rank_aggregate": (502, 221, 107, 28, 43.29405361274824),
        "quality_control_aggregate": (205, 129, 30, 13, 43.905547394411805),
        "cv_occupancy_aggregate": (220, 120, 43, 13, 45.20842296723279),
        "local_response_field_aggregate": (218, 122, 37, 9, 54.57972428889805),
        "plan_response_field_aggregate": (217, 121, 38, 9, 56.55117310290402),
    }
    for k, e in exp.items():
        d = n.get(k, {})
        got = (d.get("selected_count"), d.get("selected_positive_count"), d.get("no_positive_opportunity_false_intervention_count"), d.get("catastrophic_count"), d.get("teacher_improvement_sum"))
        if any(got[i] != e[i] for i in range(4)) or abs(float(got[4]) - e[4]) > 1.0e-9:
            raise RuntimeError(f"V46 ENGINEERING STOP: V45 signature mismatch {k}: {got}")
    if r.get("train_gate_pass") is not False or n.get("failure_diagnosis") != "plan_conditioned_response_is_identifiable_but_absolute_zero_or_remaining_consequence_family_is_insufficient":
        raise RuntimeError("V46 ENGINEERING STOP: V45 scientific-stop signature changed")
    rr = json.loads(response_report.read_text())
    a = rr.get("aggregate", {})
    target = (0.30137842796229286, 0.12486025654085724, 0.12385468573917016)
    got = (float(a.get("cv_mse", -1)), float(a.get("local_mse", -1)), float(a.get("plan_mse", -1)))
    if any(abs(x - y) > 1.0e-12 for x, y in zip(got, target)) or int(rr.get("plan_better_than_local_fold_count", 0)) != 5:
        raise RuntimeError(f"V46 ENGINEERING STOP: V45 response-identification signature changed {got}")


def _nested(groups: dict[str, list[dict[str, Any]]], side: dict[str, np.ndarray], audit_csv: Path, dist_report: dict[str, Any]) -> dict[str, Any]:
    scene = _scene(groups, side)
    arms = ["rsmr", "quality"] + list(ARM_FEATURES)
    agg = {a: [] for a in arms}; caps = {a: 0 for a in arms}; noops = {a: 0 for a in arms}; oppsels = {a: 0 for a in arms}
    total_opp = total_noop = 0; folds = []; aud = []; vy = []; vp = {a: [] for a in arms}
    for k in range(FOLDS):
        test = [t for t in scene if _fold(t) == k]
        cf = (k + 1) % FOLDS
        fit = [t for t in scene if _fold(t) not in {k, cf}]
        cal = [t for t in scene if _fold(t) == cf]
        rsm = _fit_regret_structured_margin(scene, fit)
        epv = _fit_zero_ridge(scene, fit, "epv")
        q = _fit_weighted_zero_ridge(scene, fit, lambda a: float(a["y"]) - _epv_pred(a, epv), _quality_x, QUALITY_NAMES)
        rs = {
            arm: _fit_weighted_zero_ridge(scene, fit, lambda a: float(a["y"]) - _quality_value(a, epv, q), lambda a, nn=names: _rx(a, nn), names)
            for arm, names in ARM_FEATURES.items()
        }
        fv = {a: [] for a in arms}; fc = {a: 0 for a in arms}; fn = {a: 0 for a in arms}; fo = {a: 0 for a in arms}
        opp = noopsc = 0; subset = identity = True
        for t in test:
            ss = scene[t]; yy = np.asarray([float(a["y"]) for a in ss]); has = bool(np.any(yy > 0)); opp += int(has); noopsc += int(not has)
            score = _structured_scores(ss, rsm); idx = _select(ss, score)
            vals = {a: float("nan") for a in arms}; chosen = {a: None for a in arms}; chosen["rsmr"] = idx
            if idx is not None:
                row = ss[idx]; vals["rsmr"] = float(score[idx]); qv = _quality_value(row, epv, q); vals["quality"] = qv; chosen["quality"] = idx if qv > 0 else None
                for arm, names in ARM_FEATURES.items():
                    v = _rval(row, epv, q, rs[arm], names); vals[arm] = v; chosen[arm] = idx if v > 0 else None
                vy.append(float(yy[idx])); [vp[n].append(float(vals[n])) for n in arms]
            subset = subset and all(chosen[n] is None or idx is not None for n in arms if n != "rsmr")
            identity = identity and all(chosen[n] is None or chosen[n] == idx for n in arms if n != "rsmr")
            for n, ii in chosen.items():
                if ii is None: continue
                v = float(yy[ii]); fv[n].append(v); fc[n] += int(has and v > 0); fn[n] += int(not has); fo[n] += int(has)
            aud.append({
                "scenario_token": t, "outer_test_fold": k, "calibration_fold": cf, "candidate_count": len(ss), "positive_opportunity": int(has),
                "rsm_selected_action": -1 if idx is None else int(ss[idx]["action"]), "rsm_selected_teacher_improvement": float("nan") if idx is None else float(yy[idx]),
                **{f"{n}_selected_action": -1 if chosen[n] is None else int(ss[chosen[n]]["action"]) for n in arms if n != "rsmr"},
                **{f"{n}_value": float(vals[n]) for n in arms if n != "rsmr"},
            })
        total_opp += opp; total_noop += noopsc; fd = {}
        for n in arms:
            fd[n] = _metrics(fv[n], fc[n], opp, fn[n], fo[n], noopsc); agg[n] += fv[n]; caps[n] += fc[n]; noops[n] += fn[n]; oppsels[n] += fo[n]
        folds.append({"fold": k, "fit_scenes": len(fit), "value_calibration_scenes": len(cal), "test_scenes": len(test), **{n: fd[n] for n in arms}, "monotone_subset_valid": subset, "frozen_winner_identity_valid": identity})
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(aud[0])); w.writeheader(); w.writerows(aud)
    A = {n: _metrics(agg[n], caps[n], total_opp, noops[n], oppsels[n], total_noop) for n in arms}
    vd = {n: _value_diag(vy, vp[n]) for n in arms}
    g = {n: _gate(A[n], A["rsmr"], folds, n) for n in arms if n != "rsmr"}
    contracts = all(f["monotone_subset_valid"] and f["frozen_winner_identity_valid"] for f in folds)
    dist_ident = bool(dist_report.get("local_second_moment_identified", False) and dist_report.get("plan_second_moment_identified", False))
    promotion = {
        "dist_mean": bool(g["dist_mean"]["pass"] and dist_ident),
        "temporal_profile": bool(g["temporal_profile"]["pass"]),
        "dirp_joint": bool(g["dirp_joint"]["pass"] and dist_ident),
    }
    preferred = "dist_mean" if promotion["dist_mean"] else ("temporal_profile" if promotion["temporal_profile"] else ("dirp_joint" if promotion["dirp_joint"] else None))
    if preferred == "dist_mean": diag = "identified_response_distribution_is_sufficient_beyond_point_response"
    elif preferred == "temporal_profile": diag = "temporal_interaction_profile_is_sufficient_beyond_scalar_time_mean"
    elif preferred == "dirp_joint": diag = "response_distribution_and_temporal_profile_are_jointly_required"
    elif dist_ident: diag = "response_second_moment_is_identifiable_but_acceleration_distribution_or_interaction_profile_still_not_absolute_value_sufficient"
    else: diag = "response_second_moment_not_identified_close_acceleration_distribution_and_move_to_general_trajectory_response"
    return {
        "folds": folds,
        "scene_audit_csv": str(audit_csv),
        "rsmr_rank_aggregate": A["rsmr"],
        "quality_control_aggregate": A["quality"],
        "v45_plan_control_aggregate": A["plan_control"],
        "distribution_mean_aggregate": A["dist_mean"],
        "temporal_profile_aggregate": A["temporal_profile"],
        "dirp_joint_aggregate": A["dirp_joint"],
        "selected_proposal_value_prediction_diagnostics": vd,
        "gates": g,
        "distribution_identification": {"identified": dist_ident, "crossfit_report": dist_report},
        "promotion_eligible": promotion,
        "preferred_promotion_arm": preferred,
        "monotone_frozen_winner_contract_valid": contracts,
        "train_gate_pass": bool(contracts and preferred is not None),
        "failure_diagnosis": diag,
    }


def _decorate(rsmcfg: dict[str, Any], epv: dict[str, Any], q: dict[str, Any], residual: dict[str, Any], models: dict[str, Any], response_names: list[str], path: str, version: str) -> None:
    cfg = yaml.safe_load(yaml.safe_dump(rsmcfg, sort_keys=False))
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic["instrument_value_observables"] = True
    ic["instrument_future_response_observables"] = True
    ic["instrument_plan_conditioned_response_observables"] = False
    ic["instrument_interaction_response_field_observables"] = True
    ic["instrument_distributional_interaction_response_observables"] = True
    sc = ic["selection_conditioned_intervention_recovery"]
    sc.update({
        "post_selection_value_enabled": True,
        "post_selection_value_mode": "endpoint_potential_quality_distributional_response_profile",
        "post_selection_endpoint_feature_names": list(epv["names"]),
        "post_selection_endpoint_feature_scale": [float(x) for x in epv["scale"]],
        "post_selection_endpoint_weights": [float(x) for x in epv["weights"]],
        "post_selection_endpoint_bias": 0.0,
        "post_selection_observable_names": list(ALL_OBSERVABLE_NAMES),
        "post_selection_quality_observable_names": list(QUALITY_NAMES),
        "post_selection_quality_observable_scale": [float(x) for x in q["scale"]],
        "post_selection_quality_observable_weights": [float(x) for x in q["weights"]],
        "post_selection_future_response_observable_names": list(response_names),
        "post_selection_future_response_scales": [float(x) for x in residual["scale"]],
        "post_selection_future_response_weights": [float(x) for x in residual["weights"]],
        "post_selection_selected_bias": 0.0,
        "interaction_response_field": models["mean_response_model"],
        "distributional_interaction_response_field": models["second_moment_model"],
        "post_selection_value_training": "scene_equal_all_edge_EPV_plus_QUALITY_plus_DIRP_response_profile_fixed_lambda_1",
        "post_selection_operator": "freeze_RSMR_winner_then_DIRP_value_accept_same_winner_iff_positive_else_incumbent_no_rerank_no_fallback",
    })
    cfg.setdefault("metadata", {})["algorithm_version"] = version
    cfg.setdefault("provenance", {})["algorithm_version"] = version
    cfg.setdefault("experiment", {})["name"] = version.lower().replace(".", "_").replace("-", "_")
    cfg["experiment"]["algorithm"] = version
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frontier-edges", required=True)
    ap.add_argument("--dirp-sidecar", required=True)
    ap.add_argument("--dirp-model", required=True)
    ap.add_argument("--dirp-response-report", required=True)
    ap.add_argument("--v45-fit-report", required=True)
    ap.add_argument("--v45-response-report", required=True)
    ap.add_argument("--base-config", required=True)
    for n in ["rsmr", "quality", "plan_control", "dist_mean", "temporal_profile", "dirp_joint"]:
        ap.add_argument(f"--output-{n.replace('_','-')}-config", dest=f"output_{n}_config", required=True)
    ap.add_argument("--output-report", required=True)
    ap.add_argument("--output-scene-audit", required=True)
    a = ap.parse_args()

    _check_v45(Path(a.v45_fit_report), Path(a.v45_response_report))
    _, groups = _read_edges(Path(a.train_frontier_edges))
    side = _read_sidecar(Path(a.dirp_sidecar))
    dr = json.loads(Path(a.dirp_response_report).read_text())
    nested = _nested(groups, side, Path(a.output_scene_audit), dr)
    pc = nested["v45_plan_control_aggregate"]
    expected = (217, 121, 38, 9, 56.55117310290402)
    got = (pc.get("selected_count"), pc.get("selected_positive_count"), pc.get("no_positive_opportunity_false_intervention_count"), pc.get("catastrophic_count"), pc.get("teacher_improvement_sum"))
    if any(got[i] != expected[i] for i in range(4)) or abs(float(got[4]) - expected[4]) > 1.0e-9:
        raise RuntimeError(f"V46 ENGINEERING STOP: V45 PLAN control failed exact replay {got}")

    report = {
        "audit": "v64_3_46_eaf_icer_dirp_fit",
        "scientific_role": "TRAIN_only_distributional_response_and_temporal_interaction_profile_after_frozen_RSMR_QUALITY_and_V45_mean_response",
        "frozen_train_scenes": len(groups),
        "ridge_lambda": RIDGE_LAMBDA,
        "mechanism_hypothesis": "V45 identifies an agent-local plan-conditioned response mean, but deterministic mean response loses V44 ensemble-only material opportunities and scalar time-mean occupancy leaves the absolute zero unresolved. V46 independently tests conditional response second moment and temporal hazard-profile sufficiency.",
        "nested_crossfit": nested,
        "train_gate_pass": nested["train_gate_pass"],
        "train_gate_contract": {
            "V45_failure_signature_is_exact_hard_gate": True,
            "V45_plan_response_mean_is_exact_control": True,
            "RSMR_is_sole_challenger_selector": True,
            "DIRP_second_moment_supervision_is_TRAIN_only_and_uses_no_teacher_value": True,
            "deployment_uses_no_logged_future": True,
            "fixed_three_point_moment_matching_quadrature_no_tuning": True,
            "temporal_profile_uses_no_threshold_or_bandwidth": True,
            "no_selected_translation_or_CVaR_tuning": True,
            "capture_tolerance": CAPTURE_TOL,
            "noop_reduction_min": NOOP_REDUCTION_MIN,
            "catastrophe_reduction_min": CATASTROPHE_REDUCTION_MIN,
        },
    }
    Path(a.output_report).write_text(json.dumps(report, indent=2, sort_keys=True))

    scene = _scene(groups, side)
    base = _base_cfg(a.base_config)
    base["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["instrument_value_observables"] = True
    rsm = _fit_regret_structured_margin(scene, list(scene)); rsmcfg = _write_rsmr(base, a.output_rsmr_config, rsm); Path(a.output_rsmr_config).write_text(yaml.safe_dump(rsmcfg, sort_keys=False))
    epv = _fit_zero_ridge(scene, list(scene), "epv")
    q = _fit_weighted_zero_ridge(scene, list(scene), lambda x: float(x["y"]) - _epv_pred(x, epv), _quality_x, QUALITY_NAMES)
    _write_quality_control(rsmcfg, epv, q, a.output_quality_config)
    models = json.loads(Path(a.dirp_model).read_text())
    for arm, names in ARM_FEATURES.items():
        res = _fit_weighted_zero_ridge(scene, list(scene), lambda x: float(x["y"]) - _quality_value(x, epv, q), lambda x, nn=names: _rx(x, nn), names)
        _decorate(rsmcfg, epv, q, res, models, names, getattr(a, f"output_{arm}_config"), f"V64.3.46-EAF-ICER-DIRP-{arm.upper()}")

    print(json.dumps({"pass": nested["train_gate_pass"], "preferred_promotion_arm": nested["preferred_promotion_arm"], "failure_diagnosis": nested["failure_diagnosis"]}, sort_keys=True))
    if not nested["train_gate_pass"]:
        raise SystemExit(f"V64.3.46 DIRP nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before fresh selection")


if __name__ == "__main__":
    main()
