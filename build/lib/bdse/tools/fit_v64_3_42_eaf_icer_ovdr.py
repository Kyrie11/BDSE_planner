from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.planner.value_observables import QUALITY_DIM, VALUE_OBSERVABLE_NAMES
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS, RIDGE_LAMBDA, _finite, _fold, _read_edges, _select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _extended_diag, _fit_regret_structured_margin, _structured_scores
from bdse.tools.fit_v64_3_38_eaf_icer_davr import (
    CAPTURE_TOL,
    CATASTROPHE_REDUCTION_MIN,
    MIN_VALUE_CAL_PROPOSALS,
    NOOP_REDUCTION_MIN,
    _value_diag,
    _write_rsmr,
)
from bdse.tools.fit_v64_3_39_eaf_icer_cfsr import _fit_translation
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import (
    EPV_NAMES,
    _base_cfg,
    _endpoint_scene,
    _fit_zero_ridge,
    _pred as _epv_pred,
)

EPS = 1.0e-12


def _scene_with_observables(groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    scene = _endpoint_scene(groups)
    out: dict[str, list[dict[str, Any]]] = {}
    keys = [f"icer_value_observable_{n}" for n in VALUE_OBSERVABLE_NAMES]
    for tok, ss in scene.items():
        rows = groups[tok]
        inc = int(rows[0].get("raw_top_action", -1))
        byact = {int(r.get("challenger_action", -2)): r for r in rows}
        ir = byact.get(inc)
        if ir is None:
            continue
        try:
            oi = np.asarray([_finite(ir, k) for k in keys], dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(f"V42 missing incumbent deployment observables for scene {tok}") from exc
        if oi.size != len(VALUE_OBSERVABLE_NAMES) or not np.all(np.isfinite(oi)):
            raise RuntimeError(f"V42 incumbent observable schema invalid for scene {tok}")
        rr = []
        for a in ss:
            row = byact.get(int(a["action"]))
            if row is None:
                raise RuntimeError("V42 deployment observable replay missing candidate row")
            try:
                ob = np.asarray([_finite(row, k) for k in keys], dtype=np.float64)
            except Exception as exc:
                raise RuntimeError(f"V42 missing candidate deployment observables for scene {tok}") from exc
            if ob.size != len(VALUE_OBSERVABLE_NAMES) or not np.all(np.isfinite(ob)):
                raise RuntimeError(f"V42 candidate observable schema invalid for scene {tok}")
            b = dict(a)
            b["observable_inc"] = oi.copy()
            b["observable_cand"] = ob
            # Every observable is lower-is-better.  Positive means the candidate
            # improves the deployment-observable consequence relative to incumbent.
            b["observable_improvement"] = oi - ob
            rr.append(b)
        out[tok] = rr
    return out


def _obs_feature(a: dict[str, Any], block: str) -> np.ndarray:
    x = np.asarray(a["observable_improvement"], dtype=np.float64)
    if x.size != len(VALUE_OBSERVABLE_NAMES):
        raise ValueError("V42 deployment observable vector has wrong dimension")
    if block == "quality":
        return x[:QUALITY_DIM]
    if block == "risk":
        return x[QUALITY_DIM:]
    if block == "joint":
        return x
    raise ValueError(block)


def _fit_observable_residual(
    scene: dict[str, list[dict[str, Any]]],
    tokens: list[str],
    epv: dict[str, Any],
    block: str,
) -> dict[str, Any]:
    X = []
    y = []
    w = []
    for tok in tokens:
        ss = scene[tok]
        n = len(ss)
        if n <= 0:
            continue
        for a in ss:
            X.append(_obs_feature(a, block))
            y.append(float(a["y"]) - _epv_pred(a, epv))
            w.append(1.0 / n)
    if not X:
        raise ValueError("V42 observable residual fit has no samples")
    X = np.stack(X).astype(np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    # Moment normalization may use global normalized weights because a common
    # scalar leaves moments invariant.  The ridge objective itself uses the
    # *unnormalized* scene-total-one weights, preserving the V32.1 correction.
    pm = w / max(float(w.sum()), EPS)
    scale = np.sqrt(np.sum((X * X) * pm[:, None], axis=0))
    scale = np.maximum(scale, 1.0e-6)
    Z = X / scale[None, :]
    root = np.sqrt(w)[:, None]
    Zw = Z * root
    yw = y * root[:, 0]
    coef = np.linalg.solve(Zw.T @ Zw + np.eye(Z.shape[1]) * RIDGE_LAMBDA, Zw.T @ yw)
    names = (
        VALUE_OBSERVABLE_NAMES[:QUALITY_DIM]
        if block == "quality"
        else VALUE_OBSERVABLE_NAMES[QUALITY_DIM:]
        if block == "risk"
        else VALUE_OBSERVABLE_NAMES
    )
    return {
        "block": block,
        "names": list(names),
        "scale": scale,
        "weights": coef,
        "bias": 0.0,
        "sample_count": int(len(y)),
        "scene_weight_sum": float(w.sum()),
    }


def _obs_residual(a: dict[str, Any], model: dict[str, Any]) -> float:
    x = _obs_feature(a, str(model["block"]))
    return float((x / np.maximum(np.asarray(model["scale"], dtype=np.float64), 1.0e-6)) @ np.asarray(model["weights"], dtype=np.float64))


def _value(a: dict[str, Any], epv: dict[str, Any], obs: dict[str, Any] | None = None, shift: float = 0.0) -> float:
    v = _epv_pred(a, epv)
    if obs is not None:
        v += _obs_residual(a, obs)
    return float(np.clip(v + float(shift), -40.0, 40.0))


def _metrics(vals, captured, opp, noop_selected, opp_selected, noop_scenes):
    return _extended_diag(vals, captured, opp, noop_selected, opp_selected, noop_scenes)


def _gate(m, r, folds, key):
    existence = (
        m["no_positive_opportunity_false_intervention_count"]
        <= (1.0 - NOOP_REDUCTION_MIN) * r["no_positive_opportunity_false_intervention_count"] + EPS
        and m["positive_capture_rate"] >= r["positive_capture_rate"] - CAPTURE_TOL - EPS
    )
    tail = (
        m["catastrophic_count"] <= (1.0 - CATASTROPHE_REDUCTION_MIN) * r["catastrophic_count"] + EPS
        and m["teacher_negative_rms"] <= r["teacher_negative_rms"] + EPS
        and m["teacher_improvement_sum"] >= -EPS
    )
    all_folds = all(float(f[key]["teacher_improvement_sum"]) >= -EPS for f in folds)
    population = m["selected_count"] >= 64 and m["selected_positive_count"] >= 32
    return {
        "existence_and_capture": bool(existence),
        "tail": bool(tail),
        "all_folds_sum_nonnegative": bool(all_folds),
        "population": bool(population),
        "pass": bool(existence and tail and all_folds and population),
    }


def _eval(ss, rsm, epv, q, risk, joint, shift):
    score = _structured_scores(ss, rsm)
    idx = _select(ss, score)
    names = ["rsmr", "epv_raw", "quality", "risk", "joint_raw", "joint_main"]
    if idx is None:
        return {n: (None, float("nan")) for n in names}
    ev = _value(ss[idx], epv)
    qv = _value(ss[idx], epv, q)
    rv = _value(ss[idx], epv, risk)
    jv = _value(ss[idx], epv, joint)
    mv = _value(ss[idx], epv, joint, shift)
    return {
        "rsmr": (idx, float(score[idx])),
        "epv_raw": (idx if ev > 0 else None, ev),
        "quality": (idx if qv > 0 else None, qv),
        "risk": (idx if rv > 0 else None, rv),
        "joint_raw": (idx if jv > 0 else None, jv),
        "joint_main": (idx if mv > 0 else None, mv),
    }


def _nested(groups, audit_csv: Path):
    scene = _scene_with_observables(groups)
    names = ["rsmr", "epv_raw", "quality", "risk", "joint_raw", "joint_main"]
    agg = {n: [] for n in names}
    caps = {n: 0 for n in names}
    noops = {n: 0 for n in names}
    oppsels = {n: 0 for n in names}
    total_opp = total_noop = 0
    folds = []
    audits = []
    vy = []
    vp = {n: [] for n in names}

    for k in range(FOLDS):
        test = [t for t in scene if _fold(t) == k]
        cf = (k + 1) % FOLDS
        cal = [t for t in scene if _fold(t) == cf]
        fit = [t for t in scene if _fold(t) not in {k, cf}]
        rsm = _fit_regret_structured_margin(scene, fit)
        epv = _fit_zero_ridge(scene, fit, "epv")
        q = _fit_observable_residual(scene, fit, epv, "quality")
        risk = _fit_observable_residual(scene, fit, epv, "risk")
        joint = _fit_observable_residual(scene, fit, epv, "joint")

        cy = []
        cp = []
        used = []
        for t in cal:
            ss = scene[t]
            score = _structured_scores(ss, rsm)
            idx = _select(ss, score)
            if idx is None:
                continue
            cy.append(float(ss[idx]["y"]))
            cp.append(_value(ss[idx], epv, joint))
            used.append(t)
        if len(used) < MIN_VALUE_CAL_PROPOSALS:
            raise ValueError(f"V42 calibration proposals {len(used)} < {MIN_VALUE_CAL_PROPOSALS}")
        shift_fit = _fit_translation(np.asarray(cp), np.asarray(cy), "endpoint_plus_joint_observable")
        shift = float(shift_fit["selected_policy_bias"])

        fv = {n: [] for n in names}
        fc = {n: 0 for n in names}
        fn = {n: 0 for n in names}
        fo = {n: 0 for n in names}
        opp = noopsc = 0
        subset = identity = True
        for t in test:
            ss = scene[t]
            yy = np.asarray([float(a["y"]) for a in ss])
            has = bool(np.any(yy > 0.0))
            opp += int(has)
            noopsc += int(not has)
            ev = _eval(ss, rsm, epv, q, risk, joint, shift)
            ridx = ev["rsmr"][0]
            if ridx is not None:
                vy.append(float(yy[ridx]))
                for n in names:
                    vp[n].append(float(ev[n][1]))
            chosen = {n: ev[n][0] for n in names}
            subset = subset and all(chosen[n] is None or ridx is not None for n in names if n != "rsmr")
            identity = identity and all(chosen[n] is None or chosen[n] == ridx for n in names if n != "rsmr")
            for n, idx in chosen.items():
                if idx is None:
                    continue
                val = float(yy[idx])
                fv[n].append(val)
                fc[n] += int(has and val > 0.0)
                fn[n] += int(not has)
                fo[n] += int(has)
            audits.append(
                {
                    "scenario_token": t,
                    "outer_test_fold": k,
                    "calibration_fold": cf,
                    "candidate_count": len(ss),
                    "positive_opportunity": int(has),
                    "rsm_selected_action": -1 if ridx is None else int(ss[ridx]["action"]),
                    "rsm_selected_score": float(ev["rsmr"][1]),
                    "rsm_selected_teacher_improvement": float("nan") if ridx is None else float(yy[ridx]),
                    **{f"{n}_selected_action": -1 if chosen[n] is None else int(ss[chosen[n]]["action"]) for n in names if n != "rsmr"},
                    **{f"{n}_value": float(ev[n][1]) for n in names if n != "rsmr"},
                }
            )
        total_opp += opp
        total_noop += noopsc
        fd = {}
        for n in names:
            fd[n] = _metrics(fv[n], fc[n], opp, fn[n], fo[n], noopsc)
            agg[n] += fv[n]
            caps[n] += fc[n]
            noops[n] += fn[n]
            oppsels[n] += fo[n]
        folds.append(
            {
                "fold": k,
                "fit_scenes": len(fit),
                "value_calibration_scenes": len(cal),
                "test_scenes": len(test),
                "value_calibration_proposal_count": len(used),
                "joint_selected_translation_fit": shift_fit,
                "quality_fit": {"sample_count": q["sample_count"], "scene_weight_sum": q["scene_weight_sum"]},
                "risk_fit": {"sample_count": risk["sample_count"], "scene_weight_sum": risk["scene_weight_sum"]},
                "joint_fit": {"sample_count": joint["sample_count"], "scene_weight_sum": joint["scene_weight_sum"]},
                **{n: fd[n] for n in names},
                "monotone_subset_valid": subset,
                "frozen_winner_identity_valid": identity,
            }
        )

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(audits[0])
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(audits)
    A = {n: _metrics(agg[n], caps[n], total_opp, noops[n], oppsels[n], total_noop) for n in names}
    vd = {n: _value_diag(vy, vp[n]) for n in names}
    gates = {n: _gate(A[n], A["rsmr"], folds, n) for n in names if n != "rsmr"}
    contracts = all(f["monotone_subset_valid"] and f["frozen_winner_identity_valid"] for f in folds)
    passed = bool(contracts and gates["joint_main"]["pass"])

    # Pre-registered mechanism diagnosis.  Thresholds are diagnostic effect-size
    # floors, not deployment thresholds and do not alter any arm.
    epv_noncat = float(vd["epv_raw"].get("noncatastrophe_auc", -9.0))
    q_noncat = float(vd["quality"].get("noncatastrophe_auc", -9.0))
    r_noncat = float(vd["risk"].get("noncatastrophe_auc", -9.0))
    j_noncat = float(vd["joint_raw"].get("noncatastrophe_auc", -9.0))
    epv_pos = float(vd["epv_raw"].get("positive_auc", -9.0))
    q_pos = float(vd["quality"].get("positive_auc", -9.0))
    r_pos = float(vd["risk"].get("positive_auc", -9.0))
    j_pos = float(vd["joint_raw"].get("positive_auc", -9.0))
    if passed:
        diagnosis = "deployment_observable_value_decomposition_closes_selected_intervention_boundary"
    elif r_noncat > epv_noncat + 0.05 and r_noncat >= q_noncat + 0.02:
        diagnosis = "current_physical_risk_observable_is_primary_missing_tail_mediator"
    elif q_pos > epv_pos + 0.03 and q_pos >= r_pos + 0.02:
        diagnosis = "trajectory_quality_observable_is_primary_missing_cardinal_mediator"
    elif (j_noncat > epv_noncat + 0.04 or j_pos > epv_pos + 0.03) and (
        A["joint_raw"]["teacher_improvement_sum"] > A["epv_raw"]["teacher_improvement_sum"] + 1.0
        or A["joint_raw"]["catastrophic_count"] < A["epv_raw"]["catastrophic_count"]
    ):
        diagnosis = "observable_partition_adds_complementary_value_signal_but_zero_or_unobserved_future_tail_remains"
    else:
        diagnosis = "current_deployment_observable_partition_insufficient_require_new_future_sensitive_value_observable"

    return {
        "folds": folds,
        "scene_audit_csv": str(audit_csv),
        "rsmr_rank_aggregate": A["rsmr"],
        "endpoint_potential_raw_aggregate": A["epv_raw"],
        "quality_observable_aggregate": A["quality"],
        "risk_observable_aggregate": A["risk"],
        "joint_observable_raw_aggregate": A["joint_raw"],
        "joint_observable_main_aggregate": A["joint_main"],
        "selected_proposal_value_prediction_diagnostics": vd,
        "gates": gates,
        "monotone_frozen_winner_contract_valid": contracts,
        "train_gate_pass": passed,
        "failure_diagnosis": diagnosis,
    }


def _decorate_observable(rsm_cfg, epv, obs, mode, path, version, selected_bias=0.0):
    cfg = yaml.safe_load(yaml.safe_dump(rsm_cfg, sort_keys=False))
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic["instrument_value_observables"] = True
    sc = ic["selection_conditioned_intervention_recovery"]
    sc.update(
        {
            "post_selection_value_enabled": True,
            "post_selection_value_mode": mode,
            "post_selection_endpoint_feature_names": list(epv["names"]),
            "post_selection_endpoint_feature_scale": [float(x) for x in epv["scale"]],
            "post_selection_endpoint_weights": [float(x) for x in epv["weights"]],
            "post_selection_endpoint_bias": 0.0,
            "post_selection_observable_names": list(VALUE_OBSERVABLE_NAMES),
            "post_selection_observable_quality_dim": int(QUALITY_DIM),
            "post_selection_observable_scale": [float(x) for x in obs["scale"]],
            "post_selection_observable_weights": [float(x) for x in obs["weights"]],
            "post_selection_selected_bias": float(selected_bias),
            "post_selection_value_training": "scene_equal_all_edge_EPV_plus_deployment_observable_residual_fixed_lambda_1",
            "post_selection_operator": "freeze_RSMR_winner_then_value_specific_observable_accept_same_winner_iff_positive_else_incumbent_no_rerank_no_fallback",
        }
    )
    cfg.setdefault("metadata", {})["algorithm_version"] = version
    cfg.setdefault("provenance", {})["algorithm_version"] = version
    cfg.setdefault("experiment", {})["name"] = version.lower().replace(".", "_").replace("-", "_")
    cfg["experiment"]["algorithm"] = version
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frontier-edges", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-preserve-config", required=True)
    ap.add_argument("--output-rsmr-config", required=True)
    ap.add_argument("--output-epv-config", required=True)
    ap.add_argument("--output-quality-config", required=True)
    ap.add_argument("--output-risk-config", required=True)
    ap.add_argument("--output-joint-config", required=True)
    ap.add_argument("--output-report", required=True)
    ap.add_argument("--output-scene-audit", required=True)
    a = ap.parse_args()

    _, groups = _read_edges(Path(a.train_frontier_edges))
    nested = _nested(groups, Path(a.output_scene_audit))
    # Engineering/replay gate: V42 instruments new label-free observables but
    # must not perturb the frozen V34 RSMR operator or the historical 782-scene
    # TRAIN population.  A mismatch is an engineering failure, not a mechanism
    # result, and therefore aborts before scientific interpretation.
    _r = nested["rsmr_rank_aggregate"]
    _expected = {
        "selected_count": 502,
        "selected_positive_count": 221,
        "no_positive_opportunity_false_intervention_count": 107,
        "catastrophic_count": 28,
    }
    if any(int(_r[k]) != int(v) for k, v in _expected.items()) or abs(float(_r["teacher_improvement_sum"]) - 43.29405361274824) > 1.0e-9 or abs(float(_r["positive_capture_rate"]) - 0.38501742160278746) > 1.0e-12:
        raise RuntimeError("V42 ENGINEERING STOP: value-observable replay changed frozen RSMR TRAIN signature")
    report = {
        "audit": "v64_3_42_eaf_icer_ovdr_fit",
        "scientific_role": "TRAIN_only_frozen_RSMR_plus_value_specific_deployment_observable_decomposition",
        "frozen_train_scenes": len(groups),
        "direct_support_positive_training_scenes": len(_scene_with_observables(groups)),
        "ridge_lambda": RIDGE_LAMBDA,
        "observable_names": list(VALUE_OBSERVABLE_NAMES),
        "mechanism_hypothesis": (
            "V41 shows endpoint/basepoint geometry identifies a high-precision core but cannot transfer an absolute incumbent-exit zero. "
            "Test whether the missing selected value information is carried by deployment-observable trajectory-quality or physical-risk consequences. "
            "RSMR alone freezes winner identity; EPV models endpoint latent value; value-specific observable improvements only fit the EPV residual and can veto that same winner, never re-rank."
        ),
        "nested_crossfit": nested,
        "train_gate_pass": nested["train_gate_pass"],
        "train_gate_contract": {
            "RSMR_is_sole_challenger_selector": True,
            "all_value_arms_are_same_winner_subsets": True,
            "new_observables_are_deployment_available_and_label_free": True,
            "quality_block_is_teacher_aligned_route_progress_comfort_without_demo_future": True,
            "risk_block_is_current_map_agent_continuous_physical_risk": True,
            "observable_features_are_incumbent_minus_candidate_cost_improvements": True,
            "scene_equal_all_edge_fixed_lambda_1": True,
            "selected_policy_calibration_is_translation_only": True,
            "noop_false_intervention_reduction_fraction_min": NOOP_REDUCTION_MIN,
            "capture_tolerance": CAPTURE_TOL,
            "catastrophe_reduction_fraction_min": CATASTROPHE_REDUCTION_MIN,
            "all_test_folds_selected_sum_nonnegative": True,
            "selected_min": 64,
            "positive_min": 32,
            "no_threshold_lambda_alpha_candidate_count_topk_or_capacity_sweep": True,
        },
    }
    Path(a.output_report).write_text(json.dumps(report, indent=2, sort_keys=True))

    # Always serialize full-TRAIN raw diagnostic configs.  They are explicitly
    # tagged non-promoted and enable a small *diagnostic* closed-loop run on an
    # already-design-excluded population without weakening the TRAIN gate.
    scene = _scene_with_observables(groups)
    base = _base_cfg(a.base_config)
    base["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["instrument_value_observables"] = True
    pcfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    pcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"] = {"enabled": False}
    Path(a.output_preserve_config).write_text(yaml.safe_dump(pcfg, sort_keys=False))
    rsm = _fit_regret_structured_margin(scene, list(scene))
    rsmcfg = _write_rsmr(base, a.output_rsmr_config, rsm)
    rsmcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["instrument_value_observables"] = True
    Path(a.output_rsmr_config).write_text(yaml.safe_dump(rsmcfg, sort_keys=False))
    epv = _fit_zero_ridge(scene, list(scene), "epv")
    # Raw EPV config for causal control / diagnostic closed loop.
    epvcfg = yaml.safe_load(yaml.safe_dump(rsmcfg, sort_keys=False))
    sc = epvcfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    sc.update(
        {
            "post_selection_value_enabled": True,
            "post_selection_value_mode": "endpoint_potential_value",
            "post_selection_endpoint_feature_names": list(epv["names"]),
            "post_selection_endpoint_feature_scale": [float(x) for x in epv["scale"]],
            "post_selection_endpoint_weights": [float(x) for x in epv["weights"]],
            "post_selection_endpoint_bias": 0.0,
            "post_selection_value_training": "scene_equal_all_edge_endpoint_potential_fixed_lambda_1",
        }
    )
    epvcfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.42-EAF-ICER-EPV-CONTROL"
    Path(a.output_epv_config).write_text(yaml.safe_dump(epvcfg, sort_keys=False))
    q = _fit_observable_residual(scene, list(scene), epv, "quality")
    risk = _fit_observable_residual(scene, list(scene), epv, "risk")
    joint = _fit_observable_residual(scene, list(scene), epv, "joint")
    _decorate_observable(rsmcfg, epv, q, "endpoint_potential_quality_observable", a.output_quality_config, "V64.3.42-EAF-ICER-EPV-QOR")
    _decorate_observable(rsmcfg, epv, risk, "endpoint_potential_risk_observable", a.output_risk_config, "V64.3.42-EAF-ICER-EPV-ROR")
    _decorate_observable(rsmcfg, epv, joint, "endpoint_potential_joint_observable", a.output_joint_config, "V64.3.42-EAF-ICER-OVDR-RAW")

    if not nested["train_gate_pass"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(
            f"V64.3.42 OVDR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before CAL/fresh selection"
        )
    print(json.dumps({"pass": True, "output_joint_config": a.output_joint_config}, sort_keys=True))


if __name__ == "__main__":
    main()
