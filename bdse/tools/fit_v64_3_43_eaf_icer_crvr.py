from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.planner.response_value_observables import RESPONSE_VALUE_OBSERVABLE_NAMES
from bdse.planner.value_observables import QUALITY_DIM, VALUE_OBSERVABLE_NAMES
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FOLDS, RIDGE_LAMBDA, _finite, _fold, _read_edges, _select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _extended_diag, _fit_regret_structured_margin, _structured_scores
from bdse.tools.fit_v64_3_38_eaf_icer_davr import (
    CAPTURE_TOL,
    CATASTROPHE_REDUCTION_MIN,
    NOOP_REDUCTION_MIN,
    _value_diag,
    _write_rsmr,
)
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES, _base_cfg, _phi
from bdse.tools.fit_v64_3_42_eaf_icer_ovdr import (
    _fit_observable_residual,
    _scene_with_observables,
    _value as _v42_value,
)

EPS = 1.0e-12
ALL_OBSERVABLE_NAMES = list(VALUE_OBSERVABLE_NAMES) + list(RESPONSE_VALUE_OBSERVABLE_NAMES)
ANCHOR_ORDER = ["q_anchor", "cv_anchor", "mean_anchor", "robust_anchor"]


def _scene_with_response_observables(groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    scene = _scene_with_observables(groups)
    rkeys = [f"icer_value_observable_{n}" for n in RESPONSE_VALUE_OBSERVABLE_NAMES]
    oracle_keys = [
        "v43_oracle_teacher_base_cost",
        "v43_oracle_teacher_evidence_cost",
        "v43_oracle_teacher_selected_evidence_cost",
        "v43_oracle_teacher_unselected_evidence_cost",
    ]
    out: dict[str, list[dict[str, Any]]] = {}
    for tok, ss in scene.items():
        rows = groups[tok]
        inc = int(rows[0].get("raw_top_action", -1))
        byact = {int(r.get("challenger_action", -2)): r for r in rows}
        ir = byact.get(inc)
        if ir is None:
            continue
        try:
            ri = np.asarray([_finite(ir, k) for k in rkeys], dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(f"V43 missing incumbent response-envelope observables for scene {tok}") from exc
        if ri.size != len(RESPONSE_VALUE_OBSERVABLE_NAMES) or not np.all(np.isfinite(ri)):
            raise RuntimeError(f"V43 incumbent response-envelope schema invalid for scene {tok}")
        target_scale = _finite(ir, "v43_value_target_scale")
        if not np.isfinite(target_scale) or target_scale <= 0.0:
            raise RuntimeError(f"V43 missing/invalid normalized teacher target scale for scene {tok}")
        try:
            oracle_i = np.asarray([_finite(ir, k) for k in oracle_keys], dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(f"V43 missing TRAIN-only teacher decomposition for scene {tok}") from exc
        if oracle_i.size != len(oracle_keys) or not np.all(np.isfinite(oracle_i)):
            raise RuntimeError(f"V43 incumbent teacher decomposition invalid for scene {tok}")
        rr = []
        for a in ss:
            row = byact.get(int(a["action"]))
            if row is None:
                raise RuntimeError("V43 response-envelope replay missing candidate row")
            try:
                rb = np.asarray([_finite(row, k) for k in rkeys], dtype=np.float64)
            except Exception as exc:
                raise RuntimeError(f"V43 missing candidate response-envelope observables for scene {tok}") from exc
            if rb.size != len(RESPONSE_VALUE_OBSERVABLE_NAMES) or not np.all(np.isfinite(rb)):
                raise RuntimeError(f"V43 candidate response-envelope schema invalid for scene {tok}")
            row_scale = _finite(row, "v43_value_target_scale")
            if not np.isfinite(row_scale) or abs(row_scale - target_scale) > 1.0e-9 * max(1.0, abs(target_scale)):
                raise RuntimeError(f"V43 per-scene target scale changed across candidate rows for scene {tok}")
            try:
                oracle_b = np.asarray([_finite(row, k) for k in oracle_keys], dtype=np.float64)
            except Exception as exc:
                raise RuntimeError(f"V43 missing candidate teacher decomposition for scene {tok}") from exc
            if oracle_b.size != len(oracle_keys) or not np.all(np.isfinite(oracle_b)):
                raise RuntimeError(f"V43 candidate teacher decomposition invalid for scene {tok}")
            b = dict(a)
            b["value_target_scale"] = float(target_scale)
            b["response_observable_inc"] = ri.copy()
            b["response_observable_cand"] = rb
            b["response_observable_improvement"] = ri - rb
            oracle_imp = (oracle_i - oracle_b) / float(target_scale)
            q_norm = float(np.asarray(b["observable_improvement"], dtype=np.float64)[:QUALITY_DIM].sum() / float(target_scale))
            base_norm, evid_norm, selected_norm, unselected_norm = [float(x) for x in oracle_imp]
            demo_norm = float(base_norm - q_norm)
            closure = q_norm + demo_norm + selected_norm + unselected_norm
            if abs(closure - float(b["y"])) > 2.0e-5 * max(1.0, abs(float(b["y"])), abs(closure)):
                raise RuntimeError(
                    f"V43 normalized teacher decomposition does not close for scene {tok}: y={float(b['y'])} closure={closure}"
                )
            b["oracle_improvement"] = {
                "quality": q_norm,
                "demo_label_only": demo_norm,
                "selected_evidence_teacher": selected_norm,
                "unselected_evidence_teacher": unselected_norm,
                "base_total": base_norm,
                "evidence_total": evid_norm,
                "closure": closure,
            }
            rr.append(b)
        out[tok] = rr
    return out


def _anchor_improvement(a: dict[str, Any], mode: str) -> float:
    q = np.asarray(a["observable_improvement"], dtype=np.float64)
    if q.size != len(VALUE_OBSERVABLE_NAMES):
        raise ValueError("V43 current observable vector has wrong dimension")
    # These three columns already include the exact teacher route/progress/comfort
    # weights and scales, but the learned EAF value target is normalized by the
    # per-scene pair-margin scale.  Convert into target units first; coefficients
    # are structurally +1 only after this normalization.
    target_scale = float(a.get("value_target_scale", 1.0))
    if not np.isfinite(target_scale) or target_scale <= 0.0:
        raise ValueError("V43 analytic anchor requires positive finite value_target_scale")
    value = float(q[:QUALITY_DIM].sum() / target_scale)
    if mode == "q_anchor":
        return value
    r = np.asarray(a["response_observable_improvement"], dtype=np.float64)
    if r.size != len(RESPONSE_VALUE_OBSERVABLE_NAMES):
        raise ValueError("V43 response observable vector has wrong dimension")
    idx = {"cv_anchor": 0, "mean_anchor": 1, "robust_anchor": 2}.get(mode)
    if idx is None:
        raise ValueError(mode)
    return value + float(r[idx]) / target_scale


def _fit_endpoint_remainder(
    scene: dict[str, list[dict[str, Any]]], tokens: list[str], mode: str
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
            X.append(_phi(a, "epv"))
            y.append(float(a["y"]) - _anchor_improvement(a, mode))
            w.append(1.0 / n)
    if not X:
        raise ValueError("V43 anchored endpoint-remainder fit has no samples")
    X = np.stack(X).astype(np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    pm = w / max(float(w.sum()), EPS)
    scale = np.sqrt(np.sum((X * X) * pm[:, None], axis=0))
    scale = np.maximum(scale, 1.0e-6)
    Z = X / scale[None, :]
    root = np.sqrt(w)[:, None]
    Zw = Z * root
    yw = y * root[:, 0]
    coef = np.linalg.solve(Zw.T @ Zw + np.eye(Z.shape[1]) * RIDGE_LAMBDA, Zw.T @ yw)
    return {
        "mode": mode,
        "names": list(EPV_NAMES),
        "scale": scale,
        "weights": coef,
        "bias": 0.0,
        "sample_count": int(len(y)),
        "scene_weight_sum": float(w.sum()),
    }


def _remainder_pred(a: dict[str, Any], model: dict[str, Any]) -> float:
    phi = _phi(a, "epv")
    return float((phi / np.maximum(np.asarray(model["scale"], dtype=np.float64), 1.0e-6)) @ np.asarray(model["weights"], dtype=np.float64))


def _anchored_value(a: dict[str, Any], model: dict[str, Any]) -> float:
    return float(np.clip(_anchor_improvement(a, str(model["mode"])) + _remainder_pred(a, model), -40.0, 40.0))


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


def _eval(ss, rsm, epv_full, v42_q, anchored):
    score = _structured_scores(ss, rsm)
    idx = _select(ss, score)
    names = ["rsmr", "v42_quality"] + ANCHOR_ORDER
    if idx is None:
        return {n: (None, float("nan")) for n in names}
    qv = _v42_value(ss[idx], epv_full, v42_q)
    out = {
        "rsmr": (idx, float(score[idx])),
        "v42_quality": (idx if qv > 0.0 else None, float(qv)),
    }
    for n in ANCHOR_ORDER:
        v = _anchored_value(ss[idx], anchored[n])
        out[n] = (idx if v > 0.0 else None, v)
    return out


def _oracle_decomposition_diag(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [
        "quality",
        "demo_label_only",
        "selected_evidence_teacher",
        "unselected_evidence_teacher",
        "runtime_response_cv",
        "runtime_response_mean",
        "runtime_response_robust",
    ]
    y = np.asarray([float(r["y"]) for r in rows], dtype=np.float64)

    def pack(mask: np.ndarray) -> dict[str, Any]:
        idx = np.flatnonzero(mask)
        out: dict[str, Any] = {"count": int(idx.size), "teacher_sum": float(y[idx].sum()) if idx.size else 0.0}
        for k in keys:
            x = np.asarray([float(rows[i][k]) for i in idx], dtype=np.float64) if idx.size else np.zeros((0,), dtype=np.float64)
            out[k] = {
                "sum": float(x.sum()) if x.size else 0.0,
                "mean": float(x.mean()) if x.size else float("nan"),
                "mean_abs": float(np.abs(x).mean()) if x.size else float("nan"),
            }
        return out

    all_mask = np.ones_like(y, dtype=bool)
    cat = y <= -0.5
    material_pos = y > 0.2
    near_zero_pos = (y > 0.0) & (y <= 0.01)
    # Teacher selected-evidence minus runtime-only robust selected-evidence is a
    # direct diagnostic of logged-future / response-model mismatch on the same
    # bounded evidence subset; it is not a runtime feature.
    sel = np.asarray([float(r["selected_evidence_teacher"]) for r in rows], dtype=np.float64)
    rr = np.asarray([float(r["runtime_response_robust"]) for r in rows], dtype=np.float64)
    response_gap = sel - rr
    return {
        "all_rsmr_proposals": pack(all_mask),
        "catastrophes": pack(cat),
        "material_positives_gt_0p2": pack(material_pos),
        "near_zero_positives_le_0p01": pack(near_zero_pos),
        "selected_teacher_minus_runtime_robust_response_gap": {
            "mean": float(response_gap.mean()),
            "mean_abs": float(np.abs(response_gap).mean()),
            "rms": float(np.sqrt(np.mean(response_gap * response_gap))),
            "catastrophe_mean_abs": float(np.abs(response_gap[cat]).mean()) if np.any(cat) else float("nan"),
            "material_positive_mean_abs": float(np.abs(response_gap[material_pos]).mean()) if np.any(material_pos) else float("nan"),
        },
        "exact_normalized_teacher_partition": "y = quality + demo_label_only + selected_evidence_teacher + unselected_evidence_teacher",
        "oracle_fields_are_train_diagnostic_only": True,
    }


def _nested(groups, audit_csv: Path):
    scene = _scene_with_response_observables(groups)
    names = ["rsmr", "v42_quality"] + ANCHOR_ORDER
    agg = {n: [] for n in names}
    caps = {n: 0 for n in names}
    noops = {n: 0 for n in names}
    oppsels = {n: 0 for n in names}
    total_opp = total_noop = 0
    folds = []
    audits = []
    vy = []
    vp = {n: [] for n in names}
    oracle_selected: list[dict[str, Any]] = []

    for k in range(FOLDS):
        test = [t for t in scene if _fold(t) == k]
        cf = (k + 1) % FOLDS
        cal = [t for t in scene if _fold(t) == cf]
        fit = [t for t in scene if _fold(t) not in {k, cf}]
        # Keep the V41/V42 3-fit/1-cal/1-test population exactly even though V43
        # no longer estimates a selected-policy translation.  This isolates the
        # mechanism change from an effective-sample-size change.
        rsm = _fit_regret_structured_margin(scene, fit)
        from bdse.tools.fit_v64_3_41_eaf_icer_epvr import _fit_zero_ridge
        epv_full = _fit_zero_ridge(scene, fit, "epv")
        v42_q = _fit_observable_residual(scene, fit, epv_full, "quality")
        anchored = {n: _fit_endpoint_remainder(scene, fit, n) for n in ANCHOR_ORDER}

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
            ev = _eval(ss, rsm, epv_full, v42_q, anchored)
            ridx = ev["rsmr"][0]
            if ridx is not None:
                vy.append(float(yy[ridx]))
                for n in names:
                    vp[n].append(float(ev[n][1]))
                oracle_selected.append({
                    "scenario_token": t,
                    "y": float(yy[ridx]),
                    **{str(k0): float(v0) for k0, v0 in ss[ridx]["oracle_improvement"].items()},
                    "runtime_response_cv": float(np.asarray(ss[ridx]["response_observable_improvement"], dtype=np.float64)[0] / ss[ridx]["value_target_scale"]),
                    "runtime_response_mean": float(np.asarray(ss[ridx]["response_observable_improvement"], dtype=np.float64)[1] / ss[ridx]["value_target_scale"]),
                    "runtime_response_robust": float(np.asarray(ss[ridx]["response_observable_improvement"], dtype=np.float64)[2] / ss[ridx]["value_target_scale"]),
                })
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
                    "reserved_calibration_fold": cf,
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
                "reserved_calibration_scenes": len(cal),
                "test_scenes": len(test),
                "no_selected_policy_translation": True,
                **{f"{n}_fit": {"sample_count": anchored[n]["sample_count"], "scene_weight_sum": anchored[n]["scene_weight_sum"]} for n in ANCHOR_ORDER},
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

    passing = [n for n in ANCHOR_ORDER if gates[n]["pass"]]
    promoted = passing[0] if passing else None
    passed = bool(contracts and promoted is not None)
    if promoted == "q_anchor":
        diagnosis = "analytic_current_quality_accounting_is_sufficient_response_envelope_not_required"
    elif promoted == "cv_anchor":
        diagnosis = "selected_evidence_physical_cost_reconstruction_is_sufficient_multimodal_response_not_required"
    elif promoted == "mean_anchor":
        diagnosis = "multimodal_response_expectation_is_required_tail_functional_not_required"
    elif promoted == "robust_anchor":
        diagnosis = "response_tail_aggregation_is_required_for_deployment_sufficient_absolute_value"
    else:
        diagnosis = "handcrafted_runtime_response_envelope_incomplete_require_data_conditioned_future_consequence_observable"

    return {
        "folds": folds,
        "scene_audit_csv": str(audit_csv),
        "rsmr_rank_aggregate": A["rsmr"],
        "v42_quality_control_aggregate": A["v42_quality"],
        "quality_anchor_aggregate": A["q_anchor"],
        "cv_evidence_anchor_aggregate": A["cv_anchor"],
        "response_mean_anchor_aggregate": A["mean_anchor"],
        "response_robust_anchor_aggregate": A["robust_anchor"],
        "selected_proposal_value_prediction_diagnostics": vd,
        "teacher_component_oracle_diagnostics": _oracle_decomposition_diag(oracle_selected),
        "gates": gates,
        "monotone_frozen_winner_contract_valid": contracts,
        "train_gate_pass": passed,
        "promoted_arm": promoted,
        "failure_diagnosis": diagnosis,
    }


def _decorate_anchor(rsm_cfg, model, mode, path, version):
    mode_map = {
        "q_anchor": "endpoint_residual_quality_anchor",
        "cv_anchor": "endpoint_residual_quality_cv_evidence_anchor",
        "mean_anchor": "endpoint_residual_quality_mean_response_anchor",
        "robust_anchor": "endpoint_residual_quality_robust_response_anchor",
    }
    cfg = yaml.safe_load(yaml.safe_dump(rsm_cfg, sort_keys=False))
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic["instrument_value_observables"] = True
    needs_response = mode != "q_anchor"
    ic["instrument_response_value_observables"] = bool(needs_response)
    sc = ic["selection_conditioned_intervention_recovery"]
    sc.update(
        {
            "post_selection_value_enabled": True,
            "post_selection_value_mode": mode_map[mode],
            "post_selection_endpoint_feature_names": list(model["names"]),
            "post_selection_endpoint_feature_scale": [float(x) for x in model["scale"]],
            "post_selection_endpoint_weights": [float(x) for x in model["weights"]],
            "post_selection_endpoint_bias": 0.0,
            "post_selection_observable_names": list(ALL_OBSERVABLE_NAMES if needs_response else VALUE_OBSERVABLE_NAMES),
            "post_selection_observable_quality_dim": int(QUALITY_DIM),
            "post_selection_value_training": "scene_equal_all_edge_analytic_observable_anchor_plus_zero_bias_endpoint_remainder_fixed_lambda_1",
            "post_selection_operator": "freeze_RSMR_winner_then_analytic_quality_and_selected_evidence_response_anchor_plus_endpoint_remainder_accept_same_winner_iff_positive_else_incumbent_no_rerank_no_fallback",
            "post_selection_no_learned_observable_coefficients": True,
            "post_selection_no_selected_translation": True,
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
    ap.add_argument("--output-v42-quality-config", required=True)
    ap.add_argument("--output-q-anchor-config", required=True)
    ap.add_argument("--output-cv-anchor-config", required=True)
    ap.add_argument("--output-mean-anchor-config", required=True)
    ap.add_argument("--output-robust-anchor-config", required=True)
    ap.add_argument("--output-promoted-config", required=True)
    ap.add_argument("--output-report", required=True)
    ap.add_argument("--output-scene-audit", required=True)
    a = ap.parse_args()

    _, groups = _read_edges(Path(a.train_frontier_edges))
    nested = _nested(groups, Path(a.output_scene_audit))
    r = nested["rsmr_rank_aggregate"]
    expected = {"selected_count": 502, "selected_positive_count": 221, "no_positive_opportunity_false_intervention_count": 107, "catastrophic_count": 28}
    if any(int(r[k]) != int(v) for k, v in expected.items()) or abs(float(r["teacher_improvement_sum"]) - 43.29405361274824) > 1.0e-9 or abs(float(r["positive_capture_rate"]) - 0.38501742160278746) > 1.0e-12:
        raise RuntimeError("V43 ENGINEERING STOP: response-envelope instrumentation changed frozen RSMR TRAIN signature")

    report = {
        "audit": "v64_3_43_eaf_icer_crvr_fit",
        "scientific_role": "TRAIN_only_frozen_RSMR_counterfactual_response_anchored_absolute_value_decomposition",
        "frozen_train_scenes": len(groups),
        "direct_support_positive_training_scenes": len(_scene_with_response_observables(groups)),
        "ridge_lambda": RIDGE_LAMBDA,
        "observable_names": list(ALL_OBSERVABLE_NAMES),
        "mechanism_hypothesis": (
            "V42 proves current deployment consequence is real but incomplete and also shows learned joint regression can reverse known cost semantics. "
            "V43 therefore fixes exact observable cost coefficients to +1, evaluates the already selected evidence under label-free future response modes, "
            "and trains endpoint potential only on the unexplained remainder. Analytic costs are divided by the exact runtime pair-margin scale before fixed +1 accounting. CV/mean/robust response arms isolate physical reconstruction, response expectation, and response-tail aggregation."
        ),
        "nested_crossfit": nested,
        "train_gate_pass": nested["train_gate_pass"],
        "train_gate_contract": {
            "RSMR_is_sole_challenger_selector": True,
            "all_value_arms_are_same_winner_subsets": True,
            "selected_evidence_only_for_response_cost": True,
            "response_modes_are_runtime_only_and_label_future_forbidden": True,
            "quality_cost_coefficients_fixed_plus_one_after_exact_pair_margin_normalization": True,
            "response_cost_coefficient_fixed_plus_one_after_exact_pair_margin_normalization": True,
            "train_only_teacher_component_oracle_never_enters_runtime": True,
            "endpoint_model_fits_only_unexplained_remainder": True,
            "scene_equal_fixed_lambda_1_zero_bias": True,
            "no_selected_policy_translation": True,
            "simplicity_first_promotion_order": list(ANCHOR_ORDER),
            "noop_false_intervention_reduction_fraction_min": NOOP_REDUCTION_MIN,
            "capture_tolerance": CAPTURE_TOL,
            "catastrophe_reduction_fraction_min": CATASTROPHE_REDUCTION_MIN,
            "all_test_folds_selected_sum_nonnegative": True,
            "selected_min": 64,
            "positive_min": 32,
            "no_threshold_lambda_alpha_feature_candidate_count_topk_or_capacity_sweep": True,
        },
    }
    Path(a.output_report).write_text(json.dumps(report, indent=2, sort_keys=True))

    # Always serialize full-TRAIN configs for deterministic replay/diagnosis.
    scene = _scene_with_response_observables(groups)
    base = _base_cfg(a.base_config)
    preserve = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    preserve["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"] = {"enabled": False}
    preserve.setdefault("metadata", {})["algorithm_version"] = "V64.3.43-EAF-ICER-PRESERVE"
    preserve.setdefault("provenance", {})["algorithm_version"] = "V64.3.43-EAF-ICER-PRESERVE"
    preserve.setdefault("experiment", {})["name"] = "v64_3_43_eaf_icer_preserve"
    preserve["experiment"]["algorithm"] = "V64.3.43-EAF-ICER-PRESERVE"
    Path(a.output_preserve_config).write_text(yaml.safe_dump(preserve, sort_keys=False))
    rsm = _fit_regret_structured_margin(scene, list(scene))
    rsmcfg = _write_rsmr(base, a.output_rsmr_config, rsm)

    from bdse.tools.fit_v64_3_41_eaf_icer_epvr import _fit_zero_ridge
    epv_full = _fit_zero_ridge(scene, list(scene), "epv")
    v42q = _fit_observable_residual(scene, list(scene), epv_full, "quality")
    # Reuse the V42 decorator only for a non-promoted causal control.  Disable
    # V43 response instrumentation in this control so its serialized 9-D V42
    # observable schema remains exactly self-consistent at runtime.
    from bdse.tools.fit_v64_3_42_eaf_icer_ovdr import _decorate_observable
    v42_base = yaml.safe_load(yaml.safe_dump(rsmcfg, sort_keys=False))
    v42_base["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["instrument_response_value_observables"] = False
    _decorate_observable(v42_base, epv_full, v42q, "endpoint_potential_quality_observable", a.output_v42_quality_config, "V64.3.43-EAF-ICER-V42-Q-CONTROL")

    paths = {
        "q_anchor": a.output_q_anchor_config,
        "cv_anchor": a.output_cv_anchor_config,
        "mean_anchor": a.output_mean_anchor_config,
        "robust_anchor": a.output_robust_anchor_config,
    }
    versions = {
        "q_anchor": "V64.3.43-EAF-ICER-QA",
        "cv_anchor": "V64.3.43-EAF-ICER-CVA",
        "mean_anchor": "V64.3.43-EAF-ICER-RMA",
        "robust_anchor": "V64.3.43-EAF-ICER-CRVR",
    }
    built = {}
    for n in ANCHOR_ORDER:
        m = _fit_endpoint_remainder(scene, list(scene), n)
        built[n] = _decorate_anchor(rsmcfg, m, n, paths[n], versions[n])

    promoted = nested.get("promoted_arm")
    if promoted is not None:
        Path(a.output_promoted_config).write_text(yaml.safe_dump(built[promoted], sort_keys=False))
    if not nested["train_gate_pass"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(
            f"V64.3.43 CRVR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before any fresh selection"
        )
    print(json.dumps({"pass": True, "promoted_arm": promoted, "output_promoted_config": a.output_promoted_config}, sort_keys=True))


if __name__ == "__main__":
    main()
