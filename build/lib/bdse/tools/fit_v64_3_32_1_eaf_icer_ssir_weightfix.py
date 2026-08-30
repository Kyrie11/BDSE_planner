from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES

BASE_FEATURE_NAMES = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
FEATURE_NAMES = [f"delta::{n}" for n in BASE_FEATURE_NAMES] + ["delta::support_logit"]
EPS = 1.0e-12
EXPECTED_FRONTIER_ROWS = 75133
EXPECTED_SCENES = 3000
FOLDS = 5
RIDGE_LAMBDA = 1.0
ALPHA = 0.05


def _finite(r: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        v = float(r.get(key, default))
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _fold(token: str) -> int:
    h = hashlib.sha256(("v64.3.32-ssir-train-fold-v1::" + token).encode()).digest()
    return int.from_bytes(h[:8], "big") % FOLDS


def _read_edges(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            groups.setdefault(str(r.get("scenario_token", "")), []).append(r)
    if len(rows) != EXPECTED_FRONTIER_ROWS:
        raise SystemExit(f"V64.3.32.1 requires frozen B16 TRAIN frontier rows={len(rows)} expected={EXPECTED_FRONTIER_ROWS}")
    if len(groups) != EXPECTED_SCENES:
        raise SystemExit(f"V64.3.32.1 requires frozen B16 TRAIN scenes={len(groups)} expected={EXPECTED_SCENES}")
    return rows, groups


def _scene_samples(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not group:
        return []
    inc = int(group[0].get("raw_top_action", -1))
    inc_rows = [r for r in group if int(r.get("challenger_action", -2)) == inc]
    if not inc_rows:
        return []
    ir = inc_rows[0]
    if _finite(ir, "icer_admissible", 0.0) < 0.5:
        return []
    inc_tm = _finite(ir, "teacher_margin")
    inc_sup = _finite(ir, "icer_support_logit")
    inc_base = np.asarray([_finite(ir, f"icer_feature_{n}") for n in BASE_FEATURE_NAMES], dtype=np.float64)
    if not math.isfinite(inc_tm) or not math.isfinite(inc_sup) or not np.all(np.isfinite(inc_base)):
        return []
    out: list[dict[str, Any]] = []
    for r in group:
        act = int(r.get("challenger_action", -1))
        if act == inc:
            continue
        if _finite(r, "icer_admissible", 0.0) < 0.5 or _finite(r, "icer_support_logit", -math.inf) <= 0.0:
            continue
        cand_base = np.asarray([_finite(r, f"icer_feature_{n}") for n in BASE_FEATURE_NAMES], dtype=np.float64)
        sup = _finite(r, "icer_support_logit")
        tm = _finite(r, "teacher_margin")
        if not np.all(np.isfinite(cand_base)) or not math.isfinite(sup) or not math.isfinite(tm):
            continue
        x = np.concatenate([cand_base - inc_base, np.asarray([sup - inc_sup], dtype=np.float64)])
        out.append({
            "token": str(r.get("scenario_token", "")),
            "action": act,
            "x": x,
            "y": float(tm - inc_tm),
            "support": float(sup),
            "margin": _finite(r, "raw_margin", -math.inf),
            "utility_prior": int(_finite(r, "dacer_utility_prior", 0.0) >= 0.5),
        })
    return out


def _fit_ridge(samples: list[dict[str, Any]]) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("SSIR ridge fit has no samples")
    X = np.stack([a["x"] for a in samples])
    y = np.asarray([float(a["y"]) for a in samples], dtype=np.float64)
    # V64.3.32.1 engineering hotfix.  The frozen V31/V32 design contract says
    # that every direct-eligible scene contributes total squared-loss weight 1.
    # The historical implementation first assigned 1/n_scene per edge and then
    # normalized those weights to sum to 1 globally.  Because ridge lambda=1 is
    # not scale invariant, that changed the intended objective from
    #   sum_scene mean_edge(loss) + lambda ||w||^2
    # to
    #   (1/N_scene) sum_scene mean_edge(loss) + lambda ||w||^2,
    # i.e. it inflated the effective ridge strength by approximately N_scene.
    # Standardization still needs probability weights, so moments use a normalized
    # copy while the ridge loss and leverage Gram matrix use the unnormalized
    # scene-mass weights.
    counts: dict[str, int] = {}
    for a in samples:
        counts[a["token"]] = counts.get(a["token"], 0) + 1
    loss_w = np.asarray([1.0 / counts[a["token"]] for a in samples], dtype=np.float64)
    moment_w = loss_w / max(float(loss_w.sum()), EPS)
    mean = np.sum(X * moment_w[:, None], axis=0)
    var = np.sum(((X - mean[None, :]) ** 2) * moment_w[:, None], axis=0)
    std = np.maximum(np.sqrt(var), 1.0e-6)
    Z = (X - mean[None, :]) / std[None, :]
    A = np.concatenate([np.ones((len(Z), 1), dtype=np.float64), Z], axis=1)
    root = np.sqrt(loss_w)[:, None]
    Aw = A * root
    yw = y * root[:, 0]
    reg = np.eye(A.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    reg[0, 0] = 0.0
    coef = np.linalg.solve(Aw.T @ Aw + reg, Aw.T @ yw)
    # Auditable candidate-specific scale: ridge leverage in the standardized
    # intervention space.  It is only a normalization function; conformal
    # calibration, not a Gaussian assumption, supplies coverage.
    G = (Z * root) .T @ (Z * root) + np.eye(Z.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    Ginv = np.linalg.inv(G)
    return coef[1:], float(coef[0]), mean, std, Ginv


def _predict(samples: list[dict[str, Any]], model: tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.zeros((0,), dtype=np.float64), np.ones((0,), dtype=np.float64)
    w, b, mean, std, Ginv = model
    X = np.stack([a["x"] for a in samples])
    Z = (X - mean[None, :]) / np.maximum(std[None, :], 1.0e-6)
    mu = np.clip(Z @ w + b, -40.0, 40.0)
    h = np.einsum("bi,ij,bj->b", Z, Ginv, Z)
    h = np.maximum(np.where(np.isfinite(h), h, 0.0), 0.0)
    scale = np.sqrt(1.0 + h)
    return mu, np.maximum(scale, 1.0)


def _conformal_q(scores: list[float], alpha: float = ALPHA) -> tuple[float, int]:
    arr = np.asarray([float(x) for x in scores if math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        raise ValueError("SSIR conformal calibration has no finite scene scores")
    arr.sort()
    k = int(math.ceil((arr.size + 1) * (1.0 - alpha)))
    k = min(max(k, 1), int(arr.size))
    return max(0.0, float(arr[k - 1])), k


def _select(ss: list[dict[str, Any]], score: np.ndarray) -> int | None:
    cand = [j for j, v in enumerate(score.tolist()) if math.isfinite(v) and v > 0.0]
    if not cand:
        return None
    return sorted(cand, key=lambda j: (-float(score[j]), -float(ss[j]["support"]), -float(ss[j]["margin"]), -int(ss[j]["utility_prior"]), int(ss[j]["action"])))[0]


def _nested_diagnostics(groups: dict[str, list[dict[str, Any]]], audit_csv: Path) -> dict[str, Any]:
    ts = {t: _scene_samples(g) for t, g in groups.items()}
    ts = {t: ss for t, ss in ts.items() if ss}
    fold_reports: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    all_mean_y: list[float] = []
    all_v31_veto_y: list[float] = []
    all_lcb_y: list[float] = []
    mean_opp = mean_cap = v31_opp = v31_cap = lcb_opp = lcb_cap = 0

    for k in range(FOLDS):
        test_tokens = [t for t in ts if _fold(t) == k]
        cal_fold = (k + 1) % FOLDS
        cal_tokens = [t for t in ts if _fold(t) == cal_fold]
        fit_tokens = [t for t in ts if _fold(t) not in {k, cal_fold}]
        fit_samples = [a for t in fit_tokens for a in ts[t]]
        model = _fit_ridge(fit_samples)

        scene_scores: list[float] = []
        v31_selected_residuals: list[float] = []
        for t in cal_tokens:
            ss = ts[t]
            mu, scale = _predict(ss, model)
            y = np.asarray([a["y"] for a in ss], dtype=np.float64)
            scene_scores.append(float(np.max((mu - y) / np.maximum(scale, 1.0e-6))))
            mi = _select(ss, mu)
            if mi is not None:
                v31_selected_residuals.append(float(mu[mi] - y[mi]))
        q, q_index = _conformal_q(scene_scores)
        v31_q, v31_q_index = _conformal_q(v31_selected_residuals)

        mean_y: list[float] = []
        v31_veto_y: list[float] = []
        lcb_y: list[float] = []
        opp = mean_c = v31_c = lcb_c = 0
        for t in test_tokens:
            ss = ts[t]
            mu, scale = _predict(ss, model)
            y = np.asarray([a["y"] for a in ss], dtype=np.float64)
            lcb = mu - q * scale
            has_opp = bool(np.any(y > 0.0)); opp += int(has_opp)
            mi = _select(ss, mu)
            # Exact V31 semantic diagnostic: choose the mean winner first, then
            # apply one common selected-proposal conformal offset.  Because the
            # offset is candidate-independent it can abstain but cannot repair
            # a wrong within-scene winner.
            vi = mi if (mi is not None and float(mu[mi] - v31_q) > 0.0) else None
            li = _select(ss, lcb)
            my = float(y[mi]) if mi is not None else float("nan")
            vy = float(y[vi]) if vi is not None else float("nan")
            ly = float(y[li]) if li is not None else float("nan")
            if mi is not None:
                mean_y.append(my); mean_c += int(has_opp and my > 0.0)
            if vi is not None:
                v31_veto_y.append(vy); v31_c += int(has_opp and vy > 0.0)
            if li is not None:
                lcb_y.append(ly); lcb_c += int(has_opp and ly > 0.0)
            audit_rows.append({
                "scenario_token": t,
                "outer_test_fold": k,
                "calibration_fold": cal_fold,
                "candidate_count": len(ss),
                "positive_opportunity": int(has_opp),
                "scene_conformal_q": q,
                "v31_selected_proposal_conformal_q": v31_q,
                "mean_selected_action": -1 if mi is None else int(ss[mi]["action"]),
                "mean_selected_prediction": float("nan") if mi is None else float(mu[mi]),
                "mean_selected_scale": float("nan") if mi is None else float(scale[mi]),
                "mean_selected_teacher_improvement": my,
                "v31_veto_selected_action": -1 if vi is None else int(ss[vi]["action"]),
                "v31_veto_selected_lower_bound": float("nan") if mi is None else float(mu[mi] - v31_q),
                "v31_veto_selected_teacher_improvement": vy,
                "lcb_selected_action": -1 if li is None else int(ss[li]["action"]),
                "lcb_selected_prediction": float("nan") if li is None else float(mu[li]),
                "lcb_selected_scale": float("nan") if li is None else float(scale[li]),
                "lcb_selected_lower_bound": float("nan") if li is None else float(lcb[li]),
                "lcb_selected_teacher_improvement": ly,
                "lcb_changes_mean_winner": int(mi != li),
            })

        def diag(vals: list[float], captured: int) -> dict[str, Any]:
            arr = np.asarray(vals, dtype=np.float64)
            neg = np.minimum(arr, 0.0) if arr.size else arr
            return {
                "selected_count": int(arr.size),
                "selected_positive_count": int((arr > 0.0).sum()) if arr.size else 0,
                "selected_precision": float((arr > 0.0).mean()) if arr.size else float("nan"),
                "teacher_improvement_sum": float(arr.sum()) if arr.size else 0.0,
                "teacher_improvement_worst": float(arr.min()) if arr.size else float("nan"),
                "teacher_negative_rms": float(np.sqrt(np.mean(neg * neg))) if arr.size else 0.0,
                "positive_opportunity_scenes": int(opp),
                "positive_capture_count": int(captured),
                "positive_capture_rate": float(captured / max(opp, 1)),
                "path_nonharmful": bool(arr.size > 0 and float(arr.sum()) >= -1.0e-9),
            }

        md = diag(mean_y, mean_c); vd = diag(v31_veto_y, v31_c); ld = diag(lcb_y, lcb_c)
        fold_reports.append({
            "fold": k,
            "fit_scenes": len(fit_tokens),
            "fit_loss_weight_sum": float(len(fit_tokens)),
            "fit_effective_legacy_lambda_if_globally_normalized": float(RIDGE_LAMBDA * len(fit_tokens)),
            "calibration_scenes": len(cal_tokens),
            "test_scenes": len(test_tokens),
            "calibration_scene_score_count": len(scene_scores),
            "conformal_order_index_1based": q_index,
            "scene_simultaneous_quantile": q,
            "v31_selected_proposal_calibration_count": len(v31_selected_residuals),
            "v31_selected_proposal_order_index_1based": v31_q_index,
            "v31_selected_proposal_quantile": v31_q,
            "mean_rank": md,
            "v31_post_selection_common_offset_veto": vd,
            "simultaneous_lcb": ld,
        })
        all_mean_y.extend(mean_y); all_v31_veto_y.extend(v31_veto_y); all_lcb_y.extend(lcb_y)
        mean_opp += opp; v31_opp += opp; lcb_opp += opp
        mean_cap += mean_c; v31_cap += v31_c; lcb_cap += lcb_c

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    if audit_rows:
        with audit_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0]))
            w.writeheader(); w.writerows(audit_rows)

    def agg(vals: list[float], cap: int, opp: int) -> dict[str, Any]:
        arr = np.asarray(vals, dtype=np.float64); neg = np.minimum(arr, 0.0) if arr.size else arr
        return {
            "selected_count": int(arr.size),
            "selected_positive_count": int((arr > 0.0).sum()) if arr.size else 0,
            "selected_precision": float((arr > 0.0).mean()) if arr.size else float("nan"),
            "teacher_improvement_sum": float(arr.sum()) if arr.size else 0.0,
            "teacher_improvement_worst": float(arr.min()) if arr.size else float("nan"),
            "teacher_negative_rms": float(np.sqrt(np.mean(neg * neg))) if arr.size else 0.0,
            "positive_opportunity_scenes": int(opp),
            "positive_capture_count": int(cap),
            "positive_capture_rate": float(cap / max(opp, 1)),
        }
    ma = agg(all_mean_y, mean_cap, mean_opp)
    va = agg(all_v31_veto_y, v31_cap, v31_opp)
    la = agg(all_lcb_y, lcb_cap, lcb_opp)
    gate = bool(
        all(fr["simultaneous_lcb"]["path_nonharmful"] for fr in fold_reports)
        and la["selected_count"] >= 64
        and la["selected_positive_count"] >= 32
    )
    return {
        "folds": fold_reports,
        "mean_rank_aggregate": ma,
        "v31_post_selection_common_offset_veto_aggregate": va,
        "simultaneous_lcb_aggregate": la,
        "all_folds_simultaneous_lcb_nonharmful": all(fr["simultaneous_lcb"]["path_nonharmful"] for fr in fold_reports),
        "fold_pass_count": sum(int(fr["simultaneous_lcb"]["path_nonharmful"]) for fr in fold_reports),
        "train_gate_pass": gate,
        "scene_audit_csv": str(audit_csv),
    }


def _all_samples(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [a for g in groups.values() for a in _scene_samples(g)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.32.1 SSIR and run nested TRAIN-only simultaneous-bound gate.")
    ap.add_argument("--train-frontier-edges", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-preserve-config", required=True)
    ap.add_argument("--output-mean-config", required=True)
    ap.add_argument("--output-report", required=True)
    ap.add_argument("--output-scene-audit", required=True)
    args = ap.parse_args()

    _, groups = _read_edges(Path(args.train_frontier_edges))
    nested = _nested_diagnostics(groups, Path(args.output_scene_audit))
    samples = _all_samples(groups)
    y = np.asarray([a["y"] for a in samples], dtype=np.float64)
    report: dict[str, Any] = {
        "audit": "v64_3_32_1_eaf_icer_ssir_weightfix_fit",
        "scientific_role": "TRAIN_only_nested_fit_calibrate_test_gate_for_scene_simultaneous_selection_stable_intervention_bounds",
        "frozen_train_scenes": len(groups),
        "direct_support_positive_training_edges": len(samples),
        "direct_support_positive_training_scenes": len({a["token"] for a in samples}),
        "teacher_improvement_positive_fraction": float((y > 0).mean()) if y.size else float("nan"),
        "teacher_improvement_sum": float(y.sum()) if y.size else 0.0,
        "feature_names": FEATURE_NAMES,
        "base_feature_names": BASE_FEATURE_NAMES,
        "ridge_lambda": RIDGE_LAMBDA,
        "conformal_alpha": ALPHA,
        "selection_scale": "ridge_leverage_sqrt_1_plus_h",
        "weightfix_contract": {
            "intended_objective": "sum_over_direct_scenes(mean_squared_error_over_scene_candidates)+lambda_times_l2",
            "each_scene_total_loss_weight": 1.0,
            "moment_weights_normalized_only_for_standardization": True,
            "ridge_and_leverage_use_unnormalized_scene_mass": True,
            "historical_v31_v32_bug": "global_weight_normalization_to_sum_1_while_lambda_remained_1_inflated_effective_regularization_by_number_of_fit_scenes",
        },
        "calibration_nonconformity": "one_score_per_direct_eligible_scene=max_over_all_direct_admissible_support_positive_candidates_of_(mu-y)/scale",
        "v31_counterfactual_diagnostic": "same_nested_splits_replay_original_selected_mean_winner_then_common_conformal_offset_veto; diagnostic_only_not_a_rescue_path",
        "nested_crossfit": nested,
        "train_gate_pass": bool(nested["train_gate_pass"]),
        "train_gate_contract": {
            "outer_folds": FOLDS,
            "within_outer_split": "3_folds_fit_1_fold_calibrate_1_fold_test_rotating",
            "all_folds_simultaneous_lcb_teacher_improvement_sum_min": 0.0,
            "aggregate_simultaneous_lcb_selected_count_min": 64,
            "aggregate_simultaneous_lcb_selected_positive_count_min": 32,
            "no_alpha_ridge_feature_or_threshold_sweep": True,
        },
        "fit_uses_validation": False,
        "fit_uses_test": False,
    }
    rp = Path(args.output_report); rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["train_gate_pass"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("V64.3.32.1 SSIR nested TRAIN simultaneous-bound gate failed; STOP before calibration/fresh selection")

    model = _fit_ridge(samples)
    w, b, mean, std, Ginv = model
    cfg = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    ic = cfg.setdefault("runtime", {}).setdefault("decisive_frontier_value", {}).setdefault("incumbent_contrastive_extremal_recovery", {})
    ic["incumbent_retention_policy"] = "preserve_admissible_incumbent"
    ic["regret_risk_enabled"] = False
    ic["retention_regret_risk_enabled"] = False
    ic["replacement_regret_risk_enabled"] = False

    preserve_cfg = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False))
    preserve_ic = preserve_cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    preserve_ic["selection_conditioned_intervention_recovery"] = {"enabled": False}
    preserve_cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.32.1-EAF-ICER-PRESERVE-CONTROL"
    preserve_cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.32.1-EAF-ICER-PRESERVE-CONTROL"
    preserve_cfg.setdefault("experiment", {})["name"] = "v64_3_32_1_eaf_icer_preserve_control"
    preserve_cfg["experiment"]["algorithm"] = "V64.3.32.1 preservation control: frozen V20 direct dominance with admissible-incumbent default"
    Path(args.output_preserve_config).write_text(yaml.safe_dump(preserve_cfg, sort_keys=False), encoding="utf-8")

    scir = ic.setdefault("selection_conditioned_intervention_recovery", {})
    scir.update({
        "enabled": True,
        "mode": "mean_rank",
        "model_type": "scene_equal_total_mass1_linear_ridge_same_scene_incumbent_contrastive_improvement_with_leverage_scale_weightfix",
        "base_feature_names": BASE_FEATURE_NAMES,
        "feature_names": FEATURE_NAMES,
        "feature_mean": [float(x) for x in mean],
        "feature_std": [float(x) for x in std],
        "weights": [float(x) for x in w],
        "bias": float(b),
        "ridge_lambda": RIDGE_LAMBDA,
        "leverage_inverse": [[float(x) for x in row] for row in Ginv],
        "selection_scale_floor": 1.0,
        "training_population": "TRAIN_only_incumbent_deployment_admissible_support_positive_alternatives",
        "training_weighting": "each_scene_total_squared_loss_weight_1; global normalization used only for feature moments, not ridge/leverage objective",
        "training_target": "continuous_teacher_candidate_minus_same_scene_incumbent_improvement",
        "proposal_operator": "mean_control_argmax_predicted_improvement_before_scene_simultaneous_calibration",
        "require_positive_predicted_improvement": True,
        "conformal_alpha": ALPHA,
        "simultaneous_conformal_quantile": 0.0,
        "simultaneous_calibration_status": "not_yet_calibrated_mean_control",
        "no_fallback": True,
    })
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.32.1-EAF-ICER-SSIR-MEAN"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.32.1-EAF-ICER-SSIR-MEAN"
    exp = cfg.setdefault("experiment", {})
    exp["name"] = "v64_3_32_1_eaf_icer_ssir_mean_weightfix"
    exp["algorithm"] = "V64.3.32.1 SSIR mean-ranking causal control before simultaneous conformal bound"
    exp["mechanism_chain"] = "B16 interface -> exact EAF -> admissible direct intervention -> same-scene mean improvement + ridge leverage scale -> mean argmax control -> incumbent default"
    Path(args.output_mean_config).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    report["model"] = {
        "weights": [float(x) for x in w], "bias": float(b),
        "feature_mean": [float(x) for x in mean], "feature_std": [float(x) for x in std],
        "leverage_inverse": [[float(x) for x in row] for row in Ginv],
    }
    rp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "pass": True, "train_gate_pass": True,
        "fold_pass_count": nested["fold_pass_count"],
        "simultaneous_lcb_selected_count": nested["simultaneous_lcb_aggregate"]["selected_count"],
        "output_preserve_config": args.output_preserve_config,
        "output_mean_config": args.output_mean_config,
        "scene_audit_csv": args.output_scene_audit,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
