from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from bdse.tools.fit_v64_3_32_1_eaf_icer_ssir_weightfix import _fit_ridge as _fit_mean_ridge, _predict as _predict_mean
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import (
    ALPHA,
    BASE_FEATURE_NAMES,
    CAT,
    EXPECTED_FRONTIER_ROWS,
    EXPECTED_SCENES,
    FEATURE_NAMES,
    FOLDS,
    RIDGE_LAMBDA,
    _conformal_q,
    _diag,
    _finite,
    _fit_pair_gap,
    _fold,
    _pair_scores,
    _read_edges,
    _scene_samples,
    _select,
    _teacher_best_index,
)

EPS = 1.0e-12
MIN_NESTED_CAL_PROPOSALS = 32
LBFGS_MAX_ITER = 250
LBFGS_HISTORY = 50


def _scene_equal_zero_rms_scale(scene_map: dict[str, list[dict[str, Any]]], tokens: list[str]) -> np.ndarray:
    """Zero-preserving feature scale with one total moment mass per scene.

    This scale is label-free and does not encode V33's asymmetric pair counts.
    The incumbent pseudo-item therefore remains exactly x=0 / score=0.
    """
    xs: list[np.ndarray] = []
    ws: list[float] = []
    for tok in tokens:
        ss = scene_map.get(tok, [])
        if not ss:
            continue
        m = len(ss)
        for a in ss:
            xs.append(np.asarray(a["x"], dtype=np.float64))
            ws.append(1.0 / m)
    if not xs:
        raise ValueError("V64.3.34 structured-margin scale has no candidate features")
    X = np.stack(xs)
    w = np.asarray(ws, dtype=np.float64)
    pm = w / max(float(w.sum()), EPS)
    scale = np.sqrt(np.sum((X * X) * pm[:, None], axis=0))
    return np.maximum(scale, 1.0e-6)


def _scene_margin_blocks(
    scene_map: dict[str, list[dict[str, Any]]], tokens: list[str], scale: np.ndarray
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Build cost-sensitive structured-margin constraints.

    For teacher-optimal item t over {incumbent} U challengers and every rival r,
      d_tr = (x_t - x_r) / scale
      g_tr = Delta_t - Delta_r >= 0.

    Runtime score is s(a)=w^T x_a/scale and s(incumbent)=0.  Per-scene loss is
      [max_r (g_tr - w^T d_tr)]_+^2.
    Thus each scene contributes exactly one *worst decision violation* rather
    than averaging all rivals.  This removes V33's candidate-count-dependent
    dilution of the challenger-vs-incumbent boundary.
    """
    z0 = np.zeros((len(FEATURE_NAMES),), dtype=np.float64)
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for tok in tokens:
        ss = scene_map.get(tok, [])
        if not ss:
            continue
        bi = _teacher_best_index(ss)
        tx = z0 if bi < 0 else np.asarray(ss[bi]["x"], dtype=np.float64)
        ty = 0.0 if bi < 0 else float(ss[bi]["y"])
        rivals: list[tuple[np.ndarray, float, int]] = [(z0, 0.0, -1)]
        rivals.extend((np.asarray(a["x"], dtype=np.float64), float(a["y"]), j) for j, a in enumerate(ss))
        rivals = [r for r in rivals if r[2] != bi]
        if not rivals:
            continue
        D = np.stack([(tx - rx) / scale for rx, _, _ in rivals])
        g = np.asarray([ty - ry for _, ry, _ in rivals], dtype=np.float64)
        if np.any(g < -1.0e-10):
            raise RuntimeError("V64.3.34 teacher-best regret margin became negative")
        out.append((tok, D, np.maximum(g, 0.0)))
    return out


def _structured_objective_numpy(w: np.ndarray, blocks: list[tuple[str, np.ndarray, np.ndarray]]) -> float:
    total = RIDGE_LAMBDA * float(np.dot(w, w))
    for _, D, g in blocks:
        v = g - D @ w
        m = max(float(np.max(v)), 0.0)
        total += m * m
    return float(total)


def _fit_regret_structured_margin(
    scene_map: dict[str, list[dict[str, Any]]], tokens: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scale = _scene_equal_zero_rms_scale(scene_map, tokens)
    blocks = _scene_margin_blocks(scene_map, tokens, scale)
    if not blocks:
        raise ValueError("V64.3.34 regret-structured margin fit has no scene constraints")

    # The objective is convex: sum of squared positive parts of maxima of affine
    # functions plus fixed L2 regularization.  CPU float64 LBFGS is used only as
    # a deterministic numerical solver; its settings are not validation-tuned.
    tw = torch.zeros((len(FEATURE_NAMES),), dtype=torch.float64, requires_grad=True)
    tblocks = [
        (torch.from_numpy(D).to(dtype=torch.float64), torch.from_numpy(g).to(dtype=torch.float64))
        for _, D, g in blocks
    ]
    opt = torch.optim.LBFGS(
        [tw],
        lr=1.0,
        max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )
    eval_count = 0

    def closure() -> torch.Tensor:
        nonlocal eval_count
        opt.zero_grad()
        loss = RIDGE_LAMBDA * torch.dot(tw, tw)
        for D, g in tblocks:
            viol = g - D.mv(tw)
            m = torch.clamp(torch.max(viol), min=0.0)
            loss = loss + m * m
        loss.backward()
        eval_count += 1
        return loss

    before = _structured_objective_numpy(np.zeros((len(FEATURE_NAMES),), dtype=np.float64), blocks)
    opt.step(closure)
    w = tw.detach().cpu().numpy().astype(np.float64)
    after = _structured_objective_numpy(w, blocks)
    if not np.all(np.isfinite(w)) or not math.isfinite(after):
        raise RuntimeError("V64.3.34 structured-margin solver produced non-finite parameters")
    if after > before + 1.0e-7 * max(1.0, abs(before)):
        raise RuntimeError(f"V64.3.34 structured-margin solver increased objective: {before} -> {after}")
    info = {
        "solver": "torch_cpu_float64_lbfgs_strong_wolfe",
        "max_iter": LBFGS_MAX_ITER,
        "history_size": LBFGS_HISTORY,
        "closure_evaluations": int(eval_count),
        "objective_at_zero": float(before),
        "objective_final": float(after),
        "scene_constraint_count": len(blocks),
        "weight_l2": float(np.linalg.norm(w)),
    }
    return w, scale, info


def _structured_scores(ss: list[dict[str, Any]], model: tuple[np.ndarray, np.ndarray, dict[str, Any]]) -> np.ndarray:
    if not ss:
        return np.zeros((0,), dtype=np.float64)
    w, scale, _ = model
    X = np.stack([np.asarray(a["x"], dtype=np.float64) for a in ss])
    return np.clip((X / scale[None, :]) @ w, -40.0, 40.0)


def _extended_diag(vals: list[float], captured: int, opp: int, noopp_selected: int, opp_selected: int, noopp_scenes: int) -> dict[str, Any]:
    d = _diag(vals, captured, opp, noopp_selected)
    opp_vals_n = int(opp_selected)
    d.update({
        "positive_opportunity_selected_count": opp_vals_n,
        "positive_opportunity_selected_precision": float(captured / max(opp_vals_n, 1)),
        "positive_opportunity_proposal_rate": float(opp_vals_n / max(opp, 1)),
        "no_positive_opportunity_scene_count": int(noopp_scenes),
        "no_positive_opportunity_false_intervention_rate": float(noopp_selected / max(noopp_scenes, 1)),
    })
    return d


def _pair_boundary_mass_audit(scene_map: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    opp_mass: list[float] = []
    noopp_mass: list[float] = []
    opp_counts: list[int] = []
    noopp_counts: list[int] = []
    for ss in scene_map.values():
        if not ss:
            continue
        m = len(ss)
        if max(float(a["y"]) for a in ss) > 0.0:
            # V33 averages teacher-best vs all m rivals (incumbent + m-1 alternatives),
            # so the intervention-existence boundary gets exactly 1/m scene mass.
            opp_mass.append(1.0 / m)
            opp_counts.append(m)
        else:
            # Incumbent is teacher-best; all pairs are incumbent-vs-challenger.
            noopp_mass.append(1.0)
            noopp_counts.append(m)
    return {
        "opportunity_scene_count": len(opp_mass),
        "no_opportunity_scene_count": len(noopp_mass),
        "opportunity_candidate_count_mean": float(np.mean(opp_counts)),
        "no_opportunity_candidate_count_mean": float(np.mean(noopp_counts)),
        "v33_incumbent_boundary_loss_mass_per_opportunity_scene_mean": float(np.mean(opp_mass)),
        "v33_incumbent_boundary_loss_mass_per_no_opportunity_scene_mean": float(np.mean(noopp_mass)),
        "v33_noop_to_opportunity_boundary_mass_ratio": float(np.mean(noopp_mass) / max(float(np.mean(opp_mass)), EPS)),
        "interpretation": "V33 all-rivals averaging puts all scene loss on incumbent separation in no-opportunity scenes but only 1/m on challenger-vs-incumbent separation in opportunity scenes; V34 removes this candidate-count-dependent asymmetry by optimizing one worst regret violation per scene.",
    }


def _nested_diagnostics(groups: dict[str, list[dict[str, Any]]], audit_csv: Path) -> dict[str, Any]:
    scene_map = {t: _scene_samples(g) for t, g in groups.items()}
    scene_map = {t: ss for t, ss in scene_map.items() if ss}
    folds: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    names = ["mean", "pair", "rsm", "main"]
    agg = {n: [] for n in names}
    caps = {n: 0 for n in names}
    noops = {n: 0 for n in names}
    oppsels = {n: 0 for n in names}
    total_opp = 0
    total_noopp = 0

    for k in range(FOLDS):
        test = [t for t in scene_map if _fold(t) == k]
        cal_fold = (k + 1) % FOLDS
        cal = [t for t in scene_map if _fold(t) == cal_fold]
        fit = [t for t in scene_map if _fold(t) not in {k, cal_fold}]
        fit_samples = [a for t in fit for a in scene_map[t]]
        mean_model = _fit_mean_ridge(fit_samples)
        pair_model = _fit_pair_gap(scene_map, fit)
        rsm_model = _fit_regret_structured_margin(scene_map, fit)

        cal_res: list[float] = []
        for t in cal:
            ss = scene_map[t]
            score = _structured_scores(ss, rsm_model)
            pi = _select(ss, score)
            if pi is not None:
                cal_res.append(float(score[pi] - float(ss[pi]["y"])))
        if len(cal_res) < MIN_NESTED_CAL_PROPOSALS:
            q = float("inf")
            qidx = -1
        else:
            q, qidx = _conformal_q(cal_res)

        fold_vals = {n: [] for n in names}
        fold_cap = {n: 0 for n in names}
        fold_noop = {n: 0 for n in names}
        fold_oppsel = {n: 0 for n in names}
        opp = 0
        noop_scenes = 0
        for t in test:
            ss = scene_map[t]
            y = np.asarray([float(a["y"]) for a in ss], dtype=np.float64)
            has_opp = bool(np.any(y > 0.0))
            opp += int(has_opp)
            noop_scenes += int(not has_opp)

            mmu, _ = _predict_mean(ss, mean_model)
            mi = _select(ss, mmu)
            pscore = _pair_scores(ss, pair_model)
            pi = _select(ss, pscore)
            rscore = _structured_scores(ss, rsm_model)
            ri = _select(ss, rscore)
            main_i = ri if (ri is not None and math.isfinite(q) and float(rscore[ri] - q) > 0.0) else None
            chosen = {"mean": mi, "pair": pi, "rsm": ri, "main": main_i}

            for name, idx in chosen.items():
                if idx is None:
                    continue
                yy = float(y[idx])
                fold_vals[name].append(yy)
                fold_cap[name] += int(has_opp and yy > 0.0)
                fold_noop[name] += int(not has_opp)
                fold_oppsel[name] += int(has_opp)

            audits.append({
                "scenario_token": t,
                "outer_test_fold": k,
                "calibration_fold": cal_fold,
                "candidate_count": len(ss),
                "positive_opportunity": int(has_opp),
                "teacher_best_action": int(ss[_teacher_best_index(ss)]["action"]) if _teacher_best_index(ss) >= 0 else -1,
                "teacher_best_improvement": max(0.0, float(np.max(y))),
                "mean_selected_action": -1 if mi is None else int(ss[mi]["action"]),
                "mean_selected_score": float("nan") if mi is None else float(mmu[mi]),
                "mean_selected_teacher_improvement": float("nan") if mi is None else float(y[mi]),
                "pair_selected_action": -1 if pi is None else int(ss[pi]["action"]),
                "pair_selected_score": float("nan") if pi is None else float(pscore[pi]),
                "pair_selected_teacher_improvement": float("nan") if pi is None else float(y[pi]),
                "rsm_selected_action": -1 if ri is None else int(ss[ri]["action"]),
                "rsm_selected_score": float("nan") if ri is None else float(rscore[ri]),
                "rsm_selected_teacher_improvement": float("nan") if ri is None else float(y[ri]),
                "selected_policy_conformal_q": q,
                "main_selected_action": -1 if main_i is None else int(ss[main_i]["action"]),
                "main_selected_lower_bound": float("nan") if ri is None or not math.isfinite(q) else float(rscore[ri] - q),
                "main_selected_teacher_improvement": float("nan") if main_i is None else float(y[main_i]),
                "pair_changes_mean_winner": int(mi != pi),
                "rsm_changes_mean_winner": int(mi != ri),
                "rsm_changes_pair_winner": int(pi != ri),
            })

        fd = {
            name: _extended_diag(fold_vals[name], fold_cap[name], opp, fold_noop[name], fold_oppsel[name], noop_scenes)
            for name in names
        }
        folds.append({
            "fold": k,
            "fit_scenes": len(fit),
            "calibration_scenes": len(cal),
            "test_scenes": len(test),
            "selected_policy_calibration_proposal_count": len(cal_res),
            "selected_policy_conformal_order_index_1based": qidx,
            "selected_policy_conformal_quantile": q,
            "structured_margin_solver": rsm_model[2],
            "mean_rank": fd["mean"],
            "v33_pair_gap_rank": fd["pair"],
            "regret_structured_margin_rank": fd["rsm"],
            "regret_structured_margin_policy_conformal": fd["main"],
        })
        total_opp += opp
        total_noopp += noop_scenes
        for name in names:
            agg[name].extend(fold_vals[name])
            caps[name] += fold_cap[name]
            noops[name] += fold_noop[name]
            oppsels[name] += fold_oppsel[name]

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(audits[0].keys()) if audits else []
    with audit_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(audits)

    A = {
        name: _extended_diag(agg[name], caps[name], total_opp, noops[name], oppsels[name], total_noopp)
        for name in names
    }
    mean, pair, rsm, main = A["mean"], A["pair"], A["rsm"], A["main"]
    intervention_existence_gain = bool(rsm["no_positive_opportunity_false_intervention_count"] < mean["no_positive_opportunity_false_intervention_count"])
    opportunity_recovery_gain = bool(rsm["positive_capture_count"] > pair["positive_capture_count"])
    selected_path_direction_gain = bool(
        rsm["teacher_improvement_sum"] > pair["teacher_improvement_sum"] + 1.0e-9
        and rsm["catastrophic_count"] < mean["catastrophic_count"]
    )
    rank_mechanism_pass = bool(
        rsm["selected_count"] > 0
        and intervention_existence_gain
        and opportunity_recovery_gain
        and selected_path_direction_gain
    )
    train_pass = bool(
        rank_mechanism_pass
        and all(fr["regret_structured_margin_policy_conformal"]["path_nonharmful"] for fr in folds)
        and main["selected_count"] >= 64
        and main["selected_positive_count"] >= 32
        and main["catastrophic_count"] == 0
    )

    if not intervention_existence_gain:
        diagnosis = "structured_margin_does_not_improve_should_we_intervene_vs_corrected_mean"
    elif not opportunity_recovery_gain:
        diagnosis = "structured_margin_still_over_suppresses_positive_opportunity_recovery_vs_v33_pair"
    elif not selected_path_direction_gain:
        diagnosis = "structured_margin_recovers_coverage_but_which_intervention_ordering_or_tail_remains_unreliable"
    elif any(fr["selected_policy_calibration_proposal_count"] < MIN_NESTED_CAL_PROPOSALS for fr in folds):
        diagnosis = "ranker_direction_improves_but_selected_policy_calibration_support_is_too_sparse_for_frozen_nested_protocol"
    elif not all(fr["regret_structured_margin_policy_conformal"]["path_nonharmful"] for fr in folds):
        diagnosis = "ranker_improves_both_factors_but_selected_policy_certificate_does_not_control_all_fold_tail"
    elif main["selected_count"] < 64 or main["selected_positive_count"] < 32:
        diagnosis = "certificate_controls_tail_but_recovery_coverage_is_insufficient"
    else:
        diagnosis = "complete_nested_RSMR_gate_pass"

    return {
        "folds": folds,
        "mean_rank_aggregate": mean,
        "v33_pair_gap_rank_aggregate": pair,
        "regret_structured_margin_rank_aggregate": rsm,
        "regret_structured_margin_policy_conformal_aggregate": main,
        "fold_pass_count": sum(int(fr["regret_structured_margin_policy_conformal"]["path_nonharmful"]) for fr in folds),
        "intervention_existence_gain_vs_mean": intervention_existence_gain,
        "opportunity_recovery_gain_vs_v33_pair": opportunity_recovery_gain,
        "selected_path_direction_gain_vs_v33_pair_and_mean_tail": selected_path_direction_gain,
        "rank_mechanism_pass": rank_mechanism_pass,
        "train_gate_pass": train_pass,
        "failure_diagnosis": diagnosis,
        "scene_audit_csv": str(audit_csv),
    }


def _all_samples(groups: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    scene_map = {t: _scene_samples(g) for t, g in groups.items()}
    scene_map = {t: ss for t, ss in scene_map.items() if ss}
    return scene_map, [a for ss in scene_map.values() for a in ss]


def _base_cfg(path: str) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    ic = cfg.setdefault("runtime", {}).setdefault("decisive_frontier_value", {}).setdefault("incumbent_contrastive_extremal_recovery", {})
    ic["incumbent_retention_policy"] = "preserve_admissible_incumbent"
    ic["regret_risk_enabled"] = False
    ic["retention_regret_risk_enabled"] = False
    ic["replacement_regret_risk_enabled"] = False
    return cfg


def _write_preserve(base: dict[str, Any], path: str) -> None:
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic["selection_conditioned_intervention_recovery"] = {"enabled": False}
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.34-EAF-ICER-PRESERVE-CONTROL"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.34-EAF-ICER-PRESERVE-CONTROL"
    cfg.setdefault("experiment", {})["name"] = "v64_3_34_preserve_control"
    cfg["experiment"]["algorithm"] = "V64.3.34 preservation control: admissible-incumbent default with frozen V20 direct semantics"
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _write_linear_cfg(
    base: dict[str, Any], path: str, *, version: str, name: str, algorithm: str, mode: str,
    w: np.ndarray, mean: np.ndarray, std: np.ndarray, bias: float, model_type: str,
    training_target: str, training_weighting: str,
) -> None:
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    scir = ic.setdefault("selection_conditioned_intervention_recovery", {})
    scir.update({
        "enabled": True,
        "mode": mode,
        "model_type": model_type,
        "base_feature_names": BASE_FEATURE_NAMES,
        "feature_names": FEATURE_NAMES,
        "feature_mean": [float(x) for x in mean],
        "feature_std": [float(x) for x in std],
        "weights": [float(x) for x in w],
        "bias": float(bias),
        "ridge_lambda": RIDGE_LAMBDA,
        "leverage_inverse": [],
        "selection_scale_floor": 1.0,
        "training_population": "TRAIN_only_incumbent_deployment_admissible_support_positive_direct_scenes",
        "training_weighting": training_weighting,
        "training_target": training_target,
        "proposal_operator": "argmax_positive_structured_candidate_score_with_incumbent_zero_pseudoitem",
        "require_positive_predicted_improvement": True,
        "conformal_alpha": ALPHA,
        "conformal_overprediction_quantile": 0.0,
        "calibration_status": "not_yet_selected_policy_calibrated" if mode == "rank_only" else "control",
        "no_fallback": True,
    })
    cfg.setdefault("metadata", {})["algorithm_version"] = version
    cfg.setdefault("provenance", {})["algorithm_version"] = version
    exp = cfg.setdefault("experiment", {})
    exp["name"] = name
    exp["algorithm"] = algorithm
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.34 regret-structured margin selector and nested selected-policy conformal gate.")
    ap.add_argument("--train-frontier-edges", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-preserve-config", required=True)
    ap.add_argument("--output-mean-config", required=True)
    ap.add_argument("--output-pair-config", required=True)
    ap.add_argument("--output-rank-config", required=True)
    ap.add_argument("--output-report", required=True)
    ap.add_argument("--output-scene-audit", required=True)
    a = ap.parse_args()

    _, groups = _read_edges(Path(a.train_frontier_edges))
    nested = _nested_diagnostics(groups, Path(a.output_scene_audit))
    scene_map, samples = _all_samples(groups)
    y = np.asarray([float(x["y"]) for x in samples], dtype=np.float64)
    report: dict[str, Any] = {
        "audit": "v64_3_34_eaf_icer_rsmr_fit",
        "scientific_role": "TRAIN_only_incumbent_augmented_regret_structured_margin_ordering_plus_selected_policy_conformal_gate",
        "frozen_train_scenes": len(groups),
        "direct_support_positive_training_edges": len(samples),
        "direct_support_positive_training_scenes": len(scene_map),
        "teacher_improvement_positive_fraction": float((y > 0).mean()),
        "teacher_improvement_sum": float(y.sum()),
        "feature_names": FEATURE_NAMES,
        "ridge_lambda": RIDGE_LAMBDA,
        "conformal_alpha": ALPHA,
        "v33_pair_objective_boundary_mass_audit": _pair_boundary_mass_audit(scene_map),
        "structured_objective": "cost_sensitive_incumbent_augmented_scene_max_squared_hinge_teacher_regret_margin_plus_fixed_l2",
        "structured_regret_upper_bound": "For teacher-best t and runtime argmax a_hat over incumbent plus challengers, max_r[Delta_t-Delta_r-(s_t-s_r)]_+ >= Delta_t-Delta_a_hat because s_a_hat>=s_t. The training loss therefore directly upper-bounds selected-action teacher regret gap before statistical calibration.",
        "incumbent_pseudoitem": "x_i=0, Delta_i=0, score_i=0; no-opportunity scenes explicitly optimize incumbent as teacher-best",
        "policy_calibration": "freeze regret-structured selector; on disjoint calibration fold collect one score-y residual for its emitted positive proposal; one-sided split conformal q; MAIN may execute only the same proposal or return incumbent",
        "nested_crossfit": nested,
        "train_gate_pass": bool(nested["train_gate_pass"]),
        "train_gate_contract": {
            "outer_folds": 5,
            "within_outer_split": "3_folds_fit_1_fold_selected_policy_calibrate_1_fold_test",
            "rank_must_reduce_no_opportunity_false_interventions_vs_corrected_mean": True,
            "rank_must_increase_positive_capture_vs_v33_pair_gap": True,
            "rank_must_improve_selected_sum_vs_v33_pair_and_catastrophes_vs_mean": True,
            "nested_selected_policy_calibration_proposals_min_per_fold": MIN_NESTED_CAL_PROPOSALS,
            "all_folds_main_selected_path_sum_nonnegative_and_no_catastrophe": True,
            "aggregate_main_selected_count_min": 64,
            "aggregate_main_selected_positive_count_min": 32,
            "catastrophic_threshold": CAT,
            "no_lambda_alpha_feature_threshold_or_optimizer_sweep": True,
        },
        "fit_uses_validation": False,
        "fit_uses_test": False,
    }
    rp = Path(a.output_report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["train_gate_pass"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(f"V64.3.34 RSMR nested TRAIN gate failed ({nested['failure_diagnosis']}); STOP before CAL/fresh selection")

    base = _base_cfg(a.base_config)
    _write_preserve(base, a.output_preserve_config)

    mean_model = _fit_mean_ridge(samples)
    mw, mb, mmean, mstd, _ = mean_model
    _write_linear_cfg(
        base, a.output_mean_config,
        version="V64.3.34-EAF-ICER-MEAN-WEIGHTFIX",
        name="v64_3_34_mean_weightfix_control",
        algorithm="V64.3.34 causal control: corrected V32.1 same-scene edge-mean ridge ordering",
        mode="mean_rank", w=mw, mean=mmean, std=mstd, bias=mb,
        model_type="v32_1_scene_equal_edge_mean_ridge_control",
        training_target="continuous_teacher_candidate_minus_incumbent",
        training_weighting="each_scene_total_edge_loss_mass_1",
    )

    pw, pscale = _fit_pair_gap(scene_map, list(scene_map))
    zeros = np.zeros_like(pscale)
    _write_linear_cfg(
        base, a.output_pair_config,
        version="V64.3.34-EAF-ICER-SPCR-PAIR-CONTROL",
        name="v64_3_34_spcr_pair_control",
        algorithm="V64.3.34 causal control: exact V33 incumbent-augmented all-rivals pair-gap structured selector",
        mode="rank_only", w=pw, mean=zeros, std=pscale, bias=0.0,
        model_type="v33_scene_equal_incumbent_augmented_teacher_best_pair_gap_control",
        training_target="teacher_best_including_incumbent_vs_each_rival_continuous_gap",
        training_weighting="each_direct_scene_total_all_rivals_pair_loss_mass_1",
    )

    rw, rscale, rinfo = _fit_regret_structured_margin(scene_map, list(scene_map))
    rzeros = np.zeros_like(rscale)
    _write_linear_cfg(
        base, a.output_rank_config,
        version="V64.3.34-EAF-ICER-RSMR-RANK",
        name="v64_3_34_eaf_icer_rsmr_rank",
        algorithm="V64.3.34 RSMR rank: incumbent-augmented cost-sensitive scene-max regret-structured margin ordering",
        mode="rank_only", w=rw, mean=rzeros, std=rscale, bias=0.0,
        model_type="incumbent_augmented_scene_max_teacher_regret_structured_margin",
        training_target="teacher_best_vs_worst_cost_augmented_rival_structured_regret_margin",
        training_weighting="one_max_regret_violation_per_direct_scene_plus_fixed_l2",
    )
    report["full_rsm_model"] = {
        "weights": [float(x) for x in rw],
        "feature_mean": [0.0 for _ in rscale],
        "feature_std": [float(x) for x in rscale],
        "bias": 0.0,
        "incumbent_runtime_score": 0.0,
        "solver": rinfo,
    }
    rp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "pass": True,
        "train_gate_pass": True,
        "fold_pass_count": nested["fold_pass_count"],
        "main_selected_count": nested["regret_structured_margin_policy_conformal_aggregate"]["selected_count"],
        "output_rank_config": a.output_rank_config,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
