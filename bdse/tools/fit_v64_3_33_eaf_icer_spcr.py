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
from bdse.tools.fit_v64_3_32_1_eaf_icer_ssir_weightfix import _fit_ridge as _fit_mean_ridge, _predict as _predict_mean

BASE_FEATURE_NAMES = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
FEATURE_NAMES = [f"delta::{n}" for n in BASE_FEATURE_NAMES] + ["delta::support_logit"]
EPS = 1.0e-12
EXPECTED_FRONTIER_ROWS = 75133
EXPECTED_SCENES = 3000
FOLDS = 5
RIDGE_LAMBDA = 1.0
ALPHA = 0.05
CAT = -0.5
MIN_NESTED_CAL_PROPOSALS = 32


def _finite(r: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        v = float(r.get(key, default))
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _fold(token: str) -> int:
    # Reuse the exact V32 split so every V33 diagnostic is paired with the repaired
    # V32.1 mechanism result rather than gaining from a new development partition.
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
        raise SystemExit(f"V64.3.33 requires frozen B16 TRAIN frontier rows={len(rows)} expected={EXPECTED_FRONTIER_ROWS}")
    if len(groups) != EXPECTED_SCENES:
        raise SystemExit(f"V64.3.33 requires frozen B16 TRAIN scenes={len(groups)} expected={EXPECTED_SCENES}")
    return rows, groups


def _scene_samples(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not group:
        return []
    inc = int(group[0].get("raw_top_action", -1))
    ir = next((r for r in group if int(r.get("challenger_action", -2)) == inc), None)
    if ir is None or _finite(ir, "icer_admissible", 0.0) < 0.5:
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


def _teacher_best_index(ss: list[dict[str, Any]]) -> int:
    """Return -1 for incumbent pseudo-item, otherwise candidate list index.

    The incumbent has exact teacher improvement 0 and wins every non-positive or
    zero-tie scene.  This is the crucial V33 selection-consistency change: a
    no-opportunity scene explicitly teaches *no intervention* rather than merely
    contributing many unrelated negative edge regression targets.
    """
    if not ss:
        return -1
    best_y = max(float(a["y"]) for a in ss)
    if best_y <= 0.0:
        return -1
    cand = [j for j, a in enumerate(ss) if abs(float(a["y"]) - best_y) <= 1.0e-12]
    return sorted(cand, key=lambda j: (-float(ss[j]["support"]), -float(ss[j]["margin"]), -int(ss[j]["utility_prior"]), int(ss[j]["action"])))[0]


def _pair_rows(scene_map: dict[str, list[dict[str, Any]]], tokens: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    z0 = np.zeros((len(FEATURE_NAMES),), dtype=np.float64)
    for tok in tokens:
        ss = scene_map.get(tok, [])
        if not ss:
            continue
        bi = _teacher_best_index(ss)
        bx = z0 if bi < 0 else np.asarray(ss[bi]["x"], dtype=np.float64)
        by = 0.0 if bi < 0 else float(ss[bi]["y"])
        # Rivals are the incumbent plus every alternative except the teacher-best.
        rivals: list[tuple[np.ndarray, float, int]] = [(z0, 0.0, -1)]
        rivals.extend((np.asarray(a["x"], dtype=np.float64), float(a["y"]), j) for j, a in enumerate(ss))
        rivals = [r for r in rivals if r[2] != bi]
        n = len(rivals)
        if n <= 0:
            continue
        for rx, ry, _ in rivals:
            gap = float(by - ry)
            if gap < -1.0e-10:
                raise RuntimeError("teacher-best pair gap became negative")
            rows.append({"token": tok, "d": bx - rx, "gap": max(gap, 0.0), "weight": 1.0 / n})
    return rows


def _fit_pair_gap(scene_map: dict[str, list[dict[str, Any]]], tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pairs = _pair_rows(scene_map, tokens)
    if not pairs:
        raise ValueError("V33 structured pair-gap fit has no pair samples")
    D = np.stack([r["d"] for r in pairs])
    g = np.asarray([float(r["gap"]) for r in pairs], dtype=np.float64)
    wloss = np.asarray([float(r["weight"]) for r in pairs], dtype=np.float64)
    # Each direct scene contributes total pair-loss mass exactly one.  Use a
    # zero-preserving RMS scale instead of mean centering so the incumbent
    # pseudo-item remains exactly score 0 at runtime.
    pm = wloss / max(float(wloss.sum()), EPS)
    scale = np.sqrt(np.sum((D * D) * pm[:, None], axis=0))
    scale = np.maximum(scale, 1.0e-6)
    Z = D / scale[None, :]
    root = np.sqrt(wloss)[:, None]
    Zw = Z * root
    gw = g * root[:, 0]
    A = Zw.T @ Zw + np.eye(Z.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    rhs = Zw.T @ gw
    coef = np.linalg.solve(A, rhs)
    return coef, scale


def _pair_scores(ss: list[dict[str, Any]], model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if not ss:
        return np.zeros((0,), dtype=np.float64)
    w, scale = model
    X = np.stack([a["x"] for a in ss])
    return np.clip((X / scale[None, :]) @ w, -40.0, 40.0)


def _select(ss: list[dict[str, Any]], score: np.ndarray) -> int | None:
    cand = [j for j, v in enumerate(score.tolist()) if math.isfinite(v) and v > 0.0]
    if not cand:
        return None
    return sorted(cand, key=lambda j: (-float(score[j]), -float(ss[j]["support"]), -float(ss[j]["margin"]), -int(ss[j]["utility_prior"]), int(ss[j]["action"])))[0]


def _conformal_q(vals: list[float], alpha: float = ALPHA) -> tuple[float, int]:
    arr = np.asarray([float(x) for x in vals if math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        raise ValueError("V33 selected-policy conformal calibration has no finite proposal residuals")
    arr.sort()
    k = int(math.ceil((arr.size + 1) * (1.0 - alpha)))
    k = min(max(k, 1), int(arr.size))
    return max(0.0, float(arr[k - 1])), k


def _diag(vals: list[float], captured: int, opp: int, noopp_selected: int) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    neg = np.minimum(arr, 0.0) if arr.size else arr
    return {
        "selected_count": int(arr.size),
        "selected_positive_count": int((arr > 0.0).sum()) if arr.size else 0,
        "selected_precision": float((arr > 0.0).mean()) if arr.size else float("nan"),
        "teacher_improvement_sum": float(arr.sum()) if arr.size else 0.0,
        "teacher_improvement_worst": float(arr.min()) if arr.size else float("nan"),
        "teacher_negative_rms": float(np.sqrt(np.mean(neg * neg))) if arr.size else 0.0,
        "catastrophic_count": int((arr <= CAT).sum()) if arr.size else 0,
        "positive_opportunity_scenes": int(opp),
        "positive_capture_count": int(captured),
        "positive_capture_rate": float(captured / max(opp, 1)),
        "no_positive_opportunity_false_intervention_count": int(noopp_selected),
        "path_nonharmful": bool(arr.size > 0 and float(arr.sum()) >= -1.0e-9 and int((arr <= CAT).sum()) == 0),
    }


def _nested_diagnostics(groups: dict[str, list[dict[str, Any]]], audit_csv: Path) -> dict[str, Any]:
    scene_map = {t: _scene_samples(g) for t, g in groups.items()}
    scene_map = {t: ss for t, ss in scene_map.items() if ss}
    folds: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    agg = {"mean": [], "pair": [], "main": []}
    caps = {"mean": 0, "pair": 0, "main": 0}
    noops = {"mean": 0, "pair": 0, "main": 0}
    total_opp = 0

    for k in range(FOLDS):
        test = [t for t in scene_map if _fold(t) == k]
        cal_fold = (k + 1) % FOLDS
        cal = [t for t in scene_map if _fold(t) == cal_fold]
        fit = [t for t in scene_map if _fold(t) not in {k, cal_fold}]
        fit_samples = [a for t in fit for a in scene_map[t]]
        mean_model = _fit_mean_ridge(fit_samples)
        pair_model = _fit_pair_gap(scene_map, fit)

        cal_res: list[float] = []
        for t in cal:
            ss = scene_map[t]
            ps = _pair_scores(ss, pair_model)
            pi = _select(ss, ps)
            if pi is not None:
                cal_res.append(float(ps[pi] - float(ss[pi]["y"])))
        if len(cal_res) < MIN_NESTED_CAL_PROPOSALS:
            q = float("inf"); qidx = -1
        else:
            q, qidx = _conformal_q(cal_res)

        fold_vals = {"mean": [], "pair": [], "main": []}
        fold_cap = {"mean": 0, "pair": 0, "main": 0}
        fold_noop = {"mean": 0, "pair": 0, "main": 0}
        opp = 0
        for t in test:
            ss = scene_map[t]
            y = np.asarray([float(a["y"]) for a in ss], dtype=np.float64)
            has_opp = bool(np.any(y > 0.0)); opp += int(has_opp)
            mmu, _ = _predict_mean(ss, mean_model)
            mi = _select(ss, mmu)
            pscore = _pair_scores(ss, pair_model)
            pi = _select(ss, pscore)
            main_i = pi if (pi is not None and math.isfinite(q) and float(pscore[pi] - q) > 0.0) else None
            chosen = {"mean": mi, "pair": pi, "main": main_i}
            for name, idx in chosen.items():
                if idx is None:
                    continue
                yy = float(y[idx]); fold_vals[name].append(yy)
                fold_cap[name] += int(has_opp and yy > 0.0)
                fold_noop[name] += int((not has_opp))
            audits.append({
                "scenario_token": t,
                "outer_test_fold": k,
                "calibration_fold": cal_fold,
                "candidate_count": len(ss),
                "positive_opportunity": int(has_opp),
                "mean_selected_action": -1 if mi is None else int(ss[mi]["action"]),
                "mean_selected_score": float("nan") if mi is None else float(mmu[mi]),
                "mean_selected_teacher_improvement": float("nan") if mi is None else float(y[mi]),
                "pair_selected_action": -1 if pi is None else int(ss[pi]["action"]),
                "pair_selected_score": float("nan") if pi is None else float(pscore[pi]),
                "pair_selected_teacher_improvement": float("nan") if pi is None else float(y[pi]),
                "selected_policy_conformal_q": q,
                "main_selected_action": -1 if main_i is None else int(ss[main_i]["action"]),
                "main_selected_lower_bound": float("nan") if pi is None or not math.isfinite(q) else float(pscore[pi] - q),
                "main_selected_teacher_improvement": float("nan") if main_i is None else float(y[main_i]),
                "pair_changes_mean_winner": int(mi != pi),
            })

        fd = {name: _diag(fold_vals[name], fold_cap[name], opp, fold_noop[name]) for name in fold_vals}
        folds.append({
            "fold": k, "fit_scenes": len(fit), "calibration_scenes": len(cal), "test_scenes": len(test),
            "selected_policy_calibration_proposal_count": len(cal_res),
            "selected_policy_conformal_order_index_1based": qidx,
            "selected_policy_conformal_quantile": q,
            "mean_rank": fd["mean"], "pair_gap_rank": fd["pair"], "pair_gap_policy_conformal": fd["main"],
        })
        total_opp += opp
        for name in agg:
            agg[name].extend(fold_vals[name]); caps[name] += fold_cap[name]; noops[name] += fold_noop[name]

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(audits[0].keys()) if audits else []
    with audit_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(audits)

    A = {name: _diag(agg[name], caps[name], total_opp, noops[name]) for name in agg}
    main = A["main"]
    mean = A["mean"]
    pair = A["pair"]
    structured_ordering_pass = bool(
        pair["selected_count"] > 0
        and pair["no_positive_opportunity_false_intervention_count"] < mean["no_positive_opportunity_false_intervention_count"]
        and pair["catastrophic_count"] <= mean["catastrophic_count"]
        and (
            pair["teacher_improvement_sum"] > mean["teacher_improvement_sum"] + 1.0e-9
            or pair["catastrophic_count"] < mean["catastrophic_count"]
        )
    )
    train_pass = bool(
        structured_ordering_pass
        and all(fr["pair_gap_policy_conformal"]["path_nonharmful"] for fr in folds)
        and main["selected_count"] >= 64
        and main["selected_positive_count"] >= 32
        and main["catastrophic_count"] == 0
    )
    return {
        "folds": folds,
        "mean_rank_aggregate": A["mean"],
        "pair_gap_rank_aggregate": A["pair"],
        "pair_gap_policy_conformal_aggregate": A["main"],
        "fold_pass_count": sum(int(fr["pair_gap_policy_conformal"]["path_nonharmful"]) for fr in folds),
        "structured_ordering_mechanism_pass": structured_ordering_pass,
        "train_gate_pass": train_pass,
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
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.33-EAF-ICER-PRESERVE-CONTROL"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.33-EAF-ICER-PRESERVE-CONTROL"
    cfg.setdefault("experiment", {})["name"] = "v64_3_33_eaf_icer_preserve_control"
    cfg["experiment"]["algorithm"] = "V64.3.33 preservation control: admissible-incumbent default with frozen V20 direct semantics"
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _write_linear_cfg(base: dict[str, Any], path: str, *, version: str, name: str, algorithm: str, mode: str, w: np.ndarray, mean: np.ndarray, std: np.ndarray, bias: float, model_type: str, training_target: str, training_weighting: str) -> None:
    cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    scir = ic.setdefault("selection_conditioned_intervention_recovery", {})
    scir.update({
        "enabled": True, "mode": mode, "model_type": model_type,
        "base_feature_names": BASE_FEATURE_NAMES, "feature_names": FEATURE_NAMES,
        "feature_mean": [float(x) for x in mean], "feature_std": [float(x) for x in std],
        "weights": [float(x) for x in w], "bias": float(bias), "ridge_lambda": RIDGE_LAMBDA,
        "leverage_inverse": [], "selection_scale_floor": 1.0,
        "training_population": "TRAIN_only_incumbent_deployment_admissible_support_positive_direct_scenes",
        "training_weighting": training_weighting, "training_target": training_target,
        "proposal_operator": "argmax_positive_structured_candidate_score_with_incumbent_zero_pseudoitem",
        "require_positive_predicted_improvement": True, "conformal_alpha": ALPHA,
        "conformal_overprediction_quantile": 0.0,
        "calibration_status": "not_yet_selected_policy_calibrated" if mode == "rank_only" else "control",
        "no_fallback": True,
    })
    cfg.setdefault("metadata", {})["algorithm_version"] = version
    cfg.setdefault("provenance", {})["algorithm_version"] = version
    exp = cfg.setdefault("experiment", {}); exp["name"] = name; exp["algorithm"] = algorithm
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.33 selection-consistent pair-gap ranker and nested selected-policy conformal gate.")
    ap.add_argument("--train-frontier-edges", required=True); ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-preserve-config", required=True); ap.add_argument("--output-mean-config", required=True); ap.add_argument("--output-pair-config", required=True)
    ap.add_argument("--output-report", required=True); ap.add_argument("--output-scene-audit", required=True)
    a = ap.parse_args()
    _, groups = _read_edges(Path(a.train_frontier_edges))
    nested = _nested_diagnostics(groups, Path(a.output_scene_audit))
    scene_map, samples = _all_samples(groups)
    y = np.asarray([float(x["y"]) for x in samples], dtype=np.float64)
    report: dict[str, Any] = {
        "audit": "v64_3_33_eaf_icer_spcr_fit",
        "scientific_role": "TRAIN_only_selection_consistent_incumbent_augmented_pair_gap_ordering_plus_selected_policy_conformal_gate",
        "frozen_train_scenes": len(groups), "direct_support_positive_training_edges": len(samples), "direct_support_positive_training_scenes": len(scene_map),
        "teacher_improvement_positive_fraction": float((y > 0).mean()), "teacher_improvement_sum": float(y.sum()),
        "feature_names": FEATURE_NAMES, "ridge_lambda": RIDGE_LAMBDA, "conformal_alpha": ALPHA,
        "structured_objective": "scene_equal_teacher_best_including_incumbent_pseudoitem_vs_every_rival_pair_gap_ridge",
        "incumbent_pseudoitem": "x_i=0,y_i=0; incumbent wins every scene whose best alternative teacher improvement is <=0",
        "policy_calibration": "fit selector first; on disjoint calibration fold collect one residual score-y for the selector's positive proposal; one-sided split conformal q; deployment executes same proposal iff score-q>0",
        "nested_crossfit": nested, "train_gate_pass": bool(nested["train_gate_pass"]),
        "train_gate_contract": {"outer_folds": 5, "within_outer_split": "3_folds_fit_1_fold_selected_policy_calibrate_1_fold_test", "structured_rank_must_reduce_no_positive_opportunity_false_interventions_vs_corrected_mean": True, "structured_rank_catastrophic_count_nonworse_than_mean": True, "structured_rank_requires_strict_sum_or_catastrophe_improvement": True, "all_folds_selected_path_sum_nonnegative_and_no_catastrophe": True, "aggregate_selected_count_min": 64, "aggregate_selected_positive_count_min": 32, "catastrophic_threshold": CAT, "no_lambda_alpha_feature_or_threshold_sweep": True},
        "fit_uses_validation": False, "fit_uses_test": False,
    }
    rp = Path(a.output_report); rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["train_gate_pass"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("V64.3.33 SPCR nested TRAIN selected-policy gate failed; STOP before CAL/fresh selection")

    base = _base_cfg(a.base_config); _write_preserve(base, a.output_preserve_config)
    mean_model = _fit_mean_ridge(samples); mw, mb, mmean, mstd, _ = mean_model
    _write_linear_cfg(base, a.output_mean_config, version="V64.3.33-EAF-ICER-MEAN-WEIGHTFIX", name="v64_3_33_mean_weightfix_control", algorithm="V64.3.33 causal control: corrected V32.1 same-scene edge-mean ridge ordering", mode="mean_rank", w=mw, mean=mmean, std=mstd, bias=mb, model_type="v32_1_scene_equal_edge_mean_ridge_control", training_target="continuous_teacher_candidate_minus_incumbent", training_weighting="each_scene_total_edge_loss_mass_1")
    pw, pscale = _fit_pair_gap(scene_map, list(scene_map))
    zeros = np.zeros_like(pscale)
    _write_linear_cfg(base, a.output_pair_config, version="V64.3.33-EAF-ICER-SPCR-RANK", name="v64_3_33_eaf_icer_spcr_rank", algorithm="V64.3.33 SPCR rank control: incumbent-augmented teacher-best-vs-rivals pair-gap structured ordering", mode="rank_only", w=pw, mean=zeros, std=pscale, bias=0.0, model_type="scene_equal_incumbent_augmented_teacher_best_pair_gap_ridge", training_target="teacher_best_including_incumbent_vs_each_rival_continuous_gap", training_weighting="each_direct_scene_total_pair_loss_mass_1")
    report["full_pair_model"] = {"weights": [float(x) for x in pw], "feature_mean": [0.0 for _ in pscale], "feature_std": [float(x) for x in pscale], "bias": 0.0, "incumbent_runtime_score": 0.0}
    rp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"pass": True, "train_gate_pass": True, "fold_pass_count": nested["fold_pass_count"], "main_selected_count": nested["pair_gap_policy_conformal_aggregate"]["selected_count"], "output_pair_config": a.output_pair_config}, sort_keys=True))


if __name__ == "__main__":
    main()
