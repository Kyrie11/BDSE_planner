from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ALPHA = 0.05
EXPECTED_SCENES = 500
MIN_DIRECT_ELIGIBLE_SCENES = 64


def _f(r: dict[str, Any], k: str, d: float = float("nan")) -> float:
    try:
        v = float(r.get(k, d))
    except (TypeError, ValueError):
        return d
    return v if math.isfinite(v) else d


def _q(scores: list[float], alpha: float) -> tuple[float, int]:
    arr = np.asarray([float(x) for x in scores if math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        raise SystemExit("V64.3.32 independent calibration has no finite direct-eligible scene scores")
    arr.sort()
    k = int(math.ceil((arr.size + 1) * (1.0 - alpha)))
    k = min(max(k, 1), int(arr.size))
    # Non-negative clipping can only make the lower bound more conservative when
    # the finite-sample quantile happens to be negative.
    return max(0.0, float(arr[k - 1])), k


def _row_tokens(path: Path) -> list[str]:
    toks: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                toks.append(str(json.loads(line).get("scenario_token", "")))
    return toks


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate V64.3.32 SSIR scene-simultaneous candidate-specific lower bounds on independent CAL500.")
    ap.add_argument("--calibration-rows", required=True)
    ap.add_argument("--calibration-edges", required=True)
    ap.add_argument("--mean-config", required=True)
    ap.add_argument("--output-main-config", required=True)
    ap.add_argument("--output-report", required=True)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()
    if abs(float(args.alpha) - ALPHA) > 1.0e-12:
        raise SystemExit("V64.3.32 conformal alpha is frozen at 0.05; no sweep permitted")

    row_tokens = _row_tokens(Path(args.calibration_rows))
    if len(row_tokens) != EXPECTED_SCENES or len(set(row_tokens)) != EXPECTED_SCENES:
        raise SystemExit(f"V64.3.32 CAL rows must contain exactly {EXPECTED_SCENES} unique scenes")
    row_set = set(row_tokens)

    groups: dict[str, list[dict[str, Any]]] = {t: [] for t in row_tokens}
    with Path(args.calibration_edges).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            tok = str(r.get("scenario_token", ""))
            if tok not in row_set:
                raise SystemExit(f"V64.3.32 CAL edge contains token outside CAL500 rows: {tok}")
            groups[tok].append(r)

    # Calibration population is the deployment-relevant direct domain: the raw
    # incumbent is deployment-admissible and there is at least one admissible,
    # support-positive alternative.  Exactly one max score is contributed by
    # each such scene, so candidate multiplicity cannot overweight a scene.
    scores: list[float] = []
    eligible_counts: list[int] = []
    unscaled_scores: list[float] = []
    eligible_tokens: list[str] = []
    for tok in row_tokens:
        rs = groups[tok]
        if not rs:
            continue
        inc = int(rs[0].get("raw_top_action", -1))
        by = {int(r.get("challenger_action", -2)): r for r in rs}
        ir = by.get(inc)
        if ir is None or _f(ir, "icer_admissible", 0.0) < 0.5:
            continue
        inc_tm = _f(ir, "teacher_margin")
        if not math.isfinite(inc_tm):
            raise SystemExit(f"non-finite incumbent teacher margin in CAL scene {tok}")
        vals: list[float] = []
        vals_raw: list[float] = []
        for r in rs:
            act = int(r.get("challenger_action", -2))
            if act == inc or _f(r, "icer_admissible", 0.0) < 0.5 or _f(r, "icer_support_logit", -math.inf) <= 0.0:
                continue
            y = _f(r, "teacher_margin") - inc_tm
            mu = _f(r, "icer_scir_predicted_improvement")
            scale = _f(r, "icer_scir_selection_scale", 1.0)
            if not (math.isfinite(y) and math.isfinite(mu) and math.isfinite(scale) and scale > 0.0):
                raise SystemExit(f"non-finite SSIR calibration edge in scene {tok}")
            vals.append((mu - y) / scale)
            vals_raw.append(mu - y)
        if not vals:
            continue
        eligible_tokens.append(tok)
        scores.append(float(max(vals)))
        unscaled_scores.append(float(max(vals_raw)))
        eligible_counts.append(len(vals))

    n = len(scores)
    if n < MIN_DIRECT_ELIGIBLE_SCENES:
        raise SystemExit(
            f"V64.3.32 independent CAL500 has too few direct-eligible scenes: {n} < {MIN_DIRECT_ELIGIBLE_SCENES}"
        )
    q, k = _q(scores, ALPHA)
    q_unscaled, _ = _q(unscaled_scores, ALPHA)

    cfg = yaml.safe_load(Path(args.mean_config).read_text(encoding="utf-8"))
    ic = cfg.get("runtime", {}).get("decisive_frontier_value", {}).get("incumbent_contrastive_extremal_recovery", {}) or {}
    scir = ic.get("selection_conditioned_intervention_recovery", {}) or {}
    if not bool(scir.get("enabled", False)) or str(scir.get("mode", "")) not in {"mean_rank", "rank_only"}:
        raise SystemExit("mean config is not a V64.3.32 SSIR mean-control artifact")
    if abs(float(scir.get("conformal_alpha", ALPHA)) - ALPHA) > 1.0e-12:
        raise SystemExit("mean config alpha mismatch")
    names = list(scir.get("feature_names", []))
    lev = np.asarray(scir.get("leverage_inverse", []), dtype=np.float64)
    if lev.shape != (len(names), len(names)) or not np.all(np.isfinite(lev)):
        raise SystemExit("mean config missing valid finite leverage matrix")

    scir["mode"] = "simultaneous_lcb"
    scir["simultaneous_conformal_quantile"] = float(q)
    scir["simultaneous_calibration_status"] = "independent_cal500_direct_eligible_scene_uniform_all_candidates_frozen_before_double_fresh"
    scir["simultaneous_calibration_total_scene_count"] = EXPECTED_SCENES
    scir["simultaneous_calibration_direct_eligible_scene_count"] = n
    scir["simultaneous_conformal_order_index_1based"] = int(k)
    scir["proposal_operator"] = "argmax_positive_scene_simultaneous_candidate_specific_lower_bound"
    ic["selection_conditioned_intervention_recovery"] = scir
    cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"] = ic
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.32-EAF-ICER-SSIR"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.32-EAF-ICER-SSIR"
    exp = cfg.setdefault("experiment", {})
    exp["name"] = "v64_3_32_eaf_icer_ssir"
    exp["algorithm"] = "V64.3.32 EAF-ICER-SSIR: Selection-Stable Intervention Recovery with direct-domain scene-simultaneous conformal lower bounds"
    exp["mechanism_chain"] = "bounded B16 interface -> exact EAF attribution -> deployment-admissible same-scene incumbent contrast -> continuous mean improvement + frozen ridge-leverage normalization -> one scene-level simultaneous conformal score over all direct candidates -> extremal selection by positive candidate-specific lower bound -> incumbent default -> unchanged final/structural guards"
    Path(args.output_main_config).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    arr = np.asarray(scores, dtype=np.float64)
    report = {
        "audit": "v64_3_32_eaf_icer_ssir_independent_calibration",
        "alpha": ALPHA,
        "calibration_total_scene_count": EXPECTED_SCENES,
        "direct_eligible_scene_count": n,
        "direct_eligible_scene_fraction": float(n / EXPECTED_SCENES),
        "direct_eligible_scene_count_min": MIN_DIRECT_ELIGIBLE_SCENES,
        "scene_score_definition": "for each direct-eligible scene: max_over_all_direct_admissible_support_positive_candidates_of_(predicted_improvement-teacher_improvement)/ridge_leverage_scale",
        "scene_simultaneous_quantile": float(q),
        "unscaled_scene_max_quantile_diagnostic": float(q_unscaled),
        "conformal_order_index_1based": int(k),
        "eligible_candidate_count_mean": float(np.mean(eligible_counts)),
        "eligible_candidate_count_max": int(max(eligible_counts) if eligible_counts else 0),
        "scene_score_mean": float(arr.mean()),
        "scene_score_max": float(arr.max()),
        "fit_uses_calibration_labels": False,
        "calibration_uses_promotion_labels": False,
        "calibration_tokens_sha256": __import__("hashlib").sha256(("\n".join(row_tokens) + "\n").encode()).hexdigest(),
        "direct_eligible_tokens_sha256": __import__("hashlib").sha256(("\n".join(eligible_tokens) + "\n").encode()).hexdigest(),
        "theorem_scope": "With the mean model, leverage normalization, direct-domain candidate-set rule and alpha frozen before CAL, exchangeability of CAL and future direct-eligible scenes gives marginal simultaneous coverage over that direct intervention population: with probability at least 1-alpha, every admissible support-positive candidate in a future direct-eligible scene satisfies Delta >= mu-q*scale. Therefore any extremal selection made after observing those lower bounds, restricted to LCB>0, is non-harmful except on the scene-level simultaneous miscoverage event. This is not conditional per-scene, distribution-shift, or closed-loop absolute safety.",
    }
    Path(args.output_report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"pass": True, "scene_simultaneous_quantile": q, "direct_eligible_calibration_scenes": n, "output_main_config": args.output_main_config}, sort_keys=True))


if __name__ == "__main__":
    main()
