from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FEATURE_NAMES, _scene_samples, _select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _structured_scores
from bdse.tools.fit_v64_3_37_eaf_icer_pvr import (
    MIN_VALUE_CAL_PROPOSALS,
    _affine_value,
    _fit_affine_value,
    _fit_orthogonal_residual,
    _orthogonal_value,
    _proposal_geometry,
)

EXPECTED_SCENES = 500


def _tokens(path: Path) -> list[str]:
    return [str(json.loads(x).get("scenario_token", "")) for x in path.read_text().splitlines() if x.strip()]


def _rsm_model(cfg: dict[str, Any]):
    sc = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    if not bool(sc.get("enabled")) or bool(sc.get("scene_reservation_enabled", False)) or bool(sc.get("post_selection_value_enabled", False)):
        raise SystemExit("V37 CAL requires frozen RSMR without pre-existing reservation/value head")
    if list(sc.get("feature_names", [])) != FEATURE_NAMES:
        raise SystemExit("V37 CAL frozen RSMR feature schema mismatch")
    mean = np.asarray(sc.get("feature_mean", []), dtype=np.float64)
    if mean.size != len(FEATURE_NAMES) or np.max(np.abs(mean)) > 1.0e-12:
        raise SystemExit("V37 CAL requires V34 zero-preserving RSMR feature mean")
    w = np.asarray(sc.get("weights", []), dtype=np.float64)
    scale = np.asarray(sc.get("feature_std", []), dtype=np.float64)
    if w.size != len(FEATURE_NAMES) or scale.size != len(FEATURE_NAMES) or np.any(~np.isfinite(w)) or np.any(~np.isfinite(scale)):
        raise SystemExit("V37 CAL frozen RSMR parameters invalid")
    if abs(float(sc.get("bias", 0.0))) > 1.0e-12:
        raise SystemExit("V37 CAL frozen RSMR bias must be zero")
    return (w, scale, {"source": "frozen_full_TRAIN_RSMR"})


def _decorate(cfg: dict[str, Any], affine: dict[str, Any], residual: dict[str, Any] | None, mode: str, version: str, expname: str) -> dict[str, Any]:
    out = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False))
    sc = out["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    if bool(sc.get("scene_reservation_enabled", False)) or bool(sc.get("post_selection_value_enabled", False)):
        raise SystemExit("V37 CAL refuses pre-existing post-selection operator")
    sc.update({
        "post_selection_value_enabled": True,
        "post_selection_value_mode": mode,
        "post_selection_score_mean": float(affine["score_mean"]),
        "post_selection_score_std": float(affine["score_std"]),
        "post_selection_affine_intercept": float(affine["intercept"]),
        "post_selection_affine_score_weight": float(affine["score_weight"]),
        "post_selection_value_max_abs": 40.0,
        "post_selection_value_training": "independent_CAL500_frozen_RSMR_selected_policy_outputs_only",
        "post_selection_value_target": "absolute_teacher_improvement_of_frozen_RSMR_proposal",
        "post_selection_operator": "freeze_RSMR_winner_then_accept_same_winner_iff_value_positive_else_incumbent_no_rerank_no_fallback",
    })
    if residual is not None:
        sc.update({
            "post_selection_residual_feature_mean": list(residual["feature_mean"]),
            "post_selection_residual_feature_std": list(residual["feature_std"]),
            "post_selection_residual_weights": list(residual["weights"]),
            "post_selection_residual_bias": float(residual.get("bias", 0.0)),
            "post_selection_residual_feature_definition": "selected_standardized_19D_projected_orthogonal_to_frozen_RSMR_score_direction",
        })
    out.setdefault("metadata", {})["algorithm_version"] = version
    out.setdefault("provenance", {})["algorithm_version"] = version
    out.setdefault("experiment", {})["name"] = expname
    out["experiment"]["algorithm"] = f"V64.3.37 frozen-RSMR post-selection {mode} absolute value recovery"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.37 affine and score-orthogonal post-selection value heads on independent CAL500")
    ap.add_argument("--calibration-rows", required=True); ap.add_argument("--calibration-edges", required=True); ap.add_argument("--rsmr-config", required=True)
    ap.add_argument("--output-affine-config", required=True); ap.add_argument("--output-orthogonal-config", required=True); ap.add_argument("--output-report", required=True)
    a = ap.parse_args()
    rt = _tokens(Path(a.calibration_rows))
    if len(rt) != EXPECTED_SCENES or len(set(rt)) != EXPECTED_SCENES:
        raise SystemExit("V37 CAL rows must contain exactly 500 unique scenes")
    groups: dict[str, list[dict[str, Any]]] = {t: [] for t in rt}; allowed = set(rt)
    for line in Path(a.calibration_edges).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line); t = str(r.get("scenario_token", ""))
        if t not in allowed:
            raise SystemExit(f"V37 CAL edge token outside rows: {t}")
        groups[t].append(r)

    cfg = yaml.safe_load(Path(a.rsmr_config).read_text()); rsm = _rsm_model(cfg)
    us: list[float] = []; perps: list[np.ndarray] = []; ys: list[float] = []; used: list[str] = []
    replay_max = 0.0
    for t in rt:
        ss = _scene_samples(groups[t])
        if not ss:
            continue
        score = _structured_scores(ss, rsm); idx = _select(ss, score)
        if idx is None:
            continue
        u, _, zp = _proposal_geometry(ss, rsm, idx)
        rr = next((r for r in groups[t] if int(r.get("challenger_action", -2)) == int(ss[idx]["action"])), None)
        if rr is None:
            raise SystemExit("V37 CAL selected action missing from edge replay")
        try:
            logged = float(rr.get("icer_scir_raw_predicted_improvement", rr.get("icer_scir_predicted_improvement", float("nan"))))
        except (TypeError, ValueError):
            logged = float("nan")
        if math.isfinite(logged):
            replay_max = max(replay_max, abs(logged - u))
        us.append(u); perps.append(zp); ys.append(float(ss[idx]["y"])); used.append(t)
    if len(used) < MIN_VALUE_CAL_PROPOSALS:
        raise SystemExit(f"V37 CAL500 frozen RSMR produced too few selected-policy proposals: {len(used)} < {MIN_VALUE_CAL_PROPOSALS}")
    if replay_max > 1.0e-5:
        raise SystemExit(f"V37 CAL frozen RSMR score replay mismatch max_abs={replay_max}")

    u = np.asarray(us, dtype=np.float64); zp = np.stack(perps); y = np.asarray(ys, dtype=np.float64)
    affine = _fit_affine_value(u, y)
    av = np.asarray([_affine_value(v, affine) for v in u], dtype=np.float64)
    residual = _fit_orthogonal_residual(zp, y - av)
    ov = np.asarray([_orthogonal_value(v, q, affine, residual) for v, q in zip(u, zp)], dtype=np.float64)
    residual["combined_fit_mse"] = float(np.mean((ov - y) ** 2))

    acfg = _decorate(cfg, affine, None, "score_affine", "V64.3.37-EAF-ICER-AVR", "v64_3_37_affine_value_recovery")
    ocfg = _decorate(cfg, affine, residual, "orthogonal_proposal_value", "V64.3.37-EAF-ICER-OPVR", "v64_3_37_eaf_icer_opvr")
    Path(a.output_affine_config).write_text(yaml.safe_dump(acfg, sort_keys=False)); Path(a.output_orthogonal_config).write_text(yaml.safe_dump(ocfg, sort_keys=False))
    report = {
        "audit": "v64_3_37_eaf_icer_pvr_independent_CAL500_value_fit",
        "calibration_total_scene_count": EXPECTED_SCENES, "selected_policy_proposal_count": len(used),
        "selected_policy_proposal_count_min": MIN_VALUE_CAL_PROPOSALS,
        "calibration_tokens_sha256": hashlib.sha256(("\n".join(rt) + "\n").encode()).hexdigest(),
        "selected_policy_tokens_sha256": hashlib.sha256(("\n".join(used) + "\n").encode()).hexdigest(),
        "frozen_rsmr_score_replay_max_abs": float(replay_max),
        "proposal_teacher_improvement_sum": float(y.sum()), "proposal_precision": float(np.mean(y > 0.0)), "proposal_worst": float(y.min()),
        "affine_score_value_fit": affine, "orthogonal_residual_fit": residual,
        "affine_calibration_sign_accuracy": float(np.mean((av > 0.0) == (y > 0.0))),
        "orthogonal_calibration_sign_accuracy": float(np.mean((ov > 0.0) == (y > 0.0))),
        "causal_contract": "RSMR ranking weights and winner identity are frozen before CAL. Both value arms evaluate only that selected proposal. OPVR residual features are mathematically orthogonal to the frozen RSMR score direction. Either arm can only accept the exact RSMR winner or return incumbent; neither can re-rank or fall through.",
    }
    Path(a.output_report).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"pass": True, "selected_policy_proposals": len(used), "affine_config": a.output_affine_config, "orthogonal_config": a.output_orthogonal_config}, sort_keys=True))


if __name__ == "__main__":
    main()
