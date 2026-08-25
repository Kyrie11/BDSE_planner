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
from bdse.tools.fit_v64_3_38_eaf_icer_davr import MIN_VALUE_CAL_PROPOSALS, _affine_scalar, _fit_affine_scalar

EXPECTED_SCENES = 500


def _tokens(path: Path) -> list[str]:
    return [str(json.loads(x).get("scenario_token", "")) for x in path.read_text().splitlines() if x.strip()]


def _cfg_parts(cfg: dict[str, Any]):
    sc = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    if not bool(sc.get("enabled")) or bool(sc.get("scene_reservation_enabled", False)):
        raise SystemExit("V38 CAL requires frozen RSMR + dense value config")
    if str(sc.get("post_selection_value_mode", "")) != "dense_edge_value" or not bool(sc.get("post_selection_value_enabled", False)):
        raise SystemExit("V38 CAL requires uncalibrated dense_edge_value input config")
    if list(sc.get("feature_names", [])) != FEATURE_NAMES:
        raise SystemExit("V38 CAL frozen RSMR feature schema mismatch")
    rmean = np.asarray(sc.get("feature_mean", []), dtype=np.float64); rw = np.asarray(sc.get("weights", []), dtype=np.float64); rstd = np.asarray(sc.get("feature_std", []), dtype=np.float64)
    if any(v.size != len(FEATURE_NAMES) for v in [rmean, rw, rstd]) or np.max(np.abs(rmean)) > 1.0e-12 or abs(float(sc.get("bias", 0.0))) > 1.0e-12:
        raise SystemExit("V38 CAL frozen RSMR parameters invalid")
    dmean = np.asarray(sc.get("post_selection_dense_feature_mean", []), dtype=np.float64); dstd = np.asarray(sc.get("post_selection_dense_feature_std", []), dtype=np.float64); dw = np.asarray(sc.get("post_selection_dense_weights", []), dtype=np.float64); db = float(sc.get("post_selection_dense_bias", float("nan")))
    if any(v.size != len(FEATURE_NAMES) for v in [dmean, dstd, dw]) or not math.isfinite(db):
        raise SystemExit("V38 CAL dense value parameters invalid")
    return sc, (rw, rstd, {"source": "frozen_full_TRAIN_RSMR"}), (dw, db, dmean, dstd)


def _dense_value(x: np.ndarray, model) -> float:
    w, b, mean, std = model
    z = (np.asarray(x, dtype=np.float64) - mean) / np.maximum(std, 1.0e-6)
    return float(np.clip(z @ w + b, -40.0, 40.0))


def _decorate_avr(cfg: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    out = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False)); sc = out["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    sc.update({
        "post_selection_value_enabled": True, "post_selection_value_mode": "score_affine",
        "post_selection_score_mean": float(model["input_mean"]), "post_selection_score_std": float(model["input_std"]),
        "post_selection_affine_intercept": float(model["intercept"]), "post_selection_affine_score_weight": float(model["input_weight"]),
        "post_selection_value_max_abs": 40.0,
        "post_selection_value_training": "independent_CAL500_frozen_RSMR_selected_policy_outputs_only",
        "post_selection_value_target": "absolute_teacher_improvement_of_frozen_RSMR_proposal",
        "post_selection_operator": "freeze_RSMR_winner_then_accept_same_winner_iff_value_positive_else_incumbent_no_rerank_no_fallback",
    })
    out.setdefault("metadata", {})["algorithm_version"] = "V64.3.38-EAF-ICER-AVR-CONTROL"; out.setdefault("provenance", {})["algorithm_version"] = "V64.3.38-EAF-ICER-AVR-CONTROL"
    out.setdefault("experiment", {})["name"] = "v64_3_38_avr_control"; out["experiment"]["algorithm"] = "V64.3.38 score-affine selected-policy value control"
    return out


def _decorate_davr(cfg: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    out = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False)); sc = out["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    sc.update({
        "post_selection_value_enabled": True, "post_selection_value_mode": "dense_edge_affine",
        "post_selection_dense_cal_mean": float(model["input_mean"]), "post_selection_dense_cal_std": float(model["input_std"]),
        "post_selection_dense_cal_intercept": float(model["intercept"]), "post_selection_dense_cal_weight": float(model["input_weight"]),
        "post_selection_value_max_abs": 40.0,
        "post_selection_value_training": "dense_all_TRAIN_candidate_supervision_then_independent_CAL500_one_dimensional_selected_policy_recalibration",
        "post_selection_value_target": "absolute_teacher_improvement_of_frozen_RSMR_proposal",
        "post_selection_operator": "freeze_RSMR_winner_then_dense_cardinal_value_then_selected_policy_affine_then_accept_same_winner_iff_positive_else_incumbent",
    })
    out.setdefault("metadata", {})["algorithm_version"] = "V64.3.38-EAF-ICER-DAVR"; out.setdefault("provenance", {})["algorithm_version"] = "V64.3.38-EAF-ICER-DAVR"
    out.setdefault("experiment", {})["name"] = "v64_3_38_eaf_icer_davr"; out["experiment"]["algorithm"] = "V64.3.38 decoupled dense all-edge absolute value + selected-policy scalar recalibration"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.38 selected-policy scalar calibrators on independent CAL500")
    ap.add_argument("--calibration-rows", required=True); ap.add_argument("--calibration-edges", required=True); ap.add_argument("--dense-config", required=True)
    ap.add_argument("--output-affine-config", required=True); ap.add_argument("--output-davr-config", required=True); ap.add_argument("--output-report", required=True)
    a = ap.parse_args()
    rt = _tokens(Path(a.calibration_rows))
    if len(rt) != EXPECTED_SCENES or len(set(rt)) != EXPECTED_SCENES:
        raise SystemExit("V38 CAL rows must contain exactly 500 unique scenes")
    groups: dict[str, list[dict[str, Any]]] = {t: [] for t in rt}; allowed = set(rt)
    for line in Path(a.calibration_edges).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line); t = str(r.get("scenario_token", ""))
        if t not in allowed: raise SystemExit(f"V38 CAL edge token outside rows: {t}")
        groups[t].append(r)
    cfg = yaml.safe_load(Path(a.dense_config).read_text()); sc, rsm, dense = _cfg_parts(cfg)
    us: list[float] = []; dvs: list[float] = []; ys: list[float] = []; used: list[str] = []; replay_max = 0.0
    for t in rt:
        ss = _scene_samples(groups[t])
        if not ss: continue
        score = _structured_scores(ss, rsm); idx = _select(ss, score)
        if idx is None: continue
        u = float(score[idx]); dv = _dense_value(np.asarray(ss[idx]["x"], dtype=np.float64), dense)
        rr = next((r for r in groups[t] if int(r.get("challenger_action", -2)) == int(ss[idx]["action"])), None)
        if rr is None: raise SystemExit("V38 CAL selected action missing from edge replay")
        try: logged = float(rr.get("icer_scir_raw_predicted_improvement", rr.get("icer_scir_predicted_improvement", float("nan"))))
        except (TypeError, ValueError): logged = float("nan")
        if math.isfinite(logged): replay_max = max(replay_max, abs(logged - u))
        us.append(u); dvs.append(dv); ys.append(float(ss[idx]["y"])); used.append(t)
    if len(used) < MIN_VALUE_CAL_PROPOSALS:
        raise SystemExit(f"V38 CAL500 frozen RSMR produced too few selected-policy proposals: {len(used)} < {MIN_VALUE_CAL_PROPOSALS}")
    if replay_max > 1.0e-5: raise SystemExit(f"V38 CAL frozen RSMR score replay mismatch max_abs={replay_max}")
    u=np.asarray(us); dv=np.asarray(dvs); y=np.asarray(ys)
    avr=_fit_affine_scalar(u,y,"frozen_RSMR_scalar_score"); davr=_fit_affine_scalar(dv,y,"dense_all_edge_absolute_value_prediction")
    av=np.asarray([_affine_scalar(x,avr) for x in u]); cv=np.asarray([_affine_scalar(x,davr) for x in dv])
    Path(a.output_affine_config).write_text(yaml.safe_dump(_decorate_avr(cfg,avr),sort_keys=False)); Path(a.output_davr_config).write_text(yaml.safe_dump(_decorate_davr(cfg,davr),sort_keys=False))
    report={
        "audit":"v64_3_38_eaf_icer_davr_independent_CAL500_value_fit", "calibration_total_scene_count":EXPECTED_SCENES,
        "selected_policy_proposal_count":len(used), "selected_policy_proposal_count_min":MIN_VALUE_CAL_PROPOSALS,
        "calibration_tokens_sha256":hashlib.sha256(("\n".join(rt)+"\n").encode()).hexdigest(), "selected_policy_tokens_sha256":hashlib.sha256(("\n".join(used)+"\n").encode()).hexdigest(),
        "frozen_rsmr_score_replay_max_abs":float(replay_max), "proposal_teacher_improvement_sum":float(y.sum()), "proposal_precision":float(np.mean(y>0.0)), "proposal_worst":float(y.min()),
        "score_affine_control_fit":avr, "dense_selected_policy_affine_fit":davr,
        "score_affine_sign_accuracy":float(np.mean((av>0.0)==(y>0.0))), "dense_calibrated_sign_accuracy":float(np.mean((cv>0.0)==(y>0.0))),
        "causal_contract":"RSMR proposal identity is frozen. Dense value was trained on all TRAIN candidate edges with the corrected scene-equal objective and never participates in ranking. CAL500 fits only one scalar affine map from dense value to selected-proposal teacher improvement. DAVR can only accept that exact RSMR winner or return incumbent.",
    }
    Path(a.output_report).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({"pass":True,"selected_policy_proposals":len(used),"avr_config":a.output_affine_config,"davr_config":a.output_davr_config},sort_keys=True))

if __name__=="__main__": main()
