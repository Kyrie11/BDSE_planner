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
from bdse.tools.fit_v64_3_38_eaf_icer_davr import MIN_VALUE_CAL_PROPOSALS
from bdse.tools.fit_v64_3_39_eaf_icer_cfsr import _fit_translation

EXPECTED_SCENES = 500


def _tokens(path: Path) -> list[str]:
    return [str(json.loads(x).get("scenario_token", "")) for x in path.read_text().splitlines() if x.strip()]


def _parts(cfg: dict[str, Any], require_cfsr: bool):
    sc = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    if not bool(sc.get("enabled")) or bool(sc.get("scene_reservation_enabled", False)):
        raise SystemExit("V39 CAL requires frozen RSMR value config")
    if list(sc.get("feature_names", [])) != FEATURE_NAMES:
        raise SystemExit("V39 CAL frozen RSMR feature schema mismatch")
    rmean = np.asarray(sc.get("feature_mean", []), dtype=np.float64)
    rw = np.asarray(sc.get("weights", []), dtype=np.float64)
    rstd = np.asarray(sc.get("feature_std", []), dtype=np.float64)
    if any(v.size != len(FEATURE_NAMES) for v in [rmean, rw, rstd]) or np.max(np.abs(rmean)) > 1.0e-12 or abs(float(sc.get("bias", 0.0))) > 1.0e-12:
        raise SystemExit("V39 CAL frozen RSMR parameters invalid")
    dmean = np.asarray(sc.get("post_selection_dense_feature_mean", []), dtype=np.float64)
    dstd = np.asarray(sc.get("post_selection_dense_feature_std", []), dtype=np.float64)
    dw = np.asarray(sc.get("post_selection_dense_weights", []), dtype=np.float64)
    db = float(sc.get("post_selection_dense_bias", float("nan")))
    if any(v.size != len(FEATURE_NAMES) for v in [dmean, dstd, dw]) or not math.isfinite(db):
        raise SystemExit("V39 CAL dense parameters invalid")
    cfsr = None
    if require_cfsr:
        if str(sc.get("post_selection_value_mode", "")) != "dense_edge_cfsr":
            raise SystemExit("V39 CAL CFSR input must be raw dense_edge_cfsr")
        cmean = np.asarray(sc.get("post_selection_cfsr_feature_mean", []), dtype=np.float64)
        cstd = np.asarray(sc.get("post_selection_cfsr_feature_std", []), dtype=np.float64)
        cw = np.asarray(sc.get("post_selection_cfsr_weights", []), dtype=np.float64)
        cb = float(sc.get("post_selection_cfsr_bias", float("nan")))
        if any(v.size != len(FEATURE_NAMES) for v in [cmean, cstd, cw]) or not math.isfinite(cb):
            raise SystemExit("V39 CAL CFSR residual parameters invalid")
        cfsr = (cmean, cstd, cw, cb)
    return sc, (rw, rstd, {"source": "frozen_full_TRAIN_RSMR"}), (dmean, dstd, dw, db), cfsr


def _dense(x: np.ndarray, m) -> float:
    mean, std, w, b = m
    z = (np.asarray(x, dtype=np.float64) - mean) / np.maximum(std, 1.0e-6)
    return float(np.clip(z @ w + b, -40.0, 40.0))


def _cfsr(x: np.ndarray, dense, residual) -> float:
    dv = _dense(x, dense)
    mean, std, w, b = residual
    z = (np.asarray(x, dtype=np.float64) - mean) / np.maximum(std, 1.0e-6)
    return float(np.clip(dv + z @ w + b, -40.0, 40.0))


def _decorate_shift(cfg: dict[str, Any], bias: float) -> dict[str, Any]:
    out = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False))
    sc = out["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    sc.update({
        "post_selection_value_enabled": True,
        "post_selection_value_mode": "dense_edge_shift",
        "post_selection_selected_bias": float(bias),
        "post_selection_value_training": "dense_all_TRAIN_candidate_supervision_plus_independent_CAL500_unit_slope_translation",
        "post_selection_operator": "freeze_RSMR_winner_then_dense_value_plus_translation_then_accept_same_winner_iff_positive_else_incumbent",
    })
    out.setdefault("metadata", {})["algorithm_version"] = "V64.3.39-EAF-ICER-DENSE-SHIFT-CONTROL"
    out.setdefault("provenance", {})["algorithm_version"] = "V64.3.39-EAF-ICER-DENSE-SHIFT-CONTROL"
    out.setdefault("experiment", {})["name"] = "v64_3_39_dense_shift_control"
    out["experiment"]["algorithm"] = "V64.3.39 dense value + selected-policy translation-only control"
    return out


def _decorate_main(cfg: dict[str, Any], bias: float) -> dict[str, Any]:
    out = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False))
    sc = out["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    sc.update({
        "post_selection_value_enabled": True,
        "post_selection_value_mode": "dense_edge_cfsr_shift",
        "post_selection_selected_bias": float(bias),
        "post_selection_value_training": "dense_all_TRAIN_candidate_supervision_plus_cross_fitted_TRAIN_selected_policy_residual_plus_independent_CAL500_translation",
        "post_selection_operator": "freeze_RSMR_winner_then_dense_value_plus_cross_fitted_selection_residual_plus_translation_then_accept_same_winner_iff_positive_else_incumbent",
    })
    out.setdefault("metadata", {})["algorithm_version"] = "V64.3.39-EAF-ICER-CFSR"
    out.setdefault("provenance", {})["algorithm_version"] = "V64.3.39-EAF-ICER-CFSR"
    out.setdefault("experiment", {})["name"] = "v64_3_39_eaf_icer_cfsr"
    out["experiment"]["algorithm"] = "V64.3.39 cross-fitted selection residual value recovery"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit V64.3.39 translation-only selected-policy calibration on independent CAL500")
    ap.add_argument("--calibration-rows", required=True); ap.add_argument("--calibration-edges", required=True)
    ap.add_argument("--dense-config", required=True); ap.add_argument("--cfsr-config", required=True)
    ap.add_argument("--output-dense-shift-config", required=True); ap.add_argument("--output-cfsr-main-config", required=True); ap.add_argument("--output-report", required=True)
    a = ap.parse_args()
    rt = _tokens(Path(a.calibration_rows))
    if len(rt) != EXPECTED_SCENES or len(set(rt)) != EXPECTED_SCENES:
        raise SystemExit("V39 CAL rows must contain exactly 500 unique scenes")
    groups: dict[str, list[dict[str, Any]]] = {t: [] for t in rt}; allowed = set(rt)
    for line in Path(a.calibration_edges).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line); t = str(r.get("scenario_token", ""))
        if t not in allowed: raise SystemExit(f"V39 CAL edge token outside rows: {t}")
        groups[t].append(r)
    dcfg = yaml.safe_load(Path(a.dense_config).read_text()); dsc, rsm, dense, _ = _parts(dcfg, False)
    ccfg = yaml.safe_load(Path(a.cfsr_config).read_text()); csc, rsm2, dense2, residual = _parts(ccfg, True)
    if np.max(np.abs(rsm[0]-rsm2[0])) > 1e-12 or np.max(np.abs(rsm[1]-rsm2[1])) > 1e-12:
        raise SystemExit("V39 CAL dense/CFSR frozen RSMR mismatch")
    if any(np.max(np.abs(a-b)) > 1e-12 for a,b in zip(dense[:3], dense2[:3])) or abs(float(dense[3])-float(dense2[3])) > 1e-12:
        raise SystemExit("V39 CAL dense/CFSR base value mismatch")
    ys: list[float] = []; dvs: list[float] = []; cvs: list[float] = []; used: list[str] = []; replay_max = 0.0
    for t in rt:
        ss = _scene_samples(groups[t])
        if not ss: continue
        score = _structured_scores(ss, rsm); idx = _select(ss, score)
        if idx is None: continue
        x = np.asarray(ss[idx]["x"], dtype=np.float64); dv = _dense(x, dense); cv = _cfsr(x, dense, residual)
        rr = next((r for r in groups[t] if int(r.get("challenger_action", -2)) == int(ss[idx]["action"])), None)
        if rr is None: raise SystemExit("V39 CAL selected action missing from edge replay")
        try: logged = float(rr.get("icer_scir_raw_predicted_improvement", rr.get("icer_scir_predicted_improvement", float("nan"))))
        except (TypeError, ValueError): logged = float("nan")
        if math.isfinite(logged): replay_max = max(replay_max, abs(logged - float(score[idx])))
        ys.append(float(ss[idx]["y"])); dvs.append(dv); cvs.append(cv); used.append(t)
    if len(used) < MIN_VALUE_CAL_PROPOSALS:
        raise SystemExit(f"V39 CAL500 frozen RSMR produced too few proposals: {len(used)} < {MIN_VALUE_CAL_PROPOSALS}")
    if replay_max > 1.0e-5: raise SystemExit(f"V39 CAL frozen RSMR replay mismatch max_abs={replay_max}")
    y=np.asarray(ys); dv=np.asarray(dvs); cv=np.asarray(cvs)
    ds=_fit_translation(dv,y,"dense_all_edge_absolute_value_prediction")
    cs=_fit_translation(cv,y,"cross_fitted_selection_residual_corrected_value")
    dp=dv+float(ds["selected_policy_bias"]); cp=cv+float(cs["selected_policy_bias"])
    Path(a.output_dense_shift_config).write_text(yaml.safe_dump(_decorate_shift(dcfg,ds["selected_policy_bias"]),sort_keys=False))
    Path(a.output_cfsr_main_config).write_text(yaml.safe_dump(_decorate_main(ccfg,cs["selected_policy_bias"]),sort_keys=False))
    report={
        "audit":"v64_3_39_eaf_icer_cfsr_independent_CAL500_translation_fit",
        "calibration_total_scene_count":EXPECTED_SCENES,
        "selected_policy_proposal_count":len(used), "selected_policy_proposal_count_min":MIN_VALUE_CAL_PROPOSALS,
        "calibration_tokens_sha256":hashlib.sha256(("\n".join(rt)+"\n").encode()).hexdigest(),
        "selected_policy_tokens_sha256":hashlib.sha256(("\n".join(used)+"\n").encode()).hexdigest(),
        "frozen_rsmr_score_replay_max_abs":float(replay_max),
        "proposal_teacher_improvement_sum":float(y.sum()), "proposal_precision":float(np.mean(y>0.0)), "proposal_worst":float(y.min()),
        "dense_translation_fit":ds, "cfsr_translation_fit":cs,
        "dense_translation_sign_accuracy":float(np.mean((dp>0.0)==(y>0.0))),
        "cfsr_translation_sign_accuracy":float(np.mean((cp>0.0)==(y>0.0))),
        "causal_contract":"RSMR proposal identity is frozen before all value readouts. DENSE learns all-edge cardinal value; CFSR adds only a cross-fitted TRAIN selected-policy residual orthogonal to rank/dense directions. CAL500 estimates translation only, preserving learned value ordering. Every arm can only accept the exact RSMR winner or return incumbent.",
    }
    Path(a.output_report).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({"pass":True,"selected_policy_proposals":len(used),"dense_shift_config":a.output_dense_shift_config,"cfsr_main_config":a.output_cfsr_main_config},sort_keys=True))


if __name__ == "__main__":
    main()
