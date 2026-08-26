from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from bdse.tools.fit_v64_3_33_eaf_icer_spcr import FEATURE_NAMES, _select
from bdse.tools.fit_v64_3_34_eaf_icer_rsmr import _structured_scores
from bdse.tools.fit_v64_3_38_eaf_icer_davr import MIN_VALUE_CAL_PROPOSALS
from bdse.tools.fit_v64_3_39_eaf_icer_cfsr import _fit_translation
from bdse.tools.fit_v64_3_44_eaf_icer_pcor import (
    PC_OCC_ROBUST,
    _pc_value,
    _scene_v44,
)
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import _pred as _epv_pred

EXPECTED_SCENES = 500


def _tokens(path: str | Path) -> list[str]:
    return [str(json.loads(x).get("scenario_token", "")) for x in Path(path).read_text().splitlines() if x.strip()]


def _array_model(sc: dict) -> tuple[dict, dict, dict, dict]:
    epv = {
        "mode": "epv",
        "names": list(sc["post_selection_endpoint_feature_names"]),
        "scale": np.asarray(sc["post_selection_endpoint_feature_scale"], dtype=np.float64),
        "weights": np.asarray(sc["post_selection_endpoint_weights"], dtype=np.float64),
        "bias": 0.0,
    }
    q = {
        "names": list(sc["post_selection_quality_observable_names"]),
        "scale": np.asarray(sc["post_selection_quality_observable_scale"], dtype=np.float64),
        "weights": np.asarray(sc["post_selection_quality_observable_weights"], dtype=np.float64),
        "bias": 0.0,
    }
    residual = {
        "names": [str(sc["post_selection_future_response_observable_name"])],
        "scale": np.asarray([float(sc["post_selection_future_response_scale"])], dtype=np.float64),
        "weights": np.asarray([float(sc["post_selection_future_response_weight"])], dtype=np.float64),
        "bias": 0.0,
    }
    bm = sc.get("plan_conditioned_response_posterior", {}) or {}
    behavior = {
        **bm,
        "feature_scale": np.asarray(bm.get("feature_scale", []), dtype=np.float64),
        "weights": np.asarray(bm.get("weights", []), dtype=np.float64),
        "bias": np.asarray(bm.get("bias", []), dtype=np.float64),
    }
    return epv, q, residual, behavior


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-rows", required=True)
    ap.add_argument("--calibration-edges", required=True)
    ap.add_argument("--pc-occupancy-robust-config", required=True)
    ap.add_argument("--output-main-config", required=True)
    ap.add_argument("--output-report", required=True)
    args = ap.parse_args()

    rt = _tokens(args.calibration_rows)
    if len(rt) != EXPECTED_SCENES or len(set(rt)) != EXPECTED_SCENES:
        raise SystemExit("V44 CAL rows must be 500 unique scenes")
    allowed = set(rt)
    groups = {t: [] for t in rt}
    for line in Path(args.calibration_edges).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tok = str(r.get("scenario_token", ""))
        if tok not in allowed:
            raise SystemExit("V44 CAL edge outside row set")
        groups[tok].append(r)
    scene = _scene_v44(groups)

    cfg = yaml.safe_load(Path(args.pc_occupancy_robust_config).read_text())
    sc = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    if str(sc.get("post_selection_value_mode", "")) != "endpoint_potential_quality_plan_conditioned_response":
        raise SystemExit("V44 CAL requires raw plan-conditioned response config")
    if str(sc.get("post_selection_future_response_observable_name", "")) != PC_OCC_ROBUST:
        raise SystemExit("V44 CAL requires plan-conditioned occupancy robust arm")

    rmean = np.asarray(sc["feature_mean"], dtype=np.float64)
    rstd = np.asarray(sc["feature_std"], dtype=np.float64)
    rw = np.asarray(sc["weights"], dtype=np.float64)
    if (
        list(sc.get("feature_names", [])) != FEATURE_NAMES
        or rmean.size != len(FEATURE_NAMES)
        or rstd.size != len(FEATURE_NAMES)
        or rw.size != len(FEATURE_NAMES)
        or np.max(np.abs(rmean)) > 1e-12
        or abs(float(sc.get("bias", 0.0))) > 1e-12
    ):
        raise SystemExit("V44 CAL frozen RSMR schema invalid")
    rsm = (rw, rstd, {"source": "frozen_full_TRAIN_RSMR"})
    epv, q, residual, behavior = _array_model(sc)

    ys: list[float] = []
    pred: list[float] = []
    used: list[str] = []
    for tok in rt:
        ss = scene.get(tok, [])
        if not ss:
            continue
        idx = _select(ss, _structured_scores(ss, rsm))
        if idx is None:
            continue
        row = ss[idx]
        v = _pc_value(row, epv, q, residual, behavior, "occupancy_robust", cfg)
        ys.append(float(row["y"]))
        pred.append(float(v))
        used.append(tok)
    if len(used) < MIN_VALUE_CAL_PROPOSALS:
        raise SystemExit(f"V44 CAL proposals {len(used)} < {MIN_VALUE_CAL_PROPOSALS}")

    fit = _fit_translation(np.asarray(pred), np.asarray(ys), "quality_plus_plan_conditioned_occupancy_robust")
    sc["post_selection_value_mode"] = "endpoint_potential_quality_plan_conditioned_response_shift"
    sc["post_selection_selected_bias"] = float(fit["selected_policy_bias"])
    sc["post_selection_value_training"] = (
        "TRAIN_behavior_posterior_plus_dense_all_TRAIN_quality_plan_conditioned_occupancy_robust_"
        "plus_independent_CAL500_unit_slope_translation"
    )
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.44-EAF-ICER-PCOR"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.44-EAF-ICER-PCOR"
    cfg.setdefault("experiment", {})["name"] = "v64_3_44_eaf_icer_pcor"
    cfg["experiment"]["algorithm"] = "V64.3.44 plan-conditioned occupancy response"
    Path(args.output_main_config).write_text(yaml.safe_dump(cfg, sort_keys=False))

    rep = {
        "audit": "v64_3_44_pcor_independent_CAL500_translation",
        "selected_policy_proposal_count": len(used),
        "selected_policy_proposal_count_min": MIN_VALUE_CAL_PROPOSALS,
        "calibration_tokens_sha256": hashlib.sha256(("\n".join(rt) + "\n").encode()).hexdigest(),
        "selected_policy_tokens_sha256": hashlib.sha256(("\n".join(used) + "\n").encode()).hexdigest(),
        "translation_fit": fit,
        "causal_contract": (
            "RSMR winner, TRAIN-fitted QUALITY, TRAIN-only plan-conditioned behavior posterior, and "
            "TRAIN-fitted occupancy residual are frozen; CAL500 learns a unit-slope translation only."
        ),
    }
    Path(args.output_report).write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": True, "selected_policy_proposals": len(used), "output": args.output_main_config}, sort_keys=True))


if __name__ == "__main__":
    main()
