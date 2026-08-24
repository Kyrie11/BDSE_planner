from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES
from bdse.tests.test_v64_3_31_eaf_icer_scir import _run
from bdse.tools.fit_v64_3_33_eaf_icer_spcr import (
    FEATURE_NAMES,
    RIDGE_LAMBDA,
    _fit_pair_gap,
    _pair_rows,
    _pair_scores,
    _teacher_best_index,
)


def _alt(token: str, action: int, x0: float, y: float, support: float = 1.0) -> dict:
    x = np.zeros((len(FEATURE_NAMES),), dtype=np.float64)
    x[0] = x0
    return {
        "token": token,
        "action": action,
        "x": x,
        "y": y,
        "support": support,
        "margin": 0.0,
        "utility_prior": 0,
    }


def test_spcr_teacher_best_explicitly_uses_incumbent_as_no_intervention_item():
    noopp = [_alt("n", 1, 1.0, -3.0), _alt("n", 2, 2.0, -0.1)]
    assert _teacher_best_index(noopp) == -1
    opp = [_alt("p", 1, 1.0, 0.2), _alt("p", 2, 2.0, 1.5)]
    assert _teacher_best_index(opp) == 1


def test_spcr_pair_gap_objective_is_scene_equal_and_teaches_noop_scores_negative():
    scene_map = {
        "noopp": [_alt("noopp", 1, 1.0, -2.0), _alt("noopp", 2, 2.0, -1.0)],
        "opp": [_alt("opp", 1, -1.0, 1.0), _alt("opp", 2, -2.0, 2.0)],
    }
    rows = _pair_rows(scene_map, ["noopp", "opp"])
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r["token"], 0.0)
        by_scene[r["token"]] += float(r["weight"])
        assert float(r["gap"]) >= 0.0
    assert by_scene == pytest.approx({"noopp": 1.0, "opp": 1.0})

    model = _fit_pair_gap(scene_map, ["noopp", "opp"])
    noopp_scores = _pair_scores(scene_map["noopp"], model)
    # Incumbent pseudo-score is exactly zero. A useful structured fit must learn
    # that both challengers in a no-opportunity scene are below the incumbent.
    assert float(noopp_scores.max()) < 0.0
    opp_scores = _pair_scores(scene_map["opp"], model)
    assert int(np.argmax(opp_scores)) == 1
    assert float(opp_scores[1]) > 0.0


def test_spcr_pair_gap_fit_uses_frozen_lambda_without_global_loss_normalization():
    # Two scenes, one pair each. Because total scene mass is 1, the closed-form
    # ridge normal equations use a total loss mass of 2 rather than silently
    # normalizing to 1 (the V31/V32 bug that V32.1 repaired).
    scene_map = {
        "a": [_alt("a", 1, 1.0, 1.0)],
        "b": [_alt("b", 1, 2.0, 2.0)],
    }
    coef, scale = _fit_pair_gap(scene_map, ["a", "b"])
    pairs = _pair_rows(scene_map, ["a", "b"])
    D = np.stack([r["d"] for r in pairs])
    g = np.asarray([r["gap"] for r in pairs], dtype=np.float64)
    loss_w = np.asarray([r["weight"] for r in pairs], dtype=np.float64)
    assert loss_w.sum() == pytest.approx(2.0)
    pm = loss_w / loss_w.sum()
    exp_scale = np.maximum(np.sqrt(np.sum((D * D) * pm[:, None], axis=0)), 1e-6)
    Z = D / exp_scale[None, :]
    root = np.sqrt(loss_w)[:, None]
    exp = np.linalg.solve((Z * root).T @ (Z * root) + np.eye(Z.shape[1]) * RIDGE_LAMBDA, (Z * root).T @ (g * root[:, 0]))
    assert np.allclose(scale, exp_scale)
    assert np.allclose(coef, exp)


def _runtime_cfg(*, mode: str, q: float = 0.0) -> dict:
    # Reuse the V31 tournament fixture, then replace the linear score with a
    # zero-anchored structured SPCR score.  The incumbent feature difference is
    # exactly zero and therefore has exact score zero.
    from bdse.tests.test_v64_3_31_eaf_icer_scir import _scir_cfg
    c = _scir_cfg(mode="rank_only")
    sc = c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    names = [f"delta::{n}" for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES] + ["delta::support_logit"]
    sc.update({
        "mode": mode,
        "feature_names": names,
        "feature_mean": [0.0] * len(names),
        "feature_std": [1.0] * len(names),
        "weights": [0.0] * len(names),
        "bias": 0.0,
        "leverage_inverse": [],
        "conformal_overprediction_quantile": q,
        "no_fallback": True,
    })
    # Fixture raw_margin differences make action 1 preferable with this weight.
    sc["weights"][names.index("delta::raw_margin")] = -10.0
    return c


def test_spcr_runtime_rank_and_main_keep_exact_same_proposal_and_no_fallback():
    rank_action, rank = _run(_runtime_cfg(mode="rank_only"))
    assert float(rank["decisive_frontier_icer_scir_proposal_exists"]) == 1.0
    main_action, main = _run(_runtime_cfg(mode="conformal_veto", q=10.0))
    assert int(main["decisive_frontier_icer_scir_proposal_action"]) == int(rank["decisive_frontier_icer_scir_proposal_action"])
    assert rank_action == int(rank["decisive_frontier_icer_scir_proposal_action"])
    assert float(main["decisive_frontier_icer_scir_certificate_accepted"]) == 0.0
    assert main_action == 2  # exact incumbent in the frozen fixture, no second-best fallback


def test_spcr_selected_policy_calibrator_uses_one_frozen_proposal_per_scene(tmp_path):
    rows = tmp_path / "rows.jsonl"
    edges = tmp_path / "edges.jsonl"
    rank_cfg = tmp_path / "rank.yaml"
    main_cfg = tmp_path / "main.yaml"
    report = tmp_path / "report.json"
    with rows.open("w", encoding="utf-8") as f:
        for i in range(500):
            f.write(json.dumps({"scenario_token": f"t{i:03d}"}) + "\n")
    with edges.open("w", encoding="utf-8") as f:
        for i in range(100):
            tok = f"t{i:03d}"
            # incumbent + two positive-score alternatives: calibrator must emit
            # exactly one residual for the highest-scoring frozen proposal.
            f.write(json.dumps({"scenario_token":tok,"raw_top_action":0,"challenger_action":0,"icer_admissible":1.0,"icer_support_logit":1.0,"teacher_margin":0.0,"icer_scir_predicted_improvement":0.0})+"\n")
            f.write(json.dumps({"scenario_token":tok,"raw_top_action":0,"challenger_action":1,"icer_admissible":1.0,"icer_support_logit":1.0,"teacher_margin":1.0,"icer_scir_predicted_improvement":1.4,"raw_margin":0.1,"dacer_utility_prior":0})+"\n")
            f.write(json.dumps({"scenario_token":tok,"raw_top_action":0,"challenger_action":2,"icer_admissible":1.0,"icer_support_logit":1.0,"teacher_margin":0.2,"icer_scir_predicted_improvement":0.3,"raw_margin":0.1,"dacer_utility_prior":0})+"\n")
    cfg = {
        "runtime":{"decisive_frontier_value":{"incumbent_contrastive_extremal_recovery":{"selection_conditioned_intervention_recovery":{
            "enabled":True,"mode":"rank_only","no_fallback":True,"base_feature_names":list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES),
            "feature_names":FEATURE_NAMES,"feature_mean":[0.0]*len(FEATURE_NAMES),"feature_std":[1.0]*len(FEATURE_NAMES),"weights":[0.0]*len(FEATURE_NAMES),"bias":0.0,
        }}}},"metadata":{},"provenance":{},"experiment":{}
    }
    rank_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    cp = subprocess.run([
        sys.executable,"-m","bdse.tools.calibrate_v64_3_33_eaf_icer_spcr",
        "--calibration-rows",str(rows),"--calibration-edges",str(edges),"--pair-config",str(rank_cfg),
        "--output-main-config",str(main_cfg),"--output-report",str(report),"--alpha","0.05",
    ],check=True,text=True,capture_output=True)
    assert '"pass": true' in cp.stdout.lower()
    rep=json.loads(report.read_text(encoding="utf-8"))
    assert rep["selected_policy_proposal_count"] == 100
    assert rep["selected_policy_conformal_quantile"] == pytest.approx(0.4)
    main=yaml.safe_load(main_cfg.read_text(encoding="utf-8"))
    sc=main["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    assert sc["mode"] == "conformal_veto"
    assert sc["conformal_overprediction_quantile"] == pytest.approx(0.4)


def test_spcr_contract_checker_requires_same_rank_selector_in_main(tmp_path):
    base_ic = {
        "all_flagged_policy": "preserve_legacy_for_structural_guard",
        "incumbent_retention_policy": "preserve_admissible_incumbent",
        "regret_risk_enabled": False,
        "replacement_regret_risk_enabled": False,
        "retention_regret_risk_enabled": False,
    }
    def cfg(sc):
        ic = dict(base_ic)
        ic["selection_conditioned_intervention_recovery"] = sc
        return {"evidence": {"budget": 16}, "runtime": {"decisive_frontier_value": {"incumbent_contrastive_extremal_recovery": ic}}}
    mean_sc={"enabled":True,"mode":"mean_rank"}
    rank_sc={
        "enabled":True,"mode":"rank_only","model_type":"scene_equal_incumbent_augmented_teacher_best_pair_gap_ridge",
        "feature_names":FEATURE_NAMES,"base_feature_names":list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES),
        "feature_mean":[0.0]*len(FEATURE_NAMES),"feature_std":[1.0]*len(FEATURE_NAMES),"weights":[0.0]*len(FEATURE_NAMES),
        "bias":0.0,"ridge_lambda":1.0,"training_target":"gap","training_weighting":"scene","leverage_inverse":[],"no_fallback":True,
    }
    main_sc=dict(rank_sc); main_sc.update({"mode":"conformal_veto","conformal_overprediction_quantile":0.4})
    files={
        "v20": {"evidence":{"budget":16},"runtime":{"decisive_frontier_value":{"incumbent_contrastive_extremal_recovery":{}}}},
        "preserve": cfg({"enabled":False}), "mean":cfg(mean_sc), "rank":cfg(rank_sc), "main":cfg(main_sc),
    }
    paths={}
    for k,v in files.items():
        p=tmp_path/f"{k}.yaml"; p.write_text(yaml.safe_dump(v,sort_keys=False),encoding="utf-8"); paths[k]=p
    cal=tmp_path/"cal.json"; cal.write_text(json.dumps({"calibration_total_scene_count":500,"selected_policy_proposal_count":100,"alpha":0.05,"selected_policy_conformal_quantile":0.4,"calibration_uses_promotion_labels":False}),encoding="utf-8")
    out=tmp_path/"contract.json"
    cp=subprocess.run([
        sys.executable,"-m","bdse.tools.check_v64_3_33_eaf_icer_spcr_contract",
        "--v20-config",str(paths["v20"]),"--preserve-config",str(paths["preserve"]),"--mean-config",str(paths["mean"]),
        "--rank-config",str(paths["rank"]),"--main-config",str(paths["main"]),"--calibration-report",str(cal),"--output",str(out),
    ],check=True,text=True,capture_output=True)
    assert json.loads(out.read_text(encoding="utf-8"))["pass"] is True
