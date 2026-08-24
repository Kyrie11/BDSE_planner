from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES
from bdse.tools.fit_v64_3_32_1_eaf_icer_ssir_weightfix import (
    RIDGE_LAMBDA,
    _fit_ridge,
)


def _sample(token: str, x0: float, y: float) -> dict:
    x = np.zeros((19,), dtype=np.float64)
    x[0] = x0
    return {"token": token, "x": x, "y": y}


def test_weightfix_matches_frozen_each_scene_total_loss_weight_one_objective():
    # Scene A has two alternatives; scene B has one.  The frozen design says
    # each scene contributes total loss mass 1, hence edge weights 1/2,1/2,1.
    samples = [
        _sample("A", 0.0, 0.0),
        _sample("A", 2.0, 2.0),
        _sample("B", 10.0, 10.0),
    ]
    w, b, mean, std, ginv = _fit_ridge(samples)

    X = np.stack([s["x"] for s in samples])
    y = np.asarray([s["y"] for s in samples], dtype=np.float64)
    loss_w = np.asarray([0.5, 0.5, 1.0], dtype=np.float64)
    moment_w = loss_w / loss_w.sum()
    exp_mean = np.sum(X * moment_w[:, None], axis=0)
    exp_var = np.sum((X - exp_mean[None, :]) ** 2 * moment_w[:, None], axis=0)
    exp_std = np.maximum(np.sqrt(exp_var), 1.0e-6)
    Z = (X - exp_mean[None, :]) / exp_std[None, :]
    A = np.concatenate([np.ones((len(Z), 1)), Z], axis=1)
    root = np.sqrt(loss_w)[:, None]
    reg = np.eye(A.shape[1]) * RIDGE_LAMBDA
    reg[0, 0] = 0.0
    coef = np.linalg.solve((A * root).T @ (A * root) + reg, (A * root).T @ (y * root[:, 0]))
    G = (Z * root).T @ (Z * root) + np.eye(Z.shape[1]) * RIDGE_LAMBDA

    assert np.allclose(mean, exp_mean)
    assert np.allclose(std, exp_std)
    assert b == pytest.approx(float(coef[0]))
    assert np.allclose(w, coef[1:])
    assert np.allclose(ginv, np.linalg.inv(G))

    # Historical V31/V32 normalized the loss weights to sum 1 but left lambda=1.
    # That is a different ridge objective; the hotfix must not silently regress.
    legacy_w = loss_w / loss_w.sum()
    legacy_root = np.sqrt(legacy_w)[:, None]
    legacy_coef = np.linalg.solve(
        (A * legacy_root).T @ (A * legacy_root) + reg,
        (A * legacy_root).T @ (y * legacy_root[:, 0]),
    )
    assert not np.allclose(coef[1:], legacy_coef[1:])


def test_weightfix_independent_calibrator_keeps_same_scene_simultaneous_contract(tmp_path):
    rows = tmp_path / "rows.jsonl"
    edges = tmp_path / "edges.jsonl"
    cfgp = tmp_path / "mean.yaml"
    outcfg = tmp_path / "main.yaml"
    report = tmp_path / "report.json"
    with rows.open("w", encoding="utf-8") as f:
        for i in range(500):
            f.write(json.dumps({"scenario_token": f"t{i:03d}"}) + "\n")
    with edges.open("w", encoding="utf-8") as f:
        for i in range(100):
            tok = f"t{i:03d}"
            f.write(json.dumps({"scenario_token": tok, "raw_top_action": 0, "challenger_action": 0, "icer_admissible": 1.0, "icer_support_logit": 1.0, "teacher_margin": 0.0, "icer_scir_predicted_improvement": 0.0, "icer_scir_selection_scale": 1.0}) + "\n")
            f.write(json.dumps({"scenario_token": tok, "raw_top_action": 0, "challenger_action": 1, "icer_admissible": 1.0, "icer_support_logit": 1.0, "teacher_margin": 1.0, "icer_scir_predicted_improvement": 1.5, "icer_scir_selection_scale": 1.0}) + "\n")
    names = [f"delta::{n}" for n in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES] + ["delta::support_logit"]
    cfg = {
        "runtime": {"decisive_frontier_value": {"incumbent_contrastive_extremal_recovery": {"selection_conditioned_intervention_recovery": {
            "enabled": True,
            "mode": "mean_rank",
            "feature_names": names,
            "leverage_inverse": np.eye(len(names)).tolist(),
            "conformal_alpha": 0.05,
        }}}},
        "metadata": {},
        "provenance": {},
        "experiment": {},
    }
    cfgp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    cp = subprocess.run([
        sys.executable, "-m", "bdse.tools.calibrate_v64_3_32_1_eaf_icer_ssir_weightfix",
        "--calibration-rows", str(rows),
        "--calibration-edges", str(edges),
        "--mean-config", str(cfgp),
        "--output-main-config", str(outcfg),
        "--output-report", str(report),
        "--alpha", "0.05",
    ], check=True, text=True, capture_output=True)
    assert '"pass": true' in cp.stdout.lower()
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["direct_eligible_scene_count"] == 100
    assert rep["scene_simultaneous_quantile"] == pytest.approx(0.5)
    main = yaml.safe_load(outcfg.read_text(encoding="utf-8"))
    assert main["metadata"]["algorithm_version"] == "V64.3.32.1-EAF-ICER-SSIR-WEIGHTFIX"
