from __future__ import annotations

import copy

import numpy as np
import pytest

from bdse.tests.test_v64_3_31_eaf_icer_scir import _run, _scir_cfg


def _ssir_cfg(*, q: float = 1.0, leverage_diag: float = 400.0) -> dict:
    c = _scir_cfg(mode="simultaneous_lcb")
    scir = c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    n = len(scir["feature_names"])
    lev = np.zeros((n, n), dtype=float)
    raw_idx = scir["feature_names"].index("delta::raw_margin")
    lev[raw_idx, raw_idx] = leverage_diag
    scir["leverage_inverse"] = lev.tolist()
    scir["selection_scale_floor"] = 1.0
    scir["simultaneous_conformal_quantile"] = q
    return c


def test_ssir_candidate_specific_bound_can_reorder_mean_winner_before_selection():
    mean_action, mean = _run(_scir_cfg(mode="mean_rank"))
    ssir_action, ssir = _run(_ssir_cfg(q=1.0, leverage_diag=400.0))
    assert mean_action == 1
    # Candidate 1 has larger mean but larger leverage penalty; candidate 3 has
    # the larger calibrated lower bound and is therefore the SSIR winner.
    assert ssir_action == 3
    mu = np.asarray(ssir["_decisive_frontier_icer_scir_predicted_improvement_star"], dtype=float)
    scale = np.asarray(ssir["_decisive_frontier_icer_scir_selection_scale_star"], dtype=float)
    lcb = np.asarray(ssir["_decisive_frontier_icer_scir_lower_bound_star"], dtype=float)
    assert mu[1] > mu[3]
    assert scale[1] > scale[3]
    assert lcb[3] > lcb[1] > 0.0
    assert int(ssir["decisive_frontier_icer_scir_proposal_action"]) == 3
    assert float(ssir["decisive_frontier_icer_scir_certificate_accepted"]) == 1.0


def test_ssir_no_positive_lower_bound_returns_incumbent_without_fallback():
    action, d = _run(_ssir_cfg(q=100.0, leverage_diag=0.0))
    assert action == 2
    assert float(d["decisive_frontier_icer_scir_proposal_exists"]) == 0.0
    assert int(d["decisive_frontier_icer_selected_action"]) == 2
    assert float(d["decisive_frontier_icer_scir_certificate_accepted"]) == 0.0


def test_ssir_all_flagged_structural_domain_is_exact_incumbent_delegation():
    action, d = _run(_ssir_cfg(), safety=np.ones(4, dtype=bool))
    assert action == 2
    assert float(d["decisive_frontier_icer_structural_domain_delegated"]) == 1.0
    assert float(d["decisive_frontier_icer_scir_proposal_exists"]) == 0.0


def test_ssir_malformed_leverage_matrix_fails_closed():
    c = _ssir_cfg()
    scir = c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    scir["leverage_inverse"] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="leverage_inverse shape"):
        _run(c)


def test_v31_artifact_without_leverage_matrix_remains_exact_mean_rank_behavior():
    c = _scir_cfg(mode="mean_rank")
    a0, d0 = _run(c)
    c2 = copy.deepcopy(c)
    scir = c2["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    scir["leverage_inverse"] = []
    a1, d1 = _run(c2)
    assert a0 == a1 == 1
    assert np.allclose(d0["_decisive_frontier_icer_scir_predicted_improvement_star"], d1["_decisive_frontier_icer_scir_predicted_improvement_star"])
    assert np.allclose(d1["_decisive_frontier_icer_scir_selection_scale_star"], np.ones(4))


def test_ssir_calibrator_uses_one_score_per_direct_eligible_scene(tmp_path):
    import json, subprocess, sys, yaml
    from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES

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
            "enabled": True, "mode": "mean_rank", "feature_names": names,
            "leverage_inverse": np.eye(len(names)).tolist(), "conformal_alpha": 0.05,
        }}}},
        "metadata": {}, "provenance": {}, "experiment": {},
    }
    cfgp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    cp = subprocess.run([
        sys.executable, "-m", "bdse.tools.calibrate_v64_3_32_eaf_icer_ssir",
        "--calibration-rows", str(rows), "--calibration-edges", str(edges),
        "--mean-config", str(cfgp), "--output-main-config", str(outcfg),
        "--output-report", str(report), "--alpha", "0.05",
    ], check=True, text=True, capture_output=True)
    assert '"pass": true' in cp.stdout.lower()
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["calibration_total_scene_count"] == 500
    assert rep["direct_eligible_scene_count"] == 100
    assert rep["scene_simultaneous_quantile"] == pytest.approx(0.5)
    main = yaml.safe_load(outcfg.read_text(encoding="utf-8"))
    scir = main["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]["selection_conditioned_intervention_recovery"]
    assert scir["mode"] == "simultaneous_lcb"
