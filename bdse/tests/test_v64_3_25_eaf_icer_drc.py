from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import bdse.tools.fit_v64_3_25_eaf_icer_drc as drc
from bdse.tools.check_v64_3_25_eaf_icer_drc_split import _replacement_tail_diag


def test_downside_certificate_rejects_positive_mean_with_catastrophic_tail():
    # Mean is positive, but the direct downside RMS is larger.  This is the V23
    # failure mode that DRC is designed to reject without a validation threshold.
    train_x = np.zeros((64, 18), dtype=float)
    train_y = np.r_[np.full(63, 0.05), -1.0]
    query = np.zeros((1, 18), dtype=float)
    meanse = drc._score(train_x, train_y, query, "mean_se")[0]
    downside = drc._score(train_x, train_y, query, "downside_rms")[0]
    assert meanse > downside
    assert downside < 0.0


def test_streaming_loader_retains_only_drc_contract_fields(tmp_path: Path):
    p = tmp_path / "edges.jsonl"
    row = {
        "scenario_token": "abc", "anchor_action": 0, "raw_top_action": 1,
        "challenger_action": 2, "icer_admissible": 1.0, "teacher_margin": 0.1,
        "icer_support_logit": 1.0, "icer_scalar_dominance_logit": 2.0,
        "icer_attribution_resolved_candidate_atom_signed_spectrum_00": 999.0,
    }
    row.update({f"icer_feature_{n}": float(i) for i, n in enumerate(drc._BASE_NAMES)})
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    by, n = drc._load_minimal_scenes(p)
    assert n == 1
    kept = by["abc"][0]
    assert len([k for k in kept if k.startswith("icer_feature_")]) == 18
    assert not any("attribution_resolved" in k for k in kept)


def test_v25_config_is_aggregate_scalar_only_incumbent_default(tmp_path: Path):
    base = yaml.safe_load(Path("bdse/configs/v64_3_20_icer_dc_dual.yaml").read_text(encoding="utf-8"))
    mem = {"path": str(tmp_path / "m.npz"), "sha256": "0" * 64}
    cfg = drc._cfg(base, mem, "downside_rms", "aggregate_downside")
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    assert ic["regret_risk_feature_mode"] == "evidence_only"
    assert ic["dominance_policy"] == "scalar_only"
    assert ic["incumbent_retention_policy"] == "preserve_admissible_incumbent"
    assert ic["replacement_regret_risk_enabled"] is True
    assert ic["retention_regret_risk_enabled"] is False
    assert ic["replacement_local_regret_neighbor_k_values"] == [32, 64]
    assert "attribution_resolved" not in ic["model_type"]


def test_failed_train_gate_still_writes_audit_and_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    edge = tmp_path / "edge.jsonl"
    edge.write_text("{}\n", encoding="utf-8")
    report = tmp_path / "fit.json"
    tokens = tmp_path / "tokens.txt"
    out = tmp_path / "cfg"

    by = {"scene": [{}]}
    fake_data = {
        "delta": np.asarray([0.1]),
        "replacement_scene_count": 1,
    }
    monkeypatch.setattr(drc, "EXPECTED_TRAIN_SCENES", 1)
    monkeypatch.setattr(drc, "_load_minimal_scenes", lambda _: (by, 1))
    monkeypatch.setattr(drc, "_build", lambda _: fake_data)

    def fake_cf(_data, cert):
        passed = cert == "mean_se"
        return {
            "mode": "aggregate_evidence_only", "certificate": cert, "folds": [],
            "all_folds_path_safe": passed, "fold_pass_count": 5 if passed else 4,
            "selected_count": 100, "teacher_improvement_sum": 1.0,
            "mean_precision": 0.8, "mean_capture": 0.2,
        }

    monkeypatch.setattr(drc, "_crossfit", fake_cf)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fit", "--train-frontier-edges", str(edge),
            "--base-v20-dual-config", "unused.yaml", "--output-dir", str(out),
            "--output-train-token-file", str(tokens), "--output-report", str(report),
        ],
    )
    with pytest.raises(SystemExit, match="STOP TRAIN DRC"):
        drc.main()
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["train_gate_pass"] is False
    assert saved["crossfit"]["aggregate_downside"]["fold_pass_count"] == 4
    assert tokens.read_text(encoding="utf-8").strip() == "scene"
    assert saved["configs"] == {}
    assert saved["memories"] == {}


def test_replacement_tail_diag_reports_actual_regret_and_frontier_downside(tmp_path: Path):
    edge = tmp_path / "edges.jsonl"
    rows = [
        {"scenario_token": "s", "anchor_action": 0, "raw_top_action": 1, "challenger_action": 1,
         "icer_selected_action": 2, "icer_admissible": 1.0, "teacher_margin": 0.2},
        {"scenario_token": "s", "anchor_action": 0, "raw_top_action": 1, "challenger_action": 2,
         "icer_selected_action": 2, "icer_admissible": 1.0, "teacher_margin": -0.8},
    ]
    edge.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    raw = {"s": {"teacher_regret": 10.0}}
    alg = {"s": {"teacher_regret": 20.0}}
    d = _replacement_tail_diag(raw, alg, str(edge), {"s"})
    assert d["count"] == 1
    assert d["regret_delta_sum"] == 10.0
    assert d["regret_positive_rms"] == 10.0
    assert np.isclose(d["teacher_improvement_sum"], -1.0)
    assert np.isclose(d["teacher_negative_rms"], 1.0)
