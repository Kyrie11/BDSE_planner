from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from bdse.planner.tournament import _icer_local_regret_lower_bound, _load_icer_local_regret_memory
import bdse.tools.fit_v64_3_31_eaf_icer_omcer as omcer


def test_operator_margin_is_exact_weakest_joint_eligibility_margin():
    data = {
        "X": np.zeros((3, 18), dtype=float),
        "support": np.asarray([2.0, -0.2, 0.4]),
        "scalar": np.asarray([0.1, 3.0, -0.5]),
    }
    out = omcer._with_operator_margin(data)
    assert out["X"].shape == (3, 19)
    assert np.allclose(out["X"][:, -1], [0.1, -0.2, -0.5])


def test_catastrophic_excess_ablation_differs_from_all_negative_downside():
    # Tiny negative outcomes are below zero but not below the already-frozen
    # catastrophic boundary.  Catastrophic-excess therefore does not add an RMS
    # penalty for them; all-negative downside does.
    train_x = np.zeros((64, 19), dtype=float)
    train_y = np.r_[np.full(60, 0.05), np.full(4, -0.10)]
    q = np.zeros((1, 19), dtype=float)
    downside = omcer._score(train_x, train_y, q, "downside_rms")[0]
    cat_excess = omcer._score(train_x, train_y, q, "catastrophic_excess_rms")[0]
    assert cat_excess > downside


def test_runtime_catastrophic_excess_certificate_matches_fitter(tmp_path: Path):
    x = np.zeros((64, 19), dtype=np.float32)
    y = np.r_[np.full(63, 0.05, dtype=np.float32), np.asarray([-1.0], dtype=np.float32)]
    names = [f"f{i}" for i in range(19)]
    mem = tmp_path / "m.npz"
    np.savez_compressed(
        mem,
        memory_metric_z=x,
        teacher_improvement=y,
        feature_mean=np.zeros(19, dtype=np.float32),
        feature_std=np.ones(19, dtype=np.float32),
        feature_names=np.asarray(names, dtype="U32"),
        feature_metric_weight=np.full(19, 1.0 / 19.0, dtype=np.float32),
        neighbor_k_values=np.asarray([32, 64], dtype=np.int32),
        se_multiplier=np.asarray([1.0], dtype=np.float32),
        certificate_kind=np.asarray(["mean_minus_catastrophic_excess_rms"], dtype="U64"),
        downside_multiplier=np.asarray([1.0], dtype=np.float32),
        catastrophic_delta_threshold=np.asarray([-0.5], dtype=np.float32),
    )
    import hashlib
    sha = hashlib.sha256(mem.read_bytes()).hexdigest()
    runtime = _icer_local_regret_lower_bound(np.zeros((1, 19)), names, str(mem), sha)[0]
    fitted = omcer._score(np.zeros((64, 19)), y.astype(float), np.zeros((1, 19)), "catastrophic_excess_rms")[0]
    assert np.isclose(runtime, fitted, atol=1e-7)


def test_runtime_loader_rejects_nonnegative_catastrophic_boundary(tmp_path: Path):
    mem = tmp_path / "bad.npz"
    np.savez_compressed(
        mem,
        memory_metric_z=np.zeros((64, 1), dtype=np.float32),
        teacher_improvement=np.zeros(64, dtype=np.float32),
        feature_mean=np.zeros(1, dtype=np.float32),
        feature_std=np.ones(1, dtype=np.float32),
        feature_names=np.asarray(["x"], dtype="U4"),
        feature_metric_weight=np.ones(1, dtype=np.float32),
        neighbor_k_values=np.asarray([32, 64], dtype=np.int32),
        se_multiplier=np.asarray([1.0], dtype=np.float32),
        certificate_kind=np.asarray(["mean_minus_catastrophic_excess_rms"], dtype="U64"),
        downside_multiplier=np.asarray([1.0], dtype=np.float32),
        catastrophic_delta_threshold=np.asarray([0.0], dtype=np.float32),
    )
    import hashlib, pytest
    sha = hashlib.sha256(mem.read_bytes()).hexdigest()
    _load_icer_local_regret_memory.cache_clear()
    with pytest.raises(ValueError, match="catastrophic delta threshold"):
        _load_icer_local_regret_memory(str(mem), sha)


def test_v31_config_removes_rebinding_and_keeps_fixed_budget_and_preextremal_risk(tmp_path: Path):
    base = yaml.safe_load(Path("bdse/configs/v64_3_20_icer_dc_dual.yaml").read_text(encoding="utf-8"))
    memory = {"path": str(tmp_path / "m.npz"), "sha256": "0" * 64}
    cfg = omcer._cfg(base, memory)
    assert cfg["evidence"]["budget"] == 16
    assert cfg["selector"]["proposal_top_m"] == 24
    assert "proposal_conditioned_witness_rebinding" not in cfg["selector"]
    assert "frontier_contrast_rebinding" not in cfg["selector"]
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    assert ic["dominance_policy"] == "scalar_only"
    assert ic["incumbent_retention_policy"] == "preserve_admissible_incumbent"
    assert ic["regret_risk_feature_mode"] == "operator_margin_evidence"
    assert ic["regret_risk_model_type"] == "local_multiscale_catastrophic_excess_regret_certificate"
    assert ic["replacement_local_regret_neighbor_k_values"] == [32, 64]
    assert ic["replacement_local_regret_catastrophic_delta_threshold"] == -0.5
    assert "before extremization" in ic["replacement_operator"]


def test_v31_exclusion_stays_8700_because_corrected_v30_consumed_no_fresh():
    p = Path("bdse/configs/v64_3_31_design_exclude_v64_3_30_train_stop_tokens.txt")
    toks = [x.strip() for x in p.read_text().splitlines() if x.strip()]
    assert len(toks) == 8700 and len(set(toks)) == 8700
