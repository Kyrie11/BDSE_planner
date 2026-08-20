from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

import bdse.tools.fit_v64_3_26_eaf_icer_sarc as sarc
from bdse.planner import tournament as tour


def test_semantic_family_features_preserve_family_identity_and_incumbent_correspondence():
    # Rows are selected evidence atoms; columns are actions.  Anchor=0,
    # incumbent=1, candidate=2.  The two selected atoms have equal magnitude but
    # different semantic families.  A V24 abs-sorted spectrum would erase this
    # identity; SARC keeps it in fixed coordinates.
    contrib = np.asarray([
        [0.0, 0.10, 0.50],  # feasibility
        [0.0, 0.20, -0.50], # interaction
    ], dtype=float)
    out, names = tour._icer_semantic_family_feature_matrix(
        contrib, np.asarray([1, 2]), np.asarray([True, True, True]), 0, 1
    )
    assert names == list(tour._ICER_SEMANTIC_FAMILY_FEATURE_NAMES)
    # Candidate family signed sums.
    assert np.isclose(out[2, 0], 0.50)
    assert np.isclose(out[2, 1], -0.50)
    # Candidate-minus-incumbent on the exact same family-aligned atoms.
    assert np.isclose(out[2, 5], 0.40)
    assert np.isclose(out[2, 6], -0.70)
    # No L1 normalization: physical contribution scale is preserved.
    assert np.isclose(np.abs(out[2, :5]).sum(), 1.0)


def test_semantic_family_swapping_atom_families_changes_representation():
    contrib = np.asarray([[0.0, 0.0, 0.8], [0.0, 0.0, -0.2]], dtype=float)
    valid = np.asarray([True, True, True])
    a, _ = tour._icer_semantic_family_feature_matrix(contrib, [1, 2], valid, 0, 1)
    b, _ = tour._icer_semantic_family_feature_matrix(contrib, [2, 1], valid, 0, 1)
    assert not np.allclose(a[2], b[2])
    # Yet an abs-sorted/L1 spectrum would be identical, demonstrating the exact
    # V24 information loss that this representation is designed to avoid.
    assert np.allclose(tour._signed_attribution_spectrum(contrib[:, 2]), tour._signed_attribution_spectrum(contrib[:, 2]))


def test_regret_risk_semantic_mode_is_18_plus_10_with_no_transition_or_sorted_spectrum():
    K = 3
    feature_names = list(tour._ICER_DOMINANCE_PROFILE_BASE_NAMES)
    feat = np.zeros((K, len(feature_names)), dtype=float)
    tr = np.zeros((K, 1), dtype=float)
    sf = np.arange(K * 10, dtype=float).reshape(K, 10)
    x, names = tour._icer_regret_risk_feature_matrix(
        feat, feature_names, tr, ["dummy"], "semantic_family_aligned",
        np.zeros((K, 32)), list(tour._ICER_ATTRIBUTION_RESOLVED_FEATURE_NAMES),
        sf, list(tour._ICER_SEMANTIC_FAMILY_FEATURE_NAMES),
    )
    assert x.shape == (K, 28)
    assert len(names) == 28
    assert all(n.startswith("evidence::") for n in names[:18])
    assert all(n.startswith("semantic_family::") for n in names[18:])
    assert not any(n.startswith("attribution::") or n.startswith("transition::") for n in names)


def test_v26_config_changes_only_risk_representation(tmp_path: Path):
    base = yaml.safe_load(Path("bdse/configs/v64_3_20_icer_dc_dual.yaml").read_text(encoding="utf-8"))
    mem = {"path": str(tmp_path / "m.npz"), "sha256": "0" * 64}
    agg = sarc._cfg(base, mem, "evidence_only", "aggregate_downside")
    sem = sarc._cfg(base, mem, "semantic_family_aligned", "semantic_family_downside")
    ai = agg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    si = sem["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    assert ai["regret_risk_feature_mode"] == "evidence_only"
    assert si["regret_risk_feature_mode"] == "semantic_family_aligned"
    for key in ["dominance_policy", "incumbent_retention_policy", "regret_risk_model_type",
                "replacement_local_regret_neighbor_k_values", "replacement_local_regret_certificate"]:
        assert ai[key] == si[key]
    assert si["dominance_policy"] == "scalar_only"
    assert si["incumbent_retention_policy"] == "preserve_admissible_incumbent"
    assert si["replacement_local_regret_neighbor_k_values"] == [32, 64]
    assert "semantic-family coordinates affect only the certificate neighborhood" in si["replacement_operator"]


def test_streaming_loader_requires_semantic_family_instrumentation(tmp_path: Path):
    p = tmp_path / "edges.jsonl"
    row = {
        "scenario_token": "abc", "anchor_action": 0, "raw_top_action": 1,
        "challenger_action": 2, "icer_admissible": 1.0, "teacher_margin": 0.1,
        "icer_support_logit": 1.0, "icer_scalar_dominance_logit": 2.0,
    }
    row.update({f"icer_feature_{n}": float(i) for i, n in enumerate(sarc._BASE_NAMES)})
    row.update({f"icer_semantic_family_{n}": float(i) for i, n in enumerate(sarc._SEMANTIC_NAMES)})
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    by, n = sarc._load_minimal_scenes(p)
    assert n == 1
    kept = by["abc"][0]
    assert len([k for k in kept if k.startswith("icer_feature_")]) == 18
    assert len([k for k in kept if k.startswith("icer_semantic_family_")]) == 10


def test_equal_metric_weights_no_family_group_weight():
    X = np.arange(56, dtype=float).reshape(2, 28)
    _, _, _, w = sarc._memory(X)
    assert np.allclose(w, np.full(28, 1.0 / 28.0))
    assert np.isclose(w.sum(), 1.0)
