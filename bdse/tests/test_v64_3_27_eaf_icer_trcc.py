from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import bdse.tools.fit_v64_3_27_eaf_icer_trcc as trcc
from bdse.planner import tournament as tour


def test_type_resolved_features_keep_fixed_type_coordinates_and_same_atom_contrast():
    # selected rows: occupancy and ttc. Equal magnitudes must not collapse into a
    # magnitude-sorted spectrum; candidate and candidate-incumbent use the same
    # fixed semantic type coordinate.
    contrib = np.asarray([
        [0.0, 0.10, 0.50],   # occupancy
        [0.0, 0.20, -0.50],  # ttc
    ], dtype=float)
    out, names = tour._icer_semantic_type_feature_matrix(
        contrib, ["occupancy", "ttc"], np.asarray([True, True, True]), 0, 1
    )
    assert names == list(tour._ICER_SEMANTIC_TYPE_FEATURE_NAMES)
    occ = list(tour._ICER_SEMANTIC_TYPE_NAMES).index("occupancy")
    ttc = list(tour._ICER_SEMANTIC_TYPE_NAMES).index("ttc")
    T = len(tour._ICER_SEMANTIC_TYPE_NAMES)
    assert np.isclose(out[2, occ], 0.50)
    assert np.isclose(out[2, ttc], -0.50)
    assert np.isclose(out[2, T + occ], 0.40)
    assert np.isclose(out[2, T + ttc], -0.70)
    assert out.shape[1] == 24


def test_type_resolved_features_fail_closed_on_unknown_atom_type():
    contrib = np.zeros((1, 3), dtype=float)
    with pytest.raises(ValueError, match="unknown selected atom types"):
        tour._icer_semantic_type_feature_matrix(
            contrib, ["mystery_atom"], np.asarray([True, True, True]), 0, 1
        )


def test_semantic_type_risk_view_is_type_only_not_flat_concatenation():
    K = 3
    feature_names = list(tour._ICER_DOMINANCE_PROFILE_BASE_NAMES)
    feat = np.zeros((K, len(feature_names)), dtype=float)
    type_feat = np.arange(K * 24, dtype=float).reshape(K, 24)
    x, names = tour._icer_regret_risk_feature_matrix(
        feat, feature_names, np.zeros((K, 1)), ["dummy"], "semantic_type_only",
        np.zeros((K, 32)), list(tour._ICER_ATTRIBUTION_RESOLVED_FEATURE_NAMES),
        np.zeros((K, 10)), list(tour._ICER_SEMANTIC_FAMILY_FEATURE_NAMES),
        type_feat, list(tour._ICER_SEMANTIC_TYPE_FEATURE_NAMES),
    )
    assert x.shape == (K, 24)
    assert names == [f"semantic_type::{n}" for n in tour._ICER_SEMANTIC_TYPE_FEATURE_NAMES]
    assert not any(n.startswith("evidence::") or n.startswith("semantic_family::") for n in names)


def test_confirmation_is_monotone_and_has_no_fallback_to_second_candidate():
    cand = np.asarray([2, 3], dtype=np.int64)
    dominance = np.asarray([0.0, 0.0, 2.0, 1.0])
    aggregate_risk = np.asarray([0.0, 0.0, 0.4, 0.8])
    support = np.asarray([0.0, 0.0, 1.0, 1.0])
    margin = np.asarray([0.0, 0.0, 0.3, 0.2])
    utility = np.asarray([0, 0, 1, 1])
    # Aggregate proposal is action 2 by frozen scalar dominance. Type view vetoes
    # action 2 but would approve action 3. TRCC MUST return None, never action 3.
    confirmation = np.asarray([np.nan, np.nan, -0.1, +1.0])
    got = tour._icer_select_extremal_candidate_with_optional_confirmation(
        cand, dominance, aggregate_risk, support, margin, utility,
        confirmation_logits=confirmation,
    )
    assert got is None
    # Without confirmation this is exactly the aggregate proposal.
    assert tour._icer_select_extremal_candidate_with_optional_confirmation(
        cand, dominance, aggregate_risk, support, margin, utility,
    ) == 2


def test_v27_main_config_freezes_aggregate_proposal_and_separate_type_confirmation(tmp_path: Path):
    base = yaml.safe_load(Path("bdse/configs/v64_3_20_icer_dc_dual.yaml").read_text(encoding="utf-8"))
    am = {"path": str(tmp_path / "agg.npz"), "sha256": "a" * 64}
    tm = {"path": str(tmp_path / "type.npz"), "sha256": "b" * 64}
    cfg = trcc._cfg_main(base, am, tm)
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    assert ic["dominance_policy"] == "scalar_only"
    assert ic["incumbent_retention_policy"] == "preserve_admissible_incumbent"
    assert ic["regret_risk_feature_mode"] == "evidence_only"
    assert ic["replacement_confirmation_regret_risk_feature_mode"] == "semantic_type_only"
    assert ic["regret_risk_model_type"] == "local_multiscale_downside_regret_with_type_confirmation"
    assert ic["replacement_local_regret_neighbor_k_values"] == [32, 64]
    assert ic["replacement_confirmation_local_regret_neighbor_k_values"] == [32, 64]
    assert "NO fallback/reselection" in ic["replacement_operator"]
    assert "subset_of" in ic["replacement_selection_monotonicity"]


def test_streaming_loader_requires_all_24_type_resolved_instrumentation_fields(tmp_path: Path):
    p = tmp_path / "edges.jsonl"
    row = {
        "scenario_token": "abc", "anchor_action": 0, "raw_top_action": 1,
        "challenger_action": 2, "icer_admissible": 1.0, "teacher_margin": 0.1,
        "icer_support_logit": 1.0, "icer_scalar_dominance_logit": 2.0,
    }
    row.update({f"icer_feature_{n}": float(i) for i, n in enumerate(trcc._BASE_NAMES)})
    row.update({f"icer_semantic_type_{n}": float(i) for i, n in enumerate(trcc._TYPE_NAMES)})
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    by, n = trcc._load_minimal_scenes(p)
    assert n == 1
    kept = by["abc"][0]
    assert len([k for k in kept if k.startswith("icer_feature_")]) == 18
    assert len([k for k in kept if k.startswith("icer_semantic_type_")]) == 24


def test_crossfit_main_selection_is_structural_subset_of_aggregate_control():
    # Directly exercise the offline selection invariant with one scene and two
    # alternatives: type confirmation can delete the aggregate proposal only.
    data = {
        "tok": np.asarray(["s", "s"], dtype=object),
        "support": np.asarray([1.0, 1.0]),
        "scalar": np.asarray([2.0, 1.0]),
        "delta": np.asarray([-1.0, +1.0]),
        "action": np.asarray([2, 3]),
    }
    aggregate = np.asarray([+0.5, +0.5])
    type_score = np.asarray([-0.2, +0.9])
    out, selected, proposed = trcc._confirmed_selection(data, aggregate, type_score, {"s"})
    assert proposed == {("s", 2)}
    assert selected == set()
    assert out["type_confirmation_veto_count"] == 1
