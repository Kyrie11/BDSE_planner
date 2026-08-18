from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
from bdse.planner.tournament import (
    _DACER_FEATURE_NAMES,
    _DALER_FEATURE_NAMES,
    _apply_decisive_frontier_icer,
    _icer_quadratic_interaction_features,
)


def _matrix() -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float32)
    for b, v in [(1, 0.20), (2, 0.40), (3, 0.30)]:
        m[b, 0] = v; m[0, b] = -v
    return m


def _diag() -> dict:
    return {
        "decisive_frontier_value_active": 1.0,
        "decisive_frontier_value_residual_rms": 0.2,
        "decisive_frontier_value_residual_abs_mean": 0.15,
        "decisive_frontier_value_attribution_scale_rms": 0.12,
        "decisive_frontier_value_attribution_scale_mean": 0.10,
    }


def _head_schema(mode: str) -> tuple[list[str], list[str]]:
    dummy = np.zeros((1, len(_DACER_FEATURE_NAMES)), dtype=np.float64)
    _, names, base = _icer_quadratic_interaction_features(dummy, list(_DACER_FEATURE_NAMES), mode)
    return names, base


def _cfg(*, support_bias: float = 1.0, dominance_bias: float = -1.0, raw_margin_weight: float = 0.0, policy: str = "scalar_only") -> dict:
    sn, sb = _head_schema("scalar_interaction"); pn, pb = _head_schema("profile_interaction")
    sw = [0.0] * len(_DALER_FEATURE_NAMES)
    dw = [0.0] * len(sn)
    if raw_margin_weight:
        dw[sn.index("lin::raw_margin")] = raw_margin_weight
    return {
        "runtime": {
            "pair_action_anchor_guard": {"enabled": True, "flip_margin": 0.015, "score_margin": 0.0},
            "dual_certificate": {"enabled": True, "require_evidence_certificate_before_residual_flip": True, "min_evidence_certificate_fraction_for_residual_flip": 1.0, "residual_beta_uncertainty": 0.0, "residual_epsilon": 0.0},
            "decisive_frontier_value": {
                "incumbent_contrastive_extremal_recovery": {
                    "enabled": True, "instrument_features": True, "dominance_policy": policy,
                    "support_feature_names": list(_DALER_FEATURE_NAMES), "support_feature_mean": [0.0] * len(sw), "support_feature_std": [1.0] * len(sw), "support_weights": sw, "support_bias": support_bias,
                    "scalar_dominance_base_feature_names": sb, "scalar_dominance_feature_names": sn, "scalar_dominance_feature_mean": [0.0] * len(sn), "scalar_dominance_feature_std": [1.0] * len(sn), "scalar_dominance_weights": dw, "scalar_dominance_bias": dominance_bias,
                    "profile_dominance_base_feature_names": pb, "profile_dominance_feature_names": pn, "profile_dominance_feature_mean": [0.0] * len(pn), "profile_dominance_feature_std": [1.0] * len(pn), "profile_dominance_weights": [0.0] * len(pn), "profile_dominance_bias": dominance_bias,
                    "require_guard_admissible": True, "require_safe_available_for_learned_intervention": True,
                }
            },
        },
        "tournament": {"utility_refinement": {"enabled": False}},
    }


def _run(cfg: dict, evidence_fraction: float = 1.0, safety: np.ndarray | None = None) -> tuple[int, dict]:
    return _apply_decisive_frontier_icer(
        2, 0, _matrix(), np.asarray([0.0, 0.10, 0.05, 0.08]), np.zeros((3, 4), dtype=np.float32),
        np.asarray([0.0, 0.20, 0.40, 0.30]), np.ones(4, bool), np.zeros(4, bool) if safety is None else safety,
        _diag(), evidence_fraction, None, cfg,
    )


def test_icer_keeps_supported_admissible_incumbent_without_positive_dominance() -> None:
    selected, d = _run(_cfg(dominance_bias=-2.0))
    assert selected == 2
    assert float(d["decisive_frontier_icer_legacy_admissible"]) == 1.0


def test_icer_replaces_incumbent_only_with_support_and_positive_direct_dominance() -> None:
    selected, d = _run(_cfg(dominance_bias=-1.5, raw_margin_weight=10.0))
    assert selected in {1, 3}
    assert selected != 2
    assert float(d["decisive_frontier_icer_selected_dominance_logit"]) > 0.0


def test_icer_negative_support_cannot_bypass_to_alternative() -> None:
    selected, _ = _run(_cfg(support_bias=-1.0, dominance_bias=10.0, raw_margin_weight=10.0))
    assert selected == 0


def test_icer_fails_closed_on_evidence_certificate_and_all_flagged_scene() -> None:
    selected, d = _run(_cfg(dominance_bias=10.0), evidence_fraction=0.5)
    assert int(np.asarray(d["_decisive_frontier_icer_admissible_mask"]).sum()) == 0
    assert selected == 0
    selected2, d2 = _run(_cfg(dominance_bias=10.0), safety=np.ones(4, dtype=bool))
    assert int(np.asarray(d2["_decisive_frontier_icer_admissible_mask"]).sum()) == 0
    assert selected2 == 0


def test_icer_quadratic_map_is_fixed_and_profile_strictly_contains_signed_atom_view() -> None:
    X = np.arange(2 * len(_DACER_FEATURE_NAMES), dtype=np.float64).reshape(2, -1) / 100.0
    sx, sn, sb = _icer_quadratic_interaction_features(X, list(_DACER_FEATURE_NAMES), "scalar_interaction")
    px, pn, pb = _icer_quadratic_interaction_features(X, list(_DACER_FEATURE_NAMES), "profile_interaction")
    assert sx.shape[1] == len(sb) + len(sb) * (len(sb) + 1) // 2
    assert px.shape[1] == len(pb) + len(pb) * (len(pb) + 1) // 2
    assert len(px[0]) > len(sx[0])
    assert "lin::delta_atom_top1_signed_norm" in pn
    assert np.all(np.isfinite(px)) and np.all(np.isfinite(sx))


def test_preprocessed_token_filter_resolves_manifest_before_npz_deserialization(tmp_path: Path) -> None:
    val = tmp_path / "val"; val.mkdir()
    a = val / "tokA_it000001.npz"; b = val / "tokB_it000001.npz"; a.touch(); b.touch()
    manifest = val / "manifest.jsonl"
    manifest.write_text("\n".join([
        json.dumps({"split": "val", "scenario_token": "tokA", "path": str(a)}),
        json.dumps({"split": "val", "scenario_token": "tokB", "path": str(b)}),
    ]) + "\n", encoding="utf-8")
    ds = PreprocessedBDSEDataset(tmp_path, split="val", scenario_tokens={"tokB"})
    paths = ds.build_index()
    assert paths == [b]
    assert getattr(ds, "_scenario_token_matches", set()) == {"tokB"}


def test_icer_screen_separates_direct_incumbent_replacement_from_anchor_recovery(tmp_path: Path) -> None:
    from bdse.tools.check_v64_3_19_eaf_icer_screen import _icer_edge_diag

    rows = [
        # scene 1: raw incumbent is admissible; selected alternative truly beats it.
        {"scenario_token": "s1", "anchor_action": 0, "raw_top_action": 1, "challenger_action": 1,
         "teacher_margin": 0.20, "icer_admissible": 1.0, "icer_selected_action": 2,
         "icer_support_logit": 1.0, "icer_dominance_logit": 0.0,
         "icer_scalar_dominance_logit": 0.0, "icer_profile_dominance_logit": 0.0},
        {"scenario_token": "s1", "anchor_action": 0, "raw_top_action": 1, "challenger_action": 2,
         "teacher_margin": 0.35, "icer_admissible": 1.0, "icer_selected_action": 2,
         "icer_support_logit": 1.0, "icer_dominance_logit": 1.0,
         "icer_scalar_dominance_logit": 1.0, "icer_profile_dominance_logit": 1.0},
        # scene 2: raw top is inadmissible; selecting action 2 is anchor-relative recovery,
        # not evidence for the incumbent-contrastive replacement claim.
        {"scenario_token": "s2", "anchor_action": 0, "raw_top_action": 1, "challenger_action": 1,
         "teacher_margin": 0.40, "icer_admissible": 0.0, "icer_selected_action": 2,
         "icer_support_logit": 0.0, "icer_dominance_logit": 0.0,
         "icer_scalar_dominance_logit": 0.0, "icer_profile_dominance_logit": 0.0},
        {"scenario_token": "s2", "anchor_action": 0, "raw_top_action": 1, "challenger_action": 2,
         "teacher_margin": 0.10, "icer_admissible": 1.0, "icer_selected_action": 2,
         "icer_support_logit": 1.0, "icer_dominance_logit": 0.0,
         "icer_scalar_dominance_logit": 0.0, "icer_profile_dominance_logit": 0.0},
    ]
    p = tmp_path / "edges.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    d = _icer_edge_diag(p)
    assert d["direct_incumbent_proposal_count"] == 1.0
    assert d["direct_incumbent_replacement_rate"] == 1.0
    assert d["direct_incumbent_replacement_precision"] == 1.0
    assert d["anchor_recovery_rate_on_proposals"] == 0.5
    assert d["anchor_recovery_precision"] == 1.0
