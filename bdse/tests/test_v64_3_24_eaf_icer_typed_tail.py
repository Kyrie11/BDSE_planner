from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import bdse.planner.tournament as tournament
from bdse.planner.tournament import (
    _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES,
    _ICER_TYPED_EVIDENCE_FEATURE_NAMES,
    _icer_local_regret_lower_bound,
    _icer_local_tail_coherence_lower_bound,
    _icer_typed_selected_evidence_feature_matrix,
)
from bdse.tests.test_v64_3_19_eaf_icer import _cfg, _diag, _matrix


def test_typed_selected_evidence_exposes_incumbent_contrast_without_new_query() -> None:
    g = np.asarray([
        [0.8, 0.2, 0.7],   # occupancy: candidate 1 improves strongly vs incumbent 0
        [0.5, 0.4, 0.9],   # ttc: candidate 2 worsens
        [0.1, 0.2, 0.1],   # speed limit
    ], dtype=np.float32)
    x, names = _icer_typed_selected_evidence_feature_matrix(
        g, [0, 1, 2], ["occupancy", "ttc", "speed_limit"], np.ones(3, bool), 0
    )
    assert names == list(_ICER_TYPED_EVIDENCE_FEATURE_NAMES)
    assert x.shape == (3, len(names)) and np.all(np.isfinite(x))
    p = {n: i for i, n in enumerate(names)}
    assert x[1, p["occupancy_improvement_sum_norm"]] > 0.0
    assert x[2, p["ttc_downside_mass_norm"]] > 0.0
    assert np.isclose(x[0, p["occupancy_improvement_sum_norm"]], 0.0)
    assert np.isclose(x[1, p["occupancy_selected_fraction"]], 1.0 / 3.0)


def _tail_memory(tmp_path: Path) -> tuple[str, str]:
    n = len(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    path = tmp_path / "tail.npz"
    # Query z=0 sees the material negative as a genuine nearest-neighbor tail,
    # while the local average remains positive.  This isolates the V24 lower
    # partial moment from the unchanged V23 mean-1SE term.
    memory = np.zeros((96, n), dtype=np.float32)
    memory[:, 0] = np.linspace(1.0, 1.095, 96, dtype=np.float32)
    delta = np.full((96,), 0.02, dtype=np.float32)
    delta[0] = -0.20
    names = np.asarray([f"evidence::{x}" for x in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES], dtype="U128")
    np.savez_compressed(
        path,
        memory_metric_z=memory,
        teacher_improvement=delta,
        feature_mean=np.zeros((n,), dtype=np.float32),
        feature_std=np.ones((n,), dtype=np.float32),
        feature_names=names,
        feature_metric_weight=np.full((n,), 1.0 / n, dtype=np.float32),
        neighbor_k_values=np.asarray([32, 64], dtype=np.int32),
        se_multiplier=np.asarray([1.0], dtype=np.float32),
        material_delta_threshold=np.asarray([0.004], dtype=np.float32),
        tail_se_multiplier=np.asarray([1.0], dtype=np.float32),
    )
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def test_tail_coherence_penalizes_rare_material_downside_beyond_positive_mean_lcb(tmp_path: Path) -> None:
    p, h = _tail_memory(tmp_path)
    n = len(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    names = [f"evidence::{x}" for x in _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]
    q = np.zeros((1, n), dtype=np.float64)
    mean_lcb = float(_icer_local_regret_lower_bound(q, names, p, h)[0])
    tail = float(_icer_local_tail_coherence_lower_bound(q, names, p, h)[0])
    assert mean_lcb > 0.0
    assert tail < mean_lcb
    assert tail < 0.0


def _integrated_cfg(rank_policy: str) -> dict:
    c = _cfg(support_bias=2.0, dominance_bias=2.0, raw_margin_weight=10.0, policy="scalar_only")
    ic = c["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    ic.update({
        "all_flagged_policy": "preserve_legacy_for_structural_guard",
        "incumbent_retention_policy": "preserve_admissible_incumbent",
        "regret_risk_enabled": True,
        "retention_regret_risk_enabled": False,
        "replacement_regret_risk_enabled": True,
        "regret_risk_model_type": "local_multiscale_tail_coherence",
        "regret_risk_feature_mode": "typed_interaction",
        "replacement_rank_policy": rank_policy,
        "replacement_local_regret_memory_path": "unused-by-monkeypatch",
        "replacement_local_regret_memory_sha256": "",
    })
    return c


def test_full_v24_operator_ranks_by_regret_certificate_not_dominance_after_gate(monkeypatch) -> None:
    def fake_tail(x, runtime_names, memory_path, memory_sha256):
        # Alternative 3 has the larger frozen dominance logit, but alternative 1
        # has the substantially stronger tail-coherent replacement score.
        return np.asarray([0.0, 1.00, 0.0, 0.10], dtype=np.float64)
    monkeypatch.setattr(tournament, "_icer_local_tail_coherence_lower_bound", fake_tail)
    kwargs = dict(
        legacy_action=2, anchor_action=0, margin_matrix=_matrix(),
        attribution_star=np.asarray([0.0, 0.10, 0.05, 0.08]),
        atom_contrib_star=np.zeros((3, 4), dtype=np.float32),
        scores=np.asarray([0.0, 0.20, 0.40, 0.30]),
        valid_mask=np.ones(4, bool), runtime_safety_flags=np.zeros(4, bool),
        potential_diag=_diag(), evidence_certificate_fraction=1.0,
        utility_refinement_diag=None, selected_atoms=[0],
        predicted_atom_costs=np.zeros((1, 4), dtype=np.float32), atom_types=["occupancy"],
    )
    risk_first, d = tournament._apply_decisive_frontier_icer(cfg=_integrated_cfg("regret_risk_first"), **kwargs)
    dominance_first, _ = tournament._apply_decisive_frontier_icer(cfg=_integrated_cfg("dominance_first"), **kwargs)
    assert risk_first == 1
    assert dominance_first == 3
    assert float(d["decisive_frontier_icer_replacement_rank_regret_first"]) == 1.0
    assert float(d["decisive_frontier_icer_regret_risk_typed_interaction"]) == 1.0


def test_v24_design_exclusion_adds_exact_v23_double_fresh() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    old = {x.strip() for x in (root / "v64_3_23_design_exclude_v64_3_21_screen_tokens.txt").read_text().splitlines() if x.strip()}
    v23_fresh_path = Path("/mnt/data/exp_work/provenance/val_screen_fresh_1000_tokens.txt")
    # Repository tests must also work outside this analysis workspace, so only
    # the frozen count/subset invariant is mandatory there.
    new = {x.strip() for x in (root / "v64_3_24_design_exclude_v64_3_23_screen_tokens.txt").read_text().splitlines() if x.strip()}
    assert len(old) == 4700 and len(new) == 5700 and old <= new and len(new - old) == 1000
    if v23_fresh_path.is_file():
        fresh = {x.strip() for x in v23_fresh_path.read_text().splitlines() if x.strip()}
        assert fresh == new - old
