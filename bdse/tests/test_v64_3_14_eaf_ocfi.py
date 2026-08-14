from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from bdse.planner.tournament import (
    _decisive_frontier_value_star_residual_numpy,
    run_pair_conditioned_tournament,
)
from bdse.tools.calibrate_v64_3_14_eaf_ocfi import calibrate, _write_calibrated_config
from bdse.tools.check_v64_3_14_eaf_ocfi_contract import run_contract
from bdse.tools.check_v64_3_14_eaf_ocfi_screen import build as build_screen


def _cfg(*, ocfi: bool, q: float = 0.0, normalization: str = "attribution") -> dict:
    return {
        "runtime": {
            "pair_tournament_anchor_mode": "selected_local",
            "pair_tournament_pair_delta_includes_local": True,
            "pair_tournament_aggregation_mode": "decisive_anchor_margin",
            "pair_action_anchor_guard": {"enabled": True, "flip_margin": 0.01, "score_margin": 0.0},
            "dual_certificate": {"enabled": False},
            "decisive_frontier_value": {
                "enabled": True,
                "scale": 1.0,
                "one_sided_intervention": {
                    "enabled": ocfi,
                    "normalization": normalization,
                    "calibration_quantile": q,
                    "attribution_scale_floor": 0.005,
                    "additive_radius": 0.0,
                    "require_frontier_active": True,
                },
            },
        },
        "model": {"pair_margin_normalized": False},
        "tournament": {"epsilon_cal": 0.0, "beta_uncertainty": 0.0, "use_softmin": True, "softmin_tau": 1.0},
        "selector": {"pair_screen_top_l": 3, "pair_screen_near_eta": 10.0},
    }


def _frontier_inputs():
    J0 = np.array([0.0, 2.0, 5.0], dtype=np.float32)
    g = np.zeros((1, 3), dtype=np.float32)
    pairs = np.array([[0, 1]], dtype=np.int64)
    pair_delta = np.zeros((1, 1), dtype=np.float32)
    valid = np.ones(3, dtype=bool)
    safety = np.zeros(3, dtype=bool)
    atom = np.ones((1, 1), dtype=np.float32) * 3.0
    signed = np.array([[0.0], [0.0], [-10.0]], dtype=np.float32)
    context = np.ones((3, 1), dtype=np.float32)
    return J0, g, pairs, pair_delta, valid, safety, atom, signed, context


def _run(cfg: dict):
    J0, g, pairs, pair_delta, valid, safety, atom, signed, context = _frontier_inputs()
    return run_pair_conditioned_tournament(
        J0, pair_delta, pairs, [0], valid, safety, cfg,
        predicted_atom_costs=g,
        frontier_value_atom_factors=atom,
        frontier_value_action_signed_factors=signed,
        frontier_value_action_context_factors=context,
    )


def test_ocfi_zero_quantile_is_exact_v64_3_13_runtime_noop() -> None:
    raw = _run(_cfg(ocfi=False))
    zero = _run(_cfg(ocfi=True, q=0.0))
    assert raw.action_index == zero.action_index == 2
    np.testing.assert_allclose(raw.margins, zero.margins, rtol=0.0, atol=0.0)
    assert zero.diagnostics["decisive_frontier_ocfi_active"] == 1.0
    assert zero.diagnostics["decisive_frontier_ocfi_calibration_radius"] == 0.0


def test_ocfi_large_one_sided_radius_blocks_overconfident_frontier_flip() -> None:
    raw = _run(_cfg(ocfi=False))
    guarded = _run(_cfg(ocfi=True, q=100.0))
    assert raw.action_index == 2
    assert guarded.action_index == 0
    assert guarded.diagnostics["decisive_frontier_ocfi_active"] == 1.0
    assert guarded.diagnostics["decisive_frontier_ocfi_calibration_radius"] > 0.0
    assert guarded.diagnostics["pair_action_anchor_guard_blocked_flip"]


def test_ocfi_is_noop_when_eaf_is_intentionally_absent_from_pairfull_ceiling() -> None:
    J0 = np.array([0.0, 2.0, 5.0], dtype=np.float32)
    g = np.zeros((1, 3), dtype=np.float32)
    pairs = np.array([[0, 1]], dtype=np.int64)
    delta = np.zeros((1, 1), dtype=np.float32)
    valid = np.ones(3, dtype=bool)
    safety = np.zeros(3, dtype=bool)
    raw = run_pair_conditioned_tournament(J0, delta, pairs, [0], valid, safety, _cfg(ocfi=False), predicted_atom_costs=g)
    ocfi = run_pair_conditioned_tournament(J0, delta, pairs, [0], valid, safety, _cfg(ocfi=True, q=100.0), predicted_atom_costs=g)
    assert raw.action_index == ocfi.action_index
    np.testing.assert_allclose(raw.margins, ocfi.margins, rtol=0.0, atol=0.0)
    assert ocfi.diagnostics["decisive_frontier_value_active"] == 0.0
    assert ocfi.diagnostics["decisive_frontier_ocfi_active"] == 0.0


def test_eaf_attribution_scale_uses_same_additive_atom_contributions() -> None:
    rng = np.random.default_rng(7)
    atom = rng.normal(size=(4, 5)).astype(np.float32)
    signed = rng.normal(size=(3, 5)).astype(np.float32)
    context = rng.normal(size=(3, 5)).astype(np.float32)
    selected = [0, 2, 3]
    out, diag = _decisive_frontier_value_star_residual_numpy(selected, np.ones(3, bool), 0, atom, signed, context)
    scale_star = np.asarray(diag["_decisive_frontier_value_attribution_scale_star"], dtype=np.float32)
    assert out.shape == scale_star.shape == (3,)
    assert np.all(scale_star >= 0.0)
    assert scale_star[0] == 0.0
    assert diag["decisive_frontier_value_attribution_scale_rms"] > 0.0

    # The deployed residual must remain the exact V64.3.13 arithmetic; the new
    # per-atom decomposition is side information used only to scale calibration.
    selected_arr = np.asarray(selected, dtype=np.int64).reshape(-1)
    bounded = np.tanh(atom[selected_arr])
    pooled = bounded.sum(axis=0) / np.sqrt(max(float(selected_arr.size), 1.0))
    pooled = pooled.astype(np.float32)
    pair_sym = np.tanh(context[0][None, :] + context + context[0][None, :] * context).astype(np.float32)
    signed_diff = signed - signed[0][None, :]
    old_v13 = ((pooled[None, :] * pair_sym * signed_diff).sum(axis=1) / np.sqrt(max(float(atom.shape[1]), 1.0))).astype(np.float32) * float(1.0)
    old_v13[0] = 0.0
    np.testing.assert_array_equal(out, old_v13)

    pair_vec = pair_sym * signed_diff
    denom = np.sqrt(max(float(selected_arr.size * atom.shape[1]), 1.0))
    contrib = np.einsum("nr,kr->nk", bounded.astype(np.float32), pair_vec, optimize=True).astype(np.float32) / denom
    contrib_sum = contrib.sum(axis=0).astype(np.float32)
    contrib_sum[0] = 0.0
    np.testing.assert_allclose(contrib_sum, out, rtol=2e-6, atol=2e-7)
    expected_scale = np.sqrt(np.sum(contrib * contrib, axis=0)).astype(np.float32)
    expected_scale[0] = 0.0
    np.testing.assert_allclose(scale_star, expected_scale, rtol=1e-6, atol=1e-7)


def _synthetic_rows(n: int = 240) -> list[dict]:
    rows = []
    for i in range(n):
        attr = 0.05 + 0.001 * (i % 40)
        teacher_margin = -0.10 + 0.01 * (i % 30)
        # Controlled positive over-estimation proportional to attribution scale.
        pred_margin = teacher_margin + (0.4 + 0.01 * (i % 10)) * attr
        teacher_action = 1 if i % 5 == 0 else 0
        bdse_action = teacher_action if i % 4 else 2
        rows.append({
            "scenario_token": f"scene-{i:04d}",
            "timestamp_us": i,
            "raw_frontier_anchor_action": 0,
            "raw_frontier_proposed_action": 1,
            "pair_action_anchor_raw_margin": pred_margin,
            "decisive_frontier_value_teacher_proposed_vs_anchor_margin": teacher_margin,
            "decisive_frontier_ocfi_proposed_attribution_scale": attr,
            "decisive_frontier_value_active": 1.0,
            "decisive_frontier_value_complete_star_coverage": 1.0,
            "teacher_action_match": float(bdse_action == teacher_action),
            "teacher_regret": float(i % 13 + 1),
            "pair_full_interface_action_match": 0.2,
            "local_pair_full_interface_action_match": 0.2,
            "evidence_certificate_fraction": 0.93,
            "selected_local_anchor_action_match": 0.17,
            "deployed_vs_selected_local_anchor_match": 0.4,
            "pair_potential_deployed_flip_rate": 0.6,
            "beneficial_pair_potential_intervention_rate": 0.08,
            "harmful_pair_potential_intervention_rate": 0.09,
            "decisive_frontier_value_residual_rms": 0.2,
        })
    return rows


def test_ocfi_calibration_is_group_disjoint_and_writes_frozen_config(tmp_path: Path) -> None:
    report = calibrate(
        _synthetic_rows(), normalization="attribution", alpha=0.10,
        calibration_fraction=0.4, split_seed="unit", scale_floor_quantile=0.1, min_scale_floor=0.005,
    )
    assert report["calibration_proposal_edge_count"] >= 32
    assert report["calibration_quantile"] > 0.0
    assert not (set(report["calibration_tokens"]) & set(report["evaluation_tokens"]))
    out_cfg = tmp_path / "calibrated.yaml"
    report_path = tmp_path / "calibration.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _write_calibrated_config(
        Path("bdse/configs/v64_3_14_eaf_ocfi_raw_calibration.yaml"), out_cfg, report, report_path
    )
    cfg = yaml.safe_load(out_cfg.read_text(encoding="utf-8"))
    ocfi = cfg["runtime"]["decisive_frontier_value"]["one_sided_intervention"]
    assert ocfi["enabled"] is True
    assert ocfi["normalization"] == "attribution"
    assert ocfi["calibration_quantile"] == report["calibration_quantile"]
    contract = run_contract(str(out_cfg), "calibrated")
    assert contract["pass"], [k for k, v in contract["checks"].items() if not v]


def _candidate_metrics(*, teacher: float, regret: float, harm: float, benefit: float, flip: float, attr: bool) -> dict:
    return {
        "teacher_action_match": teacher,
        "teacher_regret": regret,
        "pair_full_interface_action_match": 0.20,
        "local_pair_full_interface_action_match": 0.20,
        "evidence_certificate_fraction": 0.93,
        "selected_local_anchor_action_match": 0.17,
        "pair_potential_deployed_flip_rate": flip,
        "beneficial_pair_potential_intervention_rate": benefit,
        "harmful_pair_potential_intervention_rate": harm,
        "decisive_frontier_value_active": 1.0,
        "decisive_frontier_value_complete_star_coverage": 1.0,
        "decisive_frontier_ocfi_active": 1.0,
        "decisive_frontier_ocfi_calibration_radius": 0.08,
        "decisive_frontier_ocfi_calibration_quantile": 1.2 if attr else 0.2,
    }


def _cal_report(norm: str) -> dict:
    return {
        "normalization": norm,
        "alpha": 0.1,
        "calibration_quantile": 1.2 if norm == "attribution" else 0.2,
        "attribution_scale_floor": 0.02 if norm == "attribution" else 1.0,
        "calibration_proposal_edge_count": 100,
        "calibration_group_count": 100,
        "evaluation_group_count": 140,
        "raw_eval_subset_metrics": {
            "teacher_action_match": 0.15,
            "teacher_regret": 15000.0,
            "pair_full_interface_action_match": 0.20,
            "local_pair_full_interface_action_match": 0.20,
            "evidence_certificate_fraction": 0.93,
            "selected_local_anchor_action_match": 0.17,
            "pair_potential_deployed_flip_rate": 0.58,
            "beneficial_pair_potential_intervention_rate": 0.08,
            "harmful_pair_potential_intervention_rate": 0.09,
        },
    }


def test_ocfi_screen_prefers_attribution_when_it_preserves_and_improves_endpoint() -> None:
    attr = _candidate_metrics(teacher=0.17, regret=14000.0, harm=0.03, benefit=0.06, flip=0.32, attr=True)
    const = _candidate_metrics(teacher=0.155, regret=14900.0, harm=0.05, benefit=0.055, flip=0.37, attr=False)
    r = build_screen(attr, _cal_report("attribution"), const, _cal_report("none"))
    assert r["selected_normalization"] == "attribution"
    assert r["full_promotion"]
    assert r["attribution_specific_gain"]


def test_ocfi_screen_does_not_call_parity_attribution_specific_novelty() -> None:
    attr = _candidate_metrics(teacher=0.17, regret=14000.0, harm=0.03, benefit=0.06, flip=0.32, attr=True)
    const = dict(attr)
    const["decisive_frontier_ocfi_calibration_quantile"] = 0.2
    r = build_screen(attr, _cal_report("attribution"), const, _cal_report("none"))
    assert r["full_promotion"]
    assert not r["attribution_specific_gain"]
