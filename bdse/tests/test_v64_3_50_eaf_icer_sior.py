from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
import numpy as np
import torch

from bdse.planner.selected_outcome_probe import SelectedOutcomeProbeState, apply_selected_outcome_probe
from bdse.model.bdse_model import BDSEModel
from bdse.planner.future_state_factorization import FSFR_OBSERVABLE_NAMES, runtime_future_state_factorization_observable_costs
from bdse.planner.tournament import _ICER_REGRET_RISK_EVIDENCE_BASE_NAMES, _icer_post_selection_value
from bdse.planner.value_observables import VALUE_OBSERVABLE_NAMES, runtime_value_observable_costs
from bdse.tools.fit_v64_3_41_eaf_icer_epvr import EPV_NAMES
from bdse.tests.test_v64_3_47_eaf_icer_fsfr import _runtime as _v47_runtime, _bank as _v47_bank, _cfg as _v47_cfg
from bdse.tools.fit_v64_3_50_eaf_icer_sior import ALPHA_RET, PAIR_COLLECTION_PROTOCOL_VERSION, V49_FAILURE, _check_v49, _statistical_admissibility
from bdse.tools.prepare_v64_3_50_eaf_icer_sior_probe_configs import _make
from bdse.tools.run_v64_3_50_paired_selected_outcome_collection import (
    COLLECTION_PROTOCOL_VERSION,
    COLLECTION_ENGINE_VERSION,
    HARD_METRICS,
    _probe_rows,
    _scenario_metric_rows,
    _build_token_db_index,
    _hard_noninferiority,
    _validate_native_nuplan_inputs,
    _validate_pair,
    _BatchProgressTracker,
)

ROOT = Path(__file__).resolve().parents[2]


def _diag(proposal: int = 7, baseline: int = 2, selected: int = 7) -> dict:
    return {
        "decisive_frontier_icer_scir_proposal_exists": 1.0,
        "decisive_frontier_icer_scir_proposal_action": float(proposal),
        "decisive_frontier_icer_baseline_action": float(baseline),
        "decisive_frontier_icer_selected_action": float(selected),
    }


def _cfg(role: str) -> dict:
    return {"fallback": {"enabled": False}, "selected_outcome_probe": {"enabled": True, "role": role}}


def test_v50_probe_disabled_is_exact_noop() -> None:
    s = SelectedOutcomeProbeState()
    out, d = apply_selected_outcome_probe(7, _diag(), {}, s)
    assert out == 7
    assert d["enabled"] is False
    assert s.executed_intervention_count == 0


def test_v50_control_always_preserves_incumbent_at_rsmr_proposal() -> None:
    s = SelectedOutcomeProbeState()
    out, d = apply_selected_outcome_probe(7, _diag(), _cfg("control"), s)
    assert out == 2
    assert d["first_proposal_now"] is True
    assert d["intervention_executed"] is False
    assert s.executed_intervention_count == 0


def test_v50_treatment_executes_exact_live_winner_once_then_incumbent() -> None:
    s = SelectedOutcomeProbeState()
    out1, d1 = apply_selected_outcome_probe(7, _diag(), _cfg("treatment"), s)
    out2, d2 = apply_selected_outcome_probe(7, _diag(), _cfg("treatment"), s)
    assert out1 == 7 and d1["intervention_executed"] is True
    assert out2 == 2 and d2["intervention_executed"] is False
    assert s.executed_intervention_count == 1
    s.reset()
    assert s.executed_intervention_count == 0 and s.first_proposal_seen is False


def test_v50_treatment_overrides_historical_veto_but_rejects_rerank() -> None:
    # Historical post-selection may veto the frozen proposal to baseline.  V50
    # treatment assignment must still execute the pre-post-selection RSMR proposal.
    s = SelectedOutcomeProbeState()
    out, d = apply_selected_outcome_probe(2, _diag(proposal=7, baseline=2, selected=2), _cfg("treatment"), s)
    assert out == 7 and d["rsmr_selected_action"] == 7 and d["pre_probe_selected_action"] == 2
    s.reset()
    with pytest.raises(ValueError, match=r"outside \{live RSMR proposal, incumbent\}"):
        apply_selected_outcome_probe(9, _diag(proposal=7, baseline=2, selected=9), _cfg("treatment"), s)


def test_v50_probe_requires_no_fallback() -> None:
    s = SelectedOutcomeProbeState()
    c = _cfg("control"); c["fallback"]["enabled"] = True
    with pytest.raises(ValueError, match="fallback.enabled=false"):
        apply_selected_outcome_probe(7, _diag(), c, s)


def test_v50_probe_config_is_evidence_collection_only() -> None:
    src = {
        "fallback": {"enabled": True},
        "runtime": {"decisive_frontier_value": {"incumbent_contrastive_extremal_recovery": {
            "enabled": True,
            "selection_conditioned_intervention_recovery": {"post_selection_value_enabled": False},
        }}},
        "metadata": {}, "provenance": {}, "experiment": {},
    }
    c = _make(src, "control"); t = _make(src, "treatment")
    assert c["fallback"]["enabled"] is False and t["fallback"]["enabled"] is False
    assert c["selected_outcome_probe"]["role"] == "control"
    assert t["selected_outcome_probe"]["role"] == "treatment"
    assert c["metadata"]["selected_outcome_probe_role"] == "control"
    assert t["metadata"]["selected_outcome_probe_role"] == "treatment"
    assert src["metadata"] == {}  # config preparation must not mutate the frozen source object
    assert t["selected_outcome_probe"]["proposal_source"] == "first_live_full_set_RSMR_winner_from_frozen_selector_before_post_selection"
    assert t["selected_outcome_probe"]["require_live_qpe"] is True
    assert t["selected_outcome_probe"]["teacher_or_logged_future_inputs"] is False




def _write_probe_diag(path: Path, role: str, *, first_it: int = 7, proposal: int = 23, pre_action: int | None = None) -> None:
    rows = []
    for it in range(first_it + 2):
        first = it == first_it
        pexists = it >= first_it
        baseline = 5
        pre = baseline if pre_action is None else pre_action
        d = {
            "enabled": True,
            "role": role,
            "proposal_exists": pexists,
            "proposal_action": proposal if pexists else -1,
            "baseline_action": baseline if pexists else -1,
            "rsmr_selected_action": proposal if pexists else -1,
            "pre_probe_selected_action": pre if pexists else 4,
            "pre_probe_action": pre if pexists else 4,
            "post_probe_action": (proposal if role == "treatment" and first else baseline) if pexists else 4,
            "first_proposal_now": first,
            "first_proposal_seen": pexists,
            "intervention_executed": bool(role == "treatment" and first),
            "executed_intervention_count": 1 if role == "treatment" and it >= first_it else 0,
            "proposal_event_count": max(0, it - first_it + 1),
        }
        d.update({
            "v50_pre_probe_action_fingerprint": f"pre-{it}-{pre if pexists else 4}",
            "v50_post_probe_action_fingerprint": (
                f"post-{it}-{proposal if role == 'treatment' and first else baseline}" if pexists else f"post-{it}-4"
            ),
        })
        if pexists:
            d.update({
                "live_quality_value": 0.1,
                "live_plan_control_value": 0.2,
                "live_ego_ref_value": 0.3,
                "v50_live_proposal_fingerprint": f"fp-{proposal}",
                "v50_live_proposal_maneuver_id": 0,
                "v50_live_proposal_pool_original_index": 11,
                "v50_live_proposal_maneuver": "keep_follow",
                "v50_live_proposal_theta": "{}",
            })
        rows.append({"iteration_index": it, "time_s": 0.1 * it, "diagnostics": {"selected_outcome_probe": d}})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_v50_pair_validation_accepts_late_live_winner_with_different_offline_slot(tmp_path: Path) -> None:
    c = tmp_path / "c.jsonl"; t = tmp_path / "t.jsonl"
    # The live candidate slot (6) is state-local and need not equal the frozen
    # V49 offline cohort slot (23).  Pair validity is established at the live
    # event by identical timing, live winner, fingerprint, Q/P/E and pretrace.
    _write_probe_diag(c, "control", first_it=7, proposal=6)
    _write_probe_diag(t, "treatment", first_it=7, proposal=6)
    got = _validate_pair("tok", c, t, 23)
    assert got["intervention_iteration"] == 7
    assert got["proposal_action"] == 6
    assert got["offline_v49_action_slot"] == 23
    assert got["live_vs_offline_action_slot_equal"] == 0
    assert got["live_proposal_fingerprint"] == "fp-6"
    assert got["preintervention_pair_aligned"] == 1
    assert got["live_quality_value"] == pytest.approx(0.1)
    assert got["live_plan_control_value"] == pytest.approx(0.2)
    assert got["live_ego_ref_value"] == pytest.approx(0.3)


def test_v50_pair_validation_rejects_live_candidate_or_timing_mismatch(tmp_path: Path) -> None:
    c = tmp_path / "c.jsonl"; t = tmp_path / "t.jsonl"
    _write_probe_diag(c, "control", first_it=7, proposal=6)
    _write_probe_diag(t, "treatment", first_it=7, proposal=7)
    with pytest.raises(RuntimeError, match="paired action identity mismatch"):
        _validate_pair("tok", c, t, 23)
    _write_probe_diag(t, "treatment", first_it=8, proposal=6)
    with pytest.raises(RuntimeError, match="first-proposal iteration mismatch"):
        _validate_pair("tok", c, t, 23)


def _write_no_proposal_diag(path: Path, role: str, *, n: int = 10, action: int = 4) -> None:
    rows = []
    for it in range(n):
        d = {
            "enabled": True,
            "role": role,
            "proposal_exists": False,
            "proposal_action": -1,
            "baseline_action": -1,
            "rsmr_selected_action": -1,
            "pre_probe_selected_action": action,
            "pre_probe_action": action,
            "post_probe_action": action,
            "first_proposal_now": False,
            "first_proposal_seen": False,
            "intervention_executed": False,
            "intervention_consumed": False,
            "executed_intervention_count": 0,
            "proposal_event_count": 0,
            "v50_pre_probe_action_fingerprint": f"same-{it}",
            "v50_post_probe_action_fingerprint": f"same-{it}",
        }
        rows.append({"iteration_index": it, "time_s": 0.1 * it, "diagnostics": {"selected_outcome_probe": d}})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_v50_pair_validation_records_symmetric_no_live_proposal_as_ineligible_not_failure(tmp_path: Path) -> None:
    c = tmp_path / "c.jsonl"; t = tmp_path / "t.jsonl"
    _write_no_proposal_diag(c, "control")
    _write_no_proposal_diag(t, "treatment")
    got = _validate_pair("tok", c, t, 23)
    assert got["collection_protocol_version"] == COLLECTION_PROTOCOL_VERSION
    assert got["pair_status"] == "no_live_proposal"
    assert got["live_intervention_eligible"] == 0
    assert got["proposal_action"] == -1
    assert got["intervention_iteration"] == -1
    assert got["preintervention_pair_aligned"] == 1


def test_v50_pair_validation_rejects_asymmetric_live_eligibility(tmp_path: Path) -> None:
    c = tmp_path / "c.jsonl"; t = tmp_path / "t.jsonl"
    _write_no_proposal_diag(c, "control")
    _write_probe_diag(t, "treatment", first_it=7, proposal=6)
    with pytest.raises(RuntimeError, match="asymmetric/malformed first proposal markers"):
        _validate_pair("tok", c, t, 23)


def test_v50_treatment_must_return_to_incumbent_after_first_live_intervention(tmp_path: Path) -> None:
    c = tmp_path / "c.jsonl"; t = tmp_path / "t.jsonl"
    _write_probe_diag(c, "control", first_it=7, proposal=6)
    _write_probe_diag(t, "treatment", first_it=7, proposal=6)
    rows = [json.loads(x) for x in t.read_text().splitlines() if x.strip()]
    # Corrupt the later proposal event to re-execute the proposal.  The one-shot
    # counter could still remain 1 in a buggy implementation, so validate the
    # full post-intervention state machine explicitly.
    d = rows[-1]["diagnostics"]["selected_outcome_probe"]
    d["post_probe_action"] = d["proposal_action"]
    t.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(RuntimeError, match="did not return to incumbent"):
        _validate_pair("tok", c, t, 23)


def test_v50_paired_hard_metrics_fail_closed_on_schema_drift() -> None:
    control = {m: 1.0 for m in HARD_METRICS}
    treatment = dict(control)
    ok, bad = _hard_noninferiority(control, treatment)
    assert ok is True and bad == []
    treatment[HARD_METRICS[0]] = 0.0
    ok, bad = _hard_noninferiority(control, treatment)
    assert ok is False and bad == [HARD_METRICS[0]]
    missing = dict(treatment); missing.pop(HARD_METRICS[-1])
    with pytest.raises(RuntimeError, match="missing preregistered hard metrics"):
        _hard_noninferiority(control, missing)

def test_v50_live_population_admissibility_uses_frozen_estimator_minima() -> None:
    rows = []
    # 100 events/fold with 40 benefits gives every three-fold fit >=120/180
    # positive/nonpositive and every one-fold calibration >=16 positives.
    for k in range(5):
        for i in range(100):
            rows.append({"outer_test_fold": k, "safe_benefit": i < 40})
    got = _statistical_admissibility(rows)
    assert got["pass"] is True
    # Remove almost all positives from fold 1; it is used as a calibration fold
    # for outer fold 0 and must fail the inherited >=16-positive requirement.
    bad = [dict(r) for r in rows]
    seen = 0
    for r in bad:
        if r["outer_test_fold"] == 1 and r["safe_benefit"]:
            seen += 1
            if seen > 10:
                r["safe_benefit"] = False
    got2 = _statistical_admissibility(bad)
    assert got2["pass"] is False
    assert got2["folds"][0]["cal_safe_benefit"] == 10


def test_v50_fit_hard_locks_v49_preregistered_failure(tmp_path: Path) -> None:
    p = tmp_path / "v49.json"
    p.write_text(json.dumps({"nested_crossfit": {
        "train_gate_pass": False,
        "failure_diagnosis": V49_FAILURE,
        "risk_identification": {
            "aggregate_ego_ref_auc": 0.6298288272330558,
            "aggregate_obs_sign_auc": 0.6139192605594113,
            "aggregate_siir_auc": 0.6081222524597028,
            "siir_better_ego_fold_count": 1,
            "siir_better_obs_fold_count": 3,
            "identified": False,
        },
    }}))
    _check_v49(p)
    bad = json.loads(p.read_text()); bad["nested_crossfit"]["train_gate_pass"] = True
    p.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="V49 preregistered offline-family failure signature changed"):
        _check_v49(p)
    bad = json.loads(p.read_text()); bad["nested_crossfit"]["train_gate_pass"] = False
    bad["nested_crossfit"]["risk_identification"]["identified"] = True
    p.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="SIIR identification-failure semantics changed"):
        _check_v49(p)
    assert ALPHA_RET == pytest.approx(0.0779185520361991)


def test_v50_v49_prerequisite_lock_uses_persisted_flat_risk_identification_schema() -> None:
    p = ROOT / "RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh"
    text = p.read_text()
    assert "aggregate_siir_auc" in text
    assert "aggregate_obs_sign_auc" in text
    assert "aggregate_ego_ref_auc" in text
    assert "aggregate_nonpositive_risk_auc" not in text


def test_v50_launcher_is_closed_loop_evidence_source_and_stops_before_fresh_on_fit_failure() -> None:
    p = ROOT / "RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh"
    if not p.exists():
        pytest.skip("launcher generated after focused unit layer")
    text = p.read_text()
    assert "run_v64_3_50_paired_selected_outcome_collection" in text
    assert "closed_loop_reactive_agents" in text
    assert "fit_v64_3_50_eaf_icer_sior" in text
    assert "[[ $FIT_STATUS -eq 0 ]] || exit \"$FIT_STATUS\"" in text
    assert "select_fresh_preprocessed_tokens" not in text


def test_v50_native_nuplan_layout_accepts_flat_city_db_directories(tmp_path: Path) -> None:
    data_root = tmp_path / "native_nuplan"
    map_root = data_root / "maps"
    split_root = data_root / "nuplan-v1.1" / "splits"
    map_root.mkdir(parents=True)
    (map_root / "nuplan-maps-v1.0.json").write_text("{}")
    db_dirs = []
    for city in ("train_boston", "train_pittsburgh", "train_singapore", "train_vegas"):
        d = split_root / city
        d.mkdir(parents=True)
        (d / f"{city}.db").write_bytes(b"sqlite-placeholder")
        db_dirs.append(d)
    args = type("Args", (), {
        "nuplan_data_root": data_root,
        "nuplan_map_root": map_root,
        "nuplan_exp_root": tmp_path / "exp",
        "nuplan_db_files": db_dirs,
        "nuplan_db_root": None,
    })()
    out = _validate_native_nuplan_inputs(args)
    assert out["mode"] == "db_files"
    assert all(out["direct_db_counts"][str(d)] == 1 for d in db_dirs)
    assert Path(out["exp_root"]).is_dir()


def test_v50_launcher_separates_npz_cache_from_native_closed_loop_db_layout() -> None:
    p = ROOT / "RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh"
    text = p.read_text()
    assert "/data0/senzeyu2/dataset/nuplan/data/cache" in text
    assert "/data0/senzeyu2/dataset/CapPlan/data/nuplan" in text
    assert '"$NUPLAN_SPLITS_ROOT/train_boston"' in text
    assert '"$NUPLAN_SPLITS_ROOT/train_pittsburgh"' in text
    assert '"$NUPLAN_SPLITS_ROOT/train_singapore"' in text
    assert '"$NUPLAN_SPLITS_ROOT/train_vegas"' in text
    assert '--nuplan-exp-root "$NUPLAN_EXP_ROOT" "${NUPLAN_DB_ARGS[@]}" --batch-size "$V50_BATCH_SIZE" --heartbeat-seconds "$V50_HEARTBEAT_SECONDS" --resume' in text
    assert 'NUPLAN_DB_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits/train' not in text


def test_v50_packaging_metadata_restored_for_clean_server_checkout() -> None:
    pyproject = ROOT / "pyproject.toml"
    setup = ROOT / "setup.py"
    assert pyproject.is_file() and setup.is_file()
    ptxt = pyproject.read_text(encoding="utf-8")
    stxt = setup.read_text(encoding="utf-8")
    assert 'name = "bdse-planner"' in ptxt
    assert 'version = "64.3.50"' in ptxt
    assert 'include = ["bdse*"]' in ptxt
    assert "setup()" in stxt


def test_v50_launcher_checks_current_checkout_import_provenance() -> None:
    text = (ROOT / "RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh").read_text(encoding="utf-8")
    assert "PASS V50 import provenance" in text
    assert "python -m pip install -e ." in text


def test_v50_fit_locks_live_selection_pair_protocol_version() -> None:
    assert PAIR_COLLECTION_PROTOCOL_VERSION == COLLECTION_PROTOCOL_VERSION
    text = (ROOT / "bdse/tools/fit_v64_3_50_eaf_icer_sior.py").read_text(encoding="utf-8")
    assert "paired outcome protocol version mismatch" in text
    assert "never mix old and live-eligibility rows" in text



def test_v50_batched_metric_rows_use_scenario_token_as_exact_join(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "run" / "closed_loop_reactive_agents" / "aggregator_metric"
    root.mkdir(parents=True)
    import pandas as pd
    df = pd.DataFrame([
        {"scenario": "tok_a", "score": 0.7, "no_ego_at_fault_collisions": 1.0},
        {"scenario": "tok_b", "score": 0.4, "no_ego_at_fault_collisions": 0.5},
        {"scenario": "some_type", "score": 0.55, "no_ego_at_fault_collisions": 0.75},
        {"scenario": "final_score", "score": 0.55, "no_ego_at_fault_collisions": 0.75},
    ])
    parquet = root / "summary.parquet"
    parquet.write_bytes(b"placeholder")
    monkeypatch.setattr("bdse.tools.run_v64_3_50_paired_selected_outcome_collection.pd.read_parquet", lambda _p: df)
    rows, path = _scenario_metric_rows(tmp_path / "run", ["tok_a", "tok_b"])
    assert path.name == "summary.parquet"
    assert rows["tok_a"]["score"] == pytest.approx(0.7)
    assert rows["tok_b"]["score"] == pytest.approx(0.4)
    assert "final_score" not in rows


def test_v50_batched_probe_rows_are_token_scoped(tmp_path: Path) -> None:
    p = tmp_path / "diag.jsonl"
    rows = []
    for tok in ("tok_a", "tok_b"):
        for it in range(2):
            rows.append({
                "scenario_token": tok,
                "iteration_index": it,
                "time_s": 0.1 * it,
                "diagnostics": {"selected_outcome_probe": {"enabled": True, "role": "control", "proposal_exists": False}},
            })
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    a = _probe_rows(p, "tok_a")
    b = _probe_rows(p, "tok_b")
    assert len(a) == len(b) == 2
    assert {r["scenario_token"] for r in a} == {"tok_a"}
    assert {r["scenario_token"] for r in b} == {"tok_b"}


def test_v50_batched_engine_is_execution_only_and_protocol_is_unchanged() -> None:
    assert COLLECTION_PROTOCOL_VERSION == "v50-live-selected-event-cohort-v1"
    assert COLLECTION_ENGINE_VERSION == "v50-batched-nuplan-lossless-opt-v4"
    text = (ROOT / "bdse/tools/run_v64_3_50_paired_selected_outcome_collection.py").read_text(encoding="utf-8")
    assert '"BDSE_FORCE_REPLAN_EVERY_TICK": "1"' in text
    assert '"BDSE_SHARE_MODEL_PER_PROCESS": "1"' in text
    assert '"BDSE_REQUIRE_SCENARIO_FOR_DIAG": "1"' in text
    assert '"BDSE_SELECTED_OUTCOME_DIAG_ONLY": "1"' in text
    assert '"BDSE_V50_TIMING_TELEMETRY": "1"' in text
    assert '"BDSE_PROFILE_CLOSED_LOOP": "1"' in text
    assert "heartbeat-seconds" in text
    assert 'worker.max_workers=1' in text



def test_v50_token_db_index_maps_lidar_tokens_without_outcome_data(tmp_path: Path) -> None:
    import sqlite3
    toks = ["0011223344556677", "8899aabbccddeeff"]
    db1 = tmp_path / "a.db"; db2 = tmp_path / "b.db"
    for db, tok in ((db1, toks[0]), (db2, toks[1])):
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE lidar_pc (token BLOB PRIMARY KEY)")
        con.execute("INSERT INTO lidar_pc(token) VALUES (?)", (bytes.fromhex(tok),))
        con.commit(); con.close()
    cache = tmp_path / "index.json"
    got = _build_token_db_index(toks, [db1, db2], cache)
    assert got == {toks[0]: str(db1), toks[1]: str(db2)}
    got2 = _build_token_db_index(toks, [db1, db2], cache)
    assert got2 == got


def test_v50_batched_resume_locks_requested_batch_size() -> None:
    text = (ROOT / "bdse/tools/run_v64_3_50_paired_selected_outcome_collection.py").read_text(encoding="utf-8")
    assert '"requested_batch_size": batch_size' in text
    assert 'different requested batch size' in text
    assert 'V50 paired output predates the frozen batched-engine provenance field requested_batch_size' in text


def test_v50_progress_tracker_extracts_compact_tick_timing(tmp_path: Path) -> None:
    diag = tmp_path / "diag.jsonl"
    log = tmp_path / "run.log"
    telemetry = tmp_path / "timing.jsonl"
    row = {
        "scenario_token": "tok_a",
        "iteration_index": 7,
        "time_s": 0.7,
        "diagnostics": {
            "selected_outcome_probe": {"enabled": True, "proposal_exists": False},
            "v50_timing": {
                "timing": {"compute_planner_trajectory_total_s": 0.5, "core_plan_s": 0.4},
                "timing_core": {"candidate_generation_s": 0.1, "certificate_stages_s": 0.2},
                "model_timing": {"model_encode_context_s": 0.05},
            },
        },
    }
    diag.write_text(json.dumps(row) + "\n", encoding="utf-8")
    log.write_text("[planner-ready] reused=True\n", encoding="utf-8")
    import time
    tr = _BatchProgressTracker(role="control", diag=diag, log=log, telemetry=telemetry, started=time.perf_counter())
    snap = tr.snapshot(pid=123)
    assert snap["diag_rows"] == 1
    assert snap["scenarios_seen"] == 1
    assert snap["latest_token"] == "tok_a"
    assert snap["latest_iteration"] == 7
    assert snap["planner_ready_count"] == 1
    assert snap["timing_mean_s"]["timing.compute_planner_trajectory_total_s"] == pytest.approx(0.5)
    assert snap["timing_mean_s"]["core.candidate_generation_s"] == pytest.approx(0.1)
    assert snap["timing_mean_s"]["model.model_encode_context_s"] == pytest.approx(0.05)
    assert telemetry.is_file() and telemetry.stat().st_size > 0


def test_v50_diag_only_profile_keeps_probe_and_compact_timing_only() -> None:
    text = (ROOT / "bdse/planner/nuplan_planner.py").read_text(encoding="utf-8")
    assert 'diag_payload["v50_timing"]' in text
    assert '"selected_outcome_probe": (diagnostics.get("selected_outcome_probe", {}) or {})' in text
    assert '_append_diag_line(path, json.dumps(row, sort_keys=True))' in text
    assert 'timing_core["v50_probe_instrumentation_s"]' in text
    assert 'identity_cache: dict[int, dict[str, Any]]' in text


def test_v50_persistent_diag_writer_is_line_visible(tmp_path: Path) -> None:
    from bdse.planner.nuplan_planner import _append_diag_line
    p = tmp_path / "live.jsonl"
    _append_diag_line(p, '{"x":1}')
    assert p.read_text(encoding="utf-8") == '{"x":1}\n'


def test_v50_packed_runtime_batch_transfer_is_bit_exact_on_all_dtype_shapes() -> None:
    arrays = {
        "f": np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 7.0,
        "f32": np.arange(5, dtype=np.float32),
        "i16": np.asarray([[1, -2], [3, 4]], dtype=np.int16),
        "i64": np.asarray([5, 6, 7], dtype=np.int64),
        "b": np.asarray([[True, False], [False, True]], dtype=bool),
        "scalar": np.asarray(True),
        "empty": np.zeros((0, 3), dtype=np.float32),
    }
    legacy = BDSEModel._runtime_arrays_to_torch_batch(arrays, torch.device("cpu"), packed_transfer=False)
    packed = BDSEModel._runtime_arrays_to_torch_batch(arrays, torch.device("cpu"), packed_transfer=True)
    assert list(legacy) == list(packed) == list(arrays)
    for key in arrays:
        assert legacy[key].dtype == packed[key].dtype
        assert legacy[key].shape == packed[key].shape
        assert legacy[key].stride() == packed[key].stride()
        assert legacy[key].is_contiguous() and packed[key].is_contiguous()
        assert torch.equal(legacy[key], packed[key])


def _v50_ocrr_sc_for_projection() -> dict:
    raw_names = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    feature_names = [f"delta::{n}" for n in raw_names] + ["delta::support_logit"]
    obs_names = VALUE_OBSERVABLE_NAMES + FSFR_OBSERVABLE_NAMES
    return {
        "feature_names": feature_names,
        "feature_mean": [0.0] * 19,
        "feature_std": [1.0] * 19,
        "weights": [0.0] * 19,
        "bias": 0.0,
        "base_feature_names": raw_names,
        "post_selection_value_enabled": True,
        "post_selection_value_mode": "endpoint_potential_quality_operator_conditioned_risk_retention",
        "post_selection_endpoint_feature_names": EPV_NAMES,
        "post_selection_endpoint_feature_scale": [1.0] * 38,
        "post_selection_endpoint_weights": [0.0] * 38,
        "post_selection_endpoint_bias": 0.0,
        "post_selection_observable_names": obs_names,
        "post_selection_quality_observable_names": ["route_deviation_cost", "progress_deficit_cost", "global_comfort_cost"],
        "post_selection_quality_observable_scale": [1.0] * 3,
        "post_selection_quality_observable_weights": [0.3, -0.2, 0.1],
        "selected_policy_risk_plan_response_names": ["fsfr_plan_1d_occupancy_cost"],
        "selected_policy_risk_plan_response_scales": [1.0],
        "selected_policy_risk_plan_response_weights": [0.4],
        "selected_policy_risk_ego_reference_names": ["fsfr_plan_1d_occupancy_cost", "fsfr_predicted_demo_cost"],
        "selected_policy_risk_ego_reference_scales": [1.0, 1.0],
        "selected_policy_risk_ego_reference_weights": [0.4, 0.2],
        "operator_conditioned_risk_retention": {
            "feature_names": ["quality_value", "prospective_response_increment", "ego_reference_increment", "log_extremal_multiplicity"],
            "aggregation": "sign_only",
            "use_extremal_multiplicity": False,
            "components": {
                "sign_risk": {
                    "model": "zero_bias_pairwise_selected_sign_risk",
                    "feature_names": ["quality_value", "prospective_response_increment", "ego_reference_increment", "log_extremal_multiplicity"],
                    "feature_mean": [0.0] * 4,
                    "feature_std": [1.0] * 4,
                    "weights": [-0.2, -0.3, -0.4, 0.0],
                    "bias": 0.0,
                    "lambda": 1.0,
                    "fit_positive_score_mean": 0.0,
                    "fit_positive_score_std": 1.0,
                }
            },
            "retention_threshold": 0.7,
        },
    }


def test_v50_fsfr_projection_exactly_reuses_full_plan1d_and_egoref_columns() -> None:
    rt, bank, cfg = _v47_runtime(), _v47_bank(), _v47_cfg()
    full, names = runtime_future_state_factorization_observable_costs(rt, bank, cfg)
    requested = ["fsfr_plan_1d_occupancy_cost", "fsfr_predicted_demo_cost"]
    projected, pnames = runtime_future_state_factorization_observable_costs(rt, bank, cfg, requested_names=requested)
    assert pnames == requested
    assert np.array_equal(projected[:, 0], full[:, names.index(requested[0])])
    assert np.array_equal(projected[:, 1], full[:, names.index(requested[1])])


def test_v50_eliding_closed_plan2d_keeps_frozen_ocrr_certificate_bit_exact() -> None:
    sc = _v50_ocrr_sc_for_projection()
    full_names = list(sc["post_selection_observable_names"])
    full = np.arange(2 * len(full_names), dtype=np.float64).reshape(2, len(full_names)) / 17.0
    elided = full.copy()
    elided[:, full_names.index("fsfr_plan_2d_occupancy_cost")] = 0.0
    raw_names = list(_ICER_REGRET_RISK_EVIDENCE_BASE_NAMES)
    feature_names = list(sc["feature_names"])
    raw = np.zeros((2, len(raw_names)), dtype=np.float64)
    sup = np.zeros(2, dtype=np.float64)
    X = np.zeros((2, 19), dtype=np.float64)
    mu = np.zeros(2, dtype=np.float64)
    vf, ff, nf = _icer_post_selection_value(
        1, mu, X, feature_names, sc, raw_feat=raw, raw_feature_names=raw_names, support_logits=sup,
        legacy_action=0, value_observable_matrix=full, value_observable_names=full_names, selection_multiplicity=9,
    )
    ve, fe, ne = _icer_post_selection_value(
        1, mu, X, feature_names, sc, raw_feat=raw, raw_feature_names=raw_names, support_logits=sup,
        legacy_action=0, value_observable_matrix=elided, value_observable_names=full_names, selection_multiplicity=9,
    )
    assert ve == pytest.approx(vf, abs=0.0, rel=0.0)
    assert np.array_equal(fe, ff)
    assert ne == nf


def test_v50_runtime_elide_fsfr2d_matches_full_observables_on_all_consumed_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    rt, bank, cfg = _v47_runtime(), _v47_bank(), _v47_cfg()
    ic = cfg["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
    sc = ic["selection_conditioned_intervention_recovery"]
    sc.update({
        "post_selection_value_mode": "endpoint_potential_quality_operator_conditioned_risk_retention",
        "post_selection_observable_names": list(VALUE_OBSERVABLE_NAMES) + list(FSFR_OBSERVABLE_NAMES),
        "post_selection_quality_observable_names": ["route_deviation_cost", "progress_deficit_cost", "global_comfort_cost"],
        "selected_policy_risk_plan_response_names": ["fsfr_plan_1d_occupancy_cost"],
        "selected_policy_risk_ego_reference_names": ["fsfr_plan_1d_occupancy_cost", "fsfr_predicted_demo_cost"],
    })
    monkeypatch.delenv("BDSE_V50_ELIDE_UNUSED_FSFR_2D", raising=False)
    full, full_names = runtime_value_observable_costs(rt, bank, cfg)
    monkeypatch.setenv("BDSE_V50_ELIDE_UNUSED_FSFR_2D", "1")
    elided, names = runtime_value_observable_costs(rt, bank, cfg)
    assert names == full_names
    assert elided.shape == full.shape
    dead = "fsfr_plan_2d_occupancy_cost"
    dead_i = full_names.index(dead)
    for i, name in enumerate(names):
        if name == dead:
            assert np.array_equal(elided[:, i], np.zeros((len(elided),), dtype=np.float64))
        else:
            assert np.array_equal(elided[:, i], full[:, i]), name
    assert np.any(full[:, dead_i] != 0.0)
