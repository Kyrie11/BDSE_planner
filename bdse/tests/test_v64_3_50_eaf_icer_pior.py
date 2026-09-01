from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from bdse.planner.nuplan_planner import BDSEPlannerCore
from bdse.tools.build_v64_3_50_pior_train_manifest import CITY_TO_RAW, _stable_log_name, _resolve_raw_db_files
from bdse.tools.run_v64_3_50_pior_paired_closed_loop import (
    _pair, SAFETY_METRICS, _batch_certificate_valid, _batch_raw_files, _validate_probe_events, _collision_safe_batches
)

ROOT = Path(__file__).resolve().parents[2]


def _tournament(proposal: int = 3, baseline: int = 1, exists: bool = True, action: int = 1):
    return SimpleNamespace(
        action_index=action,
        diagnostics={
            "decisive_frontier_icer_scir_proposal_exists": float(exists),
            "decisive_frontier_icer_scir_proposal_action": float(proposal),
            "decisive_frontier_icer_baseline_action": float(baseline),
        },
    )


def _candidates():
    return SimpleNamespace(K=5, valid_mask=np.asarray([True, True, True, True, True], dtype=bool))


def _cfg(arm: str | None, *, fallback: bool = False):
    c = {"fallback": {"enabled": fallback, "rule_rerank_top_k": 0}}
    if arm is not None:
        c["selected_outcome_probe"] = {"enabled": True, "arm": arm, "one_shot": True}
    return c


def test_v50_probe_is_semantic_noop_when_disabled() -> None:
    core = BDSEPlannerCore(cfg=_cfg(None))
    action, diag = core._apply_selected_outcome_probe(_tournament(action=4), 4, _candidates())
    assert action == 4
    assert diag["pior_probe_enabled"] is False
    assert core._pior_probe_used is False


def test_v50_treatment_executes_exact_frozen_proposal_once_then_incumbent() -> None:
    core = BDSEPlannerCore(cfg=_cfg("treatment"))
    action, d1 = core._apply_selected_outcome_probe(_tournament(proposal=3, baseline=1, action=1), 1, _candidates())
    assert action == 3
    assert d1["pior_probe_fired"] is True
    assert d1["pior_probe_contract_same_frozen_proposal_or_incumbent"] is True
    action2, d2 = core._apply_selected_outcome_probe(_tournament(proposal=4, baseline=1, action=4), 4, _candidates())
    assert action2 == 1
    assert d2["pior_probe_fired"] is False
    assert d2["pior_probe_phase"] == "post_intervention_incumbent"
    assert core._pior_probe_event_count == 1


def test_v50_control_marks_same_event_but_never_leaves_incumbent() -> None:
    core = BDSEPlannerCore(cfg=_cfg("control"))
    action, d = core._apply_selected_outcome_probe(_tournament(proposal=3, baseline=1, action=3), 3, _candidates())
    assert action == 1
    assert d["pior_probe_fired"] is True
    assert d["pior_probe_proposal_action"] == 3
    assert d["pior_probe_baseline_action"] == 1



def test_v50_manifest_bound_probe_fires_exact_v49_action_at_iteration0_even_when_online_proposal_absent(tmp_path, monkeypatch) -> None:
    import json
    ts = 1_629_745_157_950_134
    target = tmp_path / "targets.json"
    target.write_text(json.dumps({"targets": [{
        "scenario_token": "tok", "timestamp_us": ts, "full_selected_action": 3,
    }]}))
    monkeypatch.setenv("BDSE_PIOR_TARGETS_FILE", str(target))
    cfg = _cfg("treatment")
    cfg["selected_outcome_probe"]["proposal_source"] = "preregistered_V49_manifest_iteration0_proposal"
    core = BDSEPlannerCore(cfg=cfg)
    current = SimpleNamespace(iteration=SimpleNamespace(index=0, time_s=ts / 1e6))
    action, d = core._apply_selected_outcome_probe(
        _tournament(proposal=-1, baseline=1, exists=False, action=1), 1, _candidates(), current
    )
    assert action == 3
    assert d["pior_probe_fired"] is True
    assert d["pior_probe_scenario_token"] == "tok"
    assert d["pior_probe_proposal_action"] == 3
    assert d["pior_probe_online_proposal_exists"] is False
    assert d["pior_probe_online_proposal_matches_target"] is False
    assert d["pior_probe_target_timestamp_us"] == ts
    assert abs(d["pior_probe_current_timestamp_us"] - ts) <= 4


def test_v50_manifest_bound_probe_refuses_nonzero_first_iteration(tmp_path, monkeypatch) -> None:
    import json
    ts = 1_629_745_157_950_134
    target = tmp_path / "targets.json"
    target.write_text(json.dumps({"targets": [{
        "scenario_token": "tok", "timestamp_us": ts, "full_selected_action": 3,
    }]}))
    monkeypatch.setenv("BDSE_PIOR_TARGETS_FILE", str(target))
    cfg = _cfg("control")
    cfg["selected_outcome_probe"]["proposal_source"] = "preregistered_V49_manifest_iteration0_proposal"
    core = BDSEPlannerCore(cfg=cfg)
    current = SimpleNamespace(iteration=SimpleNamespace(index=5, time_s=ts / 1e6))
    with pytest.raises(RuntimeError, match="iteration 0"):
        core._apply_selected_outcome_probe(_tournament(baseline=1), 1, _candidates(), current)


def test_v50_probe_refuses_fallback_or_second_path() -> None:
    core = BDSEPlannerCore(cfg=_cfg("treatment", fallback=True))
    with pytest.raises(RuntimeError, match="fallback.enabled=false"):
        core._apply_selected_outcome_probe(_tournament(), 1, _candidates())


def test_v50_dataset_contract_matches_user_train_and_raw_db_layout() -> None:
    assert CITY_TO_RAW == {
        "train_boston": "train_boston",
        "train_pittsburgh": "train_pittsburgh",
        "train_singapore": "train_singapore",
        "train_vegas_2": "train_vegas",
    }


def test_v50_paired_outcome_is_positive_only_when_score_improves_without_hard_safety_degradation() -> None:
    metrics = {"score": 0.7, **{k: 1.0 for k in SAFETY_METRICS}}
    control = {"a": {"identity": {}, "metrics": metrics}}
    treat_good = {"a": {"identity": {}, "metrics": {**metrics, "score": 0.8}}}
    row = _pair(control, treat_good, ["a"])[0]
    assert row["closed_loop_beneficial"] is True
    assert row["pior_interventional_outcome"] == 1.0
    treat_harm = {"a": {"identity": {}, "metrics": {**metrics, "score": 0.9, "time_to_collision_within_bound": 0.0}}}
    row2 = _pair(control, treat_harm, ["a"])[0]
    assert row2["closed_loop_hard_harm"] is True
    assert row2["closed_loop_beneficial"] is False
    assert row2["pior_interventional_outcome"] == -1.0


def test_v50_launcher_changes_evidence_source_and_stops_before_untouched_validation() -> None:
    text = (ROOT / "RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh").read_text()
    assert "v64_3_49_siir_fit.json" in text
    assert "805c4f8088051413edeb568623bc6d225d1b3c301c52612f89109216b38be296" in text
    assert "bdse_train_v2" in text
    assert "train_vegas_2" in text and "train_vegas" in text
    assert "run_v64_3_50_pior_paired_closed_loop" in text
    assert "closed_loop_nonreactive_agents" in text
    assert "STOP before untouched closed-loop validation" in text or "do not consume untouched closed-loop validation" in text
    assert "A/B pooling" not in text  # no offline fresh-rescue stage exists in V50 launcher


def test_v50_probe_only_diag_is_minimal_and_event_only(tmp_path, monkeypatch) -> None:
    from bdse.planner.nuplan_planner import BDSEnuPlanPlanner

    diag_path = tmp_path / "probe.jsonl"
    monkeypatch.setenv("BDSE_CLOSED_LOOP_DIAG", str(diag_path))
    monkeypatch.setenv("BDSE_CLOSED_LOOP_DIAG_MODE", "pior_probe_events")
    fake_planner = SimpleNamespace(_name="BDSEPlanner")
    current = SimpleNamespace(iteration=SimpleNamespace(index=7, time_s=0.7))
    disabled = {"tournament": {"pior_probe_fired": False}}
    BDSEnuPlanPlanner._write_closed_loop_diag(fake_planner, current, 1, disabled)
    assert not diag_path.exists()
    fired = {
        "tournament": {
            "pior_probe_fired": True,
            "pior_probe_arm": "treatment",
            "pior_probe_event_count": 1,
            "pior_probe_scenario_token": "tok",
            "pior_probe_target_timestamp_us": 700000,
            "pior_probe_current_timestamp_us": 700000,
            "pior_probe_timestamp_error_us": 0,
            "pior_probe_target_source": "preregistered_V49_manifest_iteration0_proposal",
            "pior_probe_online_proposal_exists": False,
            "pior_probe_online_proposal_action": -1,
            "pior_probe_online_proposal_matches_target": False,
            "pior_probe_baseline_action": 1,
            "pior_probe_proposal_action": 3,
            "pior_probe_final_action": 3,
            "pior_probe_contract_same_frozen_proposal_or_incumbent": True,
            "pior_probe_contract_no_rerank_second_best_fallback": True,
            "large_unused_payload": list(range(1000)),
        }
    }
    BDSEnuPlanPlanner._write_closed_loop_diag(fake_planner, current, 3, fired)
    rows = [__import__("json").loads(x) for x in diag_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["pior_probe_fired"] is True
    assert rows[0]["pior_probe_proposal_action"] == 3
    assert rows[0]["scenario_token"] == "tok"
    assert rows[0]["target_timestamp_us"] == 700000
    assert "diagnostics" not in rows[0]
    assert "large_unused_payload" not in rows[0]


def test_v50_optimized_launcher_has_exact_db_resume_and_ticks() -> None:
    text = (ROOT / "RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh").read_text()
    assert "PIOR_BATCH_SIZE" in text
    assert "PIOR_HEARTBEAT_SECONDS" in text
    assert "PIOR_RESUME" in text
    assert "PIOR_SERIALIZE_GPU_INFERENCE" in text
    assert "--resume" in text and "--allow-legacy-full-arm-resume" in text
    runner = (ROOT / "bdse/tools/run_v64_3_50_pior_paired_closed_loop.py").read_text()
    assert "[PIOR-TICK]" in runner
    assert "BDSE_CLOSED_LOOP_DIAG_MODE" in runner
    assert '"pior_probe_events"' in runner
    assert '"BDSE_SERIALIZE_GPU_INFERENCE": "1" if serialize_gpu_inference else "0"' in runner
    manifest = (ROOT / "bdse/tools/build_v64_3_50_pior_train_manifest.py").read_text()
    assert '_stable_log_name' in manifest
    assert 'row["raw_db_files"]' in manifest
    assert '"raw_db_files": raw_files' in manifest
    assert 'city_split_fallback' in manifest


def test_v50_resume_certificate_requires_exact_hashes(tmp_path) -> None:
    import hashlib, json
    root = tmp_path / "batch_0000"
    root.mkdir()
    tokens = ["a", "b"]
    metrics = root / "scenario_metrics.jsonl"
    metrics.write_text(
        "\n".join(json.dumps({"scenario_token": t, "identity": {}, "metrics": {"score": 1.0}}, sort_keys=True) for t in tokens) + "\n"
    )
    diag = root / "pior_probe_events.jsonl"
    diag.write_text(
        "\n".join(json.dumps({"pior_probe_fired": True, "pior_probe_arm": "control"}, sort_keys=True) for _ in tokens) + "\n"
    )
    (root / "pior_probe_targets.json").write_text(json.dumps({"targets": []}) + "\n")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    token_sha = hashlib.sha256(("\n".join(tokens) + "\n").encode()).hexdigest()
    cert = {
        "complete": True, "scenario_count": 2, "scenario_token_sha256": token_sha,
        "config_sha256": "cfg", "checkpoint_sha256": "ckpt",
        "challenge": "closed_loop_nonreactive_agents", "successful": 2, "failed": 0,
        "raw_db_file_list_sha256": "dbhash",
        "probe_fired_count": 2, "scenario_metrics_sha256": sha(metrics),
        "probe_events_sha256": sha(diag),
    }
    (root / ".pior_batch_complete.json").write_text(json.dumps(cert))
    got, _ = _batch_certificate_valid(
        root=root, tokens=tokens, cfg_sha="cfg", ckpt_sha="ckpt", challenge="closed_loop_nonreactive_agents",
        raw_db_file_list_sha256="dbhash",
    )
    assert got is not None and set(got) == set(tokens)
    # A different raw-DB subset invalidates reuse even if every result file is unchanged.
    got_db, cert_db = _batch_certificate_valid(
        root=root, tokens=tokens, cfg_sha="cfg", ckpt_sha="ckpt", challenge="closed_loop_nonreactive_agents",
        raw_db_file_list_sha256="different-dbhash",
    )
    assert got_db is None and cert_db is None
    # Any metric mutation invalidates the batch rather than silently reusing it.
    metrics.write_text(metrics.read_text() + " ")
    got2, cert2 = _batch_certificate_valid(
        root=root, tokens=tokens, cfg_sha="cfg", ckpt_sha="ckpt", challenge="closed_loop_nonreactive_agents",
        raw_db_file_list_sha256="dbhash",
    )
    assert got2 is None and cert2 is None


def test_v50_raw_db_resolution_understands_nuplan_crop_suffix_and_exact_token(tmp_path: Path) -> None:
    import sqlite3

    token = "00f4aedf9b3c5f65"
    log = "2021.08.23.18.41.38_veh-28"
    a = tmp_path / f"{log}_00001_00099.db"
    b = tmp_path / f"{log}_00100_00199.db"
    for db in (a, b):
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE lidar_pc (token BLOB PRIMARY KEY)")
        conn.commit(); conn.close()
    conn = sqlite3.connect(b)
    conn.execute("INSERT INTO lidar_pc(token) VALUES (?)", (bytes.fromhex(token),))
    conn.commit(); conn.close()

    assert _stable_log_name(a.name) == log
    idx = {
        "train_boston": {
            "all": [a, b],
            "by_stem": {a.stem: a, b.stem: b},
            "by_stable": {log: [a, b]},
        }
    }
    row = {"raw_db_split": "train_boston", "log_name": log, "npz_path": str(tmp_path / log / "x.npz")}
    got, mode = _resolve_raw_db_files(row=row, token=token, db_index=idx)
    assert got == [b]
    assert mode == "stable_log_sqlite_token_exact"


def test_v50_raw_db_resolution_falls_back_without_changing_token_population(tmp_path: Path) -> None:
    a = tmp_path / "unrelated_a.db"; a.write_bytes(b"")
    b = tmp_path / "unrelated_b.db"; b.write_bytes(b"")
    idx = {
        "train_boston": {
            "all": [a, b],
            "by_stem": {a.stem: a, b.stem: b},
            "by_stable": {a.stem: [a], b.stem: [b]},
        }
    }
    row = {"raw_db_split": "train_boston", "log_name": "2021.08.23.18.41.38_veh-28", "npz_path": str(tmp_path / "log" / "x.npz")}
    got, mode = _resolve_raw_db_files(row=row, token="00f4aedf9b3c5f65", db_index=idx)
    assert got == [a, b]
    assert mode == "city_split_fallback"

    meta = {
        "tok": {"raw_db_files": [str(a), str(b)]},
        "tok2": {"raw_db_files": [str(b)]},
    }
    assert _batch_raw_files(["tok", "tok2"], meta) == [str(a), str(b)]


def test_v50_probe_event_validation_is_token_bound(tmp_path: Path) -> None:
    import json
    tokens = ["a", "b"]
    meta = {
        "a": {"timestamp_us": 1000000, "cache_iteration": 0, "full_selected_action": 3},
        "b": {"timestamp_us": 2000000, "cache_iteration": 0, "full_selected_action": 4},
    }
    path = tmp_path / "events.jsonl"
    rows = []
    for tok, base in [("a", 1), ("b", 2)]:
        m = meta[tok]
        rows.append({
            "scenario_token": tok, "iteration_index": 0,
            "pior_probe_fired": True, "pior_probe_event_count": 1, "pior_probe_arm": "treatment",
            "pior_probe_target_source": "preregistered_V49_manifest_iteration0_proposal",
            "target_timestamp_us": m["timestamp_us"], "current_timestamp_us": m["timestamp_us"],
            "pior_probe_proposal_action": m["full_selected_action"], "pior_probe_baseline_action": base,
            "pior_probe_final_action": m["full_selected_action"],
            "pior_probe_contract_same_frozen_proposal_or_incumbent": True,
            "pior_probe_contract_no_rerank_second_best_fallback": True,
            "pior_probe_online_proposal_exists": False, "pior_probe_online_proposal_matches_target": False,
        })
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got, audit = _validate_probe_events(path, tokens=tokens, meta=meta, arm="treatment")
    assert set(got) == set(tokens)
    assert audit["online_proposal_matches_frozen_target_count"] == 0
    rows[1]["scenario_token"] = "a"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(RuntimeError, match="duplicate probe event"):
        _validate_probe_events(path, tokens=tokens, meta=meta, arm="treatment")


def test_v50_collision_safe_batches_separate_equal_start_timestamps() -> None:
    tokens = ["a", "b", "c", "d"]
    meta = {
        "a": {"timestamp_us": 1},
        "b": {"timestamp_us": 2},
        "c": {"timestamp_us": 1},
        "d": {"timestamp_us": 3},
    }
    batches = _collision_safe_batches(tokens, meta, 4)
    assert batches == [["a", "b"], ["c", "d"]]
    for batch in batches:
        assert len({meta[t]["timestamp_us"] for t in batch}) == len(batch)


def test_v50_first_batch_is_small_paired_preflight_without_changing_population() -> None:
    tokens = [f"t{i}" for i in range(10)]
    meta = {t: {"timestamp_us": i + 1} for i, t in enumerate(tokens)}
    batches = _collision_safe_batches(tokens, meta, batch_size=6, first_batch_size=3)
    assert batches == [tokens[:3], tokens[3:9], tokens[9:]]
    assert [t for batch in batches for t in batch] == tokens
    text = (ROOT / "RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh").read_text()
    assert 'PIOR_FIRST_BATCH_SIZE="${PIOR_FIRST_BATCH_SIZE:-4}"' in text
    assert '--first-batch-size "$PIOR_FIRST_BATCH_SIZE"' in text
