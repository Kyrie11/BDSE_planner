import numpy as np

from bdse.data.cache_schema import CandidateBank, RuntimeFeatures
from bdse.experiments.train import _validation_fixed_budget_critical_score
from bdse.planner import fallback as fb


def _bank() -> CandidateBank:
    t = np.linspace(0.0, 4.0, 9, dtype=np.float32)
    traj = np.zeros((3, len(t), 5), dtype=np.float32)
    traj[0, :, 0] = 5.0 * t  # highest progress, unsafe TTC
    traj[1, :, 0] = 3.0 * t
    traj[2, :, 0] = 2.0 * t
    traj[:, :, 3] = np.asarray([[5.0], [3.0], [2.0]], dtype=np.float32)
    traj[:, :, 4] = t
    return CandidateBank(
        trajectories=traj,
        valid_mask=np.ones((3,), dtype=bool),
        maneuver_ids=np.zeros((3,), dtype=np.int64),
        theta=[{}, {}, {}],
        dynamic_flags=[{}, {}, {}],
        metadata=[{}, {}, {}],
    )


def _runtime() -> RuntimeFeatures:
    return RuntimeFeatures(
        ego_history=np.zeros((1, 5), dtype=np.float32),
        agent_history=np.zeros((0, 1, 10), dtype=np.float32),
        agent_valid=np.zeros((0,), dtype=bool),
        current_agents=np.zeros((0, 10), dtype=np.float32),
        traffic_lights=[],
        map_features={
            "route_centerline": np.asarray([[0.0, 0.0], [40.0, 0.0]], dtype=np.float32),
            "route_corridor_centerlines": [
                np.asarray([[0.0, 0.0], [40.0, 0.0]], dtype=np.float32),
                np.asarray([[0.0, 10.0], [40.0, 10.0]], dtype=np.float32),
            ],
            "route_corridor_width": 2.0,
            "stop_lines": [],
        },
        route_roadblock_ids=[],
        mission_goal=None,
    )


def _cfg() -> dict:
    return {
        "fallback": {
            "safe_progress_recovery": {
                "viability_frontier": {
                    "min_pool": 1,
                    "max_pool": 3,
                    "absolute_ttc_floor_s": 1.5,
                    "absolute_agent_overlap_cap": 0.02,
                    "relative_ttc_slack_s": 0.35,
                    "certificate_guard_epsilon_norm": 0.35,
                    "joint_viability_epsilon_norm": 0.2,
                    "progress_weight": 1.0,
                    "path_length_weight": 0.0,
                    "certificate_weight": 0.2,
                    "lateral_weight": 0.0,
                    "agent_risk_weight": 0.5,
                    "ttc_risk_weight": 0.6,
                    "offroute_risk_weight": 0.2,
                    "soft_risk_weight": 0.0,
                    "hard_risk_weight": 0.0,
                    "low_speed_penalty": 0.0,
                }
            }
        }
    }


def test_route_graph_distance_accepts_valid_intersection_branch():
    runtime = _runtime()
    points = np.asarray([[5.0, 10.0], [20.0, 10.0]], dtype=np.float32)
    base = runtime.map_features["route_centerline"]
    distance = fb._route_graph_distance(points, runtime.map_features, base)
    assert np.all(distance < 1e-5)


def test_absolute_ttc_barrier_prevents_progress_from_overriding_safety(monkeypatch):
    bank = _bank()
    runtime = _runtime()

    def fake_risks(*_args, **_kwargs):
        return {
            "hard": np.asarray([2.0, 0.1, 0.2], dtype=np.float32),
            "soft": np.zeros((3,), dtype=np.float32),
            "hard_agent": np.asarray([0.4, 0.0, 0.0], dtype=np.float32),
            "soft_agent": np.zeros((3,), dtype=np.float32),
            "agent_ttc": np.asarray([0.9, 0.2, 0.1], dtype=np.float32),
            "min_ttc_s": np.asarray([0.55, 2.2, 3.0], dtype=np.float32),
            "hard_horizon_s": np.full((3,), 4.0, dtype=np.float32),
            "hard_off_route": np.zeros((3,), dtype=np.float32),
            "soft_off_route": np.zeros((3,), dtype=np.float32),
            "red_light": np.zeros((3,), dtype=np.float32),
        }

    monkeypatch.setattr(fb, "runtime_risk_scores", fake_risks)
    decision = fb.viability_frontier_recovery_action(
        bank,
        safety_flags=np.ones((3,), dtype=bool),
        cfg=_cfg(),
        runtime=runtime,
        tournament_scores=np.asarray([10.0, 0.0, -0.1], dtype=np.float32),
        reference_action=0,
    )
    assert decision.action_index in {1, 2}
    assert decision.diagnostics["absolute_barrier_applied"] is True
    assert decision.diagnostics["selected_min_ttc_s"] >= 1.5


def test_adaptive_horizon_is_active_at_moderate_high_speed():
    bank = _bank()
    rsc = {
        "hard_check_horizon_s": 4.0,
        "soft_check_horizon_s": 6.0,
        "speed_adaptive_horizon": True,
        "min_hard_horizon_s": 4.0,
        "max_hard_horizon_s": 6.5,
        "reaction_time_s": 0.9,
        "comfortable_emergency_decel_mps2": 3.0,
        "stopping_horizon_margin_s": 0.55,
    }
    bank.trajectories[0, :, 3] = 10.0
    _, _, _, hard_h, _ = fb._candidate_safety_time_masks(bank.trajectories, rsc, 0.1)
    assert hard_h[0] > 4.5
    assert hard_h[2] == 4.0


def test_checkpoint_score_rejects_hard_recall_regression():
    safe = {
        "val_budget_vs_full_match": 0.171,
        "val_teacher_action_match": 0.225,
        "val_selected_interaction_decisive_recall": 0.33,
        "val_selected_hard_decisive_recall": 0.64,
        "val_pair_sign_acc_near_tie": 0.58,
        "val_evidence_sufficiency": 0.4,
        "val_fallback_would_trigger_rate": 0.01,
        "val_teacher_regret": 13000.0,
        "val_effective_query_count": 8500.0,
        "val_total_sparse_query_count": 33000.0,
    }
    unsafe = dict(safe)
    unsafe.update({
        "val_budget_vs_full_match": 0.176,
        "val_teacher_action_match": 0.232,
        "val_selected_hard_decisive_recall": 0.47,
        "val_fallback_would_trigger_rate": 0.046,
    })
    assert _validation_fixed_budget_critical_score(safe) > _validation_fixed_budget_critical_score(unsafe)


def test_closed_loop_packager_selects_only_requested_artifacts(tmp_path):
    from tools.package_closed_loop_results import discover_artifacts

    challenge = tmp_path / "run_a" / "simulation" / "closed_loop_nonreactive_agents"
    (challenge / "aggregator_metric").mkdir(parents=True)
    (challenge / "metrics").mkdir()
    (challenge / "aggregator_metric" / "summary.parquet").write_bytes(b"a")
    (challenge / "metrics" / "metric.parquet").write_bytes(b"b")
    (challenge / "runner_report.parquet").write_bytes(b"c")
    (challenge / "nuboard_abc.nuboard").write_bytes(b"d")
    (challenge / "unrelated.txt").write_bytes(b"x")
    files, warnings = discover_artifacts(tmp_path)
    assert not warnings
    names = {path.name for path in files}
    assert names == {"summary.parquet", "metric.parquet", "runner_report.parquet", "nuboard_abc.nuboard"}
