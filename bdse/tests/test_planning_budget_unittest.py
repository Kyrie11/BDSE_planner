"""Run with python -m unittest; torch/nuPlan not required for these checks."""
from __future__ import annotations
import json
import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from bdse.experiments import planning_budget as pb
from bdse.tools.profile_planning_budget import choose_workload
from bdse.tools.summarize_planning_budget import summarize_files


def runtime_fixture():
    from bdse.data.cache_schema import RuntimeFeatures
    return RuntimeFeatures(ego_history=np.array([[0, 0, 0, 5, -1], [0, 0, 0, 5, 0]], dtype=np.float32),
                           agent_history=np.zeros((1, 2, 10), dtype=np.float32),
                           agent_valid=np.array([True]), current_agents=np.zeros((1, 10), dtype=np.float32),
                           traffic_lights=[], map_features={"route_centerline": np.array([[0, 0], [160, 0]], dtype=np.float32),
                                                         "route_corridor_width": 4., "speed_limit_mps": 13.4},
                           route_roadblock_ids=[], mission_goal=None)


class BudgetProfileTests(unittest.TestCase):
    def test_deadline_and_result_preserved(self):
        with patch.object(pb.time, "perf_counter_ns", side_effect=[0, 0, 0, 120_000_000]):
            result, row = pb.measure_call(lambda: (7, "same trajectory"), deadline_ms=100)
        self.assertEqual(result, (7, "same trajectory"))
        self.assertEqual(row["wall_ms"], 120.)
        self.assertTrue(row["service_exceeded_deadline"])
        self.assertFalse(row["deadline_is_enforced"])

    def test_invalid_deadline_and_mode(self):
        for kwargs in [{"deadline_ms": -1}, {"deadline_ms": float("nan")}, {"mode": "wrong"}]:
            with self.assertRaises(ValueError):
                pb.measure_call(lambda: None, **kwargs)

    def test_exception_recorded_and_context_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fail():
                raise ValueError("intentional")
            with self.assertRaisesRegex(ValueError, "intentional"):
                pb.measure_call(fail, output_prefix=Path(tmp) / "trace")
            row = json.loads(pb.trace_path(Path(tmp) / "trace").read_text())
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["error_type"], "ValueError")
            self.assertIsNone(pb._ACTIVE.get())

    def test_nested_spans_exclusive(self):
        def run():
            with pb.span("outer"):
                with pb.span("inner"):
                    pass
        with patch.object(pb.time, "perf_counter_ns", side_effect=[0, 0, 0, 1_000_000, 2_000_000, 4_000_000, 6_000_000, 8_000_000]):
            _, row = pb.measure_call(run, mode="attribution")
        self.assertEqual(row["spans"]["inner"]["inclusive_ms"], 2.)
        self.assertEqual(row["spans"]["outer"]["inclusive_ms"], 5.)
        self.assertEqual(row["spans"]["outer"]["exclusive_ms"], 3.)

    def test_native_does_not_add_internal_barriers(self):
        with patch.object(pb, "synchronize") as sync:
            _, row = pb.measure_call(lambda: self._span_once(), device="cpu", mode="native")
        self.assertEqual(sync.call_count, 2)
        self.assertEqual(row["spans"], {})

    def _span_once(self):
        with pb.span("test"):
            return 3

    def test_queue_deadline_at_boundary(self):
        row = pb.queue_diagnostic([120, 80, 120], 100, 100)
        self.assertAlmostEqual(row["miss_rate"], 2/3)
        self.assertEqual(row["wait_ms"]["max"], 20)
        self.assertEqual(row["response_ms"]["max"], 120)

    def test_quantiles(self):
        self.assertEqual(pb.latency_stats([0, 100])["p95"], 95.)
        self.assertEqual(pb.latency_stats([]), {"n": 0})

    def test_npz_runtime_only_no_label_or_metadata_read(self):
        runtime = runtime_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            from bdse.data.cache_schema import _json_dumps
            np.savez(path, runtime_ego_history=runtime.ego_history,
                     runtime_agent_history=runtime.agent_history, runtime_agent_valid=runtime.agent_valid,
                     runtime_current_agents=runtime.current_agents, runtime_traffic_lights_json=_json_dumps([]),
                     runtime_map_features_json=_json_dumps(runtime.map_features),
                     route_roadblock_ids=np.asarray([], dtype=str), mission_goal=np.asarray([]),
                     scenario_token=np.asarray("tok"), timestamp_us=np.asarray(123),
                     teacher_poison=np.asarray([object()], dtype=object),
                     label_logged_ego=np.asarray([object()], dtype=object),
                     runtime_metadata_json=_json_dumps({"teacher_leak": "poison"}))
            loaded, identity = pb.read_runtime_npz(path)
        self.assertEqual(loaded.metadata, {})
        self.assertEqual(identity, {"scenario_token": "tok", "timestamp_us": 123})
        np.testing.assert_equal(loaded.map_features["route_centerline"], runtime.map_features["route_centerline"])

    def test_manifest_log_balanced_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for log in ("log0", "log1", "log2"):
                (root / log).mkdir()
                for index in range(6):
                    (root / log / f"{index}.npz").touch()
            a = choose_workload(root, None, 3, 42)
            b = choose_workload(root, None, 3, 42)
        self.assertEqual(a, b)
        self.assertEqual(len({p.parent for p in a}), 3)

    def test_summary_excludes_warmup_and_separates_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "trace"
            for warmup, cached, duration in [(True, False, 999), (False, False, 120), (False, True, 1)]:
                pb.write_row(prefix, {"schema": "bdse_planning_budget_v1", "status": "ok", "warmup": warmup,
                                      "cached_plan": cached, "wall_ms": duration, "scope": "test", "mode": "native",
                                      "device": "cpu", "pid": 1, "start_perf_ns": 1})
            report = summarize_files([pb.trace_path(prefix)], [100])
        self.assertEqual(report["measured_ok_calls"], 2)
        self.assertEqual(len(report["groups"]), 2)
        non_cached = next(g for g in report["groups"] if not g["cached_plan"])
        self.assertEqual(non_cached["wall_ms"]["mean"], 120)
        self.assertEqual(non_cached["warmup_wall_ms"]["mean"], 999)

    def test_thread_contexts_isolated_and_complete_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "trace"
            barrier = threading.Barrier(2)
            def run(number):
                pb.measure_call(lambda: barrier.wait(timeout=3), context={"worker": number}, output_prefix=prefix)
            threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            rows = [json.loads(line) for line in pb.trace_path(prefix).read_text().splitlines()]
        self.assertEqual({r["worker"] for r in rows}, {0, 1})
        self.assertTrue(all(r["status"] == "ok" for r in rows))

    def test_core_capture_and_result_preservation(self):
        trajectory = np.asarray([[1., 2.]])
        core = SimpleNamespace(model=None, plan_from_runtime=lambda runtime: (3, trajectory, {"effective_query_count": 17}))
        pb.instrument_core(core)
        result, row = pb.measure_call(lambda: core.plan_from_runtime(runtime_fixture()))
        self.assertEqual(result[0], 3)
        self.assertIs(result[1], trajectory)
        self.assertEqual(row["counts"]["effective_query_count"], 17)
        self.assertEqual(row["runtime_shape"]["active_agent_count"], 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "original frozen core imports torch; run in target environment")
    def test_real_core_result_invariant_under_instrumentation(self):
        from bdse.config import load_config
        from bdse.planner.nuplan_planner import BDSEPlannerCore
        cfg = load_config(overrides={"runtime": {"max_agents": 1}, "evidence": {"max_atoms": 24, "budget": 4},
                                     "selector": {"proposal_top_m": 8}, "fallback": {"enabled": False}})
        core = BDSEPlannerCore(model=None, cfg=cfg)
        runtime = runtime_fixture()
        a, trajectory, _ = core.plan_from_runtime(runtime)
        pb.instrument_core(core, attribution=True)
        result, row = pb.measure_call(lambda: core.plan_from_runtime(runtime), mode="attribution")
        self.assertEqual(result[0], a)
        np.testing.assert_equal(result[1], trajectory)
        self.assertEqual(row["core_calls"], 1)
        self.assertFalse(row["cached_plan"])
        self.assertIn("certificate_prediction", row["spans"])


if __name__ == "__main__":
    unittest.main()
