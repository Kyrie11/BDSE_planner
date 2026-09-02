from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from bdse.tools import nuplan_metric_safe_run_simulation as metric_safe
from bdse.tools import run_v64_3_50_5_pior_paired_closed_loop as shim


def test_metric_engine_wrapper_serializes_concurrent_callbacks() -> None:
    active = 0
    peak = 0
    guard = threading.Lock()

    def fake_run_metric_engine(*args, **kwargs):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with guard:
            active -= 1
        return "ok"

    fake_module = SimpleNamespace(run_metric_engine=fake_run_metric_engine)
    wrapped = metric_safe.install_metric_engine_serialization(fake_module)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: wrapped(), range(16)))
    assert results == ["ok"] * 16
    assert peak == 1


def test_metric_engine_wrapper_is_idempotent() -> None:
    fake_module = SimpleNamespace(run_metric_engine=lambda: 7)
    first = metric_safe.install_metric_engine_serialization(fake_module)
    second = metric_safe.install_metric_engine_serialization(fake_module)
    assert first is second
    assert second() == 7


def test_spawn_shim_rewrites_only_nuplan_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class DummyProc:
        pass

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return DummyProc()

    monkeypatch.setattr(shim._subprocess, "Popen", fake_popen)
    proxy = shim._SubprocessProxy()
    cmd = [
        "python", "-m", "bdse.experiments.evaluate_closed_loop",
        "--nuplan-module", "nuplan.planning.script.run_simulation",
        "--worker", "single_machine_thread_pool",
    ]
    proc = proxy.Popen(cmd, env={"X": "1"})
    assert isinstance(proc, DummyProc)
    i = captured["cmd"].index("--nuplan-module")
    assert captured["cmd"][i + 1] == "bdse.tools.nuplan_metric_safe_run_simulation"
    assert captured["env"]["BDSE_PIOR_METRIC_ENGINE_SERIALIZATION"] == "1"
    assert captured["env"]["X"] == "1"


def test_spawn_shim_fails_closed_on_changed_nuplan_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shim._subprocess, "Popen", lambda *a, **k: object())
    proxy = shim._SubprocessProxy()
    with pytest.raises(RuntimeError, match="unexpected frozen nuPlan entrypoint"):
        proxy.Popen([
            "python", "-m", "bdse.experiments.evaluate_closed_loop",
            "--nuplan-module", "some.other.module",
        ])


def test_science_critical_v50_files_remain_byte_identical() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = {
        "bdse/tools/run_v64_3_50_pior_paired_closed_loop.py": "7c5472442e5a76ee6cbb6ef3189e086e4e800e8337deb14cacb26488b9050d53",
        "bdse/planner/nuplan_planner.py": "c3a6e37901349408b7c8e6ab7b3811f905f3a81b0c441e6aa7ddf4dde92131ef",
        "bdse/planner/tournament.py": "291b3b77202974b74fe42431ee7954de8c401d927591c19a12a5837f18374044",
        "bdse/tools/fit_v64_3_50_eaf_icer_pior.py": "c1c1d297766d8a2e43430739d639ff1ed73f866ec193b47a5dc9e7cb997727aa",
        "bdse/tools/build_v64_3_50_pior_train_manifest.py": "02b027447bd5dbfbe44e8d58ca39380df44c709a48a8e1523057826b354e5225",
        "bdse/tools/make_v64_3_50_pior_probe_configs.py": "6d8cd2f869cde7856d07cb09a9260c08907e2f11042d49ea8603668c838b9fad",
    }
    for rel, want in expected.items():
        got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        assert got == want, (rel, got, want)
