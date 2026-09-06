from __future__ import annotations

import json
from pathlib import Path

from bdse.tools import run_fixed_budget_closed_loop_suite as suite
from bdse.tools.summarize_v64_3_56_converged_benchmark import main as _summary_main  # noqa: F401


def test_external_closed_loop_suite_requires_metric_safe_provenance() -> None:
    base = {
        "scenario_count": 100,
        "scenario_token_sha256": "tok",
        "config_sha256": "cfg",
        "checkpoint_sha256": "ckpt",
    }
    assert not suite._resume_summary_compatible(base, scenario_count=100, token_sha="tok", config_sha="cfg", checkpoint_sha="ckpt")
    base["metric_engine_serialized"] = True
    base["nuplan_module"] = suite.METRIC_SAFE_NUPLAN_MODULE
    assert suite._resume_summary_compatible(base, scenario_count=100, token_sha="tok", config_sha="cfg", checkpoint_sha="ckpt")


def test_external_closed_loop_suite_uses_existing_metric_safe_wrapper() -> None:
    assert suite.METRIC_SAFE_NUPLAN_MODULE == "bdse.tools.nuplan_metric_safe_run_simulation"
    assert suite.METRIC_SAFE_ENV_KEY == "BDSE_PIOR_METRIC_ENGINE_SERIALIZATION"
    src = Path(suite.__file__).read_text(encoding="utf-8")
    assert '"--nuplan-module", METRIC_SAFE_NUPLAN_MODULE' in src
    assert 'METRIC_SAFE_ENV_KEY: "1"' in src


def test_converged_benchmark_launcher_is_b16_primary_and_cross_budget_honest() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "RUN_V64_3_56_CONVERGED_CLOSED_LOOP_B8_B16_B24_2GPU.sh").read_text(encoding="utf-8")
    assert 'V47_ROOT' in text and 'v64_3_47_rsmr.yaml' in text
    assert '--systems bdse gameformer dtpp plantf pluto pdm_closed_style' in text
    assert 'BUDGETS="${BUDGETS:-8 16 24}"' in text
    assert 'Primary matched-interface comparison: B=16.' in text
    assert 'B=8 and B=24 for BDSE are frozen-policy cross-budget robustness ablations' in text
    assert 'sha256sum -c V64_3_56_SCIENCE_MANIFEST.sha256' in text


def test_converged_benchmark_summary_rejects_unsafe_rows(tmp_path: Path, monkeypatch) -> None:
    from bdse.tools import summarize_v64_3_56_converged_benchmark as sm
    src = tmp_path / "x.json"
    src.write_text(json.dumps([{
        "budget":16,"system":"bdse","scenario_count":1,"successful":1,"failed":0,
        "scenario_token_sha256":"x","metric_engine_serialized":False,
    }]))
    out=tmp_path/"o"
    monkeypatch.setattr("sys.argv", ["x", "--input-json", str(src), "--output-root", str(out)])
    try:
        sm.main()
    except SystemExit as exc:
        assert "metric-safety" in str(exc)
    else:
        raise AssertionError("unsafe rows must fail closed")


def test_converged_suite_uses_file_backed_manifests_not_giant_token_argv() -> None:
    src = Path(suite.__file__).read_text(encoding="utf-8")
    assert '"--scenario-tokens-file", str(token_manifest)' in src
    assert '"--scenario-tokens-sha256", str(token_sha)' in src
    assert '"--nuplan-db-files-file", str(db_manifest)' in src
    assert 'token_override = "scenario_filter.scenario_tokens="' not in src


def test_own_frozen_budget_sweep_is_evaluation_only_and_uses_queue_transport() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "RUN_V64_3_56_OWN_FROZEN_B8_B16_B24_CLOSED_LOOP_2GPU.sh").read_text(encoding="utf-8")
    assert '--systems bdse --resume' in text
    assert '--schedule-mode queue' in text
    assert 'BUDGETS="${BUDGETS:-8 16 24}"' in text
    assert 'frozen B16 learned policy; cross-budget interface robustness' in text
    assert 'RUN_FAIR_EXTERNAL_BASELINES_TRAIN' not in text
