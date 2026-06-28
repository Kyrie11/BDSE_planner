from __future__ import annotations

import argparse
from pathlib import Path

from bdse.experiments.evaluate_closed_loop import build_nuplan_command


def test_closed_loop_command_keeps_splitter_by_default(tmp_path: Path):
    ckpt = tmp_path / "m.pt"
    cfg = tmp_path / "c.yaml"
    ckpt.write_bytes(b"x")
    cfg.write_text("{}")
    args = argparse.Namespace(
        checkpoint=str(ckpt),
        config=str(cfg),
        device="cpu",
        nuplan_data_root=None,
        nuplan_map_root=None,
        nuplan_exp_root=None,
        hydra_full_error=False,
        nuplan_module="nuplan.planning.script.run_simulation",
        challenge="closed_loop_nonreactive_agents",
        output_dir="out",
        experiment_uid="eid",
        scenario_builder="nuplan",
        scenario_filter=None,
        worker="sequential",
        metric_aggregator=None,
        disable_splitter=False,
    )
    env, cmd = build_nuplan_command(args, ["scenario_filter.limit_total_scenarios=2"])
    assert any(x.startswith("BDSE_CHECKPOINT=") for x in env)
    assert "~splitter" not in cmd
    assert "planner=bdse_planner" in cmd
    searchpath = next(x for x in cmd if x.startswith("hydra.searchpath="))
    assert "pkg://nuplan.planning.script.config.common" in searchpath
    assert "pkg://nuplan.planning.script.experiments" in searchpath
    assert "pkg://bdse.nuplan_config" in searchpath


def test_closed_loop_command_can_pass_nuplan_roots(tmp_path: Path):
    ckpt = tmp_path / "m.pt"
    cfg = tmp_path / "c.yaml"
    ckpt.write_bytes(b"x")
    cfg.write_text("{}")
    args = argparse.Namespace(
        checkpoint=str(ckpt),
        config=str(cfg),
        device="cpu",
        nuplan_data_root="/data/nuplan",
        nuplan_map_root="/data/maps",
        nuplan_exp_root="/data/exp",
        hydra_full_error=True,
        nuplan_module="nuplan.planning.script.run_simulation",
        challenge="closed_loop_nonreactive_agents",
        output_dir="out",
        experiment_uid="eid",
        scenario_builder="nuplan",
        scenario_filter=None,
        worker=None,
        metric_aggregator=None,
        disable_splitter=False,
    )
    env, cmd = build_nuplan_command(args, [])
    assert "NUPLAN_DATA_ROOT=/data/nuplan" in env
    assert "NUPLAN_MAPS_ROOT=/data/maps" in env
    assert "NUPLAN_EXP_ROOT=/data/exp" in env
    assert "HYDRA_FULL_ERROR=1" in env
    assert "~splitter" not in cmd

def test_nuplan_splitter_compat_config_is_packaged():
    from importlib.resources import files

    cfg_path = files("bdse.nuplan_config").joinpath("splitter", "nuplan.yaml")
    assert cfg_path.is_file()
    assert "null" in cfg_path.read_text()

def test_closed_loop_runner_does_not_use_invalid_splitter_null_retry():
    import inspect
    from bdse.experiments import evaluate_closed_loop

    source = inspect.getsource(evaluate_closed_loop._run_nuplan_command)
    assert "splitter=null" not in source
