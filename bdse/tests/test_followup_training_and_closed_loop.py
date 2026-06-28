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
        nuplan_db_root=None,
        nuplan_db_files=None,
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
        nuplan_db_root=None,
        nuplan_db_files=None,
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


def test_closed_loop_command_appends_simulation_group_for_current_nuplan(tmp_path: Path):
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
        nuplan_db_root=None,
        nuplan_db_files=None,
        hydra_full_error=False,
        nuplan_module="nuplan.planning.script.run_simulation",
        challenge="closed_loop_nonreactive_agents",
        output_dir="out",
        experiment_uid="eid",
        scenario_builder="nuplan",
        scenario_filter=None,
        worker="sequential",
        metric_aggregator="closed_loop_nonreactive_agents_weighted_average",
        disable_splitter=False,
    )
    _env, cmd = build_nuplan_command(args, [])
    assert "+simulation=closed_loop_nonreactive_agents" in cmd
    assert "simulation=closed_loop_nonreactive_agents" not in cmd
    assert "metric_aggregator=closed_loop_nonreactive_agents_weighted_average" in cmd


def test_closed_loop_command_can_override_nuplan_db_root(tmp_path: Path):
    ckpt = tmp_path / "m.pt"
    cfg = tmp_path / "c.yaml"
    ckpt.write_bytes(b"x")
    cfg.write_text("{}")
    args = argparse.Namespace(
        checkpoint=str(ckpt),
        config=str(cfg),
        device="cpu",
        nuplan_data_root="/data0/senzeyu2/dataset/nuplan",
        nuplan_map_root="/data0/senzeyu2/dataset/nuplan/maps",
        nuplan_exp_root="/data0/senzeyu2/dataset/nuplan/exp",
        nuplan_db_root="/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/val",
        nuplan_db_files=None,
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
    _env, cmd = build_nuplan_command(args, [])
    assert "scenario_builder.data_root=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/val" in cmd


def test_closed_loop_command_can_override_multiple_db_dirs(tmp_path: Path):
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
        nuplan_db_root=None,
        nuplan_db_files=["/cache/train_boston", "/cache/train_pittsburgh"],
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
    _env, cmd = build_nuplan_command(args, [])
    assert "scenario_builder.db_files=[/cache/train_boston,/cache/train_pittsburgh]" in cmd


def test_closed_loop_raw_data_root_override_rewrites_nested_db_dirs(tmp_path: Path):
    ckpt = tmp_path / "m.pt"
    cfg = tmp_path / "c.yaml"
    root = tmp_path / "bdse_val_v2" / "val"
    nested = root / "city_bucket"
    nested.mkdir(parents=True)
    (nested / "log_a.db").write_bytes(b"sqlite-placeholder")
    ckpt.write_bytes(b"x")
    cfg.write_text("{}")
    args = argparse.Namespace(
        checkpoint=str(ckpt),
        config=str(cfg),
        device="cpu",
        nuplan_data_root=None,
        nuplan_map_root=None,
        nuplan_exp_root=None,
        nuplan_db_root=None,
        nuplan_db_files=None,
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
    _env, cmd = build_nuplan_command(args, [f"scenario_builder.data_root={root}"])
    assert f"scenario_builder.data_root={root}" not in cmd
    assert f"scenario_builder.db_files=[{nested}]" in cmd
    assert "scenario_filter.log_names=null" in cmd


def test_closed_loop_nuplan_db_root_rewrites_direct_db_dir_to_db_files(tmp_path: Path):
    ckpt = tmp_path / "m.pt"
    cfg = tmp_path / "c.yaml"
    root = tmp_path / "val"
    root.mkdir()
    (root / "log_a.db").write_bytes(b"sqlite-placeholder")
    ckpt.write_bytes(b"x")
    cfg.write_text("{}")
    args = argparse.Namespace(
        checkpoint=str(ckpt),
        config=str(cfg),
        device="cpu",
        nuplan_data_root=None,
        nuplan_map_root=None,
        nuplan_exp_root=None,
        nuplan_db_root=str(root),
        nuplan_db_files=None,
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
    _env, cmd = build_nuplan_command(args, [])
    assert f"scenario_builder.db_files=[{root}]" in cmd
    assert "scenario_filter.log_names=null" in cmd


def test_closed_loop_does_not_clear_explicit_log_names(tmp_path: Path):
    ckpt = tmp_path / "m.pt"
    cfg = tmp_path / "c.yaml"
    root = tmp_path / "val"
    root.mkdir()
    (root / "log_a.db").write_bytes(b"sqlite-placeholder")
    ckpt.write_bytes(b"x")
    cfg.write_text("{}")
    args = argparse.Namespace(
        checkpoint=str(ckpt),
        config=str(cfg),
        device="cpu",
        nuplan_data_root=None,
        nuplan_map_root=None,
        nuplan_exp_root=None,
        nuplan_db_root=str(root),
        nuplan_db_files=None,
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
    _env, cmd = build_nuplan_command(args, ["scenario_filter.log_names=[log_a]"])
    assert "scenario_filter.log_names=null" not in cmd
    assert "scenario_filter.log_names=[log_a]" in cmd


def test_closed_loop_nuplan_db_root_expands_log_subfolders(tmp_path: Path):
    ckpt = tmp_path / "m.pt"
    cfg = tmp_path / "c.yaml"
    val_root = tmp_path / "bdse_val_v2" / "val"
    log_a = val_root / "2021.06.07.11.59.52_veh-35"
    log_b = val_root / "2021.06.08.19.16.23_veh-26"
    log_a.mkdir(parents=True)
    log_b.mkdir(parents=True)
    (log_a / "log_a.db").write_bytes(b"sqlite-placeholder")
    (log_b / "log_b.db").write_bytes(b"sqlite-placeholder")
    ckpt.write_bytes(b"x")
    cfg.write_text("{}")
    args = argparse.Namespace(
        checkpoint=str(ckpt),
        config=str(cfg),
        device="cpu",
        nuplan_data_root=None,
        nuplan_map_root=None,
        nuplan_exp_root=None,
        nuplan_db_root=str(val_root),
        nuplan_db_files=None,
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
    _env, cmd = build_nuplan_command(args, [])
    assert f"scenario_builder.db_files=[{log_a},{log_b}]" in cmd
    assert f"scenario_builder.data_root={val_root}" not in cmd
    assert "scenario_filter.log_names=null" in cmd


def test_closed_loop_nuplan_db_files_expands_roots_too(tmp_path: Path):
    ckpt = tmp_path / "m.pt"
    cfg = tmp_path / "c.yaml"
    val_root = tmp_path / "bdse_val_v2" / "val"
    log_a = val_root / "2021.06.07.11.59.52_veh-35"
    log_a.mkdir(parents=True)
    (log_a / "log_a.db").write_bytes(b"sqlite-placeholder")
    ckpt.write_bytes(b"x")
    cfg.write_text("{}")
    args = argparse.Namespace(
        checkpoint=str(ckpt),
        config=str(cfg),
        device="cpu",
        nuplan_data_root=None,
        nuplan_map_root=None,
        nuplan_exp_root=None,
        nuplan_db_root=None,
        nuplan_db_files=[str(val_root)],
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
    _env, cmd = build_nuplan_command(args, [])
    assert f"scenario_builder.db_files=[{log_a}]" in cmd
    assert "scenario_filter.log_names=null" in cmd


def test_closed_loop_normalizes_singular_simulation_metric_override(tmp_path: Path):
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
        nuplan_db_root=None,
        nuplan_db_files=None,
        hydra_full_error=False,
        nuplan_module="nuplan.planning.script.run_simulation",
        challenge="closed_loop_nonreactive_agents",
        output_dir="out",
        experiment_uid="eid",
        scenario_builder="nuplan",
        scenario_filter=None,
        worker="sequential",
        metric_aggregator="closed_loop_nonreactive_agent_weighted_average",
        disable_splitter=False,
    )
    _env, cmd = build_nuplan_command(args, ["simulation_metric=simulation_closed_loop_nonreactive_agent"])
    assert "simulation_metric=simulation_closed_loop_nonreactive_agents" in cmd
    assert "simulation_metric=simulation_closed_loop_nonreactive_agent" not in cmd
    assert "metric_aggregator=closed_loop_nonreactive_agents_weighted_average" in cmd


def test_closed_loop_normalizes_singular_challenge_name(tmp_path: Path):
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
        nuplan_db_root=None,
        nuplan_db_files=None,
        hydra_full_error=False,
        nuplan_module="nuplan.planning.script.run_simulation",
        challenge="closed_loop_nonreactive_agent",
        output_dir="out",
        experiment_uid="eid",
        scenario_builder="nuplan",
        scenario_filter=None,
        worker="sequential",
        metric_aggregator=None,
        disable_splitter=False,
    )
    _env, cmd = build_nuplan_command(args, [])
    assert "+simulation=closed_loop_nonreactive_agents" in cmd
    assert "+simulation=closed_loop_nonreactive_agent" not in cmd
