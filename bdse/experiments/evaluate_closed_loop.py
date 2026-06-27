from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _split_overrides(raw: list[str]) -> list[str]:
    if raw and raw[0] == "--":
        return raw[1:]
    return raw


def _append_if_missing(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def build_nuplan_command(args: argparse.Namespace, overrides: list[str]) -> tuple[list[str], list[str]]:
    env_overrides = [
        f"BDSE_CHECKPOINT={str(Path(args.checkpoint).resolve())}",
        f"BDSE_CONFIG={str(Path(args.config).resolve()) if args.config else ''}",
        f"BDSE_DEVICE={args.device}",
    ]
    if args.nuplan_data_root:
        env_overrides.append(f"NUPLAN_DATA_ROOT={str(Path(args.nuplan_data_root).expanduser())}")
    if args.nuplan_map_root:
        env_overrides.append(f"NUPLAN_MAPS_ROOT={str(Path(args.nuplan_map_root).expanduser())}")
    if args.nuplan_exp_root:
        env_overrides.append(f"NUPLAN_EXP_ROOT={str(Path(args.nuplan_exp_root).expanduser())}")
    if args.hydra_full_error:
        env_overrides.append("HYDRA_FULL_ERROR=1")

    base = [
        sys.executable,
        "-m",
        args.nuplan_module,
        "hydra.searchpath=[pkg://bdse.nuplan_config]",
        "planner=bdse_planner",
        f"simulation={args.challenge}",
        f"output_dir={args.output_dir}",
        f"experiment_uid={args.experiment_uid}",
    ]
    if args.scenario_builder:
        base.append(f"scenario_builder={args.scenario_builder}")
    if args.scenario_filter:
        base.append(f"scenario_filter={args.scenario_filter}")
    if args.worker:
        base.append(f"worker={args.worker}")
    if args.metric_aggregator:
        base.append(f"metric_aggregator={args.metric_aggregator}")

    final_overrides = list(overrides)
    # Some nuPlan-devkit installs ship a default_simulation.yaml that references
    # splitter/nuplan while the cloned config tree does not include that group.
    # Removing the splitter default is safe for small sequential smoke runs and
    # avoids the Hydra composition error before simulation starts.
    if args.disable_splitter and not any(x.startswith("splitter=") or x == "~splitter" for x in final_overrides):
        final_overrides.insert(0, "~splitter")
    return env_overrides, base + final_overrides


def _run_nuplan_command(cmd: list[str], env: dict[str, str], *, retry_without_splitter: bool) -> None:
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        # Fallback for Hydra versions that do not support defaults-list deletion
        # with '~splitter'.  This keeps the original error visible if both fail.
        if retry_without_splitter and "~splitter" in cmd:
            alt = [x for x in cmd if x != "~splitter"] + ["splitter=null"]
            print("[bdse] nuPlan command failed with '~splitter'; retrying with splitter=null", flush=True)
            subprocess.run(alt, check=True, env=env)
            return
        raise exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a nuPlan closed-loop simulation with BDSEPlanner through Hydra. "
            "Arguments after '--' are forwarded as raw nuPlan Hydra overrides."
        )
    )
    parser.add_argument("--config", type=str, default="bdse/configs/full_preprocess.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", help="Planner device passed to BDSEnuPlanPlanner. auto uses CUDA when available.")
    parser.add_argument("--challenge", type=str, default="closed_loop_nonreactive_agents")
    parser.add_argument("--output-dir", type=str, default="outputs/closed_loop_bdse")
    parser.add_argument("--experiment-uid", type=str, default="bdse_closed_loop")
    parser.add_argument("--nuplan-module", type=str, default="nuplan.planning.script.run_simulation")
    parser.add_argument("--scenario-builder", type=str, default="nuplan")
    parser.add_argument("--scenario-filter", type=str, default=None)
    parser.add_argument("--worker", type=str, default=None)
    parser.add_argument("--metric-aggregator", type=str, default=None)
    parser.add_argument("--nuplan-data-root", type=str, default=None, help="Optional NUPLAN_DATA_ROOT passed to nuPlan.")
    parser.add_argument("--nuplan-map-root", type=str, default=None, help="Optional NUPLAN_MAPS_ROOT passed to nuPlan.")
    parser.add_argument("--nuplan-exp-root", type=str, default=None, help="Optional NUPLAN_EXP_ROOT passed to nuPlan.")
    parser.add_argument("--hydra-full-error", action="store_true", help="Set HYDRA_FULL_ERROR=1 for full Hydra stack traces.")
    parser.add_argument("--disable-splitter", dest="disable_splitter", action="store_true", default=True, help="Remove nuPlan's splitter default; fixes missing splitter/nuplan configs in many local devkit clones.")
    parser.add_argument("--keep-splitter", dest="disable_splitter", action="store_false", help="Do not remove nuPlan's splitter default.")
    parser.add_argument("--dry-run", action="store_true", help="Print the nuPlan command without executing it.")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.config and not Path(args.config).exists():
        raise FileNotFoundError(f"BDSE config not found: {args.config}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    overrides = _split_overrides(args.overrides)
    env_assignments, cmd = build_nuplan_command(args, overrides)
    print(" ".join(env_assignments + cmd))
    if args.dry_run:
        return
    env = os.environ.copy()
    for item in env_assignments:
        k, v = item.split("=", 1)
        env[k] = v
    _run_nuplan_command(cmd, env, retry_without_splitter=bool(args.disable_splitter))


if __name__ == "__main__":
    main()
