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


def build_nuplan_command(args: argparse.Namespace, overrides: list[str]) -> list[str]:
    env_overrides = [
        f"BDSE_CHECKPOINT={str(Path(args.checkpoint).resolve())}",
        f"BDSE_CONFIG={str(Path(args.config).resolve()) if args.config else ''}",
        f"BDSE_DEVICE={args.device}",
    ]
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
    return env_overrides + base + overrides


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
    parser.add_argument("--dry-run", action="store_true", help="Print the nuPlan command without executing it.")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.config and not Path(args.config).exists():
        raise FileNotFoundError(f"BDSE config not found: {args.config}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    overrides = _split_overrides(args.overrides)
    cmd_with_env = build_nuplan_command(args, overrides)
    env_assignments = cmd_with_env[:3]
    cmd = cmd_with_env[3:]
    print(" ".join(env_assignments + cmd))
    if args.dry_run:
        return
    env = os.environ.copy()
    for item in env_assignments:
        k, v = item.split("=", 1)
        env[k] = v
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
