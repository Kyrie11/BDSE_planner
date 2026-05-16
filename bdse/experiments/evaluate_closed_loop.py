from __future__ import annotations

import argparse
from pathlib import Path

from bdse.config import load_config
from bdse.planner.nuplan_planner import BDSEnuPlanPlanner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--challenge", type=str, default="closed_loop_nonreactive_agents")
    parser.add_argument("--output-dir", type=str, default="outputs/closed_loop")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = None
    if args.checkpoint:
        import torch
        from bdse.model.bdse_model import BDSEModel

        model = BDSEModel(cfg)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    planner = BDSEnuPlanPlanner(model=model, cfg=cfg)
    try:
        from nuplan.planning.script.run_simulation import main as run_simulation
    except ImportError as exc:
        raise RuntimeError("Install nuplan-devkit and run this script inside a nuPlan Hydra environment.") from exc
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Planner ready: {planner.name()}. Use nuPlan Hydra configs to pass it as pre_built_planners for {args.challenge}.")
    print("For full closed-loop evaluation, import BDSEnuPlanPlanner in a nuPlan run_simulation entrypoint and provide the planner instance as pre_built_planners.")


if __name__ == "__main__":
    main()
