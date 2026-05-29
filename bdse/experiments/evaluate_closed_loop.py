from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from bdse.config import load_config
from bdse.metrics.nuplan_metrics import load_nuplan_metric_summary, validate_metric_summary
from bdse.planner.nuplan_planner import BDSEnuPlanPlanner


def _load_model(checkpoint: str | None, cfg: dict[str, Any]):
    if not checkpoint:
        return None
    import torch
    from bdse.model.bdse_model import BDSEModel

    model = BDSEModel(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _load_callable(path: str):
    mod_name, _, fn_name = path.partition(":")
    if not mod_name or not fn_name:
        raise ValueError("runner must be formatted as module:function")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"{path} is not callable")
    return fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--challenge", type=str, default="closed_loop_nonreactive_agents")
    parser.add_argument("--output-dir", type=str, default="outputs/closed_loop")
    parser.add_argument("--metric-summary", type=str, default=None, help="Validate/export an existing nuPlan metric summary JSON.")
    parser.add_argument("--run", action="store_true", help="Call a project-provided nuPlan runner with the prebuilt BDSE planner.")
    parser.add_argument("--runner", type=str, default=None, help="Callable module:function accepting planner=..., cfg=..., challenge=..., output_dir=....")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = _load_model(args.checkpoint, cfg)
    planner = BDSEnuPlanPlanner(model=model, cfg=cfg)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {"planner_name": planner.name(), "challenge": args.challenge, "output_dir": str(out)}
    if args.run:
        if not args.runner:
            raise RuntimeError(
                "Closed-loop execution needs a project-specific Hydra/devkit runner. "
                "Pass --runner module:function; the callable will receive planner, cfg, challenge and output_dir."
            )
        runner = _load_callable(args.runner)
        result = runner(planner=planner, cfg=cfg, challenge=args.challenge, output_dir=str(out))
        payload["runner_result"] = result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else str(result)

    if args.metric_summary:
        summary = load_nuplan_metric_summary(args.metric_summary)
        validate_metric_summary(summary)
        payload["nuplan_metrics"] = summary
        (out / "closed_loop_metric_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if not args.run and not args.metric_summary:
        payload["integration_note"] = (
            "BDSE planner was constructed successfully. To execute simulation, pass --run "
            "with --runner module:function from your nuPlan Hydra project, or pass --metric-summary "
            "after running nuPlan externally."
        )
    (out / "bdse_closed_loop_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
