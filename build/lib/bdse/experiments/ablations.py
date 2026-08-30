from __future__ import annotations

import argparse
import json
from pathlib import Path

from bdse.config import deep_update, load_config
from bdse.experiments.diagnostics import main as diagnostics_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/ablation_plan.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    budgets = [4, 8, 16, 24, 32]
    rivals = [4, 8, 16, 24, 31]
    selector_modes = ["runtime_predicted", "random", "top_magnitude", "diversity", "interaction_only", "rule_map_only"]
    plan = []
    for B in budgets:
        plan.append(deep_update(cfg, {"evidence": {"budget": B}}))
    for L in rivals:
        plan.append(deep_update(cfg, {"tournament": {"L_infer": L}}))
    for mode in selector_modes:
        plan.append(deep_update(cfg, {"selector": {"mode": mode}}))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"num_runs": len(plan), "budgets": budgets, "rivals": rivals, "selector_modes": selector_modes}, indent=2), encoding="utf-8")
    print(f"Wrote ablation plan with {len(plan)} configs to {out}")


if __name__ == "__main__":
    main()
