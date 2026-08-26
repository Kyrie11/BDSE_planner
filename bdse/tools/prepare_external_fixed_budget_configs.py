from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


SYSTEMS = ("gameformer", "dtpp", "plantf", "pluto", "pdm_closed")


def resolve_budget_config(source: Path, output: Path, *, budget: int, proposal_top_m: int) -> None:
    cfg: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    cfg.setdefault("evidence", {})["budget"] = int(budget)
    selector = cfg.setdefault("selector", {})
    selector["min_selected_atoms"] = int(budget)
    selector["force_fill_budget"] = True
    selector["proposal_top_m"] = int(proposal_top_m)
    fallback = cfg.setdefault("fallback", {})
    fallback["enabled"] = False
    fallback["max_additional_stages"] = 0
    fallback["budget_stages"] = [int(budget)]
    external = cfg.get("external_baseline")
    if isinstance(external, dict):
        external["budget"] = int(budget)
    cfg.setdefault("experiment", {})["fixed_budget_protocol"] = {
        "budget": int(budget),
        "proposal_top_m": int(proposal_top_m),
        "fallback_disabled": True,
        "training_budget_specific": True,
        "planner_supervision": str((external or {}).get("planner_supervision", "deterministic")) if isinstance(external, dict) else "deterministic",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate strict B-specific external baseline train/closed-loop configs with fixed upstream M.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 24])
    p.add_argument("--proposal-top-m", type=int, default=24)
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    args = p.parse_args()
    unknown = [s for s in args.systems if s not in SYSTEMS]
    if unknown:
        raise ValueError(f"unknown systems: {unknown}")
    if any(b <= 0 for b in args.budgets):
        raise ValueError("budgets must be positive")
    if args.proposal_top_m <= 0:
        raise ValueError("proposal_top_m must be positive")

    for budget in args.budgets:
        out_dir = args.output_root / f"B{budget}"
        for system in args.systems:
            if system == "pdm_closed":
                sources = [(Path("bdse/configs/external_pdm_closed_budgeted_fast_cl.yaml"), "external_pdm_closed_budgeted_fast_cl.yaml")]
            else:
                sources = [
                    (Path(f"bdse/configs/external_{system}_budgeted.yaml"), f"external_{system}_budgeted.yaml"),
                    (Path(f"bdse/configs/external_{system}_budgeted_fast_cl.yaml"), f"external_{system}_budgeted_fast_cl.yaml"),
                ]
            for source, name in sources:
                if not source.is_file():
                    raise FileNotFoundError(source)
                resolve_budget_config(source, out_dir / name, budget=budget, proposal_top_m=args.proposal_top_m)
    print(args.output_root.resolve())


if __name__ == "__main__":
    main()
