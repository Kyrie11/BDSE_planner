from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from bdse.config import deep_update, load_config


def _variant(name: str, cfg: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "cfg": deep_update(cfg, patch), "patch": patch}


def build_ablation_plan(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for B in [4, 8, 16, 24, 32]:
        M = min(int(cfg.get("evidence", {}).get("max_atoms", 128)), max(2 * B, 1))
        plan.append(_variant(f"budget_B{B}_M{M}", cfg, {"evidence": {"budget": B}, "selector": {"proposal_top_m": M}}))
    for mult in [2, 3, 4]:
        B = int(cfg.get("evidence", {}).get("budget", 16))
        M = min(int(cfg.get("evidence", {}).get("max_atoms", 128)), mult * B)
        plan.append(_variant(f"proposal_M{mult}B_{M}", cfg, {"selector": {"proposal_top_m": M}}))
    plan.append(_variant("proposal_full_E", cfg, {"selector": {"proposal_top_m": int(cfg.get("evidence", {}).get("max_atoms", 128))}}))
    for L in [4, 8, 16, 24, 31]:
        plan.append(_variant(f"rivals_L{L}", cfg, {"tournament": {"L_infer": L}}))
    for mode in ["runtime_predicted", "random", "top_magnitude", "diversity", "interaction_only", "rule_map_only", "risk_only", "full_prescore_ablation"]:
        plan.append(_variant(f"selector_{mode}", cfg, {"selector": {"mode": mode}}))
    family_patches = {
        "no_interaction": {"evidence": {"include_interaction": False}},
        "no_rule_map": {"evidence": {"include_rule_map": False}},
        "no_kinematic": {"evidence": {"include_kinematic": False}},
        "no_hard_atoms": {"evidence": {"include_hard_atoms": False}},
    }
    for name, patch in family_patches.items():
        plan.append(_variant(name, cfg, patch))
    for enabled in [False, True]:
        plan.append(_variant(f"fallback_{'on' if enabled else 'off'}", cfg, {"fallback": {"enabled": enabled}}))
    for demo_w in [0.0, 1.0, 3.0, 10.0]:
        plan.append(_variant(f"demo_weight_{demo_w:g}", cfg, {"teacher": {"demo_weight": demo_w}}))
    for include in [False, True]:
        plan.append(_variant(f"drivable_polygons_{'on' if include else 'off'}", cfg, {"runtime": {"include_drivable_polygons": include}}))
    return plan


def _write_variant_configs(plan: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    cfg_dir = output_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in plan:
        path = cfg_dir / f"{item['name']}.yaml"
        path.write_text(yaml.safe_dump(item["cfg"], sort_keys=False), encoding="utf-8")
        rows.append({"name": item["name"], "config": str(path), "patch": item["patch"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/ablations")
    parser.add_argument("--plan-output", type=str, default=None)
    parser.add_argument("--run", action="store_true", help="Run open-loop evaluation for each ablation config.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--preprocessed-dir", type=str, default=None)
    parser.add_argument("--split", type=str, nargs="+", default=["val"])
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--no-dense-full-interface", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_ablation_plan(cfg)
    rows = _write_variant_configs(plan, out_dir)
    plan_path = Path(args.plan_output) if args.plan_output else out_dir / "ablation_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"num_runs": len(rows), "runs": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote ablation plan with {len(rows)} configs to {plan_path}")
    if not args.run:
        return
    if not args.checkpoint or not args.preprocessed_dir:
        raise ValueError("--run requires --checkpoint and --preprocessed-dir")
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for row in rows:
        metric_path = metrics_dir / f"{row['name']}.json"
        cmd = [
            sys.executable,
            "-m",
            "bdse.experiments.evaluate_open_loop",
            "--config",
            row["config"],
            "--checkpoint",
            args.checkpoint,
            "--preprocessed-dir",
            args.preprocessed_dir,
            "--output",
            str(metric_path),
        ]
        for split in args.split:
            cmd.extend(["--split", split])
        if args.max_scenarios is not None:
            cmd.extend(["--max-scenarios", str(args.max_scenarios)])
        if args.no_dense_full_interface:
            cmd.append("--no-dense-full-interface")
        print("[bdse ablation]", " ".join(cmd), flush=True)
        completed = subprocess.run(cmd, check=False)
        rec = {"name": row["name"], "metric_path": str(metric_path), "returncode": int(completed.returncode)}
        if metric_path.exists():
            try:
                payload = json.loads(metric_path.read_text(encoding="utf-8"))
                rec.update(payload.get("summary", payload))
            except Exception as exc:
                rec["parse_error"] = repr(exc)
        summary_rows.append(rec)
    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps({"runs": summary_rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
