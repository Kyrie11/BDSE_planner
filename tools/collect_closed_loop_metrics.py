from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

DEFAULT_COLS = [
    "score",
    "ego_is_making_progress",
    "ego_progress_along_expert_route",
    "no_ego_at_fault_collisions",
    "time_to_collision_within_bound",
    "ego_is_comfortable",
    "speed_limit_compliance",
    "drivable_area_compliance",
    "driving_direction_compliance",
]


def _candidate_metric_files(output_dir: str) -> list[str]:
    base = Path(output_dir)
    name = base.name
    # nuPlan sometimes nests output_dir/challenge/outputs/.../challenge.
    exact = base / "closed_loop_nonreactive_agents" / "outputs" / "closed_loop" / name / "closed_loop_nonreactive_agents" / "aggregator_metric"
    files = sorted(glob.glob(str(exact / "*.parquet")))
    if files:
        return files
    return sorted(glob.glob(str(base / "**" / "aggregator_metric" / "*.parquet"), recursive=True))


def read_final_score(output_dir: str, cols: list[str]) -> dict[str, float | str]:
    files = _candidate_metric_files(output_dir)
    if not files:
        raise FileNotFoundError(f"No aggregator_metric parquet found under {output_dir!r}")
    df = pd.read_parquet(files[0])
    row = df[df["scenario"] == "final_score"]
    if row.empty:
        raise ValueError(f"No final_score row in {files[0]}")
    r = row.iloc[0]
    out: dict[str, float | str] = {"output_dir": output_dir, "metric_file": files[0]}
    for c in cols:
        if c in r:
            out[c] = float(r[c])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", nargs="+", help="Closed-loop output-dir folders, e.g. outputs/closed_loop/v12_rule_prior_w50_20")
    ap.add_argument("--cols", nargs="*", default=DEFAULT_COLS)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()
    rows = []
    for folder in args.outputs:
        try:
            rows.append(read_final_score(folder, args.cols))
        except Exception as exc:
            rows.append({"output_dir": folder, "error": str(exc)})
    df = pd.DataFrame(rows)
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
