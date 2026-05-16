from __future__ import annotations

import argparse
from pathlib import Path

from bdse.config import load_config
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, discover_available_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--folders", type=str, nargs="*", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--list-splits", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.list_splits:
        print(discover_available_splits(cfg["paths"]["data_cache_root"]))
        return
    dataset = NuPlanBDSEDataset(
        cfg=cfg,
        split=args.split,
        folders=args.folders,
        max_files=args.max_files,
        max_scenarios=args.max_scenarios,
        use_devkit=True,
        preprocessed_dir=args.output_dir or cfg["paths"]["preprocessed_cache"],
    )
    paths = dataset.write_preprocessed_cache(args.output_dir)
    print(f"Wrote {len(paths)} samples under {Path(args.output_dir or cfg['paths']['preprocessed_cache'])}")


if __name__ == "__main__":
    main()
