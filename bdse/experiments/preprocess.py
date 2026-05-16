from __future__ import annotations

import argparse
from pathlib import Path

from bdse.config import load_config
from bdse.data.nuplan_dataset import NuPlanBDSEDataset, discover_available_splits


def _folders_for_split(cfg: dict, split: str, cli_folders: list[str] | None) -> list[str] | None:
    if cli_folders:
        return cli_folders
    split_folders = cfg.get("data", {}).get("split_folders", {})
    val = split_folders.get(split)
    return list(val) if val else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, help="Single split, kept for backward compatibility.")
    parser.add_argument("--splits", type=str, nargs="*", default=None, help="One or more splits, e.g. train val. Defaults to --split or train.")
    parser.add_argument("--folders", type=str, nargs="*", default=None, help="Explicit DB subfolders to use for every selected split.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--maps-root", type=str, default=None)
    parser.add_argument("--map-version", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-files-per-split", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-scenarios-per-split", type=int, default=None)
    parser.add_argument("--scenario-stride", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--list-splits", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("paths", {})
    cfg.setdefault("preprocess", {})
    if args.data_root:
        cfg["paths"]["data_cache_root"] = args.data_root
    if args.maps_root:
        cfg["paths"]["maps_root"] = args.maps_root
    if args.map_version:
        cfg["paths"]["map_version"] = args.map_version
    if args.output_dir:
        cfg["paths"]["preprocessed_cache"] = args.output_dir
    if args.scenario_stride is not None:
        cfg["preprocess"]["scenario_stride"] = int(args.scenario_stride)

    if args.list_splits:
        print(discover_available_splits(cfg["paths"]["data_cache_root"]))
        return

    splits = args.splits or ([args.split] if args.split else ["train"])
    resume = bool(cfg.get("preprocess", {}).get("resume", True)) if args.resume is None else bool(args.resume)
    out_dir = Path(args.output_dir or cfg["paths"].get("preprocessed_cache", "cache"))
    total_paths = []
    for split in splits:
        folders = _folders_for_split(cfg, split, args.folders)
        dataset = NuPlanBDSEDataset(
            cfg=cfg,
            split=split,
            folders=folders,
            max_files=args.max_files_per_split if args.max_files_per_split is not None else args.max_files,
            max_scenarios=args.max_scenarios_per_split if args.max_scenarios_per_split is not None else args.max_scenarios,
            stride=args.scenario_stride,
            use_devkit=True,
            preprocessed_dir=out_dir,
        )
        paths = dataset.write_preprocessed_cache(out_dir, resume=resume, overwrite=args.overwrite, show_progress=not args.no_tqdm)
        total_paths.extend(paths)
        print(f"split={split}: materialized {len(paths)} sample paths under {out_dir}")
    print(f"Done. materialized {len(total_paths)} sample paths under {out_dir}")


if __name__ == "__main__":
    main()
