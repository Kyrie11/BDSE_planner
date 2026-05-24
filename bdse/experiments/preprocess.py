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
    parser.add_argument("--split", type=str, nargs="+", default=None, help="One or more splits. Kept for backward compatibility with a single value.")
    parser.add_argument("--splits", type=str, nargs="*", default=None, help="One or more splits, e.g. train val. Defaults to --split or train.")
    parser.add_argument("--folders", type=str, nargs="*", default=None, help="Explicit DB subfolders to use for every selected split.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--maps-root", type=str, default=None)
    parser.add_argument("--map-version", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-files-per-split", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-scenarios-per-split", type=int, default=None)
    parser.add_argument("--max-samples-per-log", type=int, default=None,
                        help="Cap materialized samples per nuPlan DB/log after stride thinning. Useful for balanced time tests.")
    parser.add_argument("--max-samples-per-log-strategy", type=str, default=None, choices=["first", "uniform", "uniform_blocks"],
                        help="How to choose capped samples inside each log. 'uniform_blocks' keeps broad coverage while grouping nearby samples for exact temporal-cache reuse.")
    parser.add_argument("--max-samples-per-log-block-size", type=int, default=None,
                        help="For --max-samples-per-log-strategy uniform_blocks, number of consecutive stride-thinned samples per log block.")
    parser.add_argument("--scenario-stride", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--list-splits", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-in-flight", type=int, default=None,
                        help="Maximum sample-materialization futures submitted at once. Set 1 to preserve strict cache-local order for profiling/resume debugging.")
    parser.add_argument("--scenario-builder-workers", type=int, default=None,
                        help="Override nuPlan ScenarioBuilder worker count used during scenario discovery/index construction.")
    parser.add_argument("--temporal-frame-cache-max-entries", type=int, default=None,
                        help="Override exact ego/tracked-frame cache size. Larger values help high-worker preprocessing reuse warm frames across logs.")
    parser.add_argument("--temporal-frame-cache-individual-miss-threshold", type=int, default=None,
                        help="Max future-frame misses to fill by individual iteration calls before falling back to exact bulk nuPlan APIs.")
    parser.add_argument("--cache-local-scheduler", action="store_true", default=None,
                        help="Keep at most one in-flight materialization per nuPlan log to preserve temporal cache locality.")
    parser.add_argument("--no-cache-local-scheduler", action="store_false", dest="cache_local_scheduler",
                        help="Disable per-log cache-local scheduling. This is rarely faster for nuPlan exact labels.")
    parser.add_argument("--temporal-frame-cache-coalesce-bulk", action="store_true", default=None,
                        help="Serialize exact bulk frame-cache fills per log/direction so overlapping windows do not duplicate nuPlan DB work.")
    parser.add_argument("--no-temporal-frame-cache-coalesce-bulk", action="store_false", dest="temporal_frame_cache_coalesce_bulk",
                        help="Disable per-log coalescing of exact bulk cache fills. This is mainly for debugging contention.")
    parser.add_argument("--use-process-pool", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Print per-sample preprocessing time breakdowns.")
    parser.add_argument("--profile-threshold-s", type=float, default=None,
                        help="Only print profiling rows slower than this many seconds.")
    parser.add_argument("--candidate-aware-agent-selection", action="store_true", default=None,
                        help="Do a second runtime extraction after candidate generation to sort agents by candidate proximity. Slower.")
    parser.add_argument("--no-candidate-aware-agent-selection", action="store_false", dest="candidate_aware_agent_selection",
                        help="Disable candidate-aware agent reordering even when the config enables it.")
    parser.add_argument("--include-drivable-polygons", action="store_true", default=None,
                        help="Extract full drivable-area polygons from map API. Much slower; route-corridor fallback is used otherwise.")
    parser.add_argument("--no-include-drivable-polygons", action="store_false", dest="include_drivable_polygons",
                        help="Disable full drivable-area polygon extraction and use the route-corridor fallback.")
    parser.add_argument("--candidate-k", type=int, default=None,
                        help="Override the finite candidate-bank size K. Use 32 for the paper/default BDSE setting.")
    parser.add_argument("--teacher-cost-eval-stride", type=int, default=None,
                        help="Evaluate expensive teacher costs every N candidate timesteps; candidates are still saved at full resolution.")
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
    if args.max_samples_per_log is not None:
        cfg["preprocess"]["max_samples_per_log"] = int(args.max_samples_per_log)
    if args.max_samples_per_log_strategy is not None:
        cfg["preprocess"]["max_samples_per_log_strategy"] = str(args.max_samples_per_log_strategy)
    if args.max_samples_per_log_block_size is not None:
        cfg["preprocess"]["max_samples_per_log_block_size"] = max(1, int(args.max_samples_per_log_block_size))
    if args.num_workers is not None:
        cfg["preprocess"]["num_workers"] = int(args.num_workers)
    if args.max_in_flight is not None:
        cfg["preprocess"]["max_in_flight"] = max(1, int(args.max_in_flight))
    if args.scenario_builder_workers is not None:
        cfg["preprocess"]["scenario_builder_workers"] = max(1, int(args.scenario_builder_workers))
    if args.temporal_frame_cache_max_entries is not None:
        cfg["preprocess"]["temporal_frame_cache_max_entries"] = max(128, int(args.temporal_frame_cache_max_entries))
    if args.temporal_frame_cache_individual_miss_threshold is not None:
        cfg["preprocess"]["temporal_frame_cache_individual_miss_threshold"] = max(0, int(args.temporal_frame_cache_individual_miss_threshold))
    if args.cache_local_scheduler is not None:
        cfg["preprocess"]["cache_local_scheduler"] = bool(args.cache_local_scheduler)
    if args.temporal_frame_cache_coalesce_bulk is not None:
        cfg["preprocess"]["temporal_frame_cache_coalesce_bulk"] = bool(args.temporal_frame_cache_coalesce_bulk)
    if args.use_process_pool:
        cfg["preprocess"]["use_process_pool"] = True
        cfg["preprocess"].setdefault("scenario_builder_use_process_pool", False)
    if args.profile:
        cfg["preprocess"]["profile"] = True
    if args.profile_threshold_s is not None:
        cfg["preprocess"]["profile_threshold_s"] = float(args.profile_threshold_s)
    if args.candidate_aware_agent_selection is not None:
        cfg["preprocess"]["candidate_aware_agent_selection"] = bool(args.candidate_aware_agent_selection)
    if args.include_drivable_polygons is not None:
        cfg.setdefault("runtime", {})["include_drivable_polygons"] = bool(args.include_drivable_polygons)
    if args.candidate_k is not None:
        cfg.setdefault("candidate", {})["K"] = max(1, int(args.candidate_k))
    if args.teacher_cost_eval_stride is not None:
        cfg.setdefault("teacher", {})["cost_eval_stride"] = max(1, int(args.teacher_cost_eval_stride))
    if args.list_splits:
        print(discover_available_splits(cfg["paths"]["data_cache_root"]))
        return

    splits = args.splits or (args.split if args.split else ["train"])
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
            num_workers=cfg.get("preprocess", {}).get("num_workers", 1),
            use_process_pool=cfg.get("preprocess", {}).get("use_process_pool", False),
        )
        print(f"[bdse] start preprocessing split={split} folders={folders} data_root={cfg['paths']['data_cache_root']} out={out_dir / split}", flush=True)
        paths = dataset.write_preprocessed_cache(
            out_dir,
            resume=resume,
            overwrite=args.overwrite,
            show_progress=not args.no_tqdm,
            num_workers=cfg.get("preprocess", {}).get("num_workers", 1),
            use_process_pool=cfg.get("preprocess", {}).get("use_process_pool", False),
        )
        total_paths.extend(paths)
        print(f"split={split}: materialized {len(paths)} sample paths under {out_dir}", flush=True)
    print(f"Done. materialized {len(total_paths)} sample paths under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
