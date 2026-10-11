"""Batch-one frozen-core latency benchmark on the user's recursive NPZ caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

from bdse.experiments.planning_budget import instrument_core, measure_call, read_runtime_npz, write_row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--subdirs", nargs="+", default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rule-only", action="store_true", help="Engineering smoke only; no learned planner claims")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--proposal-top-m", type=int, default=24)
    parser.add_argument("--mode", choices=["native", "attribution"], default="native")
    parser.add_argument("--deadline-ms", type=float, default=None)
    parser.add_argument("--limit", type=int, default=1000, help="0 means every NPZ; sample is log-balanced")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--manifest", type=Path, help="Reuse exact selected paths; overrides sampling")
    parser.add_argument("--output", type=Path, required=True, help="Must be new/empty; prevents run mixing")
    parser.add_argument("--list-only", action="store_true", help="Create workload manifest without torch/GPU")
    return parser.parse_args()


def choose_workload(root, subdirs, limit, seed):
    roots = [root / s for s in subdirs] if subdirs else [root]
    if any(not p.is_dir() for p in roots):
        raise ValueError("cache root/subdirectory missing")
    # Parent paths encode collection logs in the supplied directory layout.
    buckets = {}
    for search_root in roots:
        for path in search_root.rglob("*.npz"):
            buckets.setdefault(str(path.parent.resolve()), []).append(path.resolve())
    rng = random.Random(seed)
    logs = sorted(buckets)
    rng.shuffle(logs)
    for log in logs:
        buckets[log].sort()
        rng.shuffle(buckets[log])
    result = []
    while logs and (limit == 0 or len(result) < limit):
        active = []
        for log in logs:
            result.append(buckets[log].pop())
            if buckets[log]:
                active.append(log)
            if limit and len(result) >= limit:
                break
        logs = active
    if not result:
        raise ValueError("no NPZ files found")
    return result


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.budget <= 0 or args.proposal_top_m < args.budget:
        raise SystemExit("require 0 < budget <= proposal-top-m")
    if args.limit < 0 or args.warmup < 0 or args.repeats < 1:
        raise SystemExit("invalid limit/warmup/repeats")
    if args.deadline_ms is not None and args.deadline_ms <= 0:
        raise SystemExit("deadline must be positive")
    if not args.config.is_file():
        raise SystemExit("missing config")
    if not args.list_only and not args.rule_only and (args.checkpoint is None or not args.checkpoint.is_file()):
        raise SystemExit("provide --checkpoint, or --rule-only for engineering smoke")
    if args.rule_only and (args.checkpoint is not None or args.device != "cpu"):
        raise SystemExit("rule-only requires --device cpu and no checkpoint")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be new/empty; use a new run directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        paths = [Path(x) for x in payload["files"]]
        if not paths or any(not p.is_file() for p in paths) or len(set(paths)) != len(paths):
            raise SystemExit("workload manifest has missing/duplicate/empty paths")
    else:
        paths = choose_workload(args.cache_root, args.subdirs, args.limit, args.seed)
    paths_text = "\n".join(str(p.resolve()) for p in paths)
    manifest = {"files": [str(p.resolve()) for p in paths], "seed": args.seed,
                "selection": "reused_manifest" if args.manifest else "round_robin_shuffled_log_parents",
                "selected_path_sha256": hashlib.sha256(paths_text.encode()).hexdigest(),
                "count": len(paths), "log_count": len({p.parent for p in paths}),
                "cache_root": str(args.cache_root), "subdirs": args.subdirs}
    (args.output / "workload_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.list_only:
        print(json.dumps({"manifest": str(args.output / "workload_manifest.json"), "count": len(paths)}))
        return
    # No unrelated cache labels, batch-of-64 training, or simulated rollouts are
    # part of this benchmark. Loading is reported separately from core service.
    os.environ["BDSE_PROFILE_CLOSED_LOOP"] = "1"
    from bdse.config import load_config
    from bdse.planner.nuplan_planner import BDSEPlannerCore
    cfg = load_config(args.config, {"evidence": {"budget": args.budget},
                                   "selector": {"proposal_top_m": args.proposal_top_m}})
    if (cfg.get("selected_outcome_probe", {}) or {}).get("enabled", False):
        raise SystemExit("refuse TRAIN-only selected_outcome_probe config; use frozen deployment config")
    # Also refuse any alternative config that nests a collection-only probe.
    if (cfg.get("planner", {}).get("selected_outcome_probe", {}) or {}).get("enabled", False):
        raise SystemExit("refuse TRAIN-only probe config")
    setup_start = time.perf_counter()
    model = None
    device = args.device
    hw = {"platform": platform.platform(), "python": sys.version,
          "cpu_count": os.cpu_count(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "thread_settings": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    if not args.rule_only:
        import torch
        from bdse.external_baselines.model_factory import load_model_for_config
        from bdse.utils import resolve_torch_device, configure_torch_for_device
        torch.manual_seed(args.seed)
        device = resolve_torch_device(args.device, context="budget profiling")
        configure_torch_for_device(device)
        model = load_model_for_config(str(args.checkpoint), cfg, device)
        hw.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(), "torch_threads": torch.get_num_threads()})
        if str(device).startswith("cuda"):
            hw["gpu"] = str(torch.cuda.get_device_properties(device))
    core = BDSEPlannerCore(model=model, cfg=cfg)
    instrument_core(core, args.mode == "attribution")
    resolved = {"configuration": cfg, "hardware": hw, "mode": args.mode,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "checkpoint_sha256": file_sha(args.checkpoint) if args.checkpoint else None,
                "config_file_sha256": file_sha(args.config), "rule_only": args.rule_only,
                "setup_ms_excluded": (time.perf_counter() - setup_start) * 1000,
                "workload": manifest, "arguments": vars(args),
                "claim_limit": "cached-runtime core service only; not live end-to-end deadline or safety proof"}
    (args.output / "provenance.json").write_text(json.dumps(resolved, indent=2, default=str), encoding="utf-8")
    prefix = args.output / "trace"
    # Explicit warmup rows, never silently discard failed or first-scene calls.
    runtime0, identity0 = read_runtime_npz(paths[0])
    for i in range(args.warmup):
        measure_call(lambda: core.plan_from_runtime(runtime0), device=device, mode=args.mode,
                     context={"warmup": True, "sequence_index": i, "stream_id": "cache_core",
                              "budget": args.budget, "proposal_top_m": args.proposal_top_m,
                              "rule_only": args.rule_only, **identity0},
                     deadline_ms=args.deadline_ms, output_prefix=prefix)
    measured = 0
    for repeat in range(args.repeats):
        for path in paths:
            load_start = time.perf_counter()
            runtime, identity = read_runtime_npz(path)
            load_ms = (time.perf_counter() - load_start) * 1000
            context = {"warmup": False, "sequence_index": measured, "stream_id": "cache_core",
                       "budget": args.budget, "proposal_top_m": args.proposal_top_m,
                       "repeat": repeat, "npz_load_ms_excluded": load_ms,
                       "workload_path": str(path), "log_id": str(path.parent),
                       "rule_only": args.rule_only, "includes_runtime_feature_build": False,
                       "includes_nuplan_output_conversion": False, "includes_profile_trace_write": False,
                       **identity}
            _, row = measure_call(lambda: core.plan_from_runtime(runtime), device=device, mode=args.mode,
                                  context=context, deadline_ms=args.deadline_ms, output_prefix=prefix)
            measured += 1
            if measured % 50 == 0:
                print(json.dumps({"calls": measured, "of": len(paths) * args.repeats,
                                  "last_wall_ms": round(row["wall_ms"], 3)}), flush=True)
    from bdse.tools.summarize_planning_budget import summarize_files
    summary = summarize_files(sorted(args.output.glob("trace.*.jsonl")), deadlines=[50, 100, 200])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"finished_calls": measured, "summary": str(args.output / "summary.json")}))


if __name__ == "__main__":
    main()
