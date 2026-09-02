"""Small server-side sentinel for the V64.3.50.5 nuPlan metric-safety repair.

The sentinel is engineering-only and is never consumed by PIOR fitting. It
replays the two tokens on which V50.4 exposed the nuPlan metric race plus two
fixed timestamp-compatible TRAIN tokens under the same 4-worker simulation
setting and the metric-safe child entrypoint.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bdse.tools import run_v64_3_50_pior_paired_closed_loop as base
from bdse.tools import run_v64_3_50_5_pior_paired_closed_loop as safe_shim

KNOWN_FAILURE_TOKENS = ["a2326fd4694d5191", "fbd37ced14a15418"]
SAFE_MARKER = "[BDSE-PIOR-METRIC-SAFE]"


def _pick_tokens(tokens: list[str], meta: dict[str, dict], n: int = 4) -> list[str]:
    chosen: list[str] = []
    for tok in KNOWN_FAILURE_TOKENS:
        if tok not in meta:
            raise RuntimeError(f"STOP V64.3.50.5 preflight: known V50.4 failure token missing: {tok}")
        chosen.append(tok)

    def compatible(tok: str) -> bool:
        ts = int(meta[tok]["timestamp_us"])
        return all(
            abs(ts - int(meta[x]["timestamp_us"])) > base.PIOR_ANCHOR_MATCH_TOLERANCE_US
            for x in chosen
        )

    for tok in tokens:
        if tok not in chosen and compatible(tok):
            chosen.append(tok)
        if len(chosen) >= n:
            break
    if len(chosen) != n:
        raise RuntimeError(f"STOP V64.3.50.5 preflight: could not construct {n} timestamp-compatible sentinel tokens")
    return chosen


def _marker_ok(root: Path) -> bool:
    log = root / "batches" / "batch_0000" / "run.log"
    return log.is_file() and SAFE_MARKER in log.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--treatment-config", type=Path, required=True)
    ap.add_argument("--control-config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--nuplan-root", type=Path, required=True)
    ap.add_argument("--challenge", default="closed_loop_nonreactive_agents")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--workers-per-arm", type=int, default=4)
    ap.add_argument("--heartbeat-seconds", type=float, default=60.0)
    ap.add_argument("--output-report", type=Path, required=True)
    args = ap.parse_args()

    safe_shim._assert_frozen_base_runner()
    base.subprocess = safe_shim._SubprocessProxy()
    tokens, meta, _ = base._manifest(args.manifest)
    sentinels = _pick_tokens(tokens, meta)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if len(gpus) < 2:
        raise RuntimeError("STOP V64.3.50.5 preflight requires two GPUs, one per causal arm")

    args.output_root.mkdir(parents=True, exist_ok=True)
    cfgs = {"treatment": args.treatment_config, "control": args.control_config}

    def run_arm(arm: str, gpu: str):
        cfg = cfgs[arm]
        return base._run_batch(
            arm=arm,
            gpu=gpu,
            cfg=cfg,
            checkpoint=args.checkpoint,
            tokens=sentinels,
            meta=meta,
            nuplan_root=args.nuplan_root,
            challenge=args.challenge,
            arm_root=args.output_root / arm,
            workers=args.workers_per_arm,
            batch_index=0,
            batch_count=1,
            resume=False,
            heartbeat_seconds=args.heartbeat_seconds,
            serialize_gpu_inference=False,
            profile_closed_loop=False,
            cfg_sha=base._sha256(cfg),
            ckpt_sha=base._sha256(args.checkpoint),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_t = pool.submit(run_arm, "treatment", gpus[0])
        fut_c = pool.submit(run_arm, "control", gpus[1])
        _, cert_t = fut_t.result()
        _, cert_c = fut_c.result()

    marker = {
        "treatment": _marker_ok(args.output_root / "treatment"),
        "control": _marker_ok(args.output_root / "control"),
    }
    if not all(marker.values()):
        raise RuntimeError(f"STOP V64.3.50.5 preflight: metric-safe child marker missing: {marker}")

    report = {
        "schema": "v64.3.50.5-pior-metric-safety-preflight-v1",
        "engineering_only_not_consumed_by_fit": True,
        "sentinel_tokens": sentinels,
        "known_v50_4_metric_failure_tokens": KNOWN_FAILURE_TOKENS,
        "workers_per_arm": args.workers_per_arm,
        "metric_safe_marker": marker,
        "treatment_certificate": cert_t,
        "control_certificate": cert_c,
        "pass": True,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"pass": True, "sentinel_tokens": sentinels, "workers_per_arm": args.workers_per_arm}, indent=2))


if __name__ == "__main__":
    main()
