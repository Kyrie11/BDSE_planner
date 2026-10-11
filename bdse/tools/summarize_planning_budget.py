"""Tail latency/deadline sensitivity from raw per-call traces (stdlib only)."""
from __future__ import annotations
import argparse
import collections
import json
import math
from pathlib import Path
from bdse.experiments.planning_budget import latency_stats, queue_diagnostic


def summarize_files(paths, deadlines=None, periods=None):
    rows, errors, malformed = [], [], []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if row.get("schema") != "bdse_planning_budget_v1":
                        raise ValueError("unknown schema")
                    if not math.isfinite(float(row["wall_ms"])) or float(row["wall_ms"]) < 0:
                        raise ValueError("invalid service duration")
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    malformed.append({"file": str(path), "line": number, "error": str(exc)})
                    continue
                if row.get("status") != "ok":
                    errors.append(row)
                elif not row.get("warmup", False):
                    rows.append(row)
    groups = collections.defaultdict(list)
    for row in rows:
        key = (row["scope"], row["mode"], row.get("budget"), row.get("proposal_top_m"),
               row.get("cached_plan", False), row.get("rule_only", False), row.get("device"))
        groups[key].append(row)
    reports = []
    for key, bucket in sorted(groups.items(), key=lambda x: repr(x[0])):
        values = [float(r["wall_ms"]) for r in bucket]
        count_fields = sorted({k for row in bucket for k, v in row.get("counts", {}).items()
                               if isinstance(v, (int, float)) and not isinstance(v, bool)})
        stages = sorted({k for row in bucket for k in row.get("spans", {})})
        logs = {r.get("log_id") for r in bucket if r.get("log_id")}
        warmup = [r["wall_ms"] for r in _all_warmup(paths, key)]
        item = {"scope": key[0], "mode": key[1], "budget": key[2], "proposal_top_m": key[3],
                "cached_plan": key[4], "rule_only": key[5], "device": key[6],
                "wall_ms": latency_stats(values), "warmup_wall_ms": latency_stats(warmup),
                "log_count": len(logs), "low_tail_sample_warning": len(values) < 1000,
                "deadline_sensitivity_service_only": {
                    str(d): {"miss_count": sum(t > d for t in values),
                             "miss_rate": sum(t > d for t in values) / len(values)}
                    for d in (deadlines or [])},
                "counts": {field: latency_stats([r["counts"][field] for r in bucket if field in r.get("counts", {})])
                           for field in count_fields},
                "stage_exclusive_ms_attribution_only": {
                    stage: latency_stats([r["spans"][stage]["exclusive_ms"] for r in bucket if stage in r.get("spans", {})])
                    for stage in stages},
                "periodic_FIFO_diagnostics_not_real_latency": []}
        if key[1] == "native" and periods:
            streams = collections.defaultdict(list)
            for row in bucket:
                streams[(row["pid"], row.get("stream_id"))].append(row)
            for stream, sequence in sorted(streams.items(), key=lambda x: repr(x[0])):
                sequence.sort(key=lambda r: r["start_perf_ns"])
                for period in periods:
                    item["periodic_FIFO_diagnostics_not_real_latency"].append({
                        "stream": str(stream), "sequence_is_complete": False,
                        **queue_diagnostic([r["wall_ms"] for r in sequence], period, period)})
        reports.append(item)
    return {"schema": "bdse_budget_summary_v1", "trace_files": [str(p) for p in paths],
            "measured_ok_calls": len(rows), "errors": errors, "malformed_lines": malformed,
            "groups": reports,
            "limitations": ["No real deadline enforcement, input arrival or queue wait is measured.",
                            "Attribution synchronization perturbs execution; use native for service latency.",
                            "Core NPZ profile excludes live feature construction, sensors, output conversion and cache I/O.",
                            "FIFO replay is idealized; no closed-loop safety effect follows from its numbers.",
                            "Warmup and cached calls are separate; process peak CUDA memory is not per-frame peak."]}


def _all_warmup(paths, key):
    # Called only once per workload group, retains cold-start information.
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                row_key = (r.get("scope"), r.get("mode"), r.get("budget"), r.get("proposal_top_m"),
                           r.get("cached_plan", False), r.get("rule_only", False), r.get("device"))
                if row_key == key and r.get("warmup", False) and r.get("status") == "ok":
                    yield r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True, help="Files or directories (recursive)")
    parser.add_argument("--deadlines-ms", type=float, nargs="+", default=[50, 100, 200])
    parser.add_argument("--periods-ms", type=float, nargs="+", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in args.deadlines_ms + args.periods_ms):
        raise SystemExit("deadlines/periods must be positive")
    paths = set()
    for p in args.traces:
        if p.is_dir():
            paths.update(p.rglob("*.jsonl"))
        elif p.is_file():
            paths.add(p)
        else:
            raise SystemExit(f"missing trace path: {p}")
    if not paths:
        raise SystemExit("no trace files")
    result = summarize_files(sorted(paths), args.deadlines_ms, args.periods_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"measured_ok_calls": result["measured_ok_calls"], "groups": len(result["groups"]),
                      "errors": len(result["errors"]), "malformed_lines": len(result["malformed_lines"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
