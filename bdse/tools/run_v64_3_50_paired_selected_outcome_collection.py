from __future__ import annotations

"""Collect paired closed-loop outcomes for the exact frozen RSMR winner.

Each scenario is run twice from the identical nuPlan scenario start:
  CONTROL   : every direct RSMR proposal is vetoed to the incumbent;
  TREATMENT : the first direct RSMR proposal is executed exactly once, then all
              later direct proposals are vetoed to the incumbent.

Running one scenario per nuPlan invocation is intentional: it makes the final
nuPlan aggregate row a scenario-level paired outcome without relying on private
metric-file schemas or heuristic scenario-name/token joins.  The runner is
resumable and schedules the two arms concurrently on two GPUs.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from bdse.tools.run_fixed_budget_closed_loop_suite import final_metric_row

HARD_METRICS = [
    "no_ego_at_fault_collisions",
    "time_to_collision_within_bound",
    "drivable_area_compliance",
    "driving_direction_compliance",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_tokens(path: Path) -> list[str]:
    xs = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not xs or len(xs) != len(set(xs)):
        raise ValueError("V50 token file must contain unique nonempty tokens")
    return xs


def _parse_success(text: str) -> tuple[int, int]:
    s = re.findall(r"Number of successful simulations:\s*(\d+)", text)
    f = re.findall(r"Number of failed simulations:\s*(\d+)", text)
    return (int(s[-1]) if s else -1, int(f[-1]) if f else -1)


def _score(metrics: dict[str, float], challenge: str) -> float:
    names = (["closed_loop_reactive_agents_weighted_average", "score", "final_score"]
             if challenge == "closed_loop_reactive_agents"
             else ["closed_loop_nonreactive_agents_weighted_average", "score", "final_score"])
    for name in names:
        if name in metrics and math.isfinite(float(metrics[name])):
            return float(metrics[name])
    raise RuntimeError(f"V50 cannot locate official aggregate score in metric row keys={sorted(metrics)}")


def _hard_noninferiority(control: dict[str, float], treatment: dict[str, float]) -> tuple[bool, list[str]]:
    """Return the preregistered paired hard-safety non-inferiority result.

    Missing hard metrics are an engineering failure, not an implicit PASS.  This
    prevents a nuPlan metric-schema/configuration drift from silently turning the
    causal label into score-only supervision.
    """
    missing = [m for m in HARD_METRICS if m not in control or m not in treatment]
    if missing:
        raise RuntimeError(f"V50 missing preregistered hard metrics in paired aggregate row: {missing}")
    regressions: list[str] = []
    for m in HARD_METRICS:
        c = float(control[m]); t = float(treatment[m])
        if not math.isfinite(c) or not math.isfinite(t):
            raise RuntimeError(f"V50 non-finite preregistered hard metric {m}: control={c} treatment={t}")
        if t + 1e-12 < c:
            regressions.append(m)
    return len(regressions) == 0, regressions


def _probe_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        d = ((r.get("diagnostics", {}) or {}).get("selected_outcome_probe", {}) or {})
        if d.get("enabled"):
            out.append({"iteration_index": int(r.get("iteration_index", -1)), **d})
    return out


def _validate_pair(token: str, control_diag: Path, treatment_diag: Path) -> dict[str, Any]:
    c = _probe_rows(control_diag); t = _probe_rows(treatment_diag)
    if not c or not t:
        raise RuntimeError(f"V50 {token}: missing strict selected-outcome probe diagnostics")
    c0 = [r for r in c if bool(r.get("first_proposal_now", False))]
    t0 = [r for r in t if bool(r.get("first_proposal_now", False))]
    if len(c0) != 1 or len(t0) != 1:
        raise RuntimeError(f"V50 {token}: expected exactly one first proposal marker in both arms, got {len(c0)}/{len(t0)}")
    cr, tr = c0[0], t0[0]
    # The offline feature state is the scenario-start selected event.  Refuse a
    # later first proposal because it would silently pair a different state with
    # the frozen Q/P/E OOF feature.
    if int(cr["iteration_index"]) != 0 or int(tr["iteration_index"]) != 0:
        raise RuntimeError(f"V50 {token}: first RSMR proposal must occur at planner iteration 0, got {cr['iteration_index']}/{tr['iteration_index']}")
    for key in ("proposal_action", "baseline_action", "rsmr_selected_action"):
        if int(cr[key]) != int(tr[key]):
            raise RuntimeError(f"V50 {token}: paired action identity mismatch for {key}: {cr[key]} vs {tr[key]}")
    if int(cr["post_probe_action"]) != int(cr["baseline_action"]):
        raise RuntimeError(f"V50 {token}: CONTROL did not preserve incumbent at intervention event")
    if not bool(tr.get("intervention_executed", False)) or int(tr["post_probe_action"]) != int(tr["proposal_action"]):
        raise RuntimeError(f"V50 {token}: TREATMENT did not execute exact first RSMR proposal")
    if max(int(r.get("executed_intervention_count", 0)) for r in t) != 1:
        raise RuntimeError(f"V50 {token}: TREATMENT executed more/less than one selected intervention")
    if any(bool(r.get("intervention_executed", False)) for r in c):
        raise RuntimeError(f"V50 {token}: CONTROL executed a selected intervention")
    return {
        "proposal_action": int(tr["proposal_action"]),
        "baseline_action": int(tr["baseline_action"]),
        "intervention_iteration": int(tr["iteration_index"]),
        "treatment_proposal_events": max(int(r.get("proposal_event_count", 0)) for r in t),
        "control_proposal_events": max(int(r.get("proposal_event_count", 0)) for r in c),
    }


def _run_arm(*, token: str, role: str, config: Path, checkpoint: Path, gpu: str, root: Path, args: argparse.Namespace) -> tuple[dict[str, float], Path, float]:
    # A token is skipped entirely only after its paired row was committed.  For
    # an incomplete/retried token, delete any partial nuPlan output first so a
    # stale aggregator parquet can never be mistaken for the new run.
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    diag = root / "probe_diag.jsonl"
    log = root / "run.log"
    uid = f"v50_{role}_{token}_{args.challenge}"
    cmd = [
        sys.executable, "-m", "bdse.experiments.evaluate_closed_loop",
        "--config", str(config), "--checkpoint", str(checkpoint), "--device", "cuda",
        "--challenge", args.challenge, "--metric-aggregator", args.metric_aggregator,
        "--output-dir", str(root), "--experiment-uid", uid,
        "--nuplan-module", "nuplan.planning.script.run_simulation",
        "--scenario-builder", "nuplan", "--worker", "single_machine_thread_pool",
        "--hydra-full-error", "--nuplan-data-root", str(args.nuplan_data_root),
        "--nuplan-map-root", str(args.nuplan_map_root), "--nuplan-exp-root", str(args.nuplan_exp_root),
    ]
    if args.nuplan_db_files:
        cmd += ["--nuplan-db-files", *[str(x) for x in args.nuplan_db_files]]
    else:
        cmd += ["--nuplan-db-root", str(args.nuplan_db_root)]
    token_override = "scenario_filter.scenario_tokens=" + json.dumps([token], separators=(",", ":"))
    cmd += ["--", token_override, "scenario_filter.limit_total_scenarios=1", "scenario_filter.shuffle=false",
            "scenario_filter.log_names=null", "worker.max_workers=1", "run_metric=true", "~callback.simulation_log_callback"]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "BDSE_CLOSED_LOOP_DIAG": str(diag.resolve()),
        "BDSE_STRICT_CLOSED_LOOP_DIAG": "1",
        "BDSE_REPLAN_INTERVAL_TICKS": "1",
        "BDSE_FORCE_REPLAN_EVERY_TICK": "1",
        "BDSE_SHARE_MODEL_PER_PROCESS": "1",
        "BDSE_SHARD_PLANNERS_ACROSS_GPUS": "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    started = time.time()
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    wall = time.time() - started
    text = log.read_text(encoding="utf-8", errors="replace")
    succ, fail = _parse_success(text)
    if proc.returncode != 0 or succ != 1 or fail != 0:
        tail = "\n".join(text.replace("\r", "\n").splitlines()[-30:])
        raise RuntimeError(f"V50 {role} {token} failed return={proc.returncode} successful={succ} failed={fail}\n{tail}")
    metrics, metric_file = final_metric_row(root)
    if not diag.is_file() or not diag.stat().st_size:
        raise RuntimeError(f"V50 {role} {token}: strict probe diagnostic missing")
    return metrics, diag, wall


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="V64.3.50 paired closed-loop selected-outcome collection")
    ap.add_argument("--control-config", type=Path, required=True)
    ap.add_argument("--treatment-config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--scenario-token-file", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--challenge", choices=["closed_loop_reactive_agents", "closed_loop_nonreactive_agents"], default="closed_loop_reactive_agents")
    ap.add_argument("--metric-aggregator", default=None)
    ap.add_argument("--nuplan-data-root", type=Path, required=True)
    ap.add_argument("--nuplan-map-root", type=Path, required=True)
    ap.add_argument("--nuplan-exp-root", type=Path, required=True)
    ap.add_argument("--nuplan-db-root", type=Path, default=None)
    ap.add_argument("--nuplan-db-files", type=Path, nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if not a.checkpoint.is_file(): raise FileNotFoundError(a.checkpoint)
    for p in (a.control_config, a.treatment_config, a.scenario_token_file):
        if not p.is_file(): raise FileNotFoundError(p)
    if not a.nuplan_db_files and a.nuplan_db_root is None:
        raise ValueError("provide --nuplan-db-root or --nuplan-db-files")
    if a.metric_aggregator is None:
        a.metric_aggregator = ("closed_loop_reactive_agents_weighted_average" if a.challenge == "closed_loop_reactive_agents" else "closed_loop_nonreactive_agents_weighted_average")
    gpus = [x.strip() for x in a.gpus.split(",") if x.strip()]
    if len(gpus) != 2:
        raise ValueError("V50 paired collection requires exactly two GPU ids so CONTROL/TREATMENT start concurrently")
    tokens = _read_tokens(a.scenario_token_file)
    a.output_root.mkdir(parents=True, exist_ok=True)
    rows_path = a.output_root / "paired_selected_outcomes.csv"
    existing: dict[str, dict[str, Any]] = {}
    if a.resume and rows_path.is_file():
        for r in csv.DictReader(rows_path.open(newline="", encoding="utf-8")):
            existing[str(r["scenario_token"])] = dict(r)
    rows: list[dict[str, Any]] = [existing[t] for t in tokens if t in existing]

    for idx, token in enumerate(tokens):
        if token in existing:
            print(f"[V50 resume {idx+1}/{len(tokens)}] {token}", flush=True); continue
        print(f"[V50 pair {idx+1}/{len(tokens)}] {token}", flush=True)
        roots = {role: a.output_root / "scenarios" / token / role for role in ("control", "treatment")}
        procs: dict[str, Any] = {}; results: dict[str, Any] = {}; errors: list[str] = []
        import threading
        def target(role: str, config: Path, gpu: str) -> None:
            try: results[role] = _run_arm(token=token, role=role, config=config, checkpoint=a.checkpoint, gpu=gpu, root=roots[role], args=a)
            except Exception as exc: errors.append(f"{role}: {type(exc).__name__}: {exc}")
        th = [threading.Thread(target=target, args=("control", a.control_config, gpus[0]), daemon=True),
              threading.Thread(target=target, args=("treatment", a.treatment_config, gpus[1]), daemon=True)]
        [x.start() for x in th]; [x.join() for x in th]
        if errors: raise RuntimeError(" | ".join(errors))
        cm, cd, cw = results["control"]; tm, td, tw = results["treatment"]
        ident = _validate_pair(token, cd, td)
        cs = _score(cm, a.challenge); ts = _score(tm, a.challenge); delta = ts - cs
        hard_ok, hard_reg = _hard_noninferiority(cm, tm)
        row: dict[str, Any] = {
            "scenario_token": token, **ident,
            "challenge": a.challenge, "control_score": cs, "treatment_score": ts,
            "paired_score_delta": delta, "hard_noninferior": int(hard_ok),
            "safe_benefit": int(hard_ok and delta > 0.0),
            "hard_regressions": ";".join(hard_reg), "control_wall_s": cw, "treatment_wall_s": tw,
        }
        for m in HARD_METRICS:
            if m in cm: row[f"control::{m}"] = float(cm[m])
            if m in tm: row[f"treatment::{m}"] = float(tm[m])
        rows.append(row); existing[token] = row
        ordered = [existing[t] for t in tokens if t in existing]
        _write_csv(rows_path, ordered)
        (a.output_root / "progress.json").write_text(json.dumps({"completed": len(ordered), "expected": len(tokens), "challenge": a.challenge}, indent=2), encoding="utf-8")

    ordered = [existing[t] for t in tokens]
    _write_csv(rows_path, ordered)
    summary = {
        "complete": True, "scenario_count": len(tokens), "challenge": a.challenge,
        "token_file_sha256": _sha(a.scenario_token_file),
        "control_config_sha256": _sha(a.control_config), "treatment_config_sha256": _sha(a.treatment_config),
        "checkpoint_sha256": _sha(a.checkpoint),
        "safe_benefit_count": sum(int(float(r["safe_benefit"])) for r in ordered),
        "hard_regression_count": sum(1 for r in ordered if not bool(int(float(r["hard_noninferior"])))),
        "paired_score_delta_sum": sum(float(r["paired_score_delta"]) for r in ordered),
    }
    (a.output_root / "paired_collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
