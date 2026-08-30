from __future__ import annotations

"""Collect paired closed-loop outcomes for the frozen RSMR selection policy.

Each scenario is run twice from the identical nuPlan scenario start:
  CONTROL   : every direct RSMR proposal is vetoed to the incumbent;
  TREATMENT : the first live direct RSMR winner is executed exactly once, then all
              later direct proposals are vetoed to the incumbent.

For throughput, scenarios are collected in small deterministic batches.  nuPlan's
weighted metric aggregator already writes one row per scenario and
``NuPlanScenario.scenario_name`` is the scenario token, so batching preserves the
exact per-scenario official score while amortizing Hydra/ScenarioBuilder/model
startup.  CONTROL and TREATMENT still run in separate processes on separate GPUs,
use identical token batches, replan every tick, and are validated scenario by
scenario.  The V49 action integer stored for each token is only an offline cohort
slot: candidate banks are regenerated and pruned at every live state, so
cross-state slot equality is not a valid proposal-identity test.
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
import sqlite3
import subprocess
import sys
import time
from typing import Any

import pandas as pd

HARD_METRICS = [
    "no_ego_at_fault_collisions",
    "time_to_collision_within_bound",
    "drivable_area_compliance",
    "driving_direction_compliance",
]
COLLECTION_PROTOCOL_VERSION = "v50-live-selected-event-cohort-v1"
COLLECTION_ENGINE_VERSION = "v50-batched-nuplan-v1"
NO_TREATMENT_SCORE_TOL = 1.0e-9


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


def _probe_rows(path: Path, scenario_token: str | None = None) -> list[dict[str, Any]]:
    """Read selected-outcome diagnostics, optionally for one batched scenario."""
    out: list[dict[str, Any]] = []
    saw_tagged = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        row_token = str(r.get("scenario_token", "") or "").strip()
        if row_token:
            saw_tagged = True
        if scenario_token is not None and row_token and row_token != str(scenario_token):
            continue
        d = ((r.get("diagnostics", {}) or {}).get("selected_outcome_probe", {}) or {})
        if d.get("enabled"):
            out.append({
                "scenario_token": row_token,
                "iteration_index": int(r.get("iteration_index", -1)),
                "time_s": float(r.get("time_s", 0.0)),
                **d,
            })
    # Legacy single-scenario unit fixtures have no scenario_token field.  Batch
    # outputs are required to be tagged; the caller checks this contract.
    if scenario_token is not None and saw_tagged:
        out = [r for r in out if str(r.get("scenario_token", "")) == str(scenario_token)]
    return out


def _scenario_metric_rows(root: Path, tokens: list[str]) -> tuple[dict[str, dict[str, float]], Path]:
    """Read official nuPlan per-scenario aggregate rows for a batched run.

    WeightedAverageMetricAggregator stores the individual scenario rows in the
    same parquet as scenario-type and final_score rows.  For NuPlanScenario the
    ``scenario`` column is the initial lidar token, so it is an exact join key.
    """
    wanted = set(map(str, tokens))
    found: dict[str, dict[str, float]] = {}
    used: Path | None = None
    files = sorted(root.glob("**/aggregator_metric/*.parquet"))
    if not files:
        raise RuntimeError(f"V50 no aggregator_metric parquet under {root}")
    for path in files:
        df = pd.read_parquet(path)
        if "scenario" not in df.columns:
            continue
        scenario_col = df["scenario"].astype(str)
        sub = df[scenario_col.isin(wanted)]
        if sub.empty:
            continue
        used = path
        for _, sr in sub.iterrows():
            tok = str(sr["scenario"])
            if tok in found:
                raise RuntimeError(f"V50 duplicate per-scenario aggregate row for token {tok}")
            row: dict[str, float] = {}
            for key, value in sr.items():
                if isinstance(value, (bool, int, float)):
                    x = float(value)
                    if math.isfinite(x):
                        row[str(key)] = x
                else:
                    # numpy scalar numerics are common in parquet rows.
                    try:
                        x = float(value)
                    except Exception:
                        continue
                    if math.isfinite(x):
                        row[str(key)] = x
            found[tok] = row
    missing = sorted(wanted - set(found))
    extra = sorted(set(found) - wanted)
    if missing or extra or used is None:
        raise RuntimeError(
            f"V50 batched per-scenario metric join failed missing={missing[:10]} extra={extra[:10]} "
            f"found={len(found)} expected={len(wanted)} root={root}"
        )
    return found, used


def _load_frozen_proposals(path: Path) -> dict[str, int]:
    """Read the exact V49 full-set RSMR winner identity per TRAIN token."""
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tok = str(r.get("scenario_token", "")).strip()
        act = int(r.get("full_selected_action", -1))
        if tok and act >= 0:
            if tok in out:
                raise RuntimeError(f"V50 duplicate frozen proposal token in V49 candidate audit: {tok}")
            out[tok] = act
    if len(out) != 502:
        raise RuntimeError(f"V50 expected 502 frozen V49 full-set RSMR proposals, got {len(out)}")
    return out


def _finite_qpe(row: dict[str, Any], token: str, role: str) -> tuple[float, float, float]:
    vals = []
    for key in ("live_quality_value", "live_plan_control_value", "live_ego_ref_value"):
        if key not in row:
            raise RuntimeError(f"V50 {token}: {role} intervention event missing {key}")
        x = float(row[key])
        if not math.isfinite(x):
            raise RuntimeError(f"V50 {token}: {role} intervention event has non-finite {key}={x}")
        vals.append(x)
    return float(vals[0]), float(vals[1]), float(vals[2])


def _trace_identity(row: dict[str, Any], prefix: str) -> str:
    value = str(row.get(f"{prefix}_fingerprint", "")).strip()
    if not value:
        raise RuntimeError(
            f"V50 paired trace is missing {prefix}_fingerprint; "
            "use the current V50 planner instrumentation and rerun this token"
        )
    return value


def _validate_aligned_trace_prefix(
    token: str,
    control: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    *,
    stop_before_iteration: int | None,
) -> None:
    """Certify CONTROL/TREATMENT equality before any causal intervention.

    ``stop_before_iteration=None`` means certify the whole scenario.  This is
    used for the scientifically valid no-live-proposal stratum, where neither arm
    ever receives a treatment opportunity and therefore both policies must remain
    identical for the entire native nuPlan rollout.
    """
    crows = [r for r in control if stop_before_iteration is None or int(r["iteration_index"]) < int(stop_before_iteration)]
    trows = [r for r in treatment if stop_before_iteration is None or int(r["iteration_index"]) < int(stop_before_iteration)]
    if len(crows) != len(trows):
        raise RuntimeError(
            f"V50 {token}: CONTROL/TREATMENT aligned-trace length mismatch "
            f"{len(crows)}/{len(trows)} before intervention"
        )
    for j, (a, b) in enumerate(zip(crows, trows)):
        ai = int(a["iteration_index"]); bi = int(b["iteration_index"])
        if ai != bi:
            raise RuntimeError(f"V50 {token}: CONTROL/TREATMENT iteration mismatch at trace row {j}: {ai}/{bi}")
        if abs(float(a.get("time_s", 0.0)) - float(b.get("time_s", 0.0))) > 1.0e-9:
            raise RuntimeError(f"V50 {token}: CONTROL/TREATMENT time mismatch at iteration {ai}")
        if bool(a.get("proposal_exists", False)) or bool(b.get("proposal_exists", False)):
            raise RuntimeError(f"V50 {token}: proposal occurred inside certified no-treatment prefix at iteration {ai}")
        for key in ("pre_probe_action", "post_probe_action"):
            if int(a.get(key, -1)) != int(b.get(key, -2)):
                raise RuntimeError(
                    f"V50 {token}: CONTROL/TREATMENT {key} diverged at iteration {ai}: "
                    f"{a.get(key)}/{b.get(key)}"
                )
        # Integer action slots are state-local.  For same-state paired equality we
        # additionally require the full quantized trajectory/semantic certificate.
        for prefix in ("v50_pre_probe_action", "v50_post_probe_action"):
            if _trace_identity(a, prefix) != _trace_identity(b, prefix):
                raise RuntimeError(
                    f"V50 {token}: CONTROL/TREATMENT {prefix} candidate fingerprint diverged at iteration {ai}"
                )
        if bool(a.get("intervention_executed", False)) or bool(b.get("intervention_executed", False)):
            raise RuntimeError(f"V50 {token}: intervention executed inside certified no-treatment prefix at iteration {ai}")


def _validate_pair(token: str, control_diag: Path, treatment_diag: Path, offline_v49_action_slot: int) -> dict[str, Any]:
    """Validate one V50 pair and classify its live treatment-opportunity stratum.

    The 502 V49 scenes are a frozen *offline discovery cohort*.  Native closed-loop
    replay can legitimately contain scenes in which the frozen RSMR policy never
    emits a live direct proposal.  Such scenes have no defined selected-action
    treatment and must not be mislabeled as negative outcomes.  They are retained
    as an audited, label-free ``no_live_proposal`` transport stratum and excluded
    from SIOR outcome fitting.
    """
    c = _probe_rows(control_diag, token); t = _probe_rows(treatment_diag, token)
    if not c or not t:
        raise RuntimeError(f"V50 {token}: missing strict selected-outcome probe diagnostics")
    c0 = [r for r in c if bool(r.get("first_proposal_now", False))]
    t0 = [r for r in t if bool(r.get("first_proposal_now", False))]

    # Scientifically valid no-treatment stratum: neither arm ever observes a live
    # RSMR proposal.  This is not a classifier negative.  Prove the two full
    # rollouts were the same policy/state trajectory, then record eligibility=0.
    if len(c0) == 0 and len(t0) == 0:
        if any(bool(r.get("proposal_exists", False)) for r in c + t):
            raise RuntimeError(f"V50 {token}: proposal_exists occurred without a first_proposal_now marker")
        if any(int(r.get("proposal_event_count", 0)) != 0 for r in c + t):
            raise RuntimeError(f"V50 {token}: nonzero proposal_event_count without a live proposal marker")
        if any(bool(r.get("intervention_consumed", False)) or bool(r.get("intervention_executed", False)) for r in c + t):
            raise RuntimeError(f"V50 {token}: intervention state changed although no live proposal existed")
        _validate_aligned_trace_prefix(token, c, t, stop_before_iteration=None)
        return {
            "collection_protocol_version": COLLECTION_PROTOCOL_VERSION,
            "pair_status": "no_live_proposal",
            "live_intervention_eligible": 0,
            "proposal_action": -1,
            "live_proposal_action": -1,
            "offline_v49_action_slot": int(offline_v49_action_slot),
            "frozen_v49_proposal_action": int(offline_v49_action_slot),
            "live_vs_offline_action_slot_equal": 0,
            "live_proposal_fingerprint": "",
            "baseline_action": -1,
            "intervention_iteration": -1,
            "intervention_time_s": float("nan"),
            "preintervention_pair_aligned": 1,
            "treatment_proposal_events": 0,
            "control_proposal_events": 0,
        }

    # One arm seeing a proposal while the other does not is a true paired-runtime
    # violation, not a valid transport stratum.
    if len(c0) != 1 or len(t0) != 1:
        raise RuntimeError(
            f"V50 {token}: asymmetric/malformed first proposal markers CONTROL/TREATMENT={len(c0)}/{len(t0)}"
        )
    cr, tr = c0[0], t0[0]
    ci = int(cr["iteration_index"]); ti = int(tr["iteration_index"])
    if ci < 0 or ti < 0 or ci != ti:
        raise RuntimeError(f"V50 {token}: paired first-proposal iteration mismatch {ci}/{ti}")
    if abs(float(cr.get("time_s", 0.0)) - float(tr.get("time_s", 0.0))) > 1.0e-9:
        raise RuntimeError(f"V50 {token}: paired first-proposal simulation time mismatch {cr.get('time_s')}/{tr.get('time_s')}")

    _validate_aligned_trace_prefix(token, c, t, stop_before_iteration=ci)

    for key in ("proposal_action", "baseline_action", "rsmr_selected_action"):
        if int(cr[key]) != int(tr[key]):
            raise RuntimeError(f"V50 {token}: paired action identity mismatch for {key}: {cr[key]} vs {tr[key]}")

    cf = str(cr.get("v50_live_proposal_fingerprint", "")).strip()
    tf = str(tr.get("v50_live_proposal_fingerprint", "")).strip()
    if not cf or not tf:
        raise RuntimeError(f"V50 {token}: live proposal fingerprint missing in CONTROL/TREATMENT diagnostics")
    if cf != tf:
        raise RuntimeError(f"V50 {token}: CONTROL/TREATMENT live proposal fingerprint mismatch: {cf}/{tf}")
    for key in (
        "v50_live_proposal_maneuver_id",
        "v50_live_proposal_pool_original_index",
        "v50_live_proposal_maneuver",
        "v50_live_proposal_theta",
    ):
        if str(cr.get(key, "")) != str(tr.get(key, "")):
            raise RuntimeError(f"V50 {token}: CONTROL/TREATMENT live proposal semantic mismatch for {key}")

    cq, cp, ce = _finite_qpe(cr, token, "CONTROL")
    tq, tp, te = _finite_qpe(tr, token, "TREATMENT")
    for name, a, b in (("Q", cq, tq), ("P", cp, tp), ("E", ce, te)):
        if not math.isclose(a, b, rel_tol=1.0e-8, abs_tol=1.0e-10):
            raise RuntimeError(f"V50 {token}: CONTROL/TREATMENT live {name} mismatch at intervention event: {a}/{b}")

    if bool(cr.get("intervention_executed", False)) or int(cr["post_probe_action"]) != int(cr["baseline_action"]):
        raise RuntimeError(f"V50 {token}: CONTROL did not preserve incumbent at intervention event")
    if not bool(tr.get("intervention_executed", False)) or int(tr["post_probe_action"]) != int(tr["proposal_action"]):
        raise RuntimeError(f"V50 {token}: TREATMENT did not execute exact first live RSMR proposal")
    if max(int(r.get("executed_intervention_count", 0)) for r in t) != 1:
        raise RuntimeError(f"V50 {token}: TREATMENT executed more/less than one selected intervention")
    if any(bool(r.get("intervention_executed", False)) for r in c):
        raise RuntimeError(f"V50 {token}: CONTROL executed a selected intervention")
    # Fail closed on the complete one-shot state machine, not only its counter.
    for r in c:
        if bool(r.get("proposal_exists", False)) and int(r.get("post_probe_action", -1)) != int(r.get("baseline_action", -2)):
            raise RuntimeError(f"V50 {token}: CONTROL failed to preserve incumbent at a later proposal event")
    for r in t:
        if not bool(r.get("proposal_exists", False)):
            continue
        if bool(r.get("first_proposal_now", False)):
            if int(r.get("post_probe_action", -1)) != int(r.get("proposal_action", -2)):
                raise RuntimeError(f"V50 {token}: TREATMENT first proposal event did not execute the live winner")
        elif int(r.get("post_probe_action", -1)) != int(r.get("baseline_action", -2)):
            raise RuntimeError(f"V50 {token}: TREATMENT did not return to incumbent after the one-shot intervention")

    return {
        "collection_protocol_version": COLLECTION_PROTOCOL_VERSION,
        "pair_status": "eligible_intervened",
        "live_intervention_eligible": 1,
        "proposal_action": int(tr["proposal_action"]),
        "live_proposal_action": int(tr["proposal_action"]),
        "offline_v49_action_slot": int(offline_v49_action_slot),
        "frozen_v49_proposal_action": int(offline_v49_action_slot),
        "live_vs_offline_action_slot_equal": int(int(tr["proposal_action"]) == int(offline_v49_action_slot)),
        "live_proposal_fingerprint": tf,
        "live_proposal_maneuver_id": int(tr.get("v50_live_proposal_maneuver_id", -1)),
        "live_proposal_pool_original_index": int(tr.get("v50_live_proposal_pool_original_index", -1)),
        "live_proposal_maneuver": str(tr.get("v50_live_proposal_maneuver", "")),
        "live_proposal_theta": str(tr.get("v50_live_proposal_theta", "")),
        "baseline_action": int(tr["baseline_action"]),
        "intervention_iteration": int(ti),
        "intervention_time_s": float(tr.get("time_s", 0.0)),
        "preintervention_pair_aligned": 1,
        "live_quality_value": tq,
        "live_plan_control_value": tp,
        "live_ego_ref_value": te,
        "treatment_proposal_events": max(int(r.get("proposal_event_count", 0)) for r in t),
        "control_proposal_events": max(int(r.get("proposal_event_count", 0)) for r in c),
    }


def _native_db_files(args: argparse.Namespace) -> list[Path]:
    """Flatten configured native nuPlan DB inputs to exact SQLite files."""
    inputs: list[Path]
    if args.nuplan_db_files:
        inputs = [Path(x).expanduser() for x in args.nuplan_db_files]
    elif args.nuplan_db_root is not None:
        inputs = [Path(args.nuplan_db_root).expanduser()]
    else:
        return []
    files: list[Path] = []
    for p in inputs:
        if p.is_file() and p.suffix == ".db":
            files.append(p.resolve())
        elif p.is_dir():
            direct = sorted(p.glob("*.db"))
            files.extend(x.resolve() for x in (direct if direct else p.rglob("*.db")))
    return sorted(set(files), key=lambda x: str(x))


def _build_token_db_index(tokens: list[str], db_files: list[Path], cache_path: Path) -> dict[str, str] | None:
    """Build an outcome-blind token -> exact nuPlan DB index.

    NuPlanScenario.token is the initial lidar_pc token.  Querying the primary-key
    ``lidar_pc.token`` column is therefore sufficient and avoids repeatedly asking
    ScenarioBuilder to discover every TRAIN DB for every batch.  Failure to build
    the optimization index falls back to the original DB-directory inputs rather
    than changing scientific behavior.
    """
    token_sha = hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()
    db_sig = [
        {"path": str(p), "size": int(p.stat().st_size), "mtime_ns": int(p.stat().st_mtime_ns)}
        for p in db_files
    ]
    signature = hashlib.sha256(json.dumps({"token_sha": token_sha, "db": db_sig}, sort_keys=True).encode("utf-8")).hexdigest()
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            mapping = {str(k): str(v) for k, v in (cached.get("mapping", {}) or {}).items()}
            if cached.get("signature") == signature and set(mapping) == set(tokens) and all(Path(v).is_file() for v in mapping.values()):
                print(f"[V50 DB index] reused exact token->DB cache: {cache_path}", flush=True)
                return mapping
        except Exception:
            pass

    wanted: dict[bytes, str] = {}
    try:
        for tok in tokens:
            raw = bytes.fromhex(tok)
            wanted[raw] = tok
    except Exception as exc:
        print(f"[V50 DB index] disabled: scenario token is not canonical hex: {exc}", flush=True)
        return None
    mapping: dict[str, str] = {}
    unresolved = set(tokens)
    try:
        for db in db_files:
            if not unresolved:
                break
            raw_unresolved = [bytes.fromhex(t) for t in sorted(unresolved)]
            uri = f"file:{db}?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=5.0)
            try:
                # Stay comfortably below common SQLite variable limits.
                for off in range(0, len(raw_unresolved), 400):
                    chunk = raw_unresolved[off:off + 400]
                    marks = ",".join("?" for _ in chunk)
                    rows = con.execute(f"SELECT token FROM lidar_pc WHERE token IN ({marks})", chunk).fetchall()
                    for (raw_token,) in rows:
                        raw_bytes = bytes(raw_token)
                        tok = wanted.get(raw_bytes, raw_bytes.hex())
                        if tok not in unresolved:
                            continue
                        if tok in mapping and mapping[tok] != str(db):
                            raise RuntimeError(f"token {tok} appeared in multiple nuPlan DBs: {mapping[tok]} and {db}")
                        mapping[tok] = str(db)
                        unresolved.discard(tok)
            finally:
                con.close()
    except Exception as exc:
        print(f"[V50 DB index] disabled after SQLite lookup error: {type(exc).__name__}: {exc}", flush=True)
        return None
    if unresolved:
        print(
            f"[V50 DB index] disabled: could not map {len(unresolved)}/{len(tokens)} tokens; "
            f"examples={sorted(unresolved)[:5]}. Falling back to configured DB directories.",
            flush=True,
        )
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"signature": signature, "mapping": mapping}, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[V50 DB index] mapped {len(mapping)}/{len(tokens)} tokens across "
        f"{len(set(mapping.values()))} exact DB files; cache={cache_path}",
        flush=True,
    )
    return mapping


def _run_arm_batch(
    *,
    tokens: list[str],
    role: str,
    config: Path,
    checkpoint: Path,
    gpu: str,
    root: Path,
    args: argparse.Namespace,
    db_files_override: list[Path] | None = None,
) -> tuple[dict[str, dict[str, float]], Path, float]:
    """Run one CONTROL/TREATMENT batch in a single nuPlan process."""
    if not tokens:
        raise ValueError("V50 empty batch")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    diag = root / "probe_diag.jsonl"
    log = root / "run.log"
    uid = f"v50_{role}_batch_{tokens[0]}_{len(tokens)}_{args.challenge}"
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
    if db_files_override:
        cmd += ["--nuplan-db-files", *[str(x) for x in db_files_override]]
    elif args.nuplan_db_files:
        cmd += ["--nuplan-db-files", *[str(x) for x in args.nuplan_db_files]]
    else:
        cmd += ["--nuplan-db-root", str(args.nuplan_db_root)]
    token_override = "scenario_filter.scenario_tokens=" + json.dumps(tokens, separators=(",", ":"))
    cmd += [
        "--", token_override, f"scenario_filter.limit_total_scenarios={len(tokens)}",
        "scenario_filter.shuffle=false", "scenario_filter.log_names=null",
        "worker.max_workers=1", "run_metric=true", "~callback.simulation_log_callback",
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "BDSE_CLOSED_LOOP_DIAG": str(diag.resolve()),
        "BDSE_STRICT_CLOSED_LOOP_DIAG": "1",
        "BDSE_REQUIRE_SCENARIO_FOR_DIAG": "1",
        "BDSE_SELECTED_OUTCOME_DIAG_ONLY": "1",
        "BDSE_REPLAN_INTERVAL_TICKS": "1",
        "BDSE_FORCE_REPLAN_EVERY_TICK": "1",
        # This is now useful: all planners in the batch reuse one read-only CUDA
        # model inside the same process.
        "BDSE_SHARE_MODEL_PER_PROCESS": "1",
        "BDSE_SERIALIZE_GPU_INFERENCE": "0",
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
    if proc.returncode != 0 or succ != len(tokens) or fail != 0:
        tail = "\n".join(text.replace("\r", "\n").splitlines()[-40:])
        raise RuntimeError(
            f"V50 {role} batch failed return={proc.returncode} successful={succ} failed={fail} "
            f"expected={len(tokens)} tokens={tokens[:4]}...\n{tail}"
        )
    metrics_by_token, _ = _scenario_metric_rows(root, tokens)
    if not diag.is_file() or not diag.stat().st_size:
        raise RuntimeError(f"V50 {role} batch: strict probe diagnostic missing")
    tagged = {str(r.get("scenario_token", "")) for r in _probe_rows(diag) if str(r.get("scenario_token", ""))}
    missing_diag = sorted(set(tokens) - tagged)
    if missing_diag:
        raise RuntimeError(
            f"V50 {role} batch diagnostics missing native scenario-token tags for {missing_diag[:10]}"
        )
    return metrics_by_token, diag, wall


def _validate_native_nuplan_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed on the V50 native nuPlan input contract.

    BDSE NPZ caches are deliberately outside this function.  Closed-loop
    ScenarioBuilder must consume original nuPlan SQLite log DBs and maps.
    Flat split directories containing ``*.db`` directly are the preferred
    layout; nested roots remain supported through evaluate_closed_loop's DB
    expansion layer.
    """
    data_root = Path(args.nuplan_data_root).expanduser()
    map_root = Path(args.nuplan_map_root).expanduser()
    exp_root = Path(args.nuplan_exp_root).expanduser()
    if not data_root.is_dir():
        raise FileNotFoundError(f"V50 native nuPlan data root not found: {data_root}")
    if not map_root.is_dir():
        raise FileNotFoundError(f"V50 nuPlan map root not found: {map_root}")
    meta = map_root / "nuplan-maps-v1.0.json"
    if not meta.is_file():
        raise FileNotFoundError(f"V50 nuPlan map metadata not found: {meta}")
    exp_root.mkdir(parents=True, exist_ok=True)

    db_inputs: list[Path] = []
    mode: str
    if args.nuplan_db_files:
        mode = "db_files"
        db_inputs = [Path(x).expanduser() for x in args.nuplan_db_files]
    elif args.nuplan_db_root is not None:
        mode = "db_root"
        db_inputs = [Path(args.nuplan_db_root).expanduser()]
    else:
        raise ValueError("provide --nuplan-db-root or --nuplan-db-files")

    missing = [str(p) for p in db_inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"V50 native nuPlan DB input(s) not found: {missing}")
    direct_db_counts: dict[str, int] = {}
    recursive_db_counts: dict[str, int] = {}
    for p in db_inputs:
        if p.is_file():
            if p.suffix != ".db":
                raise ValueError(f"V50 DB file input is not .db: {p}")
            direct_db_counts[str(p)] = 1
            recursive_db_counts[str(p)] = 1
            continue
        direct = sum(1 for _ in p.glob("*.db"))
        recursive = direct if direct else sum(1 for _ in p.rglob("*.db"))
        if recursive <= 0:
            raise FileNotFoundError(
                f"V50 found no native nuPlan .db files under {p}. "
                "Do not pass a BDSE NPZ cache directory to closed-loop ScenarioBuilder."
            )
        direct_db_counts[str(p)] = direct
        recursive_db_counts[str(p)] = recursive
    return {
        "mode": mode,
        "data_root": str(data_root),
        "map_root": str(map_root),
        "exp_root": str(exp_root),
        "db_inputs": [str(p) for p in db_inputs],
        "direct_db_counts": direct_db_counts,
        "recursive_db_counts": recursive_db_counts,
    }


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
    ap.add_argument("--frozen-proposal-audit", type=Path, required=True, help="V49 OOF candidate-state audit with exact full_selected_action per token")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--challenge", choices=["closed_loop_reactive_agents", "closed_loop_nonreactive_agents"], default="closed_loop_reactive_agents")
    ap.add_argument("--metric-aggregator", default=None)
    ap.add_argument("--nuplan-data-root", type=Path, required=True)
    ap.add_argument("--nuplan-map-root", type=Path, required=True)
    ap.add_argument("--nuplan-exp-root", type=Path, required=True)
    ap.add_argument("--nuplan-db-root", type=Path, default=None)
    ap.add_argument("--nuplan-db-files", type=Path, nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=16, help="Scenarios per nuPlan process/arm; scientific protocol is unchanged")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if not a.checkpoint.is_file(): raise FileNotFoundError(a.checkpoint)
    for p in (a.control_config, a.treatment_config, a.scenario_token_file, a.frozen_proposal_audit):
        if not p.is_file(): raise FileNotFoundError(p)
    native_layout = _validate_native_nuplan_inputs(a)
    print("[V50 native nuPlan input contract] " + json.dumps(native_layout, sort_keys=True), flush=True)
    if a.metric_aggregator is None:
        a.metric_aggregator = ("closed_loop_reactive_agents_weighted_average" if a.challenge == "closed_loop_reactive_agents" else "closed_loop_nonreactive_agents_weighted_average")
    gpus = [x.strip() for x in a.gpus.split(",") if x.strip()]
    if len(gpus) != 2:
        raise ValueError("V50 paired collection requires exactly two GPU ids so CONTROL/TREATMENT start concurrently")
    if int(a.batch_size) < 1:
        raise ValueError("V50 --batch-size must be >=1")
    tokens = _read_tokens(a.scenario_token_file)
    frozen_proposals = _load_frozen_proposals(a.frozen_proposal_audit)
    if set(tokens) != set(frozen_proposals):
        raise RuntimeError(f"V50 frozen token/proposal population mismatch tokens={len(tokens)} proposals={len(frozen_proposals)}")
    a.output_root.mkdir(parents=True, exist_ok=True)
    all_db_files = _native_db_files(a)
    token_db_index = _build_token_db_index(tokens, all_db_files, a.output_root / "token_db_index.json") if all_db_files else None
    rows_path = a.output_root / "paired_selected_outcomes.csv"
    existing: dict[str, dict[str, Any]] = {}
    if a.resume and rows_path.is_file():
        for r in csv.DictReader(rows_path.open(newline="", encoding="utf-8")):
            version = str(r.get("collection_protocol_version", "")).strip()
            if version != COLLECTION_PROTOCOL_VERSION:
                raise RuntimeError(
                    "V50 paired output was created by an older protocol revision "
                    f"({version or 'missing version'}). Remove only the V50 paired_train directory and rerun; "
                    "do not mix pre-amendment and live-eligibility rows."
                )
            engine = str(r.get("collection_engine_version", "")).strip()
            if engine != COLLECTION_ENGINE_VERSION:
                raise RuntimeError(
                    "V50 paired output was created by a different execution engine "
                    f"({engine or 'legacy single-scenario engine'}). Remove only the V50 paired_train directory and rerun; "
                    "the scientific protocol is unchanged but engine outputs must not be mixed."
                )
            requested_batch = str(r.get("requested_batch_size", "")).strip()
            if requested_batch and int(float(requested_batch)) != int(a.batch_size):
                raise RuntimeError(
                    "V50 paired output was created with a different requested batch size "
                    f"({requested_batch} vs current {a.batch_size}). Remove only the V50 paired_train directory and rerun; "
                    "do not mix execution-engine settings inside one frozen evidence collection."
                )
            if not requested_batch:
                raise RuntimeError(
                    "V50 paired output predates the frozen batched-engine provenance field requested_batch_size. "
                    "Remove only the V50 paired_train directory and rerun."
                )
            existing[str(r["scenario_token"])] = dict(r)
    rows: list[dict[str, Any]] = [existing[t] for t in tokens if t in existing]

    pending = [t for t in tokens if t not in existing]
    batch_size = int(a.batch_size)
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    completed_before = len(existing)
    for batch_idx, batch_tokens in enumerate(batches):
        print(
            f"[V50 batch {batch_idx+1}/{len(batches)}] tokens={len(batch_tokens)} "
            f"progress={completed_before + batch_idx * batch_size}/{len(tokens)} first={batch_tokens[0]}",
            flush=True,
        )
        batch_root = a.output_root / "batches" / f"batch_{batch_idx:04d}_{batch_tokens[0]}"
        roots = {role: batch_root / role for role in ("control", "treatment")}
        batch_db_files = (
            sorted({Path(token_db_index[t]) for t in batch_tokens}, key=lambda x: str(x))
            if token_db_index is not None else None
        )
        if batch_db_files is not None:
            print(f"[V50 batch DB restriction] exact_db_files={len(batch_db_files)}", flush=True)
        results: dict[str, Any] = {}
        errors: list[str] = []
        import threading

        def target(role: str, config: Path, gpu: str) -> None:
            try:
                results[role] = _run_arm_batch(
                    tokens=batch_tokens, role=role, config=config, checkpoint=a.checkpoint,
                    gpu=gpu, root=roots[role], args=a, db_files_override=batch_db_files,
                )
            except Exception as exc:
                errors.append(f"{role}: {type(exc).__name__}: {exc}")

        th = [
            threading.Thread(target=target, args=("control", a.control_config, gpus[0]), daemon=True),
            threading.Thread(target=target, args=("treatment", a.treatment_config, gpus[1]), daemon=True),
        ]
        [x.start() for x in th]
        [x.join() for x in th]
        if errors:
            raise RuntimeError(" | ".join(errors))
        cm_by, cd, cw = results["control"]
        tm_by, td, tw = results["treatment"]
        print(
            f"[V50 batch done] size={len(batch_tokens)} wall_control={cw:.1f}s wall_treatment={tw:.1f}s "
            f"amortized_pair_wall={max(cw, tw)/max(len(batch_tokens),1):.1f}s/scenario",
            flush=True,
        )
        for local_idx, token in enumerate(batch_tokens):
            print(
                f"[V50 validate {completed_before + batch_idx * batch_size + local_idx + 1}/{len(tokens)}] {token}",
                flush=True,
            )
            cm = cm_by[token]
            tm = tm_by[token]
            ident = _validate_pair(token, cd, td, frozen_proposals[token])
            cs = _score(cm, a.challenge)
            ts = _score(tm, a.challenge)
            delta = ts - cs
            row: dict[str, Any] = {
                "scenario_token": token,
                **ident,
                "collection_engine_version": COLLECTION_ENGINE_VERSION,
                "challenge": a.challenge,
                "control_score": cs,
                "treatment_score": ts,
                "batch_size": len(batch_tokens),
                "requested_batch_size": batch_size,
                "batch_index": batch_idx,
                "control_batch_wall_s": cw,
                "treatment_batch_wall_s": tw,
                "amortized_pair_wall_s": max(cw, tw) / max(len(batch_tokens), 1),
            }
            if int(ident["live_intervention_eligible"]) == 1:
                hard_ok, hard_reg = _hard_noninferiority(cm, tm)
                row.update({
                    "paired_score_delta": delta, "hard_noninferior": int(hard_ok),
                    "safe_benefit": int(hard_ok and delta > 0.0),
                    "hard_regressions": ";".join(hard_reg),
                    "no_treatment_metric_equivalent": "",
                })
            else:
                hard_same, hard_diff = _hard_noninferiority(cm, tm)
                reverse_same, reverse_diff = _hard_noninferiority(tm, cm)
                metric_equiv = bool(abs(delta) <= NO_TREATMENT_SCORE_TOL and hard_same and reverse_same)
                if not metric_equiv:
                    raise RuntimeError(
                        f"V50 {token}: no-live-proposal arms should be identical but official metrics diverged: "
                        f"score_delta={delta} hard_control_to_treatment={hard_diff} "
                        f"hard_treatment_to_control={reverse_diff}"
                    )
                row.update({
                    "paired_score_delta": "", "hard_noninferior": "", "safe_benefit": "",
                    "hard_regressions": "", "no_treatment_metric_equivalent": 1,
                })
            for m in HARD_METRICS:
                if m in cm:
                    row[f"control::{m}"] = float(cm[m])
                if m in tm:
                    row[f"treatment::{m}"] = float(tm[m])
            existing[token] = row
            ordered_now = [existing[t] for t in tokens if t in existing]
            _write_csv(rows_path, ordered_now)
            (a.output_root / "progress.json").write_text(
                json.dumps({
                    "completed": len(ordered_now), "expected": len(tokens), "challenge": a.challenge,
                    "collection_engine_version": COLLECTION_ENGINE_VERSION, "batch_size": batch_size,
                }, indent=2),
                encoding="utf-8",
            )

    ordered = [existing[t] for t in tokens]
    _write_csv(rows_path, ordered)
    summary = {
        "complete": True, "scenario_count": len(tokens), "challenge": a.challenge,
        "token_file_sha256": _sha(a.scenario_token_file),
        "frozen_proposal_audit_sha256": _sha(a.frozen_proposal_audit),
        "collection_protocol_version": COLLECTION_PROTOCOL_VERSION,
        "collection_engine_version": COLLECTION_ENGINE_VERSION,
        "batch_size": int(a.batch_size),
        "nuplan_process_count_per_arm": int(math.ceil(len(tokens) / max(int(a.batch_size), 1))),
        "exact_token_db_index_enabled": bool(token_db_index is not None),
        "offline_discovery_cohort_count": len(tokens),
        "live_intervention_eligible_count": sum(int(float(r.get("live_intervention_eligible", 0))) for r in ordered),
        "no_live_proposal_count": sum(1 for r in ordered if str(r.get("pair_status", "")) == "no_live_proposal"),
        "live_intervention_eligibility_rate": float(sum(int(float(r.get("live_intervention_eligible", 0))) for r in ordered) / max(len(tokens), 1)),
        "intervention_iteration_min": min([int(float(r["intervention_iteration"])) for r in ordered if int(float(r.get("live_intervention_eligible", 0))) == 1], default=-1),
        "intervention_iteration_max": max([int(float(r["intervention_iteration"])) for r in ordered if int(float(r.get("live_intervention_eligible", 0))) == 1], default=-1),
        "control_config_sha256": _sha(a.control_config), "treatment_config_sha256": _sha(a.treatment_config),
        "checkpoint_sha256": _sha(a.checkpoint),
        "native_nuplan_layout": native_layout,
        "safe_benefit_count": sum(int(float(r["safe_benefit"])) for r in ordered if str(r.get("safe_benefit", "")).strip() != ""),
        "hard_regression_count": sum(1 for r in ordered if str(r.get("hard_noninferior", "")).strip() != "" and not bool(int(float(r["hard_noninferior"])))),
        "paired_score_delta_sum_live_eligible": sum(float(r["paired_score_delta"]) for r in ordered if str(r.get("paired_score_delta", "")).strip() != ""),
    }
    (a.output_root / "paired_collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
