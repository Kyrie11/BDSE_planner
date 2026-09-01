from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SAFETY_METRICS = (
    "no_ego_at_fault_collisions",
    "time_to_collision_within_bound",
    "drivable_area_compliance",
)
SCORE_CANDIDATES = (
    "score",
    "closed_loop_nonreactive_agents_weighted_average",
    "closed_loop_reactive_agents_weighted_average",
    "final_score",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _token_sha(tokens: list[str]) -> str:
    return hashlib.sha256(("\n".join(tokens) + "\n").encode("utf-8")).hexdigest()


def _parse_success(text: str) -> tuple[int, int]:
    s = re.findall(r"Number of successful simulations:\s*(\d+)", text)
    f = re.findall(r"Number of failed simulations:\s*(\d+)", text)
    return (int(s[-1]) if s else -1, int(f[-1]) if f else -1)


def _manifest(path: Path) -> tuple[list[str], dict[str, dict[str, Any]], list[str]]:
    r = json.loads(path.read_text(encoding="utf-8"))
    rows = list(r.get("rows", []))
    tokens = [str(x["scenario_token"]) for x in rows]
    if len(tokens) != 502 or len(set(tokens)) != 502:
        raise RuntimeError(f"V64.3.50 PIOR manifest must be exact 502 unique frozen RSMR proposals, got {len(tokens)}/{len(set(tokens))}")
    meta = {str(x["scenario_token"]): dict(x) for x in rows}
    raw_files = [str(x) for x in r.get("raw_db_files", [])]
    # A row may map to one exact DB or to a small safe DB set (e.g. multiple
    # nuPlan crop DBs sharing the same stable log name). The *scenario token*
    # remains the scientific identity and is still passed as an exact nuPlan
    # scenario_filter. Never require a guessed filename == log_name equality.
    for t in tokens:
        row_files = [str(x) for x in meta[t].get("raw_db_files", []) if str(x)]
        if not row_files:
            one = str(meta[t].get("raw_db_file", ""))
            row_files = [one] if one else []
        if not row_files:
            raise RuntimeError(
                f"V64.3.50 PIOR manifest has no safe raw DB candidate set for token={t}; "
                "regenerate the manifest with build_v64_3_50_pior_train_manifest."
            )
        meta[t]["raw_db_files"] = row_files
        for db in row_files:
            if not Path(db).is_file():
                raise FileNotFoundError(f"V64.3.50 PIOR raw DB file disappeared for token={t}: {db}")
    if not raw_files:
        seen: set[str] = set()
        for t in tokens:
            for db in meta[t]["raw_db_files"]:
                if db not in seen:
                    seen.add(db); raw_files.append(db)
    for db in raw_files:
        if not Path(db).is_file():
            raise FileNotFoundError(f"V64.3.50 PIOR raw DB file disappeared: {db}")
    return tokens, meta, raw_files


def _as_str(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v).strip()


def _numeric_row(row: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in row.items():
        if isinstance(v, (bool, int, float, np.integer, np.floating)):
            try:
                x = float(v)
            except Exception:
                continue
            if math.isfinite(x):
                out[str(k)] = x
    return out


def _extract_scenario_metrics(root: Path, tokens: list[str], meta: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], str]:
    expected = set(tokens)
    alias: dict[tuple[str, str], str] = {}
    single_alias: dict[str, set[str]] = {}
    for t in tokens:
        r = meta[t]
        log = _as_str(r.get("log_name"))
        sc = _as_str(r.get("scenario_name"))
        if log and sc:
            alias[(log, sc)] = t
        if sc:
            single_alias.setdefault(sc, set()).add(t)

    candidates: list[tuple[int, Path, dict[str, dict[str, Any]]]] = []
    for path in sorted(root.glob("**/aggregator_metric/*.parquet")):
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if "scenario" in df.columns:
            df = df[df["scenario"].astype(str) != "final_score"]
        got: dict[str, dict[str, Any]] = {}
        ambiguous = False
        for _, row in df.iterrows():
            tok = ""
            for c in ("scenario_token", "token", "scenario", "scenario_name"):
                if c in row.index:
                    val = _as_str(row[c])
                    if val in expected:
                        tok = val
                        break
            if not tok:
                log = _as_str(row["log_name"]) if "log_name" in row.index else ""
                sc = _as_str(row["scenario"]) if "scenario" in row.index else _as_str(row["scenario_name"]) if "scenario_name" in row.index else ""
                tok = alias.get((log, sc), "")
                if not tok and sc in single_alias and len(single_alias[sc]) == 1:
                    tok = next(iter(single_alias[sc]))
            if not tok:
                matches = {_as_str(v) for v in row.values if _as_str(v) in expected}
                if len(matches) == 1:
                    tok = next(iter(matches))
            if not tok:
                continue
            if tok in got:
                ambiguous = True
                break
            ident = {
                str(k): _as_str(row[k])
                for k in row.index
                if not isinstance(row[k], (bool, int, float, np.integer, np.floating))
            }
            got[tok] = {"scenario_token": tok, "identity": ident, "metrics": _numeric_row(row)}
        if not ambiguous:
            candidates.append((len(got), path, got))
    if not candidates:
        raise RuntimeError(f"no usable per-scenario aggregator parquet under {root}")
    candidates.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    count, path, got = candidates[0]
    missing = sorted(expected - set(got))
    extra = sorted(set(got) - expected)
    if count != len(tokens) or missing or extra:
        raise RuntimeError(
            f"PIOR cannot establish exact token identity from aggregator metrics: matched={count}/{len(tokens)} "
            f"missing={missing[:10]} extra={extra[:10]} file={path}. STOP instead of relying on row order."
        )
    return got, str(path)


def _write_normalized_metrics(path: Path, metrics: dict[str, dict[str, Any]], tokens: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(json.dumps(metrics[tok], sort_keys=True) + "\n")


def _read_normalized_metrics(path: Path, tokens: list[str]) -> dict[str, dict[str, Any]]:
    got: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            tok = str(r.get("scenario_token", ""))
            if not tok or tok in got:
                raise RuntimeError(f"invalid/duplicate normalized PIOR metric token in {path}: {tok!r}")
            got[tok] = r
    if set(got) != set(tokens):
        raise RuntimeError(f"normalized PIOR metrics token mismatch in {path}: {len(got)}/{len(tokens)}")
    return got


def _count_probe_fires(diag: Path) -> int:
    if not diag.is_file():
        return -1
    fired = 0
    with diag.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if bool(row.get("pior_probe_fired", False)):
                fired += 1
                continue
            td = ((row.get("diagnostics", {}) or {}).get("tournament", {}) or {})
            fired += int(bool(td.get("pior_probe_fired", False)))
    return fired


def _tail_log(path: Path, max_bytes: int = 32768) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    text = text.replace("\r", "\n")
    ansi = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
    lines = [ansi.sub("", x).strip() for x in text.splitlines() if x.strip()]
    return lines[-1][-240:] if lines else ""


def _phase(log: Path) -> str:
    try:
        size = log.stat().st_size
        with log.open("rb") as f:
            f.seek(max(0, size - 131072))
            text = f.read().decode("utf-8", errors="replace")
    except Exception:
        return "process-starting"
    if "Number of successful simulations:" in text:
        return "metrics/finalizing"
    if "[planner-ready]" in text or "BDSEnuPlanPlanner device:" in text:
        return "planner-loaded/simulating"
    return "nuplan-init/scenario-build" if text.strip() else "process-starting"


def _gpu_stats(gpu: str) -> tuple[str, str]:
    try:
        p = subprocess.run(
            [
                "nvidia-smi", "-i", str(gpu),
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=2.0,
        )
        if p.returncode == 0 and p.stdout.strip():
            util, mem = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")[:2]]
            return f"{util}%", f"{mem}MB"
    except Exception:
        pass
    return "n/a", "n/a"


def _batch_raw_files(tokens: list[str], meta: dict[str, dict[str, Any]]) -> list[str]:
    """Union the safe per-token DB candidate sets for one deterministic batch."""
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        row_files = [str(x) for x in meta[tok].get("raw_db_files", []) if str(x)]
        if not row_files:
            one = str(meta[tok].get("raw_db_file", ""))
            row_files = [one] if one else []
        if not row_files:
            raise RuntimeError(f"PIOR batch has no raw DB candidates for token={tok}")
        for p in row_files:
            if not Path(p).is_file():
                raise RuntimeError(f"PIOR batch raw DB candidate disappeared for token={tok}: {p}")
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out



def _probe_target_payload(tokens: list[str], meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_ts: set[int] = set()
    for tok in tokens:
        row = meta[tok]
        ts = int(row.get("timestamp_us", 0) or 0)
        cache_it = int(row.get("cache_iteration", -1))
        act = int(row.get("full_selected_action", -1))
        if ts <= 0 or cache_it != 0 or act < 0:
            raise RuntimeError(
                f"V64.3.50.1 PIOR invalid frozen target token={tok}: timestamp_us={ts} cache_iteration={cache_it} action={act}"
            )
        if ts in seen_ts:
            raise RuntimeError(f"V64.3.50.1 PIOR duplicate start timestamp_us={ts} inside batch")
        seen_ts.add(ts)
        rows.append({"scenario_token": str(tok), "timestamp_us": ts, "full_selected_action": act})
    return {
        "algorithm_version": "V64.3.50.1-EAF-ICER-PIOR-ENGINEERING-REPAIR",
        "identity": "exact_V49_manifest_it000000_timestamp_and_frozen_action",
        "targets": rows,
    }


def _payload_sha256(payload: dict[str, Any]) -> str:
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_probe_target_file(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return _sha256(path)


def _validate_probe_events(
    path: Path,
    *,
    tokens: list[str],
    meta: dict[str, dict[str, Any]],
    arm: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"V64.3.50.1 PIOR missing probe event file: {path}")
    by_token: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            tok = str(row.get("scenario_token", ""))
            if not tok or tok not in meta:
                raise RuntimeError(f"V64.3.50.1 PIOR invalid probe token at {path}:{line_no}: {tok!r}")
            if tok in by_token:
                raise RuntimeError(f"V64.3.50.1 PIOR duplicate probe event for token={tok}")
            if not bool(row.get("pior_probe_fired", False)) or int(row.get("pior_probe_event_count", 0)) != 1:
                raise RuntimeError(f"V64.3.50.1 PIOR token={tok} does not have exactly one fired event")
            if str(row.get("pior_probe_arm", "")) != arm:
                raise RuntimeError(f"V64.3.50.1 PIOR token={tok} arm mismatch {row.get('pior_probe_arm')} vs {arm}")
            if int(row.get("iteration_index", -1)) != 0:
                raise RuntimeError(f"V64.3.50.1 PIOR token={tok} fired outside iteration 0")
            if str(row.get("pior_probe_target_source", "")) != "preregistered_V49_manifest_iteration0_proposal":
                raise RuntimeError(f"V64.3.50.1 PIOR token={tok} target source mismatch")
            expected_ts = int(meta[tok].get("timestamp_us", 0) or 0)
            target_ts = int(row.get("target_timestamp_us", 0) or 0)
            current_ts = int(row.get("current_timestamp_us", 0) or 0)
            if target_ts != expected_ts or abs(current_ts - expected_ts) > 4:
                raise RuntimeError(
                    f"V64.3.50.1 PIOR token={tok} timestamp mismatch target/current/expected="
                    f"{target_ts}/{current_ts}/{expected_ts}"
                )
            expected_action = int(meta[tok].get("full_selected_action", -1))
            proposal = int(row.get("pior_probe_proposal_action", -1))
            baseline = int(row.get("pior_probe_baseline_action", -1))
            final = int(row.get("pior_probe_final_action", -1))
            if proposal != expected_action or proposal < 0 or proposal == baseline:
                raise RuntimeError(
                    f"V64.3.50.1 PIOR token={tok} frozen action mismatch proposal={proposal} expected={expected_action} baseline={baseline}"
                )
            if arm == "treatment" and final != proposal:
                raise RuntimeError(f"V64.3.50.1 PIOR treatment token={tok} did not execute frozen proposal")
            if arm == "control" and final != baseline:
                raise RuntimeError(f"V64.3.50.1 PIOR control token={tok} did not preserve incumbent")
            if not bool(row.get("pior_probe_contract_same_frozen_proposal_or_incumbent", False)):
                raise RuntimeError(f"V64.3.50.1 PIOR token={tok} same-winner containment certificate failed")
            if not bool(row.get("pior_probe_contract_no_rerank_second_best_fallback", False)):
                raise RuntimeError(f"V64.3.50.1 PIOR token={tok} no-fallback certificate failed")
            by_token[tok] = row
    if set(by_token) != set(tokens):
        missing = sorted(set(tokens) - set(by_token))[:20]
        extra = sorted(set(by_token) - set(tokens))[:20]
        raise RuntimeError(
            f"V64.3.50.1 PIOR probe-event token mismatch {len(by_token)}/{len(tokens)} missing={missing} extra={extra}"
        )
    online_exists = sum(bool(r.get("pior_probe_online_proposal_exists", False)) for r in by_token.values())
    online_match = sum(bool(r.get("pior_probe_online_proposal_matches_target", False)) for r in by_token.values())
    audit = {
        "scenario_count": len(tokens),
        "token_sha256": _token_sha(tokens),
        "online_proposal_exists_count": int(online_exists),
        "online_proposal_matches_frozen_target_count": int(online_match),
        "online_replay_fraction": float(online_match / max(len(tokens), 1)),
        "scientific_intervention_identity": "manifest_frozen_V49_action_not_online_reselection",
    }
    return by_token, audit


def _batch_certificate_valid(
    *, root: Path, tokens: list[str], cfg_sha: str, ckpt_sha: str, challenge: str, raw_db_file_list_sha256: str = "",
    meta: dict[str, dict[str, Any]] | None = None, arm: str = "", probe_target_spec_sha256: str = "",
) -> tuple[dict[str, dict[str, Any]] | None, dict[str, Any] | None]:
    cert_path = root / ".pior_batch_complete.json"
    metrics_path = root / "scenario_metrics.jsonl"
    diag = root / "pior_probe_events.jsonl"
    target_path = root / "pior_probe_targets.json"
    if not cert_path.is_file() or not metrics_path.is_file() or not diag.is_file() or not target_path.is_file():
        return None, None
    try:
        cert = json.loads(cert_path.read_text(encoding="utf-8"))
        if (
            cert.get("complete") is not True
            or int(cert.get("scenario_count", -1)) != len(tokens)
            or str(cert.get("scenario_token_sha256", "")) != _token_sha(tokens)
            or str(cert.get("config_sha256", "")) != cfg_sha
            or str(cert.get("checkpoint_sha256", "")) != ckpt_sha
            or str(cert.get("challenge", "")) != challenge
            or (raw_db_file_list_sha256 and str(cert.get("raw_db_file_list_sha256", "")) != raw_db_file_list_sha256)
            or (probe_target_spec_sha256 and str(cert.get("probe_target_spec_sha256", "")) != probe_target_spec_sha256)
            or (probe_target_spec_sha256 and _sha256(target_path) != probe_target_spec_sha256)
            or int(cert.get("successful", -1)) != len(tokens)
            or int(cert.get("failed", -1)) != 0
            or int(cert.get("probe_fired_count", -1)) != len(tokens)
            or str(cert.get("scenario_metrics_sha256", "")) != _sha256(metrics_path)
            or str(cert.get("probe_events_sha256", "")) != _sha256(diag)
        ):
            return None, None
        metrics = _read_normalized_metrics(metrics_path, tokens)
        if meta is not None and arm:
            _validate_probe_events(diag, tokens=tokens, meta=meta, arm=arm)
        return metrics, cert
    except Exception:
        return None, None


def _validate_legacy_full_arm(
    *, root: Path, tokens: list[str], meta: dict[str, dict[str, Any]], cfg_sha: str, ckpt_sha: str, challenge: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]] | None:
    """Safely salvage only a *fully complete* pre-optimization V50 arm.

    The old runner has no token-bound partial completion certificate, so an
    interrupted 502-scene arm cannot be resumed scene-by-scene without weakening
    the one-probe-per-scenario attribution. A fully finished arm, however, can be
    proven by its nuPlan success summary, exact 502 probe count, and exact 502
    per-scenario metric identity and is scientifically safe to reuse.
    """
    log = root / "run.log"
    diag = root / "pior_closed_loop_diag.jsonl"
    if not log.is_file() or not diag.is_file():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    succ, fail = _parse_success(text)
    if succ != len(tokens) or fail != 0 or _count_probe_fires(diag) != len(tokens):
        return None
    metrics, metric_file = _extract_scenario_metrics(root, tokens, meta)
    normalized = root / "legacy_full_arm_scenario_metrics.jsonl"
    _write_normalized_metrics(normalized, metrics, tokens)
    cert = {
        "complete": True,
        "resume_source": "legacy_pre_batch_full_arm",
        "scenario_count": len(tokens),
        "scenario_token_sha256": _token_sha(tokens),
        "config_sha256": cfg_sha,
        "checkpoint_sha256": ckpt_sha,
        "challenge": challenge,
        "successful": succ,
        "failed": fail,
        "probe_fired_count": len(tokens),
        "metric_file": metric_file,
        "normalized_metric_file": str(normalized),
        "scenario_metrics_sha256": _sha256(normalized),
    }
    (root / ".pior_legacy_full_arm_reuse.json").write_text(json.dumps(cert, indent=2, sort_keys=True), encoding="utf-8")
    return metrics, cert


def _run_batch(
    *, arm: str, gpu: str, cfg: Path, checkpoint: Path, tokens: list[str], meta: dict[str, dict[str, Any]],
    nuplan_root: Path, challenge: str, arm_root: Path, workers: int, batch_index: int, batch_count: int,
    resume: bool, heartbeat_seconds: float, serialize_gpu_inference: bool, profile_closed_loop: bool,
    cfg_sha: str, ckpt_sha: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = arm_root / "batches" / f"batch_{batch_index:04d}"
    raw_files = _batch_raw_files(tokens, meta)
    raw_db_file_list_sha256 = hashlib.sha256(("\n".join(raw_files) + "\n").encode("utf-8")).hexdigest()
    target_payload = _probe_target_payload(tokens, meta)
    probe_target_spec_sha256 = _payload_sha256(target_payload)
    if resume:
        metrics, cert = _batch_certificate_valid(
            root=root, tokens=tokens, cfg_sha=cfg_sha, ckpt_sha=ckpt_sha, challenge=challenge,
            raw_db_file_list_sha256=raw_db_file_list_sha256, meta=meta, arm=arm,
            probe_target_spec_sha256=probe_target_spec_sha256,
        )
        if metrics is not None and cert is not None:
            print(
                f"[PIOR-RESUME] arm={arm} batch={batch_index + 1}/{batch_count} scenarios={len(tokens)} "
                f"validated_complete=true root={root}",
                flush=True,
            )
            return metrics, cert

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "scenario_tokens.json").write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    target_file = root / "pior_probe_targets.json"
    written_target_sha = _write_probe_target_file(target_file, target_payload)
    if written_target_sha != probe_target_spec_sha256:
        raise RuntimeError("V64.3.50.1 PIOR target-spec serialization hash mismatch")
    token_override = "scenario_filter.scenario_tokens=" + json.dumps(tokens, separators=(",", ":"))
    diag = root / "pior_probe_events.jsonl"
    profile = root / "bdse_closed_loop_profile.json"
    cmd = [
        sys.executable, "-m", "bdse.experiments.evaluate_closed_loop",
        "--config", str(cfg), "--checkpoint", str(checkpoint), "--device", "cuda",
        "--challenge", challenge,
        "--metric-aggregator", f"{challenge}_weighted_average",
        "--output-dir", str(root), "--experiment-uid", f"v64_3_50_pior_{arm}_b{batch_index:04d}",
        "--nuplan-module", "nuplan.planning.script.run_simulation",
        "--scenario-builder", "nuplan", "--worker", "single_machine_thread_pool", "--hydra-full-error",
        "--nuplan-data-root", str(nuplan_root), "--nuplan-map-root", str(nuplan_root / "maps"),
        "--nuplan-exp-root", str(nuplan_root / "exp"), "--nuplan-db-files", *raw_files,
        "--", token_override, f"scenario_filter.limit_total_scenarios={len(tokens)}",
        "scenario_filter.shuffle=false", "scenario_filter.log_names=null",
        f"worker.max_workers={int(workers)}", "run_metric=true", "~callback.simulation_log_callback",
    ]
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "BDSE_SHARE_MODEL_PER_PROCESS": "1",
        # The shared model is read-only/eval. The repository's optimized
        # fixed-budget closed-loop suite already runs this mode unlocked. Keep a
        # knob for conservative debugging, but do not serialize by default.
        "BDSE_SERIALIZE_GPU_INFERENCE": "1" if serialize_gpu_inference else "0",
        "BDSE_SHARD_PLANNERS_ACROSS_GPUS": "0",
        # PIOR needs only one durable probe certificate per scenario. Full
        # per-tick diagnostics are pure I/O overhead for this experiment.
        "BDSE_CLOSED_LOOP_DIAG": str(diag.resolve()),
        "BDSE_CLOSED_LOOP_DIAG_MODE": "pior_probe_events",
        "BDSE_PIOR_TARGETS_FILE": str(target_file.resolve()),
        "BDSE_STRICT_CLOSED_LOOP_DIAG": "1",
        "BDSE_PROFILE_CLOSED_LOOP": "1" if profile_closed_loop else "0",
        "BDSE_CLOSED_LOOP_PROFILE_JSON": str(profile.resolve()),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    log = root / "run.log"
    started = time.time()
    print(
        f"[PIOR-BATCH-START] arm={arm} batch={batch_index + 1}/{batch_count} gpu={gpu} "
        f"scenarios={len(tokens)} raw_dbs={len(raw_files)} workers={workers} "
        f"serialize_gpu={int(serialize_gpu_inference)} root={root}",
        flush=True,
    )
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    heartbeat = max(5.0, float(heartbeat_seconds))
    while True:
        try:
            returncode = proc.wait(timeout=heartbeat)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.time() - started
            fires = max(0, _count_probe_fires(diag))
            util, mem = _gpu_stats(gpu)
            tail = _tail_log(log)
            print(
                f"[PIOR-TICK] arm={arm} batch={batch_index + 1}/{batch_count} pid={proc.pid} "
                f"elapsed={elapsed / 60.0:.1f}min probe_events={fires}/{len(tokens)} "
                f"gpu_util={util} gpu_mem={mem} phase={_phase(log)}"
                + (f" last='{tail}'" if tail else ""),
                flush=True,
            )
    wall = time.time() - started
    text = log.read_text(encoding="utf-8", errors="replace")
    succ, fail = _parse_success(text)
    fired = _count_probe_fires(diag)
    if returncode != 0 or succ != len(tokens) or fail != 0 or fired != len(tokens):
        tail = "\n".join(text.replace("\r", "\n").splitlines()[-30:])
        raise RuntimeError(
            f"PIOR {arm} batch={batch_index} invalid return={returncode} success={succ} failed={fail} "
            f"probe_fired={fired} expected={len(tokens)}; see {log}\n--- tail ---\n{tail}"
        )
    _, probe_audit = _validate_probe_events(diag, tokens=tokens, meta=meta, arm=arm)
    metrics, metric_file = _extract_scenario_metrics(root, tokens, meta)
    normalized = root / "scenario_metrics.jsonl"
    _write_normalized_metrics(normalized, metrics, tokens)
    cert = {
        "complete": True,
        "arm": arm,
        "batch_index": int(batch_index),
        "batch_count": int(batch_count),
        "scenario_count": len(tokens),
        "scenario_token_sha256": _token_sha(tokens),
        "config_sha256": cfg_sha,
        "checkpoint_sha256": ckpt_sha,
        "challenge": challenge,
        "successful": succ,
        "failed": fail,
        "probe_fired_count": fired,
        "raw_db_file_count": len(raw_files),
        "raw_db_files": raw_files,
        "raw_db_file_list_sha256": raw_db_file_list_sha256,
        "probe_target_spec_file": str(target_file),
        "probe_target_spec_sha256": probe_target_spec_sha256,
        "probe_event_token_sha256": _token_sha(tokens),
        "probe_identity_audit": probe_audit,
        "workers": int(workers),
        "serialize_gpu_inference": bool(serialize_gpu_inference),
        "wall_time_s": float(wall),
        "scenarios_per_wall_hour": len(tokens) / max(wall, 1e-9) * 3600.0,
        "metric_file": metric_file,
        "scenario_metrics_file": str(normalized),
        "scenario_metrics_sha256": _sha256(normalized),
        "probe_events_file": str(diag),
        "probe_events_sha256": _sha256(diag),
        "profile_file": str(profile) if profile.is_file() else "",
    }
    (root / ".pior_batch_complete.json").write_text(json.dumps(cert, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[PIOR-BATCH-DONE] arm={arm} batch={batch_index + 1}/{batch_count} scenarios={len(tokens)} "
        f"wall={wall / 60.0:.1f}min throughput={cert['scenarios_per_wall_hour']:.1f}/h",
        flush=True,
    )
    return metrics, cert



def _collision_safe_batches(
    tokens: list[str], meta: dict[str, dict[str, Any]], batch_size: int, first_batch_size: int | None = None
) -> list[list[str]]:
    """Deterministically batch tokens while keeping start timestamps unique per subprocess.

    ``first_batch_size`` is an engineering-only paired preflight.  It changes no
    token, intervention, metric, label, or fold; it merely makes the first
    completion certificate small enough to fail fast on a server-side identity
    mismatch before the remaining expensive TRAIN collection is allowed to run.
    """
    normal_limit = max(1, int(batch_size))
    first_limit = normal_limit if first_batch_size is None else max(1, min(int(first_batch_size), normal_limit))
    out: list[list[str]] = []
    cur: list[str] = []
    seen_ts: set[int] = set()
    limit = first_limit
    for tok in tokens:
        ts = int(meta[tok].get("timestamp_us", 0) or 0)
        if ts <= 0:
            raise RuntimeError(f"V64.3.50.1 PIOR token={tok} missing timestamp_us for batching")
        if cur and (len(cur) >= limit or ts in seen_ts):
            out.append(cur)
            cur = []
            seen_ts = set()
            limit = normal_limit
        cur.append(tok)
        seen_ts.add(ts)
    if cur:
        out.append(cur)
    return out


def _run_arm(
    *, arm: str, gpu: str, cfg: Path, checkpoint: Path, tokens: list[str], meta: dict[str, dict[str, Any]],
    nuplan_root: Path, challenge: str, output_root: Path, workers: int, batch_size: int, resume: bool,
    heartbeat_seconds: float, serialize_gpu_inference: bool, profile_closed_loop: bool,
    allow_legacy_full_arm_resume: bool, first_batch_size: int, preflight_barrier: threading.Barrier | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = output_root / arm
    root.mkdir(parents=True, exist_ok=True)
    cfg_sha, ckpt_sha = _sha256(cfg), _sha256(checkpoint)
    if resume and allow_legacy_full_arm_resume:
        legacy = _validate_legacy_full_arm(
            root=root, tokens=tokens, meta=meta, cfg_sha=cfg_sha, ckpt_sha=ckpt_sha, challenge=challenge,
        )
        if legacy is not None:
            metrics, cert = legacy
            print(
                f"[PIOR-RESUME] arm={arm} reused fully complete legacy 502-scene arm; "
                "partial legacy arms are intentionally not reused",
                flush=True,
            )
            return metrics, {"arm": arm, "resume_mode": "legacy_full_arm", "batches": [cert]}

    batch_size = max(1, int(batch_size))
    batches = _collision_safe_batches(tokens, meta, batch_size, first_batch_size=first_batch_size)
    combined: dict[str, dict[str, Any]] = {}
    certs: list[dict[str, Any]] = []
    for bi, batch_tokens in enumerate(batches):
        try:
            metrics, cert = _run_batch(
                arm=arm, gpu=gpu, cfg=cfg, checkpoint=checkpoint, tokens=batch_tokens, meta=meta,
                nuplan_root=nuplan_root, challenge=challenge, arm_root=root, workers=workers,
                batch_index=bi, batch_count=len(batches), resume=resume, heartbeat_seconds=heartbeat_seconds,
                serialize_gpu_inference=serialize_gpu_inference, profile_closed_loop=profile_closed_loop,
                cfg_sha=cfg_sha, ckpt_sha=ckpt_sha,
            )
        except Exception:
            if bi == 0 and preflight_barrier is not None:
                try:
                    preflight_barrier.abort()
                except Exception:
                    pass
            raise
        overlap = set(combined) & set(metrics)
        if overlap:
            raise RuntimeError(f"PIOR duplicate tokens across completed batches: {sorted(overlap)[:10]}")
        combined.update(metrics)
        certs.append(cert)
        if bi == 0 and preflight_barrier is not None:
            print(
                f"[PIOR-PREFLIGHT-PASS] arm={arm} scenarios={len(batch_tokens)}; "
                "waiting for paired arm before full TRAIN collection",
                flush=True,
            )
            try:
                preflight_barrier.wait(timeout=300.0)
            except threading.BrokenBarrierError as exc:
                raise RuntimeError(
                    f"V64.3.50.1 PIOR paired preflight peer failed; arm={arm} will not continue expensive collection"
                ) from exc
            print(f"[PIOR-PREFLIGHT-PAIR-PASS] arm={arm}; continuing full TRAIN collection", flush=True)
    if set(combined) != set(tokens):
        raise RuntimeError(f"PIOR {arm} combined batch metrics mismatch: {len(combined)}/{len(tokens)}")
    arm_summary = {
        "arm": arm,
        "resume_mode": "validated_batches",
        "batch_size": int(batch_size),
        "first_batch_size": int(first_batch_size),
        "batch_count": len(batches),
        "scenario_count": len(tokens),
        "scenario_token_sha256": _token_sha(tokens),
        "config_sha256": cfg_sha,
        "checkpoint_sha256": ckpt_sha,
        "challenge": challenge,
        "batches": certs,
        "total_batch_wall_time_s": float(sum(float(x.get("wall_time_s", 0.0)) for x in certs)),
    }
    (root / "pior_arm_summary.json").write_text(json.dumps(arm_summary, indent=2, sort_keys=True), encoding="utf-8")
    return combined, arm_summary


def _score_key(c: dict[str, float], t: dict[str, float]) -> str:
    for k in SCORE_CANDIDATES:
        if k in c and k in t:
            return k
    common = sorted(set(c) & set(t))
    weighted = [k for k in common if "weighted_average" in k.lower() and "scenario" not in k.lower()]
    if len(weighted) == 1:
        return weighted[0]
    raise RuntimeError(f"PIOR per-scenario aggregate has no unambiguous official score column; common={common}")


def _pair(control: dict[str, dict[str, Any]], treatment: dict[str, dict[str, Any]], tokens: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tok in tokens:
        cm = dict(control[tok]["metrics"])
        tm = dict(treatment[tok]["metrics"])
        score_key = _score_key(cm, tm)
        missing_safety = [k for k in SAFETY_METRICS if k not in cm or k not in tm]
        if missing_safety:
            raise RuntimeError(f"PIOR STOP {tok}: required hard-safety metrics missing {missing_safety}")
        score_delta = float(tm[score_key] - cm[score_key])
        safety_delta = {k: float(tm[k] - cm[k]) for k in SAFETY_METRICS}
        hard_harm = any(v < -1.0e-12 for v in safety_delta.values())
        beneficial = bool((not hard_harm) and score_delta > 1.0e-12)
        common = sorted(set(cm) & set(tm))
        rows.append({
            "scenario_token": tok,
            "official_score_metric": score_key,
            "control_score": float(cm[score_key]),
            "treatment_score": float(tm[score_key]),
            "closed_loop_score_delta": score_delta,
            "closed_loop_hard_harm": bool(hard_harm),
            "closed_loop_beneficial": bool(beneficial),
            "pior_interventional_outcome": 1.0 if beneficial else -1.0,
            "safety_delta": safety_delta,
            "metric_delta": {k: float(tm[k] - cm[k]) for k in common},
            "control_identity": control[tok]["identity"],
            "treatment_identity": treatment[tok]["identity"],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Run V64.3.50 paired one-shot closed-loop proposal-vs-incumbent interventions on the exact frozen TRAIN RSMR proposal population.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--treatment-config", type=Path, required=True)
    ap.add_argument("--control-config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--nuplan-root", type=Path, required=True)
    ap.add_argument("--challenge", choices=["closed_loop_nonreactive_agents", "closed_loop_reactive_agents"], default="closed_loop_nonreactive_agents")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--workers-per-arm", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--first-batch-size", type=int, default=4, help="Engineering-only paired preflight batch; both arms must pass before expensive collection continues.")
    ap.add_argument("--heartbeat-seconds", type=float, default=30.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-legacy-full-arm-resume", action="store_true")
    ap.add_argument("--serialize-gpu-inference", type=int, choices=[0, 1], default=0)
    ap.add_argument("--profile-closed-loop", type=int, choices=[0, 1], default=1)
    ap.add_argument("--output-paired-outcomes", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    a = ap.parse_args()

    tokens, meta, raw_files = _manifest(a.manifest)
    gpus = [x.strip() for x in a.gpus.split(",") if x.strip()]
    if len(gpus) < 2:
        raise ValueError("V64.3.50 PIOR requires two GPU ids so paired arms can run without sharing one model/GPU process")
    a.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[PIOR-COLLECT] scenarios={len(tokens)} exact_raw_dbs={len(raw_files)} batch_size={a.batch_size} "
        f"first_batch_size={a.first_batch_size} "
        f"workers_per_arm={a.workers_per_arm} resume={int(a.resume)} heartbeat={a.heartbeat_seconds}s "
        f"serialize_gpu={a.serialize_gpu_inference}",
        flush=True,
    )

    results: dict[str, dict[str, Any]] = {}
    metrics_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    errors: list[str] = []
    lock = threading.Lock()
    preflight_barrier = threading.Barrier(2) if int(a.first_batch_size) > 0 else None

    def work(arm: str, gpu: str, cfg: Path) -> None:
        try:
            metrics, summary = _run_arm(
                arm=arm, gpu=gpu, cfg=cfg, checkpoint=a.checkpoint, tokens=tokens, meta=meta,
                nuplan_root=a.nuplan_root, challenge=a.challenge, output_root=a.output_root,
                workers=a.workers_per_arm, batch_size=a.batch_size, resume=a.resume,
                heartbeat_seconds=a.heartbeat_seconds,
                serialize_gpu_inference=bool(a.serialize_gpu_inference),
                profile_closed_loop=bool(a.profile_closed_loop),
                allow_legacy_full_arm_resume=bool(a.allow_legacy_full_arm_resume),
                first_batch_size=int(a.first_batch_size), preflight_barrier=preflight_barrier,
            )
            with lock:
                metrics_by_arm[arm] = metrics
                results[arm] = summary
        except Exception as exc:
            with lock:
                errors.append(f"{arm}: {type(exc).__name__}: {exc}")

    threads = [
        threading.Thread(target=work, args=("control", gpus[0], a.control_config), daemon=True),
        threading.Thread(target=work, args=("treatment", gpus[1], a.treatment_config), daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError("V64.3.50 PIOR paired closed-loop failed: " + " | ".join(errors))

    control = metrics_by_arm["control"]
    treatment = metrics_by_arm["treatment"]
    paired = _pair(control, treatment, tokens)
    a.output_paired_outcomes.parent.mkdir(parents=True, exist_ok=True)
    with a.output_paired_outcomes.open("w", encoding="utf-8") as f:
        for r in paired:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    pos = sum(int(r["closed_loop_beneficial"]) for r in paired)
    harm = sum(int(r["closed_loop_hard_harm"]) for r in paired)
    report = {
        "audit": "v64_3_50_pior_paired_closed_loop",
        "algorithm_version": "V64.3.50-EAF-ICER-PIOR",
        "engineering_revision": "v64_3_50_1_manifest_bound_iteration0_probe_identity_repair",
        "challenge": a.challenge,
        "scientific_intervention": "paired_one_shot_actual_full_set_RSMR_proposal_vs_same_incumbent_then_incumbent_only",
        "scenario_count": len(paired),
        "beneficial_count": pos,
        "nonbeneficial_count": len(paired) - pos,
        "hard_harm_count": harm,
        "exact_raw_db_file_count": len(raw_files),
        "batch_size": int(a.batch_size),
        "first_batch_size": int(a.first_batch_size),
        "workers_per_arm": int(a.workers_per_arm),
        "serialize_gpu_inference": bool(a.serialize_gpu_inference),
        "resume_enabled": bool(a.resume),
        "control": results["control"],
        "treatment": results["treatment"],
        "label_contract": {
            "beneficial": "official_closed_loop_score_delta>0 AND no degradation in collision/TTC/drivable hard-safety metrics",
            "nonbeneficial": "otherwise",
            "learner_consumes_sign_only": True,
            "no_teacher_or_logged_future_in_runtime": True,
        },
        "resume_contract": {
            "scientifically_safe_unit": "only a fully completed batch with exact token/config/checkpoint/challenge/probe/metric hashes",
            "partial_pre_optimization_arm_reuse": False,
            "fully_completed_pre_optimization_arm_reuse": bool(a.allow_legacy_full_arm_resume),
            "sample_membership_or_fold_assignment_changed": False,
        },
        "pass": True,
    }
    a.output_report.parent.mkdir(parents=True, exist_ok=True)
    a.output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "pass": True,
        "scenario_count": len(paired),
        "beneficial_count": pos,
        "hard_harm_count": harm,
        "paired_outcomes": str(a.output_paired_outcomes),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
