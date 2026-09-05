from __future__ import annotations

"""Collect V64.3.54 paired realized ego response over the one-shot exposure window.

This collector replays both V50.5 arms but terminates immediately after the
first scheduled planner replan and disables metric computation.  Full-horizon
paired outcome labels are *reused* from V50.5; they are not recollected.  The
short replay therefore changes only the mediator/state evidence source.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from bdse.planner.paired_dynamic_response_mediation import (
    DYNAMIC_PROFILE_SCHEMA,
    paired_realized_profile,
)
from bdse.tools import run_v64_3_50_pior_paired_closed_loop as base
from bdse.tools import run_v64_3_50_5_pior_paired_closed_loop as safe

NUPLAN_MODULE = "bdse.tools.nuplan_v54_dynamic_response_run_simulation"


def _sha256(path: Path) -> str:
    return base._sha256(path)


def _config_replan_ticks(path: Path) -> int:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    n = int(((d.get("planner", {}) or {}).get("replan_interval_ticks", -1)))
    if n < 1:
        raise RuntimeError(f"V54 PDRM config lacks valid planner.replan_interval_ticks: {path}")
    return n


def _read_trace(path: Path, tokens: list[str], arm: str, exposure: int) -> dict[str, list[dict[str, Any]]]:
    expected = set(tokens)
    by: dict[str, dict[int, dict[str, Any]]] = {}
    if not path.is_file():
        raise RuntimeError(f"V54 PDRM missing dynamic sidecar {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tok = str(r.get("scenario_token", "")); a = str(r.get("arm", ""))
        idx = int(r.get("iteration_index", -1))
        if tok not in expected or a != arm or idx < 0 or idx > exposure:
            raise RuntimeError(f"V54 PDRM invalid dynamic row token={tok} arm={a} idx={idx}")
        if int(r.get("exposure_ticks", -1)) != exposure:
            raise RuntimeError(f"V54 PDRM exposure drift token={tok}")
        vals = [float(v) for v in r.get("ego_world", [])]
        if len(vals) != 4:
            raise RuntimeError(f"V54 PDRM ego_world shape error token={tok} idx={idx}")
        slot = by.setdefault(tok, {})
        if idx in slot:
            raise RuntimeError(f"V54 PDRM duplicate dynamic sample token={tok} idx={idx}")
        slot[idx] = r
    want_idx = list(range(exposure + 1))
    out: dict[str, list[dict[str, Any]]] = {}
    for tok in tokens:
        got = by.get(tok, {})
        if sorted(got) != want_idx:
            raise RuntimeError(f"V54 PDRM incomplete dynamic trace token={tok} indices={sorted(got)} expected={want_idx}")
        rows = [got[i] for i in want_idx]
        ts = [int(r["time_us"]) for r in rows]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise RuntimeError(f"V54 PDRM non-increasing time token={tok}: {ts}")
        out[tok] = rows
    return out


def _batch_cert_valid(root: Path, *, tokens: list[str], arm: str, cfg_sha: str, ckpt_sha: str, exposure: int, meta: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    cp = root / ".v54_dynamic_batch_complete.json"
    side = root / "v54_dynamic_response_events.jsonl"
    diag = root / "pior_probe_events.jsonl"
    target = root / "pior_probe_targets.json"
    if not all(p.is_file() for p in (cp, side, diag, target)):
        return None
    try:
        c = json.loads(cp.read_text(encoding="utf-8"))
        if (
            c.get("complete") is not True
            or str(c.get("arm", "")) != arm
            or int(c.get("scenario_count", -1)) != len(tokens)
            or str(c.get("scenario_token_sha256", "")) != base._token_sha(tokens)
            or str(c.get("config_sha256", "")) != cfg_sha
            or str(c.get("checkpoint_sha256", "")) != ckpt_sha
            or int(c.get("exposure_ticks", -1)) != exposure
            or int(c.get("successful", -1)) != len(tokens)
            or int(c.get("failed", -1)) != 0
            or int(c.get("probe_fired_count", -1)) != len(tokens)
            or str(c.get("dynamic_sidecar_sha256", "")) != _sha256(side)
            or str(c.get("probe_events_sha256", "")) != _sha256(diag)
        ):
            return None
        base._validate_probe_events(diag, tokens=tokens, meta=meta, arm=arm)
        _read_trace(side, tokens, arm, exposure)
        return c
    except Exception:
        return None


def _run_batch(
    *, arm: str, gpu: str, cfg: Path, checkpoint: Path, tokens: list[str], meta: dict[str, dict[str, Any]],
    nuplan_root: Path, challenge: str, arm_root: Path, workers: int, batch_index: int, batch_count: int,
    exposure: int, heartbeat_seconds: float, resume: bool,
) -> dict[str, Any]:
    root = arm_root / "batches" / f"batch_{batch_index:04d}"
    cfg_sha = _sha256(cfg); ckpt_sha = _sha256(checkpoint)
    if resume:
        cert = _batch_cert_valid(root, tokens=tokens, arm=arm, cfg_sha=cfg_sha, ckpt_sha=ckpt_sha, exposure=exposure, meta=meta)
        if cert is not None:
            print(f"[V54-RESUME] arm={arm} batch={batch_index+1}/{batch_count} n={len(tokens)}", flush=True)
            return cert
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    raw_files = base._batch_raw_files(tokens, meta)
    target_payload = base._probe_target_payload(tokens, meta)
    target_file = root / "pior_probe_targets.json"
    semantic_sha = base._write_probe_target_file(target_file, target_payload)
    if semantic_sha != base._payload_sha256(target_payload):
        raise RuntimeError("V54 PDRM target semantic hash drift")
    (root / "scenario_tokens.json").write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    diag = root / "pior_probe_events.jsonl"
    dynamic = root / "v54_dynamic_response_events.jsonl"
    token_override = "scenario_filter.scenario_tokens=" + json.dumps(tokens, separators=(",", ":"))
    cmd = [
        sys.executable, "-m", "bdse.experiments.evaluate_closed_loop",
        "--config", str(cfg), "--checkpoint", str(checkpoint), "--device", "cuda",
        "--challenge", challenge, "--metric-aggregator", f"{challenge}_weighted_average",
        "--output-dir", str(root), "--experiment-uid", f"v64_3_54_pdrm_{arm}_b{batch_index:04d}",
        "--nuplan-module", NUPLAN_MODULE,
        "--scenario-builder", "nuplan", "--worker", "single_machine_thread_pool", "--hydra-full-error",
        "--nuplan-data-root", str(nuplan_root), "--nuplan-map-root", str(nuplan_root / "maps"),
        "--nuplan-exp-root", str(nuplan_root / "exp"), "--nuplan-db-files", *raw_files,
        "--", token_override, f"scenario_filter.limit_total_scenarios={len(tokens)}", "scenario_filter.shuffle=false",
        "scenario_filter.log_names=null", *base._anchor_start_mapping_overrides(), f"worker.max_workers={int(workers)}",
        "run_metric=false", "~callback.simulation_log_callback",
    ]
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(gpu), "BDSE_SHARE_MODEL_PER_PROCESS": "1",
        "BDSE_SERIALIZE_GPU_INFERENCE": "0", "BDSE_SHARD_PLANNERS_ACROSS_GPUS": "0",
        "BDSE_CLOSED_LOOP_DIAG": str(diag.resolve()), "BDSE_CLOSED_LOOP_DIAG_MODE": "pior_probe_events",
        "BDSE_PIOR_TARGETS_FILE": str(target_file.resolve()), "BDSE_STRICT_CLOSED_LOOP_DIAG": "1",
        "BDSE_PROFILE_CLOSED_LOOP": "0", "BDSE_V54_DYNAMIC_RESPONSE": "1",
        "BDSE_V54_EXPOSURE_TICKS": str(exposure),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    log = root / "run.log"; started = time.time()
    print(f"[V54-BATCH-START] arm={arm} batch={batch_index+1}/{batch_count} gpu={gpu} n={len(tokens)} exposure={exposure}", flush=True)
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    heartbeat = max(5.0, float(heartbeat_seconds))
    while True:
        try:
            rc = proc.wait(timeout=heartbeat); break
        except subprocess.TimeoutExpired:
            elapsed = time.time() - started
            fires = max(0, base._count_probe_fires(diag))
            util, mem = base._gpu_stats(gpu)
            print(f"[V54-TICK] arm={arm} batch={batch_index+1}/{batch_count} elapsed={elapsed/60:.1f}m probes={fires}/{len(tokens)} gpu={util}/{mem}", flush=True)
    text = log.read_text(encoding="utf-8", errors="replace")
    succ, fail = base._parse_success(text); fired = base._count_probe_fires(diag)
    if rc != 0 or succ != len(tokens) or fail != 0 or fired != len(tokens):
        tail = "\n".join(text.replace("\r", "\n").splitlines()[-40:])
        raise RuntimeError(f"V54 PDRM {arm} batch failed rc={rc} success={succ} failed={fail} probes={fired}/{len(tokens)}\n{tail}")
    base._validate_probe_events(diag, tokens=tokens, meta=meta, arm=arm)
    _read_trace(dynamic, tokens, arm, exposure)
    wall = time.time() - started
    cert = {
        "complete": True, "algorithm_version": "V64.3.54-EAF-ICER-PDRM", "arm": arm,
        "batch_index": batch_index, "batch_count": batch_count, "scenario_count": len(tokens),
        "scenario_token_sha256": base._token_sha(tokens), "config_sha256": cfg_sha, "checkpoint_sha256": ckpt_sha,
        "challenge": challenge, "successful": succ, "failed": fail, "probe_fired_count": fired,
        "exposure_ticks": exposure, "run_metric": False, "outcome_labels_recollected": False,
        "dynamic_sidecar": str(dynamic), "dynamic_sidecar_sha256": _sha256(dynamic),
        "probe_events_sha256": _sha256(diag), "wall_time_s": wall,
        "scenarios_per_wall_hour": len(tokens) / max(wall, 1e-9) * 3600.0,
    }
    (root / ".v54_dynamic_batch_complete.json").write_text(json.dumps(cert, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[V54-BATCH-DONE] arm={arm} batch={batch_index+1}/{batch_count} wall={wall/60:.1f}m", flush=True)
    return cert


def _run_arm(*, arm: str, gpu: str, cfg: Path, checkpoint: Path, tokens: list[str], meta: dict[str, dict[str, Any]], nuplan_root: Path, challenge: str, output_root: Path, workers: int, batch_size: int, exposure: int, heartbeat_seconds: float, resume: bool) -> dict[str, Any]:
    batches = base._collision_safe_batches(tokens, meta, batch_size=max(1, batch_size), first_batch_size=min(8, len(tokens)))
    root = output_root / arm; root.mkdir(parents=True, exist_ok=True)
    certs=[]
    for bi, bt in enumerate(batches):
        certs.append(_run_batch(arm=arm, gpu=gpu, cfg=cfg, checkpoint=checkpoint, tokens=bt, meta=meta,
            nuplan_root=nuplan_root, challenge=challenge, arm_root=root, workers=workers, batch_index=bi,
            batch_count=len(batches), exposure=exposure, heartbeat_seconds=heartbeat_seconds, resume=resume))
    return {"arm": arm, "gpu": str(gpu), "batch_count": len(certs), "scenario_count": sum(int(c["scenario_count"]) for c in certs),
            "total_batch_wall_time_s": sum(float(c["wall_time_s"]) for c in certs), "batches": certs}


def _collect_arm_traces(output_root: Path, arm: str, tokens: list[str], exposure: int) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    for p in sorted((output_root / arm / "batches").glob("batch_*/v54_dynamic_response_events.jsonl")):
        local_tokens=[]
        # derive token set directly from sidecar and validate with _read_trace
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip(): local_tokens.append(str(json.loads(line).get("scenario_token", "")))
        uniq=sorted(set(local_tokens))
        part=_read_trace(p, uniq, arm, exposure)
        for tok, rows in part.items():
            if tok in merged: raise RuntimeError(f"V54 duplicate token across batches arm={arm} token={tok}")
            merged[tok]=rows
    if set(merged) != set(tokens):
        raise RuntimeError(f"V54 arm trace population mismatch arm={arm} {len(merged)}/{len(tokens)}")
    return merged


def _load_planned_d(path: Path, tokens: list[str]) -> dict[str, float]:
    out={}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r=json.loads(line); tok=str(r.get("scenario_token", ""))
        if tok in out: raise RuntimeError(f"V54 duplicate V53 profile token={tok}")
        out[tok]=float(r.get("execution_contrast_linf", float("nan")))
    if set(out)!=set(tokens): raise RuntimeError(f"V54 V53 profile population mismatch {len(out)}/{len(tokens)}")
    return out


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True); p.add_argument("--treatment-config",type=Path,required=True); p.add_argument("--control-config",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--nuplan-root",type=Path,required=True); p.add_argument("--v53-operator-profiles",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True); p.add_argument("--output-profiles",type=Path,required=True); p.add_argument("--output-report",type=Path,required=True)
    p.add_argument("--gpu-treatment",default="0"); p.add_argument("--gpu-control",default="1"); p.add_argument("--workers",type=int,default=4); p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--heartbeat-seconds",type=float,default=30.0); p.add_argument("--challenge",default="closed_loop_nonreactive_agents"); p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    safe._assert_frozen_base_runner()
    tokens,meta,_raw=base._manifest(a.manifest)
    if len(tokens)!=502 or len(set(tokens))!=502: raise RuntimeError("V54 PDRM requires exact frozen 502 population")
    et=_config_replan_ticks(a.treatment_config); ec=_config_replan_ticks(a.control_config)
    if et!=ec: raise RuntimeError(f"V54 PDRM treatment/control replan interval mismatch {et} vs {ec}")
    exposure=et
    planned_d=_load_planned_d(a.v53_operator_profiles,tokens)

    kw=dict(checkpoint=a.checkpoint,tokens=tokens,meta=meta,nuplan_root=a.nuplan_root,challenge=str(a.challenge),output_root=a.output_root,
            workers=int(a.workers),batch_size=int(a.batch_size),exposure=exposure,heartbeat_seconds=float(a.heartbeat_seconds),resume=bool(a.resume))
    if str(a.gpu_treatment)!=str(a.gpu_control):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            ft=ex.submit(_run_arm,arm="treatment",gpu=str(a.gpu_treatment),cfg=a.treatment_config,**kw)
            fc=ex.submit(_run_arm,arm="control",gpu=str(a.gpu_control),cfg=a.control_config,**kw)
            ts,cs=ft.result(),fc.result()
    else:
        ts=_run_arm(arm="treatment",gpu=str(a.gpu_treatment),cfg=a.treatment_config,**kw)
        cs=_run_arm(arm="control",gpu=str(a.gpu_control),cfg=a.control_config,**kw)

    tr=_collect_arm_traces(a.output_root,"treatment",tokens,exposure)
    cr=_collect_arm_traces(a.output_root,"control",tokens,exposure)
    profiles=[]; equal_count=0; realized_nonzero=0
    for tok in sorted(tokens):
        rt,rc=tr[tok],cr[tok]
        its=[int(x["iteration_index"]) for x in rt]; ics=[int(x["iteration_index"]) for x in rc]
        tts=[int(x["time_us"]) for x in rt]; cts=[int(x["time_us"]) for x in rc]
        if its!=ics or tts!=cts: raise RuntimeError(f"V54 PDRM arm synchronization mismatch token={tok}")
        ta=[x["ego_world"] for x in rt]; ca=[x["ego_world"] for x in rc]
        prof=paired_realized_profile(ta,ca,iteration_indices=its,timestamps_us=tts,planned_execution_contrast_linf=planned_d[tok])
        if planned_d[tok] <= 1e-10:
            equal_count += 1
            if float(prof["realized_response_linf"]) > 1e-6:
                raise RuntimeError(f"V54 PDRM planned-physical-equal treatment/control diverged token={tok} realized={prof['realized_response_linf']}")
        if float(prof["realized_response_linf"]) > 1e-8: realized_nonzero += 1
        profiles.append({"scenario_token":tok,**prof})
    a.output_profiles.parent.mkdir(parents=True,exist_ok=True)
    with a.output_profiles.open("w",encoding="utf-8") as f:
        for r in profiles: f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")
    report={
        "audit":"v64_3_54_paired_dynamic_response_probe","algorithm_version":"V64.3.54-EAF-ICER-PDRM","pass":True,
        "scientific_role":"paired_post_intervention_realized_ego_response_mediator_state_only",
        "scenario_count":502,"scenario_token_sha256":base._token_sha(tokens),"exposure_ticks":exposure,"sample_count_per_arm_per_scene":exposure+1,
        "exposure_window_definition":"exact frozen planner replan interval from one-shot intervention anchor through first scheduled replan",
        "paired_outcome_labels_recollected":False,"run_metric":False,"short_horizon_only":True,"profile_schema":DYNAMIC_PROFILE_SCHEMA,
        "profile_sha256":_sha256(a.output_profiles),"planned_physical_equal_count":equal_count,"realized_nonzero_count":realized_nonzero,
        "treatment_summary":ts,"control_summary":cs,"two_gpu_parallel":str(a.gpu_treatment)!=str(a.gpu_control),
    }
    a.output_report.parent.mkdir(parents=True,exist_ok=True); a.output_report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"pass":True,"profiles":502,"exposure_ticks":exposure,"profile_sha256":report["profile_sha256"]},sort_keys=True))

if __name__=="__main__": main()
