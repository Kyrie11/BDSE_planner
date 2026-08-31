from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
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
    raw = [str(x) for x in r.get("raw_db_directories", [])]
    if len(raw) != 4:
        raise RuntimeError(f"V64.3.50 PIOR expected 4 TRAIN raw DB directories, got {raw}")
    return tokens, {str(x["scenario_token"]): x for x in rows}, raw


def _run_arm(
    *, arm: str, gpu: str, cfg: Path, checkpoint: Path, tokens: list[str], raw_dirs: list[str],
    nuplan_root: Path, challenge: str, output_root: Path, workers: int,
) -> dict[str, Any]:
    root = output_root / arm
    root.mkdir(parents=True, exist_ok=True)
    diag = root / "pior_closed_loop_diag.jsonl"
    if diag.exists():
        diag.unlink()
    token_override = "scenario_filter.scenario_tokens=" + json.dumps(tokens, separators=(",", ":"))
    cmd = [
        sys.executable, "-m", "bdse.experiments.evaluate_closed_loop",
        "--config", str(cfg), "--checkpoint", str(checkpoint), "--device", "cuda",
        "--challenge", challenge,
        "--metric-aggregator", f"{challenge}_weighted_average",
        "--output-dir", str(root), "--experiment-uid", f"v64_3_50_pior_{arm}",
        "--nuplan-module", "nuplan.planning.script.run_simulation",
        "--scenario-builder", "nuplan", "--worker", "single_machine_thread_pool", "--hydra-full-error",
        "--nuplan-data-root", str(nuplan_root), "--nuplan-map-root", str(nuplan_root / "maps"),
        "--nuplan-exp-root", str(nuplan_root / "exp"), "--nuplan-db-files", *raw_dirs,
        "--", token_override, f"scenario_filter.limit_total_scenarios={len(tokens)}",
        "scenario_filter.shuffle=false", f"worker.max_workers={int(workers)}", "run_metric=true",
        "~callback.simulation_log_callback",
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "BDSE_SHARE_MODEL_PER_PROCESS": "1",
        "BDSE_SERIALIZE_GPU_INFERENCE": "1",
        "BDSE_CLOSED_LOOP_DIAG": str(diag),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    log = root / "run.log"
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    txt = log.read_text(encoding="utf-8", errors="replace")
    succ, fail = _parse_success(txt)
    if proc.returncode != 0 or succ != len(tokens) or fail != 0:
        raise RuntimeError(f"PIOR {arm} closed-loop invalid return={proc.returncode} success={succ} failed={fail} expected={len(tokens)}; see {log}")
    if not diag.is_file():
        raise RuntimeError(f"PIOR {arm} missing BDSE_CLOSED_LOOP_DIAG {diag}")
    fired = 0
    with diag.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            td = ((row.get("diagnostics", {}) or {}).get("tournament", {}) or {})
            fired += int(bool(td.get("pior_probe_fired", False)))
    # Each nuPlan scenario owns an independently initialized planner/core and the
    # probe can fire at most once per scenario. Exact count therefore proves the
    # full paired population actually received its designated intervention.
    if fired != len(tokens):
        raise RuntimeError(f"PIOR {arm} probe coverage mismatch: fired={fired}, expected={len(tokens)}. STOP rather than training on untreated scenes.")
    return {"arm": arm, "root": str(root), "successful": succ, "failed": fail, "probe_fired_count": fired, "log": str(log), "diag": str(diag)}


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
    for t, r in meta.items():
        log = _as_str(r.get("log_name"))
        sc = _as_str(r.get("scenario_name"))
        if log and sc:
            alias[(log, sc)] = t
        for a in (sc,):
            if a:
                single_alias.setdefault(a, set()).add(t)

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
            # nuPlan versions differ in the exact identity column. Prefer direct
            # token columns, then exact token-valued scenario names, then the
            # (log_name, scenario_name) aliases captured from the NPZ cache.
            for c in ("scenario_token", "token", "scenario", "scenario_name"):
                if c in row.index:
                    val = _as_str(row[c])
                    if val in expected:
                        tok = val; break
            if not tok:
                log = _as_str(row["log_name"]) if "log_name" in row.index else ""
                sc = _as_str(row["scenario"]) if "scenario" in row.index else _as_str(row["scenario_name"]) if "scenario_name" in row.index else ""
                tok = alias.get((log, sc), "")
                if not tok and sc in single_alias and len(single_alias[sc]) == 1:
                    tok = next(iter(single_alias[sc]))
            if not tok:
                # Last exact-match fallback: some devkit versions use a custom
                # string identity column. Never substring-match or infer order.
                matches = {str(v) for v in row.values if _as_str(v) in expected}
                if len(matches) == 1:
                    tok = next(iter(matches))
            if not tok:
                continue
            if tok in got:
                ambiguous = True; break
            ident = {k: _as_str(row[k]) for k in row.index if not isinstance(row[k], (bool, int, float, np.integer, np.floating))}
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
            f"PIOR cannot establish exact token identity from aggregator metrics: matched={count}/{len(tokens)} missing={missing[:10]} extra={extra[:10]} file={path}. "
            "STOP instead of relying on row order."
        )
    return got, str(path)


def _score_key(c: dict[str, float], t: dict[str, float]) -> str:
    for k in SCORE_CANDIDATES:
        if k in c and k in t:
            return k
    # Accept a challenge-specific weighted-average column discovered at runtime.
    common = sorted(set(c) & set(t))
    weighted = [k for k in common if "weighted_average" in k.lower() and "scenario" not in k.lower()]
    if len(weighted) == 1:
        return weighted[0]
    raise RuntimeError(f"PIOR per-scenario aggregate has no unambiguous official score column; common={common}")


def _pair(control: dict[str, dict[str, Any]], treatment: dict[str, dict[str, Any]], tokens: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tok in tokens:
        cm = dict(control[tok]["metrics"]); tm = dict(treatment[tok]["metrics"])
        score_key = _score_key(cm, tm)
        missing_safety = [k for k in SAFETY_METRICS if k not in cm or k not in tm]
        if missing_safety:
            raise RuntimeError(f"PIOR STOP {tok}: required hard-safety metrics missing {missing_safety}")
        score_delta = float(tm[score_key] - cm[score_key])
        safety_delta = {k: float(tm[k] - cm[k]) for k in SAFETY_METRICS}
        hard_harm = any(v < -1.0e-12 for v in safety_delta.values())
        beneficial = bool((not hard_harm) and score_delta > 1.0e-12)
        # The low-capacity ranker consumes sign only. Keep the physical score
        # delta separately and expose a +/-1 outcome sign so no arbitrary
        # magnitude or catastrophe weight enters learning.
        common = sorted(set(cm) & set(tm))
        rows.append({
            "scenario_token": tok,
            "official_score_metric": score_key,
            "control_score": float(cm[score_key]), "treatment_score": float(tm[score_key]),
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
    ap.add_argument("--output-paired-outcomes", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    a = ap.parse_args()
    tokens, meta, raw_dirs = _manifest(a.manifest)
    gpus = [x.strip() for x in a.gpus.split(",") if x.strip()]
    if len(gpus) < 2:
        raise ValueError("V64.3.50 PIOR requires two GPU ids so paired arms can run without sharing one model/GPU process")
    a.output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    lock = threading.Lock()
    def work(arm: str, gpu: str, cfg: Path) -> None:
        try:
            r = _run_arm(arm=arm, gpu=gpu, cfg=cfg, checkpoint=a.checkpoint, tokens=tokens, raw_dirs=raw_dirs,
                         nuplan_root=a.nuplan_root, challenge=a.challenge, output_root=a.output_root, workers=a.workers_per_arm)
            with lock: results[arm] = r
        except Exception as exc:
            with lock: errors.append(f"{arm}: {type(exc).__name__}: {exc}")
    threads = [
        threading.Thread(target=work, args=("control", gpus[0], a.control_config), daemon=True),
        threading.Thread(target=work, args=("treatment", gpus[1], a.treatment_config), daemon=True),
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    if errors:
        raise RuntimeError("V64.3.50 PIOR paired closed-loop failed: " + " | ".join(errors))
    control, cfile = _extract_scenario_metrics(Path(results["control"]["root"]), tokens, meta)
    treatment, tfile = _extract_scenario_metrics(Path(results["treatment"]["root"]), tokens, meta)
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
        "challenge": a.challenge,
        "scientific_intervention": "paired_one_shot_actual_full_set_RSMR_proposal_vs_same_incumbent_then_incumbent_only",
        "scenario_count": len(paired), "beneficial_count": pos, "nonbeneficial_count": len(paired)-pos, "hard_harm_count": harm,
        "control": results["control"], "treatment": results["treatment"],
        "control_metric_file": cfile, "treatment_metric_file": tfile,
        "label_contract": {
            "beneficial": "official_closed_loop_score_delta>0 AND no degradation in collision/TTC/drivable hard-safety metrics",
            "nonbeneficial": "otherwise",
            "learner_consumes_sign_only": True,
            "no_teacher_or_logged_future_in_runtime": True,
        },
        "pass": True,
    }
    a.output_report.parent.mkdir(parents=True, exist_ok=True)
    a.output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"pass": True, "scenario_count": len(paired), "beneficial_count": pos, "hard_harm_count": harm, "paired_outcomes": str(a.output_paired_outcomes)}, sort_keys=True))


if __name__ == "__main__":
    main()
