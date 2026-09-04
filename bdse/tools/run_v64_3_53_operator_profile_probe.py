from __future__ import annotations

"""Collect V64.3.53 pre-execution proposal-vs-incumbent trajectory state.

This is a state-only replay of the frozen V50.5 treatment arm.  It does not
redefine or recollect the paired outcome label.  The scientific outcome rows
remain the metric-safe V50.5 502/502 treatment-control pairs; this probe only
records the exact runtime incumbent trajectory contrast already present at the
one-shot intervention event.
"""

import argparse
import hashlib
import json
import os
import subprocess as _subprocess
from pathlib import Path
from typing import Any

from bdse.planner.paired_operator_trajectory_retention import PROFILE_SCHEMA
from bdse.tools import run_v64_3_50_pior_paired_closed_loop as _base
from bdse.tools import run_v64_3_50_5_pior_paired_closed_loop as _safe

_OFFICIAL_MODULE = "nuplan.planning.script.run_simulation"
_V53_MODULE = "bdse.tools.nuplan_v53_operator_profile_run_simulation"

class _ProfileSubprocessProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(_subprocess, name)
    @staticmethod
    def Popen(cmd: Any, *args: Any, **kwargs: Any):  # noqa: N802
        if isinstance(cmd, (list, tuple)) and "bdse.experiments.evaluate_closed_loop" in cmd:
            rewritten=list(cmd)
            try: i=rewritten.index("--nuplan-module")
            except ValueError as exc: raise RuntimeError("V53 profile child lacks --nuplan-module") from exc
            if i+1>=len(rewritten) or rewritten[i+1] != _OFFICIAL_MODULE:
                raise RuntimeError(f"V53 profile unexpected frozen nuPlan module {rewritten[i+1] if i+1<len(rewritten) else '<missing>'}")
            rewritten[i+1]=_V53_MODULE
            env=dict(kwargs.get("env") or os.environ)
            env["BDSE_PIOR_METRIC_ENGINE_SERIALIZATION"]="1"
            env["BDSE_V53_OPERATOR_PROFILE"]="1"
            kwargs["env"]=env
            cmd=rewritten
        return _subprocess.Popen(cmd,*args,**kwargs)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_profile_events(root: Path, expected_tokens: set[str]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "treatment" / "batches").glob("batch_*/v53_operator_profile_events.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not bool(r.get("pior_probe_fired", False)):
                continue
            tok = str(r.get("scenario_token") or r.get("pior_probe_scenario_token") or "")
            if not tok:
                raise RuntimeError(f"V53 profile event lacks scenario_token: {path}")
            if tok in rows:
                raise RuntimeError(f"V53 duplicate fired profile event for token={tok}")
            p = r.get("operator_trajectory_profile")
            if not isinstance(p, dict) or str(p.get("schema", "")) != PROFILE_SCHEMA:
                raise RuntimeError(f"V53 missing/invalid operator profile token={tok} file={path}")
            d = float(p.get("execution_contrast_linf", float("nan")))
            legacy_d = float(r.get("pior_probe_frozen_vs_runtime_incumbent_geometry_max_abs_error", float("nan")))
            if not (d >= 0.0 and abs(d - legacy_d) <= 1.0e-9):
                raise RuntimeError(f"V53 profile D replay mismatch token={tok}: profile={d} legacy={legacy_d}")
            endpoint = [float(x) for x in p.get("endpoint_signed", [])]
            temporal = [float(x) for x in p.get("cosine_modes_1_2", [])]
            if len(endpoint) != 4 or len(temporal) != 8 or int(p.get("trajectory_steps", 0)) < 2:
                raise RuntimeError(f"V53 profile shape invalid token={tok}: {p}")
            physical_equal = bool(r.get("pior_probe_frozen_equals_runtime_incumbent_physical", False))
            if physical_equal and (max([abs(x) for x in endpoint + temporal] + [abs(d)]) > 1.0e-8):
                raise RuntimeError(f"V53 physical-equal proposal has nonzero trajectory contrast token={tok}")
            rows[tok] = {
                "scenario_token": tok,
                "schema": PROFILE_SCHEMA,
                "execution_contrast_linf": d,
                "endpoint_signed": endpoint,
                "cosine_modes_1_2": temporal,
                "trajectory_steps": int(p["trajectory_steps"]),
                "frozen_equals_runtime_incumbent_physical": physical_equal,
                "frozen_proposal_trajectory_sha256": str(r.get("pior_probe_frozen_proposal_trajectory_sha256", "")),
                "runtime_incumbent_trajectory_sha256": str(r.get("pior_probe_runtime_incumbent_trajectory_sha256", "")),
            }
    if set(rows) != expected_tokens:
        miss = sorted(expected_tokens - set(rows))[:10]
        extra = sorted(set(rows) - expected_tokens)[:10]
        raise RuntimeError(f"V53 profile population mismatch {len(rows)}/{len(expected_tokens)} missing={miss} extra={extra}")
    return [rows[t] for t in sorted(rows)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--treatment-config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--nuplan-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--output-profiles", type=Path, required=True)
    p.add_argument("--output-report", type=Path, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--heartbeat-seconds", type=float, default=30.0)
    p.add_argument("--challenge", default="closed_loop_nonreactive_agents")
    a = p.parse_args()

    _safe._assert_frozen_base_runner()
    _base.subprocess = _ProfileSubprocessProxy()
    tokens, meta, raw_files = _base._manifest(a.manifest)
    if len(tokens) != 502 or len(set(tokens)) != 502:
        raise RuntimeError(f"V53 requires exact frozen 502 proposal population, got {len(tokens)}/{len(set(tokens))}")

    _metrics, summary = _base._run_arm(
        arm="treatment", gpu=str(a.gpu), cfg=a.treatment_config, checkpoint=a.checkpoint,
        tokens=tokens, meta=meta, nuplan_root=a.nuplan_root, challenge=str(a.challenge),
        output_root=a.output_root, workers=int(a.workers), batch_size=int(a.batch_size),
        resume=False, heartbeat_seconds=float(a.heartbeat_seconds), serialize_gpu_inference=False,
        profile_closed_loop=False, allow_legacy_full_arm_resume=False,
        first_batch_size=0, preflight_barrier=None,
    )

    profiles = _read_profile_events(a.output_root, set(tokens))
    a.output_profiles.parent.mkdir(parents=True, exist_ok=True)
    with a.output_profiles.open("w", encoding="utf-8") as f:
        for r in profiles:
            f.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    digest = _sha256(a.output_profiles)
    equal_count = sum(bool(r["frozen_equals_runtime_incumbent_physical"]) for r in profiles)
    report = {
        "audit": "v64_3_53_operator_profile_probe",
        "algorithm_version": "V64.3.53-EAF-ICER-POTR",
        "scientific_role": "state_only_preexecution_operator_trajectory_contrast_replay",
        "paired_outcome_labels_recollected": False,
        "arm_replayed": "treatment_only",
        "scenario_count": len(profiles),
        "scenario_token_sha256": _base._token_sha(tokens),
        "raw_db_count": len(raw_files),
        "profile_schema": PROFILE_SCHEMA,
        "profile_sha256": digest,
        "physical_equal_count": equal_count,
        "scalar_D_exact_replay_all": True,
        "metric_safe_child_runner": True,
        "arm_summary": summary,
        "pass": True,
    }
    a.output_report.parent.mkdir(parents=True, exist_ok=True)
    a.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "profiles": len(profiles), "profile_sha256": digest, "physical_equal": equal_count}, sort_keys=True))


if __name__ == "__main__":
    main()
