"""V53 state-only nuPlan entrypoint.

Preserves the byte-identical historical planner and installs two process-local
engineering hooks only for the V53 treatment replay:
  1) V50.5 metric-engine serialization;
  2) a read-only wrapper around the frozen selected-outcome probe that records
     proposal-vs-runtime-incumbent trajectory contrast to a sidecar *before*
     the treatment action is executed.

The hook never changes the returned action or diagnostic dictionary and never
reads closed-loop outcome labels.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import runpy
import threading
from typing import Any

import numpy as np

from bdse.planner.paired_operator_trajectory_retention import trajectory_contrast_profile
from bdse.tools.nuplan_metric_safe_run_simulation import install_metric_engine_serialization

_WRITE_LOCK = threading.RLock()
_PATCH_MARKER = "_bdse_v64_3_53_operator_profile_sidecar"


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes(order="C")).hexdigest()


def install_operator_profile_sidecar() -> None:
    from bdse.planner.nuplan_planner import BDSEPlannerCore

    original = BDSEPlannerCore._apply_selected_outcome_probe
    if bool(getattr(original, _PATCH_MARKER, False)):
        return

    @functools.wraps(original)
    def wrapped(self: Any, tournament: Any, action: int, candidates: Any, current_input: Any | None = None):
        chosen, diag = original(self, tournament, action, candidates, current_input)
        if os.environ.get("BDSE_V53_OPERATOR_PROFILE", "0").strip() != "1":
            return chosen, diag
        if not bool(diag.get("pior_probe_fired", False)) or str(diag.get("pior_probe_arm", "")) != "treatment":
            return chosen, diag
        target = getattr(self, "_pior_bound_target", None)
        if not isinstance(target, dict):
            raise RuntimeError("V53 profile hook missing frozen manifest-bound target")
        expected = np.asarray(target.get("frozen_proposal_trajectory", []), dtype=np.float32)
        ct = np.asarray(getattr(candidates, "trajectories", []), dtype=np.float32)
        baseline = int(diag.get("pior_probe_baseline_action", -1))
        if expected.ndim != 2 or expected.shape[0] < 2 or ct.ndim != 3 or not (0 <= baseline < ct.shape[0]):
            raise RuntimeError("V53 profile hook cannot recover proposal/runtime-incumbent trajectories")
        incumbent = np.ascontiguousarray(ct[baseline], dtype=np.float32)
        if incumbent.shape != expected.shape:
            raise RuntimeError(f"V53 profile hook shape mismatch {expected.shape} vs {incumbent.shape}")
        profile = trajectory_contrast_profile(expected, incumbent)
        d = float(profile["execution_contrast_linf"])
        legacy_d = float(diag.get("pior_probe_frozen_vs_runtime_incumbent_geometry_max_abs_error", float("nan")))
        if not np.isfinite(legacy_d) or abs(d - legacy_d) > 1.0e-9:
            raise RuntimeError(f"V53 profile hook D replay mismatch profile={d} legacy={legacy_d}")
        diag_path = os.environ.get("BDSE_CLOSED_LOOP_DIAG", "").strip()
        if not diag_path:
            raise RuntimeError("V53 profile hook requires BDSE_CLOSED_LOOP_DIAG sidecar anchor")
        out = Path(diag_path).with_name("v53_operator_profile_events.jsonl")
        row = {
            "scenario_token": str(diag.get("pior_probe_scenario_token", "")),
            "pior_probe_arm": "treatment",
            "pior_probe_fired": True,
            "pior_probe_event_count": int(diag.get("pior_probe_event_count", 0)),
            "pior_probe_physical_identity_contract": str(diag.get("pior_probe_physical_identity_contract", "")),
            "pior_probe_contract_same_frozen_proposal_or_incumbent": bool(diag.get("pior_probe_contract_same_frozen_proposal_or_incumbent", False)),
            "pior_probe_contract_no_rerank_second_best_fallback": bool(diag.get("pior_probe_contract_no_rerank_second_best_fallback", False)),
            "pior_probe_frozen_equals_runtime_incumbent_physical": bool(diag.get("pior_probe_frozen_equals_runtime_incumbent_physical", False)),
            "pior_probe_frozen_proposal_trajectory_sha256": _sha(expected),
            "pior_probe_runtime_incumbent_trajectory_sha256": _sha(incumbent),
            "pior_probe_frozen_vs_runtime_incumbent_geometry_max_abs_error": legacy_d,
            "operator_trajectory_profile": profile,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return chosen, diag

    setattr(wrapped, _PATCH_MARKER, True)
    setattr(BDSEPlannerCore, "_apply_selected_outcome_probe", wrapped)


def main() -> None:
    if os.environ.get("BDSE_PIOR_METRIC_ENGINE_SERIALIZATION") != "1":
        raise RuntimeError("V53 state probe requires V50.5 metric-engine serialization")
    if os.environ.get("BDSE_V53_OPERATOR_PROFILE") != "1":
        raise RuntimeError("V53 operator-profile wrapper requires BDSE_V53_OPERATOR_PROFILE=1")
    install_metric_engine_serialization()
    install_operator_profile_sidecar()
    print("[BDSE-V53-POTR-PROFILE] metric-safe pre-execution operator-profile sidecar enabled; planner source/action unchanged", flush=True)
    runpy.run_module("nuplan.planning.script.run_simulation", run_name="__main__")

if __name__ == "__main__":
    main()
