from __future__ import annotations

"""Prepare V64.3.50 paired selected-outcome closed-loop probe configs."""

import argparse
import copy
from pathlib import Path
import yaml


def _make(src: dict, role: str) -> dict:
    cfg = copy.deepcopy(src)
    cfg["fallback"] = dict(cfg.get("fallback", {}) or {})
    cfg["fallback"]["enabled"] = False
    # The probe is evidence collection only.  Proposal generation remains the
    # exact full-set RSMR config supplied by the previous frozen version.
    cfg["selected_outcome_probe"] = {
        "enabled": True,
        "role": role,
        "one_shot": True,
        "proposal_source": "frozen_full_set_RSMR",
        "post_intervention_policy": "preserve_incumbent",
        "teacher_or_logged_future_inputs": False,
    }
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.50-EAF-ICER-SIOR-PROBE"
    cfg.setdefault("metadata", {})["selected_outcome_probe_role"] = role
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.50-EAF-ICER-SIOR-PROBE"
    cfg.setdefault("experiment", {})["algorithm"] = "V64.3.50-EAF-ICER-SIOR-PROBE"
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare V50 control/treatment probe configs from frozen RSMR config")
    ap.add_argument("--rsmr-config", type=Path, required=True)
    ap.add_argument("--control-output", type=Path, required=True)
    ap.add_argument("--treatment-output", type=Path, required=True)
    a = ap.parse_args()
    src = yaml.safe_load(a.rsmr_config.read_text(encoding="utf-8"))
    if not isinstance(src, dict):
        raise SystemExit("V50 ENGINEERING STOP: RSMR config is not a mapping")
    # Require the frozen direct RSMR path rather than a downstream risk head.
    try:
        ic = src["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
        sc = ic["selection_conditioned_intervention_recovery"]
    except Exception as exc:
        raise SystemExit("V50 ENGINEERING STOP: source config does not expose frozen ICER/RSMR path") from exc
    if not bool(ic.get("enabled", False)):
        raise SystemExit("V50 ENGINEERING STOP: source config does not enable ICER/RSMR path")
    if bool(sc.get("post_selection_value_enabled", False)):
        raise SystemExit("V50 ENGINEERING STOP: probe source must be pure frozen RSMR, not a post-selection value/risk config")
    # The RSMR config itself is frozen by its upstream result artifact.  The V50
    # probe only adds an external paired-outcome collection operator.
    for role, out in [("control", a.control_output), ("treatment", a.treatment_output)]:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(_make(src, role), sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"wrote {role}: {out}")


if __name__ == "__main__":
    main()
