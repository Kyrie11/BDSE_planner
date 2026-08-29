from __future__ import annotations

"""Prepare V64.3.50 paired selected-outcome closed-loop probe configs.

V50 needs two things at the intervention event:
  1) the exact full-set RSMR proposal identity, frozen before post-selection; and
  2) the already-defined V47/V48/V49 Q/P/E consequence coordinates evaluated at
     that same *live* event state.

The persisted V49 SIIR runtime config supplies both.  The V50 probe overrides its
historical accept/veto action with the paired CONTROL/TREATMENT assignment, so
V49 risk cannot influence the causal treatment while its Q/P/E instrumentation
remains available.
"""

import argparse
import copy
from pathlib import Path
import yaml


def _make(src: dict, role: str) -> dict:
    cfg = copy.deepcopy(src)
    cfg["fallback"] = dict(cfg.get("fallback", {}) or {})
    cfg["fallback"]["enabled"] = False
    cfg["selected_outcome_probe"] = {
        "enabled": True,
        "role": role,
        "one_shot": True,
        "proposal_source": "frozen_full_set_RSMR_before_post_selection",
        "state_source": "live_same_event_QPE_from_frozen_V49_coordinate_definitions",
        "require_live_qpe": True,
        "historical_post_selection_action_ignored_for_treatment_assignment": True,
        "post_intervention_policy": "preserve_incumbent",
        "teacher_or_logged_future_inputs": False,
    }
    cfg.setdefault("metadata", {})["algorithm_version"] = "V64.3.50-EAF-ICER-SIOR-PROBE"
    cfg.setdefault("metadata", {})["selected_outcome_probe_role"] = role
    cfg.setdefault("metadata", {})["selected_outcome_event_state_alignment"] = "first_live_RSMR_proposal"
    cfg.setdefault("provenance", {})["algorithm_version"] = "V64.3.50-EAF-ICER-SIOR-PROBE"
    cfg.setdefault("experiment", {})["algorithm"] = "V64.3.50-EAF-ICER-SIOR-PROBE"
    return cfg


def _validate_source(src: dict) -> None:
    try:
        ic = src["runtime"]["decisive_frontier_value"]["incumbent_contrastive_extremal_recovery"]
        sc = ic["selection_conditioned_intervention_recovery"]
    except Exception as exc:
        raise SystemExit("V50 ENGINEERING STOP: source config does not expose frozen ICER/RSMR path") from exc
    if not bool(ic.get("enabled", False)):
        raise SystemExit("V50 ENGINEERING STOP: source config does not enable ICER/RSMR path")
    # Event-state alignment intentionally uses V49 solely because it carries the
    # frozen Q/P/E runtime coordinate computation.  Proposal identity is still
    # scir_proposal_action, which is frozen before this post-selection stage.
    if not bool(sc.get("post_selection_value_enabled", False)):
        raise SystemExit("V50 ENGINEERING STOP: probe source must expose frozen Q/P/E post-selection instrumentation")
    if str(sc.get("post_selection_value_mode", "")).strip().lower() != "endpoint_potential_quality_operator_conditioned_risk_retention":
        raise SystemExit("V50 ENGINEERING STOP: probe source is not the frozen V49 Q/P/E operator-state runtime mode")
    rcfg = sc.get("operator_conditioned_risk_retention", {}) or {}
    if bool(rcfg.get("use_extremal_multiplicity", True)):
        raise SystemExit("V50 ENGINEERING STOP: source config unexpectedly re-enables V48 multiplicity")
    names = [str(x) for x in rcfg.get("feature_names", [])]
    expected = ["quality_value", "prospective_response_increment", "ego_reference_increment", "log_extremal_multiplicity"]
    if names != expected:
        raise SystemExit(f"V50 ENGINEERING STOP: source Q/P/E operator-state schema changed: {names}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare V50 control/treatment probe configs from frozen V49 Q/P/E runtime config")
    ap.add_argument("--source-config", "--rsmr-config", dest="source_config", type=Path, required=True,
                    help="Frozen V49 SIIR config. --rsmr-config is retained as a CLI alias only.")
    ap.add_argument("--control-output", type=Path, required=True)
    ap.add_argument("--treatment-output", type=Path, required=True)
    a = ap.parse_args()
    src = yaml.safe_load(a.source_config.read_text(encoding="utf-8"))
    if not isinstance(src, dict):
        raise SystemExit("V50 ENGINEERING STOP: source config is not a mapping")
    _validate_source(src)
    for role, out in [("control", a.control_output), ("treatment", a.treatment_output)]:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(_make(src, role), sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"wrote {role}: {out}")


if __name__ == "__main__":
    main()
