from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from bdse.config import load_config


def run_contract(config_path: str, expect: str = "raw") -> dict:
    cfg = load_config(config_path)
    runtime = cfg.get("runtime", {}) or {}
    frontier = runtime.get("decisive_frontier_value", {}) or {}
    ocfi = frontier.get("one_sided_intervention", {}) or {}
    model = cfg.get("model", {}) or {}
    training = cfg.get("training", {}) or {}
    selector = cfg.get("selector", {}) or {}
    evidence = cfg.get("evidence", {}) or {}
    meta = str((cfg.get("metadata", {}) or {}).get("algorithm_version", ""))
    prov = str((cfg.get("provenance", {}) or {}).get("algorithm_version", ""))
    positive_losses = {str(k) for k, v in (training.get("loss_weights", {}) or {}).items() if abs(float(v)) > 1e-12}
    calibrated = expect == "calibrated"
    q = ocfi.get("calibration_quantile", None)
    q_finite = q is not None and math.isfinite(float(q)) and float(q) >= 0.0
    checks = {
        "version_is_v64_3_14_eaf_ocfi": meta == "V64.3.14-EAF-OCFI-DARM-DBR" and prov == meta,
        "fixed_evidence_budget_B16": int(evidence.get("budget", -1)) == 16,
        "fixed_proposal_topm_M24": int(selector.get("proposal_top_m", -1)) == 24,
        "darm_runtime_unchanged": str(runtime.get("pair_tournament_aggregation_mode", "")).strip().lower() == "decisive_anchor_margin",
        "eaf_model_enabled": bool((model.get("decisive_anchor_frontier_value", {}) or {}).get("enabled", False)),
        "eaf_runtime_enabled": bool(frontier.get("enabled", False)),
        "frontier_consumes_fixed_selected_B": str(frontier.get("evidence_source", "")) == "fixed_runtime_selected_B",
        "no_training_in_v64_3_14": list(training.get("trainable_modules", [])) == [] and positive_losses == set(),
        "acquisition_training_terminally_disabled": not bool((training.get("budgeted_decisive_margin_utility", {}) or {}).get("enabled", False)),
        "ocfi_state_matches_role": bool(ocfi.get("enabled", False)) if calibrated else not bool(ocfi.get("enabled", False)),
        "ocfi_requires_frontier": bool(ocfi.get("require_frontier_active", False)),
        "ocfi_method_is_one_sided_split_conformal": str(ocfi.get("calibration_method", "")).startswith("split_conformal_overprediction"),
        "ocfi_normalization_supported": str(ocfi.get("normalization", "")).strip().lower() in {"attribution", "none"},
        "calibration_quantile_valid": q_finite,
        "attribution_floor_positive": float(ocfi.get("attribution_scale_floor", 0.0)) > 0.0,
        "existing_evidence_certificate_not_relaxed": bool((runtime.get("dual_certificate", {}) or {}).get("require_evidence_certificate_before_residual_flip", False))
            and float((runtime.get("dual_certificate", {}) or {}).get("min_evidence_certificate_fraction_for_residual_flip", 0.0)) >= 1.0,
    }
    return {
        "audit": "v64_3_14_eaf_ocfi_contract",
        "config": config_path,
        "expect": expect,
        "checks": checks,
        "pass": all(checks.values()),
        "note": (
            "V64.3.14 is evaluation/calibration-only. It reuses the frozen V64.3.13 EAF value checkpoint, "
            "keeps B=16/M=24 and the terminal acquisition stop, and changes only the one-sided condition "
            "under which an EAF challenger may replace the selected-local/DARM anchor."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect", choices=["raw", "calibrated"], default="raw")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    r = run_contract(args.config, args.expect)
    text = json.dumps(r, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not r["pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
