from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from bdse.tools.check_v64_3_19_eaf_icer_contract import _finite_vector
from bdse.planner.tournament import _DALER_FEATURE_NAMES


def _icer(cfg: dict[str, Any]) -> dict[str, Any]:
    return ((((cfg.get("runtime", {}) or {}).get("decisive_frontier_value", {}) or {}).get(
        "incumbent_contrastive_extremal_recovery", {}
    ) or {}))


def _frozen_head_payload(ic: dict[str, Any]) -> dict[str, Any]:
    prefixes = ["support_", "scalar_dominance_", "profile_dominance_"]
    keep = {
        "model_type", "dominance_policy", "anchor_logit", "incumbent_logit",
        "require_guard_admissible", "require_safe_available_for_learned_intervention",
        "training_support_target", "training_dominance_target", "threshold_policy",
        "utility_equivalence_role", "training_support_objective", "training_dominance_objective",
        "objective_weights", "selection_operator",
    }
    out: dict[str, Any] = {}
    for k, v in ic.items():
        if k in keep or any(k.startswith(p) for p in prefixes):
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit V64.3.20 deployment-complete ICER contract.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect", choices=["raw", "icer-dc-scalar", "icer-dc-dual"], required=True)
    ap.add_argument("--frozen-v19-config")
    ap.add_argument("--output")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    ic = _icer(cfg)
    runtime = cfg.get("runtime", {}) or {}
    frontier = runtime.get("decisive_frontier_value", {}) or {}
    metadata = cfg.get("metadata", {}) or {}
    provenance = cfg.get("provenance", {}) or {}
    fitted = a.expect != "raw"
    expected_policy = "scalar_only" if a.expect == "icer-dc-scalar" else "dual_equal_mean"
    checks: dict[str, bool] = {
        "version": metadata.get("algorithm_version") == "V64.3.20-EAF-ICER-DC-DARM-DBR" and provenance.get("algorithm_version") == metadata.get("algorithm_version"),
        "frontier_enabled": bool(frontier.get("enabled", False)),
        "icer_state": bool(ic.get("enabled", False)) == fitted,
        "policy": (str(ic.get("dominance_policy", "")) == expected_policy) if fitted else True,
        "zero_reference_logits": abs(float(ic.get("anchor_logit", 99.0))) < 1e-12 and abs(float(ic.get("incumbent_logit", 99.0))) < 1e-12,
        "safe_domain_learning_only": bool(ic.get("require_safe_available_for_learned_intervention", False)),
        "deployment_complete_all_flagged_policy": (str(ic.get("all_flagged_policy", "")) == "preserve_legacy_for_structural_guard") if fitted else True,
        "no_threshold_sweep": str(ic.get("threshold_policy", "")) == "fixed_zero_direct_counterfactual_log_odds_no_validation_threshold_sweep",
        "support_schema": list(ic.get("support_feature_names", []) or []) == list(_DALER_FEATURE_NAMES) if fitted else list(ic.get("support_feature_names", []) or []) == [],
        "evaluation_only": bool((cfg.get("training", {}) or {}).get("evaluation_only", False)),
    }

    if fitted:
        if not a.frozen_v19_config:
            raise SystemExit("--frozen-v19-config is required for fitted V64.3.20 configs")
        old = yaml.safe_load(Path(a.frozen_v19_config).read_text(encoding="utf-8"))
        old_ic = _icer(old)
        checks["v19_train_only_heads_bitwise_semantic_identity"] = _frozen_head_payload(ic) == _frozen_head_payload(old_ic)
        checks["no_refit_marker"] = str(ic.get("training_reuse", "")) == "exact_v64_3_19_train_only_heads_no_refit"
        # Ensure all learned numeric vectors remain finite after copying.
        numeric = []
        for k, v in ic.items():
            if any(k.startswith(p) for p in ["support_", "scalar_dominance_", "profile_dominance_"]):
                if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                    numeric.extend(v)
                elif k.endswith("bias") and isinstance(v, (int, float)):
                    numeric.append(v)
        checks["frozen_head_numeric_finite"] = _finite_vector(numeric)
    report = {"audit": "v64_3_20_eaf_icer_dc_contract", "expect": a.expect, "passed": all(checks.values()), "checks": checks}
    if a.output:
        p = Path(a.output); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
