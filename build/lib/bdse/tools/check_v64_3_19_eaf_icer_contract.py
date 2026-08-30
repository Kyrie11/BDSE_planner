from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from bdse.planner.tournament import (
    _DACER_FEATURE_NAMES,
    _DALER_FEATURE_NAMES,
    _icer_quadratic_interaction_features,
)


def _finite_vector(x: list[object]) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in x)
    except (TypeError, ValueError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit the frozen V64.3.19 EAF-ICER deployment contract.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect", choices=["raw", "icer-scalar", "icer-dual"], required=True)
    ap.add_argument("--output")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    runtime = cfg.get("runtime", {}) or {}
    frontier = runtime.get("decisive_frontier_value", {}) or {}
    icer = frontier.get("incumbent_contrastive_extremal_recovery", {}) or {}
    legacy_blocks = [
        frontier.get("one_sided_intervention", {}) or {},
        frontier.get("learned_intervention_reliability", {}) or {},
        frontier.get("reliability_aware_extremal_reranking", {}) or {},
        frontier.get("deployment_aligned_listwise_extremal_reliability", {}) or {},
        frontier.get("deployment_admissible_counterfactual_extremal_recovery", {}) or {},
    ]
    metadata = cfg.get("metadata", {}) or {}
    evidence = cfg.get("evidence", {}) or {}
    selector = cfg.get("selector", {}) or {}
    utility = ((cfg.get("tournament", {}) or {}).get("utility_refinement", {}) or {})
    guard = runtime.get("pair_action_anchor_guard", {}) or {}
    dual = runtime.get("dual_certificate", {}) or {}

    fitted = a.expect != "raw"
    expected_policy = "scalar_only" if a.expect == "icer-scalar" else "dual_equal_mean"
    dummy = np.zeros((1, len(_DACER_FEATURE_NAMES)), dtype=np.float64)
    _, scalar_names, scalar_base = _icer_quadratic_interaction_features(dummy, list(_DACER_FEATURE_NAMES), "scalar_interaction")
    _, profile_names, profile_base = _icer_quadratic_interaction_features(dummy, list(_DACER_FEATURE_NAMES), "profile_interaction")

    sn = list(icer.get("support_feature_names", []) or [])
    sm = list(icer.get("support_feature_mean", []) or [])
    ss = list(icer.get("support_feature_std", []) or [])
    sw = list(icer.get("support_weights", []) or [])
    sdn = list(icer.get("scalar_dominance_feature_names", []) or [])
    sdb = list(icer.get("scalar_dominance_base_feature_names", []) or [])
    sdm = list(icer.get("scalar_dominance_feature_mean", []) or [])
    sds = list(icer.get("scalar_dominance_feature_std", []) or [])
    sdw = list(icer.get("scalar_dominance_weights", []) or [])
    pdn = list(icer.get("profile_dominance_feature_names", []) or [])
    pdb = list(icer.get("profile_dominance_base_feature_names", []) or [])
    pdm = list(icer.get("profile_dominance_feature_mean", []) or [])
    pds = list(icer.get("profile_dominance_feature_std", []) or [])
    pdw = list(icer.get("profile_dominance_weights", []) or [])

    version = metadata.get("algorithm_version")
    provenance_version = (cfg.get("provenance", {}) or {}).get("algorithm_version")
    checks = {
        "version": version == "V64.3.19-EAF-ICER-DARM-DBR" and provenance_version == version,
        "frontier_enabled": bool(frontier.get("enabled", False)),
        "legacy_learned_arms_disabled": not any(bool(x.get("enabled", False)) for x in legacy_blocks),
        "icer_state": bool(icer.get("enabled", False)) == fitted,
        "feature_instrumentation": bool(icer.get("instrument_features", False)),
        "model_type": (
            str(icer.get("model_type", "")) == "decomposed_anchor_support_plus_dual_view_quadratic_incumbent_contrastive_reliability"
            if fitted else True
        ),
        "guard_admissible_contract": bool(icer.get("require_guard_admissible", False)) and bool(icer.get("require_safe_available_for_learned_intervention", False)),
        "utility_not_gate_or_feature": str(icer.get("utility_equivalence_role", "")) == "diagnostic_exact_tiebreak_only_not_hard_mask_not_learned_feature",
        "zero_reference_logits": abs(float(icer.get("anchor_logit", 99.0))) < 1e-12 and abs(float(icer.get("incumbent_logit", 99.0))) < 1e-12,
        "fixed_threshold_policy": str(icer.get("threshold_policy", "")) == "fixed_zero_direct_counterfactual_log_odds_no_validation_threshold_sweep",
        "dominance_policy": str(icer.get("dominance_policy", "dual_equal_mean")) == expected_policy if fitted else True,
        "support_schema": sn == list(_DALER_FEATURE_NAMES) if fitted else sn == [],
        "support_vector_contract": (
            len(sn) == len(sm) == len(ss) == len(sw) == len(_DALER_FEATURE_NAMES)
            and _finite_vector([*sm, *ss, *sw, icer.get("support_bias", 0.0)])
            and all(float(x) > 0 for x in ss)
        ) if fitted else len(sn) == len(sm) == len(ss) == len(sw) == 0,
        "scalar_interaction_schema": sdn == scalar_names and sdb == scalar_base if fitted else sdn == [] and sdb == [],
        "scalar_interaction_vector_contract": (
            len(sdn) == len(sdm) == len(sds) == len(sdw) == len(scalar_names)
            and _finite_vector([*sdm, *sds, *sdw, icer.get("scalar_dominance_bias", 0.0)])
            and all(float(x) > 0 for x in sds)
        ) if fitted else len(sdn) == len(sdm) == len(sds) == len(sdw) == 0,
        "profile_interaction_schema": pdn == profile_names and pdb == profile_base if fitted else pdn == [] and pdb == [],
        "profile_interaction_vector_contract": (
            len(pdn) == len(pdm) == len(pds) == len(pdw) == len(profile_names)
            and _finite_vector([*pdm, *pds, *pdw, icer.get("profile_dominance_bias", 0.0)])
            and all(float(x) > 0 for x in pds)
        ) if fitted else len(pdn) == len(pdm) == len(pds) == len(pdw) == 0,
        "direct_counterfactual_objective": (
            str(icer.get("training_dominance_target", "")) == "teacher_alternative_better_than_max_anchor_and_frozen_incumbent"
            and str(icer.get("training_dominance_objective", "")) == "unweighted_direct_counterfactual_bce_over_fixed_quadratic_evidence_interaction_map"
        ) if fitted else True,
        "objective_weights_fixed": (
            (icer.get("objective_weights", {}) or {}) == {"support": 1.0, "dominance_bce": 1.0}
        ) if fitted else True,
        "utility_refinement_frozen_on": bool(utility.get("enabled", False)) and bool(utility.get("pair_certificate_enabled", False)),
        "one_sided_guard_frozen_on": bool(guard.get("enabled", False)) and abs(float(guard.get("flip_margin", 99.0)) - 0.015) < 1e-12 and abs(float(guard.get("score_margin", 99.0))) < 1e-12,
        "evidence_certificate_frozen_on": bool(dual.get("enabled", False)) and bool(dual.get("require_evidence_certificate_before_residual_flip", False)) and abs(float(dual.get("min_evidence_certificate_fraction_for_residual_flip", 0.0)) - 1.0) < 1e-12,
        "robust_margin_corrections_frozen_zero": abs(float(dual.get("residual_beta_uncertainty", 99.0))) < 1e-12 and abs(float(dual.get("residual_epsilon_cal", dual.get("residual_epsilon", 99.0)))) < 1e-12,
        "budget_cap_B16": int(evidence.get("budget", -1)) == 16 and int(metadata.get("fixed_planner_interface_evidence_budget", -1)) == 16,
        "topM24": int(selector.get("proposal_top_m", -1)) == 24 and int(metadata.get("fixed_proposal_top_m", -1)) == 24,
        "evaluation_only": bool((cfg.get("training", {}) or {}).get("evaluation_only", False)),
        "no_probability_or_margin_tuning_fields": not any(k in icer for k in ["min_probability", "probability_threshold", "dominance_threshold", "support_threshold", "margin_threshold"]),
    }
    report = {"audit": "v64_3_19_eaf_icer_contract", "expect": a.expect, "passed": all(checks.values()), "checks": checks}
    if a.output:
        out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
