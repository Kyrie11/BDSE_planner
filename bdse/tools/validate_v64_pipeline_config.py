from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


CACHE_SOURCES = {
    "cache",
    "cached",
    "cache_only",
    "cache_prefix_runtime_extension",
    "verified_prefix_runtime_extension",
    "cache_supported_prefix",
}
RUNTIME_SOURCES = {
    "runtime",
    "runtime_recompute",
    "recompute",
    "canonical_runtime",
    "cache_verified",
    "verified_cache",
}
V64_3_PREFIXES = (
    "v64_3_cc_aocc_apwcca",
    "v64_3_2_cc_aocc_apwcca", "v64_3_2_cc_aocc_apwrcca",
    "v64_3_3_cc_aocc_apwcca", "v64_3_3_cc_aocc_apwrcca",
    "v64_3_4_cc_aocc_mrbcca",
    "v64_3_4_cc_aocc_fpcca",
    "v64_3_4_cc_aocc_apwrcca",
    "v64_3_5_cc_aocc_ccbr",
    "v64_3_6_cc_aocc_ccbr",
    "v64_3_7_cc_aocc_darm_dbr",
    "v64_3_8_cc_aocc_bdmu",
    "v64_3_9_cc_aocc_af_bdmu",
)
V64_3_ALGORITHM_VERSIONS = {
    "V64.3-CC-AOCC-AP-WCCA",
    "V64.3.1-CC-AOCC-AP-WCCA",
    "V64.3.1-CC-AOCC-AP-WCCA-DA-EPC",
    "V64.3.2-CC-AOCC-AP-WCCA-DA-EPC-SCREENFIX",
    "V64.3.2-CC-AOCC-AP-WRCCA-DA-EPC",
    "V64.3.3-CC-AOCC-AP-WCCA-DA-EPC-FULLSUPPORT",
    "V64.3.3-CC-AOCC-AP-WRCCA-DA-EPC-FULLSUPPORT",
    "V64.3.3-CC-AOCC-AP-WRCCA-LCV-DA-EPC",
    "V64.3.4-CC-AOCC-MR-BCCA-DA-EPC",
    "V64.3.4-CC-AOCC-MR-BCCA-LCV-DA-EPC",
    "V64.3.4-CC-AOCC-FPCCA-DA-EPC",
    "V64.3.4-CC-AOCC-FPCCA-LBA-DA-EPC",
    "V64.3.4-CC-AOCC-AP-WRCCA-LCV-DA-EPC",
    "V64.3.5-CC-AOCC-CCBR-DA-EPC",
    "V64.3.5-CC-AOCC-CCBR-LEA-DA-EPC",
    "V64.3.6-CC-AOCC-CCBR-LEA-LOCAL-DA-EPC",
    "V64.3.6-CC-AOCC-CCBR-BCHA-LEA-DA-EPC",
    "V64.3.6-CC-AOCC-CCBR-LBPR-LEA-DA-EPC",
    "V64.3.6-CC-AOCC-CCBR-BCHA-LBPR-LEA-DA-EPC",
    "V64.3.7-CC-AOCC-DARM-DBR-DA-EPC",
    "V64.3.7-CC-AOCC-DARM-DBR-DA-EPC-LITERAL",
    "V64.3.8-CC-AOCC-BDMU-DARM-DBR",
    "V64.3.8-CC-AOCC-BDMU-R1-DARM-DBR",
    "V64.3.8-CC-AOCC-BDMU-NOCOST-DARM-DBR",
    "V64.3.9-CC-AOCC-AF-BDMU-DARM-DBR",
}
V64_3_REQUIRED_TRAINABLE = {
    "critical_proposal_adapter",
    "residual_action_head",
    "residual_action_var_head",
    "residual_set_atom_head",
    "residual_set_action_head",
}
V64_3_FORBIDDEN_TRAINABLE = {
    "proposal_head",
    "family_head",
    "family_embed",
    "family_activity_proj",
    "proposal_feature_proj",
    "query_extension_proj",
}


def _load(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return obj


def _algorithm_version(cfg: dict[str, Any], section: str) -> str:
    value = (cfg.get(section, {}) or {}).get("algorithm_version", "")
    return str(value).strip()


def _check(path: Path, role: str, expected_family: str | None) -> dict[str, Any]:
    cfg = _load(path)
    exp = cfg.get("experiment", {}) or {}
    model = cfg.get("model", {}) or {}
    evidence = cfg.get("evidence", {}) or {}
    runtime = cfg.get("runtime", {}) or {}
    adapter = model.get("query_extension_adapter", {}) or {}
    critical_adapter = model.get("critical_proposal_adapter", {}) or {}
    lbpr = model.get("literal_boundary_pair_adapter", {}) or {}
    dbr = model.get("decisive_boundary_pair_adapter", {}) or {}
    family_coupling = critical_adapter.get("family_coupling", {}) or {}
    training_cfg = cfg.get("training", {}) or {}
    checkpoint_cfg = cfg.get("checkpoint_loading", {}) or {}
    metadata_cfg = cfg.get("metadata", {}) or {}
    provenance_cfg = cfg.get("provenance", {}) or {}
    crit = training_cfg.get("exact_winner_flip_criticality", {}) or {}
    source = str(runtime.get("dense_query_feature_source", "cache_or_recompute")).strip().lower()
    exp_name = str(exp.get("name", ""))
    metadata_version = _algorithm_version(cfg, "metadata")
    provenance_version = _algorithm_version(cfg, "provenance")

    checks: dict[str, bool] = {
        "experiment_is_v64_family": exp_name.startswith(("v64_", "v64.")),
        "fixed_evidence_budget_16": int(evidence.get("budget", -1)) == 16,
        "query_feature_dim_18": int(model.get("query_feature_dim", -1)) == 18,
        "legacy_support_dim_12": int(model.get("query_legacy_support_dim", -1)) == 12,
        "query_extension_adapter_enabled": bool(adapter.get("enabled", False)),
        "query_extension_adapter_zero_init": bool(adapter.get("zero_init", False)),
        "query_source_supported": source in CACHE_SOURCES | RUNTIME_SOURCES,
    }
    if role == "train":
        v6437_value_isolation = metadata_version.startswith("V64.3.7-")
        v6438_bdmu_isolation = metadata_version.startswith(("V64.3.8-", "V64.3.9-"))
        checks.update(
            {
                "literal_winner_flip_criticality_enabled": bool(crit.get("enabled", False)) or v6437_value_isolation or v6438_bdmu_isolation,
                "teacher_interface_target": str(crit.get("target_source", "")).lower()
                in {"teacher", "teacher_interface", "teacher_exact"},
            }
        )

    strict_v64_3 = expected_family in {"v64.3", "v64.3.1", "v64.3.2", "v64.3.3", "v64.3.4", "v64.3.5", "v64.3.6", "v64.3.7", "v64.3.8", "v64.3.9", "v64_3"}
    if strict_v64_3:
        trainable = {str(x) for x in training_cfg.get("trainable_modules", [])}
        checks.update(
            {
                # Do not merely test "v64 family": the V64.3 wrapper must never
                # accept a stale V64.2 config inherited through generic shell vars.
                "v64_3_exact_experiment_family": exp_name.startswith(V64_3_PREFIXES),
                "v64_3_metadata_version": metadata_version in V64_3_ALGORITHM_VERSIONS,
                "v64_3_provenance_version": provenance_version in V64_3_ALGORITHM_VERSIONS,
                "v64_3_metadata_provenance_match": metadata_version == provenance_version,
                "v64_3_query_extension_nominally_noop": abs(float(adapter.get("scale", 1.0))) <= 1.0e-12,
                "v64_3_critical_proposal_adapter_enabled": (
                    (not metadata_version.startswith("V64.3.7-"))
                    or (not bool(critical_adapter.get("enabled", False)))
                ),
                "v64_3_critical_proposal_adapter_zero_init": bool(critical_adapter.get("zero_init", False)),
                "v64_3_winner_conditioning": (
                    str(critical_adapter.get("conditioning", "")) in {
                        "frozen_base_winner_action",
                        "frozen_base_winner_rival_actions",
                        "frozen_base_multi_rival_frontier",
                        "frozen_base_frontier_pair_set",
                        "complete_candidate_boundary_routing",
                    }
                    or (metadata_version.startswith("V64.3.7-") and str(critical_adapter.get("conditioning", "")).startswith("disabled_v64_3_7"))
                ),
                "v64_3_decision_aligned_exact_certificate": str(
                    (cfg.get("selector", {}) or {}).get("evidence_certificate_mode", "")
                ).strip().lower()
                in {
                    "exact_downstream_winner_preservation",
                    "exact_winner_preservation",
                    "decision_aligned_exact",
                    "da_epc",
                },
            }
        )
        if role == "train":
            if metadata_version.startswith(("V64.3.8-", "V64.3.9-")):
                loss_weights = training_cfg.get("loss_weights", {}) or {}
                positive_losses = {
                    str(k)
                    for k, v in loss_weights.items()
                    if abs(float(v)) > 1.0e-12
                }
                bdmu_cfg = training_cfg.get("budgeted_decisive_margin_utility", {}) or {}
                checks.update(
                    {
                        "v64_3_8_bdmu_enabled": bool(bdmu_cfg.get("enabled", False)),
                        "v64_3_8_bdmu_is_only_positive_loss": positive_losses
                        == {"budgeted_decisive_margin_utility"},
                        "v64_3_8_bdmu_budget_feasible_exchange": str(
                            bdmu_cfg.get("exchange_mode", "")
                        ).strip().lower()
                        == "best_budget_feasible_single_exchange",
                        "v64_3_8_literal_critical_objective_disabled": not bool(crit.get("enabled", False)),
                        "v64_3_8_family_coupling_disabled": not bool(family_coupling.get("enabled", False)),
                        "v64_3_8_critical_adapter_enabled": bool(critical_adapter.get("enabled", False)),
                    }
                )
                if metadata_version.startswith("V64.3.9-"):
                    checks.update(
                        {
                            "v64_3_9_adaptive_frontier": str(bdmu_cfg.get("rival_mode", "")).strip().lower() == "adaptive_frontier",
                            "v64_3_9_frontier_expands_rivals": int(bdmu_cfg.get("rival_max_count", 0)) > int(bdmu_cfg.get("rival_min_count", 0)) >= 1,
                            "v64_3_9_worst_rival_term_active": float(bdmu_cfg.get("worst_rival_weight", 0.0)) > 0.0,
                            "v64_3_9_topm_swap_rank_active": float(bdmu_cfg.get("topm_swap_rank_weight", 0.0)) > 0.0,
                            "v64_3_9_exact_runtime_topm_training_contract": str(
                                bdmu_cfg.get("topm_membership_source", "")
                            ).strip().lower() == "exact_runtime_hab",
                            "v64_3_9_exact_runtime_topm_reference_contract": str(
                                bdmu_cfg.get("reference_topm_pool_source", "")
                            ).strip().lower() == "exact_runtime_hab",
                        }
                    )

            checks.update(
                {
                    "v64_3_required_trainable_exact": (
                        (trainable == {"critical_proposal_adapter"} and bool(dbr.get("enabled", False)))
                        if metadata_version.startswith(("V64.3.8-", "V64.3.9-"))
                        else (({"decisive_boundary_pair_adapter"}.issubset(trainable) and bool(dbr.get("enabled", False)))
                              if metadata_version.startswith("V64.3.7-")
                              else (({"critical_proposal_adapter"}.issubset(trainable) and
                                     ((not bool(lbpr.get("enabled", False))) or "literal_boundary_pair_adapter" in trainable))
                                    if metadata_version.startswith("V64.3.6-")
                                    else V64_3_REQUIRED_TRAINABLE.issubset(trainable)))
                    ),
                    "v64_3_forbidden_legacy_trainables_absent": not bool(trainable & V64_3_FORBIDDEN_TRAINABLE),
                    "v64_3_critical_adapter_trainable": (
                        "critical_proposal_adapter" in trainable if not metadata_version.startswith("V64.3.7-") else "critical_proposal_adapter" not in trainable
                    ),
                    "v64_3_6_lbpr_trainable_when_enabled": (
                        (not metadata_version.startswith("V64.3.6-"))
                        or (not bool(lbpr.get("enabled", False)))
                        or ("literal_boundary_pair_adapter" in trainable)
                    ),
                    "v64_3_legacy_proposal_frozen": not bool(
                        trainable
                        & {"proposal_head", "family_head", "family_embed", "family_activity_proj", "proposal_feature_proj"}
                    ),
                    "v64_3_query_extension_frozen": "query_extension_proj" not in trainable,
                    "v64_3_7_dbr_checkpoint_optional_for_warm_start": (
                        (not metadata_version.startswith("V64.3.7-"))
                        or any(str(x) == "decisive_boundary_pair_adapter." for x in checkpoint_cfg.get("allowed_missing_prefixes", []))
                    ),
                    "v64_3_8_requires_pretrained_dbr_checkpoint": (
                        (not metadata_version.startswith(("V64.3.8-", "V64.3.9-")))
                        or (bool(dbr.get("enabled", False)) and not any(
                            str(x) == "decisive_boundary_pair_adapter."
                            for x in checkpoint_cfg.get("allowed_missing_prefixes", [])
                        ))
                    ),
                    "v64_3_7_screening_provenance_match": (
                        (not metadata_version.startswith(("V64.3.7-", "V64.3.8-", "V64.3.9-")))
                        or metadata_cfg.get("screening_only", None) == provenance_cfg.get("screening_only", None)
                    ),
                }
            )

    critical_signature = {
        "enabled": bool(critical_adapter.get("enabled", False)),
        "conditioning": str(critical_adapter.get("conditioning", "")).strip().lower(),
        "rank": int(critical_adapter.get("rank", 0)),
        "zero_init": bool(critical_adapter.get("zero_init", False)),
        "scale": float(critical_adapter.get("scale", 1.0)),
        "frontier_size": int(critical_adapter.get("frontier_size", 0)),
        "frontier_action_count": int(critical_adapter.get("frontier_action_count", 0)),
        "frontier_gap_bias_scale": float(critical_adapter.get("frontier_gap_bias_scale", 0.0)),
        "endpoint_cost_bias_scale": float(critical_adapter.get("endpoint_cost_bias_scale", 0.0)),
        "family_coupling_enabled": bool(family_coupling.get("enabled", False)),
        "family_coupling_scale": float(family_coupling.get("scale", 0.0)),
        "family_coupling_clip": float(family_coupling.get("clip", 0.0)),
        "lbpr_enabled": bool(lbpr.get("enabled", False)),
        "lbpr_rank": int(lbpr.get("rank", 0)),
        "lbpr_scale": float(lbpr.get("scale", 0.0)),
        "lbpr_endpoint_gate_floor": float(lbpr.get("endpoint_gate_floor", 0.0)),
        "lbpr_endpoint_gate_temperature": float(lbpr.get("endpoint_gate_temperature", 1.0)),
        "dbr_enabled": bool(dbr.get("enabled", False)),
        "dbr_rank": int(dbr.get("rank", 0)),
        "dbr_scale": float(dbr.get("scale", 0.0)),
    }
    failures = [name for name, ok in checks.items() if not ok]
    return {
        "path": str(path),
        "role": role,
        "experiment_name": exp.get("name"),
        "metadata_algorithm_version": metadata_version,
        "provenance_algorithm_version": provenance_version,
        "dense_query_feature_source": source,
        "critical_proposal_conditioning": str(critical_adapter.get("conditioning", "")),
        "critical_proposal_signature": critical_signature,
        "checks": checks,
        "pass": not failures,
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail fast on stale/non-V64 pipeline configs.")
    ap.add_argument("--train-config", type=Path, required=True)
    ap.add_argument("--eval-config", type=Path, required=True)
    ap.add_argument(
        "--expected-family",
        choices=["v64", "v64.3", "v64.3.1", "v64.3.2", "v64.3.3", "v64.3.4", "v64.3.5", "v64.3.6", "v64.3.7", "v64.3.8", "v64.3.9", "v64_3"],
        default="v64",
        help="Use v64.3+ to reject any stale V64.2 config even if generic V64 checks pass.",
    )
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    report = {
        "audit": "v64_pipeline_config_contract",
        "expected_family": args.expected_family,
        "train": _check(args.train_config, "train", args.expected_family),
        "eval": _check(args.eval_config, "eval", args.expected_family),
    }
    strict_v64_3 = args.expected_family in {"v64.3", "v64.3.1", "v64.3.2", "v64.3.3", "v64.3.4", "v64.3.5", "v64.3.6", "v64.3.7", "v64.3.8", "v64.3.9", "v64_3"}
    train_cond = report["train"].get("critical_proposal_conditioning", "")
    eval_cond = report["eval"].get("critical_proposal_conditioning", "")
    train_signature = report["train"].get("critical_proposal_signature", {})
    eval_signature = report["eval"].get("critical_proposal_signature", {})
    report["cross_config_checks"] = {
        "critical_proposal_conditioning_match": (not strict_v64_3) or train_cond == eval_cond,
        # V64.3.4 frontier representations have deployment semantics beyond the
        # conditioning string.  A train/eval frontier-size or rank mismatch can
        # silently change the acquisition policy without changing checkpoint
        # tensor shapes, so bind the complete adapter signature in provenance.
        "critical_proposal_signature_match": (not strict_v64_3) or train_signature == eval_signature,
    }
    report["pass"] = bool(
        report["train"]["pass"]
        and report["eval"]["pass"]
        and all(report["cross_config_checks"].values())
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
