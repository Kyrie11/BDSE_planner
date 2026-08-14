from __future__ import annotations

import argparse
import json
from pathlib import Path

from bdse.config import load_config
from bdse.tools.check_v64_3_11_btp_bdmu_contract import _exact_budget_override_fixture
from bdse.tools.check_v64_3_10_hap_bdmu_contract import run_contract as run_hap_contract


EXACT_SOURCES = {"exact_runtime_sampled", "sampled_exact_runtime", "runtime_exact_sampled"}


def run_contract(config_path: str) -> dict:
    cfg = load_config(config_path)
    util = ((cfg.get("training", {}) or {}).get("budgeted_decisive_margin_utility", {}) or {})
    metadata_version = str(((cfg.get("metadata", {}) or {}).get("algorithm_version", "")))
    variant = "cet" if "-CET-BDMU-" in metadata_version else "ret" if "-RET-BDMU-" in metadata_version else "unknown"

    hap = run_hap_contract(config_path)
    hap_checks = hap.get("checks", {}) or {}
    runtime_hab_semantics = all(bool(hap_checks.get(k, False)) for k in (
        "runtime_topm_contract",
        "admission_projection_exact_hab",
        "utility_projection_matches_canonical_runtime_policy",
        "utility_projection_respects_frozen_family_slots",
    ))
    exact_budget_fixture = _exact_budget_override_fixture()
    source = str(util.get("budget_transmission_selector_source", "")).strip().lower()
    allow_exchange = bool(util.get("budget_transmission_allow_controlled_budget_exchange", False))
    checks = {
        "known_v64_3_12_variant": variant in {"ret", "cet"},
        "runtime_and_hab_projection_semantics": runtime_hab_semantics,
        "exact_budget_projection_nested_in_injected_topm": bool(exact_budget_fixture["pass"]),
        "bdmu_enabled": bool(util.get("enabled", False)),
        "exact_runtime_topm_training": str(util.get("topm_membership_source", "")).strip().lower() == "exact_runtime_hab",
        "exact_runtime_topm_reference": str(util.get("reference_topm_pool_source", "")).strip().lower() == "exact_runtime_hab",
        "exact_hab_utility_projection": str(util.get("admission_projection_mode", "")).strip().lower()
        in {"exact_hab_utility", "hab_utility_projection", "feasible_hab"},
        "budget_transmission_rank_active": float(util.get("budget_transmission_rank_weight", 0.0)) > 0.0,
        "training_projection_is_sampled_exact_runtime": source in EXACT_SOURCES,
        "sampled_exact_scene_budget_positive": int(util.get("budget_transmission_exact_scenes_per_rank", 0)) > 0,
        "sampled_exact_runs_each_step": int(util.get("budget_transmission_exact_every_n_steps", 0)) == 1,
        "exact_sampling_is_actionable_scene_only": bool(util.get("budget_transmission_exact_candidate_only", False)),
        "exact_runtime_budget_projection_required_for_validation": bool(util.get("budget_transmission_exact_eval", False)),
        "same_family_only": bool(util.get("budget_transmission_same_family", False)),
        "cross_family_fallback_disabled": not bool(util.get("budget_transmission_cross_family_fallback", True)),
        "legacy_af_swap_disabled": abs(float(util.get("topm_swap_rank_weight", 0.0))) <= 1e-12,
        "legacy_hap_rank_disabled": abs(float(util.get("feasible_admission_rank_weight", 0.0))) <= 1e-12,
        "broad_listwise_disabled": abs(float(util.get("listwise_weight", 0.0))) <= 1e-12,
        "variant_exchange_semantics": allow_exchange if variant == "cet" else (not allow_exchange),
        "controlled_exchange_weight_bounded": 0.0 <= float(util.get("budget_transmission_controlled_exchange_weight", 0.5)) <= 1.0,
    }
    return {
        "audit": "v64_3_12_ret_cet_bdmu_contract",
        "config": config_path,
        "variant": variant,
        "checks": checks,
        "pass": all(checks.values()),
        "exact_budget_override_fixture": exact_budget_fixture,
        "inherited_hap_runtime_semantics": {
            "checks": {k: bool(hap_checks.get(k, False)) for k in (
                "runtime_topm_contract",
                "admission_projection_exact_hab",
                "utility_projection_matches_canonical_runtime_policy",
                "utility_projection_respects_frozen_family_slots",
            )},
            "utility_projection_fixture": hap.get("utility_projection_fixture"),
            "expected_fixture": hap.get("expected_fixture"),
            "family_slot_fixture": hap.get("family_slot_fixture"),
            "family_slot_expected": hap.get("family_slot_expected"),
        },
        "note": (
            "RET/CET-BDMU uses stop-gradient exact runtime B=16 selection during training on rotating "
            "actionable scenes. CET extends RET only by allowing an already-transmitted current-B atom "
            "to be a ranking negative when the exact oracle-B intervention drops that atom; broad "
            "current-B unprotection, cross-family fallback, AF/HAP/listwise objectives remain disabled."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output")
    args = ap.parse_args()
    report = run_contract(args.config)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
