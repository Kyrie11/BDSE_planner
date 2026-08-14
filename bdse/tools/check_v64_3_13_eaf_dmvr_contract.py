from __future__ import annotations

import argparse
import json
from pathlib import Path

from bdse.config import load_config
from bdse.tools.check_v64_3_11_btp_bdmu_contract import _exact_budget_override_fixture


def run_contract(config_path: str) -> dict:
    cfg = load_config(config_path)
    model = cfg.get("model", {}) or {}
    runtime = cfg.get("runtime", {}) or {}
    training = cfg.get("training", {}) or {}
    selector = cfg.get("selector", {}) or {}
    evidence = cfg.get("evidence", {}) or {}
    lw = training.get("loss_weights", {}) or {}
    fv = model.get("decisive_anchor_frontier_value", {}) or {}
    rfv = runtime.get("decisive_frontier_value", {}) or {}
    tfv = training.get("decisive_frontier_value", {}) or {}
    bdmu = training.get("budgeted_decisive_margin_utility", {}) or {}
    metadata_version = str((cfg.get("metadata", {}) or {}).get("algorithm_version", ""))
    positive_losses = {str(k) for k, v in lw.items() if abs(float(v)) > 1e-12}
    trainable = [str(x) for x in training.get("trainable_modules", [])]
    allowed_missing = [str(x) for x in (cfg.get("checkpoint_loading", {}) or {}).get("allowed_missing_prefixes", [])]
    exact_fixture = _exact_budget_override_fixture()

    checks = {
        "version_is_v64_3_13_eaf_dmvr": metadata_version == "V64.3.13-EAF-DMVR-DARM-DBR",
        "fixed_evidence_budget_B16": int(evidence.get("budget", -1)) == 16,
        "fixed_proposal_topm_M24": int(selector.get("proposal_top_m", -1)) == 24,
        "acquisition_bdmu_terminally_disabled": not bool(bdmu.get("enabled", False)),
        "no_acquisition_loss_positive": all(abs(float(lw.get(k, 0.0))) <= 1e-12 for k in (
            "proposal", "critical_proposal", "counterfactual_critical_proposal",
            "exact_winner_flip_critical_proposal", "deployment_selection",
            "budgeted_decisive_margin_utility", "family",
        )),
        "frontier_value_only_positive_loss": positive_losses == {"decisive_frontier_value"},
        "frontier_model_enabled": bool(fv.get("enabled", False)),
        "frontier_runtime_enabled": bool(rfv.get("enabled", False)),
        "frontier_training_enabled": bool(tfv.get("enabled", False)),
        "complete_anchor_frontier_required": bool(tfv.get("complete_frontier_required", False)),
        "selection_stop_gradient": bool(tfv.get("stop_gradient_selection", False)),
        "runtime_aggregation_remains_darm": str(runtime.get("pair_tournament_aggregation_mode", "")).strip().lower() == "decisive_anchor_margin",
        "dbr_remains_enabled": bool((model.get("decisive_boundary_pair_adapter", {}) or {}).get("enabled", False)),
        "only_new_frontier_head_trainable": trainable == ["decisive_anchor_frontier_value_adapter"],
        "exact_B16_training_only": list(training.get("deployment_budgets", [])) == [16]
            and int(training.get("deployment_primary_budget", -1)) == 16,
        "sampled_exact_runtime_selector": str(training.get("deployment_selector_backend", "")).strip().lower() == "exact_cpu"
            and int(training.get("deployment_selector_scenes_per_rank", 0)) > 0
            and int(training.get("deployment_selector_every_n_steps", 0)) == 1,
        "new_head_is_warm_start_optional": "decisive_anchor_frontier_value_adapter." in allowed_missing,
        "exact_B_nested_in_topm_fixture": bool(exact_fixture.get("pass", False)),
        "generic_set_potential_disabled": int(model.get("set_residual_rank", 0)) == 0,
        "legacy_evidence_action_potential_disabled": not bool(model.get("evidence_action_residual", False)),
    }
    return {
        "audit": "v64_3_13_eaf_dmvr_contract",
        "config": config_path,
        "checks": checks,
        "pass": all(checks.values()),
        "exact_budget_override_fixture": exact_fixture,
        "note": (
            "V64.3.13 freezes proposal/HAB/acquisition after the V64.3.12 terminal stop. "
            "Only the evidence-attributed complete anchor-frontier value head is trainable; "
            "it consumes the exact runtime-selected B=16 set and does not modify selector scores or budgets."
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
