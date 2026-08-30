from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bdse.config import load_config
from bdse.model.losses import _predicted_pair_certificate_masks
from bdse.tools.check_v64_3_10_hap_bdmu_contract import run_contract as run_hap_contract


def _exact_budget_override_fixture() -> dict:
    """Verify exact C1-B/C2-B selection is nested in an injected final Top-M.

    This fixture intentionally places high-scoring soft-interaction atoms outside
    the injected Top-M.  A historical evaluation-adapter bug re-ran the Top-M
    reservation policy after injection and pulled those atoms back into the B
    candidate domain.  The paper-level mediation contract requires strict
    B subset Top-M semantics, so catch that regression before any GPU training.
    """
    outputs = {
        "J0": torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32),
        "proposal_logits": torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32),
        "pair_atom_delta": torch.tensor(
            [[[0.2, 0.1], [0.1, -0.2], [0.3, 0.1], [-0.1, 0.2]]],
            dtype=torch.float32,
        ),
    }
    batch = {
        "pair_indices": torch.tensor([[[0, 1], [0, 2]]], dtype=torch.long),
        "pair_valid": torch.tensor([[True, True]]),
        "pair_weights": torch.ones((1, 2), dtype=torch.float32),
        "candidate_valid": torch.ones((1, 3), dtype=torch.bool),
        "evidence_active": torch.ones((1, 4), dtype=torch.bool),
        "evidence_budget_costs": torch.ones((1, 4), dtype=torch.float32),
        "evidence_family_ids": torch.tensor([[2, 2, 3, 4]], dtype=torch.long),
        "evidence_agent_group_ids": torch.tensor([[10, 10, 11, -1]], dtype=torch.long),
        "runtime_safety_flags": torch.zeros((1, 3), dtype=torch.bool),
        "evidence_features": torch.zeros((1, 4, 2), dtype=torch.float32),
    }
    fixture_cfg = {
        "evidence": {"budget": 2},
        "model": {"pair_margin_normalized": True},
        "tournament": {},
        "selector": {
            "proposal_top_m": 4,
            "hab_enabled": False,
            "selector_cap_mode": "margin_coreset",
            "soft_interaction_quota": 1,
            "min_soft_interaction_topm_slots": 2,
            "interaction_family_ids": [2, 3],
        },
    }
    override = torch.tensor([[False, False, True, True]], dtype=torch.bool)
    oracle_scores = torch.tensor([[0.1, 0.2, 2.0, 1.0]], dtype=torch.float32)
    selected = _predicted_pair_certificate_masks(
        outputs, batch, fixture_cfg,
        topm_mask_override=override,
        proposal_scores_override=oracle_scores,
    )
    violation = selected & ~override
    return {
        "pass": bool(not violation.any().item()),
        "injected_topm": torch.nonzero(override[0], as_tuple=False).squeeze(1).tolist(),
        "selected_budget": torch.nonzero(selected[0], as_tuple=False).squeeze(1).tolist(),
        "outside_topm": torch.nonzero(violation[0], as_tuple=False).squeeze(1).tolist(),
    }


def run_contract(config_path: str) -> dict:
    cfg = load_config(config_path)
    util = ((cfg.get("training", {}) or {}).get("budgeted_decisive_margin_utility", {}) or {})
    hap = run_hap_contract(config_path)
    # HAP's legacy contract intentionally requires the V64.3.10 feasible rank to
    # be active, so for BTP reuse only its runtime/HAB semantic subchecks.
    hap_checks = hap.get("checks", {}) or {}
    runtime_hab_semantics = all(bool(hap_checks.get(k, False)) for k in (
        "runtime_topm_contract",
        "admission_projection_exact_hab",
        "utility_projection_matches_canonical_runtime_policy",
        "utility_projection_respects_frozen_family_slots",
    ))
    exact_budget_fixture = _exact_budget_override_fixture()
    checks = {
        "runtime_and_hab_projection_semantics": runtime_hab_semantics,
        "exact_budget_projection_nested_in_injected_topm": bool(exact_budget_fixture["pass"]),
        "bdmu_enabled": bool(util.get("enabled", False)),
        "exact_runtime_topm_training": str(util.get("topm_membership_source", "")).strip().lower() == "exact_runtime_hab",
        "exact_runtime_topm_reference": str(util.get("reference_topm_pool_source", "")).strip().lower() == "exact_runtime_hab",
        "exact_hab_utility_projection": str(util.get("admission_projection_mode", "")).strip().lower()
        in {"exact_hab_utility", "hab_utility_projection", "feasible_hab"},
        "budget_transmission_rank_active": float(util.get("budget_transmission_rank_weight", 0.0)) > 0.0,
        "budget_projection_uses_frozen_pair_selector": str(util.get("budget_transmission_selector_source", "")).strip().lower()
        == "frozen_pair_margin_surrogate",
        "exact_runtime_budget_projection_required_for_validation": bool(util.get("budget_transmission_exact_eval", False)),
        "same_family_only": bool(util.get("budget_transmission_same_family", False)),
        "cross_family_fallback_disabled": not bool(util.get("budget_transmission_cross_family_fallback", True)),
        "current_budget_evidence_protected": bool(util.get("budget_transmission_protect_current_budget", False)),
        "legacy_af_swap_disabled": abs(float(util.get("topm_swap_rank_weight", 0.0))) <= 1e-12,
        "legacy_hap_rank_disabled": abs(float(util.get("feasible_admission_rank_weight", 0.0))) <= 1e-12,
        "broad_listwise_disabled": abs(float(util.get("listwise_weight", 0.0))) <= 1e-12,
    }
    return {
        "audit": "v64_3_11_btp_bdmu_contract",
        "config": config_path,
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
            "BTP-BDMU may supervise proposal logits only with evidence that is exact-HAB feasible and "
            "survives the frozen B=16 pair-margin selector. Current budget-selected evidence is protected; "
            "AF/HAP broad ranking objectives are disabled."
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
