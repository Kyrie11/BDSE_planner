from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bdse.config import load_config
from bdse.model.losses import _runtime_hab_topm_mask_from_scores
from bdse.tools.check_v64_3_9_runtime_topm_contract import run_contract as run_runtime_contract


def run_contract(config_path: str) -> dict:
    cfg = load_config(config_path)
    util = ((cfg.get("training", {}) or {}).get("budgeted_decisive_margin_utility", {}) or {})
    base = run_runtime_contract(config_path)

    fixture_cfg = {
        **cfg,
        "evidence": {**(cfg.get("evidence", {}) or {}), "budget": 1},
        "selector": {
            **(cfg.get("selector", {}) or {}),
            "proposal_top_m": 3,
            "hab_enabled": False,
            "interaction_family_ids": [2, 3],
            "min_soft_interaction_topm_slots": 2,
            "decision_budget_excludes_structural_safety": True,
            "structural_safety_include_feasibility": True,
            "force_hard_topm": True,
        },
    }
    utility = torch.tensor([[100.0, 9.0, 8.0, 7.0, 6.5, 6.0]], dtype=torch.float32)
    family_ids = torch.tensor([[1, 4, 4, 2, 2, 2]], dtype=torch.long)
    batch = {
        "evidence_active": torch.ones_like(utility, dtype=torch.bool),
        "evidence_budget_costs": torch.ones_like(utility),
        "evidence_family_ids": family_ids,
        "evidence_features": torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        "evidence_agent_group_ids": torch.tensor([[-1, -1, -1, 10, 10, 20]], dtype=torch.long),
    }
    projected = _runtime_hab_topm_mask_from_scores(utility, None, batch, fixture_cfg)
    projected_ids = set(torch.nonzero(projected[0], as_tuple=False).squeeze(1).tolist())
    expected = {1, 3, 5}

    family_fixture_cfg = {
        **cfg,
        "evidence": {**(cfg.get("evidence", {}) or {}), "budget": 2},
        "selector": {
            **(cfg.get("selector", {}) or {}),
            "proposal_top_m": 3,
            "hab_enabled": True,
            "hab_reserve_fraction": 0.0,
            "min_family_topm_slots": {1: 2, 2: 1},
            "min_soft_interaction_topm_slots": 0,
            "decision_budget_excludes_structural_safety": False,
            "force_hard_topm": False,
        },
    }
    family_utility = torch.tensor([[9.0, 8.0, 7.0, 100.0, 99.0, 98.0]], dtype=torch.float32)
    family_ids_2 = torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.long)
    family_logits = torch.tensor([[0.0, 10.0, -10.0, -20.0, -20.0, -20.0]], dtype=torch.float32)
    family_batch = {
        "evidence_active": torch.ones_like(family_utility, dtype=torch.bool),
        "evidence_budget_costs": torch.ones_like(family_utility),
        "evidence_family_ids": family_ids_2,
        "evidence_features": torch.zeros((1, 6, 2), dtype=torch.float32),
        "evidence_agent_group_ids": torch.full((1, 6), -1, dtype=torch.long),
    }
    family_projected = _runtime_hab_topm_mask_from_scores(
        family_utility, family_logits, family_batch, family_fixture_cfg
    )
    family_projected_ids = set(torch.nonzero(family_projected[0], as_tuple=False).squeeze(1).tolist())
    family_expected = {0, 1, 3}
    checks = {
        "runtime_topm_contract": bool(base.get("pass", False)),
        "admission_projection_exact_hab": str(util.get("admission_projection_mode", "")).strip().lower()
        in {"exact_hab_utility", "hab_utility_projection", "feasible_hab"},
        "feasible_admission_rank_active": float(util.get("feasible_admission_rank_weight", 0.0)) > 0.0,
        "feasible_admission_respects_family_strata": bool(util.get("feasible_admission_same_family", False)),
        "legacy_unconstrained_swap_disabled": abs(float(util.get("topm_swap_rank_weight", 0.0))) <= 1e-12,
        "utility_projection_matches_canonical_runtime_policy": projected_ids == expected,
        "utility_projection_respects_frozen_family_slots": family_projected_ids == family_expected,
    }
    return {
        "audit": "v64_3_10_hap_bdmu_contract",
        "config": config_path,
        "checks": checks,
        "pass": all(checks.values()),
        "utility_projection_fixture": sorted(projected_ids),
        "expected_fixture": sorted(expected),
        "family_slot_fixture": sorted(family_projected_ids),
        "family_slot_expected": sorted(family_expected),
        "runtime_topm_contract": base,
        "note": "HAP-BDMU may train only against an oracle Top-M that is realizable by the exact deployed HAB/post-processing policy.",
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
