from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bdse.config import load_config
from bdse.model.losses import _fast_topm_mask_torch, _runtime_hab_topm_hard_mask
from bdse.planner.hab import select_topm_atoms_hab
from bdse.planner.selector import finalize_runtime_topm_policy


def _fixture_cfg(base: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(base)
    cfg["evidence"] = dict(base.get("evidence", {}))
    cfg["selector"] = dict(base.get("selector", {}))
    cfg["training"] = dict(base.get("training", {}))
    cfg["evidence"]["budget"] = 1
    cfg["selector"].update(
        {
            "proposal_top_m": 3,
            "hab_enabled": False,
            "interaction_family_ids": [2, 3],
            "min_soft_interaction_topm_slots": 2,
            "decision_budget_excludes_structural_safety": True,
            "structural_safety_include_feasibility": True,
            "force_hard_topm": True,
        }
    )
    return cfg


def run_contract(config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    util = ((cfg.get("training", {}) or {}).get("budgeted_decisive_margin_utility", {}) or {})
    selector = cfg.get("selector", {}) or {}

    fixture_cfg = _fixture_cfg(cfg)
    logits = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.5, 6.0]], dtype=torch.float32)
    active = torch.ones_like(logits, dtype=torch.bool)
    costs = torch.ones_like(logits)
    family_ids = torch.tensor([[1, 4, 4, 2, 2, 2]], dtype=torch.long)
    features = torch.zeros((1, 6, 2), dtype=torch.float32)
    group_ids = torch.tensor([[-1, -1, -1, 10, 10, 20]], dtype=torch.long)

    initial, _, _ = select_topm_atoms_hab(
        logits[0].numpy(),
        family_ids[0].numpy(),
        active[0].numpy(),
        costs[0].numpy(),
        1.0,
        3,
        enabled=False,
    )
    canonical, _, _, diag = finalize_runtime_topm_policy(
        initial,
        proposal_scores=logits[0].numpy(),
        family_ids=family_ids[0].numpy(),
        active_mask=active[0].numpy(),
        max_size=3,
        selector_cfg=fixture_cfg["selector"],
        raw_hard_mask=features[0, :, 0].numpy() > 0.5,
        interaction_group_ids=group_ids[0].numpy(),
    )
    batch = {
        "evidence_active": active,
        "evidence_budget_costs": costs,
        "evidence_family_ids": family_ids,
        "evidence_features": features,
        "evidence_agent_group_ids": group_ids,
    }
    outputs = {"proposal_logits": logits}
    exact_mask = _runtime_hab_topm_hard_mask(outputs, batch, fixture_cfg)
    exact_ids = torch.nonzero(exact_mask[0], as_tuple=False).squeeze(1).cpu().numpy()
    fast_mask, _ = _fast_topm_mask_torch(
        logits, active, costs, family_ids, features, fixture_cfg, family_scores=None
    )
    fast_ids = torch.nonzero(fast_mask[0], as_tuple=False).squeeze(1).cpu().numpy()

    expected = {1, 3, 5}
    canonical_set = set(int(x) for x in canonical.tolist())
    exact_set = set(int(x) for x in exact_ids.tolist())
    fast_set = set(int(x) for x in fast_ids.tolist())
    checks = {
        "bdmu_enabled": bool(util.get("enabled", False)),
        "exact_runtime_topm_membership_configured": str(
            util.get("topm_membership_source", "")
        ).strip().lower() == "exact_runtime_hab",
        "exact_runtime_reference_pool_configured": str(
            util.get("reference_topm_pool_source", "")
        ).strip().lower() == "exact_runtime_hab",
        "screen_structural_bypass_active": bool(
            selector.get("decision_budget_excludes_structural_safety", False)
        ),
        "screen_soft_interaction_reserve_active": int(
            selector.get("min_soft_interaction_topm_slots", 0)
        ) > 0,
        "canonical_policy_is_group_aware": canonical_set == expected,
        "training_exact_mask_matches_canonical_policy": exact_set == canonical_set,
    }
    # This is diagnostic rather than a pass condition.  The fixture is designed
    # to expose why the historical fast surrogate cannot be used as the AF-BDMU
    # hard membership target: it selects two atoms from one agent group.
    report = {
        "audit": "v64_3_9_runtime_topm_contract",
        "config": str(config_path),
        "checks": checks,
        "pass": all(checks.values()),
        "fixture": {
            "canonical_topm": sorted(canonical_set),
            "exact_training_topm": sorted(exact_set),
            "legacy_fast_surrogate_topm": sorted(fast_set),
            "legacy_fast_surrogate_differs": fast_set != canonical_set,
            "policy_diagnostics": {k: int(v) for k, v in diag.items()},
        },
        "note": (
            "AF-BDMU hard membership supervision must use exact runtime HAB Top-M. "
            "The legacy fast surrogate is retained only as a speed/instrumentation "
            "diagnostic and is not allowed to define swap-ranking membership or the "
            "frozen reference Top-M pool."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = run_contract(args.config)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
