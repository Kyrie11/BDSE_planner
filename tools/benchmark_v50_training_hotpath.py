from __future__ import annotations

import argparse
import copy
import json
import time
from typing import Any

import numpy as np
import torch

from bdse.model.losses import (
    _config_with_evidence_budget,
    _predicted_pair_certificate_masks_multi_budget,
    _shutdown_exact_selector_process_pools,
)


def _synthetic_case(batch_size: int, atoms: int, actions: int, pair_count: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    rng = np.random.default_rng(50)
    pairs = np.stack(
        [rng.integers(0, actions, (batch_size, pair_count)), rng.integers(0, actions, (batch_size, pair_count))],
        axis=-1,
    )
    same = pairs[:, :, 0] == pairs[:, :, 1]
    pairs[:, :, 1][same] = (pairs[:, :, 1][same] + 1) % actions
    outputs = {
        "J0": torch.tensor(rng.normal(size=(batch_size, actions)), dtype=torch.float32),
        "proposal_logits": torch.tensor(rng.normal(size=(batch_size, atoms)), dtype=torch.float32),
        "pair_atom_delta": torch.tensor(
            rng.normal(scale=0.2, size=(batch_size, atoms, pair_count)), dtype=torch.float32
        ),
        "pair_atom_var": torch.full((batch_size, atoms, pair_count), 0.04, dtype=torch.float32),
    }
    batch = {
        "pair_indices": torch.tensor(pairs, dtype=torch.long),
        "pair_valid": torch.ones((batch_size, pair_count), dtype=torch.bool),
        "pair_weights": torch.ones((batch_size, pair_count), dtype=torch.float32),
        "candidate_valid": torch.ones((batch_size, actions), dtype=torch.bool),
        "evidence_active": torch.ones((batch_size, atoms), dtype=torch.bool),
        "evidence_budget_costs": torch.ones((batch_size, atoms), dtype=torch.float32),
        "evidence_family_ids": torch.tensor(
            np.tile(np.arange(atoms) % 6, (batch_size, 1)), dtype=torch.long
        ),
        "evidence_agent_group_ids": torch.full((batch_size, atoms), -1, dtype=torch.long),
        "runtime_safety_flags": torch.zeros((batch_size, actions), dtype=torch.bool),
        "evidence_features": torch.zeros((batch_size, atoms, 2), dtype=torch.float32),
    }
    cfg: dict[str, Any] = {
        "evidence": {"budget": 16},
        "model": {"pair_margin_normalized": True},
        "runtime": {"pair_delta_hybrid_local_weight": 0.0},
        "calibration": {"epsilon_cal": 0.05},
        "tournament": {"beta_uncertainty": 0.5, "epsilon_cal": 0.05},
        "training": {},
        "selector": {
            "proposal_top_m": min(24, atoms),
            "hab_enabled": True,
            "selector_cap_mode": "anytime_adverse_certificate",
            "decision_budget_excludes_structural_safety": True,
            "collapse_reciprocal_pairs": True,
            "force_fill_budget": True,
            "min_selected_atoms": min(16, atoms),
            "soft_interaction_quota": 0,
            "adverse_certificate_beta": 1.0,
            "adverse_certificate_epsilon": 0.05,
            "adverse_certificate_prior_radius": 0.02,
            "adverse_certificate_stop_when_certified": False,
            "adverse_certificate_max_target_rivals": 6,
            "max_selector_pairs": pair_count,
            "adverse_certificate_fill_to_budget_after_certified": True,
            "adverse_certificate_max_interaction_prefix_fraction": 0.75,
        },
    }
    return outputs, batch, cfg


def _run(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    *,
    backend: str,
    workers: int,
    repeats: int,
) -> tuple[list[float], list[torch.Tensor]]:
    cfgs = [_config_with_evidence_budget(copy.deepcopy(cfg), budget) for budget in (16.0, 8.0)]
    for local_cfg in cfgs:
        local_cfg.setdefault("training", {})["deployment_selector_cpu_backend"] = backend
        local_cfg["training"]["deployment_selector_cpu_workers"] = int(workers)
    times: list[float] = []
    result: list[torch.Tensor] = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        result = _predicted_pair_certificate_masks_multi_budget(outputs, batch, cfgs)
        times.append(time.perf_counter() - started)
    return times, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--actions", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=56)
    parser.add_argument("--process-workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    outputs, batch, cfg = _synthetic_case(args.batch_size, args.atoms, args.actions, args.pairs)
    sequential_times, sequential = _run(
        outputs, batch, cfg, backend="sequential", workers=1, repeats=args.repeats
    )
    process_times, process = _run(
        outputs,
        batch,
        cfg,
        backend="process",
        workers=args.process_workers,
        repeats=args.repeats,
    )
    exact_equal = all(torch.equal(a, b) for a, b in zip(sequential, process))
    sequential_steady = sequential_times[-1]
    process_steady = process_times[-1]
    report = {
        "shape": {
            "batch_size": args.batch_size,
            "atoms": args.atoms,
            "actions": args.actions,
            "pairs": args.pairs,
            "budgets": [16, 8],
        },
        "sequential_seconds": sequential_times,
        "process_seconds": process_times,
        "process_workers": args.process_workers,
        "steady_state_speedup": sequential_steady / max(process_steady, 1e-12),
        "masks_exactly_equal": exact_equal,
        "note": "The first process call includes one-time spawn/import startup; compare the final repeat for steady state.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    _shutdown_exact_selector_process_pools()
    if not exact_equal:
        raise SystemExit("process backend changed exact selector masks")


if __name__ == "__main__":
    main()
