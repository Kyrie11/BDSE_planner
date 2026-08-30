from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from bdse.config import load_config
from bdse.model.losses import _fast_pair_margin_surrogate_masks, _predicted_pair_certificate_masks


def _mean_time(fn, repeats: int) -> float:
    values: list[float] = []
    for _ in range(max(repeats, 1)):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic selector benchmark matching the v45 training tensor sizes.")
    parser.add_argument("--config", default="bdse/configs/v45_bdse_pb_rads_train_fast_2gpu.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--actions", type=int, default=32)
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    B, K, E, P = args.batch, args.actions, args.atoms, args.pairs
    gen = torch.Generator(device="cpu").manual_seed(17)
    pairs = torch.randint(0, K, (B, P, 2), generator=gen)
    pairs[..., 1] = (pairs[..., 0] + torch.randint(1, K, (B, P), generator=gen)) % K
    batch = {
        "pair_indices": pairs.to(device),
        "pair_valid": torch.ones(B, P, dtype=torch.bool, device=device),
        "pair_weights": (torch.rand(B, P, generator=gen) + 0.1).to(device),
        "candidate_valid": torch.ones(B, K, dtype=torch.bool, device=device),
        "evidence_active": torch.ones(B, E, dtype=torch.bool, device=device),
        "evidence_budget_costs": torch.ones(B, E, device=device),
        "evidence_family_ids": torch.randint(1, 6, (B, E), generator=gen).to(device),
        "evidence_agent_group_ids": torch.randint(-1, 20, (B, E), generator=gen).to(device),
        "runtime_safety_flags": torch.zeros(B, K, dtype=torch.bool, device=device),
        "evidence_features": torch.randn(B, E, 18, generator=gen).to(device),
        "decisive_hard_mask": torch.zeros(B, E, dtype=torch.bool, device=device),
    }
    batch["evidence_features"][..., 0] = (torch.rand(B, E, generator=gen) < 0.1).to(device)
    outputs = {
        "J0": (torch.randn(B, K, generator=gen) * 1000.0).to(device),
        "pair_atom_delta": (torch.randn(B, E, P, generator=gen) * 0.1).to(device),
        "proposal_logits": torch.randn(B, E, generator=gen).to(device),
        "family_logits": torch.randn(B, 8, generator=gen).to(device),
    }

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    _fast_pair_margin_surrogate_masks(outputs, batch, cfg, [8.0, 16.0])
    _predicted_pair_certificate_masks(outputs, batch, cfg, scene_indices=torch.tensor([0], device=device))
    sync()

    def fast() -> None:
        _fast_pair_margin_surrogate_masks(outputs, batch, cfg, [8.0, 16.0])
        sync()

    def exact_full() -> None:
        _predicted_pair_certificate_masks(outputs, batch, cfg)
        sync()

    def exact_one() -> None:
        _predicted_pair_certificate_masks(outputs, batch, cfg, scene_indices=torch.tensor([0], device=device))
        sync()

    fast_s = _mean_time(fast, args.repeats)
    exact_one_s = _mean_time(exact_one, args.repeats)
    exact_full_s = _mean_time(exact_full, max(1, min(args.repeats, 3)))
    amortized = fast_s + exact_one_s / 4.0
    legacy_two_budget = 2.0 * exact_full_s
    print(f"device={device} B={B} K={K} E={E} P={P}")
    print(f"fast_two_budget_s={fast_s:.6f}")
    print(f"exact_one_scene_s={exact_one_s:.6f}")
    print(f"exact_full_batch_one_budget_s={exact_full_s:.6f}")
    print(f"legacy_exact_two_budget_s_per_step~={legacy_two_budget:.6f}")
    print(f"hybrid_fast_amortized_s_per_step~={amortized:.6f}")
    if amortized > 0:
        print(f"selector_path_speedup~={legacy_two_budget / amortized:.2f}x")


if __name__ == "__main__":
    main()
